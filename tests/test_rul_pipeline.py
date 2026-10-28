"""Pytest checks for FD001 engineering assumptions and saved model results."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from scripts.rul_utils import (
    calculate_rul, split_engine_indices, select_sensor_columns,
    last_observation_per_engine, predict_nonnegative_rul,
)

ROOT = Path(__file__).resolve().parents[1]

def sample():
    """Supply three small, interleaved, complete synthetic engine histories."""
    return pd.DataFrame({'engine_id':[1,2,1,2,1,3,3],
                         'cycle':[1,1,2,2,3,1,2],
                         'sensor_1':[100]*7,
                         'sensor_2':[11,22,12,23,13,33,34],
                         'sensor_3':[0,1,0,1,0,0,1]})

def test_uncapped_rul_per_engine():
    """Correctly label each complete engine history, ending at RUL zero."""
    assert calculate_rul(sample()).tolist()==[2,1,1,0,0,1,0]

def test_rul_keeps_row_index():
    """Training labels preserve their source row identity."""
    data=sample()
    data.index=[30,31,32,33,34,35,36]
    assert calculate_rul(data).index.tolist()==data.index.tolist()

@pytest.mark.parametrize('invalid',[
    pd.DataFrame({'engine_id':[1,1],'cycle':[1,1]}),
    pd.DataFrame({'engine_id':[1,1],'cycle':[0,1]}),
    pd.DataFrame({'engine_id':[1,None],'cycle':[1,2]}),
    pd.DataFrame({'engine_id':[1],'cycle':[None]}),
])
def test_reject_invalid_training_histories(invalid):
    """Reject missing or duplicate engine cycles and invalid cycle numbers."""
    with pytest.raises(ValueError): calculate_rul(invalid)

def test_rul_requires_ids():
    """An engine identifier is needed to calculate a per-engine lifetime."""
    with pytest.raises(ValueError):
        calculate_rul(pd.DataFrame({'cycle':[1,2]}))

def test_engine_level_split_prevents_leakage():
    """All cycles of one engine must belong to only one fold."""
    data=pd.DataFrame({'engine_id':np.repeat(np.arange(1,31),4),
                       'cycle':np.tile([1,2,3,4],30)})
    a,b=split_engine_indices(data,test_size=.2,random_state=42)
    assert len(a)+len(b)==len(data)
    assert data.iloc[a].engine_id.nunique()==24
    assert data.iloc[b].engine_id.nunique()==6
    assert set(data.iloc[a].engine_id).isdisjoint(data.iloc[b].engine_id)
    a2,b2=split_engine_indices(data,test_size=.2,random_state=42)
    assert np.array_equal(a,a2) and np.array_equal(b,b2)

def test_single_engine_cannot_be_split():
    """A grouped holdout needs at least two independent engines."""
    with pytest.raises(ValueError):
        split_engine_indices(pd.DataFrame({'engine_id':[1,1],'cycle':[1,2]}))

def test_sensor_selection_uses_training_only():
    """Exclude constant/near-constant channels from development data only."""
    selected,excluded=select_sensor_columns(sample())
    assert selected==['sensor_2']
    assert excluded==['sensor_1','sensor_3']

def test_no_sensor_columns_rejected():
    """The feature selection rule fails safely on invalid input."""
    with pytest.raises(ValueError):
        select_sensor_columns(pd.DataFrame({'cycle':[1,2]}))

def test_sensor_threshold_must_be_positive():
    """Feature rule thresholds must be meaningful."""
    with pytest.raises(ValueError):
        select_sensor_columns(sample(),max_unique=0)

def test_latest_snapshot_is_label_free():
    """Choose last observed test cycle per engine without future information."""
    endpoints=last_observation_per_engine(sample().sample(frac=1,random_state=7))
    assert endpoints.engine_id.tolist()==[1,2,3]
    assert endpoints.cycle.tolist()==[3,2,2]
    assert 'RUL' not in endpoints

def test_duplicate_snapshot_cycle_rejected():
    """Test-time duplicate engine-cycle measurements cannot be silently dropped."""
    with pytest.raises(ValueError):
        last_observation_per_engine(pd.DataFrame({'engine_id':[1,1],'cycle':[1,1]}))

def test_scaler_fits_only_on_training():
    """Validation values cannot influence fitted feature means."""
    train=np.array([[1.,5.],[2.,7.],[3.,9.]])
    holdout=np.array([[999.,999.]])
    scaler=StandardScaler().fit(train)
    np.testing.assert_allclose(scaler.mean_,train.mean(axis=0))
    assert scaler.n_samples_seen_==len(train)
    assert (scaler.transform(holdout)>100).all()

def test_negative_life_is_clipped():
    """Negative model forecasts cannot become negative cycles of useful life."""
    class Example:
        def predict(self,x):
            """Produce two illustrative predictions."""
            return [-5.,7.5]
    np.testing.assert_array_equal(predict_nonnegative_rul(Example(),[[1],[2]]),[0.,7.5])

def test_nan_predictions_fail():
    """Non-finite predictions fail rather than being hidden."""
    class Broken:
        def predict(self,x):
            """Deliberately produce NaN."""
            return [np.nan]
    with pytest.raises(ValueError):
        predict_nonnegative_rul(Broken(),[[1]])

def test_sklearn_linear_regression_clip():
    """The helper works with a real fitted scikit-learn estimator."""
    model=LinearRegression().fit([[0],[1],[2]],[2,1,0])
    np.testing.assert_allclose(predict_nonnegative_rul(model,[[0],[3]]),[2.,0.],atol=1e-10)

def test_all_saved_predictions_present():
    """Official NASA results contain one prediction per independent test engine."""
    data=pd.read_csv(ROOT/'stage4/results/official_test_predictions_by_engine.csv')
    assert len(data)==100 and data.engine_id.is_unique
    assert data.engine_id.tolist()==list(range(1,101))
    assert np.isfinite(data.select_dtypes('number').to_numpy()).all()
    assert (data.filter(like='pred_')>=0).all().all()

def test_saved_error_metrics_reproduce():
    """Recalculate official MAE/RMSE/R² from actual saved per-engine predictions."""
    values=pd.read_csv(ROOT/'stage4/results/official_test_predictions_by_engine.csv')
    summary=pd.read_csv(ROOT/'stage4/results/official_test_metrics.csv')
    actual=values.actual_RUL.to_numpy()
    for _,row in summary.iterrows():
        name='pred_'+row['model'].lower().replace(' ','_').replace('-','_')
        pred=values[name].to_numpy()
        assert np.isclose(mean_absolute_error(actual,pred),row.MAE_cycles,atol=1e-7)
        assert np.isclose(np.sqrt(mean_squared_error(actual,pred)),row.RMSE_cycles,atol=1e-7)
        assert np.isclose(r2_score(actual,pred),row.R2,atol=1e-7)
        assert int((pred>actual).sum())==row.overestimation_count

def test_expected_fd001_neural_metrics():
    """Guard against accidentally replacing the original official test results."""
    values=pd.read_csv(ROOT/'stage4/results/official_test_metrics.csv').set_index('model')
    assert abs(values.loc['Neural Network','MAE_cycles']-18.24)<.02
    assert values.loc['Neural Network','overestimation_count']==64

def test_scripts_are_syntactically_valid_and_documented():
    """Ensure source modules and functions have docstrings and parse correctly."""
    files=[ROOT/'run_all.py',*sorted((ROOT/'scripts').glob('*.py'))]
    files += [*sorted((ROOT/'stage1').glob('*.py')),
              *sorted((ROOT/'stage2').glob('*.py')),
              *sorted((ROOT/'stage3').glob('*.py')),
              *sorted((ROOT/'stage4').glob('*.py'))]
    assert len(files)>=9
    for f in files:
        tree=ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
        assert ast.get_docstring(tree),f'Missing module docstring: {f}'
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                assert ast.get_docstring(node),f'Missing function docstring: {f}:{node.name}'

def test_license_readme_and_generated_docs_exist():
    """Ensure the required repository deliverables have identifiable locations."""
    assert (ROOT/'README.md').exists()
    assert (ROOT/'LICENSE').read_text().startswith('MIT License')
    assert (ROOT/'docs/html/index.html').exists()
    assert (ROOT/'docs/html/scripts_rul_utils.html').exists()
