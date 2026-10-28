# %%
"""MMA3001 Stage 3: FD001 regression model development.
Run from the Stage3 directory. No official NASA test labels/files are loaded.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from scripts.rul_utils import predict_nonnegative_rul

BASE = Path(__file__).resolve().parent
DATA, RESULTS, FIG, MODELS = (BASE / sub for sub in ('data', 'results', 'figures', 'models'))
for folder in (RESULTS, FIG, MODELS): folder.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.14, 'figure.dpi': 110})
print('Stage 3 project:', BASE)

# %%
manifest = json.loads((DATA / 'feature_manifest.json').read_text())
FEATURES = manifest['feature_names']
train = pd.read_csv(DATA / 'train_engine_rows.csv')
val = pd.read_csv(DATA / 'validation_snapshots.csv')
val_histories = pd.read_csv(DATA / 'validation_engine_rows.csv')
assert FEATURES == ['cycle'] + [f'sensor_{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
assert train.shape[0] == 16561 and val.shape[0] == 20 and len(val_histories) == 4070
assert train.engine_id.nunique() == 80 and val.engine_id.nunique() == 20
assert set(train.engine_id).isdisjoint(set(val.engine_id))
assert set(val.engine_id) == set(val_histories.engine_id)
assert all(name in train.columns and name in val.columns for name in FEATURES)
assert train[FEATURES+['RUL']].notna().all().all() and val[FEATURES+['RUL']].notna().all().all()
assert (train.RUL >= 0).all() and (val.RUL > 0).all()
assert 'engine_id' not in FEATURES and 'RUL' not in FEATURES
assert 'test' not in ' '.join(p.name.lower() for p in DATA.iterdir())
X_train = train[FEATURES]
y_train = train.RUL.to_numpy()
X_val = val[FEATURES]
y_val = val.RUL.to_numpy()
print('Training:',len(train),'rows from',train.engine_id.nunique(),'engines')
print('Validation:',len(val),'snapshots from',val.engine_id.nunique(),'unseen engines')
print('Inputs:',len(FEATURES),'= cycle + 14 retained sensor values')
print('Target: original uncapped RUL in cycles')
print('Validation RUL: min',y_val.min(),'/ median',np.median(y_val),'/ max',y_val.max())

# %%
# Fixed preselected configurations; no grid search on 20 validation snapshots.
# Each pipeline independently fits its scaler using the TRAINING engines only.
# MLP early stopping uses a fraction of those training-engine rows; NEVER the 20
# entirely held-out validation engines. Its internal row-level stopping split
# is not a reliable estimate of generalisation and is not reported as such.
# Clip all RUL estimates at 0 cycles because negative useful life is not meaningful.
models = {
    'Mean baseline': DummyRegressor(strategy='mean'),
    'Cycle-only baseline': LinearRegression(),
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())]),
    'Random Forest': RandomForestRegressor(
        n_estimators=160, max_depth=14, min_samples_leaf=5,
        max_features=0.8, random_state=42, n_jobs=-1),
    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(48,24), activation='relu', solver='adam',
            alpha=0.01, batch_size=256, learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, tol=0.0008,
            random_state=42, shuffle=True))]),
}
PREDICTOR_SETS = {
    'Mean baseline': FEATURES,
    'Cycle-only baseline': ['cycle'],
    'Linear Regression': FEATURES,
    'Random Forest': FEATURES,
    'Neural Network': FEATURES,
}
PRED_ORDER = list(models)

# %%
def predict_rul(model, X):
    """Return physical nonnegative remaining-life predictions in cycles."""
    return predict_nonnegative_rul(model, X)

records=[]
all_predictions={'engine_id':val.engine_id.to_numpy(dtype=int),
                 'observed_cycle':val.cycle.to_numpy(dtype=int),
                 'actual_RUL': y_val.astype(float)}
for name, model in models.items():
    cols = PREDICTOR_SETS[name]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X_train[cols], y_train)
    elapsed = time.perf_counter() - t0
    preds = predict_rul(model, X_val[cols])
    train_pred = predict_rul(model, X_train[cols])
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    train_mae = mean_absolute_error(y_train, train_pred)
    records.append({'model':name, 'MAE_cycles':mae, 'RMSE_cycles':rmse,
                    'R2':r2, 'train_MAE_cycles':train_mae,
                    'fit_time_seconds':elapsed, 'n_validation_engines':len(val)})
    all_predictions[f'pred_{name.replace(" ", "_")}'] = preds
    joblib.dump(model, MODELS / (name.lower().replace(' ','_')+'.joblib'))
    for w in observed_warnings:
        if issubclass(w.category, ConvergenceWarning):
            print('Model warning:',name, str(w.message))
    print(f'{name:20s} MAE={mae:7.2f} RMSE={rmse:7.2f} R²={r2:7.3f}'
          f' | training MAE={train_mae:7.2f} | fit={elapsed:6.2f}s')
metrics = pd.DataFrame(records).sort_values('RMSE_cycles').reset_index(drop=True)
predictions = pd.DataFrame(all_predictions)
assert len(predictions)==20 and predictions.engine_id.is_unique
assert predictions.isna().sum().sum()==0
assert np.isfinite(metrics[['MAE_cycles','RMSE_cycles','R2']].to_numpy()).all()
metrics.to_csv(RESULTS / 'model_comparison_validation.csv', index=False)
predictions.to_csv(RESULTS / 'validation_engine_predictions.csv', index=False)
print('\nRanked by validation RMSE (20 independent engine endpoints):')
print(metrics[['model','MAE_cycles','RMSE_cycles','R2','train_MAE_cycles']].round(3).to_string(index=False))

# %%
# Uncertainty: nonparametric bootstrap of 20 paired engine endpoints.
# Uncertainty intervals describe finite validation-sample variation, not real-world reliability.
rng = np.random.default_rng(2026)
idx = rng.integers(0,len(val),size=(2000,len(val)))
uncertainty=[]
for name in PRED_ORDER:
    errors = np.abs(predictions[f'pred_{name.replace(" ", "_")}'].to_numpy() - y_val)
    sample_maes = errors[idx].mean(axis=1)
    low, high = np.quantile(sample_maes,[0.025,0.975])
    uncertainty.append({'model':name, 'MAE_CI_low_cycles':low, 'MAE_CI_high_cycles':high})
uncertainty=pd.DataFrame(uncertainty)
metrics=metrics.merge(uncertainty,on='model')
metrics.to_csv(RESULTS / 'model_comparison_validation.csv',index=False)
print('\nApproximate 95% bootstrap uncertainty intervals for validation MAE (n=20):')
print(metrics[['model','MAE_cycles','MAE_CI_low_cycles','MAE_CI_high_cycles']].round(2).to_string(index=False))

# %%
fig, axs=plt.subplots(1,2, figsize=(12.0,4.3))
plot_metrics=metrics.sort_values('RMSE_cycles',ascending=False)
colors=['#7E9FC0' if 'baseline' in m.lower() else '#3179AA' for m in plot_metrics.model]
for ax, metric, ttl in [(axs[0],'MAE_cycles','Mean Absolute Error'),(axs[1],'RMSE_cycles','Root Mean Squared Error')]:
    vals=plot_metrics[metric].to_numpy()
    ax.barh(plot_metrics.model,vals,color=colors,edgecolor='white')
    ax.set(xlabel='Error (remaining cycles)', title=ttl)
    for y,x in enumerate(vals): ax.text(x+.8,y,f'{x:.1f}',va='center',fontsize=9)
    ax.set_xlim(0,max(vals)*1.20)
fig.suptitle('NASA FD001: 20 held-out engine snapshots')
fig.tight_layout()
fig.savefig(FIG/'01_validation_MAE_RMSE_comparison.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# %%
selected=['Linear Regression','Random Forest','Neural Network']
fig,axs=plt.subplots(1,3,figsize=(13,4.3),sharex=True,sharey=True)
for ax,name in zip(axs,selected):
    pred=predictions[f'pred_{name.replace(" ", "_")}'].to_numpy()
    ax.scatter(y_val,pred,alpha=.83,color='#277DA8',edgecolor='white',linewidth=.35,s=48)
    lim=max(y_val.max(),max(pred)) *1.05
    ax.plot([0,lim],[0,lim],'--',color='#C65E42',linewidth=1.3,label='Ideal')
    ax.set(xlim=(0,lim),ylim=(0,lim),title=name, xlabel='Actual RUL (cycles)')
axs[0].set_ylabel('Predicted RUL (cycles)')
fig.suptitle('Predicted versus actual RUL — same 20 validation engines')
fig.tight_layout()
fig.savefig(FIG/'02_validation_actual_vs_predicted.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# %%
fig,ax=plt.subplots(figsize=(9.5,4.8))
for name,colour in [('Cycle-only baseline','#8D8D8D'),('Linear Regression','#277DA8'),('Random Forest','#CE8A32'),('Neural Network','#3D8C75')]:
    residual = predictions[f'pred_{name.replace(" ", "_")}'].to_numpy()-y_val
    ax.scatter(y_val,residual,label=name,alpha=.80,s=42,color=colour)
ax.axhline(0,color='black',linewidth=1,linestyle='--')
ax.set(xlabel='Actual RUL (cycles)',ylabel='Predicted − actual RUL (cycles)',
       title='Validation residuals: positive = overestimated RUL')
ax.legend(ncol=2,fontsize=9)
fig.tight_layout()
fig.savefig(FIG/'03_validation_residuals.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# %%
rf = models['Random Forest']
importance = pd.Series(rf.feature_importances_,index=FEATURES).sort_values(ascending=True)
fig,ax=plt.subplots(figsize=(8.3,5.3))
ax.barh(importance.index,importance.values,color='#3179AA')
ax.set(xlabel='Random Forest impurity-based importance (relative)',
       title='Indicative feature importance — fitted 80-engine model')
fig.tight_layout()
fig.savefig(FIG/'04_random_forest_feature_importance.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('Most important Random Forest features (indicative only):')
print(importance.sort_values(ascending=False).head(8).round(4).to_string())

# %%
# Optional diagnostic: model predictions along the full history of ONE held-out engine.
# This is for visualization, NOT for selecting or comparing models.
example_engine=int(val.iloc[0].engine_id)
example=val_histories[val_histories.engine_id==example_engine].sort_values('cycle')
fig,ax=plt.subplots(figsize=(9.4,4.7))
ax.plot(example.cycle, example.RUL, color='black',linewidth=2,label='Actual RUL')
for name,color in [('Linear Regression','#277DA8'),('Random Forest','#D3942B'),('Neural Network','#37967B')]:
    p=predict_rul(models[name],example[PREDICTOR_SETS[name]])
    ax.plot(example.cycle,p,label=name,alpha=.82,linewidth=1.4,color=color)
ax.axvline(int(val.iloc[0].cycle),ls=':',color='#A33737',label='Prechosen validation snapshot')
ax.set(xlabel='Engine operating cycle',ylabel='RUL (cycles)',
       title=f'Diagnostic prediction trajectory — validation engine {example_engine}')
ax.legend(fontsize=8,ncol=2)
fig.tight_layout()
fig.savefig(FIG/'05_one_validation_engine_trajectory_DIAGNOSTIC.png',dpi=180,bbox_inches='tight')
plt.close(fig)

# %%
# Stage 3 model choice is provisional: only 20 snapshots are evaluated.
# No NASA official test data or RUL labels are used here.
primary_winner = str(metrics.iloc[0]['model'])
choice = {
 'dataset':'NASA C-MAPSS FD001',
 'development_training_engines':80,
 'validation_engines':20,
 'validation_scoring':'one artificially censored endpoint per validation engine',
 'ranking_metric':'lowest validation RMSE (cycles)',
 'provisional_winner':primary_winner,
 'all_predictions_clipped_below_zero':True,
 'features':FEATURES,
 'model_parameters':{
     'cycle_only_baseline':'ordinary least squares fit to cycle only',
     'linear_regression':'StandardScaler + ordinary least squares',
     'random_forest':'160 trees, max_depth=14, min_samples_leaf=5, max_features=0.8, random_state=42',
     'neural_network':'StandardScaler + MLP (48,24), ReLU, Adam, alpha=0.01, max_iter=500, early_stopping=True, random_state=42'},
 'important_limitations':['20 validation snapshots; uncertain model rankings',
       'FD001 is simulated and represents one operational condition and one fault mode',
       'same static features at each cycle; no explicit time-series memory',
       'MLP internal early stopping uses rows from development engines, not a separate group-held-out set',
       'validation censoring is synthetic, not identical to official test censoring'],
 'official_test_labels_read':False,
 'next_stage':'Stage 4: refit selected configurations on all 100 original training engines, then score 100 official NASA test endpoints once.'}
(RESULTS/'stage3_model_decision.json').write_text(json.dumps(choice,indent=2))
print('Provisional lowest-RMSE model:',primary_winner)
print('PASS: all project Stage 3 integrity checks. No official NASA test data/labels used.')
