# %% [markdown]
# # MMA3001 · Stage 4: NASA C-MAPSS FD001 final test evaluation
# This script is self-contained within the Stage4 package. Stage 3 model
# choices/features are frozen before accessing official test RUL labels.

# %%
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

ROOT = Path.cwd()
if not (ROOT/'data'/'train_FD001.txt').is_file():
    ROOT = Path(__file__).resolve().parent
DATA, RESULTS, FIG, MODELS = (ROOT/item for item in ('data','results','figures','models'))
for d in (DATA, RESULTS, FIG, MODELS): d.mkdir(exist_ok=True)
plt.rcParams.update({'axes.grid':True,'grid.alpha':.15,'figure.dpi':110})
print('Stage 4 root:',ROOT)

# %% [markdown]
# ## 1. Read the frozen Stage 3 decision and load the raw NASA FD001 data
# The NASA labels will be read *after* all model definitions/parameters are frozen.

# %%
stage3 = json.loads((DATA/'stage3_model_decision.json').read_text())
FEATURES = stage3['features']
assert stage3['provisional_winner'] == 'Neural Network'
assert stage3['official_test_labels_read'] is False
assert FEATURES == ['cycle'] + [f'sensor_{j}' for j in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
COLS = ['engine_id','cycle'] + [f'operating_setting_{j}' for j in range(1,4)] + [f'sensor_{j}' for j in range(1,22)]
train = pd.read_csv(DATA/'train_FD001.txt', sep=r'\s+', header=None, names=COLS)
test = pd.read_csv(DATA/'test_FD001.txt', sep=r'\s+', header=None, names=COLS)
assert train.shape==(20631,26) and test.shape==(13096,26)
assert train.engine_id.nunique()==100 and test.engine_id.nunique()==100
assert not train.isna().any().any() and not test.isna().any().any()
assert not train.duplicated(subset=['engine_id','cycle']).any()
assert not test.duplicated(subset=['engine_id','cycle']).any()
assert (train.groupby('engine_id').cycle.min()==1).all()
assert (test.groupby('engine_id').cycle.min()==1).all()
train['RUL'] = train.groupby('engine_id')['cycle'].transform('max')-train['cycle']
assert train.groupby('engine_id').RUL.min().eq(0).all()
# Latest available test observation for each engine; no future measurements used.
endpoint = (test.sort_values(['engine_id','cycle'])
            .groupby('engine_id',as_index=False).tail(1)
            .sort_values('engine_id').reset_index(drop=True))
assert len(endpoint)==100 and np.array_equal(endpoint.engine_id.values,np.arange(1,101))
assert not endpoint[FEATURES].isna().any().any()
print('Training: 20,631 cycles, 100 complete engine histories')
print('Official test: 100 last-observed engine snapshots (from 13,096 rows)')
print('Feature count:',len(FEATURES),' | Fixed from Stage 3')

# %% [markdown]
# ## 2. Refit the exact Stage 3 model configurations on all 100 training engines
# Includes a mean baseline and a cycle-only baseline. All scalers fit on the
# training engines; no official NASA test labels are used to fit anything.

# %%
models = {
    'Mean baseline': DummyRegressor(strategy='mean'),
    'Cycle-only baseline': LinearRegression(),
    'Linear Regression': Pipeline([
        ('scaler',StandardScaler()),('regressor',LinearRegression())]),
    'Random Forest': RandomForestRegressor(
        n_estimators=160,max_depth=14,min_samples_leaf=5,
        max_features=.8,random_state=42,n_jobs=-1),
    'Neural Network': Pipeline([
        ('scaler',StandardScaler()),
        ('regressor',MLPRegressor(
            hidden_layer_sizes=(48,24),activation='relu',solver='adam',
            alpha=.01,batch_size=256,learning_rate_init=.001,
            max_iter=500,early_stopping=True,validation_fraction=.15,
            n_iter_no_change=20,tol=.0008,random_state=42,shuffle=True))]),
}
PREDICTORS = {name:(['cycle'] if name=='Cycle-only baseline' else FEATURES)
              for name in models}
X_train, y_train = train[FEATURES], train.RUL.to_numpy(dtype=float)
X_test = endpoint[FEATURES]
predictions = pd.DataFrame({'engine_id':endpoint.engine_id.astype(int),
                             'last_observed_cycle':endpoint.cycle.astype(int)})
fit_info=[]
for name,model in models.items():
    t0=time.perf_counter()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(train[PREDICTORS[name]],y_train)
    fit_secs = time.perf_counter()-t0
    p = np.maximum(0,model.predict(endpoint[PREDICTORS[name]]).astype(float))
    assert len(p)==100 and np.isfinite(p).all()
    col='pred_'+name.lower().replace(' ','_').replace('-','_')
    predictions[col]=p
    joblib.dump(model,MODELS/(name.lower().replace(' ','_').replace('-','_')+'.joblib'))
    fit_info.append({'model':name,'refit_time_seconds':fit_secs,
                     'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in ws)})
    print(name,': fitted in',round(fit_secs,2),'sec, test predictions generated')
assert not any(x['convergence_warning'] for x in fit_info), 'MLP did not converge cleanly'

# %% [markdown]
# ## 3. One-time official NASA test label evaluation
# Model settings were frozen in Stage 3. Do not use these test results to tune
# parameters or select features: that would invalidate the held-out evaluation.

# %%
y_test=pd.read_csv(DATA/'RUL_FD001.txt',sep=r'\s+',header=None).iloc[:,0].to_numpy(float)
assert len(y_test)==100 and np.all(np.isfinite(y_test)) and np.all(y_test>=0)
predictions['actual_RUL']=y_test
records=[]
for name in models:
    col='pred_'+name.lower().replace(' ','_').replace('-','_')
    p=predictions[col].to_numpy()
    err=p-y_test
    records.append({'model':name,'MAE_cycles':mean_absolute_error(y_test,p),
                    'RMSE_cycles':np.sqrt(mean_squared_error(y_test,p)),
                    'R2':r2_score(y_test,p),'mean_signed_error_cycles':err.mean(),
                    'overestimation_count':int((err>0).sum()),
                    'underestimation_count':int((err<0).sum()),
                    'overestimate_20plus_count':int((err>20).sum()),
                    'underestimate_20plus_count':int((err<-20).sum()),
                    'max_absolute_error_cycles':np.abs(err).max(),
                    'n_test_engines':len(y_test)})
results=pd.DataFrame(records).merge(pd.DataFrame(fit_info),on='model').sort_values('RMSE_cycles').reset_index(drop=True)
assert results.n_test_engines.eq(100).all()
print('\nOFFICIAL NASA TEST (100 engine endpoints):')
print(results[['model','MAE_cycles','RMSE_cycles','R2','mean_signed_error_cycles']].round(3).to_string(index=False))
print('\nSelected before test evaluation:',stage3['provisional_winner'])
print('Best TEST RMSE (descriptive only):',results.iloc[0]['model'])

# %% [markdown]
# ## 4. Bootstrap descriptive uncertainty + paired comparisons
# 95% bootstrap intervals show test-sample variation, not real-world safety
# guarantees. Paired bootstrap comparisons use the same resampled engines.

# %%
rng=np.random.default_rng(202604)
idx=rng.integers(0,100,size=(5000,100))
bootstrap_mae={}
for name in models:
    col='pred_'+name.lower().replace(' ','_').replace('-','_')
    error=np.abs(predictions[col].to_numpy()-y_test)
    bootstrap_mae[name]=error[idx].mean(axis=1)
ci=pd.DataFrame([{'model':name,
                  'MAE_95_bootstrap_low_cycles':np.quantile(vals,.025),
                  'MAE_95_bootstrap_high_cycles':np.quantile(vals,.975)}
                 for name,vals in bootstrap_mae.items()])
results=results.merge(ci,on='model')
selected_name=stage3['provisional_winner']
pairwise=[]
for other in ('Cycle-only baseline','Linear Regression','Random Forest'):
    diff=bootstrap_mae[other]-bootstrap_mae[selected_name]  # positive favours prechosen NN
    pairwise.append({'comparison':f'{selected_name} vs {other}',
                     'MAE_advantage_NN_cycles':float(diff.mean()),
                     '95_bootstrap_low_cycles':float(np.quantile(diff,.025)),
                     '95_bootstrap_high_cycles':float(np.quantile(diff,.975))})
paired=pd.DataFrame(pairwise)
print('\nPaired descriptive 95% bootstrap intervals for MAE difference (positive = NN better):')
print(paired.round(2).to_string(index=False))

# %% [markdown]
# ## 5. Save machine-readable results and make figures

# %%
results.to_csv(RESULTS/'official_test_metrics.csv',index=False)
predictions.to_csv(RESULTS/'official_test_predictions_by_engine.csv',index=False)
paired.to_csv(RESULTS/'paired_MAE_comparisons.csv',index=False)
with (RESULTS/'evaluation_protocol.json').open('w') as f:
    json.dump({'data':'NASA FD001','stage3_selected_model':selected_name,
       'frozen_features':FEATURES,'training_engines':100,'train_rows':len(train),
       'test_engines':100,'test_scoring':'one last-observed cycle per engine',
       'target':'uncapped RUL cycles','negatives_clipped_to_zero':True,
       'data_usage':'all 100 training engine histories used to fit; NASA official test RUL opened after predictions produced',
       'selection_warning':'DO NOT revise hyperparameters or select features from these official test results',
       'bootstraps':5000,'seed':202604},f,indent=2)

colors={'Mean baseline':'#969696','Cycle-only baseline':'#7799B6',
        'Linear Regression':'#2677A9','Random Forest':'#C48A2D','Neural Network':'#368877'}
ordered=list(models)
fig,axs=plt.subplots(1,2,figsize=(12.4,4.8))
for ax,metric,title in zip(axs,['MAE_cycles','RMSE_cycles'],['Mean absolute error','Root mean squared error']):
    vals=results.set_index('model').loc[ordered,metric]
    bars=ax.barh(ordered,vals,color=[colors[n] for n in ordered])
    ax.invert_yaxis()
    ax.set(xlabel='Error (cycles)',title=title,xlim=(0,vals.max()*1.24))
    for bar,v in zip(bars,vals): ax.text(v+0.5,bar.get_y()+bar.get_height()/2,f'{v:.1f}',va='center',fontsize=9)
fig.suptitle('NASA FD001: official independent test — 100 engine endpoints')
fig.tight_layout()
fig.savefig(FIG/'01_official_test_MAE_RMSE.png',dpi=180,bbox_inches='tight')
plt.show()

# %%
chosen=['Linear Regression','Random Forest','Neural Network']
fig,axs=plt.subplots(1,3,figsize=(13.3,4.5),sharex=True,sharey=True)
maxv=max(y_test.max(),max(predictions['pred_'+n.lower().replace(' ','_')].max() for n in chosen))*1.04
for ax,name in zip(axs,chosen):
    p=predictions['pred_'+name.lower().replace(' ','_')]
    ax.scatter(y_test,p,color=colors[name],s=33,alpha=.76,edgecolor='white',linewidth=.35)
    ax.plot([0,maxv],[0,maxv],ls='--',color='#B34432',lw=1)
    ax.set(xlim=(0,maxv),ylim=(0,maxv),title=name,xlabel='Actual RUL (cycles)')
axs[0].set_ylabel('Predicted RUL (cycles)')
fig.suptitle('NASA FD001: official test predictions versus actual RUL (n=100)')
fig.tight_layout()
fig.savefig(FIG/'02_official_test_actual_vs_predicted.png',dpi=180,bbox_inches='tight')
plt.show()

# %%
fig,ax=plt.subplots(figsize=(10,5))
for name in ['Cycle-only baseline']+chosen:
    p=predictions['pred_'+name.lower().replace(' ','_').replace('-','_')]
    ax.scatter(y_test,p-y_test,s=35,alpha=.67,color=colors[name],label=name)
ax.axhline(0,ls='--',c='black',lw=1)
ax.set(xlabel='Actual RUL (cycles)',ylabel='Predicted − actual RUL (cycles)',
       title='Official test signed errors: positive = overestimated engine life')
ax.legend(ncol=2,fontsize=9)
fig.tight_layout()
fig.savefig(FIG/'03_official_test_signed_errors.png',dpi=180,bbox_inches='tight')
plt.show()

# %%
# Display MAE by actual-life group, a descriptive diagnostic only (no retuning).
bins=[-0.01,30,60,90,np.inf]
labels=['0–30','31–60','61–90','>90']
parts=pd.cut(y_test,bins=bins,labels=labels)
rows=[]
for group in labels:
    mask=np.array(parts==group)
    for name in chosen:
        col='pred_'+name.lower().replace(' ','_')
        rows.append({'RUL_band':group,'model':name,'engine_count':int(mask.sum()),
                     'MAE_cycles':float(np.abs(predictions[col].to_numpy()[mask]-y_test[mask]).mean()) if mask.sum() else np.nan})
banded=pd.DataFrame(rows)
banded.to_csv(RESULTS/'test_MAE_by_actual_RUL_band.csv',index=False)
fig,ax=plt.subplots(figsize=(9.6,4.8))
x=np.arange(len(labels)); w=.25
for j,name in enumerate(chosen):
    df=banded[banded.model==name]
    ax.bar(x+(j-1)*w,df.MAE_cycles,width=w,color=colors[name],label=name)
ax.set_xticks(x,labels)
counts=banded[banded.model==chosen[0]].engine_count.values
ax.set(xlabel='Actual RUL group (cycles)',ylabel='MAE within group (cycles)',
       title='Official test: errors differ by how close engines are to failure')
for i,count in enumerate(counts): ax.text(i,ax.get_ylim()[1]*.94,f'n={count}',ha='center',fontsize=9)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG/'04_test_MAE_by_RUL_band.png',dpi=180,bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Final checks and engineering interpretation
# This is a simulation-based predictive maintenance benchmark. No claim is made
# that these errors meet aircraft maintenance or certification requirements.

# %%
assert len(predictions)==100 and predictions.engine_id.is_unique
assert predictions.actual_RUL.notna().all()
assert np.all(np.isfinite(predictions.select_dtypes('number').values))
assert (predictions[[c for c in predictions if c.startswith('pred_')]]>=0).all().all()
print('\nFINAL INTEGRITY CHECKS PASSED')
print('Official NASA test labels only evaluated after final model fitting.')
print('Stage 3 selected model:',selected_name)
print('Official test ranking shown descriptively: not used to retune models.')
