"""Stage 1: Explore NASA C-MAPSS FD001 sensor data and engine lifetimes.

Prerequisite: python scripts/setup_dataset.py /path/to/CMAPSSData.zip
Usage: python stage1/01_Data_Exploration.py
Outputs: stage1/figures/
"""


# MMA3001 — Stage 1: NASA C-MAPSS FD001 Exploratory Data Analysis


# 1 — Imports and file loading

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.rul_utils import calculate_rul

# Locate the data relative to this script, independent of working directory.
BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
FIGURES = BASE / 'figures'
FIGURES.mkdir(exist_ok=True, parents=True)
assert (DATA / 'train_FD001.txt').exists(), f'Cannot find FD001 files in {DATA}'

cols = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
train = pd.read_csv(DATA / 'train_FD001.txt', sep=r'\s+', header=None, names=cols)
test = pd.read_csv(DATA / 'test_FD001.txt', sep=r'\s+', header=None, names=cols)
sensors = [f'sensor_{i}' for i in range(1, 22)]

print('Train shape:', train.shape, '| Test shape:', test.shape)
print('Unique engines (train/test):', train.engine_id.nunique(), '/', test.engine_id.nunique())
print(train.head().to_string(index=False))


# 2 — Data quality checks

def audit(df, name):
    """Report data-quality, engine-count and cycle-continuity diagnostics."""
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated(subset=['engine_id', 'cycle']).sum())
    counts = df.groupby('engine_id').size()
    consecutive = all(
        np.array_equal(g['cycle'].to_numpy(), np.arange(1, len(g) + 1))
        for _, g in df.groupby('engine_id', sort=False)
    )
    print(f'{name}: {len(df):,} rows; {df.engine_id.nunique()} engines; '
          f'{missing} missing cells; {duplicates} duplicate engine/cycles; '
          f'sequential cycles: {consecutive}')
    print(f'  Observed cycles/engine: min {counts.min()}, median {counts.median():.1f}, max {counts.max()}')
    return counts

train_lengths = audit(train, 'TRAIN')
test_lengths = audit(test, 'TEST')


# 3 — Engine life distribution

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hist(train_lengths, bins=np.arange(120, 381, 15), color='#2176ae', edgecolor='white', alpha=.9)
ax.axvline(train_lengths.median(), color='#c04b40', lw=2,
           label=f'Median = {train_lengths.median():.0f} cycles')
ax.set(title='FD001 training-engine lifetime distribution', xlabel='Cycles until failure', ylabel='Number of engines')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.18)
fig.tight_layout()
fig.savefig(FIGURES / '01_engine_lifetime_distribution.png', dpi=180)
plt.close(fig)
print(train_lengths.describe().to_string())


# 4 — Construct training RUL target

train['RUL'] = calculate_rul(train)
assert (train.groupby('engine_id')['RUL'].min() == 0).all()
print(train[['engine_id', 'cycle', 'RUL']].head(7).to_string(index=False))
print('RUL range:', int(train.RUL.min()), 'to', int(train.RUL.max()), 'cycles')


# 5 — Example degradation patterns

examples = [1, 35, 70]
show_sensors = ['sensor_4', 'sensor_7', 'sensor_11', 'sensor_12']
fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), sharex=False)
for ax, sensor in zip(axes.flat, show_sensors):
    mu, std = train[sensor].mean(), train[sensor].std()
    for engine in examples:
        g = train.loc[train.engine_id == engine]
        normalized_life = (g.cycle - 1) / (g.cycle.max() - 1)
        ax.plot(normalized_life, (g[sensor] - mu) / std, lw=1.2, alpha=.85,
                label=f'Engine {engine} ({g.cycle.max()} cycles)')
    ax.set(title=sensor.replace('_', ' ').title(), xlabel='Fraction of engine life (0 = start, 1 = failure)',
           ylabel='Sensor value (training z-score)')
    ax.grid(alpha=.18)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
fig.suptitle('Example engine sensor trajectories', y=1.07, fontsize=15)
fig.tight_layout()
fig.savefig(FIGURES / '02_example_sensor_trajectories.png', dpi=180, bbox_inches='tight')
plt.close(fig)


# 6 — Identify uninformative sensors

summary = pd.DataFrame({
    'unique_values': train[sensors].nunique(),
    'min': train[sensors].min(),
    'max': train[sensors].max(),
    'std': train[sensors].std(),
})
summary.index.name = 'sensor'
summary['constant'] = summary.unique_values == 1
summary['range'] = summary['max'] - summary['min']
constant_sensors = summary.index[summary.constant].tolist()
near_constant_sensors = summary.index[(summary.unique_values <= 2) & (~summary.constant)].tolist()
print('Constant sensors:', constant_sensors)
print('Near-constant sensors (2 distinct values):', near_constant_sensors)
print('Constant operating settings:', [c for c in ['setting_1','setting_2','setting_3'] if train[c].nunique() == 1])
print(summary.round(5).to_string())
summary.to_csv(FIGURES / 'sensor_variability_summary.csv')


# 7 — Sensor–RUL relationships

varying = [s for s in sensors if train[s].nunique() > 1]
pearson = train[varying].corrwith(train['RUL']).dropna().sort_values()
fig, ax = plt.subplots(figsize=(9.5, 6.2))
bar_colors = np.where(pearson.values >= 0, '#1b8793', '#c05d44')
ax.barh(pearson.index.str.replace('sensor_', 'S'), pearson.values, color=bar_colors)
ax.axvline(0, color='#333333', lw=.8)
ax.set(xlabel='Pearson correlation with training RUL', title='Association between individual sensors and RUL', xlim=(-.85,.85))
ax.grid(axis='x', alpha=.16)
fig.tight_layout()
fig.savefig(FIGURES / '03_sensor_RUL_correlations.png', dpi=180)
plt.close(fig)
print('Strongest absolute pooled linear associations:')
print(pearson.abs().sort_values(ascending=False).head(10).round(3).to_string())


# 8 — Correlation and redundancy between candidate sensors

candidates = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12',
              'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
mat = train[candidates + ['RUL']].corr()
fig, ax = plt.subplots(figsize=(9.4, 8.1))
im = ax.imshow(mat.to_numpy(), vmin=-1, vmax=1, cmap='RdBu_r')
labels = [x.replace('sensor_', 'S') for x in mat.columns]
ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha='right')
ax.set_yticks(range(len(labels)), labels=labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = mat.iloc[i,j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5,
                color='white' if abs(v) > .63 else '#262626')
ax.set_title('Correlation matrix: shortlisted sensor inputs and RUL')
fig.colorbar(im, ax=ax, label='Pearson correlation', shrink=.8)
fig.tight_layout()
fig.savefig(FIGURES / '04_candidate_sensor_correlation_matrix.png', dpi=180)
plt.close(fig)


# 9 — Stage 1 findings and Stage 2 decisions


# Reference
