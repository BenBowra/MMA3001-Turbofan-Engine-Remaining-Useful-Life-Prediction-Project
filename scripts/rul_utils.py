"""Import-safe RUL data preparation and prediction helpers for NASA C-MAPSS FD001.

Remaining Useful Life (RUL) is measured in operating cycles. Compute it only
for COMPLETE training histories, not truncated official test histories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def calculate_rul(history: pd.DataFrame) -> pd.Series:
    """Calculate uncapped RUL from each complete training engine's failure cycle.

    Requires engine_id and positive, unique cycle numbers. The result retains
    the input index and is zero at each engine's final (failure) cycle.
    Never apply to truncated test histories.
    """
    if not {'engine_id', 'cycle'}.issubset(history.columns):
        raise ValueError('history must have engine_id and cycle columns')
    if history[['engine_id', 'cycle']].isna().any().any():
        raise ValueError('engine_id and cycle cannot contain missing values')
    if (history['cycle'] < 1).any() or history.duplicated(['engine_id', 'cycle']).any():
        raise ValueError('cycle values must be positive and unique per engine')
    # Only a complete training history establishes the true failure cycle.
    return history.groupby('engine_id')['cycle'].transform('max') - history['cycle']

def split_engine_indices(history: pd.DataFrame, *, test_size: float = 0.2,
                         random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Make a reproducible grouped split; no engine may appear in both sets."""
    if 'engine_id' not in history or history.engine_id.isna().any() or history.engine_id.nunique() < 2:
        raise ValueError('at least two nonmissing engine IDs are required')
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(history, groups=history.engine_id))
    assert set(history.iloc[train_idx].engine_id).isdisjoint(set(history.iloc[val_idx].engine_id))
    return train_idx, val_idx

def select_sensor_columns(development: pd.DataFrame, *, max_unique: int = 2,
                          sensor_prefix: str = 'sensor_') -> tuple[list[str], list[str]]:
    """Select sensors using development data ONLY; return (selected, excluded).

    Exclude channels having at most max_unique distinct training-fold values.
    The selected list is frozen for validation and official test predictions.
    """
    if max_unique < 1:
        raise ValueError('max_unique must be at least 1')
    sensors = [c for c in development if c.startswith(sensor_prefix)]
    if not sensors:
        raise ValueError('no sensor columns found')
    counts = development[sensors].nunique(dropna=True)
    excluded = [s for s in sensors if counts[s] <= max_unique]
    return [s for s in sensors if s not in excluded], excluded

def last_observation_per_engine(history: pd.DataFrame) -> pd.DataFrame:
    """Return the latest observed engine cycles without accessing any RUL labels."""
    if not {'engine_id', 'cycle'}.issubset(history.columns):
        raise ValueError('history must have engine_id and cycle columns')
    if history.duplicated(['engine_id', 'cycle']).any():
        raise ValueError('duplicate engine/cycle observations')
    return (history.sort_values(['engine_id', 'cycle'])
            .groupby('engine_id', as_index=False).tail(1)
            .sort_values('engine_id').reset_index(drop=True))

def predict_nonnegative_rul(model, features) -> np.ndarray:
    """Predict RUL and clip negative forecasts to zero, rejecting non-finite values."""
    raw = np.asarray(model.predict(features), dtype=float).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError('model produced non-finite RUL predictions')
    return np.maximum(0.0, raw)
