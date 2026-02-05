"""XGBoost forecaster utilities (hits + bytes).

This module trains (if needed) and loads models for 1m/5m/15m forecasts.
It expects the processed time-series CSVs created by src/data/build_timeseries.py:
  data/processed/{split}_ts_{1m|5m|15m}.csv

Targets:
  - hits
  - total_bytes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Optional, Dict

import pandas as pd
import numpy as np
from joblib import dump, load

try:
    from xgboost import XGBRegressor
except Exception as e:  # pragma: no cover
    XGBRegressor = None  # type: ignore


Granularity = Literal["1m", "5m", "15m"]
Split = Literal["train", "test"]


@dataclass(frozen=True)
class ModelPaths:
    hits: Path
    bytes: Path


def _project_root() -> Path:
    # .../src/models/xgb.py -> project root
    return Path(__file__).resolve().parents[2]


def _processed_ts_path(split: Split, granularity: Granularity) -> Path:
    return _project_root() / "data" / "processed" / f"{split}_ts_{granularity}.csv"


def _model_dir(granularity: Granularity) -> Path:
    return _project_root() / "artifacts" / "models" / "xgb" / granularity


def model_paths(granularity: Granularity) -> ModelPaths:
    mdir = _model_dir(granularity)
    return ModelPaths(
        hits=mdir / "xgb_hits.joblib",
        bytes=mdir / "xgb_bytes.joblib",
    )


def load_timeseries(split: Split, granularity: Granularity) -> pd.DataFrame:
    path = _processed_ts_path(split, granularity)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed time-series file: {path}. " 
            "Run preprocessing first: python main.py (or python main.py --skip-db)"
        )

    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError(f"Expected 'datetime' column in {path}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def make_supervised(
    df: pd.DataFrame,
    horizon_steps: int = 1,
    target_hits: str = "hits",
    target_bytes: str = "total_bytes",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create supervised dataset using a forward shift.

    Returns:
      X, y_hits, y_bytes, dt_index
    """
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be >= 1")

    if target_hits not in df.columns or target_bytes not in df.columns:
        raise ValueError("DataFrame must contain 'hits' and 'total_bytes'")

    df2 = df.copy()

    # targets = future values
    df2["y_hits"] = df2[target_hits].shift(-horizon_steps)
    df2["y_bytes"] = df2[target_bytes].shift(-horizon_steps)

    # feature columns: everything except targets + datetime + future labels
    drop_cols = {"datetime", "y_hits", "y_bytes"}
    X = df2.drop(columns=[c for c in drop_cols if c in df2.columns])

    # Also drop the original targets from features to avoid trivial leakage.
    # (You can keep lags/rolling features that already encode history.)
    for leak_col in [target_hits, target_bytes]:
        if leak_col in X.columns:
            X = X.drop(columns=[leak_col])

    # remove rows where shifted target is NaN
    mask = (~df2["y_hits"].isna()) & (~df2["y_bytes"].isna())
    X = X.loc[mask].reset_index(drop=True)
    y_hits = df2.loc[mask, "y_hits"].reset_index(drop=True)
    y_bytes = df2.loc[mask, "y_bytes"].reset_index(drop=True)
    dt = df2.loc[mask, "datetime"].reset_index(drop=True)

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y_hits, y_bytes, dt


def _default_xgb_params(seed: int = 42) -> Dict:
    return dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=0,
    )


def train_and_save(
    granularity: Granularity,
    horizon_steps: int = 1,
    seed: int = 42,
    force: bool = False,
) -> ModelPaths:
    """Train XGBoost models on TRAIN split and save to artifacts."""
    if XGBRegressor is None:
        raise RuntimeError(
            "xgboost is not installed. Install dependencies: pip install -r requirements.txt"
        )

    paths = model_paths(granularity)
    paths.hits.parent.mkdir(parents=True, exist_ok=True)

    if (not force) and paths.hits.exists() and paths.bytes.exists():
        return paths

    df_train = load_timeseries("train", granularity)
    Xtr, y_hits, y_bytes, _ = make_supervised(df_train, horizon_steps=horizon_steps)

    params = _default_xgb_params(seed=seed)

    m_hits = XGBRegressor(**params)
    m_bytes = XGBRegressor(**params)

    m_hits.fit(Xtr, y_hits)
    m_bytes.fit(Xtr, y_bytes)

    dump(m_hits, paths.hits)
    dump(m_bytes, paths.bytes)

    return paths


def load_or_train(
    granularity: Granularity,
    horizon_steps: int = 1,
    force_train: bool = False,
):
    paths = model_paths(granularity)
    if force_train or (not paths.hits.exists()) or (not paths.bytes.exists()):
        train_and_save(granularity, horizon_steps=horizon_steps, force=True)

    m_hits = load(paths.hits)
    m_bytes = load(paths.bytes)
    return m_hits, m_bytes


def predict_on_split(
    granularity: Granularity,
    split: Split = "test",
    horizon_steps: int = 1,
    force_train: bool = False,
) -> pd.DataFrame:
    """Predict for the chosen split (default: test) and return a tidy DataFrame."""
    m_hits, m_bytes = load_or_train(granularity, horizon_steps=horizon_steps, force_train=force_train)
    df = load_timeseries(split, granularity)
    X, y_hits, y_bytes, dt = make_supervised(df, horizon_steps=horizon_steps)

    pred_hits = np.maximum(0.0, m_hits.predict(X))
    pred_bytes = np.maximum(0.0, m_bytes.predict(X))

    out = pd.DataFrame(
        {
            "datetime": dt,
            "granularity": granularity,
            "horizon_steps": horizon_steps,
            "y_hits_true": y_hits.astype(float),
            "y_hits_pred": pred_hits.astype(float),
            "y_bytes_true": y_bytes.astype(float),
            "y_bytes_pred": pred_bytes.astype(float),
        }
    )
    return out
