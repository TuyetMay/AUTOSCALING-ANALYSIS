from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

Granularity = Literal["1m", "5m", "15m"]
Mode = Literal["evaluate", "future"]
Target = Literal["hits", "total_bytes"]

ARTIFACT_DIR = Path("artifacts/models/xgb")
DATA_DIR = Path("data/processed")

# -----------------------------
# Feature engineering (future-safe)
# -----------------------------
def _freq_from_granularity(g: Granularity) -> str:
    return {"1m": "1T", "5m": "5T", "15m": "15T"}[g]


def _make_time_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    # safe time features
    df = pd.DataFrame(index=dt_index)
    df["hour"] = dt_index.hour
    df["dow"] = dt_index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def _build_future_safe_features(
    df: pd.DataFrame,
    y_col: str,
    lags: Tuple[int, ...] = (1, 2, 3, 6, 12),
    rolls: Tuple[int, ...] = (3, 6, 12, 24),
) -> pd.DataFrame:
    """
    Build features that can also be generated for future predictions.
    Requires df indexed by datetime, with y_col present.
    """
    x = _make_time_features(df.index)

    y = df[y_col].astype(float)

    for k in lags:
        x[f"{y_col}_lag_{k}"] = y.shift(k)

    for w in rolls:
        x[f"{y_col}_roll_mean_{w}"] = y.rolling(w).mean()
        x[f"{y_col}_roll_std_{w}"] = y.rolling(w).std()

    # simple rate features
    x[f"{y_col}_diff_1"] = y.diff(1)
    x[f"{y_col}_pct_1"] = y.pct_change(1).replace([np.inf, -np.inf], np.nan)

    return x


def _load_ts(split: Literal["train", "test"], g: Granularity) -> pd.DataFrame:
    path = DATA_DIR / f"{split}_ts_{g}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing timeseries file: {path}")

    df = pd.read_csv(path)
    # try common datetime column names
    dt_col = None
    for c in ["datetime", "timestamp", "time", "ds"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        raise ValueError(f"Cannot find datetime column in {path}. Columns={list(df.columns)[:20]}")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)
    return df


@dataclass
class XGBBundle:
    granularity: Granularity
    models: Dict[Target, XGBRegressor]
    feature_cols: Dict[Target, list]
    y_cols: Dict[Target, str]


def _default_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )


def train_xgb_bundle(
    granularity: Granularity,
    force: bool = False,
) -> XGBBundle:
    """
    Train two models:
      - hits
      - total_bytes
    using only future-safe features.
    """
    out_dir = ARTIFACT_DIR / granularity
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "bundle.joblib"

    if bundle_path.exists() and not force:
        return joblib.load(bundle_path)

    df_train = _load_ts("train", granularity)

    # Ensure target columns exist
    y_map: Dict[Target, str] = {
        "hits": "hits",
        "total_bytes": "total_bytes",
    }
    for t, col in y_map.items():
        if col not in df_train.columns:
            raise ValueError(f"Missing column '{col}' in train_ts_{granularity}.csv")

    models: Dict[Target, XGBRegressor] = {}
    feature_cols: Dict[Target, list] = {}

    for target, y_col in y_map.items():
        feats = _build_future_safe_features(df_train, y_col=y_col)
        d = feats.copy()
        d["y"] = df_train[y_col].astype(float)

        d = d.dropna()
        X = d.drop(columns=["y"])
        y = d["y"]

        m = _default_model()
        m.fit(X, y)

        models[target] = m
        feature_cols[target] = list(X.columns)

    bundle = XGBBundle(
        granularity=granularity,
        models=models,
        feature_cols=feature_cols,
        y_cols=y_map,
    )
    joblib.dump(bundle, bundle_path)
    return bundle


def evaluate_on_split(
    granularity: Granularity,
    split: Literal["train", "test"],
    last_n: Optional[int] = None,
    force_train: bool = False,
) -> pd.DataFrame:
    """
    Backtest on existing split; returns dataframe with y_true/y_pred for both targets.
    """
    bundle = train_xgb_bundle(granularity, force=force_train)
    df = _load_ts(split, granularity)

    rows = []
    for target, y_col in bundle.y_cols.items():
        feats = _build_future_safe_features(df, y_col=y_col)
        d = feats.copy()
        d["y_true"] = df[y_col].astype(float)
        d = d.dropna()

        X = d[bundle.feature_cols[target]]
        pred = bundle.models[target].predict(X)

        out = pd.DataFrame(index=d.index)
        out[f"{target}_true"] = d["y_true"].values
        out[f"{target}_pred"] = pred
        rows.append(out)

    out_all = pd.concat(rows, axis=1)
    out_all = out_all.sort_index()

    if last_n is not None:
        out_all = out_all.tail(int(last_n))

    out_all = out_all.reset_index().rename(columns={out_all.index.name: "datetime"})
    return out_all


def forecast_future_recursive(
    granularity: Granularity,
    horizon_steps: int,
    last_n_context: int = 500,
    force_train: bool = False,
) -> pd.DataFrame:
    """
    True future forecast for BOTH hits and bytes using recursive strategy.

    - Uses TRAIN split as history context.
    - Generates next timestamps based on granularity.
    - For each step: build future-safe features from (history + predicted),
      then predict next value.
    """
    bundle = train_xgb_bundle(granularity, force=force_train)
    df_hist = _load_ts("train", granularity)

    freq = _freq_from_granularity(granularity)

    # context: keep last N for stable rolling/lag
    df_hist = df_hist.tail(max(int(last_n_context), 200))

    # work copies for recursive updates
    hist_hits = df_hist[bundle.y_cols["hits"]].astype(float).copy()
    hist_bytes = df_hist[bundle.y_cols["total_bytes"]].astype(float).copy()

    start_ts = df_hist.index.max()
    future_index = pd.date_range(start=start_ts + pd.tseries.frequencies.to_offset(freq),
                                 periods=int(horizon_steps), freq=freq)

    preds_hits = []
    preds_bytes = []

    # Build a temp DF that we append to
    df_tmp = df_hist[[bundle.y_cols["hits"], bundle.y_cols["total_bytes"]]].copy()

    for ts in future_index:
        # append placeholder row
        df_tmp.loc[ts, bundle.y_cols["hits"]] = np.nan
        df_tmp.loc[ts, bundle.y_cols["total_bytes"]] = np.nan

        # predict hits
        feats_hits = _build_future_safe_features(df_tmp, y_col=bundle.y_cols["hits"])
        xh = feats_hits.loc[[ts]].copy()
        xh = xh[bundle.feature_cols["hits"]]
        ph = float(bundle.models["hits"].predict(xh)[0])

        # set predicted hits in df_tmp (so future steps can use lags/rolls)
        df_tmp.loc[ts, bundle.y_cols["hits"]] = ph
        preds_hits.append(ph)

        # predict bytes
        feats_b = _build_future_safe_features(df_tmp, y_col=bundle.y_cols["total_bytes"])
        xb = feats_b.loc[[ts]].copy()
        xb = xb[bundle.feature_cols["total_bytes"]]
        pb = float(bundle.models["total_bytes"].predict(xb)[0])

        df_tmp.loc[ts, bundle.y_cols["total_bytes"]] = pb
        preds_bytes.append(pb)

    out = pd.DataFrame(
        {
            "datetime": future_index,
            "hits_pred": preds_hits,
            "total_bytes_pred": preds_bytes,
        }
    )
    return out
