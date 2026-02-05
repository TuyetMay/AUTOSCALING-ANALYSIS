"""
Build time series features from parsed log data
Creates 1-minute, 5-minute, and 15-minute aggregated features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


def load_parsed_data(csv_path: str) -> pd.DataFrame:
    """Load parsed CSV and prepare for time series aggregation"""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df


def build_1m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 1-minute time series features"""
    df_1m = df.resample('1T').agg(
        hits=('request', 'count'),
        total_bytes=('bytes', 'sum'),
        
        unique_hosts=('host', 'nunique'),
        unique_urls=('url', 'nunique'),
        unique_methods=('method', 'nunique'),
        unique_versions=('version', 'nunique'),
        
        min_bytes=('bytes', 'min'),
        max_bytes=('bytes', 'max'),
        mean_bytes=('bytes', 'mean'),
        std_bytes=('bytes', 'std'),
        median_bytes=('bytes', 'median'),
        
        count_success=('status', lambda x: (x == 200).sum()),
        count_redirect_301=('status', lambda x: (x == 301).sum()),
        count_redirect_302=('status', lambda x: (x == 302).sum()),
        count_cache_304=('status', lambda x: (x == 304).sum()),
        count_client_err=('status', lambda x: ((x >= 400) & (x < 500)).sum()),
        count_server_err=('status', lambda x: (x >= 500).sum()),
    )
    
    df_1m = df_1m.fillna(0)
    denom = df_1m['hits'].replace(0, 1)
    
    # Lag features (1m step)
    df_1m['hits_lag_1'] = df_1m['hits'].shift(1)
    df_1m['hits_lag_5'] = df_1m['hits'].shift(5)
    df_1m['hits_lag_10'] = df_1m['hits'].shift(10)
    
    df_1m['bytes_lag_1'] = df_1m['total_bytes'].shift(1)
    df_1m['bytes_lag_5'] = df_1m['total_bytes'].shift(5)
    df_1m['bytes_lag_10'] = df_1m['total_bytes'].shift(10)
    
    # Rolling features
    df_1m['hits_roll_mean_5'] = df_1m['hits'].rolling(5).mean()
    df_1m['hits_roll_std_5'] = df_1m['hits'].rolling(5).std()
    df_1m['hits_roll_mean_10'] = df_1m['hits'].rolling(10).mean()
    
    df_1m['bytes_roll_mean_5'] = df_1m['total_bytes'].rolling(5).mean()
    df_1m['bytes_roll_mean_10'] = df_1m['total_bytes'].rolling(10).mean()
    
    # Status rates
    df_1m['success_rate'] = df_1m['count_success'] / denom
    df_1m['cache_304_rate'] = df_1m['count_cache_304'] / denom
    df_1m['redirect_rate'] = (df_1m['count_redirect_301'] + df_1m['count_redirect_302']) / denom
    df_1m['error_rate'] = (df_1m['count_client_err'] + df_1m['count_server_err']) / denom
    
    # Rate lags
    df_1m['error_rate_lag_1'] = df_1m['error_rate'].shift(1)
    df_1m['redirect_rate_lag_1'] = df_1m['redirect_rate'].shift(1)
    df_1m['cache_rate_lag_1'] = df_1m['cache_304_rate'].shift(1)
    
    # Time features
    df_1m['hour'] = df_1m.index.hour
    df_1m['day_of_week'] = df_1m.index.dayofweek
    df_1m['is_weekend'] = (df_1m['day_of_week'] >= 5).astype(int)
    
    df_1m['sin_hour'] = np.sin(2 * np.pi * df_1m['hour'] / 24)
    df_1m['cos_hour'] = np.cos(2 * np.pi * df_1m['hour'] / 24)
    df_1m['sin_dow'] = np.sin(2 * np.pi * df_1m['day_of_week'] / 7)
    df_1m['cos_dow'] = np.cos(2 * np.pi * df_1m['day_of_week'] / 7)
    
    # Momentum features
    df_1m['hits_diff_1'] = df_1m['hits'].diff(1)
    df_1m['bytes_diff_1'] = df_1m['total_bytes'].diff(1)
    
    df_1m['hits_pct_change_1'] = (
        df_1m['hits'].pct_change(1)
    ).replace([np.inf, -np.inf], 0).clip(-5, 5)
    
    df_1m['bytes_pct_change_1'] = (
        df_1m['total_bytes'].pct_change(1)
    ).replace([np.inf, -np.inf], 0).clip(-5, 5)
    
    # Composition
    df_1m['bytes_per_hit'] = df_1m['total_bytes'] / denom
    
    # Gap features
    df_1m['is_gap'] = (df_1m['hits'] == 0).astype(int)
    df_1m['time_gap_sec'] = df_1m.index.to_series().diff().dt.total_seconds()
    
    return df_1m


def build_5m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 5-minute time series features"""
    df_5m = df.resample('5T').agg(
        hits=('request', 'count'),
        total_bytes=('bytes', 'sum'),
        
        unique_hosts=('host', 'nunique'),
        unique_urls=('url', 'nunique'),
        unique_methods=('method', 'nunique'),
        unique_versions=('version', 'nunique'),
        
        min_bytes=('bytes', 'min'),
        max_bytes=('bytes', 'max'),
        mean_bytes=('bytes', 'mean'),
        std_bytes=('bytes', 'std'),
        median_bytes=('bytes', 'median'),
        
        count_success=('status', lambda x: (x == 200).sum()),
        count_redirect_301=('status', lambda x: (x == 301).sum()),
        count_redirect_302=('status', lambda x: (x == 302).sum()),
        count_cache_304=('status', lambda x: (x == 304).sum()),
        count_client_err=('status', lambda x: ((x >= 400) & (x < 500)).sum()),
        count_server_err=('status', lambda x: (x >= 500).sum()),
    )
    
    df_5m = df_5m.fillna(0)
    denom = df_5m['hits'].replace(0, 1)
    
    # Lag features (5m step)
    df_5m['hits_lag_1'] = df_5m['hits'].shift(1)    # 5 min
    df_5m['hits_lag_3'] = df_5m['hits'].shift(3)    # 15 min
    df_5m['hits_lag_6'] = df_5m['hits'].shift(6)    # 30 min
    
    df_5m['bytes_lag_1'] = df_5m['total_bytes'].shift(1)
    df_5m['bytes_lag_3'] = df_5m['total_bytes'].shift(3)
    df_5m['bytes_lag_6'] = df_5m['total_bytes'].shift(6)
    
    # Rolling features
    df_5m['hits_roll_mean_3'] = df_5m['hits'].rolling(3).mean()    # 15 min
    df_5m['hits_roll_std_3'] = df_5m['hits'].rolling(3).std()
    df_5m['hits_roll_mean_6'] = df_5m['hits'].rolling(6).mean()    # 30 min
    
    df_5m['bytes_roll_mean_3'] = df_5m['total_bytes'].rolling(3).mean()
    df_5m['bytes_roll_mean_6'] = df_5m['total_bytes'].rolling(6).mean()
    
    # Status rates
    df_5m['success_rate'] = df_5m['count_success'] / denom
    df_5m['redirect_301_rate'] = df_5m['count_redirect_301'] / denom
    df_5m['redirect_302_rate'] = df_5m['count_redirect_302'] / denom
    df_5m['cache_304_rate'] = df_5m['count_cache_304'] / denom
    df_5m['client_error_rate'] = df_5m['count_client_err'] / denom
    df_5m['server_error_rate'] = df_5m['count_server_err'] / denom
    
    df_5m['redirect_rate'] = (
        df_5m['count_redirect_301'] + df_5m['count_redirect_302']
    ) / denom
    
    df_5m['error_rate'] = (
        df_5m['count_client_err'] + df_5m['count_server_err']
    ) / denom
    
    # Rate lags
    df_5m['error_rate_lag_1'] = df_5m['error_rate'].shift(1)
    df_5m['redirect_rate_lag_1'] = df_5m['redirect_rate'].shift(1)
    df_5m['cache_rate_lag_1'] = df_5m['cache_304_rate'].shift(1)
    
    # Time features
    df_5m['hour'] = df_5m.index.hour
    df_5m['day_of_week'] = df_5m.index.dayofweek
    df_5m['is_weekend'] = (df_5m['day_of_week'] >= 5).astype(int)
    
    df_5m['sin_hour'] = np.sin(2 * np.pi * df_5m['hour'] / 24)
    df_5m['cos_hour'] = np.cos(2 * np.pi * df_5m['hour'] / 24)
    df_5m['sin_dow'] = np.sin(2 * np.pi * df_5m['day_of_week'] / 7)
    df_5m['cos_dow'] = np.cos(2 * np.pi * df_5m['day_of_week'] / 7)
    
    # Momentum features
    df_5m['hits_diff_1'] = df_5m['hits'].diff(1)
    df_5m['bytes_diff_1'] = df_5m['total_bytes'].diff(1)
    
    df_5m['hits_pct_change_1'] = (
        df_5m['hits'].pct_change(1)
    ).replace([np.inf, -np.inf], 0).clip(-5, 5)
    
    df_5m['bytes_pct_change_1'] = (
        df_5m['total_bytes'].pct_change(1)
    ).replace([np.inf, -np.inf], 0).clip(-5, 5)
    
    # Composition
    df_5m['bytes_per_hit'] = df_5m['total_bytes'] / denom
    
    # Gap features
    df_5m['is_gap'] = (df_5m['hits'] == 0).astype(int)
    df_5m['time_gap_sec'] = df_5m.index.to_series().diff().dt.total_seconds()
    
    return df_5m


def build_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 15-minute time series features"""
    df_15m = df.resample('15T').agg(
        hits=('request', 'count'),
        total_bytes=('bytes', 'sum'),
        
        unique_hosts=('host', 'nunique'),
        unique_urls=('url', 'nunique'),
        unique_methods=('method', 'nunique'),
        unique_versions=('version', 'nunique'),
        
        min_bytes=('bytes', 'min'),
        max_bytes=('bytes', 'max'),
        mean_bytes=('bytes', 'mean'),
        std_bytes=('bytes', 'std'),
        median_bytes=('bytes', 'median'),
        
        count_success=('status', lambda x: (x == 200).sum()),
        count_redirect_301=('status', lambda x: (x == 301).sum()),
        count_redirect_302=('status', lambda x: (x == 302).sum()),
        count_cache_304=('status', lambda x: (x == 304).sum()),
        count_client_err=('status', lambda x: ((x >= 400) & (x < 500)).sum()),
        count_server_err=('status', lambda x: (x >= 500).sum()),
    )
    
    df_15m = df_15m.fillna(0)
    denom = df_15m['hits'].replace(0, 1)
    
    # Lag features (15m step)
    df_15m['hits_lag_1'] = df_15m['hits'].shift(1)   # 15 min
    df_15m['hits_lag_2'] = df_15m['hits'].shift(2)   # 30 min
    df_15m['hits_lag_4'] = df_15m['hits'].shift(4)   # 60 min
    
    df_15m['bytes_lag_1'] = df_15m['total_bytes'].shift(1)
    df_15m['bytes_lag_2'] = df_15m['total_bytes'].shift(2)
    df_15m['bytes_lag_4'] = df_15m['total_bytes'].shift(4)
    
    # Rolling features
    df_15m['hits_roll_mean_2'] = df_15m['hits'].rolling(2).mean()  # 30 min
    df_15m['hits_roll_std_2'] = df_15m['hits'].rolling(2).std()
    df_15m['hits_roll_mean_4'] = df_15m['hits'].rolling(4).mean()  # 60 min
    df_15m['hits_roll_std_4'] = df_15m['hits'].rolling(4).std()
    
    df_15m['bytes_roll_mean_2'] = df_15m['total_bytes'].rolling(2).mean()
    df_15m['bytes_roll_mean_4'] = df_15m['total_bytes'].rolling(4).mean()
    df_15m['bytes_roll_std_2'] = df_15m['total_bytes'].rolling(2).std()
    df_15m['bytes_roll_std_4'] = df_15m['total_bytes'].rolling(4).std()
    
    # Status rates
    df_15m['success_rate'] = df_15m['count_success'] / denom
    df_15m['redirect_301_rate'] = df_15m['count_redirect_301'] / denom
    df_15m['redirect_302_rate'] = df_15m['count_redirect_302'] / denom
    df_15m['cache_304_rate'] = df_15m['count_cache_304'] / denom
    df_15m['client_error_rate'] = df_15m['count_client_err'] / denom
    df_15m['server_error_rate'] = df_15m['count_server_err'] / denom
    
    df_15m['redirect_rate'] = (
        df_15m['count_redirect_301'] + df_15m['count_redirect_302']
    ) / denom
    
    df_15m['error_rate'] = (
        df_15m['count_client_err'] + df_15m['count_server_err']
    ) / denom
    
    # Rate lags
    df_15m['error_rate_lag_1'] = df_15m['error_rate'].shift(1)
    df_15m['redirect_rate_lag_1'] = df_15m['redirect_rate'].shift(1)
    df_15m['cache_rate_lag_1'] = df_15m['cache_304_rate'].shift(1)
    
    # Time features
    df_15m['hour'] = df_15m.index.hour
    df_15m['day_of_week'] = df_15m.index.dayofweek
    df_15m['is_weekend'] = (df_15m['day_of_week'] >= 5).astype(int)
    
    df_15m['sin_hour'] = np.sin(2 * np.pi * df_15m['hour'] / 24)
    df_15m['cos_hour'] = np.cos(2 * np.pi * df_15m['hour'] / 24)
    df_15m['sin_dow'] = np.sin(2 * np.pi * df_15m['day_of_week'] / 7)
    df_15m['cos_dow'] = np.cos(2 * np.pi * df_15m['day_of_week'] / 7)
    
    # Momentum features
    df_15m['hits_diff_1'] = df_15m['hits'].diff(1)
    df_15m['bytes_diff_1'] = df_15m['total_bytes'].diff(1)
    
    df_15m['hits_pct_change_1'] = (
        df_15m['hits'].pct_change(1)
    ).replace([np.inf, -np.inf], 0).clip(-5, 5)
    
    df_15m['bytes_pct_change_1'] = (
        df_15m['total_bytes'].pct_change(1)
    ).replace([np.inf, -np.inf], 0).clip(-5, 5)
    
    # Composition
    df_15m['bytes_per_hit'] = df_15m['total_bytes'] / denom
    
    # Gap features
    df_15m['is_gap'] = (df_15m['hits'] == 0).astype(int)
    df_15m['time_gap_sec'] = df_15m.index.to_series().diff().dt.total_seconds()
    
    return df_15m


def build_all_timeseries(
    input_csv: str,
    output_dir: str,
    split: Literal["train", "test"]
):
    """
    Build all time series features (1m, 5m, 15m) from parsed CSV
    
    Args:
        input_csv: Path to parsed CSV file
        output_dir: Directory to save processed files
        split: Either "train" or "test"
    """
    print(f"\n{'='*60}")
    print(f"Building time series features: {split}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading: {input_csv}")
    df = load_parsed_data(input_csv)
    print(f"✓ Loaded {len(df):,} rows")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build 1-minute features
    print("\nBuilding 1-minute features...")
    df_1m = build_1m_features(df).reset_index()
    output_1m = output_dir / f"{split}_ts_1m.csv"
    df_1m.to_csv(output_1m, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {output_1m} ({len(df_1m):,} rows, {len(df_1m.columns)} cols)")
    
    # Build 5-minute features
    print("\nBuilding 5-minute features...")
    df_5m = build_5m_features(df).reset_index()
    output_5m = output_dir / f"{split}_ts_5m.csv"
    df_5m.to_csv(output_5m, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {output_5m} ({len(df_5m):,} rows, {len(df_5m.columns)} cols)")
    
    # Build 15-minute features
    print("\nBuilding 15-minute features...")
    df_15m = build_15m_features(df).reset_index()
    output_15m = output_dir / f"{split}_ts_15m.csv"
    df_15m.to_csv(output_15m, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {output_15m} ({len(df_15m):,} rows, {len(df_15m.columns)} cols)")
    
    print(f"\n✓ Complete: {split} time series features")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        input_csv = sys.argv[1]
        output_dir = sys.argv[2]
        split = sys.argv[3] if len(sys.argv) > 3 else "train"
        build_all_timeseries(input_csv, output_dir, split)
    else:
        # Default: process both train and test
        build_all_timeseries(
            "data/interim/train_parsed.csv",
            "data/processed",
            "train"
        )
        build_all_timeseries(
            "data/interim/test_parsed.csv",
            "data/processed",
            "test"
        )