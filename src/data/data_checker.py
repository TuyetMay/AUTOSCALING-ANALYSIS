"""
Database and data validation utilities
Check if data exists in PostgreSQL or CSV files
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sqlalchemy import create_engine, text

from configs.config import config


def check_postgres_data() -> Tuple[bool, Dict[str, int]]:
    """
    Check if processed data exists in PostgreSQL
    
    Returns:
        (has_data, row_counts) where row_counts is dict of {table: count}
    """
    if not config.SAVE_TO_POSTGRES:
        return False, {}
    
    try:
        engine = create_engine(config.get_connection_string())
        
        with engine.connect() as conn:
            # Check if tables exist and have data
            tables = {
                'timeseries_1m': 0,
                'timeseries_5m': 0,
                'timeseries_15m': 0
            }
            
            for table in tables.keys():
                try:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {config.POSTGRES_SCHEMA}.{table}")
                    )
                    count = result.scalar()
                    tables[table] = count
                except Exception:
                    tables[table] = 0
            
        engine.dispose()
        
        # Has data if all tables have rows
        has_data = all(count > 0 for count in tables.values())
        
        return has_data, tables
        
    except Exception as e:
        print(f"⚠️  Could not check PostgreSQL: {e}")
        return False, {}


def check_csv_data() -> Tuple[bool, Dict[str, int]]:
    """
    Check if processed CSV files exist
    
    Returns:
        (has_data, file_info) where file_info is dict of {filename: row_count}
    """
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        return False, {}
    
    files = {
        'train_ts_1m.csv': 0,
        'train_ts_5m.csv': 0,
        'train_ts_15m.csv': 0,
        'test_ts_1m.csv': 0,
        'test_ts_5m.csv': 0,
        'test_ts_15m.csv': 0
    }
    
    for filename in files.keys():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                files[filename] = len(df)
            except Exception:
                files[filename] = 0
        else:
            files[filename] = 0
    
    # Has data if all files exist and have rows
    has_data = all(count > 0 for count in files.values())
    
    return has_data, files


def check_raw_data() -> Tuple[bool, Dict[str, bool]]:
    """
    Check if raw data files exist
    
    Returns:
        (has_data, file_exists) where file_exists is dict of {filename: exists}
    """
    data_dir = Path("data/raw")
    
    files = {
        'train.txt': False,
        'test.txt': False
    }
    
    for filename in files.keys():
        filepath = data_dir / filename
        files[filename] = filepath.exists()
    
    has_data = all(files.values())
    
    return has_data, files


def check_trained_models() -> Tuple[bool, Dict[str, int]]:
    """
    Check if trained models exist
    
    Returns:
        (has_models, model_info) where model_info is dict of {window: count}
    """
    models_dir = Path("artifacts/models")
    
    if not models_dir.exists():
        return False, {}
    
    windows = {
        '1m': 0,
        '5m': 0,
        '15m': 0
    }
    
    for window in windows.keys():
        window_dir = models_dir / window
        if window_dir.exists():
            # Count .json model files
            model_files = list(window_dir.glob("model_xgb_*.json"))
            windows[window] = len(model_files)
        else:
            windows[window] = 0
    
    # Has models if each window has at least 2 models (hits + bytes)
    has_models = all(count >= 2 for count in windows.values())
    
    return has_models, windows


def print_data_status():
    """Print comprehensive data status"""
    print("\n" + "="*70)
    print("DATA STATUS CHECK")
    print("="*70)
    
    # Check raw data
    has_raw, raw_files = check_raw_data()
    print("\n📁 Raw Data:")
    for filename, exists in raw_files.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")
    
    # Check processed CSV
    has_csv, csv_files = check_csv_data()
    print("\n📊 Processed CSV Files:")
    if has_csv:
        print("  ✓ All processed files exist")
        for filename, count in csv_files.items():
            if count > 0:
                print(f"    - {filename}: {count:,} rows")
    else:
        print("  ✗ Processed files missing or incomplete")
        for filename, count in csv_files.items():
            status = "✓" if count > 0 else "✗"
            print(f"    {status} {filename}: {count:,} rows")
    
    # Check PostgreSQL
    has_pg, pg_tables = check_postgres_data()
    print("\n🗄️  PostgreSQL Database:")
    if config.SAVE_TO_POSTGRES:
        if has_pg:
            print(f"  ✓ Database: {config.POSTGRES_DB}")
            print(f"  ✓ Schema: {config.POSTGRES_SCHEMA}")
            for table, count in pg_tables.items():
                print(f"    - {table}: {count:,} rows")
        else:
            print("  ✗ Database tables missing or empty")
            for table, count in pg_tables.items():
                status = "✓" if count > 0 else "✗"
                print(f"    {status} {table}: {count:,} rows")
    else:
        print("  ⚠️  PostgreSQL saving disabled")
    
    # Check trained models
    has_models, model_info = check_trained_models()
    print("\n🤖 Trained Models:")
    if has_models:
        print("  ✓ All models trained")
        for window, count in model_info.items():
            print(f"    - {window}: {count} models")
    else:
        print("  ✗ Models missing or incomplete")
        for window, count in model_info.items():
            status = "✓" if count >= 2 else "✗"
            expected = "2 models expected (hits + bytes)"
            print(f"    {status} {window}: {count} models ({expected})")
    
    print("\n" + "="*70)
    
    return {
        'has_raw': has_raw,
        'has_csv': has_csv,
        'has_pg': has_pg,
        'has_models': has_models
    }


if __name__ == "__main__":
    status = print_data_status()
    
    print("\nSummary:")
    if status['has_raw']:
        print("  ✓ Raw data available")
    else:
        print("  ✗ Raw data missing - place train.txt and test.txt in data/raw/")
    
    if status['has_csv']:
        print("  ✓ Processed data available")
    else:
        print("  ✗ Processed data missing - run preprocessing first")
    
    if status['has_models']:
        print("  ✓ Models trained")
    else:
        print("  ✗ Models not trained - run training pipeline")
    
    print()