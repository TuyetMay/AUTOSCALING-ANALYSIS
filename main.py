"""
Main pipeline for preprocessing NASA log data
Runs complete workflow from raw logs to time series features
Optionally saves to PostgreSQL database
"""

import sys
from pathlib import Path
from datetime import datetime

from configs.config import config as app_config
from db_connector import PostgreSQLConnector
from src.data.build_timeseries import build_all_timeseries
from src.data.parse_log import parse_and_save

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        "data/raw",
        "data/interim", 
        "data/processed",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready\n")


def save_to_postgres(db: PostgreSQLConnector, split: str):
    """
    Save preprocessed data to PostgreSQL
    
    Args:
        db: PostgreSQL connector instance
        split: Data split (train or test)
    """
    print(f"\nSaving {split} data to PostgreSQL...")
    
    import pandas as pd
    
    # Clear existing data for this split
    db.clear_split_data(split)
    
    # Load and save parsed logs
    parsed_path = f"data/interim/{split}_parsed.csv"
    if Path(parsed_path).exists():
        df_parsed = pd.read_csv(parsed_path)
        db.save_dataframe(df_parsed, "parsed_logs", split, if_exists="append")
    
    # Load and save time series data
    for window in ["1m", "5m", "15m"]:
        ts_path = f"data/processed/{split}_ts_{window}.csv"
        if Path(ts_path).exists():
            df_ts = pd.read_csv(ts_path)
            table_name = f"timeseries_{window}"
            db.save_dataframe(df_ts, table_name, split, if_exists="append")
    
    print(f"✓ Saved {split} data to PostgreSQL")


def run_preprocessing_pipeline(save_to_db: bool = None):
    """
    Complete preprocessing pipeline:
    1. Parse raw log files (train.txt, test.txt) → CSV
    2. Build time series features (1m, 5m, 15m) for each split
    3. Optionally save to PostgreSQL database
    
    Args:
        save_to_db: Override config setting for saving to PostgreSQL
    """
    start_time = datetime.now()
    
    print_header("NASA LOG PREPROCESSING PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Determine if we should save to PostgreSQL
    if save_to_db is None:
        save_to_db = app_config.SAVE_TO_POSTGRES

    
    # Ensure directories exist
    ensure_directories()
    
    # Initialize database connector if needed
    db = None
    if save_to_db:
        print_header("Initializing PostgreSQL Connection")
        db = PostgreSQLConnector()
        if db.connect():
            db.create_tables()
        else:
            print("⚠️  PostgreSQL connection failed - will only save to CSV")
            save_to_db = False
            db = None
    
    # =============================
    # STEP 1: Parse raw logs to CSV
    # =============================
    print_header("STEP 1: Parse Raw Logs → CSV")
    
    # Parse train.txt
    train_raw = "data/raw/train.txt"
    train_parsed = "data/interim/train_parsed.csv"
    
    if not Path(train_raw).exists():
        print(f"❌ ERROR: {train_raw} not found!")
        print("Please place train.txt in data/raw/ directory")
        return
    
    parse_and_save(train_raw, train_parsed)
    
    # Parse test.txt
    test_raw = "data/raw/test.txt"
    test_parsed = "data/interim/test_parsed.csv"
    
    if not Path(test_raw).exists():
        print(f"❌ ERROR: {test_raw} not found!")
        print("Please place test.txt in data/raw/ directory")
        return
    
    parse_and_save(test_raw, test_parsed)
    
    # ========================================
    # STEP 2: Build time series features
    # ========================================
    print_header("STEP 2: Build Time Series Features")
    
    # Build train features (1m, 5m, 15m)
    build_all_timeseries(
        input_csv=train_parsed,
        output_dir="data/processed",
        split="train"
    )
    
    # Build test features (1m, 5m, 15m)
    build_all_timeseries(
        input_csv=test_parsed,
        output_dir="data/processed",
        split="test"
    )
    
    # ========================================
    # STEP 3: Save to PostgreSQL (optional)
    # ========================================
    if save_to_db and db:
        print_header("STEP 3: Save to PostgreSQL")
        
        try:
            save_to_postgres(db, "train")
            save_to_postgres(db, "test")
            
            # Show database statistics
            db.get_table_stats()
            
        except Exception as e:
            print(f"❌ Error saving to PostgreSQL: {e}")
        finally:
            db.close()
    
    # =============================
    # SUMMARY
    # =============================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("PIPELINE COMPLETE")
    
    print("✓ Generated files:")
    print("\n  Parsed CSV:")
    print(f"    - {train_parsed}")
    print(f"    - {test_parsed}")
    
    print("\n  Time Series Features:")
    for split in ["train", "test"]:
        for window in ["1m", "5m", "15m"]:
            path = f"data/processed/{split}_ts_{window}.csv"
            if Path(path).exists():
                size = Path(path).stat().st_size / (1024 * 1024)  # MB
                print(f"    - {path} ({size:.2f} MB)")
    
    if save_to_db:
        print("\n  PostgreSQL Database:")
        print(f"    - Database: {app_config.POSTGRES_DB}")
        print(f"    - Schema: {app_config.POSTGRES_SCHEMA}")
        print("    - Tables: parsed_logs, timeseries_1m, timeseries_5m, timeseries_15m")
    
    print(f"\n✓ Total time: {duration:.1f} seconds")
    print(f"✓ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Train models using data/processed/*_ts_*.csv files")
    if save_to_db:
        print("  - Query PostgreSQL database for analysis and modeling")
    print("  - Use 1m data for high-frequency predictions")
    print("  - Use 5m data for medium-term forecasting")
    print("  - Use 15m data for longer-term planning")
    print()


if __name__ == "__main__":
    try:
        # Check for command line arguments
        save_to_db = None
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            if arg in ["--no-db", "--skip-db"]:
                save_to_db = False
                print("📝 PostgreSQL saving disabled via command line\n")
            elif arg in ["--db", "--save-db"]:
                save_to_db = True
                print("📝 PostgreSQL saving enabled via command line\n")
        
        run_preprocessing_pipeline(save_to_db=save_to_db)
        
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)