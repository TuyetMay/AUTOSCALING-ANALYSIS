"""
Main pipeline for NASA log data analysis
Complete workflow:
1. Check if data already exists (skip preprocessing if found)
2. Parse raw logs and build time series features (if needed)
3. Train forecasting models (XGBoost + Seasonal Naive)
4. Save to PostgreSQL (optional)
"""

import sys
from pathlib import Path
from datetime import datetime

from configs.config import config as app_config
from db_connector import PostgreSQLConnector
from src.data.build_timeseries import build_all_timeseries
from src.data.data_checker import check_csv_data, check_postgres_data, check_raw_data, check_trained_models, print_data_status
from src.data.parse_log import parse_and_save
from src.training.train_models import run_training_pipeline


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
        "artifacts/models/1m",
        "artifacts/models/5m",
        "artifacts/models/15m",
        "artifacts/predictions",
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


def run_preprocessing_pipeline(save_to_db: bool = None, force: bool = False):
    """
    Preprocessing pipeline: parse logs and build features
    
    Args:
        save_to_db: Override config setting for saving to PostgreSQL
        force: Force reprocessing even if data exists
    """
    print_header("STEP 1: DATA PREPROCESSING")
    
    # Check if data already exists
    if not force:
        has_csv, csv_info = check_csv_data()
        has_pg, pg_info = check_postgres_data()
        
        if has_csv:
            print("✓ Processed CSV data already exists:")
            for filename, count in csv_info.items():
                if count > 0:
                    print(f"  - {filename}: {count:,} rows")
            
            if has_pg:
                print("\n✓ PostgreSQL data already exists:")
                for table, count in pg_info.items():
                    if count > 0:
                        print(f"  - {table}: {count:,} rows")
            
            print("\n⏭️  Skipping preprocessing (use --force to reprocess)")
            return True
    
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
    
    # Check raw data
    has_raw, raw_files = check_raw_data()
    if not has_raw:
        print("❌ ERROR: Raw data files not found!")
        print("Please place the following files in data/raw/:")
        for filename, exists in raw_files.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {filename}")
        return False
    
    # Parse raw logs to CSV
    print_header("Parsing Raw Logs → CSV")
    
    train_raw = "data/raw/train.txt"
    train_parsed = "data/interim/train_parsed.csv"
    parse_and_save(train_raw, train_parsed)
    
    test_raw = "data/raw/test.txt"
    test_parsed = "data/interim/test_parsed.csv"
    parse_and_save(test_raw, test_parsed)
    
    # Build time series features
    print_header("Building Time Series Features")
    
    build_all_timeseries(
        input_csv=train_parsed,
        output_dir="data/processed",
        split="train"
    )
    
    build_all_timeseries(
        input_csv=test_parsed,
        output_dir="data/processed",
        split="test"
    )
    
    # Save to PostgreSQL (optional)
    if save_to_db and db:
        print_header("Saving to PostgreSQL")
        
        try:
            save_to_postgres(db, "train")
            save_to_postgres(db, "test")
            
            # Show database statistics
            db.get_table_stats()
            
        except Exception as e:
            print(f"❌ Error saving to PostgreSQL: {e}")
        finally:
            db.close()
    
    print_header("Preprocessing Complete")
    print("✓ Data ready for training")
    
    return True


def run_full_pipeline(
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    force_preprocessing: bool = False,
    save_to_db: bool = None
):
    """
    Run complete pipeline: preprocessing + training
    
    Args:
        skip_preprocessing: Skip data preprocessing
        skip_training: Skip model training
        force_preprocessing: Force reprocessing even if data exists
        save_to_db: Override config for PostgreSQL saving
    """
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print(" "*15 + "NASA LOG ANALYSIS PIPELINE")
    print("="*70)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Show initial status
    status = print_data_status()
    
    # Step 1: Preprocessing
    if not skip_preprocessing:
        success = run_preprocessing_pipeline(
            save_to_db=save_to_db,
            force=force_preprocessing
        )
        if not success:
            print("\n❌ Pipeline failed at preprocessing stage")
            return
    else:
        print_header("STEP 1: DATA PREPROCESSING")
        print("⏭️  Skipping preprocessing (--skip-preprocessing)")
        
        # Check if data exists
        has_csv, _ = check_csv_data()
        if not has_csv:
            print("\n❌ ERROR: No processed data found!")
            print("Cannot skip preprocessing without existing data")
            return
    
    # Step 2: Model Training
    if not skip_training:
        print_header("STEP 2: MODEL TRAINING")
        
        # Check if data exists
        has_csv, _ = check_csv_data()
        if not has_csv:
            print("❌ ERROR: No processed data found!")
            print("Run preprocessing first or remove --skip-preprocessing")
            return
        
        try:
            run_training_pipeline(
                tags=["1m", "5m", "15m"],
                targets=["hits", "total_bytes"],
                models=["xgb", "seasonal_naive"]
            )
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print_header("STEP 2: MODEL TRAINING")
        print("⏭️  Skipping training (--skip-training)")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETE")
    print("="*70)
    
    # Show final status
    print("\n📊 Final Status:")
    
    has_csv, csv_info = check_csv_data()
    if has_csv:
        print("  ✓ Processed data: 6 files ready")
    
    has_models, model_info = check_trained_models()
    if has_models:
        total_models = sum(model_info.values())
        print(f"  ✓ Trained models: {total_models} models ready")
    
    metrics_file = Path("artifacts/metrics.csv")
    if metrics_file.exists():
        print(f"  ✓ Metrics: {metrics_file}")
    
    predictions_dir = Path("artifacts/predictions")
    if predictions_dir.exists():
        pred_files = list(predictions_dir.glob("*.csv"))
        print(f"  ✓ Predictions: {len(pred_files)} files")
    
    print(f"\n⏱️  Total time: {duration:.1f} seconds")
    print(f"✓ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("="*70)
    
    print("\n📝 Next steps:")
    if has_models:
        print("  - Review metrics in artifacts/metrics.csv")
        print("  - Check predictions in artifacts/predictions/")
        print("  - Use models for API deployment")
    else:
        print("  - Run training: python main.py --skip-preprocessing")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NASA Log Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (preprocessing + training)
  python main.py
  
  # Skip preprocessing if data exists
  python main.py --skip-preprocessing
  
  # Force reprocess data
  python main.py --force
  
  # Only train models
  python main.py --skip-preprocessing
  
  # Only preprocess data
  python main.py --skip-training
  
  # Disable PostgreSQL
  python main.py --no-db
        """
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing (use existing processed data)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (only preprocess data)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if data exists"
    )
    
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable PostgreSQL saving"
    )
    
    parser.add_argument(
        "--db",
        action="store_true",
        help="Enable PostgreSQL saving"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show data status and exit"
    )
    
    args = parser.parse_args()
    
    # Handle --status flag
    if args.status:
        print_data_status()
        sys.exit(0)
    
    # Determine PostgreSQL setting
    save_to_db = None
    if args.no_db:
        save_to_db = False
    elif args.db:
        save_to_db = True
    
    try:
        run_full_pipeline(
            skip_preprocessing=args.skip_preprocessing,
            skip_training=args.skip_training,
            force_preprocessing=args.force,
            save_to_db=save_to_db
        )
        
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)