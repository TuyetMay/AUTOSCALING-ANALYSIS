"""
Main pipeline for preprocessing NASA log data
Runs complete workflow from raw logs to time series features
"""

import sys
from pathlib import Path
from datetime import datetime

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


def run_preprocessing_pipeline():
    """
    Complete preprocessing pipeline:
    1. Parse raw log files (train.txt, test.txt) → CSV
    2. Build time series features (1m, 5m, 15m) for each split
    """
    start_time = datetime.now()
    
    print_header("NASA LOG PREPROCESSING PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Ensure directories exist
    ensure_directories()
    
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
    
    print(f"\n✓ Total time: {duration:.1f} seconds")
    print(f"✓ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Train models using data/processed/*_ts_*.csv files")
    print("  - Use 1m data for high-frequency predictions")
    print("  - Use 5m data for medium-term forecasting")
    print("  - Use 15m data for longer-term planning")
    print()


if __name__ == "__main__":
    try:
        run_preprocessing_pipeline()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)