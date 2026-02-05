"""
Training pipeline for forecasting models
Trains XGBoost and Seasonal Naive models on all time windows
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.models.seasonal_native import SeasonalNaiveTrainer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import MetricsLogger
from src.models.xgb import XGBTrainer


def prepare_training_data(tag: str) -> tuple:
    """
    Load and prepare training/test data for a given time window
    
    Args:
        tag: Time window (1m, 5m, 15m)
        
    Returns:
        (train_df, test_df, feature_cols_hits, feature_cols_bytes)
    """
    # Load data
    train_path = f"data/processed/train_ts_{tag}.csv"
    test_path = f"data/processed/test_ts_{tag}.csv"
    
    print(f"\nLoading data for {tag}:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Convert datetime
    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    test_df["datetime"] = pd.to_datetime(test_df["datetime"])
    
    print(f"  Train: {len(train_df):,} rows | {train_df['datetime'].min()} → {train_df['datetime'].max()}")
    print(f"  Test: {len(test_df):,} rows | {test_df['datetime'].min()} → {test_df['datetime'].max()}")
    
    # Define feature columns
    # Exclude datetime, targets, and other non-feature columns
    exclude_cols = {
        'datetime', 'hits', 'total_bytes', 'split',
        'hits_lag_1', 'hits_lag_2', 'hits_lag_3', 'hits_lag_4', 
        'hits_lag_5', 'hits_lag_6', 'hits_lag_10',
        'bytes_lag_1', 'bytes_lag_2', 'bytes_lag_3', 'bytes_lag_4',
        'bytes_lag_5', 'bytes_lag_6', 'bytes_lag_10'
    }
    
    # All columns except excluded ones
    all_cols = set(train_df.columns)
    base_features = list(all_cols - exclude_cols)
    
    # Features for hits prediction (include bytes features)
    feature_cols_hits = base_features.copy()
    
    # Features for bytes prediction (exclude hits-related features to avoid leakage)
    hits_related = {col for col in base_features if 'hits' in col.lower()}
    feature_cols_bytes = [col for col in base_features if col not in hits_related]
    
    print(f"  Features for hits: {len(feature_cols_hits)}")
    print(f"  Features for bytes: {len(feature_cols_bytes)}")
    
    return train_df, test_df, feature_cols_hits, feature_cols_bytes


def run_training_pipeline(
    tags: list = None,
    targets: list = None,
    models: list = None
):
    """
    Run complete training pipeline
    
    Args:
        tags: Time windows to train (default: ['1m', '5m', '15m'])
        targets: Targets to predict (default: ['hits', 'total_bytes'])
        models: Models to train (default: ['xgb', 'seasonal_naive'])
    """
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Defaults
    if tags is None:
        tags = ["1m", "5m", "15m"]
    if targets is None:
        targets = ["hits", "total_bytes"]
    if models is None:
        models = ["xgb", "seasonal_naive"]
    
    print(f"Configuration:")
    print(f"  Time windows: {tags}")
    print(f"  Targets: {targets}")
    print(f"  Models: {models}")
    
    # Setup directories
    artifacts_dir = Path("artifacts")
    models_dir = artifacts_dir / "models"
    predictions_dir = artifacts_dir / "predictions"
    metrics_path = artifacts_dir / "metrics.csv"
    
    for d in [artifacts_dir, models_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(str(metrics_path))
    
    # XGBoost parameters
    xgb_params = {
        "booster": "gbtree",
        "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 42,
    }
    
    # Initialize trainers
    xgb_trainer = XGBTrainer(
        model_dir=str(models_dir),
        predictions_dir=str(predictions_dir),
        metrics_logger=metrics_logger,
        xgb_params=xgb_params,
        cv_splits=5,
        cv_test_days=2,
        cv_gap_steps=1
    )
    
    seasonal_trainer = SeasonalNaiveTrainer(
        predictions_dir=str(predictions_dir),
        metrics_logger=metrics_logger
    )
    
    # Train models for each combination
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    for tag in tags:
        print(f"\n{'#'*70}")
        print(f"# Time Window: {tag}")
        print(f"{'#'*70}")
        
        # Load data
        try:
            train_df, test_df, feat_hits, feat_bytes = prepare_training_data(tag)
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {tag}: {e}")
            continue
        
        for target in targets:
            # Determine feature columns and log transform
            if target == "hits":
                feature_cols = feat_hits
                use_log = False
            else:  # total_bytes
                feature_cols = feat_bytes
                use_log = True
            
            # Train XGBoost
            if "xgb" in models:
                try:
                    xgb_trainer.train_one_model(
                        train_df=train_df,
                        test_df=test_df,
                        feature_cols=feature_cols,
                        target=target,
                        tag=tag,
                        use_log=use_log
                    )
                except Exception as e:
                    print(f"❌ XGBoost training failed for {target} @ {tag}: {e}")
            
            # Train Seasonal Naive
            if "seasonal_naive" in models:
                try:
                    seasonal_trainer.train_one_model(
                        train_df=train_df,
                        test_df=test_df,
                        target=target,
                        tag=tag
                    )
                except Exception as e:
                    print(f"❌ Seasonal Naive training failed for {target} @ {tag}: {e}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Print metrics summary
    metrics_logger.print_summary()
    
    print(f"\n✓ Models saved to: {models_dir}")
    print(f"✓ Predictions saved to: {predictions_dir}")
    print(f"✓ Metrics saved to: {metrics_path}")
    print(f"\n✓ Total time: {duration:.1f} seconds")
    print(f"✓ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["1m", "5m", "15m"],
        help="Time windows to train (default: 1m 5m 15m)"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["hits", "total_bytes"],
        help="Targets to predict (default: hits total_bytes)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "seasonal_naive"],
        help="Models to train (default: xgb seasonal_naive)"
    )
    
    args = parser.parse_args()
    
    try:
        run_training_pipeline(
            tags=args.tags,
            targets=args.targets,
            models=args.models
        )
    except KeyboardInterrupt:
        print("\n\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)