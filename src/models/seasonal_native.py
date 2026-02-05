"""
Seasonal Naive baseline model for time series forecasting
Uses value from same time in previous season (day)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from src.evaluation.metrics import compute_metrics, MetricsLogger


class SeasonalNaiveTrainer:
    """Seasonal Naive baseline model trainer"""
    
    def __init__(
        self,
        predictions_dir: str,
        metrics_logger: MetricsLogger
    ):
        """
        Initialize Seasonal Naive trainer
        
        Args:
            predictions_dir: Directory to save predictions
            metrics_logger: Metrics logger instance
        """
        self.predictions_dir = Path(predictions_dir)
        self.metrics_logger = metrics_logger
        
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_tag_minutes(self, tag: str) -> int:
        """Get minutes per time window"""
        return {"1m": 1, "5m": 5, "15m": 15}[tag]
    
    def _steps_per_day(self, tag: str) -> int:
        """Get number of steps per day"""
        return int(24 * 60 / self._get_tag_minutes(tag))
    
    def _seasonal_naive_forecast(self, hist: np.ndarray, season_len: int) -> float:
        """
        Seasonal naive forecast: use value from same time in previous season
        
        Args:
            hist: Historical values
            season_len: Season length (steps per day)
            
        Returns:
            Forecasted value
        """
        hist = np.asarray(hist, dtype=float)
        
        if len(hist) == 0:
            return 0.0
        
        if len(hist) < season_len:
            # Not enough history, use last value
            return float(hist[-1])
        
        # Use value from same time yesterday
        return float(hist[-season_len])
    
    def train_one_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: str,
        tag: str
    ) -> Dict[str, float]:
        """
        Train Seasonal Naive model (no actual training, just rolling forecast)
        
        Args:
            train_df: Training data
            test_df: Test data
            target: Target column name
            tag: Time window tag (1m, 5m, 15m)
            
        Returns:
            Test metrics dictionary
        """
        TIME_COL = "datetime"
        
        print(f"\n{'='*60}")
        print(f"Training Seasonal Naive: {target} @ {tag}")
        print(f"{'='*60}")
        print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")
        
        # Prepare test data
        test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)
        te = test_df[[TIME_COL, target]].copy()
        
        # Create true_next (shift target by -1 for t -> t+1 forecast)
        te["true_next"] = pd.to_numeric(te[target], errors="coerce").astype(float).shift(-1)
        eval_df = te[te["true_next"].notna()].copy().reset_index(drop=True)
        
        if len(eval_df) == 0:
            print("⚠️  No valid test data for evaluation")
            test_metrics = {k: np.nan for k in ["RMSE", "MSE", "MAE", "MAPE"]}
            out_df = te[[TIME_COL, target, "true_next"]].head(0).copy()
            out_df["pred"] = np.nan
            return test_metrics
        
        # Initialize history with training data
        hist = pd.to_numeric(train_df[target], errors="coerce").astype(float).fillna(0.0).values.tolist()
        season_len = self._steps_per_day(tag)
        
        print(f"Season length: {season_len} steps (1 day)")
        print(f"Initial history: {len(hist)} points")
        
        # Rolling forecast on test set
        preds = []
        for i in range(len(eval_df)):
            # Get current true value
            y_t = float(pd.to_numeric(te.iloc[i][target], errors="coerce"))
            if not np.isfinite(y_t):
                y_t = 0.0
            
            # Add to history
            hist.append(y_t)
            
            # Forecast next value
            forecast = max(0.0, self._seasonal_naive_forecast(hist, season_len))
            preds.append(forecast)
        
        eval_df["pred"] = np.asarray(preds, dtype=float)
        
        # Compute metrics
        test_metrics = compute_metrics(
            eval_df["true_next"].values,
            eval_df["pred"].values,
            target
        )
        
        print(f"Test: RMSE={test_metrics['RMSE']:.2f}, MAE={test_metrics['MAE']:.2f}, MAPE={test_metrics['MAPE']:.2f}%")
        
        # Save predictions
        csv_path = self.predictions_dir / f"pred_{target}_{tag}_seasonal_naive.csv"
        pq_path = self.predictions_dir / f"pred_{target}_{tag}_seasonal_naive.parquet"
        
        out_df = eval_df[[TIME_COL, target, "true_next", "pred"]]
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out_df.to_parquet(pq_path, index=False)
        
        print(f"✓ Predictions saved: {csv_path}")
        
        # Log metrics
        self.metrics_logger.log_metrics(
            model_name="seasonal_naive",
            target=target,
            window=tag,
            split="test",
            metrics=test_metrics
        )
        
        return test_metrics