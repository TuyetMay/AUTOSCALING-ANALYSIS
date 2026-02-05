"""
Evaluation metrics for forecasting models
Computes RMSE, MSE, MAE, MAPE and saves to CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target: str) -> Dict[str, float]:
    """
    Compute forecasting metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        target: Target name (for logging)
        
    Returns:
        Dictionary with RMSE, MSE, MAE, MAPE
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Filter out NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            "RMSE": np.nan,
            "MSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan
        }
    
    # MSE, RMSE, MAE
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # MAPE - avoid division by zero
    epsilon = 1e-10
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
    
    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape
    }


class MetricsLogger:
    """Logger for model metrics - saves to CSV in long format"""
    
    def __init__(self, metrics_path: str):
        """
        Initialize metrics logger
        
        Args:
            metrics_path: Path to save metrics CSV
        """
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load existing metrics
        if self.metrics_path.exists():
            self.df = pd.read_csv(self.metrics_path)
        else:
            self.df = pd.DataFrame(columns=[
                "model", "target", "window", "split", "metric", "value"
            ])
    
    def log_metrics(
        self,
        model_name: str,
        target: str,
        window: str,
        split: str,
        metrics: Dict[str, float]
    ):
        """
        Log metrics for a model
        
        Args:
            model_name: Name of model (e.g., 'xgb', 'seasonal_naive')
            target: Target variable (e.g., 'hits', 'bytes_sum')
            window: Time window (e.g., '1m', '5m', '15m')
            split: Data split (e.g., 'cv_mean', 'test')
            metrics: Dictionary of metric values
        """
        rows = []
        for metric_name, value in metrics.items():
            rows.append({
                "model": model_name,
                "target": target,
                "window": window,
                "split": split,
                "metric": metric_name,
                "value": float(value) if value is not None else np.nan
            })
        
        # Append to dataframe
        new_df = pd.DataFrame(rows)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        # Save to CSV
        self.save()
    
    def save(self):
        """Save metrics to CSV"""
        self.df.to_csv(self.metrics_path, index=False, encoding='utf-8-sig')
    
    def get_benchmark(self, split: str = "test") -> pd.DataFrame:
        """
        Get benchmark comparison table
        
        Args:
            split: Which split to show (default: 'test')
            
        Returns:
            Pivot table comparing models
        """
        if len(self.df) == 0:
            return pd.DataFrame()
        
        df_split = self.df[self.df["split"].str.lower() == split.lower()].copy()
        
        if len(df_split) == 0:
            return pd.DataFrame()
        
        # Pivot to wide format for comparison
        bench = df_split.pivot_table(
            index=["target", "window", "metric"],
            columns=["model"],
            values="value",
            aggfunc="first"
        ).reset_index()
        
        return bench.sort_values(["target", "window", "metric"])
    
    def print_summary(self):
        """Print summary of logged metrics"""
        if len(self.df) == 0:
            print("No metrics logged yet")
            return
        
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        
        # Show test set benchmark
        bench = self.get_benchmark(split="test")
        if len(bench) > 0:
            print("\nTest Set Performance:")
            print(bench.to_string(index=False))
        
        # Show CV results if available
        bench_cv = self.get_benchmark(split="cv_mean")
        if len(bench_cv) > 0:
            print("\nCross-Validation Performance:")
            print(bench_cv.to_string(index=False))
        
        print("\n" + "="*70)