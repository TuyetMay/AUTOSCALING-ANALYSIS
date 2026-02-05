
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from src.evaluation.metrics import compute_metrics, MetricsLogger


class XGBTrainer:
    """XGBoost trainer for time series forecasting"""
    
    def __init__(
        self,
        model_dir: str,
        predictions_dir: str,
        metrics_logger: MetricsLogger,
        xgb_params: Optional[Dict] = None,
        cv_splits: int = 5,
        cv_test_days: int = 2,
        cv_gap_steps: int = 1
    ):
        """
        Initialize XGBoost trainer
        
        Args:
            model_dir: Directory to save trained models
            predictions_dir: Directory to save predictions
            metrics_logger: Metrics logger instance
            xgb_params: XGBoost parameters
            cv_splits: Number of CV splits
            cv_test_days: Test period in days for CV
            cv_gap_steps: Gap between train and validation
        """
        self.model_dir = Path(model_dir)
        self.predictions_dir = Path(predictions_dir)
        self.metrics_logger = metrics_logger
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
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
        
        self.cv_splits = cv_splits
        self.cv_test_days = cv_test_days
        self.cv_gap_steps = cv_gap_steps
    
    def _get_tag_minutes(self, tag: str) -> int:
        """Get minutes per time window"""
        return {"1m": 1, "5m": 5, "15m": 15}[tag]
    
    def train_one_model(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target: str,
        tag: str,
        use_log: bool = False
    ) -> Dict[str, float]:
        """
        Train one XGBoost model with cross-validation
        
        Args:
            train_df: Training data
            test_df: Test data
            feature_cols: Feature column names
            target: Target column name
            tag: Time window tag (1m, 5m, 15m)
            use_log: Whether to use log transform for target
            
        Returns:
            Test metrics dictionary
        """
        TIME_COL = "datetime"
        
        # Ensure data is sorted
        train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)
        test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"Training XGBoost: {target} @ {tag}")
        print(f"{'='*60}")
        print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")
        print(f"Features: {len(feature_cols)} | Use log: {use_log}")
        
        # Cross-validation
        freq_min = self._get_tag_minutes(tag)
        test_size = int(self.cv_test_days * 24 * 60 / freq_min)
        gap = self.cv_gap_steps
        
        n = len(train_df)
        max_splits = (n - gap) // test_size - 1
        n_splits_eff = min(self.cv_splits, max(0, max_splits))
        
        cv_metrics = []
        
        if n_splits_eff >= 2:
            print(f"Running {n_splits_eff}-fold time series CV...")
            tss = TimeSeriesSplit(n_splits=n_splits_eff, test_size=test_size, gap=gap) # type: ignore
            
            for fold, (tr_idx, va_idx) in enumerate(tss.split(train_df)):
                tr = train_df.iloc[tr_idx]
                va = train_df.iloc[va_idx]
                
                X_tr, X_va = tr[feature_cols], va[feature_cols]
                y_tr = tr[target].astype(float).values
                y_va = va[target].astype(float).values
                
                # Apply log transform if needed
                if use_log:
                    y_tr_fit = np.log1p(np.maximum(y_tr, 0.0))
                    y_va_fit = np.log1p(np.maximum(y_va, 0.0))
                else:
                    y_tr_fit, y_va_fit = y_tr, y_va
                
                # Train fold model
                reg = xgb.XGBRegressor(**self.xgb_params)
                reg.fit(X_tr, y_tr_fit, eval_set=[(X_va, y_va_fit)], verbose=False)
                
                # Predict
                pred_fit = reg.predict(X_va)
                pred = np.expm1(pred_fit) if use_log else pred_fit
                pred = np.maximum(pred, 0.0)
                
                # Compute metrics
                fold_metrics = compute_metrics(y_va, pred, target)
                cv_metrics.append(fold_metrics)
                
                print(f"  Fold {fold+1}/{n_splits_eff}: RMSE={fold_metrics['RMSE']:.2f}, MAE={fold_metrics['MAE']:.2f}")
        
        # Compute CV mean metrics
        cv_mean = {
            k: float(np.mean([m[k] for m in cv_metrics])) if cv_metrics else np.nan
            for k in ["RMSE", "MSE", "MAE", "MAPE"]
        }
        
        if cv_metrics:
            print(f"\nCV Mean: RMSE={cv_mean['RMSE']:.2f}, MAE={cv_mean['MAE']:.2f}, MAPE={cv_mean['MAPE']:.2f}%")
            self.metrics_logger.log_metrics(
                model_name="xgb",
                target=target,
                window=tag,
                split="cv_mean",
                metrics=cv_mean
            )
        
        # Train final model on full training data
        print("\nTraining final model on full training data...")
        X_all = train_df[feature_cols]
        y_all = train_df[target].astype(float).values
        y_fit = np.log1p(np.maximum(y_all, 0.0)) if use_log else y_all
        
        # Remove early stopping for final model
        final_params = {k: v for k, v in self.xgb_params.items() if k != "early_stopping_rounds"}
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_all, y_fit, eval_set=[(X_all, y_fit)], verbose=False)
        
        # Save model
        model_path = self.model_dir / tag / f"model_xgb_{target}_{tag}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.get_booster().save_model(str(model_path))
        
        # Save feature columns
        feat_path = self.model_dir / tag / f"feat_cols_xgb_{target}_{tag}.json"
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Model saved: {model_path}")
        
        # Predict on test set (t -> t+1 forecast)
        print("\nPredicting on test set...")
        df = test_df[[TIME_COL, target] + feature_cols].copy()
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        
        # Shift to get next value as target
        df["true_next"] = pd.to_numeric(df[target], errors="coerce").astype(float).shift(-1)
        eval_df = df[df["true_next"].notna()].copy()
        
        if len(eval_df) == 0:
            print("⚠️  No valid test data for evaluation")
            test_metrics = {k: np.nan for k in ["RMSE", "MSE", "MAE", "MAPE"]}
            out_df = df[[TIME_COL, target, "true_next"]].head(0).copy()
            out_df["pred"] = np.nan
        else:
            # Predict
            pred_fit = model.predict(eval_df[feature_cols])
            pred = np.expm1(pred_fit) if use_log else pred_fit
            pred = np.maximum(pred, 0.0)
            
            eval_df["pred"] = pred
            
            # Compute test metrics
            test_metrics = compute_metrics(
                eval_df["true_next"].values,
                eval_df["pred"].values,
                target
            )
            
            print(f"Test: RMSE={test_metrics['RMSE']:.2f}, MAE={test_metrics['MAE']:.2f}, MAPE={test_metrics['MAPE']:.2f}%")
            
            out_df = eval_df[[TIME_COL, target, "true_next", "pred"]]
        
        # Save predictions
        csv_path = self.predictions_dir / f"pred_{target}_{tag}_xgb.csv"
        pq_path = self.predictions_dir / f"pred_{target}_{tag}_xgb.parquet"
        
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        out_df.to_parquet(pq_path, index=False)
        
        print(f"✓ Predictions saved: {csv_path}")
        
        # Log test metrics
        self.metrics_logger.log_metrics(
            model_name="xgb",
            target=target,
            window=tag,
            split="test",
            metrics=test_metrics
        )
        
        return test_metrics