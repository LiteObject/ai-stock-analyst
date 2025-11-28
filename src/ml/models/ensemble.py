"""
Ensemble ML models.

Implements multi-model ensemble predictors with:
- XGBoost, LightGBM, and CatBoost support
- GPU acceleration
- Optuna hyperparameter optimization
- Time-series aware cross-validation
- Stacking ensemble with meta-learner
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

from core.exceptions import MLModelError
from ml.models.base import BaseMLPredictor

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU configuration for ensemble training."""

    enabled: bool = True
    device_id: int = 0
    memory_fraction: float = 0.9


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    model_type: str = "xgboost"
    task: str = "classification"
    use_gpu: bool = True
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    early_stopping_rounds: int = 50
    auto_tune: bool = False
    tune_trials: int = 50
    random_state: int = 42


class EnsemblePredictor(BaseMLPredictor):
    """
    Enhanced ensemble predictor with GPU support and multiple model types.

    Supports:
    - Random Forest (sklearn)
    - XGBoost (with GPU)
    - LightGBM (with GPU)
    - CatBoost (with GPU)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        task: str = "classification",
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Initialize.

        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm', or 'catboost'
            task: 'classification' (direction) or 'regression' (returns)
            config: Optional configuration object
        """
        super().__init__(name=f"{model_type}_{task}")

        self.config = config or EnsembleConfig(model_type=model_type, task=task)
        self.model_type = self.config.model_type
        self.task = self.config.task

        # GPU detection
        self.gpu_available = self._detect_gpu()
        self.use_gpu = self.config.use_gpu and self.gpu_available

        # Validate model availability
        self._validate_model_availability()

        # Optimized parameters (populated by auto_tune)
        self.best_params_: Optional[Dict[str, Any]] = None

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            pass

        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"GPU detected: {result.stdout.strip()}")
                return True
        except Exception:
            pass

        return False

    def _validate_model_availability(self) -> None:
        """Validate that the requested model type is available."""
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to Random Forest")
            self.model_type = "random_forest"
        elif self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to XGBoost")
            self.model_type = "xgboost" if XGBOOST_AVAILABLE else "random_forest"
        elif self.model_type == "catboost" and not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available, falling back to XGBoost")
            self.model_type = "xgboost" if XGBOOST_AVAILABLE else "random_forest"

    def _get_gpu_params(self) -> Dict[str, Any]:
        """Get GPU parameters for the model."""
        if not self.use_gpu:
            return {}

        if self.model_type == "xgboost":
            return {
                "tree_method": "gpu_hist",
                "device": f"cuda:{self.config.use_gpu if isinstance(self.config.use_gpu, int) else 0}",
            }
        elif self.model_type == "lightgbm":
            return {
                "device": "gpu",
                "gpu_device_id": 0,
            }
        elif self.model_type == "catboost":
            return {
                "task_type": "GPU",
                "devices": "0",
            }

        return {}

    def _create_model(self, params: Optional[Dict[str, Any]] = None):
        """Create the underlying model."""
        base_params = params or {}
        gpu_params = self._get_gpu_params()

        if self.model_type == "random_forest":
            if self.task == "classification":
                return RandomForestClassifier(
                    n_estimators=base_params.get(
                        "n_estimators", self.config.n_estimators
                    ),
                    max_depth=base_params.get("max_depth", self.config.max_depth),
                    min_samples_split=base_params.get("min_samples_split", 5),
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )
            else:
                return RandomForestRegressor(
                    n_estimators=base_params.get(
                        "n_estimators", self.config.n_estimators
                    ),
                    max_depth=base_params.get("max_depth", self.config.max_depth),
                    min_samples_split=base_params.get("min_samples_split", 5),
                    random_state=self.config.random_state,
                    n_jobs=-1,
                )

        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            model_params = {
                "n_estimators": base_params.get(
                    "n_estimators", self.config.n_estimators
                ),
                "max_depth": base_params.get("max_depth", self.config.max_depth),
                "learning_rate": base_params.get(
                    "learning_rate", self.config.learning_rate
                ),
                "subsample": base_params.get("subsample", 0.8),
                "colsample_bytree": base_params.get("colsample_bytree", 0.8),
                "reg_alpha": base_params.get("reg_alpha", 0.1),
                "reg_lambda": base_params.get("reg_lambda", 0.1),
                "random_state": self.config.random_state,
                "n_jobs": -1 if not self.use_gpu else 1,
                **gpu_params,
            }

            if self.task == "classification":
                return xgb.XGBClassifier(**model_params)  # type: ignore
            else:
                return xgb.XGBRegressor(**model_params)  # type: ignore

        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            model_params = {
                "n_estimators": base_params.get(
                    "n_estimators", self.config.n_estimators
                ),
                "num_leaves": base_params.get("num_leaves", 63),
                "max_depth": base_params.get("max_depth", self.config.max_depth),
                "learning_rate": base_params.get(
                    "learning_rate", self.config.learning_rate
                ),
                "subsample": base_params.get("subsample", 0.8),
                "colsample_bytree": base_params.get("colsample_bytree", 0.8),
                "reg_alpha": base_params.get("reg_alpha", 0.1),
                "reg_lambda": base_params.get("reg_lambda", 0.1),
                "random_state": self.config.random_state,
                "verbosity": -1,
                "n_jobs": -1 if not self.use_gpu else 1,
                **gpu_params,
            }

            if self.task == "classification":
                return lgb.LGBMClassifier(**model_params)  # type: ignore
            else:
                return lgb.LGBMRegressor(**model_params)  # type: ignore

        elif self.model_type == "catboost" and CATBOOST_AVAILABLE:
            model_params = {
                "iterations": base_params.get("n_estimators", self.config.n_estimators),
                "depth": base_params.get("max_depth", self.config.max_depth),
                "learning_rate": base_params.get(
                    "learning_rate", self.config.learning_rate
                ),
                "l2_leaf_reg": base_params.get("l2_leaf_reg", 3.0),
                "random_seed": self.config.random_state,
                "verbose": False,
                **gpu_params,
            }

            if self.task == "classification":
                return cb.CatBoostClassifier(**model_params)  # type: ignore
            else:
                return cb.CatBoostRegressor(**model_params)  # type: ignore

        return None

    def auto_tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Automatically tune hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of optimization trials

        Returns:
            Dictionary with best parameters
        """
        try:
            from ml.tuning import HyperparameterOptimizer, GPUConfig as TuneGPUConfig
        except ImportError:
            logger.warning("Optuna tuning not available, using default parameters")
            return {}

        gpu_config = TuneGPUConfig(enabled=self.use_gpu)
        optimizer = HyperparameterOptimizer(
            n_trials=n_trials,
            gpu_config=gpu_config,
            random_state=self.config.random_state,
        )

        if self.model_type == "xgboost":
            result = optimizer.optimize_xgboost(X, y)
        elif self.model_type == "lightgbm":
            result = optimizer.optimize_lightgbm(X, y)
        elif self.model_type == "catboost":
            result = optimizer.optimize_catboost(X, y)
        else:
            logger.warning(f"Auto-tuning not supported for {self.model_type}")
            return {}

        self.best_params_ = result.best_params
        logger.info(f"Best parameters found: {self.best_params_}")
        logger.info(f"Best score: {result.best_score}")

        return self.best_params_

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
        use_time_series_split: bool = True,
        auto_tune: bool = False,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            features: Feature DataFrame
            target: Target Series
            validation_split: Fraction for validation
            use_time_series_split: Whether to use time-series aware split
            auto_tune: Whether to auto-tune hyperparameters first

        Returns:
            Dictionary of evaluation metrics
        """
        self._feature_names = features.columns.tolist()

        # Auto-tune if requested
        if auto_tune or self.config.auto_tune:
            self.auto_tune(features, target, self.config.tune_trials)

        # Create model with best params or defaults
        self._model = self._create_model(self.best_params_)

        if self._model is None:
            raise MLModelError("Failed to create model", model_name=self.name)

        # Split data
        if use_time_series_split:
            split_idx = int(len(features) * (1 - validation_split))
            X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_val = target.iloc[:split_idx], target.iloc[split_idx:]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                features, target, test_size=validation_split, shuffle=False
            )

        # Train with early stopping if supported
        if (
            self.model_type in ["xgboost", "lightgbm"]
            and self.config.early_stopping_rounds > 0
        ):
            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
            )
        elif self.model_type == "catboost":
            self._model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.config.early_stopping_rounds,
            )
        else:
            self._model.fit(X_train, y_train)

        self._is_trained = True

        # Evaluate
        metrics = self._evaluate(X_val, y_val)

        logger.info(f"Trained {self.name} with metrics: {metrics}")
        return metrics

    def _evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate the model on validation data."""
        metrics = {}

        if self.task == "classification":
            y_pred = self._model.predict(X)  # type: ignore
            y_proba = None

            if hasattr(self._model, "predict_proba"):
                y_proba = self._model.predict_proba(X)[:, 1]  # type: ignore

            metrics["accuracy"] = accuracy_score(y, y_pred)
            metrics["precision"] = precision_score(
                y, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(
                y, y_pred, average="weighted", zero_division=0
            )
            metrics["f1"] = f1_score(y, y_pred, average="weighted", zero_division=0)

            if y_proba is not None:
                try:
                    metrics["auc"] = roc_auc_score(y, y_proba)
                    metrics["log_loss"] = log_loss(y, y_proba)
                except Exception:
                    pass
        else:
            y_pred = self._model.predict(X)  # type: ignore
            metrics["mse"] = np.mean((y - y_pred) ** 2)
            metrics["mae"] = np.mean(np.abs(y - y_pred))
            metrics["rmse"] = np.sqrt(metrics["mse"])

        return metrics

    def cross_validate(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Perform time-series cross-validation.

        Args:
            features: Feature DataFrame
            target: Target Series
            n_splits: Number of CV splits

        Returns:
            Dictionary with mean and std of metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_metrics: Dict[str, List[float]] = {}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

            model = self._create_model(self.best_params_)
            if model is None:
                continue

            model.fit(X_train, y_train)

            if self.task == "classification":
                y_pred = model.predict(X_val)

                fold_metrics = {
                    "accuracy": accuracy_score(y_val, y_pred),
                    "precision": precision_score(
                        y_val, y_pred, average="weighted", zero_division=0
                    ),
                    "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0),
                }
            else:
                y_pred = model.predict(X_val)
                fold_metrics = {
                    "mse": np.mean((y_val - y_pred) ** 2),
                    "mae": np.mean(np.abs(y_val - y_pred)),
                }

            for key, value in fold_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Calculate mean and std
        results = {}
        for key, values in all_metrics.items():
            results[f"{key}_mean"] = np.mean(values)
            results[f"{key}_std"] = np.std(values)

        return results

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if not self._is_trained or self._model is None:
            raise MLModelError("Model not trained", model_name=self.name)

        missing = [f for f in self._feature_names if f not in features.columns]
        if missing:
            raise MLModelError(f"Missing features: {missing}", model_name=self.name)

        X = features[self._feature_names]
        predictions = self._model.predict(X)

        return pd.Series(predictions, index=features.index)

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities (classification only)."""
        if not self._is_trained or self._model is None:
            raise MLModelError("Model not trained", model_name=self.name)

        if self.task != "classification":
            raise MLModelError(
                "predict_proba only available for classification", model_name=self.name
            )

        X = features[self._feature_names]

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X)  # type: ignore
            if hasattr(self._model, "classes_"):
                classes = self._model.classes_  # type: ignore
                return pd.DataFrame(probs, columns=classes, index=features.index)
            else:
                return pd.DataFrame(probs, index=features.index)
        else:
            raise MLModelError(
                "Model does not support predict_proba", model_name=self.name
            )

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self._is_trained or self._model is None:
            return None

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            return dict(zip(self._feature_names, importances))
        return None


class MultiModelEnsemble:
    """
    Multi-model ensemble that combines predictions from multiple models.

    Uses weighted averaging or stacking meta-learner.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        use_stacking: bool = False,
        use_gpu: bool = True,
    ):
        """
        Initialize multi-model ensemble.

        Args:
            models: List of model types to include
            weights: Optional weights for each model
            use_stacking: Whether to use stacking meta-learner
            use_gpu: Whether to use GPU
        """
        self.model_types = models or ["xgboost", "lightgbm", "catboost"]
        self.weights = weights or {
            m: 1.0 / len(self.model_types) for m in self.model_types
        }
        self.use_stacking = use_stacking
        self.use_gpu = use_gpu

        self.models_: Dict[str, EnsemblePredictor] = {}
        self.meta_learner_ = None
        self._is_trained = False

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        auto_tune: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.

        Args:
            features: Feature DataFrame
            target: Target Series
            auto_tune: Whether to auto-tune each model

        Returns:
            Dictionary of metrics for each model
        """
        all_metrics = {}

        for model_type in self.model_types:
            logger.info(f"Training {model_type}...")

            config = EnsembleConfig(
                model_type=model_type,
                use_gpu=self.use_gpu,
                auto_tune=auto_tune,
            )

            model = EnsemblePredictor(model_type=model_type, config=config)
            metrics = model.train(features, target, auto_tune=auto_tune)

            self.models_[model_type] = model
            all_metrics[model_type] = metrics

        if self.use_stacking:
            self._train_stacking_meta_learner(features, target)

        self._is_trained = True
        return all_metrics

    def _train_stacking_meta_learner(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Train stacking meta-learner."""
        try:
            from ml.models.meta_learner import MetaLearner

            self.meta_learner_ = MetaLearner(use_gpu=self.use_gpu)
            self.meta_learner_.fit(features, target)
        except ImportError:
            logger.warning("Meta-learner not available, using weighted average")
            self.use_stacking = False

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """
        Get ensemble prediction probabilities.

        Args:
            features: Feature DataFrame

        Returns:
            Series of prediction probabilities
        """
        if not self._is_trained:
            raise MLModelError("Ensemble not trained", model_name="MultiModelEnsemble")

        if self.use_stacking and self.meta_learner_ is not None:
            return pd.Series(
                self.meta_learner_.predict_proba(features), index=features.index
            )

        # Weighted average
        all_preds = []
        total_weight = 0

        for model_type, model in self.models_.items():
            weight = self.weights.get(model_type, 1.0)
            proba_df = model.predict_proba(features)

            # Get positive class probability
            if proba_df.shape[1] > 1:
                proba = proba_df.iloc[:, 1]
            else:
                proba = proba_df.iloc[:, 0]

            all_preds.append(proba * weight)
            total_weight += weight

        ensemble_pred = sum(all_preds) / total_weight
        return ensemble_pred

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Make ensemble predictions."""
        proba = self.predict_proba(features)
        return (proba >= threshold).astype(int)
