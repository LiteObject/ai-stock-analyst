"""
Meta-Learning for Stacking Ensembles.

This module provides:
- Stacking ensemble with multiple base models
- Meta-learner training on out-of-fold predictions
- Confidence-weighted predictions
- Regime-adaptive model selection
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    lgb = None

try:
    import catboost as cb

    HAS_CB = True
except ImportError:
    HAS_CB = False
    cb = None

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    """Configuration for a base model in the ensemble."""

    name: str
    model_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    enabled: bool = True


@dataclass
class StackingResult:
    """Result of stacking ensemble training."""

    meta_model: Any
    base_models: Dict[str, Any]
    feature_importance: Dict[str, float]
    cv_score: float
    oof_predictions: np.ndarray


class MetaLearner:
    """
    Meta-learner for stacking ensemble.

    Trains multiple base models and combines their predictions
    using a meta-learner that learns optimal weights.
    """

    def __init__(
        self,
        base_configs: Optional[List[BaseModelConfig]] = None,
        meta_type: str = "logistic",
        n_cv_splits: int = 5,
        use_gpu: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize the meta-learner.

        Args:
            base_configs: List of base model configurations
            meta_type: Type of meta-learner (logistic, xgboost, lightgbm)
            n_cv_splits: Number of CV splits for stacking
            use_gpu: Whether to use GPU for training
            random_state: Random seed
        """
        self.base_configs = base_configs or self._default_configs()
        self.meta_type = meta_type
        self.n_cv_splits = n_cv_splits
        self.use_gpu = use_gpu
        self.random_state = random_state

        self.base_models_: Dict[str, List[Any]] = {}
        self.meta_model_: Any = None
        self.feature_importance_: Dict[str, float] = {}

    def _default_configs(self) -> List[BaseModelConfig]:
        """Get default base model configurations."""
        configs = []

        if HAS_XGB:
            configs.append(
                BaseModelConfig(
                    name="xgboost",
                    model_type="xgboost",
                    params={
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.1,
                    },
                )
            )

        if HAS_LGB:
            configs.append(
                BaseModelConfig(
                    name="lightgbm",
                    model_type="lightgbm",
                    params={
                        "n_estimators": 200,
                        "num_leaves": 63,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 0.1,
                    },
                )
            )

        if HAS_CB:
            configs.append(
                BaseModelConfig(
                    name="catboost",
                    model_type="catboost",
                    params={
                        "iterations": 200,
                        "depth": 6,
                        "learning_rate": 0.1,
                        "l2_leaf_reg": 3.0,
                    },
                )
            )

        return configs

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "MetaLearner":
        """
        Train the stacking ensemble.

        Uses out-of-fold predictions to train the meta-learner,
        avoiding data leakage.

        Args:
            X: Feature matrix
            y: Target variable
            eval_set: Optional validation set

        Returns:
            Self
        """
        logger.info("Training stacking ensemble...")

        # Generate out-of-fold predictions
        oof_preds = self._generate_oof_predictions(X, y)

        # Create meta-features
        meta_X = self._create_meta_features(oof_preds, X)

        # Train meta-learner
        logger.info(f"Training meta-learner: {self.meta_type}")
        self.meta_model_ = self._train_meta_learner(meta_X, y)

        # Train final base models on full data
        logger.info("Training final base models on full data...")
        self._train_final_base_models(X, y)

        # Calculate feature importance
        self._calculate_importance(meta_X)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the stacking ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        if self.meta_model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = {}
        for name, models in self.base_models_.items():
            # Average predictions across CV models
            preds = []
            for model in models:
                pred = self._get_prediction(model, X, name)
                preds.append(pred)
            base_preds[name] = np.mean(preds, axis=0)

        # Create meta-features
        meta_X = self._create_meta_features(base_preds, X)

        # Get meta-learner prediction
        if hasattr(self.meta_model_, "predict_proba"):
            proba = self.meta_model_.predict_proba(meta_X)
            return proba[:, 1] if proba.ndim > 1 else proba
        else:
            return self.meta_model_.predict(meta_X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def _generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, np.ndarray]:
        """Generate out-of-fold predictions for meta-learner training."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        oof_preds = {
            config.name: np.zeros(len(X))
            for config in self.base_configs
            if config.enabled
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.n_cv_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for config in self.base_configs:
                if not config.enabled:
                    continue

                # Train base model
                model = self._create_model(config)
                model = self._fit_model(
                    model, X_train, y_train, X_val, y_val, config.name
                )

                # Store model for later
                if config.name not in self.base_models_:
                    self.base_models_[config.name] = []
                self.base_models_[config.name].append(model)

                # Get OOF prediction
                pred = self._get_prediction(model, X_val, config.name)
                oof_preds[config.name][val_idx] = pred

        return oof_preds

    def _create_model(self, config: BaseModelConfig) -> Any:
        """Create a model based on configuration."""
        params = config.params.copy()

        if config.model_type == "xgboost":
            if not HAS_XGB:
                raise ImportError("XGBoost not available")

            assert xgb is not None  # Type narrowing for Pylance
            if self.use_gpu:
                params["tree_method"] = "gpu_hist"
                params["device"] = "cuda:0"
            else:
                params["tree_method"] = "hist"

            params["random_state"] = self.random_state
            params["objective"] = "binary:logistic"

            return xgb.XGBClassifier(**params)

        elif config.model_type == "lightgbm":
            if not HAS_LGB:
                raise ImportError("LightGBM not available")

            assert lgb is not None  # Type narrowing for Pylance
            if self.use_gpu:
                params["device"] = "gpu"
            else:
                params["device"] = "cpu"

            params["random_state"] = self.random_state
            params["objective"] = "binary"
            params["verbosity"] = -1

            return lgb.LGBMClassifier(**params)

        elif config.model_type == "catboost":
            if not HAS_CB:
                raise ImportError("CatBoost not available")

            assert cb is not None  # Type narrowing for Pylance
            if self.use_gpu:
                params["task_type"] = "GPU"
            else:
                params["task_type"] = "CPU"

            params["random_seed"] = self.random_state
            params["verbose"] = False

            return cb.CatBoostClassifier(**params)

        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def _fit_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
    ) -> Any:
        """Fit a base model."""
        if model_name in ["xgboost", "lightgbm"]:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
            )
        elif model_name == "catboost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
            )
        else:
            model.fit(X_train, y_train)

        return model

    def _get_prediction(
        self,
        model: Any,
        X: pd.DataFrame,
        model_name: str,
    ) -> np.ndarray:
        """Get prediction probability from a model."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return proba[:, 1] if proba.ndim > 1 else proba
        else:
            return model.predict(X)

    def _create_meta_features(
        self,
        base_preds: Dict[str, np.ndarray],
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Create meta-features from base model predictions."""
        # Stack base predictions
        pred_matrix = np.column_stack(
            [base_preds[name] for name in sorted(base_preds.keys())]
        )

        # Add interaction features
        if pred_matrix.shape[1] >= 2:
            # Average prediction
            avg_pred = pred_matrix.mean(axis=1, keepdims=True)

            # Prediction variance (disagreement)
            var_pred = pred_matrix.var(axis=1, keepdims=True)

            # Min and max predictions
            min_pred = pred_matrix.min(axis=1, keepdims=True)
            max_pred = pred_matrix.max(axis=1, keepdims=True)

            # Range
            range_pred = max_pred - min_pred

            meta_features = np.hstack(
                [
                    pred_matrix,
                    avg_pred,
                    var_pred,
                    min_pred,
                    max_pred,
                    range_pred,
                ]
            )
        else:
            meta_features = pred_matrix

        return meta_features

    def _train_meta_learner(
        self,
        meta_X: np.ndarray,
        y: pd.Series,
    ) -> Any:
        """Train the meta-learner on OOF predictions."""
        # Remove NaN rows (from first fold)
        valid_mask = ~np.isnan(meta_X).any(axis=1)
        meta_X_clean = meta_X[valid_mask]
        y_clean = y.iloc[valid_mask]

        if self.meta_type == "logistic":
            model = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            )
            model.fit(meta_X_clean, y_clean)

        elif self.meta_type == "ridge":
            model = RidgeClassifier(
                alpha=1.0,
                random_state=self.random_state,
            )
            model.fit(meta_X_clean, y_clean)

        elif self.meta_type == "xgboost":
            if not HAS_XGB:
                raise ImportError("XGBoost not available")

            assert xgb is not None  # Type narrowing for Pylance
            params = {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": self.random_state,
            }
            if self.use_gpu:
                params["tree_method"] = "gpu_hist"

            model = xgb.XGBClassifier(**params)
            model.fit(meta_X_clean, y_clean)

        elif self.meta_type == "lightgbm":
            if not HAS_LGB:
                raise ImportError("LightGBM not available")

            assert lgb is not None  # Type narrowing for Pylance
            params = {
                "n_estimators": 50,
                "num_leaves": 15,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": self.random_state,
                "verbosity": -1,
            }
            if self.use_gpu:
                params["device"] = "gpu"

            model = lgb.LGBMClassifier(**params)
            model.fit(meta_X_clean, y_clean)

        else:
            raise ValueError(f"Unknown meta type: {self.meta_type}")

        return model

    def _train_final_base_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train final base models on full data."""
        # Clear CV models and train on full data
        for config in self.base_configs:
            if not config.enabled:
                continue

            model = self._create_model(config)
            model.fit(X, y)

            # Replace CV models with single full model
            self.base_models_[config.name] = [model]

    def _calculate_importance(self, meta_X: np.ndarray) -> None:
        """Calculate feature importance for meta-learner."""
        if hasattr(self.meta_model_, "coef_"):
            # Linear model
            coefs = np.abs(self.meta_model_.coef_).flatten()
            n_base = len([c for c in self.base_configs if c.enabled])

            for i, config in enumerate([c for c in self.base_configs if c.enabled]):
                if i < len(coefs):
                    self.feature_importance_[config.name] = float(coefs[i])

        elif hasattr(self.meta_model_, "feature_importances_"):
            importances = self.meta_model_.feature_importances_

            for i, config in enumerate([c for c in self.base_configs if c.enabled]):
                if i < len(importances):
                    self.feature_importance_[config.name] = float(importances[i])


class RegimeAdaptiveMetaLearner(MetaLearner):
    """
    Meta-learner that adapts to market regimes.

    Trains separate meta-learners for different market regimes
    and selects the appropriate one at prediction time.
    """

    def __init__(
        self,
        base_configs: Optional[List[BaseModelConfig]] = None,
        meta_type: str = "logistic",
        n_cv_splits: int = 5,
        use_gpu: bool = True,
        random_state: int = 42,
    ):
        super().__init__(base_configs, meta_type, n_cv_splits, use_gpu, random_state)
        self.regime_models_: Dict[str, Any] = {}

    def fit_with_regimes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.Series,
    ) -> "RegimeAdaptiveMetaLearner":
        """
        Train regime-specific meta-learners.

        Args:
            X: Feature matrix
            y: Target variable
            regimes: Regime labels for each sample

        Returns:
            Self
        """
        unique_regimes = regimes.unique()

        for regime in unique_regimes:
            logger.info(f"Training meta-learner for regime: {regime}")

            mask = regimes == regime
            X_regime = X[mask]
            y_regime = y[mask]

            if len(X_regime) < 100:
                logger.warning(f"Skipping regime {regime}: insufficient data")
                continue

            # Create and train regime-specific model
            regime_model = MetaLearner(
                base_configs=self.base_configs,
                meta_type=self.meta_type,
                n_cv_splits=self.n_cv_splits,
                use_gpu=self.use_gpu,
                random_state=self.random_state,
            )
            regime_model.fit(X_regime, y_regime)

            self.regime_models_[str(regime)] = regime_model

        # Also train a default model on all data
        self.fit(X, y)

        return self

    def predict_proba_with_regime(
        self,
        X: pd.DataFrame,
        regime: str,
    ) -> np.ndarray:
        """
        Predict using regime-specific model.

        Args:
            X: Feature matrix
            regime: Current market regime

        Returns:
            Predicted probabilities
        """
        if regime in self.regime_models_:
            return self.regime_models_[regime].predict_proba(X)
        else:
            return self.predict_proba(X)
