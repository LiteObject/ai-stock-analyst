"""
Ensemble ML models.

Implements Random Forest and XGBoost predictors.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from core.exceptions import MLModelError
from ml.models.base import BaseMLPredictor

logger = logging.getLogger(__name__)


class EnsemblePredictor(BaseMLPredictor):
    """
    Ensemble predictor using Random Forest or XGBoost.
    """

    def __init__(self, model_type: str = "random_forest", task: str = "classification"):
        """
        Initialize.

        Args:
            model_type: 'random_forest' or 'xgboost'
            task: 'classification' (direction) or 'regression' (returns)
        """
        super().__init__(name=f"{model_type}_{task}")
        self.model_type = model_type
        self.task = task

        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to Random Forest")
            self.model_type = "random_forest"

    def _create_model(self):
        """Create the underlying sklearn/xgb model."""
        if self.model_type == "random_forest":
            if self.task == "classification":
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                )
        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            if self.task == "classification":
                return xgb.XGBClassifier(  # type: ignore[possibly-undefined]
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                return xgb.XGBRegressor(  # type: ignore[possibly-undefined]
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                )
        return None

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """Train the model."""
        self._feature_names = features.columns.tolist()
        self._model = self._create_model()

        if self._model is None:
            raise MLModelError("Failed to create model", model_name=self.name)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=validation_split, shuffle=False)

        # Train
        self._model.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate
        metrics = {}
        if self.task == "classification":
            y_pred = self._model.predict(X_val)
            metrics["accuracy"] = accuracy_score(y_val, y_pred)
            metrics["precision"] = precision_score(y_val, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_val, y_pred, average="weighted", zero_division=0)
            metrics["f1"] = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        else:
            y_pred = self._model.predict(X_val)
            metrics["mse"] = np.mean((y_val - y_pred) ** 2)
            metrics["mae"] = np.mean(np.abs(y_val - y_pred))

        logger.info(f"Trained {self.name} with metrics: {metrics}")
        return metrics

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        if not self._is_trained or self._model is None:
            raise MLModelError("Model not trained", model_name=self.name)

        # Ensure features match
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
            raise MLModelError("predict_proba only available for classification", model_name=self.name)

        X = features[self._feature_names]

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(X)  # type: ignore[union-attr]
            # Assuming binary classification or multi-class, return as DataFrame
            if hasattr(self._model, "classes_"):
                classes = self._model.classes_  # type: ignore[union-attr]
                return pd.DataFrame(probs, columns=classes, index=features.index)
            else:
                return pd.DataFrame(probs, index=features.index)

        else:
            raise MLModelError("Model does not support predict_proba", model_name=self.name)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self._is_trained or self._model is None:
            return None

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            return dict(zip(self._feature_names, importances))
        return None
