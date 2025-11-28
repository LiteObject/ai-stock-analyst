"""
Machine Learning module for the AI Stock Analyst.

This module provides tools for:
- Feature engineering (technical, fundamental, sentiment)
- Model training and prediction
- Ensemble models (Random Forest, XGBoost)
"""

from ml.features.technical import TechnicalFeatures
from ml.models.ensemble import EnsemblePredictor
from ml.training.trainer import ModelTrainer

__all__ = [
    "TechnicalFeatures",
    "EnsemblePredictor",
    "ModelTrainer",
]
