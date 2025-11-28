"""
Machine Learning module for the AI Stock Analyst.

This module provides tools for:
- Feature engineering (technical, fundamental, sentiment, regime, cross-asset)
- Model training and prediction with GPU acceleration
- Ensemble models (Random Forest, XGBoost, LightGBM, CatBoost)
- Meta-learning and model stacking
- Hyperparameter optimization with Optuna
- Advanced cross-validation strategies
- Comprehensive ML validation metrics
"""

# Features
from ml.features.technical import TechnicalFeatures
from ml.features.advanced import AdvancedFeatureEngineer
from ml.features.regime import RegimeDetector
from ml.features.cross_asset import CrossAssetFeatures

# Models
from ml.models.ensemble import EnsemblePredictor
from ml.models.meta_learner import MetaLearner

# Training
from ml.training.trainer import ModelTrainer

# Tuning
from ml.tuning.hyperparameter import HyperparameterOptimizer
from ml.tuning.threshold import ThresholdOptimizer

# Validation
from ml.validation.cross_validation import (
    TimeSeriesSplit,
    PurgedKFold,
    CombinatorialPurgedCV,
    RegimeStratifiedCV,
    CrossValidator,
    cross_validate_model,
)
from ml.validation.validation_metrics import (
    ValidationMetrics,
    calculate_prediction_metrics,
    calculate_directional_metrics,
    calculate_confidence_metrics,
    calculate_trading_metrics,
    calculate_stability_metrics,
    calculate_overfitting_score,
)
from ml.validation.reporter import ValidationReporter

__all__ = [
    # Features
    "TechnicalFeatures",
    "AdvancedFeatureEngineer",
    "RegimeDetector",
    "CrossAssetFeatures",
    # Models
    "EnsemblePredictor",
    "MetaLearner",
    # Training
    "ModelTrainer",
    # Tuning
    "HyperparameterOptimizer",
    "ThresholdOptimizer",
    # Validation - CV strategies
    "TimeSeriesSplit",
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "RegimeStratifiedCV",
    "CrossValidator",
    "cross_validate_model",
    # Validation - Metrics
    "ValidationMetrics",
    "calculate_prediction_metrics",
    "calculate_directional_metrics",
    "calculate_confidence_metrics",
    "calculate_trading_metrics",
    "calculate_stability_metrics",
    "calculate_overfitting_score",
    # Validation - Reporter
    "ValidationReporter",
]
