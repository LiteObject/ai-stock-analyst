"""
ML Validation module.

Provides time-series appropriate cross-validation and validation metrics.
"""

from src.ml.validation.cross_validation import (
    TimeSeriesCV,
    PurgedKFold,
    BlockingTimeSeriesCV,
    MonteCarloCV,
    CrossValidationResult,
)
from src.ml.validation.validation_metrics import (
    ValidationMetrics,
    calculate_prediction_metrics,
    calculate_directional_metrics,
    calculate_confidence_metrics,
    calculate_stability_metrics,
)
from src.ml.validation.reporter import (
    ValidationReporter,
    ValidationReport,
    generate_validation_summary,
)

__all__ = [
    # Cross-validation
    "TimeSeriesCV",
    "PurgedKFold",
    "BlockingTimeSeriesCV",
    "MonteCarloCV",
    "CrossValidationResult",
    # Validation Metrics
    "ValidationMetrics",
    "calculate_prediction_metrics",
    "calculate_directional_metrics",
    "calculate_confidence_metrics",
    "calculate_stability_metrics",
    # Reporting
    "ValidationReporter",
    "ValidationReport",
    "generate_validation_summary",
]
