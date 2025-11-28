"""
ML Tuning Module.

This module provides hyperparameter optimization and threshold tuning:
- GPU-accelerated hyperparameter search with Optuna
- Optimal threshold detection for trading signals
"""

from .hyperparameter import (
    GPUConfig,
    HyperparameterOptimizer,
    OptimizationResult,
    quick_optimize,
)
from .threshold import (
    ThresholdOptimizer,
    ThresholdResult,
    optimize_threshold_for_trading,
)

__all__ = [
    # Hyperparameter optimization
    "HyperparameterOptimizer",
    "OptimizationResult",
    "GPUConfig",
    "quick_optimize",
    # Threshold optimization
    "ThresholdOptimizer",
    "ThresholdResult",
    "optimize_threshold_for_trading",
]
