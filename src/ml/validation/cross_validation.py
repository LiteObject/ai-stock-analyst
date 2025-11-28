"""
Time-series appropriate cross-validation strategies.

Implements cross-validation methods that respect temporal ordering and
avoid look-ahead bias in financial time series.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResult:
    """Result from cross-validation."""

    fold_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    oof_predictions: Optional[np.ndarray] = None
    oof_indices: Optional[np.ndarray] = None
    models: List[Any] = field(default_factory=list)
    feature_importance: Optional[pd.DataFrame] = None
    metric_stability: Dict[str, float] = field(default_factory=dict)
    is_stable: bool = True


class TimeSeriesCV(BaseCrossValidator):
    """Expanding window time-series cross-validation."""

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # type: ignore[override]
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # type: ignore[override]
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = self.test_size or n_samples // (self.n_splits + 1)
        min_train = self.min_train_size or test_size

        for fold in range(self.n_splits):
            test_start = min_train + (fold * test_size) + self.gap
            test_end = test_start + test_size
            if test_end > n_samples:
                break
            train_end = test_start - self.gap
            train_start = (
                max(0, train_end - self.max_train_size) if self.max_train_size else 0
            )
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            if len(train_indices) >= min_train:
                yield train_indices, test_indices


class PurgedKFold(BaseCrossValidator):
    """K-Fold CV with purging and embargo for financial data."""

    def __init__(
        self,
        n_splits: int = 5,
        purge_periods: int = 5,
        embargo_periods: int = 5,
        shuffle: bool = False,
    ):
        self.n_splits = n_splits
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods
        self.shuffle = shuffle

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # type: ignore[override]
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # type: ignore[override]
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        fold_size = n_samples // self.n_splits

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            purge_start = max(0, test_start - self.purge_periods)
            embargo_end = min(n_samples, test_end + self.embargo_periods)
            test_indices = indices[test_start:test_end]
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False
            train_indices = indices[train_mask]
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class BlockingTimeSeriesCV(BaseCrossValidator):
    """Blocking time-series CV with non-overlapping blocks."""

    def __init__(self, n_blocks: int = 5, gap: int = 0):
        self.n_blocks = n_blocks
        self.gap = gap

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # type: ignore[override]
        return self.n_blocks

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # type: ignore[override]
        n_samples = len(X)
        indices = np.arange(n_samples)
        block_size = n_samples // self.n_blocks
        blocks: List[np.ndarray] = []
        for i in range(self.n_blocks):
            start = i * block_size
            end = (i + 1) * block_size if i < self.n_blocks - 1 else n_samples
            blocks.append(indices[start:end])

        for test_block_idx in range(self.n_blocks):
            test_indices = blocks[test_block_idx]
            train_blocks: List[np.ndarray] = []
            for i, block in enumerate(blocks):
                if i == test_block_idx:
                    continue
                if abs(i - test_block_idx) == 1 and self.gap > 0:
                    block = (
                        block[: -self.gap] if i < test_block_idx else block[self.gap :]
                    )
                    if len(block) == 0:
                        continue
                train_blocks.append(block)
            if train_blocks:
                train_indices = np.concatenate(train_blocks)
                yield train_indices, test_indices


class MonteCarloCV(BaseCrossValidator):
    """Monte Carlo cross-validation for time series."""

    def __init__(
        self,
        n_splits: int = 10,
        train_ratio: float = 0.7,
        gap: int = 0,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap = gap
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:  # type: ignore[override]
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # type: ignore[override]
        n_samples = len(X)
        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.random_state)
        train_size = int(n_samples * self.train_ratio)
        test_size = n_samples - train_size - self.gap
        if test_size <= 0:
            raise ValueError("train_ratio too high or gap too large")

        for _ in range(self.n_splits):
            max_split = n_samples - test_size
            split_point = rng.randint(train_size, max_split + 1)
            train_start = max(0, split_point - train_size)
            train_indices = indices[train_start:split_point]
            test_start = split_point + self.gap
            test_end = min(n_samples, test_start + test_size)
            test_indices = indices[test_start:test_end]
            yield train_indices, test_indices


def cross_validate_model(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: BaseCrossValidator,
    scoring: Optional[Dict[str, Callable[[Any, Any], float]]] = None,
    return_estimator: bool = False,
    return_predictions: bool = True,
) -> CrossValidationResult:
    """Perform cross-validation with comprehensive metrics."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.base import clone

    if scoring is None:
        # Use explicit typing to avoid invariance issues
        default_scoring: Dict[str, Callable[[Any, Any], float]] = {
            "accuracy": lambda y_t, y_p: float(accuracy_score(y_t, y_p)),
            "f1": lambda y_t, y_p: float(
                f1_score(y_t, y_p, average="binary", zero_division=0)
            ),
        }
        scoring = default_scoring

    fold_results: List[Dict[str, float]] = []
    models: List[Any] = []
    n_samples = len(y)

    oof_predictions: Optional[np.ndarray] = (
        np.zeros(n_samples) if return_predictions else None
    )
    oof_indices: Optional[np.ndarray] = (
        np.zeros(n_samples, dtype=int) if return_predictions else None
    )
    feature_importance_list: List[pd.DataFrame] = []

    feature_names: List[str] = (
        X.columns.tolist()
        if isinstance(X, pd.DataFrame)
        else [f"feature_{i}" for i in range(X.shape[1] if hasattr(X, "shape") else 0)]
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        if hasattr(fold_model, "predict_proba"):
            y_pred_proba = fold_model.predict_proba(X_test)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = fold_model.predict(X_test)
            y_pred_proba = y_pred.astype(float)

        fold_metrics: Dict[str, float] = {}
        for metric_name, metric_fn in scoring.items():
            try:
                fold_metrics[metric_name] = float(metric_fn(y_test, y_pred))
            except Exception as e:
                logger.warning(f"Could not calculate {metric_name}: {e}")
                fold_metrics[metric_name] = float("nan")

        fold_results.append(fold_metrics)

        if (
            return_predictions
            and oof_predictions is not None
            and oof_indices is not None
        ):
            oof_predictions[test_idx] = y_pred_proba
            oof_indices[test_idx] = fold_idx + 1

        if return_estimator:
            models.append(fold_model)

        if hasattr(fold_model, "feature_importances_"):
            importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": fold_model.feature_importances_,
                    "fold": fold_idx,
                }
            )
            feature_importance_list.append(importance)

        logger.info(f"Fold {fold_idx + 1}: {fold_metrics}")

    metric_names = list(fold_results[0].keys()) if fold_results else []
    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}

    for metric in metric_names:
        values = [
            f[metric] for f in fold_results if not np.isnan(f.get(metric, float("nan")))
        ]
        mean_metrics[metric] = float(np.mean(values)) if values else float("nan")
        std_metrics[metric] = float(np.std(values)) if values else float("nan")

    metric_stability: Dict[str, float] = {}
    for metric in metric_names:
        if mean_metrics[metric] != 0 and not np.isnan(mean_metrics[metric]):
            metric_stability[metric] = std_metrics[metric] / abs(mean_metrics[metric])
        else:
            metric_stability[metric] = float("inf")

    is_stable = all(v < 0.3 for v in metric_stability.values() if not np.isinf(v))

    feature_importance: Optional[pd.DataFrame] = None
    if feature_importance_list:
        feature_importance = pd.concat(feature_importance_list)
        feature_importance = (
            feature_importance.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

    return CrossValidationResult(
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        oof_predictions=oof_predictions,
        oof_indices=oof_indices,
        models=models if return_estimator else [],
        feature_importance=feature_importance,
        metric_stability=metric_stability,
        is_stable=is_stable,
    )
