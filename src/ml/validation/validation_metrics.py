"""
Validation metrics for ML trading models.

Comprehensive metrics beyond basic accuracy:
- Directional accuracy
- Signal quality metrics
- Confidence calibration
- Prediction stability
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Complete validation metrics for a trading model."""

    # Basic classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Directional metrics (most important for trading)
    directional_accuracy: float
    up_accuracy: float
    down_accuracy: float

    # Signal quality
    hit_ratio: float  # % of correct directional predictions
    profit_ratio: float  # Avg profit on hits / avg loss on misses
    information_coefficient: float  # Correlation with returns

    # Confidence metrics
    calibration_error: float  # How well probabilities match actual outcomes
    brier_score: float
    log_loss: float

    # Stability metrics
    prediction_stability: float  # Consistency across time
    feature_stability: float  # Stability of feature importance

    # AUC metrics
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


def calculate_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate basic prediction metrics.

    Args:
        y_true: True labels (0/1 or -1/1)
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        log_loss,
        brier_score_loss,
    )

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    # Probability-based metrics
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = np.nan

        try:
            metrics["log_loss"] = log_loss(y_true, y_prob)
        except ValueError:
            metrics["log_loss"] = np.nan

        try:
            metrics["brier_score"] = brier_score_loss(y_true, y_prob)
        except ValueError:
            metrics["brier_score"] = np.nan

    return metrics


def calculate_directional_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    actual_directions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate directional accuracy metrics for trading.

    Args:
        returns: Actual returns following predictions
        predictions: Predicted direction (1=up, -1=down, 0=neutral)
        actual_directions: Actual direction labels (optional)

    Returns:
        Dictionary of directional metrics
    """
    metrics = {}

    # Calculate actual directions if not provided
    if actual_directions is None:
        actual_directions = np.sign(returns)

    # Overall directional accuracy
    non_zero = predictions != 0
    if non_zero.sum() > 0:
        correct = np.sign(predictions[non_zero]) == actual_directions[non_zero]
        metrics["directional_accuracy"] = correct.mean()
    else:
        metrics["directional_accuracy"] = 0.5

    # Up accuracy (when predicting up)
    up_preds = predictions > 0
    if up_preds.sum() > 0:
        up_correct = actual_directions[up_preds] > 0
        metrics["up_accuracy"] = up_correct.mean()
    else:
        metrics["up_accuracy"] = 0.5

    # Down accuracy (when predicting down)
    down_preds = predictions < 0
    if down_preds.sum() > 0:
        down_correct = actual_directions[down_preds] < 0
        metrics["down_accuracy"] = down_correct.mean()
    else:
        metrics["down_accuracy"] = 0.5

    # Hit ratio
    metrics["hit_ratio"] = metrics["directional_accuracy"]

    # Profit ratio
    correct_mask = np.sign(predictions) == actual_directions
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        avg_profit_on_hits = np.abs(returns[correct_mask]).mean()
        avg_loss_on_misses = np.abs(returns[~correct_mask]).mean()
        if avg_loss_on_misses > 0:
            metrics["profit_ratio"] = avg_profit_on_hits / avg_loss_on_misses
        else:
            metrics["profit_ratio"] = float("inf")
    else:
        metrics["profit_ratio"] = 1.0

    # Information Coefficient (rank correlation)
    if len(predictions) > 2:
        try:
            result = stats.spearmanr(predictions, returns)
            # spearmanr returns SpearmanrResult - use getattr for safe access
            ic_correlation: float = 0.0
            if hasattr(result, "statistic"):
                ic_correlation = float(getattr(result, "statistic"))
            elif hasattr(result, "correlation"):
                ic_correlation = float(getattr(result, "correlation"))
            else:
                # Fallback for tuple return
                ic_correlation = float(result[0])  # type: ignore[arg-type]
            metrics["information_coefficient"] = (
                ic_correlation if not np.isnan(ic_correlation) else 0.0
            )
        except Exception:
            metrics["information_coefficient"] = 0.0
    else:
        metrics["information_coefficient"] = 0.0

    # Confusion-based metrics
    tp = ((predictions > 0) & (actual_directions > 0)).sum()
    tn = ((predictions < 0) & (actual_directions < 0)).sum()
    fp = ((predictions > 0) & (actual_directions <= 0)).sum()
    fn = ((predictions < 0) & (actual_directions >= 0)).sum()

    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    # Matthews Correlation Coefficient
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom > 0:
        metrics["mcc"] = (tp * tn - fp * fn) / denom
    else:
        metrics["mcc"] = 0.0

    return metrics


def calculate_confidence_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Calculate confidence calibration metrics.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        Dictionary of calibration metrics
    """
    metrics = {}

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0  # Maximum Calibration Error

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = in_bin.sum()

        if bin_size > 0:
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_prob[in_bin].mean()
            bin_error = abs(bin_accuracy - bin_confidence)

            ece += (bin_size / len(y_true)) * bin_error
            mce = max(mce, bin_error)

    metrics["expected_calibration_error"] = ece
    metrics["max_calibration_error"] = mce

    # Reliability (how often high-confidence predictions are correct)
    high_conf_mask = (y_prob > 0.7) | (y_prob < 0.3)
    if high_conf_mask.sum() > 0:
        high_conf_preds = (y_prob[high_conf_mask] > 0.5).astype(int)
        high_conf_correct = high_conf_preds == y_true[high_conf_mask]
        metrics["high_confidence_accuracy"] = high_conf_correct.mean()
    else:
        metrics["high_confidence_accuracy"] = 0.0

    # Sharpness (spread of predictions)
    metrics["sharpness"] = y_prob.std()

    # Resolution
    metrics["resolution"] = abs(y_prob.mean() - 0.5)

    return metrics


def calculate_stability_metrics(
    predictions_over_time: List[np.ndarray],
    feature_importances: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Calculate stability metrics across time periods.

    Args:
        predictions_over_time: List of prediction arrays from different periods
        feature_importances: List of feature importance arrays (optional)

    Returns:
        Dictionary of stability metrics
    """
    metrics: Dict[str, float] = {}

    # Prediction stability (correlation between consecutive periods)
    if len(predictions_over_time) >= 2:
        correlations: List[float] = []
        for i in range(1, len(predictions_over_time)):
            prev = predictions_over_time[i - 1]
            curr = predictions_over_time[i]

            # Align lengths if different
            min_len = min(len(prev), len(curr))
            if min_len > 1:
                try:
                    result = stats.pearsonr(prev[:min_len], curr[:min_len])
                    corr_val = float(result[0])  # type: ignore[arg-type]
                    if not np.isnan(corr_val):
                        correlations.append(corr_val)
                except Exception:
                    pass

        if correlations:
            metrics["prediction_stability"] = float(np.mean(correlations))
            metrics["prediction_stability_std"] = float(np.std(correlations))
        else:
            metrics["prediction_stability"] = 0.0
            metrics["prediction_stability_std"] = 1.0
    else:
        metrics["prediction_stability"] = 0.0
        metrics["prediction_stability_std"] = 1.0

    # Feature importance stability
    if feature_importances and len(feature_importances) >= 2:
        fi_correlations: List[float] = []
        for i in range(1, len(feature_importances)):
            prev = feature_importances[i - 1]
            curr = feature_importances[i]

            if len(prev) == len(curr) and len(prev) > 1:
                try:
                    result = stats.spearmanr(prev, curr)
                    corr_val = float(result[0])  # type: ignore[arg-type]
                    if not np.isnan(corr_val):
                        fi_correlations.append(corr_val)
                except Exception:
                    pass

        if fi_correlations:
            metrics["feature_stability"] = float(np.mean(fi_correlations))
            metrics["feature_stability_std"] = float(np.std(fi_correlations))
        else:
            metrics["feature_stability"] = 0.0
            metrics["feature_stability_std"] = 1.0
    else:
        metrics["feature_stability"] = 0.0
        metrics["feature_stability_std"] = 1.0

    return metrics


def calculate_trading_metrics(
    signals: np.ndarray,
    returns: np.ndarray,
    positions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.

    Args:
        signals: Trading signals (1, -1, 0)
        returns: Subsequent returns
        positions: Position sizes (optional, defaults to signal)

    Returns:
        Dictionary of trading metrics
    """
    if positions is None:
        positions = signals

    # Strategy returns
    strategy_returns = positions * returns

    metrics = {}

    # Win rate
    trades = positions != 0
    if trades.sum() > 0:
        wins = strategy_returns[trades] > 0
        metrics["win_rate"] = wins.mean()
    else:
        metrics["win_rate"] = 0.0

    # Average win/loss
    winning_trades = strategy_returns > 0
    losing_trades = strategy_returns < 0

    if winning_trades.sum() > 0:
        metrics["avg_win"] = strategy_returns[winning_trades].mean()
    else:
        metrics["avg_win"] = 0.0

    if losing_trades.sum() > 0:
        metrics["avg_loss"] = strategy_returns[losing_trades].mean()
    else:
        metrics["avg_loss"] = 0.0

    # Profit factor
    gross_profit = (
        strategy_returns[winning_trades].sum() if winning_trades.sum() > 0 else 0
    )
    gross_loss = (
        abs(strategy_returns[losing_trades].sum()) if losing_trades.sum() > 0 else 0
    )

    if gross_loss > 0:
        metrics["profit_factor"] = gross_profit / gross_loss
    else:
        metrics["profit_factor"] = float("inf") if gross_profit > 0 else 1.0

    # Expectancy
    if trades.sum() > 0:
        metrics["expectancy"] = strategy_returns[trades].mean()
    else:
        metrics["expectancy"] = 0.0

    # Sharpe ratio (annualized, assuming daily data)
    if len(strategy_returns) > 1 and strategy_returns.std() > 0:
        metrics["sharpe_ratio"] = (
            strategy_returns.mean() / strategy_returns.std()
        ) * np.sqrt(252)
    else:
        metrics["sharpe_ratio"] = 0.0

    # Maximum drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    metrics["max_drawdown"] = drawdown.min()

    # Calmar ratio
    if metrics["max_drawdown"] != 0:
        annualized_return = strategy_returns.mean() * 252
        metrics["calmar_ratio"] = annualized_return / abs(metrics["max_drawdown"])
    else:
        metrics["calmar_ratio"] = 0.0

    # Sortino ratio
    downside = strategy_returns[strategy_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        metrics["sortino_ratio"] = (strategy_returns.mean() / downside.std()) * np.sqrt(
            252
        )
    else:
        metrics["sortino_ratio"] = 0.0

    return metrics


def aggregate_validation_metrics(
    fold_metrics: List[Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Aggregate metrics across cross-validation folds.

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        Tuple of (mean_metrics, std_metrics)
    """
    if not fold_metrics:
        return {}, {}

    all_keys = set()
    for fm in fold_metrics:
        all_keys.update(fm.keys())

    mean_metrics = {}
    std_metrics = {}

    for key in all_keys:
        values = []
        for fm in fold_metrics:
            if key in fm and not np.isnan(fm[key]) and not np.isinf(fm[key]):
                values.append(fm[key])

        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        else:
            mean_metrics[key] = np.nan
            std_metrics[key] = np.nan

    return mean_metrics, std_metrics
