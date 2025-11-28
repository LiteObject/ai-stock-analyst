"""
Threshold Optimization for Trading Predictions.

This module optimizes prediction thresholds for:
- Maximum profit factor
- Maximum Sharpe ratio
- Minimum false positive rate
- Trading cost awareness
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""

    optimal_threshold: float
    buy_threshold: float
    sell_threshold: float
    metrics: Dict[str, float]
    confusion: np.ndarray
    threshold_curve: Optional[pd.DataFrame] = None


class ThresholdOptimizer:
    """
    Optimizes prediction thresholds for trading signals.

    Instead of using default 0.5 threshold, finds optimal thresholds
    that maximize trading performance metrics.
    """

    def __init__(
        self,
        trading_cost: float = 0.001,
        min_confidence: float = 0.55,
        max_positions: int = 100,
    ):
        """
        Initialize the threshold optimizer.

        Args:
            trading_cost: Transaction cost as fraction of trade value
            min_confidence: Minimum prediction confidence to consider
            max_positions: Maximum simultaneous positions for scaling
        """
        self.trading_cost = trading_cost
        self.min_confidence = min_confidence
        self.max_positions = max_positions

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        returns: Optional[np.ndarray] = None,
        objective: str = "profit_factor",
    ) -> ThresholdResult:
        """
        Find optimal prediction threshold.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            returns: Actual returns for each prediction (optional)
            objective: Optimization objective
                - profit_factor: Maximize profit factor
                - sharpe: Maximize Sharpe ratio
                - f1: Maximize F1 score
                - accuracy: Maximize accuracy
                - precision: Maximize precision (minimize false positives)

        Returns:
            ThresholdResult with optimal thresholds and metrics
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        if returns is not None:
            returns = np.array(returns)

        # Generate threshold curve
        thresholds = np.linspace(0.3, 0.7, 81)
        results = []

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = self._calculate_metrics(
                y_true, y_pred, y_prob, returns, threshold
            )
            metrics["threshold"] = threshold
            results.append(metrics)

        threshold_df = pd.DataFrame(results)

        # Find optimal threshold based on objective
        if objective == "profit_factor":
            optimal_idx = threshold_df["profit_factor"].idxmax()
        elif objective == "sharpe":
            optimal_idx = threshold_df["sharpe_ratio"].idxmax()
        elif objective == "f1":
            optimal_idx = threshold_df["f1"].idxmax()
        elif objective == "accuracy":
            optimal_idx = threshold_df["accuracy"].idxmax()
        elif objective == "precision":
            optimal_idx = threshold_df["precision"].idxmax()
        else:
            raise ValueError(f"Unknown objective: {objective}")

        optimal_threshold_val: float = float(threshold_df.loc[optimal_idx, "threshold"])  # type: ignore

        # Calculate final metrics at optimal threshold
        y_pred = (y_prob >= optimal_threshold_val).astype(int)
        final_metrics = self._calculate_metrics(
            y_true, y_pred, y_prob, returns, optimal_threshold_val
        )

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Find asymmetric thresholds (buy high, sell low)
        buy_threshold, sell_threshold = self._find_asymmetric_thresholds(
            y_true, y_prob, returns
        )

        return ThresholdResult(
            optimal_threshold=optimal_threshold_val,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            metrics=final_metrics,
            confusion=cm,
            threshold_curve=threshold_df,
        )

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        returns: Optional[np.ndarray],
        threshold: float,
    ) -> Dict[str, float]:
        """Calculate performance metrics for a given threshold."""
        metrics = {}

        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Handle edge cases
        if y_pred.sum() == 0:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        else:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # Trading metrics
        n_trades = y_pred.sum()
        metrics["n_trades"] = float(n_trades)
        metrics["trade_frequency"] = (
            float(n_trades / len(y_pred)) if len(y_pred) > 0 else 0.0
        )

        if returns is not None and n_trades > 0:
            # Calculate trading performance
            trade_returns = returns[y_pred == 1]

            # Gross returns
            gross_return = trade_returns.sum()

            # Net returns (after costs)
            net_return = gross_return - (n_trades * self.trading_cost * 2)  # Buy + sell

            metrics["gross_return"] = float(gross_return)
            metrics["net_return"] = float(net_return)

            # Profit factor
            winning_trades = trade_returns[trade_returns > 0].sum()
            losing_trades = abs(trade_returns[trade_returns < 0].sum())

            if losing_trades > 0:
                metrics["profit_factor"] = float(winning_trades / losing_trades)
            else:
                metrics["profit_factor"] = (
                    float(winning_trades) if winning_trades > 0 else 1.0
                )

            # Win rate
            metrics["win_rate"] = float((trade_returns > 0).mean())

            # Sharpe ratio (annualized, assuming daily returns)
            if len(trade_returns) > 1 and trade_returns.std() > 0:
                metrics["sharpe_ratio"] = float(
                    trade_returns.mean() / trade_returns.std() * np.sqrt(252)
                )
            else:
                metrics["sharpe_ratio"] = 0.0

            # Average trade
            metrics["avg_trade"] = float(trade_returns.mean())

            # Max consecutive losses
            is_loss = (trade_returns < 0).astype(int)
            max_consec_loss = 0
            current_streak = 0
            for loss in is_loss:
                if loss:
                    current_streak += 1
                    max_consec_loss = max(max_consec_loss, current_streak)
                else:
                    current_streak = 0
            metrics["max_consecutive_losses"] = float(max_consec_loss)

        else:
            metrics["gross_return"] = 0.0
            metrics["net_return"] = 0.0
            metrics["profit_factor"] = 1.0
            metrics["win_rate"] = 0.0
            metrics["sharpe_ratio"] = 0.0
            metrics["avg_trade"] = 0.0
            metrics["max_consecutive_losses"] = 0.0

        # Confidence metrics
        high_conf_mask = y_prob >= threshold
        if high_conf_mask.sum() > 0:
            metrics["avg_confidence"] = float(y_prob[high_conf_mask].mean())
        else:
            metrics["avg_confidence"] = 0.0

        return metrics

    def _find_asymmetric_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        returns: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Find asymmetric thresholds for buy and sell signals.

        Buy signals require higher confidence than sell signals
        to account for the asymmetric nature of gains/losses.
        """
        if returns is None:
            # Default to symmetric
            return 0.55, 0.45

        # Grid search for asymmetric thresholds
        best_sharpe = -np.inf
        best_buy = 0.55
        best_sell = 0.45

        for buy_thresh in np.linspace(0.55, 0.7, 16):
            for sell_thresh in np.linspace(0.3, 0.45, 16):
                # Buy when prob > buy_thresh
                # Sell when prob < sell_thresh

                signals = np.zeros(len(y_prob))
                signals[y_prob >= buy_thresh] = 1  # Buy
                signals[y_prob <= sell_thresh] = -1  # Sell

                # Calculate returns
                signal_returns = signals * returns

                if signal_returns.std() > 0:
                    sharpe = signal_returns.mean() / signal_returns.std() * np.sqrt(252)

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_buy = buy_thresh
                        best_sell = sell_thresh

        return float(best_buy), float(best_sell)

    def calibrate_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_precision: float = 0.55,
    ) -> float:
        """
        Calibrate threshold to achieve target precision.

        Useful when you want to control false positive rate.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            target_precision: Desired precision level

        Returns:
            Threshold that achieves approximately target precision
        """
        thresholds = np.linspace(0.3, 0.9, 121)

        for threshold in sorted(thresholds, reverse=True):
            y_pred = (y_prob >= threshold).astype(int)
            if y_pred.sum() > 0:
                prec = precision_score(y_true, y_pred, zero_division=0)
                if prec >= target_precision:
                    return float(threshold)

        # Return highest threshold if target not achievable
        return float(thresholds[-1])

    def dynamic_threshold(
        self,
        y_prob: np.ndarray,
        volatility: np.ndarray,
        base_threshold: float = 0.55,
    ) -> np.ndarray:
        """
        Calculate dynamic thresholds based on market volatility.

        In high volatility, require higher confidence.
        In low volatility, can use lower thresholds.

        Args:
            y_prob: Predicted probabilities
            volatility: Volatility measure for each prediction
            base_threshold: Base threshold for normal volatility

        Returns:
            Array of adjusted thresholds
        """
        # Normalize volatility
        vol_zscore = (volatility - volatility.mean()) / (volatility.std() + 1e-10)

        # Adjust threshold based on volatility
        # Higher vol -> higher threshold
        adjustment = 0.05 * vol_zscore
        adjusted_thresholds = base_threshold + adjustment

        # Clip to reasonable range
        adjusted_thresholds = np.clip(adjusted_thresholds, 0.5, 0.7)

        return adjusted_thresholds

    def regime_aware_threshold(
        self,
        y_prob: np.ndarray,
        regime: np.ndarray,
        regime_thresholds: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Apply regime-specific thresholds.

        Different market regimes may require different thresholds.

        Args:
            y_prob: Predicted probabilities
            regime: Regime labels for each prediction
            regime_thresholds: Dictionary mapping regime -> threshold

        Returns:
            Array of regime-adjusted thresholds
        """
        if regime_thresholds is None:
            regime_thresholds = {
                "bullish": 0.52,
                "neutral": 0.55,
                "bearish": 0.58,
                "high_volatility": 0.60,
            }

        thresholds = np.full(len(y_prob), 0.55)

        for regime_name, threshold in regime_thresholds.items():
            mask = regime == regime_name
            thresholds[mask] = threshold

        return thresholds


def optimize_threshold_for_trading(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    trading_cost: float = 0.001,
) -> Dict[str, Any]:
    """
    Convenience function to optimize threshold for trading.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        returns: Actual returns
        trading_cost: Transaction cost fraction

    Returns:
        Dictionary with optimal threshold and metrics
    """
    optimizer = ThresholdOptimizer(trading_cost=trading_cost)
    result = optimizer.find_optimal_threshold(
        y_true, y_prob, returns, objective="profit_factor"
    )

    return {
        "optimal_threshold": result.optimal_threshold,
        "buy_threshold": result.buy_threshold,
        "sell_threshold": result.sell_threshold,
        "profit_factor": result.metrics.get("profit_factor", 1.0),
        "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
        "win_rate": result.metrics.get("win_rate", 0.5),
        "n_trades": result.metrics.get("n_trades", 0),
    }
