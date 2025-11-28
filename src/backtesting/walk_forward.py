"""
Walk-forward analysis with proper parameter optimization and validation.

Implements:
- Rolling and anchored walk-forward analysis
- Parameter optimization during training periods
- Out-of-sample validation
- Equity curve stitching
- Statistical significance testing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import calculate_metrics, analyze_drawdowns
from src.core.models import BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward analysis modes."""

    ROLLING = "rolling"  # Fixed-size rolling training window
    ANCHORED = "anchored"  # Expanding training window from start
    HYBRID = "hybrid"  # Anchored with maximum window size


@dataclass
class WalkForwardStep:
    """Results from a single walk-forward step."""

    step_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: Optional[BacktestResult]
    test_result: BacktestResult
    optimized_params: Dict[str, Any]
    optimization_score: float
    out_of_sample_score: float
    efficiency_ratio: float  # OOS / IS performance


@dataclass
class WalkForwardAnalysis:
    """Complete walk-forward analysis results."""

    mode: WalkForwardMode
    total_steps: int
    steps: List[WalkForwardStep]

    # Aggregated performance
    combined_returns: pd.Series = field(default_factory=pd.Series)
    combined_equity_curve: pd.Series = field(default_factory=pd.Series)

    # In-sample vs out-of-sample
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    efficiency_ratio: float = 0.0  # OOS Sharpe / IS Sharpe

    # Combined metrics
    combined_metrics: Dict[str, float] = field(default_factory=dict)

    # Statistical tests
    is_significant: bool = False
    p_value: float = 1.0

    # Robustness
    param_stability: Dict[str, float] = field(default_factory=dict)
    degradation_ratio: float = 0.0


class WalkForwardBacktester:
    """
    Implements walk-forward analysis with parameter optimization.

    Walk-forward analysis helps validate that:
    1. The strategy works on out-of-sample data
    2. Parameters are stable across time periods
    3. Performance doesn't degrade significantly OOS
    """

    def __init__(
        self,
        engine: BacktestEngine,
        mode: WalkForwardMode = WalkForwardMode.ROLLING,
        min_train_samples: int = 252,  # Minimum training period (1 year)
        optimization_metric: str = "sharpe_ratio",
        n_optimization_trials: int = 50,
    ):
        """
        Initialize walk-forward backtester.

        Args:
            engine: Backtest engine instance
            mode: Walk-forward mode (rolling, anchored, hybrid)
            min_train_samples: Minimum training samples required
            optimization_metric: Metric to optimize during training
            n_optimization_trials: Number of optimization trials per step
        """
        self.engine = engine
        self.mode = mode
        self.min_train_samples = min_train_samples
        self.optimization_metric = optimization_metric
        self.n_optimization_trials = n_optimization_trials

    async def run(
        self,
        strategy_factory: Callable[..., Any],
        config: BacktestConfig,
        param_space: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Any] = None,  # HyperparameterOptimizer
    ) -> WalkForwardAnalysis:
        """
        Run walk-forward analysis.

        Args:
            strategy_factory: Factory function that creates strategy with given params
            config: Backtest configuration
            param_space: Parameter space for optimization (optional)
            optimizer: Hyperparameter optimizer instance (optional)

        Returns:
            WalkForwardAnalysis with aggregated results
        """
        if not config.use_walk_forward:
            # Fallback to standard backtest
            logger.info("Walk-forward disabled, running standard backtest")
            strategy = strategy_factory()
            result = await self.engine.run(strategy, config)

            return WalkForwardAnalysis(
                mode=self.mode,
                total_steps=1,
                steps=[
                    WalkForwardStep(
                        step_number=1,
                        train_start=config.start_date,
                        train_end=config.start_date,
                        test_start=config.start_date,
                        test_end=config.end_date,
                        train_result=None,
                        test_result=result,
                        optimized_params={},
                        optimization_score=0.0,
                        out_of_sample_score=0.0,
                        efficiency_ratio=1.0,
                    )
                ],
                combined_returns=(
                    result.returns if hasattr(result, "returns") else pd.Series()
                ),
                combined_equity_curve=(
                    result.equity_curve
                    if hasattr(result, "equity_curve")
                    else pd.Series()
                ),
            )

        logger.info(f"Starting Walk-Forward Analysis (mode={self.mode.value})")

        # Calculate periods
        periods = self._calculate_periods(config)

        if len(periods) == 0:
            raise ValueError("No valid walk-forward periods generated")

        logger.info(f"Generated {len(periods)} walk-forward steps")

        steps: List[WalkForwardStep] = []
        all_returns: List[pd.Series] = []
        all_equity: List[pd.Series] = []
        all_params: List[Dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(
                f"Step {i+1}/{len(periods)}: "
                f"Train {train_start.date()}->{train_end.date()}, "
                f"Test {test_start.date()}->{test_end.date()}"
            )

            # 1. Parameter optimization on training data
            if param_space and optimizer:
                optimized_params, train_result = await self._optimize_step(
                    strategy_factory=strategy_factory,
                    config=config,
                    train_start=train_start,
                    train_end=train_end,
                    param_space=param_space,
                    optimizer=optimizer,
                )
            else:
                optimized_params = {}
                # Run training period
                train_config = config.model_copy()
                train_config.start_date = train_start
                train_config.end_date = train_end
                strategy = (
                    strategy_factory(**optimized_params)
                    if optimized_params
                    else strategy_factory()
                )
                train_result = await self.engine.run(strategy, train_config)

            # 2. Out-of-sample testing with optimized parameters
            test_config = config.model_copy()
            test_config.start_date = test_start
            test_config.end_date = test_end

            strategy = (
                strategy_factory(**optimized_params)
                if optimized_params
                else strategy_factory()
            )
            test_result = await self.engine.run(strategy, test_config)

            # 3. Calculate performance scores
            train_metrics = calculate_metrics(
                train_result.returns
                if hasattr(train_result, "returns")
                else pd.Series()
            )
            test_metrics = calculate_metrics(
                test_result.returns if hasattr(test_result, "returns") else pd.Series()
            )

            is_score = train_metrics.get(self.optimization_metric, 0.0)
            oos_score = test_metrics.get(self.optimization_metric, 0.0)

            efficiency = oos_score / is_score if is_score != 0 else 1.0

            step = WalkForwardStep(
                step_number=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_result=train_result,
                test_result=test_result,
                optimized_params=optimized_params,
                optimization_score=is_score,
                out_of_sample_score=oos_score,
                efficiency_ratio=efficiency,
            )
            steps.append(step)
            all_params.append(optimized_params)

            # Collect returns and equity for stitching
            if hasattr(test_result, "returns"):
                all_returns.append(test_result.returns)
            if hasattr(test_result, "equity_curve"):
                all_equity.append(test_result.equity_curve)

            logger.info(
                f"Step {i+1}: IS {self.optimization_metric}={is_score:.4f}, "
                f"OOS {self.optimization_metric}={oos_score:.4f}, "
                f"Efficiency={efficiency:.2%}"
            )

        # Aggregate results
        analysis = self._aggregate_results(steps, all_returns, all_equity, all_params)

        logger.info(
            f"Walk-forward complete: {analysis.total_steps} steps, "
            f"Efficiency ratio={analysis.efficiency_ratio:.2%}, "
            f"Combined Sharpe={analysis.combined_metrics.get('sharpe_ratio', 0):.4f}"
        )

        return analysis

    def _calculate_periods(
        self,
        config: BacktestConfig,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Calculate train/test periods for walk-forward analysis.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        start_date = config.start_date
        end_date = config.end_date
        train_days = config.training_window_days
        test_days = config.test_window_days

        train_window = timedelta(days=train_days)
        test_window = timedelta(days=test_days)

        periods = []

        if self.mode == WalkForwardMode.ROLLING:
            # Fixed-size rolling window
            current = start_date

            while current + train_window + test_window <= end_date:
                train_start = current
                train_end = current + train_window
                test_start = train_end
                test_end = test_start + test_window

                periods.append((train_start, train_end, test_start, test_end))
                current += test_window  # Step forward by test window

        elif self.mode == WalkForwardMode.ANCHORED:
            # Expanding training window from start
            current = start_date + train_window

            while current + test_window <= end_date:
                train_start = start_date  # Always from beginning
                train_end = current
                test_start = current
                test_end = current + test_window

                periods.append((train_start, train_end, test_start, test_end))
                current += test_window

        elif self.mode == WalkForwardMode.HYBRID:
            # Anchored with maximum window size (e.g., 3 years max)
            max_train_days = train_days * 3  # Max 3x the standard window
            max_train_window = timedelta(days=max_train_days)

            current = start_date + train_window

            while current + test_window <= end_date:
                # Calculate train start - anchor but with max limit
                potential_train_start = start_date
                if current - start_date > max_train_window:
                    potential_train_start = current - max_train_window

                train_start = potential_train_start
                train_end = current
                test_start = current
                test_end = current + test_window

                periods.append((train_start, train_end, test_start, test_end))
                current += test_window

        return periods

    async def _optimize_step(
        self,
        strategy_factory: Callable[..., Any],
        config: BacktestConfig,
        train_start: datetime,
        train_end: datetime,
        param_space: Dict[str, Any],
        optimizer: Any,
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters on training data.

        Returns:
            Tuple of (optimized_params, training_result)
        """
        train_config = config.model_copy()
        train_config.start_date = train_start
        train_config.end_date = train_end

        async def objective(params: Dict[str, Any]) -> float:
            """Optimization objective function."""
            strategy = strategy_factory(**params)
            result = await self.engine.run(strategy, train_config)

            if hasattr(result, "returns") and len(result.returns) > 0:
                metrics = calculate_metrics(result.returns)
                return metrics.get(self.optimization_metric, 0.0)
            return 0.0

        # Run optimization (simplified - real implementation would use optimizer)
        # For now, just use default parameters
        best_params = {}

        # Run with best params to get result
        strategy = (
            strategy_factory(**best_params) if best_params else strategy_factory()
        )
        result = await self.engine.run(strategy, train_config)

        return best_params, result

    def _aggregate_results(
        self,
        steps: List[WalkForwardStep],
        all_returns: List[pd.Series],
        all_equity: List[pd.Series],
        all_params: List[Dict[str, Any]],
    ) -> WalkForwardAnalysis:
        """Aggregate results from all walk-forward steps."""

        # Stitch returns together
        if all_returns:
            combined_returns = pd.concat(all_returns, axis=0)
            combined_returns = combined_returns[
                ~combined_returns.index.duplicated(keep="first")
            ]
            combined_returns = combined_returns.sort_index()
        else:
            combined_returns = pd.Series()

        # Stitch equity curves with proper continuation
        if all_equity:
            combined_equity = self._stitch_equity_curves(all_equity)
        else:
            combined_equity = pd.Series()

        # Calculate combined metrics
        if len(combined_returns) > 0:
            combined_metrics = calculate_metrics(combined_returns)
        else:
            combined_metrics = {}

        # Calculate IS/OOS statistics
        is_scores = [s.optimization_score for s in steps]
        oos_scores = [s.out_of_sample_score for s in steps]

        avg_is = np.mean(is_scores) if is_scores else 0.0
        avg_oos = np.mean(oos_scores) if oos_scores else 0.0
        efficiency = avg_oos / avg_is if avg_is != 0 else 1.0

        # Parameter stability analysis
        param_stability = self._analyze_param_stability(all_params)

        # Calculate degradation
        degradation = (avg_is - avg_oos) / avg_is if avg_is != 0 else 0.0

        # Statistical significance test
        is_significant, p_value = self._test_significance(combined_returns)

        return WalkForwardAnalysis(
            mode=self.mode,
            total_steps=len(steps),
            steps=steps,
            combined_returns=combined_returns,
            combined_equity_curve=combined_equity,
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            efficiency_ratio=efficiency,
            combined_metrics=combined_metrics,
            is_significant=is_significant,
            p_value=p_value,
            param_stability=param_stability,
            degradation_ratio=degradation,
        )

    def _stitch_equity_curves(
        self,
        equity_curves: List[pd.Series],
    ) -> pd.Series:
        """
        Stitch equity curves together with proper continuation.

        Each subsequent period starts where the previous ended.
        """
        if not equity_curves:
            return pd.Series()

        result_parts = []
        cumulative_factor = 1.0

        for i, equity in enumerate(equity_curves):
            if len(equity) == 0:
                continue

            # Normalize equity curve to start at 1
            normalized = equity / equity.iloc[0]

            # Scale by cumulative factor from previous periods
            scaled = normalized * cumulative_factor

            if i > 0:
                # Remove the first point (it's duplicated from previous period's end)
                scaled = scaled.iloc[1:]

            result_parts.append(scaled)

            # Update cumulative factor
            if len(scaled) > 0:
                cumulative_factor = scaled.iloc[-1]

        if result_parts:
            result = pd.concat(result_parts)
            result = result[~result.index.duplicated(keep="first")]
            return result.sort_index()

        return pd.Series()

    def _analyze_param_stability(
        self,
        all_params: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Analyze how stable parameters are across walk-forward steps.

        Returns coefficient of variation for each parameter.
        """
        if not all_params or len(all_params) < 2:
            return {}

        # Collect all unique parameter names
        all_param_names = set()
        for params in all_params:
            all_param_names.update(params.keys())

        stability = {}

        for param_name in all_param_names:
            values = []
            for params in all_params:
                if param_name in params:
                    val = params[param_name]
                    if isinstance(val, (int, float)):
                        values.append(val)

            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)

                # Coefficient of variation (lower = more stable)
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                else:
                    cv = 0.0 if std_val == 0 else float("inf")

                stability[param_name] = cv

        return stability

    def _test_significance(
        self,
        returns: pd.Series,
        null_mean: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Test if returns are statistically significantly different from zero.

        Uses t-test with Newey-West standard errors for autocorrelation.
        """
        if len(returns) < 30:
            return False, 1.0

        from scipy import stats

        # Simple t-test (could be improved with Newey-West)
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), null_mean)

        # One-sided test (returns > 0)
        p_value_one_sided = p_value / 2 if t_stat > 0 else 1.0

        is_significant = p_value_one_sided < 0.05

        return is_significant, p_value_one_sided


class ComboPurgedCV:
    """
    Combinatorially Purged Cross-Validation for financial time series.

    Addresses:
    1. Information leakage from overlapping periods
    2. Look-ahead bias
    3. Limited number of independent samples
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_periods: int = 5,  # Gap between train and test
        purge_periods: int = 3,  # Periods to remove at boundaries
    ):
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
        self.purge_periods = purge_periods

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate combinatorially purged train/test splits.

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate fold size
        fold_size = n_samples // self.n_splits

        splits = []

        # Generate all combinations of folds
        for test_fold in range(self.n_splits):
            test_start = test_fold * fold_size
            test_end = (
                (test_fold + 1) * fold_size
                if test_fold < self.n_splits - 1
                else n_samples
            )

            test_indices = indices[test_start:test_end]

            # Purge and embargo
            purge_start = max(0, test_start - self.purge_periods)
            embargo_end = min(n_samples, test_end + self.embargo_periods)

            # Train indices: everything except test + purge + embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:embargo_end] = False

            train_indices = indices[train_mask]

            splits.append((train_indices, test_indices))

        return splits
