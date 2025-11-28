"""
Walk-forward analysis.
"""

import logging
from datetime import timedelta
from typing import Any, List

from backtesting.engine import BacktestEngine
from core.models import BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Implements walk-forward analysis.
    """

    def __init__(self, engine: BacktestEngine):
        self.engine = engine

    async def run(
        self,
        strategy_factory: Any,  # Callable returning a Strategy
        config: BacktestConfig,
    ) -> BacktestResult:
        """
        Run walk-forward analysis.
        """
        if not config.use_walk_forward:
            # Fallback to standard backtest
            strategy = strategy_factory()
            return await self.engine.run(strategy, config)

        logger.info("Starting Walk-Forward Analysis")

        start_date = config.start_date
        end_date = config.end_date
        train_window = timedelta(days=config.training_window_days)
        test_window = timedelta(days=config.test_window_days)

        current_date = start_date
        results: List[BacktestResult] = []

        while current_date + train_window + test_window <= end_date:
            train_start = current_date
            train_end = current_date + train_window
            test_start = train_end
            test_end = test_start + test_window

            logger.info(
                f"Walk-forward step: Train {train_start.date()}->{train_end.date()}, Test {test_start.date()}->{test_end.date()}"
            )

            # 1. Train (Optimize)
            # In a real scenario, we would optimize parameters here using the training data.
            # For now, we just instantiate the strategy.
            # TODO: Implement parameter optimization interface
            strategy = strategy_factory()

            # 2. Test (Out-of-sample)
            step_config = config.model_copy()
            step_config.start_date = test_start
            step_config.end_date = test_end
            # Reset capital for each step or carry over?
            # Usually walk-forward stitches returns.
            # For simplicity, we'll run independent tests and aggregate.

            result = await self.engine.run(strategy, step_config)
            results.append(result)

            # Move forward
            current_date += test_window

        # Aggregate results
        # This is complex because we need to stitch the equity curves.
        # For now, we will return the last result or a dummy result with aggregated metrics.

        if not results:
            raise ValueError("No walk-forward steps completed")

        # TODO: Proper aggregation
        logger.info(f"Completed {len(results)} walk-forward steps")

        return results[-1]  # Return last step for now
