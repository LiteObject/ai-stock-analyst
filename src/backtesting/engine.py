"""
Event-driven backtesting engine with realistic execution modeling.

Features:
- Event-driven architecture (market data, signals, fills)
- Realistic execution with slippage and transaction costs
- Multi-asset portfolio support
- Position sizing integration
- Risk management hooks
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import (
    calculate_metrics,
    calculate_trade_metrics,
    analyze_drawdowns,
)
from src.backtesting.position_manager import PositionManager, SlippageModel
from src.backtesting.strategy import Strategy
from src.core.interfaces import DataProvider
from src.core.models import (
    BacktestConfig,
    BacktestResult,
    DailyPerformance,
    OrderSide,
    PerformanceMetrics,
    Portfolio,
    Position,
    Signal,
    SignalType,
    Trade,
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution timing modes."""

    NEXT_OPEN = "next_open"  # Execute at next bar's open
    NEXT_CLOSE = "next_close"  # Execute at next bar's close
    SAME_CLOSE = "same_close"  # Execute at same bar's close


class BacktestEngine:
    """
    Event-driven backtesting engine with enhanced execution modeling.

    Supports:
    - Realistic slippage and transaction costs via PositionManager
    - Multi-asset portfolios with proper rebalancing
    - Multiple position sizing strategies
    - Walk-forward integration
    """

    def __init__(
        self,
        data_provider: DataProvider,
        position_manager: Optional[PositionManager] = None,
        execution_mode: ExecutionMode = ExecutionMode.SAME_CLOSE,
    ):
        """
        Initialize backtest engine.

        Args:
            data_provider: Data provider for market data
            position_manager: Position manager for sizing and execution (optional)
            execution_mode: When to execute orders
        """
        self.data_provider = data_provider
        self.position_manager = position_manager
        self.execution_mode = execution_mode

    async def run(self, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
        """
        Run the backtest.
        """
        start_time = time.time()
        logger.info(
            f"Starting backtest for {config.tickers} from {config.start_date} to {config.end_date}"
        )

        # 1. Fetch Data
        market_data: Dict[str, pd.DataFrame] = {}
        for ticker in config.tickers:
            df = await self.data_provider.get_historical_data(
                ticker, config.start_date, config.end_date
            )
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue
            market_data[ticker] = df

        if not market_data:
            raise ValueError("No market data available for backtest")

        # Align data to a common index (dates)
        combined_index = pd.Index([])
        for df in market_data.values():
            combined_index = combined_index.union(df.index)
        combined_index = combined_index.sort_values()

        # 2. Initialize Portfolio
        portfolio = Portfolio(
            name=f"Backtest_{strategy.name}", cash=config.initial_capital
        )

        trades: List[Trade] = []
        daily_perf: List[DailyPerformance] = []
        trade_details: List[Dict[str, Any]] = []  # For enhanced trade metrics

        # 3. Simulation Loop
        for current_date in combined_index:
            # Update current prices for all positions
            current_prices = {}
            volumes = {}
            for ticker, df in market_data.items():
                if current_date in df.index:
                    price_val = df.loc[current_date, "close"]
                    if isinstance(price_val, pd.Series):
                        price_val = price_val.iloc[0]

                    try:
                        price = float(price_val)
                    except (ValueError, TypeError):
                        price = float(str(price_val))

                    current_prices[ticker] = price

                    # Get volume for position manager
                    if "volume" in df.columns:
                        vol_val = df.loc[current_date, "volume"]
                        if isinstance(vol_val, pd.Series):
                            vol_val = vol_val.iloc[0]
                        volumes[ticker] = float(vol_val)
                    else:
                        volumes[ticker] = 1_000_000  # Default volume

                    if ticker in portfolio.positions:
                        portfolio.positions[ticker].current_price = price
                        portfolio.positions[ticker].last_updated = current_date

            # Calculate Portfolio Value before trades
            portfolio_value = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            # Strategy Execution
            signals: List[Signal] = []
            for ticker, df in market_data.items():
                if current_date in df.index:
                    bar = df.loc[current_date]
                    signal = await strategy.on_bar(bar, portfolio.model_dump())
                    if signal:
                        signals.append(signal)

            # Order Execution with enhanced modeling
            for signal in signals:
                trade = await self._execute_signal(
                    signal,
                    portfolio,
                    current_prices,
                    volumes,
                    config,
                    trades,
                    current_date,
                )
                if trade:
                    trade_details.append(trade)

            # Record Daily Performance
            portfolio_value = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            daily_return = 0.0
            daily_return_pct = 0.0
            if daily_perf:
                prev_value = daily_perf[-1].portfolio_value
                daily_return = portfolio_value - prev_value
                daily_return_pct = daily_return / prev_value if prev_value > 0 else 0.0
            elif portfolio_value != config.initial_capital:
                daily_return = portfolio_value - config.initial_capital
                daily_return_pct = (
                    daily_return / config.initial_capital
                    if config.initial_capital > 0
                    else 0.0
                )

            # Calculate Drawdown
            peak = (
                max(d.portfolio_value for d in daily_perf)
                if daily_perf
                else config.initial_capital
            )
            peak = max(peak, portfolio_value)
            drawdown = (portfolio_value - peak) / peak if peak > 0 else 0.0

            daily_perf.append(
                DailyPerformance(
                    date=current_date,
                    portfolio_value=portfolio_value,
                    daily_return=daily_return,
                    daily_return_pct=daily_return_pct,
                    cumulative_return=(portfolio_value / config.initial_capital) - 1,
                    drawdown=drawdown,
                    cash=portfolio.cash,
                    positions_value=portfolio_value - portfolio.cash,
                )
            )

        # 4. Final Metrics with enhancements
        execution_time = time.time() - start_time

        returns_series = pd.Series([d.daily_return_pct for d in daily_perf])
        metrics_dict = calculate_metrics(returns_series)

        # Enhanced trade metrics
        trade_metrics = calculate_trade_metrics(trade_details) if trade_details else {}

        # Drawdown analysis
        dd_analysis = analyze_drawdowns(returns_series)

        total_return = metrics_dict.get("total_return", 0.0)
        max_drawdown = metrics_dict.get("max_drawdown", 0.0)

        metrics = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return * 100,
            cagr=metrics_dict.get("cagr", 0.0),
            sharpe_ratio=metrics_dict.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics_dict.get("sortino_ratio", 0.0),
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown * 100,
            volatility=metrics_dict.get("volatility", 0.0),
            win_rate=metrics_dict.get("win_rate", 0.0),
            profit_factor=metrics_dict.get("profit_factor", 0.0),
            total_trades=len(trades),
            start_date=config.start_date,
            end_date=config.end_date,
            trading_days=len(daily_perf),
            # Enhanced fields from comprehensive metrics
            calmar_ratio=metrics_dict.get("calmar_ratio"),
            avg_drawdown=metrics_dict.get("avg_drawdown"),
            max_drawdown_duration_days=dd_analysis.max_drawdown_duration_days,
            avg_win=metrics_dict.get("avg_win"),
            avg_loss=metrics_dict.get("avg_loss"),
            expectancy=metrics_dict.get("expectancy"),
            var_95=metrics_dict.get("var_95"),
            cvar_95=metrics_dict.get("cvar_95"),
            benchmark_return=None,
            alpha=None,
            beta=None,
        )

        return BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            daily_performance=daily_perf,
            final_portfolio=portfolio,
            execution_time_seconds=execution_time,
        )

    async def run_multi_asset(
        self,
        strategy: Strategy,
        config: BacktestConfig,
        rebalance_frequency: str = "monthly",
    ) -> BacktestResult:
        """
        Run multi-asset portfolio backtest with rebalancing.

        Args:
            strategy: Strategy that generates signals for multiple assets
            config: Base configuration
            rebalance_frequency: How often to rebalance (daily, weekly, monthly, quarterly)

        Returns:
            BacktestResult for the portfolio
        """
        logger.info(f"Starting multi-asset backtest with {len(config.tickers)} assets")

        # Load data for all tickers
        market_data: Dict[str, pd.DataFrame] = {}
        for ticker in config.tickers:
            df = await self.data_provider.get_historical_data(
                ticker, config.start_date, config.end_date
            )
            if not df.empty:
                market_data[ticker] = df

        if not market_data:
            raise ValueError("No market data available")

        # Get combined index
        combined_index = pd.Index([])
        for df in market_data.values():
            combined_index = combined_index.union(df.index)
        combined_index = combined_index.sort_values()

        # Calculate rebalance dates
        rebalance_dates = self._get_rebalance_dates(combined_index, rebalance_frequency)

        # Initialize portfolio
        portfolio = Portfolio(
            name=f"MultiAsset_{strategy.name}", cash=config.initial_capital
        )
        trades: List[Trade] = []
        daily_perf: List[DailyPerformance] = []

        for current_date in combined_index:
            # Update prices
            current_prices = {}
            volumes = {}
            for ticker, df in market_data.items():
                if current_date in df.index:
                    price = float(df.loc[current_date, "close"])
                    current_prices[ticker] = price
                    volumes[ticker] = (
                        float(df.loc[current_date, "volume"])
                        if "volume" in df.columns
                        else 1_000_000
                    )

                    if ticker in portfolio.positions:
                        portfolio.positions[ticker].current_price = price

            portfolio_value = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            # Check for rebalance
            if current_date in rebalance_dates:
                # Get target weights from strategy
                if hasattr(strategy, "get_target_weights"):
                    target_weights = await strategy.get_target_weights(
                        market_data, current_date, portfolio
                    )
                    await self._rebalance_portfolio(
                        portfolio,
                        target_weights,
                        current_prices,
                        volumes,
                        config,
                        trades,
                        current_date,
                    )

            # Record daily performance
            portfolio_value = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )

            daily_return_pct = 0.0
            if daily_perf:
                prev_value = daily_perf[-1].portfolio_value
                daily_return_pct = (
                    (portfolio_value - prev_value) / prev_value
                    if prev_value > 0
                    else 0.0
                )

            peak = (
                max(d.portfolio_value for d in daily_perf)
                if daily_perf
                else config.initial_capital
            )
            peak = max(peak, portfolio_value)
            drawdown = (portfolio_value - peak) / peak if peak > 0 else 0.0

            daily_perf.append(
                DailyPerformance(
                    date=current_date,
                    portfolio_value=portfolio_value,
                    daily_return=portfolio_value
                    - (
                        daily_perf[-1].portfolio_value
                        if daily_perf
                        else config.initial_capital
                    ),
                    daily_return_pct=daily_return_pct,
                    cumulative_return=(portfolio_value / config.initial_capital) - 1,
                    drawdown=drawdown,
                    cash=portfolio.cash,
                    positions_value=portfolio_value - portfolio.cash,
                )
            )

        # Calculate final metrics
        returns_series = pd.Series([d.daily_return_pct for d in daily_perf])
        metrics_dict = calculate_metrics(returns_series)
        dd_analysis = analyze_drawdowns(returns_series)

        metrics = PerformanceMetrics(
            total_return=metrics_dict.get("total_return", 0.0),
            total_return_pct=metrics_dict.get("total_return", 0.0) * 100,
            cagr=metrics_dict.get("cagr", 0.0),
            sharpe_ratio=metrics_dict.get("sharpe_ratio", 0.0),
            sortino_ratio=metrics_dict.get("sortino_ratio", 0.0),
            max_drawdown=metrics_dict.get("max_drawdown", 0.0),
            max_drawdown_pct=metrics_dict.get("max_drawdown", 0.0) * 100,
            volatility=metrics_dict.get("volatility", 0.0),
            win_rate=metrics_dict.get("win_rate", 0.0),
            profit_factor=metrics_dict.get("profit_factor", 0.0),
            total_trades=len(trades),
            start_date=config.start_date,
            end_date=config.end_date,
            trading_days=len(daily_perf),
            calmar_ratio=metrics_dict.get("calmar_ratio"),
            avg_drawdown=metrics_dict.get("avg_drawdown"),
            max_drawdown_duration_days=dd_analysis.max_drawdown_duration_days,
            avg_win=metrics_dict.get("avg_win"),
            avg_loss=metrics_dict.get("avg_loss"),
            expectancy=metrics_dict.get("expectancy"),
            var_95=metrics_dict.get("var_95"),
            cvar_95=metrics_dict.get("cvar_95"),
            benchmark_return=None,
            alpha=None,
            beta=None,
        )

        return BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            daily_performance=daily_perf,
            final_portfolio=portfolio,
            execution_time_seconds=0.0,
        )

    async def _execute_signal(
        self,
        signal: Signal,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
        volumes: Dict[str, float],
        config: BacktestConfig,
        trades: List[Trade],
        current_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a signal against the portfolio with enhanced modeling.
        """
        ticker = signal.ticker
        price = current_prices.get(ticker)
        volume = volumes.get(ticker, 1_000_000)

        if not price:
            return None

        # Use position manager if available for realistic execution
        if self.position_manager:
            return await self._execute_with_position_manager(
                signal, portfolio, price, volume, config, trades, current_date
            )

        # Fallback to simple execution
        slippage_mult = (
            1 + config.slippage_rate
            if signal.signal_type == SignalType.BUY
            else 1 - config.slippage_rate
        )
        execution_price = float(price * slippage_mult)

        trade_detail = None

        if signal.signal_type == SignalType.BUY:
            if ticker in portfolio.positions:
                return None

            target_allocation = portfolio.cash * config.max_position_size_pct
            quantity = int(target_allocation / execution_price)

            if quantity <= 0:
                return None

            cost = quantity * execution_price
            commission = cost * config.commission_rate
            total_cost = cost + commission

            if portfolio.cash >= total_cost:
                portfolio.cash -= total_cost
                portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    average_cost=execution_price,
                    current_price=float(price),
                    opened_at=current_date,
                    last_updated=current_date,
                )

                trades.append(
                    Trade(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        price=execution_price,
                        timestamp=current_date,
                        commission=commission,
                        strategy_name=signal.source,
                        order_id=None,
                    )
                )

                trade_detail = {
                    "ticker": ticker,
                    "side": "buy",
                    "quantity": quantity,
                    "price": execution_price,
                    "commission": commission,
                    "slippage": abs(execution_price - price) * quantity,
                    "entry_date": current_date,
                }

        elif signal.signal_type == SignalType.SELL:
            if ticker not in portfolio.positions:
                return None

            position = portfolio.positions[ticker]
            quantity = position.quantity

            proceeds = quantity * execution_price
            commission = proceeds * config.commission_rate
            net_proceeds = proceeds - commission

            # Calculate PnL
            entry_cost = position.average_cost * quantity
            pnl = net_proceeds - entry_cost

            # Duration
            duration_days = (
                (current_date - position.opened_at).days if position.opened_at else 0
            )

            portfolio.cash += net_proceeds
            del portfolio.positions[ticker]

            trades.append(
                Trade(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    price=execution_price,
                    timestamp=current_date,
                    commission=commission,
                    strategy_name=signal.source,
                    order_id=None,
                )
            )

            trade_detail = {
                "ticker": ticker,
                "side": "sell",
                "quantity": quantity,
                "price": execution_price,
                "commission": commission,
                "slippage": abs(price - execution_price) * quantity,
                "pnl": pnl,
                "duration_days": duration_days,
            }

        return trade_detail

    async def _execute_with_position_manager(
        self,
        signal: Signal,
        portfolio: Portfolio,
        price: float,
        volume: float,
        config: BacktestConfig,
        trades: List[Trade],
        current_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Execute using position manager for realistic costs."""
        ticker = signal.ticker

        if signal.signal_type == SignalType.BUY:
            if ticker in portfolio.positions:
                return None

            # Calculate position size
            portfolio_value = portfolio.cash + sum(
                p.market_value for p in portfolio.positions.values()
            )
            size_result = self.position_manager.calculate_position_size(
                portfolio_value=portfolio_value,
                current_price=price,
                strategy="equal_weight",
            )

            quantity = size_result.shares
            if quantity <= 0:
                return None

            # Execute with realistic costs
            execution = self.position_manager.execute_trade(
                ticker=ticker,
                shares=quantity,
                price=price,
                side="buy",
                volume=volume,
            )

            if not execution.success or execution.total_cost > portfolio.cash:
                return None

            portfolio.cash -= execution.total_cost
            portfolio.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                average_cost=execution.fill_price,
                current_price=price,
                opened_at=current_date,
                last_updated=current_date,
            )

            trades.append(
                Trade(
                    ticker=ticker,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    price=execution.fill_price,
                    timestamp=current_date,
                    commission=execution.transaction_costs.total_cost,
                    strategy_name=signal.source,
                    order_id=None,
                )
            )

            return {
                "ticker": ticker,
                "side": "buy",
                "quantity": quantity,
                "price": execution.fill_price,
                "commission": execution.transaction_costs.total_cost,
                "slippage": execution.slippage_cost,
                "market_impact": execution.market_impact_cost,
                "entry_date": current_date,
            }

        elif signal.signal_type == SignalType.SELL:
            if ticker not in portfolio.positions:
                return None

            position = portfolio.positions[ticker]
            quantity = position.quantity

            execution = self.position_manager.execute_trade(
                ticker=ticker,
                shares=quantity,
                price=price,
                side="sell",
                volume=volume,
            )

            if not execution.success:
                return None

            proceeds = execution.total_proceeds
            pnl = proceeds - (position.average_cost * quantity)
            duration_days = (
                (current_date - position.opened_at).days if position.opened_at else 0
            )

            portfolio.cash += proceeds
            del portfolio.positions[ticker]

            trades.append(
                Trade(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    price=execution.fill_price,
                    timestamp=current_date,
                    commission=execution.transaction_costs.total_cost,
                    strategy_name=signal.source,
                    order_id=None,
                )
            )

            return {
                "ticker": ticker,
                "side": "sell",
                "quantity": quantity,
                "price": execution.fill_price,
                "commission": execution.transaction_costs.total_cost,
                "slippage": execution.slippage_cost,
                "market_impact": execution.market_impact_cost,
                "pnl": pnl,
                "duration_days": duration_days,
            }

        return None

    async def _rebalance_portfolio(
        self,
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        volumes: Dict[str, float],
        config: BacktestConfig,
        trades: List[Trade],
        current_date: datetime,
    ) -> None:
        """Rebalance portfolio to target weights."""
        portfolio_value = portfolio.cash + sum(
            p.market_value for p in portfolio.positions.values()
        )

        # Calculate target positions
        for ticker, target_weight in target_weights.items():
            if ticker not in current_prices:
                continue

            price = current_prices[ticker]
            volume = volumes.get(ticker, 1_000_000)
            target_value = portfolio_value * target_weight
            target_shares = int(target_value / price)

            current_shares = 0
            if ticker in portfolio.positions:
                current_shares = portfolio.positions[ticker].quantity

            diff_shares = target_shares - current_shares

            if abs(diff_shares) < 1:
                continue

            # Execute rebalancing trade
            if self.position_manager:
                side = "buy" if diff_shares > 0 else "sell"
                execution = self.position_manager.execute_trade(
                    ticker=ticker,
                    shares=abs(diff_shares),
                    price=price,
                    side=side,
                    volume=volume,
                )

                if execution.success:
                    if diff_shares > 0:
                        if portfolio.cash >= execution.total_cost:
                            portfolio.cash -= execution.total_cost
                            if ticker in portfolio.positions:
                                pos = portfolio.positions[ticker]
                                old_value = pos.quantity * pos.average_cost
                                new_value = diff_shares * execution.fill_price
                                pos.quantity += diff_shares
                                pos.average_cost = (
                                    old_value + new_value
                                ) / pos.quantity
                            else:
                                portfolio.positions[ticker] = Position(
                                    ticker=ticker,
                                    quantity=diff_shares,
                                    average_cost=execution.fill_price,
                                    current_price=price,
                                    opened_at=current_date,
                                    last_updated=current_date,
                                )
                    else:
                        portfolio.cash += execution.total_proceeds
                        portfolio.positions[
                            ticker
                        ].quantity += diff_shares  # diff_shares is negative
                        if portfolio.positions[ticker].quantity <= 0:
                            del portfolio.positions[ticker]

                    trades.append(
                        Trade(
                            ticker=ticker,
                            side=OrderSide.BUY if diff_shares > 0 else OrderSide.SELL,
                            quantity=abs(diff_shares),
                            price=execution.fill_price,
                            timestamp=current_date,
                            commission=execution.transaction_costs.total_cost,
                            strategy_name="rebalance",
                            order_id=None,
                        )
                    )

    def _get_rebalance_dates(
        self,
        index: pd.Index,
        frequency: str,
    ) -> set:
        """Get dates when rebalancing should occur."""
        rebalance_dates = set()

        if frequency == "daily":
            return set(index)

        prev_period = None

        for date in index:
            if frequency == "weekly":
                period = date.isocalendar()[1]
            elif frequency == "monthly":
                period = date.month
            elif frequency == "quarterly":
                period = (date.month - 1) // 3
            else:
                period = date.month

            if prev_period is not None and period != prev_period:
                rebalance_dates.add(date)

            prev_period = period

        return rebalance_dates
