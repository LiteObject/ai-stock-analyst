"""
Backtesting Engine.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd

from backtesting.metrics import calculate_metrics
from backtesting.strategy import Strategy
from core.interfaces import DataProvider
from core.models import (
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


class BacktestEngine:
    """
    Event-driven backtesting engine.
    """

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    async def run(self, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
        """
        Run the backtest.
        """
        start_time = time.time()
        logger.info(f"Starting backtest for {config.tickers} from {config.start_date} to {config.end_date}")

        # 1. Fetch Data
        # For simplicity, we'll fetch all data upfront.
        # In a real streaming engine, we'd fetch chunk by chunk or use a generator.
        market_data: Dict[str, pd.DataFrame] = {}
        for ticker in config.tickers:
            df = await self.data_provider.get_historical_data(ticker, config.start_date, config.end_date)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue
            market_data[ticker] = df

        if not market_data:
            raise ValueError("No market data available for backtest")

        # Align data to a common index (dates)
        # We need to iterate day by day.
        combined_index = pd.Index([])
        for df in market_data.values():
            combined_index = combined_index.union(df.index)
        combined_index = combined_index.sort_values()

        # 2. Initialize Portfolio
        portfolio = Portfolio(name=f"Backtest_{strategy.name}", cash=config.initial_capital)

        trades: List[Trade] = []
        daily_perf: List[DailyPerformance] = []

        # 3. Simulation Loop
        for current_date in combined_index:
            # Update current prices for all positions
            current_prices = {}
            for ticker, df in market_data.items():
                if current_date in df.index:
                    # Use .item() to get python scalar if it's a single value series/dataframe
                    price_val = df.loc[current_date, "close"]
                    # Handle potential Series if duplicate index (shouldn't happen but good to be safe)
                    if isinstance(price_val, pd.Series):
                        price_val = price_val.iloc[0]

                    # Safe conversion
                    try:
                        price = float(price_val)  # type: ignore[arg-type]
                    except (ValueError, TypeError):
                        # Fallback for complex types or others
                        price = float(str(price_val))

                    current_prices[ticker] = price
                    if ticker in portfolio.positions:
                        portfolio.positions[ticker].current_price = price
                        portfolio.positions[ticker].last_updated = current_date

            # Calculate Portfolio Value before trades
            portfolio_value = portfolio.cash + sum(p.market_value for p in portfolio.positions.values())

            # Strategy Execution
            # We pass the data available UP TO this point (or just the current bar if strategy is stateless)
            # For this implementation, we'll pass the current bar for each ticker.

            signals: List[Signal] = []
            for ticker, df in market_data.items():
                if current_date in df.index:
                    bar = df.loc[current_date]
                    # TODO: Pass historical context if needed by strategy
                    # For now, we assume strategy might need to look up history itself or we pass a slice
                    # But Strategy.on_bar signature takes a single bar.
                    # We might need to enhance Strategy to accept history.

                    # Let's assume the strategy can handle single bar updates or we'll improve this later.
                    signal = await strategy.on_bar(bar, portfolio.model_dump())
                    if signal:
                        signals.append(signal)

            # Order Execution
            for signal in signals:
                await self._execute_signal(signal, portfolio, current_prices, config, trades, current_date)

            # Record Daily Performance
            # Recalculate value after trades
            portfolio_value = portfolio.cash + sum(p.market_value for p in portfolio.positions.values())

            daily_return = 0.0
            daily_return_pct = 0.0
            if daily_perf:
                prev_value = daily_perf[-1].portfolio_value
                daily_return = portfolio_value - prev_value
                daily_return_pct = daily_return / prev_value
            elif portfolio_value != config.initial_capital:
                daily_return = portfolio_value - config.initial_capital
                daily_return_pct = daily_return / config.initial_capital

            # Calculate Drawdown
            peak = max(d.portfolio_value for d in daily_perf) if daily_perf else config.initial_capital
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

        # 4. Final Metrics
        execution_time = time.time() - start_time

        returns_series = pd.Series([d.daily_return_pct for d in daily_perf])
        metrics_dict = calculate_metrics(returns_series)

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
            # Optional fields
            calmar_ratio=None,
            avg_drawdown=None,
            max_drawdown_duration_days=None,
            avg_win=None,
            avg_loss=None,
            expectancy=None,
            var_95=None,
            cvar_95=None,
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

    async def _execute_signal(
        self,
        signal: Signal,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
        config: BacktestConfig,
        trades: List[Trade],
        current_date: datetime,
    ):
        """
        Execute a signal against the portfolio.
        """
        ticker = signal.ticker
        price = current_prices.get(ticker)

        if not price:
            return

        # Apply slippage
        # Buy: Price increases. Sell: Price decreases (worse execution)
        # Actually, Sell: Price decreases means we get less money.
        slippage_mult = 1 + config.slippage_rate if signal.signal_type == SignalType.BUY else 1 - config.slippage_rate
        execution_price = float(price * slippage_mult)

        # Calculate Quantity
        # Simple logic: Use signal.strength or fixed size
        # Here we use max_position_size_pct from config

        if signal.signal_type == SignalType.BUY:
            # Check if we already have a position
            if ticker in portfolio.positions:
                # Already long, maybe add more? For now, ignore.
                return

            # Calculate max capital to allocate
            target_allocation = portfolio.cash * config.max_position_size_pct
            quantity = int(target_allocation / execution_price)

            if quantity <= 0:
                return

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

        elif signal.signal_type == SignalType.SELL:
            if ticker not in portfolio.positions:
                return

            position = portfolio.positions[ticker]
            quantity = position.quantity  # Sell all

            proceeds = quantity * execution_price
            commission = proceeds * config.commission_rate
            net_proceeds = proceeds - commission

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
