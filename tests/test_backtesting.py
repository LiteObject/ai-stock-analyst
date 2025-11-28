from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.strategy import Strategy
from core.interfaces import DataProvider
from core.models import BacktestConfig, Signal, SignalType


class MockDataProvider(DataProvider):
    @property
    def name(self) -> str:
        return "Mock"

    async def get_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        # Create a simple uptrend
        prices = np.linspace(100, 200, len(dates))
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": 1000,
            },
            index=dates,
        )
        return df

    async def get_current_price(self, ticker: str) -> float:
        return 150.0

    async def get_quote(self, ticker: str) -> Dict[str, Any]:
        return {"price": 150.0, "volume": 1000}


class BuyAndHoldStrategy(Strategy):
    async def on_bar(self, bar: pd.Series, portfolio: Dict[str, Any]) -> Optional[Signal]:
        # Buy on the first day if we don't have a position
        # We can check portfolio cash or positions
        # For simplicity, just emit BUY signal always, engine handles position limits
        return Signal(
            ticker="AAPL",
            signal_type=SignalType.BUY,
            confidence=1.0,
            source="BuyAndHold",
            timestamp=datetime.now(),
            reasoning="Test signal",
        )

    async def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


@pytest.mark.asyncio
async def test_backtest_engine():
    provider = MockDataProvider()
    engine = BacktestEngine(provider)
    strategy = BuyAndHoldStrategy(name="TestStrategy")

    config = BacktestConfig(
        tickers=["AAPL"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_capital=10000.0,
        stop_loss_pct=None,
        take_profit_pct=None,
    )

    result = await engine.run(strategy, config)

    assert result.metrics.total_return > 0
    assert len(result.trades) > 0
    assert result.final_portfolio.cash < 10000.0
    assert "AAPL" in result.final_portfolio.positions
