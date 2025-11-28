"""
Base Strategy class for backtesting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from core.models import Signal


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    async def on_bar(self, bar: pd.Series, portfolio: Dict[str, Any]) -> Optional[Signal]:
        """
        Called on every new bar (candle).

        Args:
            bar: The current OHLCV bar (Series)
            portfolio: Current portfolio state

        Returns:
            Signal or None
        """
        pass

    @abstractmethod
    async def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate indicators or signals on the whole dataset (Vectorized approach).

        Args:
            data: Full historical DataFrame

        Returns:
            DataFrame with added indicators/signals
        """
        pass
