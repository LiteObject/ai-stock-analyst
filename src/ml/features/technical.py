"""
Technical analysis feature engineering.

This module generates technical indicators from OHLCV data
to be used as features for ML models.

Uses pandas-ta for cross-platform compatibility (no TA-Lib dependency).
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    ta = None

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Generates technical indicators for ML features.

    Uses pandas-ta for indicator calculations when available,
    with fallback to manual implementations.
    """

    def __init__(
        self,
        ma_periods: Optional[List[int]] = None,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
    ):
        """
        Initialize TechnicalFeatures with configurable parameters.

        Args:
            ma_periods: Moving average periods (default: [10, 20, 50, 200])
            rsi_period: RSI calculation period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            atr_period: ATR period
            stoch_k: Stochastic %K period
            stoch_d: Stochastic %D smoothing period
        """
        self.ma_periods = ma_periods or [10, 20, 50, 200]
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical features to the DataFrame.

        Args:
            df: DataFrame with open, high, low, close, volume columns

        Returns:
            DataFrame with added features
        """
        df = df.copy()

        # Ensure columns are lowercase
        df.columns = df.columns.str.lower()

        if HAS_PANDAS_TA:
            df = self._add_features_pandas_ta(df)
        else:
            logger.warning("pandas-ta not available, using fallback implementations")
            df = self._add_features_fallback(df)

        # Returns and Log Returns (Targets)
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Target: Next day return
        df["target_return"] = df["returns"].shift(-1)
        df["target_direction"] = (df["target_return"] > 0).astype(int)

        return df

    def _add_features_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features using pandas-ta library."""

        # Moving Averages
        for p in self.ma_periods:
            df[f"sma_{p}"] = ta.sma(df["close"], length=p)
            df[f"ema_{p}"] = ta.ema(df["close"], length=p)
            df[f"dist_sma_{p}"] = (df["close"] - df[f"sma_{p}"]) / (
                df[f"sma_{p}"] + 1e-10
            )

        # RSI
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)

        # MACD
        macd_df = ta.macd(
            df["close"],
            fast=self.macd_fast,
            slow=self.macd_slow,
            signal=self.macd_signal,
        )
        if macd_df is not None:
            df["macd"] = macd_df.iloc[:, 0]
            df["macd_hist"] = macd_df.iloc[:, 1]
            df["macd_signal"] = macd_df.iloc[:, 2]

        # Bollinger Bands
        bb_df = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        if bb_df is not None:
            df["bb_lower"] = bb_df.iloc[:, 0]
            df["bb_mid"] = bb_df.iloc[:, 1]
            df["bb_upper"] = bb_df.iloc[:, 2]
            df["bb_width"] = bb_df.iloc[:, 3]
            df["bb_pct"] = bb_df.iloc[:, 4]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"] + 1e-10
            )

        # Stochastic
        stoch_df = ta.stoch(
            df["high"], df["low"], df["close"], k=self.stoch_k, d=self.stoch_d
        )
        if stoch_df is not None:
            df["stoch_k"] = stoch_df.iloc[:, 0]
            df["stoch_d"] = stoch_df.iloc[:, 1]

        # ATR
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-10)

        # ADX (trend strength)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None:
            df["adx"] = adx_df.iloc[:, 0]
            df["dmp"] = adx_df.iloc[:, 1]
            df["dmn"] = adx_df.iloc[:, 2]

        # OBV
        df["obv"] = ta.obv(df["close"], df["volume"])

        # CCI
        df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

        # Williams %R
        df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)

        # MFI (Money Flow Index)
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

        # VWAP (Volume Weighted Average Price)
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        if vwap is not None:
            df["vwap"] = vwap
            df["dist_vwap"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-10)

        # Additional momentum indicators
        df["roc"] = ta.roc(df["close"], length=10)  # Rate of Change
        df["mom"] = ta.mom(df["close"], length=10)  # Momentum

        # Volatility indicators
        df["natr"] = ta.natr(
            df["high"], df["low"], df["close"], length=14
        )  # Normalized ATR

        # Volume indicators
        ad = ta.ad(df["high"], df["low"], df["close"], df["volume"])
        if ad is not None:
            df["ad_line"] = ad

        # Trend indicators
        aroon_df = ta.aroon(df["high"], df["low"], length=25)
        if aroon_df is not None:
            df["aroon_up"] = aroon_df.iloc[:, 0]
            df["aroon_down"] = aroon_df.iloc[:, 1]
            df["aroon_osc"] = aroon_df.iloc[:, 2]

        return df

    def _add_features_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback implementations without pandas-ta."""

        # Moving Averages
        for p in self.ma_periods:
            df[f"sma_{p}"] = df["close"].rolling(window=p).mean()
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
            df[f"dist_sma_{p}"] = (df["close"] - df[f"sma_{p}"]) / (
                df[f"sma_{p}"] + 1e-10
            )

        # RSI
        df = self._add_rsi_fallback(df)

        # MACD
        df = self._add_macd_fallback(df)

        # Bollinger Bands
        df = self._add_bollinger_fallback(df)

        # Stochastic
        df = self._add_stochastic_fallback(df)

        # ATR
        df = self._add_atr_fallback(df)

        # OBV
        df = self._add_obv_fallback(df)

        return df

    def _add_rsi_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI without pandas-ta."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD without pandas-ta."""
        exp1 = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def _add_bollinger_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands without pandas-ta."""
        sma = df["close"].rolling(window=self.bb_period).mean()
        std = df["close"].rolling(window=self.bb_period).std()
        df["bb_upper"] = sma + (std * self.bb_std)
        df["bb_lower"] = sma - (std * self.bb_std)
        df["bb_mid"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma + 1e-10)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )
        return df

    def _add_stochastic_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stochastic Oscillator without pandas-ta."""
        low_min = df["low"].rolling(window=self.stoch_k).min()
        high_max = df["high"].rolling(window=self.stoch_k).max()
        df["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min + 1e-10))
        df["stoch_d"] = df["stoch_k"].rolling(window=self.stoch_d).mean()
        return df

    def _add_atr_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR without pandas-ta."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(window=self.atr_period).mean()
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-10)
        return df

    def _add_obv_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV without pandas-ta."""
        close_diff = df["close"].diff()
        direction = np.sign(close_diff)
        df["obv"] = (direction * df["volume"]).fillna(0).cumsum()
        return df

    # Keep static methods for backward compatibility
    @staticmethod
    def add_all_features_static(df: pd.DataFrame) -> pd.DataFrame:
        """Static method for backward compatibility."""
        return TechnicalFeatures().add_all_features(df)

    # Alias for backward compatibility with existing code
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        periods = [10, 20, 50, 200]
        for p in periods:
            df[f"sma_{p}"] = df["close"].rolling(window=p).mean()
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
            df[f"dist_sma_{p}"] = (df["close"] - df[f"sma_{p}"]) / (
                df[f"sma_{p}"] + 1e-10
            )
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index (RSI)."""
        delta = df["close"].diff()
        delta = pd.to_numeric(delta, errors="coerce")
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add Moving Average Convergence Divergence (MACD)."""
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        df["bb_upper"] = sma + (std * 2)
        df["bb_lower"] = sma - (std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma + 1e-10)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )
        return df

    @staticmethod
    def add_stochastic(
        df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        df["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min + 1e-10))
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (ATR)."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-10)
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume (OBV)."""
        close_diff = df["close"].diff()
        direction = np.sign(close_diff)
        direction_series = pd.Series(direction, index=df.index)
        df["obv"] = (direction_series * df["volume"]).fillna(0).cumsum()
        return df
