"""
Technical analysis feature engineering.

This module generates technical indicators from OHLCV data
to be used as features for ML models.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """
    Generates technical indicators for ML features.
    """

    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
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

        # Trend Indicators
        df = TechnicalFeatures.add_moving_averages(df)
        df = TechnicalFeatures.add_macd(df)

        # Momentum Indicators
        df = TechnicalFeatures.add_rsi(df)
        df = TechnicalFeatures.add_stochastic(df)

        # Volatility Indicators
        df = TechnicalFeatures.add_bollinger_bands(df)
        df = TechnicalFeatures.add_atr(df)

        # Volume Indicators
        df = TechnicalFeatures.add_obv(df)

        # Returns and Log Returns (Targets)
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Target: Next day return
        df["target_return"] = df["returns"].shift(-1)
        df["target_direction"] = (df["target_return"] > 0).astype(int)

        return df

    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        periods = [10, 20, 50, 200]
        for p in periods:
            df[f"sma_{p}"] = df["close"].rolling(window=p).mean()
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

            # Distance from MA
            df[f"dist_sma_{p}"] = (df["close"] - df[f"sma_{p}"]) / df[f"sma_{p}"]

        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index (RSI)."""
        delta = df["close"].diff()
        # Ensure delta is numeric
        delta = pd.to_numeric(delta, errors="coerce")

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
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
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        return df

    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        df["stoch_k"] = 100 * ((df["close"] - low_min) / (high_max - low_min))
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
        df["atr_pct"] = df["atr"] / df["close"]
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume (OBV)."""
        # Ensure we are working with Series
        close_diff = df["close"].diff()
        direction = np.sign(close_diff)
        # Convert to Series to ensure fillna works
        direction_series = pd.Series(direction, index=df.index)

        df["obv"] = (direction_series * df["volume"]).fillna(0).cumsum()
        return df
