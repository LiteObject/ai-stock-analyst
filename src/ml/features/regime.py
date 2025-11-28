"""
Market Regime Detection for ML Models.

This module provides regime detection capabilities including:
- Trend regime detection (bullish/bearish/neutral)
- Volatility regime classification
- Market stress indicators
- Hurst exponent for mean-reversion vs momentum
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    """Trend regime classification."""

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class VolatilityRegime(Enum):
    """Volatility regime classification."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class MarketRegime(Enum):
    """Overall market regime."""

    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITIONING = "transitioning"


@dataclass
class RegimeState:
    """Current regime state."""

    trend: TrendRegime
    volatility: VolatilityRegime
    market: MarketRegime
    hurst: float
    stress_level: float
    confidence: float


class RegimeDetector:
    """
    Detects market regimes for adaptive trading strategies.

    Identifies:
    - Trend regimes using moving average crossovers and momentum
    - Volatility regimes using realized volatility percentiles
    - Market stress using composite indicators
    - Mean-reversion vs momentum using Hurst exponent
    """

    def __init__(
        self,
        short_window: int = 20,
        medium_window: int = 50,
        long_window: int = 200,
        vol_lookback: int = 252,
    ):
        """
        Initialize RegimeDetector.

        Args:
            short_window: Short-term moving average window
            medium_window: Medium-term moving average window
            long_window: Long-term moving average window
            vol_lookback: Lookback for volatility percentile calculation
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.vol_lookback = vol_lookback

    def detect_all_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all regime types and return as features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with regime features
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        features = pd.DataFrame(index=df.index)

        # Trend Regime
        logger.info("Detecting trend regime...")
        trend_features = self.detect_trend_regime(df)
        features = pd.concat([features, trend_features], axis=1)

        # Volatility Regime
        logger.info("Detecting volatility regime...")
        vol_features = self.detect_volatility_regime(df)
        features = pd.concat([features, vol_features], axis=1)

        # Market Stress
        logger.info("Calculating market stress...")
        stress_features = self.calculate_market_stress(df)
        features = pd.concat([features, stress_features], axis=1)

        # Hurst Exponent
        logger.info("Calculating Hurst exponent...")
        hurst_features = self.calculate_hurst_features(df)
        features = pd.concat([features, hurst_features], axis=1)

        # Combined Regime Score
        logger.info("Calculating combined regime score...")
        combined = self.calculate_combined_regime(features)
        features = pd.concat([features, combined], axis=1)

        return features

    def detect_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect trend regime using multiple indicators.

        Uses:
        - Moving average crossovers
        - Price momentum
        - Higher highs/lower lows
        """
        features = pd.DataFrame(index=df.index)
        close = df["close"]

        # Moving Averages
        sma_short = close.rolling(self.short_window).mean()
        sma_medium = close.rolling(self.medium_window).mean()
        sma_long = close.rolling(self.long_window).mean()

        # EMA for faster response
        ema_short = close.ewm(span=self.short_window).mean()
        ema_medium = close.ewm(span=self.medium_window).mean()

        # Trend Direction Indicators
        features["trend_sma_short_above_long"] = (sma_short > sma_long).astype(int)
        features["trend_sma_medium_above_long"] = (sma_medium > sma_long).astype(int)
        features["trend_price_above_sma_long"] = (close > sma_long).astype(int)

        # Trend Strength (distance from long MA)
        features["trend_strength"] = (close - sma_long) / (sma_long + 1e-10)

        # Trend Momentum
        features["trend_momentum_20d"] = close.pct_change(20)
        features["trend_momentum_60d"] = close.pct_change(60)

        # Rate of Change of Trend
        features["trend_acceleration"] = features["trend_strength"].diff(5)

        # ADX-like Trend Strength (simplified)
        high = df["high"]
        low = df["low"]

        # Directional Movement
        plus_dm: pd.Series = high.diff().fillna(0).astype(float)
        minus_dm: pd.Series = (-low.diff()).fillna(0).astype(float)
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        # True Range
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)

        # Smoothed values
        tr_smooth = tr.rolling(14).sum()
        plus_di = 100 * plus_dm.rolling(14).sum() / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).sum() / (tr_smooth + 1e-10)

        features["plus_di"] = plus_di
        features["minus_di"] = minus_di
        features["di_diff"] = plus_di - minus_di

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        features["adx"] = dx.rolling(14).mean()

        # Trend Regime Classification (numeric)
        # Score from -2 (strong bearish) to +2 (strong bullish)
        trend_score = (
            features["trend_sma_short_above_long"]
            + features["trend_sma_medium_above_long"]
            + features["trend_price_above_sma_long"]
            + np.sign(features["trend_momentum_20d"])
        ) / 4  # Normalize to [-1, 1]

        features["trend_regime_score"] = trend_score

        # Classify trend regime
        features["trend_regime"] = pd.cut(
            trend_score,
            bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
            labels=[
                TrendRegime.STRONG_BEARISH.value,
                TrendRegime.BEARISH.value,
                TrendRegime.NEUTRAL.value,
                TrendRegime.BULLISH.value,
                TrendRegime.STRONG_BULLISH.value,
            ],
        )

        # One-hot encode for ML
        for regime in TrendRegime:
            features[f"trend_is_{regime.value}"] = (
                features["trend_regime"] == regime.value
            ).astype(int)

        return features

    def detect_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volatility regime using realized volatility percentiles.
        """
        features = pd.DataFrame(index=df.index)
        returns = df["close"].pct_change()

        # Realized Volatility
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        features["vol_realized"] = realized_vol

        # Volatility Percentile (vs history)
        features["vol_percentile"] = realized_vol.rolling(
            self.vol_lookback, min_periods=60
        ).rank(pct=True)

        # Volatility Term Structure
        vol_5d = returns.rolling(5).std() * np.sqrt(252)
        vol_60d = returns.rolling(60).std() * np.sqrt(252)
        features["vol_term_structure"] = vol_5d / (vol_60d + 1e-10)

        # Volatility Trend
        features["vol_trend"] = realized_vol.pct_change(10)

        # VIX-like Fear Gauge (using high-low range)
        intraday_range = (df["high"] - df["low"]) / df["close"]
        features["fear_gauge"] = intraday_range.rolling(20).mean() * np.sqrt(252)

        # Volatility Regime Classification
        vol_pct = features["vol_percentile"]
        features["vol_regime_score"] = vol_pct

        # Classify volatility regime
        features["vol_regime"] = pd.cut(
            vol_pct,
            bins=[-np.inf, 0.25, 0.75, 0.90, np.inf],
            labels=[
                VolatilityRegime.LOW.value,
                VolatilityRegime.NORMAL.value,
                VolatilityRegime.HIGH.value,
                VolatilityRegime.EXTREME.value,
            ],
        )

        # One-hot encode for ML
        for regime in VolatilityRegime:
            features[f"vol_is_{regime.value}"] = (
                features["vol_regime"] == regime.value
            ).astype(int)

        return features

    def calculate_market_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market stress indicators.

        Combines multiple stress signals into a composite indicator.
        """
        features = pd.DataFrame(index=df.index)
        returns = df["close"].pct_change()

        # Component 1: Volatility Spike
        vol_20d = returns.rolling(20).std()
        vol_60d = returns.rolling(60).std()
        features["stress_vol_spike"] = vol_20d / (vol_60d + 1e-10)

        # Component 2: Tail Risk (negative returns)
        features["stress_tail_risk"] = returns.rolling(20).apply(
            lambda x: (x < x.quantile(0.05)).mean()
        )

        # Component 3: Correlation Breakdown
        # Higher correlation in stress periods
        volume_returns = df["volume"].pct_change()
        features["stress_price_vol_corr"] = returns.rolling(20).corr(volume_returns)

        # Component 4: Consecutive Down Days
        down_days = (returns < 0).astype(int)
        features["stress_consec_down"] = down_days.rolling(5).sum() / 5

        # Component 5: Distance from Recent High
        rolling_high = df["close"].rolling(60).max()
        features["stress_drawdown"] = (df["close"] - rolling_high) / (
            rolling_high + 1e-10
        )

        # Component 6: Range Expansion
        daily_range = (df["high"] - df["low"]) / df["close"]
        avg_range = daily_range.rolling(60).mean()
        features["stress_range_expansion"] = daily_range / (avg_range + 1e-10)

        # Composite Stress Score (0 to 1)
        # Normalize each component and combine
        stress_components = [
            (features["stress_vol_spike"] - 1).clip(0, 2) / 2,  # Vol spike above normal
            features["stress_tail_risk"] * 5,  # Scale up tail risk
            features["stress_consec_down"],
            -features["stress_drawdown"],  # Negative drawdown is stress
            (features["stress_range_expansion"] - 1).clip(0, 2) / 2,
        ]

        features["market_stress_score"] = pd.Series(
            sum(stress_components) / len(stress_components), index=df.index
        ).clip(0, 1)

        # Stress Level Classification
        features["market_stress_regime"] = pd.cut(
            features["market_stress_score"],
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=["low", "moderate", "high"],
        )

        return features

    def calculate_hurst_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Hurst exponent for regime detection.

        H > 0.5: Trending (momentum regime)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting
        """
        features = pd.DataFrame(index=df.index)

        # Calculate Hurst Exponent using R/S method
        def hurst_rs(ts, max_lag: int = 20):
            """Calculate Hurst exponent using rescaled range method."""
            if len(ts) < max_lag * 2:
                return 0.5

            lags = range(2, min(max_lag, len(ts) // 4))
            if len(lags) < 3:
                return 0.5

            try:
                rs_values = []
                for lag in lags:
                    # Split into subseries
                    subseries = np.array_split(ts, len(ts) // lag)
                    rs_subseries = []

                    for sub in subseries:
                        if len(sub) < 2:
                            continue
                        mean = np.mean(sub)
                        std = np.std(sub)
                        if std == 0:
                            continue

                        # Cumulative deviation from mean
                        cumdev = np.cumsum(sub - mean)
                        r = max(cumdev) - min(cumdev)
                        rs_subseries.append(r / std)

                    if rs_subseries:
                        rs_values.append(np.mean(rs_subseries))

                if len(rs_values) < 3:
                    return 0.5

                log_lags = np.log(list(lags)[: len(rs_values)])
                log_rs = np.log(rs_values)

                poly = np.polyfit(log_lags, log_rs, 1)
                return np.clip(poly[0], 0, 1)

            except Exception:
                return 0.5

        # Rolling Hurst calculation
        close = df["close"].values
        hurst_values = []

        window = 100
        for i in range(len(close)):
            if i < window:
                hurst_values.append(0.5)
            else:
                h = hurst_rs(close[i - window : i])
                hurst_values.append(h)

        features["hurst_exponent"] = hurst_values

        # Hurst-based regime
        features["hurst_momentum_regime"] = (features["hurst_exponent"] > 0.55).astype(
            int
        )
        features["hurst_meanrevert_regime"] = (
            features["hurst_exponent"] < 0.45
        ).astype(int)
        features["hurst_random_regime"] = (
            (features["hurst_exponent"] >= 0.45) & (features["hurst_exponent"] <= 0.55)
        ).astype(int)

        # Hurst trend
        features["hurst_trend"] = (
            features["hurst_exponent"].rolling(20).mean()
            - features["hurst_exponent"].rolling(60).mean()
        )

        return features

    def calculate_combined_regime(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate combined regime score from all regime indicators.
        """
        combined = pd.DataFrame(index=features.index)

        # Risk-On/Risk-Off Score
        # Positive = Risk-On, Negative = Risk-Off
        trend_score = features.get(
            "trend_regime_score", pd.Series(0, index=features.index)
        )
        vol_pct = features.get("vol_percentile", pd.Series(0.5, index=features.index))
        stress_score = features.get(
            "market_stress_score", pd.Series(0.5, index=features.index)
        )

        # Convert to Series if needed
        if not isinstance(trend_score, pd.Series):
            trend_score = pd.Series(trend_score, index=features.index)
        if not isinstance(vol_pct, pd.Series):
            vol_pct = pd.Series(vol_pct, index=features.index)
        if not isinstance(stress_score, pd.Series):
            stress_score = pd.Series(stress_score, index=features.index)

        risk_score = trend_score * 0.4 - (vol_pct - 0.5) * 0.3 - stress_score * 0.3

        combined["risk_regime_score"] = risk_score

        # Market Regime Classification
        combined["market_regime"] = pd.cut(
            risk_score.astype(float),
            bins=[-np.inf, -0.2, 0.2, np.inf],
            labels=[
                MarketRegime.RISK_OFF.value,
                MarketRegime.TRANSITIONING.value,
                MarketRegime.RISK_ON.value,
            ],
        )

        # One-hot encode for ML
        for regime in MarketRegime:
            combined[f"market_is_{regime.value}"] = (
                combined["market_regime"] == regime.value
            ).astype(int)

        # Regime Confidence (how clear is the regime?)
        combined["regime_confidence"] = risk_score.abs()

        # Regime Change Detection
        combined["regime_change"] = (
            combined["market_regime"] != combined["market_regime"].shift(1)
        ).astype(int)

        # Days Since Regime Change
        regime_changes = combined["regime_change"].cumsum()
        combined["days_in_regime"] = combined.groupby(regime_changes).cumcount() + 1

        return combined

    def get_current_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Get the current regime state.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeState with current regime information
        """
        features = self.detect_all_regimes(df)
        last_row = features.iloc[-1]

        # Parse trend regime
        trend_val = last_row.get("trend_regime", TrendRegime.NEUTRAL.value)
        try:
            trend = TrendRegime(trend_val)
        except ValueError:
            trend = TrendRegime.NEUTRAL

        # Parse volatility regime
        vol_val = last_row.get("vol_regime", VolatilityRegime.NORMAL.value)
        try:
            volatility = VolatilityRegime(vol_val)
        except ValueError:
            volatility = VolatilityRegime.NORMAL

        # Parse market regime
        market_val = last_row.get("market_regime", MarketRegime.TRANSITIONING.value)
        try:
            market = MarketRegime(market_val)
        except ValueError:
            market = MarketRegime.TRANSITIONING

        return RegimeState(
            trend=trend,
            volatility=volatility,
            market=market,
            hurst=last_row.get("hurst_exponent", 0.5),
            stress_level=last_row.get("market_stress_score", 0.5),
            confidence=last_row.get("regime_confidence", 0.5),
        )
