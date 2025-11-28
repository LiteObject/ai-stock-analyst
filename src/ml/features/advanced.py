"""
Advanced Feature Engineering for ML Models.

This module provides sophisticated feature engineering including:
- Market microstructure features
- Volume-price analysis
- Volatility clustering
- Interaction features between top predictors
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for improved ML accuracy.

    Creates features that capture market microstructure, volatility dynamics,
    and complex interactions between predictors.
    """

    def __init__(self):
        self.feature_importance: Dict[str, float] = {}
        self._feature_stats: Dict[str, Dict] = {}

    def create_all_features(
        self,
        df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Create all advanced features from OHLCV data.

        Args:
            df: DataFrame with open, high, low, close, volume columns
            market_df: Optional market benchmark data (e.g., SPY)
            sentiment_data: Optional sentiment indicators

        Returns:
            DataFrame with all advanced features
        """
        df = df.copy()
        df.columns = df.columns.str.lower()

        features = pd.DataFrame(index=df.index)

        # Market Microstructure Features
        logger.info("Creating market microstructure features...")
        microstructure = self.create_market_microstructure_features(df)
        features = pd.concat([features, microstructure], axis=1)

        # Volatility Features
        logger.info("Creating volatility features...")
        volatility = self.create_volatility_features(df)
        features = pd.concat([features, volatility], axis=1)

        # Volume-Price Features
        logger.info("Creating volume-price features...")
        volume_price = self.create_volume_price_features(df)
        features = pd.concat([features, volume_price], axis=1)

        # Cross-sectional features if market data provided
        if market_df is not None:
            logger.info("Creating cross-sectional features...")
            cross_sectional = self.create_cross_sectional_features(df, market_df)
            features = pd.concat([features, cross_sectional], axis=1)

        # Alternative data features if provided
        if sentiment_data is not None:
            logger.info("Creating alternative data features...")
            alt_data = self.create_alternative_data_features(df, sentiment_data)
            features = pd.concat([features, alt_data], axis=1)

        # Statistical features
        logger.info("Creating statistical features...")
        statistical = self.create_statistical_features(df)
        features = pd.concat([features, statistical], axis=1)

        return features

    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market microstructure features.

        These features capture the underlying market dynamics and
        order flow characteristics.
        """
        features = pd.DataFrame(index=df.index)

        # Bid-Ask Spread Proxy (using high-low range)
        features["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        features["spread_proxy_ma5"] = features["spread_proxy"].rolling(5).mean()
        features["spread_proxy_std5"] = features["spread_proxy"].rolling(5).std()

        # Kyle's Lambda (price impact measure)
        # Approximation: |return| / log(volume)
        returns = df["close"].pct_change()
        log_volume = np.log1p(df["volume"])
        features["kyle_lambda"] = np.abs(returns) / (log_volume + 1e-10)
        features["kyle_lambda_ma10"] = features["kyle_lambda"].rolling(10).mean()

        # Amihud Illiquidity Ratio
        # |return| / dollar_volume
        dollar_volume = df["close"] * df["volume"]
        features["amihud_illiquidity"] = np.abs(returns) / (dollar_volume + 1e-10)
        features["amihud_illiquidity_ma20"] = (
            features["amihud_illiquidity"].rolling(20).mean()
        )

        # Roll's Spread Estimator
        # 2 * sqrt(-cov(r_t, r_{t-1})) if negative, else 0
        def calc_roll_spread(x):
            if len(x) < 2:
                return 0
            cov = np.cov(x[1:], x[:-1])[0, 1]
            return 2 * np.sqrt(-cov) if cov < 0 else 0

        features["roll_spread"] = returns.rolling(20).apply(calc_roll_spread, raw=True)

        # Intraday Range Ratio (measures price range relative to volume)
        features["range_volume_ratio"] = (df["high"] - df["low"]) / (
            np.log1p(df["volume"]) + 1e-10
        )

        # Close Location Value (where close is within the day's range)
        features["close_location"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        return features

    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility and volatility clustering features.

        Captures different measures of volatility and their dynamics.
        """
        features = pd.DataFrame(index=df.index)
        returns = df["close"].pct_change()

        # Realized Volatility (multiple windows)
        for window in [5, 10, 20, 60]:
            features[f"realized_vol_{window}d"] = returns.rolling(
                window
            ).std() * np.sqrt(252)

        # Parkinson Volatility (uses high-low range)
        # More efficient than close-close volatility
        log_hl = np.log(df["high"] / df["low"])
        features["parkinson_vol"] = np.sqrt(
            log_hl.rolling(20).apply(lambda x: (x**2).sum() / (4 * len(x) * np.log(2)))
        ) * np.sqrt(252)

        # Garman-Klass Volatility (uses OHLC)
        log_hl_sq = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"])
        log_co_sq = log_co**2
        gk_var = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
        features["garman_klass_vol"] = np.sqrt(gk_var.rolling(20).mean()) * np.sqrt(252)

        # Volatility of Volatility
        features["vol_of_vol"] = features["realized_vol_20d"].rolling(20).std()

        # Volatility Regime (percentile-based)
        features["vol_percentile"] = (
            features["realized_vol_20d"].rolling(252, min_periods=60).rank(pct=True)
        )

        # Volatility Term Structure (short vs long term)
        features["vol_term_structure"] = features["realized_vol_5d"] / (
            features["realized_vol_60d"] + 1e-10
        )

        # GARCH-like conditional volatility (simplified)
        # Using exponential weighted moving std as proxy
        features["ewm_volatility"] = returns.ewm(span=20).std() * np.sqrt(252)

        # Volatility Skew (asymmetry in returns)
        features["vol_skew"] = returns.rolling(20).skew()

        # Volatility Kurtosis (tail risk)
        features["vol_kurtosis"] = returns.rolling(20).kurt()

        return features

    def create_volume_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-price relationship features.

        Captures the interplay between price movements and volume.
        """
        features = pd.DataFrame(index=df.index)
        returns = df["close"].pct_change()

        # On-Balance Volume (OBV)
        obv = (np.sign(returns) * df["volume"]).cumsum()
        features["obv"] = obv
        features["obv_ma20"] = obv.rolling(20).mean()
        features["obv_slope"] = obv.diff(5) / 5

        # Volume-Price Trend
        # Cumulative (volume * price_change)
        features["vpt"] = (returns * df["volume"]).cumsum()

        # Accumulation/Distribution Line
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + 1e-10
        )
        features["ad_line"] = (clv * df["volume"]).cumsum()

        # Money Flow Multiplier and Volume
        features["mf_multiplier"] = clv
        features["mf_volume"] = clv * df["volume"]

        # Volume Force Index
        features["force_index"] = returns * df["volume"]
        features["force_index_ma13"] = features["force_index"].rolling(13).mean()

        # Relative Volume (vs moving average)
        features["relative_volume"] = df["volume"] / (
            df["volume"].rolling(20).mean() + 1e-10
        )

        # Volume Momentum
        features["volume_momentum"] = df["volume"].pct_change(5)

        # Price-Volume Correlation
        features["price_volume_corr"] = returns.rolling(20).corr(
            df["volume"].pct_change()
        )

        # Volume Weighted Price Momentum
        vwap = (df["close"] * df["volume"]).rolling(20).sum() / (
            df["volume"].rolling(20).sum() + 1e-10
        )
        features["price_vs_vwap"] = (df["close"] - vwap) / (vwap + 1e-10)

        # Chaikin Money Flow
        mf_volume = clv * df["volume"]
        features["cmf"] = mf_volume.rolling(20).sum() / (
            df["volume"].rolling(20).sum() + 1e-10
        )

        return features

    def create_cross_sectional_features(
        self, df: pd.DataFrame, market_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create cross-sectional features relative to market benchmark.

        Args:
            df: Stock OHLCV data
            market_df: Market benchmark OHLCV data (e.g., SPY)
        """
        features = pd.DataFrame(index=df.index)

        # Align data
        market_df = market_df.reindex(df.index)

        stock_returns = df["close"].pct_change()
        market_returns = market_df["close"].pct_change()

        # Relative Strength
        features["relative_return_1d"] = stock_returns - market_returns
        features["relative_return_5d"] = df["close"].pct_change(5) - market_df[
            "close"
        ].pct_change(5)
        features["relative_return_20d"] = df["close"].pct_change(20) - market_df[
            "close"
        ].pct_change(20)

        # Rolling Beta
        def calc_beta(y, x):
            if len(y) < 10 or y.std() == 0 or x.std() == 0:
                return 1.0
            cov = np.cov(y, x)[0, 1]
            var = np.var(x)
            return cov / var if var > 0 else 1.0

        features["beta_60d"] = stock_returns.rolling(60).apply(
            lambda y: calc_beta(y, market_returns.loc[y.index]), raw=False
        )

        # Alpha (Jensen's Alpha approximation)
        risk_free_daily = 0.02 / 252  # 2% annual risk-free rate
        expected_return = risk_free_daily + features["beta_60d"] * (
            market_returns.rolling(60).mean() - risk_free_daily
        )
        features["alpha_60d"] = stock_returns.rolling(60).mean() - expected_return

        # Correlation with Market
        features["market_corr_30d"] = stock_returns.rolling(30).corr(market_returns)
        features["market_corr_change"] = features["market_corr_30d"].diff(10)

        # Relative Volume
        features["relative_volume_vs_market"] = (
            df["volume"] / df["volume"].rolling(20).mean()
        ) / (market_df["volume"] / market_df["volume"].rolling(20).mean() + 1e-10)

        # Idiosyncratic Volatility
        residuals = stock_returns - features["beta_60d"] * market_returns
        features["idio_vol"] = residuals.rolling(20).std() * np.sqrt(252)

        # Information Ratio (vs market)
        tracking_diff = stock_returns - market_returns
        features["info_ratio_60d"] = (
            tracking_diff.rolling(60).mean()
            / (tracking_diff.rolling(60).std() + 1e-10)
            * np.sqrt(252)
        )

        return features

    def create_alternative_data_features(
        self, df: pd.DataFrame, sentiment_data: Dict
    ) -> pd.DataFrame:
        """
        Create features from alternative data sources.

        Args:
            df: OHLCV data
            sentiment_data: Dictionary with sentiment indicators
        """
        features = pd.DataFrame(index=df.index)

        # News Sentiment
        if "news_sentiment" in sentiment_data:
            news = pd.Series(sentiment_data["news_sentiment"], index=df.index)
            features["news_sentiment"] = news
            features["news_sentiment_ma5"] = news.rolling(5).mean()
            features["news_sentiment_momentum"] = news.diff(5)
            features["news_sentiment_zscore"] = (news - news.rolling(20).mean()) / (
                news.rolling(20).std() + 1e-10
            )

        # Social Media Volume
        if "social_volume" in sentiment_data:
            social = pd.Series(sentiment_data["social_volume"], index=df.index)
            features["social_volume"] = social
            features["social_volume_zscore"] = (social - social.rolling(20).mean()) / (
                social.rolling(20).std() + 1e-10
            )

        # Options Flow
        if "put_call_ratio" in sentiment_data:
            pcr = pd.Series(sentiment_data["put_call_ratio"], index=df.index)
            features["put_call_ratio"] = pcr
            features["pcr_zscore"] = (pcr - pcr.rolling(20).mean()) / (
                pcr.rolling(20).std() + 1e-10
            )
            features["pcr_ma5"] = pcr.rolling(5).mean()

        # Fear & Greed Index
        if "fear_greed" in sentiment_data:
            fg = pd.Series(sentiment_data["fear_greed"], index=df.index)
            features["fear_greed"] = fg
            features["fear_greed_zscore"] = (fg - fg.rolling(20).mean()) / (
                fg.rolling(20).std() + 1e-10
            )

        # Analyst Ratings
        if "analyst_rating" in sentiment_data:
            rating = pd.Series(sentiment_data["analyst_rating"], index=df.index)
            features["analyst_rating"] = rating
            features["analyst_rating_change"] = rating.diff(20)

        return features

    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical and distributional features.
        """
        features = pd.DataFrame(index=df.index)
        returns = df["close"].pct_change()

        # Return Distribution Features
        features["return_skew_20d"] = returns.rolling(20).skew()
        features["return_kurtosis_20d"] = returns.rolling(20).kurt()

        # Percentile Ranks
        features["return_percentile"] = returns.rolling(252).rank(pct=True)
        features["price_percentile"] = df["close"].rolling(252).rank(pct=True)

        # Z-Scores
        features["return_zscore"] = (returns - returns.rolling(20).mean()) / (
            returns.rolling(20).std() + 1e-10
        )
        features["price_zscore"] = (df["close"] - df["close"].rolling(20).mean()) / (
            df["close"].rolling(20).std() + 1e-10
        )

        # Autocorrelation
        features["return_autocorr_1"] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        features["return_autocorr_5"] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0, raw=False
        )

        # Maximum Drawdown (rolling)
        def max_drawdown(prices):
            if len(prices) < 2:
                return 0
            peak = prices.expanding().max()
            dd = (prices - peak) / peak
            return dd.min()

        features["max_dd_20d"] = df["close"].rolling(20).apply(max_drawdown, raw=False)

        # Up/Down Days Ratio
        up_days = (returns > 0).rolling(20).sum()
        features["up_days_ratio"] = up_days / 20

        # Consecutive Up/Down Days
        sign_changes = np.sign(returns).diff().ne(0).cumsum()
        features["streak_length"] = returns.groupby(sign_changes).cumcount() + 1

        return features

    def create_interaction_features(
        self, features_df: pd.DataFrame, top_k: int = 15
    ) -> pd.DataFrame:
        """
        Create interaction features between top features.

        Args:
            features_df: DataFrame with existing features
            top_k: Number of top features to create interactions for
        """
        interaction_features = pd.DataFrame(index=features_df.index)

        # Select numeric columns only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Use variance as proxy for importance if no importance scores
        if self.feature_importance:
            sorted_features = sorted(
                [
                    (f, v)
                    for f, v in self.feature_importance.items()
                    if f in numeric_cols
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            top_feature_names = [f[0] for f in sorted_features]
        else:
            variances = features_df[numeric_cols].var()
            top_feature_names = variances.nlargest(top_k).index.tolist()

        logger.info(f"Creating interactions for top {len(top_feature_names)} features")

        # Create pairwise interactions
        for i, feat1 in enumerate(top_feature_names):
            for feat2 in top_feature_names[i + 1 :]:
                if feat1 in features_df.columns and feat2 in features_df.columns:
                    # Multiplication
                    interaction_features[f"{feat1}_x_{feat2}"] = (
                        features_df[feat1] * features_df[feat2]
                    )
                    # Ratio (with safe division)
                    denominator = features_df[feat2].replace(0, np.nan)
                    interaction_features[f"{feat1}_div_{feat2}"] = (
                        features_df[feat1] / denominator
                    )

        return interaction_features

    def calculate_hurst_exponent(
        self, series: pd.Series, max_lag: int = 20
    ) -> pd.Series:
        """
        Calculate rolling Hurst exponent for regime detection.

        H > 0.5: Trending (momentum)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting
        """

        def hurst(ts):
            if len(ts) < max_lag:
                return 0.5

            lags = range(2, min(max_lag, len(ts) // 2))
            if len(lags) < 3:
                return 0.5

            try:
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                tau = [t for t in tau if t > 0]  # Filter out zeros
                if len(tau) < 3:
                    return 0.5

                log_lags = np.log(list(lags)[: len(tau)])
                log_tau = np.log(tau)

                poly = np.polyfit(log_lags, log_tau, 1)
                return poly[0] * 2.0
            except Exception:
                return 0.5

        return series.rolling(100, min_periods=50).apply(hurst, raw=True)

    def update_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """Update feature importance scores from trained models."""
        for feature, importance in importance_dict.items():
            if feature in self.feature_importance:
                # Average with existing importance
                self.feature_importance[feature] = (
                    self.feature_importance[feature] + importance
                ) / 2
            else:
                self.feature_importance[feature] = importance

    def get_feature_stats(self, features_df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistics for all features."""
        stats = {}
        for col in features_df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": features_df[col].mean(),
                "std": features_df[col].std(),
                "min": features_df[col].min(),
                "max": features_df[col].max(),
                "null_pct": features_df[col].isnull().mean(),
                "inf_pct": np.isinf(features_df[col]).mean(),
            }
        self._feature_stats = stats
        return stats
