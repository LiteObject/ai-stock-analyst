"""
Cross-Asset Feature Engineering for Multi-Asset Portfolio Analysis.

This module provides features that capture relationships between multiple assets:
- Correlation dynamics
- Lead-lag relationships
- Sector/market factor exposures
- Pair trading signals
- Contagion indicators
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CrossAssetFeatures:
    """
    Generates features that capture cross-asset relationships.

    These features are essential for:
    - Portfolio optimization
    - Risk management
    - Identifying diversification opportunities
    - Detecting market regime changes
    """

    def __init__(
        self,
        correlation_windows: Optional[List[int]] = None,
        market_ticker: str = "SPY",
    ):
        """
        Initialize CrossAssetFeatures.

        Args:
            correlation_windows: Windows for rolling correlation calculations
            market_ticker: Market benchmark ticker
        """
        self.correlation_windows = correlation_windows or [20, 60, 120]
        self.market_ticker = market_ticker

    def create_all_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Create all cross-asset features for a target ticker.

        Args:
            price_data: Dictionary of ticker -> OHLCV DataFrames
            target_ticker: The ticker to create features for

        Returns:
            DataFrame with cross-asset features
        """
        if target_ticker not in price_data:
            raise ValueError(f"Target ticker {target_ticker} not in price_data")

        target_df = price_data[target_ticker].copy()
        target_df.columns = target_df.columns.str.lower()
        features = pd.DataFrame(index=target_df.index)

        # Get returns for all tickers
        returns_dict = self._calculate_returns(price_data)

        # Create features
        logger.info(f"Creating cross-asset features for {target_ticker}...")

        # 1. Correlation Features
        corr_features = self.create_correlation_features(returns_dict, target_ticker)
        features = pd.concat([features, corr_features], axis=1)

        # 2. Beta/Market Sensitivity Features
        beta_features = self.create_beta_features(returns_dict, target_ticker)
        features = pd.concat([features, beta_features], axis=1)

        # 3. Lead-Lag Features
        lead_lag_features = self.create_lead_lag_features(returns_dict, target_ticker)
        features = pd.concat([features, lead_lag_features], axis=1)

        # 4. Relative Strength Features
        rs_features = self.create_relative_strength_features(price_data, target_ticker)
        features = pd.concat([features, rs_features], axis=1)

        # 5. Dispersion Features
        disp_features = self.create_dispersion_features(returns_dict, target_ticker)
        features = pd.concat([features, disp_features], axis=1)

        # 6. Cointegration Features
        coint_features = self.create_cointegration_features(price_data, target_ticker)
        features = pd.concat([features, coint_features], axis=1)

        return features

    def _calculate_returns(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """Calculate returns for all tickers."""
        returns_dict = {}
        for ticker, df in price_data.items():
            df_copy = df.copy()
            df_copy.columns = df_copy.columns.str.lower()
            returns_dict[ticker] = df_copy["close"].pct_change()
        return returns_dict

    def create_correlation_features(
        self,
        returns_dict: Dict[str, pd.Series],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Create correlation-based features.

        Features:
        - Rolling correlation with market
        - Rolling correlation with each asset
        - Correlation regime changes
        - Average correlation with universe
        """
        features = pd.DataFrame(index=returns_dict[target_ticker].index)
        target_returns = returns_dict[target_ticker]

        # Correlation with market
        if self.market_ticker in returns_dict:
            market_returns = returns_dict[self.market_ticker]
            for window in self.correlation_windows:
                features[f"corr_market_{window}d"] = target_returns.rolling(
                    window, min_periods=int(window * 0.5)
                ).corr(market_returns)

            # Correlation change
            features["corr_market_change"] = features.get(
                f"corr_market_{self.correlation_windows[0]}d", 0
            ) - features.get(f"corr_market_{self.correlation_windows[-1]}d", 0)

        # Correlation with all other assets
        corr_cols = []
        for ticker, returns in returns_dict.items():
            if ticker == target_ticker:
                continue

            for window in self.correlation_windows:
                col_name = f"corr_{ticker}_{window}d"
                features[col_name] = target_returns.rolling(
                    window, min_periods=int(window * 0.5)
                ).corr(returns)
                corr_cols.append(col_name)

        # Average correlation with universe
        for window in self.correlation_windows:
            window_corr_cols = [c for c in corr_cols if f"_{window}d" in c]
            if window_corr_cols:
                features[f"avg_corr_{window}d"] = features[window_corr_cols].mean(
                    axis=1
                )

        # Correlation dispersion (how different are correlations?)
        for window in self.correlation_windows:
            window_corr_cols = [c for c in corr_cols if f"_{window}d" in c]
            if window_corr_cols:
                features[f"corr_dispersion_{window}d"] = features[window_corr_cols].std(
                    axis=1
                )

        return features

    def create_beta_features(
        self,
        returns_dict: Dict[str, pd.Series],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Create beta and market sensitivity features.

        Features:
        - Rolling beta to market
        - Up/down beta asymmetry
        - Beta stability
        - Idiosyncratic volatility
        """
        features = pd.DataFrame(index=returns_dict[target_ticker].index)
        target_returns = returns_dict[target_ticker]

        if self.market_ticker not in returns_dict:
            return features

        market_returns = returns_dict[self.market_ticker]

        for window in self.correlation_windows:
            # Rolling beta
            cov = target_returns.rolling(window, min_periods=int(window * 0.5)).cov(
                market_returns
            )
            var = market_returns.rolling(window, min_periods=int(window * 0.5)).var()
            features[f"beta_{window}d"] = cov / (var + 1e-10)

            # Up beta (beta when market is up)
            up_mask = market_returns > 0
            up_cov = (
                target_returns.where(up_mask)
                .rolling(window, min_periods=int(window * 0.25))
                .cov(market_returns.where(up_mask))
            )
            up_var = (
                market_returns.where(up_mask)
                .rolling(window, min_periods=int(window * 0.25))
                .var()
            )
            features[f"beta_up_{window}d"] = up_cov / (up_var + 1e-10)

            # Down beta (beta when market is down)
            down_mask = market_returns < 0
            down_cov = (
                target_returns.where(down_mask)
                .rolling(window, min_periods=int(window * 0.25))
                .cov(market_returns.where(down_mask))
            )
            down_var = (
                market_returns.where(down_mask)
                .rolling(window, min_periods=int(window * 0.25))
                .var()
            )
            features[f"beta_down_{window}d"] = down_cov / (down_var + 1e-10)

            # Beta asymmetry (downside capture ratio)
            features[f"beta_asymmetry_{window}d"] = (
                features[f"beta_down_{window}d"] - features[f"beta_up_{window}d"]
            )

        # Beta stability (std of rolling beta)
        beta_cols = [
            c
            for c in features.columns
            if c.startswith("beta_")
            and "d" in c
            and "up" not in c
            and "down" not in c
            and "asymmetry" not in c
        ]
        if beta_cols:
            features["beta_stability"] = features[beta_cols].std(axis=1)

        # Idiosyncratic volatility (residual vol after removing market exposure)
        for window in self.correlation_windows:
            beta = features.get(f"beta_{window}d", 0)
            if isinstance(beta, pd.Series):
                residual = target_returns - beta * market_returns
                features[f"idio_vol_{window}d"] = residual.rolling(
                    window, min_periods=int(window * 0.5)
                ).std() * np.sqrt(252)

        return features

    def create_lead_lag_features(
        self,
        returns_dict: Dict[str, pd.Series],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Create lead-lag relationship features.

        Identifies which assets lead or lag the target.
        """
        features = pd.DataFrame(index=returns_dict[target_ticker].index)
        target_returns = returns_dict[target_ticker]

        max_lag = 5
        window = 60

        for ticker, returns in returns_dict.items():
            if ticker == target_ticker:
                continue

            # Cross-correlation at different lags
            best_lag = 0
            best_corr = 0

            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue

                if lag > 0:
                    # Other asset leads target
                    lagged_returns = returns.shift(lag)
                else:
                    # Target leads other asset
                    lagged_returns = returns.shift(lag)

                corr = target_returns.rolling(window).corr(lagged_returns)
                current_corr = (
                    corr.iloc[-1] if len(corr) > 0 and not pd.isna(corr.iloc[-1]) else 0
                )

                if abs(current_corr) > abs(best_corr):
                    best_lag = lag
                    best_corr = current_corr

            # Lead-lag indicator
            # Positive = other asset leads, Negative = target leads
            features[f"lead_lag_{ticker}"] = best_lag
            features[f"lead_lag_corr_{ticker}"] = best_corr

        # Summary features
        lead_cols = [
            c
            for c in features.columns
            if c.startswith("lead_lag_") and "_corr_" not in c
        ]
        if lead_cols:
            # Average lead-lag
            features["avg_lead_lag"] = features[lead_cols].mean(axis=1)
            # Number of assets that lead the target
            features["n_leaders"] = (features[lead_cols] > 0).sum(axis=1)

        return features

    def create_relative_strength_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Create relative strength features.

        Measures performance relative to other assets and the market.
        """
        features = pd.DataFrame()
        target_df = price_data[target_ticker].copy()
        target_df.columns = target_df.columns.str.lower()
        target_close = target_df["close"]
        features.index = target_df.index

        # Relative strength vs market
        if self.market_ticker in price_data:
            market_df = price_data[self.market_ticker].copy()
            market_df.columns = market_df.columns.str.lower()
            market_close = market_df["close"]

            for window in [10, 20, 60, 120]:
                target_ret = target_close.pct_change(window)
                market_ret = market_close.pct_change(window)
                features[f"rs_vs_market_{window}d"] = target_ret - market_ret

        # Relative strength rank within universe
        all_returns = {}
        for ticker, df in price_data.items():
            df_copy = df.copy()
            df_copy.columns = df_copy.columns.str.lower()
            all_returns[ticker] = df_copy["close"].pct_change(20)

        returns_df = pd.DataFrame(all_returns)

        # Percentile rank of target
        if target_ticker in returns_df.columns:
            features["rs_rank_20d"] = returns_df.rank(axis=1, pct=True)[target_ticker]

        # Momentum rank (60-day)
        all_momentum = {}
        for ticker, df in price_data.items():
            df_copy = df.copy()
            df_copy.columns = df_copy.columns.str.lower()
            all_momentum[ticker] = df_copy["close"].pct_change(60)

        momentum_df = pd.DataFrame(all_momentum)
        if target_ticker in momentum_df.columns:
            features["momentum_rank_60d"] = momentum_df.rank(axis=1, pct=True)[
                target_ticker
            ]

        # Relative strength change
        if "rs_vs_market_20d" in features.columns:
            features["rs_change_10d"] = features["rs_vs_market_20d"].diff(10)

        return features

    def create_dispersion_features(
        self,
        returns_dict: Dict[str, pd.Series],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Create market dispersion features.

        High dispersion often indicates stock-picking opportunities.
        """
        features = pd.DataFrame(index=returns_dict[target_ticker].index)

        # Combine all returns
        returns_df = pd.DataFrame(returns_dict)

        for window in [20, 60]:
            # Cross-sectional volatility (dispersion)
            features[f"cross_sec_vol_{window}d"] = (
                returns_df.rolling(window).std().mean(axis=1)
            )

            # Cross-sectional return dispersion
            rolling_returns = returns_df.rolling(window).mean()
            features[f"return_dispersion_{window}d"] = rolling_returns.std(axis=1)

            # Cross-sectional correlation
            corr_matrix = returns_df.rolling(window).corr()
            # Average pairwise correlation (simplified)
            avg_corr = []
            for date in returns_df.index:
                try:
                    if date in corr_matrix.index.get_level_values(0):
                        corr_slice = corr_matrix.loc[date]
                        mask = ~np.eye(len(corr_slice), dtype=bool)
                        avg_corr.append(corr_slice.values[mask].mean())
                    else:
                        avg_corr.append(np.nan)
                except Exception:
                    avg_corr.append(np.nan)

            features[f"avg_pairwise_corr_{window}d"] = avg_corr

        # Target's deviation from cross-sectional mean
        target_returns = returns_dict[target_ticker]
        cross_sec_mean = returns_df.mean(axis=1)
        cross_sec_std = returns_df.std(axis=1)
        features["zscore_vs_universe"] = (target_returns - cross_sec_mean) / (
            cross_sec_std + 1e-10
        )

        return features

    def create_cointegration_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        target_ticker: str,
        window: int = 252,
    ) -> pd.DataFrame:
        """
        Create cointegration-based features for pair trading.

        Identifies mean-reverting relationships with other assets.
        """
        features = pd.DataFrame()
        target_df = price_data[target_ticker].copy()
        target_df.columns = target_df.columns.str.lower()
        target_close = target_df["close"]
        features.index = target_df.index

        # Cointegration with each asset
        for ticker, df in price_data.items():
            if ticker == target_ticker:
                continue

            df_copy = df.copy()
            df_copy.columns = df_copy.columns.str.lower()
            other_close = df_copy["close"]

            # Calculate spread z-score (simplified cointegration)
            log_target_series = pd.Series(
                np.log(target_close + 1e-10), index=target_close.index
            )
            log_other_series = pd.Series(
                np.log(other_close + 1e-10), index=other_close.index
            )

            # Rolling regression for hedge ratio
            hedge_ratios = []
            z_scores = []

            for i in range(len(target_close)):
                if i < window:
                    hedge_ratios.append(np.nan)
                    z_scores.append(np.nan)
                    continue

                y = np.array(
                    log_target_series.iloc[i - window : i].values, dtype=np.float64
                )
                x = np.array(
                    log_other_series.iloc[i - window : i].values, dtype=np.float64
                )

                if len(y) < 10 or len(x) < 10:
                    hedge_ratios.append(np.nan)
                    z_scores.append(np.nan)
                    continue

                try:
                    # Simple OLS using numpy polyfit
                    coeffs = np.polyfit(x, y, 1)
                    slope: float = coeffs[0]
                    intercept: float = coeffs[1]
                    spread = (
                        log_target_series.iloc[i]
                        - slope * log_other_series.iloc[i]
                        - intercept
                    )
                    spread_series = (
                        log_target_series.iloc[i - window : i]
                        - slope * log_other_series.iloc[i - window : i]
                        - intercept
                    )
                    spread_mean = float(spread_series.mean())
                    spread_std = float(spread_series.std())

                    if spread_std > 0:
                        z = (spread - spread_mean) / spread_std
                    else:
                        z = 0

                    hedge_ratios.append(slope)
                    z_scores.append(z)
                except Exception:
                    hedge_ratios.append(np.nan)
                    z_scores.append(np.nan)

            features[f"coint_hedge_{ticker}"] = hedge_ratios
            features[f"coint_zscore_{ticker}"] = z_scores

        # Summary features
        zscore_cols = [c for c in features.columns if "zscore" in c]
        if zscore_cols:
            # Best pair (most extreme z-score)
            features["best_pair_zscore"] = features[zscore_cols].abs().max(axis=1)
            # Average z-score (divergence from pairs)
            features["avg_pair_zscore"] = features[zscore_cols].mean(axis=1)

        return features

    def calculate_sector_exposures(
        self,
        returns_dict: Dict[str, pd.Series],
        sector_map: Dict[str, str],
        target_ticker: str,
    ) -> pd.DataFrame:
        """
        Calculate sector factor exposures.

        Args:
            returns_dict: Dictionary of ticker -> returns Series
            sector_map: Dictionary of ticker -> sector name
            target_ticker: Target ticker to calculate exposures for

        Returns:
            DataFrame with sector exposure features
        """
        features = pd.DataFrame(index=returns_dict[target_ticker].index)
        target_returns = returns_dict[target_ticker]

        # Group assets by sector
        sector_returns = {}
        for ticker, returns in returns_dict.items():
            if ticker == target_ticker:
                continue
            sector = sector_map.get(ticker, "Unknown")
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(returns)

        # Calculate sector factor returns (equal weight)
        for sector, returns_list in sector_returns.items():
            if returns_list:
                sector_df = pd.concat(returns_list, axis=1)
                sector_return = sector_df.mean(axis=1)

                # Beta to sector
                for window in [20, 60]:
                    cov = target_returns.rolling(window).cov(sector_return)
                    var = sector_return.rolling(window).var()
                    features[f"beta_{sector}_{window}d"] = cov / (var + 1e-10)

        return features
