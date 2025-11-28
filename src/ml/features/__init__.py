"""
ML Feature Engineering Module.

This module provides comprehensive feature engineering for ML models including:
- Technical indicators (using pandas-ta)
- Fundamental analysis features
- Sentiment analysis features
- Market regime detection
- Cross-asset relationship features
- Advanced market microstructure features
"""

from .advanced import AdvancedFeatureEngineer
from .cross_asset import CrossAssetFeatures
from .fundamental import FundamentalFeatures
from .regime import (
    MarketRegime,
    RegimeDetector,
    RegimeState,
    TrendRegime,
    VolatilityRegime,
)
from .sentiment import SentimentFeatures
from .technical import TechnicalFeatures

__all__ = [
    # Core feature classes
    "TechnicalFeatures",
    "FundamentalFeatures",
    "SentimentFeatures",
    # Advanced features
    "AdvancedFeatureEngineer",
    "CrossAssetFeatures",
    # Regime detection
    "RegimeDetector",
    "RegimeState",
    "TrendRegime",
    "VolatilityRegime",
    "MarketRegime",
]
