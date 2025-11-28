"""
Model training pipeline.
"""

import logging
from typing import Any, Dict, Optional

from core.interfaces import DataProvider
from ml.features.technical import TechnicalFeatures
from ml.models.ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Pipeline for training ML models.
    """

    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    async def train_model(
        self,
        ticker: str,
        start_date: Any,
        end_date: Any,
        model_type: str = "random_forest",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch data, generate features, train model, and save it.
        """
        logger.info(f"Starting training for {ticker}")

        # 1. Fetch Data
        df = await self.data_provider.get_historical_data(ticker, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")

        # 2. Feature Engineering
        df_features = TechnicalFeatures.add_all_features(df)

        # Drop NaN values created by indicators and targets
        df_features = df_features.dropna()

        if df_features.empty:
            raise ValueError("Not enough data after feature engineering")

        # Define Features and Target
        # Exclude non-feature columns
        exclude_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "returns",
            "log_returns",
            "target_return",
            "target_direction",
        ]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        X = df_features[feature_cols]
        y = df_features["target_direction"]  # Classification target

        # 3. Train Model
        predictor = EnsemblePredictor(model_type=model_type, task="classification")
        metrics = predictor.train(X, y)

        # 4. Save Model
        if save_path:
            predictor.save(save_path)
            logger.info(f"Model saved to {save_path}")

        return {
            "metrics": metrics,
            "feature_importance": predictor.get_feature_importance(),
            "data_points": len(X),
        }
