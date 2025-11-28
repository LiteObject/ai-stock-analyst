"""
Demo script for the AI Stock Analyst.

This script demonstrates the full pipeline:
1. Fetch historical data (Yahoo Finance)
2. Train an ML model (Random Forest)
3. Backtest a strategy using the trained model
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from colorama import Fore, Style, init

from backtesting.engine import BacktestEngine
from backtesting.strategy import Strategy
from core.models import BacktestConfig, Signal, SignalType
from data.providers.yahoo import YahooDataProvider
from ml.training.trainer import ModelTrainer
from ml.models.ensemble import EnsemblePredictor
from ml.features.technical import TechnicalFeatures

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

init(autoreset=True)


class MLStrategy(Strategy):
    """
    Strategy that uses a trained ML model to generate signals.
    """

    def __init__(self, name: str, model: EnsemblePredictor):
        super().__init__(name)
        self.model = model
        self.min_confidence = 0.30  # Lower confidence for demo to get trades

    async def on_bar(self, bar: pd.Series, portfolio: dict) -> Signal | None:
        # In a real scenario, we would need to reconstruct the features for this specific bar
        # using historical context.
        # For this demo, we will assume the 'bar' already contains the features
        # (which we will ensure in the backtest setup).

        # Check if we have the necessary features
        try:
            # Create a DataFrame for prediction (single row)
            features_df = pd.DataFrame([bar])

            # Predict
            probs = self.model.predict_proba(features_df)

            # Assuming binary classification: 0 = Down, 1 = Up
            # Classes might be [0, 1] or [False, True]
            up_prob = 0.0
            if 1 in probs.columns:
                up_prob = probs.iloc[0][1]
            elif True in probs.columns:
                up_prob = probs.iloc[0][True]
            elif "1" in probs.columns:
                up_prob = probs.iloc[0]["1"]
            elif "1.0" in probs.columns:
                up_prob = probs.iloc[0]["1.0"]

            down_prob = 1.0 - up_prob

            signal_type = None
            confidence = 0.0

            if up_prob > self.min_confidence:
                signal_type = SignalType.BUY
                confidence = up_prob
            elif down_prob > self.min_confidence:
                signal_type = SignalType.SELL
                confidence = down_prob

            if signal_type:
                return Signal(
                    ticker="SPY",
                    signal_type=signal_type,
                    confidence=confidence,
                    source="ML_RF",
                    timestamp=datetime.now(),  # In backtest this should be current_date
                    reasoning=f"Model probability: {confidence:.2f}",
                )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            pass

        return None

    async def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


async def main():
    print(f"{Fore.CYAN}=== AI Stock Analyst Demo ==={Style.RESET_ALL}")

    ticker = "SPY"
    start_date = datetime.now() - timedelta(days=365 * 2)  # 2 years of data
    end_date = datetime.now()

    # 1. Data Provider
    print(f"\n{Fore.YELLOW}[1] Initializing Data Provider...{Style.RESET_ALL}")
    provider = YahooDataProvider()

    # 2. Train Model
    print(f"\n{Fore.YELLOW}[2] Training ML Model...{Style.RESET_ALL}")
    trainer = ModelTrainer(provider)

    # Split data: Train on first 1.5 years, Test on last 0.5 years
    train_end_date = start_date + timedelta(days=365 * 1.5)

    print(f"Training on data from {start_date.date()} to {train_end_date.date()}")
    train_result = await trainer.train_model(
        ticker=ticker,
        start_date=start_date,
        end_date=train_end_date,
        model_type="random_forest",
    )

    metrics = train_result["metrics"]
    print(
        f"{Fore.GREEN}Model Trained! Accuracy: {metrics.get('accuracy', 0):.2%}{Style.RESET_ALL}"
    )

    # We need to load the trained model to use it
    # Since ModelTrainer saves it or we can reconstruct it.
    # For this demo, let's just use the one created inside trainer if we could access it,
    # but trainer returns metrics.
    # Let's manually train to get the object instance for the strategy.

    # Fetch data again for manual training/backtesting setup
    df = await provider.get_historical_data(ticker, start_date, end_date)
    df_features = TechnicalFeatures.add_all_features(df).dropna()

    # Split for training
    train_mask = df_features.index <= train_end_date
    train_data = df_features[train_mask]
    test_data = df_features[~train_mask]

    print(f"Test data points: {len(test_data)}")

    # Train Predictor
    predictor = EnsemblePredictor(model_type="random_forest", task="classification")

    # Prepare X and y
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

    X_train = train_data[feature_cols]
    y_train = train_data["target_direction"]

    predictor.train(X_train, y_train)

    # 3. Backtest
    print(f"\n{Fore.YELLOW}[3] Running Backtest...{Style.RESET_ALL}")

    # We need to feed the features into the backtester so the strategy can see them.
    # The current BacktestEngine fetches raw data.
    # We will subclass BacktestEngine or Mock the provider to return the feature-rich dataframe.

    class FeatureDataProvider(YahooDataProvider):
        async def get_historical_data(
            self,
            ticker: str,
            start_date: datetime,
            end_date: datetime,
            timeframe: str = "1d",
        ) -> pd.DataFrame:
            # Return the pre-calculated test data which has features
            # Filter by date
            mask = (test_data.index >= start_date) & (test_data.index <= end_date)
            return test_data[mask]

    feature_provider = FeatureDataProvider()

    engine = BacktestEngine(feature_provider)
    strategy = MLStrategy("RandomForest_Strategy", predictor)

    config = BacktestConfig(
        tickers=[ticker],
        start_date=train_end_date + timedelta(days=1),
        end_date=end_date,
        initial_capital=100000.0,
        commission_rate=0.000,  # Zero commission for simplicity
        slippage_rate=0.0005,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )

    result = await engine.run(strategy, config)

    # 4. Results
    print(f"\n{Fore.CYAN}=== Backtest Results ==={Style.RESET_ALL}")
    print(
        f"Total Return: {Fore.GREEN if result.metrics.total_return > 0 else Fore.RED}{result.metrics.total_return:.2%}{Style.RESET_ALL}"
    )
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {Fore.RED}{result.metrics.max_drawdown:.2%}{Style.RESET_ALL}")
    print(f"Total Trades: {result.metrics.total_trades}")
    print(f"Win Rate: {result.metrics.win_rate:.2%}")

    print(
        f"\n{Fore.YELLOW}Final Portfolio Value: ${result.final_portfolio.total_value:,.2f}{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    asyncio.run(main())
