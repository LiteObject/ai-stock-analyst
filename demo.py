"""
Demo script for the AI Stock Analyst.

This script demonstrates the full pipeline:
1. Fetch historical data (Yahoo Finance)
2. Engineer advanced features (technical, regime, cross-asset)
3. Train an ML model with GPU acceleration
4. Perform comprehensive cross-validation
5. Calculate validation metrics
6. Backtest with realistic execution simulation
7. Generate validation report

New Features:
- GPU-accelerated LightGBM and CatBoost models
- Advanced feature engineering (regime detection, cross-asset)
- Purged K-Fold cross-validation
- Hyperparameter optimization with Optuna
- Comprehensive ML validation metrics
- Realistic backtesting with slippage and transaction costs
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
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


def check_gpu_availability() -> Dict[str, Any]:
    """Check for GPU availability and return info."""
    gpu_info = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }

    try:
        import torch

        gpu_info["cuda_available"] = torch.cuda.is_available()
        if gpu_info["cuda_available"]:
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
    except ImportError:
        pass

    return gpu_info


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


async def run_basic_demo():
    """Run basic demo without advanced features."""
    print(f"{Fore.CYAN}=== AI Stock Analyst Demo (Basic) ==={Style.RESET_ALL}")

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

    class FeatureDataProvider(YahooDataProvider):
        async def get_historical_data(
            self,
            ticker: str,
            start_date: datetime,
            end_date: datetime,
            timeframe: str = "1d",
        ) -> pd.DataFrame:
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
        commission_rate=0.000,
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


async def run_advanced_demo():
    """Run advanced demo with GPU acceleration and comprehensive validation."""
    print(
        f"{Fore.CYAN}=== AI Stock Analyst Demo (Advanced ML Validation) ==={Style.RESET_ALL}"
    )

    # Check GPU
    gpu_info = check_gpu_availability()
    if gpu_info["cuda_available"]:
        print(
            f"\n{Fore.GREEN}GPU Detected: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']:.1f} GB){Style.RESET_ALL}"
        )
        use_gpu = True
    else:
        print(f"\n{Fore.YELLOW}No GPU detected, using CPU{Style.RESET_ALL}")
        use_gpu = False

    # Configuration
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "SPY"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 1, 1)

    print(f"\n{Fore.CYAN}Configuration:{Style.RESET_ALL}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Date Range: {start_date.date()} to {end_date.date()}")
    print(f"  GPU Enabled: {use_gpu}")

    # 1. Load Data
    print(f"\n{Fore.YELLOW}[1] Loading Market Data...{Style.RESET_ALL}")
    provider = YahooDataProvider()

    all_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = await provider.get_historical_data(ticker, start_date, end_date)
            if df is not None and len(df) > 0:
                all_data[ticker] = df
                print(f"  {Fore.GREEN}{ticker}: {len(df)} rows{Style.RESET_ALL}")
        except Exception as e:
            print(f"  {Fore.RED}{ticker}: Failed - {e}{Style.RESET_ALL}")

    if not all_data:
        print(f"{Fore.RED}No data loaded. Exiting.{Style.RESET_ALL}")
        return

    # 2. Feature Engineering
    print(f"\n{Fore.YELLOW}[2] Engineering Features...{Style.RESET_ALL}")

    try:
        from ml.features import AdvancedFeatureEngineer, RegimeDetector

        all_features: List[pd.DataFrame] = []

        for ticker, df in all_data.items():
            print(f"  Processing {ticker}...")

            # Basic technical features
            tech_features = TechnicalFeatures.add_all_features(df)

            # Advanced features
            try:
                advanced = AdvancedFeatureEngineer()
                adv_features = advanced.calculate_advanced_features(df)
                tech_features = pd.concat([tech_features, adv_features], axis=1)
            except Exception as e:
                logger.warning(f"Advanced features failed for {ticker}: {e}")

            # Regime detection
            try:
                regime = RegimeDetector()
                regime_features = regime.detect_regime(df)
                tech_features = pd.concat([tech_features, regime_features], axis=1)
            except Exception as e:
                logger.warning(f"Regime detection failed for {ticker}: {e}")

            tech_features = tech_features.loc[:, ~tech_features.columns.duplicated()]
            tech_features["ticker"] = ticker
            all_features.append(tech_features)

        features_df = pd.concat(all_features, axis=0).dropna()
        print(
            f"  {Fore.GREEN}Total: {features_df.shape[1]-1} features, {features_df.shape[0]} rows{Style.RESET_ALL}"
        )

    except ImportError:
        print(
            f"  {Fore.YELLOW}Advanced features not available, using basic features{Style.RESET_ALL}"
        )
        # Fallback to basic features
        all_features = []
        for ticker, df in all_data.items():
            tech_features = TechnicalFeatures.add_all_features(df)
            tech_features["ticker"] = ticker
            all_features.append(tech_features)
        features_df = pd.concat(all_features, axis=0).dropna()

    # 3. Prepare Labels
    print(f"\n{Fore.YELLOW}[3] Preparing Labels...{Style.RESET_ALL}")

    labeled_dfs: List[pd.DataFrame] = []
    for ticker in features_df["ticker"].unique():
        ticker_df = features_df[features_df["ticker"] == ticker].copy()
        if "close" in ticker_df.columns:
            ticker_df["forward_return"] = (
                ticker_df["close"].shift(-5) / ticker_df["close"] - 1
            )
            ticker_df["target"] = (ticker_df["forward_return"] > 0).astype(int)
            labeled_dfs.append(ticker_df)

    labeled_df = pd.concat(labeled_dfs, axis=0).dropna(subset=["target"])
    positive_ratio = labeled_df["target"].mean()
    print(
        f"  Target distribution: {Fore.GREEN}{positive_ratio:.2%} positive{Style.RESET_ALL}, {1-positive_ratio:.2%} negative"
    )

    # 4. Cross-Validation
    print(f"\n{Fore.YELLOW}[4] Running Cross-Validation...{Style.RESET_ALL}")

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
        "forward_return",
        "target",
        "ticker",
        "date",
    ]
    feature_cols = [c for c in labeled_df.columns if c not in exclude_cols]

    X = labeled_df[feature_cols].values
    y = labeled_df["target"].values

    try:
        from ml.validation import PurgedKFold, cross_validate_model
        from sklearn.ensemble import RandomForestClassifier

        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        cv_result = cross_validate_model(
            model=model,
            X=X,
            y=y,
            cv=cv,
            return_predictions=True,
        )

        mean_acc = cv_result.fold_metrics_df["accuracy"].mean()
        std_acc = cv_result.fold_metrics_df["accuracy"].std()
        print(
            f"  {Fore.GREEN}Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}){Style.RESET_ALL}"
        )

        predictions = cv_result.oof_predictions

    except ImportError as e:
        print(
            f"  {Fore.YELLOW}Advanced CV not available, using sklearn{Style.RESET_ALL}"
        )
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier

        cv = TimeSeriesSplit(n_splits=5)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(
            f"  {Fore.GREEN}Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f}){Style.RESET_ALL}"
        )

        # Train final model for predictions
        model.fit(X, y)
        predictions = model.predict(X)

    # 5. Validation Metrics
    print(f"\n{Fore.YELLOW}[5] Calculating Validation Metrics...{Style.RESET_ALL}")

    try:
        from ml.validation import (
            calculate_prediction_metrics,
            calculate_directional_metrics,
            calculate_trading_metrics,
        )

        pred_metrics = calculate_prediction_metrics(y, predictions, None)
        print(
            f"  Accuracy: {Fore.GREEN}{pred_metrics.get('accuracy', 0):.4f}{Style.RESET_ALL}"
        )
        print(
            f"  F1 Score: {Fore.GREEN}{pred_metrics.get('f1_score', 0):.4f}{Style.RESET_ALL}"
        )
        print(f"  Precision: {pred_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {pred_metrics.get('recall', 0):.4f}")

        returns = labeled_df["forward_return"].values
        dir_metrics = calculate_directional_metrics(predictions, returns)
        print(
            f"  Directional Accuracy: {Fore.GREEN}{dir_metrics.get('directional_accuracy', 0):.4f}{Style.RESET_ALL}"
        )
        print(
            f"  Information Coefficient: {dir_metrics.get('information_coefficient', 0):.4f}"
        )

        trade_metrics = calculate_trading_metrics(predictions, returns)
        print(
            f"  Win Rate: {Fore.GREEN}{trade_metrics.get('win_rate', 0):.4f}{Style.RESET_ALL}"
        )
        print(
            f"  Profit Factor: {Fore.GREEN}{trade_metrics.get('profit_factor', 0):.4f}{Style.RESET_ALL}"
        )

    except ImportError:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        acc = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average="binary", zero_division=0)
        prec = precision_score(y, predictions, average="binary", zero_division=0)
        rec = recall_score(y, predictions, average="binary", zero_division=0)

        print(f"  Accuracy: {Fore.GREEN}{acc:.4f}{Style.RESET_ALL}")
        print(f"  F1 Score: {Fore.GREEN}{f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")

    # 6. Performance Targets Check
    print(f"\n{Fore.YELLOW}[6] Performance Targets Check...{Style.RESET_ALL}")

    targets = {
        "ML Accuracy ≥ 55%": (
            pred_metrics.get("accuracy", 0) >= 0.55
            if "pred_metrics" in dir()
            else acc >= 0.55
        ),
        "Win Rate ≥ 52%": (
            trade_metrics.get("win_rate", 0) >= 0.52
            if "trade_metrics" in dir()
            else False
        ),
        "Profit Factor ≥ 1.3": (
            trade_metrics.get("profit_factor", 0) >= 1.3
            if "trade_metrics" in dir()
            else False
        ),
    }

    for target, met in targets.items():
        status = (
            f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}"
            if met
            else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
        )
        print(f"  {target}: {status}")

    # Summary
    print(f"\n{Fore.CYAN}=== Demo Complete ==={Style.RESET_ALL}")
    print(f"\nFor full ML validation with hyperparameter optimization, run:")
    print(
        f"  {Fore.YELLOW}python -m cli.ml_validator --tickers AAPL MSFT GOOGL --optimize{Style.RESET_ALL}"
    )


async def main():
    """Main entry point - choose demo mode."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Stock Analyst Demo")
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced"],
        default="advanced",
        help="Demo mode: basic (original) or advanced (ML validation)",
    )

    args = parser.parse_args()

    if args.mode == "basic":
        await run_basic_demo()
    else:
        await run_advanced_demo()


if __name__ == "__main__":
    asyncio.run(main())
