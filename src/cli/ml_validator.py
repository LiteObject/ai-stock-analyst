"""
ML Validation CLI for AI Stock Analyst.

Command-line interface for running comprehensive ML validation workflows
with GPU-accelerated models, advanced cross-validation, and detailed reporting.

Usage:
    python -m cli.ml_validator --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2025-01-01
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_gpu() -> Dict[str, Any]:
    """Setup and verify GPU availability."""
    gpu_info: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory": None,
        "lightgbm_gpu": False,
        "catboost_gpu": False,
    }

    try:
        import torch  # type: ignore[import-unresolved]

        gpu_info["cuda_available"] = torch.cuda.is_available()
        if gpu_info["cuda_available"]:
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_memory"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
            logger.info(
                f"GPU detected: {gpu_info['gpu_name']} ({gpu_info['gpu_memory']:.1f} GB)"
            )
    except ImportError:
        logger.warning("PyTorch not available for GPU detection")

    # Check LightGBM GPU support
    try:
        import lightgbm as lgb  # noqa: F401

        gpu_info["lightgbm_gpu"] = True
        logger.info("LightGBM GPU support available")
    except ImportError:
        pass

    # Check CatBoost GPU support
    try:
        import catboost  # noqa: F401

        gpu_info["catboost_gpu"] = True
        logger.info("CatBoost GPU support available")
    except ImportError:
        pass

    return gpu_info


async def load_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """Load market data for tickers."""
    from datetime import datetime as dt

    from data.providers import DataProviderFactory

    logger.info(
        f"Loading data for {len(tickers)} tickers from {start_date} to {end_date}"
    )

    provider = DataProviderFactory.get_default()
    data: Dict[str, pd.DataFrame] = {}

    # Convert string dates to datetime
    start_dt = dt.strptime(start_date, "%Y-%m-%d")
    end_dt = dt.strptime(end_date, "%Y-%m-%d")

    for ticker in tickers:
        try:
            df = await provider.get_historical_data(ticker, start_dt, end_dt)
            if df is not None and len(df) > 0:
                data[ticker] = df
                logger.info(f"  {ticker}: {len(df)} rows loaded")
            else:
                logger.warning(f"  {ticker}: No data available")
        except Exception as e:
            logger.error(f"  {ticker}: Failed to load - {e}")

    return data


def engineer_features(
    data: Dict[str, pd.DataFrame],
    use_cross_asset: bool = True,
) -> pd.DataFrame:
    """Engineer features for all tickers."""
    from ml.features import (
        AdvancedFeatureEngineer,
        CrossAssetFeatures,
        RegimeDetector,
        TechnicalFeatures,
    )

    logger.info("Engineering features...")

    all_features: List[pd.DataFrame] = []

    for ticker, df in data.items():
        logger.info(f"  Processing {ticker}...")

        # Technical features - use add_all_features method
        tech = TechnicalFeatures()
        tech_df = tech.add_all_features(df)

        # Advanced features - use create_all_features method
        advanced = AdvancedFeatureEngineer()
        adv_features = advanced.create_all_features(df)

        # Regime detection - use detect_all_regimes method
        regime = RegimeDetector()
        regime_features = regime.detect_all_regimes(df)

        # Combine
        combined = pd.concat([tech_df, adv_features, regime_features], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined["ticker"] = ticker

        all_features.append(combined)

    # Combine all tickers
    features_df = pd.concat(all_features, axis=0)

    # Cross-asset features
    if use_cross_asset and len(data) > 1:
        logger.info("  Adding cross-asset features...")
        cross_asset = CrossAssetFeatures()
        # Cross-asset features need price_data dict and target ticker
        # We'll compute for the first ticker as a demo
        tickers = list(data.keys())
        if len(tickers) > 1:
            cross_features = cross_asset.create_all_features(data, tickers[0])
            logger.info(f"  Cross-asset features calculated: {cross_features.shape}")

    # Drop NaN rows
    features_df = features_df.dropna()
    logger.info(
        f"Total features: {features_df.shape[1]-1} columns, {features_df.shape[0]} rows"
    )

    return features_df


def prepare_labels(
    features_df: pd.DataFrame,
    prediction_horizon: int = 5,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Prepare labels for supervised learning."""
    logger.info(
        f"Preparing labels (horizon={prediction_horizon} days, threshold={threshold})..."
    )

    labeled_dfs: List[pd.DataFrame] = []

    for ticker in features_df["ticker"].unique():
        ticker_df = features_df[features_df["ticker"] == ticker].copy()

        if "close" in ticker_df.columns:
            # Calculate forward returns
            ticker_df["forward_return"] = (
                ticker_df["close"].shift(-prediction_horizon) / ticker_df["close"] - 1
            )
            # Binary classification: 1 if return > threshold
            ticker_df["target"] = (ticker_df["forward_return"] > threshold).astype(int)
            labeled_dfs.append(ticker_df)

    result = pd.concat(labeled_dfs, axis=0)
    result = result.dropna(subset=["target"])

    positive_ratio = result["target"].mean()
    logger.info(
        f"  Target distribution: {positive_ratio:.2%} positive, {1-positive_ratio:.2%} negative"
    )

    return result


def run_cross_validation(
    features_df: pd.DataFrame,
    cv_strategy: str = "purged_kfold",
    n_splits: int = 5,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Run cross-validation with specified strategy."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score

    from ml.validation import PurgedKFold, TimeSeriesCV

    logger.info(f"Running cross-validation with {cv_strategy} ({n_splits} splits)...")

    # Prepare X and y
    feature_cols = [
        c
        for c in features_df.columns
        if c
        not in [
            "ticker",
            "target",
            "forward_return",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]
    X = features_df[feature_cols].values
    y = features_df["target"].values

    # Setup CV strategy
    if cv_strategy == "purged_kfold":
        cv = PurgedKFold(n_splits=n_splits, purge_periods=5, embargo_periods=5)
    elif cv_strategy == "time_series":
        cv = TimeSeriesCV(n_splits=n_splits, gap=5)
    else:
        cv = PurgedKFold(n_splits=n_splits)

    # Run cross-validation manually
    fold_metrics: List[Dict[str, Any]] = []
    oof_predictions = np.zeros(len(y))

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = np.asarray(y[train_idx]), np.asarray(y[test_idx])

        # Simple model for demonstration
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        oof_predictions[test_idx] = preds

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        fold_metrics.append({"fold": fold_idx, "accuracy": acc, "f1_score": f1})
        logger.info(f"  Fold {fold_idx + 1}: Accuracy={acc:.4f}, F1={f1:.4f}")

    mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
    std_acc = np.std([m["accuracy"] for m in fold_metrics])
    logger.info(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    return {
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_predictions,
        "cv_strategy": cv_strategy,
        "n_splits": n_splits,
    }


def run_hyperparameter_optimization(
    features_df: pd.DataFrame,
    n_trials: int = 50,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Run hyperparameter optimization with Optuna."""
    from ml.tuning import HyperparameterOptimizer

    logger.info(f"Running hyperparameter optimization ({n_trials} trials)...")

    # Prepare X and y
    feature_cols = [
        c
        for c in features_df.columns
        if c
        not in [
            "ticker",
            "target",
            "forward_return",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]
    X = features_df[feature_cols]
    y = features_df["target"]

    # Setup optimizer - note: no use_gpu parameter, it auto-detects
    optimizer = HyperparameterOptimizer(
        n_trials=n_trials,
        n_cv_splits=5,
        random_state=42,
    )

    # Run optimization for XGBoost
    result = optimizer.optimize_xgboost(
        X=X,
        y=y,
        metric="logloss",
        direction="minimize",
    )

    logger.info(f"  Best score: {result.best_score:.4f}")
    logger.info(f"  Best parameters: {result.best_params}")

    return {
        "best_params": result.best_params,
        "best_score": result.best_score,
        "n_trials": n_trials,
    }


def run_threshold_optimization(
    features_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    """Optimize classification thresholds."""
    from ml.tuning import ThresholdOptimizer

    logger.info("Optimizing classification thresholds...")

    optimizer = ThresholdOptimizer()

    # Get actual values as numpy arrays
    y_true = np.asarray(features_df["target"].values)

    # Get returns for profit-based optimization
    returns = (
        np.asarray(features_df["forward_return"].values)
        if "forward_return" in features_df.columns
        else None
    )

    # Optimize thresholds - use find_optimal_threshold method
    result = optimizer.find_optimal_threshold(
        y_true=y_true,
        y_prob=probabilities,
        returns=returns,
        objective="f1",
    )

    logger.info(f"  Optimal threshold: {result.optimal_threshold:.4f}")
    logger.info(f"  Best metrics: {result.metrics}")

    return {
        "threshold": result.optimal_threshold,
        "metrics": result.metrics,
    }


def calculate_validation_metrics(
    features_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive validation metrics."""
    from ml.validation import (
        calculate_confidence_metrics,
        calculate_directional_metrics,
        calculate_prediction_metrics,
    )

    logger.info("Calculating validation metrics...")

    y_true = np.asarray(features_df["target"].values)
    predictions_arr = np.asarray(predictions)
    returns = (
        np.asarray(features_df["forward_return"].values)
        if "forward_return" in features_df.columns
        else None
    )

    # Prediction metrics
    pred_metrics = calculate_prediction_metrics(y_true, predictions_arr, probabilities)
    logger.info(f"  Accuracy: {pred_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  F1 Score: {pred_metrics.get('f1_score', 0):.4f}")

    # Directional metrics
    dir_metrics: Dict[str, Any] = {}
    if returns is not None:
        dir_metrics = calculate_directional_metrics(predictions_arr, returns)
        logger.info(
            f"  Directional Accuracy: {dir_metrics.get('directional_accuracy', 0):.4f}"
        )

    # Confidence metrics
    conf_metrics: Dict[str, Any] = {}
    if probabilities is not None:
        conf_metrics = calculate_confidence_metrics(
            y_true=y_true,
            y_prob=np.asarray(probabilities),
            n_bins=10,
        )
        logger.info(f"  Brier Score: {conf_metrics.get('brier_score', 0):.4f}")

    return {
        "prediction": pred_metrics,
        "directional": dir_metrics,
        "confidence": conf_metrics,
    }


def generate_report(
    validation_results: Dict[str, Any],
    model_name: str = "GradientBoostingClassifier",
    output_path: Optional[str] = None,
) -> str:
    """Generate validation report."""
    logger.info("Generating validation report...")

    # Generate simple markdown report
    lines = [
        "# ML Validation Report",
        "",
        f"**Model**: {model_name}",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
    ]

    if "config" in validation_results:
        config = validation_results["config"]
        lines.extend(
            [
                f"- Tickers: {config.get('tickers', 'N/A')}",
                f"- Date Range: {config.get('date_range', 'N/A')}",
                f"- CV Strategy: {config.get('cv_strategy', 'N/A')}",
                f"- N Splits: {config.get('n_splits', 'N/A')}",
                "",
            ]
        )

    lines.append("## Cross-Validation Results")
    lines.append("")

    if "cv_results" in validation_results:
        cv = validation_results["cv_results"]
        fold_metrics = cv.get("fold_metrics", [])
        if fold_metrics:
            mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
            std_acc = np.std([m["accuracy"] for m in fold_metrics])
            mean_f1 = np.mean([m["f1_score"] for m in fold_metrics])
            std_f1 = np.std([m["f1_score"] for m in fold_metrics])
            lines.extend(
                [
                    f"- Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}",
                    f"- Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}",
                    "",
                ]
            )

    lines.append("## Validation Metrics")
    lines.append("")

    if "validation_metrics" in validation_results:
        metrics = validation_results["validation_metrics"]
        if "prediction" in metrics:
            pred = metrics["prediction"]
            lines.extend(
                [
                    f"- Accuracy: {pred.get('accuracy', 0):.4f}",
                    f"- F1 Score: {pred.get('f1_score', 0):.4f}",
                    f"- Precision: {pred.get('precision', 0):.4f}",
                    f"- Recall: {pred.get('recall', 0):.4f}",
                    "",
                ]
            )

    if "gpu_info" in validation_results:
        gpu = validation_results["gpu_info"]
        lines.extend(
            [
                "## GPU Information",
                "",
                f"- CUDA Available: {gpu.get('cuda_available', False)}",
                f"- GPU Name: {gpu.get('gpu_name', 'N/A')}",
                f"- GPU Memory: {gpu.get('gpu_memory', 'N/A')} GB",
                "",
            ]
        )

    summary = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(summary)
        logger.info(f"  Report saved to: {output_path}")

    return summary


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Validation CLI for AI Stock Analyst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation run
    python -m cli.ml_validator --tickers AAPL MSFT GOOGL

    # Full validation with GPU
    python -m cli.ml_validator --tickers AAPL MSFT GOOGL NVDA TSLA SPY \\
        --start 2020-01-01 --end 2025-01-01 --use-gpu

    # Hyperparameter optimization
    python -m cli.ml_validator --tickers AAPL --optimize --n-trials 100
        """,
    )

    # Data arguments
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "SPY"],
        help="List of ticker symbols (default: AAPL MSFT GOOGL NVDA TSLA SPY)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-01-01",
        help="End date (default: 2025-01-01)",
    )

    # Model arguments
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration if available (default: True)",
    )
    parser.add_argument(
        "--cv-strategy",
        type=str,
        choices=["purged_kfold", "time_series"],
        default="purged_kfold",
        help="Cross-validation strategy (default: purged_kfold)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for validation report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Header
    print("\n" + "=" * 60)
    print("AI Stock Analyst - ML Validation Pipeline")
    print("=" * 60)

    # Setup GPU
    gpu_info = setup_gpu()
    use_gpu = args.use_gpu and gpu_info["cuda_available"]

    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(args.tickers)}")
    print(f"  Date Range: {args.start} to {args.end}")
    print(f"  GPU Enabled: {use_gpu}")
    print(f"  CV Strategy: {args.cv_strategy}")
    print(f"  N Splits: {args.n_splits}")
    print()

    async def run_pipeline() -> None:
        try:
            # Step 1: Load data
            data = await load_data(args.tickers, args.start, args.end)
            if not data:
                logger.error("No data loaded. Exiting.")
                sys.exit(1)

            # Step 2: Engineer features
            features_df = engineer_features(data)

            # Step 3: Prepare labels
            labeled_df = prepare_labels(features_df)

            # Step 4: Run cross-validation
            cv_results = run_cross_validation(
                labeled_df,
                cv_strategy=args.cv_strategy,
                n_splits=args.n_splits,
                use_gpu=use_gpu,
            )

            # Step 5: Hyperparameter optimization (optional)
            hp_results: Optional[Dict[str, Any]] = None
            if args.optimize:
                hp_results = run_hyperparameter_optimization(
                    labeled_df,
                    n_trials=args.n_trials,
                    use_gpu=use_gpu,
                )

            # Step 6: Get predictions for metrics
            predictions = cv_results["oof_predictions"]

            # Step 7: Calculate validation metrics
            validation_metrics: Dict[str, Any] = {}
            if predictions is not None:
                validation_metrics = calculate_validation_metrics(
                    labeled_df,
                    predictions,
                    probabilities=None,
                )

            # Step 8: Generate report
            report_data = {
                "cv_results": cv_results,
                "hp_results": hp_results,
                "validation_metrics": validation_metrics,
                "gpu_info": gpu_info,
                "config": {
                    "tickers": args.tickers,
                    "date_range": f"{args.start} to {args.end}",
                    "cv_strategy": args.cv_strategy,
                    "n_splits": args.n_splits,
                },
            }

            output_path = (
                args.output
                or f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            report = generate_report(report_data, output_path=output_path)

            # Summary
            print("\n" + "=" * 60)
            print("Validation Complete!")
            print("=" * 60)
            print(f"\nReport saved to: {output_path}")

            # Print key metrics
            if validation_metrics:
                print("\nKey Metrics:")
                if "prediction" in validation_metrics:
                    print(
                        f"  Accuracy: {validation_metrics['prediction'].get('accuracy', 0):.4f}"
                    )
                    print(
                        f"  F1 Score: {validation_metrics['prediction'].get('f1_score', 0):.4f}"
                    )

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Validation failed: {e}")
            sys.exit(1)

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
