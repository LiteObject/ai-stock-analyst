"""
Validation reporting for ML trading models.

Generates comprehensive reports on model validation including:
- Performance metrics summary
- Stability analysis
- Feature importance analysis
- Recommendations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.validation.cross_validation import CrossValidationResult
from src.ml.validation.validation_metrics import (
    calculate_directional_metrics,
    calculate_confidence_metrics,
    calculate_trading_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Complete validation report for a model."""

    # Identification
    model_name: str
    timestamp: datetime

    # Data info
    n_samples: int
    n_features: int
    date_range: Tuple[str, str]

    # Cross-validation results
    cv_results: Dict[str, float]
    cv_std: Dict[str, float]
    n_folds: int

    # Directional metrics
    directional_metrics: Dict[str, float]

    # Trading performance
    trading_metrics: Dict[str, float]

    # Stability analysis
    is_stable: bool
    stability_metrics: Dict[str, float]

    # Feature importance
    top_features: List[Tuple[str, float]]
    feature_stability: float

    # Recommendations
    recommendations: List[str]
    warnings: List[str]

    # Performance targets
    meets_targets: bool
    target_summary: Dict[str, bool]

    # Raw data for further analysis
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


class ValidationReporter:
    """
    Generates comprehensive validation reports.

    Analyzes cross-validation results and provides actionable insights.
    """

    def __init__(
        self,
        performance_targets: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize reporter.

        Args:
            performance_targets: Dictionary of metric_name -> minimum acceptable value
        """
        self.performance_targets = performance_targets or {
            "accuracy": 0.52,
            "directional_accuracy": 0.52,
            "sharpe_ratio": 1.0,
            "win_rate": 0.50,
            "profit_factor": 1.2,
            "max_drawdown": -0.20,  # Maximum allowed drawdown
        }

    def generate_report(
        self,
        cv_result: CrossValidationResult,
        model_name: str,
        returns: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
    ) -> ValidationReport:
        """
        Generate comprehensive validation report.

        Args:
            cv_result: Cross-validation results
            model_name: Name of the model
            returns: Actual returns (for trading metrics)
            predictions: Model predictions (for trading metrics)
            feature_names: Names of features
            date_range: Start and end dates

        Returns:
            ValidationReport with all analysis
        """
        timestamp = datetime.now()

        # Extract basic stats
        n_folds = len(cv_result.fold_results)

        # Get feature info
        n_features = 0
        if cv_result.feature_importance is not None:
            n_features = len(cv_result.feature_importance)

        n_samples = (
            len(cv_result.oof_predictions)
            if cv_result.oof_predictions is not None
            else 0
        )

        # Directional metrics
        directional_metrics = {}
        if returns is not None and predictions is not None:
            directional_metrics = calculate_directional_metrics(returns, predictions)

        # Trading metrics
        trading_metrics = {}
        if returns is not None and predictions is not None:
            trading_metrics = calculate_trading_metrics(predictions, returns)

        # Stability analysis
        stability_metrics = {
            "metric_stability": (
                np.mean(list(cv_result.metric_stability.values()))
                if cv_result.metric_stability
                else 1.0
            ),
            "is_stable": cv_result.is_stable,
        }

        # Feature importance
        top_features = []
        feature_stability = 0.0

        if cv_result.feature_importance is not None:
            fi_df = cv_result.feature_importance
            top_n = min(20, len(fi_df))

            for _, row in fi_df.head(top_n).iterrows():
                if feature_names and row["feature"] in range(len(feature_names)):
                    name = feature_names[int(row["feature"])]
                else:
                    name = str(row["feature"])
                top_features.append((name, float(row["mean"])))

            # Feature stability (CV of importances)
            if "std" in fi_df.columns:
                with_std = fi_df[fi_df["std"] > 0]
                if len(with_std) > 0:
                    cv_values = with_std["std"] / with_std["mean"].abs()
                    feature_stability = 1.0 - cv_values.mean()  # Higher is more stable

        # Check performance targets
        target_summary = self._check_targets(
            cv_result.mean_metrics,
            directional_metrics,
            trading_metrics,
        )
        meets_targets = all(target_summary.values())

        # Generate recommendations
        recommendations, warnings = self._generate_recommendations(
            cv_result,
            directional_metrics,
            trading_metrics,
            stability_metrics,
            target_summary,
        )

        return ValidationReport(
            model_name=model_name,
            timestamp=timestamp,
            n_samples=n_samples,
            n_features=n_features,
            date_range=date_range or ("N/A", "N/A"),
            cv_results=cv_result.mean_metrics,
            cv_std=cv_result.std_metrics,
            n_folds=n_folds,
            directional_metrics=directional_metrics,
            trading_metrics=trading_metrics,
            is_stable=cv_result.is_stable,
            stability_metrics=stability_metrics,
            top_features=top_features,
            feature_stability=feature_stability,
            recommendations=recommendations,
            warnings=warnings,
            meets_targets=meets_targets,
            target_summary=target_summary,
            raw_metrics={
                "cv_result": cv_result,
            },
        )

    def _check_targets(
        self,
        cv_metrics: Dict[str, float],
        directional_metrics: Dict[str, float],
        trading_metrics: Dict[str, float],
    ) -> Dict[str, bool]:
        """Check which performance targets are met."""
        all_metrics = {**cv_metrics, **directional_metrics, **trading_metrics}
        target_summary = {}

        for metric, target in self.performance_targets.items():
            if metric in all_metrics:
                value = all_metrics[metric]
                if metric == "max_drawdown":
                    # Drawdown is negative, we want it > target (less negative)
                    target_summary[metric] = value >= target
                else:
                    target_summary[metric] = value >= target
            else:
                target_summary[metric] = False

        return target_summary

    def _generate_recommendations(
        self,
        cv_result: CrossValidationResult,
        directional_metrics: Dict[str, float],
        trading_metrics: Dict[str, float],
        stability_metrics: Dict[str, float],
        target_summary: Dict[str, bool],
    ) -> Tuple[List[str], List[str]]:
        """Generate actionable recommendations and warnings."""
        recommendations = []
        warnings = []

        # Check overfitting risk
        if not cv_result.is_stable:
            warnings.append(
                "Model shows unstable performance across folds. "
                "High variance suggests potential overfitting."
            )
            recommendations.append(
                "Consider reducing model complexity or adding regularization."
            )

        # Check directional accuracy
        dir_acc = directional_metrics.get("directional_accuracy", 0.5)
        if dir_acc < 0.52:
            warnings.append(
                f"Directional accuracy ({dir_acc:.1%}) is near random. "
                "The model may not have predictive power."
            )
            recommendations.append(
                "Review feature engineering and consider market regime features."
            )
        elif dir_acc > 0.65:
            warnings.append(
                f"Directional accuracy ({dir_acc:.1%}) is unusually high. "
                "Verify there's no look-ahead bias or data leakage."
            )

        # Check information coefficient
        ic = directional_metrics.get("information_coefficient", 0)
        if abs(ic) < 0.02:
            warnings.append(
                "Information coefficient is very low. "
                "Predictions have weak correlation with returns."
            )
            recommendations.append(
                "Experiment with different target definitions or longer horizons."
            )

        # Check profit factor
        pf = trading_metrics.get("profit_factor", 1.0)
        if pf < 1.0:
            warnings.append(
                f"Profit factor ({pf:.2f}) is below 1.0. "
                "Strategy loses money on average."
            )
        elif pf < 1.2:
            recommendations.append(
                "Profit factor is marginal. Consider optimizing entry/exit thresholds."
            )

        # Check max drawdown
        mdd = trading_metrics.get("max_drawdown", 0)
        if mdd < -0.25:
            warnings.append(
                f"Maximum drawdown ({mdd:.1%}) exceeds 25%. "
                "Risk management improvements needed."
            )
            recommendations.append("Implement position sizing or stop-loss rules.")

        # Check Sharpe ratio
        sharpe = trading_metrics.get("sharpe_ratio", 0)
        if sharpe < 0:
            warnings.append(
                "Sharpe ratio is negative. Strategy underperforms risk-free rate."
            )
        elif sharpe < 0.5:
            warnings.append(
                f"Sharpe ratio ({sharpe:.2f}) is low for a trading strategy."
            )
            recommendations.append(
                "Optimize signal generation or consider different market conditions."
            )

        # Feature stability
        feat_stability = stability_metrics.get("metric_stability", 1.0)
        if feat_stability > 0.3:
            recommendations.append(
                "Feature importance varies significantly across folds. "
                "Consider more stable feature selection."
            )

        # Check for failed targets
        failed_targets = [k for k, v in target_summary.items() if not v]
        if failed_targets:
            warnings.append(f"Performance targets not met: {', '.join(failed_targets)}")

        # Positive recommendations
        if cv_result.is_stable and dir_acc >= 0.55:
            recommendations.append(
                "Model shows stable, above-random performance. "
                "Consider paper trading before live deployment."
            )

        return recommendations, warnings

    def format_report(self, report: ValidationReport) -> str:
        """Format report as readable text."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"VALIDATION REPORT: {report.model_name}")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        lines.append("\n### DATA INFO")
        lines.append(f"Samples: {report.n_samples:,}")
        lines.append(f"Features: {report.n_features:,}")
        lines.append(f"Date Range: {report.date_range[0]} to {report.date_range[1]}")
        lines.append(f"CV Folds: {report.n_folds}")

        lines.append("\n### CROSS-VALIDATION RESULTS")
        for metric, value in report.cv_results.items():
            std = report.cv_std.get(metric, 0)
            if isinstance(value, float):
                lines.append(f"  {metric}: {value:.4f} (+/- {std:.4f})")

        if report.directional_metrics:
            lines.append("\n### DIRECTIONAL METRICS")
            for metric, value in report.directional_metrics.items():
                if isinstance(value, float):
                    if "accuracy" in metric or "ratio" in metric:
                        lines.append(f"  {metric}: {value:.2%}")
                    else:
                        lines.append(f"  {metric}: {value:.4f}")
                else:
                    lines.append(f"  {metric}: {value}")

        if report.trading_metrics:
            lines.append("\n### TRADING METRICS")
            for metric, value in report.trading_metrics.items():
                if isinstance(value, float):
                    if "rate" in metric or "drawdown" in metric:
                        lines.append(f"  {metric}: {value:.2%}")
                    else:
                        lines.append(f"  {metric}: {value:.4f}")

        lines.append("\n### STABILITY ANALYSIS")
        lines.append(f"  Model Stable: {'Yes' if report.is_stable else 'No'}")
        lines.append(f"  Feature Stability: {report.feature_stability:.2%}")

        if report.top_features:
            lines.append("\n### TOP FEATURES")
            for i, (name, importance) in enumerate(report.top_features[:10], 1):
                lines.append(f"  {i}. {name}: {importance:.4f}")

        lines.append("\n### PERFORMANCE TARGETS")
        status = "PASSED" if report.meets_targets else "FAILED"
        lines.append(f"  Overall: {status}")
        for target, met in report.target_summary.items():
            icon = "+" if met else "-"
            lines.append(f"  [{icon}] {target}")

        if report.warnings:
            lines.append("\n### WARNINGS")
            for warning in report.warnings:
                lines.append(f"  ! {warning}")

        if report.recommendations:
            lines.append("\n### RECOMMENDATIONS")
            for rec in report.recommendations:
                lines.append(f"  > {rec}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def to_dataframe(self, report: ValidationReport) -> pd.DataFrame:
        """Convert report to DataFrame for easy export."""
        data = {
            "Model": report.model_name,
            "Timestamp": report.timestamp,
            "Samples": report.n_samples,
            "Features": report.n_features,
            "Folds": report.n_folds,
            "Stable": report.is_stable,
            "Meets_Targets": report.meets_targets,
        }

        # Add CV metrics
        for metric, value in report.cv_results.items():
            data[f"CV_{metric}"] = value

        # Add directional metrics
        for metric, value in report.directional_metrics.items():
            if isinstance(value, (int, float)):
                data[f"Dir_{metric}"] = value

        # Add trading metrics
        for metric, value in report.trading_metrics.items():
            if isinstance(value, (int, float)):
                data[f"Trade_{metric}"] = value

        return pd.DataFrame([data])


def generate_validation_summary(
    reports: List[ValidationReport],
) -> pd.DataFrame:
    """
    Generate summary comparison of multiple validation reports.

    Args:
        reports: List of validation reports to compare

    Returns:
        DataFrame summarizing all reports
    """
    reporter = ValidationReporter()
    dfs = [reporter.to_dataframe(r) for r in reports]
    return pd.concat(dfs, ignore_index=True)
