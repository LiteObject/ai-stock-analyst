"""
Performance metrics calculation for backtesting.

Comprehensive metrics including:
- Return metrics (total, annualized, risk-adjusted)
- Risk metrics (VaR, CVaR, drawdown analysis)
- Trade metrics (win rate, profit factor, expectancy)
- Benchmark comparison (alpha, beta, information ratio)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DrawdownAnalysis:
    """Detailed drawdown analysis."""

    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    avg_drawdown_duration_days: float
    current_drawdown: float
    recovery_time_days: Optional[int]
    drawdown_periods: List[Dict[str, Any]]


@dataclass
class TradeAnalysis:
    """Detailed trade analysis."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    avg_trade_duration_days: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    payoff_ratio: float  # avg_win / avg_loss


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics."""

    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    tail_ratio: float
    downside_deviation: float
    upside_potential_ratio: float


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics."""

    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float
    capture_ratio: float


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from a series of returns.

    Args:
        returns: Series of percentage returns (e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods in a year (252 for daily)

    Returns:
        Dictionary of metrics
    """
    if returns.empty:
        return {}

    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    metrics = {}

    # Basic return metrics
    cumulative_prod = (1 + returns).prod()
    total_return: float = float(cumulative_prod) - 1
    metrics["total_return"] = total_return

    # CAGR
    n_years = len(returns) / periods_per_year
    if n_years > 0 and total_return > -1:
        cagr: float = (1 + total_return) ** (1 / n_years) - 1
    else:
        cagr = 0.0
    metrics["cagr"] = cagr

    # Volatility (Annualized)
    volatility = float(returns.std() * np.sqrt(periods_per_year))
    metrics["volatility"] = volatility

    # Sharpe Ratio
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    if returns.std() > 0:
        sharpe_ratio = float(
            (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        )
    else:
        sharpe_ratio = 0.0
    metrics["sharpe_ratio"] = sharpe_ratio

    # Sortino Ratio
    downside_returns = returns[returns < daily_rf]
    if len(downside_returns) > 0:
        downside_std = float(
            np.sqrt(((returns.clip(upper=daily_rf) - daily_rf) ** 2).mean())
        )
        if downside_std > 0:
            sortino_ratio = float(
                (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)
            )
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0
    metrics["sortino_ratio"] = sortino_ratio

    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = float(drawdown.min())
    metrics["max_drawdown"] = max_drawdown

    # Average drawdown
    metrics["avg_drawdown"] = (
        float(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0.0
    )

    # Calmar Ratio
    if max_drawdown != 0:
        calmar_ratio = cagr / abs(max_drawdown)
    else:
        calmar_ratio = 0.0
    metrics["calmar_ratio"] = calmar_ratio

    # Win/Loss metrics
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    metrics["win_rate"] = float(len(wins) / len(returns)) if len(returns) > 0 else 0.0
    metrics["avg_win"] = float(wins.mean()) if len(wins) > 0 else 0.0
    metrics["avg_loss"] = float(losses.mean()) if len(losses) > 0 else 0.0
    metrics["largest_win"] = float(wins.max()) if len(wins) > 0 else 0.0
    metrics["largest_loss"] = float(losses.min()) if len(losses) > 0 else 0.0

    # Profit Factor
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 1.0
    metrics["profit_factor"] = profit_factor

    # Payoff Ratio
    if metrics["avg_loss"] != 0:
        payoff_ratio = abs(metrics["avg_win"] / metrics["avg_loss"])
    else:
        payoff_ratio = float("inf") if metrics["avg_win"] > 0 else 1.0
    metrics["payoff_ratio"] = payoff_ratio

    # Expectancy
    win_rate = metrics["win_rate"]
    if metrics["avg_loss"] != 0:
        expectancy = (win_rate * metrics["avg_win"]) + (
            (1 - win_rate) * metrics["avg_loss"]
        )
    else:
        expectancy = win_rate * metrics["avg_win"]
    metrics["expectancy"] = expectancy

    # VaR and CVaR
    metrics["var_95"] = float(np.percentile(returns, 5))
    metrics["var_99"] = float(np.percentile(returns, 1))
    metrics["cvar_95"] = (
        float(returns[returns <= np.percentile(returns, 5)].mean())
        if len(returns) > 0
        else 0.0
    )
    metrics["cvar_99"] = (
        float(returns[returns <= np.percentile(returns, 1)].mean())
        if len(returns) > 0
        else 0.0
    )

    # Tail Ratio
    right_tail = float(np.percentile(returns, 95))
    left_tail = abs(float(np.percentile(returns, 5)))
    metrics["tail_ratio"] = right_tail / left_tail if left_tail != 0 else 1.0

    # Omega Ratio
    threshold = daily_rf
    gains = returns[returns > threshold] - threshold
    losses_below = threshold - returns[returns <= threshold]

    if losses_below.sum() > 0:
        omega_ratio = float(gains.sum() / losses_below.sum())
    else:
        omega_ratio = float("inf") if gains.sum() > 0 else 1.0
    metrics["omega_ratio"] = omega_ratio

    # Skewness and Kurtosis
    metrics["skewness"] = float(returns.skew())
    metrics["kurtosis"] = float(returns.kurtosis())

    # Consecutive wins/losses
    is_win = (returns > 0).astype(int)
    metrics["max_consecutive_wins"] = _max_consecutive(is_win, 1)
    metrics["max_consecutive_losses"] = _max_consecutive(is_win, 0)

    return metrics


def calculate_benchmark_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Calculate metrics comparing strategy to benchmark.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        Dictionary with benchmark comparison metrics
    """
    # Align the series
    returns, benchmark_returns = returns.align(benchmark_returns, join="inner")

    if len(returns) == 0:
        return {}

    metrics = {}

    # Beta
    covariance = returns.cov(benchmark_returns)
    benchmark_var = benchmark_returns.var()

    if benchmark_var > 0:
        beta = covariance / benchmark_var
    else:
        beta = 0.0
    metrics["beta"] = float(beta)

    # Alpha (Jensen's Alpha)
    daily_rf = risk_free_rate / periods_per_year
    strategy_excess = returns.mean() - daily_rf
    benchmark_excess = benchmark_returns.mean() - daily_rf

    alpha = (strategy_excess - beta * benchmark_excess) * periods_per_year
    metrics["alpha"] = float(alpha)

    # Correlation
    metrics["correlation"] = float(returns.corr(benchmark_returns))

    # Tracking Error
    tracking_diff = returns - benchmark_returns
    tracking_error = float(tracking_diff.std() * np.sqrt(periods_per_year))
    metrics["tracking_error"] = tracking_error

    # Information Ratio
    if tracking_error > 0:
        info_ratio = float(
            (returns.mean() - benchmark_returns.mean())
            * periods_per_year
            / tracking_error
        )
    else:
        info_ratio = 0.0
    metrics["information_ratio"] = info_ratio

    # Up/Down Capture Ratios
    up_days = benchmark_returns > 0
    down_days = benchmark_returns < 0

    if up_days.sum() > 0:
        up_capture = float(returns[up_days].mean() / benchmark_returns[up_days].mean())
    else:
        up_capture = 1.0
    metrics["up_capture"] = up_capture

    if down_days.sum() > 0:
        down_capture = float(
            returns[down_days].mean() / benchmark_returns[down_days].mean()
        )
    else:
        down_capture = 1.0
    metrics["down_capture"] = down_capture

    # Capture Ratio
    if down_capture != 0:
        metrics["capture_ratio"] = up_capture / down_capture
    else:
        metrics["capture_ratio"] = up_capture

    return metrics


def calculate_trade_metrics(
    trades: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Calculate metrics from individual trades.

    Args:
        trades: List of trade dictionaries with 'pnl' and 'duration_days'

    Returns:
        Dictionary with trade-level metrics
    """
    if not trades:
        return {}

    pnls = [t.get("pnl", 0) for t in trades]
    durations = [t.get("duration_days", 0) for t in trades]

    metrics = {}
    metrics["total_trades"] = len(trades)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    metrics["winning_trades"] = len(wins)
    metrics["losing_trades"] = len(losses)
    metrics["win_rate"] = len(wins) / len(pnls) if pnls else 0.0

    metrics["avg_win"] = np.mean(wins) if wins else 0.0
    metrics["avg_loss"] = np.mean(losses) if losses else 0.0
    metrics["largest_win"] = max(wins) if wins else 0.0
    metrics["largest_loss"] = min(losses) if losses else 0.0

    # Profit Factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        metrics["profit_factor"] = gross_profit / gross_loss
    else:
        metrics["profit_factor"] = float("inf") if gross_profit > 0 else 1.0

    # Expectancy
    metrics["expectancy"] = np.mean(pnls) if pnls else 0.0

    # Duration
    metrics["avg_trade_duration_days"] = np.mean(durations) if durations else 0.0

    # Consecutive analysis
    is_win = [1 if p > 0 else 0 for p in pnls]
    metrics["max_consecutive_wins"] = _max_consecutive_list(is_win, 1)
    metrics["max_consecutive_losses"] = _max_consecutive_list(is_win, 0)

    # Payoff Ratio
    if metrics["avg_loss"] != 0:
        metrics["payoff_ratio"] = abs(metrics["avg_win"] / metrics["avg_loss"])
    else:
        metrics["payoff_ratio"] = float("inf") if metrics["avg_win"] > 0 else 1.0

    return metrics


def analyze_drawdowns(
    returns: pd.Series,
    top_n: int = 5,
) -> DrawdownAnalysis:
    """
    Perform detailed drawdown analysis.

    Args:
        returns: Series of returns
        top_n: Number of top drawdown periods to return

    Returns:
        DrawdownAnalysis object
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    max_dd = float(drawdown.min())
    max_dd_pct = max_dd * 100
    current_dd = float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0

    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

    periods = []
    start_dates = drawdown.index[drawdown_starts]
    end_dates = drawdown.index[drawdown_ends]

    for i, start in enumerate(start_dates):
        # Find the end
        ends_after_start = end_dates[end_dates > start]
        if len(ends_after_start) > 0:
            end = ends_after_start[0]
        else:
            end = drawdown.index[-1]

        period_dd = drawdown[start:end]
        if len(period_dd) > 0:
            periods.append(
                {
                    "start": start,
                    "end": end,
                    "duration_days": (
                        (end - start).days
                        if hasattr(end - start, "days")
                        else len(period_dd)
                    ),
                    "max_drawdown": float(period_dd.min()),
                    "recovery_date": end if period_dd.iloc[-1] >= 0 else None,
                }
            )

    # Sort by max drawdown
    periods.sort(key=lambda x: x["max_drawdown"])
    top_periods = periods[:top_n]

    # Calculate max drawdown duration
    if periods:
        max_dd_period = periods[0]
        max_dd_duration = max_dd_period["duration_days"]
        avg_dd_duration = np.mean([p["duration_days"] for p in periods])
    else:
        max_dd_duration = 0
        avg_dd_duration = 0.0

    avg_dd = (
        float(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else 0.0
    )

    return DrawdownAnalysis(
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration_days=int(max_dd_duration),
        avg_drawdown=avg_dd,
        avg_drawdown_duration_days=float(avg_dd_duration),
        current_drawdown=current_dd,
        recovery_time_days=None,  # Would need more analysis
        drawdown_periods=top_periods,
    )


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Series of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame with rolling metrics
    """
    rolling = pd.DataFrame(index=returns.index)

    # Rolling return
    rolling["return"] = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)

    # Rolling volatility
    rolling["volatility"] = returns.rolling(window).std() * np.sqrt(252)

    # Rolling Sharpe
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    rolling["sharpe"] = (
        excess.rolling(window).mean() / returns.rolling(window).std()
    ) * np.sqrt(252)

    # Rolling max drawdown
    cum_returns = (1 + returns).cumprod()

    def rolling_max_dd(x):
        cum = (1 + x).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        return dd.min()

    rolling["max_drawdown"] = returns.rolling(window).apply(rolling_max_dd)

    return rolling


def _max_consecutive(series: pd.Series, value: int) -> int:
    """Find maximum consecutive occurrences of a value."""
    if len(series) == 0:
        return 0

    is_value = (series == value).astype(int)
    groups = (is_value != is_value.shift()).cumsum()
    counts = is_value.groupby(groups).sum()

    return int(counts.max()) if len(counts) > 0 else 0


def _max_consecutive_list(lst: List[int], value: int) -> int:
    """Find maximum consecutive occurrences in a list."""
    if not lst:
        return 0

    max_count = 0
    current_count = 0

    for item in lst:
        if item == value:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count
