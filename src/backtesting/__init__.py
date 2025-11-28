"""
Backtesting module with enhanced execution modeling and validation.

Components:
- BacktestEngine: Event-driven engine with realistic execution
- PositionManager: Position sizing and execution cost modeling
- WalkForwardBacktester: Walk-forward analysis with parameter optimization
- Metrics: Comprehensive performance metrics calculation
"""

from src.backtesting.engine import BacktestEngine, ExecutionMode
from src.backtesting.metrics import (
    calculate_metrics,
    calculate_benchmark_metrics,
    calculate_trade_metrics,
    analyze_drawdowns,
    calculate_rolling_metrics,
    DrawdownAnalysis,
    TradeAnalysis,
    RiskMetrics,
    BenchmarkMetrics,
)
from src.backtesting.position_manager import (
    PositionManager,
    SlippageModel,
    ExecutionQuality,
    TransactionCosts,
    ExecutionResult,
    PositionSizeResult,
)
from src.backtesting.strategy import Strategy
from src.backtesting.walk_forward import (
    WalkForwardBacktester,
    WalkForwardMode,
    WalkForwardStep,
    WalkForwardAnalysis,
    ComboPurgedCV,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "ExecutionMode",
    # Position Management
    "PositionManager",
    "SlippageModel",
    "ExecutionQuality",
    "TransactionCosts",
    "ExecutionResult",
    "PositionSizeResult",
    # Metrics
    "calculate_metrics",
    "calculate_benchmark_metrics",
    "calculate_trade_metrics",
    "analyze_drawdowns",
    "calculate_rolling_metrics",
    "DrawdownAnalysis",
    "TradeAnalysis",
    "RiskMetrics",
    "BenchmarkMetrics",
    # Strategy
    "Strategy",
    # Walk-Forward
    "WalkForwardBacktester",
    "WalkForwardMode",
    "WalkForwardStep",
    "WalkForwardAnalysis",
    "ComboPurgedCV",
]
