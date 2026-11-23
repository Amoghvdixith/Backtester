"""backtester - an educational, event-driven backtesting framework.

Research software, not a trading system.  See README.md and docs/limitations.md.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .costs import Commission, CostModel, FinancingModel, Slippage, SpreadCost, SqrtImpact
from .data import DataValidationError, LookAheadError, MarketData, PointInTimeView
from .engine import Backtest, BacktestConfig
from .events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderStatus,
    OrderStatusEvent,
    Phase,
    SignalEvent,
)
from .execution import ExecutionConfig, SimulatedExecution
from .performance import compute_metrics
from .portfolio import AccountingError, Portfolio
from .results import BacktestResult
from .sizing import (
    EqualWeight,
    FixedFraction,
    KellyCapped,
    PortfolioConstraints,
    Sizer,
    VolTarget,
)
from .strategy import Strategy
from .universe import MembershipUniverse, StaticUniverse, Universe

__all__ = [
    "__version__", "AccountingError", "Backtest", "BacktestConfig", "BacktestResult",
    "Commission", "CostModel", "DataValidationError", "EqualWeight", "ExecutionConfig",
    "FillEvent", "FinancingModel", "FixedFraction", "KellyCapped", "LookAheadError",
    "MarketData", "MarketEvent", "MembershipUniverse", "OrderEvent", "OrderStatus",
    "OrderStatusEvent", "Phase", "PointInTimeView", "Portfolio", "PortfolioConstraints",
    "SignalEvent", "SimulatedExecution", "Sizer", "Slippage", "SpreadCost", "SqrtImpact",
    "StaticUniverse", "Strategy", "Universe", "VolTarget", "compute_metrics",
]
