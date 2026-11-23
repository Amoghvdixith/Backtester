"""Immutable event types and the session-phase model.

Timing model (daily bars)
-------------------------
Each trading session ``t`` is processed in two phases, in this order:

* ``Phase.OPEN``  - only the opening price of session ``t`` is known.  Orders that
  were created at an earlier close and are eligible now are executed at the open.
  The resulting fills are applied to the portfolio immediately.
* ``Phase.CLOSE`` - the full bar of session ``t`` (open, high, low, close, volume)
  is complete.  The portfolio is marked to market, financing is accrued, the
  strategy observes the completed bar and may emit signals.  Orders created now
  are eligible no earlier than the *next* session's open.

Events are frozen dataclasses so that nothing downstream can rewrite history.
Order status changes are recorded as separate :class:`OrderStatusEvent` records
rather than by mutating the original order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd


class Phase(StrEnum):
    """Intraday phase of a daily session."""

    OPEN = "open"
    CLOSE = "close"


class OrderStatus(StrEnum):
    """Lifecycle states of an order."""

    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"  # filled in part; the remainder was cancelled
    RESIZED = "resized"  # filled at a smaller size because of cash/leverage limits
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REPLACED = "replaced"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A new piece of market information became available.

    Attributes
    ----------
    timestamp: the session date.
    phase: ``Phase.OPEN`` (only opening prices known) or ``Phase.CLOSE`` (bar complete).
    """

    timestamp: pd.Timestamp
    phase: Phase


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A strategy's view on one instrument, created after a bar closes.

    ``direction`` is the strategy's raw view (e.g. +1 long, 0 flat, -1 short, or a
    score).  Converting directions into portfolio weights is the job of the sizing
    component, not the strategy.  A batch of signals emitted on the same bar is
    interpreted as a *complete* target portfolio: held instruments that are not
    mentioned get a target of zero.
    """

    timestamp: pd.Timestamp
    symbol: str
    direction: float
    strategy_id: str = ""
    created_phase: Phase = Phase.CLOSE

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("SignalEvent.symbol must be a non-empty string")
        if not pd.notna(self.direction):
            raise ValueError("SignalEvent.direction must be finite")


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A request to trade a signed quantity.

    Attributes
    ----------
    order_id: unique, monotonically increasing integer.
    created_at: session whose close produced the order.
    eligible_from: earliest session at whose *open* the order may execute.
    quantity: signed quantity (positive = buy, negative = sell/short).
    reference_price: decision price (the close used to size the order).
    expires_after: number of sessions after ``eligible_from`` during which the order may
        wait for a tradable bar before it expires.
    reason: free-text tag ("rebalance", "universe_exit", "terminal_liquidation", ...).
    """

    order_id: int
    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    created_at: pd.Timestamp
    eligible_from: pd.Timestamp
    reference_price: float
    expires_after: int = 5
    reason: str = "rebalance"

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("OrderEvent.quantity must be non-zero")
        if self.eligible_from <= self.created_at:
            raise ValueError("An order cannot be eligible at or before the bar that created it")

    @property
    def side(self) -> str:
        return "buy" if self.quantity > 0 else "sell"


@dataclass(frozen=True, slots=True)
class FillEvent:
    """An executed trade with itemised costs.

    Price-based costs (half-spread, slippage, impact) are embedded in
    ``fill_price``; they are reported here as cash-equivalent amounts for
    reconciliation only and must NOT be deducted from cash a second time.
    ``commission`` is the only explicit cash fee.
    """

    timestamp: pd.Timestamp
    phase: Phase
    order_id: int
    symbol: str
    quantity: float
    reference_price: float  # arrival price (the session open)
    decision_price: float  # close at which the order was sized
    fill_price: float
    commission: float
    spread_cost: float
    slippage_cost: float
    impact_cost: float
    order_created_at: pd.Timestamp
    status: OrderStatus = OrderStatus.FILLED

    @property
    def notional(self) -> float:
        """Signed cash value at the execution price (positive for buys)."""
        return self.quantity * self.fill_price

    @property
    def shortfall(self) -> float:
        """Execution shortfall vs the arrival price (always >= 0 for adverse costs)."""
        return self.quantity * (self.fill_price - self.reference_price)


@dataclass(frozen=True, slots=True)
class OrderStatusEvent:
    """A change of state for an order (immutable audit record)."""

    timestamp: pd.Timestamp
    phase: Phase
    order_id: int
    symbol: str
    status: OrderStatus
    reason: str = ""
    detail: dict[str, float] = field(default_factory=dict)
