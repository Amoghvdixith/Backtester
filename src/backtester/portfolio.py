"""Portfolio accounting: cash, positions, FIFO lots, marks, financing and round trips.

Accounting identities (checked by :meth:`Portfolio.reconcile`):

* ``equity = cash + sum_i quantity_i * mark_i``  (signed market value; shorts negative)
* ``equity = initial_cash + realised_pnl + unrealised_pnl - commissions
  + interest - borrow_fees``

where realised/unrealised P&L are measured against *fill prices*, which already
contain spread, slippage and impact.  Lots are matched FIFO; a fill that crosses
zero first closes existing lots then opens a new lot in the opposite direction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .events import FillEvent

ABS_TOL = 1e-6  # currency units
REL_TOL = 1e-9


class AccountingError(RuntimeError):
    """Raised when an accounting identity is violated beyond tolerance."""


@dataclass
class Lot:
    symbol: str
    quantity: float  # signed
    entry_price: float
    entry_date: pd.Timestamp
    entry_commission_per_unit: float
    order_id: int


@dataclass(frozen=True)
class RoundTrip:
    """A completed (possibly partial) lot closure."""

    symbol: str
    direction: str  # "long" or "short"
    quantity: float  # absolute closed quantity
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_pnl: float  # price P&L at fill prices (includes spread/slippage/impact)
    commissions: float  # entry share (pro rata) + exit share
    net_pnl: float
    holding_days: int
    return_pct: float  # net_pnl / |entry notional|


class Portfolio:
    """Holds state; it does not decide what to trade (see :mod:`backtester.rebalance`)."""

    def __init__(self, initial_cash: float, fractional: bool = False) -> None:
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be a positive finite number")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fractional = fractional
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.mark_dates: dict[str, pd.Timestamp] = {}
        self.lots: dict[str, deque[Lot]] = {}
        self.realised_pnl = 0.0
        self.commissions = 0.0
        self.spread_costs = 0.0
        self.slippage_costs = 0.0
        self.impact_costs = 0.0
        self.interest = 0.0
        self.borrow_fees = 0.0
        self.traded_notional = 0.0
        self.round_trips: list[RoundTrip] = []
        self.insolvent = False

    # ----------------------------------------------------------------- valuation
    def quantity(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def market_value(self, symbol: str) -> float:
        return self.quantity(symbol) * self.marks.get(symbol, np.nan)

    @property
    def long_value(self) -> float:
        return float(sum(q * self.marks[s] for s, q in self.positions.items() if q > 0))

    @property
    def short_value(self) -> float:
        """Signed (negative) short market value."""
        return float(sum(q * self.marks[s] for s, q in self.positions.items() if q < 0))

    @property
    def equity(self) -> float:
        return self.cash + self.long_value + self.short_value

    def weights(self) -> dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {s: float("nan") for s in self.positions}
        return {s: q * self.marks[s] / eq for s, q in self.positions.items()}

    @property
    def unrealised_pnl(self) -> float:
        tot = 0.0
        for s, lots in self.lots.items():
            m = self.marks[s]
            for lot in lots:
                tot += lot.quantity * (m - lot.entry_price)
        return tot

    def update_marks(self, prices: dict[str, float], date: pd.Timestamp) -> None:
        for s, p in prices.items():
            if np.isfinite(p) and p > 0:
                self.marks[s] = float(p)
                self.mark_dates[s] = date

    # ----------------------------------------------------------------- fills
    def apply_fill(self, fill: FillEvent) -> None:
        """Apply a fill immediately: cash, position, lots, realised P&L, cost tallies."""
        s, q, px = fill.symbol, float(fill.quantity), float(fill.fill_price)
        if q == 0:
            return
        if not np.isfinite(px) or px <= 0:
            raise AccountingError(f"invalid fill price {px} for {s}")
        self.cash -= q * px + fill.commission
        self.commissions += fill.commission
        self.spread_costs += fill.spread_cost
        self.slippage_costs += fill.slippage_cost
        self.impact_costs += fill.impact_cost
        self.traded_notional += abs(q) * px
        comm_per_unit = fill.commission / abs(q)
        lots = self.lots.setdefault(s, deque())
        remaining = q
        while remaining != 0 and lots and np.sign(lots[0].quantity) != np.sign(remaining):
            lot = lots[0]
            close_qty = min(abs(remaining), abs(lot.quantity))
            sign = np.sign(lot.quantity)  # +1 long lot, -1 short lot
            gross = sign * close_qty * (px - lot.entry_price)
            comm = close_qty * (lot.entry_commission_per_unit + comm_per_unit)
            self.realised_pnl += gross
            entry_notional = close_qty * lot.entry_price
            self.round_trips.append(RoundTrip(
                symbol=s, direction="long" if sign > 0 else "short", quantity=close_qty,
                entry_date=lot.entry_date, exit_date=fill.timestamp,
                entry_price=lot.entry_price, exit_price=px, gross_pnl=gross,
                commissions=comm, net_pnl=gross - comm,
                holding_days=int((fill.timestamp - lot.entry_date).days),
                return_pct=(gross - comm) / entry_notional if entry_notional else np.nan,
            ))
            lot.quantity -= sign * close_qty
            remaining += sign * close_qty
            if abs(lot.quantity) <= 1e-12:
                lots.popleft()
            if abs(remaining) <= 1e-12:
                remaining = 0.0
        if remaining != 0:
            lots.append(Lot(s, remaining, px, fill.timestamp, comm_per_unit, fill.order_id))
        newq = self.positions.get(s, 0.0) + q
        if abs(newq) <= 1e-12:
            self.positions.pop(s, None)
            self.lots.pop(s, None)
        else:
            self.positions[s] = newq
        if s not in self.marks:
            self.marks[s] = px
            self.mark_dates[s] = fill.timestamp

    # ----------------------------------------------------------------- financing
    def accrue_financing(self, interest: float, borrow_fee: float) -> None:
        self.cash += interest - borrow_fee
        self.interest += interest
        self.borrow_fees += borrow_fee

    # ----------------------------------------------------------------- checks
    def reconcile(self) -> float:
        """Check both accounting identities; return the larger absolute discrepancy."""
        eq = self.equity
        lot_qty = {s: sum(lot.quantity for lot in lots) for s, lots in self.lots.items()}
        for s, q in self.positions.items():
            if abs(lot_qty.get(s, 0.0) - q) > 1e-9 * max(1.0, abs(q)):
                raise AccountingError(f"lot quantities do not sum to position for {s}")
        alt = (self.initial_cash + self.realised_pnl + self.unrealised_pnl
               - self.commissions + self.interest - self.borrow_fees)
        tol = ABS_TOL + REL_TOL * (abs(eq) + self.traded_notional)
        diff = abs(eq - alt)
        if diff > tol:
            raise AccountingError(f"equity {eq:.6f} != P&L decomposition {alt:.6f}")
        return diff

    def round_trips_frame(self) -> pd.DataFrame:
        cols = list(RoundTrip.__dataclass_fields__)
        return pd.DataFrame([rt.__dict__ for rt in self.round_trips], columns=cols)
