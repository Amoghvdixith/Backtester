"""Simulated execution at the session open.

Rules
-----
* An order created at the close of session t is eligible from the open of the next
  session in the master calendar.  It fills at the first open, on or after that
  session, at which **its own symbol** has a tradable bar.  It never fills at the
  close used to create it.
* If the symbol has no tradable bar, the order waits; after ``expires_after``
  further sessions without a tradable bar it EXPIRES.
* Only the session open (and lagged estimates) are used to fill: no same-day
  high, low, close or volume.
* Within one open, cash-raising orders (sells/shorts) are processed before buys,
  each group in symbol order (deterministic).  If buys need more cash than is
  available, all buys are scaled down pro rata (policy ``resize``) or rejected
  (policy ``reject``).  Quantities are rounded toward zero to whole shares unless
  the portfolio is fractional.
* Optional ``max_participation`` caps |q| at a fraction of *lagged* ADV; the
  remainder is cancelled (status PARTIAL).  Default: unlimited liquidity
  approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .data import MarketData
from .events import FillEvent, OrderEvent, OrderStatus, OrderStatusEvent, Phase
from .portfolio import Portfolio


@dataclass(frozen=True)
class ExecutionConfig:
    expires_after: int = 5
    insufficient_cash: str = "resize"  # "resize" | "reject"
    max_participation: float | None = None
    adv_window: int = 20
    vol_window: int = 20

    def __post_init__(self) -> None:
        if self.insufficient_cash not in ("resize", "reject"):
            raise ValueError("insufficient_cash must be 'resize' or 'reject'")
        if self.expires_after < 0:
            raise ValueError("expires_after must be >= 0")
        if self.max_participation is not None and not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")


def round_qty(q: float, fractional: bool) -> float:
    if fractional:
        return float(q)
    return float(math.trunc(q))


class SimulatedExecution:
    """Holds pending orders and turns eligible ones into fills at the open."""

    def __init__(self, data: MarketData, costs: CostModel | None = None,
                 config: ExecutionConfig | None = None) -> None:
        self.data = data
        self.costs = costs or CostModel.zero()
        self.config = config or ExecutionConfig()
        self.pending: dict[str, OrderEvent] = {}  # one live order per symbol
        self._est = data.estimates(self.config.adv_window, self.config.vol_window)
        self.impact_fallbacks = 0

    # ---------------------------------------------------------------- order book
    def submit(self, order: OrderEvent, phase: Phase = Phase.CLOSE) -> list[OrderStatusEvent]:
        """Queue an order; an existing pending order for the same symbol is REPLACED."""
        out: list[OrderStatusEvent] = []
        old = self.pending.get(order.symbol)
        if old is not None:
            out.append(OrderStatusEvent(order.timestamp, phase, old.order_id, old.symbol,
                                        OrderStatus.REPLACED, f"replaced by {order.order_id}"))
        self.pending[order.symbol] = order
        out.append(OrderStatusEvent(order.timestamp, phase, order.order_id, order.symbol,
                                    OrderStatus.SUBMITTED, order.reason,
                                    {"quantity": order.quantity}))
        return out

    def cancel_all(self, ts: pd.Timestamp, phase: Phase, reason: str) -> list[OrderStatusEvent]:
        out = [OrderStatusEvent(ts, phase, o.order_id, o.symbol, OrderStatus.CANCELLED, reason)
               for o in sorted(self.pending.values(), key=lambda o: o.order_id)]
        self.pending.clear()
        return out

    def cancel(self, symbol: str, ts: pd.Timestamp, phase: Phase,
               reason: str) -> list[OrderStatusEvent]:
        o = self.pending.pop(symbol, None)
        if o is None:
            return []
        return [OrderStatusEvent(ts, phase, o.order_id, o.symbol, OrderStatus.CANCELLED, reason)]

    # ---------------------------------------------------------------- pricing
    def _price(self, i: int, j: int, q: float, slip_frac: float) -> tuple[float, float, float,
                                                                         float, float]:
        ref = float(self.data.arrays["open"][i, j])
        hs = self.costs.spread.fraction()
        imp, fb = self.costs.impact.fraction(q, float(self._est.adv[i, j]),
                                             float(self._est.sigma[i, j]))
        if fb:
            self.impact_fallbacks += 1
        sign = 1.0 if q > 0 else -1.0
        px = ref * (1.0 + sign * (hs + slip_frac + imp))
        if px <= 0:
            px = ref * 1e-6
        notional_ref = abs(q) * ref
        return px, ref, notional_ref * hs, notional_ref * slip_frac, notional_ref * imp

    def _make_fill(self, i: int, order: OrderEvent, q: float, slip_frac: float,
                   status: OrderStatus) -> FillEvent:
        j = self.data.sym_index(order.symbol)
        px, ref, sc, slc, ic = self._price(i, j, q, slip_frac)
        comm = self.costs.commission(q, px)
        return FillEvent(
            timestamp=self.data.index[i], phase=Phase.OPEN, order_id=order.order_id,
            symbol=order.symbol, quantity=q, reference_price=ref,
            decision_price=order.reference_price, fill_price=px, commission=comm,
            spread_cost=sc, slippage_cost=slc, impact_cost=ic,
            order_created_at=order.created_at, status=status,
        )

    # ---------------------------------------------------------------- execution
    def execute_open(self, i: int, portfolio: Portfolio,
                     cash_floor: float = 0.0) -> tuple[list[FillEvent], list[OrderStatusEvent]]:
        """Fill eligible pending orders at the open of row ``i``.

        Fills are applied to ``portfolio`` immediately (so later orders in the same open
        see updated cash).  ``cash_floor`` is the lowest cash balance buys may leave
        (0 without leverage; negative when margin borrowing is permitted).
        """
        date = self.data.index[i]
        fills: list[FillEvent] = []
        status: list[OrderStatusEvent] = []
        ready: list[OrderEvent] = []
        for sym in sorted(self.pending):
            o = self.pending[sym]
            if o.eligible_from > date:
                continue
            j = self.data.sym_index(sym)
            if self.data.tradable[i, j] and np.isfinite(self.data.arrays["open"][i, j]):
                ready.append(o)
                continue
            waited = int(self.data.index.searchsorted(date) -
                         self.data.index.searchsorted(o.eligible_from))
            if waited >= o.expires_after:
                status.append(OrderStatusEvent(date, Phase.OPEN, o.order_id, sym,
                                               OrderStatus.EXPIRED,
                                               "no tradable bar before expiry"))
                del self.pending[sym]
        for o in ready:
            del self.pending[o.symbol]
        sells = [o for o in ready if o.quantity < 0]
        buys = [o for o in ready if o.quantity > 0]
        frac = portfolio.fractional

        def participation_cap(o: OrderEvent) -> tuple[float, bool]:
            q = o.quantity
            mp = self.config.max_participation
            if mp is None:
                return q, False
            adv = float(self._est.adv[i, self.data.sym_index(o.symbol)])
            if not math.isfinite(adv) or adv <= 0:
                return q, False
            cap = round_qty(mp * adv, frac)
            if abs(q) > cap:
                return math.copysign(cap, q), True
            return q, False

        for o in sells:
            q, partial = participation_cap(o)
            q = round_qty(q, frac)
            if q == 0:
                status.append(OrderStatusEvent(date, Phase.OPEN, o.order_id, o.symbol,
                                               OrderStatus.CANCELLED, "zero after participation cap"))
                continue
            st = OrderStatus.PARTIAL if partial else OrderStatus.FILLED
            f = self._make_fill(i, o, q, self.costs.slippage.fraction(), st)
            portfolio.apply_fill(f)
            fills.append(f)
            status.append(OrderStatusEvent(date, Phase.OPEN, o.order_id, o.symbol, st,
                                           "", {"filled": q, "requested": o.quantity}))

        if not buys:
            return fills, status
        slips = {o.order_id: self.costs.slippage.fraction() for o in buys}
        qty: dict[int, float] = {}
        partials: dict[int, bool] = {}
        for o in buys:
            q, partial = participation_cap(o)
            qty[o.order_id] = round_qty(q, frac)
            partials[o.order_id] = partial

        def cash_needed(qs: dict[int, float]) -> float:
            tot = 0.0
            for o in buys:
                q = qs[o.order_id]
                if q == 0:
                    continue
                j = self.data.sym_index(o.symbol)
                px = self._price(i, j, q, slips[o.order_id])[0]
                tot += q * px + self.costs.commission(q, px)
            return tot

        available = portfolio.cash - cash_floor
        need = cash_needed(qty)
        resized = False
        if need > available + 1e-9:
            if self.config.insufficient_cash == "reject" or available <= 0:
                for o in buys:
                    status.append(OrderStatusEvent(
                        date, Phase.OPEN, o.order_id, o.symbol, OrderStatus.REJECTED,
                        "insufficient cash", {"needed": need, "available": available}))
                return fills, status
            resized = True
            base = dict(qty)
            k = available / need
            for _ in range(50):
                qty = {oid: round_qty(base[oid] * k, frac) for oid in base}
                need = cash_needed(qty)
                if need <= available + 1e-9:
                    break
                k *= 0.999 if frac else max(0.0, 1 - 1.0 / max(1.0, max(base.values())))
            else:  # pragma: no cover - defensive
                qty = {oid: 0.0 for oid in base}
        for o in buys:
            q = qty[o.order_id]
            if q <= 0:
                status.append(OrderStatusEvent(date, Phase.OPEN, o.order_id, o.symbol,
                                               OrderStatus.REJECTED,
                                               "insufficient cash after resizing"))
                continue
            if resized and q < o.quantity:
                st = OrderStatus.RESIZED
            elif partials[o.order_id]:
                st = OrderStatus.PARTIAL
            else:
                st = OrderStatus.FILLED
            f = self._make_fill(i, o, q, slips[o.order_id], st)
            portfolio.apply_fill(f)
            fills.append(f)
            status.append(OrderStatusEvent(date, Phase.OPEN, o.order_id, o.symbol, st, "",
                                           {"filled": q, "requested": o.quantity}))
        return fills, status
