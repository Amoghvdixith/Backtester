"""The chronological, phase-aware event loop.

For every session ``t`` (row ``i`` of the master calendar):

OPEN phase
    1. The view advances to (t, OPEN): only the open of t is newly visible.
    2. ``MarketEvent(t, OPEN)`` -> execution fills eligible pending orders at the open
       of t.  Each fill is applied to the portfolio immediately (cash, positions, lots).
    3. ``FillEvent`` / ``OrderStatusEvent`` records are appended to the audit log.

CLOSE phase
    4. The view advances to (t, CLOSE): the full bar of t is visible.
    5. Financing accrues for the calendar days since the previous close, on the
       balances held at the previous close (documented simplification).
    6. Positions are marked at the close of t (stale last-close marks where a held
       instrument has no bar; the number of stale marks is recorded).
    7. Accounting identities are reconciled; an equity snapshot is stored.
    8. ``MarketEvent(t, CLOSE)`` -> strategy.on_bar -> ``SignalEvent`` batch ->
       sizer + constraints -> target weights -> ``OrderEvent`` objects eligible no
       earlier than the open of t+1.

Hence: no order can fill at the close used to compute its signal, fills are
never left unapplied while the strategy computes new targets, and the strategy
always sees post-fill holdings.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .costs import CostModel, FinancingModel
from .data import MarketData, PointInTimeView
from .events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderStatus,
    OrderStatusEvent,
    Phase,
    SignalEvent,
)
from .execution import ExecutionConfig, SimulatedExecution, round_qty
from .portfolio import Portfolio
from .results import BacktestResult
from .sizing import EqualWeight, PortfolioConstraints, Sizer
from .strategy import Strategy
from .universe import StaticUniverse, Universe


@dataclass
class BacktestConfig:
    """Engine-level settings.

    initial_cash: starting cash.
    fractional: allow fractional quantities (False = whole shares, rounded toward zero).
    rebalance_band: skip a trade if |target_w - current_w| < band and both are non-zero.
    cash_buffer: fraction of equity left unallocated to absorb costs and opening gaps.
    trade_start: first session at whose OPEN trading may occur (warm-up before it).
    liquidate_on_final_open: target zero at the second-to-last close so positions are
        sold (with costs) at the final session's open.
    universe_exit: 'liquidate' (target zero when a holding leaves the universe) or 'hold'.
    check_accounting: reconcile accounting identities at every close.
    execution_delay: sessions between the signal close and the earliest eligible open
        (1 = next open, the default; 2 = one extra session of delay).
    """

    initial_cash: float = 1_000_000.0
    fractional: bool = False
    rebalance_band: float = 0.0
    cash_buffer: float = 0.0
    trade_start: pd.Timestamp | str | None = None
    liquidate_on_final_open: bool = False
    universe_exit: str = "liquidate"
    check_accounting: bool = True
    execution_delay: int = 1

    def __post_init__(self) -> None:
        if self.execution_delay < 1:
            raise ValueError("execution_delay must be >= 1 session (no same-close fills)")
        if self.universe_exit not in ("liquidate", "hold"):
            raise ValueError("universe_exit must be 'liquidate' or 'hold'")
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        if self.rebalance_band < 0:
            raise ValueError("rebalance_band must be >= 0")


@dataclass
class _Logs:
    events: list[dict[str, Any]] = field(default_factory=list)
    orders: list[OrderEvent] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    status: list[OrderStatusEvent] = field(default_factory=list)
    signals: list[SignalEvent] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)


class Backtest:
    """Run one strategy on one dataset with injected sizing, costs and execution."""

    def __init__(
        self,
        data: MarketData,
        strategy: Strategy,
        sizer: Sizer | None = None,
        constraints: PortfolioConstraints | None = None,
        costs: CostModel | None = None,
        execution: ExecutionConfig | None = None,
        financing: FinancingModel | None = None,
        universe: Universe | None = None,
        config: BacktestConfig | None = None,
        label: str = "",
        data_mode: str = "unspecified",
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.sizer = sizer or EqualWeight()
        self.constraints = constraints or PortfolioConstraints()
        self.costs = costs or CostModel.zero()
        self.exec_config = execution or ExecutionConfig()
        self.financing = financing or FinancingModel()
        self.universe = universe or StaticUniverse(data.symbols)
        self.config = config or BacktestConfig()
        self.label = label or strategy.name
        self.data_mode = data_mode
        unknown = set(self.universe.all_symbols()) - set(data.symbols)
        if unknown:
            raise ValueError(f"universe symbols missing from data: {sorted(unknown)[:5]}")

    # ------------------------------------------------------------------ helpers
    def _log(self, ts: pd.Timestamp, phase: Phase, kind: str, **kw: Any) -> None:
        self._logs.events.append({"timestamp": ts, "phase": phase.value, "event": kind, **kw})

    def _cash_floor(self, pf: Portfolio) -> float:
        if self.constraints.max_gross <= 1.0:
            return 0.0
        return -(self.constraints.max_gross - 1.0) * max(pf.equity, 0.0)

    def _targets_to_orders(self, i: int, weights: dict[str, float], reason: str) -> None:
        pf, data, ex = self.portfolio, self.data, self.execution
        date = data.index[i]
        eq = pf.equity
        k = i + self.config.execution_delay
        nxt = data.index[k] if k < len(data.index) else None
        close = data.arrays["close"]
        invest = 1.0 - self.config.cash_buffer
        cur_w = pf.weights()
        symbols = sorted(set(weights) | set(pf.positions))
        for s in symbols:
            w = float(weights.get(s, 0.0))
            j = data.sym_index(s)
            price = close[i, j]
            if not np.isfinite(price):
                price = pf.marks.get(s, np.nan)  # stale reference for sizing only
            if not np.isfinite(price) or price <= 0:
                if w != 0:
                    self._log(date, Phase.CLOSE, "target_skipped", symbol=s,
                              detail="no price available to size order")
                continue
            target_q = round_qty(w * invest * eq / price, pf.fractional)
            cur_q = pf.quantity(s)
            delta = target_q - cur_q
            if abs(delta) < 1e-12:
                self._logs.status.extend(ex.cancel(s, date, Phase.CLOSE, "target already held"))
                continue
            cw = cur_w.get(s, 0.0)
            if (target_q != 0 and cur_q != 0 and np.sign(target_q) == np.sign(cur_q)
                    and abs(w * invest - cw) < self.config.rebalance_band):
                self._logs.status.extend(ex.cancel(s, date, Phase.CLOSE, "within rebalance band"))
                continue
            if nxt is None:
                self._log(date, Phase.CLOSE, "order_not_created", symbol=s,
                          detail="final bar: no later session to execute")
                continue
            self._order_id += 1
            order = OrderEvent(
                order_id=self._order_id, timestamp=date, symbol=s, quantity=float(delta),
                created_at=date, eligible_from=nxt, reference_price=float(price),
                expires_after=self.exec_config.expires_after, reason=reason,
            )
            self._queue.append(order)

    # ------------------------------------------------------------------ main loop
    def run(self) -> BacktestResult:
        t0 = time.perf_counter()
        data, cfg = self.data, self.config
        n = len(data.index)
        if n < 2:
            raise ValueError("need at least two sessions")
        self.view = PointInTimeView(data, self.universe)
        self.strategy.bind(self.view)
        self.portfolio = Portfolio(cfg.initial_cash, cfg.fractional)
        self.execution = SimulatedExecution(data, self.costs, self.exec_config)
        self._logs = _Logs()
        self._queue: deque[Any] = deque()
        self._order_id = 0
        ts_idx = 0
        if cfg.trade_start is not None:
            ts_idx = int(data.index.searchsorted(pd.Timestamp(cfg.trade_start)))
            if ts_idx >= n:
                raise ValueError("trade_start is after the last session")
        first_signal_idx = max(0, ts_idx - 1)
        prev_date: pd.Timestamp | None = None
        prev_cash = cfg.initial_cash
        prev_short = 0.0
        close = data.arrays["close"]
        pf = self.portfolio

        for i in range(n):
            date = data.index[i]
            # ------------------------------------------------------------ OPEN
            self.view._advance(i, Phase.OPEN)
            self._queue.append(MarketEvent(date, Phase.OPEN))
            self._drain(i)
            # ------------------------------------------------------------ CLOSE
            self.view._advance(i, Phase.CLOSE)
            if prev_date is not None:
                days = int((date - prev_date).days)
                interest, fee = self.financing.accrue(prev_cash, prev_short, days)
                if interest or fee:
                    pf.accrue_financing(interest, fee)
            marks = {s: float(close[i, data.sym_index(s)]) for s in pf.positions}
            pf.update_marks({s: p for s, p in marks.items() if np.isfinite(p)}, date)
            n_stale = sum(1 for s in pf.positions if pf.mark_dates.get(s) != date)
            if cfg.check_accounting:
                pf.reconcile()
            eq = pf.equity
            if eq <= 0 and not pf.insolvent:
                pf.insolvent = True
                self._logs.status.extend(self.execution.cancel_all(date, Phase.CLOSE,
                                                                   "insolvency"))
                self._log(date, Phase.CLOSE, "insolvent", detail=f"equity={eq:.2f}")
            lv, sv = pf.long_value, pf.short_value
            self._logs.snapshots.append({
                "date": date, "cash": pf.cash, "long_value": lv, "short_value": sv,
                "equity": eq,
                "gross_exposure": (lv - sv) / eq if eq > 0 else np.nan,
                "net_exposure": (lv + sv) / eq if eq > 0 else np.nan,
                "n_positions": len(pf.positions), "n_stale_marks": n_stale,
                "commissions": pf.commissions, "spread_costs": pf.spread_costs,
                "slippage_costs": pf.slippage_costs, "impact_costs": pf.impact_costs,
                "interest": pf.interest, "borrow_fees": pf.borrow_fees,
                "traded_notional": pf.traded_notional, "realised_pnl": pf.realised_pnl,
                "trading": i >= ts_idx,
            })
            prev_date, prev_cash, prev_short = date, pf.cash, abs(sv)
            if i >= first_signal_idx and not pf.insolvent:
                self._queue.append(MarketEvent(date, Phase.CLOSE))
                self._drain(i)

        for o in sorted(self.execution.pending.values(), key=lambda o: o.order_id):
            self._logs.status.append(OrderStatusEvent(data.index[-1], Phase.CLOSE, o.order_id,
                                                      o.symbol, OrderStatus.EXPIRED,
                                                      "end of data"))
        self.execution.pending.clear()
        runtime = time.perf_counter() - t0
        meta = {
            "label": self.label,
            "data_mode": self.data_mode,
            "strategy": self.strategy.describe(),
            "sizer": {"type": type(self.sizer).__name__, **_safe_asdict(self.sizer)},
            "constraints": asdict(self.constraints),
            "costs": _describe_costs(self.costs),
            "execution": asdict(self.exec_config),
            "financing": asdict(self.financing),
            "config": {k: (str(v) if isinstance(v, pd.Timestamp) else v)
                       for k, v in asdict(cfg).items()},
            "execution_convention": "signal at close t, fill at next tradable open (t+1)",
            "data_start": str(data.index[0].date()),
            "data_end": str(data.index[-1].date()),
            "trade_start": str(data.index[ts_idx].date()),
            "n_symbols": len(data.symbols),
            "provenance": data.provenance,
            "impact_fallbacks": self.execution.impact_fallbacks,
            "insolvent": pf.insolvent,
            "runtime_seconds": round(runtime, 3),
        }
        return BacktestResult.from_logs(self._logs, pf, meta, data.index[ts_idx])

    def _drain(self, i: int) -> None:
        """Process the FIFO queue until empty (all events of the current phase)."""
        pf = self.portfolio
        while self._queue:
            ev = self._queue.popleft()
            if isinstance(ev, MarketEvent) and ev.phase is Phase.OPEN:
                fills, status = self.execution.execute_open(i, pf, self._cash_floor(pf))
                self._logs.status.extend(status)
                for f in fills:
                    self._queue.append(f)
            elif isinstance(ev, MarketEvent):
                sigs = self.strategy.on_bar(ev)
                weights = self._resolve_targets(i, sigs)
                if weights is not None:
                    self._targets_to_orders(i, weights[0], weights[1])
            elif isinstance(ev, FillEvent):
                # already applied to the portfolio inside execute_open (sequential cash);
                # recorded here for the audit trail.
                self._logs.fills.append(ev)
                self._log(ev.timestamp, ev.phase, "fill", symbol=ev.symbol,
                          order_id=ev.order_id, quantity=ev.quantity, price=ev.fill_price)
            elif isinstance(ev, OrderEvent):
                self._logs.orders.append(ev)
                self._logs.status.extend(self.execution.submit(ev, Phase.CLOSE))
                self._log(ev.timestamp, Phase.CLOSE, "order", symbol=ev.symbol,
                          order_id=ev.order_id, quantity=ev.quantity,
                          eligible_from=str(ev.eligible_from.date()), reason=ev.reason)
            else:  # pragma: no cover - defensive
                raise TypeError(f"unknown event {ev!r}")

    def _resolve_targets(self, i: int, sigs: list[SignalEvent] | None
                         ) -> tuple[dict[str, float], str] | None:
        """Combine strategy output, universe exits and terminal liquidation into targets."""
        pf, data, cfg = self.portfolio, self.data, self.config
        date = data.index[i]
        members = set(self.view.universe())
        exits = [s for s in pf.positions if s not in members]
        liquidate = cfg.liquidate_on_final_open and i == len(data.index) - 2
        if liquidate:
            if sigs:
                self._logs.signals.extend(sigs)
            if not pf.positions and not self.execution.pending:
                return None
            weights = {s: 0.0 for s in pf.positions}
            self._log(date, Phase.CLOSE, "terminal_liquidation")
            return weights, "terminal_liquidation"
        if sigs is None:
            if exits and cfg.universe_exit == "liquidate":
                w = {s: v for s, v in pf.weights().items() if math.isfinite(v)}
                for s in exits:
                    w[s] = 0.0
                    self._log(date, Phase.CLOSE, "universe_exit", symbol=s)
                return w, "universe_exit"
            return None
        for sg in sigs:
            if sg.timestamp != date:
                raise RuntimeError("strategy emitted a signal with a stale/future timestamp")
            self._logs.signals.append(sg)
            self._log(date, Phase.CLOSE, "signal", symbol=sg.symbol, direction=sg.direction)
        sigs_ok = [s for s in sigs if s.symbol in members or s.direction == 0]
        raw = self.sizer.weights(sigs_ok, self.view)
        weights = self.constraints.apply(raw)
        if cfg.universe_exit == "liquidate":
            for s in exits:
                weights[s] = 0.0
        elif cfg.universe_exit == "hold":
            for s in exits:
                weights.setdefault(s, pf.weights().get(s, 0.0))
        for s, tw in sorted(weights.items()):
            self._logs.targets.append({"date": date, "symbol": s, "target_weight": tw})
        return weights, "rebalance"


def _safe_asdict(obj: Any) -> dict[str, Any]:
    try:
        return asdict(obj)
    except TypeError:
        return {}


def _describe_costs(c: CostModel) -> dict[str, Any]:
    return {
        "commission": asdict(c.commission),
        "spread_bps_full": c.spread.spread_bps,
        "slippage_bps": c.slippage.bps,
        "slippage_stochastic_sd_bps": c.slippage.stochastic_sd_bps,
        "slippage_seed": c.slippage.seed,
        "impact_coefficient": c.impact.coefficient,
        "impact_fallback_bps": c.impact.fallback_bps,
    }
