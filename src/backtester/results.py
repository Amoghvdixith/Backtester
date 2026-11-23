"""Backtest result container, audit tables and persistence."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .portfolio import Portfolio


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if hasattr(obj, "value") and obj.__class__.__module__.endswith("events"):
        return obj.value
    return obj


@dataclass
class BacktestResult:
    equity: pd.DataFrame  # one row per session (index = date)
    fills: pd.DataFrame
    orders: pd.DataFrame
    order_status: pd.DataFrame
    signals: pd.DataFrame
    targets: pd.DataFrame
    round_trips: pd.DataFrame
    events: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)
    trade_start: pd.Timestamp | None = None

    @classmethod
    def from_logs(cls, logs: Any, pf: Portfolio, meta: dict[str, Any],
                  trade_start: pd.Timestamp) -> BacktestResult:
        eq = pd.DataFrame(logs.snapshots).set_index("date")
        fills = pd.DataFrame([{**asdict(f), "status": f.status.value, "phase": f.phase.value,
                               "notional": f.notional, "shortfall": f.shortfall}
                              for f in logs.fills])
        orders = pd.DataFrame([asdict(o) for o in logs.orders])
        status = pd.DataFrame([{**asdict(s), "status": s.status.value, "phase": s.phase.value,
                                "detail": json.dumps(_jsonable(s.detail))}
                               for s in logs.status])
        signals = pd.DataFrame([{**asdict(s), "created_phase": s.created_phase.value}
                                for s in logs.signals])
        return cls(
            equity=eq, fills=fills, orders=orders, order_status=status, signals=signals,
            targets=pd.DataFrame(logs.targets), round_trips=pf.round_trips_frame(),
            events=pd.DataFrame(logs.events), meta=meta, trade_start=trade_start,
        )

    # ------------------------------------------------------------------ series
    def returns(self, start: pd.Timestamp | str | None = None,
                end: pd.Timestamp | str | None = None) -> pd.Series:
        """Daily close-to-close equity returns for sessions in [start, end].

        Defaults to the trading window (from ``trade_start``): the first scored return
        is the session of ``trade_start`` relative to the previous close.
        """
        e = self.equity["equity"]
        r = e.pct_change()
        start = pd.Timestamp(start) if start is not None else self.trade_start
        if start is not None:
            r = r[r.index >= start]
        if end is not None:
            r = r[r.index <= pd.Timestamp(end)]
        r = r.dropna()
        r.name = "return"
        return r

    def equity_curve(self, start: pd.Timestamp | str | None = None) -> pd.Series:
        e = self.equity["equity"]
        start = pd.Timestamp(start) if start is not None else self.trade_start
        if start is not None:
            pos = e.index.searchsorted(start)
            e = e.iloc[max(0, pos - 1):]
        return e

    def cost_summary(self) -> dict[str, float]:
        last = self.equity.iloc[-1]
        out = {k: float(last[k]) for k in ("commissions", "spread_costs", "slippage_costs",
                                           "impact_costs", "interest", "borrow_fees",
                                           "traded_notional")}
        out["total_trading_costs"] = (out["commissions"] + out["spread_costs"]
                                      + out["slippage_costs"] + out["impact_costs"])
        out["net_financing"] = out["interest"] - out["borrow_fees"]
        return out

    def reconciliation(self) -> pd.DataFrame:
        """Per-fill reconciliation of reference vs execution value and fees."""
        if self.fills.empty:
            return pd.DataFrame()
        f = self.fills
        out = pd.DataFrame({
            "timestamp": f["timestamp"], "order_id": f["order_id"], "symbol": f["symbol"],
            "quantity": f["quantity"], "decision_price": f["decision_price"],
            "reference_price": f["reference_price"], "fill_price": f["fill_price"],
            "reference_value": f["quantity"] * f["reference_price"],
            "execution_value": f["quantity"] * f["fill_price"],
            "execution_shortfall": f["shortfall"],
            "spread_cost": f["spread_cost"], "slippage_cost": f["slippage_cost"],
            "impact_cost": f["impact_cost"], "commission": f["commission"],
            "delay_cost_vs_decision": f["quantity"] * (f["reference_price"]
                                                       - f["decision_price"]),
        })
        out["components_minus_shortfall"] = (out["spread_cost"] + out["slippage_cost"]
                                             + out["impact_cost"] - out["execution_shortfall"])
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path, prefix: str = "") -> dict[str, str]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = f"{prefix}_" if prefix else ""
        written = {}
        tables = {"equity": self.equity, "fills": self.fills, "orders": self.orders,
                  "order_status": self.order_status, "signals": self.signals,
                  "targets": self.targets, "round_trips": self.round_trips,
                  "events": self.events, "reconciliation": self.reconciliation()}
        for name, df in tables.items():
            path = d / f"{p}{name}.csv"
            df.to_csv(path, index=(name == "equity"))
            written[name] = path.name
        meta_path = d / f"{p}meta.json"
        meta_path.write_text(json.dumps(_jsonable(self.meta), indent=2, default=str),
                             encoding="utf-8")
        written["meta"] = meta_path.name
        return written


def environment_info() -> dict[str, Any]:
    """Versions of Python and key packages (for run metadata)."""
    import importlib

    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "scipy", "matplotlib", "yaml", "backtester"):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:  # pragma: no cover
            info[mod] = None
    return info
