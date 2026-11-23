"""Simple moving-average crossover (long/cash by default).

For each point-in-time universe member at the close of session t:

    fast_t = mean of the last ``fast`` observed closes (including t)
    slow_t = mean of the last ``slow`` observed closes (including t)
    direction = +1 if fast_t > slow_t else 0   (or -1 if ``allow_short``)

No signal (direction 0) is produced until ``slow`` observed closes exist.  The
strategy only emits a new target set when at least one direction changes, so the
portfolio does not churn while views are unchanged.  Allocation across active
signals is done by the injected sizer (equal weight by default), never by
assigning 100% to every bullish asset.
"""

from __future__ import annotations

import numpy as np

from ..events import MarketEvent, SignalEvent
from ..strategy import Strategy


class SMACrossover(Strategy):
    name = "sma_crossover"

    def __init__(self, fast: int = 50, slow: int = 200, allow_short: bool = False) -> None:
        super().__init__(fast=int(fast), slow=int(slow), allow_short=bool(allow_short))

    def validate_params(self) -> None:
        f, s = self.params["fast"], self.params["slow"]
        if f < 1 or s < 2:
            raise ValueError("windows must be positive")
        if f >= s:
            raise ValueError(f"fast ({f}) must be < slow ({s})")

    @property
    def warmup(self) -> int:
        return int(self.params["slow"])

    def reset(self) -> None:
        self._last: dict[str, float] | None = None

    def directions(self) -> dict[str, float]:
        f, s = self.params["fast"], self.params["slow"]
        out: dict[str, float] = {}
        for sym in self.view.universe():
            px = self.view.history(sym, "close", n=s)
            if len(px) < s:
                out[sym] = 0.0
                continue
            fast_ma, slow_ma = float(np.mean(px[-f:])), float(np.mean(px))
            if fast_ma > slow_ma:
                out[sym] = 1.0
            else:
                out[sym] = -1.0 if self.params["allow_short"] else 0.0
        return out

    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        d = self.directions()
        if d == self._last:
            return None
        self._last = d
        return [self.signal(event, s, v) for s, v in sorted(d.items())]
