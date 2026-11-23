"""Buy-and-hold validation strategy and an equal-weight universe benchmark."""

from __future__ import annotations

from ..events import MarketEvent, SignalEvent
from ..strategy import Strategy


class BuyAndHold(Strategy):
    """Emit a long signal for ``symbols`` once, at the first close it is called.

    With the default engine convention the purchase fills at the NEXT session's open.
    Weights are set by the sizer (equal weight by default).  Nothing is emitted after
    the first bar, so no rebalancing occurs.
    """

    name = "buy_and_hold"

    def __init__(self, symbols: list[str] | None = None) -> None:
        super().__init__(symbols=list(symbols) if symbols else None)

    def reset(self) -> None:
        self._done = False

    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        if self._done:
            return None
        syms = self.params["symbols"] or self.view.universe()
        # only buy instruments with an observed bar today (a price to size against)
        syms = [s for s in syms if len(self.view.history(s, "close", n=1))]
        if not syms:
            return None
        self._done = True
        return [self.signal(event, s, 1.0) for s in syms]


class EqualWeightUniverse(Strategy):
    """Benchmark: hold all eligible point-in-time members equally, rebalanced monthly.

    ``min_history`` lets the benchmark use the same eligibility filter as a strategy
    (e.g. 253 observed closes for 12-1 momentum) so that universes are comparable.
    """

    name = "equal_weight_universe"

    def __init__(self, min_history: int = 1, rebalance: str = "monthly") -> None:
        super().__init__(min_history=int(min_history), rebalance=rebalance)

    def validate_params(self) -> None:
        if self.params["rebalance"] not in ("monthly", "once"):
            raise ValueError("rebalance must be 'monthly' or 'once'")
        if self.params["min_history"] < 1:
            raise ValueError("min_history must be >= 1")

    def reset(self) -> None:
        self._started = False

    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        if self._started and (self.params["rebalance"] == "once"
                              or not self.view.is_last_session_of_month()):
            return None
        n = self.params["min_history"]
        elig = [s for s in self.view.universe()
                if len(self.view.history(s, "close", n=n)) >= n
                and self.view.value_at_offset(s, 0, max_stale=5) > 0]
        if not elig:
            return None
        self._started = True
        return [self.signal(event, s, 1.0) for s in elig]
