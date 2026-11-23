"""12-1 cross-sectional momentum (Jegadeesh-Titman style, trading-day approximation).

Signal (computed at the close of the last session of each calendar month, t):

    MOM_i(t) = P_i(t - skip) / P_i(t - lookback) - 1

with offsets counted in sessions of the master calendar (defaults lookback=252,
skip=21, i.e. roughly months t-12 .. t-1, skipping the most recent month).
P are (adjusted) closes.  If the exact session is missing for an instrument, the
most recent observation at most ``max_stale`` sessions earlier is used;
otherwise the instrument is ineligible this month.

Eligibility: point-in-time member at t, an observed close within ``max_stale``
sessions of t (a short halt does not force a sale), both endpoint prices available, and at least ``min_obs_frac * lookback`` observed closes in the window.

Selection: rank by MOM descending; ties broken by symbol name ascending
(deterministic).  ``k = max(min_names, ceil(quantile * N_eligible))``.
Long-only: long the top k.  Long-short: also short the bottom k (requires
``2k <= N_eligible``).  If fewer than ``min_universe`` instruments are eligible,
the strategy targets cash.  Orders execute at the next session's open.
"""

from __future__ import annotations

import math

from ..events import MarketEvent, SignalEvent
from ..strategy import Strategy


class Momentum12_1(Strategy):
    name = "momentum_12_1"

    def __init__(self, lookback: int = 252, skip: int = 21, quantile: float = 0.1,
                 mode: str = "long_only", min_names: int = 1, min_universe: int = 5,
                 max_stale: int = 5, min_obs_frac: float = 0.9) -> None:
        super().__init__(lookback=int(lookback), skip=int(skip), quantile=float(quantile),
                         mode=mode, min_names=int(min_names), min_universe=int(min_universe),
                         max_stale=int(max_stale), min_obs_frac=float(min_obs_frac))

    def validate_params(self) -> None:
        p = self.params
        if not 0 <= p["skip"] < p["lookback"]:
            raise ValueError("need 0 <= skip < lookback")
        if not 0 < p["quantile"] <= 0.5:
            raise ValueError("quantile must be in (0, 0.5]")
        if p["mode"] not in ("long_only", "long_short"):
            raise ValueError("mode must be 'long_only' or 'long_short'")
        if p["min_names"] < 1 or p["min_universe"] < 1:
            raise ValueError("min_names and min_universe must be >= 1")

    @property
    def warmup(self) -> int:
        return int(self.params["lookback"]) + 1

    def reset(self) -> None:
        self.last_scores: dict[str, float] = {}

    def scores(self) -> dict[str, float]:
        p = self.params
        v = self.view
        out: dict[str, float] = {}
        need = math.ceil(p["min_obs_frac"] * p["lookback"])
        for s in v.universe():
            # a recent observed close is required (halts of <= max_stale sessions are
            # tolerated; the order itself still waits for a tradable bar)
            if not v.value_at_offset(s, 0, max_stale=p["max_stale"]) > 0:
                continue
            window = v.history(s, "close", n=p["lookback"] + 1, include_missing=True)
            if len(window) < p["lookback"] + 1:
                continue
            if sum(1 for x in window if x == x) < need:  # NaN != NaN
                continue
            p_end = v.value_at_offset(s, p["skip"], max_stale=p["max_stale"])
            p_start = v.value_at_offset(s, p["lookback"], max_stale=p["max_stale"])
            if not (p_end > 0 and p_start > 0):
                continue
            out[s] = p_end / p_start - 1.0
        return out

    def select(self, scores: dict[str, float]) -> tuple[list[str], list[str]]:
        """Return (longs, shorts) given scores; deterministic ordering."""
        p = self.params
        n = len(scores)
        if n < p["min_universe"]:
            return [], []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        k = max(p["min_names"], math.ceil(p["quantile"] * n))
        if p["mode"] == "long_short":
            k = min(k, n // 2)
            if k == 0:
                return [], []
            longs = [s for s, _ in ranked[:k]]
            shorts_ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]))
            shorts = [s for s, _ in shorts_ranked[:k]]
            return longs, shorts
        k = min(k, n)
        return [s for s, _ in ranked[:k]], []

    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        if not self.view.is_last_session_of_month():
            return None
        sc = self.scores()
        self.last_scores = sc
        longs, shorts = self.select(sc)
        sigs = [self.signal(event, s, 1.0) for s in longs]
        sigs += [self.signal(event, s, -1.0) for s in shorts]
        return sigs
