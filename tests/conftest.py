"""Small deterministic datasets and helper strategies for tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import MarketData, Strategy
from backtester.events import MarketEvent, Phase, SignalEvent


def make_data(closes: dict[str, list[float]], opens: dict[str, list[float]] | None = None,
              start: str = "2021-01-04", volume: float = 1_000_000.0,
              dates: pd.DatetimeIndex | None = None) -> MarketData:
    """Wide OHLCV from close (and optional open) lists; NaN close = missing bar."""
    n = len(next(iter(closes.values())))
    idx = dates if dates is not None else pd.bdate_range(start, periods=n)
    frames: dict[str, dict[str, list[float]]] = {f: {} for f in
                                                 ("open", "high", "low", "close", "volume")}
    for s, c in closes.items():
        c = np.asarray(c, float)
        o = np.asarray(opens[s], float) if opens and s in opens else c.copy()
        o = np.where(np.isnan(c), np.nan, o)
        frames["open"][s] = list(o)
        frames["close"][s] = list(c)
        frames["high"][s] = list(np.fmax(o, c) * 1.01)
        frames["low"][s] = list(np.fmin(o, c) * 0.99)
        frames["volume"][s] = list(np.where(np.isnan(c), np.nan, volume))
    dfs = {f: pd.DataFrame(v, index=idx) for f, v in frames.items()}
    return MarketData(dfs, {"data_mode": "test"})


def random_walk(n: int = 300, symbols: tuple[str, ...] = ("A", "B"), seed: int = 0,
                start: str = "2015-01-01") -> MarketData:
    rng = np.random.default_rng(seed)
    closes, opens = {}, {}
    for s in symbols:
        r = rng.normal(0.0003, 0.012, n)
        c = 100 * np.exp(np.cumsum(r))
        o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, 0.003, n))
        closes[s], opens[s] = list(c), list(o)
    return make_data(closes, opens, start=start)


class Recorder(Strategy):
    """Emits pre-programmed targets {date: {symbol: direction}} and records what it saw."""

    name = "recorder"

    def __init__(self, plan: dict[pd.Timestamp, dict[str, float]] | None = None) -> None:
        super().__init__()
        self.plan = {pd.Timestamp(k): v for k, v in (plan or {}).items()}
        self.seen: list[tuple[pd.Timestamp, Phase, float]] = []

    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        sym = self.view.all_symbols[0]
        self.seen.append((event.timestamp, event.phase, self.view.history(sym, "close", 1)[-1]))
        tgt = self.plan.get(event.timestamp)
        if tgt is None:
            return None
        return [self.signal(event, s, d) for s, d in tgt.items()]


@pytest.fixture
def rw() -> MarketData:
    return random_walk()
