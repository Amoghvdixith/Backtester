"""Point-in-time universe definitions.

A universe answers one question: *which instruments was the strategy allowed to
consider on date t, using only information known on date t?*

``MembershipUniverse`` takes membership intervals with an optional
``known_from`` date (when the membership change became public).  An instrument
is a member on ``t`` when ``max(start, known_from) <= t <= end``.

Important: a universe built from *today's* index constituents is NOT
survivorship-bias-free, whatever interface is used to load it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd


class Universe(ABC):
    """Interface for point-in-time membership."""

    @abstractmethod
    def members(self, date: pd.Timestamp) -> list[str]:
        """Sorted list of members known on ``date``."""

    def all_symbols(self) -> list[str]:  # pragma: no cover - overridden
        return []


class StaticUniverse(Universe):
    """Fixed membership (all symbols always eligible)."""

    def __init__(self, symbols: Iterable[str]) -> None:
        self._symbols = sorted({str(s) for s in symbols})
        if not self._symbols:
            raise ValueError("universe must contain at least one symbol")

    def members(self, date: pd.Timestamp) -> list[str]:
        return list(self._symbols)

    def all_symbols(self) -> list[str]:
        return list(self._symbols)


class MembershipUniverse(Universe):
    """Membership from dated intervals.

    Parameters
    ----------
    intervals: DataFrame with columns ``symbol``, ``start``, ``end`` (NaT = still a member)
        and optionally ``known_from`` (date the information became available).
    """

    def __init__(self, intervals: pd.DataFrame) -> None:
        need = {"symbol", "start", "end"}
        if not need.issubset(intervals.columns):
            raise ValueError(f"intervals need columns {sorted(need)}")
        df = intervals.copy()
        df["symbol"] = df["symbol"].astype(str)
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
        if "known_from" in df.columns:
            df["known_from"] = pd.to_datetime(df["known_from"]).fillna(df["start"])
        else:
            df["known_from"] = df["start"]
        bad = df["end"].notna() & (df["end"] < df["start"])
        if bad.any():
            raise ValueError("membership interval with end < start")
        df["effective"] = df[["start", "known_from"]].max(axis=1)
        self._df = df.sort_values(["symbol", "start"]).reset_index(drop=True)
        self._cache: dict[pd.Timestamp, list[str]] = {}

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> MembershipUniverse:
        """Build from dated events with columns ``date``, ``symbol``, ``in_index`` (bool)."""
        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"])
        ev = ev.sort_values(["symbol", "date"])
        rows = []
        for sym, g in ev.groupby("symbol"):
            start = None
            for _, r in g.iterrows():
                if bool(r["in_index"]) and start is None:
                    start = r["date"]
                elif not bool(r["in_index"]) and start is not None:
                    rows.append({"symbol": sym, "start": start,
                                 "end": r["date"] - pd.Timedelta(days=1)})
                    start = None
            if start is not None:
                rows.append({"symbol": sym, "start": start, "end": pd.NaT})
        return cls(pd.DataFrame(rows, columns=["symbol", "start", "end"]))

    def members(self, date: pd.Timestamp) -> list[str]:
        date = pd.Timestamp(date)
        if date in self._cache:
            return list(self._cache[date])
        d = self._df
        m = (d["effective"] <= date) & (d["end"].isna() | (d["end"] >= date))
        out = sorted(set(d.loc[m, "symbol"]))
        self._cache[date] = out
        return list(out)

    def all_symbols(self) -> list[str]:
        return sorted(set(self._df["symbol"]))

    @property
    def intervals(self) -> pd.DataFrame:
        return self._df.copy()
