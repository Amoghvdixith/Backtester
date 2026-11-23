"""Market-data container, validation, CSV loading and the point-in-time view.

Conventions
-----------
* Index: tz-naive ``DatetimeIndex`` of exchange session dates (local exchange
  time is implied).  The *master calendar* is the union of all session dates in
  the file; it is treated as the exchange calendar.
* A **missing bar** is a session in the master calendar for which a symbol has
  no row, or all OHLCV fields are NaN.  Missing bars are never forward-filled for
  execution.  A bar is **tradable** when OHLC are finite and volume > 0.
* Prices are an *adjusted-price research approximation* (convention B in the
  docs): OHLC and volume are expected to be on the same adjustment basis
  (e.g. split- and dividend-adjusted OHLC with split-adjusted volume).  Share
  quantities are therefore in "adjusted share units"; the framework does not
  model discrete split or dividend cash events.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .events import Phase
from .universe import StaticUniverse, Universe

FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when market data violate a structural or OHLC constraint."""


class LookAheadError(RuntimeError):
    """Raised when code asks the point-in-time view for information not yet available."""


@dataclass(frozen=True)
class ExecutionEstimates:
    """Lagged liquidity/volatility estimates (computed from bars strictly before t)."""

    adv: np.ndarray  # shape (T, N), average daily volume over the lookback, lagged one session
    sigma: np.ndarray  # shape (T, N), daily log-return std over the lookback, lagged


class MarketData:
    """Validated wide-format OHLCV panel.

    Parameters
    ----------
    frames: mapping of field name to DataFrame (index = dates, columns = symbols).
    provenance: free-form metadata (source, retrieval date, adjustment convention...).
    """

    def __init__(self, frames: dict[str, pd.DataFrame], provenance: dict | None = None,
                 validate: bool = True) -> None:
        missing = [f for f in FIELDS if f not in frames]
        if missing:
            raise DataValidationError(f"missing fields: {missing}")
        close = frames["close"]
        index = close.index
        columns = list(close.columns)
        self._frames: dict[str, pd.DataFrame] = {}
        for f in FIELDS:
            df = frames[f]
            if not df.index.equals(index) or list(df.columns) != columns:
                raise DataValidationError(f"field {f!r} is not aligned with 'close'")
            self._frames[f] = df.astype(float)
        self.provenance: dict = dict(provenance or {})
        if validate:
            self._validate()
        self.index: pd.DatetimeIndex = pd.DatetimeIndex(index)
        self.symbols: list[str] = [str(c) for c in columns]
        self._sym_pos = {s: i for i, s in enumerate(self.symbols)}
        self.arrays: dict[str, np.ndarray] = {f: self._frames[f].to_numpy() for f in FIELDS}
        a = self.arrays
        self.tradable: np.ndarray = (
            np.isfinite(a["open"]) & np.isfinite(a["high"]) & np.isfinite(a["low"])
            & np.isfinite(a["close"]) & np.isfinite(a["volume"]) & (a["volume"] > 0)
        )
        self._estimates: dict[tuple[int, int], ExecutionEstimates] = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_long(cls, df: pd.DataFrame, provenance: dict | None = None) -> MarketData:
        """Build from a long table with columns date, symbol, open, high, low, close, volume."""
        required = {"date", "symbol", *FIELDS}
        if not required.issubset(df.columns):
            raise DataValidationError(f"long data needs columns {sorted(required)}")
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        if d["date"].dt.tz is not None:
            raise DataValidationError("timestamps must be tz-naive session dates")
        d["symbol"] = d["symbol"].astype(str)
        dup = d.duplicated(["date", "symbol"])
        if dup.any():
            first = d.loc[dup, ["date", "symbol"]].iloc[0].tolist()
            raise DataValidationError(f"duplicate symbol-date record, e.g. {first}")
        for sym, g in d.groupby("symbol", sort=False):
            if not g["date"].is_monotonic_increasing:
                raise DataValidationError(f"dates for {sym} are not sorted ascending")
        frames = {
            f: d.pivot(index="date", columns="symbol", values=f).sort_index() for f in FIELDS
        }
        cols = sorted(frames["close"].columns)
        frames = {f: v.reindex(columns=cols) for f, v in frames.items()}
        return cls(frames, provenance)

    @classmethod
    def from_csv(cls, path: str | Path, provenance: dict | None = None) -> MarketData:
        """Load a long-format CSV (date, symbol, open, high, low, close, volume)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        prov = {"source_file": p.name}
        meta = p.with_suffix(".meta.json")
        if meta.exists():
            import json

            prov.update(json.loads(meta.read_text(encoding="utf-8")))
        prov.update(provenance or {})
        return cls.from_long(df, prov)

    def to_long(self) -> pd.DataFrame:
        """Long-format table of observed (non-missing) rows."""
        parts = []
        for f in FIELDS:
            df = self._frames[f]
            s = pd.Series(
                df.to_numpy().ravel(),
                index=pd.MultiIndex.from_product([df.index, df.columns]),
                name=f,
            )
            parts.append(s)
        out = pd.concat(parts, axis=1)
        out.index.names = ["date", "symbol"]
        out = out.dropna(how="all").reset_index()
        return out

    # ------------------------------------------------------------------ validation
    def _validate(self) -> None:
        idx = self._frames["close"].index
        if not isinstance(idx, pd.DatetimeIndex):
            raise DataValidationError("index must be a DatetimeIndex")
        if idx.tz is not None:
            raise DataValidationError("index must be tz-naive session dates")
        if not idx.is_monotonic_increasing:
            raise DataValidationError("timestamps must be sorted ascending")
        if idx.has_duplicates:
            raise DataValidationError("duplicate timestamps")
        if self._frames["close"].columns.has_duplicates:
            raise DataValidationError("duplicate symbols")
        o, h, low, c, v = (self._frames[f].to_numpy() for f in FIELDS)
        obs = np.isfinite(c)
        any_obs = np.isfinite(o) | np.isfinite(h) | np.isfinite(low) | obs | np.isfinite(v)
        partial = any_obs & ~(np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & obs
                              & np.isfinite(v))
        if partial.any():
            raise DataValidationError("bars with some but not all OHLCV fields present")
        with np.errstate(invalid="ignore"):
            if np.any(o[obs] <= 0) or np.any(low[obs] <= 0) or np.any(c[obs] <= 0):
                raise DataValidationError("prices must be strictly positive")
            if np.any(v[obs] < 0):
                raise DataValidationError("volumes must be non-negative")
            tol = 1e-9
            hi_ok = h[obs] >= np.maximum(o[obs], c[obs]) * (1 - tol)
            lo_ok = low[obs] <= np.minimum(o[obs], c[obs]) * (1 + tol)
            if not (hi_ok.all() and lo_ok.all()):
                raise DataValidationError("OHLC inconsistency: need low <= open,close <= high")

    # ------------------------------------------------------------------ helpers
    def frame(self, field: str) -> pd.DataFrame:
        if field not in self._frames:
            raise KeyError(field)
        return self._frames[field]

    def sym_index(self, symbol: str) -> int:
        try:
            return self._sym_pos[symbol]
        except KeyError as exc:
            raise KeyError(f"unknown symbol {symbol!r}") from exc

    def slice(self, start: pd.Timestamp | str | None = None,
              end: pd.Timestamp | str | None = None) -> MarketData:
        """Return a new MarketData restricted to ``start <= date <= end``.

        Used to *physically remove* the protected hold-out from research data.
        """
        mask = np.ones(len(self.index), dtype=bool)
        if start is not None:
            mask &= self.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.index <= pd.Timestamp(end)
        frames = {f: self._frames[f].loc[mask] for f in FIELDS}
        prov = dict(self.provenance)
        prov["slice"] = [str(start), str(end)]
        return MarketData(frames, prov, validate=False)

    def select(self, symbols: Iterable[str]) -> MarketData:
        syms = list(symbols)
        frames = {f: self._frames[f][syms] for f in FIELDS}
        return MarketData(frames, dict(self.provenance), validate=False)

    def estimates(self, adv_window: int = 20, vol_window: int = 20) -> ExecutionEstimates:
        """Lagged ADV and daily volatility per (session, symbol).

        Value at row t uses only bars strictly before session t (no same-day volume
        or close), so it is legitimately known at the open of t.
        """
        key = (adv_window, vol_window)
        if key in self._estimates:
            return self._estimates[key]
        adv = pd.DataFrame(index=self.index, columns=self.symbols, dtype=float)
        sig = pd.DataFrame(index=self.index, columns=self.symbols, dtype=float)
        vol_f, close_f = self._frames["volume"], self._frames["close"]
        for s in self.symbols:
            obs = close_f[s].notna()
            v = vol_f.loc[obs, s]
            c = close_f.loc[obs, s]
            a = v.rolling(adv_window, min_periods=max(2, adv_window // 2)).mean()
            r = np.log(c).diff()
            sd = r.rolling(vol_window, min_periods=max(2, vol_window // 2)).std(ddof=1)
            # value known after the close of each observed bar; usable from the next session
            adv[s] = a.reindex(self.index).ffill().shift(1)
            sig[s] = sd.reindex(self.index).ffill().shift(1)
        est = ExecutionEstimates(adv=adv.to_numpy(), sigma=sig.to_numpy())
        self._estimates[key] = est
        return est

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return (f"MarketData({len(self.symbols)} symbols, {len(self.index)} sessions, "
                f"{self.index[0].date()}..{self.index[-1].date()})")


class PointInTimeView:
    """The only market-data interface a strategy or sizer receives.

    It exposes data that would be known at the current (session, phase):

    * ``Phase.CLOSE`` at row ``i``: all fields of rows ``0..i``.
    * ``Phase.OPEN``  at row ``i``: all fields of rows ``0..i-1`` plus the ``open`` of row ``i``.

    Python cannot make attributes truly private; the guarantee is enforced by this
    API and by the look-ahead tests, not by the language.
    """

    def __init__(self, data: MarketData, universe: Universe | None = None) -> None:
        self.__data = data
        self.__universe = universe or StaticUniverse(data.symbols)
        self._i = -1
        self._phase = Phase.CLOSE

    # engine-only ----------------------------------------------------------------
    def _advance(self, i: int, phase: Phase) -> None:
        self._i = i
        self._phase = phase

    # public ----------------------------------------------------------------------
    @property
    def now(self) -> pd.Timestamp:
        if self._i < 0:
            raise LookAheadError("view has not been advanced yet")
        return self.__data.index[self._i]

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def position(self) -> int:
        """Number of sessions elapsed minus one (row index of the current session)."""
        return self._i

    @property
    def all_symbols(self) -> list[str]:
        return list(self.__data.symbols)

    def universe(self) -> list[str]:
        """Point-in-time members known at the current session."""
        return self.__universe.members(self.now)

    def _rows_visible(self, field: str) -> int:
        if field not in FIELDS:
            raise KeyError(f"unknown field {field!r}")
        if self._phase is Phase.CLOSE:
            return self._i + 1
        return self._i + 1 if field == "open" else self._i

    def history(self, symbol: str, field: str = "close", n: int | None = None,
                include_missing: bool = False) -> np.ndarray:
        """Last ``n`` *observed* values of ``field`` (oldest first).

        Missing bars are skipped (never forward-filled) unless ``include_missing``
        is True, in which case the last ``n`` calendar sessions are returned with NaN.
        """
        rows = self._rows_visible(field)
        j = self.__data.sym_index(symbol)
        col = self.__data.arrays[field][:rows, j]
        if include_missing:
            return col[-n:].copy() if n else col.copy()
        obs = col[np.isfinite(col)]
        if n is not None:
            if n <= 0:
                raise ValueError("n must be positive")
            obs = obs[-n:]
        return obs.copy()

    def value_at_offset(self, symbol: str, offset: int, field: str = "close",
                        max_stale: int = 0) -> float:
        """Value ``offset`` calendar sessions before the latest visible session.

        If that session is missing for the symbol, the most recent observation at most
        ``max_stale`` sessions earlier is used; otherwise NaN.
        """
        rows = self._rows_visible(field)
        k = rows - 1 - offset
        if k < 0:
            return float("nan")
        col = self.__data.arrays[field][:, self.__data.sym_index(symbol)]
        for back in range(max_stale + 1):
            if k - back < 0:
                break
            val = col[k - back]
            if np.isfinite(val):
                return float(val)
        return float("nan")

    def history_frame(self, field: str = "close", n: int | None = None,
                      symbols: list[str] | None = None) -> pd.DataFrame:
        """Calendar-aligned frame (NaN for missing bars) of visible history."""
        rows = self._rows_visible(field)
        df = self.__data.frame(field).iloc[:rows]
        if symbols is not None:
            df = df[symbols]
        return df.iloc[-n:].copy() if n else df.copy()

    def calendar(self) -> pd.DatetimeIndex:
        """Session dates up to and including the current session."""
        return self.__data.index[: self._i + 1]

    def is_tradable_today(self, symbol: str) -> bool:
        """Whether the current session has a tradable bar (CLOSE phase only)."""
        if self._phase is not Phase.CLOSE:
            raise LookAheadError("full-bar tradability is unknown before the close")
        return bool(self.__data.tradable[self._i, self.__data.sym_index(symbol)])

    def next_session(self) -> pd.Timestamp | None:
        """Next exchange session date (calendar knowledge, not price knowledge)."""
        if self._i + 1 < len(self.__data.index):
            return self.__data.index[self._i + 1]
        return None

    def is_last_session_of_month(self) -> bool:
        """True if the next session falls in a different calendar month.

        For the final row of the data, the next business day is used as a proxy.
        """
        nxt = self.next_session()
        if nxt is None:
            nxt = self.now + pd.offsets.BDay(1)
        return (nxt.year, nxt.month) != (self.now.year, self.now.month)
