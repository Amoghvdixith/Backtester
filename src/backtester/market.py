"""Optional market-data download and cache (yfinance).  Never used by tests.

* Provider: Yahoo Finance via the unofficial ``yfinance`` package.  Data are for
  personal research; check Yahoo's terms before redistributing cached files.
* Adjustment: ``auto_adjust=True`` (OHLC adjusted for splits and dividends;
  volume as returned by the provider).  This is the adjusted-price research
  approximation documented in docs/execution_and_accounting.md.
* Missing data: no forward fill.  A symbol-date with any missing field is dropped
  (it becomes a missing bar).  Small OHLC inconsistencies created by price
  adjustment/rounding (<= ``repair_tol``) are repaired by widening high/low and
  counted; larger violations drop the row and are counted.
* Failure policy: if the download fails, an exception is raised.  Synthetic data
  are NEVER substituted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import MarketData


def download(symbols: list[str], start: str, end: str, directory: str | Path, name: str,
             currency: str = "USD", repair_tol: float = 0.005) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("yfinance is not installed: pip install -e .[market]") from exc
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False,
                      group_by="ticker", threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("download returned no data (network or provider failure)")
    rows, report = [], {"repaired_rows": 0, "dropped_rows": 0, "missing_rows": 0}
    for s in symbols:
        if s not in raw.columns.get_level_values(0):
            raise RuntimeError(f"no data returned for {s}")
        df = raw[s].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna(how="all")
        bad = df.isna().any(axis=1)
        report["missing_rows"] += int(bad.sum())
        df = df[~bad].copy()
        hi_need = df[["open", "close"]].max(axis=1)
        lo_need = df[["open", "close"]].min(axis=1)
        viol = (df["high"] < hi_need) | (df["low"] > lo_need)
        small = viol & (np.maximum(hi_need / df["high"] - 1, 1 - lo_need / df["low"]) <= repair_tol)
        df.loc[small, "high"] = np.maximum(df.loc[small, "high"], hi_need[small])
        df.loc[small, "low"] = np.minimum(df.loc[small, "low"], lo_need[small])
        report["repaired_rows"] += int(small.sum())
        big = viol & ~small
        report["dropped_rows"] += int(big.sum())
        df = df[~big & (df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        df["symbol"] = s
        rows.append(df)
    long = pd.concat(rows, ignore_index=True)[["date", "symbol", "open", "high", "low",
                                               "close", "volume"]]
    long = long.sort_values(["date", "symbol"]).reset_index(drop=True)
    MarketData.from_long(long)  # validate before writing
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    long.to_csv(d / f"{name}.csv", index=False, float_format="%.6f")
    meta = {
        "data_mode": "market", "provider": "Yahoo Finance via yfinance",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "symbols": symbols, "requested_period": [start, end],
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "currency": currency, "adjustment_convention": "auto_adjust=True (split+dividend "
        "adjusted OHLC); research approximation, not raw tradable prices",
        "missing_data_policy": "no forward fill; incomplete rows dropped (missing bars)",
        "ohlc_repair_tolerance": repair_tol, "cleaning_report": report,
        "universe_note": "current ETF tickers chosen today: NOT a point-in-time universe; "
                         "survivorship/selection bias is not addressed",
        "usage_note": "for personal research; check provider terms before redistribution",
        "n_rows": int(len(long)),
    }
    (d / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
