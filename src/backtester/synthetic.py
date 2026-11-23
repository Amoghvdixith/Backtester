"""Reproducible SYNTHETIC multi-asset OHLCV generator.

SYNTHETIC DATA - NOT MARKET DATA.  Any strategy performance on these series
demonstrates software behaviour only, not an investment edge.

Model (daily, business-day calendar):

* A market factor follows a 3-state Markov regime model
  (calm: +9%/yr drift, 12% vol; stressed: -20%/yr, 30% vol; sideways: 0%/yr, 16% vol).
* Asset i log return = beta_i * market + alpha_i(t) + idiosyncratic noise, where
  alpha_i(t) is a slow Ornstein-Uhlenbeck drift (annualised sd ~6%, half-life ~1y).
  This persistence is a modelling ASSUMPTION; it can make trend/momentum rules look
  better than they would on random-walk data.  It is disclosed, not hidden.
* Close-to-close return is split into an overnight gap and an intraday move:
  open_t = close_{t-1} * exp(gap), close_t = open_t * exp(intraday).
  high/low extend beyond max/min(open, close) by half-normal amounts, so
  low <= open, close <= high always holds.
* Volume is log-normal around an asset-specific level and rises with |return|.
* Optional events: late listings (data start later), delistings (data stop; some
  after a crash), and short trading halts (missing bars).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REGIMES = {  # name: (annual drift, annual vol)
    "calm": (0.09, 0.12),
    "stressed": (-0.20, 0.30),
    "sideways": (0.00, 0.16),
}
TRANSITIONS = np.array([  # daily transition probabilities (rows sum to 1)
    [0.995, 0.002, 0.003],
    [0.010, 0.980, 0.010],
    [0.004, 0.002, 0.994],
])


@dataclass(frozen=True)
class SyntheticSpec:
    n_assets: int = 30
    start: str = "2008-01-01"
    end: str = "2024-12-31"
    seed: int = 42
    prefix: str = "SYN"
    late_listing_frac: float = 0.15
    delist_frac: float = 0.10
    crash_delist_frac: float = 0.5  # share of delistings preceded by a crash
    halt_prob: float = 0.0005  # per asset-day probability that a halt starts
    alpha_ann_sd: float = 0.06
    alpha_halflife_days: float = 252.0


def generate(spec: SyntheticSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (long OHLCV frame, membership intervals, regime series)."""
    rng = np.random.default_rng(spec.seed)
    dates = pd.bdate_range(spec.start, spec.end)
    T, N = len(dates), spec.n_assets
    if T < 300:
        raise ValueError("need at least ~300 sessions")
    names = list(REGIMES)
    state = np.empty(T, dtype=int)
    state[0] = 0
    for t in range(1, T):
        state[t] = rng.choice(3, p=TRANSITIONS[state[t - 1]])
    drift = np.array([REGIMES[n][0] for n in names])[state] / 252
    vol = np.array([REGIMES[n][1] for n in names])[state] / np.sqrt(252)
    mkt = drift - 0.5 * vol ** 2 + vol * rng.standard_normal(T)

    beta = rng.uniform(0.6, 1.4, N)
    idio_vol = rng.uniform(0.15, 0.40, N) / np.sqrt(252)
    phi = 0.5 ** (1 / spec.alpha_halflife_days)
    a_sd = spec.alpha_ann_sd / 252
    alpha = np.zeros((T, N))
    alpha[0] = rng.normal(0, a_sd, N)
    eps_sd = a_sd * np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        alpha[t] = phi * alpha[t - 1] + rng.normal(0, eps_sd, N)
    ret = beta[None, :] * mkt[:, None] + alpha + idio_vol[None, :] * rng.standard_normal((T, N))
    ret -= 0.5 * (idio_vol ** 2)[None, :]

    # lifecycle: late listings and delistings
    first = np.zeros(N, dtype=int)
    last = np.full(N, T - 1, dtype=int)
    n_late = int(round(spec.late_listing_frac * N))
    n_del = int(round(spec.delist_frac * N))
    order = rng.permutation(N)
    for i in order[:n_late]:
        first[i] = int(rng.integers(T // 10, T // 2))
    for k, i in enumerate(order[n_late:n_late + n_del]):
        last[i] = int(rng.integers(T // 2, T - 60))
        if k < int(round(spec.crash_delist_frac * n_del)):
            # crash over the final 40 sessions before delisting (~ -80%)
            ret[last[i] - 39: last[i] + 1, i] += np.log(0.2) / 40

    gap_share = rng.uniform(0.2, 0.4, N)
    gap = ret * gap_share[None, :] + rng.normal(0, 0.002, (T, N))
    intraday = ret - gap
    log_close = np.cumsum(ret, axis=0) + np.log(rng.uniform(20, 200, N))[None, :]
    close = np.exp(log_close)
    prev_close = np.vstack([close[0] / np.exp(ret[0]), close[:-1]])
    open_ = prev_close * np.exp(gap)
    close = open_ * np.exp(intraday)
    rng_hl = np.abs(rng.normal(0, 1, (T, N))) * (idio_vol[None, :] * 0.6)
    high = np.maximum(open_, close) * np.exp(rng_hl)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 1, (T, N)))
                                            * idio_vol[None, :] * 0.6)
    base_vol = rng.uniform(2e5, 5e6, N)
    volume = np.round(base_vol[None, :] * np.exp(0.3 * rng.standard_normal((T, N)))
                      * (1 + 20 * np.abs(ret)))

    present = np.zeros((T, N), dtype=bool)
    for i in range(N):
        present[first[i]: last[i] + 1, i] = True
    halts = rng.random((T, N)) < spec.halt_prob
    for t, i in zip(*np.nonzero(halts), strict=True):
        present[t: t + int(rng.integers(1, 4)), i] = False
    for i in range(N):  # keep listing/delisting bars so lifecycle dates are exact
        present[first[i], i] = True
        present[last[i], i] = True
        present[last[i] + 1:, i] = False

    syms = [f"{spec.prefix}{k:03d}" for k in range(N)]
    rows = []
    for i, s in enumerate(syms):
        m = present[:, i]
        rows.append(pd.DataFrame({
            "date": dates[m], "symbol": s, "open": open_[m, i], "high": high[m, i],
            "low": low[m, i], "close": close[m, i], "volume": volume[m, i],
        }))
    long = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    long = long.reset_index(drop=True)
    membership = pd.DataFrame({
        "symbol": syms,
        "start": dates[first],
        "end": [dates[last[i]] if last[i] < T - 1 else pd.NaT for i in range(N)],
    })
    # membership becomes known on the listing date itself (no back-dating)
    membership["known_from"] = membership["start"]
    regimes = pd.Series(np.array(names)[state], index=dates, name="regime")
    return long, membership, regimes


def write_dataset(spec: SyntheticSpec, directory: str | Path, name: str) -> dict[str, str]:
    """Generate and write ``<name>.csv``, ``<name>_membership.csv`` and metadata."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    long, mem, regimes = generate(spec)
    long.to_csv(d / f"{name}.csv", index=False, float_format="%.6f")
    mem.to_csv(d / f"{name}_membership.csv", index=False)
    regimes.to_csv(d / f"{name}_regimes.csv")
    meta = {
        "data_mode": "synthetic",
        "label": "SYNTHETIC DATA - NOT MARKET DATA",
        "generator": "backtester.synthetic.generate",
        "spec": asdict(spec),
        "regimes": {k: {"annual_drift": v[0], "annual_vol": v[1]} for k, v in REGIMES.items()},
        "calendar": "pandas business days (Mon-Fri), no holidays",
        "timezone": "tz-naive session dates",
        "adjustment_convention": "no corporate actions are simulated; prices are total-return-like",
        "currency": "synthetic units",
        "n_rows": int(len(long)),
    }
    (d / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                          encoding="utf-8")
    return {"prices": f"{name}.csv", "membership": f"{name}_membership.csv"}
