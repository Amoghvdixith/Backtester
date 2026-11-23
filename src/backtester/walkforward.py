"""Walk-forward parameter selection, trial logging and the protected hold-out.

Design
------
* The protected hold-out (``holdout_start`` onward) is removed physically from the
  data passed to any search (:func:`research_data`).  ``WalkForward`` refuses
  data that reaches into the hold-out.
* Folds are chronological: train [a, a + train_years), test [a + train_years,
  a + train_years + test_years), rolled forward by ``step_years``.  The last
  test window is truncated at the session before the hold-out.
* Training trials: one backtest per (fold, parameter set) on data up to the
  training end, trading from the training start (earlier data are warm-up only;
  the first fold has no earlier data, so long indicators consume part of it).
  Every trial is logged, including invalid parameter sets and errors.
* Selection: best training objective; ties broken by the parameter tuple in
  ascending order (deterministic).  Non-finite objectives never win.
* Out-of-sample: ONE continuous backtest over all test windows with the selected
  parameters switched at fold boundaries (:class:`ScheduledStrategy`).  The
  portfolio carries across boundaries; transition trades pay costs; there are
  no free resets and no overlapping test dates.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .data import MarketData, PointInTimeView
from .events import MarketEvent, SignalEvent
from .performance import cagr, compute_metrics, max_drawdown, sharpe_ratio
from .results import BacktestResult
from .statistics import (
    average_offdiag_correlation,
    deflated_sharpe_ratio,
    implied_independent_trials,
)
from .strategy import Strategy

StrategyFactory = Callable[[dict[str, Any]], Strategy]
RunFactory = Callable[[MarketData, Strategy, pd.Timestamp], BacktestResult]


class HoldoutViolation(RuntimeError):
    """Raised when research code could touch the protected hold-out."""


# ---------------------------------------------------------------------- hold-out
def research_data(data: MarketData, holdout_start: str | pd.Timestamp) -> MarketData:
    """Return data strictly before the hold-out (the only data searches may see)."""
    hs = pd.Timestamp(holdout_start)
    if data.index[0] >= hs:
        raise HoldoutViolation("no research data before the hold-out start")
    out = data.slice(end=hs - pd.Timedelta(days=1))
    out.provenance["holdout_start_excluded"] = str(hs.date())
    return out


def check_holdout_lock(lock_path: str | Path, holdout_start: str | pd.Timestamp,
                       dataset: str) -> dict[str, Any]:
    """Create or verify the hold-out lock file.

    The first research run writes the lock.  Any later run with a different hold-out
    start for the same dataset raises :class:`HoldoutViolation`.
    """
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    locks: dict[str, Any] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    hs = str(pd.Timestamp(holdout_start).date())
    if dataset in locks:
        if locks[dataset]["holdout_start"] != hs:
            raise HoldoutViolation(
                f"hold-out for {dataset!r} is locked at {locks[dataset]['holdout_start']}, "
                f"config asks for {hs}")
        return locks[dataset]
    locks[dataset] = {"holdout_start": hs,
                      "locked_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    p.write_text(json.dumps(locks, indent=2), encoding="utf-8")
    return locks[dataset]


def holdout_run_count(log_path: str | Path, dataset: str) -> int:
    p = Path(log_path)
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("dataset") == dataset:
            n += 1
    return n


def evaluate_holdout(data: MarketData, holdout_start: str | pd.Timestamp,
                     strategy: Strategy, run_factory: RunFactory, *, confirm: bool,
                     log_path: str | Path, dataset: str,
                     extra: dict[str, Any] | None = None) -> BacktestResult:
    """Evaluate frozen parameters on the hold-out.  Requires ``confirm=True``.

    The run uses earlier data only as warm-up and trades from ``holdout_start``.
    Every evaluation is appended to ``log_path`` (JSON lines).
    """
    if not confirm:
        raise HoldoutViolation("hold-out evaluation requires explicit confirmation")
    hs = pd.Timestamp(holdout_start)
    prior = holdout_run_count(log_path, dataset)
    res = run_factory(data, strategy, hs)
    r = res.returns()
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset, "holdout_start": str(hs.date()),
        "data_end": str(data.index[-1].date()), "strategy": strategy.describe(),
        "evaluation_number": prior + 1, "sharpe": sharpe_ratio(r), "cagr": cagr(r),
        "max_drawdown": max_drawdown(r), **(extra or {}),
    }
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    res.meta["holdout_evaluation_number"] = prior + 1
    res.meta["sample"] = "HOLD-OUT"
    return res


# ---------------------------------------------------------------------- folds
@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(index: pd.DatetimeIndex, train_years: int, test_years: int,
               step_years: int | None = None, start: pd.Timestamp | None = None
               ) -> list[Fold]:
    """Rolling folds snapped to sessions in ``index``; last test truncated at index end."""
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years and test_years must be >= 1")
    step = step_years or test_years
    a = pd.Timestamp(start) if start is not None else index[0]
    end = index[-1]
    folds: list[Fold] = []
    k = 0
    while True:
        tr_s = a + pd.DateOffset(years=k * step)
        te_s = tr_s + pd.DateOffset(years=train_years)
        te_e = te_s + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if te_s > end:
            break
        sess = lambda lo, hi: index[(index >= lo) & (index <= hi)]  # noqa: E731
        tr = sess(tr_s, te_s - pd.Timedelta(days=1))
        te = sess(te_s, min(te_e, end))
        if len(tr) == 0 or len(te) == 0:
            break
        folds.append(Fold(k, tr[0], tr[-1], te[0], te[-1]))
        k += 1
    if not folds:
        raise ValueError("sample too short for the requested walk-forward windows")
    for f1, f2 in itertools.pairwise(folds):
        if f2.test_start <= f1.test_end:
            raise AssertionError("overlapping test windows")  # pragma: no cover
    return folds


def param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product in deterministic (sorted-key, listed-value) order."""
    keys = sorted(grid)
    return [dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(grid[k] for k in keys))]


def _param_key(p: dict[str, Any]) -> tuple:
    return tuple((k, p[k]) for k in sorted(p))


# ---------------------------------------------------------------------- scheduled strategy
class ScheduledStrategy(Strategy):
    """Delegates to the strategy whose segment contains the NEXT session.

    Decisions at the close of t execute at the open of t+1, so the parameters in
    force for t+1 make the decision.  A segment with ``None`` targets cash.
    """

    name = "walk_forward_schedule"

    def __init__(self, segments: list[tuple[pd.Timestamp, Strategy | None]]) -> None:
        segs = sorted(segments, key=lambda s: s[0])
        super().__init__(segments=[(str(t.date()), s.describe() if s else None)
                                   for t, s in segs])
        self._segs = segs

    def bind(self, view: PointInTimeView) -> None:
        super().bind(view)
        for _, s in self._segs:
            if s is not None:
                s.bind(view)

    def reset(self) -> None:
        self._active: int | None = None

    def on_bar(self, event: MarketEvent) -> list[SignalEvent] | None:
        nxt = self.view.next_session() or event.timestamp
        idx = None
        for k, (t, _) in enumerate(self._segs):
            if t <= nxt:
                idx = k
        if idx is None:
            return None
        switched = idx != self._active
        self._active = idx
        strat = self._segs[idx][1]
        if strat is None:
            return [] if switched else None
        return strat.on_bar(event)


# ---------------------------------------------------------------------- trial registry
class TrialRegistry:
    """Append-only JSON-lines log of every trial executed through the framework.

    It cannot know about experiments run outside the framework (e.g. ad-hoc
    notebooks that bypass it); the trial count is a lower bound in that sense.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, trials: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for rec in trials.to_dict(orient="records"):
                fh.write(json.dumps(rec, default=str) + "\n")

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    trials: pd.DataFrame
    oos: BacktestResult
    oos_returns: pd.Series
    fold_metrics: pd.DataFrame
    dsr: pd.DataFrame
    trial_returns: dict[int, pd.Series] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


class WalkForward:
    def __init__(self, data: MarketData, strategy_factory: StrategyFactory,
                 grid: dict[str, list[Any]], run_factory: RunFactory, *,
                 train_years: int = 3, test_years: int = 1, step_years: int | None = None,
                 objective: str = "sharpe", holdout_start: str | pd.Timestamp | None = None,
                 experiment: str = "walkforward", min_train_obs: int = 126,
                 periods_per_year: int = 252) -> None:
        if holdout_start is not None and data.index[-1] >= pd.Timestamp(holdout_start):
            raise HoldoutViolation(
                "walk-forward data reach into the protected hold-out; use research_data()")
        if objective not in ("sharpe", "cagr", "calmar"):
            raise ValueError("objective must be 'sharpe', 'cagr' or 'calmar'")
        self.data = data
        self.strategy_factory = strategy_factory
        self.grid = grid
        self.params = param_grid(grid)
        self.run_factory = run_factory
        self.train_years, self.test_years = train_years, test_years
        self.step_years = step_years or test_years
        self.objective = objective
        self.holdout_start = holdout_start
        self.experiment = experiment
        self.min_train_obs = min_train_obs
        self.ppy = periods_per_year

    def _score(self, r: pd.Series) -> float:
        if len(r) < self.min_train_obs:
            return float("nan")
        if self.objective == "sharpe":
            return sharpe_ratio(r, periods_per_year=self.ppy)
        if self.objective == "cagr":
            return cagr(r, self.ppy)
        mdd = max_drawdown(r)
        return cagr(r, self.ppy) / mdd if mdd > 0 else float("nan")

    def run(self, progress: Callable[[str], None] | None = None) -> WalkForwardResult:
        t0 = time.perf_counter()
        folds = make_folds(self.data.index, self.train_years, self.test_years, self.step_years)
        trials: list[dict[str, Any]] = []
        trial_returns: dict[int, pd.Series] = {}
        selected: dict[int, dict[str, Any] | None] = {}
        tid = 0
        for f in folds:
            train_data = self.data.slice(end=f.train_end)
            best: tuple[float, tuple, dict[str, Any]] | None = None
            for p in self.params:
                tid += 1
                rec: dict[str, Any] = {
                    "trial_id": tid, "experiment": self.experiment, "fold": f.fold,
                    "params": json.dumps(p, sort_keys=True),
                    "data_start": str(train_data.index[0].date()),
                    "train_start": str(f.train_start.date()),
                    "train_end": str(f.train_end.date()),
                    "objective": self.objective, "sample": "training",
                    "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                t1 = time.perf_counter()
                try:
                    strat = self.strategy_factory(p)
                except ValueError as exc:
                    rec.update(status="invalid", error=str(exc))
                    trials.append(rec)
                    continue
                try:
                    res = self.run_factory(train_data, strat, f.train_start)
                    r = res.returns(f.train_start, f.train_end)
                    score = self._score(r)
                    m = compute_metrics(r)
                    rec.update(status="ok" if math.isfinite(score) else "undefined_objective",
                               objective_value=score, sharpe=m["sharpe"],
                               sharpe_per_period=m["sharpe_per_period"], cagr=m["cagr"],
                               max_drawdown=m["max_drawdown"], n_obs=m["n_periods"],
                               skew=m["skew"], kurtosis_pearson=m["kurtosis_pearson"],
                               n_round_trips=len(res.round_trips))
                    trial_returns[tid] = r
                    if math.isfinite(score):
                        key = _param_key(p)
                        if (best is None or score > best[0]
                                or (score == best[0] and key < best[1])):
                            best = (score, key, p)
                except Exception as exc:  # noqa: BLE001 - logged, never hidden
                    rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                rec["runtime_seconds"] = round(time.perf_counter() - t1, 3)
                trials.append(rec)
            selected[f.fold] = best[2] if best else None
            if progress:
                progress(f"fold {f.fold}: selected {selected[f.fold]}")
        tdf = pd.DataFrame(trials)
        tdf["selected"] = False
        for f in folds:
            chosen = selected[f.fold]
            if chosen is not None:
                mask = ((tdf["fold"] == f.fold)
                        & (tdf["params"] == json.dumps(chosen, sort_keys=True)))
                tdf.loc[mask, "selected"] = True

        segments: list[tuple[pd.Timestamp, Strategy | None]] = []
        for f in folds:
            chosen = selected[f.fold]
            segments.append((f.test_start,
                             self.strategy_factory(chosen) if chosen is not None else None))
        oos = self.run_factory(self.data, ScheduledStrategy(segments), folds[0].test_start)
        oos.meta["sample"] = "walk-forward out-of-sample (stitched)"
        oos_r = oos.returns(folds[0].test_start, folds[-1].test_end)
        fold_rows = []
        for f in folds:
            r = oos_r[(oos_r.index >= f.test_start) & (oos_r.index <= f.test_end)]
            m = compute_metrics(r)
            fold_rows.append({
                "fold": f.fold, "train_start": f.train_start.date(),
                "train_end": f.train_end.date(), "test_start": f.test_start.date(),
                "test_end": f.test_end.date(), "selected_params": json.dumps(selected[f.fold]),
                "train_objective": float(tdf.loc[(tdf["fold"] == f.fold) & tdf["selected"],
                                                 "objective_value"].max())
                if selected[f.fold] else float("nan"),
                "test_return": m["total_return"], "test_sharpe": m["sharpe"],
                "test_max_drawdown": m["max_drawdown"], "test_n_obs": m["n_periods"],
            })
        fold_df = pd.DataFrame(fold_rows)
        dsr_df = self._dsr_table(tdf, trial_returns, folds, oos_r)
        folds_df = pd.DataFrame([f.__dict__ for f in folds])
        meta = {"experiment": self.experiment, "grid": self.grid,
                "train_years": self.train_years, "test_years": self.test_years,
                "step_years": self.step_years, "objective": self.objective,
                "holdout_start": str(self.holdout_start) if self.holdout_start else None,
                "research_data_end": str(self.data.index[-1].date()),
                "n_trials_total": int(len(tdf)),
                "n_trials_invalid": int((tdf["status"] != "ok").sum()),
                "runtime_seconds": round(time.perf_counter() - t0, 2)}
        return WalkForwardResult(folds_df, tdf, oos, oos_r, fold_df, dsr_df, trial_returns, meta)

    def _dsr_table(self, tdf: pd.DataFrame, trial_returns: dict[int, pd.Series],
                   folds: list[Fold], oos_r: pd.Series) -> pd.DataFrame:
        rows = []
        for f in folds:
            ft = tdf[(tdf["fold"] == f.fold)]
            ok = ft[ft["status"] == "ok"]
            sel = ok[ok["selected"]]
            if sel.empty:
                continue
            s = sel.iloc[0]
            rets = pd.DataFrame({t: trial_returns[t] for t in ok["trial_id"]
                                 if t in trial_returns})
            rho = average_offdiag_correlation(rets) if rets.shape[1] > 1 else float("nan")
            m_raw = len(ft)  # all attempted configurations, including invalid ones
            n_eff = implied_independent_trials(len(ok), rho) if math.isfinite(rho) else len(ok)
            for label, n_tr in (("raw_attempted", m_raw), ("implied_independent", n_eff)):
                d = deflated_sharpe_ratio(float(s["sharpe_per_period"]),
                                          ok["sharpe_per_period"].tolist(), int(s["n_obs"]),
                                          float(s["skew"]), float(s["kurtosis_pearson"]),
                                          n_trials=n_tr)
                rows.append({"scope": f"fold {f.fold} training selection", "n_trials_basis": label,
                             "n_trials": n_tr, "avg_trial_corr": rho,
                             "sr_annualised": float(s["sharpe"]),
                             "sr0_annualised": d["sr0_per_period"] * math.sqrt(self.ppy),
                             "dsr": d["dsr"], "psr_vs_zero": d["psr_vs_zero"],
                             "n_obs": d["n_obs"], "warnings": "; ".join(d["warnings"])})
        # conservative diagnostic for the stitched OOS track (see docs)
        ok = tdf[tdf["status"] == "ok"]
        m = compute_metrics(oos_r)
        if len(oos_r) > 30 and len(ok) > 1:
            d = deflated_sharpe_ratio(m["sharpe_per_period"], ok["sharpe_per_period"].tolist(),
                                      m["n_periods"], m["skew"], m["kurtosis_pearson"],
                                      n_trials=len(tdf))
            rows.append({"scope": "stitched OOS (conservative diagnostic)",
                         "n_trials_basis": "all attempted trials, all folds",
                         "n_trials": len(tdf), "avg_trial_corr": float("nan"),
                         "sr_annualised": m["sharpe"],
                         "sr0_annualised": d["sr0_per_period"] * math.sqrt(self.ppy),
                         "dsr": d["dsr"], "psr_vs_zero": d["psr_vs_zero"],
                         "n_obs": d["n_obs"], "warnings": "; ".join(d["warnings"])})
        return pd.DataFrame(rows)
