"""Performance and risk metrics.

Definitions (daily returns r_t of equity, n observations, P periods per year):

* total_return  = prod(1 + r_t) - 1                       (compounded)
* CAGR          = (1 + total_return)^(P / n) - 1           (geometric; undefined if equity <= 0)
* mean_arith_ann = P * mean(r)                              (arithmetic, NOT a compounded growth rate)
* ann_vol       = sqrt(P) * sd(r), sample sd (ddof = 1)
* excess x_t    = r_t - rf_t, rf_t the per-period risk-free return aligned on dates
* Sharpe        = sqrt(P) * mean(x) / sd(x)               (IID annualisation)
* Sortino       = sqrt(P) * mean(x) / DD,  DD = sqrt(mean(min(x_t, 0)^2)) over ALL n periods;
                  target = the risk-free rate.
* max_drawdown  = max_t (1 - V_t / max_{s<=t} V_s), with V_0 = 1 included
* Calmar        = CAGR / max_drawdown over the full sample (not the 36-month variant)
* skew          = sample skewness (bias-adjusted); kurtosis reported both as Pearson
                  (normal = 3) and excess (normal = 0).
* turnover      = one-way annual turnover = (sum_t traded_notional_t / equity_{t-1}) / 2 / years;
                  cash is not counted; a full replacement of a fully invested book = 100%.
* time_invested = share of sessions with gross exposure > 0 at the close.

Trade statistics come from completed FIFO round trips (lot closures), not from
daily returns.  Metrics that are not meaningful return NaN and add a message to
``warnings``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PERIODS_PER_YEAR = 252


def annual_to_periodic(rate: float | pd.Series, periods_per_year: int = PERIODS_PER_YEAR,
                       compounding: str = "simple") -> float | pd.Series:
    """Convert an annual rate to a per-period return.

    ``simple``: rate / P (money-market style, consistent with simple interest accrual).
    ``compound``: (1 + rate)^(1/P) - 1.
    """
    if compounding == "simple":
        return rate / periods_per_year
    if compounding == "compound":
        return (1 + rate) ** (1.0 / periods_per_year) - 1
    raise ValueError("compounding must be 'simple' or 'compound'")


def rf_from_annual(index: pd.DatetimeIndex, annual_rate: float,
                   day_count_basis: int = 365) -> pd.Series:
    """Per-session risk-free return consistent with the engine's cash accrual.

    rf_t = annual_rate * (date_t - date_{t-1}).days / basis  (simple interest), so a
    Monday return carries three days of interest, exactly like the cash account.
    The first session gets the median spacing.
    """
    days = pd.Series(index, index=index).diff().dt.days
    days = days.fillna(days.median() if len(days) > 1 else 1.0)
    return annual_rate * days / day_count_basis


def _align_rf(r: pd.Series, rf: float | pd.Series | None) -> pd.Series:
    if rf is None:
        return pd.Series(0.0, index=r.index)
    if isinstance(rf, pd.Series):
        out = rf.reindex(r.index)
        if out.isna().any():
            raise ValueError("risk-free series does not cover all return dates")
        return out
    return pd.Series(float(rf), index=r.index)


def wealth_index(r: pd.Series) -> pd.Series:
    return (1.0 + r).cumprod()


def drawdown_series(r: pd.Series) -> pd.Series:
    w = wealth_index(r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], w.to_numpy()]))[1:]
    return pd.Series(w.to_numpy() / peak - 1.0, index=r.index, name="drawdown")


def max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    return float(-drawdown_series(r).min())


def longest_drawdown(r: pd.Series) -> tuple[int, int]:
    """(sessions, calendar days) of the longest period below a previous peak."""
    dd = drawdown_series(r)
    best_n, best_days, start = 0, 0, None
    idx = dd.index
    for k, v in enumerate(dd.to_numpy()):
        if v < -1e-12:
            if start is None:
                start = k
            n = k - start + 1
            if n > best_n:
                best_n = n
                prev = idx[start - 1] if start > 0 else idx[start]
                best_days = int((idx[k] - prev).days)
        else:
            start = None
    return best_n, best_days


def sharpe_ratio(r: pd.Series, rf: float | pd.Series | None = None,
                 periods_per_year: int = PERIODS_PER_YEAR, annualise: bool = True) -> float:
    x = r - _align_rf(r, rf)
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if not math.isfinite(sd) or sd <= 1e-14:
        return float("nan")
    s = float(x.mean()) / sd
    return s * math.sqrt(periods_per_year) if annualise else s


def sortino_ratio(r: pd.Series, rf: float | pd.Series | None = None,
                  periods_per_year: int = PERIODS_PER_YEAR) -> float:
    x = r - _align_rf(r, rf)
    if len(x) < 2:
        return float("nan")
    dd = math.sqrt(float(np.mean(np.minimum(x.to_numpy(), 0.0) ** 2)))
    if dd <= 1e-14:
        return float("nan")
    return math.sqrt(periods_per_year) * float(x.mean()) / dd


def cagr(r: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    if r.empty:
        return float("nan")
    growth = float(np.prod(1.0 + r.to_numpy()))
    if growth <= 0:
        return float("nan")
    return growth ** (periods_per_year / len(r)) - 1.0


def period_returns(r: pd.Series, freq: str) -> pd.Series:
    """Compounded returns per calendar period ('ME' month, 'YE' year)."""
    return (1.0 + r).resample(freq).prod() - 1.0


def monthly_table(r: pd.Series) -> pd.DataFrame:
    m = period_returns(r, "ME")
    di = pd.DatetimeIndex(m.index)
    df = pd.DataFrame({"year": di.year, "month": di.month, "ret": m.to_numpy()})
    return df.pivot(index="year", columns="month", values="ret")


def trade_stats(round_trips: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {"n_round_trips": 0, "hit_rate": np.nan, "avg_win": np.nan,
                           "avg_loss": np.nan, "win_loss_ratio": np.nan,
                           "profit_factor": np.nan, "avg_holding_days": np.nan}
    if round_trips is None or round_trips.empty:
        return out
    pnl = round_trips["net_pnl"].to_numpy(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    out["n_round_trips"] = len(pnl)
    out["hit_rate"] = float(len(wins)) / float(len(pnl))
    out["avg_win"] = float(wins.mean()) if len(wins) else np.nan
    out["avg_loss"] = float(losses.mean()) if len(losses) else np.nan
    if len(wins) and len(losses):
        out["win_loss_ratio"] = out["avg_win"] / abs(out["avg_loss"])
        out["profit_factor"] = float(wins.sum()) / abs(float(losses.sum()))
    out["avg_holding_days"] = float(round_trips["holding_days"].mean())
    return out


def turnover_series(equity_frame: pd.DataFrame) -> pd.Series:
    """Daily one-way turnover = traded notional / (2 * previous-close equity)."""
    traded = equity_frame["traded_notional"].diff().fillna(equity_frame["traded_notional"])
    prev_eq = equity_frame["equity"].shift(1).fillna(equity_frame["equity"])
    return (traded / prev_eq.where(prev_eq > 0)) / 2.0


def compute_metrics(
    r: pd.Series,
    rf: float | pd.Series | None = None,
    benchmark: pd.Series | None = None,
    equity_frame: pd.DataFrame | None = None,
    round_trips: pd.DataFrame | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Compute the full metric set; NaN + warning when a metric is undefined."""
    warnings: list[str] = []
    r = r.dropna()
    n = len(r)
    m: dict[str, Any] = {"n_periods": n}
    if n < 2:
        warnings.append("fewer than two return observations")
    if n:
        m["start"] = str(r.index[0].date())
        m["end"] = str(r.index[-1].date())
    growth = float(np.prod(1.0 + r.to_numpy())) if n else np.nan
    m["total_return"] = growth - 1.0 if n else np.nan
    m["cagr"] = cagr(r, periods_per_year)
    if n and growth <= 0:
        warnings.append("equity reached zero or below: CAGR/Calmar undefined")
    m["mean_daily_arith"] = float(r.mean()) if n else np.nan
    m["mean_ann_arith"] = m["mean_daily_arith"] * periods_per_year
    m["ann_vol"] = float(r.std(ddof=1)) * math.sqrt(periods_per_year) if n > 1 else np.nan
    m["sharpe"] = sharpe_ratio(r, rf, periods_per_year)
    m["sharpe_per_period"] = sharpe_ratio(r, rf, periods_per_year, annualise=False)
    if n > 1 and not math.isfinite(m["sharpe"]):
        warnings.append("zero return volatility: Sharpe undefined")
    m["sortino"] = sortino_ratio(r, rf, periods_per_year)
    if n > 1 and not math.isfinite(m["sortino"]):
        warnings.append("no downside deviation: Sortino undefined")
    m["max_drawdown"] = max_drawdown(r)
    m["calmar"] = (m["cagr"] / m["max_drawdown"]
                   if m["max_drawdown"] and math.isfinite(m["cagr"]) and m["max_drawdown"] > 0
                   else np.nan)
    ld = longest_drawdown(r) if n else (0, 0)
    m["longest_dd_sessions"], m["longest_dd_days"] = ld
    if n > 3 and float(r.std()) > 1e-12:
        m["skew"] = float(stats.skew(r, bias=False))
        m["kurtosis_excess"] = float(stats.kurtosis(r, fisher=True, bias=False))
        m["kurtosis_pearson"] = m["kurtosis_excess"] + 3.0
    else:
        m["skew"] = m["kurtosis_excess"] = m["kurtosis_pearson"] = np.nan
    m["best_day"] = float(r.max()) if n else np.nan
    m["worst_day"] = float(r.min()) if n else np.nan
    if benchmark is not None:
        b = benchmark.reindex(r.index)
        both = pd.concat([r, b], axis=1, join="inner").dropna()
        if len(both) > 2 and both.iloc[:, 1].std() > 0:
            m["benchmark_corr"] = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
            m["beta"] = float(np.cov(both.iloc[:, 0], both.iloc[:, 1], ddof=1)[0, 1]
                              / both.iloc[:, 1].var(ddof=1))
            m["benchmark_cagr"] = cagr(both.iloc[:, 1], periods_per_year)
            m["benchmark_sharpe"] = sharpe_ratio(both.iloc[:, 1], 0.0 if rf is None else
                                                 _align_rf(both.iloc[:, 1], rf), periods_per_year)
        else:
            m["benchmark_corr"] = m["beta"] = np.nan
            warnings.append("benchmark overlap too short or constant")
    if equity_frame is not None:
        ef = equity_frame.loc[equity_frame.index.isin(r.index)]
        if len(ef):
            m["time_invested"] = float((ef["gross_exposure"] > 1e-9).mean())
            m["avg_gross_exposure"] = float(ef["gross_exposure"].mean())
            m["avg_net_exposure"] = float(ef["net_exposure"].mean())
            to = turnover_series(equity_frame).reindex(ef.index)
            years = n / periods_per_year
            m["turnover_annual"] = float(to.sum() / years) if years > 0 else np.nan
            first = equity_frame.index.searchsorted(ef.index[0])
            base = equity_frame.iloc[max(0, first - 1)]
            last = ef.iloc[-1]
            for k in ("commissions", "spread_costs", "slippage_costs", "impact_costs",
                      "interest", "borrow_fees"):
                m[k] = float(last[k] - base[k]) if first > 0 else float(last[k])
            m["total_trading_costs"] = (m["commissions"] + m["spread_costs"]
                                        + m["slippage_costs"] + m["impact_costs"])
            m["net_financing"] = m["interest"] - m["borrow_fees"]
            m["max_stale_marks"] = int(ef["n_stale_marks"].max())
    m.update(trade_stats(round_trips))
    if m["n_round_trips"] == 0:
        warnings.append("no completed round trips: trade statistics undefined")
    m["warnings"] = warnings
    return m


def metrics_frame(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Combine several metric dicts into a table (warnings joined as text)."""
    out = {}
    for k, v in rows.items():
        v = dict(v)
        v["warnings"] = "; ".join(v.get("warnings", []))
        out[k] = v
    return pd.DataFrame(out).T
