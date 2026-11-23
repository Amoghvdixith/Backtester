"""Performance metrics (known values) and Sharpe statistics / DSR (paper reference cases)."""

import math

import numpy as np
import pandas as pd
import pytest

from backtester.performance import (
    annual_to_periodic,
    cagr,
    compute_metrics,
    longest_drawdown,
    max_drawdown,
    monthly_table,
    sharpe_ratio,
    sortino_ratio,
    trade_stats,
)
from backtester.statistics import (
    average_offdiag_correlation,
    circular_block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    implied_independent_trials,
    probabilistic_sharpe_ratio,
    sharpe_se_iid_normal,
    sharpe_se_nonnormal,
)

IDX = pd.bdate_range("2020-01-01", periods=8)


def test_constant_positive_returns():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    m = compute_metrics(r)
    assert m["max_drawdown"] == 0
    assert math.isnan(m["sharpe"]) and math.isnan(m["sortino"]) and math.isnan(m["calmar"])
    assert m["cagr"] == pytest.approx(1.001 ** 252 - 1)
    assert any("volatility" in w for w in m["warnings"])


def test_known_values_small_series():
    r = pd.Series([0.10, -0.05, 0.02, -0.03, 0.04, 0.0, -0.01, 0.03], index=IDX)
    x = r.to_numpy()
    assert sharpe_ratio(r) == pytest.approx(math.sqrt(252) * x.mean() / x.std(ddof=1))
    dd = math.sqrt(np.mean(np.minimum(x, 0) ** 2))
    assert sortino_ratio(r) == pytest.approx(math.sqrt(252) * x.mean() / dd)
    w = np.cumprod(1 + x)
    peak = np.maximum.accumulate(np.concatenate([[1], w]))[1:]
    assert max_drawdown(r) == pytest.approx(np.max(1 - w / peak))
    assert cagr(r) == pytest.approx(np.prod(1 + x) ** (252 / 8) - 1)
    rf = 0.0001
    assert sharpe_ratio(r, rf) == pytest.approx(math.sqrt(252) * (x - rf).mean()
                                               / (x - rf).std(ddof=1))
    n, days = longest_drawdown(r)
    assert n >= 1 and days >= 1


def test_nonpositive_equity_and_rf_alignment():
    r = pd.Series([0.1, -1.0, 0.0], index=IDX[:3])
    m = compute_metrics(r)
    assert math.isnan(m["cagr"])
    assert any("zero or below" in w for w in m["warnings"])
    with pytest.raises(ValueError):
        sharpe_ratio(r, pd.Series(0.0, index=IDX[:2]))
    assert annual_to_periodic(0.0252) == pytest.approx(0.0001)
    assert annual_to_periodic(0.05, 252, "compound") == pytest.approx(1.05 ** (1 / 252) - 1)


def test_monthly_table_compounds():
    r = pd.Series([0.1, 0.1], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    assert monthly_table(r).loc[2020, 1] == pytest.approx(0.21)


def test_trade_stats():
    rt = pd.DataFrame({"net_pnl": [10.0, -5.0, 20.0, -5.0], "holding_days": [1, 2, 3, 4]})
    s = trade_stats(rt)
    assert s["hit_rate"] == 0.5 and s["profit_factor"] == pytest.approx(3.0)
    assert s["avg_win"] == 15 and s["avg_loss"] == -5
    s2 = trade_stats(pd.DataFrame({"net_pnl": [1.0], "holding_days": [1]}))
    assert math.isnan(s2["profit_factor"])  # no losses -> undefined, not infinite
    assert trade_stats(None)["n_round_trips"] == 0


def test_benchmark_beta_and_correlation():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    r = 2 * b + 0.0001
    m = compute_metrics(r, benchmark=b)
    assert m["beta"] == pytest.approx(2.0) and m["benchmark_corr"] == pytest.approx(1.0)


# ------------------------------------------------------------------ statistics
def test_sharpe_standard_errors():
    assert sharpe_se_iid_normal(0.1, 1000) == pytest.approx(math.sqrt((1 + 0.005) / 1000))
    # non-normal formula reduces to the IID-normal one at skew 0, Pearson kurtosis 3
    assert sharpe_se_nonnormal(0.1, 1000, 0.0, 3.0) == pytest.approx(
        sharpe_se_iid_normal(0.1, 1000))
    assert math.isnan(sharpe_se_iid_normal(0.1, 1))


def test_dsr_reproduces_bailey_lopez_de_prado_2014_example():
    """Numerical example in Bailey & Lopez de Prado (2014), JPM 40(5).

    Annualised SR 2.5 over T = 1250 daily obs (250/yr), N = 100 trials,
    V[SR_n] = 1/2 (annualised), skew -3, kurtosis 10.  The paper reports
    SR0 = 0.1132 (non-annualised), DSR = 0.9004, DSR = 0.9505 had N been 46,
    and that with Normal returns DSR stays >= 0.95 up to N = 88.
    """
    sr = 2.5 / math.sqrt(250)
    v = 0.5 / 250
    sr0 = expected_max_sharpe(v, 100)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    assert probabilistic_sharpe_ratio(sr, sr0, 1250, -3, 10) == pytest.approx(0.9004, abs=5e-5)
    sr0_46 = expected_max_sharpe(v, 46)
    assert probabilistic_sharpe_ratio(sr, sr0_46, 1250, -3, 10) == pytest.approx(0.9505,
                                                                               abs=5e-5)
    sr0_88 = expected_max_sharpe(v, 88)
    assert probabilistic_sharpe_ratio(sr, sr0_88, 1250, 0, 3) == pytest.approx(0.9505, abs=5e-4)


def test_dsr_special_cases():
    assert expected_max_sharpe(0.01, 1) == 0.0  # one trial: no selection
    d = deflated_sharpe_ratio(0.05, [0.05], 500, 0.0, 3.0)
    assert d["sr0_per_period"] == 0.0 and any("variance" in w for w in d["warnings"])
    assert d["dsr"] == pytest.approx(d["psr_vs_zero"])
    with pytest.raises(ValueError):
        expected_max_sharpe(-1, 10)
    bad = probabilistic_sharpe_ratio(0.5, 0.0, 100, 10.0, 3.0)  # non-positive denominator
    assert math.isnan(bad)
    few = deflated_sharpe_ratio(0.1, [0.1, 0.0, 0.05], 20, 0, 3)
    assert any("30 observations" in w for w in few["warnings"])


def test_effective_trials_and_correlation():
    assert implied_independent_trials(100, 0.0) == 100
    assert implied_independent_trials(100, 1.0) == 1
    assert implied_independent_trials(100, 0.5) == pytest.approx(50.5)
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    df = pd.DataFrame({"a": base, "b": base + 0.01 * rng.normal(size=500),
                       "c": base + 0.01 * rng.normal(size=500)})
    assert average_offdiag_correlation(df) > 0.99


def test_block_bootstrap_reproducible():
    r = np.random.default_rng(3).normal(0.0005, 0.01, 1000)
    a = circular_block_bootstrap_sharpe(r, block=20, n_boot=300, seed=1)
    b = circular_block_bootstrap_sharpe(r, block=20, n_boot=300, seed=1)
    assert a == b and a["ci_low"] < a["ci_high"]
    assert math.isnan(circular_block_bootstrap_sharpe(r[:10], block=20)["se"])
