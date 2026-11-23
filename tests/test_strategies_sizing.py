"""Momentum scheduling/lag convention, SMA behaviour and sizing rules."""

import math

import numpy as np
import pandas as pd
import pytest
from conftest import make_data, random_walk

from backtester import (
    Backtest,
    BacktestConfig,
    EqualWeight,
    KellyCapped,
    MembershipUniverse,
    PointInTimeView,
    VolTarget,
)
from backtester.events import MarketEvent, Phase, SignalEvent
from backtester.sizing import FixedFraction, build_sizer, trailing_vol
from backtester.strategies import EqualWeightUniverse, Momentum12_1, SMACrossover, build_strategy


def _trend_data(n=320, k=8, start="2019-01-01"):
    idx = pd.bdate_range(start, periods=n)
    closes = {f"S{j}": list(100 * np.exp(np.arange(n) * 0.0005 * (j + 1))) for j in range(k)}
    return make_data(closes, dates=idx)


def test_momentum_formula_and_lag():
    d = _trend_data()
    m = Momentum12_1(lookback=252, skip=21, quantile=0.25, min_universe=2)
    v = PointInTimeView(d)
    m.bind(v)
    i = 300
    v._advance(i, Phase.CLOSE)
    sc = m.scores()
    c = d.arrays["close"]
    for j, s in enumerate(d.symbols):
        assert sc[s] == pytest.approx(c[i - 21, j] / c[i - 252, j] - 1)
    longs, shorts = m.select(sc)
    assert longs == ["S7", "S6"] and shorts == []  # ceil(0.25 * 8) = 2


def test_momentum_monthly_schedule_and_next_open_execution():
    d = _trend_data(400)
    res = Backtest(d, Momentum12_1(quantile=0.25, min_universe=2)).run()
    sig_dates = sorted(set(res.signals["timestamp"]))
    for t in sig_dates:
        nxt = d.index[d.index.get_loc(t) + 1] if d.index.get_loc(t) + 1 < len(d.index) else None
        assert nxt is None or nxt.month != t.month  # last session of month
    assert sig_dates[0] >= d.index[252]  # insufficient history before
    f = res.fills.iloc[0]
    assert f["timestamp"].month != f["order_created_at"].month or \
        f["timestamp"].year != f["order_created_at"].year


def test_momentum_short_halt_does_not_force_exclusion():
    d0 = _trend_data(300, 4)
    frames = {f: d0.frame(f).copy() for f in ("open", "high", "low", "close", "volume")}
    for f in frames:
        frames[f].iloc[298:300, 0] = np.nan  # S0 halted on the last two sessions
    from backtester import MarketData

    d = MarketData(frames)
    m = Momentum12_1(quantile=0.25, min_universe=2)
    v = PointInTimeView(d)
    m.bind(v)
    v._advance(299, Phase.CLOSE)
    assert "S0" in m.scores()
    frames["close"].iloc[290:300, 0] = np.nan
    for f in frames:
        frames[f].iloc[290:300, 0] = np.nan  # halted longer than max_stale
    v2 = PointInTimeView(MarketData(frames))
    m.bind(v2)
    v2._advance(299, Phase.CLOSE)
    assert "S0" not in m.scores()


def test_momentum_ties_small_universe_and_long_short():
    m = Momentum12_1(quantile=0.1, min_universe=3, mode="long_short")
    assert m.select({"B": 0.1, "A": 0.1, "C": -0.2, "D": 0.3}) == (["D"], ["C"])
    m2 = Momentum12_1(quantile=0.5, min_universe=3)
    assert m2.select({"B": 0.1, "A": 0.1, "C": 0.0}) == (["A", "B"], [])  # tie -> name order
    assert m2.select({"A": 0.1}) == ([], [])  # too small -> cash
    for bad in ({"skip": 300}, {"quantile": 0.0}, {"mode": "x"}, {"min_names": 0}):
        with pytest.raises(ValueError):
            Momentum12_1(**bad)


def test_momentum_respects_point_in_time_membership():
    d = _trend_data(330)
    iv = pd.DataFrame({"symbol": d.symbols, "start": [d.index[0]] * 8,
                       "end": [None] * 8, "known_from": [d.index[0]] * 7 + [d.index[-1]]})
    m = Momentum12_1(quantile=0.25, min_universe=2)
    res = Backtest(d, m, universe=MembershipUniverse(iv)).run()
    assert "S7" not in set(res.fills["symbol"])  # strongest asset is not yet a member


def test_equal_weight_benchmark_and_registry():
    d = _trend_data(300, 4)
    res = Backtest(d, EqualWeightUniverse(min_history=1)).run()
    assert res.equity["gross_exposure"].iloc[-1] > 0.95
    assert isinstance(build_strategy("sma_crossover", fast=5, slow=10), SMACrossover)
    with pytest.raises(ValueError):
        build_strategy("nope")
    with pytest.raises(ValueError):
        EqualWeightUniverse(rebalance="weekly")


def test_sma_only_emits_on_change():
    d = random_walk(300, ("A", "B"), seed=2)
    res = Backtest(d, SMACrossover(10, 30)).run()
    by_day = res.signals.groupby("timestamp")
    assert by_day.ngroups < 300 / 2


def _view_at(d, i):
    v = PointInTimeView(d)
    v._advance(i, Phase.CLOSE)
    return v


def test_vol_target_uses_trailing_data_only():
    d = random_walk(200, ("A", "B"), seed=4)
    v = _view_at(d, 120)
    vol = trailing_vol(v, "A", 60)
    px = d.arrays["close"][60:121, 0]
    assert vol == pytest.approx(np.std(np.diff(np.log(px)), ddof=1) * math.sqrt(252))
    t = d.index[120]
    sigs = [SignalEvent(t, "A", 1.0), SignalEvent(t, "B", 1.0)]
    w = VolTarget(target_vol=0.10, lookback=60).weights(sigs, v)
    assert w["A"] == pytest.approx(min(1.0, 0.10 / vol) / 2)
    early = VolTarget(lookback=60).weights(sigs, _view_at(d, 5))
    assert early == {"A": 0.0, "B": 0.0}  # insufficient history
    ew = trailing_vol(v, "A", 60, halflife=20)
    assert ew > 0


def test_kelly_capped_and_shrunk():
    idx = pd.bdate_range("2019-01-01", periods=300)
    rng = np.random.default_rng(0)
    r = rng.normal(0.002, 0.01, 300)
    d = make_data({"A": list(100 * np.cumprod(1 + r))}, dates=idx)
    v = _view_at(d, 299)
    k = KellyCapped(lookback=252, kelly_fraction=0.5, shrinkage=0.5, cap=1.0)
    w = k.weights([SignalEvent(idx[299], "A", 1.0)], v)
    assert 0 < w["A"] <= 1.0
    px = d.arrays["close"][299 - 252:300, 0]
    rr = np.diff(px) / px[:-1]
    raw = 0.25 * rr.mean() / rr.var(ddof=1)
    assert w["A"] == pytest.approx(min(raw, 1.0))
    assert KellyCapped().weights([SignalEvent(idx[10], "A", 1.0)], _view_at(d, 10)) == {"A": 0.0}


def test_sizer_factory_and_equal_weight():
    t = pd.Timestamp("2020-01-02")
    sigs = [SignalEvent(t, "A", 1.0), SignalEvent(t, "B", -1.0), SignalEvent(t, "C", 0.0)]
    w = EqualWeight(1.0).weights(sigs, None)  # type: ignore[arg-type]
    assert w == {"A": 0.5, "B": -0.5, "C": 0.0}
    assert isinstance(build_sizer({"type": "vol_target", "target_vol": 0.2}), VolTarget)
    assert FixedFraction(0.2).weights(sigs, None)["A"] == 0.2  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_sizer({"type": "bogus"})
    with pytest.raises(ValueError):
        FixedFraction(0)


def test_vol_target_backtest_runs_with_caps():
    d = random_walk(300, ("A", "B"), seed=8)
    res = Backtest(d, SMACrossover(10, 50), sizer=VolTarget(0.1, 60),
                   config=BacktestConfig(rebalance_band=0.02)).run()
    assert res.equity["gross_exposure"].max() <= 1.0 + 1e-9
    MarketEvent  # noqa: B018 - imported for type completeness
