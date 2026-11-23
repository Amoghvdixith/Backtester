"""Data validation, missing/stale bars, lagged estimates and point-in-time membership."""

import numpy as np
import pandas as pd
import pytest
from conftest import Recorder, make_data, random_walk

from backtester import (
    Backtest,
    BacktestConfig,
    DataValidationError,
    ExecutionConfig,
    MarketData,
    MembershipUniverse,
    StaticUniverse,
)
from backtester.synthetic import SyntheticSpec, generate


def _long(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])


def test_validation_rejects_bad_inputs():
    good = [("2021-01-04", "A", 10, 11, 9, 10.5, 100), ("2021-01-05", "A", 10, 11, 9, 10, 100)]
    MarketData.from_long(_long(good))
    with pytest.raises(DataValidationError, match="duplicate"):
        MarketData.from_long(_long(good + [good[0]]))
    with pytest.raises(DataValidationError, match="sorted"):
        MarketData.from_long(_long(good[::-1]))
    with pytest.raises(DataValidationError, match="OHLC"):
        MarketData.from_long(_long([("2021-01-04", "A", 10, 10.2, 9, 10.5, 100)]))
    with pytest.raises(DataValidationError, match="positive"):
        MarketData.from_long(_long([("2021-01-04", "A", -1, 11, -2, 10, 100)]))
    with pytest.raises(DataValidationError, match="non-negative"):
        MarketData.from_long(_long([("2021-01-04", "A", 10, 11, 9, 10, -5)]))
    with pytest.raises(DataValidationError, match="some but not all"):
        MarketData.from_long(_long([("2021-01-04", "A", 10, 11, 9, None, 100)]))


def test_long_roundtrip_and_csv(tmp_path):
    d = random_walk(20)
    p = tmp_path / "x.csv"
    d.to_long().to_csv(p, index=False)
    d2 = MarketData.from_csv(p)
    np.testing.assert_allclose(d2.arrays["close"], d.arrays["close"])
    with pytest.raises(FileNotFoundError):
        MarketData.from_csv(tmp_path / "missing.csv")


def test_missing_bar_order_waits_and_is_not_filled_at_ffilled_price():
    c = {"A": [10, 10, np.nan, np.nan, 12, 12, 12], "B": [5] * 7}
    d = make_data(c, {"A": [10, 10, np.nan, np.nan, 11.8, 12, 12]})
    assert not d.tradable[2, 0]
    res = Backtest(d, Recorder({d.index[1]: {"A": 1.0}}),
                   config=BacktestConfig(fractional=True)).run()
    f = res.fills.iloc[0]
    assert f["timestamp"] == d.index[4]  # first tradable open for A
    assert f["reference_price"] == pytest.approx(11.8)
    # stale marks: A held? No - bought on index 4; B never held. Check stale marking path:
    assert res.equity["n_stale_marks"].max() == 0


def test_order_expires_without_tradable_bar():
    c = {"A": [10, 10] + [np.nan] * 6 + [10], "B": [5] * 9}
    d = make_data(c)
    res = Backtest(d, Recorder({d.index[1]: {"A": 1.0}}),
                   execution=ExecutionConfig(expires_after=3)).run()
    assert res.fills.empty
    st = res.order_status
    assert (st["status"] == "expired").sum() == 1


def test_stale_marks_are_counted():
    c = {"A": [10, 10, 10, np.nan, np.nan, 11, 11], "B": [5] * 7}
    d = make_data(c)
    res = Backtest(d, Recorder({d.index[0]: {"A": 1.0}})).run()
    assert res.equity["n_stale_marks"].iloc[3] == 1
    assert res.equity["n_stale_marks"].iloc[5] == 0


def test_lagged_estimates_ignore_same_day_volume():
    d = random_walk(60)
    est1 = d.estimates(10, 10)
    frames = {f: d.frame(f).copy() for f in ("open", "high", "low", "close", "volume")}
    frames["volume"].iloc[40] *= 50
    d2 = MarketData(frames)
    est2 = d2.estimates(10, 10)
    np.testing.assert_allclose(est1.adv[:41], est2.adv[:41])  # row 40 unaffected by its volume
    assert not np.allclose(est1.adv[41], est2.adv[41])


def test_membership_known_from_and_end():
    iv = pd.DataFrame({"symbol": ["A", "B"], "start": ["2020-01-01", "2020-01-01"],
                       "end": [None, "2020-06-30"], "known_from": ["2020-03-01", None]})
    u = MembershipUniverse(iv)
    assert u.members(pd.Timestamp("2020-02-01")) == ["B"]  # A not yet known
    assert u.members(pd.Timestamp("2020-04-01")) == ["A", "B"]
    assert u.members(pd.Timestamp("2020-07-01")) == ["A"]
    ev = pd.DataFrame({"date": ["2020-01-01", "2020-05-01", "2020-01-01"],
                       "symbol": ["A", "A", "B"], "in_index": [True, False, True]})
    u2 = MembershipUniverse.from_events(ev)
    assert u2.members(pd.Timestamp("2020-06-01")) == ["B"]
    with pytest.raises(ValueError):
        StaticUniverse([])


def test_universe_exit_liquidates_holding():
    d = random_walk(30, ("A", "B"))
    iv = pd.DataFrame({"symbol": ["A", "B"], "start": [d.index[0]] * 2,
                       "end": [d.index[10], None]})
    res = Backtest(d, Recorder({d.index[2]: {"A": 1.0, "B": 1.0}}),
                   universe=MembershipUniverse(iv)).run()
    sells = res.fills[(res.fills["symbol"] == "A") & (res.fills["quantity"] < 0)]
    assert sells.iloc[0]["timestamp"] == d.index[12]  # exit seen at close 11, sold at open 12
    assert res.equity["n_positions"].iloc[-1] == 1


def test_synthetic_generator_is_reproducible_and_valid():
    spec = SyntheticSpec(n_assets=8, start="2015-01-01", end="2017-12-31", seed=9)
    a, m1, _ = generate(spec)
    b, m2, _ = generate(spec)
    pd.testing.assert_frame_equal(a, b)
    d = MarketData.from_long(a)  # passes OHLC validation
    assert (d.frame("close").notna().sum() > 0).all()
    assert m1["end"].notna().any()  # some delistings present
