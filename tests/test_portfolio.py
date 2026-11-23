"""Portfolio accounting, lots, constraints and order handling."""

import pandas as pd
import pytest
from conftest import Recorder, make_data, random_walk

from backtester import (
    AccountingError,
    Backtest,
    BacktestConfig,
    ExecutionConfig,
    Portfolio,
    PortfolioConstraints,
)
from backtester.events import FillEvent, OrderEvent, Phase
from backtester.execution import SimulatedExecution

D = pd.bdate_range("2021-01-04", periods=10)


def fill(k, sym, q, px, comm=0.0, oid=1):
    return FillEvent(D[k], Phase.OPEN, oid, sym, q, px, px, px, comm, 0.0, 0.0, 0.0, D[k - 1])


def test_long_buy_sell_realised_pnl_and_identity():
    pf = Portfolio(10_000, fractional=True)
    pf.apply_fill(fill(1, "A", 10, 100, comm=1.0))
    pf.update_marks({"A": 110}, D[1])
    assert pf.equity == pytest.approx(10_000 - 1000 - 1 + 1100)
    pf.reconcile()
    pf.apply_fill(fill(2, "A", -10, 120, comm=1.0))
    assert pf.realised_pnl == pytest.approx(200)
    rt = pf.round_trips_frame().iloc[0]
    assert rt["net_pnl"] == pytest.approx(198) and rt["direction"] == "long"
    assert pf.cash == pytest.approx(10_198)
    pf.reconcile()


def test_short_and_cover():
    pf = Portfolio(10_000, fractional=True)
    pf.apply_fill(fill(1, "A", -10, 100))
    assert pf.cash == pytest.approx(11_000)
    pf.update_marks({"A": 90}, D[1])
    assert pf.short_value == pytest.approx(-900)
    assert pf.equity == pytest.approx(10_100)
    pf.reconcile()
    pf.apply_fill(fill(2, "A", 10, 90))
    assert pf.realised_pnl == pytest.approx(100)
    assert pf.round_trips_frame().iloc[0]["direction"] == "short"
    assert "A" not in pf.positions


def test_partial_exit_and_reversal_fifo():
    pf = Portfolio(100_000, fractional=True)
    pf.apply_fill(fill(1, "A", 10, 100, oid=1))
    pf.apply_fill(fill(2, "A", 10, 110, oid=2))
    pf.apply_fill(fill(3, "A", -15, 120, oid=3))  # closes lot1 (10) + 5 of lot2
    rts = pf.round_trips_frame()
    assert list(rts["quantity"]) == [10, 5]
    assert list(rts["gross_pnl"]) == pytest.approx([200, 50])
    assert pf.quantity("A") == 5
    pf.apply_fill(fill(4, "A", -10, 130, oid=4))  # closes 5, opens short 5
    assert pf.quantity("A") == -5
    assert pf.lots["A"][0].entry_price == 130 and pf.lots["A"][0].quantity == -5
    pf.update_marks({"A": 125}, D[4])
    pf.reconcile()
    assert pf.unrealised_pnl == pytest.approx(25)


def test_commission_allocated_to_round_trip():
    pf = Portfolio(100_000, fractional=True)
    pf.apply_fill(fill(1, "A", 10, 100, comm=10.0))
    pf.apply_fill(fill(2, "A", -5, 100, comm=4.0))
    rt = pf.round_trips_frame().iloc[0]
    assert rt["commissions"] == pytest.approx(5.0 + 4.0)


def test_reconcile_detects_corruption():
    pf = Portfolio(1000)
    pf.apply_fill(fill(1, "A", 1, 100))
    pf.cash += 50  # corrupt
    with pytest.raises(AccountingError):
        pf.reconcile()


def test_invalid_portfolio_inputs():
    with pytest.raises(ValueError):
        Portfolio(0)
    pf = Portfolio(100)
    with pytest.raises(AccountingError):
        pf.apply_fill(fill(1, "A", 1, float("nan")))


def test_insufficient_cash_resize_and_reject():
    d = make_data({"A": [100] * 5, "B": [100] * 5}, {"A": [100, 100, 150, 100, 100],
                                                      "B": [100, 100, 150, 100, 100]})
    for policy, expected in (("resize", "resized"), ("reject", "rejected")):
        res = Backtest(d, Recorder({d.index[1]: {"A": 1.0, "B": 1.0}}),
                       execution=ExecutionConfig(insufficient_cash=policy),
                       config=BacktestConfig(initial_cash=10_000)).run()
        st = res.order_status
        assert (st["status"] == expected).sum() == 2
        assert res.equity["cash"].min() >= -1e-9
        if policy == "resize":
            # gap +50%: 50 shares each wanted, cash 10,000 at 150 -> 33 shares each, pro rata
            assert sorted(res.fills["quantity"]) == [33, 33]


def test_constraints_gross_net_concentration():
    c = PortfolioConstraints(max_gross=1.0, max_net=0.5, max_position=0.3, allow_short=True)
    w = c.apply({"A": 0.8, "B": 0.8, "C": -0.2})
    assert max(abs(v) for v in w.values()) <= 0.3 + 1e-12
    assert sum(abs(v) for v in w.values()) <= 1.0 + 1e-12
    assert abs(sum(w.values())) <= 0.5 + 1e-12
    lo = PortfolioConstraints().apply({"A": -0.5, "B": 0.5})
    assert lo == {"A": 0.0, "B": 0.5}
    with pytest.raises(ValueError):
        PortfolioConstraints(max_gross=0)


def test_multi_asset_targets_are_one_coherent_rebalance():
    d = random_walk(30, ("A", "B", "C", "D"))
    res = Backtest(d, Recorder({d.index[3]: {s: 1.0 for s in "ABCD"}})).run()
    tw = res.targets[res.targets["date"] == d.index[3]]["target_weight"]
    assert tw.sum() == pytest.approx(1.0)  # not 100% to each bullish asset
    assert res.equity["gross_exposure"].max() <= 1.0 + 1e-9


def test_leverage_cap_and_margin():
    d = random_walk(30, ("A", "B"))
    res = Backtest(d, Recorder({d.index[3]: {"A": 1.0, "B": 1.0}}),
                   sizer=__import__("backtester").FixedFraction(1.0),
                   constraints=PortfolioConstraints(max_gross=1.5, max_net=1.5,
                                                    max_position=1.0)).run()
    g = res.equity["gross_exposure"].iloc[5]
    assert 1.3 < g <= 1.6
    assert res.equity["cash"].iloc[5] < 0  # borrowed


def test_duplicate_targets_do_not_trade_and_pending_replaced():
    d = random_walk(30)
    plan = {d.index[3]: {"A": 1.0}, d.index[6]: {"A": 1.0}}
    res = Backtest(d, Recorder(plan), config=BacktestConfig(rebalance_band=0.05)).run()
    assert len(res.fills) == 1  # second identical target within band -> no trade
    ex = SimulatedExecution(d)
    o1 = OrderEvent(1, d.index[3], "A", 5, d.index[3], d.index[4], 100.0)
    o2 = OrderEvent(2, d.index[3], "A", 7, d.index[3], d.index[4], 100.0)
    ex.submit(o1)
    st = ex.submit(o2)
    assert st[0].status.value == "replaced" and ex.pending["A"].order_id == 2


def test_insolvency_stops_trading():
    c = [100, 100, 100, 1.0, 1.0, 1.0]
    d = make_data({"A": c, "B": [10] * 6})
    res = Backtest(d, Recorder({d.index[0]: {"A": 1.0}}), sizer=__import__(
        "backtester").FixedFraction(3.0), constraints=PortfolioConstraints(
        max_gross=3.0, max_net=3.0, max_position=3.0),
        config=BacktestConfig(fractional=True)).run()
    assert res.meta["insolvent"]
    assert (res.events["event"] == "insolvent").any()
