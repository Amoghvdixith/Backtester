"""Cost and financing models against hand calculations."""

import math

import numpy as np
import pandas as pd
import pytest
from conftest import Recorder, make_data, random_walk

from backtester import (
    Backtest,
    BacktestConfig,
    Commission,
    CostModel,
    FinancingModel,
    Portfolio,
    PortfolioConstraints,
    Slippage,
    SpreadCost,
    SqrtImpact,
)
from backtester.events import OrderEvent
from backtester.execution import SimulatedExecution


def test_commission_hand_calculation():
    c = Commission(fixed=1.0, per_share=0.005, rate_bps=2.0, minimum=5.0, maximum=50.0)
    # 1 + 0.005*1000 + 2e-4 * 1000*50 = 1 + 5 + 10 = 16
    assert c(1000, 50.0) == pytest.approx(16.0)
    assert c(10, 50.0) == pytest.approx(5.0)  # minimum
    assert c(100000, 50.0) == pytest.approx(50.0)  # maximum
    assert c(0, 50.0) == 0.0
    with pytest.raises(ValueError):
        Commission(rate_bps=-1)


def test_spread_slippage_impact_fill_price():
    # 10 bps full spread -> 5 bps half; 3 bps slippage; impact 0.5 * 0.02 * sqrt(1000/100000)
    hs = SpreadCost(10).fraction()
    assert hs == pytest.approx(5e-4)
    imp, fb = SqrtImpact(0.5).fraction(1000, 100_000, 0.02)
    assert imp == pytest.approx(0.5 * 0.02 * math.sqrt(0.01)) and not fb
    imp2, fb2 = SqrtImpact(0.5, fallback_bps=7).fraction(1000, float("nan"), 0.02)
    assert imp2 == pytest.approx(7e-4) and fb2


def test_execution_itemised_costs_reconcile():
    d = random_walk(60)
    costs = CostModel(commission=Commission(rate_bps=1.0), spread=SpreadCost(10),
                      slippage=Slippage(3), impact=SqrtImpact(0.5))
    ex = SimulatedExecution(d, costs)
    pf = Portfolio(1e7, fractional=True)
    i = 40
    o = OrderEvent(1, d.index[i - 1], "A", 1000.0, d.index[i - 1], d.index[i], 100.0)
    ex.submit(o)
    fills, _ = ex.execute_open(i, pf)
    f = fills[0]
    ref = d.arrays["open"][i, 0]
    est = d.estimates(20, 20)
    imp = 0.5 * est.sigma[i, 0] * math.sqrt(1000 / est.adv[i, 0])
    expected_px = ref * (1 + 5e-4 + 3e-4 + imp)
    assert f.fill_price == pytest.approx(expected_px, rel=1e-12)
    assert f.spread_cost == pytest.approx(1000 * ref * 5e-4)
    assert f.slippage_cost == pytest.approx(1000 * ref * 3e-4)
    assert f.impact_cost == pytest.approx(1000 * ref * imp)
    assert f.shortfall == pytest.approx(f.spread_cost + f.slippage_cost + f.impact_cost)
    assert f.commission == pytest.approx(1e-4 * 1000 * expected_px)
    # cash deducted once: execution value + commission (no second spread charge)
    assert pf.cash == pytest.approx(1e7 - 1000 * expected_px - f.commission)


def test_sell_fill_price_is_below_reference():
    d = random_walk(60)
    ex = SimulatedExecution(d, CostModel(spread=SpreadCost(20)))
    pf = Portfolio(1e6, fractional=True)
    ex.submit(OrderEvent(1, d.index[29], "A", -10.0, d.index[29], d.index[30], 100.0))
    f = ex.execute_open(30, pf)[0][0]
    assert f.fill_price == pytest.approx(d.arrays["open"][30, 0] * (1 - 10e-4))
    assert f.shortfall == pytest.approx(f.spread_cost)  # adverse for sells too


def test_stochastic_slippage_is_reproducible():
    a = [Slippage(1, 5, seed=11).fraction() for _ in range(1)]
    s1, s2 = Slippage(1, 5, seed=11), Slippage(1, 5, seed=11)
    assert [s1.fraction() for _ in range(5)] == [s2.fraction() for _ in range(5)]
    assert a[0] == Slippage(1, 5, seed=11).fraction()


def test_proportional_scenario_replaces_detail():
    c = CostModel.proportional(20)
    assert c.spread.spread_bps == 0 and c.impact.coefficient == 0
    assert c.commission(100, 50.0) == pytest.approx(100 * 50 * 20e-4)
    assert CostModel.from_dict({"proportional_bps": 5}).commission.rate_bps == 5


def test_financing_accrual_hand_calculation():
    fm = FinancingModel(credit_rate=0.0365, debit_rate=0.073, borrow_fee=0.0365,
                        day_count_basis=365)
    i, fee = fm.accrue(cash=1_000_000, short_mv_abs=0, days=1)
    assert i == pytest.approx(100.0) and fee == 0
    i, fee = fm.accrue(cash=-500_000, short_mv_abs=0, days=2)
    assert i == pytest.approx(-500_000 * 0.073 * 2 / 365)
    # short: proceeds excluded from interest base by default
    i, fee = fm.accrue(cash=1_200_000, short_mv_abs=200_000, days=3)
    assert i == pytest.approx(1_000_000 * 0.0365 * 3 / 365)
    assert fee == pytest.approx(200_000 * 0.0365 * 3 / 365)
    with pytest.raises(ValueError):
        FinancingModel(day_count_basis=252)


def test_engine_borrow_fee_and_interest_accrue_over_weekend():
    c = [100.0] * 8
    d = make_data({"A": c, "B": c}, dates=pd.bdate_range("2021-01-04", periods=8))
    fin = FinancingModel(credit_rate=0.0365, borrow_fee=0.0365)
    res = Backtest(d, Recorder({d.index[0]: {"A": -1.0}}), financing=fin,
                   constraints=PortfolioConstraints(allow_short=True),
                   config=BacktestConfig(fractional=True, initial_cash=1_000_000)).run()
    eq = res.equity
    # short 1,000,000 of A from open of index 1; Fri->Mon accrual = 3 calendar days
    fee_mon = eq["borrow_fees"].iloc[5] - eq["borrow_fees"].iloc[4]
    assert d.index[5].day_name() == "Monday"
    assert fee_mon == pytest.approx(1_000_000 * 0.0365 * 3 / 365, rel=1e-9)
    # interest base = previous-close cash minus short proceeds (excluded from interest)
    int_mon = eq["interest"].iloc[5] - eq["interest"].iloc[4]
    base = eq["cash"].iloc[4] - abs(eq["short_value"].iloc[4])
    assert base == pytest.approx(1_000_100, abs=1.0)  # +100 from day-1 interest on all cash
    assert int_mon == pytest.approx(base * 0.0365 * 3 / 365, rel=1e-12)
    assert np.isfinite(eq["equity"]).all()
