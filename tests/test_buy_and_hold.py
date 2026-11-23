"""Buy-and-hold against a timing-consistent independent reference."""

import math

import numpy as np
import pandas as pd
import pytest
from conftest import random_walk

from backtester import Backtest, BacktestConfig
from backtester.strategies import BuyAndHold


def reference(d, cash, fractional):
    """Independent pandas reference with the engine's documented convention.

    Signal at the close of session 0; quantity sized on that close (the decision
    price); filled at the open of session 1.  Without leverage cash may not go
    negative, so if the open gaps above the close the order is resized to what the
    cash buys at the open.  The residual stays in cash.
    """
    o = d.frame("open")["A"]
    c = d.frame("close")["A"]
    if fractional:
        shares = min(cash / c.iloc[0], cash / o.iloc[1])
    else:
        shares = min(math.floor(cash / c.iloc[0]), math.floor(cash / o.iloc[1]))
    resid = cash - shares * o.iloc[1]
    eq = shares * c + resid
    eq.iloc[0] = cash  # before the purchase
    return eq


def test_fractional_buy_and_hold_matches_reference_exactly():
    d = random_walk(500, ("A",), seed=5)
    res = Backtest(d, BuyAndHold(["A"]), config=BacktestConfig(fractional=True)).run()
    ref = reference(d, 1_000_000, True)
    diff = np.abs(res.equity["equity"].to_numpy() - ref.to_numpy()).max()
    assert diff < 1e-6  # currency units on 1e6 (relative ~1e-12)
    rel = np.abs(res.equity["equity"] / ref - 1).max()
    assert rel < 1e-10


def test_integer_buy_and_hold_keeps_residual_cash():
    d = random_walk(300, ("A",), seed=6)
    res = Backtest(d, BuyAndHold(["A"])).run()
    ref = reference(d, 1_000_000, False)
    pd.testing.assert_series_equal(res.equity["equity"], ref, check_names=False,
                                   check_freq=False, rtol=1e-12, atol=1e-6)
    assert res.fills.iloc[0]["quantity"] == min(math.floor(1_000_000 / d.arrays["close"][0, 0]),
                                                math.floor(1_000_000 / d.arrays["open"][1, 0]))


@pytest.mark.parametrize("seed", range(6))
def test_buy_and_hold_reference_over_many_paths(seed):
    """Gap-up and gap-down first fills both reconcile with the reference."""
    d = random_walk(120, ("A",), seed=100 + seed)
    for frac in (True, False):
        res = Backtest(d, BuyAndHold(["A"]), config=BacktestConfig(fractional=frac)).run()
        ref = reference(d, 1_000_000, frac)
        assert np.abs(res.equity["equity"].to_numpy() / ref.to_numpy() - 1).max() < 1e-10


def test_naive_first_close_benchmark_is_not_comparable():
    """The guide's initial_cash * close / close[0] reference ignores next-open execution."""
    d = random_walk(300, ("A",), seed=5)
    res = Backtest(d, BuyAndHold(["A"]), config=BacktestConfig(fractional=True)).run()
    c = d.frame("close")["A"]
    naive = 1_000_000 * c / c.iloc[0]
    assert np.abs(res.equity["equity"] - naive).max() > 1.0  # differs by the open/close gap
