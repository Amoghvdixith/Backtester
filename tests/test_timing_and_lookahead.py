"""Execution timing and look-ahead protection."""

import numpy as np
import pandas as pd
import pytest
from conftest import Recorder, make_data, random_walk

from backtester import Backtest, BacktestConfig, LookAheadError, PointInTimeView
from backtester.events import MarketEvent, Phase
from backtester.strategies import SMACrossover
from backtester.strategy import Strategy


def test_fill_at_next_open_never_same_close():
    closes = {"A": [10, 11, 12, 13, 14, 15]}
    opens = {"A": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]}
    d = make_data(closes, opens)
    plan = {d.index[1]: {"A": 1.0}}
    res = Backtest(d, Recorder(plan), config=BacktestConfig(fractional=True)).run()
    f = res.fills.iloc[0]
    assert f["timestamp"] == d.index[2]
    assert f["order_created_at"] == d.index[1]
    assert f["reference_price"] == pytest.approx(11.5)  # open of t+1, not close of t (12)
    assert f["fill_price"] == pytest.approx(11.5)
    assert (res.fills["timestamp"] > res.fills["order_created_at"]).all()


def test_strategy_sees_post_fill_holdings():
    d = random_walk(40)

    class Probe(Strategy):
        name = "probe"

        def reset(self):
            self.qty_seen = {}

        def on_bar(self, event):
            self.qty_seen[event.timestamp] = engine.portfolio.quantity("A")
            if event.timestamp == d.index[5]:
                return [self.signal(event, "A", 1.0)]
            return None

    s = Probe()
    engine = Backtest(d, s, config=BacktestConfig(fractional=True))
    engine.run()
    assert s.qty_seen[d.index[5]] == 0
    assert s.qty_seen[d.index[6]] > 0  # fill at open of index[6] applied before its close


def test_open_phase_view_hides_current_close():
    d = make_data({"A": [1.0, 2.0, 3.0]}, {"A": [0.9, 1.9, 2.9]})
    v = PointInTimeView(d)
    v._advance(1, Phase.OPEN)
    assert v.history("A", "close")[-1] == 1.0  # yesterday's close only
    assert v.history("A", "open")[-1] == 1.9  # today's open is known
    assert len(v.history("A", "high")) == 1
    assert len(v.history("A", "volume")) == 1
    with pytest.raises(LookAheadError):
        v.is_tradable_today("A")
    v._advance(1, Phase.CLOSE)
    assert v.history("A", "close")[-1] == 2.0


def test_events_processed_in_phase_order():
    d = random_walk(30)
    res = Backtest(d, Recorder({d.index[3]: {"A": 1.0}}),
                   config=BacktestConfig(fractional=True)).run()
    ev = res.events
    order_row = ev[ev["event"] == "order"].index[0]
    fill_row = ev[ev["event"] == "fill"].index[0]
    assert order_row < fill_row
    assert ev.loc[fill_row, "phase"] == "open"
    assert pd.Timestamp(ev.loc[fill_row, "timestamp"]) == d.index[4]


class Threshold(Strategy):
    name = "threshold"

    def on_bar(self, event):
        px = self.view.history("A", "close", 1)[-1]
        return [self.signal(event, "A", 1.0 if px > 1000 else 0.0)]


def test_future_spike_cannot_be_traded_early():
    n = 30
    c = [100.0] * n
    c[20] = 5000.0  # spike in the close of session 20
    d = make_data({"A": c})
    res = Backtest(d, Threshold(), config=BacktestConfig(fractional=True)).run()
    buys = res.fills[res.fills["quantity"] > 0]
    assert len(buys) == 1
    assert buys.iloc[0]["timestamp"] == d.index[21]  # first possible fill: next open
    eq = res.equity["equity"]
    assert (eq.iloc[:21] == 1_000_000).all()


@pytest.mark.parametrize("mutation", ["shuffle", "scale"])
def test_future_data_mutation_does_not_change_past(mutation):
    d = random_walk(400, ("A", "B", "C"), seed=3)
    cut = 250
    frames = {f: d.frame(f).copy() for f in ("open", "high", "low", "close", "volume")}
    rng = np.random.default_rng(7)
    perm = rng.permutation(np.arange(cut + 1, len(d.index)))
    for f in frames:
        vals = frames[f].to_numpy().copy()
        if mutation == "shuffle":
            vals[cut + 1:] = vals[perm]
        else:
            vals[cut + 1:] = vals[cut + 1:] * (10.0 if f != "volume" else 3.0)
        frames[f] = pd.DataFrame(vals, index=d.index, columns=d.symbols)
    from backtester import CostModel, MarketData

    d2 = MarketData(frames)
    costs = {"spread": {"spread_bps": 5}, "impact": {"coefficient": 0.5}}
    r1 = Backtest(d, SMACrossover(10, 40), costs=CostModel.from_dict(costs)).run()
    r2 = Backtest(d2, SMACrossover(10, 40), costs=CostModel.from_dict(costs)).run()
    e1, e2 = r1.equity["equity"].iloc[: cut + 1], r2.equity["equity"].iloc[: cut + 1]
    pd.testing.assert_series_equal(e1, e2)
    f1 = r1.fills[r1.fills["timestamp"] <= d.index[cut]].reset_index(drop=True)
    f2 = r2.fills[r2.fills["timestamp"] <= d.index[cut]].reset_index(drop=True)
    pd.testing.assert_frame_equal(f1, f2)


def test_sma_warmup_no_signal_before_slow_window():
    d = random_walk(120)
    s = SMACrossover(5, 50)
    res = Backtest(d, s).run()
    first_active = res.signals[res.signals["direction"] != 0]["timestamp"].min()
    assert first_active >= d.index[49]
    with pytest.raises(ValueError):
        SMACrossover(50, 50)


def test_final_bar_orders_are_not_created():
    d = random_walk(10)
    res = Backtest(d, Recorder({d.index[-1]: {"A": 1.0}})).run()
    assert res.fills.empty
    assert (res.events["event"] == "order_not_created").any()


def test_terminal_liquidation_on_final_open():
    d = random_walk(20)
    res = Backtest(d, Recorder({d.index[2]: {"A": 1.0}}),
                   config=BacktestConfig(liquidate_on_final_open=True)).run()
    last = res.fills.iloc[-1]
    assert last["timestamp"] == d.index[-1] and last["quantity"] < 0
    assert res.equity["n_positions"].iloc[-1] == 0


def test_strategy_signal_timestamp_must_be_current():
    d = random_walk(10)

    class Bad(Strategy):
        name = "bad"

        def on_bar(self, event):
            return [self.signal(MarketEvent(event.timestamp - pd.Timedelta(days=1),
                                            event.phase), "A", 1.0)]

    with pytest.raises(RuntimeError, match="timestamp"):
        Backtest(d, Bad()).run()
