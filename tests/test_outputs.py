"""Result persistence, reconciliation tables, full-metric path and figure generation."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import random_walk

from backtester import Backtest, BacktestConfig, CostModel, compute_metrics, plots
from backtester.config import build_engine_parts, deep_update, load_dataset, load_yaml
from backtester.performance import metrics_frame, turnover_series
from backtester.results import environment_info
from backtester.strategies import SMACrossover
from backtester.synthetic import SyntheticSpec, write_dataset


@pytest.fixture(scope="module")
def result():
    d = random_walk(400, ("A", "B", "C"), seed=21)
    costs = CostModel.from_dict({"commission": {"rate_bps": 1, "minimum": 1.0},
                                 "spread": {"spread_bps": 8}, "slippage": {"bps": 2},
                                 "impact": {"coefficient": 0.5}})
    return Backtest(d, SMACrossover(10, 40), costs=costs,
                    config=BacktestConfig(rebalance_band=0.02)).run()


def test_reconciliation_table_closes(result):
    rec = result.reconciliation()
    assert np.abs(rec["components_minus_shortfall"]).max() < 1e-6
    assert (rec["execution_shortfall"] >= -1e-9).all()  # deterministic costs are adverse
    cs = result.cost_summary()
    assert cs["total_trading_costs"] == pytest.approx(
        rec[["spread_cost", "slippage_cost", "impact_cost", "commission"]].to_numpy().sum())


def test_full_metrics_with_equity_frame(result):
    r = result.returns()
    m = compute_metrics(r, equity_frame=result.equity, round_trips=result.round_trips,
                        benchmark=r * 0.5 + 0.0001)
    assert 0 <= m["time_invested"] <= 1
    assert m["turnover_annual"] > 0
    assert m["total_trading_costs"] == pytest.approx(result.cost_summary()["total_trading_costs"])
    to = turnover_series(result.equity)
    assert to.iloc[0] == 0
    tbl = metrics_frame({"x": m})
    assert "sharpe" in tbl.columns


def test_save_writes_all_tables(tmp_path, result):
    written = result.save(tmp_path, prefix="t")
    for name in ("equity", "fills", "orders", "order_status", "round_trips", "events",
                 "reconciliation", "meta"):
        assert (tmp_path / written[name]).exists()
    meta = json.loads((tmp_path / written["meta"]).read_text())
    assert meta["execution_convention"].startswith("signal at close")
    assert "numpy" in environment_info()


def test_figures_are_generated(tmp_path, result):
    r = result.returns()
    stamp = plots.Stamp("SMA", "synthetic", "x", costs="test", sample="in-sample")
    figs = tmp_path / "figures"
    p1 = plots.tearsheet(r, figs / "tear.png", stamp, bench=r * 0.8)
    p2 = plots.exposure(result.equity, figs / "exp.png", stamp)
    p3 = plots.turnover_and_costs(result.equity, figs / "cost.png", stamp)
    tbl = pd.DataFrame([[0.1, np.nan], [-0.2, 0.3]], index=[5, 10], columns=[20, 40])
    p4 = plots.heatmap(tbl, figs / "hm.png", stamp, "t", "slow", "fast")
    p5 = plots.lines({"a": (1 + r).cumprod()}, figs / "l.png", stamp, "t",
                     spans=[(r.index[0], r.index[50], "fold 0")])
    p6 = plots.bars(pd.DataFrame({"x": [0.1, 0.2]}, index=["a", "b"]), figs / "b.png", stamp,
                    "t", ["x"], pct=[True])
    p7 = plots.architecture_diagram(figs / "arch.png")
    for k in ("equity", "drawdown", "monthly", "rolling_sharpe", "distribution", "annual"):
        assert plots.single(k, r, figs / f"{k}.png", stamp).stat().st_size > 1000
    for p in (p1, p2, p3, p4, p5, p6, p7):
        assert p.exists() and p.stat().st_size > 1000
    assert (tmp_path / "tables" / "figure_data" / "tear.csv").exists()


def test_config_loading_and_dataset(tmp_path):
    write_dataset(SyntheticSpec(n_assets=4, start="2018-01-01", end="2019-12-31", seed=1),
                  tmp_path / "data", "mini")
    cfg = {"data": {"mode": "synthetic", "path": "data/mini.csv",
                    "membership": "data/mini_membership.csv"},
           "costs": {"spread": {"spread_bps": 5}}, "sizing": {"type": "equal_weight"}}
    data, uni, mode = load_dataset(cfg, tmp_path)
    assert mode == "synthetic" and data.provenance["data_mode"] == "synthetic"
    assert uni.all_symbols() == data.symbols
    parts = build_engine_parts(cfg)
    assert parts["costs"].spread.spread_bps == 5
    assert deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    (tmp_path / "c.yaml").write_text("a: 1\n")
    assert load_yaml(tmp_path / "c.yaml") == {"a": 1}
    with pytest.raises(FileNotFoundError, match="synthetic"):
        load_dataset({"data": {"mode": "synthetic", "path": "nope.csv"}}, tmp_path)
    with pytest.raises(FileNotFoundError, match="network"):
        load_dataset({"data": {"mode": "market", "path": "nope.csv"}}, tmp_path)
