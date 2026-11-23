"""Walk-forward splits, hold-out protection, fold boundaries and trial logging."""

import json

import pandas as pd
import pytest
from conftest import random_walk

from backtester import CostModel
from backtester.config import make_run_factory
from backtester.strategies import SMACrossover
from backtester.walkforward import (
    HoldoutViolation,
    ScheduledStrategy,
    TrialRegistry,
    WalkForward,
    check_holdout_lock,
    evaluate_holdout,
    make_folds,
    param_grid,
    research_data,
)

CFG = {"costs": {"spread": {"spread_bps": 10}}, "backtest": {"rebalance_band": 0.02}}


@pytest.fixture(scope="module")
def data():
    return random_walk(252 * 7, ("A", "B", "C"), seed=11, start="2010-01-01")


def test_folds_are_chronological_and_disjoint(data):
    folds = make_folds(data.index, 2, 1)
    for f in folds:
        assert f.train_start < f.train_end < f.test_start <= f.test_end
    for a, b in zip(folds, folds[1:], strict=False):
        assert b.test_start > a.test_end
        assert b.train_start > a.train_start
    with pytest.raises(ValueError):
        make_folds(data.index[:100], 3, 1)


def test_param_grid_deterministic():
    g = param_grid({"slow": [50, 20], "fast": [5, 10]})
    assert g[0] == {"fast": 5, "slow": 50} and len(g) == 4


def test_holdout_is_physically_excluded(data):
    hs = data.index[-252]
    rd = research_data(data, hs)
    assert rd.index[-1] < hs
    with pytest.raises(HoldoutViolation):
        WalkForward(data, lambda p: SMACrossover(**p), {"fast": [5]}, make_run_factory(CFG),
                    holdout_start=hs)
    with pytest.raises(HoldoutViolation):
        research_data(data, data.index[0])


@pytest.fixture(scope="module")
def wf_result(data):
    hs = data.index[-252]
    rd = research_data(data, hs)
    grid = {"fast": [5, 20, 60], "slow": [40, 60]}  # includes invalid (60, 40) and (60, 60)
    wf = WalkForward(rd, lambda p: SMACrossover(**p), grid, make_run_factory(CFG),
                     train_years=2, test_years=1, holdout_start=hs, min_train_obs=50)
    return wf.run(), rd, hs


def test_trial_log_records_every_configuration(wf_result):
    res, rd, hs = wf_result
    t = res.trials
    n_folds = len(res.folds)
    assert len(t) == n_folds * 6
    assert (t["status"] == "invalid").sum() == n_folds * 2
    assert t.groupby("fold")["selected"].sum().max() == 1
    # no trial used data at or after its fold's test start / the hold-out
    for _, row in t.iterrows():
        f = res.folds.iloc[row["fold"]]
        assert pd.Timestamp(row["train_end"]) < f["test_start"]
        assert pd.Timestamp(row["train_end"]) < hs


def test_selection_uses_training_only_with_tie_break(wf_result):
    res, _, _ = wf_result
    for _fold, g in res.trials[res.trials["status"] == "ok"].groupby("fold"):
        sel = g[g["selected"]].iloc[0]
        best = g["objective_value"].max()
        assert sel["objective_value"] == best
        ties = g[g["objective_value"] == best]
        assert sel["params"] == sorted(ties["params"], key=lambda s: tuple(
            sorted(json.loads(s).items())))[0]


def test_oos_is_one_continuous_run_with_transition_costs(wf_result):
    res, rd, _ = wf_result
    r = res.oos_returns
    assert r.index.is_unique and r.index.is_monotonic_increasing
    assert r.index[0] == res.folds.iloc[0]["test_start"]
    eq = res.oos.equity
    # no trading or fills before the first test window (warm-up only)
    assert (res.oos.fills["timestamp"] >= res.folds.iloc[0]["test_start"]).all()
    # portfolio state carries over boundaries: no reset to initial cash at fold starts
    for ts in res.folds["test_start"].iloc[1:]:
        pos = eq.index.get_loc(ts)
        assert eq["equity"].iloc[pos - 1] != 1_000_000
    assert res.oos.cost_summary()["spread_costs"] > 0
    assert len(res.fold_metrics) == len(res.folds)
    assert not res.dsr.empty and res.dsr["dsr"].between(0, 1).all()


def test_scheduled_strategy_switches_on_next_session(data):
    a, b = SMACrossover(5, 20), SMACrossover(10, 40)
    s = ScheduledStrategy([(data.index[100], a), (data.index[200], None), (data.index[50], b)])
    run = make_run_factory(CFG)
    res = run(data.slice(end=data.index[300]), s, data.index[50])
    assert res.equity.loc[data.index[210]:, "n_positions"].max() == 0  # None segment -> cash


def test_holdout_evaluation_requires_confirmation_and_is_logged(tmp_path, data):
    hs = data.index[-252]
    run = make_run_factory(CFG)
    log = tmp_path / "holdout.jsonl"
    with pytest.raises(HoldoutViolation):
        evaluate_holdout(data, hs, SMACrossover(5, 20), run, confirm=False, log_path=log,
                         dataset="t")
    res = evaluate_holdout(data, hs, SMACrossover(5, 20), run, confirm=True, log_path=log,
                           dataset="t")
    assert res.returns().index[0] == hs
    assert res.meta["holdout_evaluation_number"] == 1
    lines = log.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["evaluation_number"] == 1


def test_holdout_lock(tmp_path):
    p = tmp_path / "lock.json"
    check_holdout_lock(p, "2022-01-01", "ds")
    check_holdout_lock(p, "2022-01-01", "ds")
    with pytest.raises(HoldoutViolation):
        check_holdout_lock(p, "2021-01-01", "ds")


def test_trial_registry_roundtrip(tmp_path, wf_result):
    res, _, _ = wf_result
    reg = TrialRegistry(tmp_path / "trials.jsonl")
    reg.append(res.trials)
    assert len(reg.read()) == len(res.trials)
    assert TrialRegistry(tmp_path / "none.jsonl").read().empty


def test_reproducibility_with_stochastic_slippage(data):
    cfg = {"costs": {"slippage": {"bps": 2, "stochastic_sd_bps": 5}}, "seed": 3}
    run = make_run_factory(cfg)
    r1 = run(data.slice(end=data.index[400]), SMACrossover(5, 20), data.index[50])
    r2 = run(data.slice(end=data.index[400]), SMACrossover(5, 20), data.index[50])
    pd.testing.assert_frame_equal(r1.equity, r2.equity)
    assert CostModel.from_dict(cfg["costs"], seed=3).slippage.stochastic_sd_bps == 5
