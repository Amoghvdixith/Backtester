"""End-to-end experiment pipeline on a small temporary project (slow integration test)."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from backtester import experiments as ex
from backtester.cli import main
from backtester.synthetic import SyntheticSpec, write_dataset
from backtester.walkforward import HoldoutViolation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    shutil.copytree(ROOT / "configs", root / "configs")
    wf = yaml.safe_load((root / "configs" / "walkforward.yaml").read_text())
    wf["train_years"] = 2
    wf["grids"] = {"sma_crossover": {"fast": [10, 50], "slow": [50, 100]},
                   "momentum_12_1": {"lookback": [126], "skip": [21], "quantile": [0.2]}}
    (root / "configs" / "walkforward.yaml").write_text(yaml.safe_dump(wf))
    cs = yaml.safe_load((root / "configs" / "cost_scenarios.yaml").read_text())
    cs["one_way_bps"] = [0, 20]
    cs["robustness"] = {"leverage_caps": [1.0], "execution_delay_sessions": [1, 2],
                        "sizers": [{"type": "equal_weight"}], "train_years": [2]}
    (root / "configs" / "cost_scenarios.yaml").write_text(yaml.safe_dump(cs))
    syn = root / "data" / "synthetic"
    write_dataset(SyntheticSpec(n_assets=3, start="2016-01-01", end="2023-12-31", seed=1,
                                prefix="ETF", late_listing_frac=0, delist_frac=0), syn,
                  "etf_like")
    write_dataset(SyntheticSpec(n_assets=12, start="2016-01-01", end="2023-12-31", seed=2,
                                prefix="STK"), syn, "stock_universe")
    return root


@pytest.mark.slow
def test_full_pipeline(project):
    main(["--root", str(project), "run-all"])
    meta = json.loads((project / "results" / "run_metadata.json").read_text())
    assert meta["experiments"]["holdout_consumed_by_this_run"] is False
    assert meta["experiments"]["buy_and_hold_validation"]["max_abs_diff_vs_reference"] < 1e-6
    figs = {p.name for p in (project / "results" / "figures").glob("*.png")}
    for f in ("01_architecture_event_flow.png", "02_buy_and_hold_validation.png",
              "03_sma_tearsheet.png", "04_momentum_tearsheet.png",
              "05_sma_wf_stitched_oos_equity.png", "05_sma_wf_param_heatmap_training.png",
              "12_cost_sensitivity.png"):
        assert f in figs
    trials = pd.read_csv(project / "results" / "tables" / "sma_wf_trial_log.csv")
    assert (trials["train_end"] < "2022-01-01").all()  # hold-out never searched
    assert not (project / "results" / "logs" / "holdout_runs.jsonl").exists()
    # protected hold-out: refused without confirmation, dry run evaluates nothing
    with pytest.raises(SystemExit):
        main(["--root", str(project), "holdout"])
    dry = main(["--root", str(project), "holdout", "--dry-run"])
    assert "frozen_params" in dry["sma"]
    assert not (project / "results" / "logs" / "holdout_runs.jsonl").exists()
    out = main(["--root", str(project), "holdout", "--confirm"])
    assert out["sma"]["evaluation_number"] == 1
    assert (project / "results" / "logs" / "holdout_runs.jsonl").exists()


@pytest.mark.slow
def test_holdout_lock_blocks_moved_boundary(project):
    ctx = ex.Context(project)
    ctx.load(ctx.config("sma"))  # creates (or verifies) the lock at the configured date
    ctx.holdout_start = pd.Timestamp("2021-01-01")
    with pytest.raises(HoldoutViolation):
        ctx.load(ctx.config("sma"))
    with pytest.raises(FileNotFoundError):
        ex.project_root(Path(project).anchor)
