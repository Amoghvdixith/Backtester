"""YAML configuration loading and component construction (dependency injection)."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .costs import CostModel, FinancingModel
from .data import MarketData
from .engine import Backtest, BacktestConfig
from .execution import ExecutionConfig
from .results import BacktestResult
from .sizing import PortfolioConstraints, build_sizer
from .strategy import Strategy
from .universe import MembershipUniverse, StaticUniverse, Universe


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_dataset(cfg: dict[str, Any], root: str | Path) -> tuple[MarketData, Universe, str]:
    """Load prices (and optional membership) described by ``cfg['data']``.

    Never substitutes synthetic data for a missing market file.
    """
    d = cfg["data"]
    root = Path(root)
    mode = d.get("mode", "synthetic")
    path = root / d["path"]
    if not path.exists():
        hint = ("run scripts/generate_synthetic_data.py" if mode == "synthetic"
                else "run scripts/download_market_data.py (requires network)")
        raise FileNotFoundError(f"{path} not found; {hint}")
    data = MarketData.from_csv(path)
    data.provenance.setdefault("data_mode", mode)
    if d.get("symbols"):
        data = data.select(d["symbols"])
    if d.get("start") or d.get("end"):
        data = data.slice(d.get("start"), d.get("end"))
    universe: Universe
    if d.get("membership"):
        mem = pd.read_csv(root / d["membership"])
        if d.get("symbols"):
            mem = mem[mem["symbol"].isin(d["symbols"])]
        universe = MembershipUniverse(mem)
    else:
        universe = StaticUniverse(data.symbols)
    return data, universe, mode


def build_engine_parts(cfg: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    seed = cfg.get("seed", 0) if seed is None else seed
    bt = dict(cfg.get("backtest", {}))
    return {
        "sizer": build_sizer(cfg.get("sizing")),
        "constraints": PortfolioConstraints(**cfg.get("constraints", {})),
        "costs": CostModel.from_dict(cfg.get("costs"), seed=seed),
        "execution": ExecutionConfig(**cfg.get("execution", {})),
        "financing": FinancingModel(**cfg.get("financing", {})),
        "config": BacktestConfig(**bt),
    }


def make_run_factory(cfg: dict[str, Any], universe: Universe | None = None,
                     data_mode: str = "unspecified", label: str = ""):
    """Return f(data, strategy, trade_start) -> BacktestResult with fresh components.

    Fresh components per run keep stochastic slippage reproducible (same seed each run).
    """

    def run(data: MarketData, strategy: Strategy, trade_start: pd.Timestamp | None
            ) -> BacktestResult:
        parts = build_engine_parts(cfg)
        conf = replace(parts.pop("config"), trade_start=trade_start)
        return Backtest(data, strategy, universe=universe, config=conf,
                        label=label or strategy.name,
                        data_mode=data_mode, **parts).run()

    return run
