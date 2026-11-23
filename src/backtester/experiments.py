"""Reproducible demonstration experiments (used by scripts/ and notebooks/).

Every experiment:
* uses only data BEFORE the protected hold-out (``research_data``), except
  :func:`holdout_evaluation`, which requires explicit confirmation;
* writes CSV tables to ``results/tables``, PNG figures to ``results/figures`` and
  the numbers behind each figure to ``results/tables/figure_data``;
* returns a metadata dict that ``run_all`` stores in ``results/run_metadata.json``.
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import plots
from .config import deep_update, load_dataset, load_yaml, make_run_factory
from .data import MarketData
from .performance import compute_metrics, metrics_frame, rf_from_annual
from .plots import Stamp
from .results import BacktestResult, environment_info
from .statistics import (
    circular_block_bootstrap_sharpe,
    sharpe_se_iid_normal,
    sharpe_se_nonnormal,
)
from .strategies import build_strategy
from .synthetic import SyntheticSpec, write_dataset
from .universe import Universe
from .walkforward import (
    TrialRegistry,
    WalkForward,
    check_holdout_lock,
    evaluate_holdout,
    research_data,
)

DATASETS = {
    "etf_like": SyntheticSpec(n_assets=6, start="2005-01-01", end="2024-12-31", seed=101,
                              prefix="ETF", late_listing_frac=0.0, delist_frac=0.0,
                              halt_prob=0.0002),
    "stock_universe": SyntheticSpec(n_assets=40, start="2005-01-01", end="2024-12-31",
                                    seed=202, prefix="STK"),
    "stock_universe_large": SyntheticSpec(n_assets=150, start="2005-01-01", end="2024-12-31",
                                          seed=303, prefix="STK"),
}


def project_root(start: str | Path | None = None) -> Path:
    """Locate the project folder (the one containing configs/ and pyproject.toml)."""
    p = Path(start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").exists() and (cand / "configs").is_dir():
            return cand
    raise FileNotFoundError("could not find the backtester project root")


def ensure_datasets(root: Path, names: tuple[str, ...] = ("etf_like", "stock_universe")
                    ) -> dict[str, Any]:
    out = {}
    for n in names:
        if not (root / "data" / "synthetic" / f"{n}.csv").exists():
            write_dataset(DATASETS[n], root / "data" / "synthetic", n)
        out[n] = DATASETS[n].__dict__
    return out


class Context:
    """Paths, configs and the hold-out boundary shared by experiments."""

    def __init__(self, root: str | Path | None = None, research: bool = False) -> None:
        self.root = project_root(root)
        self.research = research
        # research-mode outputs are kept apart from the fast offline demo
        self.results = self.root / "results" / ("research" if research else "")
        self.figs = self.results / "figures"
        self.tables = self.results / "tables"
        self.logs = self.root / "results" / "logs" / ("research" if research else "")
        self.shared_logs = self.root / "results" / "logs"  # hold-out lock + evaluation log
        for p in (self.figs, self.tables, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        self.wf = load_yaml(self.root / "configs" / "walkforward.yaml")
        self.holdout_start = pd.Timestamp(self.wf["holdout_start"])

    def config(self, name: str) -> dict[str, Any]:
        cfg = load_yaml(self.root / "configs" / f"{name}.yaml")
        if self.research and name == "momentum":
            cfg = deep_update(cfg, {"data": {
                "path": "data/synthetic/stock_universe_large.csv",
                "membership": "data/synthetic/stock_universe_large_membership.csv"}})
        return cfg

    def load(self, cfg: dict[str, Any]) -> tuple[MarketData, MarketData, Universe, str]:
        """Return (full data, research data before the hold-out, universe, data mode)."""
        data, uni, mode = load_dataset(cfg, self.root)
        check_holdout_lock(self.shared_logs / "holdout_lock.json", self.holdout_start,
                           cfg["data"]["path"])
        return data, research_data(data, self.holdout_start), uni, mode

    def grid(self, strategy: str) -> dict[str, list[Any]]:
        if self.research:
            return self.wf["research_mode"]["grids"][strategy]
        return self.wf["grids"][strategy]


def _range(r: pd.Series) -> str:
    return f"{r.index[0].date()} to {r.index[-1].date()}" if len(r) else "empty"


def _cost_label(cfg: dict[str, Any]) -> str:
    c = cfg.get("costs", {})
    if "proportional_bps" in c:
        return f"flat {c['proportional_bps']} bps one-way"
    return (f"comm {c.get('commission', {}).get('rate_bps', 0)}bp, spread "
            f"{c.get('spread', {}).get('spread_bps', 0)}bp full, slip "
            f"{c.get('slippage', {}).get('bps', 0)}bp, sqrt-impact c="
            f"{c.get('impact', {}).get('coefficient', 0)}")


def _rf(cfg: dict[str, Any], r: pd.Series) -> pd.Series:
    return rf_from_annual(pd.DatetimeIndex(r.index), float(cfg.get("risk_free_annual", 0.0)))


def sharpe_uncertainty(r: pd.Series, rf: pd.Series | None = None, seed: int = 0
                       ) -> dict[str, float]:
    x = r - (rf if rf is not None else 0.0)
    m = compute_metrics(x)
    sr = m["sharpe_per_period"]
    n = m["n_periods"]
    ann = math.sqrt(252)
    boot = circular_block_bootstrap_sharpe(x, block=20, n_boot=1000, seed=seed)
    return {
        "sharpe_annualised": m["sharpe"], "n_obs": n,
        "se_iid_normal_annualised": sharpe_se_iid_normal(sr, n) * ann,
        "se_iid_nonnormal_annualised": sharpe_se_nonnormal(sr, n, m["skew"],
                                                           m["kurtosis_pearson"]) * ann,
        "se_block_bootstrap_annualised": boot["se"], "bootstrap_ci_low": boot["ci_low"],
        "bootstrap_ci_high": boot["ci_high"], "bootstrap_block": boot["block"],
    }


# ---------------------------------------------------------------------- buy and hold
def buy_and_hold_validation(ctx: Context) -> dict[str, Any]:
    """Engine buy-and-hold vs an independent pandas reference with the same timing."""
    cfg = deep_update(ctx.config("sma"), {"costs": {"proportional_bps": 0},
                                          "financing": {"credit_rate": 0.0},
                                          "constraints": {"max_position": 1.0},
                                          "backtest": {"fractional": True, "cash_buffer": 0.0}})
    _, rd, _, mode = ctx.load(cfg)
    sym = rd.symbols[0]
    d1 = rd.select([sym])
    run = make_run_factory(cfg, None, mode, "buy_and_hold")
    res = run(d1, build_strategy("buy_and_hold", symbols=[sym]), None)
    o, c = d1.frame("open")[sym], d1.frame("close")[sym]
    if pd.isna(c.iloc[0]) or pd.isna(o.iloc[1]):
        raise ValueError("validation symbol must trade on the first two sessions")
    c = c.ffill()  # valuation policy on halted days: last close (never used for fills)
    cash = float(cfg["backtest"]["initial_cash"])
    shares = min(cash / c.iloc[0], cash / o.iloc[1])
    ref = shares * c + (cash - shares * o.iloc[1])
    ref.iloc[0] = cash
    naive = cash * c / c.iloc[0]
    eng = res.equity["equity"]
    diff = (eng - ref).abs()
    tbl = pd.DataFrame({"engine_equity": eng, "reference_equity": ref,
                        "abs_difference": diff, "naive_first_close_reference": naive})
    tbl.to_csv(ctx.tables / "buy_and_hold_validation.csv")
    stamp = Stamp(f"Buy & hold {sym}", mode, _range(eng), costs="zero",
                  sample="development (pre-hold-out); validation only")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    axs[0].plot(eng.index, eng, color=plots.C1, label="engine (next-open fill)")
    axs[0].plot(ref.index, ref, color=plots.C2, linestyle="--",
                label="independent reference (same fill)")
    axs[0].plot(naive.index, naive, color=plots.C3, linewidth=1.0,
                label="naive cash*close/close[0] (incompatible timing)")
    axs[0].set_title("Buy-and-hold validation: equity")
    axs[0].legend()
    axs[1].plot(diff.index, diff, color=plots.C1, label="|engine - reference|")
    axs[1].plot(diff.index, (eng - naive).abs(), color=plots.C3, linewidth=1.0,
                label="|engine - naive|")
    axs[1].set_yscale("symlog", linthresh=1e-9)
    axs[1].set_title("Absolute difference (currency units, symlog)")
    axs[1].legend()
    plots._finish(fig, ctx.figs / "02_buy_and_hold_validation.png", stamp, tbl)
    out = {"symbol": sym, "max_abs_diff_vs_reference": float(diff.max()),
           "max_rel_diff_vs_reference": float((eng / ref - 1).abs().max()),
           "max_abs_diff_vs_naive_first_close": float((eng - naive).abs().max()),
           "n_sessions": len(eng)}
    (ctx.tables / "buy_and_hold_validation_summary.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------- single strategy
def strategy_demo(ctx: Context, name: str, fig_prefix: str) -> dict[str, Any]:
    """Fixed-parameter run over the development sample (pre-hold-out) with benchmark."""
    cfg = ctx.config(name)
    _, rd, uni, mode = ctx.load(cfg)
    run = make_run_factory(cfg, uni, mode)
    strat = build_strategy(cfg["strategy"]["name"], **cfg["strategy"]["params"])
    t0 = time.perf_counter()
    res = run(rd, strat, None)
    bench_res = run(rd, build_strategy(cfg["benchmark"]["name"], **cfg["benchmark"]["params"]),
                    None)
    # score from the first session with a non-zero position in either book
    first = min(res.equity.index[res.equity["gross_exposure"] > 0][0],
                bench_res.equity.index[bench_res.equity["gross_exposure"] > 0][0])
    r = res.returns(first)
    b = bench_res.returns(first)
    rf = _rf(cfg, r)
    m = compute_metrics(r, rf=rf, benchmark=b, equity_frame=res.equity,
                        round_trips=res.round_trips)
    mb = compute_metrics(b, rf=rf, equity_frame=bench_res.equity,
                         round_trips=bench_res.round_trips)
    label = f"{cfg['strategy']['name']} {cfg['strategy']['params']}"
    tbl = metrics_frame({name: m, f"{name}_benchmark": mb})
    tbl.to_csv(ctx.tables / f"{name}_metrics.csv")
    unc = sharpe_uncertainty(r, rf, seed=cfg.get("seed", 0))
    pd.Series(unc).to_csv(ctx.tables / f"{name}_sharpe_uncertainty.csv", header=["value"])
    res.save(ctx.logs / name, prefix=name)
    sample = "development sample before hold-out; fixed a-priori parameters (not OOS)"
    stamp = Stamp(label, mode, _range(r), costs=_cost_label(cfg), sample=sample)
    title = f"{name} vs equal-weight benchmark"
    plots.tearsheet(r, ctx.figs / f"{fig_prefix}_tearsheet.png", stamp, bench=b,
                    title=f"Tearsheet: {title}")
    plots.single("equity", r, ctx.figs / f"{fig_prefix}_equity_vs_benchmark.png", stamp,
                 bench=b, title=f"{title}: growth of 1 (log)")
    plots.single("drawdown", r, ctx.figs / f"{fig_prefix}_drawdown.png", stamp, bench=b)
    plots.single("monthly", r, ctx.figs / f"{fig_prefix}_monthly_returns.png", stamp)
    plots.single("rolling_sharpe", r, ctx.figs / f"{fig_prefix}_rolling_sharpe.png", stamp)
    plots.single("annual", r, ctx.figs / f"{fig_prefix}_annual_returns.png", stamp, bench=b)
    plots.single("distribution", r, ctx.figs / f"{fig_prefix}_return_distribution.png", stamp)
    plots.exposure(res.equity.loc[res.equity.index >= first],
                   ctx.figs / f"{fig_prefix}_exposure.png", stamp)
    plots.turnover_and_costs(res.equity.loc[res.equity.index >= first],
                             ctx.figs / f"{fig_prefix}_turnover_costs.png", stamp)
    audit = _audit_example(res)
    audit.to_csv(ctx.tables / f"{name}_order_fill_audit_example.csv", index=False)
    return {"metrics": {k: v for k, v in m.items() if k != "warnings"},
            "benchmark_metrics": {k: v for k, v in mb.items() if k != "warnings"},
            "warnings": m["warnings"], "sharpe_uncertainty": unc,
            "order_status_counts": res.order_status["status"].value_counts().to_dict(),
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "scored_range": _range(r), "data_mode": mode, "sample": sample}


def _audit_example(res: BacktestResult, n_orders: int = 12) -> pd.DataFrame:
    """Join orders, status changes, fills and per-fill reconciliation for the first orders."""
    orders = res.orders.head(n_orders)
    st = res.order_status[res.order_status["order_id"].isin(orders["order_id"])]
    final = st.groupby("order_id").last()[["status", "reason"]].rename(
        columns={"status": "final_status", "reason": "status_reason"})
    rec = res.reconciliation()
    cols = ["order_id", "timestamp", "fill_price", "reference_price", "execution_shortfall",
            "spread_cost", "slippage_cost", "impact_cost", "commission",
            "delay_cost_vs_decision"]
    rec = rec[cols].rename(columns={"timestamp": "fill_time"}) if len(rec) else pd.DataFrame(
        columns=cols)
    out = orders[["order_id", "symbol", "created_at", "eligible_from", "quantity",
                  "reference_price", "reason"]].rename(
        columns={"reference_price": "decision_price", "quantity": "order_quantity"})
    out = out.merge(final, left_on="order_id", right_index=True, how="left")
    fq = res.fills[["order_id", "quantity"]].rename(columns={"quantity": "filled_quantity"}) \
        if len(res.fills) else pd.DataFrame(columns=["order_id", "filled_quantity"])
    out = out.merge(fq, on="order_id", how="left").merge(rec, on="order_id", how="left")
    return out


# ---------------------------------------------------------------------- walk-forward
def walkforward_experiment(ctx: Context, name: str, fig_prefix: str,
                           train_years: int | None = None, tag: str = "",
                           make_figures: bool = True) -> dict[str, Any]:
    cfg = ctx.config(name)
    _, rd, uni, mode = ctx.load(cfg)
    run = make_run_factory(cfg, uni, mode)
    sname = cfg["strategy"]["name"]
    fixed = {k: v for k, v in cfg["strategy"]["params"].items()}
    grid = ctx.grid(sname)

    def factory(p: dict[str, Any]):
        return build_strategy(sname, **{**fixed, **p})

    ty = train_years or int(ctx.wf["train_years"])
    exp_name = f"{name}_wf{tag}"
    wf = WalkForward(rd, factory, grid, run, train_years=ty,
                     test_years=int(ctx.wf["test_years"]), step_years=int(ctx.wf["step_years"]),
                     objective=ctx.wf["objective"], holdout_start=ctx.holdout_start,
                     experiment=exp_name, min_train_obs=int(ctx.wf["min_train_obs"]))
    res = wf.run()
    TrialRegistry(ctx.logs / "trial_registry.jsonl").append(res.trials)
    res.trials.to_csv(ctx.tables / f"{exp_name}_trial_log.csv", index=False)
    res.fold_metrics.to_csv(ctx.tables / f"{exp_name}_fold_summary.csv", index=False)
    res.dsr.to_csv(ctx.tables / f"{exp_name}_deflated_sharpe.csv", index=False)
    out: dict[str, Any] = {"meta": res.meta, "n_folds": len(res.folds)}
    if not make_figures:
        m = compute_metrics(res.oos_returns)
        out["oos_metrics"] = {k: m[k] for k in ("sharpe", "cagr", "max_drawdown", "n_periods",
                                                "start", "end")}
        out["oos_returns"] = res.oos_returns
        return out
    # benchmark over the same stitched OOS dates, same engine and costs
    bench = run(rd, build_strategy(cfg["benchmark"]["name"], **cfg["benchmark"]["params"]),
                res.folds.iloc[0]["test_start"])
    b = bench.returns(res.oos_returns.index[0], res.oos_returns.index[-1])
    r = res.oos_returns
    rf = _rf(cfg, r)
    m = compute_metrics(r, rf=rf, benchmark=b, equity_frame=res.oos.equity,
                        round_trips=res.oos.round_trips)
    mb = compute_metrics(b, rf=rf)
    metrics_frame({f"{exp_name}_stitched_oos": m, "benchmark_same_dates": mb}).to_csv(
        ctx.tables / f"{exp_name}_oos_metrics.csv")
    unc = sharpe_uncertainty(r, rf)
    pd.Series(unc).to_csv(ctx.tables / f"{exp_name}_oos_sharpe_uncertainty.csv",
                          header=["value"])
    res.oos.save(ctx.logs / exp_name, prefix="oos")
    stamp = Stamp(f"{sname} walk-forward ({ty}y train / {ctx.wf['test_years']}y test)", mode,
                  _range(r), costs=_cost_label(cfg),
                  sample="stitched walk-forward OUT-OF-SAMPLE (hold-out untouched)")
    spans = [(pd.Timestamp(str(a)), pd.Timestamp(str(b)), f"F{k}") for k, a, b in zip(
        res.fold_metrics["fold"], res.fold_metrics["test_start"], res.fold_metrics["test_end"],
        strict=True)]
    wealth = {"walk-forward OOS": (1 + r).cumprod(), "benchmark": (1 + b).cumprod()}
    plots.lines(wealth, ctx.figs / f"{fig_prefix}_stitched_oos_equity.png", stamp,
                f"{sname}: stitched out-of-sample equity (shaded = test folds)",
                ylabel="growth of 1", log=True, spans=spans)
    plots.tearsheet(r, ctx.figs / f"{fig_prefix}_oos_tearsheet.png", stamp, bench=b,
                    title=f"Tearsheet: {sname} walk-forward OOS",
                    shade=(r.index[0], r.index[-1]))
    fm = res.fold_metrics.set_index("fold")
    plots.bars(fm, ctx.figs / f"{fig_prefix}_fold_summary.png", stamp,
               f"{sname}: per-fold test results (selected on training only)",
               ["test_return", "test_sharpe", "train_objective"], pct=[True, False, False])
    out.update({"oos_metrics": {k: v for k, v in m.items() if k != "warnings"},
                "benchmark_metrics": {k: v for k, v in mb.items() if k != "warnings"},
                "sharpe_uncertainty": unc,
                "dsr": res.dsr.to_dict(orient="records"),
                "selected_params": res.fold_metrics[["fold", "selected_params"]]
                .to_dict(orient="records")})
    # training-window heatmap (in-sample) for two-parameter grids
    ok = res.trials[res.trials["status"] == "ok"].copy()
    keys = sorted(grid)
    if sname == "sma_crossover" and len(ok):
        ok["fast"] = ok["params"].map(lambda s: json.loads(s)["fast"])
        ok["slow"] = ok["params"].map(lambda s: json.loads(s)["slow"])
        hm = ok.pivot_table(index="fast", columns="slow", values="objective_value",
                            aggfunc="mean").reindex(index=grid["fast"], columns=grid["slow"])
        hm.to_csv(ctx.tables / f"{exp_name}_training_sharpe_heatmap.csv")
        plots.heatmap(hm, ctx.figs / f"{fig_prefix}_param_heatmap_training.png",
                      Stamp(f"{sname} grid", mode, f"training windows {res.folds.iloc[0]['train_start'].date()} to {res.folds.iloc[-1]['train_end'].date()}",
                            costs=_cost_label(cfg), sample="TRAINING windows (in-sample), mean over folds"),
                      "Mean training Sharpe across folds (fast x slow)", "slow window",
                      "fast window")
        out["training_heatmap"] = hm.round(3).to_dict()
        # exploratory: every grid point over the outer-test period (NOT used for selection)
        rows = []
        for p in wf.params:
            try:
                s = factory(p)
            except ValueError:
                rows.append({**p, "sharpe": np.nan})
                continue
            rr = run(rd, s, res.folds.iloc[0]["test_start"]).returns()
            rows.append({**p, "sharpe": compute_metrics(rr)["sharpe"]})
        ex = pd.DataFrame(rows).pivot(index="fast", columns="slow", values="sharpe")
        ex = ex.reindex(index=grid["fast"], columns=grid["slow"])
        ex.to_csv(ctx.tables / f"{exp_name}_EXPLORATORY_outer_test_heatmap.csv")
        plots.heatmap(ex, ctx.figs / f"{fig_prefix}_param_heatmap_EXPLORATORY_outer_test.png",
                      Stamp(f"{sname} grid", mode, _range(r), costs=_cost_label(cfg),
                            sample="EXPLORATORY outer-test period, all params; not used for "
                                   "selection; do not retune on it"),
                      "EXPLORATORY: Sharpe over the outer-test period for every grid point",
                      "slow window", "fast window")
        TrialRegistry(ctx.logs / "trial_registry.jsonl").append(pd.DataFrame(
            [{"experiment": f"{exp_name}_exploratory_outer_test", "params": json.dumps(
                {k: r_[k] for k in keys}), "sharpe": r_["sharpe"],
              "sample": "outer-test (exploratory)"} for r_ in rows]))
        out["exploratory_outer_test_heatmap"] = ex.round(3).to_dict()
    return out


# ---------------------------------------------------------------------- costs and robustness
def cost_sensitivity(ctx: Context) -> dict[str, Any]:
    sc = load_yaml(ctx.root / "configs" / "cost_scenarios.yaml")
    rows = []
    for name in ("sma", "momentum"):
        base = ctx.config(name)
        _, rd, uni, mode = ctx.load(base)
        strat_cfg = base["strategy"]
        for bps in sc["one_way_bps"]:
            cfg = deep_update({k: v for k, v in base.items() if k != "costs"},
                              {"costs": {"proportional_bps": bps}})
            res = make_run_factory(cfg, uni, mode)(
                rd, build_strategy(strat_cfg["name"], **strat_cfg["params"]), None)
            first = res.equity.index[res.equity["gross_exposure"] > 0][0]
            r = res.returns(first)
            m = compute_metrics(r, rf=_rf(cfg, r), equity_frame=res.equity,
                                round_trips=res.round_trips)
            rows.append({"strategy": name, "one_way_bps": bps, "cagr": m["cagr"],
                         "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                         "turnover_annual": m["turnover_annual"],
                         "total_trading_costs": m["total_trading_costs"],
                         "net_financing": m["net_financing"],
                         "date_range": _range(r)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(ctx.tables / "cost_sensitivity.csv", index=False)
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for (sname, g), col in zip(tbl.groupby("strategy"), (plots.C1, plots.C2), strict=False):
        for ax, k in zip(axs, ("sharpe", "cagr", "max_drawdown"), strict=True):
            ax.plot(g["one_way_bps"], g[k], marker="o", color=col, label=str(sname))
            ax.set_title(k)
            ax.set_xlabel("one-way cost (bps)")
    axs[0].legend()
    fig.suptitle("Cost sensitivity (flat cost replaces detailed model; params & dates fixed)",
                 fontweight="bold")
    plots._finish(fig, ctx.figs / "12_cost_sensitivity.png",
                  Stamp("SMA 50/200 & momentum 12-1 (fixed params)", "synthetic",
                        "development sample before hold-out", costs="0/5/10/20/50 bps",
                        sample="development sample (not OOS)"), tbl)
    return {"table": tbl.to_dict(orient="records")}


def robustness(ctx: Context) -> dict[str, Any]:
    sc = load_yaml(ctx.root / "configs" / "cost_scenarios.yaml")["robustness"]
    base = ctx.config("sma")
    _, rd, uni, mode = ctx.load(base)
    sp = base["strategy"]["params"]
    rows: list[dict[str, Any]] = []

    def one(label: str, value: Any, cfg: dict[str, Any]) -> None:
        res = make_run_factory(cfg, uni, mode)(rd, build_strategy("sma_crossover", **sp), None)
        first = res.equity.index[res.equity["gross_exposure"] > 0][0]
        r = res.returns(first)
        m = compute_metrics(r, rf=_rf(cfg, r), equity_frame=res.equity,
                            round_trips=res.round_trips)
        rows.append({"experiment": label, "setting": str(value), "cagr": m["cagr"],
                     "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "ann_vol": m["ann_vol"], "avg_gross_exposure": m["avg_gross_exposure"],
                     "turnover_annual": m["turnover_annual"],
                     "total_trading_costs": m["total_trading_costs"]})

    for cap in sc["leverage_caps"]:
        no_sizing = {k: v for k, v in base.items() if k != "sizing"}
        one("leverage_cap (fixed_fraction 0.5/asset)", cap, deep_update(no_sizing, {
            "sizing": {"type": "fixed_fraction", "fraction": 0.5},
            "constraints": {"max_gross": cap, "max_net": cap, "max_position": 1.0}}))
    for dly in sc["execution_delay_sessions"]:
        one("execution_delay_sessions", dly, deep_update(base, {"backtest": {
            "execution_delay": dly}}))
    for sz in sc["sizers"]:
        one("sizing_rule", sz["type"], deep_update({k: v for k, v in base.items()
                                                    if k != "sizing"}, {"sizing": sz}))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(ctx.tables / "robustness_sma.csv", index=False)
    # training-window sensitivity: compare stitched OOS on COMMON dates
    wf_rows = []
    series = {}
    for ty in sc["train_years"]:
        o = walkforward_experiment(ctx, "sma", "", train_years=ty, tag=f"_train{ty}y",
                                   make_figures=False)
        series[ty] = o["oos_returns"]
    common_start = max(s.index[0] for s in series.values())
    for ty, s in series.items():
        s2 = s[s.index >= common_start]
        m = compute_metrics(s2)
        wf_rows.append({"experiment": "walk-forward train_years (common OOS dates)",
                        "setting": ty, "cagr": m["cagr"], "sharpe": m["sharpe"],
                        "max_drawdown": m["max_drawdown"], "ann_vol": m["ann_vol"],
                        "date_range": _range(s2)})
    wtbl = pd.DataFrame(wf_rows)
    wtbl.to_csv(ctx.tables / "robustness_sma_train_window.csv", index=False)
    return {"table": tbl.to_dict(orient="records"),
            "train_window": wtbl.to_dict(orient="records")}


# ---------------------------------------------------------------------- hold-out
def holdout_evaluation(ctx: Context, confirm: bool, dry_run: bool = False) -> dict[str, Any]:
    """Evaluate the last walk-forward selections on the protected hold-out."""
    out: dict[str, Any] = {}
    for name in ("sma", "momentum"):
        cfg = ctx.config(name)
        folds = ctx.tables / f"{name}_wf_fold_summary.csv"
        if not folds.exists():
            raise FileNotFoundError(f"{folds} missing: run the walk-forward first")
        fs = pd.read_csv(folds)
        params = json.loads(fs.iloc[-1]["selected_params"])
        if params is None:
            out[name] = {"status": "no parameters selected in last fold; skipped"}
            continue
        sp = {**cfg["strategy"]["params"], **params}
        out[name] = {"frozen_params": sp, "source": folds.name}
        if dry_run:
            continue
        data, _, uni, mode = ctx.load(cfg)
        run = make_run_factory(cfg, uni, mode)
        res = evaluate_holdout(data, ctx.holdout_start,
                               build_strategy(cfg["strategy"]["name"], **sp), run,
                               confirm=confirm, log_path=ctx.shared_logs / "holdout_runs.jsonl",
                               dataset=cfg["data"]["path"])
        r = res.returns()
        m = compute_metrics(r, rf=_rf(cfg, r), equity_frame=res.equity,
                            round_trips=res.round_trips)
        n = res.meta["holdout_evaluation_number"]
        metrics_frame({f"{name}_holdout": m}).to_csv(
            ctx.tables / f"HOLDOUT_{name}_metrics_eval{n}.csv")
        plots.tearsheet(r, ctx.figs / f"HOLDOUT_{name}_tearsheet_eval{n}.png",
                        Stamp(f"{cfg['strategy']['name']} {sp}", mode, _range(r),
                              costs=_cost_label(cfg),
                              sample=f"FINAL HOLD-OUT, evaluation #{n}"),
                        title=f"HOLD-OUT evaluation #{n}: {name}")
        out[name].update({"evaluation_number": n, "sharpe": m["sharpe"], "cagr": m["cagr"],
                          "max_drawdown": m["max_drawdown"]})
    return out


def write_run_metadata(ctx: Context, entries: dict[str, Any], datasets: dict[str, Any]) -> Path:
    from .walkforward import holdout_run_count

    meta = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "research" if ctx.research else "fast",
        "environment": environment_info(),
        "holdout_start": str(ctx.holdout_start.date()),
        "holdout_evaluations_logged": {
            ds: holdout_run_count(ctx.shared_logs / "holdout_runs.jsonl", ds)
            for ds in ("data/synthetic/etf_like.csv", "data/synthetic/stock_universe.csv",
                       "data/synthetic/stock_universe_large.csv")},
        "datasets": datasets,
        "configs": {n: load_yaml(ctx.root / "configs" / f"{n}.yaml")
                    for n in ("sma", "momentum", "walkforward", "cost_scenarios")},
        "experiments": entries,
    }

    def clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list | tuple):
            return [clean(v) for v in o]
        if isinstance(o, pd.Series):
            return None
        if isinstance(o, float | np.floating):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, pd.Timestamp):
            return str(o)
        return o

    p = ctx.results / "run_metadata.json"
    if p.exists():  # merge: a partial run (one script) must not erase other experiments
        try:
            old = json.loads(p.read_text(encoding="utf-8"))
            for key in ("experiments", "datasets"):
                merged: dict[str, Any] = dict(old.get(key, {}))
                merged.update(entries if key == "experiments" else datasets)
                meta[key] = merged
        except (json.JSONDecodeError, AttributeError):
            pass
    p.write_text(json.dumps(clean(meta), indent=2, default=str), encoding="utf-8")
    return p
