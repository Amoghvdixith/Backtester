"""Command-line entry point: ``backtester <command>`` (also used by scripts/)."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from . import experiments as ex
from . import plots


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=lambda o: None if hasattr(o, "index") else str(o)))


def cmd_generate(args: argparse.Namespace) -> dict[str, Any]:
    root = ex.project_root(args.root)
    names = ("etf_like", "stock_universe") + (("stock_universe_large",) if args.research else ())
    return ex.ensure_datasets(root, names)


def cmd_run_all(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    root = ex.project_root(args.root)
    names = ("etf_like", "stock_universe") + (("stock_universe_large",) if args.research else ())
    datasets = ex.ensure_datasets(root, names)
    ctx = ex.Context(root, research=args.research)
    plots.architecture_diagram(ctx.figs / "01_architecture_event_flow.png")
    entries: dict[str, Any] = {}
    steps = [
        ("buy_and_hold_validation", lambda: ex.buy_and_hold_validation(ctx)),
        ("sma_demo", lambda: ex.strategy_demo(ctx, "sma", "03_sma")),
        ("momentum_demo", lambda: ex.strategy_demo(ctx, "momentum", "04_momentum")),
        ("sma_walkforward", lambda: ex.walkforward_experiment(ctx, "sma", "05_sma_wf")),
        ("momentum_walkforward",
         lambda: ex.walkforward_experiment(ctx, "momentum", "06_momentum_wf")),
        ("cost_sensitivity", lambda: ex.cost_sensitivity(ctx)),
        ("robustness", lambda: ex.robustness(ctx)),
    ]
    for name, fn in steps:
        if args.only and name not in args.only:
            continue
        t1 = time.perf_counter()
        print(f"[run_all] {name} ...", flush=True)
        entries[name] = fn()
        entries[name]["step_runtime_seconds"] = round(time.perf_counter() - t1, 2)
        print(f"[run_all] {name} done in {entries[name]['step_runtime_seconds']}s", flush=True)
    entries["total_runtime_seconds"] = round(time.perf_counter() - t0, 2)
    entries["holdout_consumed_by_this_run"] = False
    path = ex.write_run_metadata(ctx, entries, datasets)
    print(f"[run_all] metadata written to {path.relative_to(root)}")
    return entries


def cmd_holdout(args: argparse.Namespace) -> dict[str, Any]:
    ctx = ex.Context(args.root, research=args.research)
    if not args.confirm and not args.dry_run:
        raise SystemExit("Refusing to evaluate the protected hold-out without --confirm. "
                         "Use --dry-run to see what would be evaluated.")
    return ex.holdout_evaluation(ctx, confirm=args.confirm, dry_run=args.dry_run)


def cmd_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Fixed-parameter demo for any config (e.g. the optional market config)."""
    ctx = ex.Context(args.root, research=args.research)
    out = ex.strategy_demo(ctx, args.config, args.prefix or args.config)
    cfg = ctx.config(args.config)
    meta_path = ctx.root / cfg["data"]["path"]
    prov = json.loads(meta_path.with_suffix(".meta.json").read_text(encoding="utf-8")) \
        if meta_path.with_suffix(".meta.json").exists() else {}
    ex.write_run_metadata(ctx, {f"demo_{args.config}": out}, {args.config: prov})
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backtester", description=__doc__)
    p.add_argument("--root", default=None, help="project folder (default: search from cwd)")
    p.add_argument("--research", action="store_true",
                   help="larger dataset/grids (slower research mode)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-data", help="write the synthetic datasets").set_defaults(
        func=cmd_generate)
    ra = sub.add_parser("run-all", help="all demonstration experiments (no hold-out)")
    ra.add_argument("--only", nargs="*", default=None)
    ra.set_defaults(func=cmd_run_all)
    de = sub.add_parser("demo", help="fixed-parameter demo for configs/<name>.yaml")
    de.add_argument("config")
    de.add_argument("--prefix", default=None)
    de.set_defaults(func=cmd_demo)
    ho = sub.add_parser("holdout", help="PROTECTED: evaluate frozen params on the hold-out")
    ho.add_argument("--confirm", action="store_true")
    ho.add_argument("--dry-run", action="store_true")
    ho.set_defaults(func=cmd_holdout)
    return p


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    out = args.func(args)
    if args.command != "run-all":
        _print(out)
    return out


if __name__ == "__main__":  # pragma: no cover
    main()
