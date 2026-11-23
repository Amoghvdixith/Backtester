"""PROTECTED final hold-out evaluation.

Evaluates the parameters selected in the LAST walk-forward fold (frozen) on the
hold-out period defined in configs/walkforward.yaml.  Every evaluation is appended
to results/logs/holdout_runs.jsonl with an evaluation number.

    python scripts/run_holdout.py --dry-run    # show what would be evaluated
    python scripts/run_holdout.py --confirm    # actually evaluate (consumes the hold-out)
"""

import sys

from backtester.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    flags = ["--research"] if "--research" in args else []
    rest = [a for a in args if a in ("--confirm", "--dry-run")]
    main([*flags, "holdout", *rest])
