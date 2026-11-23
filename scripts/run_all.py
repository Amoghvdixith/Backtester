"""Run every demonstration experiment (fast offline mode by default).

Does NOT evaluate the protected hold-out (see scripts/run_holdout.py).

    python scripts/run_all.py              # fast mode (a few minutes on a laptop)
    python scripts/run_all.py --research   # larger universe and grids (slower)
"""

import sys

from backtester.cli import main

if __name__ == "__main__":
    flags = ["--research"] if "--research" in sys.argv[1:] else []
    main([*flags, "run-all"])
