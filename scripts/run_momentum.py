"""12-1 momentum demonstration vs an equal-weight point-in-time universe benchmark.

Thin wrapper around the installed package; run from the project folder:
    python scripts/run_momentum.py
"""

from backtester.cli import main

if __name__ == "__main__":
    main(["run-all", "--only", "momentum_demo"])
