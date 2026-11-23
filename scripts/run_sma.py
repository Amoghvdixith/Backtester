"""SMA crossover demonstration (fixed a-priori params, development sample before the hold-out).

Thin wrapper around the installed package; run from the project folder:
    python scripts/run_sma.py
"""

from backtester.cli import main

if __name__ == "__main__":
    main(["run-all", "--only", "sma_demo"])
