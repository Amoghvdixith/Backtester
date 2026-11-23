"""Walk-forward selection for SMA and momentum (trial log, stitched OOS, DSR). Hold-out excluded.

Thin wrapper around the installed package; run from the project folder:
    python scripts/run_walkforward.py
"""

from backtester.cli import main

if __name__ == "__main__":
    main(["run-all", "--only", "sma_walkforward", "momentum_walkforward"])
