"""Buy-and-hold validation against an independent, timing-consistent reference.

Thin wrapper around the installed package; run from the project folder:
    python scripts/run_buy_and_hold.py
"""

from backtester.cli import main

if __name__ == "__main__":
    main(["run-all", "--only", "buy_and_hold_validation"])
