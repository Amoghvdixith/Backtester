"""Cost-sensitivity table (0/5/10/20/50 bps) and robustness checks.

Thin wrapper around the installed package; run from the project folder:
    python scripts/run_cost_sensitivity.py
"""

from backtester.cli import main

if __name__ == "__main__":
    main(["run-all", "--only", "cost_sensitivity", "robustness"])
