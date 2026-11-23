"""OPTIONAL: download and cache a modest single-currency market dataset (needs network).

    pip install -e .[market]
    python scripts/download_market_data.py

Writes data/market/us_sector_etfs.csv and .meta.json (provider, symbols, retrieval
date, period, currency, adjustment convention, cleaning report).  On failure it
raises; it never substitutes synthetic data.  Tests never call this.
"""

import json
import sys
from pathlib import Path

from backtester.market import download

SYMBOLS = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]  # all USD

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    start, end = "2005-01-01", "2025-01-01"
    if len(sys.argv) > 2:
        start, end = sys.argv[1], sys.argv[2]
    meta = download(SYMBOLS, start, end, root / "data" / "market", "us_sector_etfs")
    print(json.dumps(meta, indent=2))
