"""Write the reproducible SYNTHETIC datasets to data/synthetic/.

    python scripts/generate_synthetic_data.py              # fast-mode datasets
    python scripts/generate_synthetic_data.py --research   # also the 150-asset universe
"""

import sys

from backtester.cli import main

if __name__ == "__main__":
    flags = ["--research"] if "--research" in sys.argv[1:] else []
    main([*flags, "generate-data"])
