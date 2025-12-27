"""
Sanity Regression Check

Pick one asset and one date; print:
- Price (P_t)
- Previous price (P_{t-1})
- Return r_t
Then assert r_t == P_t/P_{t-1} - 1.

Why: persistence bugs are sneaky; verify post-save consistency.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Import settings for centralized paths and universe
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.loader import load_settings


def main() -> int:
    settings = load_settings()

    prices_path = settings.paths.prices_file
    returns_path = settings.paths.returns_file

    prices = pd.read_parquet(prices_path, engine="pyarrow")
    returns = pd.read_parquet(returns_path, engine="pyarrow")

    # Choose asset and a stable date (middle of dataset to avoid holidays)
    asset = settings.universe_symbols[0]
    date_idx = len(returns) // 2
    date = returns.index[date_idx]
    prev_date = returns.index[date_idx - 1]

    P_t = float(prices.loc[date, asset])
    P_prev = float(prices.loc[prev_date, asset])
    r_t = float(returns.loc[date, asset])
    expected = (P_t / P_prev) - 1.0
    diff = float(expected - r_t)

    print("\nSanity Regression Check")
    print(f"Asset: {asset}")
    print(f"Date: {pd.Timestamp(date).date()}  (Prev: {pd.Timestamp(prev_date).date()})")
    print(f"Price (P_t):        {P_t:.6f}")
    print(f"Prev Price (P_{ '{' }t-1{ '}' }): {P_prev:.6f}")
    print(f"Return (r_t):       {r_t:.8f}")
    print(f"Expected:           {expected:.8f}")
    print(f"Difference:         {diff:.12f}")

    # Assert exact consistency to machine precision tolerance
    if not np.isclose(expected, r_t, rtol=0, atol=1e-12):
        print("\n✗ Consistency check FAILED: return does not match price ratio")
        return 1

    print("\n✓ Consistency check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
