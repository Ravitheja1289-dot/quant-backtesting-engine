"""
Final Sanity Sweep (One-Time)

Print shapes and date ranges, then assert:
- No NaNs
- No infinities

If all checks pass, print FREEZE OK and exit 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Prefer local packages
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_settings


def main() -> int:
    settings = load_settings()

    prices = pd.read_parquet(settings.paths.prices_file, engine="pyarrow")
    returns = pd.read_parquet(settings.paths.returns_file, engine="pyarrow")

    # Print shapes
    print("\nFinal Sanity Sweep")
    print(f"prices → {prices.shape}")
    print(f"returns → {returns.shape}")

    # Print first/last dates
    print(f"prices dates → {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"returns dates → {returns.index.min().date()} → {returns.index.max().date()}")

    # Basic shape expectations
    n_assets = len(settings.universe_symbols)
    if prices.shape[1] != n_assets:
        print(f"✗ prices columns = {prices.shape[1]} != universe {n_assets}")
        return 1
    if returns.shape[1] != n_assets:
        print(f"✗ returns columns = {returns.shape[1]} != universe {n_assets}")
        return 1
    if len(returns) != len(prices) - 1:
        print(f"✗ returns length = {len(returns)} != len(prices)-1 = {len(prices)-1}")
        return 1

    # No NaNs
    if prices.isna().any().any():
        print(f"✗ prices contain NaNs: {prices.isna().sum().sum()}")
        return 1
    if returns.isna().any().any():
        print(f"✗ returns contain NaNs: {returns.isna().sum().sum()}")
        return 1

    # No infinities
    if np.isinf(prices.values).any():
        print("✗ prices contain infinities")
        return 1
    if np.isinf(returns.values).any():
        print("✗ returns contain infinities")
        return 1

    print("\n✓ All checks green → FREEZE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
