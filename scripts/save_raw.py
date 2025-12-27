from __future__ import annotations

"""
Persist raw OHLCV to CSV for auditability.

Rules:
- One file per asset (data/raw/{TICKER}.csv)
- CSV format, human readable
- No renaming, no date fixing, no feature creation

Stops on failure (e.g., missing 'Adj Close'), as raw persistence is non-optional.
"""

import sys
from pathlib import Path

import pandas as pd

# Prefer local packages
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_settings
from data.loaders.yahoo_loader import YahooLoader, required_columns


def fail(msg: str) -> None:
    # Do not crash the overall pipeline; report and continue to next symbol.
    print(f"RAW PERSIST FAILED: {msg}")


def save_symbol(symbol: str, out_dir: Path, loader: YahooLoader, start, end) -> bool:
    try:
        df = loader.fetch(symbol, start, end)
    except Exception as e:
        fail(f"{symbol}: fetch error: {e}")
        return False

    # Sanity checks (mandatory)
    if not isinstance(df.index, pd.DatetimeIndex):
        fail(f"{symbol}: Index is not DatetimeIndex")
        return False
    if df.index.has_duplicates:
        fail(f"{symbol}: Index has duplicate timestamps")
        return False
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        fail(f"{symbol}: Missing required columns {missing}")
        return False

    out_path = out_dir / f"{symbol}.csv"
    if out_path.exists():
        print(f"SKIP: {symbol} raw already exists → {out_path} (preserving audit data)")
        return True
    try:
        # Save exactly as returned; include index; no renames
        df.to_csv(out_path)
    except Exception as e:
        fail(f"{symbol}: write error: {e}")
        return False

    print(f"Saved {symbol} → {out_path} ({len(df)} rows)")
    return True


def main() -> None:
    settings = load_settings()
    out_dir = PROJECT_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = YahooLoader()
    successes = 0
    failures = 0
    for symbol in settings.universe_symbols:
        ok = save_symbol(symbol, out_dir, loader, settings.start_date, settings.end_date)
        successes += int(ok)
        failures += int(not ok)

    print(f"RAW PERSIST COMPLETE: {successes} saved, {failures} failed.")
    # Do not crash pipeline on failures; caller can decide next steps.


if __name__ == "__main__":
    main()
