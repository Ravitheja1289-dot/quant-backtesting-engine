"""
Pre-Flight Checks (Fail Fast)

Crash early if prerequisites are not met:
- Config loads (handled by caller)
- Universe is non-empty
- Raw CSV exists for every symbol
- Raw coverage intersects configured date range for every symbol

Keeps Pandas logic out of `main.py` to preserve pure orchestration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config.loader import Settings


def _ensure_universe_non_empty(settings: Settings) -> None:
    if not settings.universe_symbols:
        raise ValueError("Preflight: universe_symbols is empty in config")


def _ensure_raw_files_exist(settings: Settings) -> None:
    raw_dir = Path(settings.paths.raw_dir)
    missing = [s for s in settings.universe_symbols if not (raw_dir / f"{s}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Preflight: Raw CSVs missing for symbols: {missing}. "
            "Run scripts/save_raw.py to fetch and persist."
        )


def _ensure_date_coverage(settings: Settings) -> None:
    raw_dir = Path(settings.paths.raw_dir)
    start = pd.Timestamp(settings.start_date)
    end = pd.Timestamp(settings.end_date)

    for symbol in settings.universe_symbols:
        path = raw_dir / f"{symbol}.csv"
        try:
            # Raw CSVs have multi-row header; skip ticker and label rows
            df = pd.read_csv(path, skiprows=[1, 2])
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df.set_index(date_col, inplace=True)
            df.index.name = "Date"
            df = df[df.index.notna()]
        except Exception as e:
            raise ValueError(f"Preflight: Failed to read raw CSV for {symbol}: {e}") from e

        if df.empty:
            raise ValueError(f"Preflight: Raw CSV for {symbol} is empty: {path}")

        min_dt = pd.Timestamp(df.index.min())
        max_dt = pd.Timestamp(df.index.max())

        # Ensure coverage intersects config window: there must be data on or after start, and on or before end
        if max_dt < start:
            raise ValueError(
                f"Preflight: {symbol} has no data on/after start_date. "
                f"Coverage {min_dt.date()}→{max_dt.date()}, start={start.date()}"
            )
        if min_dt > end:
            raise ValueError(
                f"Preflight: {symbol} has no data on/before end_date. "
                f"Coverage {min_dt.date()}→{max_dt.date()}, end={end.date()}"
            )


def run_preflight(settings: Settings) -> None:
    """Run all pre-flight checks; raise on first failure."""
    _ensure_universe_non_empty(settings)
    _ensure_raw_files_exist(settings)
    _ensure_date_coverage(settings)
