"""
Utilities to load backtest settings from YAML.

Avoid hardcoding run parameters in Python; use this loader to read
start/end dates, universe symbols, and data frequency from settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import yaml

__all__ = ["Settings", "Paths", "Rules", "load_settings"]


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    prices_file: Path
    returns_file: Path


@dataclass(frozen=True)
class Rules:
    price_column: str
    missing_data_policy: str
    allow_backward_fill: bool
    drop_all_nan_rows: bool
    min_days_required: int
    returns_method: str
    enforce_processed_source_for_returns: bool
    parquet_compression: str


@dataclass(frozen=True)
class Settings:
    start_date: date
    end_date: date
    universe_symbols: List[str]
    data_frequency: str
    paths: Paths
    rules: Rules


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}

    try:
        start = _to_date(data["start_date"])  # ISO string expected
        end = _to_date(data["end_date"])      # ISO string expected
        symbols = list(data["universe_symbols"])  # list[str]
        freq = str(data["data_frequency"])        # e.g., 'daily'
        paths_cfg = data["paths"]
        rules_cfg = data["rules"]
    except KeyError as e:
        raise KeyError(f"Missing required config key in {p}: {e}") from e

    if not symbols:
        raise ValueError("universe_symbols must contain at least one symbol")

    # Build Paths
    paths = Paths(
        raw_dir=Path(paths_cfg["raw_dir"]),
        processed_dir=Path(paths_cfg["processed_dir"]),
        prices_file=Path(paths_cfg["prices_file"]),
        returns_file=Path(paths_cfg["returns_file"]),
    )

    # Build Rules
    rules = Rules(
        price_column=str(rules_cfg["price_column"]),
        missing_data_policy=str(rules_cfg["missing_data_policy"]),
        allow_backward_fill=bool(rules_cfg["allow_backward_fill"]),
        drop_all_nan_rows=bool(rules_cfg["drop_all_nan_rows"]),
        min_days_required=int(rules_cfg["min_days_required"]),
        returns_method=str(rules_cfg["returns_method"]),
        enforce_processed_source_for_returns=bool(rules_cfg["enforce_processed_source_for_returns"]),
        parquet_compression=str(rules_cfg["parquet_compression"]),
    )

    return Settings(
        start_date=start,
        end_date=end,
        universe_symbols=symbols,
        data_frequency=freq,
        paths=paths,
        rules=rules,
    )


def _to_date(s: str | date) -> date:
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s))
