"""Attribution plotting utilities.

Responsibilities:
- Load persisted attribution data (parquet)
- Plot-only (no attribution math)
- No smoothing, no strategy logic
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # ensure non-interactive backend
import matplotlib.pyplot as plt

__all__ = [
    "load_return_attribution",
    "plot_cumulative_return_attribution",
    "plot_drawdown_period_contribution",
    "load_risk_attribution",
    "plot_risk_contribution_bar",
]


def load_return_attribution(path: str | Path) -> pd.DataFrame:
    """Load persisted return attribution parquet.

    Expects columns: asset contributions + 'portfolio_return'.
    Index should be datelike; converted to datetime index.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def plot_cumulative_return_attribution(
    attribution: pd.DataFrame,
    output_path: str | Path,
    title: str = "Cumulative Return Attribution",
    overlay_portfolio: bool = True,
) -> None:
    """Stacked area of cumulative contributions by asset.

    - Uses raw cumulative sum (no smoothing)
    - Stacked height should align with cumulative portfolio return
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate contributions vs portfolio return
    if "portfolio_return" not in attribution.columns:
        raise ValueError("Expected 'portfolio_return' column in attribution data")

    contrib = attribution.drop(columns=["portfolio_return"])
    cum_contrib = contrib.cumsum()

    # Optional overlay: cumulative portfolio return for visual alignment
    portfolio_curve = attribution["portfolio_return"].cumsum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(cum_contrib.index, cum_contrib.T, labels=cum_contrib.columns, alpha=0.8)

    if overlay_portfolio:
        ax.plot(portfolio_curve.index, portfolio_curve.values, color="black", linewidth=1.6, label="Portfolio cumulative")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative contribution")
    ax.legend(loc="upper left", ncol=2, fontsize="small")
    ax.grid(True, linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_drawdown_period_contribution(
    attribution: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    output_path: str | Path,
    title: str = "2022 Drawdown Attribution",
) -> None:
    """Bar chart of total contribution over a specified period (e.g., 2022 drawdown).

    Summed contributions per asset; portfolio_return column is ignored for bars.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "portfolio_return" not in attribution.columns:
        raise ValueError("Expected 'portfolio_return' column in attribution data")

    # Slice period
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    window = attribution.loc[start_ts:end_ts]
    if window.empty:
        raise ValueError("Selected window has no data")

    contrib = window.drop(columns=["portfolio_return"])
    totals = contrib.sum().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(totals.index, totals.values, color=["crimson" if v < 0 else "steelblue" for v in totals.values])

    ax.set_title(title)
    ax.set_xlabel("Asset")
    ax.set_ylabel("Total contribution (window)")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_risk_attribution(path: str | Path) -> pd.DataFrame:
    """Load persisted risk attribution parquet.

    Expects columns: avg_weight, marginal_contribution, risk_contribution, pct_of_portfolio_vol
    plus portfolio_volatility column (same scalar per row).
    """
    df = pd.read_parquet(path)
    return df


def plot_risk_contribution_bar(
    risk_attr: pd.DataFrame,
    output_path: str | Path,
    title: str = "Risk Contribution (Volatility)",
    use_pct: bool = True,
) -> None:
    """Bar chart of risk contribution per asset.

    - use_pct: plot percentage of portfolio volatility if True; otherwise absolute contribution.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    col = "pct_of_portfolio_vol" if use_pct else "risk_contribution"
    if col not in risk_attr.columns:
        raise ValueError(f"Expected column '{col}' in risk attribution data")

    values = risk_attr[col].sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(values.index, values.values, color="steelblue")

    ax.set_title(title)
    ax.set_xlabel("Asset")
    ylabel = "% of portfolio volatility" if use_pct else "Risk contribution (vol units)"
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.6)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("This module provides plotting helpers. Use from a script to generate figures.")
