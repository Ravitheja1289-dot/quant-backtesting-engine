"""
Visualization module (dumb plotting only)

Responsibilities:
- Accept precomputed data (equity, drawdown, rolling metrics, rebalance dates)
- Plot only; no calculations or strategy logic
"""

from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd


def plot_equity_curve(
    dates: pd.Index,
    equity: pd.Series,
    rebalance_dates: Optional[pd.Index] = None,
    log_scale: bool = True,
    ax: Optional[Axes] = None,
    title: str = "Equity Curve"
) -> Axes:
    """Plot the equity curve; optionally mark rebalance dates lightly."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, equity, color="#1f77b4", linewidth=1.5, label="Equity")
    if log_scale:
        ax.set_yscale("log")
    if rebalance_dates is not None:
        for d in rebalance_dates:
            if d in dates:
                ax.axvline(d, color="#999999", linewidth=0.5, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle=":", alpha=0.3)
    return ax


def plot_drawdown(
    dates: pd.Index,
    drawdown: pd.Series,
    ax: Optional[Axes] = None,
    title: str = "Drawdown"
) -> Axes:
    """Plot the drawdown series on the same time axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(dates, drawdown, color="#d62728", linewidth=1.2, label="Drawdown")
    ax.fill_between(dates, drawdown, 0, color="#d62728", alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("DD")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_ylim(min(drawdown.min(), -1.0), 0.0)
    return ax


def plot_rolling_sharpe(
    dates: pd.Index,
    rolling_sharpe: pd.Series,
    ax: Optional[Axes] = None,
    title: str = "Rolling Sharpe (63d)"
) -> Axes:
    """Plot rolling Sharpe ratio."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(dates, rolling_sharpe, color="#2ca02c", linewidth=1.2, label="Rolling Sharpe")
    ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Sharpe")
    ax.grid(True, linestyle=":", alpha=0.3)
    return ax


def plot_rolling_volatility(
    dates: pd.Index,
    rolling_vol: pd.Series,
    ax: Optional[Axes] = None,
    title: str = "Rolling Volatility (ann, 63d)"
) -> Axes:
    """Plot rolling annualized volatility."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(dates, rolling_vol, color="#9467bd", linewidth=1.2, label="Rolling Vol (ann)")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Vol")
    ax.grid(True, linestyle=":", alpha=0.3)
    return ax


def create_system_output_figure(
    dates: pd.Index,
    equity: pd.Series,
    drawdown: pd.Series,
    rolling_sharpe: pd.Series,
    rebalance_dates: Optional[pd.Index] = None,
    rolling_vol: Optional[pd.Series] = None,
    title: str = "System Output"
) -> Figure:
    """Create a one-page system output view.

    Default panes:
    1) Equity curve
    2) Drawdown
    3) Rolling Sharpe
    Optionally add rolling volatility as a 4th pane if provided.
    """
    nrows = 4 if rolling_vol is not None else 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Equity
    plot_equity_curve(dates, equity, rebalance_dates=rebalance_dates, log_scale=True, ax=axes[0])
    # Drawdown
    plot_drawdown(dates, drawdown, ax=axes[1])
    # Rolling Sharpe
    plot_rolling_sharpe(dates, rolling_sharpe, ax=axes[2])
    # Optional rolling volatility
    if rolling_vol is not None:
        plot_rolling_volatility(dates, rolling_vol, ax=axes[3])

    axes[-1].set_xlabel("Date")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
