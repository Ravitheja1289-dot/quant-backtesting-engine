"""
Risk module

Computes and analyzes risk metrics from portfolio returns and equity curves.

Key functions:
- compute_risk_metrics: Main entry point for all risk calculations
- annualized_volatility: Daily volatility scaled to annual
- annualized_return_cagr: Compound annual growth rate from equity curve
- sharpe_ratio: Risk-adjusted return metric
- compute_drawdown: Drawdown series and statistics
- rolling_volatility: Volatility in rolling windows
- rolling_sharpe: Sharpe ratio in rolling windows
- rolling_max_drawdown: Max drawdown in rolling windows
- save_risk_metrics_summary: Persist metrics to JSON
"""

from risk.metrics import (
    annualized_volatility,
    annualized_return_cagr,
    sharpe_ratio,
    compute_drawdown,
    rolling_volatility,
    rolling_sharpe,
    rolling_max_drawdown,
    compute_risk_metrics,
    save_risk_metrics_summary,
)

__all__ = [
    "annualized_volatility",
    "annualized_return_cagr",
    "sharpe_ratio",
    "compute_drawdown",
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_max_drawdown",
    "compute_risk_metrics",
    "save_risk_metrics_summary",
]
