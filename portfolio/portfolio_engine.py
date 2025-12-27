"""
Portfolio Engine

Responsibility:
- Convert daily weights + returns → portfolio PnL
- Apply transaction costs
- Track equity over time

Does NOT contain:
- Strategy logic (that's in strategies/)
- Execution logic (that's in execution/)

Key Formula:
- portfolio_return_t = sum(weight_{t-1}^i * return_t^i) - transaction_cost_t
- equity_t = equity_{t-1} * (1 + portfolio_return_t)

Critical Note:
- Use PREVIOUS day's weights with today's returns
- Transaction costs are subtracted on rebalance dates only
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
import numpy as np

__all__ = ["calculate_portfolio_returns", "calculate_equity_curve"]


def calculate_portfolio_returns(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: pd.Series,
) -> Dict[str, pd.Series]:
    """Calculate portfolio returns from weights, asset returns, and costs.
    
    Parameters
    ----------
    daily_weights : pd.DataFrame
        Daily portfolio weights (dates × assets)
        Shape: (n_days, n_assets)
    returns : pd.DataFrame
        Daily asset returns (dates × assets)
        Shape: (n_days, n_assets)
    transaction_costs : pd.Series
        Transaction costs on rebalance dates (as fraction of portfolio)
        Index: rebalance_dates
        
    Returns
    -------
    dict
        {
            'gross_returns': pd.Series (before costs),
            'net_returns': pd.Series (after costs),
            'daily_costs': pd.Series (costs on each day, 0 on non-rebalance),
        }
        
    Notes
    -----
    Portfolio return formula:
        r_portfolio_t = sum(w_{t-1}^i * r_t^i)
        
    ⚠️ CRITICAL: Use YESTERDAY's weights with TODAY's returns to avoid look-ahead bias
        
    Transaction costs are tracked separately and subtracted from returns for clean accounting.
    This makes cost attribution easier later.
    
    Example:
        If turnover = 5% and cost = 10 bps, then tc = 0.05 * 0.001 = 0.0005 (5 bps of portfolio)
        This is subtracted from gross return on that rebalance date.
    """
    # Align indices
    common_dates = daily_weights.index.intersection(returns.index)
    daily_weights = daily_weights.loc[common_dates]
    returns = returns.loc[common_dates]
    
    # Validation
    assert daily_weights.shape == returns.shape, "Weights and returns must have same shape"
    assert daily_weights.columns.equals(returns.columns), "Weights and returns must have same columns"
    
    # Calculate gross portfolio returns (before costs)
    # r_portfolio_t = sum(w_{t-1}^i * r_t^i)
    # ⚠️ CRITICAL: Use YESTERDAY's weights with TODAY's returns to avoid look-ahead bias
    # daily_weights[t] = position at END of day t
    # returns[t] = return DURING day t
    # So we need weights[t-1] with returns[t]
    
    # Shift weights forward by 1 to align with returns
    lagged_weights = daily_weights.shift(1)
    
    # Drop first row (no lagged weights for first day)
    lagged_weights = lagged_weights.dropna(how='all')
    returns_aligned = returns.loc[lagged_weights.index]
    
    gross_returns = (lagged_weights * returns_aligned).sum(axis=1)
    
    # Create transaction cost series aligned to daily returns
    # (zero on non-rebalance dates)
    daily_costs = pd.Series(0.0, index=gross_returns.index)
    for date, cost in transaction_costs.items():
        if date in daily_costs.index:
            daily_costs.loc[date] = cost
    
    # Net portfolio returns = gross returns - transaction costs
    net_returns = gross_returns - daily_costs
    
    # Sanity checks
    assert not gross_returns.isna().any(), "Gross returns contain NaNs"
    assert not gross_returns.isin([np.inf, -np.inf]).any(), "Gross returns contain infinities"
    assert not net_returns.isna().any(), "Net returns contain NaNs"
    assert not net_returns.isin([np.inf, -np.inf]).any(), "Net returns contain infinities"
    
    gross_returns.name = 'gross_returns'
    net_returns.name = 'net_returns'
    daily_costs.name = 'daily_costs'
    
    return {
        'gross_returns': gross_returns,
        'net_returns': net_returns,
        'daily_costs': daily_costs,
    }


def calculate_equity_curve(
    portfolio_returns: pd.Series,
    daily_costs: pd.Series,
    initial_capital: float = 1.0,
) -> pd.Series:
    """Calculate equity curve from portfolio returns and costs.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily gross portfolio returns (before costs)
    daily_costs : pd.Series
        Daily transaction costs (as fraction of portfolio)
    initial_capital : float, default=1.0
        Starting capital (normalized to 1.0)
        
    Returns
    -------
    pd.Series
        Equity curve (cumulative wealth)
        
    Notes
    -----
    Equity formula (iterative):
        equity_0 = initial_capital
        equity_t = equity_{t-1} * (1 + R_t^portfolio) - equity_{t-1} * cost_t
        
    Or equivalently:
        equity_t = equity_{t-1} * (1 + R_t^portfolio - cost_t)
        
    This applies returns first, then subtracts costs (which are percentages of portfolio value).
    """
    # Align indices
    assert portfolio_returns.index.equals(daily_costs.index), "Returns and costs must have same index"
    
    # Initialize equity array
    equity_values = np.zeros(len(portfolio_returns))
    
    # Starting equity
    current_equity = initial_capital
    
    # Iterate through each day
    for i, (date, ret) in enumerate(portfolio_returns.items()):
        cost = daily_costs.loc[date]
        
        # Apply return and subtract cost
        # equity_t = equity_{t-1} * (1 + R_t) - equity_{t-1} * cost_t
        # Equivalently: equity_t = equity_{t-1} * (1 + R_t - cost_t)
        current_equity = current_equity * (1.0 + ret - cost)
        
        equity_values[i] = current_equity
    
    # Create Series
    equity = pd.Series(equity_values, index=portfolio_returns.index, name='equity')
    
    # Sanity checks
    assert not equity.isna().any(), "Equity curve contains NaNs"
    assert not equity.isin([np.inf, -np.inf]).any(), "Equity curve contains infinities"
    assert (equity > 0).all(), "Equity curve must remain positive"
    
    return equity


def run_backtest(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: pd.Series,
    initial_capital: float = 1.0,
) -> Dict[str, pd.Series]:
    """Run full backtest: weights + returns → equity curve.
    
    Parameters
    ----------
    daily_weights : pd.DataFrame
        Daily portfolio weights (dates × assets)
    returns : pd.DataFrame
        Daily asset returns (dates × assets)
    transaction_costs : pd.Series
        Transaction costs on rebalance dates
    initial_capital : float, default=1.0
        Starting capital
        
    Returns
    -------
    dict
        {
            'gross_returns': pd.Series (before costs),
            'net_returns': pd.Series (after costs),
            'daily_costs': pd.Series (costs each day),
            'equity': pd.Series (cumulative wealth),
        }
    """
    # Calculate portfolio returns (gross and net)
    returns_output = calculate_portfolio_returns(
        daily_weights=daily_weights,
        returns=returns,
        transaction_costs=transaction_costs,
    )
    
    gross_returns = returns_output['gross_returns']
    net_returns = returns_output['net_returns']
    daily_costs = returns_output['daily_costs']
    
    # Calculate equity curve from gross returns and costs
    # (apply returns first, then subtract costs)
    equity = calculate_equity_curve(
        portfolio_returns=gross_returns,
        daily_costs=daily_costs,
        initial_capital=initial_capital,
    )
    
    # Sanity checks (MANDATORY)
    print("\nPortfolio engine sanity checks...")
    
    # Check 1: Equity never becomes NaN
    assert not equity.isna().any(), "Equity contains NaNs"
    print("  [OK] Equity never becomes NaN")
    
    # Check 2: Equity never becomes negative (unless leverage later)
    assert (equity > 0).all(), "Equity becomes negative"
    print("  [OK] Equity never becomes negative")
    
    # Check 3: Portfolio returns length = number of trading days - 1
    # (we lose one day from weight lagging)
    assert len(gross_returns) == len(returns.index.intersection(daily_weights.index)) - 1, \
        f"Portfolio returns length incorrect: {len(gross_returns)}"
    print("  [OK] Portfolio returns length correct (trading days - 1)")
    
    # Temporary diagnostic prints
    print("\nDiagnostic prints:")
    print("  First 5 gross returns:")
    for i, (date, ret) in enumerate(gross_returns.head(5).items()):
        cost = daily_costs.loc[date]
        print(f"    {date.date()}: {ret:+.6f} (cost: {cost:.6f})")
    
    print("  First 5 equity values:")
    for date, eq in equity.head(5).items():
        print(f"    {date.date()}: ${eq:.6f}")
    
    # Find first rebalance date in equity index
    first_rebalance = None
    for date in equity.index:
        if date in transaction_costs.index:
            first_rebalance = date
            break
    
    if first_rebalance is not None:
        first_rebal_idx = equity.index.get_loc(first_rebalance)
        if first_rebal_idx > 0:
            equity_before = equity.iloc[first_rebal_idx - 1]
            equity_at = equity.loc[first_rebalance]
            print(f"\n  Equity before first rebalance ({equity.index[first_rebal_idx - 1].date()}): ${equity_before:.6f}")
            print(f"  Equity at first rebalance ({first_rebalance.date()}): ${equity_at:.6f}")
            print(f"  Change: {(equity_at - equity_before) / equity_before * 100:+.4f}%")
    
    return {
        'gross_returns': gross_returns,
        'net_returns': net_returns,
        'daily_costs': daily_costs,
        'equity': equity,
    }
