"""
Execution Module

Responsibility:
- Convert target weights (sparse, weekly) into daily weights (daily drift)
- Apply transaction costs at rebalance
- Track turnover

Does NOT compute returns or PnL.

Key Concepts:
- On rebalance dates: trade to target weights, incur costs
- Between rebalances: weights drift due to returns
- Weight drift formula: w_{t+1}^i = w_t^i * (1 + r_{t+1}^i) / sum(w_t^j * (1 + r_{t+1}^j))
- Renormalization is NON-NEGOTIABLE (prevents weight explosion)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from execution.costs import calculate_transaction_costs

__all__ = ["execute_strategy"]


def _drift_weights(
    current_weights: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """Drift weights by one day using returns.
    
    Parameters
    ----------
    current_weights : pd.Series
        Current portfolio weights (asset → weight)
    returns : pd.Series
        Daily returns for all assets
        
    Returns
    -------
    pd.Series
        Next-day weights after drift (renormalized)
        
    Notes
    -----
    Formula:
        w_{t+1}^i = w_t^i * (1 + r_{t+1}^i) / sum(w_t^j * (1 + r_{t+1}^j))
        
    This maintains sum(w) = 1.0 after return drift.
    """
    # Compute numerator: w_t^i * (1 + r_{t+1}^i)
    numerator = current_weights * (1.0 + returns)
    
    # Compute denominator: sum(w_t^j * (1 + r_{t+1}^j))
    denominator = numerator.sum()
    
    # Avoid division by zero (should never happen with positive weights + returns)
    if abs(denominator) < 1e-12:
        raise ValueError(f"Denominator near zero in weight drift: {denominator}")
    
    # Renormalize
    next_weights = numerator / denominator
    
    return next_weights


def _compute_turnover(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> float:
    """Compute turnover as sum of absolute weight changes.
    
    Parameters
    ----------
    current_weights : pd.Series
        Weights before rebalance
    target_weights : pd.Series
        Target weights after rebalance
        
    Returns
    -------
    float
        Turnover (sum of |w_target - w_current|)
        
    Notes
    -----
    Turnover = sum(|w_target^i - w_current^i|)
    
    This measures the fraction of portfolio that needs to be traded.
    """
    turnover = (target_weights - current_weights).abs().sum()
    return turnover


def execute_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    cost_bps: float = 10.0,
) -> Dict[str, pd.DataFrame]:
    """Execute strategy with daily weight drift and transaction costs.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices (dates × assets)
    returns : pd.DataFrame
        Daily returns (dates × assets)
    target_weights : pd.DataFrame
        Target weights on rebalance dates (rebalance_dates × assets)
    rebalance_dates : List[pd.Timestamp]
        Rebalance dates (subset of trading dates)
    cost_bps : float, default=10.0
        Transaction cost in basis points
        
    Returns
    -------
    dict
        {
            'daily_weights': pd.DataFrame (daily × assets),
            'turnover': pd.Series (rebalance_dates),
            'transaction_costs': pd.Series (rebalance_dates),
        }
        
    Algorithm
    ---------
    1. Initialize weights at first rebalance
    2. For each trading day:
        - If rebalance day:
            - Compute turnover
            - Apply transaction cost
            - Set weights to target
        - Else:
            - Drift weights using returns
    3. Return daily weights, turnover, costs
    
    Notes
    -----
    - Daily weights are stored BEFORE costs are applied
    - Transaction costs reduce portfolio value but do NOT affect weight calculation
    - Weight drift uses renormalization to maintain sum(w) = 1.0
    """
    # Validation
    # Note: Returns has one less row than prices (first day has no return)
    # We keep full prices index for daily_weights, but only drift after first day
    assert prices.columns.equals(returns.columns), "Prices and returns must have same columns"
    assert target_weights.columns.equals(prices.columns), "Target weights columns must match prices"
    
    # Returns index should be subset of prices index
    assert returns.index[0] > prices.index[0], "Returns should start after first price date"
    assert returns.index[-1] == prices.index[-1], "Returns and prices should end on same date"
    
    rebalance_set = set(rebalance_dates)
    
    # Initialize containers (use full prices index for all 1505 days)
    daily_weights = pd.DataFrame(
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )
    turnover_list = []
    turnover_dates = []
    
    # Initialize weights at first rebalance
    current_weights = None
    first_rebalance = rebalance_dates[0]
    
    print("Executing strategy...")
    print(f"  Trading days: {len(prices)}")
    print(f"  Rebalance dates: {len(rebalance_dates)}")
    print(f"  Transaction cost: {cost_bps} bps")
    
    # Loop over all trading days
    for i, date in enumerate(prices.index):
        
        if date in rebalance_set:
            # Rebalance day
            target = target_weights.loc[date]
            
            if current_weights is None:
                # First rebalance: initialize
                current_weights = target.copy()
                turnover = 0.0  # No turnover on first day (no previous position)
            else:
                # Compute turnover before rebalance
                turnover = _compute_turnover(current_weights, target)
                
                # Update to target weights
                current_weights = target.copy()
            
            # Store turnover
            turnover_list.append(turnover)
            turnover_dates.append(date)
            
        else:
            # Non-rebalance day: drift weights
            if current_weights is not None and date in returns.index:
                # Only drift if we have a return for this date
                # (first price date has no return, so can't drift)
                daily_return = returns.loc[date]
                
                # Drift weights
                current_weights = _drift_weights(current_weights, daily_return)
        
        # Store daily weights (after rebalance or drift)
        if current_weights is not None:
            daily_weights.loc[date] = current_weights
    
    # Convert turnover to Series
    turnover_series = pd.Series(turnover_list, index=turnover_dates, name='turnover')
    
    # Calculate transaction costs
    transaction_costs = calculate_transaction_costs(turnover_series, cost_bps=cost_bps)
    
    # Sanity checks (MANDATORY)
    print("\nRunning sanity checks...")
    
    # Check 1: Daily weights sum ≈ 1 (for non-NaN rows)
    # Note: Days before first rebalance may be NaN
    weight_sums = daily_weights.sum(axis=1, skipna=True)
    # Only check rows where we have valid weights (at least one non-NaN value)
    has_weights = ~daily_weights.isna().all(axis=1)
    valid_sums = weight_sums[has_weights]
    
    if len(valid_sums) > 0:
        max_deviation = (valid_sums - 1.0).abs().max()
        assert max_deviation < 1e-6, f"Daily weight sums deviate from 1.0: max={max_deviation}"
        print(f"  [OK] Daily weights sum ~= 1.0 (max deviation: {max_deviation:.2e})")
    else:
        raise ValueError("No valid daily weights found")
    
    # Check 2: No NaNs after first rebalance
    first_rebalance = rebalance_dates[0]
    weights_after_first = daily_weights.loc[first_rebalance:]
    assert not weights_after_first.isna().any().any(), "Daily weights contain NaNs after first rebalance"
    print("  [OK] No NaNs in daily weights (after first rebalance)")
    
    # Check 3: No infs
    assert not daily_weights.isin([np.inf, -np.inf]).any().any(), "Daily weights contain infinities"
    print("  [OK] No infinities in daily weights")
    
    # Check 4: Turnover >= 0
    assert (turnover_series >= 0).all(), "Turnover must be non-negative"
    print("  [OK] Turnover >= 0")
    
    # Check 5: Transaction costs >= 0
    assert (transaction_costs >= 0).all(), "Transaction costs must be non-negative"
    print("  [OK] Transaction costs >= 0")
    
    # Print diagnostics (temporary)
    print("\nExecution diagnostics:")
    print(f"  First 3 rebalance turnovers:")
    for i, (date, turnover) in enumerate(zip(turnover_dates[:3], turnover_list[:3])):
        print(f"    {date.date()}: {turnover:.4f}")
    
    print(f"  First 3 daily weight sums:")
    for i, date in enumerate(prices.index[:3]):
        if not daily_weights.loc[date].isna().all():
            weight_sum = daily_weights.loc[date].sum()
            print(f"    {date.date()}: {weight_sum:.6f}")
    
    return {
        'daily_weights': daily_weights,
        'turnover': turnover_series,
        'transaction_costs': transaction_costs,
    }
