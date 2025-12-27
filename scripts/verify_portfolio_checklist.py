"""
Portfolio Engine - End-of-Day Checklist

Verify all critical requirements before proceeding:
1. Portfolio returns computed using lagged weights
2. Transaction costs deducted correctly
3. Equity curve monotonic except for returns
4. No NaNs / no silent errors
5. Equal-weight behaves sensibly

If any box fails → fix before proceeding.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prices import load_processed_prices
from data.returns import load_processed_returns
from features.feature_engine import build_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy
from execution.executor import execute_strategy
from portfolio.portfolio_engine import run_backtest
import numpy as np

print("=" * 60)
print("Portfolio Engine - End-of-Day Checklist")
print("=" * 60)

# Load data and run backtest
prices = load_processed_prices("data/processed/prices.parquet")
returns = load_processed_returns("data/processed/returns.parquet")
features = build_features(prices, returns)
rebalance_dates = get_weekly_rebalance_dates(prices.index)
strategy = EqualWeightStrategy()
target_weights = strategy.generate_weights(features, rebalance_dates)

execution_output = execute_strategy(
    prices=prices,
    returns=returns,
    target_weights=target_weights,
    rebalance_dates=rebalance_dates,
    cost_bps=10.0,
)

backtest_results = run_backtest(
    daily_weights=execution_output['daily_weights'],
    returns=returns,
    transaction_costs=execution_output['transaction_costs'],
    initial_capital=1.0,
)

gross_returns = backtest_results['gross_returns']
net_returns = backtest_results['net_returns']
daily_costs = backtest_results['daily_costs']
equity = backtest_results['equity']
daily_weights = execution_output['daily_weights']

print("\n" + "=" * 60)
print("CHECKLIST VERIFICATION")
print("=" * 60)

# Check 1: Portfolio returns computed using lagged weights
print("\n[1/5] Portfolio returns computed using lagged weights")
# Verify by checking if weights[t] with returns[t] would give different result
# Compare with weights[t-1] with returns[t] (correct)

# Get aligned data
common_dates = daily_weights.index.intersection(returns.index)
weights_aligned = daily_weights.loc[common_dates]
returns_aligned = returns.loc[common_dates]

# Calculate with SAME-day weights (WRONG - look-ahead bias)
same_day_returns = (weights_aligned * returns_aligned).sum(axis=1).dropna()

# Calculate with LAGGED weights (CORRECT)
lagged_weights = weights_aligned.shift(1).dropna(how='all')
returns_for_lagged = returns_aligned.loc[lagged_weights.index]
lagged_returns = (lagged_weights * returns_for_lagged).sum(axis=1)

# They should be different
are_different = not gross_returns.equals(same_day_returns.loc[gross_returns.index])
print(f"  Same-day vs lagged returns are different: {are_different}")
print(f"  Gross returns match lagged calculation: {gross_returns.equals(lagged_returns)}")

if are_different and gross_returns.equals(lagged_returns):
    print("  ✓ PASS: Portfolio returns use lagged weights (no look-ahead bias)")
    check1 = True
else:
    print("  ✗ FAIL: Portfolio returns NOT using lagged weights correctly")
    check1 = False

# Check 2: Transaction costs deducted correctly
print("\n[2/5] Transaction costs deducted correctly")
# Net returns should equal gross returns minus costs
computed_net = gross_returns - daily_costs
matches = np.allclose(net_returns, computed_net)

total_costs_in_returns = (gross_returns - net_returns).sum()
total_costs_series = daily_costs.sum()
costs_match = abs(total_costs_in_returns - total_costs_series) < 1e-10

print(f"  Net = Gross - Costs: {matches}")
print(f"  Total costs in returns: {total_costs_in_returns:.6f}")
print(f"  Total costs from series: {total_costs_series:.6f}")
print(f"  Costs match: {costs_match}")

if matches and costs_match:
    print("  ✓ PASS: Transaction costs deducted correctly")
    check2 = True
else:
    print("  ✗ FAIL: Transaction cost deduction incorrect")
    check2 = False

# Check 3: Equity curve monotonic except for returns
print("\n[3/5] Equity curve monotonic except for returns")
# Equity changes should exactly match net returns (after costs)
equity_pct_change = equity.pct_change().dropna()
# Account for costs in the equity formula: equity_t = equity_{t-1} * (1 + gross_ret - cost)
# So pct_change should be approximately (gross_ret - cost)
expected_equity_change = gross_returns[1:] - daily_costs[1:]

# Check if equity changes match expected
equity_changes_match = np.allclose(equity_pct_change.values, expected_equity_change.values, rtol=1e-5)

print(f"  Equity pct_change matches (gross_ret - cost): {equity_changes_match}")
print(f"  Sample equity changes (first 3):")
for i in range(min(3, len(equity_pct_change))):
    date = equity_pct_change.index[i]
    actual = equity_pct_change.iloc[i]
    expected = expected_equity_change.iloc[i]
    print(f"    {date.date()}: actual={actual*100:+.4f}%, expected={expected*100:+.4f}%")

if equity_changes_match:
    print("  ✓ PASS: Equity curve changes match returns (monotonic except for market moves)")
    check3 = True
else:
    print("  ✗ FAIL: Equity curve has unexpected changes")
    check3 = False

# Check 4: No NaNs / no silent errors
print("\n[4/5] No NaNs / no silent errors")
has_nan_gross = gross_returns.isna().any()
has_nan_net = net_returns.isna().any()
has_nan_equity = equity.isna().any()
has_nan_costs = daily_costs.isna().any()

has_inf_gross = gross_returns.isin([np.inf, -np.inf]).any()
has_inf_net = net_returns.isin([np.inf, -np.inf]).any()
has_inf_equity = equity.isin([np.inf, -np.inf]).any()

no_errors = not (has_nan_gross or has_nan_net or has_nan_equity or has_nan_costs or 
                 has_inf_gross or has_inf_net or has_inf_equity)

print(f"  Gross returns have NaNs: {has_nan_gross}")
print(f"  Net returns have NaNs: {has_nan_net}")
print(f"  Equity has NaNs: {has_nan_equity}")
print(f"  Daily costs have NaNs: {has_nan_costs}")
print(f"  Gross returns have Infs: {has_inf_gross}")
print(f"  Net returns have Infs: {has_inf_net}")
print(f"  Equity has Infs: {has_inf_equity}")

if no_errors:
    print("  ✓ PASS: No NaNs or Infs in any series")
    check4 = True
else:
    print("  ✗ FAIL: NaNs or Infs detected")
    check4 = False

# Check 5: Equal-weight behaves sensibly
print("\n[5/5] Equal-weight behaves sensibly")
# Portfolio returns should roughly match market average
market_avg_return = returns.mean(axis=1).mean()
portfolio_avg_return = gross_returns.mean()
return_diff = abs(portfolio_avg_return - market_avg_return)

# Volatility should be similar to market
market_vol = returns.mean(axis=1).std()
portfolio_vol = gross_returns.std()
vol_ratio = portfolio_vol / market_vol

# Final equity should be positive (tech stocks 2020-2025)
final_equity = equity.iloc[-1]
positive_return = final_equity > 1.0

print(f"  Portfolio avg return: {portfolio_avg_return*100:.4f}%")
print(f"  Market avg return: {market_avg_return*100:.4f}%")
print(f"  Difference: {return_diff*100:.4f}%")
print(f"  Portfolio vol / Market vol: {vol_ratio:.2f}x")
print(f"  Final equity: ${final_equity:.4f}")

sensible = (return_diff < 0.001) and (0.5 < vol_ratio < 2.0) and positive_return

if sensible:
    print("  ✓ PASS: Equal-weight behaves sensibly")
    check5 = True
else:
    print("  ✗ FAIL: Equal-weight behavior is suspicious")
    check5 = False

# Final result
print("\n" + "=" * 60)
all_pass = check1 and check2 and check3 and check4 and check5

if all_pass:
    print("✓ ALL CHECKS PASSED")
    print("=" * 60)
    print("\nPortfolio engine ready. Safe to proceed.")
else:
    print("✗ SOME CHECKS FAILED")
    print("=" * 60)
    print("\n⚠ FIX BEFORE PROCEEDING")
    exit(1)
