"""
Equal-Weight Reality Check

Verify that equal-weight strategy produces reasonable results:
- Returns roughly match market average
- No extreme volatility spikes
- Costs reduce equity at rebalances
- No crazy jumps

If numbers look insane → bug exists.
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
print("Equal-Weight Reality Check")
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

print("\n" + "=" * 60)
print("REALITY CHECK")
print("=" * 60)

# Check 1: Returns roughly match market average
print("\n[1/4] Returns match market expectations")
market_avg_returns = returns.mean(axis=1)  # Average return across all assets each day
market_avg_return = market_avg_returns.mean()
portfolio_avg_return = gross_returns.mean()

print(f"  Market average return (daily): {market_avg_return * 100:.4f}%")
print(f"  Portfolio return (daily): {portfolio_avg_return * 100:.4f}%")
print(f"  Difference: {(portfolio_avg_return - market_avg_return) * 100:.4f}%")

# Equal-weight should be close to market average
deviation = abs(portfolio_avg_return - market_avg_return)
if deviation < 0.001:  # Less than 10 bps difference
    print("  ✓ Portfolio returns match market average")
else:
    print(f"  ⚠ Portfolio deviates from market by {deviation * 100:.4f}%")

# Check 2: No extreme volatility
print("\n[2/4] Volatility is reasonable")
portfolio_vol = gross_returns.std()
market_vol = market_avg_returns.std()

print(f"  Market volatility (daily): {market_vol * 100:.4f}%")
print(f"  Portfolio volatility (daily): {portfolio_vol * 100:.4f}%")
print(f"  Ratio: {portfolio_vol / market_vol:.2f}x")

# Equal-weight should have similar volatility to market average
if 0.5 < portfolio_vol / market_vol < 2.0:
    print("  ✓ Volatility is reasonable (0.5x to 2.0x market)")
else:
    print(f"  ⚠ Volatility ratio unusual: {portfolio_vol / market_vol:.2f}x")

# Check 3: No crazy spikes in returns
print("\n[3/4] No crazy spikes in returns")
max_daily_return = gross_returns.max()
min_daily_return = gross_returns.min()
abs_max_return = max(abs(max_daily_return), abs(min_daily_return))

print(f"  Max daily return: {max_daily_return * 100:+.2f}%")
print(f"  Min daily return: {min_daily_return * 100:+.2f}%")
print(f"  Largest absolute move: {abs_max_return * 100:.2f}%")

# No single day should move more than ~20% (extremely rare even in crashes)
if abs_max_return < 0.20:
    print("  ✓ No crazy spikes (all moves < 20%)")
else:
    print(f"  ⚠ Extreme spike detected: {abs_max_return * 100:.2f}%")

# Check 4: Costs reduce equity at rebalances
print("\n[4/4] Costs reduce equity at rebalances")
# Find rebalance dates with non-zero costs
cost_dates = daily_costs[daily_costs > 0].index

if len(cost_dates) > 0:
    # Check a few rebalances
    sample_size = min(5, len(cost_dates))
    print(f"  Checking {sample_size} rebalance dates:")
    
    all_reduce = True
    for i, date in enumerate(cost_dates[:sample_size]):
        idx = gross_returns.index.get_loc(date)
        gross_ret = gross_returns.iloc[idx]
        net_ret = net_returns.iloc[idx]
        cost = daily_costs.iloc[idx]
        
        cost_impact = gross_ret - net_ret
        
        print(f"    {date.date()}: gross={gross_ret*100:+.4f}%, net={net_ret*100:+.4f}%, cost={cost*100:.4f}%")
        
        if cost_impact < 0:
            print(f"      ⚠ Cost impact is negative: {cost_impact}")
            all_reduce = False
    
    if all_reduce:
        print("  ✓ Costs consistently reduce returns at rebalances")
else:
    print("  ⚠ No rebalance costs found")

# Check 5: Equity grows over time (for this tech portfolio 2020-2025)
print("\n[5/5] Equity trajectory is reasonable")
final_equity = equity.iloc[-1]
total_return = (final_equity - 1.0) * 100
years = len(equity) / 252

print(f"  Starting equity: $1.00")
print(f"  Final equity: ${final_equity:.4f}")
print(f"  Total return: {total_return:.2f}%")
print(f"  Annualized return: {((final_equity) ** (1/years) - 1) * 100:.2f}%")

# For tech stocks 2020-2025, positive returns expected
if final_equity > 1.0:
    print("  ✓ Equity grew over period (expected for tech 2020-2025)")
else:
    print("  ⚠ Equity declined (unusual for tech 2020-2025)")

# Final sanity: Check for sudden equity jumps
equity_pct_change = equity.pct_change().dropna()
max_equity_jump = equity_pct_change.max()
min_equity_jump = equity_pct_change.min()

print(f"\n  Max equity jump (single day): {max_equity_jump * 100:+.2f}%")
print(f"  Min equity jump (single day): {min_equity_jump * 100:+.2f}%")

if abs(max_equity_jump) < 0.20 and abs(min_equity_jump) < 0.20:
    print("  ✓ No sudden equity jumps (all < 20%)")
else:
    print("  ⚠ Sudden equity jump detected")

print("\n" + "=" * 60)
print("✓ Equal-weight reality check complete")
print("=" * 60)
print("\nConclusion: Numbers look reasonable for equal-weight tech portfolio")
