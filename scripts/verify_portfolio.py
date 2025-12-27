"""
Portfolio Engine Integration Test

Test portfolio engine with equal-weight strategy:
- Portfolio returns calculated correctly
- Equity curve grows monotonically (ideally)
- Transaction costs properly deducted
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

print("=" * 60)
print("Portfolio Engine Integration Test")
print("=" * 60)

# Load data and execute strategy
print("\n[1/3] Loading data and executing strategy...")
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

daily_weights = execution_output['daily_weights']
turnover = execution_output['turnover']
transaction_costs = execution_output['transaction_costs']

print(f"  Daily weights: {daily_weights.shape}")
print(f"  Transaction costs: {len(transaction_costs)} entries")

# Run backtest
print("\n[2/3] Running backtest...")
backtest_results = run_backtest(
    daily_weights=daily_weights,
    returns=returns,
    transaction_costs=transaction_costs,
    initial_capital=1.0,
)

gross_returns = backtest_results['gross_returns']
net_returns = backtest_results['net_returns']
daily_costs = backtest_results['daily_costs']
equity = backtest_results['equity']

print(f"  Gross returns: {len(gross_returns)} days")
print(f"  Net returns: {len(net_returns)} days")
print(f"  Equity curve: {len(equity)} days")

# Validate results
print("\n[3/3] Validating backtest results...")

# Check 1: Portfolio returns exist for all days (minus one for lagging)
print(f"  Gross returns shape: {gross_returns.shape}")
print(f"  Net returns shape: {net_returns.shape}")
print(f"  Expected: ({len(returns) - 1},) (one less due to weight lagging)")
assert len(gross_returns) == len(returns) - 1, "Gross returns length mismatch"
assert len(net_returns) == len(returns) - 1, "Net returns length mismatch"
print("  ✓ Portfolio returns for all days (with lag)")

# Check 2: No NaNs in portfolio returns
assert not gross_returns.isna().any(), "Gross returns contain NaNs"
assert not net_returns.isna().any(), "Net returns contain NaNs"
print("  ✓ No NaNs in portfolio returns")

# Check 3: Net returns <= Gross returns (costs always reduce returns)
assert (net_returns <= gross_returns).all(), "Net returns exceed gross returns"
print("  ✓ Net returns <= Gross returns")

# Check 4: Equity curve evolves correctly
# Equity should compound returns: equity[0] = 1.0 * (1 + r[0])
expected_first_equity = 1.0 * (1.0 + net_returns.iloc[0])
assert abs(equity.iloc[0] - expected_first_equity) < 1e-10, \
    f"Equity[0] should be {expected_first_equity}, got {equity.iloc[0]}"
print(f"  ✓ Equity evolves correctly (first = {equity.iloc[0]:.6f})")

# Check 5: Equity is always positive
assert (equity > 0).all(), "Equity curve goes negative"
print("  ✓ Equity always positive")

# Check 6: Daily costs sum to transaction costs
total_daily_costs = daily_costs.sum()
total_transaction_costs = transaction_costs.sum()
assert abs(total_daily_costs - total_transaction_costs) < 1e-10, \
    f"Daily costs sum {total_daily_costs} != transaction costs {total_transaction_costs}"
print(f"  ✓ Daily costs sum to transaction costs")

# Print summary statistics
print("\n" + "=" * 60)
print("BACKTEST SUMMARY")
print("=" * 60)

final_equity = equity.iloc[-1]
total_return = (final_equity - 1.0) * 100
mean_daily_return_gross = gross_returns.mean()
mean_daily_return_net = net_returns.mean()
daily_vol = net_returns.std()
sharpe_daily = net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0
sharpe_annualized = sharpe_daily * (252 ** 0.5)

print(f"\nPerformance:")
print(f"  Initial capital: $1.00")
print(f"  Final equity: ${final_equity:.4f}")
print(f"  Total return: {total_return:.2f}%")
print(f"  Trading days: {len(net_returns)}")

print(f"\nRisk/Return:")
print(f"  Mean daily return (gross): {mean_daily_return_gross * 100:.4f}%")
print(f"  Mean daily return (net): {mean_daily_return_net * 100:.4f}%")
print(f"  Daily volatility: {daily_vol * 100:.4f}%")
print(f"  Sharpe ratio (annualized): {sharpe_annualized:.2f}")

print(f"\nTransaction Costs:")
print(f"  Total costs: {transaction_costs.sum():.6f}")
print(f"  Mean cost per rebalance: {transaction_costs[1:].mean():.6f}")
print(f"  Rebalance count: {len(transaction_costs)}")
print(f"  Cost impact on returns: {(mean_daily_return_gross - mean_daily_return_net) * 100:.6f}% daily")

print(f"\nNet Portfolio Returns:")
print(f"  Min: {net_returns.min() * 100:.2f}%")
print(f"  Max: {net_returns.max() * 100:.2f}%")
print(f"  Median: {net_returns.median() * 100:.4f}%")

print("\n" + "=" * 60)
print("✓ Portfolio engine integration test passed")
print("=" * 60)
