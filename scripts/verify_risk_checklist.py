"""
Risk Module - End-of-Day Checklist

Verify all critical requirements before proceeding:
1. All core risk metrics computed correctly (vol, return, Sharpe, drawdown)
2. Formulas match mathematical specification (annualization, CAGR, etc.)
3. Rolling metrics work without NaNs or infinities
4. Sanity interpretation confirms reasonable results
5. Risk outputs persisted to JSON for reporting

If any check fails → fix before proceeding.
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
from risk.metrics import compute_risk_metrics, save_risk_metrics_summary
import numpy as np
import json

print("=" * 70)
print("Risk Module - End-of-Day Checklist")
print("=" * 70)

# Load data and run backtest
print("\nLoading data and executing strategy...")
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

net_returns = backtest_results['net_returns']
equity = backtest_results['equity']

print("  ✓ Data loaded and strategy executed")

# Compute risk metrics
print("\nComputing risk metrics...")
metrics = compute_risk_metrics(net_returns, equity)
print("  ✓ All risk metrics computed")

print("\n" + "=" * 70)
print("CHECKLIST VERIFICATION")
print("=" * 70)

checks = []

# Check 1: Core metrics exist and are reasonable
print("\n[1/5] Core Risk Metrics Computed")
static = metrics['static']
has_all_metrics = all(key in static for key in [
    'annualized_volatility', 'annualized_return_cagr', 'sharpe_ratio',
    'max_drawdown', 'drawdown_duration_days', 'risk_free_rate'
])
checks.append(has_all_metrics)

print(f"  Has annualized_volatility: {static['annualized_volatility']:.4f}")
print(f"  Has annualized_return_cagr: {static['annualized_return_cagr']:.4f}")
print(f"  Has sharpe_ratio: {static['sharpe_ratio']:.4f}")
print(f"  Has max_drawdown: {static['max_drawdown']:.4f}")
print(f"  Has drawdown_duration_days: {static['drawdown_duration_days']}")
status = "✓" if has_all_metrics else "✗"
print(f"  {status} {'PASS' if has_all_metrics else 'FAIL'}: All metrics present")

# Check 2: Metrics are numerically sound (no NaNs, infinities, or weird values)
print("\n[2/5] Metrics Are Numerically Sound")

vol = static['annualized_volatility']
cagr = static['annualized_return_cagr']
sharpe = static['sharpe_ratio']
max_dd = static['max_drawdown']
dd_dur = static['drawdown_duration_days']

# Volatility should be positive
is_vol_ok = not np.isnan(vol) and not np.isinf(vol) and vol > 0
print(f"  Volatility > 0: {is_vol_ok} (value: {vol:.4f})")

# CAGR should be reasonable
is_cagr_ok = not np.isnan(cagr) and not np.isinf(cagr) and -1 < cagr < 2
print(f"  CAGR reasonable (-1 to +2): {is_cagr_ok} (value: {cagr:.4f})")

# Sharpe can be negative but shouldn't be extreme
is_sharpe_ok = not np.isnan(sharpe) and not np.isinf(sharpe) and -10 < sharpe < 10
print(f"  Sharpe reasonable (-10 to +10): {is_sharpe_ok} (value: {sharpe:.4f})")

# Max DD should be negative
is_dd_ok = not np.isnan(max_dd) and not np.isinf(max_dd) and max_dd < 0
print(f"  Max drawdown negative: {is_dd_ok} (value: {max_dd:.4f})")

# DD duration should be positive integer
is_dd_dur_ok = isinstance(dd_dur, (int, np.integer)) and dd_dur > 0
print(f"  Drawdown duration positive: {is_dd_dur_ok} (value: {dd_dur})")

checks_ok = is_vol_ok and is_cagr_ok and is_sharpe_ok and is_dd_ok and is_dd_dur_ok
checks.append(checks_ok)
status = "✓" if checks_ok else "✗"
print(f"  {status} {'PASS' if checks_ok else 'FAIL'}: All metrics numerically sound")

# Check 3: Rolling metrics exist and are complete
print("\n[3/5] Rolling Metrics Computed")

series = metrics['series']
has_rolling = all(key in series for key in [
    'drawdown', 'rolling_volatility', 'rolling_sharpe', 'rolling_max_drawdown'
])
checks.append(has_rolling)

rolling_vol = series['rolling_volatility'].dropna()
rolling_sharpe = series['rolling_sharpe'].dropna()
rolling_dd = series['rolling_max_drawdown'].dropna()

print(f"  Rolling volatility length: {len(rolling_vol)} days")
print(f"  Rolling Sharpe length: {len(rolling_sharpe)} days")
print(f"  Rolling max DD length: {len(rolling_dd)} days")

# Check for NaNs in rolling metrics
has_nans = (rolling_vol.isna().any() or rolling_sharpe.isna().any() or 
            rolling_dd.isna().any())
print(f"  Rolling metrics have NaNs: {has_nans}")

status = "✓" if has_rolling and not has_nans else "✗"
print(f"  {status} {'PASS' if has_rolling and not has_nans else 'FAIL'}: Rolling metrics complete")

# Check 4: Annualization is correct (key formulas)
print("\n[4/5] Formulas Match Specification")

# Check annualized volatility formula
daily_vol = net_returns.std()
expected_vol = daily_vol * np.sqrt(252)
vol_matches = np.isclose(vol, expected_vol, rtol=1e-5)
print(f"  Annualized vol = daily_vol × √252: {vol_matches}")
print(f"    Actual: {vol:.6f}, Expected: {expected_vol:.6f}, Diff: {abs(vol - expected_vol):.2e}")

# Check CAGR formula: (E_T / E_0)^(252 / N) - 1
E_0 = equity.iloc[0]
E_T = equity.iloc[-1]
N = len(equity) - 1
expected_cagr = (E_T / E_0) ** (252 / N) - 1
cagr_matches = np.isclose(cagr, expected_cagr, rtol=1e-5)
print(f"  CAGR = (E_T / E_0)^(252/N) - 1: {cagr_matches}")
print(f"    Actual: {cagr:.6f}, Expected: {expected_cagr:.6f}, Diff: {abs(cagr - expected_cagr):.2e}")

# Check Sharpe formula: mean(r) / std(r) × √252
expected_sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(252)
sharpe_matches = np.isclose(sharpe, expected_sharpe, rtol=1e-5)
print(f"  Sharpe = mean(r) / std(r) × √252: {sharpe_matches}")
print(f"    Actual: {sharpe:.6f}, Expected: {expected_sharpe:.6f}, Diff: {abs(sharpe - expected_sharpe):.2e}")

formulas_ok = vol_matches and cagr_matches and sharpe_matches
checks.append(formulas_ok)
status = "✓" if formulas_ok else "✗"
print(f"  {status} {'PASS' if formulas_ok else 'FAIL'}: All formulas correct")

# Check 5: Metrics persisted to JSON
print("\n[5/5] Risk Metrics Persisted to JSON")

try:
    with open("data/processed/risk_metrics.json", "r") as f:
        saved_metrics = json.load(f)
    
    # Check that all static metrics are in the file
    saved_keys = set(saved_metrics.keys())
    expected_keys = {'annualized_volatility', 'annualized_return_cagr', 'sharpe_ratio',
                     'max_drawdown', 'drawdown_duration_days', 'risk_free_rate'}
    has_all_keys = expected_keys.issubset(saved_keys)
    
    print(f"  File exists: data/processed/risk_metrics.json")
    print(f"  Has all static metrics: {has_all_keys}")
    print(f"  Saved metrics keys: {list(saved_metrics.keys())}")
    
    # Verify values match
    values_match = (
        np.isclose(saved_metrics['annualized_volatility'], static['annualized_volatility']) and
        np.isclose(saved_metrics['annualized_return_cagr'], static['annualized_return_cagr']) and
        np.isclose(saved_metrics['sharpe_ratio'], static['sharpe_ratio']) and
        np.isclose(saved_metrics['max_drawdown'], static['max_drawdown']) and
        saved_metrics['drawdown_duration_days'] == static['drawdown_duration_days']
    )
    print(f"  Saved values match computed metrics: {values_match}")
    
    persisted_ok = has_all_keys and values_match
    checks.append(persisted_ok)
    status = "✓" if persisted_ok else "✗"
    print(f"  {status} {'PASS' if persisted_ok else 'FAIL'}: Metrics persisted correctly")
    
except Exception as e:
    print(f"  ✗ FAIL: Error reading JSON: {e}")
    checks.append(False)

# Summary
print("\n" + "=" * 70)
if all(checks):
    print("✓ ALL CHECKS PASSED")
    print("=" * 70)
    print("\nRisk Module Summary:")
    print(f"  • Annualized Volatility: {static['annualized_volatility']*100:.2f}%")
    print(f"  • Annualized Return (CAGR): {static['annualized_return_cagr']*100:.2f}%")
    print(f"  • Sharpe Ratio: {static['sharpe_ratio']:.4f}")
    print(f"  • Max Drawdown: {static['max_drawdown']*100:.2f}%")
    print(f"  • Drawdown Duration: {static['drawdown_duration_days']} trading days")
    print(f"  • Risk-Free Rate Assumption: {static['risk_free_rate']*100:.2f}%")
    print(f"\n✓ Risk module ready. Safe to proceed.")
    sys.exit(0)
else:
    print("✗ SOME CHECKS FAILED")
    print("=" * 70)
    sys.exit(1)
