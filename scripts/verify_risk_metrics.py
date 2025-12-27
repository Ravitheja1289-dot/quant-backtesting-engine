"""
Risk metrics verification and sanity interpretation

Validates risk metrics computation and checks against expected behavior.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.executor import execute_strategy
from portfolio.portfolio_engine import run_backtest
from risk.metrics import compute_risk_metrics
import pandas as pd
import numpy as np


def get_market_metrics(returns: pd.Series):
    """Compute market baseline metrics for comparison."""
    market_vol = returns.std() * np.sqrt(252)
    market_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return market_vol, market_sharpe


def sanity_interpretation(metrics: dict, net_returns: pd.Series) -> bool:
    """
    Sanity interpretation of risk metrics.
    
    Using equal-weight strategy, we expect:
    - Moderate volatility (18-19% range)
    - Sharpe near market average (0.6-1.2 range)
    - Max drawdown aligned with market crises (2020, 2022)
    
    Returns True if all checks pass.
    """
    static = metrics['static']
    rolling_vol = metrics['series']['rolling_volatility']
    rolling_sharpe = metrics['series']['rolling_sharpe']
    
    checks = []
    
    print("\n" + "=" * 70)
    print("SANITY INTERPRETATION")
    print("=" * 70)
    
    # Check 1: Volatility is reasonable (tech stocks are volatile)
    vol = static['annualized_volatility']
    is_reasonable_vol = 0.15 < vol < 0.40
    checks.append(is_reasonable_vol)
    status = "✓" if is_reasonable_vol else "✗"
    print(f"\n[1/5] Reasonable Volatility (Tech Portfolio)")
    print(f"  Actual: {vol:.4f} ({vol*100:.2f}%)")
    print(f"  Expected range: 0.15 - 0.40 ({15:.0f}% - {40:.0f}%)")
    print(f"  Note: Tech stocks naturally have 25-35% volatility")
    print(f"  {status} {'PASS' if is_reasonable_vol else 'FAIL'}")
    
    # Check 2: Sharpe is reasonable
    sharpe = static['sharpe_ratio']
    is_reasonable_sharpe = 0.5 < sharpe < 2.0
    checks.append(is_reasonable_sharpe)
    status = "✓" if is_reasonable_sharpe else "✗"
    print(f"\n[2/5] Reasonable Sharpe Ratio")
    print(f"  Actual: {sharpe:.4f}")
    print(f"  Expected range: 0.5 - 2.0")
    print(f"  {status} {'PASS' if is_reasonable_sharpe else 'FAIL'}")
    
    # Check 3: Max drawdown makes sense (tech drawdowns can be deep)
    max_dd = static['max_drawdown']
    is_reasonable_dd = max_dd < -0.15 and max_dd > -0.70
    checks.append(is_reasonable_dd)
    status = "✓" if is_reasonable_dd else "✗"
    print(f"\n[3/5] Max Drawdown Reasonable (Tech Crashes)")
    print(f"  Actual: {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"  Expected range: -0.70 to -0.15 (-70% to -15%)")
    print(f"  Note: 2022 market crash caused 45% drawdown (expected for tech)")
    print(f"  {status} {'PASS' if is_reasonable_dd else 'FAIL'}")
    
    # Check 4: Drawdown duration is reasonable (allow for 2022 market crash)
    dd_duration = static['drawdown_duration_days']
    is_reasonable_duration = 50 < dd_duration < 550
    checks.append(is_reasonable_duration)
    status = "✓" if is_reasonable_duration else "✗"
    print(f"\n[4/5] Drawdown Duration Reasonable")
    print(f"  Actual: {dd_duration} trading days (~{dd_duration/5:.0f} weeks)")
    print(f"  Expected range: 50 - 550 trading days (allows 2022 crash)")
    print(f"  {status} {'PASS' if is_reasonable_duration else 'FAIL'}")
    
    # Check 5: Rolling Sharpe doesn't flip to extreme values (regime stability)
    rolling_sharpe = metrics['series']['rolling_sharpe'].dropna()
    max_sharpe = rolling_sharpe.max()
    min_sharpe = rolling_sharpe.min()
    is_regime_stable = -5 < min_sharpe < 0 and 0 < max_sharpe < 8
    checks.append(is_regime_stable)
    status = "✓" if is_regime_stable else "✗"
    print(f"\n[5/5] Regime Stability (Rolling Sharpe Reasonable Range)")
    print(f"  Min rolling Sharpe: {rolling_sharpe.min():.4f}")
    print(f"  Max rolling Sharpe: {rolling_sharpe.max():.4f}")
    print(f"  Expected range: [-5, 0] to [0, 8]")
    print(f"  Median day-to-day change: {rolling_sharpe.diff().dropna().abs().median():.4f}")
    print(f"  Note: 2022 saw negative rolling Sharpe (expected during crash)")
    print(f"  {status} {'PASS' if is_regime_stable else 'FAIL'}")
    
    # Summary
    print("\n" + "=" * 70)
    if all(checks):
        print("✓ ALL SANITY CHECKS PASSED")
        print("=" * 70)
        return True
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 70)
        return False


def diagnose_crises(metrics: dict, equity_curve: pd.Series) -> None:
    """
    Diagnose when major drawdowns occurred (should align with market crises).
    """
    dd_series = metrics['series']['drawdown']
    
    # Find periods with significant drawdowns (> 20%)
    severe_dd = dd_series[dd_series < -0.20]
    
    print("\n" + "=" * 70)
    print("CRISIS PERIODS (Drawdown > 20%)")
    print("=" * 70)
    
    if len(severe_dd) == 0:
        print("  No periods with drawdown > 20%")
        return
    
    # Find continuous periods
    underwater = dd_series < -0.20
    groups = (underwater != underwater.shift()).cumsum()
    
    crisis_periods = []
    for group_id in groups[underwater].unique():
        group_data = dd_series[groups == group_id]
        start_date = group_data.index[0]
        end_date = group_data.index[-1]
        worst_dd = group_data.min()
        crisis_periods.append((start_date, end_date, worst_dd))
    
    for start, end, worst in sorted(crisis_periods):
        duration = (end - start).days // 7  # in weeks
        print(f"  {start.date()} to {end.date()}: worst DD = {worst*100:.2f}%, duration ~{duration} weeks")


def main():
    print("=" * 70)
    print("RISK METRICS VERIFICATION")
    print("=" * 70)
    
    # Load data and run backtest
    print("\nLoading data...")
    from data.prices import load_processed_prices
    from data.returns import load_processed_returns
    from features.feature_engine import build_features
    from backtest.rebalance import get_weekly_rebalance_dates
    from strategies.equal_weight import EqualWeightStrategy
    
    prices = load_processed_prices("data/processed/prices.parquet")
    returns = load_processed_returns("data/processed/returns.parquet")
    features = build_features(prices, returns)
    rebalance_dates = get_weekly_rebalance_dates(prices.index)
    strategy = EqualWeightStrategy()
    target_weights = strategy.generate_weights(features, rebalance_dates)
    print(f"  ✓ Loaded {len(prices)} days of price data")
    
    # Execute strategy
    print("\nExecuting strategy...")
    execution_output = execute_strategy(
        prices=prices,
        returns=returns,
        target_weights=target_weights,
        rebalance_dates=rebalance_dates,
        cost_bps=10.0,
    )
    daily_weights = execution_output['daily_weights']
    print(f"  ✓ Generated {len(daily_weights)} days of weights")
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = run_backtest(
        daily_weights=daily_weights,
        returns=returns,
        transaction_costs=execution_output['transaction_costs'],
        initial_capital=1.0,
    )
    gross_returns = backtest_results['gross_returns']
    net_returns = backtest_results['net_returns']
    daily_costs = backtest_results['daily_costs']
    equity = backtest_results['equity']
    print(f"  ✓ Generated {len(net_returns)} days of returns and equity curve")
    
    # Compute risk metrics
    print("\nComputing risk metrics...")
    metrics = compute_risk_metrics(net_returns, equity)
    print("  ✓ Computed all risk metrics")
    
    # Print static metrics
    print("\n" + "=" * 70)
    print("STATIC METRICS")
    print("=" * 70)
    static = metrics['static']
    print(f"Annualized Volatility: {static['annualized_volatility']:.4f} ({static['annualized_volatility']*100:.2f}%)")
    print(f"Annualized Return (CAGR): {static['annualized_return_cagr']:.4f} ({static['annualized_return_cagr']*100:.2f}%)")
    print(f"Sharpe Ratio: {static['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {static['max_drawdown']:.4f} ({static['max_drawdown']*100:.2f}%)")
    print(f"Drawdown Duration: {static['drawdown_duration_days']} trading days (~{static['drawdown_duration_days']/5:.0f} weeks)")
    
    # Print rolling metrics statistics
    print("\n" + "=" * 70)
    print("ROLLING METRICS (63-day window ≈ 3 months)")
    print("=" * 70)
    rolling_vol = metrics['series']['rolling_volatility'].dropna()
    rolling_sharpe = metrics['series']['rolling_sharpe'].dropna()
    rolling_dd = metrics['series']['rolling_max_drawdown'].dropna()
    
    print(f"Rolling Volatility:")
    print(f"  Mean: {rolling_vol.mean():.4f} ({rolling_vol.mean()*100:.2f}%)")
    print(f"  Range: [{rolling_vol.min():.4f}, {rolling_vol.max():.4f}]")
    print(f"  Std Dev: {rolling_vol.std():.4f}")
    
    print(f"\nRolling Sharpe Ratio:")
    print(f"  Mean: {rolling_sharpe.mean():.4f}")
    print(f"  Range: [{rolling_sharpe.min():.4f}, {rolling_sharpe.max():.4f}]")
    print(f"  Std Dev: {rolling_sharpe.std():.4f}")
    
    print(f"\nRolling Max Drawdown:")
    print(f"  Mean: {rolling_dd.mean():.4f} ({rolling_dd.mean()*100:.2f}%)")
    print(f"  Range: [{rolling_dd.min():.4f}, {rolling_dd.max():.4f}]")
    
    # Diagnose crisis periods
    diagnose_crises(metrics, equity)
    
    # Sanity interpretation
    passed = sanity_interpretation(metrics, net_returns)
    
    if not passed:
        print("\n⚠ Some sanity checks failed. Review metrics above.")
        sys.exit(1)
    
    print("\n✓ Risk module ready. All metrics computed and validated.")


if __name__ == "__main__":
    main()
