"""One-Page System Output View

Produces:
- Equity curve (log scale)
- Drawdown series
- Rolling Sharpe (63-day)
Optionally rolling volatility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt

from data.prices import load_processed_prices
from data.returns import load_processed_returns
from features.feature_engine import build_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy
from execution.executor import execute_strategy
from portfolio.portfolio_engine import run_backtest
from risk.metrics import compute_risk_metrics
from visualisation.plots import create_system_output_figure

OUTPUT_PATH = Path("data/processed/system_output.svg")


def main():
    print("Generating system output view...")

    # Load data
    prices = load_processed_prices("data/processed/prices.parquet")
    returns = load_processed_returns("data/processed/returns.parquet")
    features = build_features(prices, returns)
    rebalance_dates = get_weekly_rebalance_dates(prices.index)

    # Strategy: equal-weight baseline
    strategy = EqualWeightStrategy()
    target_weights = strategy.generate_weights(features, rebalance_dates)

    # Execute and backtest
    exec_out = execute_strategy(prices, returns, target_weights, rebalance_dates, cost_bps=10.0)
    backtest = run_backtest(exec_out["daily_weights"], returns, exec_out["transaction_costs"], initial_capital=1.0)

    # Risk metrics (for drawdown and rolling)
    metrics = compute_risk_metrics(backtest["net_returns"], backtest["equity"])\

    # Prepare figure
    fig = create_system_output_figure(
        dates=backtest["equity"].index,
        equity=backtest["equity"],
        drawdown=metrics["series"]["drawdown"],
        rolling_sharpe=metrics["series"]["rolling_sharpe"],
        rebalance_dates=rebalance_dates,
        # Include rolling volatility if desired:
        # rolling_vol=metrics["series"]["rolling_volatility"],
        title="System Output: Equal-Weight Baseline (Week 2)"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save as SVG to avoid PIL/PNG encoder issues in some environments
    fig.savefig(OUTPUT_PATH, format="svg")
    print(f"[OK] Saved system output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
