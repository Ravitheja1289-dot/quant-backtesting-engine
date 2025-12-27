"""Single-run experiment script.

- Loads processed data
- Builds features and weekly target weights (equal-weight baseline)
- Executes strategy with drift + costs
- Runs portfolio accounting
- Computes risk + attribution
- Saves required plots to outputs/

Usage:
    python run_experiment.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from data.prices import load_processed_prices
from data.returns import load_processed_returns
from features.feature_engine import build_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy
from execution.executor import execute_strategy
from portfolio.portfolio_engine import run_backtest
from risk.metrics import compute_risk_metrics
from risk.attribution import compute_return_attribution, compute_risk_attribution
from visualisation.plots import create_system_output_figure, plot_rolling_sharpe
from visualization.attribution_plots import (
    load_return_attribution,
    plot_drawdown_period_contribution,
    plot_risk_contribution_bar,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Running end-to-end experiment...")

    # Load processed data
    prices = load_processed_prices("data/processed/prices.parquet")
    returns = load_processed_returns("data/processed/returns.parquet")

    # Features and target weights
    features = build_features(prices, returns)
    rebalance_dates = get_weekly_rebalance_dates(prices.index)
    strategy = EqualWeightStrategy()
    target_weights = strategy.generate_weights(features, rebalance_dates)

    # Execute strategy and backtest
    exec_out = execute_strategy(prices, returns, target_weights, rebalance_dates, cost_bps=10.0)
    backtest = run_backtest(exec_out["daily_weights"], returns, exec_out["transaction_costs"], initial_capital=1.0)

    # Risk metrics
    metrics = compute_risk_metrics(backtest["net_returns"], backtest["equity"])
    drawdown = metrics["series"]["drawdown"]
    rolling_sharpe = metrics["series"]["rolling_sharpe"]

    # Return attribution (full period, gross to match weight × return definition)
    ret_attr = compute_return_attribution(
        daily_weights=exec_out["daily_weights"],
        returns=returns,
        portfolio_returns=backtest["gross_returns"],
    )

    # Risk attribution (full period, average weights)
    risk_attr = compute_risk_attribution(returns=returns, daily_weights=exec_out["daily_weights"])

    # Plots: equity + drawdown + rolling sharpe (one page)
    fig = create_system_output_figure(
        dates=backtest["equity"].index,
        equity=backtest["equity"],
        drawdown=drawdown,
        rolling_sharpe=rolling_sharpe,
        rebalance_dates=rebalance_dates,
        title="Equity + Drawdown + Rolling Sharpe",
    )
    fig.savefig(OUTPUT_DIR / "equity_drawdown.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'equity_drawdown.png'}")

    # Rolling Sharpe (solo panel)
    fig_rs = plot_rolling_sharpe(backtest["equity"].index, rolling_sharpe)
    fig_rs.figure.tight_layout()
    fig_rs.figure.savefig(OUTPUT_DIR / "rolling_sharpe.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'rolling_sharpe.png'}")

    # 2022 drawdown attribution (bar chart)
    ret_attr_df = ret_attr["daily_contributions"].copy()
    ret_attr_df["portfolio_return"] = ret_attr["portfolio_returns"]
    plot_drawdown_period_contribution(
        attribution=ret_attr_df,
        start="2022-01-01",
        end="2022-12-31",
        output_path=OUTPUT_DIR / "return_attribution_2022.png",
        title="2022 Drawdown Attribution",
    )
    print(f"Saved: {OUTPUT_DIR / 'return_attribution_2022.png'}")

    # Risk contribution bar (percentage of vol)
    plot_risk_contribution_bar(
        risk_attr=risk_attr["summary"],
        output_path=OUTPUT_DIR / "risk_contribution.png",
        title="Risk Contribution (% of Vol)",
        use_pct=True,
    )
    print(f"Saved: {OUTPUT_DIR / 'risk_contribution.png'}")

    # Persist attribution outputs for reproducibility (already saved by scripts/compute_attribution if needed)
    # But keep deterministic path: reuse existing parquet if present; else save quick snapshot here.
    attr_path = Path("data/processed/return_attribution.parquet")
    if not attr_path.exists():
        ret_attr_df.to_parquet(attr_path)
    risk_path = Path("data/processed/risk_attribution.parquet")
    if not risk_path.exists():
        risk_attr["summary"].assign(portfolio_volatility=risk_attr["portfolio_volatility"]).to_parquet(risk_path)

    print("Experiment complete. All outputs are reproducible via this script.")


if __name__ == "__main__":
    # Ensure project root on path
    sys.path.insert(0, str(Path(__file__).parent))
    main()
