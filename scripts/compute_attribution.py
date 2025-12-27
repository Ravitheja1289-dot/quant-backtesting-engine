"""Compute return and risk attribution and persist outputs.

Steps:
- Load processed prices/returns
- Build features and weekly target weights (equal-weight baseline)
- Execute strategy to get daily weights and costs
- Run backtest to obtain portfolio returns
- Compute return and risk attribution
- Optionally compute segmented attribution (pre-2022, 2022, post-2022)
- Persist Parquet outputs for reuse
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prices import load_processed_prices
from data.returns import load_processed_returns
from features.feature_engine import build_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy
from execution.executor import execute_strategy
from portfolio.portfolio_engine import run_backtest
from risk.attribution import (
    compute_return_attribution,
    compute_risk_attribution,
    compute_segmented_attribution,
    save_return_attribution,
    save_risk_attribution,
)

RETURN_ATTR_PATH = Path("data/processed/return_attribution.parquet")
RISK_ATTR_PATH = Path("data/processed/risk_attribution.parquet")


def main() -> None:
    print("Running attribution pipeline...")

    # Load data
    prices = load_processed_prices("data/processed/prices.parquet")
    returns = load_processed_returns("data/processed/returns.parquet")
    features = build_features(prices, returns)
    rebalance_dates = get_weekly_rebalance_dates(prices.index)

    # Strategy + execution
    strategy = EqualWeightStrategy()
    target_weights = strategy.generate_weights(features, rebalance_dates)
    exec_out = execute_strategy(prices, returns, target_weights, rebalance_dates, cost_bps=10.0)

    # Backtest (needed for portfolio returns)
    backtest = run_backtest(exec_out["daily_weights"], returns, exec_out["transaction_costs"], initial_capital=1.0)
    gross_returns = backtest["gross_returns"]

    # Return attribution (uses gross returns to match weight × return definition)
    ret_attr = compute_return_attribution(
        daily_weights=exec_out["daily_weights"],
        returns=returns,
        portfolio_returns=gross_returns,
    )

    # Checks: contributions sum and presence of negative contributors
    portfolio_from_contrib = ret_attr["portfolio_returns"]
    max_diff = (portfolio_from_contrib - gross_returns).abs().max()
    print(f"Return attribution sum check (max diff vs gross returns): {max_diff:.2e}")

    has_negative = ret_attr["summary"]["cumulative_contribution"].lt(0).any()
    print(f"Negative contributors present: {has_negative}")

    # Risk attribution (volatility-based, average weights)
    risk_attr = compute_risk_attribution(returns=returns, daily_weights=exec_out["daily_weights"])
    rc_sum = risk_attr["summary"]["risk_contribution"].sum()
    port_vol = risk_attr["portfolio_volatility"]
    print(f"Risk attribution sum check: rc_sum={rc_sum:.6f}, portfolio_vol={port_vol:.6f}")

    # Segmented attribution (pre-2022, 2022, post-2022)
    segments = [
        ("pre_2022", None, pd.Timestamp("2021-12-31")),
        ("during_2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("post_2022", pd.Timestamp("2023-01-01"), None),
    ]
    segmented = compute_segmented_attribution(returns, exec_out["daily_weights"], segments)
    for label, out in segmented.items():
        top_asset = out["return_summary"].head(1).index[0]
        top_rc = out["risk_summary"].head(1).index[0]
        print(f"Segment {label}: top return contributor={top_asset}, top risk contributor={top_rc}")

    # Persist outputs
    save_return_attribution(ret_attr, output_path=str(RETURN_ATTR_PATH))
    save_risk_attribution(risk_attr, output_path=str(RISK_ATTR_PATH))

    print("Attribution pipeline complete.")


if __name__ == "__main__":
    main()
