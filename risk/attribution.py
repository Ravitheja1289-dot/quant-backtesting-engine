"""
Attribution Module

Responsibility:
- Decompose portfolio return and risk by asset using actual daily weights
- No plotting. No strategy logic.

Outputs:
- Return attribution parquet: data/processed/return_attribution.parquet
- Risk attribution parquet: data/processed/risk_attribution.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
	"compute_return_attribution",
	"compute_risk_attribution",
	"compute_segmented_attribution",
	"save_return_attribution",
	"save_risk_attribution",
]


def _align_assets(
	daily_weights: pd.DataFrame, returns: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Align weights/returns to common assets and date index."""
	common_assets = returns.columns.intersection(daily_weights.columns)
	aligned_returns = returns[common_assets]
	aligned_weights = daily_weights[common_assets]
	return aligned_weights, aligned_returns


def compute_return_attribution(
	daily_weights: pd.DataFrame,
	returns: pd.DataFrame,
	portfolio_returns: Optional[pd.Series] = None,
	min_days: int = 1,
) -> Dict[str, pd.DataFrame]:
	"""Compute asset-level return contributions using lagged weights.

	Contribution_{i,t} = w_{t-1}^i * r_t^i
	Sum_i Contribution_{i,t} = portfolio return_t (validated if provided).
	"""
	weights, rets = _align_assets(daily_weights, returns)

	# Lag weights so each day's return uses prior-day weights
	lagged_weights = weights.reindex(rets.index).shift(1)

	# Daily contributions per asset
	contributions = lagged_weights * rets
	contributions = contributions.dropna(how="all")

	if len(contributions) < min_days:
		raise ValueError(f"Not enough observations for attribution: {len(contributions)} < {min_days}")

	# Portfolio return from contributions
	portfolio_from_contrib = contributions.sum(axis=1)

	if portfolio_returns is not None:
		aligned_portfolio = portfolio_returns.reindex(portfolio_from_contrib.index)
		max_diff = (portfolio_from_contrib - aligned_portfolio).abs().max()
		tol = 1e-10 + 1e-6 * aligned_portfolio.abs().max()
		assert max_diff < tol, f"Return attribution does not sum to portfolio returns (max diff: {max_diff})"

	# Aggregate contribution over time and percentage share
	cumulative = contributions.sum()
	total_portfolio_return = portfolio_from_contrib.sum()
	if abs(total_portfolio_return) > 1e-12:
		pct_contribution = cumulative / total_portfolio_return
	else:
		pct_contribution = cumulative * 0.0

	summary = pd.DataFrame(
		{
			"cumulative_contribution": cumulative,
			"pct_contribution": pct_contribution,
		}
	).sort_values("cumulative_contribution", ascending=False)

	return {
		"daily_contributions": contributions,
		"portfolio_returns": portfolio_from_contrib,
		"summary": summary,
	}


def compute_risk_attribution(
	returns: pd.DataFrame,
	daily_weights: pd.DataFrame,
) -> Dict[str, object]:
	"""Volatility-based risk attribution using average weights.

	σ_p = sqrt(w' Σ w)
	MC_i = (Σ w)_i / σ_p
	RC_i = w_i * MC_i
	"""
	weights, rets = _align_assets(daily_weights, returns)

	# Align to return index and use average weights over the sample
	weights_on_returns = weights.reindex(rets.index)
	avg_weights = weights_on_returns.mean(skipna=True)

	# Covariance of asset returns
	cov = rets.cov()

	# Portfolio volatility
	cov_w = cov.dot(avg_weights)
	portfolio_var = float(avg_weights.dot(cov_w))
	portfolio_vol = float(np.sqrt(max(portfolio_var, 0.0)))

	if portfolio_vol <= 0:
		raise ValueError("Portfolio volatility is zero; cannot attribute risk")

	marginal_contrib = cov_w / portfolio_vol
	risk_contrib = avg_weights * marginal_contrib

	# Sanity: risk contributions should sum to portfolio vol
	rc_sum = risk_contrib.sum()
	diff = abs(rc_sum - portfolio_vol)
	rel_diff = diff / portfolio_vol
	assert diff < 1e-10 or rel_diff < 1e-4, (
		f"Risk contributions do not sum to portfolio volatility (diff={diff}, vol={portfolio_vol})"
	)

	summary = pd.DataFrame(
		{
			"avg_weight": avg_weights,
			"marginal_contribution": marginal_contrib,
			"risk_contribution": risk_contrib,
			"pct_of_portfolio_vol": risk_contrib / portfolio_vol,
		}
	).sort_values("risk_contribution", ascending=False)

	return {
		"summary": summary,
		"portfolio_volatility": portfolio_vol,
		"covariance": cov,
	}


def compute_segmented_attribution(
	returns: pd.DataFrame,
	daily_weights: pd.DataFrame,
	segments: List[Tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp]]],
) -> Dict[str, Dict[str, object]]:
	"""Compute return and risk attribution for time segments."""
	outputs: Dict[str, Dict[str, object]] = {}
	for label, start, end in segments:
		mask = pd.Series(True, index=returns.index)
		if start is not None:
			mask &= returns.index >= start
		if end is not None:
			mask &= returns.index <= end

		seg_returns = returns.loc[mask]
		if len(seg_returns) == 0:
			continue

		seg_weights = daily_weights.loc[seg_returns.index]
		ret_attr = compute_return_attribution(seg_weights, seg_returns)
		risk_attr = compute_risk_attribution(seg_returns, seg_weights)

		outputs[label] = {
			"return_summary": ret_attr["summary"],
			"risk_summary": risk_attr["summary"],
			"portfolio_volatility": risk_attr["portfolio_volatility"],
		}

	return outputs


def save_return_attribution(
	attribution: Dict[str, pd.DataFrame],
	output_path: str = "data/processed/return_attribution.parquet",
) -> None:
	"""Persist daily return contributions + portfolio returns."""
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)

	to_save = attribution["daily_contributions"].copy()
	to_save["portfolio_return"] = attribution["portfolio_returns"]
	to_save.to_parquet(output_file)

	print(f"Return attribution saved to {output_file}")


def save_risk_attribution(
	risk_attr: Dict[str, object],
	output_path: str = "data/processed/risk_attribution.parquet",
) -> None:
	"""Persist risk contribution table."""
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)

	summary = risk_attr["summary"].copy()
	summary["portfolio_volatility"] = risk_attr["portfolio_volatility"]
	summary.to_parquet(output_file)

	print(f"Risk attribution saved to {output_file}")


if __name__ == "__main__":
	# Example wiring: replace with actual data loaders in scripts
	print("Attribution module is library-only. Use scripts to feed data and persist outputs.")
