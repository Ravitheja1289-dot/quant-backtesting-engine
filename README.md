# VectorAlpha

Institutional-grade, multi-asset backtesting and risk attribution engine designed to evaluate portfolio strategies under realistic execution and market regimes. Built for quant research and PM evaluation.

## 1️⃣ Overview
- What it does: End-to-end pipeline from raw prices to processed returns, weekly target weights, execution with drift/costs, portfolio accounting, and return/risk attribution.
- Problem it solves: Gives PMs and quants a deterministic, bias-aware harness to test strategies with realistic execution and clean risk/attribution reporting.
- Who it’s for: Quant researchers, PMs, risk/attribution analysts needing reproducible evaluation of portfolio strategies.

## 2️⃣ System Architecture
- Data ingestion: Raw CSVs (audit layer) → processed Parquet (`prices.parquet`, `returns.parquet`), immutable contract; deterministic and idempotent.
- Feature engine: Builds aligned features on processed prices/returns.
- Strategy interface: Pluggable strategies (baseline equal-weight) emitting weekly target weights.
- Execution: Converts weekly targets to daily weights via return drift; applies turnover and linear costs (bps) on rebalance days; renormalizes weights.
- Portfolio accounting: Uses lagged weights with same-day returns; separates gross returns, costs, net returns; iterative equity curve.
- Risk & attribution: Vol/CAGR/Sharpe/drawdown (static + rolling); return attribution (lagged weights × returns, sums exactly to portfolio returns); risk attribution (covariance-based, sums to portfolio volatility); persistence to Parquet for plots.

## 3️⃣ Methodology & Assumptions
- Weekly rebalancing (313 periods in sample).
- Linear transaction costs: 10 bps × turnover on rebalance days.
- No leverage; long-only weights; sum to 1.0 with drift renormalization.
- Daily bars; simple returns; no intraday modeling.
- Equal-weight baseline across 15 large-cap tech/growth names; deterministic pipelines (no randomness, no wall-clock dependence).

## 4️⃣ Validation & Sanity Checks
- Lagged weights for PnL: portfolio return_t = Σ w_{t-1} * r_t.
- Return attribution: Per-asset contributions sum to daily portfolio returns (float tolerance checked, max diff 0.00e+00).
- Risk attribution: Per-asset risk contributions sum to portfolio volatility (covariance-based, sum check enforced).
- Execution: Daily weights sum ≈ 1.0 after drift; no NaNs/infs; non-negative turnover/costs.
- Determinism: Processed data and attribution outputs are reproducible from raw; idempotent regeneration.

## 5️⃣ Key Findings
- 2022 drawdown (~45%) driven by concentrated, correlated growth/tech exposure; diversification failed when correlations spiked.
- NVDA and TSLA dominated both volatility contribution and return drag in 2022; high beta and high covariance amplified losses.
- META added material losses with elevated risk; ORCL provided a modest diversification benefit during the same period.
- Full-period performance (equal-weight baseline): CAGR ~27.4%, vol ~29.7%, Sharpe ~0.97, max drawdown ~45% (prolonged 2022 stress).

## 6️⃣ Limitations & Next Steps
- Current gaps: No factor attribution; no dynamic universe; no intraday execution; linear cost model only; no stress-testing harness; attribution costs not allocated by asset.
- Next steps:
  - Add factor-based risk/return attribution and regime-aware rebalancing.
  - Introduce dynamic universe handling and sector/correlation caps.
  - Implement stress testing and scenario analysis; add nonlinear/impact cost models.
  - Optional: allocate costs by asset for net-of-cost attribution and add rolling risk contribution plots.
