"""
Risk metrics module

Computes risk statistics from portfolio returns and equity curve.
No plotting, no strategy logic.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def annualized_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility of daily returns.
    
    σ_ann = σ_daily × √252
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    
    Returns
    -------
    float
        Annualized volatility (as decimal, e.g., 0.1865 = 18.65%)
    """
    daily_vol = returns.std()
    return daily_vol * np.sqrt(252)


def annualized_return_cagr(equity_curve: pd.Series) -> float:
    """
    Compound Annual Growth Rate from equity curve.
    
    CAGR = (E_T / E_0)^(252 / N) - 1
    
    Where:
    - E_T: final equity
    - E_0: initial equity
    - N: number of trading days
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily equity values
    
    Returns
    -------
    float
        CAGR (as decimal, e.g., 0.276 = 27.6%)
    """
    E_0 = equity_curve.iloc[0]
    E_T = equity_curve.iloc[-1]
    N = len(equity_curve) - 1  # number of periods (trading days)
    
    # Handle edge case: no growth
    if E_T <= 0 or E_0 <= 0:
        return 0.0
    
    cagr = (E_T / E_0) ** (252 / N) - 1
    return cagr


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio with annualization.
    
    Sharpe = [mean(r_daily) - rf] / σ_daily × √252
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    risk_free_rate : float, optional
        Annual risk-free rate (default 0.0)
        Note: Using 0% RF is acceptable for strategy comparison
    
    Returns
    -------
    float
        Sharpe ratio
    """
    daily_rf = risk_free_rate / 252
    excess_return = returns.mean() - daily_rf
    daily_vol = returns.std()
    
    if daily_vol == 0:
        return 0.0
    
    sharpe = (excess_return / daily_vol) * np.sqrt(252)
    return sharpe


def compute_drawdown(equity_curve: pd.Series) -> tuple:
    """
    Compute drawdown series and statistics.
    
    Drawdown at time t:
    DD_t = E_t / max(E_0..t) - 1
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily equity values (indexed by date)
    
    Returns
    -------
    tuple
        - drawdown_series: pd.Series of drawdowns over time
        - max_drawdown: float, deepest drawdown
        - drawdown_duration: int, longest underwater period (trading days)
    """
    # Cumulative maximum of equity
    cummax = equity_curve.cummax()
    
    # Drawdown series: (current - cummax) / cummax
    drawdown_series = (equity_curve - cummax) / cummax
    
    # Max drawdown (most negative)
    max_drawdown = drawdown_series.min()
    
    # Drawdown duration: longest consecutive period where equity < cummax
    # Create a boolean mask: True when underwater
    underwater = drawdown_series < 0
    
    # Find consecutive stretches
    # Use cumsum to identify groups
    groups = (underwater != underwater.shift()).cumsum()
    durations = underwater.groupby(groups).sum()
    
    # Max duration is longest underwater period
    drawdown_duration = int(durations.max()) if len(durations) > 0 else 0
    
    return drawdown_series, max_drawdown, drawdown_duration


def rolling_volatility(returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling volatility (annualized).
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    window : int, optional
        Rolling window in trading days (default 63 ≈ 3 months)
    
    Returns
    -------
    pd.Series
        Rolling annualized volatility
    """
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol


def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free_rate: float = 0.0) -> pd.Series:
    """
    Rolling Sharpe ratio (annualized).
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns
    window : int, optional
        Rolling window in trading days (default 63 ≈ 3 months)
    risk_free_rate : float, optional
        Annual risk-free rate (default 0.0)
    
    Returns
    -------
    pd.Series
        Rolling Sharpe ratio
    """
    daily_rf = risk_free_rate / 252
    
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    rolling_sharpe_values = ((rolling_mean - daily_rf) / rolling_std) * np.sqrt(252)
    
    return rolling_sharpe_values


def rolling_max_drawdown(equity_curve: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling maximum drawdown.
    
    Parameters
    ----------
    equity_curve : pd.Series
        Daily equity values (indexed by date)
    window : int, optional
        Rolling window in trading days (default 63 ≈ 3 months)
    
    Returns
    -------
    pd.Series
        Rolling max drawdown for each period
    """
    def max_dd_in_window(series):
        cummax = series.cummax()
        dd = (series - cummax) / cummax
        return dd.min()
    
    rolling_max_dd = equity_curve.rolling(window=window).apply(max_dd_in_window, raw=False)
    
    return rolling_max_dd


def compute_risk_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    rolling_window: int = 63
) -> dict:
    """
    Compute all risk metrics at once.
    
    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (net of costs)
    equity_curve : pd.Series
        Daily equity values
    risk_free_rate : float, optional
        Annual risk-free rate (default 0.0)
    rolling_window : int, optional
        Rolling window in trading days (default 63 ≈ 3 months)
    
    Returns
    -------
    dict
        Dictionary with:
        - Static metrics: volatility, cagr, sharpe, max_drawdown, dd_duration
        - Rolling metrics: rolling_vol, rolling_sharpe, rolling_max_dd
        - Series: drawdown (full series)
    """
    # Static metrics
    vol = annualized_volatility(returns)
    cagr = annualized_return_cagr(equity_curve)
    sharpe = sharpe_ratio(returns, risk_free_rate)
    
    # Drawdown
    dd_series, max_dd, dd_duration = compute_drawdown(equity_curve)
    
    # Rolling metrics
    rolling_vol = rolling_volatility(returns, rolling_window)
    rolling_sharpe_series = rolling_sharpe(returns, rolling_window, risk_free_rate)
    rolling_max_dd = rolling_max_drawdown(equity_curve, rolling_window)
    
    return {
        "static": {
            "annualized_volatility": float(vol),
            "annualized_return_cagr": float(cagr),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "drawdown_duration_days": int(dd_duration),
            "risk_free_rate": float(risk_free_rate),
        },
        "series": {
            "drawdown": dd_series,
            "rolling_volatility": rolling_vol,
            "rolling_sharpe": rolling_sharpe_series,
            "rolling_max_drawdown": rolling_max_dd,
        }
    }


def save_risk_metrics_summary(
    metrics: dict,
    output_path: str = "data/processed/risk_metrics.json"
) -> None:
    """
    Save static risk metrics to JSON for reporting.
    
    Note: Series are not persisted (use pickle/parquet if needed).
    
    Parameters
    ----------
    metrics : dict
        Risk metrics dictionary from compute_risk_metrics()
    output_path : str, optional
        Path to save JSON file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save only static metrics (JSON-serializable)
    with open(output_path, "w") as f:
        json.dump(metrics["static"], f, indent=2)
    
    print(f"Risk metrics summary saved to {output_path}")


if __name__ == "__main__":
    # Example usage (run from project root)
    from execution.executor import execute_strategy
    from portfolio.portfolio_engine import run_backtest
    
    print("Computing risk metrics for equal-weight strategy...")
    
    # Execute strategy
    daily_weights, turnover, costs = execute_strategy()
    
    # Run backtest
    gross_returns, net_returns, daily_costs, equity = run_backtest(daily_weights, costs)
    
    # Compute risk metrics
    metrics = compute_risk_metrics(net_returns, equity)
    
    # Print static metrics
    print("\n" + "=" * 60)
    print("RISK METRICS SUMMARY")
    print("=" * 60)
    print(f"Annualized Volatility: {metrics['static']['annualized_volatility']:.4f} ({metrics['static']['annualized_volatility']*100:.2f}%)")
    print(f"Annualized Return (CAGR): {metrics['static']['annualized_return_cagr']:.4f} ({metrics['static']['annualized_return_cagr']*100:.2f}%)")
    print(f"Sharpe Ratio: {metrics['static']['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['static']['max_drawdown']:.4f} ({metrics['static']['max_drawdown']*100:.2f}%)")
    print(f"Drawdown Duration: {metrics['static']['drawdown_duration_days']} trading days")
    print("=" * 60)
    
    # Print rolling metrics statistics
    print("\nROLLING METRICS (63-day window):")
    print("-" * 60)
    rolling_vol = metrics['series']['rolling_volatility']
    rolling_sharpe = metrics['series']['rolling_sharpe']
    rolling_dd = metrics['series']['rolling_max_drawdown']
    
    print(f"Rolling Volatility: {rolling_vol.mean():.4f} (mean), range [{rolling_vol.min():.4f}, {rolling_vol.max():.4f}]")
    print(f"Rolling Sharpe: {rolling_sharpe.mean():.4f} (mean), range [{rolling_sharpe.min():.4f}, {rolling_sharpe.max():.4f}]")
    print(f"Rolling Max DD: {rolling_dd.mean():.4f} (mean), range [{rolling_dd.min():.4f}, {rolling_dd.max():.4f}]")
    print("-" * 60)
    
    # Save summary
    save_risk_metrics_summary(metrics)
