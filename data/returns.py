"""
Returns Module

Single Responsibility:
    Convert aligned prices → aligned returns

Pipeline Discipline (ENFORCED):
    MUST load from data/processed/prices.parquet
    NEVER recompute from raw CSVs here
    
This enforces proper data flow: raw → processed prices → returns

Nothing else belongs here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


# === Return Type Decision (LOCKED IN) ===
# DEFAULT: Simple returns
# Formula: r_t = (P_t / P_{t-1}) - 1
#
# Why Simple Returns:
# 1. Easier to interpret: "5% return" is intuitive
# 2. Industry-acceptable: Standard in portfolio management
# 3. Works cleanly with portfolio weights: Portfolio return = weighted average
# 4. No complexity overhead: Log returns add complexity without practical benefit


def load_prices_for_returns(
    prices_path: str | Path = "data/processed/prices.parquet",
) -> pd.DataFrame:
    """
    Load processed prices for return calculation.
    
    Pipeline Discipline (ENFORCED):
    --------------------------------
    You MUST load: data/processed/prices.parquet
    Never recompute prices from raw here.
    
    This enforces proper data flow:
    1. Raw CSVs → processed prices (via data/prices.py)
    2. Processed prices → returns (via data/returns.py)
    
    Parameters
    ----------
    prices_path : str or Path, default="data/processed/prices.parquet"
        Path to processed prices Parquet file
    
    Returns
    -------
    pd.DataFrame
        Processed price matrix ready for return calculation
        
    Raises
    ------
    FileNotFoundError
        If processed prices file doesn't exist. Run main.py first.
    
    Examples
    --------
    >>> prices = load_prices_for_returns()
    >>> returns = calculate_returns(prices)
    """
    prices_path = Path(prices_path)
    
    if not prices_path.exists():
        raise FileNotFoundError(
            f"Processed prices not found at {prices_path}. "
            "Run main.py or data/prices.py first to generate processed prices."
        )
    
    return pd.read_parquet(prices_path, engine='pyarrow')


def calculate_returns(
    price_matrix: pd.DataFrame,
    method: Literal["simple"] = "simple",
) -> pd.DataFrame:
    """
    Convert aligned prices → aligned returns.
    
    Return Computation Rules (MANDATORY):
    --------------------------------------
    1. Use .pct_change() for return calculation
    2. Drop the first row (NaN from pct_change)
    3. NEVER fill returns (NaNs in returns = data quality issue)
    4. NEVER shift prices forward (creates look-ahead bias)
    
    Result Shape:
    - index   → Date (starts one day after first price date)
    - columns → Assets (same as price matrix)
    - values  → Returns (simple returns: r_t = P_t/P_{t-1} - 1)
    
    Example: If prices start 2020-01-02, returns start 2020-01-03
    
    Return Type (LOCKED IN): Simple returns (default and recommended)
    Formula: r_t = (P_t / P_{t-1}) - 1
    
    Note: Use load_prices_for_returns() to get price_matrix.
          Never pass raw/recomputed prices - use processed Parquet only.
    
    Parameters
    ----------
    price_matrix : pd.DataFrame
        Aligned price matrix with DatetimeIndex and ticker columns
        Must have no NaNs and strictly increasing dates
        Should be loaded from data/processed/prices.parquet
    method : {"simple"}, default="simple"
        Return calculation method. Only "simple" is supported.
        Simple returns are the industry standard for portfolio backtesting.
    
    Returns
    -------
    pd.DataFrame
        Returns matrix with same structure as price matrix.
        First row will be NaN (no prior price to compare) and will be dropped.
        Length will be len(price_matrix) - 1
        
    Raises
    ------
    ValueError
        If price_matrix contains NaNs or invalid method specified
    
    Examples
    --------
    >>> prices = load_prices_for_returns()  # Load from processed Parquet
    >>> returns = calculate_returns(prices)
    >>> returns.head()
    >>> # If prices start 2020-01-02, returns start 2020-01-03
    """
    # Validate input
    if price_matrix.isna().any().any():
        nan_count = price_matrix.isna().sum().sum()
        raise ValueError(
            f"Price matrix contains {nan_count} NaNs. "
            "Use load_price_matrix() with apply_missing_data_policy=True to fix."
        )
    
    if method != "simple":
        raise ValueError(
            f"Invalid method: '{method}'. "
            "Only 'simple' returns are supported for portfolio backtesting."
        )
    
    # Rule 1: Use .pct_change() for return calculation
    # Formula: r_t = (P_t / P_{t-1}) - 1
    returns = price_matrix.pct_change()
    
    # Rule 2: Drop the first row (NaN from pct_change)
    # This is correct: first return date = second price date
    returns = returns.iloc[1:]
    
    # Rule 3: NEVER fill returns
    # If NaNs exist in returns (after dropping first row), that's a data quality issue
    # Don't hide it with fillna()
    
    # Rule 4: NEVER shift prices forward
    # We don't do: price_matrix.shift(-1) - that would be look-ahead bias
    # pct_change() correctly uses prior price: P_t / P_{t-1}
    
    # Validate output
    if len(returns) == 0:
        raise ValueError("Returns matrix is empty after removing first NaN row")
    
    # === Sanity Assertions (MANDATORY) ===
    # Fail loudly - silent errors are fatal
    
    # Assertion 1: No NaNs
    if returns.isna().any().any():
        nan_count = returns.isna().sum().sum()
        nan_cols = returns.isna().sum()[returns.isna().sum() > 0]
        raise ValueError(
            f"ASSERTION FAILED: Found {nan_count} NaNs in returns matrix. "
            f"Columns with NaNs: {nan_cols.to_dict()}. "
            "This indicates a data quality issue in the price matrix."
        )
    
    # Assertion 2: No infinite values
    import numpy as np
    if np.isinf(returns.values).any():
        inf_count = np.isinf(returns.values).sum()
        # Find which columns have infinite values
        inf_cols = [col for col in returns.columns if np.isinf(returns[col]).any()]
        raise ValueError(
            f"ASSERTION FAILED: Found {inf_count} infinite values in returns matrix. "
            f"Columns with infinities: {inf_cols}. "
            "This typically indicates zero or near-zero prices (division by zero)."
        )
    
    # Assertion 3: Same columns as prices
    if not returns.columns.equals(price_matrix.columns):
        missing_in_returns = set(price_matrix.columns) - set(returns.columns)
        extra_in_returns = set(returns.columns) - set(price_matrix.columns)
        raise ValueError(
            f"ASSERTION FAILED: Returns columns don't match price columns. "
            f"Missing in returns: {missing_in_returns}. "
            f"Extra in returns: {extra_in_returns}."
        )
    
    # Assertion 4: Length = len(prices) - 1
    expected_length = len(price_matrix) - 1
    if len(returns) != expected_length:
        raise ValueError(
            f"ASSERTION FAILED: Expected {expected_length} returns (len(prices) - 1), "
            f"but got {len(returns)}. "
            f"Price matrix has {len(price_matrix)} rows."
        )
    
    return returns


def save_returns(
    returns: pd.DataFrame,
    output_path: str | Path = "data/processed/returns.parquet",
) -> None:
    """
    Persist returns matrix to Parquet format.
    
    File Format Policy:
    - Raw prices → CSV (audit trail)
    - Processed prices → Parquet (performance)
    - Returns → Parquet (performance)
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns matrix with DatetimeIndex and ticker columns
    output_path : str or Path, default="data/processed/returns.parquet"
        Path where the Parquet file will be saved
    
    Examples
    --------
    >>> returns = calculate_returns(prices)
    >>> save_returns(returns)
    Saved 1504 rows × 15 assets to data/processed/returns.parquet
    """
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet with compression
    print(f"Saving processed returns to {output_path}")
    returns.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',
        index=True,
    )
    
    print(f"Saved {len(returns)} rows × {len(returns.columns)} assets to {output_path}")


def load_processed_returns(
    input_path: str | Path = "data/processed/returns.parquet",
) -> pd.DataFrame:
    """
    Load processed returns matrix from Parquet file.
    
    This is much faster than re-calculating from prices every time.
    
    Parameters
    ----------
    input_path : str or Path, default="data/processed/returns.parquet"
        Path to the saved Parquet file
    
    Returns
    -------
    pd.DataFrame
        Returns matrix with DatetimeIndex and ticker columns
    
    Raises
    ------
    FileNotFoundError
        If the returns file doesn't exist
    
    Examples
    --------
    >>> returns = load_processed_returns()
    >>> returns.head()
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed returns not found at {input_path}. "
            f"Run calculate_returns() and save_returns() first."
        )
    
    return pd.read_parquet(input_path, engine='pyarrow')


if __name__ == "__main__":
    # Demo: Convert prices → returns
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("RETURNS MODULE - DEMO")
    print("=" * 60)
    
    print("\n[1/3] Loading prices from Parquet (ENFORCED)...")
    print("  Source: data/processed/prices.parquet")
    prices = load_prices_for_returns()
    print(f"  ✓ Loaded {len(prices)} days × {len(prices.columns)} assets")
    print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print("  (Never recomputing from raw - pipeline discipline enforced)")
    
    print("\n[2/3] Calculating returns...")
    returns = calculate_returns(prices)
    print(f"  ✓ Calculated {len(returns)} days × {len(returns.columns)} assets")
    print(f"  Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  (Lost 1 day due to pct_change)")
    
    print("\n[3/3] Sample returns (first 5 days, first 3 assets):")
    print(returns.iloc[:5, :3])
    
    print(f"\n[4/4] Saving to Parquet...")
    save_returns(returns)
    
    print("\n[5/5] Verification...")
    loaded_returns = load_processed_returns()
    assert loaded_returns.shape == returns.shape
    print("  ✓ Verification passed")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
