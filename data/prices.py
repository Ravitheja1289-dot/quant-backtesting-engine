"""
Prices Module

Responsibility:
    Convert raw per-asset CSVs → clean multi-asset price matrix

This module reads individual CSV files from the raw data directory and combines
them into a single DataFrame with dates as index and tickers as columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, List
from datetime import date as DateType

import pandas as pd


# === File Format Policy ===
# Raw → CSV (human-readable, audit trail)
# Processed → Parquet (columnar, fast, industry standard)
# This separation matters: you're past the audit layer now.


def load_price_matrix(
    data_dir: str | Path = "data/raw",
    price_column: str = "Adj Close",
    apply_missing_data_policy: bool = True,
    min_required_days: int = 252,
    universe: Optional[List[str]] = None,
    start_date: Optional[DateType] = None,
    end_date: Optional[DateType] = None,
) -> pd.DataFrame:
    """
    Load and combine per-asset CSV files into a clean multi-asset price matrix.

    Missing Data Policy (Non-Negotiable):
    --------------------------------------
    When apply_missing_data_policy=True (RECOMMENDED):
    1. Drop dates before each asset's first valid price
    2. Forward-fill (ffill) missing prices within an asset's trading history
    3. NEVER backward-fill (would introduce future information → illegal)

    Why This Policy:
    - Forward-fill mimics "last traded price" (realistic market behavior)
    - Backward-fill leaks future data into the past (look-ahead bias)
    - Assets start trading on different dates → NaNs before first price are expected

    Parameters
    ----------
    data_dir : str or Path, default="data/raw"
        Directory containing the raw CSV files
    price_column : str, default="Adj Close"
        Column name to extract from each CSV (e.g., "Close", "Adj Close", "Open")
    apply_missing_data_policy : bool, default=True
        If True, apply the recommended missing data policy (drop leading NaNs, ffill).
        If False, return raw data with NaNs as-is (for debugging only).

    Returns
    -------
    pd.DataFrame
        Price matrix with DatetimeIndex and ticker symbols as column names.
        Each column contains the price series for one asset.

    Raises
    ------
    ValueError
        If no CSV files are found in the data directory
    FileNotFoundError
        If the specified data directory does not exist

    Examples
    --------
    >>> prices = load_price_matrix("data/raw", price_column="Adj Close")
    >>> prices.head()
                  AAPL    GOOGL     MSFT    NVDA
    Date
    2020-01-02  300.35  1374.82  160.62   59.47
    2020-01-03  297.43  1362.66  158.62   59.03
    ...
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find all CSV files in the directory
    print(f"Loading raw data from disk: {data_path}")
    csv_files = list(data_path.glob("*.csv"))
    print(f"  Found {len(csv_files)} CSV files")
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    
    # Dictionary to store price series for each ticker
    price_data = {}
    
    for csv_file in csv_files:
        ticker = csv_file.stem  # Get filename without extension (e.g., "AAPL.csv" → "AAPL")
        
        try:
            # Step 1: Read CSV from data/raw/
            # CSVs have multi-row header: row 0=Price types, row 1=Ticker names, row 2="Date" label
            df = pd.read_csv(csv_file, skiprows=[1, 2])  # Skip ticker row and "Date" label row
            
            # Step 2 & 3: Parse dates and set as index
            # First column "Price" contains dates as strings (e.g., "2020-01-02")
            date_col = df.columns[0]  # Should be "Price"
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.set_index(date_col, inplace=True)
            df.index.name = 'Date'
            
            # Remove any rows with invalid dates (NaT)
            df = df[df.index.notna()]
            
            # Step 4: Extract only the specified price column (e.g., Adj Close)
            if price_column not in df.columns:
                print(f"Warning: '{price_column}' column not found in {csv_file.name}")
                continue
            
            # Result per asset: Date index + Single column: price
            price_series = df[[price_column]].copy()
            price_series[price_column] = pd.to_numeric(price_series[price_column], errors='coerce')
            
            # Step 5: Rename column → ticker symbol
            price_series.rename(columns={price_column: ticker}, inplace=True)
            
            # Store the price series
            price_data[ticker] = price_series[ticker]
                
        except Exception as e:
            print(f"Warning: Failed to load {csv_file.name}: {e}")
            continue
    
    if not price_data:
        raise ValueError(f"No valid price data loaded from {data_path}")
    
    # Combine all assets column-wise into one DataFrame
    # Target shape:
    #   index   → DatetimeIndex (trading days)
    #   columns → AAPL, MSFT, ...
    #   values  → Adjusted Close prices
    price_matrix = pd.DataFrame(price_data)
    print("Combining assets into price matrix (dates × tickers)...")
    
    # At this stage: NaNs are EXPECTED
    # Date ranges differ across assets → that's fine
    # NaN handling comes later based on fill_method
    
    # === Apply Alignment (Carefully) ===
    # Order matters: If you skip step order, you introduce bias.
    
    # Step 1: Sort index chronologically
    # Why: Must be sorted before any forward-fill operations
    price_matrix.sort_index(inplace=True)
    
    # Step 1.5: Remove any duplicate dates (keep first occurrence)
    # Why: Some data sources may have duplicate timestamps
    if price_matrix.index.duplicated().any():
        print(f"Warning: Found {price_matrix.index.duplicated().sum()} duplicate dates. Keeping first occurrence.")
        price_matrix = price_matrix[~price_matrix.index.duplicated(keep='first')]
    
    # Optional: Filter to requested universe (in provided order)
    if universe is not None:
        missing = sorted(set(universe) - set(price_matrix.columns))
        if missing:
            raise ValueError(
                f"Missing symbols in raw data: {missing}. Ensure CSVs exist in {data_path}."
            )
        # Preserve requested order
        price_matrix = price_matrix[universe]

    # Step 2: Sort columns alphabetically for consistency
    # Note: If you prefer preserving universe order, remove this line
    price_matrix = price_matrix[sorted(price_matrix.columns)]

    # Optional: Apply date range filtering
    if start_date is not None or end_date is not None:
        start_ts = pd.Timestamp(start_date) if start_date is not None else price_matrix.index.min()
        end_ts = pd.Timestamp(end_date) if end_date is not None else price_matrix.index.max()
        price_matrix = price_matrix.loc[(price_matrix.index >= start_ts) & (price_matrix.index <= end_ts)]
    
    # === Missing Data Policy (Non-Negotiable) ===
    if apply_missing_data_policy:
        # Step 3: Drop rows where ALL assets are NaN
        # Why: Removes dates before ANY asset started trading
        # This prevents forward-filling into the void
        price_matrix.dropna(how='all', inplace=True)
        
        # Step 4: Forward-fill missing prices
        # Why: Mimics "last traded price" (realistic market behavior)
        # Note: Only fills WITHIN the common date range (after step 3)
        # NEVER backward-fill → would introduce future information → illegal
        price_matrix.ffill(inplace=True)
        
        # Step 5: Drop any remaining NaNs
        # Why: These are NaNs at the START of an asset's history (before first valid price)
        # We cannot fill these forward (no previous price exists)
        price_matrix.dropna(inplace=True)
    
    # === Hard Assertions (MANDATORY) ===
    # These checks MUST pass. If they fail → FIX DATA, don't silence the error.
    
    # Assertion 1: Index is strictly increasing
    if not price_matrix.index.is_monotonic_increasing:
        raise ValueError(
            "Index is not strictly increasing. "
            "Found unsorted or duplicate dates. "
            "FIX: Check raw data for duplicate or out-of-order timestamps."
        )
    
    # Assertion 2: No duplicate dates
    if price_matrix.index.has_duplicates:
        duplicates = price_matrix.index[price_matrix.index.duplicated()].unique()
        raise ValueError(
            f"Duplicate dates found in index: {duplicates.tolist()}. "
            "FIX: Remove or aggregate duplicate rows in raw data."
        )
    
    # Assertion 3: No NaNs in final DataFrame
    nan_count = price_matrix.isna().sum().sum()
    if nan_count > 0:
        nan_by_col = price_matrix.isna().sum()
        nan_cols = nan_by_col[nan_by_col > 0]
        raise ValueError(
            f"Found {nan_count} NaNs in final DataFrame. "
            f"Columns with NaNs: {nan_cols.to_dict()}. "
            "FIX: Check missing data policy or raw data quality."
        )
    
    # Assertion 4: Minimum length > 1 year of data (configurable)
    if len(price_matrix) < min_required_days:
        raise ValueError(
            f"Insufficient data: {len(price_matrix)} days < {min_required_days} days (1 year). "
            f"Date range: {price_matrix.index.min()} to {price_matrix.index.max()}. "
            "FIX: Extend date range or use different data source."
        )
    
    return price_matrix


def save_price_matrix(
    price_matrix: pd.DataFrame,
    output_path: str | Path = "data/processed/prices.parquet",
) -> None:
    """
    Persist processed price matrix to Parquet format.

    Why Parquet for Processed Data?
    --------------------------------
    - Columnar: Optimized for column-wise operations (typical in quant)
    - Fast: Binary format with compression, much faster than CSV
    - Industry standard: Used by Spark, Pandas, Arrow ecosystem
    - Type preservation: Maintains datetime, float64 types perfectly

    File Format Policy:
    -------------------
    Raw → CSV (human-readable, audit trail, transparency)
    Processed → Parquet (performance, efficiency, production-ready)
    
    This separation matters: you're past the audit layer now.

    Parameters
    ----------
    price_matrix : pd.DataFrame
        Cleaned price matrix with DatetimeIndex and ticker columns
    output_path : str or Path, default="data/processed/prices.parquet"
        Path where the Parquet file will be saved

    Examples
    --------
    >>> prices = load_price_matrix()
    >>> save_price_matrix(prices)
    Saved 1234 rows × 15 assets to data/processed/prices.parquet
    """
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet with compression
    print(f"Saving processed prices to {output_path}")
    price_matrix.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',  # Good balance of speed vs size
        index=True,  # Preserve DatetimeIndex
    )
    
    print(f"Saved {len(price_matrix)} rows × {len(price_matrix.columns)} assets to {output_path}")


def load_processed_prices(
    input_path: str | Path = "data/processed/prices.parquet",
) -> pd.DataFrame:
    """
    Load processed price matrix from Parquet file.

    This is MUCH faster than re-processing raw CSVs every time.

    Parameters
    ----------
    input_path : str or Path, default="data/processed/prices.parquet"
        Path to the saved Parquet file

    Returns
    -------
    pd.DataFrame
        Price matrix with DatetimeIndex and ticker columns

    Examples
    --------
    >>> prices = load_processed_prices()
    >>> prices.head()
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed prices not found at {input_path}. "
            f"Run load_price_matrix() and save_price_matrix() first."
        )
    
    return pd.read_parquet(input_path, engine='pyarrow')


# Returns logic is enforced in data/returns.py (hard boundary)


if __name__ == "__main__":
    # Simple test/demo
    print("Loading price matrix...")
    prices = load_price_matrix()
    print(f"\nLoaded {len(prices.columns)} assets with {len(prices)} observations")
    print(f"\nAssets: {', '.join(prices.columns)}")
    print(f"\nDate range: {prices.index.min()} to {prices.index.max()}")
    
    # === Visual Sanity Check (One Time Only) ===
    # Pick 2 assets and print prices for same 5 dates
    # Eyeball to ensure values make sense (no plotting)
    if len(prices.columns) >= 2:
        # Pick first 2 assets alphabetically
        asset_1, asset_2 = prices.columns[0], prices.columns[1]
        
        print(f"\n{'='*60}")
        print("VISUAL SANITY CHECK")
        print(f"{'='*60}")
        print(f"\nComparing {asset_1} vs {asset_2} for 5 sample dates:")
        print(f"\n{prices[[asset_1, asset_2]].head()}")
        
        # Check middle of dataset
        mid_point = len(prices) // 2
        print(f"\nMiddle of dataset (around day {mid_point}):")
        print(f"\n{prices[[asset_1, asset_2]].iloc[mid_point:mid_point+5]}")
        
        # Check end of dataset
        print(f"\nLast 5 observations:")
        print(f"\n{prices[[asset_1, asset_2]].tail()}")
        
        print(f"\n{'='*60}")
        print("Sanity check: Do these prices look reasonable?")
        print("- Are values in expected range? (e.g., $1-$1000 for stocks)")
        print("- Do they change gradually? (no sudden 10x jumps)")
        print("- Are both assets moving independently?")
        print(f"{'='*60}")
    
    print(f"\nMissing values after policy: {prices.isna().sum().sum()}")
    
    # === Persist Processed Prices ===
    print(f"\n{'='*60}")
    print("PERSISTING TO PARQUET")
    print(f"{'='*60}")
    save_price_matrix(prices)
    
    # Test loading back
    print("\nVerifying saved file...")
    loaded_prices = load_processed_prices()
    assert loaded_prices.shape == prices.shape, "Shape mismatch after save/load"
    assert (loaded_prices.columns == prices.columns).all(), "Columns mismatch"
    assert (loaded_prices.index == prices.index).all(), "Index mismatch"
    print("✓ Verification passed: Parquet save/load works correctly")
    print(f"\nFile location: data/processed/prices.parquet")
    print("\nNext time, use load_processed_prices() for instant loading (10-100x faster)")
