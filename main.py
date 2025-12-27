"""
Main entry point for quant backtesting engine.

Orchestration only - no logic leaked here.
"""

from config.loader import load_settings
import argparse
from pathlib import Path
from validation.preflight import run_preflight
from data.prices import load_price_matrix, save_price_matrix
from data.returns import calculate_returns, save_returns


def _handle_rebuild(settings) -> None:
    """Delete processed artifacts without touching raw data."""
    prices_path = Path(settings.paths.prices_file)
    returns_path = Path(settings.paths.returns_file)
    processed_dir = Path(settings.paths.processed_dir)
    print("\n[0/6] Rebuild requested: cleaning processed outputs...")
    for p in [prices_path, returns_path]:
        if p.exists():
            try:
                p.unlink()
                print(f"  Deleted: {p}")
            except Exception as e:
                raise RuntimeError(f"Failed to delete processed file {p}: {e}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    print("  + Clean slate ready")


def main(rebuild: bool = False) -> None:
    """
    Build and persist clean market data layer.
    
    Steps:
    1. Load config
    2. Load universe (from config)
    3. Load raw CSVs
    4. Build aligned price DataFrame
    5. Save prices to Parquet
    6. Build returns
    7. Save returns to Parquet
    8. Exit
    
    No strategies, no portfolio logic yet.
    """
    print("=" * 60)
    print("QUANT BACKTESTING ENGINE - DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load config
    print("\n[1/6] Loading configuration...")
    settings = load_settings()
    print(f"  + Date range: {settings.start_date} to {settings.end_date}")
    print(f"  + Universe: {len(settings.universe_symbols)} assets")
    print(f"  + Frequency: {settings.data_frequency}")

    if rebuild:
        _handle_rebuild(settings)
    
    # Step 2: Pre-flight checks (fail fast)
    print("\n[2/6] Running pre-flight checks...")
    run_preflight(settings)
    print("  + Pre-flight checks passed")

    # Step 3-4: Build aligned prices (Day 3)
    print("\n[3/6] Loading and aligning price data...")
    print(f"  Loading raw data from disk: {settings.paths.raw_dir}")
    prices = load_price_matrix(
        data_dir=settings.paths.raw_dir,
        price_column=settings.rules.price_column,
        min_required_days=settings.rules.min_days_required,
        universe=settings.universe_symbols,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    print(f"  + Loaded {len(prices.columns)} assets")
    print(f"  + Total observations: {len(prices)}")
    
    # Step 5: Save prices to Parquet
    print("\n[4/6] Persisting prices to Parquet...")
    print(f"  Saving processed prices to {settings.paths.prices_file}")
    save_price_matrix(prices, output_path=settings.paths.prices_file)
    
    # Step 6: Build returns
    print("\n[5/6] Calculating returns...")
    returns = calculate_returns(prices)
    print(f"  + Calculated {len(returns)} returns")
    print(f"  + Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  + Return type: Simple returns (r_t = P_t / P_{{t-1}} - 1)")
    
    # Step 7: Save returns to Parquet
    print("\n[6/6] Persisting returns to Parquet...")
    print(f"  Saving processed returns to {settings.paths.returns_file}")
    save_returns(returns, output_path=settings.paths.returns_file)
    
    # Step 8: Exit
    print("\nComplete!")
    print("=" * 60)
    print("\nClean market data layer ready:")
    print(f"  - {settings.paths.prices_file}")
    print(f"  - {settings.paths.returns_file}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant backtesting engine orchestrator")
    parser.add_argument("--rebuild", action="store_true", help="Recompute processed data (delete Parquet, do not touch raw)")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
