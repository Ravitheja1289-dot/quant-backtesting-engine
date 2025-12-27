"""
Determinism verification

Ensures processed pipeline uses only disk I/O, with:
- No randomness
- No system-time dependency
- No API calls in processed layers

Checks:
1) `data/prices.py` and `data/returns.py` do not import or reference
   Yahoo APIs, requests, or any network libraries.
2) `main.py` does not use system time or randomness.
3) Required disk I/O functions are present (CSV/Parquet reads/writes).

Run:
    python scripts/verify_determinism.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES_TO_CHECK = [
    ROOT / "main.py",
    ROOT / "data" / "prices.py",
    ROOT / "data" / "returns.py",
]

BANNED_PATTERNS = [
    # network / APIs
    "yfinance", "YahooLoader", "requests", "http", "urllib",
    # randomness
    "import random", "np.random", "torch.random", "random.random", "random.seed",
    # system time
    "datetime.now", "time.time", "time.perf_counter", "np.datetime64('now')",
]

REQUIRED_PATTERNS = {
    ROOT / "data" / "prices.py": ["read_csv"],
    ROOT / "data" / "returns.py": ["read_parquet", "pct_change"],
}


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    violations = []
    for pat in BANNED_PATTERNS:
        if pat in text:
            violations.append(f"BANNED pattern '{pat}' found in {path}")
    # Required patterns
    req = REQUIRED_PATTERNS.get(path, [])
    for pat in req:
        if pat not in text:
            violations.append(f"REQUIRED pattern '{pat}' missing in {path}")
    return violations


def main() -> int:
    all_violations: list[str] = []
    for f in FILES_TO_CHECK:
        if not f.exists():
            all_violations.append(f"File missing: {f}")
            continue
        all_violations.extend(check_file(f))

    if all_violations:
        print("\n✗ Determinism checks FAILED:")
        for v in all_violations:
            print("  -", v)
        return 1

    print("\n✓ Determinism checks PASSED:")
    print("  - No randomness in processed pipeline")
    print("  - No system-time dependency in processed pipeline")
    print("  - No API calls in processed layers (prices/returns/main)")
    print("  - Disk-only I/O confirmed: CSV→prices, Parquet→returns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
