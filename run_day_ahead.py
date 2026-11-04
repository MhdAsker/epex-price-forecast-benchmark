"""
Train and score the day-ahead auction protocol.

Does not retrain SARIMA, LSTM, or the walk-forward folds. Those stay in
``run_benchmark.py``. This entry point only refits LightGBM on the original
feature matrix and on the auction-safe matrix.

Usage
-----
    python run_day_ahead.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.day_ahead import run_comparison


def main() -> None:
    features_path = Path("data/processed/features.parquet")
    if not features_path.exists():
        print("ERROR: data/processed/features.parquet not found.")
        print("Run:  python src/features.py")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    print(f"Loaded features: {df.shape[0]:,} rows × {df.shape[1]} columns")
    leaderboard = run_comparison(df)
    print()
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
