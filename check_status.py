"""
check_status.py -- pipeline health check for epex-price-forecast-benchmark
"""

import importlib
import io
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows so box-drawing chars don't crash cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# resolve project root regardless of cwd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

SEP = "═" * 44

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────

def ok(msg):  return f"✅ {msg}"
def fail(msg): return f"❌ {msg}"


def check_parquet(rel_path: str) -> tuple[bool, str]:
    p = ROOT / rel_path
    if not p.exists():
        return False, fail(f"{rel_path}  MISSING")
    size_mb = p.stat().st_size / 1_048_576
    try:
        df = pd.read_parquet(p)
        return True, ok(f"{rel_path:<42} ({size_mb:.1f} MB, {len(df):,} rows)")
    except Exception as e:
        return False, fail(f"{rel_path}  exists but unreadable: {e}")


def check_pkl(rel_path: str) -> tuple[bool, str]:
    p = ROOT / rel_path
    if not p.exists():
        return False, fail(f"{rel_path}  MISSING")
    try:
        obj = joblib.load(p)
        size_mb = p.stat().st_size / 1_048_576
        type_name = type(obj).__name__
        return True, ok(f"{rel_path:<42} ({size_mb:.2f} MB, type={type_name})")
    except Exception as e:
        return False, fail(f"{rel_path}  exists but unreadable: {e}")


def check_pt(rel_path: str) -> tuple[bool, str]:
    p = ROOT / rel_path
    if not p.exists():
        return False, fail(f"{rel_path}  MISSING")
    size_mb = p.stat().st_size / 1_048_576
    return True, ok(f"{rel_path:<42} ({size_mb:.2f} MB)")


def check_csv(rel_path: str) -> tuple[bool, str]:
    p = ROOT / rel_path
    if not p.exists():
        return False, fail(f"{rel_path}  MISSING")
    try:
        df = pd.read_csv(p)
        return True, ok(f"{rel_path:<42} ({df.shape[0]} rows × {df.shape[1]} cols)")
    except Exception as e:
        return False, fail(f"{rel_path}  exists but unreadable: {e}")


def check_dir(rel_path: str) -> tuple[bool, str]:
    p = ROOT / rel_path
    exists = p.exists() and p.is_dir()
    label = ok(rel_path) if exists else fail(f"{rel_path}  MISSING")
    return exists, label


def check_file(rel_path: str) -> tuple[bool, str]:
    p = ROOT / rel_path
    exists = p.exists()
    label = ok(rel_path) if exists else fail(f"{rel_path}  MISSING")
    return exists, label


# ────────────────────────────────────────────────────────────────────────────
# 1. File / folder checks
# ────────────────────────────────────────────────────────────────────────────
print(SEP)
print("FILE & FOLDER CHECK")
print(SEP)

checks = {}

# directories
for d in ["data/raw", "data/processed", "models", "results/metrics", "results/plots"]:
    ok_flag, msg = check_dir(d)
    checks[d] = ok_flag
    print(msg)

print()

# parquet files
for rel in ["data/raw/entso_raw.parquet", "data/processed/features.parquet"]:
    ok_flag, msg = check_parquet(rel)
    checks[rel] = ok_flag
    print(msg)

print()

# source files
for rel in [
    "src/fetch.py", "src/features.py", "src/evaluate.py", "src/visualize.py",
    "src/cv.py", "src/day_ahead.py", "src/protocol.py",
    "src/models/sarima_model.py", "src/models/lgbm_model.py",
    "src/models/xgb_model.py",   "src/models/lstm_model.py",
    "src/models/day_ahead_models.py",
    "run_benchmark.py", "run_day_ahead.py", "config/forecast_protocol.json",
]:
    ok_flag, msg = check_file(rel)
    checks[rel] = ok_flag
    print(msg)

print()

# model artefacts
for rel, fn in [("models/sarima_params.joblib", check_pkl),
                ("models/lgbm.pkl",   check_pkl),
                ("models/xgb.pkl",    check_pkl),
                ("models/lstm.pt",    check_pt)]:
    ok_flag, msg = fn(rel)
    checks[rel] = ok_flag
    print(msg)

print()

# results
ok_flag, msg = check_csv("results/metrics/leaderboard.csv")
checks["results/metrics/leaderboard.csv"] = ok_flag
print(msg)

plots_dir = ROOT / "results/plots"
plot_names = ["actual_vs_predicted.png", "error_distribution.png",
              "feature_importance.png", "negative_price_analysis.png"]
n_plots = sum(1 for name in plot_names if (plots_dir / name).exists())
if n_plots == 4:
    print(ok(f"results/plots/ ({n_plots}/4 PNGs present)"))
    checks["plots"] = True
else:
    print(fail(f"results/plots/  only {n_plots}/4 PNGs present"))
    checks["plots"] = False

# ────────────────────────────────────────────────────────────────────────────
# 2. .env check
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print(".ENV CHECK")
print(SEP)

api_key = os.environ.get("ENTSOE_API_KEY", "")
if api_key:
    preview = api_key[:8] + "..." if len(api_key) > 8 else api_key
    print(ok(f"ENTSOE_API_KEY is set (starts with: {preview})"))
    checks["env"] = True
else:
    print(fail("ENTSOE_API_KEY missing — add it to .env file"))
    checks["env"] = False

# ────────────────────────────────────────────────────────────────────────────
# 3. Import checks
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("IMPORT CHECK")
print(SEP)

LIBRARY_MAP = {
    "entsoe-py":      "entsoe",
    "pandas":         "pandas",
    "numpy":          "numpy",
    "scikit-learn":   "sklearn",
    "lightgbm":       "lightgbm",
    "xgboost":        "xgboost",
    "torch":          "torch",
    "matplotlib":     "matplotlib",
    "seaborn":        "seaborn",
    "python-dotenv":  "dotenv",
    "joblib":         "joblib",
    "statsmodels":    "statsmodels",
    "shap":           "shap",
}

all_imports_ok = True
for pkg_name, import_name in LIBRARY_MAP.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(ok(f"{pkg_name:<20} {version}"))
    except ImportError:
        print(fail(f"{pkg_name:<20} NOT INSTALLED"))
        all_imports_ok = False

# ────────────────────────────────────────────────────────────────────────────
# 4 & 5. Pipeline status + next action
# ────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("PIPELINE STATUS CHECK")
print(SEP)

steps = [
    ("Step 1: Data fetched",       checks.get("data/raw/entso_raw.parquet", False)),
    ("Step 2: Features built",     checks.get("data/processed/features.parquet", False)),
    ("Step 3: SARIMA trained",     checks.get("models/sarima_params.joblib", False)),
    ("Step 4: LightGBM trained",   checks.get("models/lgbm.pkl", False)),
    ("Step 5: XGBoost trained",    checks.get("models/xgb.pkl", False)),
    ("Step 6: LSTM trained",       checks.get("models/lstm.pt", False)),
    ("Step 7: Benchmark complete", checks.get("results/metrics/leaderboard.csv", False)),
    ("Step 8: Plots generated",    checks.get("plots", False)),
    ("Step 9: Day-ahead protocol", (ROOT / "results/metrics/day_ahead_primary_leaderboard.csv").exists()),
]

for label, passed in steps:
    icon = "✅" if passed else "❌"
    print(f"[{icon}] {label}")

print(SEP)

# Determine next action
if not checks.get("data/raw/entso_raw.parquet", False):
    next_action = 'Run: python src/fetch.py'
elif not checks.get("data/processed/features.parquet", False):
    next_action = 'Run: python src/features.py'
elif not all([
    checks.get("models/sarima_params.joblib", False),
    checks.get("models/lgbm.pkl",   False),
    checks.get("models/xgb.pkl",    False),
    checks.get("models/lstm.pt",    False),
]):
    next_action = 'Run: python run_benchmark.py'
elif not checks.get("results/metrics/leaderboard.csv", False):
    next_action = 'Run: python run_benchmark.py'
elif not checks.get("plots", False):
    next_action = 'Run: python run_benchmark.py'
elif not (ROOT / "results/metrics/day_ahead_primary_leaderboard.csv").exists():
    next_action = 'Run: python run_day_ahead.py'
else:
    next_action = '✅ ALL STEPS COMPLETE — primary day-ahead results below'

print(f"NEXT ACTION: {next_action}")
print(SEP)

# ────────────────────────────────────────────────────────────────────────────
# 6. Leaderboard (if exists)
# ────────────────────────────────────────────────────────────────────────────
lb_path = ROOT / "results/metrics/day_ahead_primary_leaderboard.csv"
if lb_path.exists():
    print()
    print(SEP + "════")
    print("PRIMARY NOON D-1 BENCHMARK RESULTS")
    print(SEP + "════")

    lb = pd.read_csv(lb_path)
    print(lb.to_string(index=False))
    print()

    best = lb.loc[lb["MAE"].idxmin()]
    print(f"Winner: {best['model']}  (MAE = {best['MAE']:.2f} €/MWh)")
    naive_row = lb[lb["model"] == "Naive D-1"]
    if not naive_row.empty:
        naive_mae = float(naive_row["MAE"].iloc[0])
        improvement = (naive_mae - float(best["MAE"])) / naive_mae * 100
        print(f"MAE improvement over Naive D-1: {improvement:.1f}%")
    print("Legacy hourly scores are diagnostic only; they use same-auction prices.")

    print(SEP + "════")
