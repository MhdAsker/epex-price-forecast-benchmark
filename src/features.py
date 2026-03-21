import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. DERIVED COLUMNS
    # ------------------------------------------------------------------
    df["wind_total_mw"] = df["wind_onshore_mw"] + df["wind_offshore_mw"]

    # residual_load and renewable_share require load_mw — skip if entirely missing
    if df["load_mw"].isna().all():
        print("WARNING: load_mw is entirely NaN — skipping residual_load_mw and renewable_share")
        df["residual_load_mw"] = float("nan")
        df["renewable_share"] = float("nan")
    else:
        df["residual_load_mw"] = (
            df["load_mw"] - df["wind_onshore_mw"] - df["wind_offshore_mw"] - df["solar_mw"]
        )
        df["renewable_share"] = (
            (df["wind_total_mw"] + df["solar_mw"]) / df["load_mw"]
        ).clip(0, 1)

    # ------------------------------------------------------------------
    # 2. TIME FEATURES
    # ------------------------------------------------------------------
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ------------------------------------------------------------------
    # 3. LAG FEATURES (price_eur_mwh)
    # ------------------------------------------------------------------
    df["price_lag_1h"] = df["price_eur_mwh"].shift(1)
    df["price_lag_24h"] = df["price_eur_mwh"].shift(24)
    df["price_lag_168h"] = df["price_eur_mwh"].shift(168)
    df["price_lag_336h"] = df["price_eur_mwh"].shift(336)

    # ------------------------------------------------------------------
    # 4. ROLLING STATISTICS (shifted by 1 to avoid leakage)
    # ------------------------------------------------------------------
    price = df["price_eur_mwh"]
    df["price_roll_mean_24h"] = price.rolling(24).mean().shift(1)
    df["price_roll_std_24h"] = price.rolling(24).std().shift(1)
    df["price_roll_mean_168h"] = price.rolling(168).mean().shift(1)
    df["price_roll_std_168h"] = price.rolling(168).std().shift(1)

    # ------------------------------------------------------------------
    # 5. RESIDUAL LOAD LAGS
    # ------------------------------------------------------------------
    df["resid_lag_24h"] = df["residual_load_mw"].shift(24)
    df["resid_lag_168h"] = df["residual_load_mw"].shift(168)

    # ------------------------------------------------------------------
    # 6. NEGATIVE PRICE FLAG (target label — kept but not a feature)
    # ------------------------------------------------------------------
    df["negative_price_flag"] = (df["price_eur_mwh"] < 0).astype(int)

    # ------------------------------------------------------------------
    # 7. DROP NaN ROWS (created by lags / rolling windows)
    # ------------------------------------------------------------------
    # First: drop columns that are structurally all-NaN (e.g. load_mw if API
    # returned nothing) so they don't cause dropna() to wipe every row.
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"WARNING: dropping entirely-NaN columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # Then: drop only the warm-up rows created by lag / rolling window features
    lag_rolling_cols = [
        c for c in [
            "price_lag_1h", "price_lag_24h", "price_lag_168h", "price_lag_336h",
            "price_roll_mean_24h", "price_roll_std_24h",
            "price_roll_mean_168h", "price_roll_std_168h",
        ]
        if c in df.columns
    ]
    df = df.dropna(subset=lag_rolling_cols)

    # ------------------------------------------------------------------
    # 8. SAVE
    # ------------------------------------------------------------------
    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path)

    # ------------------------------------------------------------------
    # 9. PRINT SUMMARY
    # ------------------------------------------------------------------
    n = len(df)
    # all columns except price_eur_mwh and negative_price_flag are features
    feature_cols = [
        c for c in df.columns if c not in ("price_eur_mwh", "negative_price_flag")
    ]
    n_features = len(feature_cols)
    min_date = df.index.min()
    max_date = df.index.max()
    neg_hours = df["negative_price_flag"].sum()
    neg_pct = 100 * neg_hours / n
    mean_price = df["price_eur_mwh"].mean()
    std_price = df["price_eur_mwh"].std()

    print(f"Total rows    : {n}")
    print(f"Total features: {n_features}  (excluding target price_eur_mwh)")
    print(f"Date range    : {min_date} to {max_date}")
    print(f"Negative price hours: {neg_hours} ({neg_pct:.1f}%)")
    print(f"Mean price    : {mean_price:.2f} €/MWh")
    print(f"Std price     : {std_price:.2f} €/MWh")

    return df


if __name__ == "__main__":
    raw = pd.read_parquet("data/raw/entso_raw.parquet")
    df = build_features(raw)
    print(df.describe())
