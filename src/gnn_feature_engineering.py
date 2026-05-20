"""
Feature Engineering Module - Create features for GNN nodes
"""

import pandas as pd
import numpy as np
from gnn_database import run_query
from gnn_config import DEFAULT_ZONE, NODE_FEATURES


def load_raw_data(price_zone: str = DEFAULT_ZONE) -> pd.DataFrame:
    """
    Load and merge price + weather data from database.
    
    Args:
        price_zone: Which Danish price zone (DK1 or DK2).
    
    Returns:
        Merged DataFrame sorted by time.
    """
    print(f"📊 Loading data for {price_zone}...")
    
    # Load prices for the specified zone
    prices = run_query(
        "SELECT hour_utc, hour_dk, price_dkk FROM spot_prices WHERE price_zone = ?",
        params=[price_zone],
    )
    
    # Load weather data
    weather = run_query("SELECT * FROM weather_data")
    
    if prices.empty:
        print("⚠️  No price data found in database.")
        return pd.DataFrame()
    
    # Convert to datetime
    prices["hour_utc"] = pd.to_datetime(prices["hour_utc"])
    prices["hour_dk"] = pd.to_datetime(prices["hour_dk"])
    
    if not weather.empty:
        weather["hour_utc"] = pd.to_datetime(weather["hour_utc"])
        df = prices.merge(weather, on="hour_utc", how="left")
        overlap = df["temperature_c"].notna().sum()
        print(f"📊 Weather overlap: {overlap}/{len(df)} rows")
    else:
        df = prices.copy()
        print("⚠️  No weather data available")
    
    df = df.sort_values("hour_utc").reset_index(drop=True)
    print(f"📊 Loaded {len(df)} rows of merged data")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features for GNN nodes.
    
    Args:
        df: Merged price + weather DataFrame.
    
    Returns:
        DataFrame with all features added.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    print("🔧 Creating features...")
    
    # ── Time Features ──────────────────────────────────────────────
    df["hour"] = df["hour_dk"].dt.hour
    df["day_of_week"] = df["hour_dk"].dt.dayofweek
    df["month"] = df["hour_dk"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # ── Lag Features ───────────────────────────────────────────────
    df["price_lag_1h"] = df["price_dkk"].shift(1)
    df["price_lag_2h"] = df["price_dkk"].shift(2)
    df["price_lag_24h"] = df["price_dkk"].shift(24)
    df["price_lag_48h"] = df["price_dkk"].shift(48)
    df["price_lag_168h"] = df["price_dkk"].shift(168)
    
    # ── Rolling Statistics ─────────────────────────────────────────
    df["price_rolling_6h_mean"] = df["price_dkk"].rolling(6).mean()
    df["price_rolling_12h_mean"] = df["price_dkk"].rolling(12).mean()
    df["price_rolling_24h_mean"] = df["price_dkk"].rolling(24).mean()
    df["price_rolling_6h_std"] = df["price_dkk"].rolling(6).std()
    df["price_rolling_24h_std"] = df["price_dkk"].rolling(24).std()
    
    # ── Weather Features ───────────────────────────────────────────
    weather_cols = ["temperature_c", "wind_speed_ms", "wind_direction_deg",
                    "cloud_cover_pct", "humidity_pct"]
    
    defaults = {
        "temperature_c": 10.0,
        "wind_speed_ms": 5.0,
        "wind_direction_deg": 180.0,
        "cloud_cover_pct": 50.0,
        "humidity_pct": 70.0,
    }
    
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(defaults[col])
        else:
            df[col] = defaults[col]
    
    # ── Drop rows with NaN from lag features ───────────────────────
    lag_and_rolling_cols = [
        "price_lag_1h", "price_lag_2h", "price_lag_24h",
        "price_lag_48h", "price_lag_168h",
        "price_rolling_6h_mean", "price_rolling_12h_mean",
        "price_rolling_24h_mean", "price_rolling_6h_std",
        "price_rolling_24h_std",
    ]
    
    df = df.dropna(subset=lag_and_rolling_cols).reset_index(drop=True)
    
    print(f"✅ Created features. Final dataset: {len(df)} rows, {len(NODE_FEATURES)} features")
    
    return df


def prepare_data_for_gnn(price_zone: str = DEFAULT_ZONE):
    """
    Full pipeline: load data → create features → return prepared DataFrame.
    
    Returns:
        DataFrame with all features for GNN.
    """
    print("\n🔧 Starting feature engineering for GNN...")
    
    df = load_raw_data(price_zone)
    if df.empty:
        return None
    
    df = create_features(df)
    if df.empty:
        return None
    
    print(f"✅ Data ready for GNN: {len(df)} timesteps\n")
    
    return df


if __name__ == "__main__":
    df = prepare_data_for_gnn()
    if df is not None:
        print(f"\nFeature summary:")
        print(df[NODE_FEATURES].describe().round(3))