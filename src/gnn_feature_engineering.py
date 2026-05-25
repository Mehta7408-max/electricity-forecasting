# gnn_feature_engineering.py
import pandas as pd
from gnn_database import run_query

def load_raw_data(price_zone: str = "DK1"):
    """Load price data safely."""
    query = """
        SELECT hour_utc, price_dkk 
        FROM spot_prices 
        WHERE price_zone = ?
        ORDER BY hour_utc ASC
    """
    df = run_query(query, params=[price_zone])
    
    if df.empty:
        print(f"⚠️ No data found for {price_zone}")
        return pd.DataFrame()
    
    df['hour_utc'] = pd.to_datetime(df['hour_utc'])
    print(f"✅ Loaded {len(df)} records for {price_zone}")
    return df


def create_features(df: pd.DataFrame):
    """Create features safely."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Time features
    df['hour_of_day'] = df['hour_utc'].dt.hour
    df['day_of_week'] = df['hour_utc'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Price lags
    df['price_lag_1h'] = df['price_dkk'].shift(1)
    df['price_lag_24h'] = df['price_dkk'].shift(24)
    df['price_lag_168h'] = df['price_dkk'].shift(168)
    
    # Rolling statistics
    df['price_rolling_6h_mean'] = df['price_dkk'].rolling(6).mean()
    df['price_rolling_24h_mean'] = df['price_dkk'].rolling(24).mean()
    
    # Drop initial NaNs
    df = df.dropna().reset_index(drop=True)
    
    return df


def prepare_data_for_gnn(price_zone: str = "DK1"):
    df = load_raw_data(price_zone)
    if df.empty:
        return None
    return create_features(df)


if __name__ == "__main__":
    df = prepare_data_for_gnn("DK1")
    print(df.head() if df is not None else "No data")