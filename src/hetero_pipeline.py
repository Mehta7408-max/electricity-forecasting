# OVERWRITE EXACTLY: src/hetero_pipeline.py
import pandas as pd
import numpy as np
from gnn_database import run_query

def prepare_multi_area_data():
    """Extracts, processes, and aligns multi-zone spot prices and weather inputs cleanly."""
    print("\n🔄 Extracting Production 4-Area System Data from SQLite...")
    
    # 1. Pull market spot prices
    df_prices = run_query("SELECT hour_utc as timestamp, price_zone, price_dkk FROM spot_prices")
    if df_prices.empty:
        raise ValueError("❌ spot_prices table is completely empty! Please run your scraper data ingestion scripts first.")
        
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp']).dt.strftime('%Y-%m-%d %H:00:00')
    
    # 2. Pull local atmospheric weather measurements
    df_weather = run_query("SELECT * FROM weather_data")
    if df_weather.empty:
        print("⚠️ Warning: weather_data table is empty. Generating synthetic baseline parameters.")
        unique_times = df_prices['timestamp'].unique()
        df_weather = pd.DataFrame({
            'hour_utc': unique_times,
            'temperature_c': np.random.uniform(5.0, 18.0, len(unique_times)),
            'wind_speed_ms': np.random.uniform(2.0, 12.0, len(unique_times)),
            'cloud_cover_pct': np.random.uniform(10.0, 90.0, len(unique_times)),
            'humidity_pct': np.random.uniform(60.0, 95.0, len(unique_times))
        })
    
    df_weather['timestamp'] = pd.to_datetime(df_weather['hour_utc']).dt.strftime('%Y-%m-%d %H:00:00')
    df_weather = df_weather.drop(columns=['hour_utc'])
    
    # 3. Extract individual zone data frames
    df_dk1 = df_prices[df_prices['price_zone'] == 'DK1'].copy().reset_index(drop=True)
    df_dk2 = df_prices[df_prices['price_zone'] == 'DK2'].copy().reset_index(drop=True)
    df_de = df_prices[df_prices['price_zone'] == 'DE'].copy().reset_index(drop=True)
    df_hydro = df_prices[df_prices['price_zone'] == 'HYDRO'].copy().reset_index(drop=True)
    
    # 4. Timeline Synchronization - Strict ffill only (historical lookup), NO global bfill
    print("   Synchronizing timelines via outer-join...")
    df_master = df_dk1[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'dk1'})
    df_master = df_master.merge(df_dk2[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'dk2'}), on='timestamp', how='outer')
    df_master = df_master.merge(df_de[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'de'}), on='timestamp', how='outer')
    df_master = df_master.merge(df_hydro[['timestamp', 'price_dkk']].rename(columns={'price_dkk': 'hydro'}), on='timestamp', how='outer')
    
    # Leakage fixed: Forward-fill values up to the gap, fill remaining historic edges with neutral 0.0
    df_master = df_master.sort_values('timestamp').ffill().fillna(0.0)
    
    # Re-split back into standardized individual zone views
    df_dk1 = df_master[['timestamp', 'dk1']].rename(columns={'dk1': 'price_dkk'})
    df_dk2 = df_master[['timestamp', 'dk2']].rename(columns={'dk2': 'price_dkk'})
    df_de = df_master[['timestamp', 'de']].rename(columns={'de': 'price_dkk'})
    df_hydro = df_master[['timestamp', 'hydro']].rename(columns={'hydro': 'price_dkk'})
    
    # 5. Connect weather factors
    df_dk1 = df_dk1.merge(df_weather, on='timestamp', how='left')
    df_dk2 = df_dk2.merge(df_weather, on='timestamp', how='left')
    
    # Fill remaining gaps using safe forward-fills and neutral medians to prevent look-ahead leaks
    for df in [df_dk1, df_dk2]:
        df['temperature_c'] = df['temperature_c'].ffill().fillna(12.0)
        df['wind_speed_ms'] = df['wind_speed_ms'].ffill().fillna(5.0)
        df['cloud_cover_pct'] = df['cloud_cover_pct'].ffill().fillna(50.0)
        df['humidity_pct'] = df['humidity_pct'].ffill().fillna(75.0)
        
    # 6. Apply Feature Engineering
    print("   Engineering historical lag attributes on synchronized frames...")
    df_dk1 = _add_autoregressive_features(df_dk1)
    df_dk2 = _add_autoregressive_features(df_dk2)
    df_de = _add_autoregressive_features(df_de)
    df_hydro = _add_autoregressive_features(df_hydro)
    
    print(f"   Pipeline successfully returning data frames. Total size: {len(df_dk1)} records.")
    return df_dk1, df_dk2, df_hydro, df_de, df_weather

def _add_autoregressive_features(df):
    """Calculates temporal lag sequences securely without backwards information leakage."""
    df['price_lag_1h'] = df['price_dkk'].shift(1)
    df['price_lag_2h'] = df['price_dkk'].shift(2)
    df['price_lag_6h'] = df['price_dkk'].shift(6)
    
    # Rolling attributes derived securely from historical offsets
    df['price_rolling_6h_mean'] = df['price_dkk'].shift(1).rolling(6).mean()
    df['price_rolling_6h_std'] = df['price_dkk'].shift(1).rolling(6).std()
    
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
    
    # Leakage fixed: Replace initial cold-start NaNs with 0.0 instead of backfilling future data
    return df.fillna(0.0).reset_index(drop=True)