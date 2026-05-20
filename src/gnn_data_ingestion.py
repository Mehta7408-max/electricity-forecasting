"""
Data Ingestion Module - Fetch electricity prices and weather data from free APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from gnn_config import (
    ENERGY_CHARTS_API, OPENMETEO_ARCHIVE_API, OPENMETEO_FORECAST_API,
    DENMARK_LAT, DENMARK_LON, PRICE_ZONES, HISTORICAL_DAYS, EUR_TO_DKK
)
from gnn_database import get_connection, init_database


def fetch_spot_prices(days_back: int = HISTORICAL_DAYS) -> pd.DataFrame:
    """
    Fetch spot prices from Energy-Charts API.
    
    Args:
        days_back: Number of days of historical data to fetch.
    
    Returns:
        DataFrame with columns: hour_utc, hour_dk, price_zone, price_dkk, price_eur
    """
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00Z")
    end = now.strftime("%Y-%m-%dT23:59Z")
    
    all_records = []
    
    for zone in PRICE_ZONES:
        print(f"📡 Fetching spot prices for {zone}...")
        
        try:
            params = {
                "bzn": zone,
                "start": start,
                "end": end,
            }
            
            response = requests.get(ENERGY_CHARTS_API, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            unix_seconds = data.get("unix_seconds", [])
            prices_eur_mwh = data.get("price", [])
            
            if not unix_seconds:
                print(f"⚠️  No data returned for {zone}")
                continue
            
            print(f"📦 Received {len(unix_seconds)} hourly records for {zone}")
            
            # Convert to records
            for ts, price_eur_mwh in zip(unix_seconds, prices_eur_mwh):
                if price_eur_mwh is None:
                    continue
                
                dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
                # Danish time offset
                dk_offset = 2 if 3 <= dt_utc.month <= 10 else 1
                dt_dk = dt_utc + timedelta(hours=dk_offset)
                
                # Convert EUR/MWh to DKK/kWh
                price_eur_kwh = price_eur_mwh / 1000.0
                price_dkk_kwh = price_eur_kwh * EUR_TO_DKK
                
                all_records.append({
                    "hour_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour_dk": dt_dk.strftime("%Y-%m-%d %H:%M:%S"),
                    "price_zone": zone,
                    "price_dkk": round(price_dkk_kwh, 6),
                    "price_eur": round(price_eur_kwh, 6),
                })
            
        except Exception as e:
            print(f"❌ Request failed for {zone}: {e}")
    
    df = pd.DataFrame(all_records)
    if not df.empty:
        df["hour_utc"] = pd.to_datetime(df["hour_utc"])
        df["hour_dk"] = pd.to_datetime(df["hour_dk"])
        df = df.sort_values("hour_utc").reset_index(drop=True)
    
    print(f"✅ Fetched {len(df)} price records total.")
    return df


def fetch_weather_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo.
    
    Args:
        start_date: Start date string "YYYY-MM-DD"
        end_date: End date string "YYYY-MM-DD"
    
    Returns:
        DataFrame with hourly weather data.
    """
    print(f"📡 Fetching weather data ({start_date} to {end_date})...")
    
    hourly_vars = "temperature_2m,wind_speed_10m,wind_direction_10m,cloud_cover,relative_humidity_2m"
    all_dfs = []
    
    now = datetime.now(timezone.utc)
    archive_cutoff = (now - timedelta(days=5)).strftime("%Y-%m-%d")
    
    # Part 1: Archive API for older data
    if start_date < archive_cutoff:
        archive_end = min(end_date, archive_cutoff)
        params = {
            "latitude": DENMARK_LAT,
            "longitude": DENMARK_LON,
            "hourly": hourly_vars,
            "start_date": start_date,
            "end_date": archive_end,
            "timezone": "UTC",
        }
        try:
            response = requests.get(OPENMETEO_ARCHIVE_API, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            hourly = data["hourly"]
            df = pd.DataFrame({
                "hour_utc": pd.to_datetime(hourly["time"]),
                "temperature_c": hourly["temperature_2m"],
                "wind_speed_ms": hourly["wind_speed_10m"],
                "wind_direction_deg": hourly["wind_direction_10m"],
                "cloud_cover_pct": hourly["cloud_cover"],
                "humidity_pct": hourly["relative_humidity_2m"],
            })
            all_dfs.append(df)
            print(f"📦 Archive weather: {len(df)} records")
        except Exception as e:
            print(f"⚠️  Archive weather failed: {e}")
    
    # Part 2: Forecast API for recent data
    if end_date >= archive_cutoff:
        params = {
            "latitude": DENMARK_LAT,
            "longitude": DENMARK_LON,
            "hourly": hourly_vars,
            "past_days": 5,
            "forecast_days": 2,
            "timezone": "UTC",
        }
        try:
            response = requests.get(OPENMETEO_FORECAST_API, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            hourly = data["hourly"]
            df = pd.DataFrame({
                "hour_utc": pd.to_datetime(hourly["time"]),
                "temperature_c": hourly["temperature_2m"],
                "wind_speed_ms": hourly["wind_speed_10m"],
                "wind_direction_deg": hourly["wind_direction_10m"],
                "cloud_cover_pct": hourly["cloud_cover"],
                "humidity_pct": hourly["relative_humidity_2m"],
            })
            all_dfs.append(df)
            print(f"📦 Forecast weather: {len(df)} records")
        except Exception as e:
            print(f"⚠️  Forecast weather failed: {e}")
    
    if not all_dfs:
        print("❌ No weather data fetched.")
        return pd.DataFrame()
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["hour_utc"]).sort_values("hour_utc").reset_index(drop=True)
    
    print(f"✅ Fetched {len(result)} weather records total.")
    return result


def store_spot_prices(df: pd.DataFrame):
    """Save spot prices to database."""
    if df.empty:
        return
    
    conn = get_connection()
    
    for _, row in df.iterrows():
        try:
            conn.execute(
                """INSERT OR IGNORE INTO spot_prices 
                   (hour_utc, hour_dk, price_zone, price_dkk, price_eur)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    str(row["hour_utc"]),
                    str(row["hour_dk"]),
                    row["price_zone"],
                    row["price_dkk"],
                    row["price_eur"],
                ),
            )
        except Exception:
            pass
    
    conn.commit()
    conn.close()
    print(f"💾 Stored price records to database.")


def store_weather_data(df: pd.DataFrame):
    """Save weather data to database."""
    if df.empty:
        return
    
    conn = get_connection()
    
    for _, row in df.iterrows():
        try:
            conn.execute(
                """INSERT OR IGNORE INTO weather_data 
                   (hour_utc, temperature_c, wind_speed_ms, wind_direction_deg, 
                    cloud_cover_pct, humidity_pct)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(row["hour_utc"]),
                    row["temperature_c"],
                    row["wind_speed_ms"],
                    row["wind_direction_deg"],
                    row["cloud_cover_pct"],
                    row["humidity_pct"],
                ),
            )
        except Exception:
            pass
    
    conn.commit()
    conn.close()
    print(f"💾 Stored weather records to database.")


def run_ingestion(days_back: int = HISTORICAL_DAYS):
    """Run the complete data ingestion pipeline."""
    print("\n🔄 Starting data ingestion...")
    
    # Initialize database
    init_database()
    
    # Fetch electricity prices
    prices_df = fetch_spot_prices(days_back)
    store_spot_prices(prices_df)
    
    # Fetch weather data
    if not prices_df.empty:
        price_start = prices_df["hour_utc"].min().strftime("%Y-%m-%d")
        price_end = prices_df["hour_utc"].max().strftime("%Y-%m-%d")
        print(f"📅 Price data range: {price_start} to {price_end}")
        
        weather_df = fetch_weather_data(price_start, price_end)
        store_weather_data(weather_df)
    else:
        print("⚠️  No price data - skipping weather fetch.")
    
    print("✅ Data ingestion complete!\n")
    return prices_df


if __name__ == "__main__":
    run_ingestion()