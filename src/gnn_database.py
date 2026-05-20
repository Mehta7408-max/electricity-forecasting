"""
Database Module - SQLite database for storing electricity prices and weather data
"""

import sqlite3
import pandas as pd
from pathlib import Path
from gnn_config import DB_PATH


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_database():
    """Initialize database tables."""
    conn = get_connection()
    
    # Create spot_prices table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spot_prices (
            hour_utc TEXT NOT NULL,
            hour_dk TEXT NOT NULL,
            price_zone TEXT NOT NULL,
            price_dkk REAL NOT NULL,
            price_eur REAL NOT NULL,
            PRIMARY KEY (hour_utc, price_zone)
        )
    """)
    
    # Create weather_data table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            hour_utc TEXT PRIMARY KEY,
            temperature_c REAL,
            wind_speed_ms REAL,
            wind_direction_deg REAL,
            cloud_cover_pct REAL,
            humidity_pct REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized.")


def run_query(query: str, params=None):
    """Execute a query and return results as DataFrame."""
    conn = get_connection()
    
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df


def store_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = "append"):
    """Store a DataFrame to database table."""
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.commit()
    conn.close()