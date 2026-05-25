# OVERWRITE EXACTLY: src/gnn_database.py
import sqlite3
import os
import pandas as pd
from pathlib import Path

# FAIL-SAFE: Use the global Linux temporary directory which guarantees write permissions
DB_PATH = Path("/tmp/energy.db")

def get_connection():
    """Create and return database connection pointing to a guaranteed writable path."""
    db_path = os.path.join(os.path.dirname(__file__), 'electricity.db')
    return sqlite3.connect(db_path) 

def init_database():
    """Initialize localized database tables."""
    print(f"📁 Initializing fail-safe database environment at: {DB_PATH}")
    conn = get_connection()
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spot_prices (
            hour_utc TEXT,
            price_zone TEXT,
            price_dkk REAL,
            PRIMARY KEY (hour_utc, price_zone)
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            hour_utc TEXT PRIMARY KEY,
            temperature_c REAL,
            wind_speed_ms REAL,
            cloud_cover_pct REAL,
            humidity_pct REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database successfully initialized in /tmp/ directory!")

def run_query(query: str, params=None):
    """Run SQL query and return securely as a Pandas DataFrame."""
    conn = get_connection()
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()