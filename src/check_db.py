# SAVE AS: src/check_db.py
import sqlite3
from pathlib import Path

def list_tables():
    # Attempt to locate your database file
    # If you know the specific path, replace the string below
    db_path = Path("data/electricity_market.db") 
    
    if not db_path.exists():
        print(f"❌ Database file not found at: {db_path.absolute()}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"✅ Connected to {db_path.name}")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n📊 Tables found in database:")
    for table in tables:
        print(f" - {table[0]}")
    
    conn.close()

if __name__ == "__main__":
    list_tables().py