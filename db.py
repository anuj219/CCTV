# SQLITE db to store recognitions/detections of known/unknown people
import sqlite3
from datetime import datetime, timedelta

DB_FILE = "cctv_logs.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encounters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                headcount INTEGER,
                image_path TEXT
            )
        """)
        conn.commit()

def should_log_encounter(person_name, cooldown_seconds=60):
    """Returns True if the person hasn't been logged in the last cooldown period."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cutoff = datetime.utcnow() - timedelta(seconds=cooldown_seconds)
        cursor.execute("""
            SELECT id FROM encounters 
            WHERE person_name = ? AND timestamp >= ?
        """, (person_name, cutoff.strftime("%Y-%m-%d %H:%M:%S")))
        return cursor.fetchone() is None

def log_encounter(person_name, confidence, headcount, image_path=""):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO encounters (person_name, confidence, headcount, image_path)
            VALUES (?, ?, ?, ?)
        """, (person_name, confidence, headcount, image_path))
        conn.commit()