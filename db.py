# SQLITE db to store recognitions/detections of known/unknown people
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cctv_logs.db")

def init_db():
    with closing(sqlite3.connect(DB_FILE)) as conn, conn:
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL DEFAULT 'entry',
                person_name TEXT,
                status TEXT NOT NULL,
                confidence REAL,
                headcount INTEGER NOT NULL DEFAULT 0,
                camera_id TEXT NOT NULL DEFAULT 'front_door',
                track_id TEXT,
                image_path TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                summary_text TEXT NOT NULL,
                reason TEXT,
                headcount INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES entry_events(id)
            )
        """)
        summary_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(ai_summaries)")
        }
        for column_name, column_definition in (
            ("reason", "TEXT"),
            ("headcount", "INTEGER NOT NULL DEFAULT 0"),
            ("error", "TEXT"),
        ):
            if column_name not in summary_columns:
                cursor.execute(
                    f"ALTER TABLE ai_summaries ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()


def create_entry_event(status, headcount, track_id=None, event_time=None):
    """Create an entry record, including provisional unconfirmed entries."""
    event_time = event_time or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with closing(sqlite3.connect(DB_FILE)) as conn, conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO entry_events (event_time, status, headcount, track_id)
            VALUES (?, ?, ?, ?)
        """, (event_time, status, headcount, track_id))
        conn.commit()
        return cursor.lastrowid


def update_entry_event(event_id, status, person_name=None, confidence=None):
    """Resolve a provisional event without creating a duplicate crossing row."""
    with closing(sqlite3.connect(DB_FILE)) as conn, conn:
        conn.execute("""
            UPDATE entry_events
            SET status = ?, person_name = ?, confidence = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, person_name, confidence, event_id))
        conn.commit()


def should_log_entry(person_name, cooldown_seconds=60, exclude_event_id=None):
    """Check the durable cooldown for confirmed entries for a named person."""
    cutoff = datetime.utcnow() - timedelta(seconds=cooldown_seconds)
    query = """
        SELECT id FROM entry_events
        WHERE person_name = ? AND status = 'known' AND event_time >= ?
    """
    params = [person_name, cutoff.strftime("%Y-%m-%d %H:%M:%S")]
    if exclude_event_id is not None:
        query += " AND id != ?"
        params.append(exclude_event_id)
    with closing(sqlite3.connect(DB_FILE)) as conn:
        return conn.execute(query, params).fetchone() is None


def create_ai_summary(event_id, summary_text, reason=None, headcount=0, error=None):
    """Persist a scene summary tied to a valid entry event."""
    with closing(sqlite3.connect(DB_FILE)) as conn, conn:
        cursor = conn.cursor()
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(ai_summaries)")}
        values = {
            "event_id": event_id,
            "summary_text": summary_text,
            "reason": reason,
            "headcount": headcount,
            "error": error,
            "trigger_reason": reason,
        }
        insert_columns = [name for name in values if name in columns]
        placeholders = ", ".join("?" for _ in insert_columns)
        column_list = ", ".join(insert_columns)
        cursor.execute(
            f"INSERT INTO ai_summaries ({column_list}) VALUES ({placeholders})",
            [values[name] for name in insert_columns],
        )
        conn.commit()
        return cursor.lastrowid


def should_log_encounter(person_name, cooldown_seconds=60):
    """Returns True if the person hasn't been logged in the last cooldown period."""
    with closing(sqlite3.connect(DB_FILE)) as conn, conn:
        cursor = conn.cursor()
        cutoff = datetime.utcnow() - timedelta(seconds=cooldown_seconds)
        cursor.execute("""
            SELECT id FROM encounters 
            WHERE person_name = ? AND timestamp >= ?
        """, (person_name, cutoff.strftime("%Y-%m-%d %H:%M:%S")))
        return cursor.fetchone() is None

def log_encounter(person_name, confidence, headcount, image_path=""):
    with closing(sqlite3.connect(DB_FILE)) as conn, conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO encounters (person_name, confidence, headcount, image_path)
            VALUES (?, ?, ?, ?)
        """, (person_name, confidence, headcount, image_path))
        conn.commit()