import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from cctv import db


class DatabaseMigrationTests(unittest.TestCase):
    def test_legacy_ai_summaries_schema_accepts_current_summary_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "legacy.db")
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute("""
                    CREATE TABLE ai_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id INTEGER,
                        track_id TEXT,
                        trigger_reason TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        image_path TEXT,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            with patch.object(db, "DB_FILE", database_path):
                db.init_db()
                event_id = db.create_entry_event(
                    "confirmed_unknown",
                    1,
                    "track-test",
                )
                db.create_ai_summary(
                    event_id,
                    "fallback summary",
                    reason="unknown_linger",
                    headcount=1,
                    error="service unavailable",
                )

                with closing(sqlite3.connect(database_path)) as connection:
                    row = connection.execute(
                        "SELECT reason, trigger_reason, headcount, error "
                        "FROM ai_summaries"
                    ).fetchone()

            self.assertEqual(
                row,
                ("unknown_linger", "unknown_linger", 1, "service unavailable"),
            )


if __name__ == "__main__":
    unittest.main()