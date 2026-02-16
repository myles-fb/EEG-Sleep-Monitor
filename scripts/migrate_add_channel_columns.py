"""Add channel_index and channel_label to feature_records, channels_json to studies.

Idempotent: safe to run multiple times (ignores 'duplicate column' errors).
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "physician.db"


def migrate(db_path: Path = DB_PATH):
    if not db_path.exists():
        print(f"Database not found at {db_path}; nothing to migrate.")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    alterations = [
        "ALTER TABLE feature_records ADD COLUMN channel_index INTEGER",
        "ALTER TABLE feature_records ADD COLUMN channel_label TEXT",
        "ALTER TABLE studies ADD COLUMN channels_json TEXT",
    ]

    for sql in alterations:
        try:
            cursor.execute(sql)
            print(f"OK: {sql}")
        except sqlite3.OperationalError as e:
            if "duplicate column" in str(e).lower():
                print(f"SKIP (already exists): {sql}")
            else:
                raise

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
