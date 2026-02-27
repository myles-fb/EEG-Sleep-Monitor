#!/usr/bin/env python
"""Migration: add users table and user_id columns to patients/devices.

For existing databases that were created before the auth feature.
For fresh installs, init_db() handles everything automatically.

Usage:
    python scripts/migrate_add_users.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from sqlalchemy import inspect, text
from models.database import engine


def migrate():
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    with engine.begin() as conn:
        # 1. Create users table if it doesn't exist
        if "users" not in existing_tables:
            conn.execute(text("""
                CREATE TABLE users (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    email VARCHAR NOT NULL UNIQUE,
                    password_hash VARCHAR NOT NULL,
                    display_name VARCHAR NOT NULL,
                    created_at DATETIME
                )
            """))
            print("Created 'users' table.")
        else:
            print("'users' table already exists.")

        # 2. Add user_id column to patients if missing
        patient_cols = {c["name"] for c in inspector.get_columns("patients")}
        if "user_id" not in patient_cols:
            conn.execute(text(
                "ALTER TABLE patients ADD COLUMN user_id VARCHAR REFERENCES users(id)"
            ))
            print("Added 'user_id' column to patients.")
        else:
            print("patients.user_id already exists.")

        # 3. Add user_id column to devices if missing
        device_cols = {c["name"] for c in inspector.get_columns("devices")}
        if "user_id" not in device_cols:
            conn.execute(text(
                "ALTER TABLE devices ADD COLUMN user_id VARCHAR REFERENCES users(id)"
            ))
            print("Added 'user_id' column to devices.")
        else:
            print("devices.user_id already exists.")

    print("Migration complete.")


if __name__ == "__main__":
    migrate()
