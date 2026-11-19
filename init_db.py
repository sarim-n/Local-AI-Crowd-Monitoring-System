#!/usr/bin/env python3
"""
CLI script to initialize the crowd monitoring database schema.
"""

import sys
from src.storage.database import DB_DIALECT, SessionLocal, init_db
from src.storage.repository import MonitoringRepository


def main():
    print(f"Initializing crowd monitoring database schema (Dialect: {DB_DIALECT})...")
    init_db()

    db = SessionLocal()
    try:
        repo = MonitoringRepository(db)
        zones = repo.get_zones()
        print(f"Database schema initialized successfully. Configured custom zones: {len(zones)}.")
        return 0
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
