"""Database initialization script for Katomaran Intelligent Face Tracker.

Creates tables, constraints, and indexes. Can be run manually or during setup.
Usage:
    python scripts/initialize_database.py [--force-recreate]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.database import DatabaseManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize Katomaran Face Tracker SQLite database.")
    parser.add_argument("--force-recreate", action="store_true", help="Delete existing database before initializing.")
    args = parser.parse_args()

    config = load_config()
    db_path = config.get_resolved_database_path()

    print(f"Target database path: {db_path}")

    if args.force_recreate and db_path.exists():
        print(f"Removing existing database at {db_path} (--force-recreate)...")
        db_path.unlink()

    db_manager = DatabaseManager(db_path)
    count = db_manager.get_unique_visitor_count()
    events = db_manager.get_event_counts()

    print("\n[SUCCESS] Database initialized successfully:")
    print(f"  - Database file: {db_path}")
    print(f"  - Registered visitors: {count}")
    print(f"  - Entry events: {events['ENTRY']}")
    print(f"  - Exit events: {events['EXIT']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
