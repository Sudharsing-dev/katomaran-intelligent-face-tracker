"""Database inspection tool for Katomaran Intelligent Face Tracker.

Displays visitor identities, visit history, entry/exit logs, and verified visitor counts.
Usage:
    python scripts/inspect_database.py [--events-limit 20] [--face-id FACE_001]
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


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format tabular data into an ASCII table."""
    if not rows:
        return "  (No records found)"

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [sep, header_line, sep]
    for row in rows:
        line = "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Katomaran Face Tracker SQLite database.")
    parser.add_argument("--events-limit", type=int, default=25, help="Number of recent events to display.")
    parser.add_argument("--face-id", type=str, default=None, help="Filter by specific Face ID.")
    parser.add_argument("--db-path", type=str, default=None, help="Custom database path.")
    args = parser.parse_args()

    config = load_config()
    db_path = Path(args.db_path) if args.db_path else config.get_resolved_database_path()

    if not db_path.exists():
        print(f"Error: Database file not found at '{db_path}'. Run initialization first.")
        return 1

    db_manager = DatabaseManager(db_path)

    # 1. Summary Statistics
    visitor_count = db_manager.get_unique_visitor_count()
    event_counts = db_manager.get_event_counts()

    print("=" * 80)
    print(f"  KATOMARAN DATABASE INSPECTOR — {db_path.name}")
    print("=" * 80)
    print(f"  Database Location       : {db_path}")
  # Visitor count calculated strictly from registered identities
    print(f"  Total Unique Visitors   : {visitor_count}")
    print(f"  Total Entry Events      : {event_counts['ENTRY']}")
    print(f"  Total Exit Events       : {event_counts['EXIT']}")
    print("=" * 80)

    # 2. Registered Visitors
    print("\n--- REGISTERED VISITORS (Persistent Identities) ---")
    visitors = db_manager.get_all_visitors()
    if args.face_id:
        visitors = [v for v in visitors if v["face_id"] == args.face_id]

    v_headers = ["Face ID", "First Seen (UTC)", "Last Seen (UTC)", "Visits", "Status"]
    v_rows = [
        [
            v["face_id"],
            v["first_seen"],
            v["last_seen"],
            str(v["total_visits"]),
            v.get("status", "N/A"),
        ]
        for v in visitors
    ]
    print(format_table(v_headers, v_rows))

    # 3. Recent Events
    print(f"\n--- RECENT EVENT LOGS (Showing up to {args.events_limit}) ---")
    events = db_manager.get_events(face_id=args.face_id, limit=args.events_limit)
    e_headers = ["Event ID", "Face ID", "Type", "Timestamp (UTC)", "Track ID", "Image Path"]
    e_rows = [
        [
            str(e["event_id"]),
            e["face_id"],
            e["event_type"],
            e["timestamp"],
            str(e.get("track_id") or "-"),
            e["image_path"],
        ]
        for e in events
    ]
    print(format_table(e_headers, e_rows))
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
