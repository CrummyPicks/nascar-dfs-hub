"""Backfill team data in race_results from NASCAR API for 2022-2025 races.

Re-fetches completed race results from the API, which updates the team column
via the existing UPSERT in fetch_and_store_race().

Usage:
    python backfill_teams.py          # Backfill all 2022-2025 races
    python backfill_teams.py --year 2024  # Backfill specific year
    python backfill_teams.py --dry-run    # Show what would be backfilled
"""

import argparse
import sqlite3
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.config import DB_PATH
from src.data import fetch_and_store_race


def get_races_needing_backfill(year=None):
    """Find races with missing team data in race_results."""
    conn = sqlite3.connect(str(DB_PATH))

    year_filter = f"AND r.season = {year}" if year else "AND r.season BETWEEN 2022 AND 2025"

    races = conn.execute(f'''
        SELECT DISTINCT r.id, r.api_race_id, r.series_id, r.season,
               r.race_name, r.race_date
        FROM races r
        JOIN race_results rr ON rr.race_id = r.id
        WHERE r.api_race_id IS NOT NULL
          {year_filter}
          AND (rr.team IS NULL OR rr.team = '')
        ORDER BY r.series_id, r.race_date
    ''').fetchall()

    conn.close()
    return races


def main():
    parser = argparse.ArgumentParser(description="Backfill team data from NASCAR API")
    parser.add_argument("--year", type=int, help="Specific year to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be backfilled")
    args = parser.parse_args()

    races = get_races_needing_backfill(args.year)
    series_names = {1: "Cup", 2: "O'Reilly", 3: "Truck"}

    print(f"\n{'='*60}")
    print(f"  Team Data Backfill — {len(races)} races")
    print(f"{'='*60}\n")

    if not races:
        print("  All races already have team data!")
        return

    if args.dry_run:
        for r in races:
            series = series_names.get(r[2], str(r[2]))
            print(f"  [{series}] {r[3]} {r[4]} ({r[5][:10]})")
        print(f"\n  {len(races)} races would be updated. Run without --dry-run to execute.")
        return

    success = 0
    failed = 0
    for i, r in enumerate(races):
        db_id, api_id, series_id, season, name, date = r
        series = series_names.get(series_id, str(series_id))

        print(f"  [{i+1}/{len(races)}] {series} {season} — {name}...", end=" ", flush=True)

        try:
            result = fetch_and_store_race(series_id, api_id, season)
            if result["status"] == "success":
                print(f"OK ({result['drivers']} drivers)")
                success += 1
            else:
                print(f"FAIL: {result.get('error', 'unknown')}")
                failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

        # Rate limit: be gentle with the API
        time.sleep(0.5)

    print(f"\n  Done! {success} updated, {failed} failed.")

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    remaining = conn.execute('''
        SELECT COUNT(DISTINCT r.id)
        FROM races r
        JOIN race_results rr ON rr.race_id = r.id
        WHERE r.season BETWEEN 2022 AND 2025
          AND (rr.team IS NULL OR rr.team = '')
    ''').fetchone()[0]
    conn.close()
    print(f"  Races still missing team data: {remaining}")


if __name__ == "__main__":
    main()
