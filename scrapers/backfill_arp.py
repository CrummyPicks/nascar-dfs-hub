"""
Backfill Average Running Position (ARP) for historical races.

Fetches lap-times data from the NASCAR API for each race in the database
that doesn't yet have ARP data, computes the average running position
per driver, and updates the race_results table.

Usage:
    python scrapers/backfill_arp.py                  # all races
    python scrapers/backfill_arp.py --season 2025    # specific season
    python scrapers/backfill_arp.py --dry-run        # preview without writing
"""

import sqlite3
import requests
import time
import argparse
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "nascar.db")
NASCAR_API_BASE = "https://cf.nascar.com/cacher"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def compute_arp_from_laps(lap_data: dict) -> dict:
    """Compute average running position per driver from lap-times JSON.

    Returns {driver_fullname: avg_running_position}.
    """
    drivers = lap_data.get("laps", [])
    result = {}
    for d in drivers:
        full_name = d.get("FullName", "").strip()
        if not full_name:
            continue
        # Clean name same way as _clean_api_name
        import re
        full_name = re.sub(r'^\*\s*', '', full_name)
        full_name = re.sub(r'\s*#$', '', full_name)
        full_name = re.sub(r'\s*\([a-zA-Z]\)$', '', full_name)
        full_name = re.sub(r'\bJr\.\s*$', 'Jr', full_name)
        full_name = re.sub(r'\bSr\.\s*$', 'Sr', full_name)
        full_name = full_name.strip()

        positions = [lap["RunningPos"] for lap in d.get("Laps", [])
                     if lap.get("Lap", 0) > 0 and lap.get("RunningPos")]
        if positions:
            result[full_name] = round(np.mean(positions), 1)
    return result


def backfill(season=None, dry_run=False):
    """Backfill ARP for races missing it."""
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Find races that need ARP backfill (have results but no ARP data)
    query = """
        SELECT DISTINCT r.id as db_race_id, r.api_race_id, r.series_id, r.season,
               r.race_name, r.race_date
        FROM races r
        JOIN race_results rr ON rr.race_id = r.id
        WHERE r.api_race_id IS NOT NULL
          AND rr.avg_running_position IS NULL
    """
    params = []
    if season:
        query += " AND r.season = ?"
        params.append(season)
    query += " ORDER BY r.season, r.race_date"

    races = conn.execute(query, params).fetchall()
    print(f"Found {len(races)} races needing ARP backfill")

    updated_total = 0
    failed = 0

    for i, race in enumerate(races):
        db_race_id = race["db_race_id"]
        api_race_id = race["api_race_id"]
        series_id = race["series_id"]
        year = race["season"]
        name = race["race_name"]

        print(f"\n[{i+1}/{len(races)}] {year} {name} (series={series_id}, api_id={api_race_id})")

        # Fetch lap times from NASCAR API
        url = f"{NASCAR_API_BASE}/{year}/{series_id}/{api_race_id}/lap-times.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} — skipping")
                failed += 1
                continue
            lap_data = r.json()
        except Exception as e:
            print(f"  Error fetching: {e}")
            failed += 1
            continue

        # Compute ARP
        arp_map = compute_arp_from_laps(lap_data)
        if not arp_map:
            print(f"  No lap data available")
            failed += 1
            continue

        # Match to DB drivers and update
        db_drivers = conn.execute(
            """SELECT rr.id as rr_id, d.full_name
               FROM race_results rr
               JOIN drivers d ON d.id = rr.driver_id
               WHERE rr.race_id = ?""",
            (db_race_id,)
        ).fetchall()

        updated = 0
        for row in db_drivers:
            rr_id = row["rr_id"]
            db_name = row["full_name"]

            # Try exact match, then normalized match
            arp = arp_map.get(db_name)
            if arp is None:
                # Try normalized matching
                from sys import path as _sp
                _sp.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                from src.utils import normalize_driver_name
                norm_db = normalize_driver_name(db_name)
                for api_name, api_arp in arp_map.items():
                    if normalize_driver_name(api_name) == norm_db:
                        arp = api_arp
                        break

            if arp is not None:
                if not dry_run:
                    conn.execute(
                        "UPDATE race_results SET avg_running_position = ? WHERE id = ?",
                        (arp, rr_id)
                    )
                updated += 1

        if not dry_run:
            conn.commit()

        updated_total += updated
        print(f"  Updated {updated}/{len(db_drivers)} drivers (ARP range: "
              f"{min(arp_map.values()):.1f} - {max(arp_map.values()):.1f})")

        # Rate limit — be polite to NASCAR API
        time.sleep(0.5)

    conn.close()
    action = "Would update" if dry_run else "Updated"
    print(f"\n{'='*50}")
    print(f"{action} {updated_total} race results with ARP data")
    print(f"Failed/skipped: {failed} races")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill ARP data")
    parser.add_argument("--season", type=int, help="Specific season to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()
    backfill(season=args.season, dry_run=args.dry_run)
