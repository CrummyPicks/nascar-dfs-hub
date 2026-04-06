"""
NASCAR DFS Hub — Local Data Refresh Script

Run this script locally to populate the database with race results,
fastest laps, and DFS points. This replaces the in-app admin buttons.

Usage:
    python refresh_data.py                  # Fetch all new races for Cup 2026
    python refresh_data.py --series 2       # Fetch Xfinity series
    python refresh_data.py --year 2025      # Fetch 2025 season
    python refresh_data.py --all            # Fetch all series, all years
    python refresh_data.py --race 5596      # Fetch a specific race ID
"""

import argparse
import sys
from datetime import datetime

from src.config import SERIES_OPTIONS
from src.data import (
    fetch_race_list, filter_point_races, fetch_and_store_race,
    fetch_nascar_odds, save_odds_to_db,
    fetch_dk_salaries_live, sync_dk_salaries_to_db,
)


SERIES_MAP = {
    "cup": 1, "1": 1,
    "xfinity": 2, "oreilly": 2, "2": 2,
    "truck": 3, "craftsman": 3, "3": 3,
}

SERIES_NAMES = {1: "Cup", 2: "Xfinity", 3: "Truck"}


def fetch_season(series_id: int, year: int):
    """Fetch all completed races for a given series and year."""
    series_name = SERIES_NAMES.get(series_id, f"Series {series_id}")
    print(f"\n{'='*60}")
    print(f"  {series_name} {year}")
    print(f"{'='*60}")

    races = fetch_race_list(series_id, year)
    if not races:
        print(f"  Could not fetch race list for {series_name} {year}")
        return 0

    point_races = filter_point_races(races)
    now = datetime.now()

    completed = []
    for race in point_races:
        rd = race.get("race_date", "")
        try:
            d = datetime.fromisoformat(rd.replace("Z", "+00:00").split("+")[0].split("T")[0])
            if d.date() <= now.date():
                completed.append(race)
        except Exception:
            pass

    if not completed:
        print(f"  No completed races found for {series_name} {year}")
        return 0

    print(f"  Found {len(completed)} completed races\n")

    stored = 0
    for i, race in enumerate(completed):
        race_id = race.get("race_id")
        race_name = race.get("race_name", "Unknown")
        track = race.get("track_name", "")
        date = race.get("race_date", "")[:10]

        print(f"  [{i+1}/{len(completed)}] {date} @ {track}: {race_name}...", end=" ", flush=True)

        result = fetch_and_store_race(series_id, race_id, year)
        if result.get("status") == "success":
            print(f"OK ({result['drivers']} drivers)")
            stored += 1
        else:
            print(f"SKIP ({result.get('message', result.get('error', 'unknown'))})")

    print(f"\n  Stored {stored}/{len(completed)} races for {series_name} {year}")
    return stored


def fetch_single_race(race_id: int, series_id: int = 1, year: int = 2026):
    """Fetch a single race by ID."""
    print(f"\nFetching race {race_id}...")
    result = fetch_and_store_race(series_id, race_id, year)
    if result.get("status") == "success":
        print(f"OK: Stored {result['drivers']} drivers for {result['race_name']}")
    else:
        print(f"FAILED: {result.get('message', result.get('error', 'unknown'))}")


def fetch_and_save_odds(series_id: int, year: int):
    """Fetch current odds from Action Network and save for the next upcoming race."""
    series_name = SERIES_NAMES.get(series_id, f"Series {series_id}")
    print(f"\n  Fetching odds for {series_name}...", end=" ", flush=True)

    # We need streamlit cache disabled for CLI — call the raw function
    try:
        odds = fetch_nascar_odds.__wrapped__()
    except AttributeError:
        odds = fetch_nascar_odds()

    if not odds:
        print("No odds available (no upcoming race or Action Network down)")
        return

    print(f"Got odds for {len(odds)} drivers")

    # Find the next upcoming race to associate odds with
    races = fetch_race_list(series_id, year)
    if not races:
        print("  Could not fetch race list to find upcoming race")
        return

    point_races = filter_point_races(races)
    now = datetime.now()
    upcoming = None
    for race in point_races:
        rd = race.get("race_date", "")
        try:
            d = datetime.fromisoformat(rd.replace("Z", "+00:00").split("+")[0].split("T")[0])
            if d.date() >= now.date():
                upcoming = race
                break
        except Exception:
            pass

    if not upcoming:
        print("  No upcoming race found to associate odds with")
        return

    race_id = upcoming.get("race_id")
    race_name = upcoming.get("race_name", "Unknown")
    track = upcoming.get("track_name", "")
    count = save_odds_to_db(odds, race_id)
    print(f"  Saved {count} odds for {race_name} @ {track} (race_id={race_id})")

    # Also try to fetch and save DK salaries
    print(f"  Fetching DK salaries...", end=" ", flush=True)
    try:
        dk_df = fetch_dk_salaries_live.__wrapped__()
    except AttributeError:
        dk_df = fetch_dk_salaries_live()

    if not dk_df.empty:
        sal_count = sync_dk_salaries_to_db(dk_df, race_id, series_id, race_name)
        print(f"Saved {sal_count} salaries")
    else:
        print("No DK salary data available")


def backfill_api_race_ids():
    """Backfill api_race_id for races that don't have one yet."""
    import sqlite3
    from difflib import get_close_matches
    from src.config import DB_PATH

    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(str(DB_PATH))

    # Ensure column exists
    try:
        conn.execute("SELECT api_race_id FROM races LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE races ADD COLUMN api_race_id INTEGER")
        conn.commit()

    null_count = conn.execute(
        "SELECT COUNT(*) FROM races WHERE api_race_id IS NULL"
    ).fetchone()[0]
    if null_count == 0:
        conn.close()
        return

    print(f"\n  Backfilling api_race_id for {null_count} races...")

    combos = conn.execute(
        "SELECT DISTINCT series_id, season FROM races WHERE api_race_id IS NULL"
    ).fetchall()

    used_ids = set(
        r[0] for r in conn.execute(
            "SELECT api_race_id FROM races WHERE api_race_id IS NOT NULL"
        ).fetchall()
    )

    updated = 0
    for sid, season in combos:
        races_list = fetch_race_list(sid, season)
        if not races_list:
            continue

        # Build name -> api_race_id and date -> api_race_id maps
        api_by_name = {}
        api_by_date = {}
        for r in races_list:
            name = (r.get("race_name", "") or "").strip()
            rid = r.get("race_id")
            date = (r.get("race_date", "") or "")[:10]
            if name and rid:
                api_by_name[name] = rid
            if date and rid:
                api_by_date.setdefault(date, []).append(rid)

        db_races = conn.execute(
            "SELECT id, race_name, race_date FROM races "
            "WHERE series_id=? AND season=? AND api_race_id IS NULL",
            (sid, season),
        ).fetchall()

        api_names = list(api_by_name.keys())

        for db_id, db_name, db_date in db_races:
            api_rid = None
            # Try exact name match
            if db_name and db_name in api_by_name:
                api_rid = api_by_name[db_name]
            # Try fuzzy name match
            if not api_rid and db_name:
                matches = get_close_matches(db_name, api_names, n=1, cutoff=0.6)
                if matches:
                    api_rid = api_by_name[matches[0]]
            # Try date match
            if not api_rid and db_date:
                date_key = db_date[:10]
                candidates = api_by_date.get(date_key, [])
                candidates = [c for c in candidates if c not in used_ids]
                if len(candidates) == 1:
                    api_rid = candidates[0]

            if api_rid and api_rid not in used_ids:
                conn.execute(
                    "UPDATE races SET api_race_id=? WHERE id=?", (api_rid, db_id)
                )
                used_ids.add(api_rid)
                updated += 1

    conn.commit()
    conn.close()
    print(f"  Backfilled {updated} races with api_race_id")


def main():
    parser = argparse.ArgumentParser(description="NASCAR DFS Hub — Data Refresh")
    parser.add_argument("--series", type=str, default="cup",
                        help="Series: cup, xfinity, truck (or 1, 2, 3)")
    parser.add_argument("--year", type=int, default=2026,
                        help="Season year (default: 2026)")
    parser.add_argument("--all", action="store_true",
                        help="Fetch all series and years (2022-2026)")
    parser.add_argument("--race", type=int, default=None,
                        help="Fetch a specific race ID")
    parser.add_argument("--odds", action="store_true",
                        help="Also fetch and save current odds for the next upcoming race")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  NASCAR DFS Hub — Data Refresh")
    print("="*60)

    if args.race:
        series_id = SERIES_MAP.get(args.series.lower(), 1)
        fetch_single_race(args.race, series_id, args.year)
    elif args.all:
        total = 0
        for sid in [1, 2, 3]:
            for year in [2022, 2023, 2024, 2025, 2026]:
                total += fetch_season(sid, year)
        print(f"\n{'='*60}")
        print(f"  TOTAL: {total} races stored across all series/years")
        print(f"{'='*60}")
    else:
        series_id = SERIES_MAP.get(args.series.lower(), 1)
        fetch_season(series_id, args.year)

    # Backfill api_race_id for any races missing it
    backfill_api_race_ids()

    # Fetch odds if requested or during normal refresh
    if args.odds or not args.race:
        series_id = SERIES_MAP.get(args.series.lower(), 1)
        fetch_and_save_odds(series_id, args.year)

    print("\nDone! Start the app with: streamlit run nascar_dfs_app.py\n")


if __name__ == "__main__":
    main()
