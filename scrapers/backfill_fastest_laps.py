"""
Backfill Fastest Laps for historical races.

Fetches lap-times data from the NASCAR API for each race in the database,
computes fastest laps per driver, and updates race_results.fastest_laps
and dfs_points (DK/FD) with corrected values.

Usage:
    python scrapers/backfill_fastest_laps.py                  # all races missing FL
    python scrapers/backfill_fastest_laps.py --season 2025    # specific season
    python scrapers/backfill_fastest_laps.py --force           # re-compute ALL races
    python scrapers/backfill_fastest_laps.py --dry-run         # preview without writing
"""

import sqlite3
import requests
import time
import argparse
import os
import re
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "nascar.db")
NASCAR_API_BASE = "https://cf.nascar.com/cacher"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

# DK scoring
DK_FINISH_POINTS = {
    1: 45, 2: 42, 3: 41, 4: 40, 5: 39, 6: 38, 7: 37, 8: 36, 9: 35, 10: 34,
    11: 32, 12: 31, 13: 30, 14: 29, 15: 28, 16: 27, 17: 26, 18: 25, 19: 24, 20: 23,
    21: 22, 22: 21, 23: 20, 24: 19, 25: 18, 26: 17, 27: 16, 28: 15, 29: 14, 30: 13,
    31: 12, 32: 11, 33: 10, 34: 9, 35: 8, 36: 7, 37: 6, 38: 5, 39: 4, 40: 3,
}


def _clean_api_name(name: str) -> str:
    """Clean API name (same as src/data.py)."""
    name = re.sub(r'^\*\s*', '', name)
    name = re.sub(r'\s*#$', '', name)
    name = re.sub(r'\s*\([a-zA-Z]\)$', '', name)
    name = re.sub(r'\bJr\.\s*$', 'Jr', name)
    name = re.sub(r'\bSr\.\s*$', 'Sr', name)
    return name.strip()


def compute_fastest_laps(lap_data: dict) -> dict:
    """Count fastest laps per driver from lap-times data.

    Returns {clean_driver_name: fastest_lap_count}.
    """
    drivers = lap_data.get("laps", [])
    if not drivers:
        return {}

    driver_laps = {}
    all_laps = set()
    for d in drivers:
        name = _clean_api_name(d.get("FullName", "").strip())
        if not name:
            continue
        driver_laps[name] = {}
        for lap in d.get("Laps", []):
            if lap.get("Lap", 0) > 0 and lap.get("LapTime") and lap["LapTime"] > 0:
                driver_laps[name][lap["Lap"]] = lap["LapTime"]
                all_laps.add(lap["Lap"])

    counts = defaultdict(int)
    for lap_num in sorted(all_laps):
        best_t, best_d = float("inf"), None
        for name, laps in driver_laps.items():
            t = laps.get(lap_num)
            if t is not None and t < best_t:
                best_t, best_d = t, name
        if best_d:
            counts[best_d] += 1
    return dict(counts)


def calc_dk_points(finish, start, laps_led, fastest_laps):
    """Calculate DraftKings points."""
    finish_pts = DK_FINISH_POINTS.get(finish, max(0, 44 - finish)) if finish else 0
    diff = ((start or finish or 40) - (finish or 40)) * 1.0
    led = (laps_led or 0) * 0.25
    fl = (fastest_laps or 0) * 0.45
    return finish_pts + diff + led + fl


def calc_fd_points(finish, start, laps_led, fastest_laps):
    """Calculate FanDuel points."""
    FD = {
        1: 52, 2: 47, 3: 44, 4: 42, 5: 41, 6: 40, 7: 39, 8: 38, 9: 37, 10: 36,
        11: 34, 12: 33, 13: 32, 14: 31, 15: 30, 16: 29, 17: 28, 18: 27, 19: 26, 20: 25,
        21: 22, 22: 21, 23: 20, 24: 19, 25: 18, 26: 17, 27: 16, 28: 15, 29: 14, 30: 13,
        31: 12, 32: 11, 33: 10, 34: 9, 35: 8, 36: 7, 37: 6, 38: 5, 39: 4, 40: 3,
    }
    finish_pts = FD.get(finish, max(0, 51 - finish)) if finish else 0
    diff = ((start or finish or 40) - (finish or 40)) * 0.5
    led = (laps_led or 0) * 0.1
    fl = (fastest_laps or 0) * 0.2
    return finish_pts + diff + led + fl


def _fuzzy_get(name, fl_map):
    """Try exact match first, then case-insensitive, then last-name match."""
    if name in fl_map:
        return fl_map[name]
    # Case-insensitive
    name_lower = name.lower()
    for k, v in fl_map.items():
        if k.lower() == name_lower:
            return v
    # Last name match (for cases like "Nicholas Sanchez" vs "Nick Sanchez")
    last = name.split()[-1].lower() if name.split() else ""
    first = name.split()[0].lower() if name.split() else ""
    candidates = []
    for k, v in fl_map.items():
        parts = k.split()
        k_last = parts[-1].lower() if parts else ""
        k_first = parts[0].lower() if parts else ""
        if k_last == last and (k_first[0] == first[0] if k_first and first else False):
            candidates.append((k, v))
    if len(candidates) == 1:
        return candidates[0][1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill fastest laps from lap-times API")
    parser.add_argument("--season", type=int, default=None, help="Specific season to backfill")
    parser.add_argument("--force", action="store_true", help="Re-compute ALL races (not just missing)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Find races that need backfilling — include races where SOME drivers are missing FL
    if args.force:
        where = "1=1"
    else:
        # Any race that has at least one driver with FL=0 who MIGHT have FL data
        where = "1=1"  # Check all races — the script will skip if no changes needed

    season_filter = f"AND r.season = {args.season}" if args.season else ""

    races = conn.execute(f"""
        SELECT DISTINCT r.id as db_race_id, r.api_race_id, r.race_name, r.race_date,
               r.series_id, r.season
        FROM races r
        JOIN race_results rr ON rr.race_id = r.id
        WHERE r.api_race_id IS NOT NULL {season_filter}
        AND {where}
        ORDER BY r.race_date
    """).fetchall()

    print(f"Found {len(races)} races to backfill fastest laps")
    if not races:
        conn.close()
        return

    updated_races = 0
    updated_results = 0
    updated_dfs = 0
    skipped = 0

    for i, race in enumerate(races):
        db_race_id = race["db_race_id"]
        api_race_id = race["api_race_id"]
        series_id = race["series_id"]
        season = race["season"]
        race_name = race["race_name"]
        race_date = race["race_date"]

        print(f"  [{i+1}/{len(races)}] {race_date} {race_name} (s{series_id})...", end=" ", flush=True)

        # Fetch lap times
        url = f"{NASCAR_API_BASE}/{season}/{series_id}/{api_race_id}/lap-times.json"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                skipped += 1
                time.sleep(0.3)
                continue
            lap_data = resp.json()
        except Exception as e:
            print(f"Error: {e}")
            skipped += 1
            time.sleep(0.3)
            continue

        fl_map = compute_fastest_laps(lap_data)
        if not fl_map:
            print("no lap data")
            skipped += 1
            time.sleep(0.3)
            continue

        total_fl = sum(fl_map.values())
        print(f"{total_fl} fastest laps across {len(fl_map)} drivers", end="")

        if args.dry_run:
            print(" (dry run)")
            time.sleep(0.3)
            continue

        # Get current race_results for this race
        results = conn.execute("""
            SELECT rr.id, rr.driver_id, rr.start_pos, rr.finish_pos,
                   rr.laps_led, rr.fastest_laps, d.full_name
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            WHERE rr.race_id = ?
        """, (db_race_id,)).fetchall()

        race_updated = 0
        for rr in results:
            driver_name = rr["full_name"]
            old_fl = rr["fastest_laps"] or 0
            new_fl = _fuzzy_get(driver_name, fl_map)
            if new_fl is None:
                new_fl = 0

            if new_fl != old_fl:
                # Update fastest_laps in race_results
                conn.execute(
                    "UPDATE race_results SET fastest_laps = ? WHERE id = ?",
                    (new_fl, rr["id"])
                )
                race_updated += 1
                updated_results += 1

            # Always recalculate DFS points to ensure consistency
            finish = rr["finish_pos"] or 0
            start = rr["start_pos"] or 0
            ll = rr["laps_led"] or 0
            dk_pts = calc_dk_points(finish, start, ll, new_fl)
            fd_pts = calc_fd_points(finish, start, ll, new_fl)

            # DK score breakdown
            dk_place = DK_FINISH_POINTS.get(finish, max(0, 44 - finish)) if finish else 0
            dk_diff = ((start or finish or 40) - (finish or 40)) * 1.0
            dk_led = ll * 0.25
            dk_fl = new_fl * 0.45

            # FD score breakdown
            FD_PTS = {
                1: 52, 2: 47, 3: 44, 4: 42, 5: 41, 6: 40, 7: 39, 8: 38, 9: 37, 10: 36,
                11: 34, 12: 33, 13: 32, 14: 31, 15: 30, 16: 29, 17: 28, 18: 27, 19: 26, 20: 25,
                21: 22, 22: 21, 23: 20, 24: 19, 25: 18, 26: 17, 27: 16, 28: 15, 29: 14, 30: 13,
                31: 12, 32: 11, 33: 10, 34: 9, 35: 8, 36: 7, 37: 6, 38: 5, 39: 4, 40: 3,
            }
            fd_place = FD_PTS.get(finish, max(0, 51 - finish)) if finish else 0
            fd_diff = ((start or finish or 40) - (finish or 40)) * 0.5
            fd_led = ll * 0.1
            fd_fl = new_fl * 0.2

            # Upsert dfs_points with full breakdown
            conn.execute("""
                INSERT INTO dfs_points (race_id, driver_id, platform, dfs_score,
                    place_pts, place_diff_pts, laps_led_pts, fastest_laps_pts)
                VALUES (?, ?, 'DraftKings', ?, ?, ?, ?, ?)
                ON CONFLICT(race_id, driver_id, platform) DO UPDATE SET
                    dfs_score=excluded.dfs_score, place_pts=excluded.place_pts,
                    place_diff_pts=excluded.place_diff_pts, laps_led_pts=excluded.laps_led_pts,
                    fastest_laps_pts=excluded.fastest_laps_pts
            """, (db_race_id, rr["driver_id"], dk_pts, dk_place, dk_diff, dk_led, dk_fl))
            conn.execute("""
                INSERT INTO dfs_points (race_id, driver_id, platform, dfs_score,
                    place_pts, place_diff_pts, laps_led_pts, fastest_laps_pts)
                VALUES (?, ?, 'FanDuel', ?, ?, ?, ?, ?)
                ON CONFLICT(race_id, driver_id, platform) DO UPDATE SET
                    dfs_score=excluded.dfs_score, place_pts=excluded.place_pts,
                    place_diff_pts=excluded.place_diff_pts, laps_led_pts=excluded.laps_led_pts,
                    fastest_laps_pts=excluded.fastest_laps_pts
            """, (db_race_id, rr["driver_id"], fd_pts, fd_place, fd_diff, fd_led, fd_fl))
            updated_dfs += 1

        if race_updated > 0:
            updated_races += 1

        conn.commit()
        print(f" -> updated {race_updated} drivers")
        time.sleep(0.3)

    conn.close()

    print(f"\n{'='*60}")
    print(f"  Backfill complete!")
    print(f"  Races processed: {len(races)}")
    print(f"  Races with FL updates: {updated_races}")
    print(f"  Individual FL updates: {updated_results}")
    print(f"  DFS points recalculated: {updated_dfs}")
    print(f"  Skipped (no data): {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
