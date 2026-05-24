"""Backfill Average Running Position (ARP) for historical races.

For each race that still has NULL avg_running_position rows, re-fetch its lap
times, recompute ARP with the canonical name cleaning, and fill ONLY the NULL
rows using the same clean-name + fuzzy matching the rest of the app uses. This
handles cross-feed name differences — e.g. results "John H Nemechek" vs the
lap-times feed's "John Hunter Nemechek(i) (P)" (full name + ineligible +
Playoff tags). Races whose lap data is genuinely unavailable stay NULL.

Matches by internal DB race id (not api_race_id) to avoid any ambiguity, and
reuses src.data / src.utils so the matching stays identical to the live app.

Usage:
    python scrapers/backfill_arp.py                  # all races
    python scrapers/backfill_arp.py --season 2025    # specific season
    python scrapers/backfill_arp.py --dry-run        # preview without writing
"""
import sys
import os
import time
import argparse
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DB_PATH
from src.data import fetch_lap_times, compute_avg_running_position
from src.utils import build_norm_lookup, fuzzy_get


def _remaining_nulls(conn):
    """(races_with_null_arp, total_null_arp_rows)."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT r.id), COUNT(*) "
        "FROM race_results rr JOIN races r ON r.id = rr.race_id "
        "WHERE rr.avg_running_position IS NULL"
    ).fetchone()
    return row[0] or 0, row[1] or 0


def backfill(season=None, dry_run=False):
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    q = ("SELECT r.id, r.api_race_id, r.series_id, r.season, r.race_name, "
         "SUM(CASE WHEN rr.avg_running_position IS NULL THEN 1 ELSE 0 END) AS nulls "
         "FROM race_results rr JOIN races r ON r.id = rr.race_id "
         "WHERE r.api_race_id IS NOT NULL ")
    params = []
    if season:
        q += "AND r.season = ? "
        params.append(season)
    q += "GROUP BY r.id HAVING nulls > 0 ORDER BY r.season, r.series_id, r.race_date"
    races = conn.execute(q, params).fetchall()
    print(f"{len(races)} races have NULL ARP rows (before).")

    filled_rows = filled_races = no_lap = unresolved = 0
    no_lap_races = []
    for i, (rid, apid, sid, yr, name, nulls) in enumerate(races):
        try:
            laps = fetch_lap_times(sid, apid, yr)
            arp = compute_avg_running_position(laps) if laps else {}
        except Exception as e:
            print(f"  [{i+1}/{len(races)}] {yr} S{sid} {name[:30]:<30} fetch error: {e}")
            time.sleep(0.2)
            continue

        if not arp:
            no_lap += 1
            no_lap_races.append((yr, sid, name, nulls))
            print(f"  [{i+1}/{len(races)}] {yr} S{sid} {name[:30]:<30} NO LAP DATA ({nulls} NULL)")
            time.sleep(0.2)
            continue

        norm = build_norm_lookup(arp)
        null_rows = conn.execute(
            "SELECT rr.id, d.full_name FROM race_results rr "
            "JOIN drivers d ON d.id = rr.driver_id "
            "WHERE rr.race_id = ? AND rr.avg_running_position IS NULL", (rid,)
        ).fetchall()

        n = 0
        for rr_id, dbname in null_rows:
            val = fuzzy_get(dbname, arp, norm)
            if val is not None:
                if not dry_run:
                    conn.execute(
                        "UPDATE race_results SET avg_running_position = ? WHERE id = ?",
                        (val, rr_id))
                n += 1
            else:
                unresolved += 1
        if not dry_run:
            conn.commit()
        filled_rows += n
        filled_races += 1 if n else 0
        extra = f"  ({len(null_rows) - n} UNRESOLVED)" if n < len(null_rows) else ""
        print(f"  [{i+1}/{len(races)}] {yr} S{sid} {name[:30]:<30} "
              f"filled {n}/{len(null_rows)}{extra}{'  (dry)' if dry_run else ''}")
        time.sleep(0.2)

    races_left, rows_left = _remaining_nulls(conn)
    conn.close()

    print(f"\n{'DRY RUN — ' if dry_run else ''}DONE")
    print(f"  filled {filled_rows} rows across {filled_races} races")
    print(f"  {no_lap} races had NO lap data; {unresolved} rows had lap data but no name match")
    print(f"  REMAINING NULL: {rows_left} rows across {races_left} races")
    for yr, sid, name, nulls in no_lap_races:
        print(f"      no-lap-data: {yr} S{sid} {name[:34]:<34} {nulls} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill ARP data")
    ap.add_argument("--season", type=int, help="Specific season to backfill")
    ap.add_argument("--dry-run", action="store_true", help="Preview without writing")
    a = ap.parse_args()
    backfill(season=a.season, dry_run=a.dry_run)
