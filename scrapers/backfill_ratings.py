"""Backfill NASCAR official Driver Rating into race_results.

NASCAR publishes a per-race Driver Rating (0-150 scale) in its loop-stats feed:
    https://cf.nascar.com/loopstats/prod/{season}/{series_id}/{race_id}.json
That feed is the same one the app already uses as an ingestion fallback. It is
free, updates after every race, and (verified) covers 2022-2026 for all three
series.

The loop-stats records key on NASCAR's `driver_id` (== NASCARDriverID). Our
`drivers` table has no NASCAR id column, so — exactly like the loop-stats
ingestion path in src.data._fetch_and_store_via_loopstats — we resolve
driver_id -> name via the lap-times feed, then name -> drivers.id by
(normalized) full-name match, and UPDATE race_results.rating.

UPDATE-ONLY: never touches finish/laps_led/etc., only the rating column, so it
is safe to run repeatedly. New races are picked up automatically because
refresh_data.py calls store_all_ratings() after fetching results.

Usage:
    python scrapers/backfill_ratings.py            # only races missing rating
    python scrapers/backfill_ratings.py --force    # re-fetch every race
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import NASCAR_API_BASE
from src.data import _clean_api_name
from src.utils import normalize_driver_name

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")
# loop-stats lives at cf.nascar.com/loopstats/prod (NOT under /cacher, which is
# what NASCAR_API_BASE points to). lap-times IS under /cacher.
LOOPSTATS_BASE = "https://cf.nascar.com/loopstats/prod"


def fetch_json(url, retries=3):
    """Fetch JSON with retries + exponential backoff. Returns None on failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == retries - 1:
                print(f"  fetch failed: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def _loopstats_drivers(feed):
    """Pull the per-driver list out of the loop-stats feed (list- or dict-shaped)."""
    if isinstance(feed, list) and feed and isinstance(feed[0], dict):
        return feed[0].get("drivers") or []
    if isinstance(feed, dict):
        return feed.get("drivers") or []
    return []


def _build_id_to_name(season, series_id, api_race_id):
    """Map NASCARDriverID -> clean full name via the lap-times feed (same source
    the loop-stats ingestion uses)."""
    url = f"{NASCAR_API_BASE}/{season}/{series_id}/{api_race_id}/lap-times.json"
    feed = fetch_json(url)
    id_to_name = {}
    if isinstance(feed, dict):
        for d in feed.get("laps", []):
            nid = d.get("NASCARDriverID")
            nm = d.get("Full_Name") or d.get("FullName")
            if nid is not None and nm:
                id_to_name[nid] = _clean_api_name(nm)
    return id_to_name


def store_ratings_for_race(conn, series_id, api_race_id, season, db_race_id,
                           name_to_dbid=None):
    """Fetch one race's loop-stats ratings and UPDATE race_results.rating.

    Resolves loop-stats driver_id -> name (lap-times) -> drivers.id (exact then
    normalized). Returns (rows_updated, n_with_rating_in_feed). Reusable from the
    primary ingestion path so a freshly-stored race gets its rating in the same
    refresh run.
    """
    loop_url = f"{LOOPSTATS_BASE}/{season}/{series_id}/{api_race_id}.json"
    drivers = _loopstats_drivers(fetch_json(loop_url))
    if not drivers:
        return (0, 0)

    id_to_name = _build_id_to_name(season, series_id, api_race_id)
    if not id_to_name:
        return (0, 0)

    # Cache of driver-name -> drivers.id, normalized. Built lazily so a single
    # backfill run doesn't re-query the same name repeatedly.
    if name_to_dbid is None:
        name_to_dbid = {}
        for row in conn.execute("SELECT id, full_name FROM drivers"):
            name_to_dbid[normalize_driver_name(row[1])] = row[0]

    updated = 0
    n_rated = 0
    for d in drivers:
        nascar_id = d.get("driver_id")
        rating = d.get("rating")
        if nascar_id is None or rating is None:
            continue
        name = id_to_name.get(nascar_id)
        if not name:
            continue
        db_id = name_to_dbid.get(normalize_driver_name(name))
        if db_id is None:
            continue
        n_rated += 1
        cur = conn.execute(
            "UPDATE race_results SET rating = ? WHERE race_id = ? AND driver_id = ?",
            (rating, db_race_id, db_id),
        )
        updated += cur.rowcount
    return (updated, n_rated)


def store_all_ratings(conn, only_missing=True, verbose=False):
    """Backfill ratings for every race in the DB (or only those missing it).

    Shared entry point for the CLI and refresh_data.py. Commits per race.
    Returns (races_updated, rows_updated).
    """
    races = conn.execute('''
        SELECT id, api_race_id, series_id, season, race_name
        FROM races ORDER BY season, id
    ''').fetchall()

    # Build the name->id map once for the whole run.
    name_to_dbid = {}
    for row in conn.execute("SELECT id, full_name FROM drivers"):
        name_to_dbid[normalize_driver_name(row[1])] = row[0]

    races_updated = 0
    rows_updated = 0
    for race_id, api_race_id, series_id, season, race_name in races:
        if not api_race_id:
            continue
        if only_missing:
            missing = conn.execute(
                "SELECT COUNT(*) FROM race_results WHERE race_id = ? AND rating IS NULL",
                (race_id,),
            ).fetchone()[0]
            if missing == 0:
                continue

        upd, n_rated = store_ratings_for_race(
            conn, series_id, api_race_id, season, race_id,
            name_to_dbid=name_to_dbid,
        )
        if upd > 0:
            conn.commit()
            races_updated += 1
            rows_updated += upd
            if verbose:
                print(f"  {season} {(race_name or '')[:45]:45} "
                      f"+{upd} ratings ({n_rated} matched)")
        time.sleep(0.3)  # be polite to the feed
    return (races_updated, rows_updated)


def main():
    ap = argparse.ArgumentParser(description="Backfill NASCAR Driver Rating")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch every race, not just those missing rating")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        races_updated, rows_updated = store_all_ratings(
            conn, only_missing=not args.force, verbose=True)
    finally:
        conn.close()

    print(f"\n[DONE] ratings backfilled: {rows_updated} driver-rows "
          f"across {races_updated} races")


if __name__ == "__main__":
    main()
