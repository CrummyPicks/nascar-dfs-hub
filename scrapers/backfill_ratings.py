"""Backfill loop-stats driver metrics into race_results.

NASCAR publishes per-race driver metrics in its loop-stats feed:
    https://cf.nascar.com/loopstats/prod/{season}/{series_id}/{race_id}.json
plus per-lap speed + green/yellow flags in the lap-times feed:
    {NASCAR_API_BASE}/{season}/{series_id}/{race_id}/lap-times.json
Both are free, update after every race, and (verified) cover 2022-2026 for all
three series.

This module populates SIX columns, all UPDATE-ONLY (never touches finish /
laps_led / etc., safe to run repeatedly):
  • rating            — NASCAR Driver Rating (0-150), loop-stats
  • quality_passes    — passes of cars running in the top 15, loop-stats
  • passing_diff      — green-flag passes made minus passed, loop-stats
  • closing_pos       — avg running position over the closing laps, loop-stats
  • top15_laps        — laps run inside the top 15, loop-stats
  • green_lap_speed   — MEDIAN green-flag lap speed (mph), computed from
                        lap-times (per-lap LapSpeed filtered to FlagState==1)
  • green_speed_rank  — 1=fastest green-flag pace IN THAT RACE. Raw mph isn't
                        comparable across tracks (Atlanta ~182 vs Martinsville
                        ~100), so the rank is the track-normalized, projectable
                        form that drops into the position-units signal framework.

The loop-stats records key on NASCAR's `driver_id` (== NASCARDriverID). Our
`drivers` table has no NASCAR id column, so — like the loop-stats ingestion path
in src.data._fetch_and_store_via_loopstats — we resolve driver_id -> name via
the lap-times feed, then name -> drivers.id by (normalized) full-name match.

New races self-populate: refresh_data.py calls store_all_ratings() after
fetching results, and fetch_and_store_race() calls store_ratings_for_race()
inline at ingest. (Names kept for back-compat; they now fill ALL six columns.)

Usage:
    python scrapers/backfill_ratings.py            # only races missing metrics
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


def _green_flag_speeds(season, series_id, api_race_id):
    """Median green-flag lap speed (mph) per NASCARDriverID from the lap-times feed.

    The feed carries per-lap LapSpeed plus a `flags` change-point list
    (FlagState 1=green, 2=yellow, 8=warmup/checkered). We carry the flag state
    forward lap-by-lap, keep only green laps with a real speed, and take the
    MEDIAN (robust to a single fast/slow outlier lap). Returns {driver_id: mph}.
    Drivers with < MIN_GREEN_LAPS green laps are omitted (too small to trust).
    """
    MIN_GREEN_LAPS = 10
    url = f"{NASCAR_API_BASE}/{season}/{series_id}/{api_race_id}/lap-times.json"
    feed = fetch_json(url)
    if not isinstance(feed, dict):
        return {}
    # Build lap -> FlagState by carrying change-points forward.
    flags = feed.get("flags") or []
    change = {f["LapsCompleted"]: f["FlagState"] for f in flags
             if "LapsCompleted" in f and "FlagState" in f}
    out = {}
    for blk in feed.get("laps", []):
        nid = blk.get("NASCARDriverID")
        if nid is None:
            continue
        speeds = []
        cur_state = 0
        for lp in blk.get("Laps", []):
            lap_no = lp.get("Lap")
            if lap_no in change:
                cur_state = change[lap_no]
            spd = lp.get("LapSpeed")
            if cur_state == 1 and spd:
                try:
                    speeds.append(float(spd))
                except (TypeError, ValueError):
                    pass
        if len(speeds) >= MIN_GREEN_LAPS:
            speeds.sort()
            n = len(speeds)
            med = speeds[n // 2] if n % 2 else (speeds[n // 2 - 1] + speeds[n // 2]) / 2
            out[nid] = round(med, 3)
    return out


def store_ratings_for_race(conn, series_id, api_race_id, season, db_race_id,
                           name_to_dbid=None):
    """Fetch one race's loop-stats metrics + green-flag speed and UPDATE the six
    derived columns on race_results.

    Resolves loop-stats driver_id -> name (lap-times) -> drivers.id (exact then
    normalized). Returns (rows_updated, n_matched). Reusable from the primary
    ingestion path so a freshly-stored race gets its metrics in the same run.
    (Name retained for back-compat — it now fills all six columns, not just
    rating.)
    """
    loop_url = f"{LOOPSTATS_BASE}/{season}/{series_id}/{api_race_id}.json"
    drivers = _loopstats_drivers(fetch_json(loop_url))
    if not drivers:
        return (0, 0)

    id_to_name = _build_id_to_name(season, series_id, api_race_id)
    if not id_to_name:
        return (0, 0)

    green = _green_flag_speeds(season, series_id, api_race_id)
    # Per-race green-speed RANK (1 = fastest). Track-normalized so it's
    # comparable across venues; this is the projectable form of green pace.
    rank_by_id = {}
    for r, (nid, _spd) in enumerate(
            sorted(green.items(), key=lambda kv: kv[1], reverse=True), start=1):
        rank_by_id[nid] = r

    # Cache of normalized driver-name -> drivers.id, built lazily.
    if name_to_dbid is None:
        name_to_dbid = {}
        for row in conn.execute("SELECT id, full_name FROM drivers"):
            name_to_dbid[normalize_driver_name(row[1])] = row[0]

    updated = 0
    n_matched = 0
    for d in drivers:
        nascar_id = d.get("driver_id")
        if nascar_id is None:
            continue
        name = id_to_name.get(nascar_id)
        if not name:
            continue
        db_id = name_to_dbid.get(normalize_driver_name(name))
        if db_id is None:
            continue
        # Skip rows with nothing to write (no rating AND no green data).
        rating = d.get("rating")
        gspeed = green.get(nascar_id)
        grank = rank_by_id.get(nascar_id)
        if rating is None and gspeed is None:
            continue
        n_matched += 1
        cur = conn.execute(
            """UPDATE race_results SET
                 rating = COALESCE(?, rating),
                 quality_passes = COALESCE(?, quality_passes),
                 passing_diff = COALESCE(?, passing_diff),
                 closing_pos = COALESCE(?, closing_pos),
                 top15_laps = COALESCE(?, top15_laps),
                 green_lap_speed = COALESCE(?, green_lap_speed),
                 green_speed_rank = COALESCE(?, green_speed_rank)
               WHERE race_id = ? AND driver_id = ?""",
            (rating, d.get("quality_passes"), d.get("passing_diff"),
             d.get("closing_ps"), d.get("top15_laps"), gspeed, grank,
             db_race_id, db_id),
        )
        updated += cur.rowcount
    return (updated, n_matched)


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
            # A race needs a pass if it's missing rating OR the green-speed
            # metrics (so a rating-only DB from before this change gets the new
            # columns filled). finish_pos>0 limits to real classified rows.
            missing = conn.execute(
                "SELECT COUNT(*) FROM race_results WHERE race_id = ? "
                "AND finish_pos > 0 AND (rating IS NULL OR green_speed_rank IS NULL)",
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
