"""NASCAR DFS Hub — Data Layer (API fetches, DB queries, scraping)."""

import os
import sqlite3
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from src.config import (
    NASCAR_API_BASE, DB_PATH, DA_TRACK_IDS, EXHIBITION_KEYWORDS,
)
from src.utils import int_col, calc_dk_points, calc_fd_points, normalize_driver_name
import re


def _clean_api_name(name: str) -> str:
    """Canonicalize a driver name for display AND storage.

    Applied at every API ingress point so the app surfaces one consistent
    spelling throughout. Rules:
      1. Strip race-meta indicators (*, #, (i), (R))
      2. Fold Unicode to ASCII — "Daniel Suárez" -> "Daniel Suarez"
      3. Remove ALL periods — "A.J." -> "AJ", "John H." -> "John H"
      4. Collapse consecutive whitespace that period-stripping may introduce
      5. Normalize Jr./Sr. suffixes (already covered by rule 3, kept for clarity)

    Name matching via normalize_driver_name + DRIVER_ALIASES still works
    because that layer also strips accents and periods and applies alias
    mappings (e.g. "john h nemechek" -> "john hunter nemechek").
    """
    if not name:
        return ""
    import unicodedata as _ud
    name = name.strip()
    # Strip race-meta indicators
    name = re.sub(r'^\*\s*', '', name)            # leading asterisk (rookie)
    name = re.sub(r'\s*#$', '', name)              # trailing # (charter)
    name = re.sub(r'\s*\([a-zA-Z]\)$', '', name)   # trailing (i)/(R)
    # Unicode fold: Suárez -> Suarez, Leguizamón -> Leguizamon
    name = _ud.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Remove all periods (A.J. -> AJ, John H. -> John H, Jr. -> Jr)
    name = name.replace(".", "")
    # Collapse any accidental multiple spaces introduced by period stripping
    name = " ".join(name.split())
    return name


# ============================================================
# NASCAR API FETCHES
# ============================================================

def _default_active_year() -> int:
    """Default 'current season' for year-defaulted helpers in this module.

    Uses datetime.now() each call so the value rolls over automatically; from
    October onward we advance to the upcoming season because NASCAR posts
    next-year schedules in October. Callers should still pass `year` explicitly
    whenever the year is known — these defaults are defensive only.
    """
    from datetime import datetime
    _t = datetime.now()
    return _t.year + 1 if _t.month >= 10 else _t.year


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_race_list(series_id: int, year: int = None) -> list:
    """Fetch race list from NASCAR API."""
    if year is None:
        year = _default_active_year()
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/race_list_basic.json", timeout=15)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def sync_race_schedule_from_api(series_id: int, year: int, verbose: bool = False) -> dict:
    """Reconcile DB race rows against the live NASCAR API race_list_basic.

    Why this exists: the DB caches race metadata (date, name) at the time of
    first scrape. When NASCAR updates the schedule later — moving a race to
    a different weekend, renaming it for a new sponsor — our DB keeps the
    stale values. This function pulls the current API schedule and UPDATEs
    any DB row whose api_race_id matches, fixing dates and names in place.

    Also deletes clearly-stale placeholder rows (no api_race_id, zero
    dependent data — no race_results/salaries/odds).

    Returns a summary dict: {dates_updated, names_updated, placeholders_deleted}.
    """
    summary = {"dates_updated": 0, "names_updated": 0, "placeholders_deleted": 0}
    if not DB_PATH.exists():
        return summary
    try:
        api_data = fetch_race_list(series_id, year)
        if not api_data:
            return summary

        api_by_id = {
            r["race_id"]: {
                "date": (r.get("race_date") or "")[:10],
                "name": r.get("race_name", ""),
            }
            for r in api_data if r.get("race_id")
        }

        conn = sqlite3.connect(str(DB_PATH))
        # 1. Update stale dates/names where api_race_id matches
        rows = conn.execute(
            "SELECT id, api_race_id, race_date, race_name FROM races "
            "WHERE series_id = ? AND season = ? AND api_race_id IS NOT NULL",
            (series_id, year)
        ).fetchall()
        for db_id, api_id, db_date, db_name in rows:
            api = api_by_id.get(api_id)
            if not api:
                continue
            # Date sync
            if db_date and api["date"] and db_date[:10] != api["date"]:
                conn.execute("UPDATE races SET race_date = ? WHERE id = ?",
                             (api["date"], db_id))
                summary["dates_updated"] += 1
                if verbose:
                    print(f"  date  db={db_id}  {db_date[:10]} -> {api['date']}  ({db_name[:30]})")
            # Name sync — only when our name is stale (API has updated sponsor)
            if api["name"] and db_name != api["name"]:
                conn.execute("UPDATE races SET race_name = ? WHERE id = ?",
                             (api["name"], db_id))
                summary["names_updated"] += 1
                if verbose:
                    print(f"  name  db={db_id}  '{db_name[:30]}' -> '{api['name'][:30]}'")

        # 2. Delete placeholder rows (no api_race_id + no dependent data)
        placeholders = conn.execute('''
            SELECT id, race_name, race_date FROM races
            WHERE series_id = ? AND season = ? AND api_race_id IS NULL
              AND NOT EXISTS (SELECT 1 FROM race_results WHERE race_id = races.id)
              AND NOT EXISTS (SELECT 1 FROM salaries WHERE race_id = races.id)
              AND NOT EXISTS (SELECT 1 FROM odds WHERE race_id = races.id)
        ''', (series_id, year)).fetchall()
        for db_id, name, date in placeholders:
            conn.execute("DELETE FROM races WHERE id = ?", (db_id,))
            summary["placeholders_deleted"] += 1
            if verbose:
                print(f"  del   db={db_id}  '{name[:30]}' date={date[:10] if date else '?'}")

        conn.commit()
        conn.close()
    except Exception as e:
        if verbose:
            print(f"  sync_race_schedule_from_api error: {e}")
    return summary


def merge_duplicate_drivers(verbose: bool = False) -> dict:
    """Find and merge duplicate driver entries.

    Detection groups by a FULLY CANONICALIZED key that applies:
      - normalize_driver_name (lowercase, accent fold, strip periods, aliases)
      - nickname expansion   (Nick -> Nicholas, Bob -> Robert, etc.)
      - middle-initial strip (Jason M White -> Jason White)

    Catches all of:
      - A.J. Allmendinger / AJ Allmendinger           (periods)
      - Daniel Suárez / Daniel Suarez                 (accent)
      - Corey LaJoie / Corey Lajoie                   (case)
      - John H. Nemechek / John Hunter Nemechek       (alias)
      - Nick Sanchez / Nicholas Sanchez               (nickname)
      - Jason White / Jason M White                   (middle initial)

    Merge strategy per group:
      1. Canonical = row with most race_results (most proven real).
         Tiebreak: lower id.
      2. UPDATE foreign keys (race_results/salaries/odds/dfs_points) to
         canonical driver_id.
      3. DELETE the duplicate driver rows.

    Returns {"groups_merged": int, "drivers_deleted": int, "rows_rekeyed": int}.
    """
    from src.utils import normalize_driver_name, _nickname_canonical, _stripped_middle
    summary = {"groups_merged": 0, "drivers_deleted": 0, "rows_rekeyed": 0}
    if not DB_PATH.exists():
        return summary

    def _canonical_group_key(name):
        """Fully canonicalized key: normalize + nickname expand + strip middle."""
        n = normalize_driver_name(name)
        n = _nickname_canonical(n)
        n = _stripped_middle(n)
        return n

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Group all drivers by the fully canonical key
    all_drivers = conn.execute("SELECT id, full_name FROM drivers").fetchall()
    groups = {}
    for row in all_drivers:
        key = _canonical_group_key(row["full_name"])
        if not key:
            continue
        groups.setdefault(key, []).append(dict(row))

    def _score_row(r):
        """Higher score = better canonical choice.

        Primary: race_results count (real data wins).
        Secondary: name has accents (prefer "Suárez" over "Suarez" — official).
        Tertiary: lower id (older entry, more historically linked).
        """
        rr = conn.execute(
            "SELECT COUNT(*) FROM race_results WHERE driver_id = ?", (r["id"],)
        ).fetchone()[0]
        has_accent = any(ord(c) > 127 for c in r["full_name"])
        # Large primary, moderate tiebreakers
        return (rr, 1 if has_accent else 0, -r["id"])

    def _shares_race(a_id, b_id):
        """Return True if both driver_ids have race_results for the same race.
        A single driver cannot race two cars in one event — so if both rows
        appear in the same race they are DEFINITIVELY distinct people.
        """
        row = conn.execute(
            "SELECT 1 FROM race_results ra "
            "JOIN race_results rb ON rb.race_id = ra.race_id "
            "WHERE ra.driver_id = ? AND rb.driver_id = ? LIMIT 1",
            (a_id, b_id)
        ).fetchone()
        return row is not None

    for key, rows in groups.items():
        if len(rows) < 2:
            continue

        # SAFETY: if any pair in the group raced the same event, they're
        # distinct drivers (one car each per race) — skip the merge.
        pairs_distinct = False
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if _shares_race(rows[i]["id"], rows[j]["id"]):
                    pairs_distinct = True
                    if verbose:
                        print(f"  SKIP [{key}]: {rows[i]['full_name']} (id={rows[i]['id']}) "
                              f"and {rows[j]['full_name']} (id={rows[j]['id']}) "
                              f"share a race — they're distinct drivers, not duplicates.")
                    break
            if pairs_distinct:
                break
        if pairs_distinct:
            continue

        # Pick canonical
        canonical = max(rows, key=_score_row)
        others = [r for r in rows if r["id"] != canonical["id"]]
        if not others:
            continue

        if verbose:
            names = " | ".join(f"{r['full_name']} (id={r['id']})" for r in rows)
            print(f"  merging [{key}]: {names} -> canonical id={canonical['id']} '{canonical['full_name']}'")

        # Rekey all foreign keys to canonical
        for o in others:
            dup_id = o["id"]
            for table in ["race_results", "salaries", "odds", "dfs_points"]:
                try:
                    # UPDATE OR IGNORE to avoid duplicate PK conflicts when
                    # the canonical already has an entry for (race_id, driver_id).
                    # race_results + dfs_points + odds have UNIQUE on (race_id,
                    # driver_id[, platform/sportsbook]); in that case we keep
                    # the canonical's row and just drop the duplicate's.
                    cur = conn.execute(
                        f"UPDATE OR IGNORE {table} SET driver_id = ? WHERE driver_id = ?",
                        (canonical["id"], dup_id)
                    )
                    summary["rows_rekeyed"] += cur.rowcount
                    # Delete remaining rows (those that couldn't be rekeyed
                    # due to unique conflict — canonical already has that row)
                    conn.execute(
                        f"DELETE FROM {table} WHERE driver_id = ?", (dup_id,)
                    )
                except Exception:
                    pass

            conn.execute("DELETE FROM drivers WHERE id = ?", (dup_id,))
            summary["drivers_deleted"] += 1

        summary["groups_merged"] += 1

    conn.commit()
    conn.close()
    return summary


def sync_all_schedules(years: list = None, verbose: bool = False) -> dict:
    """Run sync_race_schedule_from_api for all 3 series across given years.

    Default: sync the current season and the prior season (fresh races only —
    historical seasons are immutable in the API). Returns aggregate summary.
    """
    from datetime import datetime
    if years is None:
        cur = datetime.now().year
        years = [cur - 1, cur]
    totals = {"dates_updated": 0, "names_updated": 0, "placeholders_deleted": 0}
    for year in years:
        for sid in [1, 2, 3]:
            res = sync_race_schedule_from_api(sid, year, verbose=verbose)
            for k in totals:
                totals[k] += res[k]
    return totals


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weekend_feed(series_id: int, race_id: int, year: int = None) -> Optional[dict]:
    """Fetch weekend feed (entry list, qualifying, practice, results)."""
    if year is None:
        year = _default_active_year()
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/weekend-feed.json", timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_lap_times(series_id: int, race_id: int, year: int = None) -> Optional[dict]:
    """Fetch lap-by-lap timing data."""
    if year is None:
        year = _default_active_year()
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/lap-times.json", timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _parse_lap_avg_session(session: dict) -> pd.DataFrame:
    """Parse a single lap-averages session into a DataFrame."""
    items = session.get("Items", [])
    if not items:
        return pd.DataFrame()
    rows = []
    for item in items:
        driver_name = _clean_api_name(item.get("FullName") or item.get("Driver") or "")
        row = {
            "Driver": driver_name,
            "Car": str(item.get("Number", "")).strip(),
            "Manufacturer": item.get("Manufacturer", ""),
            "Sponsor": item.get("Sponsor", ""),
            "Overall Avg": item.get("OverAllAvg"),
            "Overall Rank": item.get("OverAllAvgRank"),
            "Best Lap": item.get("BestLapTime"),
            "1 Lap Rank": item.get("BestLapRank"),
        }
        for n in [5, 10, 15, 20, 25, 30]:
            val = item.get(f"Con{n}Lap")
            row[f"{n} Lap"] = val if (val or 999) < 900 else None
            row[f"{n} Lap Rank"] = item.get(f"Con{n}LapRank")
        rows.append(row)

    df = pd.DataFrame(rows)
    for rc in [c for c in df.columns if "Rank" in c]:
        df[rc] = int_col(df[rc])
    if "Overall Rank" in df.columns:
        df = df.sort_values("Overall Rank", na_position="last").reset_index(drop=True)
    return df


def fetch_lap_averages(series_id: int, race_id: int, year: int = None) -> pd.DataFrame:
    """Fetch practice lap averages (Overall, 5/10/15/20/25/30 lap consecutive averages).

    Returns data from the last practice session (combined/overall view).
    """
    if year is None:
        year = _default_active_year()
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/lap-averages.json", timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        return _parse_lap_avg_session(data[-1])
    except Exception:
        return pd.DataFrame()


def fetch_all_practice_sessions(series_id: int, race_id: int, year: int = None) -> list:
    """Fetch ALL practice sessions from lap-averages endpoint.

    Returns list of (session_label, DataFrame) tuples.
    Session labels are like "Group 1", "Group 2", etc.
    """
    if year is None:
        year = _default_active_year()
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/lap-averages.json", timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        if not data or not isinstance(data, list):
            return []

        sessions = []
        for i, session in enumerate(data):
            df = _parse_lap_avg_session(session)
            if df.empty:
                continue
            # Try to get session name from the data; fall back to numbered label
            label = session.get("SessionName") or session.get("RunName") or ""
            if not label:
                if len(data) == 1:
                    label = "Practice"
                else:
                    label = f"Group {i + 1}"
            sessions.append((label, df))
        return sessions
    except Exception:
        return []


# ============================================================
# EXTRACT DATA FROM WEEKEND FEED
# ============================================================

def _build_car_driver_map(feed: dict) -> Dict[str, dict]:
    """Build car_number -> {driver, team, make, crew_chief} lookup."""
    car_map = {}
    races = feed.get("weekend_race", [])
    if races:
        for car in races[0].get("cars", []):
            cn = str(car.get("car_number", "")).strip()
            if cn and car.get("driver_fullname"):
                car_map[cn] = {
                    "driver": _clean_api_name(car["driver_fullname"]),
                    "team": car.get("team_name", ""),
                    "make": car.get("car_make", ""),
                    "crew_chief": car.get("crew_chief_fullname", ""),
                }
        for r in races[0].get("results", []):
            cn = str(r.get("car_number", "")).strip()
            if cn and cn not in car_map and r.get("driver_fullname"):
                car_map[cn] = {
                    "driver": _clean_api_name(r["driver_fullname"]),
                    "team": r.get("team_name", ""),
                    "make": r.get("car_make", ""),
                    "crew_chief": r.get("crew_chief_fullname", ""),
                }
    return car_map


def extract_entry_list(feed: dict) -> pd.DataFrame:
    """Extract entry list from weekend-feed with crew chief data."""
    if not feed:
        return pd.DataFrame()
    races = feed.get("weekend_race", [])
    if not races:
        return pd.DataFrame()

    cars = races[0].get("cars", [])
    if cars:
        rows = [{
            "Car": car.get("car_number"),
            "Driver": _clean_api_name(car.get("driver_fullname") or ""),
            "Team": car.get("team_name"),
            "Manufacturer": car.get("car_make"),
            "Crew Chief": car.get("crew_chief_fullname", ""),
        } for car in cars]
        return pd.DataFrame(rows)

    results = races[0].get("results", [])
    if results:
        rows = [{
            "Car": r.get("car_number"),
            "Driver": _clean_api_name(r.get("driver_fullname") or ""),
            "Team": r.get("team_name"),
            "Manufacturer": r.get("car_make"),
            "Crew Chief": r.get("crew_chief_fullname", ""),
            "Starting Position": r.get("starting_position"),
        } for r in results if r.get("driver_fullname")]
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame()


def extract_qualifying(feed: dict) -> pd.DataFrame:
    """Extract qualifying results from weekend-feed."""
    if not feed:
        return pd.DataFrame()
    car_map = _build_car_driver_map(feed)
    for run in feed.get("weekend_runs", []):
        rt = run.get("run_type", 0)
        rn = str(run.get("run_name", "")).lower()
        if rt == 2 or "qualif" in rn:
            results = run.get("results", [])
            if results:
                rows = []
                for r in results:
                    cn = str(r.get("car_number", "")).strip()
                    driver = r.get("driver_fullname")
                    team = r.get("team_name")
                    if not driver and cn in car_map:
                        driver = car_map[cn]["driver"]
                        team = team or car_map[cn]["team"]
                    speed = r.get("best_lap_speed") or r.get("best_speed") or r.get("qualifying_speed")
                    rows.append({
                        "Driver": driver,
                        "Qualifying Position": r.get("finishing_position"),
                        "Best Lap Time": r.get("best_lap_time"),
                        "Best Lap Speed": speed,
                        "Car": cn,
                        "Team": team,
                    })
                df = pd.DataFrame(rows)
                if not df.empty and "Qualifying Position" in df.columns:
                    df["Qualifying Position"] = int_col(df["Qualifying Position"])
                return df
    return pd.DataFrame()


def extract_practice_lap_counts(feed: dict) -> dict:
    """Extract total laps completed per driver from the practice run in weekend-feed.
    Returns {driver_name: laps_completed}.
    """
    if not feed:
        return {}
    for run in reversed(feed.get("weekend_runs", [])):
        rt = run.get("run_type", 0)
        rn = str(run.get("run_name", "")).lower()
        if rt == 1 or "practice" in rn:
            results = run.get("results", [])
            if results:
                counts = {}
                car_map = _build_car_driver_map(feed)
                for r in results:
                    cn = str(r.get("car_number", "")).strip()
                    driver = r.get("driver_fullname")
                    if not driver and cn in car_map:
                        driver = car_map[cn]["driver"]
                    laps = r.get("laps_completed")
                    if driver and laps is not None:
                        driver = _clean_api_name(driver)
                        counts[driver] = int(laps)
                if counts:
                    return counts
    return {}


def extract_practice_laps(feed: dict) -> list:
    """Extract practice lap-by-lap data from weekend_runs for the lap chart.
    Returns list of dicts: [{driver, laps: [{lap_num, lap_time}]}]
    """
    if not feed:
        return []
    car_map = _build_car_driver_map(feed)
    for run in reversed(feed.get("weekend_runs", [])):
        rt = run.get("run_type", 0)
        rn = str(run.get("run_name", "")).lower()
        if rt == 1 or "practice" in rn:
            results = run.get("results", [])
            if results:
                driver_laps = []
                for r in results:
                    cn = str(r.get("car_number", "")).strip()
                    driver = r.get("driver_fullname")
                    if not driver and cn in car_map:
                        driver = car_map[cn]["driver"]
                    laps = r.get("laps", [])
                    if driver and laps:
                        lap_data = []
                        for lap in laps:
                            lt = lap.get("lap_time") or lap.get("LapTime")
                            ln = lap.get("lap_number") or lap.get("Lap")
                            if lt and ln and lt > 0:
                                lap_data.append({"lap_num": ln, "lap_time": lt})
                        if lap_data:
                            driver_laps.append({"driver": driver, "laps": lap_data})
                if driver_laps:
                    return driver_laps
    return []


def extract_race_results(feed: dict) -> pd.DataFrame:
    """Extract race results, filtering out DNQ drivers."""
    if not feed:
        return pd.DataFrame()
    races = feed.get("weekend_race", [])
    if not races:
        return pd.DataFrame()
    results = races[0].get("results", [])
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        status = r.get("finishing_status") or ""
        fp = r.get("finishing_position", 0) or 0
        if fp == 0 and not status.strip():
            continue  # DNQ
        rows.append({
            "Finish Position": fp,
            "Driver": _clean_api_name(r.get("driver_fullname") or ""),
            "Start": r.get("starting_position") or r.get("qualifying_position"),
            "Car": r.get("car_number"),
            "Team": r.get("team_name"),
            "Manufacturer": r.get("car_make"),
            "Crew Chief": r.get("crew_chief_fullname", ""),
            "Laps": r.get("laps_completed"),
            "Laps Led": r.get("laps_led", 0),
            "Status": status,
            "QualSpeed": r.get("qualifying_speed"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["Finish Position", "Start", "Laps", "Laps Led"]:
            if col in df.columns:
                df[col] = int_col(df[col])
    return df


# ============================================================
# LAP DATA COMPUTATIONS
# ============================================================

def compute_fastest_laps(lap_data: dict) -> Dict[str, int]:
    """Count fastest laps per driver from lap-times data."""
    drivers = lap_data.get("laps", [])
    if not drivers:
        return {}
    driver_laps = {}
    all_laps = set()
    for d in drivers:
        name = _clean_api_name(d["FullName"])
        driver_laps[name] = {}
        for lap in d.get("Laps", []):
            if lap["Lap"] > 0 and lap.get("LapTime") and lap["LapTime"] > 0:
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


def compute_avg_running_position(lap_data: dict) -> Dict[str, float]:
    """Average running position across all race laps."""
    drivers = lap_data.get("laps", [])
    result = {}
    for d in drivers:
        positions = [lap["RunningPos"] for lap in d.get("Laps", [])
                     if lap["Lap"] > 0 and lap.get("RunningPos")]
        if positions:
            result[_clean_api_name(d["FullName"])] = round(np.mean(positions), 1)
    return result


def save_arp_to_db(arp_data: dict, race_id: int) -> int:
    """Persist computed ARP values to the database for a given race.

    Args:
        arp_data: {driver_display_name: avg_running_position}
        race_id: NASCAR API race_id

    Returns count of rows updated.
    """
    if not arp_data or not DB_PATH.exists():
        return 0
    try:
        conn = sqlite3.connect(str(DB_PATH))
        # Resolve API race_id to DB race_id
        db_race = conn.execute(
            "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
        ).fetchone()
        if not db_race:
            conn.close()
            return 0
        db_race_id = db_race[0]

        # Check if ARP already filled for this race
        existing = conn.execute(
            "SELECT COUNT(*) FROM race_results WHERE race_id = ? AND avg_running_position IS NOT NULL",
            (db_race_id,)
        ).fetchone()[0]
        if existing > 0:
            conn.close()
            return 0  # Already backfilled

        # Match driver names and update
        db_drivers = conn.execute(
            """SELECT rr.id, d.full_name FROM race_results rr
               JOIN drivers d ON d.id = rr.driver_id
               WHERE rr.race_id = ?""",
            (db_race_id,)
        ).fetchall()

        arp_norm = {normalize_driver_name(k): v for k, v in arp_data.items()}
        count = 0
        for rr_id, db_name in db_drivers:
            arp = arp_data.get(db_name)
            if arp is None:
                norm_key = normalize_driver_name(db_name)
                arp = arp_norm.get(norm_key)
            if arp is not None:
                conn.execute(
                    "UPDATE race_results SET avg_running_position = ? WHERE id = ?",
                    (arp, rr_id)
                )
                count += 1
        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def load_arp_from_db(race_id: int) -> dict:
    """Load saved ARP data from DB for a given API race_id.

    Returns {driver_display_name: avg_running_position}.
    """
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        db_race = conn.execute(
            "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
        ).fetchone()
        if not db_race:
            conn.close()
            return {}
        rows = conn.execute(
            """SELECT d.full_name, rr.avg_running_position
               FROM race_results rr
               JOIN drivers d ON d.id = rr.driver_id
               WHERE rr.race_id = ? AND rr.avg_running_position IS NOT NULL""",
            (db_race[0],)
        ).fetchall()
        conn.close()
        return {name: arp for name, arp in rows}
    except Exception:
        return {}


def query_db_track_history(track_name: str, series_id: int = 1,
                            min_season: int = 2022) -> pd.DataFrame:
    """Query per-driver track history from DB (Next Gen era, 2022+).

    Returns full column set so it can be displayed alongside the by-type
    aggregation: Driver, Races, Avg Finish, Avg Start, Avg Run Pos,
    Avg DFS, Best DFS, Wins, Top 5, Top 10, Laps Led, Avg Laps Led,
    Fast Laps, Avg Fastest Laps, DNF.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query('''
            SELECT d.full_name as Driver,
                   COUNT(*) as Races,
                   ROUND(AVG(rr.finish_pos), 1) as "Avg Finish",
                   ROUND(AVG(rr.start_pos), 1) as "Avg Start",
                   COALESCE(ROUND(AVG(rr.avg_running_position), 1), 99) as "Avg Run Pos",
                   ROUND(AVG(dp.dfs_score), 1) as "Avg DFS",
                   ROUND(MAX(dp.dfs_score), 1) as "Best DFS",
                   SUM(CASE WHEN rr.finish_pos = 1 THEN 1 ELSE 0 END) as Wins,
                   SUM(CASE WHEN rr.finish_pos <= 5 THEN 1 ELSE 0 END) as "Top 5",
                   SUM(CASE WHEN rr.finish_pos <= 10 THEN 1 ELSE 0 END) as "Top 10",
                   SUM(rr.laps_led) as "Laps Led",
                   ROUND(AVG(rr.laps_led), 1) as "Avg Laps Led",
                   SUM(rr.fastest_laps) as "Fast Laps",
                   ROUND(AVG(rr.fastest_laps), 1) as "Avg Fastest Laps",
                   SUM(CASE WHEN LOWER(COALESCE(rr.status,'running'))
                        NOT IN ('running','') THEN 1 ELSE 0 END) as DNF
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            LEFT JOIN dfs_points dp ON dp.race_id = rr.race_id
                AND dp.driver_id = rr.driver_id AND dp.platform = 'DraftKings'
            WHERE t.name = ?
              AND r.series_id = ?
              AND r.season >= ?
            GROUP BY d.id
            HAVING COUNT(*) >= 1
            ORDER BY "Avg Finish" ASC
        ''', conn, params=[track_name, series_id, min_season])
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_driver_dk_points_at_track(track_name: str, series_id: int = 1,
                                     min_season: int = 2022,
                                     before_date: str = None) -> dict:
    """Compute per-driver avg/best/worst DK points at a track from race_results.

    Returns {driver_name: {"avg_dk": X, "best_dk": Y, "worst_dk": Z, "races": N}}.
    DK points computed from finish, start, laps_led, fastest_laps.
    """
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        where_extra = ""
        params = [track_name, series_id, min_season]
        if before_date:
            where_extra = " AND r.race_date < ?"
            params.append(before_date)

        rows = conn.execute(f'''
            SELECT d.full_name, rr.finish_pos, rr.start_pos,
                   rr.laps_led, rr.fastest_laps
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE t.name = ?
              AND r.series_id = ?
              AND r.season >= ?
              {where_extra}
        ''', params).fetchall()
        conn.close()

        from src.utils import calc_dk_points
        driver_scores = {}
        for name, finish, start, ll, fl in rows:
            if finish is None or start is None:
                continue
            dk = calc_dk_points(finish, start, ll or 0, fl or 0)
            driver_scores.setdefault(name, []).append(dk)

        result = {}
        for name, scores in driver_scores.items():
            if scores:
                result[name] = {
                    "avg_dk": round(sum(scores) / len(scores), 1),
                    "best_dk": round(max(scores), 1),
                    "worst_dk": round(min(scores), 1),
                    "races": len(scores),
                }
        return result
    except Exception:
        return {}


def query_driver_race_log(
    driver_name: str,
    series_id: int,
    track_name: str = None,
    track_type: str = None,
    season: int = None,
    min_season: int = 2022,
    before_date: str = None,
) -> list:
    """Per-race log for a single driver, filtered to a track OR a track type
    OR an entire season.

    Returns a list of dicts (newest first), one per race, suitable for
    `render_driver_race_log` in components.py:
        {"Date": str, "Race": str, "Track": str,
         "Start": int, "Finish": int, "Laps Led": int, "Fast Laps": int,
         "Avg Run": float, "DK Pts": float, "Status": str}

    Specify EITHER track_name (single-track view), track_type (track-type
    folded view — short_concrete pulls in short, intermediate ↔ intermediate_worn),
    OR season (all races in a single year — used by the Standings tab).
    """
    if not DB_PATH.exists() or not driver_name:
        return []
    if not track_name and not track_type and not season:
        return []

    from src.config import TRACK_TYPE_MAP, TRACK_TYPE_PARENT

    where = ["d.full_name = ?", "r.series_id = ?"]
    params = [driver_name, series_id]

    # Season filter mode: ONLY this season (no min_season floor)
    if season is not None:
        where.append("r.season = ?")
        params.append(season)
    else:
        where.append("r.season >= ?")
        params.append(min_season)

    if track_name:
        where.append("t.name = ?")
        params.append(track_name)
    elif track_type:
        # Same family-folding rules used elsewhere
        parent = TRACK_TYPE_PARENT.get(track_type, track_type)
        include_types = {track_type}
        for tt, p in TRACK_TYPE_PARENT.items():
            if tt == track_type or p == parent:
                include_types.add(tt)
        if track_type == "short_concrete":
            include_types.add("short")
        matching_tracks = [t for t, tt in TRACK_TYPE_MAP.items() if tt in include_types]
        if not matching_tracks:
            return []
        placeholders = ",".join("?" for _ in matching_tracks)
        where.append(f"t.name IN ({placeholders})")
        params.extend(matching_tracks)

    if before_date:
        where.append("r.race_date < ?")
        params.append(before_date)

    where_clause = " AND ".join(where)

    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(f'''
            SELECT r.race_date, r.race_name, t.name as track,
                   rr.car_number, rr.team,
                   rr.start_pos, rr.finish_pos,
                   rr.laps_led, rr.fastest_laps,
                   rr.avg_running_position, rr.status
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r   ON r.id = rr.race_id
            JOIN tracks t  ON t.id = r.track_id
            WHERE {where_clause} AND rr.finish_pos IS NOT NULL
            ORDER BY r.race_date DESC, r.id DESC
        ''', params).fetchall()
        conn.close()
    except Exception:
        return []

    if not rows:
        return []

    from src.utils import calc_dk_points
    out = []
    for rdate, rname, track, car, team, start, finish, ll, fl, arp, status in rows:
        ll_v = ll or 0
        fl_v = fl or 0
        try:
            dk = calc_dk_points(finish, start or 0, ll_v, fl_v) if (start is not None and finish) else None
        except Exception:
            dk = None
        # Trim ISO time off the date for display
        date_str = (str(rdate) or "")[:10]
        out.append({
            "Date": date_str,
            "Race": rname,
            "Track": track,
            "Car": car,
            "Team": team,
            "Start": start,
            "Finish": finish,
            "Laps Led": ll_v,
            "Fast Laps": fl_v,
            "Avg Run": round(arp, 1) if arp is not None else None,
            "DK Pts": round(dk, 2) if dk is not None else None,
            "Status": status,
        })
    return out


def query_driver_finishes_by_track_type(
    track_type: str,
    series_id: int,
    drivers: list = None,
    last_n: int = 10,
    before_date: str = None,
) -> tuple:
    """Per-driver per-race finish positions at recent races of this track type.

    Track-type matching follows the same parent-aware logic as `query_team_stats`:
      - "short_concrete" pulls in all "short" tracks (Bristol/Dover + Martinsville/etc.)
      - "intermediate_worn" pulls in all "intermediate" tracks
      - All other types match exactly

    Returns (race_meta, driver_finishes) where:
      race_meta: list of dicts in chronological order (oldest -> newest), each
        {"race_id": int, "track": str, "short_name": str, "race_date": str,
         "label": str}  -- label is e.g. "Vegas (3/2)"
      driver_finishes: {driver_name: {race_id: finish_pos}}

    Args:
        track_type: e.g. "intermediate", "short_concrete", "intermediate_worn"
        series_id: 1=Cup, 2=O'Reilly, 3=Truck
        drivers: optional list of driver names to filter to
        last_n: max number of recent races to return per driver (default 10)
        before_date: if set, only races before this date (YYYY-MM-DD)
    """
    if not DB_PATH.exists() or not track_type:
        return ([], {})

    from src.config import TRACK_TYPE_MAP, TRACK_TYPE_PARENT
    # Build the family of types to include. Default uses parent-aware folding
    # (intermediate_worn folds into intermediate). For short_concrete we also
    # pull regular short tracks so Bristol/Dover races include comparable
    # short-track form.
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    include_types = {track_type}
    for tt, p in TRACK_TYPE_PARENT.items():
        if tt == track_type or p == parent:
            include_types.add(tt)
    if track_type == "short_concrete":
        include_types.add("short")
    matching_tracks = [t for t, tt in TRACK_TYPE_MAP.items() if tt in include_types]
    if not matching_tracks:
        return ([], {})

    try:
        conn = sqlite3.connect(str(DB_PATH))
        placeholders = ",".join("?" for _ in matching_tracks)
        where = (
            f"WHERE t.name IN ({placeholders}) AND r.series_id = ? "
            f"AND rr.finish_pos IS NOT NULL"
        )
        params = list(matching_tracks) + [series_id]
        if before_date:
            where += " AND r.race_date < ?"
            params.append(before_date)

        rows = conn.execute(f'''
            SELECT d.full_name, r.id, r.race_date, t.name, t.short_name,
                   rr.finish_pos
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            {where}
            ORDER BY r.race_date DESC, r.id DESC
        ''', params).fetchall()
        conn.close()
    except Exception:
        return ([], {})

    if not rows:
        return ([], {})

    # Per-driver: keep only the most-recent N races
    driver_filter = set(drivers) if drivers else None
    per_driver = {}                # name -> [(race_id, finish, date, track, short)]
    for name, race_id, rdate, track, short, finish in rows:
        if driver_filter and name not in driver_filter:
            continue
        if name not in per_driver:
            per_driver[name] = []
        if len(per_driver[name]) < last_n:
            per_driver[name].append((race_id, finish, rdate, track, short))

    # Build the global race set across the filtered drivers (union of races
    # any included driver participated in within their last-N window)
    race_meta = {}  # race_id -> {race_id, track, short_name, race_date}
    driver_finishes = {}
    for name, races in per_driver.items():
        finishes = {}
        for race_id, finish, rdate, track, short in races:
            finishes[race_id] = int(finish)
            if race_id not in race_meta:
                race_meta[race_id] = {
                    "race_id": race_id,
                    "track": track,
                    "short_name": short,
                    "race_date": rdate,
                }
        driver_finishes[name] = finishes

    # Order chronologically (oldest -> newest, so columns read left=old, right=new
    # → caller can reverse for newest-first display)
    meta_list = sorted(race_meta.values(),
                       key=lambda m: (m["race_date"] or "", m["race_id"]))

    # Build short labels: prefer tracks.short_name, else first word(s) of track,
    # plus M/D date. e.g. "Vegas (3/2)" / "Charlotte (5/24)"
    def _abbreviate(track_name, short):
        if short and len(short) > 0:
            return short
        # Strip "Speedway"/"Motor Speedway"/"International Speedway"/"Raceway"
        for suffix in [" International Speedway", " Motor Speedway",
                       " Speedway", " Raceway", " Course"]:
            if track_name and track_name.endswith(suffix):
                return track_name[:-len(suffix)]
        return track_name or "?"

    def _fmt_date(d):
        if not d:
            return ""
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(str(d).split("T")[0])
            return f"{dt.month}/{dt.day}"
        except Exception:
            return str(d)[:5]

    for m in meta_list:
        abbr = _abbreviate(m["track"], m["short_name"])
        date_str = _fmt_date(m["race_date"])
        m["label"] = f"{abbr} ({date_str})" if date_str else abbr

    return (meta_list, driver_finishes)


def query_driver_dk_points_by_track_type(
    track_type: str,
    series_id: int,
    season: int,
    before_date: str = None,
    exclude_track: str = None,
) -> dict:
    """Compute per-driver avg/best/worst DK points at tracks of a given TYPE
    within a single season. Useful for "how have drivers done at similar
    tracks this year?" — captures current-season form.

    Args:
        track_type: e.g. "intermediate", "short", "superspeedway"
        series_id: 1=Cup, 2=O'Reilly, 3=Truck
        season: year (e.g. 2026)
        before_date: if set, only races before this date (YYYY-MM-DD).
        exclude_track: if set, exclude races at this track name (e.g. to
            avoid double-counting the track the user is currently viewing).

    Returns {driver_name: {"avg_dk": X, "best_dk": Y, "worst_dk": Z, "races": N}}.
    """
    if not DB_PATH.exists() or not track_type:
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        where = (
            "WHERE t.track_type = ? AND r.series_id = ? AND r.season = ? "
            "AND rr.finish_pos IS NOT NULL"
        )
        params = [track_type, series_id, season]
        if before_date:
            where += " AND r.race_date < ?"
            params.append(before_date)
        if exclude_track:
            where += " AND t.name != ?"
            params.append(exclude_track)

        rows = conn.execute(f'''
            SELECT d.full_name, rr.finish_pos, rr.start_pos,
                   rr.laps_led, rr.fastest_laps, r.race_name, r.race_date
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            {where}
            ORDER BY r.race_date DESC
        ''', params).fetchall()
        conn.close()

        from src.utils import calc_dk_points
        driver_scores = {}
        for name, finish, start, ll, fl, _rname, _rdate in rows:
            if finish is None or start is None:
                continue
            dk = calc_dk_points(finish, start, ll or 0, fl or 0)
            driver_scores.setdefault(name, []).append(dk)

        result = {}
        for name, scores in driver_scores.items():
            if scores:
                result[name] = {
                    "avg_dk": round(sum(scores) / len(scores), 1),
                    "best_dk": round(max(scores), 1),
                    "worst_dk": round(min(scores), 1),
                    "races": len(scores),
                }
        return result
    except Exception:
        return {}


def query_driver_track_dnf(track_name: str, series_id: int,
                            before_date: str = None,
                            min_races: int = 3) -> dict:
    """Query DNF and crash rates for drivers at a SPECIFIC track.

    Complements query_driver_career_dnf by providing track-specific signal.
    At superspeedways, career DNF rates miss the fact that some drivers
    run the back-of-pack draft (high crash exposure) while others stay up
    front (lower exposure). Per-track data captures that.

    Returns {driver_name: {dnf_rate, crash_rate, speed_score, races}}.
    Only includes drivers with >= min_races (default 3) at this track.

    Returns empty dict if no data. Callers should fall back to career stats.
    """
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        where = "WHERE r.series_id = ? AND t.name = ?"
        params = [series_id, track_name]
        if before_date:
            where += " AND r.race_date < ?"
            params.append(before_date)

        rows = conn.execute(f'''
            SELECT d.full_name,
                   COUNT(*) as races,
                   SUM(CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END) as dnfs,
                   SUM(CASE WHEN LOWER(rr.status) IN ('accident','crash','damage') THEN 1 ELSE 0 END) as crashes,
                   1.0 * SUM(rr.laps_led) / COUNT(*) as ll_per_race,
                   1.0 * SUM(rr.fastest_laps) / COUNT(*) as fl_per_race
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            {where}
            GROUP BY d.id
            HAVING races >= ?
        ''', params + [min_races]).fetchall()
        conn.close()

        result = {}
        for r in rows:
            name, races, dnfs, crashes, ll_per, fl_per = r
            if races and races > 0:
                speed = (ll_per or 0) + (fl_per or 0)
                result[name] = {
                    "dnf_rate": (dnfs or 0) / races,
                    "crash_rate": (crashes or 0) / races,
                    "speed_score": speed,
                    "races": races,
                }
        return result
    except Exception:
        return {}


def query_driver_career_dnf(series_id: int, before_date: str = None) -> dict:
    """Query career DNF and crash rates for all drivers.

    Returns {driver_name: {dnf_rate, crash_rate, speed_score, races}}.
    Only includes drivers with 5+ career races.
    """
    if not DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        where = "WHERE r.series_id = ?"
        params = [series_id]
        if before_date:
            where += " AND r.race_date < ?"
            params.append(before_date)

        rows = conn.execute(f'''
            SELECT d.full_name,
                   COUNT(*) as races,
                   SUM(CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END) as dnfs,
                   SUM(CASE WHEN LOWER(rr.status) IN ('accident','crash','damage') THEN 1 ELSE 0 END) as crashes,
                   1.0 * SUM(rr.laps_led) / COUNT(*) as ll_per_race,
                   1.0 * SUM(rr.fastest_laps) / COUNT(*) as fl_per_race
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            {where}
            GROUP BY d.id
            HAVING races >= 5
        ''', params).fetchall()
        conn.close()

        result = {}
        for r in rows:
            name, races, dnfs, crashes, ll_per, fl_per = r
            if races and races > 0:
                speed = (ll_per or 0) + (fl_per or 0)
                result[name] = {
                    "dnf_rate": (dnfs or 0) / races,
                    "crash_rate": (crashes or 0) / races,
                    "speed_score": speed,
                    "races": races,
                }
        return result
    except Exception:
        return {}


# ============================================================
# TRACK HISTORY SCRAPING
# ============================================================

def _scrape_da_tables(track_name: str, series_id: int = 1):
    """Scrape driveraverages.com and return (recent_df, alltime_df) tuple."""
    key = track_name.lower().strip()
    trk_id = None
    # Sort by key length descending so "charlotte roval" matches before "charlotte"
    for name, tid in sorted(DA_TRACK_IDS.items(), key=lambda x: len(x[0]), reverse=True):
        if name in key:
            trk_id = tid
            break
    if trk_id is None:
        return pd.DataFrame(), pd.DataFrame()

    series_map = {1: "nascar", 2: "nascar_secondseries", 3: "nascar_truckseries"}
    series_path = series_map.get(series_id, "nascar")

    # Column layouts: site uses different layouts per series
    # Cup: 13 or 15 columns; Xfinity/Truck: 14 (recent) or 10 (alltime)
    VALID_COL_COUNTS = (10, 13, 14, 15)

    col_names_13_recent = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Laps Led", "Avg Start", "Best Finish",
        "Worst Finish", "Detail",
    ]
    col_names_13_alltime = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Laps Led", "Avg Start", "Best Finish",
        "Worst Finish", "DNF", "Detail",
    ]
    col_names_14_recent = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Top 20", "Laps Led", "Avg Start", "Best Finish",
        "DNF", "Detail",
    ]
    col_names_10_alltime = [
        "Rank", "Driver", "Wins", "Races", "Avg Finish",
        "Top 5", "Top 10", "Avg Start", "Best Finish", "DNF",
    ]
    col_names_15 = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Top 20", "Laps Led", "Avg Start", "Best Finish",
        "Worst Finish", "DNF", "Detail",
    ]

    try:
        resp = requests.get(
            f"https://www.driveraverages.com/{series_path}/track_avg.php?trk_id={trk_id}",
            timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(resp.text, "lxml")

        all_tables = []
        for t in soup.find_all("table"):
            # Skip wrapper tables that contain nested data tables
            if t.find("table"):
                has_nested_data = False
                for nt in t.find_all("table"):
                    for tr in nt.find_all("tr"):
                        if len(tr.find_all("td")) in VALID_COL_COUNTS:
                            has_nested_data = True
                            break
                    if has_nested_data:
                        break
                if has_nested_data:
                    continue
            rows = []
            ncols = None
            for tr in t.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) in VALID_COL_COUNTS:
                    vals = [c.get_text(strip=True) for c in cells]
                    if vals[1] and vals[1] != "Driver":
                        rows.append(vals)
                        ncols = len(cells)
            if rows:
                all_tables.append((rows, ncols))

        def _build_df(rows, ncols, is_alltime=False):
            if ncols == 15:
                cols = col_names_15
            elif ncols == 14:
                cols = col_names_14_recent
            elif ncols == 13:
                cols = col_names_13_alltime if is_alltime else col_names_13_recent
            elif ncols == 10:
                cols = col_names_10_alltime
            else:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=cols)
            df = df.drop(columns=["Detail"], errors="ignore")
            for col in df.columns:
                if col != "Driver":
                    # Strip commas from numbers like "1,027" before conversion
                    df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df

        recent_df = pd.DataFrame()
        alltime_df = pd.DataFrame()
        if len(all_tables) >= 1:
            recent_df = _build_df(all_tables[0][0], all_tables[0][1], is_alltime=False)
        if len(all_tables) >= 2:
            alltime_df = _build_df(all_tables[1][0], all_tables[1][1], is_alltime=True)

        return recent_df, alltime_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def scrape_track_history(track_name: str, series_id: int = 1) -> pd.DataFrame:
    """Scrape driveraverages.com recent track history (v3: comma-strip)."""
    recent, _ = _scrape_da_tables(track_name, series_id)
    return recent


@st.cache_data(ttl=3600, show_spinner=False)
def scrape_track_history_alltime(track_name: str, series_id: int = 1) -> pd.DataFrame:
    """Scrape driveraverages.com all-time track history (v3: comma-strip)."""
    _, alltime = _scrape_da_tables(track_name, series_id)
    return alltime


# ============================================================
# DATABASE QUERIES
# ============================================================

def query_gfs_stats(series_id: int = None) -> pd.DataFrame:
    """Query season GFS (Game Format Stats) from database.
    Returns: Driver, Races, Avg DK Pts, Avg FD Pts, Avg Finish, Avg Start, Wins, T5, T10.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT d.full_name as Driver,
                   COUNT(*) as Races,
                   ROUND(AVG(CASE WHEN dp.platform='DraftKings' THEN dp.dfs_score END), 1) as "Avg DK Pts",
                   ROUND(AVG(CASE WHEN dp.platform='FanDuel' THEN dp.dfs_score END), 1) as "Avg FD Pts",
                   ROUND(AVG(rr.finish_pos), 1) as "Avg Finish",
                   ROUND(AVG(rr.start_pos), 1) as "Avg Start",
                   SUM(CASE WHEN rr.finish_pos = 1 THEN 1 ELSE 0 END) as Wins,
                   SUM(CASE WHEN rr.finish_pos <= 5 THEN 1 ELSE 0 END) as "Top 5",
                   SUM(CASE WHEN rr.finish_pos <= 10 THEN 1 ELSE 0 END) as "Top 10",
                   ROUND(AVG(rr.laps_led), 1) as "Avg Laps Led",
                   ROUND(AVG(rr.fastest_laps), 1) as "Avg Fast Laps"
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            LEFT JOIN dfs_points dp ON dp.race_id = rr.race_id AND dp.driver_id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
        """
        params = ()
        if series_id:
            query += " WHERE r.series_id = ?"
            params = (series_id,)
        query += " GROUP BY d.full_name ORDER BY \"Avg DK Pts\" DESC"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_season_stats(track_name: str = None, season: int = None,
                       series_id: int = None) -> pd.DataFrame:
    """Pull aggregated season stats from local DB, optionally filtered."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT d.full_name as Driver,
                   COUNT(*) as Races,
                   ROUND(AVG(rr.finish_pos),1) as "Avg Finish",
                   ROUND(AVG(rr.start_pos),1) as "Avg Start",
                   ROUND(AVG(dp.dfs_score),1) as "Avg DFS",
                   ROUND(MAX(dp.dfs_score),1) as "Best DFS",
                   ROUND(MIN(dp.dfs_score),1) as "Worst DFS",
                   SUM(CASE WHEN rr.finish_pos=1 THEN 1 ELSE 0 END) as Wins,
                   SUM(CASE WHEN rr.finish_pos<=5 THEN 1 ELSE 0 END) as "Top 5",
                   SUM(CASE WHEN rr.finish_pos<=10 THEN 1 ELSE 0 END) as "Top 10",
                   ROUND(AVG(rr.laps_led),1) as "Avg Laps Led",
                   ROUND(AVG(rr.fastest_laps),1) as "Avg Fastest Laps",
                   SUM(CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END) as DNF
            FROM race_results rr
            JOIN drivers d ON d.id=rr.driver_id
            JOIN races r ON r.id=rr.race_id
            LEFT JOIN dfs_points dp ON dp.race_id=rr.race_id AND dp.driver_id=rr.driver_id
                                       AND dp.platform='DraftKings'
        """
        conditions = []
        params = []
        if track_name:
            query += " JOIN tracks t ON t.id=r.track_id"
            conditions.append("t.name = ?")
            params.append(track_name)
        if season:
            conditions.append("r.season = ?")
            params.append(season)
        if series_id:
            conditions.append("r.series_id = ?")
            params.append(series_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += """
            GROUP BY d.full_name
            ORDER BY "Avg DFS" DESC
        """
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_track_type_stats(track_type: str, season: int = None,
                            series_id: int = None) -> pd.DataFrame:
    """Pull stats filtered by track type from local DB.

    Handles both DB-level types (short, intermediate, road, superspeedway, dirt)
    and config-level subtypes (short_concrete, intermediate_worn) by:
    - For subtypes: filtering by track names belonging to that subtype
    - For parent groups ("All Short"): including all tracks of that parent type
    - For base types: querying by DB track_type directly

    When series_id is provided, results are filtered to that series only.
    Without it, results span all series (useful for cross-series analysis).
    """
    from src.config import TRACK_TYPE_MAP, TRACK_TYPE_PARENT
    if not DB_PATH.exists():
        return pd.DataFrame()

    # Determine query strategy
    is_parent_group = track_type.startswith("All ")
    # Detect subtypes: types in TRACK_TYPE_PARENT whose parent differs from themselves
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    is_subtype = (parent != track_type)

    if is_parent_group:
        parent_name = track_type.replace("All ", "").lower()
        # Get ALL tracks whose parent resolves to this group
        filter_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                         if TRACK_TYPE_PARENT.get(tt, tt) == parent_name]
    elif is_subtype:
        # Subtype like "short_concrete" — filter to specific tracks
        filter_tracks = [t for t, tt in TRACK_TYPE_MAP.items() if tt == track_type]
    else:
        # Base type (short, intermediate, road, superspeedway)
        # Get all tracks that belong to this type OR whose parent is this type
        filter_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                         if tt == track_type or TRACK_TYPE_PARENT.get(tt, tt) == track_type]

    if not filter_tracks:
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(str(DB_PATH))
        placeholders = ",".join("?" for _ in filter_tracks)

        query = f"""
            SELECT d.full_name as Driver,
                   COUNT(DISTINCT rr.race_id) as Races,
                   ROUND(AVG(rr.finish_pos),1) as "Avg Finish",
                   ROUND(AVG(rr.start_pos),1) as "Avg Start",
                   COALESCE(ROUND(AVG(rr.avg_running_position),1), 99) as "Avg Run Pos",
                   ROUND(AVG(dp.dfs_score),1) as "Avg DFS",
                   ROUND(MAX(dp.dfs_score),1) as "Best DFS",
                   SUM(CASE WHEN rr.finish_pos = 1 THEN 1 ELSE 0 END) as Wins,
                   SUM(CASE WHEN rr.finish_pos<=5 THEN 1 ELSE 0 END) as "Top 5",
                   SUM(CASE WHEN rr.finish_pos<=10 THEN 1 ELSE 0 END) as "Top 10",
                   SUM(rr.laps_led) as "Laps Led",
                   ROUND(AVG(rr.laps_led),1) as "Avg Laps Led",
                   SUM(rr.fastest_laps) as "Fast Laps",
                   ROUND(AVG(rr.fastest_laps),1) as "Avg Fastest Laps",
                   SUM(CASE WHEN LOWER(COALESCE(rr.status,'running'))
                        NOT IN ('running','') THEN 1 ELSE 0 END) as DNF
            FROM race_results rr
            JOIN drivers d ON d.id=rr.driver_id
            LEFT JOIN dfs_points dp ON dp.race_id=rr.race_id
                AND dp.driver_id=rr.driver_id AND dp.platform='DraftKings'
            JOIN races r ON r.id=rr.race_id
            JOIN tracks t ON t.id=r.track_id
            WHERE t.name IN ({placeholders})
        """
        params = list(filter_tracks)
        if series_id:
            query += " AND r.series_id = ?"
            params.append(series_id)
        if season:
            query += " AND r.season = ?"
            params.append(season)
        query += """
            GROUP BY d.full_name
            HAVING Races >= 1
            ORDER BY "Avg Finish" ASC
        """

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_team_stats(series_id: int, track_type: str = None,
                     min_season: int = 2022, before_date: str = None) -> dict:
    """Query team performance stats from race_results.

    Returns {team_name: {"avg_finish": X, "avg_arp": Y, "races": N}}.
    Optionally filtered by track type for track-type-specific team quality.
    """
    if not DB_PATH.exists():
        return {}

    from src.config import TRACK_TYPE_MAP, TRACK_TYPE_PARENT

    conn = sqlite3.connect(str(DB_PATH))
    params = [series_id, min_season]
    track_filter = ""

    if track_type:
        # Resolve track names for this track type (including parent groups)
        parent = TRACK_TYPE_PARENT.get(track_type, track_type)
        matching_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                           if tt == track_type or TRACK_TYPE_PARENT.get(tt, tt) == parent]
        if matching_tracks:
            placeholders = ",".join("?" for _ in matching_tracks)
            track_filter = f"AND t.name IN ({placeholders})"
            params.extend(matching_tracks)

    if before_date:
        track_filter += " AND r.race_date < ?"
        params.append(before_date)

    query = f'''
        SELECT rr.team, COUNT(*) as races,
               ROUND(AVG(rr.finish_pos), 1) as avg_finish,
               ROUND(AVG(rr.avg_running_position), 1) as avg_arp
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE r.series_id = ? AND r.season >= ?
          AND rr.team IS NOT NULL AND rr.team != ''
          {track_filter}
        GROUP BY rr.team
        HAVING races >= 3
        ORDER BY avg_finish
    '''
    try:
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return {
            row[0]: {"avg_finish": row[2], "avg_arp": row[3], "races": row[1]}
            for row in rows if row[0]
        }
    except Exception:
        conn.close()
        return {}


def query_team_quality_lookup(series_id: int, min_season: int = 2022,
                              before_date: str = None) -> dict:
    """Query overall team quality: avg finish across all races per team.

    Returns {team_name: avg_finish_pos} for teams with sufficient data.
    This is a broad measure of team competitiveness (lower = better).
    """
    if not DB_PATH.exists():
        return {}

    conn = sqlite3.connect(str(DB_PATH))
    params = [series_id, min_season]
    date_filter = ""
    if before_date:
        date_filter = "AND r.race_date < ?"
        params.append(before_date)

    query = f'''
        SELECT rr.team, ROUND(AVG(rr.finish_pos), 2) as avg_finish,
               COUNT(*) as entries
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        WHERE r.series_id = ? AND r.season >= ?
          AND rr.team IS NOT NULL AND rr.team != ''
          {date_filter}
        GROUP BY rr.team
        HAVING entries >= 20
        ORDER BY avg_finish
    '''
    try:
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows if row[0]}
    except Exception:
        conn.close()
        return {}


def query_driver_track_history_by_team(track_name: str, series_id: int,
                                        before_date: str = None) -> dict:
    """Query per-driver track history broken down by team.

    Returns {driver_name: [{"team": str, "finish_pos": int, "race_date": str}, ...]}
    Each entry is one race at this track with the team they drove for.
    """
    if not DB_PATH.exists():
        return {}

    conn = sqlite3.connect(str(DB_PATH))
    params = [track_name, series_id]
    date_filter = ""
    if before_date:
        date_filter = "AND r.race_date < ?"
        params.append(before_date)

    query = f'''
        SELECT d.full_name, rr.team, rr.finish_pos, r.race_date
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE t.name = ? AND r.series_id = ?
          AND rr.team IS NOT NULL AND rr.team != ''
          {date_filter}
        ORDER BY d.full_name, r.race_date
    '''
    try:
        rows = conn.execute(query, params).fetchall()
        conn.close()
    except Exception:
        conn.close()
        return {}

    result = defaultdict(list)
    for name, team, finish_pos, race_date in rows:
        result[name].append({
            "team": team,
            "finish_pos": finish_pos,
            "race_date": race_date,
        })
    return dict(result)


def compute_team_adjusted_track_history(track_name: str, series_id: int,
                                         driver_team_map: dict,
                                         before_date: str = None,
                                         track_type: str = None) -> dict:
    """Compute team-adjusted track history for all drivers.

    For each driver, compares the quality of their historical teams at this
    track to their current team. Adjusts avg_finish by the team quality delta.

    Args:
        track_name: track name for DB query
        series_id: series ID
        driver_team_map: {driver_name: current_team_name} from entry list
        before_date: only include races before this date
        track_type: when provided, uses track-type-specific team quality
            (e.g. Spire at superspeedway vs their career-wide average).
            Without it, falls back to career-wide quality.

    Returns {driver_name: {"team_adj": float}} where team_adj is the
    position adjustment to apply to track history (negative = improved team).
    """
    if not driver_team_map:
        return {}

    # Get team quality — track-type-specific when possible (much better
    # signal than career-wide, since teams' superspeedway vs short-track
    # performance can differ by 5+ positions).
    team_quality = {}
    if track_type:
        ts = query_team_stats(series_id, track_type=track_type,
                              before_date=before_date)
        if ts:
            # query_team_stats returns dict-of-dicts; extract avg_finish
            team_quality = {team: stats.get("avg_finish")
                            for team, stats in ts.items()
                            if stats.get("avg_finish") is not None}
    if not team_quality:
        # Fallback to career-wide
        team_quality = query_team_quality_lookup(series_id,
                                                  before_date=before_date)
    if not team_quality:
        return {}

    # Get per-driver race-by-race history with teams
    driver_history = query_driver_track_history_by_team(
        track_name, series_id, before_date=before_date)
    if not driver_history:
        return {}

    # Compute field-average team quality for normalization
    avg_team_quality = sum(team_quality.values()) / len(team_quality) if team_quality else 20.0

    from src.utils import fuzzy_match_name

    result = {}
    team_names = list(team_quality.keys())

    for driver, current_team in driver_team_map.items():
        # Match current team to team quality lookup
        matched_current = current_team if current_team in team_quality else \
            fuzzy_match_name(current_team, team_names) if team_names else None
        if not matched_current:
            continue
        current_quality = team_quality[matched_current]

        # Get this driver's race-by-race history
        # Try exact match first, then fuzzy
        hist = driver_history.get(driver)
        if not hist:
            matched_driver = fuzzy_match_name(driver, list(driver_history.keys()))
            hist = driver_history.get(matched_driver) if matched_driver else None
        if not hist:
            continue

        # Compute weighted average of historical team qualities
        # Weight recent races more heavily
        hist_team_quals = []
        for entry in hist:
            h_team = entry["team"]
            matched_h = h_team if h_team in team_quality else \
                fuzzy_match_name(h_team, team_names) if team_names else None
            if matched_h:
                hist_team_quals.append(team_quality[matched_h])

        if not hist_team_quals:
            continue

        avg_hist_team = sum(hist_team_quals) / len(hist_team_quals)

        # Team adjustment: current quality vs historical quality.
        # Negative = improved (lower avg finish = better current team).
        # Previously capped at +/- 4 and applied at 60% strength — too
        # conservative. Drivers whose history mixes good and bad teams
        # got only a fraction of the real boost/penalty they deserved.
        # Now: cap at +/- 6, apply at 80% strength.
        raw_adj = current_quality - avg_hist_team
        team_adj = max(-6.0, min(6.0, raw_adj))
        team_adj *= 0.80

        result[driver] = {"team_adj": round(team_adj, 2),
                          "current_team_quality": current_quality,
                          "hist_team_quality": round(avg_hist_team, 2)}

    return result


def query_manufacturer_stats(series_id: int, track_type: str = None,
                              min_season: int = 2022,
                              before_date: str = None) -> dict:
    """Query manufacturer performance stats from race_results.

    Returns {manufacturer: {"avg_finish": X, "races": N}}.
    Filtered by track type for track-specific manufacturer performance.
    """
    if not DB_PATH.exists():
        return {}

    from src.config import TRACK_TYPE_MAP, TRACK_TYPE_PARENT

    conn = sqlite3.connect(str(DB_PATH))
    params = [series_id, min_season]
    track_filter = ""

    if track_type:
        parent = TRACK_TYPE_PARENT.get(track_type, track_type)
        matching_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                           if tt == track_type or TRACK_TYPE_PARENT.get(tt, tt) == parent]
        if matching_tracks:
            placeholders = ",".join("?" for _ in matching_tracks)
            track_filter = f"AND t.name IN ({placeholders})"
            params.extend(matching_tracks)

    if before_date:
        track_filter += " AND r.race_date < ?"
        params.append(before_date)

    query = f'''
        SELECT rr.manufacturer, COUNT(*) as races,
               ROUND(AVG(rr.finish_pos), 1) as avg_finish
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE r.series_id = ? AND r.season >= ?
          AND rr.manufacturer IS NOT NULL AND rr.manufacturer != ''
          {track_filter}
        GROUP BY rr.manufacturer
        HAVING races >= 5
        ORDER BY avg_finish
    '''
    try:
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return {
            row[0]: {"avg_finish": row[2], "races": row[1]}
            for row in rows if row[0]
        }
    except Exception:
        conn.close()
        return {}


def query_salaries(race_id: int = None, platform: str = None) -> pd.DataFrame:
    """Query stored salaries from database.

    Args:
        race_id: NASCAR API race_id (resolved to DB race_id internally).
        platform: 'DraftKings' or 'FanDuel'.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT d.full_name as Driver, s.platform, s.salary as Salary, s.status
            FROM salaries s
            JOIN drivers d ON d.id = s.driver_id
            WHERE 1=1
        """
        params = []
        if race_id:
            # Resolve API race_id to internal DB race_id
            db_race = conn.execute(
                "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
            ).fetchone()
            if db_race:
                query += " AND s.race_id = ?"
                params.append(db_race[0])
            else:
                conn.close()
                return pd.DataFrame()
        if platform:
            query += " AND s.platform = ?"
            params.append(platform)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================
# AUTO SALARY FETCH (DraftKings API)
# ============================================================

DK_CONTEST_URLS = [
    "https://www.draftkings.com/lobby/getcontests?sport=NASCAR",
]
DK_DRAFTABLES_URL = "https://api.draftkings.com/draftgroups/v1/draftgroups/{}/draftables"


SERIES_DK_KEYWORDS = {
    1: ["cup"],
    2: ["xfinity", "o'reilly", "oreilly"],
    3: ["truck", "craftsman"],
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dk_salaries_live(series_id: int = 1) -> pd.DataFrame:
    """Fetch upcoming DraftKings NASCAR salaries directly from DK API.

    Args:
        series_id: 1=Cup, 2=Xfinity, 3=Trucks. Filters DK draft groups by
                   series keywords in the group name/suffix.

    Returns DataFrame with columns: Driver, DK Salary, Status.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    # Find draft groups
    data = None
    for url in DK_CONTEST_URLS:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            continue

    if not data or not isinstance(data, dict):
        return pd.DataFrame()

    draft_groups = data.get("DraftGroups") or data.get("draftGroups") or []
    # Lobby endpoint returns all sports — filter to NASCAR only
    draft_groups = [
        g for g in draft_groups
        if (g.get("Sport") or g.get("sport") or "").upper() == "NASCAR"
    ]
    if not draft_groups:
        return pd.DataFrame()

    # Try to match the specific series by keywords in group metadata
    keywords = SERIES_DK_KEYWORDS.get(series_id, [])
    if keywords:
        try:
            def _group_text(g):
                parts = [
                    str(g.get("ContestStartTimeSuffix") or ""),
                    str(g.get("GameType") or ""),
                    str(g.get("ContestTypeName") or ""),
                    str(g.get("GameSetKey") or ""),
                ]
                return " ".join(parts).lower()

            filtered = [g for g in draft_groups if any(kw in _group_text(g) for kw in keywords)]
            if filtered:
                draft_groups = filtered
        except Exception:
            pass  # Fall back to unfiltered groups

    # Get the first (most upcoming) matching group
    group = draft_groups[0]
    group_id = group.get("DraftGroupId") or group.get("draftGroupId")
    if not group_id:
        return pd.DataFrame()

    # Fetch draftables
    try:
        r = requests.get(DK_DRAFTABLES_URL.format(group_id), headers=headers, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        resp = r.json()
    except Exception:
        return pd.DataFrame()

    draftables = resp.get("draftables") or resp.get("Draftables") or []
    if not draftables:
        return pd.DataFrame()

    rows = []
    for player in draftables:
        name = (player.get("displayName") or player.get("DisplayName") or "").strip()
        salary = player.get("salary") or player.get("Salary")
        slot = (player.get("rosterSlotName") or player.get("RosterSlotName") or "").strip().lower()

        if not name or not salary:
            continue
        if slot in {"cpt", "captain"}:
            continue  # Skip captain rows

        is_disabled = player.get("isDisabled") or player.get("IsDisabled") or False
        status = "Out" if is_disabled else "Available"

        rows.append({"Driver": name, "DK Salary": salary, "Status": status})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Filter out "Out" drivers but keep them in data
    return df


def _ensure_race_in_db(conn, api_race_id, series_id, race_name):
    """Create a race entry in the DB if it doesn't exist yet (for upcoming races).

    Fetches race metadata from the NASCAR API to populate track, date, etc.
    Returns the DB row dict or None.
    """
    try:
        race_list = fetch_race_list(series_id)
        race_info = next((r for r in race_list if r.get("race_id") == api_race_id), None)
        if not race_info:
            return None

        track_name = race_info.get("track_name", "Unknown")
        race_date = race_info.get("race_date", "")
        scheduled_laps = race_info.get("scheduled_laps", 0)

        # Find or create track
        track_row = conn.execute("SELECT id FROM tracks WHERE name = ?", (track_name,)).fetchone()
        if not track_row:
            conn.execute("INSERT INTO tracks (name) VALUES (?)", (track_name,))
            track_row = conn.execute("SELECT id FROM tracks WHERE name = ?", (track_name,)).fetchone()

        from datetime import datetime as _dt
        season = _dt.now().year
        max_num = conn.execute(
            "SELECT COALESCE(MAX(race_num), 0) FROM races WHERE series_id = ? AND season = ?",
            (series_id, season)
        ).fetchone()[0]

        conn.execute(
            """INSERT OR IGNORE INTO races
               (series_id, track_id, season, race_num, race_name, race_date, laps, api_race_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (series_id, track_row[0], season, max_num + 1,
             race_name, race_date, scheduled_laps, api_race_id)
        )
        conn.commit()
        return conn.execute("SELECT id FROM races WHERE api_race_id = ?", (api_race_id,)).fetchone()
    except Exception:
        return None


def sync_dk_salaries_to_db(dk_df: pd.DataFrame, race_id: int, series_id: int,
                            race_name: str) -> int:
    """Save DK salary data into the database so the projection engine can use it.

    Returns count of salaries written.
    """
    if dk_df.empty or not os.path.exists(str(DB_PATH)):
        return 0

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Find DB race — api_race_id is the ONLY reliable key (unique per year+series).
        # Name-only fallback is unsafe: "Kansas Lottery 300" exists in multiple years,
        # and ORDER BY season DESC would route current-season salaries to the wrong year.
        db_race = None
        if race_id:
            db_race = conn.execute(
                "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
            ).fetchone()
            # If race not in DB yet (upcoming), auto-create from API data
            if not db_race:
                db_race = _ensure_race_in_db(conn, race_id, series_id, race_name)
        else:
            # No race_id at all — last resort, match by series+name+most-recent season
            db_race = conn.execute(
                "SELECT id FROM races WHERE series_id = ? AND race_name = ? ORDER BY season DESC LIMIT 1",
                (series_id, race_name)
            ).fetchone()

        if not db_race:
            conn.close()
            return 0

        db_race_id = db_race["id"]

        # Check if salaries already exist for this race
        existing = conn.execute(
            "SELECT COUNT(*) FROM salaries WHERE race_id = ? AND platform = 'DraftKings'",
            (db_race_id,)
        ).fetchone()[0]

        if existing > 0:
            # Delete old salaries and replace
            conn.execute(
                "DELETE FROM salaries WHERE race_id = ? AND platform = 'DraftKings'",
                (db_race_id,))

        # Pre-fetch all drivers for normalized matching
        all_drivers = conn.execute("SELECT id, full_name FROM drivers").fetchall()
        driver_norm_map = {}  # normalized_name → (id, full_name)
        for dr in all_drivers:
            nn = normalize_driver_name(dr["full_name"])
            if nn not in driver_norm_map:
                driver_norm_map[nn] = (dr["id"], dr["full_name"])

        count = 0
        for _, row in dk_df.iterrows():
            driver_name = row["Driver"]
            salary = row["DK Salary"]
            status = row.get("Status", "Available")

            # Find driver: exact match → normalized match → create new
            d = conn.execute("SELECT id FROM drivers WHERE full_name = ?",
                             (driver_name,)).fetchone()
            if not d:
                norm_key = normalize_driver_name(driver_name)
                if norm_key in driver_norm_map:
                    d = {"id": driver_norm_map[norm_key][0]}
                else:
                    conn.execute("INSERT INTO drivers (full_name) VALUES (?)", (driver_name,))
                    d = conn.execute("SELECT id FROM drivers WHERE full_name = ?",
                                     (driver_name,)).fetchone()
                    # Add to norm map for subsequent iterations
                    driver_norm_map[norm_key] = (d["id"], driver_name)

            conn.execute("""
                INSERT INTO salaries (race_id, driver_id, platform, salary, status)
                VALUES (?, ?, 'DraftKings', ?, ?)
            """, (db_race_id, d["id"], salary, status))
            count += 1

        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def sync_fd_salaries_to_db(fd_df: pd.DataFrame, race_id: int, series_id: int,
                            race_name: str) -> int:
    """Save FanDuel salary data into the database.

    Mirrors sync_dk_salaries_to_db but for the FanDuel platform.
    Returns count of salaries written.
    """
    if fd_df.empty or not os.path.exists(str(DB_PATH)):
        return 0

    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # Find DB race — api_race_id is the ONLY reliable key (unique per year+series).
        # Name-only fallback is unsafe: race names repeat across years.
        db_race = None
        if race_id:
            db_race = conn.execute(
                "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
            ).fetchone()
            # If race not in DB yet (upcoming), auto-create from API data
            if not db_race:
                db_race = _ensure_race_in_db(conn, race_id, series_id, race_name)
        else:
            # No race_id — last resort, match by series+name+most-recent season
            db_race = conn.execute(
                "SELECT id FROM races WHERE series_id = ? AND race_name = ? ORDER BY season DESC LIMIT 1",
                (series_id, race_name)
            ).fetchone()

        if not db_race:
            conn.close()
            return 0

        db_race_id = db_race["id"]

        # Delete old FD salaries and replace
        conn.execute(
            "DELETE FROM salaries WHERE race_id = ? AND platform = 'FanDuel'",
            (db_race_id,))

        # Pre-fetch all drivers for normalized matching
        all_drivers = conn.execute("SELECT id, full_name FROM drivers").fetchall()
        driver_norm_map = {}
        for dr in all_drivers:
            nn = normalize_driver_name(dr["full_name"])
            if nn not in driver_norm_map:
                driver_norm_map[nn] = (dr["id"], dr["full_name"])

        count = 0
        for _, row in fd_df.iterrows():
            driver_name = row["Driver"]
            salary = row["FD Salary"]

            d = conn.execute("SELECT id FROM drivers WHERE full_name = ?",
                             (driver_name,)).fetchone()
            if not d:
                norm_key = normalize_driver_name(driver_name)
                if norm_key in driver_norm_map:
                    d = {"id": driver_norm_map[norm_key][0]}
                else:
                    conn.execute("INSERT INTO drivers (full_name) VALUES (?)", (driver_name,))
                    d = conn.execute("SELECT id FROM drivers WHERE full_name = ?",
                                     (driver_name,)).fetchone()
                    driver_norm_map[norm_key] = (d["id"], driver_name)

            conn.execute("""
                INSERT INTO salaries (race_id, driver_id, platform, salary, status)
                VALUES (?, ?, 'FanDuel', ?, 'Available')
            """, (db_race_id, d["id"], salary))
            count += 1

        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


# ============================================================
# AUTO-FETCH NASCAR ODDS (Action Network)
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_all_nascar_odds(series_id: int = 1) -> dict:
    """Fetch upcoming NASCAR race odds from Action Network (Cup only).

    Primary odds come from the user's manual import (via UI paste/script).
    This auto-fetch is a backup for Cup races only — Action Network doesn't
    cover lower series.

    Returns {"win": {}, "top3": {}, "top5": {}, "top10": {}}.
    """
    empty = {"win": {}, "top3": {}, "top5": {}, "top10": {}}

    # Action Network (Cup only)
    if series_id == 1:
        result = _fetch_action_network_odds()
        if result.get("win"):
            return result

    return empty


def _fetch_action_network_odds() -> dict:
    """Fetch odds from Action Network's public scoreboard JSON API."""
    empty = {"win": {}, "top3": {}, "top5": {}, "top10": {}}

    url = "https://api.actionnetwork.com/web/v2/scoreboard/nascar_cup"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return empty

        data = r.json()
        comps = data.get("competitions", [])
        if not comps:
            return empty

        comp = comps[0]

        # Build player_id -> name map from competitors
        competitors = comp.get("competitors", [])
        pid_to_name = {}
        for c in competitors:
            player = c.get("player", {})
            pid = player.get("id") or c.get("player_id")
            name = player.get("full_name", "")
            if pid and name:
                pid_to_name[pid] = name

        # Extract odds from markets — prefer Consensus (book_id=15), fallback to any
        markets = comp.get("markets", {})
        win_result = {}
        top3_result = {}
        top5_result = {}
        top10_result = {}

        # Market key mapping from Action Network API
        # moneyline = win, core_bet_type_211_top_3, core_bet_type_3_top_5, core_bet_type_4_top_10
        market_targets = {
            "moneyline": win_result,
            "core_bet_type_211_top_3": top3_result,
            "top_3": top3_result,
            "core_bet_type_3_top_5": top5_result,
            "top_5": top5_result,
            "core_bet_type_4_top_10": top10_result,
            "top_10": top10_result,
        }

        for book_key, book_data in markets.items():
            event = book_data.get("event", {})
            if not isinstance(event, dict):
                continue

            for market_key, market_entries in event.items():
                if not isinstance(market_entries, list):
                    continue

                # Match market key to target dict
                mk_lower = market_key.lower().replace(" ", "_")
                target = None
                for pattern, tgt in market_targets.items():
                    if pattern in mk_lower:
                        target = tgt
                        break

                if target is None:
                    continue
                # Only fill if not already populated
                if target:
                    continue

                for entry in market_entries:
                    pid = entry.get("player_id")
                    odds_val = entry.get("odds")
                    name = pid_to_name.get(pid)
                    if name and odds_val is not None:
                        target[name] = str(odds_val)

        return {"win": win_result, "top3": top3_result, "top5": top5_result, "top10": top10_result}

    except Exception:
        return empty


def fetch_nascar_odds(series_id: int = 1) -> dict:
    """Fetch win odds from Action Network (Cup only).

    Returns {driver_name: odds_string}.
    """
    all_odds = _fetch_all_nascar_odds(series_id)
    return all_odds.get("win", {})


def fetch_nascar_prop_odds(series_id: int = 1) -> dict:
    """Fetch top3/top5/top10 prop odds from Action Network (Cup only).

    Returns {"top3": {name: odds_str}, "top5": {name: odds_str}, "top10": {name: odds_str}}.
    """
    all_odds = _fetch_all_nascar_odds(series_id)
    top3 = all_odds.get("top3", {})
    top5 = all_odds.get("top5", {})
    top10 = all_odds.get("top10", {})

    return {"top3": top3, "top5": top5, "top10": top10}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_odds_api_props() -> dict:
    """Fetch NASCAR top 5/top 10 finish odds from The Odds API.

    Requires ODDS_API_KEY set in Streamlit secrets or environment.
    Returns {"top5": {name: odds_str}, "top10": {name: odds_str}}.
    """
    from src.config import ODDS_API_KEY
    if not ODDS_API_KEY:
        return {"top5": {}, "top10": {}}

    base = "https://api.the-odds-api.com/v4"
    empty = {"top5": {}, "top10": {}}

    try:
        # First discover the NASCAR sport key
        sports_resp = requests.get(f"{base}/sports", params={"apiKey": ODDS_API_KEY}, timeout=10)
        if sports_resp.status_code != 200:
            return empty

        nascar_key = None
        for sport in sports_resp.json():
            key = sport.get("key", "")
            title = sport.get("title", "").lower()
            if "nascar" in key.lower() or "nascar" in title:
                nascar_key = key
                break

        if not nascar_key:
            return empty

        # Get events for NASCAR
        events_resp = requests.get(
            f"{base}/sports/{nascar_key}/events",
            params={"apiKey": ODDS_API_KEY},
            timeout=10,
        )
        if events_resp.status_code != 200 or not events_resp.json():
            return empty

        # Use the first (upcoming) event
        event_id = events_resp.json()[0].get("id")
        if not event_id:
            return empty

        # Try fetching outrights + top finish markets
        top5 = {}
        top10 = {}

        for market in ["outrights", "top_5_finish", "top_10_finish",
                        "top5", "top10", "driver_top_5", "driver_top_10"]:
            try:
                odds_resp = requests.get(
                    f"{base}/sports/{nascar_key}/events/{event_id}/odds",
                    params={
                        "apiKey": ODDS_API_KEY,
                        "regions": "us",
                        "markets": market,
                        "oddsFormat": "american",
                    },
                    timeout=10,
                )
                if odds_resp.status_code != 200:
                    continue

                data = odds_resp.json()
                bookmakers = data.get("bookmakers", [])
                if not bookmakers:
                    continue

                # Use first bookmaker's odds
                bm = bookmakers[0]
                for mkt in bm.get("markets", []):
                    mkt_key = mkt.get("key", "").lower()
                    outcomes = mkt.get("outcomes", [])

                    is_t5 = "top_5" in mkt_key or "top5" in mkt_key
                    is_t10 = "top_10" in mkt_key or "top10" in mkt_key

                    if is_t5 and not top5:
                        for o in outcomes:
                            name = o.get("name", "")
                            price = o.get("price")
                            if name and price is not None:
                                top5[name] = str(int(price))
                    elif is_t10 and not top10:
                        for o in outcomes:
                            name = o.get("name", "")
                            price = o.get("price")
                            if name and price is not None:
                                top10[name] = str(int(price))
            except Exception:
                continue

        return {"top5": top5, "top10": top10}

    except Exception:
        return {"top5": {}, "top10": {}}


def round_odds(odds_val: int) -> int:
    """Round American odds to clean increments for display.

    ≤500: nearest 50   (+150, +350, +500)
    501-2000: nearest 100  (+600, +1200)
    2001+: nearest 500  (+2500, +5000, +10000)
    """
    v = abs(odds_val)
    if v <= 500:
        v = round(v / 50) * 50
    elif v <= 2000:
        v = round(v / 100) * 100
    else:
        v = round(v / 500) * 500
    return max(v, 100) if odds_val >= 0 else -max(v, 100)


def estimate_odds_from_salaries(dk_df: pd.DataFrame) -> dict:
    """Derive estimated American win odds from DK salary as a fallback.

    Higher salary → lower (better) odds. Uses a log-linear mapping
    calibrated roughly to real NASCAR odds distributions:
    - $11,000+ salary ≈ +300 to +600 (favorites)
    - $8,000-$11,000 ≈ +800 to +2000
    - $6,000-$8,000 ≈ +2500 to +5000
    - Below $6,000 ≈ +6000 to +15000

    Returns dict {driver_name: odds_string} in the same format as fetch_nascar_odds.
    """
    if dk_df.empty or "DK Salary" not in dk_df.columns:
        return {}

    import math
    result = {}
    df = dk_df[dk_df["DK Salary"].notna()].copy()
    if df.empty:
        return {}

    max_sal = df["DK Salary"].max()
    min_sal = df["DK Salary"].min()
    if max_sal <= min_sal:
        return {}

    for _, row in df.iterrows():
        name = row.get("Driver", "")
        salary = row["DK Salary"]
        if not name or salary <= 0:
            continue

        # Normalized 0-1 (1 = highest salary)
        norm = (salary - min_sal) / (max_sal - min_sal)

        # Map to American odds: top salary ≈ +350, bottom ≈ +12000
        # Using exponential decay: odds = base * e^(-k * norm)
        odds = int(350 * math.exp(3.5 * (1 - norm)))
        odds = max(150, min(20000, odds))  # clamp
        odds = round_odds(odds)

        result[name] = f"+{odds}"

    return result


# ============================================================
# ODDS PERSISTENCE — save/load odds to/from database
# ============================================================

def _resolve_db_race_id(api_race_id: int, series_id: int = None):
    """Resolve a NASCAR API race_id to the internal DB race_id.

    Looks up via the api_race_id column on the races table.
    When series_id is provided, filters to that series (prevents cross-series leaks).
    Returns None if not found.
    """
    if not api_race_id or not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if series_id:
            row = conn.execute(
                "SELECT id FROM races WHERE api_race_id = ? AND series_id = ?",
                (api_race_id, series_id)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM races WHERE api_race_id = ?", (api_race_id,)
            ).fetchone()
    except Exception:
        # Column may not exist yet — add it
        try:
            conn.execute("ALTER TABLE races ADD COLUMN api_race_id INTEGER")
            conn.commit()
        except Exception:
            pass
        row = None
    conn.close()
    return row[0] if row else None


def _resolve_db_race_id_with_fallback(race_id: int, series_id: int = None):
    """Resolve API race_id to DB race_id, with date-based fallback and auto-create.

    If the race doesn't exist in the DB at all (common for upcoming races),
    creates a new entry from the API race list data so salaries/odds can be saved.
    """
    db_race_id = _resolve_db_race_id(race_id, series_id)
    if db_race_id:
        return db_race_id

    # Fallback: try matching by date from API, or create if not found
    try:
        from datetime import datetime
        current_year = datetime.now().year
        search_series = [series_id] if series_id else [1, 2, 3]
        # Try current year and previous year
        search_years = [current_year, current_year - 1]

        for year in search_years:
            for sid in search_series:
                api_url = f"{NASCAR_API_BASE}/{year}/{sid}/race_list_basic.json"
                resp = requests.get(api_url, timeout=10)
                if resp.status_code != 200:
                    continue
                for r in resp.json():
                    if r.get("race_id") == race_id:
                        race_date = (r.get("race_date") or "")[:10]
                        race_name = r.get("race_name", "")
                        track_id = r.get("track_id", 0)
                        scheduled_laps = r.get("scheduled_laps", 0)
                        scheduled_dist = r.get("scheduled_distance", 0)
                        race_num = r.get("race_season", year)

                        if not race_date:
                            break

                        conn_tmp = sqlite3.connect(str(DB_PATH))

                        # Try matching by date first
                        row = conn_tmp.execute(
                            "SELECT id FROM races WHERE race_date = ? AND series_id = ?",
                            (race_date, sid)
                        ).fetchone()
                        if row:
                            db_race_id = row[0]
                            conn_tmp.execute(
                                "UPDATE races SET api_race_id = ? WHERE id = ?",
                                (race_id, db_race_id)
                            )
                            conn_tmp.commit()
                        else:
                            # Create the race entry so odds/salaries can be saved
                            # Determine race_num (sequential within season)
                            max_num = conn_tmp.execute(
                                "SELECT COALESCE(MAX(race_num), 0) FROM races WHERE series_id = ? AND season = ?",
                                (sid, year)
                            ).fetchone()[0]
                            conn_tmp.execute('''
                                INSERT INTO races (series_id, track_id, season, race_num, race_name,
                                                   race_date, laps, miles, api_race_id)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (sid, track_id, year, max_num + 1, race_name,
                                  race_date, scheduled_laps, scheduled_dist, race_id))
                            conn_tmp.commit()
                            db_race_id = conn_tmp.execute(
                                "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
                            ).fetchone()[0]

                        conn_tmp.close()
                        break
                if db_race_id:
                    break
            if db_race_id:
                break
    except Exception:
        pass

    return db_race_id


def save_odds_to_db(odds_data: dict, race_id: int, sportsbook: str = "action_network",
                    top3_data: dict = None, top5_data: dict = None, top10_data: dict = None,
                    series_id: int = None):
    """Persist odds dict to the odds table, keyed by race_id.

    Smart update: only overwrites top3/top5/top10 odds when new valid data is provided.
    If prop data is empty/None, existing values are preserved.

    Args:
        odds_data: {driver_name: odds_string} e.g. {"Kyle Larson": "+350"}
        race_id: NASCAR API race ID (will be resolved to internal DB race_id)
        sportsbook: source label
        top3_data: {driver_name: odds_string} for top 3 finish odds
        top5_data: {driver_name: odds_string} for top 5 finish odds
        top10_data: {driver_name: odds_string} for top 10 finish odds
        series_id: series filter to prevent cross-series resolution
    """
    if not odds_data or not race_id or not DB_PATH.exists():
        return 0

    db_race_id = _resolve_db_race_id_with_fallback(race_id, series_id)
    if not db_race_id:
        return 0

    from src.utils import fuzzy_match_name

    conn = sqlite3.connect(str(DB_PATH))

    # Ensure top3_odds column exists (migration for older DBs)
    try:
        conn.execute("SELECT top3_odds FROM odds LIMIT 1")
    except Exception:
        try:
            conn.execute("ALTER TABLE odds ADD COLUMN top3_odds REAL")
            conn.commit()
        except Exception:
            pass

    # When importing from script, remove competing sportsbook entries
    # so the manually-loaded odds become the single source of truth
    if sportsbook in ("import", "bovada"):
        conn.execute(
            "DELETE FROM odds WHERE race_id = ? AND sportsbook IN ('action_network', 'auto')",
            (db_race_id,)
        )
        conn.commit()

    db_drivers = conn.execute("SELECT id, full_name FROM drivers").fetchall()
    name_to_id = {row[1]: row[0] for row in db_drivers}
    driver_names = list(name_to_id.keys())

    # Build top3/top5/top10 lookup by driver_id (fuzzy matched)
    from src.utils import parse_american_odds as _amer
    t3_by_id = {}
    t5_by_id = {}
    t10_by_id = {}
    for prop_data, target_dict in [(top3_data, t3_by_id), (top5_data, t5_by_id), (top10_data, t10_by_id)]:
        if not prop_data:
            continue
        for name, odds_str in prop_data.items():
            val = _amer(odds_str)
            if val is None:
                continue
            matched = fuzzy_match_name(name, driver_names)
            if matched:
                target_dict[name_to_id[matched]] = float(val)

    count = 0
    for name, odds_str in odds_data.items():
        odds_val = _amer(odds_str)
        if odds_val is None:
            continue
        odds_val = float(odds_val)

        matched = fuzzy_match_name(name, driver_names)
        if not matched:
            continue
        driver_id = name_to_id[matched]

        # Check if row exists with prop data we should preserve
        existing = conn.execute(
            "SELECT top3_odds, top5_odds, top10_odds FROM odds WHERE race_id = ? AND driver_id = ? AND sportsbook = ?",
            (db_race_id, driver_id, sportsbook)
        ).fetchone()

        # Use new prop data if available, else preserve existing
        t3_val = t3_by_id.get(driver_id)
        t5_val = t5_by_id.get(driver_id)
        t10_val = t10_by_id.get(driver_id)
        if existing:
            if t3_val is None and existing[0] is not None:
                t3_val = existing[0]
            if t5_val is None and existing[1] is not None:
                t5_val = existing[1]
            if t10_val is None and existing[2] is not None:
                t10_val = existing[2]

        conn.execute('''
            INSERT OR REPLACE INTO odds
            (race_id, driver_id, sportsbook, win_odds, top3_odds, top5_odds, top10_odds, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (db_race_id, driver_id, sportsbook, odds_val, t3_val, t5_val, t10_val))
        count += 1

    conn.commit()
    conn.close()
    return count


def load_race_odds(race_id: int, series_id: int = None) -> dict:
    """Load saved odds for a race from the DB.

    Args:
        race_id: NASCAR API race ID (will be resolved to internal DB race_id)
        series_id: series filter to prevent cross-series resolution

    Returns dict of {driver_name: odds_string} matching the live format.
    """
    if not DB_PATH.exists():
        return {}

    db_race_id = _resolve_db_race_id(race_id, series_id)
    if not db_race_id:
        return {}

    conn = sqlite3.connect(str(DB_PATH))
    # Sportsbook priority: import (script import) > manual > auto > action_network
    # "bovada" kept as alias for historical DB rows
    rows = conn.execute('''
        SELECT d.full_name, o.win_odds, o.sportsbook
        FROM odds o
        JOIN drivers d ON d.id = o.driver_id
        WHERE o.race_id = ?
        ORDER BY o.win_odds ASC
    ''', (db_race_id,)).fetchall()
    conn.close()

    SPORTSBOOK_PRIORITY = {"import": 0, "bovada": 0, "manual": 1, "csv_import": 2,
                           "sportsbettingdime": 3, "auto": 4, "action_network": 5}

    result = {}
    result_priority = {}
    for name, odds_val, sb in rows:
        if odds_val is not None:
            prio = SPORTSBOOK_PRIORITY.get(sb, 99)
            if name not in result or prio < result_priority[name]:
                ov = int(odds_val) if odds_val == int(odds_val) else odds_val
                result[name] = f"+{ov}" if ov > 0 else str(ov)
                result_priority[name] = prio
    return result


def clear_race_odds(race_id: int, series_id: int = None,
                    sportsbook: str = None) -> int:
    """Wipe all odds rows for a race. Returns number of rows deleted.

    Used by the import script's "Clear odds" menu and by the import flow's
    "Overwrite?" path — without an explicit clear, re-importing odds for a
    race only updates rows for drivers that appear in the new dataset and
    leaves stale rows for drivers that don't (e.g. when Cup odds were
    accidentally saved to an O'Reilly race).

    Args:
        race_id: NASCAR API race ID (resolved to internal DB id)
        series_id: series filter to prevent cross-series resolution
        sportsbook: optional — only delete rows from this sportsbook source.
                    None (default) deletes ALL odds rows for the race.
    """
    if not DB_PATH.exists() or not race_id:
        return 0
    db_race_id = _resolve_db_race_id_with_fallback(race_id, series_id)
    if not db_race_id:
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    try:
        if sportsbook:
            cur = conn.execute(
                "DELETE FROM odds WHERE race_id = ? AND sportsbook = ?",
                (db_race_id, sportsbook),
            )
        else:
            cur = conn.execute(
                "DELETE FROM odds WHERE race_id = ?",
                (db_race_id,),
            )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()


def load_race_prop_odds(race_id: int, series_id: int = None) -> dict:
    """Load top3/top5/top10 finish odds for a race from the DB.

    Args:
        race_id: NASCAR API race ID
        series_id: series filter to prevent cross-series resolution

    Returns {"top3": {name: odds_int}, "top5": {name: odds_int}, "top10": {name: odds_int}}.
    """
    if not DB_PATH.exists():
        return {"top3": {}, "top5": {}, "top10": {}}

    db_race_id = _resolve_db_race_id(race_id, series_id)
    if not db_race_id:
        return {"top3": {}, "top5": {}, "top10": {}}

    conn = sqlite3.connect(str(DB_PATH))
    # Gracefully handle missing top3_odds column in older DBs
    # Include sportsbook for priority-based dedup
    try:
        rows = conn.execute('''
            SELECT d.full_name, o.top3_odds, o.top5_odds, o.top10_odds, o.sportsbook
            FROM odds o
            JOIN drivers d ON d.id = o.driver_id
            WHERE o.race_id = ?
        ''', (db_race_id,)).fetchall()
    except Exception:
        rows = conn.execute('''
            SELECT d.full_name, NULL, o.top5_odds, o.top10_odds, o.sportsbook
            FROM odds o
            JOIN drivers d ON d.id = o.driver_id
            WHERE o.race_id = ?
        ''', (db_race_id,)).fetchall()
    conn.close()

    SPORTSBOOK_PRIORITY = {"import": 0, "bovada": 0, "manual": 1, "csv_import": 2,
                           "sportsbettingdime": 3, "auto": 4, "action_network": 5}

    top3 = {}
    top5 = {}
    top10 = {}
    top3_prio = {}
    top5_prio = {}
    top10_prio = {}
    for name, t3, t5, t10, sb in rows:
        prio = SPORTSBOOK_PRIORITY.get(sb, 99)
        if t3 is not None and (name not in top3 or prio < top3_prio[name]):
            ov = int(t3) if t3 == int(t3) else t3
            top3[name] = ov
            top3_prio[name] = prio
        if t5 is not None and (name not in top5 or prio < top5_prio[name]):
            ov = int(t5) if t5 == int(t5) else t5
            top5[name] = ov
            top5_prio[name] = prio
        if t10 is not None and (name not in top10 or prio < top10_prio[name]):
            ov = int(t10) if t10 == int(t10) else t10
            top10[name] = ov
            top10_prio[name] = prio
    return {"top3": top3, "top5": top5, "top10": top10}


# ============================================================
# AUTO-FETCH RACE RESULTS INTO DATABASE
# ============================================================

def _fetch_and_store_via_loopstats(series_id: int, race_id: int, year: int) -> dict:
    """Fallback ingestion path using NASCAR's loopstats + lap-times endpoints.

    The regular weekend-feed.json endpoint is sometimes empty for past races
    (e.g. postponed playoff races like the 2025 YellaWood 500 rescheduled
    from Oct 19 to Nov 17). In those cases the loopstats endpoint at
        https://cf.nascar.com/loopstats/prod/{year}/{series}/{race_id}.json
    still has full finishing data, and lap-times.json has the driver-name
    to NASCAR-driver-id mapping we need.

    Translates NASCAR's internal track_id to our DB's track_id by matching
    track name (our track IDs are assigned locally, not from the API).

    Returns the same dict shape as fetch_and_store_race.
    """
    from src.utils import normalize_driver_name, calc_dk_points, calc_fd_points
    import requests

    # 1. Race metadata from race_list_basic
    rlist = requests.get(
        f"{NASCAR_API_BASE}/{year}/{series_id}/race_list_basic.json", timeout=15
    ).json()
    meta = next((r for r in rlist if r.get("race_id") == race_id), None)
    if not meta:
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "race not in race_list_basic"}

    # 2. Driver stats from loopstats
    loop_resp = requests.get(
        f"https://cf.nascar.com/loopstats/prod/{year}/{series_id}/{race_id}.json",
        timeout=15,
    )
    if loop_resp.status_code != 200:
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": f"loopstats unavailable ({loop_resp.status_code})"}
    loop = loop_resp.json()
    driver_stats = loop[0].get("drivers", []) if isinstance(loop, list) and loop else []
    if not driver_stats:
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "no driver stats in loopstats"}

    # 3. Driver names from lap-times (keyed by NASCARDriverID)
    laps_resp = requests.get(
        f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/lap-times.json", timeout=30
    )
    lap_data = laps_resp.json() if laps_resp.status_code == 200 else {}
    id_to_name = {}
    for drv in lap_data.get("laps", []):
        nid = drv.get("NASCARDriverID")
        full = drv.get("FullName", "")
        if nid and full:
            id_to_name[int(nid)] = _clean_api_name(full)

    if not id_to_name:
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "could not resolve driver names from lap-times"}

    # 4. Resolve track_id by NAME (NASCAR API's internal track IDs differ from ours)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    api_track_name = (meta.get("track_name") or "").strip()
    track_row = conn.execute(
        "SELECT id FROM tracks WHERE LOWER(name) = LOWER(?)", (api_track_name,)
    ).fetchone()
    if not track_row:
        conn.execute(
            "INSERT INTO tracks (name) VALUES (?)", (api_track_name,)
        )
        track_row = conn.execute(
            "SELECT id FROM tracks WHERE LOWER(name) = LOWER(?)", (api_track_name,)
        ).fetchone()
    track_id = track_row["id"]

    # 5. Insert/lookup race row
    race_date = (meta.get("race_date") or "")[:10]
    race_name = meta.get("race_name", "")
    scheduled_laps = meta.get("scheduled_laps", 0)
    scheduled_dist = meta.get("scheduled_distance", 0)
    existing_race = conn.execute(
        "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
    ).fetchone()
    if existing_race:
        db_race_id = existing_race["id"]
        # Ensure track_id is correct in case a prior insert got it wrong
        conn.execute(
            "UPDATE races SET track_id = ?, race_date = ?, race_name = ? WHERE id = ?",
            (track_id, race_date, race_name, db_race_id)
        )
    else:
        max_num = conn.execute(
            "SELECT COALESCE(MAX(race_num), 0) FROM races WHERE series_id = ? AND season = ?",
            (series_id, year)
        ).fetchone()[0]
        conn.execute('''
            INSERT INTO races (series_id, track_id, season, race_num, race_name,
                                race_date, laps, miles, api_race_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (series_id, track_id, year, max_num + 1, race_name, race_date,
               scheduled_laps, scheduled_dist, race_id))
        db_race_id = conn.execute(
            "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
        ).fetchone()["id"]

    # 6. Insert race_results + dfs_points per driver
    count = 0
    for d in driver_stats:
        nid = d.get("driver_id")
        driver_name = id_to_name.get(nid)
        if not driver_name:
            continue

        # Find or create driver (normalized lookup first)
        drv = conn.execute(
            "SELECT id FROM drivers WHERE full_name = ?", (driver_name,)
        ).fetchone()
        if not drv:
            _norm = normalize_driver_name(driver_name)
            for row in conn.execute("SELECT id, full_name FROM drivers").fetchall():
                if normalize_driver_name(row["full_name"]) == _norm:
                    drv = {"id": row["id"]}
                    break
        if not drv:
            conn.execute("INSERT INTO drivers (full_name) VALUES (?)", (driver_name,))
            drv = conn.execute(
                "SELECT id FROM drivers WHERE full_name = ?", (driver_name,)
            ).fetchone()
        driver_id = drv["id"]

        finish_pos = d.get("ps", 0) or 0
        start_pos = d.get("start_ps", 0) or 0
        laps_led = d.get("lead_laps", 0) or 0
        fastest = d.get("fast_laps", 0) or 0
        laps_completed = d.get("laps", 0) or 0
        arp = d.get("avg_ps")

        # Loopstats doesn't include team name — infer from driver's nearest
        # same-season race. Without this, team ends up NULL and team-based
        # signals (team quality, team adjustment) can't use this race.
        team_row = conn.execute('''
            SELECT rr2.team FROM race_results rr2 JOIN races r2 ON r2.id = rr2.race_id
            WHERE rr2.driver_id = ? AND r2.season = ?
              AND rr2.team IS NOT NULL AND rr2.team != '' AND rr2.team != 'None'
            ORDER BY ABS(JULIANDAY(r2.race_date) - JULIANDAY(?)) ASC LIMIT 1
        ''', (driver_id, year, race_date)).fetchone()
        team_name = team_row["team"] if team_row else None

        conn.execute('''
            INSERT INTO race_results (race_id, driver_id, start_pos, finish_pos,
                                       laps_completed, laps_led, fastest_laps,
                                       avg_running_position, team, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Running')
            ON CONFLICT(race_id, driver_id) DO UPDATE SET
              start_pos=excluded.start_pos, finish_pos=excluded.finish_pos,
              laps_completed=excluded.laps_completed, laps_led=excluded.laps_led,
              fastest_laps=excluded.fastest_laps,
              avg_running_position=excluded.avg_running_position,
              team=COALESCE(excluded.team, race_results.team)
        ''', (db_race_id, driver_id, start_pos, finish_pos, laps_completed,
              laps_led, fastest, arp, team_name))

        dk = calc_dk_points(finish_pos, start_pos, laps_led, fastest)
        fd = calc_fd_points(finish_pos, start_pos, laps_led, fastest)
        for platform, score in [("DraftKings", dk), ("FanDuel", fd)]:
            conn.execute('''
                INSERT INTO dfs_points (race_id, driver_id, platform, dfs_score)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(race_id, driver_id, platform) DO UPDATE SET dfs_score=excluded.dfs_score
            ''', (db_race_id, driver_id, platform, score))
        count += 1

    conn.commit()
    conn.close()
    return {"drivers": count, "race_name": race_name, "status": "success",
            "error": None}


def fetch_and_store_race(series_id: int, race_id: int, year: int = None) -> dict:
    """Fetch race results + lap times from the NASCAR API and populate the DB.

    Steps:
        1. Fetch weekend feed and lap times
        2. Extract race results and fastest-lap counts
        3. Find/create the race row in the races table
        4. Find/create each driver
        5. Upsert race_results rows
        6. Compute and upsert DFS points for DraftKings and FanDuel

    Returns dict with {"drivers": int, "race_name": str, "status": str, "error": str|None}.
    """
    from src.config import DB_PATH, TRACK_TYPE_MAP

    if year is None:
        year = _default_active_year()

    # -- 1. Fetch data from API ----------------------------------------
    # Use __wrapped__ to bypass st.cache_data when available, else call directly
    _fetch_feed = getattr(fetch_weekend_feed, "__wrapped__", fetch_weekend_feed)
    _fetch_laps = getattr(fetch_lap_times, "__wrapped__", fetch_lap_times)

    feed = _fetch_feed(series_id, race_id, year)
    if feed is None:
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "Could not fetch weekend feed"}

    lap_data = _fetch_laps(series_id, race_id, year)

    # -- 2. Extract results and fastest laps ---------------------------
    races = feed.get("weekend_race", [])
    if not races:
        # NASCAR's weekend-feed endpoint is sometimes empty for races
        # (notably post-Chase playoff races that get rescheduled). Fall back
        # to the loopstats endpoint which has the same finishing data.
        try:
            fallback = _fetch_and_store_via_loopstats(series_id, race_id, year)
            if fallback.get("status") == "success":
                return fallback
        except Exception as _e:
            pass
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "No weekend_race in feed"}

    race_obj = races[0]
    results = race_obj.get("results", [])
    if not results:
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "No results in weekend_race"}

    # Check that the race is actually complete (not all finishing_position == 0)
    if all(r.get("finishing_position", 0) == 0 for r in results):
        return {"drivers": 0, "race_name": "", "status": "error",
                "error": "Race not yet completed (all positions are 0)"}

    # Race metadata
    race_name = race_obj.get("race_name", "") or ""
    track_name = race_obj.get("track_name", "") or ""
    race_date = race_obj.get("race_date") or race_obj.get("date_scheduled") or ""
    race_num = race_obj.get("race_season", 0) or race_obj.get("race_id", race_id)
    total_laps = race_obj.get("number_of_laps") or race_obj.get("laps") or 0

    # Fastest laps + avg running position from lap-times endpoint
    fastest_laps_map = {}
    arp_map = {}
    if lap_data:
        fastest_laps_map = compute_fastest_laps(lap_data)
        arp_map = compute_avg_running_position(lap_data)

    # -- 3-7. Database operations ---------------------------------------
    if not DB_PATH.exists():
        return {"drivers": 0, "race_name": race_name, "status": "error",
                "error": f"Database not found at {DB_PATH}"}

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    try:
        # --- Find or create track ---
        track_row = conn.execute(
            "SELECT id, track_type FROM tracks WHERE name = ?", (track_name,)
        ).fetchone()

        track_id = None
        track_type = TRACK_TYPE_MAP.get(track_name, "")
        if track_row:
            track_id = track_row["id"]
            track_type = track_row["track_type"] or track_type
        elif track_name:
            conn.execute(
                "INSERT OR IGNORE INTO tracks (name, track_type) VALUES (?, ?)",
                (track_name, track_type),
            )
            row = conn.execute(
                "SELECT id FROM tracks WHERE name = ?", (track_name,)
            ).fetchone()
            if row:
                track_id = row["id"]

        # --- Find or create race ---
        # First try by api_race_id (most reliable), then by series/season/race_num
        race_row = conn.execute(
            "SELECT id FROM races WHERE api_race_id = ? AND series_id = ?",
            (race_id, series_id),
        ).fetchone()
        if not race_row:
            race_row = conn.execute(
                "SELECT id FROM races WHERE series_id = ? AND season = ? AND race_num = ?",
                (series_id, year, race_num),
            ).fetchone()

        if race_row:
            db_race_id = race_row["id"]
            # Update metadata in case it was a placeholder
            conn.execute(
                "UPDATE races SET race_name=?, race_date=?, track_id=?, laps=?, api_race_id=? WHERE id=?",
                (race_name, race_date, track_id, total_laps, race_id, db_race_id),
            )
        else:
            conn.execute(
                """INSERT INTO races (series_id, track_id, season, race_num,
                   race_name, race_date, laps, api_race_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (series_id, track_id, year, race_num, race_name, race_date, total_laps, race_id),
            )
            db_race_id = conn.execute(
                "SELECT id FROM races WHERE series_id=? AND season=? AND race_num=?",
                (series_id, year, race_num),
            ).fetchone()["id"]

        # --- Process each driver result --------------------------------
        driver_count = 0

        for r in results:
            finish_pos = r.get("finishing_position", 0) or 0
            driver_name = _clean_api_name(r.get("driver_fullname") or "")
            if not driver_name or (finish_pos == 0 and not (r.get("finishing_status") or "").strip()):
                continue  # skip DNQ / empty rows

            start_pos = r.get("starting_position") or r.get("qualifying_position") or 0
            car_number = str(r.get("car_number", "")).strip()
            team = r.get("team_name", "")
            manufacturer = r.get("car_make", "")
            laps_completed = r.get("laps_completed") or 0
            laps_led = r.get("laps_led", 0) or 0
            status = r.get("finishing_status") or "Running"
            fastest = fastest_laps_map.get(driver_name)
            if fastest is None:
                # Fuzzy fallback: last name + first initial match
                _last = driver_name.split()[-1].lower() if driver_name.split() else ""
                _first = driver_name.split()[0][0].lower() if driver_name.split() and driver_name.split()[0] else ""
                for fl_name, fl_val in fastest_laps_map.items():
                    fl_parts = fl_name.split()
                    if (fl_parts and fl_parts[-1].lower() == _last
                            and fl_parts[0][0].lower() == _first):
                        fastest = fl_val
                        break
            fastest = fastest or 0

            # Find or create driver — normalized lookup prevents duplicates
            # from name variations (A.J./AJ, Suárez/Suarez, Jr./Jr, etc.).
            # See src.utils.normalize_driver_name for the normalization rules.
            from src.utils import normalize_driver_name as _norm
            _norm_key = _norm(driver_name)
            d = conn.execute(
                "SELECT id FROM drivers WHERE full_name = ?", (driver_name,)
            ).fetchone()
            if not d and _norm_key:
                # Scan existing drivers for a normalized match
                existing = conn.execute("SELECT id, full_name FROM drivers").fetchall()
                for row in existing:
                    if _norm(row["full_name"]) == _norm_key:
                        d = {"id": row["id"]}
                        break
            if not d:
                conn.execute("INSERT INTO drivers (full_name) VALUES (?)", (driver_name,))
                d = conn.execute(
                    "SELECT id FROM drivers WHERE full_name = ?", (driver_name,)
                ).fetchone()
            driver_id = d["id"]

            # Look up ARP — try direct then normalized match against the
            # lap-times names (which can differ from results feed names)
            arp_val = arp_map.get(driver_name) if arp_map else None
            if arp_val is None and arp_map:
                _nkey = _norm(driver_name)
                for an, av in arp_map.items():
                    if _norm(an) == _nkey:
                        arp_val = av
                        break

            # Upsert race_results
            conn.execute(
                """INSERT INTO race_results
                   (race_id, driver_id, car_number, team, manufacturer,
                    start_pos, finish_pos, laps_completed, laps_led,
                    fastest_laps, avg_running_position, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(race_id, driver_id) DO UPDATE SET
                    car_number=excluded.car_number, team=excluded.team,
                    manufacturer=excluded.manufacturer, start_pos=excluded.start_pos,
                    finish_pos=excluded.finish_pos, laps_completed=excluded.laps_completed,
                    laps_led=excluded.laps_led, fastest_laps=excluded.fastest_laps,
                    avg_running_position=excluded.avg_running_position,
                    status=excluded.status""",
                (db_race_id, driver_id, car_number, team, manufacturer,
                 start_pos, finish_pos, laps_completed, laps_led, fastest, arp_val, status),
            )

            # Compute and upsert DFS points for both platforms
            dk_score = calc_dk_points(finish_pos, start_pos, laps_led, fastest)
            fd_score = calc_fd_points(finish_pos, start_pos, laps_led, fastest)

            for platform, score in [("DraftKings", dk_score), ("FanDuel", fd_score)]:
                conn.execute(
                    """INSERT INTO dfs_points
                       (race_id, driver_id, platform, dfs_score)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(race_id, driver_id, platform) DO UPDATE SET
                        dfs_score=excluded.dfs_score""",
                    (db_race_id, driver_id, platform, score),
                )

            driver_count += 1

        conn.commit()
        return {"drivers": driver_count, "race_name": race_name, "status": "success",
                "error": None}

    except Exception as e:
        conn.rollback()
        return {"drivers": 0, "race_name": race_name, "status": "error",
                "error": str(e)}
    finally:
        conn.close()


# ============================================================
# CSV PARSING
# ============================================================

def _read_csv_with_fallback(file) -> pd.DataFrame:
    """Read a CSV trying multiple encodings (DK/FD exports often ship as cp1252
    when the race name contains characters like 'ü' in 'Würth')."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            if hasattr(file, "seek"):
                file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            # Non-encoding errors: re-raise so caller's except still catches
            raise
    return pd.DataFrame()


def parse_dk_csv(file) -> pd.DataFrame:
    """Parse DraftKings CSV export."""
    try:
        df = _read_csv_with_fallback(file)
        if "Name" in df.columns and "Salary" in df.columns:
            result = df[["Name", "Salary"]].copy()
            result.columns = ["Driver", "DK Salary"]
            result["Driver"] = result["Driver"].str.strip()
            return result
    except Exception:
        pass
    return pd.DataFrame()


def parse_fd_csv(file) -> pd.DataFrame:
    """Parse FanDuel CSV export."""
    try:
        df = _read_csv_with_fallback(file)
        name_col = next((c for c in df.columns if c.lower() in ["nickname", "name", "player"]), None)
        sal_col = next((c for c in df.columns if c.lower() == "salary"), None)
        if name_col and sal_col:
            result = df[[name_col, sal_col]].copy()
            result.columns = ["Driver", "FD Salary"]
            result["Driver"] = result["Driver"].str.strip()
            return result
    except Exception:
        pass
    return pd.DataFrame()


# ============================================================
# RACE LIST HELPERS
# ============================================================

def filter_point_races(races: list) -> list:
    """Filter out exhibition races (Clash, Duels, All-Star, etc.)."""
    point_races = [r for r in races
                   if not any(kw in r.get("race_name", "").lower() for kw in EXHIBITION_KEYWORDS)]
    return point_races if point_races else races


def detect_prerace(feed: dict) -> bool:
    """Detect whether a race is pre-race or post-race."""
    if not feed:
        return True
    races = feed.get("weekend_race", [])
    if not races:
        return True
    results = races[0].get("results", [])
    if not results:
        return True
    # All finishing_position=0 means pre-race
    return all(r.get("finishing_position", 0) == 0 for r in results)


def query_lapping_profile(track_type: str, series_id: int = 1,
                          min_season: int = 2022) -> dict:
    """Build a lapping profile for a track type from historical data.

    Returns dict with finish-position buckets:
        {
            "p1_10":  {"avg_laps_down": 0.1, "pct_lapped": 8, "n": 80},
            "p11_20": {"avg_laps_down": 1.1, "pct_lapped": 69, "n": 80},
            "p21_30": {"avg_laps_down": 8.9, "pct_lapped": 84, "n": 80},
            "p31_plus": {"avg_laps_down": 108.2, "pct_lapped": 100, "n": 56},
        }
    """
    from src.config import DB_PATH, TRACK_TYPE_MAP
    if not DB_PATH.exists():
        return {}

    # Collect track names for this track type
    tracks = [t for t, tt in TRACK_TYPE_MAP.items() if tt == track_type]
    if not tracks:
        return {}

    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join("?" for _ in tracks)
    rows = conn.execute(f'''
        SELECT r.id as race_id, rr.finish_pos, rr.laps_completed, rr.status
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE t.name IN ({placeholders})
          AND r.series_id = ? AND r.season >= ?
        ORDER BY r.id, rr.finish_pos
    ''', [*tracks, series_id, min_season]).fetchall()
    conn.close()

    if not rows:
        return {}

    # Group by race, compute laps down relative to leader
    from collections import defaultdict
    races = defaultdict(list)
    for race_id, finish_pos, laps_completed, status in rows:
        races[race_id].append((finish_pos, laps_completed, status))

    buckets = {"p1_10": [], "p11_20": [], "p21_30": [], "p31_plus": []}
    for race_drivers in races.values():
        max_laps = max(lc for _, lc, _ in race_drivers)
        if max_laps == 0:
            continue
        for fp, lc, status in race_drivers:
            down = max_laps - lc
            if fp <= 10:
                buckets["p1_10"].append(down)
            elif fp <= 20:
                buckets["p11_20"].append(down)
            elif fp <= 30:
                buckets["p21_30"].append(down)
            else:
                buckets["p31_plus"].append(down)

    result = {}
    for bkt, vals in buckets.items():
        if vals:
            result[bkt] = {
                "avg_laps_down": round(sum(vals) / len(vals), 1),
                "pct_lapped": round(sum(1 for v in vals if v > 0) / len(vals) * 100),
                "n": len(vals),
            }
    return result


# ---------------------------------------------------------------------------
# Season Standings (computed from weekend-feed race results)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_season_standings(series_id: int, year: int = None, _v: int = 2) -> dict:
    """Fetch season standings by aggregating results from all completed races.

    Returns dict with keys: "driver", "manufacturer", "owner", each a DataFrame.
    Uses points_earned from the API (includes stage points in total).
    """
    if year is None:
        year = _default_active_year()
    races = fetch_race_list(series_id, year)
    if not races:
        return {"driver": pd.DataFrame(), "manufacturer": pd.DataFrame(), "owner": pd.DataFrame()}

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    # Filter to completed points races (race_type_id == 1, date <= today)
    points_races = [
        r for r in races
        if r.get("race_type_id") == 1
        and (r.get("race_date") or r.get("date_scheduled", ""))[:10] <= today
    ]

    driver_rows = []
    mfr_agg = defaultdict(lambda: {"points": 0, "wins": 0, "races": 0})
    owner_agg = defaultdict(lambda: {"points": 0, "wins": 0, "top5": 0, "top10": 0,
                                      "races": 0, "car_number": "", "driver": ""})

    # Pre-build race_id → track_name map from race list (reliable fallback)
    track_map = {r["race_id"]: r.get("track_name", "") for r in races}

    for race in points_races:
        rid = race["race_id"]
        feed = fetch_weekend_feed(series_id, rid, year)
        if not feed:
            continue
        race_list = feed.get("weekend_race", [])
        if not race_list:
            continue
        race_data = race_list[0]
        results = race_data.get("results", [])
        race_name = race_data.get("race_name", "")
        track_name = race_data.get("track_name") or track_map.get(rid, "")

        # Stage results for this race
        stage_results = race_data.get("stage_results", [])
        # Build driver → stage points map
        driver_stage_pts = defaultdict(int)
        for stage in stage_results:
            for sr in stage.get("results", []):
                name = _clean_api_name(sr.get("driver_fullname", ""))
                driver_stage_pts[name] += sr.get("stage_points", 0)

        for r in results:
            fp = r.get("finishing_position", 0) or 0
            if fp == 0:
                continue
            driver = _clean_api_name(r.get("driver_fullname", ""))
            pts = r.get("points_earned", 0) or 0
            playoff_pts = r.get("playoff_points_earned", 0) or 0
            stage_pts = driver_stage_pts.get(driver, 0)
            mfr = r.get("car_make", "")
            team = r.get("team_name", "")
            owner = r.get("owner_fullname", "")
            car = r.get("car_number", "")

            driver_rows.append({
                "Driver": driver,
                "Race": race_name,
                "Track": track_name,
                "Finish": fp,
                "Start": r.get("starting_position", 0) or 0,
                "Points": pts,
                "Stage Pts": stage_pts,
                "Playoff Pts": playoff_pts,
                "Laps Led": r.get("laps_led", 0) or 0,
                "Status": r.get("finishing_status", "Running"),
                "Car": car,
                "Team": team,
                "Manufacturer": mfr,
            })

            # Manufacturer aggregate
            mfr_agg[mfr]["points"] += pts
            mfr_agg[mfr]["races"] += 1
            if fp == 1:
                mfr_agg[mfr]["wins"] += 1

            # Owner aggregate
            owner_key = f"{owner} (#{car})" if car else owner
            owner_agg[owner_key]["points"] += pts
            owner_agg[owner_key]["wins"] += (1 if fp == 1 else 0)
            owner_agg[owner_key]["top5"] += (1 if fp <= 5 else 0)
            owner_agg[owner_key]["top10"] += (1 if fp <= 10 else 0)
            owner_agg[owner_key]["races"] += 1
            owner_agg[owner_key]["car_number"] = car
            owner_agg[owner_key]["driver"] = driver

    # --- Build driver standings DataFrame ---
    if driver_rows:
        all_df = pd.DataFrame(driver_rows)
        driver_standings = all_df.groupby("Driver").agg(
            Points=("Points", "sum"),
            Wins=("Finish", lambda x: (x == 1).sum()),
            **{"Top 5": ("Finish", lambda x: (x <= 5).sum())},
            **{"Top 10": ("Finish", lambda x: (x <= 10).sum())},
            **{"Stage Pts": ("Stage Pts", "sum")},
            **{"Playoff Pts": ("Playoff Pts", "sum")},
            **{"Avg Finish": ("Finish", "mean")},
            **{"Laps Led": ("Laps Led", "sum")},
            Races=("Finish", "count"),
            **{"Best Finish": ("Finish", "min")},
        ).reset_index()
        driver_standings = driver_standings.sort_values("Points", ascending=False).reset_index(drop=True)
        driver_standings.insert(0, "Rank", range(1, len(driver_standings) + 1))
        driver_standings["Avg Finish"] = driver_standings["Avg Finish"].round(1)
    else:
        driver_standings = pd.DataFrame()

    # --- Manufacturer standings ---
    if mfr_agg:
        mfr_rows = [{"Manufacturer": m, "Points": d["points"], "Wins": d["wins"],
                      "Races": d["races"]} for m, d in mfr_agg.items()]
        mfr_standings = pd.DataFrame(mfr_rows).sort_values("Points", ascending=False).reset_index(drop=True)
        mfr_standings.insert(0, "Rank", range(1, len(mfr_standings) + 1))
    else:
        mfr_standings = pd.DataFrame()

    # --- Owner standings ---
    if owner_agg:
        owner_rows = [{"Owner": o, "Car": d["car_number"], "Driver": d["driver"],
                        "Points": d["points"], "Wins": d["wins"],
                        "Top 5": d["top5"], "Top 10": d["top10"],
                        "Races": d["races"]} for o, d in owner_agg.items()]
        owner_standings = pd.DataFrame(owner_rows).sort_values("Points", ascending=False).reset_index(drop=True)
        owner_standings.insert(0, "Rank", range(1, len(owner_standings) + 1))
    else:
        owner_standings = pd.DataFrame()

    return {
        "driver": driver_standings,
        "manufacturer": mfr_standings,
        "owner": owner_standings,
        "races": pd.DataFrame(driver_rows) if driver_rows else pd.DataFrame(),
    }
