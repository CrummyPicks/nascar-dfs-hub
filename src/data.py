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
from src.utils import int_col, calc_dk_points, calc_fd_points
import re


def _clean_api_name(name: str) -> str:
    """Normalize a driver name from any NASCAR API endpoint.

    Strips rookie indicators (*), charter indicators (#), suffixes like (i)/(R),
    and normalizes Jr./Sr. suffixes to consistent forms without trailing periods.
    """
    if not name:
        return ""
    name = name.strip()
    name = re.sub(r'^\*\s*', '', name)        # leading asterisk (rookie)
    name = re.sub(r'\s*#$', '', name)          # trailing # (charter)
    name = re.sub(r'\s*\([a-zA-Z]\)$', '', name)  # trailing (i)/(R)
    # Normalize "Jr." -> "Jr", "Sr." -> "Sr" (remove trailing period on suffixes)
    name = re.sub(r'\bJr\.\s*$', 'Jr', name)
    name = re.sub(r'\bSr\.\s*$', 'Sr', name)
    return name.strip()


# ============================================================
# NASCAR API FETCHES
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_race_list(series_id: int, year: int = 2026) -> list:
    """Fetch race list from NASCAR API."""
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/race_list_basic.json", timeout=15)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weekend_feed(series_id: int, race_id: int, year: int = 2026) -> Optional[dict]:
    """Fetch weekend feed (entry list, qualifying, practice, results)."""
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/weekend-feed.json", timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_lap_times(series_id: int, race_id: int, year: int = 2026) -> Optional[dict]:
    """Fetch lap-by-lap timing data."""
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/lap-times.json", timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_lap_averages(series_id: int, race_id: int, year: int = 2026) -> pd.DataFrame:
    """Fetch practice lap averages (Overall, 5/10/15/20/25/30 lap consecutive averages)."""
    try:
        r = requests.get(f"{NASCAR_API_BASE}/{year}/{series_id}/{race_id}/lap-averages.json", timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        session = data[-1]  # Last practice session
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
        # Convert rank columns to integers
        for rc in [c for c in df.columns if "Rank" in c]:
            df[rc] = int_col(df[rc])
        if "Overall Rank" in df.columns:
            df = df.sort_values("Overall Rank", na_position="last").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


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
        "Worst Finish", "Avg Rating", "Detail",
    ]
    col_names_13_alltime = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Laps Led", "Avg Start", "Best Finish",
        "Worst Finish", "DNF", "Detail",
    ]
    col_names_14_recent = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Top 20", "Laps Led", "Avg Start", "Best Finish",
        "DNF", "Avg Rating", "Detail",
    ]
    col_names_10_alltime = [
        "Rank", "Driver", "Wins", "Races", "Avg Finish",
        "Top 5", "Top 10", "Avg Start", "Best Finish", "DNF",
    ]
    col_names_15 = [
        "Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
        "Top 10", "Top 20", "Laps Led", "Avg Start", "Best Finish",
        "Worst Finish", "DNF", "Avg Rating", "Detail",
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


@st.cache_data(ttl=7200, show_spinner=False)
def scrape_track_history(track_name: str, series_id: int = 1) -> pd.DataFrame:
    """Scrape driveraverages.com for recent track history."""
    recent, _ = _scrape_da_tables(track_name, series_id)
    return recent


@st.cache_data(ttl=7200, show_spinner=False)
def scrape_track_history_alltime(track_name: str, series_id: int = 1) -> pd.DataFrame:
    """Scrape driveraverages.com for all-time track history."""
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


def query_track_type_stats(track_type: str, season: int = None) -> pd.DataFrame:
    """Pull stats filtered by track type from local DB.

    Handles both DB-level types (short, intermediate, road, superspeedway, dirt)
    and config-level subtypes (short_concrete, intermediate_worn) by:
    - For subtypes: filtering by track names belonging to that subtype
    - For parent groups ("All Short"): including all tracks of that parent type
    - For base types: querying by DB track_type directly
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
                   ROUND(AVG(dp.dfs_score),1) as "Avg DFS",
                   ROUND(MAX(dp.dfs_score),1) as "Best DFS",
                   SUM(CASE WHEN rr.finish_pos<=5 THEN 1 ELSE 0 END) as "Top 5",
                   SUM(CASE WHEN rr.finish_pos<=10 THEN 1 ELSE 0 END) as "Top 10",
                   ROUND(AVG(rr.laps_led),1) as "Avg Laps Led",
                   ROUND(AVG(rr.fastest_laps),1) as "Avg Fastest Laps"
            FROM race_results rr
            JOIN drivers d ON d.id=rr.driver_id
            LEFT JOIN dfs_points dp ON dp.race_id=rr.race_id
                AND dp.driver_id=rr.driver_id AND dp.platform='DraftKings'
            JOIN races r ON r.id=rr.race_id
            JOIN tracks t ON t.id=r.track_id
            WHERE t.name IN ({placeholders})
        """
        if season:
            query += f" AND r.season = ?"
            filter_tracks = filter_tracks + [season]
        query += """
            GROUP BY d.full_name
            HAVING Races >= 1
            ORDER BY "Avg DFS" DESC
        """

        df = pd.read_sql_query(query, conn, params=filter_tracks)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_salaries(race_id: int = None, platform: str = None) -> pd.DataFrame:
    """Query stored salaries from database."""
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
            query += " AND s.race_id = ?"
            params.append(race_id)
        if platform:
            query += " AND s.platform = ?"
            params.append(platform)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_projections(race_id: int = None, platform: str = "DraftKings") -> pd.DataFrame:
    """Query stored projections from database."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT d.full_name as Driver,
                   p.proj_score as "Proj Score",
                   p.salary as Salary,
                   p.value as Value,
                   p.track_score as "Track Score",
                   p.track_type_score as "Track Type Score",
                   p.form_score as "Form Score",
                   p.qual_adj as "Qual Adj",
                   p.practice_adj as "Practice Adj",
                   p.odds_adj as "Odds Adj",
                   p.track_races as "Track Races",
                   p.form_races as "Form Races",
                   p.generated_at
            FROM projections p
            JOIN drivers d ON d.id = p.driver_id
            WHERE p.platform = ?
        """
        params = [platform]
        if race_id:
            query += " AND p.race_id = ?"
            params.append(race_id)
        query += " ORDER BY p.proj_score DESC"
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dk_salaries_live() -> pd.DataFrame:
    """Fetch upcoming DraftKings NASCAR salaries directly from DK API.
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

    # Get the first (most upcoming) NASCAR group
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

        # Find or match DB race — try name, then api_race_id, then date
        db_race = conn.execute(
            "SELECT id FROM races WHERE series_id = ? AND race_name = ? ORDER BY season DESC LIMIT 1",
            (series_id, race_name)
        ).fetchone()

        if not db_race and race_id:
            db_race = conn.execute(
                "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
            ).fetchone()

        if not db_race:
            from datetime import datetime as _dt
            today = _dt.now().strftime("%Y-%m-%d")
            db_race = conn.execute(
                """SELECT id FROM races
                   WHERE series_id = ?
                   AND ABS(julianday(race_date) - julianday(?)) <= 7
                   ORDER BY ABS(julianday(race_date) - julianday(?))
                   LIMIT 1""",
                (series_id, today, today)
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

        count = 0
        for _, row in dk_df.iterrows():
            driver_name = row["Driver"]
            salary = row["DK Salary"]
            status = row.get("Status", "Available")

            # Find or create driver
            d = conn.execute("SELECT id FROM drivers WHERE full_name = ?",
                             (driver_name,)).fetchone()
            if not d:
                conn.execute("INSERT INTO drivers (full_name) VALUES (?)", (driver_name,))
                d = conn.execute("SELECT id FROM drivers WHERE full_name = ?",
                                 (driver_name,)).fetchone()

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


# ============================================================
# AUTO-FETCH NASCAR ODDS (Action Network)
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nascar_odds() -> dict:
    """Fetch upcoming NASCAR race odds from Action Network.

    Parses the __NEXT_DATA__ JSON embedded in the page which contains
    structured competition data with odds from multiple books.

    Returns dict of {driver_name: odds_string} for win odds.
    Also populates session_state with top5/top10 odds if available.
    Falls back to empty dict on failure.
    """
    import json as _json

    url = "https://www.actionnetwork.com/nascar/odds"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return {}

        # Extract __NEXT_DATA__ JSON from the page
        marker = "__NEXT_DATA__"
        idx = r.text.find(marker)
        if idx < 0:
            return {}
        json_start = r.text.find("{", idx)
        json_end = r.text.find("</script>", json_start)
        if json_start < 0 or json_end < 0:
            return {}

        data = _json.loads(r.text[json_start:json_end])
        sb = data.get("props", {}).get("pageProps", {}).get("scoreboardResponse", {})
        comps = sb.get("competitions", [])
        if not comps:
            return {}

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

        # Extract odds from all available markets
        markets = comp.get("markets", {})
        odds_result = {}
        top5_result = {}
        top10_result = {}

        for mk, mv in markets.items():
            event = mv.get("event", {})
            mk_lower = mk.lower() if mk else ""

            # Win odds (moneyline)
            ml = event.get("moneyline", [])
            if ml and not odds_result:
                for entry in ml:
                    pid = entry.get("player_id")
                    odds_val = entry.get("odds")
                    name = pid_to_name.get(pid)
                    if name and odds_val is not None:
                        odds_result[name] = str(odds_val)

            # Search for top 5 / top 10 finish markets
            # Action Network may use various keys: top_5, top5, top_5_finish, etc.
            for market_key, market_data in event.items():
                if not isinstance(market_data, list):
                    continue
                mk_key_lower = market_key.lower().replace(" ", "_")
                is_top5 = ("top_5" in mk_key_lower or "top5" in mk_key_lower) and not top5_result
                is_top10 = ("top_10" in mk_key_lower or "top10" in mk_key_lower) and not top10_result

                if is_top5 or is_top10:
                    target = top5_result if is_top5 else top10_result
                    for entry in market_data:
                        pid = entry.get("player_id")
                        odds_val = entry.get("odds")
                        name = pid_to_name.get(pid)
                        if name and odds_val is not None:
                            target[name] = str(odds_val)

        # Store top5/top10 separately — caller can retrieve via fetch_nascar_prop_odds()
        _PROP_ODDS_CACHE["top5"] = top5_result
        _PROP_ODDS_CACHE["top10"] = top10_result

        return odds_result

    except Exception:
        return {}


# Module-level cache for prop odds (top5/top10) from last fetch
_PROP_ODDS_CACHE: dict = {"top5": {}, "top10": {}}


def fetch_nascar_prop_odds() -> dict:
    """Return top5/top10 prop odds from the last fetch_nascar_odds() call.

    Returns {"top5": {name: odds_str}, "top10": {name: odds_str}}.
    """
    return dict(_PROP_ODDS_CACHE)


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

def _resolve_db_race_id(api_race_id: int):
    """Resolve a NASCAR API race_id to the internal DB race_id.

    Looks up via the api_race_id column on the races table.
    Returns None if not found.
    """
    if not api_race_id or not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    try:
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


def _resolve_db_race_id_with_fallback(race_id: int):
    """Resolve API race_id to DB race_id, with date-based fallback."""
    db_race_id = _resolve_db_race_id(race_id)
    if db_race_id:
        return db_race_id

    # Fallback: try matching by date from API
    try:
        for sid in [1, 2, 3]:
            api_url = f"{NASCAR_API_BASE}/2026/{sid}/race_list_basic.json"
            resp = requests.get(api_url, timeout=10)
            if resp.status_code != 200:
                continue
            for r in resp.json():
                if r.get("race_id") == race_id:
                    race_date = (r.get("race_date") or "")[:10]
                    if race_date:
                        conn_tmp = sqlite3.connect(str(DB_PATH))
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
                        conn_tmp.close()
                    break
            if db_race_id:
                break
    except Exception:
        pass

    return db_race_id


def save_odds_to_db(odds_data: dict, race_id: int, sportsbook: str = "action_network",
                    top5_data: dict = None, top10_data: dict = None):
    """Persist odds dict to the odds table, keyed by race_id.

    Smart update: only overwrites top5/top10 odds when new valid data is provided.
    If top5_data or top10_data is empty/None, existing values are preserved.

    Args:
        odds_data: {driver_name: odds_string} e.g. {"Kyle Larson": "+350"}
        race_id: NASCAR API race ID (will be resolved to internal DB race_id)
        sportsbook: source label
        top5_data: {driver_name: odds_string} for top 5 finish odds
        top10_data: {driver_name: odds_string} for top 10 finish odds
    """
    if not odds_data or not race_id or not DB_PATH.exists():
        return 0

    db_race_id = _resolve_db_race_id_with_fallback(race_id)
    if not db_race_id:
        return 0

    from src.utils import fuzzy_match_name

    conn = sqlite3.connect(str(DB_PATH))
    db_drivers = conn.execute("SELECT id, full_name FROM drivers").fetchall()
    name_to_id = {row[1]: row[0] for row in db_drivers}
    driver_names = list(name_to_id.keys())

    # Build top5/top10 lookup by driver_id (fuzzy matched)
    t5_by_id = {}
    t10_by_id = {}
    if top5_data:
        for name, odds_str in top5_data.items():
            try:
                val = float(str(odds_str).replace("+", ""))
                matched = fuzzy_match_name(name, driver_names)
                if matched:
                    t5_by_id[name_to_id[matched]] = val
            except (ValueError, TypeError):
                continue
    if top10_data:
        for name, odds_str in top10_data.items():
            try:
                val = float(str(odds_str).replace("+", ""))
                matched = fuzzy_match_name(name, driver_names)
                if matched:
                    t10_by_id[name_to_id[matched]] = val
            except (ValueError, TypeError):
                continue

    count = 0
    for name, odds_str in odds_data.items():
        try:
            odds_val = float(str(odds_str).replace("+", ""))
        except (ValueError, TypeError):
            continue

        matched = fuzzy_match_name(name, driver_names)
        if not matched:
            continue
        driver_id = name_to_id[matched]

        # Check if row exists with top5/top10 data we should preserve
        existing = conn.execute(
            "SELECT top5_odds, top10_odds FROM odds WHERE race_id = ? AND driver_id = ? AND sportsbook = ?",
            (db_race_id, driver_id, sportsbook)
        ).fetchone()

        # Use new prop data if available, else preserve existing
        t5_val = t5_by_id.get(driver_id)
        t10_val = t10_by_id.get(driver_id)
        if existing:
            if t5_val is None and existing[0] is not None:
                t5_val = existing[0]  # preserve existing top5
            if t10_val is None and existing[1] is not None:
                t10_val = existing[1]  # preserve existing top10

        conn.execute('''
            INSERT OR REPLACE INTO odds
            (race_id, driver_id, sportsbook, win_odds, top5_odds, top10_odds, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (db_race_id, driver_id, sportsbook, odds_val, t5_val, t10_val))
        count += 1

    conn.commit()
    conn.close()
    return count


def load_race_odds(race_id: int) -> dict:
    """Load saved odds for a race from the DB.

    Args:
        race_id: NASCAR API race ID (will be resolved to internal DB race_id)

    Returns dict of {driver_name: odds_string} matching the live format.
    """
    if not DB_PATH.exists():
        return {}

    db_race_id = _resolve_db_race_id(race_id)
    if not db_race_id:
        return {}

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute('''
        SELECT d.full_name, o.win_odds
        FROM odds o
        JOIN drivers d ON d.id = o.driver_id
        WHERE o.race_id = ?
        ORDER BY o.win_odds ASC
    ''', (db_race_id,)).fetchall()
    conn.close()

    result = {}
    for name, odds_val in rows:
        if odds_val is not None:
            ov = int(odds_val) if odds_val == int(odds_val) else odds_val
            result[name] = f"+{ov}" if ov > 0 else str(ov)
    return result


def load_race_prop_odds(race_id: int) -> dict:
    """Load top5 and top10 finish odds for a race from the DB.

    Args:
        race_id: NASCAR API race ID

    Returns {"top5": {driver_name: odds_int}, "top10": {driver_name: odds_int}}.
    """
    if not DB_PATH.exists():
        return {"top5": {}, "top10": {}}

    db_race_id = _resolve_db_race_id(race_id)
    if not db_race_id:
        return {"top5": {}, "top10": {}}

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute('''
        SELECT d.full_name, o.top5_odds, o.top10_odds
        FROM odds o
        JOIN drivers d ON d.id = o.driver_id
        WHERE o.race_id = ?
    ''', (db_race_id,)).fetchall()
    conn.close()

    top5 = {}
    top10 = {}
    for name, t5, t10 in rows:
        if t5 is not None:
            ov = int(t5) if t5 == int(t5) else t5
            top5[name] = ov
        if t10 is not None:
            ov = int(t10) if t10 == int(t10) else t10
            top10[name] = ov
    return {"top5": top5, "top10": top10}


# ============================================================
# AUTO-FETCH RACE RESULTS INTO DATABASE
# ============================================================

def fetch_and_store_race(series_id: int, race_id: int, year: int = 2026) -> dict:
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

    # Fastest laps from lap-times endpoint
    fastest_laps_map = {}
    if lap_data:
        fastest_laps_map = compute_fastest_laps(lap_data)

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
            fastest = fastest_laps_map.get(driver_name, 0)

            # Find or create driver
            d = conn.execute(
                "SELECT id FROM drivers WHERE full_name = ?", (driver_name,)
            ).fetchone()
            if not d:
                conn.execute("INSERT INTO drivers (full_name) VALUES (?)", (driver_name,))
                d = conn.execute(
                    "SELECT id FROM drivers WHERE full_name = ?", (driver_name,)
                ).fetchone()
            driver_id = d["id"]

            # Upsert race_results
            conn.execute(
                """INSERT INTO race_results
                   (race_id, driver_id, car_number, team, manufacturer,
                    start_pos, finish_pos, laps_completed, laps_led,
                    fastest_laps, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(race_id, driver_id) DO UPDATE SET
                    car_number=excluded.car_number, team=excluded.team,
                    manufacturer=excluded.manufacturer, start_pos=excluded.start_pos,
                    finish_pos=excluded.finish_pos, laps_completed=excluded.laps_completed,
                    laps_led=excluded.laps_led, fastest_laps=excluded.fastest_laps,
                    status=excluded.status""",
                (db_race_id, driver_id, car_number, team, manufacturer,
                 start_pos, finish_pos, laps_completed, laps_led, fastest, status),
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

def parse_dk_csv(file) -> pd.DataFrame:
    """Parse DraftKings CSV export."""
    try:
        df = pd.read_csv(file)
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
        df = pd.read_csv(file)
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
