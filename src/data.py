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
            driver_name = (item.get("FullName") or item.get("Driver") or "").replace(" #", "").strip()
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
                    "driver": car["driver_fullname"],
                    "team": car.get("team_name", ""),
                    "make": car.get("car_make", ""),
                    "crew_chief": car.get("crew_chief_fullname", ""),
                }
        for r in races[0].get("results", []):
            cn = str(r.get("car_number", "")).strip()
            if cn and cn not in car_map and r.get("driver_fullname"):
                car_map[cn] = {
                    "driver": r["driver_fullname"],
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
            "Driver": car.get("driver_fullname"),
            "Team": car.get("team_name"),
            "Manufacturer": car.get("car_make"),
            "Crew Chief": car.get("crew_chief_fullname", ""),
        } for car in cars]
        return pd.DataFrame(rows)

    results = races[0].get("results", [])
    if results:
        rows = [{
            "Car": r.get("car_number"),
            "Driver": r.get("driver_fullname"),
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
            "Driver": r.get("driver_fullname"),
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
        name = d["FullName"]
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
            result[d["FullName"]] = round(np.mean(positions), 1)
    return result


# ============================================================
# TRACK HISTORY SCRAPING
# ============================================================

@st.cache_data(ttl=7200, show_spinner=False)
def scrape_track_history(track_name: str, series_id: int = 1) -> pd.DataFrame:
    """Scrape driveraverages.com for recent track history."""
    key = track_name.lower().strip()
    trk_id = None
    for name, tid in DA_TRACK_IDS.items():
        if name in key:
            trk_id = tid
            break
    if trk_id is None:
        return pd.DataFrame()

    series_map = {1: "nascar", 2: "nascar_secondseries", 3: "nascar_truckseries"}
    series_path = series_map.get(series_id, "nascar")

    try:
        resp = requests.get(
            f"https://www.driveraverages.com/{series_path}/track_avg.php?trk_id={trk_id}",
            timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(resp.text, "lxml")

        col_names = ["Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
                     "Top 10", "Top 20", "Laps Led", "Avg Start", "Best Finish",
                     "Worst Finish", "DNF", "Avg Rating", "Detail"]

        all_tables = []
        for t in soup.find_all("table"):
            rows = []
            for tr in t.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) == 15:
                    vals = [c.get_text(strip=True) for c in cells]
                    if vals[1] and vals[1] != "Driver":
                        rows.append(vals)
            if rows:
                all_tables.append(rows)

        dfs = {}
        labels = ["Recent", "All-Time"]
        for i, rows in enumerate(all_tables[:2]):
            df = pd.DataFrame(rows, columns=col_names).drop(columns=["Detail"])
            for col in df.columns:
                if col != "Driver":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            dfs[labels[i] if i < len(labels) else f"Table {i}"] = df

        if dfs:
            return dfs.get("Recent", list(dfs.values())[0])
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=7200, show_spinner=False)
def scrape_track_history_alltime(track_name: str, series_id: int = 1) -> pd.DataFrame:
    """Scrape driveraverages.com for all-time track history."""
    key = track_name.lower().strip()
    trk_id = None
    for name, tid in DA_TRACK_IDS.items():
        if name in key:
            trk_id = tid
            break
    if trk_id is None:
        return pd.DataFrame()

    series_map = {1: "nascar", 2: "nascar_secondseries", 3: "nascar_truckseries"}
    series_path = series_map.get(series_id, "nascar")

    try:
        resp = requests.get(
            f"https://www.driveraverages.com/{series_path}/track_avg.php?trk_id={trk_id}",
            timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(resp.text, "lxml")
        col_names = ["Rank", "Driver", "Avg Finish", "Races", "Wins", "Top 5",
                     "Top 10", "Top 20", "Laps Led", "Avg Start", "Best Finish",
                     "Worst Finish", "DNF", "Avg Rating", "Detail"]
        all_tables = []
        for t in soup.find_all("table"):
            rows = []
            for tr in t.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) == 15:
                    vals = [c.get_text(strip=True) for c in cells]
                    if vals[1] and vals[1] != "Driver":
                        rows.append(vals)
            if rows:
                all_tables.append(rows)
        if len(all_tables) >= 2:
            rows = all_tables[1]
        elif all_tables:
            rows = all_tables[0]
        else:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=col_names).drop(columns=["Detail"])
        for col in df.columns:
            if col != "Driver":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


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


def query_season_stats() -> pd.DataFrame:
    """Pull aggregated season stats from local DB (legacy compat)."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
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
            LEFT JOIN dfs_points dp ON dp.race_id=rr.race_id AND dp.driver_id=rr.driver_id
            GROUP BY d.full_name
            ORDER BY "Avg DFS" DESC
        """, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def query_track_type_stats(track_type: str) -> pd.DataFrame:
    """Pull stats filtered by track type from local DB."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("""
            SELECT d.full_name as Driver,
                   COUNT(*) as Races,
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
            LEFT JOIN dfs_points dp ON dp.race_id=rr.race_id AND dp.driver_id=rr.driver_id
            JOIN races r ON r.id=rr.race_id
            WHERE r.track_type = ?
            GROUP BY d.full_name
            ORDER BY "Avg DFS" DESC
        """, conn, params=(track_type,))
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
    "https://api.draftkings.com/contests/v1/contests?sport=NASCAR",
    "https://api.draftkings.com/contests/v1/contests?sport=nascar",
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
    if not draft_groups:
        return pd.DataFrame()

    # Get the first (most upcoming) group
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

        # Find or match DB race
        db_race = conn.execute(
            "SELECT id FROM races WHERE series_id = ? AND race_name = ? ORDER BY season DESC LIMIT 1",
            (series_id, race_name)
        ).fetchone()

        if not db_race:
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
    """Fetch upcoming NASCAR race win odds from Action Network.

    Parses the __NEXT_DATA__ JSON embedded in the page which contains
    structured competition data with moneyline odds from multiple books.

    Returns dict of {driver_name: odds_string} e.g. {"Kyle Larson": "+350"}.
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

        # Extract moneyline odds from the first available book
        markets = comp.get("markets", {})
        for mk, mv in markets.items():
            event = mv.get("event", {})
            ml = event.get("moneyline", [])
            if not ml:
                continue

            odds_result = {}
            for entry in ml:
                pid = entry.get("player_id")
                odds_val = entry.get("odds")
                name = pid_to_name.get(pid)
                if name and odds_val is not None:
                    # Show plain number for positive, keep "-" for negative
                    odds_str = str(odds_val) if odds_val >= 0 else str(odds_val)
                    odds_result[name] = odds_str

            if odds_result:
                return odds_result

        return {}

    except Exception:
        return {}


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
                "UPDATE races SET race_name=?, race_date=?, track_id=?, laps=? WHERE id=?",
                (race_name, race_date, track_id, total_laps, db_race_id),
            )
        else:
            conn.execute(
                """INSERT INTO races (series_id, track_id, season, race_num,
                   race_name, race_date, laps)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (series_id, track_id, year, race_num, race_name, race_date, total_laps),
            )
            db_race_id = conn.execute(
                "SELECT id FROM races WHERE series_id=? AND season=? AND race_num=?",
                (series_id, year, race_num),
            ).fetchone()["id"]

        # --- Process each driver result --------------------------------
        driver_count = 0

        for r in results:
            finish_pos = r.get("finishing_position", 0) or 0
            driver_name = r.get("driver_fullname") or ""
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
