"""
Scraper for racing-reference.info
Pulls Cup, Xfinity, and Truck series race results and qualifying data.

Usage:
    python scrapers/racing_reference.py --series cup --start 2020 --end 2024
    python scrapers/racing_reference.py --series all --start 2020 --end 2024
"""

import sqlite3
import requests
import time
import re
import argparse
import os
from bs4 import BeautifulSoup
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "nascar.db")
BASE_URL = "https://www.racing-reference.info"

# racing-reference series codes
SERIES_MAP = {
    "cup":     {"rr_code": "W", "db_code": "cup"},
    "xfinity": {"rr_code": "X", "db_code": "xfinity"},
    "trucks":  {"rr_code": "C", "db_code": "trucks"},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

TRACK_TYPE_HINTS = {
    "Daytona":     "superspeedway", "Talladega":   "superspeedway",
    "Atlanta":     "intermediate",  "Bristol":     "short",
    "Martinsville":"short",         "Dover":       "intermediate",
    "Pocono":      "intermediate",  "Michigan":    "intermediate",
    "Charlotte":   "intermediate",  "Texas":       "intermediate",
    "Kansas":      "intermediate",  "Las Vegas":   "intermediate",
    "Homestead":   "intermediate",  "Phoenix":     "intermediate",
    "Richmond":    "short",         "Nashville":   "intermediate",
    "New Hampshire":"short",        "Watkins Glen":"road",
    "Sonoma":      "road",          "Road America":"road",
    "Indianapolis":"road",          "Circuit of the Americas":"road",
    "Dirt":        "dirt",          "Bristol Dirt":"dirt",
}


# ── DB helpers ───────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_series_id(conn, code):
    row = conn.execute("SELECT id FROM series WHERE code=?", (code,)).fetchone()
    return row[0] if row else None


def upsert_track(conn, name):
    track_type = "unknown"
    for hint, t in TRACK_TYPE_HINTS.items():
        if hint.lower() in name.lower():
            track_type = t
            break
    conn.execute(
        "INSERT OR IGNORE INTO tracks(name, track_type) VALUES(?,?)",
        (name, track_type)
    )
    conn.commit()
    return conn.execute("SELECT id FROM tracks WHERE name=?", (name,)).fetchone()[0]


def upsert_driver(conn, full_name):
    parts = full_name.strip().split()
    first = parts[0] if parts else ""
    last  = " ".join(parts[1:]) if len(parts) > 1 else ""
    conn.execute(
        "INSERT OR IGNORE INTO drivers(full_name,first_name,last_name) VALUES(?,?,?)",
        (full_name.strip(), first, last)
    )
    conn.commit()
    return conn.execute("SELECT id FROM drivers WHERE full_name=?", (full_name.strip(),)).fetchone()[0]


def upsert_race(conn, series_id, track_id, season, race_num, name, date, laps, miles):
    conn.execute(
        """INSERT OR IGNORE INTO races
           (series_id,track_id,season,race_num,race_name,race_date,laps,miles)
           VALUES(?,?,?,?,?,?,?,?)""",
        (series_id, track_id, season, race_num, name, date, laps, miles)
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM races WHERE series_id=? AND season=? AND race_num=?",
        (series_id, season, race_num)
    ).fetchone()[0]


# ── HTTP helper ──────────────────────────────────────────────────────────────

def fetch(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
            print(f"  [HTTP {r.status_code}] {url}")
        except Exception as e:
            print(f"  [Error] {e} — attempt {attempt+1}/{retries}")
        time.sleep(delay * (attempt + 1))
    return None


# ── Season schedule ──────────────────────────────────────────────────────────

def get_season_races(season, rr_code):
    """Returns list of (race_num, race_url, race_name, track_name, date)."""
    url = f"{BASE_URL}/raceyear/{season}/{rr_code}"
    html = fetch(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    races = []

    # racing-reference uses a table with class 'tb' for schedule
    table = soup.find("table", class_="tb")
    if not table:
        # fallback: any table containing race links
        table = soup.find("table")

    if not table:
        print(f"  [WARN] No schedule table found for {season}/{rr_code}")
        return []

    for i, row in enumerate(table.find_all("tr")[1:], start=1):
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        # First cell often has the race link
        link_tag = row.find("a", href=re.compile(r"/race/"))
        if not link_tag:
            continue
        race_url  = BASE_URL + link_tag["href"]
        race_name = link_tag.get_text(strip=True)

        # Grab date from 2nd column and track from 3rd
        date_text  = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        track_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""

        # Parse date
        parsed_date = None
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
            try:
                parsed_date = datetime.strptime(date_text, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                pass

        races.append((i, race_url, race_name, track_text, parsed_date))

    return races


# ── Individual race results ──────────────────────────────────────────────────

def scrape_race(race_url):
    """
    Returns dict with:
        track_name, laps, miles,
        results: list of dicts with driver, car_num, team, mfr,
                 start, finish, laps_completed, laps_led, status, points
    """
    html = fetch(race_url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    data = {"track_name": "", "laps": None, "miles": None, "results": []}

    # Track / race info from header
    header = soup.find("h1") or soup.find("h2")
    if header:
        header_text = header.get_text(strip=True)
        # e.g. "2024 Daytona 500 Race Results"
        data["track_name"] = header_text

    # Try to find laps/miles in the page
    page_text = soup.get_text()
    laps_match  = re.search(r"(\d{2,3})\s+laps?", page_text, re.IGNORECASE)
    miles_match = re.search(r"([\d,]+\.?\d*)\s+miles?", page_text, re.IGNORECASE)
    if laps_match:
        data["laps"] = int(laps_match.group(1))
    if miles_match:
        data["miles"] = float(miles_match.group(1).replace(",", ""))

    # Find result table — racing-reference uses class 'tb'
    result_table = None
    for tbl in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if any(h in headers for h in ["driver", "fin", "finish"]):
            result_table = tbl
            break

    if not result_table:
        return data

    header_row = result_table.find("tr")
    if not header_row:
        return data

    col_names = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]

    def col(cells, *keys):
        for k in keys:
            for i, name in enumerate(col_names):
                if k in name and i < len(cells):
                    return cells[i].get_text(strip=True)
        return ""

    for row in result_table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue

        driver_tag = row.find("a", href=re.compile(r"/driver/|/driverpage/"))
        driver_name = driver_tag.get_text(strip=True) if driver_tag else col(cells, "driver")
        if not driver_name:
            continue

        def safe_int(v):
            try: return int(re.sub(r"[^\d-]", "", v)) if v else None
            except: return None

        def safe_float(v):
            try: return float(re.sub(r"[^\d.-]", "", v)) if v else None
            except: return None

        data["results"].append({
            "driver":          driver_name,
            "car_number":      col(cells, "#", "car", "no."),
            "team":            col(cells, "team", "owner"),
            "manufacturer":    col(cells, "make", "mfr", "car make"),
            "start_pos":       safe_int(col(cells, "start", "st")),
            "finish_pos":      safe_int(col(cells, "fin", "finish", "pos")),
            "laps_completed":  safe_int(col(cells, "laps")),
            "laps_led":        safe_int(col(cells, "led", "laps led")),
            "status":          col(cells, "status", "condition"),
            "points":          safe_int(col(cells, "pts", "points")),
        })

    return data


# ── Qualifying ───────────────────────────────────────────────────────────────

def scrape_qualifying(race_url):
    """Scrape qualifying from the qual sub-page."""
    qual_url = race_url.rstrip("/") + "/qual"
    html = fetch(qual_url)
    if not html:
        return []

    soup  = BeautifulSoup(html, "html.parser")
    table = None
    for tbl in soup.find_all("table"):
        hdrs = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if any(h in hdrs for h in ["speed", "time", "pos", "position"]):
            table = tbl
            break
    if not table:
        return []

    col_names = [th.get_text(strip=True).lower() for th in table.find("tr").find_all(["th","td"])]

    def col(cells, *keys):
        for k in keys:
            for i, name in enumerate(col_names):
                if k in name and i < len(cells):
                    return cells[i].get_text(strip=True)
        return ""

    results = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td","th"])
        driver_tag  = row.find("a", href=re.compile(r"/driver/|/driverpage/"))
        driver_name = driver_tag.get_text(strip=True) if driver_tag else col(cells, "driver")
        if not driver_name:
            continue

        def safe_float(v):
            try: return float(re.sub(r"[^\d.]", "", v)) if v else None
            except: return None

        results.append({
            "driver":     driver_name,
            "q_position": int(re.sub(r"\D","",col(cells,"pos","rank","#")) or 0) or None,
            "q_speed":    safe_float(col(cells,"speed","mph")),
            "q_time":     safe_float(col(cells,"time","sec")),
        })
    return results


# ── Main scrape loop ─────────────────────────────────────────────────────────

def scrape_series(series_key, start_year, end_year, conn):
    cfg       = SERIES_MAP[series_key]
    series_id = get_series_id(conn, cfg["db_code"])
    rr_code   = cfg["rr_code"]

    for season in range(start_year, end_year + 1):
        print(f"\n{'='*60}")
        print(f"  {cfg['db_code'].upper()}  —  Season {season}")
        print(f"{'='*60}")

        schedule = get_season_races(season, rr_code)
        if not schedule:
            print(f"  No schedule found for {season}")
            continue

        print(f"  Found {len(schedule)} races")

        for race_num, race_url, race_name, track_name, race_date in schedule:
            # ── Track
            if not track_name:
                track_name = "Unknown"
            track_id = upsert_track(conn, track_name)

            # ── Race results
            print(f"  [{race_num:02d}] {race_name[:50]}  ({race_date})")
            race_data = scrape_race(race_url)
            if not race_data:
                print("       SKIP — no data")
                time.sleep(1)
                continue

            laps  = race_data.get("laps")
            miles = race_data.get("miles")
            race_id = upsert_race(conn, series_id, track_id, season,
                                  race_num, race_name, race_date, laps, miles)

            inserted = 0
            for r in race_data["results"]:
                driver_id = upsert_driver(conn, r["driver"])
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO race_results
                           (race_id,driver_id,car_number,team,manufacturer,
                            start_pos,finish_pos,laps_completed,laps_led,
                            status,points)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (race_id, driver_id,
                         r["car_number"], r["team"], r["manufacturer"],
                         r["start_pos"], r["finish_pos"],
                         r["laps_completed"], r["laps_led"],
                         r["status"], r["points"])
                    )
                    inserted += 1
                except sqlite3.Error as e:
                    print(f"       DB error: {e}")
            conn.commit()
            print(f"       {inserted} driver results saved")

            # ── Qualifying
            qual_results = scrape_qualifying(race_url)
            q_inserted = 0
            for q in qual_results:
                driver_id = upsert_driver(conn, q["driver"])
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO qualifying_results
                           (race_id,driver_id,q_position,q_speed,q_time)
                           VALUES(?,?,?,?,?)""",
                        (race_id, driver_id,
                         q["q_position"], q["q_speed"], q["q_time"])
                    )
                    q_inserted += 1
                except sqlite3.Error:
                    pass
            conn.commit()
            if q_inserted:
                print(f"       {q_inserted} qualifying results saved")

            time.sleep(1.5)  # polite delay


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape racing-reference.info")
    parser.add_argument("--series", default="all",
                        choices=["cup","xfinity","trucks","all"])
    parser.add_argument("--start",  type=int, default=2018)
    parser.add_argument("--end",    type=int, default=datetime.now().year)
    args = parser.parse_args()

    conn = get_conn()
    series_list = list(SERIES_MAP.keys()) if args.series == "all" else [args.series]

    for s in series_list:
        scrape_series(s, args.start, args.end, conn)

    conn.close()
    print("\n[DONE] racing-reference scrape complete.")


if __name__ == "__main__":
    main()
