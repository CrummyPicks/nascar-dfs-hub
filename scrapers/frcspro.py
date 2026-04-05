"""
Scraper for FRCS.pro — DraftKings & FanDuel DFS points history
Covers Cup, Xfinity, and Truck series.

Usage:
    python scrapers/frcspro.py --series cup --platform DraftKings --start 2020 --end 2024
    python scrapers/frcspro.py --series all --platform all --start 2018 --end 2024
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
BASE_URL = "https://www.frcspro.com"

SERIES_SLUGS = {
    "cup":     "cup",
    "xfinity": "xfinity",
    "trucks":  "trucks",
}

PLATFORM_SLUGS = {
    "DraftKings": "dk",
    "FanDuel":    "fd",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": BASE_URL,
}


# ── DB helpers ───────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def find_race_id(conn, series_code, season, race_name_hint=None, race_date=None):
    """Look up race_id from existing races table."""
    series_row = conn.execute(
        "SELECT id FROM series WHERE code=?", (series_code,)
    ).fetchone()
    if not series_row:
        return None
    series_id = series_row[0]

    # Try exact date match first
    if race_date:
        row = conn.execute(
            "SELECT id FROM races WHERE series_id=? AND season=? AND race_date=?",
            (series_id, season, race_date)
        ).fetchone()
        if row:
            return row[0]

    # Fuzzy race name match
    if race_name_hint:
        rows = conn.execute(
            "SELECT id, race_name FROM races WHERE series_id=? AND season=?",
            (series_id, season)
        ).fetchall()
        hint_lower = race_name_hint.lower()
        for rid, rname in rows:
            if rname and any(w in rname.lower() for w in hint_lower.split() if len(w) > 3):
                return rid
    return None


def find_driver_id(conn, name):
    row = conn.execute(
        "SELECT id FROM drivers WHERE full_name=?", (name.strip(),)
    ).fetchone()
    if row:
        return row[0]
    # Try last-name-first format (some sites use "Lastname, Firstname")
    if "," in name:
        parts = name.split(",", 1)
        alt = f"{parts[1].strip()} {parts[0].strip()}"
        row = conn.execute(
            "SELECT id FROM drivers WHERE full_name=?", (alt,)
        ).fetchone()
        if row:
            return row[0]
    # Partial match on last name
    last = name.strip().split()[-1]
    rows = conn.execute(
        "SELECT id, full_name FROM drivers WHERE last_name=?", (last,)
    ).fetchall()
    if len(rows) == 1:
        return rows[0][0]
    return None


def upsert_driver(conn, full_name):
    parts = full_name.strip().split()
    first = parts[0] if parts else ""
    last  = " ".join(parts[1:]) if len(parts) > 1 else ""
    conn.execute(
        "INSERT OR IGNORE INTO drivers(full_name,first_name,last_name) VALUES(?,?,?)",
        (full_name.strip(), first, last)
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM drivers WHERE full_name=?", (full_name.strip(),)
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


def fetch_json(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            print(f"  [HTTP {r.status_code}] {url}")
        except Exception as e:
            print(f"  [Error] {e} — attempt {attempt+1}/{retries}")
        time.sleep(delay * (attempt + 1))
    return None


# ── FRCS page discovery ──────────────────────────────────────────────────────

def get_race_list(series_slug, season):
    """
    Returns list of (race_label, race_url) for a given series/season.
    FRCS.pro URL pattern (adjust if site changes structure):
      /history/{series}/{season}/
    """
    url = f"{BASE_URL}/history/{series_slug}/{season}/"
    html = fetch(url)
    if not html:
        return []

    soup  = BeautifulSoup(html, "html.parser")
    races = []

    # Look for race links — typically in a table or list
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match race detail links
        if re.search(r"/(race|result|event)/", href, re.I):
            label = a.get_text(strip=True)
            full_url = href if href.startswith("http") else BASE_URL + href
            if label:
                races.append((label, full_url))

    # Deduplicate
    seen = set()
    unique = []
    for r in races:
        if r[1] not in seen:
            seen.add(r[1])
            unique.append(r)
    return unique


def scrape_race_dfs(race_url, platform):
    """
    Scrape DFS points for a single race page on FRCS.pro.
    Returns list of dicts: {driver, dfs_score, place_pts, laps_led_pts,
                             fastest_laps_pts, bonus_pts, race_name, race_date}
    """
    html = fetch(race_url)
    if not html:
        return []

    soup    = BeautifulSoup(html, "html.parser")
    results = []

    # Pull race meta from page title or h1
    page_title = ""
    h1 = soup.find("h1")
    if h1:
        page_title = h1.get_text(strip=True)

    # Extract date from page (look for ISO or common date formats)
    race_date = None
    page_text = soup.get_text()
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", page_text)
    if date_match:
        race_date = date_match.group(1)
    else:
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
            m = re.search(
                r"\b(" + r"\w+" + r"\s+\d{1,2},?\s+\d{4})\b",
                page_text
            )
            if m:
                try:
                    race_date = datetime.strptime(m.group(1), fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

    # Find the DFS scoring table
    # FRCS may have tabs or sections per platform — try to find the right table
    target_table = None
    for tbl in soup.find_all("table"):
        tbl_text = tbl.get_text().lower()
        hdrs = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        # Accept tables that have "pts", "score", or "points" in headers
        if any(h in hdrs for h in ["pts", "score", "points", "dk pts", "fd pts"]):
            # Prefer tables near a DK/FD heading
            parent_text = ""
            parent = tbl.find_parent()
            while parent and parent.name not in ["body", "html"]:
                parent_text = parent.get_text().lower()
                if platform.lower()[:2] in parent_text:
                    target_table = tbl
                    break
                parent = parent.find_parent()
            if not target_table:
                target_table = tbl  # fallback to first scoring table

    if not target_table:
        return []

    hdr_row   = target_table.find("tr")
    col_names = [th.get_text(strip=True).lower() for th in hdr_row.find_all(["th","td"])]

    def col(cells, *keys):
        for k in keys:
            for i, name in enumerate(col_names):
                if k in name and i < len(cells):
                    return cells[i].get_text(strip=True)
        return ""

    for row in target_table.find_all("tr")[1:]:
        cells = row.find_all(["td","th"])
        if len(cells) < 2:
            continue

        driver_tag  = row.find("a", href=re.compile(r"/driver/|/driverpage/"))
        driver_name = driver_tag.get_text(strip=True) if driver_tag else col(cells, "driver", "name")
        if not driver_name:
            continue

        def sf(v):
            try: return float(re.sub(r"[^\d.-]","",v)) if v else None
            except: return None

        results.append({
            "driver":          driver_name,
            "dfs_score":       sf(col(cells, "total", "pts", "score", "dk pts", "fd pts")),
            "place_pts":       sf(col(cells, "place", "finish pts", "pos pts")),
            "laps_led_pts":    sf(col(cells, "laps led", "led pts")),
            "fastest_laps_pts":sf(col(cells, "fastest", "fast pts", "fl pts")),
            "bonus_pts":       sf(col(cells, "bonus")),
            "race_name":       page_title,
            "race_date":       race_date,
        })

    return results


# ── Main scrape loop ─────────────────────────────────────────────────────────

def scrape_dfs(series_key, platform, start_year, end_year, conn):
    series_slug = SERIES_SLUGS[series_key]

    for season in range(start_year, end_year + 1):
        print(f"\n{'='*60}")
        print(f"  FRCS  {series_key.upper()}  {platform}  —  {season}")
        print(f"{'='*60}")

        race_list = get_race_list(series_slug, season)
        if not race_list:
            print(f"  No races found for {series_key} {season}")
            continue

        print(f"  Found {len(race_list)} races")

        for label, race_url in race_list:
            print(f"  {label[:55]}")
            dfs_rows = scrape_race_dfs(race_url, platform)
            if not dfs_rows:
                print("    SKIP — no DFS data")
                time.sleep(1)
                continue

            # Try to match this race to our DB
            sample  = dfs_rows[0]
            race_id = find_race_id(
                conn, series_key, season,
                race_name_hint=sample.get("race_name",""),
                race_date=sample.get("race_date")
            )

            if race_id is None:
                print(f"    WARN — race not found in DB, skipping DFS insert")
                time.sleep(1)
                continue

            inserted = 0
            for row in dfs_rows:
                driver_id = find_driver_id(conn, row["driver"])
                if driver_id is None:
                    driver_id = upsert_driver(conn, row["driver"])
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO dfs_points
                           (race_id,driver_id,platform,dfs_score,
                            place_pts,laps_led_pts,fastest_laps_pts,bonus_pts)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (race_id, driver_id, platform,
                         row["dfs_score"],    row["place_pts"],
                         row["laps_led_pts"], row["fastest_laps_pts"],
                         row["bonus_pts"])
                    )
                    inserted += 1
                except sqlite3.Error as e:
                    print(f"    DB error: {e}")
            conn.commit()
            print(f"    {inserted} DFS rows saved")
            time.sleep(1.5)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape FRCS.pro DFS points")
    parser.add_argument("--series",   default="all",
                        choices=["cup","xfinity","trucks","all"])
    parser.add_argument("--platform", default="all",
                        choices=["DraftKings","FanDuel","all"])
    parser.add_argument("--start",    type=int, default=2018)
    parser.add_argument("--end",      type=int, default=datetime.now().year)
    args = parser.parse_args()

    conn         = get_conn()
    series_list  = list(SERIES_SLUGS.keys()) if args.series   == "all" else [args.series]
    platform_list= list(PLATFORM_SLUGS.keys()) if args.platform == "all" else [args.platform]

    for s in series_list:
        for p in platform_list:
            scrape_dfs(s, p, args.start, args.end, conn)

    conn.close()
    print("\n[DONE] FRCS.pro DFS scrape complete.")


if __name__ == "__main__":
    main()
