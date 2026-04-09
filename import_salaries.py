"""
Import DraftKings salary CSV into the database for a specific race.

Usage:
    python import_salaries.py                          # Interactive — picks next upcoming race
    python import_salaries.py --file DKSalaries.csv    # Specify CSV file
    python import_salaries.py --series cup             # Cup series (default)
    python import_salaries.py --series xfinity         # Xfinity series
    python import_salaries.py --series truck           # Truck series
    python import_salaries.py --push                   # Auto git commit + push after import
"""

import argparse
import glob
import os
import sys
import sqlite3
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import DB_PATH
from src.data import (
    parse_dk_csv, parse_fd_csv, sync_dk_salaries_to_db, sync_fd_salaries_to_db,
    fetch_race_list, filter_point_races,
)
from src.utils import normalize_driver_name

SERIES_MAP = {
    "cup": 1, "1": 1,
    "xfinity": 2, "oreilly": 2, "2": 2,
    "truck": 3, "craftsman": 3, "3": 3,
}

SERIES_NAMES = {1: "Cup", 2: "Xfinity", 3: "Truck"}


def find_csv_file(file_arg):
    """Find the DK CSV file — from argument, or search common locations.

    Handles Windows download patterns like DKSalaries (3).csv.
    When multiple recent CSVs exist, lets the user pick.
    """
    if file_arg and os.path.exists(file_arg):
        return file_arg

    # Search common locations — glob handles "DKSalaries (3).csv" patterns
    search_dirs = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.dirname(__file__),
    ]

    candidates = []
    seen = set()
    for d in search_dirs:
        for pattern in ["DKSalaries*.csv", "DKsalaries*.csv", "dksalaries*.csv"]:
            for f in glob.glob(os.path.join(d, pattern)):
                real = os.path.realpath(f)
                if real not in seen:
                    seen.add(real)
                    candidates.append(f)

    # Filter to recent files (last 7 days) and sort newest first
    now = datetime.now().timestamp()
    recent = [(f, os.path.getmtime(f)) for f in candidates
              if now - os.path.getmtime(f) < 7 * 86400]
    recent.sort(key=lambda x: x[1], reverse=True)

    if not recent:
        return None

    if len(recent) == 1:
        return recent[0][0]

    # Multiple recent CSVs — let user pick (common when importing 3 series)
    print("Found multiple recent DK CSV files:")
    for i, (f, mtime) in enumerate(recent[:10], 1):
        mod = datetime.fromtimestamp(mtime).strftime("%m/%d %H:%M")
        size_kb = os.path.getsize(f) / 1024
        print(f"  [{i}] {os.path.basename(f):40s} ({mod}, {size_kb:.0f} KB)")

    choice = input(f"\nSelect file [1]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() else 0
    if idx < 0 or idx >= len(recent):
        idx = 0
    return recent[idx][0]


def get_upcoming_races(series_id, year=None):
    """Get list of upcoming races for the series."""
    if year is None:
        year = datetime.now().year
    races = fetch_race_list(series_id, year)
    if not races:
        return []
    point_races = filter_point_races(races)
    now = datetime.now()
    upcoming = []
    for race in point_races:
        date_str = race.get("race_date", "")
        try:
            rd = datetime.fromisoformat(date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
            if rd.date() >= now.date():
                upcoming.append(race)
        except Exception:
            upcoming.append(race)
    return upcoming


def detect_platform(csv_path):
    """Detect if CSV is DraftKings or FanDuel format."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, nrows=1)
        cols = [c.lower() for c in df.columns]
        if "name" in cols and "salary" in cols:
            return "dk"
        if "nickname" in cols or "player" in cols:
            return "fd"
        if "salary" in cols:
            return "dk"  # default guess
    except Exception:
        pass
    return "dk"


def main():
    parser = argparse.ArgumentParser(description="Import DK/FD salary CSV")
    parser.add_argument("--file", "-f", help="Path to CSV file")
    parser.add_argument("--series", "-s", default="cup", help="Series: cup, xfinity, truck")
    parser.add_argument("--push", action="store_true", help="Auto git commit + push after import")
    parser.add_argument("--race", "-r", type=int, help="Race number (1-indexed) to override auto-detect")
    args = parser.parse_args()

    series_id = SERIES_MAP.get(args.series.lower(), 1)
    series_name = SERIES_NAMES[series_id]
    print(f"\n{'='*60}")
    print(f"  NASCAR DFS — Salary Import ({series_name} Series)")
    print(f"{'='*60}\n")

    # Step 1: Find CSV file
    csv_path = find_csv_file(args.file)
    if not csv_path:
        print("No CSV file found!")
        print("Either:")
        print("  - Pass --file path/to/DKSalaries.csv")
        print("  - Place DKSalaries*.csv in Downloads or this folder")
        if not args.file:
            csv_path = input("\nEnter CSV file path (or drag file here): ").strip().strip('"')
            if not csv_path or not os.path.exists(csv_path):
                print("File not found. Exiting.")
                return
        else:
            return

    print(f"CSV file: {csv_path}")
    print(f"Modified: {datetime.fromtimestamp(os.path.getmtime(csv_path)).strftime('%Y-%m-%d %H:%M')}")

    # Step 2: Detect platform and parse
    platform = detect_platform(csv_path)
    if platform == "dk":
        df = parse_dk_csv(open(csv_path, "rb"))
        sal_col = "DK Salary"
    else:
        df = parse_fd_csv(open(csv_path, "rb"))
        sal_col = "FD Salary"

    if df.empty:
        print("Failed to parse CSV — check file format.")
        return

    print(f"Platform: {'DraftKings' if platform == 'dk' else 'FanDuel'}")
    print(f"Drivers:  {len(df)}")
    print(f"Salary range: ${df[sal_col].min():,} — ${df[sal_col].max():,}")
    print()

    # Step 3: Select race
    upcoming = get_upcoming_races(series_id)
    if not upcoming:
        print(f"No upcoming {series_name} races found.")
        return

    if args.race and 1 <= args.race <= len(upcoming):
        selected = upcoming[args.race - 1]
    else:
        print("Upcoming races:")
        for i, race in enumerate(upcoming[:8], 1):
            date = race.get("race_date", "")[:10]
            name = race.get("race_name", "Unknown")
            track = race.get("track_name", "")
            print(f"  [{i}] {date} — {name} @ {track}")

        choice = input(f"\nSelect race [1]: ").strip()
        idx = int(choice) - 1 if choice.isdigit() else 0
        if idx < 0 or idx >= len(upcoming):
            idx = 0
        selected = upcoming[idx]

    race_id = selected.get("race_id")
    race_name = selected.get("race_name", "Unknown")
    race_date = selected.get("race_date", "")[:10]
    track = selected.get("track_name", "")

    print(f"\nImporting salaries for: {race_name}")
    print(f"  Track: {track}")
    print(f"  Date:  {race_date}")
    print(f"  API Race ID: {race_id}")

    # Step 4: Sync to DB
    if platform == "dk":
        count = sync_dk_salaries_to_db(df, race_id, series_id, race_name)
    else:
        count = sync_fd_salaries_to_db(df, race_id, series_id, race_name)

    if count > 0:
        print(f"\n  Saved {count} salaries to database!")
    else:
        print(f"\n  WARNING: No salaries saved — race may not be in DB yet.")
        print(f"  Try running: python refresh_data.py --series {args.series}")
        return

    # Show top 10
    print(f"\n  Top 10 by salary:")
    top = df.sort_values(sal_col, ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"    {row['Driver']:30s} ${row[sal_col]:>6,}")

    # Step 5: Git commit + push
    if args.push:
        print(f"\n  Committing and pushing to git...")
        os.system(f'git add nascar.db')
        os.system(f'git commit -m "Add {platform.upper()} salaries for {race_name} ({race_date})"')
        os.system(f'git push')
        print(f"  Done! Salaries are now live on Streamlit Cloud.")
    else:
        print(f"\n  To deploy to Streamlit Cloud, run:")
        print(f"    git add nascar.db && git commit -m \"Add salaries for {race_name}\" && git push")
        print(f"  Or re-run with --push flag to auto-deploy.")

    print()


if __name__ == "__main__":
    main()
