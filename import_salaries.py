"""
Import DraftKings/FanDuel salary CSVs and Bovada odds into the database.

Interactive loop: import salaries and/or odds for multiple series, then commit + push once.

Usage:
    python import_salaries.py              # Interactive loop (recommended)
    python import_salaries.py --no-push    # Import without git push
"""

import argparse
import glob
import os
import re
import sys
import sqlite3
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import DB_PATH
from src.data import (
    parse_dk_csv, parse_fd_csv, sync_dk_salaries_to_db, sync_fd_salaries_to_db,
    fetch_race_list, filter_point_races, save_odds_to_db,
)

SERIES_MAP = {"1": 1, "2": 2, "3": 3, "cup": 1, "xfinity": 2, "truck": 3}
SERIES_NAMES = {1: "Cup", 2: "O'Reilly", 3: "Truck"}
MAX_UPCOMING = 2  # Only show next 2 upcoming races
RECENT_HISTORY = 3  # Show last 3 completed races for backfill


def find_csv_files():
    """Find all recent DK CSV files in Downloads/Desktop/project folder."""
    search_dirs = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.dirname(__file__),
    ]

    candidates = []
    seen = set()
    for d in search_dirs:
        for pattern in ["DKSalaries*.csv", "DKsalaries*.csv", "dksalaries*.csv",
                        "FDSalaries*.csv", "FanDuel*.csv"]:
            for f in glob.glob(os.path.join(d, pattern)):
                real = os.path.realpath(f)
                if real not in seen:
                    seen.add(real)
                    candidates.append(f)

    # Filter to recent files (last 14 days) and sort newest first
    now = datetime.now().timestamp()
    recent = [(f, os.path.getmtime(f)) for f in candidates
              if now - os.path.getmtime(f) < 14 * 86400]
    recent.sort(key=lambda x: x[1], reverse=True)
    return recent


def pick_csv(recent_files):
    """Let user pick a CSV from the list, or enter a path."""
    if not recent_files:
        path = input("No CSV files found. Enter file path (or drag file here): ").strip().strip('"')
        if path and os.path.exists(path):
            return path
        return None

    print("\nRecent CSV files:")
    for i, (f, mtime) in enumerate(recent_files[:10], 1):
        mod = datetime.fromtimestamp(mtime).strftime("%m/%d %H:%M")
        size_kb = os.path.getsize(f) / 1024
        print(f"  [{i}] {os.path.basename(f):40s}  {mod}  ({size_kb:.0f} KB)")

    choice = input(f"\nSelect file [1]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() else 0
    if idx < 0 or idx >= len(recent_files):
        idx = 0
    return recent_files[idx][0]


def get_race_options(series_id, year=None):
    """Get recent completed + upcoming races for selection."""
    if year is None:
        year = datetime.now().year
    races = fetch_race_list(series_id, year)
    if not races:
        return [], []
    point_races = filter_point_races(races)
    now = datetime.now()

    completed = []
    upcoming = []
    for race in point_races:
        date_str = race.get("race_date", "")
        try:
            rd = datetime.fromisoformat(
                date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
            if rd.date() < now.date():
                completed.append(race)
            else:
                upcoming.append(race)
        except Exception:
            upcoming.append(race)

    # Only show last N completed + next N upcoming
    recent_completed = completed[-RECENT_HISTORY:] if completed else []
    next_upcoming = upcoming[:MAX_UPCOMING] if upcoming else []
    return recent_completed, next_upcoming


def detect_platform(csv_path):
    """Detect if CSV is DraftKings or FanDuel format."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, nrows=1)
        cols = [c.lower() for c in df.columns]
        if "nickname" in cols or "player" in cols:
            return "fd"
    except Exception:
        pass
    return "dk"


def check_existing_salaries(race_id, series_id, platform="DraftKings"):
    """Check if salaries already exist in DB for this race."""
    from src.data import query_salaries
    df = query_salaries(race_id=race_id, platform=platform)
    return len(df) if not df.empty else 0


def check_existing_odds(race_id):
    """Check if odds already exist in DB for this race."""
    from src.data import load_race_odds
    odds = load_race_odds(race_id)
    return len(odds)


def pick_series():
    """Prompt user to pick a series. Returns (series_id, series_name)."""
    print("\n  Series:")
    print("    [1] Cup")
    print("    [2] O'Reilly (Xfinity)")
    print("    [3] Truck")
    series_choice = input("  Select series (1/2/3) [1]: ").strip()
    series_id = SERIES_MAP.get(series_choice, 1)
    return series_id, SERIES_NAMES[series_id]


def pick_race(series_id, series_name, check_type="salary", platform="DraftKings"):
    """Prompt user to pick a race. Returns selected race dict or None."""
    completed, upcoming = get_race_options(series_id)
    if not completed and not upcoming:
        print(f"  No {series_name} races found.")
        return None

    print(f"\n  {series_name} races:")
    all_races = []

    if completed:
        print("  -- Recent (backfill) --")
        for race in completed:
            date = race.get("race_date", "")[:10]
            name = race.get("race_name", "Unknown")
            track = race.get("track_name", "")
            if check_type == "salary":
                existing = check_existing_salaries(race.get("race_id"), series_id, platform)
                status = f" [{existing} salaries saved]" if existing else ""
            else:
                existing = check_existing_odds(race.get("race_id"))
                status = f" [{existing} odds saved]" if existing else ""
            idx = len(all_races) + 1
            all_races.append(race)
            print(f"    [{idx}] {date} — {name} @ {track}{status}")

    if upcoming:
        print("  -- Upcoming --")
        for race in upcoming:
            date = race.get("race_date", "")[:10]
            name = race.get("race_name", "Unknown")
            track = race.get("track_name", "")
            if check_type == "salary":
                existing = check_existing_salaries(race.get("race_id"), series_id, platform)
                status = f" [{existing} salaries saved]" if existing else ""
            else:
                existing = check_existing_odds(race.get("race_id"))
                status = f" [{existing} odds saved]" if existing else ""
            idx = len(all_races) + 1
            all_races.append(race)
            marker = " <-- next" if idx == len(completed) + 1 else ""
            print(f"    [{idx}] {date} — {name} @ {track}{status}{marker}")

    default_idx = len(completed)  # 0-indexed position of first upcoming
    choice = input(f"\n  Select race [{default_idx + 1}]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() else default_idx
    if idx < 0 or idx >= len(all_races):
        idx = default_idx
    return all_races[idx]


def parse_bovada_odds(text):
    """Parse odds from Bovada copy-paste text.

    Supports formats:
        Corey Heim+300          (Bovada direct copy)
        Kyle Larson, -115       (comma-separated)
        Chase Elliott +1200     (space-separated)

    Auto-skips header lines (race name, date, time, "Outright", etc.)
    """
    odds = {}
    skip_patterns = re.compile(
        r'^(outright|futures?|top\s*\d|moneyline|head.to.head'
        r'|\d{1,2}/\d{1,2}/\d{2,4}'
        r'|\d{1,2}:\d{2}\s*(am|pm)?'
        r')$', re.IGNORECASE
    )

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if skip_patterns.match(line):
            continue
        # Skip lines without odds
        if not re.search(r'[+-]\d+', line):
            continue
        # Comma-separated: "Driver Name, +350"
        if "," in line:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and re.match(r'^[+-]?\d+$', parts[1]):
                odds[parts[0]] = parts[1] if parts[1].startswith(('+', '-')) else f"+{parts[1]}"
                continue
        # Bovada no-space: "DriverName+300"
        m = re.match(r'^(.+?)([+-]\d+)$', line)
        if m and m.group(1).strip():
            odds[m.group(1).strip()] = m.group(2)
            continue
        # Space-separated: "Driver Name +350"
        m = re.match(r'^(.+?)\s+([+-]\d+)$', line)
        if m:
            odds[m.group(1).strip()] = m.group(2)

    return odds


def import_odds():
    """Import odds from pasted Bovada text. Returns (race_name, count) or None."""
    series_id, series_name = pick_series()

    print(f"\n  Paste odds from Bovada (or type them in).")
    print(f"  Supported formats:")
    print(f"    Corey Heim+300           (Bovada direct copy)")
    print(f"    Kyle Larson, -115        (comma-separated)")
    print(f"    Chase Elliott +1200      (space-separated)")
    print(f"  Header lines (race name, date, 'Outright') are auto-skipped.")
    print(f"  When done, type 'done' on a new line and press Enter.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "done":
            break
        lines.append(line)

    if not lines:
        print("  No odds entered. Skipping.")
        return None

    text = "\n".join(lines)
    odds_data = parse_bovada_odds(text)

    if not odds_data:
        print("  Could not parse any odds from input. Check format.")
        return None

    # Show parsed results
    print(f"\n  Parsed {len(odds_data)} drivers:")
    # Sort by odds value for display
    sorted_odds = sorted(odds_data.items(), key=lambda x: float(x[1].replace("+", "")))
    for name, odds_val in sorted_odds[:10]:
        print(f"    {name:30s} {odds_val}")
    if len(sorted_odds) > 10:
        print(f"    ... and {len(sorted_odds) - 10} more")

    confirm = input(f"\n  Does this look correct? (Y/n): ").strip().lower()
    if confirm == "n":
        print("  Cancelled.")
        return None

    # Pick race
    selected = pick_race(series_id, series_name, check_type="odds")
    if not selected:
        return None

    race_id = selected.get("race_id")
    race_name = selected.get("race_name", "Unknown")

    # Check for existing
    existing = check_existing_odds(race_id)
    if existing:
        confirm = input(f"\n  {existing} odds already saved for this race. Overwrite? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Skipped.")
            return None

    # Save to DB
    count = save_odds_to_db(odds_data, race_id, sportsbook="bovada")
    if count and count > 0:
        print(f"\n  Saved {count} odds for {race_name}!")
        return race_name, count, "odds"
    else:
        print(f"\n  Saved odds for {race_name}")
        return race_name, len(odds_data), "odds"


def import_salary(recent_files):
    """Import salaries for one series+race. Returns (race_name, count, type) or None."""
    series_id, series_name = pick_series()

    # Pick CSV
    csv_path = pick_csv(recent_files)
    if not csv_path:
        print("  No file selected. Skipping.")
        return None

    print(f"\n  File: {os.path.basename(csv_path)}")

    # Parse
    platform = detect_platform(csv_path)
    if platform == "dk":
        df = parse_dk_csv(open(csv_path, "rb"))
        sal_col = "DK Salary"
        plat_name = "DraftKings"
    else:
        df = parse_fd_csv(open(csv_path, "rb"))
        sal_col = "FD Salary"
        plat_name = "FanDuel"

    if df.empty:
        print("  Failed to parse CSV — check file format.")
        return None

    print(f"  Platform: {plat_name} | Drivers: {len(df)} | "
          f"Range: ${df[sal_col].min():,} — ${df[sal_col].max():,}")

    # Pick race
    selected = pick_race(series_id, series_name, check_type="salary", platform=plat_name)
    if not selected:
        return None

    race_id = selected.get("race_id")
    race_name = selected.get("race_name", "Unknown")

    # Check for existing and confirm overwrite
    existing = check_existing_salaries(race_id, series_id, plat_name)
    if existing:
        confirm = input(f"\n  {existing} salaries already saved for this race. Overwrite? (y/N): ").strip().lower()
        if confirm != "y":
            print("  Skipped.")
            return None

    # Sync to DB
    if platform == "dk":
        count = sync_dk_salaries_to_db(df, race_id, series_id, race_name)
    else:
        count = sync_fd_salaries_to_db(df, race_id, series_id, race_name)

    if count > 0:
        print(f"\n  Saved {count} {plat_name} salaries for {race_name}!")
        top = df.sort_values(sal_col, ascending=False).head(5)
        for _, row in top.iterrows():
            print(f"    {row['Driver']:30s} ${row[sal_col]:>6,}")
        return race_name, count, "salaries"
    else:
        print(f"\n  WARNING: No salaries saved — race may not be in DB yet.")
        print(f"  Try running: python refresh_data.py")
        return None


def main():
    parser = argparse.ArgumentParser(description="Import DK/FD salaries and Bovada odds")
    parser.add_argument("--no-push", action="store_true", help="Skip git commit + push")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NASCAR DFS — Data Import")
    print(f"{'='*60}")

    recent_files = find_csv_files()
    imported = []

    while True:
        print(f"\n  What would you like to import?")
        print(f"    [1] DK/FD Salaries (from CSV)")
        print(f"    [2] Bovada Odds (paste from website)")
        print(f"    [3] Both (salaries + odds for same race)")
        print(f"    [q] Done — commit & push")

        choice = input(f"\n  Select (1/2/3/q) [1]: ").strip().lower()

        if choice in ("q", "quit", "exit", "done"):
            break

        if choice == "2":
            result = import_odds()
            if result:
                imported.append(result)
        elif choice == "3":
            # Import both for same flow
            print(f"\n  --- Import Salaries ---")
            sal_result = import_salary(recent_files)
            if sal_result:
                imported.append(sal_result)
            print(f"\n  --- Import Odds ---")
            odds_result = import_odds()
            if odds_result:
                imported.append(odds_result)
        else:
            result = import_salary(recent_files)
            if result:
                imported.append(result)

        print(f"\n{'─'*40}")
        if imported:
            print(f"  Imported so far: {len(imported)} item(s)")
            for name, count, dtype in imported:
                print(f"    - {name}: {count} {dtype}")

        more = input("\n  Import more data? (y/N): ").strip().lower()
        if more != "y":
            break

    if not imported:
        print("\n  Nothing imported. Exiting.")
        return

    # Git commit + push all at once
    if not args.no_push:
        # Build commit message from what was imported
        parts = []
        sal_races = [name for name, _, dtype in imported if dtype == "salaries"]
        odds_races = [name for name, _, dtype in imported if dtype == "odds"]
        if sal_races:
            safe = ", ".join(r.replace('"', '').replace("'", "") for r in sal_races)
            parts.append(f"salaries: {safe}")
        if odds_races:
            safe = ", ".join(r.replace('"', '').replace("'", "") for r in odds_races)
            parts.append(f"odds: {safe}")
        commit_msg = "Add " + "; ".join(parts)

        print(f"\n  Committing and pushing to git...")
        ret1 = os.system('git add nascar.db')
        ret2 = os.system(f'git commit -m "{commit_msg}"')
        ret3 = os.system('git push')
        if ret3 == 0:
            print(f"\n  Done! Data is now live on Streamlit Cloud.")
        else:
            print(f"\n  WARNING: Git push may have failed (exit code {ret3}).")
            print(f"  Try manually: git add nascar.db && git commit -m \"Add data\" && git push")
    else:
        print(f"\n  Saved to local DB. To deploy:")
        print(f'    git add nascar.db && git commit -m "Add data" && git push')

    print()


if __name__ == "__main__":
    main()
