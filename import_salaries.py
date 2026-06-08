"""
Import DraftKings/FanDuel salary CSVs and sportsbook odds into the database.

Interactive loop: import salaries and/or odds for multiple series, then commit + push once.

Usage:
    python import_salaries.py              # Interactive loop (recommended)
    python import_salaries.py --no-push    # Import without git push
"""

import argparse
import glob
import os
import re
import subprocess
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


def check_existing_odds(race_id, series_id=None):
    """Check if odds already exist in DB for this race."""
    from src.data import load_race_odds
    odds = load_race_odds(race_id, series_id)
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

    # Show upcoming first (most common use case), then recent for backfill
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
            marker = " <-- next" if idx == 1 else ""
            print(f"    [{idx}] {date} — {name} @ {track}{status}{marker}")

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

    choice = input(f"\n  Select race [1]: ").strip()
    idx = int(choice) - 1 if choice.isdigit() else 0
    if idx < 0 or idx >= len(all_races):
        idx = 0
    return all_races[idx]


def parse_odds(text):
    """Parse odds from sportsbook copy-paste text.

    Supports formats:
        Corey Heim+300          (no-space direct copy)
        Kyle Larson, -115       (comma-separated)
        Chase Elliott +1200     (space-separated)
        Connor Zilisch EVEN     (pick'em — also EV / PK)

    Auto-skips header lines (race name, date, time, "Outright", etc.)
    """
    from src.utils import parse_american_odds

    odds = {}
    skip_patterns = re.compile(
        r'^(outright|futures?|top\s*\d|moneyline|head.to.head'
        r'|\d{1,2}/\d{1,2}/\d{2,4}'
        r'|\d{1,2}:\d{2}\s*(am|pm)?'
        r')$', re.IGNORECASE
    )
    # Odds tail: signed/unsigned integer OR "EVEN"/"EV"/"PK"/"Pick'em"
    ODDS_RE = r'(?:[+-]?\d+|even|evens|ev|pk|pick(?:\'?em)?)'
    has_odds_re = re.compile(ODDS_RE, re.IGNORECASE)
    csv_odds_re = re.compile(rf'^{ODDS_RE}$', re.IGNORECASE)
    trail_odds_re = re.compile(rf'^(.+?)\s*({ODDS_RE})$', re.IGNORECASE)

    def _store(name, raw):
        """Normalize the parsed odds string to "+N"/"-N" form."""
        v = parse_american_odds(raw)
        if v is None:
            return False
        odds[name] = f"+{v}" if v >= 0 else str(v)
        return True

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if skip_patterns.match(line):
            continue
        if not has_odds_re.search(line):
            continue
        # Comma-separated: "Driver Name, +350"  (or ", EVEN")
        if "," in line:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and csv_odds_re.match(parts[1]):
                if _store(parts[0], parts[1]):
                    continue
        # Trailing odds with optional whitespace — handles
        # "Driver Name +350", "DriverName+300", and "Connor Zilisch EVEN".
        m = trail_odds_re.match(line)
        if m:
            name = m.group(1).strip().rstrip(",")
            if name:
                _store(name, m.group(2))

    return odds


def import_odds():
    """Import odds from pasted sportsbook text. Returns (race_name, count) or None."""
    series_id, series_name = pick_series()

    print(f"\n  Paste odds from your sportsbook (or type them in).")
    print(f"  Supported formats:")
    print(f"    Corey Heim+300           (no-space direct copy)")
    print(f"    Kyle Larson, -115        (comma-separated)")
    print(f"    Chase Elliott +1200      (space-separated)")
    print(f"  Header lines (race name, date, 'Outright') are auto-skipped.")
    print(f"  When done, press Enter twice (blank line) or type 'done'.\n")

    lines = []
    blank_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "done":
            break
        if not line.strip():
            blank_count += 1
            # Two consecutive blanks or one blank after we have data = done
            if lines and blank_count >= 1:
                break
            continue
        blank_count = 0
        lines.append(line)

    if not lines:
        print("  No odds entered. Skipping.")
        return None

    text = "\n".join(lines)
    odds_data = parse_odds(text)

    if not odds_data:
        print("  Could not parse any odds from input. Check format.")
        return None

    # Show parsed results
    print(f"\n  Parsed {len(odds_data)} drivers:")
    # Sort by odds value (lowest = favorite first). parse_american_odds
    # returns None for unparseable values; sort those to the back.
    from src.utils import parse_american_odds
    def _sortkey(item):
        v = parse_american_odds(item[1])
        return (v is None, v if v is not None else 0)
    sorted_odds = sorted(odds_data.items(), key=_sortkey)
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
    existing = check_existing_odds(race_id, series_id)
    if existing:
        print(f"\n  {existing} odds already saved for this race.")
        print(f"    [r] Replace — clear existing, then save new (use this if Cup odds")
        print(f"        were accidentally saved to an O'Reilly slot, etc.)")
        print(f"    [m] Merge — keep existing, only update overlapping drivers")
        print(f"    [n] Cancel")
        choice = input(f"  How should we handle? (r/m/n) [r]: ").strip().lower() or "r"
        if choice == "n":
            print("  Cancelled.")
            return None
        if choice == "r":
            from src.data import clear_race_odds
            n_cleared = clear_race_odds(race_id, series_id=series_id)
            print(f"  Cleared {n_cleared} stale odds rows.")
        # else: 'm' falls through to merge (current save_odds_to_db behavior)

    # Save to DB
    count = save_odds_to_db(odds_data, race_id, sportsbook="import", series_id=series_id)
    if count and count > 0:
        print(f"\n  Saved {count} odds for {race_name}!")
        return race_name, count, "odds"
    else:
        print(f"\n  WARNING: 0 odds saved to DB!")
        print(f"  The race may not have resolved to a DB entry.")
        print(f"  Try running: python scripts/refresh_data.py")
        return None


def clear_odds():
    """Standalone flow to wipe odds for a race. Useful when odds were
    saved to the wrong race / series and need to be removed entirely.

    Returns (race_name, count, "odds-cleared") or None.
    """
    from src.data import clear_race_odds

    series_id, series_name = pick_series()
    selected = pick_race(series_id, series_name, check_type="odds")
    if not selected:
        return None

    race_id = selected.get("race_id")
    race_name = selected.get("race_name", "Unknown")

    existing = check_existing_odds(race_id, series_id)
    if existing == 0:
        print(f"\n  No odds saved for {race_name} — nothing to clear.")
        return None

    print(f"\n  {existing} odds rows currently saved for: {race_name}")
    confirm = input(f"  Permanently delete all of them? (y/N): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return None

    n = clear_race_odds(race_id, series_id=series_id)
    if n > 0:
        print(f"\n  Cleared {n} odds rows for {race_name}.")
        return race_name, n, "odds-cleared"
    print(f"\n  Nothing was deleted (race resolution may have failed).")
    return None


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
        print(f"  Try running: python scripts/refresh_data.py")
        return None


def _sync_db_with_origin():
    """Pull origin's latest nascar.db BEFORE importing, so the salary/odds rows
    you add layer on top of the daily auto-refresh Action instead of racing it.

    nascar.db is a single binary blob, so git can't merge two versions — last
    push wins. Importing into a STALE local DB and pushing would clobber the
    Action's fresh results. Fetching + resetting to origin first guarantees the
    import lands on the newest data and the push fast-forwards cleanly. Safe:
    only resets when the working tree is clean (never discards uncommitted work)."""
    project_dir = os.path.dirname(os.path.abspath(__file__))

    def _git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              cwd=project_dir)

    print("\n  Syncing local database with GitHub (so your import lands on the "
          "latest data)...")
    if _git("fetch", "origin").returncode != 0:
        print("  ! Could not reach GitHub — continuing with the local database.")
        return
    behind = _git("rev-list", "--count", "HEAD..origin/main").stdout.strip() or "0"
    ahead = _git("rev-list", "--count", "origin/main..HEAD").stdout.strip() or "0"
    if behind == "0" and ahead == "0":
        print("  Already up to date.")
        return
    if _git("status", "--porcelain").stdout.strip():
        print("  ! You have uncommitted changes — skipping auto-sync so nothing "
              "is lost. Commit/stash them (or run `git pull`) and retry.")
        return
    # Any local-ahead commits here are superseded auto-generated DB blobs; origin's
    # daily refresh is the authoritative superset, so reset to it.
    if _git("reset", "--hard", "origin/main").returncode == 0:
        print(f"  OK — local DB synced with GitHub (was behind {behind}, ahead {ahead}).")
    else:
        print("  ! Sync failed — continuing with the local database.")


def main():
    parser = argparse.ArgumentParser(description="Import DK/FD salaries and sportsbook odds")
    parser.add_argument("--no-push", action="store_true", help="Skip git commit + push")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip the pre-import git sync with origin")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NASCAR DFS — Data Import")
    print(f"{'='*60}")

    # Sync with origin FIRST so imports layer on the latest auto-refreshed DB.
    if not (args.no_push or args.no_sync):
        _sync_db_with_origin()

    recent_files = find_csv_files()
    imported = []

    while True:
        print(f"\n  What would you like to do?")
        print(f"    [1] Import DK/FD Salaries (from CSV)")
        print(f"    [2] Import Sportsbook Odds (paste from website)")
        print(f"    [3] Import Both (salaries + odds for same race)")
        print(f"    [4] Clear odds for a race (wrong race / wrong series)")
        print(f"    [q] Done — commit & push")

        choice = input(f"\n  Select (1/2/3/4/q) [1]: ").strip().lower()

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
        elif choice == "4":
            result = clear_odds()
            if result:
                imported.append(result)
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

    # Check if nascar.db has uncommitted changes (even if nothing new was imported this run)
    db_dirty = False
    try:
        result = subprocess.run(["git", "diff", "--name-only", "nascar.db"],
                                capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if "nascar.db" in result.stdout:
            db_dirty = True
        # Also check staged
        result2 = subprocess.run(["git", "diff", "--cached", "--name-only", "nascar.db"],
                                 capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if "nascar.db" in result2.stdout:
            db_dirty = True
    except Exception:
        db_dirty = bool(imported)  # fallback: assume dirty if we imported anything

    if not imported and not db_dirty:
        print("\n  Nothing imported and no pending DB changes. Exiting.")
        input("\n  Press Enter to close...")
        return

    if not imported and db_dirty:
        print("\n  No new imports this session, but nascar.db has uncommitted changes.")
        push_anyway = input("  Push pending DB changes to Streamlit Cloud? (Y/n): ").strip().lower()
        if push_anyway == "n":
            input("\n  Press Enter to close...")
            return

    # Git commit + push all at once
    if not args.no_push:
        # Build commit message from what was imported
        if imported:
            parts = []
            sal_races = [name for name, _, dtype in imported if dtype == "salaries"]
            odds_races = [name for name, _, dtype in imported if dtype == "odds"]
            cleared_races = [name for name, _, dtype in imported if dtype == "odds-cleared"]
            if sal_races:
                safe = ", ".join(r.replace('"', '').replace("'", "") for r in sal_races)
                parts.append(f"salaries: {safe}")
            if odds_races:
                safe = ", ".join(r.replace('"', '').replace("'", "") for r in odds_races)
                parts.append(f"odds: {safe}")
            if cleared_races:
                safe = ", ".join(r.replace('"', '').replace("'", "") for r in cleared_races)
                parts.append(f"cleared odds: {safe}")
            verb = "Update" if cleared_races and not (sal_races or odds_races) else "Add"
            commit_msg = f"{verb} " + "; ".join(parts)
        else:
            commit_msg = "Update nascar.db"

        project_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"\n  Committing and pushing to git...")
        print(f"  Working directory: {project_dir}")

        # Step 1: git add
        print(f"\n  [1/3] git add nascar.db")
        r1 = subprocess.run(["git", "add", "nascar.db"],
                            capture_output=True, text=True, cwd=project_dir)
        if r1.returncode != 0:
            print(f"  ERROR: git add failed (code {r1.returncode})")
            print(f"  stdout: {r1.stdout.strip()}")
            print(f"  stderr: {r1.stderr.strip()}")
            input("\n  Press Enter to close...")
            return
        print(f"  OK")

        # Step 2: git commit
        print(f"\n  [2/3] git commit -m \"{commit_msg}\"")
        r2 = subprocess.run(["git", "commit", "-m", commit_msg],
                            capture_output=True, text=True, cwd=project_dir)
        if r2.returncode != 0:
            # Check if it's just "nothing to commit"
            if "nothing to commit" in r2.stdout or "nothing to commit" in r2.stderr:
                print(f"  Nothing to commit — DB already matches last commit.")
                print(f"  Checking if push is needed...")
            else:
                print(f"  ERROR: git commit failed (code {r2.returncode})")
                print(f"  stdout: {r2.stdout.strip()}")
                print(f"  stderr: {r2.stderr.strip()}")
                input("\n  Press Enter to close...")
                return
        else:
            print(f"  OK: {r2.stdout.strip()}")

        # Step 3: git push
        print(f"\n  [3/3] git push")
        r3 = subprocess.run(["git", "push"],
                            capture_output=True, text=True, cwd=project_dir)
        if r3.returncode == 0:
            push_out = r3.stdout.strip() or r3.stderr.strip()
            print(f"  OK: {push_out}")
            print(f"\n  Done! Data is now live on Streamlit Cloud.")
        else:
            print(f"  ERROR: git push failed (code {r3.returncode})")
            print(f"  stdout: {r3.stdout.strip()}")
            print(f"  stderr: {r3.stderr.strip()}")
            if "fetch first" in (r3.stderr or "") or "rejected" in (r3.stderr or ""):
                print(f"\n  GitHub moved ahead (the daily auto-refresh pushed while "
                      f"you were importing). Just re-run this importer — it syncs "
                      f"first, so re-import the same race and it'll push cleanly.")
            else:
                print(f"\n  Try manually:")
                print(f'    cd "{project_dir}"')
                print(f'    git add nascar.db && git commit -m "Add data" && git push')
    else:
        print(f"\n  Saved to local DB. To deploy:")
        print(f'    git add nascar.db && git commit -m "Add data" && git push')

    input("\n  Press Enter to close...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n  UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\n  Press Enter to close...")
