"""
NASCAR DFS Salary Auto-Scraper
===============================
Automatically pulls upcoming DraftKings and FanDuel NASCAR salaries
directly from their APIs — no CSV export/import required.

How it works
------------
  DraftKings : Public API (no login needed). Discovers upcoming NASCAR
               slates for Cup, Xfinity, and Trucks, then fetches player
               salaries and imports them directly into the database.

  FanDuel    : Uses FD's public fixture-list API. If that endpoint
               requires auth for your account, the script will print a
               clear message and you can fall back to IMPORT_SALARIES.bat.

Usage
-----
  python scrapers/salaries.py                        # DK + FD, all series
  python scrapers/salaries.py --platform DraftKings  # DK only
  python scrapers/salaries.py --series cup           # Cup only
  python scrapers/salaries.py --dry-run              # preview, no DB writes
"""

import sqlite3
import requests
import time
import re
import os
import sys
import argparse
from difflib import get_close_matches
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "nascar.db")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# How far into the future (days) to look for upcoming races
LOOKAHEAD_DAYS = 14


# ── Series detection helpers ──────────────────────────────────────────────────

def detect_series_from_text(text: str) -> str | None:
    """
    Try to figure out which NASCAR series a contest/slate is for
    from the contest name, tags, or description.
    Returns 'cup', 'xfinity', 'trucks', or None.
    """
    t = text.lower()
    if "xfinity" in t or "o'reilly" in t or "oreilly" in t:
        # O'Reilly Auto Parts Series = rebranded Xfinity Series
        return "xfinity"
    if "truck" in t or "craftsman" in t:
        return "trucks"
    if "cup" in t or "nascar" in t:
        return "cup"          # Cup is the default / most common
    return None


# ── Database helpers ──────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def find_driver(conn, name: str) -> tuple[int | None, str | None]:
    """Exact → last-name → fuzzy. Returns (driver_id, matched_name)."""
    clean = name.strip()

    row = conn.execute("SELECT id FROM drivers WHERE full_name=?", (clean,)).fetchone()
    if row:
        return row["id"], clean

    parts = clean.split()
    if parts:
        last = parts[-1]
        rows = conn.execute(
            "SELECT id, full_name FROM drivers WHERE last_name=?", (last,)
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"], rows[0]["full_name"]

    all_rows  = conn.execute("SELECT id, full_name FROM drivers").fetchall()
    all_names = [r["full_name"] for r in all_rows]
    matches   = get_close_matches(clean, all_names, n=1, cutoff=0.75)
    if matches:
        row = conn.execute(
            "SELECT id FROM drivers WHERE full_name=?", (matches[0],)
        ).fetchone()
        return row["id"], matches[0]

    return None, None


def upsert_driver(conn, full_name: str) -> int:
    parts = full_name.strip().split()
    first = parts[0] if parts else ""
    last  = " ".join(parts[1:]) if len(parts) > 1 else ""
    conn.execute(
        "INSERT OR IGNORE INTO drivers(full_name,first_name,last_name) VALUES(?,?,?)",
        (full_name.strip(), first, last),
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM drivers WHERE full_name=?", (full_name.strip(),)
    ).fetchone()["id"]


def find_or_create_race(conn, series_code: str, race_date: str,
                         track_name: str | None) -> int:
    """Find an existing race by date+series, or create one if missing."""
    series_id = conn.execute(
        "SELECT id FROM series WHERE code=?", (series_code,)
    ).fetchone()["id"]

    # Exact date match
    row = conn.execute(
        "SELECT id FROM races WHERE series_id=? AND race_date=?",
        (series_id, race_date),
    ).fetchone()
    if row:
        return row["id"]

    # ±3 day window (handles scraped date vs API date timezone drift)
    row = conn.execute(
        """
        SELECT id, race_date FROM races
        WHERE  series_id=?
          AND  ABS(julianday(race_date) - julianday(?)) <= 3
        ORDER  BY ABS(julianday(race_date) - julianday(?))
        LIMIT  1
        """,
        (series_id, race_date, race_date),
    ).fetchone()
    if row:
        return row["id"]

    # Create minimal race entry
    track_id = None
    if track_name:
        conn.execute(
            "INSERT OR IGNORE INTO tracks(name) VALUES(?)", (track_name,)
        )
        conn.commit()
        track_id = conn.execute(
            "SELECT id FROM tracks WHERE name=?", (track_name,)
        ).fetchone()["id"]

    last_num = conn.execute(
        "SELECT COALESCE(MAX(race_num),0) FROM races WHERE series_id=? AND season=?",
        (series_id, datetime.now().year),
    ).fetchone()[0]

    conn.execute(
        """INSERT INTO races(series_id, track_id, season, race_num, race_date)
           VALUES(?,?,?,?,?)""",
        (series_id, track_id, datetime.now().year, last_num + 1, race_date),
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM races WHERE series_id=? AND season=? AND race_num=?",
        (series_id, datetime.now().year, last_num + 1),
    ).fetchone()["id"]


def upsert_salary(conn, race_id: int, driver_id: int,
                  platform: str, salary: int,
                  status: str = "Available") -> str:
    """Insert or update a salary row. Returns 'inserted', 'updated', 'unchanged'."""
    existing = conn.execute(
        "SELECT id, salary, status FROM salaries WHERE race_id=? AND driver_id=? AND platform=?",
        (race_id, driver_id, platform),
    ).fetchone()

    if existing:
        if existing["salary"] != salary or existing["status"] != status:
            conn.execute(
                "UPDATE salaries SET salary=?, status=? WHERE id=?",
                (salary, status, existing["id"]),
            )
            return "updated"
        return "unchanged"

    conn.execute(
        "INSERT INTO salaries(race_id,driver_id,platform,salary,status) VALUES(?,?,?,?,?)",
        (race_id, driver_id, platform, salary, status),
    )
    return "inserted"


# ── HTTP helper ───────────────────────────────────────────────────────────────

def get_json(url: str, extra_headers: dict | None = None,
             retries: int = 3) -> dict | list | None:
    hdrs = {**HEADERS, **(extra_headers or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=hdrs, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                print(f"    [AUTH REQUIRED] {url}")
                return None
            print(f"    [HTTP {r.status_code}] {url}")
        except Exception as e:
            print(f"    [Error] {e} — attempt {attempt+1}/{retries}")
        time.sleep(2 * (attempt + 1))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFTKINGS
# ═══════════════════════════════════════════════════════════════════════════════

# DraftKings API endpoints (try in order; DK occasionally restructures these)
DK_CONTEST_URLS = [
    "https://www.draftkings.com/lobby/getcontests?sport=NASCAR",
]
DK_DRAFTABLES_URL = "https://api.draftkings.com/draftgroups/v1/draftgroups/{}/draftables"

# DK roster slot names for captain vs flex in NASCAR captain's-choice format
DK_CAPTAIN_SLOTS = {"cpt", "captain"}


def scrape_dk(conn, series_filter: list[str], dry_run: bool) -> None:
    print("\n" + "="*60)
    print("  DraftKings Salary Scraper")
    print("="*60)

    # Try each DK endpoint in order until one succeeds
    data = None
    for url in DK_CONTEST_URLS:
        data = get_json(url)
        if data:
            break

    if not data:
        print("  [ERROR] Could not reach DraftKings API.")
        print("  Possible reasons:")
        print("    • No NASCAR slate posted yet for this week")
        print("    • DraftKings changed their API endpoint")
        print("    • Network / firewall blocking the request")
        print("  You can still use IMPORT_SALARIES.bat to load DK salaries manually.")
        return

    # DK response can be { Contests: [...], DraftGroups: [...] }
    # or { draftGroups: [...] } depending on which endpoint responded
    if isinstance(data, dict):
        draft_groups = (
            data.get("DraftGroups")
            or data.get("draftGroups")
            or []
        )
    else:
        print(f"  [WARN] Unexpected DK response type: {type(data)}")
        return

    # Lobby endpoint returns all sports — filter to NASCAR only
    draft_groups = [
        g for g in draft_groups
        if (g.get("Sport") or g.get("sport") or "").upper() == "NASCAR"
    ]

    if not draft_groups:
        print("  [i] No upcoming NASCAR draft groups found on DraftKings.")
        print("  Salaries may not be posted yet for this week's race.")
        return

    print(f"  Found {len(draft_groups)} draft group(s)\n")

    for group in draft_groups:
        group_id       = group.get("DraftGroupId") or group.get("draftGroupId")
        start_raw      = group.get("StartDate") or group.get("startDate", "")
        contest_type   = (group.get("ContestTypeName") or group.get("contestTypeName") or "")
        game_type_desc = (group.get("GameTypeDescription") or group.get("gameTypeDescription") or "")
        tags           = group.get("Tags") or group.get("tags") or []
        tag_text       = " ".join(str(t) for t in tags)

        # Combine all text for series detection
        full_desc = f"{contest_type} {game_type_desc} {tag_text}"
        series    = detect_series_from_text(full_desc)

        if series is None:
            series = "cup"   # DK defaults to Cup for most NASCAR slates

        if series not in series_filter:
            print(f"  [SKIP] DraftGroup {group_id} — {series} not in requested series")
            continue

        # Parse race date from ISO timestamp
        race_date = None
        if start_raw:
            try:
                dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                race_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                m = re.search(r"(\d{4}-\d{2}-\d{2})", start_raw)
                race_date = m.group(1) if m else None

        print(f"  DraftGroup {group_id} | {series.upper():8} | {race_date or 'date?'} | {contest_type}")

        # Fetch draftables
        url  = DK_DRAFTABLES_URL.format(group_id)
        resp = get_json(url)
        if not resp:
            print(f"    [ERROR] Could not fetch draftables for group {group_id}")
            continue

        draftables = (
            resp.get("draftables") or
            resp.get("Draftables") or
            []
        )
        if not draftables:
            print(f"    [WARN] No draftables returned for group {group_id}")
            print(f"    [FALLBACK] Existing salary data in DB is preserved.")
            continue

        # Extract track name from games info
        track_name = None
        games = resp.get("games") or resp.get("Games") or []
        if games:
            g = games[0]
            track_name = (
                g.get("awayTeamName") or g.get("homeTeamName") or
                g.get("description") or g.get("venue") or None
            )

        # Resolve race in DB
        race_id = None
        if race_date and not dry_run:
            race_id = find_or_create_race(conn, series, race_date, track_name)
            print(f"    Race ID: {race_id} | Date: {race_date}")

        # Parse and import each driver — first pass to count valid FLEX entries
        flex_entries = []
        for player in draftables:
            name   = (player.get("displayName") or player.get("DisplayName") or "").strip()
            salary = player.get("salary") or player.get("Salary")
            slot   = (
                player.get("rosterSlotName") or
                player.get("RosterSlotName") or
                player.get("position") or ""
            ).strip().lower()

            if not name or not salary:
                continue
            if slot in DK_CAPTAIN_SLOTS:
                continue   # skip inflated CPT rows

            # Availability status: DK marks scratched drivers as disabled
            is_disabled = player.get("isDisabled") or player.get("IsDisabled") or False
            draft_statuses = player.get("draftStatuses") or []
            dk_status = "Out" if is_disabled else "Available"
            # Some responses use a draftStatuses list with 'out', 'gtd', etc.
            if not is_disabled and draft_statuses:
                status_str = " ".join(str(s).lower() for s in draft_statuses)
                if "out" in status_str:
                    dk_status = "Out"
                elif "questionable" in status_str or "gtd" in status_str:
                    dk_status = "Questionable"

            flex_entries.append({
                "name":   name,
                "salary": salary,
                "status": dk_status,
            })

        # ── 0-result guard: if API returned nothing usable, keep existing data ──
        if not flex_entries:
            print(f"    [WARN] DK returned 0 valid FLEX entries for group {group_id}.")
            print(f"    [FALLBACK] Existing salary data in DB is preserved — not overwritten.")
            continue

        # ── Second pass: write to DB ───────────────────────────────────────────
        inserted = updated = unchanged = new_drivers = 0
        out_count = 0
        for entry in flex_entries:
            name, salary, dk_status = entry["name"], entry["salary"], entry["status"]

            if dry_run:
                status_tag = f" [{dk_status}]" if dk_status != "Available" else ""
                print(f"    [DRY] {name:32} ${salary:>7,}  ({series.upper()}){status_tag}")
                continue

            driver_id, matched = find_driver(conn, name)
            if driver_id is None:
                driver_id = upsert_driver(conn, name)
                new_drivers += 1

            result = upsert_salary(conn, race_id, driver_id, "DraftKings", salary, dk_status)
            if result == "inserted":    inserted  += 1
            elif result == "updated":   updated   += 1
            else:                       unchanged += 1
            if dk_status == "Out":
                out_count += 1

        if not dry_run:
            conn.commit()
            out_note = f"  Out/scratched: {out_count}" if out_count else ""
            print(f"    Inserted {inserted}  Updated {updated}  "
                  f"Unchanged {unchanged}  New drivers {new_drivers}{out_note}")


# ═══════════════════════════════════════════════════════════════════════════════
# FANDUEL
# ═══════════════════════════════════════════════════════════════════════════════

# FanDuel fixture-list discovery endpoint (public, no auth required to list slates)
FD_FIXTURE_LISTS_URL = "https://api.fanduel.com/fixture-lists"
FD_PLAYERS_URL       = "https://api.fanduel.com/fixture-lists/{}/players"

FD_HEADERS = {
    **HEADERS,
    "Referer": "https://www.fanduel.com",
}

FD_SERIES_MAP = {
    "motor racing": "cup",
    "nascar":       "cup",
}


def detect_fd_series(sport: str, name: str) -> str:
    """Map FD sport/name to our series code."""
    combined = f"{sport} {name}".lower()
    if "xfinity" in combined:
        return "xfinity"
    if "truck" in combined:
        return "trucks"
    return "cup"


def scrape_fd(conn, series_filter: list[str], dry_run: bool) -> None:
    print("\n" + "="*60)
    print("  FanDuel Salary Scraper")
    print("="*60)

    data = get_json(FD_FIXTURE_LISTS_URL, extra_headers=FD_HEADERS)

    if data is None:
        _fd_auth_message()
        return

    # FD response: { fixture_lists: [...] } or { data: { fixture_lists: [...] } }
    if isinstance(data, dict):
        fixture_lists = (
            data.get("fixture_lists") or
            data.get("fixtureLists") or
            (data.get("data") or {}).get("fixture_lists") or
            []
        )
    else:
        fixture_lists = []

    if not fixture_lists:
        print("  [i] No FanDuel fixture lists returned.")
        print("  This may mean no upcoming NASCAR slate is posted yet,")
        print("  or the API requires authentication (try IMPORT_SALARIES.bat).")
        return

    # Filter to NASCAR motor-racing slates
    nascar_slates = []
    for fl in fixture_lists:
        sport = (fl.get("sport") or "").lower()
        name  = (fl.get("label") or fl.get("name") or "").lower()
        if "nascar" in name or "motor" in sport or "racing" in sport:
            nascar_slates.append(fl)

    if not nascar_slates:
        print("  [i] No NASCAR slates found in FanDuel fixture lists.")
        return

    print(f"  Found {len(nascar_slates)} NASCAR slate(s)\n")

    for slate in nascar_slates:
        slate_id   = slate.get("id") or slate.get("fixture_list_id")
        sport      = (slate.get("sport") or "")
        label      = (slate.get("label") or slate.get("name") or "")
        start_raw  = (slate.get("start_date") or slate.get("startDate") or "")
        series     = detect_fd_series(sport, label)

        if series not in series_filter:
            continue

        # Parse race date
        race_date = None
        if start_raw:
            try:
                dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                race_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                m = re.search(r"(\d{4}-\d{2}-\d{2})", start_raw)
                race_date = m.group(1) if m else None

        print(f"  Slate {slate_id} | {series.upper():8} | {race_date or 'date?'} | {label}")

        # Fetch players for this slate
        url  = FD_PLAYERS_URL.format(slate_id)
        resp = get_json(url, extra_headers=FD_HEADERS)
        if not resp:
            print(f"    [ERROR] Could not fetch players — auth may be required.")
            _fd_auth_message()
            continue

        players = (
            resp.get("players") or
            resp.get("Players") or
            []
        )
        if not players:
            print(f"    [WARN] No players returned for slate {slate_id}")
            print(f"    [FALLBACK] Existing salary data in DB is preserved.")
            continue

        # Parse player list — first pass to build entries with status
        fd_entries = []
        for player in players:
            # FD player dict structure can vary; try common field names
            name = (
                player.get("first_name", "") + " " + player.get("last_name", "")
            ).strip()
            if not name or name == " ":
                name = (
                    player.get("name") or
                    player.get("display_name") or
                    ""
                ).strip()

            salary = (
                player.get("salary") or
                player.get("starting_salary") or
                player.get("Salary")
            )

            if not name or not salary:
                continue
            try:
                salary = int(salary)
            except (ValueError, TypeError):
                continue

            # FD injury/status field
            injury_status = (player.get("injury_status") or
                             player.get("injuryStatus") or
                             player.get("status") or "").strip().lower()
            if "out" in injury_status or "scratch" in injury_status:
                fd_status = "Out"
            elif "questionable" in injury_status or "gtd" in injury_status:
                fd_status = "Questionable"
            elif "probable" in injury_status:
                fd_status = "Probable"
            else:
                fd_status = "Available"

            fd_entries.append({"name": name, "salary": salary, "status": fd_status})

        # ── 0-result guard ────────────────────────────────────────────────────
        if not fd_entries:
            print(f"    [WARN] FD returned 0 valid players for slate {slate_id}.")
            print(f"    [FALLBACK] Existing salary data in DB is preserved.")
            continue

        # Resolve race
        race_id = None
        if race_date and not dry_run:
            race_id = find_or_create_race(conn, series, race_date, None)
            print(f"    Race ID: {race_id} | Date: {race_date}")

        inserted = updated = unchanged = new_drivers = 0
        out_count = 0
        for entry in fd_entries:
            name, salary, fd_status = entry["name"], entry["salary"], entry["status"]

            if dry_run:
                status_tag = f" [{fd_status}]" if fd_status != "Available" else ""
                print(f"    [DRY] {name:32} ${salary:>7,}  ({series.upper()}){status_tag}")
                continue

            driver_id, _ = find_driver(conn, name)
            if driver_id is None:
                driver_id = upsert_driver(conn, name)
                new_drivers += 1

            result = upsert_salary(conn, race_id, driver_id, "FanDuel", salary, fd_status)
            if result == "inserted":    inserted  += 1
            elif result == "updated":   updated   += 1
            else:                       unchanged += 1
            if fd_status == "Out":
                out_count += 1

        if not dry_run:
            conn.commit()
            out_note = f"  Out/scratched: {out_count}" if out_count else ""
            print(f"    Inserted {inserted}  Updated {updated}  "
                  f"Unchanged {unchanged}  New drivers {new_drivers}{out_note}")


def _fd_auth_message() -> None:
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  FanDuel requires login to access salary data.      │")
    print("  │                                                     │")
    print("  │  Manual steps (takes ~60 seconds):                  │")
    print("  │  1. Go to fanduel.com → DFS lobby → NASCAR contest │")
    print("  │  2. Click 'Edit Lineups' → 'Export Players List'   │")
    print("  │  3. Double-click  IMPORT_SALARIES.bat              │")
    print("  │     and drag the downloaded CSV onto the window     │")
    print("  └─────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-scrape upcoming DK/FD NASCAR salaries into nascar.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scrapers/salaries.py
  python scrapers/salaries.py --platform DraftKings
  python scrapers/salaries.py --series cup xfinity
  python scrapers/salaries.py --dry-run
        """,
    )
    parser.add_argument(
        "--platform", nargs="+", default=["DraftKings", "FanDuel"],
        choices=["DraftKings", "FanDuel"],
        help="Which platform(s) to scrape (default: both)",
    )
    parser.add_argument(
        "--series", nargs="+", default=["cup", "xfinity", "trucks"],
        choices=["cup", "xfinity", "trucks"],
        help="Which series to scrape (default: all three)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and preview without writing to database",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  NASCAR DFS  |  Salary Auto-Scraper")
    print(f"  Platforms : {', '.join(args.platform)}")
    print(f"  Series    : {', '.join(s.upper() for s in args.series)}")
    if args.dry_run:
        print(f"  Mode      : DRY RUN")
    print("="*60)

    conn = get_conn()

    if "DraftKings" in args.platform:
        scrape_dk(conn, series_filter=args.series, dry_run=args.dry_run)

    if "FanDuel" in args.platform:
        scrape_fd(conn, series_filter=args.series, dry_run=args.dry_run)

    conn.close()
    print("\n[DONE] Salary scrape complete.\n")


if __name__ == "__main__":
    main()
