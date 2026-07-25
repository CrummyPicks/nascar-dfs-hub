"""Unattended practice/qualifying lap capture — the GitHub Actions driver.

The problem this solves: NASCAR never archives per-lap practice times, so
laps must be recorded live (see capture_practice_laps.py) — but nobody can
be at a PC for every session. This script makes the capture fully
automatic when run on a schedule (the capture-practice workflow crons it
every 30 minutes around race weekends):

1. SCAN  — one fetch per series of race_list_basic.json, whose per-race
   `schedule` arrays carry every session's start_time_utc. Collect
   practice + qualifying sessions that are live now (started < 100 min
   ago) or start within --window minutes.
2. WAIT  — sleep until the earliest upcoming session start.
3. RECORD — run the capture loop for every found session (threads when
   two series run at different tracks simultaneously). Each capture
   stops itself: ~15 min after laps stop (session over), or ~35 min of
   never seeing a lap (rain delay / schedule slip — the next cron tick
   rescans and picks the session up again once it's actually live).
4. The workflow then commits nascar.db; duplicate laps across overlapping
   runs are impossible (INSERT OR IGNORE on a per-lap primary key).

Runs with only `requests` + stdlib (no pandas/streamlit) so the CI job
starts fast. Also usable manually: python scripts/capture_watchdog.py
"""
import argparse
import datetime
import json
import os
import sqlite3
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from scripts.capture_practice_laps import DDL, DB_PATH, capture

RACE_LIST = "https://cf.nascar.com/cacher/{year}/race_list_basic.json"
SERIES = {1: "Cup", 2: "Xfinity", 3: "Trucks"}


def _utcnow():
    # Naive UTC — matches the naive datetimes parsed from start_time_utc
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def scan_sessions(window_min: float, year: int):
    """Practice/qualifying sessions live now or starting within window_min.

    Returns [{series_id, race_id, race_name, event, start_utc}] sorted by
    start time. Uses each race's schedule[] from race_list_basic.json —
    run_type 1 = practice, 2 = qualifying, and start_time_utc is genuine
    UTC (verified against real session times).
    """
    try:
        data = requests.get(RACE_LIST.format(year=year), timeout=20).json()
    except Exception as e:
        print(f"race list fetch failed: {e}")
        return []
    now = _utcnow()
    lo = now - datetime.timedelta(minutes=100)   # still-live lookback
    hi = now + datetime.timedelta(minutes=window_min)
    found = []
    for sid in (1, 2, 3):
        for race in data.get(f"series_{sid}", []) or []:
            for ev in race.get("schedule") or []:
                name = str(ev.get("event_name") or "")
                rt = ev.get("run_type")
                is_session = rt in (1, 2) or any(
                    k in name.lower() for k in ("practice", "qualif"))
                if not is_session:
                    continue
                ts = ev.get("start_time_utc")
                if not ts:
                    continue
                try:
                    start = datetime.datetime.fromisoformat(str(ts)[:19])
                except ValueError:
                    continue
                if lo <= start <= hi:
                    found.append({
                        "series_id": sid,
                        "race_id": race.get("race_id"),
                        "race_name": race.get("race_name"),
                        "event": name,
                        "start_utc": start,
                    })
    found.sort(key=lambda s: s["start_utc"])
    # One capture per (series, race) even if practice AND qualifying both
    # fall in the window — the capture loop just keeps recording across runs.
    seen, unique = set(), []
    for s in found:
        k = (s["series_id"], s["race_id"])
        if k not in seen:
            seen.add(k)
            unique.append(s)
    return unique


def export_laps(path: str):
    """Dump the whole captured_laps table to JSON (workflow conflict
    recovery: export -> reset to origin -> import -> commit)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(DDL)
    rows = conn.execute("SELECT * FROM captured_laps").fetchall()
    cols = [d[1] for d in conn.execute("PRAGMA table_info(captured_laps)")]
    conn.close()
    with open(path, "w") as f:
        json.dump({"columns": cols, "rows": rows}, f)
    print(f"exported {len(rows)} captured laps -> {path}")


def import_laps(path: str):
    with open(path) as f:
        data = json.load(f)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(DDL)
    ph = ",".join("?" * len(data["columns"]))
    conn.executemany(
        f"INSERT OR IGNORE INTO captured_laps VALUES ({ph})", data["rows"])
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM captured_laps").fetchone()[0]
    conn.close()
    print(f"imported; captured_laps now holds {n} rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=float, default=40.0,
                    help="look-ahead minutes for upcoming sessions")
    ap.add_argument("--year", type=int, default=datetime.date.today().year)
    ap.add_argument("--scan-only", action="store_true",
                    help="print found sessions and exit")
    ap.add_argument("--export-json", metavar="PATH",
                    help="dump captured_laps to JSON and exit")
    ap.add_argument("--import-json", metavar="PATH",
                    help="INSERT OR IGNORE a JSON dump and exit")
    a = ap.parse_args()

    if a.export_json:
        export_laps(a.export_json)
        return
    if a.import_json:
        import_laps(a.import_json)
        return

    sessions = scan_sessions(a.window, a.year)
    if not sessions:
        print("No practice/qualifying sessions live or starting within "
              f"{a.window:.0f} min. Nothing to do.")
        return
    for s in sessions:
        print(f"found: {SERIES[s['series_id']]:<7} {s['event']:<24} "
              f"{s['start_utc']:%Y-%m-%d %H:%M} UTC  {s['race_name']}")
    if a.scan_only:
        return

    # Wait for the earliest start (sessions already live have start in past)
    earliest = min(s["start_utc"] for s in sessions)
    wait = (earliest - _utcnow()).total_seconds() - 120   # start 2 min early
    if wait > 0:
        print(f"sleeping {wait/60:.1f} min until first session...")
        time.sleep(wait)

    threads = []
    for s in sessions:
        label = f"{SERIES[s['series_id']]}"
        t = threading.Thread(
            target=capture,
            kwargs=dict(series_id=s["series_id"], race_id=s["race_id"],
                        season=a.year, interval=3.0, duration_min=150.0,
                        idle_min=15.0, no_start_min=35.0, label=label),
            daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1)   # stagger the pollers
    for t in threads:
        t.join()
    print("all captures finished")


if __name__ == "__main__":
    main()
