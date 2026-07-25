"""Capture practice/qualifying lap-by-lap times from NASCAR's live feed.

NASCAR does NOT archive per-lap practice times anywhere (verified 2026-07-24:
lap-times.json holds race laps only; the lap-averages feed has only
best-N-CONSECUTIVE windows; all run-id URL variants 403). The only per-lap
signal is the live feed's `last_lap_time`, which changes as each car crosses
the line — so lap-by-lap data can only exist if something polls the feed
DURING the session and records each new lap. That is what this script does
(and how the third-party practice spreadsheets are built).

Run it while a practice session is live:

    python scripts/capture_practice_laps.py --series-id 1 --race-id 5619

Laps land in the `captured_laps` table of nascar.db, keyed so re-runs and
overlapping captures are harmless (INSERT OR IGNORE). The Practice tab picks
them up automatically: "Best X Laps" display + Lap Chart.

Limitation: one poll sees only the most recent lap per car. At the default
3s interval no lap is missed (green-flag laps are 15s+ everywhere); if the
network hiccups longer than a lap, that lap's time is lost (the gap stays
visible as a missing lap_number).
"""
import argparse
import datetime
import os
import re
import sqlite3
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests


def _local_clean_name(name: str) -> str:
    """Standalone mirror of src.data._clean_api_name — the CI capture
    environment installs only `requests`, so src.data (pandas/streamlit)
    may not be importable. Keep the rules in sync."""
    if not name:
        return ""
    name = name.strip()
    prev = None
    while prev != name:
        prev = name
        name = re.sub(r'^\*\s*', '', name)
        name = re.sub(r'\s*#$', '', name)
        name = re.sub(r'\s*\([a-zA-Z]\)$', '', name)
        name = name.strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.replace(".", "")
    return " ".join(name.split())


try:
    from src.config import DB_PATH
    from src.data import _clean_api_name
except Exception:                                    # minimal CI environment
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "nascar.db")
    _clean_api_name = _local_clean_name

DDL = """
CREATE TABLE IF NOT EXISTS captured_laps (
    season         INTEGER NOT NULL,
    series_id      INTEGER NOT NULL,
    api_race_id    INTEGER NOT NULL,
    run_name       TEXT    NOT NULL,
    vehicle_number TEXT    NOT NULL,
    driver         TEXT,
    lap_number     INTEGER NOT NULL,
    lap_time       REAL,
    lap_speed      REAL,
    flag_state     INTEGER,
    captured_at    TEXT,
    PRIMARY KEY (api_race_id, run_name, vehicle_number, lap_number)
)
"""


def capture(series_id: int, race_id: int, season: int, interval: float,
            duration_min: float, idle_min: float, no_start_min: float = 90.0,
            label: str = ""):
    url = f"https://cf.nascar.com/cacher/live/series_{series_id}/{race_id}/live-feed.json"
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(DDL)
    conn.commit()

    tag = f"[{label}] " if label else ""
    last_laps = {}   # vehicle_number -> laps_completed at last poll
    total_new = 0
    started = time.monotonic()
    last_new_lap_at = time.monotonic()
    print(f"{tag}Polling {url} every {interval:.0f}s (Ctrl+C to stop)")

    while True:
        if time.monotonic() - started > duration_min * 60:
            print(f"{tag}Duration limit ({duration_min:.0f} min) reached.")
            break
        if total_new and time.monotonic() - last_new_lap_at > idle_min * 60:
            print(f"{tag}No new laps for {idle_min:.0f} min - session over, stopping.")
            break
        if not total_new and time.monotonic() - started > no_start_min * 60:
            print(f"{tag}No laps seen in {no_start_min:.0f} min - "
                  "session never went live here, stopping.")
            break
        try:
            feed = requests.get(url, timeout=10).json()
        except Exception as e:
            print(f"poll error: {e}")
            time.sleep(interval)
            continue

        run_name = str(feed.get("run_name") or "unknown")
        run_type = feed.get("run_type")
        flag = feed.get("flag_state")
        now = datetime.datetime.now().isoformat(timespec="seconds")
        new_this_poll = 0

        # run_type 3 = race; its laps are archived by NASCAR, skip those
        if run_type != 3:
            for v in feed.get("vehicles") or []:
                num = str(v.get("vehicle_number") or "").strip().lstrip("*")
                laps_done = v.get("laps_completed")
                lt = v.get("last_lap_time")
                if not num or laps_done is None:
                    continue
                drv = v.get("driver")
                if isinstance(drv, dict):
                    drv = drv.get("full_name")
                drv = _clean_api_name(str(drv or ""))
                prev = last_laps.get(num)
                last_laps[num] = laps_done
                if prev is None:
                    # First sight of this car. If it already posted laps
                    # before we started (mid-session join, or the session
                    # ran ahead of schedule), that history is gone from the
                    # feed — but its BEST lap is still visible. Record it so
                    # the driver isn't missing entirely (Xfinity quals
                    # 2026-07-25: 21 of 38 cars ran before the poller
                    # joined and were lost without this).
                    bt = v.get("best_lap_time")
                    if laps_done > 0 and bt and bt > 0:
                        cur = conn.execute(
                            "INSERT OR IGNORE INTO captured_laps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (season, series_id, race_id, run_name, num, drv,
                             int(laps_done), float(bt), v.get("best_lap_speed"),
                             flag, now))
                        if cur.rowcount:
                            new_this_poll += 1
                            print(f"{tag}  seed {int(laps_done):>3}  #{num:<4} {drv:<24} {bt:.3f} (best, pre-join)")
                    continue
                if laps_done <= prev:
                    continue
                if not lt or lt <= 0:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO captured_laps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (season, series_id, race_id, run_name, num, drv,
                     int(laps_done), float(lt), v.get("last_lap_speed"),
                     flag, now))
                if cur.rowcount:
                    new_this_poll += 1
                    print(f"{tag}  lap {int(laps_done):>3}  #{num:<4} {drv:<24} {lt:.3f}")
        if new_this_poll:
            conn.commit()
            total_new += new_this_poll
            last_new_lap_at = time.monotonic()
        time.sleep(interval)

    conn.commit()
    n = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT vehicle_number) FROM captured_laps "
        "WHERE api_race_id=?", (race_id,)).fetchone()
    conn.close()
    print(f"{tag}Done. {total_new} new laps this run; "
          f"{n[0]} laps / {n[1]} cars stored for race {race_id}.")
    return total_new


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--series-id", type=int, required=True, help="1=Cup 2=Xfinity 3=Trucks")
    ap.add_argument("--race-id", type=int, required=True, help="NASCAR api race id")
    ap.add_argument("--year", type=int, default=datetime.date.today().year)
    ap.add_argument("--interval", type=float, default=3.0, help="poll seconds")
    ap.add_argument("--duration", type=float, default=180.0, help="max minutes to run")
    ap.add_argument("--idle", type=float, default=20.0,
                    help="stop after this many minutes with no new laps")
    ap.add_argument("--no-start", type=float, default=90.0,
                    help="stop if NO laps at all after this many minutes")
    a = ap.parse_args()
    capture(a.series_id, a.race_id, a.year, a.interval, a.duration, a.idle,
            a.no_start)


if __name__ == "__main__":
    main()
