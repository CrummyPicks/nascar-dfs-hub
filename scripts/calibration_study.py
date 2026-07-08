"""Finish-projection calibration study.

Question (user): does the model COMPRESS the field — projecting backmarkers
too high (28th when reality is 34th) and frontrunners too low?

Method: replay every backtestable race (odds+results stored) with the same
pre-race-faithful assembly the profit sim uses; collect (projected finish,
actual finish) pairs; measure calibration by projected-finish bucket, the
regression slope of actual on projected, and the spread ratio.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")

import sqlite3
import numpy as np
from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import backtestable_races, load_race, normalize_weights
from scripts.backtest_practice_weight import fetch_practice
from src.projections import compute_projections
from src.utils import normalize_driver_name, fuzzy_match_name

races = backtestable_races()
print(f"backtestable races: {len(races)}")

pairs = []   # (proj_finish, actual_finish, dnf, series, field_size)
conn = sqlite3.connect(str(DB_PATH))
done = 0
for db_id, series_id, season, race_date, track in races:
    try:
        race = load_race(conn, db_id, series_id, track, race_date)
        if not race:
            continue
        api_row = conn.execute("SELECT api_race_id FROM races WHERE id=?",
                               (db_id,)).fetchone()
        api_id = api_row[0] if api_row else None
        prac = fetch_practice(season, series_id, api_id) or {} if api_id else {}
        if prac:
            norm = {normalize_driver_name(d): d for d in race["drivers"]}
            prac = {norm.get(normalize_driver_name(k),
                    fuzzy_match_name(k, race["drivers"]) or k): v
                    for k, v in prac.items()}
        base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        wn = normalize_weights(base, has_odds=bool(race["odds_finish"]),
                               has_prac=bool(prac))
        rows, detail = compute_projections(
            drivers=race["drivers"], field_size=race["field_size"], wn=wn,
            th_data=race["th_data"], tt_data=race["tt_data"],
            qual_pos=race["qual_pos"], practice_data=prac,
            odds_finish=race["odds_finish"], odds_display=race["odds_display"],
            team_signal=race["team_signal"], mfr_adjustment={},
            team_adj_data=race["team_adj"], dnf_data={},
            race_laps=race.get("race_laps", 200), track_name=track,
            track_type=race["track_type"], series_id=series_id,
            calibration=race["calibration"], cross_th_lookup={})
        for r in rows:
            d = r["driver"]
            af = race["actual_finish"].get(d)
            if af is None:
                continue
            status = (race["status"].get(d) or "").lower()
            dnf = bool(status and "running" not in status and status.strip())
            pairs.append((float(r["proj_finish"]), float(af), dnf,
                          series_id, race["field_size"]))
        done += 1
    except Exception:
        continue
conn.close()
print(f"replayed races: {done} | driver-race pairs: {len(pairs)}")

arr = np.array([(p, a, d, s, f) for p, a, d, s, f in pairs])
proj, act, dnf = arr[:, 0], arr[:, 1], arr[:, 2].astype(bool)

def report(mask, label):
    p, a = proj[mask], act[mask]
    if len(p) < 100:
        return
    slope, intercept = np.polyfit(p, a, 1)
    print(f"\n=== {label} (n={len(p)}) ===")
    print(f"regression: actual = {slope:.3f} x projected + {intercept:.2f}"
          f"   (slope>1 => projections compressed)")
    print(f"spread: std(projected)={p.std():.2f} vs std(actual)={a.std():.2f}"
          f"   ratio={a.std()/p.std():.2f}")
    print(f"{'proj bucket':>12} | {'n':>6} | {'mean proj':>9} | "
          f"{'mean actual':>11} | {'bias':>6}")
    for lo, hi in [(1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 25),
                   (25, 30), (30, 43)]:
        m = (p >= lo) & (p < hi)
        if m.sum() < 30:
            continue
        bias = a[m].mean() - p[m].mean()
        print(f"{f'{lo}-{hi}':>12} | {m.sum():>6} | {p[m].mean():>9.1f} | "
              f"{a[m].mean():>11.1f} | {bias:>+6.1f}")

report(np.ones(len(proj), bool), "ALL finishers (DNFs included)")
report(~dnf, "RUNNING only (DNFs excluded)")

# The user's exact claims:
print("\n=== the user's specific claims ===")
m = (proj >= 26) & (proj <= 30)
print(f"projected 26-30: mean actual = {act[m].mean():.1f} (n={m.sum()}) "
      f"| DNF rate {dnf[m].mean()*100:.0f}%")
m2 = (proj >= 1) & (proj <= 6)
print(f"projected 1-6:   mean actual = {act[m2].mean():.1f} (n={m2.sum()}) "
      f"| DNF rate {dnf[m2].mean()*100:.0f}%")
# per-race extremes
print("\nWorst-projected driver per race: how do they actually finish?")
