"""Probe: where do backmarker raw_finish scores come from?

Replays backtestable races; for every driver-race pair collects the blended
raw_finish, per-signal normalized values, presence flags and trust inputs.
Then slices deep ACTUAL finishers (>=25) and the raw~22 blob to attribute
which signals hold them mid.  READ-ONLY on the DB.
"""
import sys, warnings, sqlite3, statistics
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")

import numpy as np
from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import backtestable_races, load_race, normalize_weights
from src.projections import compute_projections

races = backtestable_races()
conn = sqlite3.connect(str(DB_PATH))
recs = []
done = 0
for db_id, series_id, season, race_date, track in races:
    try:
        race = load_race(conn, db_id, series_id, track, race_date)
        if not race:
            continue
        base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        wn = normalize_weights(base, has_odds=bool(race["odds_finish"]), has_prac=False)
        rows, detail, sig = compute_projections(
            return_signal_details=True,
            drivers=race["drivers"], field_size=race["field_size"], wn=wn,
            th_data=race["th_data"], tt_data=race["tt_data"],
            qual_pos=race["qual_pos"], practice_data={},
            odds_finish=race["odds_finish"], odds_display=race["odds_display"],
            team_signal=race["team_signal"], mfr_adjustment={},
            team_adj_data=race["team_adj"], dnf_data={},
            race_laps=race.get("race_laps", 200), track_name=track,
            track_type=race["track_type"], series_id=series_id,
            calibration=race["calibration"], cross_th_lookup={})
        fs = race["field_size"]
        for r in rows:
            d = r["driver"]
            af = race["actual_finish"].get(d)
            if af is None:
                continue
            s = sig.get(d, {})
            th = race["th_data"].get(d) or {}
            tt = race["tt_data"].get(d) or {}
            recs.append({
                "raw": r["raw_finish"], "act": float(af), "fs": fs,
                "proj_fin": r["proj_finish"],
                "sig_track": s.get("Track"), "sig_tt": s.get("TType"),
                "sig_qual": s.get("Qual"), "sig_team": s.get("Team"),
                "sig_odds": s.get("Odds"), "net": s.get("Net Sig"),
                "lowinfo": s.get("LowInfo"),
                "th_races": th.get("races", 0), "tt_races": tt.get("races", 0),
                "has_odds": s.get("Odds") is not None,
                "start": r["start"], "series": series_id,
            })
        done += 1
    except Exception:
        continue
conn.close()
print(f"races replayed: {done}  pairs: {len(recs)}")

# normalize to a 38-car scale so different field sizes compare
for r in recs:
    k = 38.0 / r["fs"]
    r["rawn"] = r["raw"] * k
    r["actn"] = r["act"] * k

deep = [r for r in recs if r["actn"] >= 25]
print(f"\ndeep actual finishers (act>=25 scaled): {len(deep)}")
import collections
hist = collections.Counter(int(r["rawn"] // 2) * 2 for r in deep)
for b in sorted(hist):
    print(f"  raw {b:>2}-{b+2:<2}: {hist[b]:>4} {'#'*(hist[b]//15)}")

blob = [r for r in deep if 20 <= r["rawn"] < 25]
deepread = [r for r in deep if r["rawn"] >= 28]
print(f"\nblob (raw 20-25, actual>=25): {len(blob)} | deep reads (raw>=28): {len(deepread)}")

def prof(rows, label):
    if not rows:
        return
    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{statistics.mean(vals):5.1f}({len(vals)})" if vals else "  -  "
    n = len(rows)
    print(f"\n-- {label} (n={n}) --")
    print(f"  mean actual {statistics.mean(r['actn'] for r in rows):.1f} | mean raw {statistics.mean(r['rawn'] for r in rows):.1f} | mean proj_fin {statistics.mean(r['proj_fin']*38/r['fs'] for r in rows):.1f}")
    print(f"  sig means: track {avg('sig_track')} tt {avg('sig_tt')} qual {avg('sig_qual')} team {avg('sig_team')} odds {avg('sig_odds')}")
    print(f"  has_odds {sum(r['has_odds'] for r in rows)/n:.0%} | lowinfo {sum(1 for r in rows if r['lowinfo'])/n:.0%}")
    print(f"  th_races med {statistics.median(r['th_races'] for r in rows):.0f} | tt_races med {statistics.median(r['tt_races'] for r in rows):.0f}")
    print(f"  start mean {statistics.mean(r['start']*38/r['fs'] for r in rows):.1f}")

prof(blob, "BLOB raw 20-25 but finished 25+")
prof(deepread, "correctly deep raw>=28, finished 25+")

# among the blob: which signal values sit mid?
mid_pull = {k: 0 for k in ["sig_track", "sig_tt", "sig_qual", "sig_team", "sig_odds"]}
for r in blob:
    for k in mid_pull:
        v = r[k]
        if v is not None and v * 38 / r["fs"] < 24:
            mid_pull[k] += 1
print("\nblob: count of drivers where signal (scaled) < 24 (i.e. signal says mid or better):")
for k, v in mid_pull.items():
    have = sum(1 for r in blob if r[k] is not None)
    print(f"  {k:>10}: {v:>4}/{have}")

# tt trust regression check: blob members with tt_races<3
low_tt = [r for r in blob if 0 < r["tt_races"] < 3]
print(f"\nblob members with 0<tt_races<3 (mid-regressed ttype): {len(low_tt)}")
no_odds_blob = [r for r in blob if not r["has_odds"]]
print(f"blob members without odds: {len(no_odds_blob)}")
print(f"blob members with odds: {len(blob)-len(no_odds_blob)}")
if no_odds_blob:
    prof(no_odds_blob, "blob, NO odds")
prof([r for r in blob if r["has_odds"]], "blob, WITH odds")
