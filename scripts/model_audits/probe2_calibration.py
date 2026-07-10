"""Probe 2: calibration of raw_finish and of individual signals.
1. raw bucket -> mean actual (all pairs, no selection filter)
2. per-signal calibration: signal value bucket -> mean actual, split by trust inputs
3. DNF-exclusion optimism at the data layer.
READ-ONLY.
"""
import sys, warnings, sqlite3, statistics, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")

from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import backtestable_races, load_race, normalize_weights
from src.projections import compute_projections

races = backtestable_races()
conn = sqlite3.connect(str(DB_PATH))
recs = []
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
        k = 38.0 / fs
        for r in rows:
            d = r["driver"]
            af = race["actual_finish"].get(d)
            if af is None:
                continue
            s = sig.get(d, {})
            th = race["th_data"].get(d) or {}
            recs.append({
                "rawn": r["raw_finish"] * k, "actn": float(af) * k,
                "projn": r["proj_finish"] * k,
                "trackn": (s.get("Track") or 0) * k if s.get("Track") is not None else None,
                "ttn": (s.get("TType") or 0) * k if s.get("TType") is not None else None,
                "oddsn": (s.get("Odds") or 0) * k if s.get("Odds") is not None else None,
                "th_races": th.get("races", 0),
                "synthetic_th": th.get("races", 0) == 2 and th.get("laps_led", 1) == 0 and th.get("th_rating", 1) is None,
            })
    except Exception:
        continue
conn.close()
print(f"pairs: {len(recs)}")

def bucket_report(key, label, split=None, splitlabel=("", "")):
    print(f"\n=== {label}: bucket -> mean actual (n) ===")
    groups = [recs] if split is None else [[r for r in recs if split(r)], [r for r in recs if not split(r)]]
    labels = [""] if split is None else list(splitlabel)
    for g, lab in zip(groups, labels):
        buckets = collections.defaultdict(list)
        for r in g:
            v = r[key]
            if v is None:
                continue
            buckets[int(min(v, 37.9) // 4) * 4].append(r["actn"])
        line = []
        for b in sorted(buckets):
            vals = buckets[b]
            line.append(f"{b:>2}-{b+4:<2}: {statistics.mean(vals):4.1f}({len(vals):>3})")
        print(f"  {lab:<22} " + " | ".join(line))

bucket_report("rawn", "raw_finish (blend output)")
bucket_report("projn", "proj_finish (post-kernel)")
bucket_report("trackn", "TRACK signal, split by races",
              split=lambda r: r["th_races"] <= 2, splitlabel=("th_races<=2", "th_races>=3"))
bucket_report("ttn", "TTYPE signal")
bucket_report("oddsn", "ODDS signal (BT)")

# synthetic team-fallback entries: how do they calibrate?
syn = [r for r in recs if r["synthetic_th"] and r["trackn"] is not None]
print(f"\nsynthetic team-fallback th entries: {len(syn)}")
if syn:
    print(f"  mean track sig {statistics.mean(r['trackn'] for r in syn):.1f} -> mean actual {statistics.mean(r['actn'] for r in syn):.1f}")

# 3. DNF exclusion optimism, straight from DB: per driver-track-type,
# clean-only avg finish vs all-races avg finish, for backmarker-ish drivers
conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute('''
    SELECT d.full_name,
           AVG(CASE WHEN LOWER(COALESCE(rr.status,'running')) IN ('running','') THEN rr.finish_pos END) AS clean_af,
           AVG(rr.finish_pos) AS all_af,
           COUNT(*) AS n,
           SUM(CASE WHEN LOWER(COALESCE(rr.status,'running')) NOT IN ('running','') THEN 1 ELSE 0 END) AS dnfs
    FROM race_results rr
    JOIN drivers d ON d.id = rr.driver_id
    JOIN races r ON r.id = rr.race_id
    WHERE r.season >= 2022 AND r.series_id = 1
    GROUP BY d.id HAVING n >= 8
''').fetchall()
conn.close()
gaps_back, gaps_front = [], []
for name, clean, alla, n, dnfs in rows:
    if clean is None:
        continue
    gap = alla - clean
    (gaps_back if alla >= 22 else gaps_front).append((gap, dnfs / n))
if gaps_back:
    print(f"\nDNF-exclusion optimism (Cup 2022+, drivers avg_all>=22, n={len(gaps_back)}):")
    print(f"  mean (all_avg - clean_avg): {statistics.mean(g for g,_ in gaps_back):+.2f} positions | mean DNF rate {statistics.mean(d for _,d in gaps_back):.0%}")
if gaps_front:
    print(f"front/mid drivers (avg_all<22, n={len(gaps_front)}):")
    print(f"  mean (all_avg - clean_avg): {statistics.mean(g for g,_ in gaps_front):+.2f} positions | mean DNF rate {statistics.mean(d for _,d in gaps_front):.0%}")
