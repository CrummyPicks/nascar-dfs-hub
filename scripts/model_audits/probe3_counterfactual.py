"""Probe 3: counterfactuals on the input assembly (no source edits).
A: baseline
B: drop synthetic team-fallback th entries
C: shrink thin track history (races<4) toward 0.68*field with w=races/(races+2)
   (and treat synthetics as races=1)
D: B+C plus shrink ttype entries with races<3 toward 0.68*field (same w)
Metrics: Spearman(raw, actual) overall + within actual>=25 tail;
mean raw among actual>=28 (tail supply); MAE of proj_finish.
READ-ONLY.
"""
import sys, warnings, sqlite3, statistics
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")

from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import backtestable_races, load_race, normalize_weights, spearman
from src.projections import compute_projections
import copy

def is_synth(th):
    return (th.get("races") == 2 and not th.get("laps_led")
            and th.get("th_rating") is None and not th.get("fastest_laps"))

def variant_inputs(race, mode):
    th = copy.deepcopy(race["th_data"])
    tt = copy.deepcopy(race["tt_data"])
    fs = race["field_size"]
    anchor = 0.68 * fs
    if mode in ("B", "D"):
        th = {d: v for d, v in th.items() if not is_synth(v)}
    if mode in ("C", "D"):
        for d, v in th.items():
            races = 1 if is_synth(v) else v.get("races", 1)
            if races < 4:
                w = races / (races + 2.0)
                v["avg_finish"] = v["avg_finish"] * w + anchor * (1 - w)
                if v.get("avg_running_pos"):
                    v["avg_running_pos"] = v["avg_running_pos"] * w + anchor * (1 - w)
                if v.get("th_rating"):
                    v["th_rating"] = None  # rating from 1-2 races: drop, let arp/finish carry
    if mode == "D":
        for d, v in tt.items():
            races = v.get("races", 1)
            if races < 3:
                w = races / (races + 2.0)
                v["avg_finish"] = v["avg_finish"] * w + anchor * (1 - w)
                if v.get("avg_running_pos"):
                    v["avg_running_pos"] = v["avg_running_pos"] * w + anchor * (1 - w)
    return th, tt

races = backtestable_races()
conn = sqlite3.connect(str(DB_PATH))
loaded = []
for db_id, series_id, season, race_date, track in races:
    try:
        r = load_race(conn, db_id, series_id, track, race_date)
        if r:
            loaded.append(r)
    except Exception:
        pass
conn.close()
print(f"races: {len(loaded)}")

for mode in ["A", "B", "C", "D"]:
    all_pairs, tail_pairs, tail_raw, abserr = [], [], [], []
    for race in loaded:
        base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        wn = normalize_weights(base, has_odds=bool(race["odds_finish"]), has_prac=False)
        th, tt = (race["th_data"], race["tt_data"]) if mode == "A" else variant_inputs(race, mode)
        rows, _, _ = compute_projections(
            return_signal_details=True,
            drivers=race["drivers"], field_size=race["field_size"], wn=wn,
            th_data=th, tt_data=tt,
            qual_pos=race["qual_pos"], practice_data={},
            odds_finish=race["odds_finish"], odds_display=race["odds_display"],
            team_signal=race["team_signal"], mfr_adjustment={},
            team_adj_data=race["team_adj"], dnf_data={},
            race_laps=race.get("race_laps", 200), track_name=race["track_name"],
            track_type=race["track_type"], series_id=race["series_id"],
            calibration=race["calibration"], cross_th_lookup={})
        fs = race["field_size"]; k = 38.0 / fs
        pr = {r_["driver"]: r_ for r_ in rows}
        race_pairs = []
        for d, rr in pr.items():
            af = race["actual_finish"].get(d)
            if af is None:
                continue
            race_pairs.append((rr["raw_finish"], af))
            abserr.append(abs(rr["proj_finish"] - af))
            if af * k >= 28:
                tail_raw.append(rr["raw_finish"] * k)
            if af * k >= 25:
                tail_pairs.append((rr["raw_finish"], af))
        if len(race_pairs) >= 4:
            all_pairs.append(spearman(race_pairs))
    # tail spearman pooled (crude, across races mixes fields but raw is scaled comparably)
    rho_all = statistics.mean(x for x in all_pairs if x is not None)
    print(f"\nmode {mode}: mean per-race Spearman(raw,act) {rho_all:.4f} | "
          f"MAE proj_finish {statistics.mean(abserr):.3f} | "
          f"tail(act>=28) mean raw {statistics.mean(tail_raw):.2f} (n={len(tail_raw)})")
