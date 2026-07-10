"""Probe 4: does an explicit expected-DNF mixture on raw_finish sharpen the tail?
raw' = (1-p)*raw + p*(0.90*field), p = shrunk pre-race DNF rate per driver.
Also variant with p from mechanical-heavy definition. READ-ONLY.
"""
import sys, warnings, sqlite3, statistics
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\codyr\OneDrive\Desktop\NASCAR DFS")

from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import backtestable_races, load_race, normalize_weights, spearman
from src.projections import compute_projections

conn = sqlite3.connect(str(DB_PATH))
races = backtestable_races()
loaded = []
for db_id, series_id, season, race_date, track in races:
    try:
        r = load_race(conn, db_id, series_id, track, race_date)
        if r:
            dnf_rows = conn.execute('''
                SELECT d.full_name, COUNT(*),
                       SUM(CASE WHEN LOWER(COALESCE(rr.status,'running')) NOT IN ('running','') THEN 1 ELSE 0 END)
                FROM race_results rr
                JOIN drivers d ON d.id = rr.driver_id
                JOIN races r2 ON r2.id = rr.race_id
                WHERE r2.series_id = ? AND r2.race_date < ? AND r2.season >= 2022
                GROUP BY d.id
            ''', (series_id, race_date)).fetchall()
            r["dnf_rate"] = {n: (dnfs + 2 * 0.12) / (cnt + 2)
                             for n, cnt, dnfs in dnf_rows}
            loaded.append(r)
    except Exception:
        pass
conn.close()
print(f"races: {len(loaded)}")

for mode, label in [(None, "A baseline"), (0.90, "E dnf-mix @0.90f"), (0.85, "E2 dnf-mix @0.85f")]:
    per_race_rho, abserr, tail_raw, deep_rho = [], [], [], []
    for race in loaded:
        base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        wn = normalize_weights(base, has_odds=bool(race["odds_finish"]), has_prac=False)
        rows, _, _ = compute_projections(
            return_signal_details=True,
            drivers=race["drivers"], field_size=race["field_size"], wn=wn,
            th_data=race["th_data"], tt_data=race["tt_data"],
            qual_pos=race["qual_pos"], practice_data={},
            odds_finish=race["odds_finish"], odds_display=race["odds_display"],
            team_signal=race["team_signal"], mfr_adjustment={},
            team_adj_data=race["team_adj"], dnf_data={},
            race_laps=race.get("race_laps", 200), track_name=race["track_name"],
            track_type=race["track_type"], series_id=race["series_id"],
            calibration=race["calibration"], cross_th_lookup={})
        fs = race["field_size"]; k = 38.0 / fs
        pairs, dpairs = [], []
        for rr in rows:
            d = rr["driver"]
            af = race["actual_finish"].get(d)
            if af is None:
                continue
            raw = rr["raw_finish"]
            if mode is not None:
                p = race["dnf_rate"].get(d, 0.12)
                raw = (1 - p) * raw + p * (mode * fs)
            pairs.append((raw, af))
            abserr.append(abs(raw - af))
            if af * k >= 28:
                tail_raw.append(raw * k)
            if race["start_pos"][d] >= 20:
                dpairs.append((raw, af))
        per_race_rho.append(spearman(pairs))
        rd = spearman(dpairs)
        if rd is not None:
            deep_rho.append(rd)
    rho = statistics.mean(x for x in per_race_rho if x is not None)
    print(f"{label}: Spearman(raw,act) {rho:.4f} | deep-start rho {statistics.mean(deep_rho):.4f} | "
          f"MAE raw {statistics.mean(abserr):.3f} | tail(act>=28) mean raw {statistics.mean(tail_raw):.2f}")
