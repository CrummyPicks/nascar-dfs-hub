"""Big-sample weight validation — every race with results since 2022.

The odds-gated harness has ~28 races; this one drops the odds signal from
the schemes under test, which removes the gate entirely: every stored race
(~450 across three series) becomes gradeable. It can't tune the ODDS weight,
but it independently tests the June-2026 intermediate redistribution
(track 25->15, ttype 15->20, team 10->15) on a sample size that can't be
argued with.

Grades each scheme three ways per race:
  rho_pts    — Spearman, projected DK pts vs actual DK pts (the deliverable)
  rho_clean  — same, running finishers only (pace-prediction skill)
  rho_finish — Spearman, projected FINISH vs actual FINISH (isolates the
               finish-ordering skill from dominator allocation)

Usage: python scripts/backtest_bigsample.py
"""
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import load_race, normalize_weights, spearman
from src.projections import compute_projections

# Pre-June-2026 intermediate/road mixes, reconstructed for the comparison
OLD_DEFAULTS = {
    "superspeedway":  {"odds": 25, "track": 20, "ttype": 30, "prac": 5,  "team": 15, "qual": 5},
    "short":          {"odds": 20, "track": 25, "ttype": 10, "prac": 10, "team": 10, "qual": 25},
    "short_concrete": {"odds": 20, "track": 30, "ttype": 5,  "prac": 10, "team": 10, "qual": 25},
    "road":           {"odds": 20, "track": 20, "ttype": 20, "prac": 15, "team": 10, "qual": 15},
    "intermediate":   {"odds": 20, "track": 25, "ttype": 20, "prac": 10, "team": 10, "qual": 15},
}


def all_result_races():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute('''
        SELECT r.id, r.series_id, r.season, r.race_date, t.name
        FROM races r JOIN tracks t ON t.id = r.track_id
        WHERE r.season >= 2022
          AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id) >= 10
        ORDER BY r.race_date
    ''').fetchall()
    conn.close()
    return rows


def project(race, raw):
    wn = normalize_weights(raw, has_odds=False, has_prac=False)
    rows, _, _ = compute_projections(
        return_signal_details=True, drivers=race["drivers"],
        field_size=race["field_size"], wn=wn, th_data=race["th_data"],
        tt_data=race["tt_data"], qual_pos=race["qual_pos"], practice_data={},
        odds_finish={}, odds_display={}, team_signal=race["team_signal"],
        mfr_adjustment={}, team_adj_data=race["team_adj"], dnf_data={},
        race_laps=200, track_name=race["track_name"],
        track_type=race["track_type"], series_id=race["series_id"],
        calibration=race["calibration"], cross_th_lookup={})
    return ({r["driver"]: r["proj_dk"] for r in rows},
            {r["driver"]: r["proj_finish"] for r in rows})


def main():
    races = all_result_races()
    print(f"gradeable races since 2022: {len(races)}")
    schemes = {"OLD": OLD_DEFAULTS, "NEW": TRACK_TYPE_WEIGHT_DEFAULTS}
    # results[scheme][parent] = list of (rho_pts, rho_clean, rho_finish)
    results = {k: {} for k in schemes}
    conn = sqlite3.connect(str(DB_PATH))
    done = 0
    for rid, sid, season, rdate, track in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        status = race.get("status", {})
        running = {d for d in race["drivers"]
                   if status.get(d, "").lower() in ("running", "finished", "")}
        for label, table in schemes.items():
            raw = table.get(race["parent"], table["intermediate"])
            proj_pts, proj_fin = project(race, raw)
            pts_pairs = [(proj_pts[d], race["actual_dk"][d])
                         for d in race["drivers"] if d in proj_pts]
            clean_pairs = [(proj_pts[d], race["actual_dk"][d])
                           for d in race["drivers"]
                           if d in proj_pts and d in running]
            fin_pairs = [(proj_fin[d], race["actual_finish"][d])
                         for d in race["drivers"]
                         if d in proj_fin and d in running
                         and race["actual_finish"].get(d)]
            results[label].setdefault(race["parent"], []).append(
                (spearman(pts_pairs), spearman(clean_pairs), spearman(fin_pairs)))
        done += 1
        if done % 100 == 0:
            print(f"  ...{done} races")
    conn.close()
    print(f"graded: {done} races")
    print()
    f = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"{'parent type':<16}{'n':>5} | {'OLD pts':>8}{'NEW pts':>8} | "
          f"{'OLD cln':>8}{'NEW cln':>8} | {'OLD fin':>8}{'NEW fin':>8}")
    parents = sorted(set(list(results["OLD"].keys()) + list(results["NEW"].keys())))
    pooled = {k: [] for k in schemes}
    for p in parents:
        line = f"{p:<16}"
        n = len(results["OLD"].get(p, []))
        line += f"{n:>5} |"
        for col in (0, 1, 2):
            for label in ("OLD", "NEW"):
                vals = [r[col] for r in results[label].get(p, []) if r[col] is not None]
                line += f"{f(vals):>8.4f}"
            line += " |" if col < 2 else ""
        print(line)
        for label in schemes:
            pooled[label].extend(results[label].get(p, []))
    print()
    for col, name in [(0, "pts (DNF incl)"), (1, "clean pts"), (2, "finish rank")]:
        o = [r[col] for r in pooled["OLD"] if r[col] is not None]
        nw = [r[col] for r in pooled["NEW"] if r[col] is not None]
        print(f"POOLED {name:<15} OLD {f(o):.4f}  NEW {f(nw):.4f}  "
              f"delta {f(nw) - f(o):+.4f}  (n={len(o)})")


if __name__ == "__main__":
    main()
