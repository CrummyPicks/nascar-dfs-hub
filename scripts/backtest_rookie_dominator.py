"""Two experiments in one pass:

A) ROOKIE TEAM-FALLBACK — for drivers with NO personal history at a track,
   substitute their TEAM's average at that track (low trust) into th_data.
   Hypothesis: equipment knows the track even when the driver doesn't.
   Graded on every race since 2022 (odds-free schemes, same as bigsample),
   overall AND on the affected-driver subset specifically.

B) DOMINATOR BASELINE — how good are the laps-led / fastest-laps projections
   themselves? Graded on the odds+practice-faithful race set (allocation
   depends on odds), reporting Spearman, MAE and top-5 capture for LL and FL.

Usage: python scripts/backtest_rookie_dominator.py
"""
import sys, os, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DB_PATH, TRACK_TYPE_WEIGHT_DEFAULTS
from scripts.backtest_weights import load_race, normalize_weights, spearman
from scripts.backtest_bigsample import all_result_races, project
from src.projections import compute_projections


def team_track_aggregates(conn, track_name, series_id, before_date):
    """{team: {avg_finish, avg_arp, races}} at one track before a date."""
    rows = conn.execute('''
        SELECT rr.team, AVG(rr.finish_pos), AVG(NULLIF(rr.avg_running_position, 99)),
               COUNT(*)
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE t.name = ? AND r.series_id = ? AND r.race_date < ?
          AND rr.team IS NOT NULL AND rr.team != ''
          AND rr.finish_pos IS NOT NULL
        GROUP BY rr.team HAVING COUNT(*) >= 2
    ''', (track_name, series_id, before_date)).fetchall()
    return {t: {"avg_finish": af, "avg_arp": arp, "races": n}
            for t, af, arp, n in rows if af is not None}


def with_team_fallback(race, team_aggs):
    """Copy of th_data with team-average entries injected for no-history
    drivers. races=2 -> partial trust via the engine's own races/MIN ramp."""
    th = dict(race["th_data"])
    filled = []
    for d in race["drivers"]:
        if d in th:
            continue
        team = race.get("team_map", {}).get(d)
        agg = team_aggs.get(team) if team else None
        if not agg:
            continue
        th[d] = {"avg_finish": agg["avg_finish"], "avg_start": 20,
                 "avg_running_pos": agg["avg_arp"], "th_rating": None,
                 "laps_led": 0, "fastest_laps": 0, "races": 2,
                 "laps_led_per_race": 0, "fastest_laps_per_race": 0}
        filled.append(d)
    return th, filled


def experiment_a():
    races = all_result_races()
    conn = sqlite3.connect(str(DB_PATH))
    pooled = {"A": [], "B": []}          # (pts, clean, finish) per race
    subset_err = {"A": [], "B": []}      # |proj_finish - actual| for filled drivers
    n_filled_total = 0
    for rid, sid, season, rdate, track in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        team_aggs = team_track_aggregates(conn, track, sid, rdate)
        th_b, filled = with_team_fallback(race, team_aggs)
        n_filled_total += len(filled)
        raw = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        status = race.get("status", {})
        running = {d for d in race["drivers"]
                   if status.get(d, "").lower() in ("running", "finished", "")}
        for label, th_use in (("A", race["th_data"]), ("B", th_b)):
            race_v = dict(race)
            race_v["th_data"] = th_use
            pts, fin = project(race_v, raw)
            pts_pairs = [(pts[d], race["actual_dk"][d]) for d in race["drivers"] if d in pts]
            cln_pairs = [(pts[d], race["actual_dk"][d]) for d in race["drivers"]
                         if d in pts and d in running]
            fin_pairs = [(fin[d], race["actual_finish"][d]) for d in race["drivers"]
                         if d in fin and d in running and race["actual_finish"].get(d)]
            pooled[label].append((spearman(pts_pairs), spearman(cln_pairs),
                                  spearman(fin_pairs)))
            for d in filled:
                if d in fin and race["actual_finish"].get(d) and d in running:
                    subset_err[label].append(abs(fin[d] - race["actual_finish"][d]))
    conn.close()

    f = lambda v: sum(v) / len(v) if v else float("nan")
    print("=== A) ROOKIE TEAM-FALLBACK (races graded:",
          len(pooled["A"]), "| fallback drivers filled:", n_filled_total, ")")
    for col, name in [(0, "pts"), (1, "clean"), (2, "finish")]:
        a = [r[col] for r in pooled["A"] if r[col] is not None]
        b = [r[col] for r in pooled["B"] if r[col] is not None]
        print(f"  pooled rho {name:<7} A {f(a):.4f}  B {f(b):.4f}  delta {f(b)-f(a):+.4f}")
    print(f"  AFFECTED-DRIVER finish MAE: A {f(subset_err['A']):.2f}  "
          f"B {f(subset_err['B']):.2f}  (n={len(subset_err['A'])} driver-races)")


def experiment_b():
    from scripts.backtest_practice_weight import (backtestable_races_with_api,
                                                  fetch_practice)
    from src.utils import normalize_driver_name, fuzzy_match_name
    races = backtestable_races_with_api()
    conn = sqlite3.connect(str(DB_PATH))
    ll_rhos, fl_rhos, ll_maes, fl_maes = [], [], [], []
    top5_ll_hits, top5_fl_hits, n_races = 0, 0, 0
    for rid, sid, season, rdate, track, api_id in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        prac = fetch_practice(season, sid, api_id) or {}
        if prac:
            norm = {normalize_driver_name(d): d for d in race["drivers"]}
            prac = {norm.get(normalize_driver_name(k), k): v for k, v in prac.items()}
        raw = TRACK_TYPE_WEIGHT_DEFAULTS.get(
            race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
        wn = normalize_weights(raw, has_odds=bool(race["odds_finish"]),
                               has_prac=bool(prac))
        rows, _, _ = compute_projections(
            return_signal_details=True, drivers=race["drivers"],
            field_size=race["field_size"], wn=wn, th_data=race["th_data"],
            tt_data=race["tt_data"], qual_pos=race["qual_pos"],
            practice_data=prac, odds_finish=race["odds_finish"],
            odds_display=race["odds_display"], team_signal=race["team_signal"],
            mfr_adjustment={}, team_adj_data=race["team_adj"], dnf_data={},
            race_laps=200, track_name=track, track_type=race["track_type"],
            series_id=sid, calibration=race["calibration"], cross_th_lookup={})
        pll = {r["driver"]: r["laps_led"] for r in rows}
        pfl = {r["driver"]: r["fast_laps"] for r in rows}
        all_d = [d for d in race["drivers"] if d in pll]
        if len(all_d) < 10:
            continue
        ll_pairs = [(pll[d], race["actual_ll"][d]) for d in all_d]
        fl_pairs = [(pfl[d], race["actual_fl"][d]) for d in all_d]
        r1, r2 = spearman(ll_pairs), spearman(fl_pairs)
        if r1 is not None:
            ll_rhos.append(r1)
        if r2 is not None:
            fl_rhos.append(r2)
        ll_maes.append(sum(abs(a - b) for a, b in ll_pairs) / len(ll_pairs))
        fl_maes.append(sum(abs(a - b) for a, b in fl_pairs) / len(fl_pairs))
        # Top-5 capture: of the actual top-5 LL/FL drivers, how many did the
        # projection's top-5 contain?
        for proj_map, actual_map, bucket in (
                (pll, race["actual_ll"], "ll"), (pfl, race["actual_fl"], "fl")):
            act5 = {d for d, _ in sorted(actual_map.items(),
                                         key=lambda x: -x[1])[:5]}
            prj5 = {d for d, _ in sorted(proj_map.items(),
                                         key=lambda x: -x[1])[:5]}
            hits = len(act5 & prj5)
            if bucket == "ll":
                top5_ll_hits += hits
            else:
                top5_fl_hits += hits
        n_races += 1
    conn.close()
    f = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"\n=== B) DOMINATOR BASELINE ({n_races} odds-faithful races)")
    print(f"  Laps Led:   rho {f(ll_rhos):.3f} | MAE {f(ll_maes):.1f} laps | "
          f"top-5 capture {top5_ll_hits / (5 * n_races) * 100:.0f}%")
    print(f"  Fast Laps:  rho {f(fl_rhos):.3f} | MAE {f(fl_maes):.1f} laps | "
          f"top-5 capture {top5_fl_hits / (5 * n_races) * 100:.0f}%")


if __name__ == "__main__":
    experiment_a()
    experiment_b()
