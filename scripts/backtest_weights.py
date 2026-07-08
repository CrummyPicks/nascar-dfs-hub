"""Offline weight backtest harness.

Replays compute_projections() PRE-RACE for every completed race that has saved
odds + results, then measures how well projected DraftKings points track the
ACTUAL DK points the drivers scored. Use it to validate any change to the
projection weights / signals BEFORE shipping it — so weight tuning is
evidence-based across many races instead of fit to one anecdote.

Faithfulness notes (held CONSTANT across weight schemes, so the relative
comparison between schemes is fair):
  - qual_pos = each driver's ACTUAL start (the real grid is the right input
    for projecting a completed race; place-differential keys off it).
  - history uses before_date = race date (no leakage).
  - practice/mfr/dnf are omitted (not reliably stored for past races); they're
    identical across schemes so they don't bias the comparison.
  - cross-series history blending is omitted (current-series only).

Metric: Spearman rank correlation between projected DK and actual DK, overall
and among DEEP starters (start >= 20) — the value-play tier we care about.

Usage:  python backtest_weights.py
"""
import sys, os, math, sqlite3
# Lives in scripts/; add the repo root so `import src.*` / `import tabs.*` resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (DB_PATH, CROSS_SERIES_HIERARCHY, TRACK_TYPE_MAP,
                        TRACK_TYPE_PARENT, TRACK_TYPE_WEIGHT_DEFAULTS)
from src.utils import fuzzy_match_name, arp_finish_blend, calc_dk_points
from src.projections import compute_projections
from tabs.tab_projections import (_query_db_track_history, _query_db_track_type_history,
                                   _get_track_dominator_calibration)
from src.data import query_team_stats, compute_team_adjusted_track_history
import pandas as pd

SERIES_NAME = {1: "Cup", 2: "ORly", 3: "Trk"}


def backtestable_races():
    """Races (DB id) that have BOTH saved odds and finishing results."""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute('''
        SELECT r.id, r.series_id, r.season, r.race_date, t.name
        FROM races r JOIN tracks t ON t.id = r.track_id
        WHERE (SELECT COUNT(*) FROM odds o WHERE o.race_id = r.id) > 0
          AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id) > 0
        ORDER BY r.series_id, r.race_date
    ''').fetchall()
    conn.close()
    return rows


def load_race(conn, race_id, series_id, track_name, race_date):
    """Assemble projection inputs + actual outcomes for one completed race."""
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)

    res = conn.execute('''
        SELECT d.full_name, rr.start_pos, rr.finish_pos, rr.laps_led,
               rr.fastest_laps, rr.team, rr.status
        FROM race_results rr JOIN drivers d ON d.id = rr.driver_id
        WHERE rr.race_id = ? AND rr.finish_pos IS NOT NULL
    ''', (race_id,)).fetchall()
    drivers, actual_dk, start_pos, team_map, status_map = [], {}, {}, {}, {}
    actual_finish, actual_ll, actual_fl = {}, {}, {}
    for name, st, fin, ll, fl, team, status in res:
        if st is None:
            st = fin
        drivers.append(name)
        actual_dk[name] = calc_dk_points(fin, st, ll or 0, fl or 0)
        actual_finish[name] = fin
        actual_ll[name] = ll or 0
        actual_fl[name] = fl or 0
        start_pos[name] = st
        status_map[name] = (status or "").strip()
        if team:
            team_map[name] = team
    field_size = len(drivers)
    if field_size < 10:
        return None

    odds_data = {n: int(w) for n, w in conn.execute(
        "SELECT d.full_name, o.win_odds FROM odds o JOIN drivers d ON d.id=o.driver_id "
        "WHERE o.race_id=? AND o.win_odds IS NOT NULL", (race_id,)).fetchall()
        if n in start_pos}

    # Odds -> implied finish (mirror tab_projections: Bradley-Terry pairwise
    # mapping, 2026-07) + implied-win % display (drives the dominator/FL odds
    # signal, like the live app). See src.projections.odds_expected_finish.
    from src.projections import odds_expected_finish
    probs = {n: (100/(o+100) if o > 0 else abs(o)/(abs(o)+100))
             for n, o in odds_data.items()}
    odds_display = {n: {"impl_pct": round(p*100, 1)} for n, p in probs.items()}
    odds_finish = odds_expected_finish(probs, field_size)

    # History (current-series, before this race)
    th_df = _query_db_track_history(track_name, series_id, before_date=race_date)
    tt_raw = _query_db_track_type_history(track_type, series_id,
                                          exclude_track=track_name, before_date=race_date)
    th_data = {}
    if not th_df.empty:
        for c in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races", "Avg Run Pos", "Avg Rating"]:
            if c in th_df.columns:
                th_df[c] = pd.to_numeric(th_df[c], errors="coerce")
        idx = th_df.drop_duplicates("Driver").set_index("Driver")
        for d in drivers:
            m = d if d in idx.index else fuzzy_match_name(d, idx.index.tolist())
            if m and m in idx.index:
                r = idx.loc[m]; races = r.get("Races", 1) or 1
                arp = r.get("Avg Run Pos") if pd.notna(r.get("Avg Run Pos")) and r.get("Avg Run Pos") != 99 else None
                rating = r.get("Avg Rating") if pd.notna(r.get("Avg Rating")) else None
                th_data[d] = {"avg_finish": r.get("Avg Finish", 20), "avg_start": r.get("Avg Start", 20),
                              "avg_running_pos": arp, "th_rating": rating, "laps_led": r.get("Laps Led", 0) or 0,
                              "fastest_laps": r.get("Fastest Laps", 0) or 0, "races": int(races),
                              "laps_led_per_race": (r.get("Laps Led", 0) or 0)/races, "fastest_laps_per_race": 0}
    # Production parity (2026-06): rookie team-fallback — no personal track
    # history -> inherit team average at the track as a soft prior.
    from src.data import apply_team_track_fallback
    th_data = apply_team_track_fallback(th_data, drivers, team_map,
                                        track_name, series_id,
                                        before_date=race_date)

    tt_data = {d: tt_raw[d] for d in drivers if d in tt_raw}

    ts = query_team_stats(series_id, track_type=track_type, before_date=race_date)
    team_signal = {}
    for d in drivers:
        tn = team_map.get(d)
        mt = tn if tn in ts else (fuzzy_match_name(tn, list(ts.keys())) if tn else None)
        if mt and mt in ts:
            t = ts[mt]
            tf = arp_finish_blend(t.get("avg_arp"), t["avg_finish"], track_type) if t.get("avg_arp") else t["avg_finish"]
            tr = min(1.0, t["races"]/10); team_signal[d] = tf*tr + (field_size*0.5)*(1-tr)
    team_adj = compute_team_adjusted_track_history(track_name, series_id, team_map,
                                                   before_date=race_date, track_type=track_type)

    # Pre-race-only calibration: without before_date the replayed race's own
    # dominator concentration leaked into the curve used to project it.
    calibration = _get_track_dominator_calibration(track_name, track_type,
                                                   series_id,
                                                   before_date=race_date)

    # Real race length — compute_projections sizes the laps-led/fastest-lap
    # point pools from it (0.7 pts/lap), so a hardcoded 200 understated
    # dominators ~60% at Bristol and overstated them ~2x at road courses.
    race_laps = None
    row = conn.execute("SELECT laps FROM races WHERE id = ?", (race_id,)).fetchone()
    if row and row[0]:
        race_laps = int(row[0])
    if not race_laps:
        row = conn.execute(
            "SELECT MAX(laps_completed) FROM race_results WHERE race_id = ?",
            (race_id,)).fetchone()
        if row and row[0]:
            race_laps = int(row[0])
    if not race_laps:
        race_laps = 200

    return dict(drivers=drivers, field_size=field_size, track_name=track_name,
                track_type=track_type, parent=parent, series_id=series_id,
                qual_pos=start_pos, odds_finish=odds_finish, odds_display=odds_display,
                th_data=th_data, tt_data=tt_data, team_signal=team_signal,
                team_adj=team_adj, calibration=calibration,
                race_laps=race_laps,
                actual_dk=actual_dk, actual_finish=actual_finish,
                actual_ll=actual_ll, actual_fl=actual_fl,
                start_pos=start_pos, status=status_map, team_map=team_map)


def normalize_weights(raw, has_odds=True, has_prac=False):
    """Mirror tab_projections weight normalization (drop unavailable signals)."""
    w_odds = raw["odds"] if has_odds else 0
    w_prac = raw["prac"] if has_prac else 0
    tot = raw["track"] + raw["ttype"] + w_prac + w_odds + raw["team"] + raw["qual"]
    if tot <= 0:
        return {"track": .6, "track_type": .4, "qual": 0, "practice": 0, "odds": 0, "team": 0}
    return {"track": raw["track"]/tot, "track_type": raw["ttype"]/tot, "qual": raw["qual"]/tot,
            "practice": w_prac/tot, "odds": w_odds/tot, "team": raw["team"]/tot}


def project_race(race, raw_weights):
    """Run compute_projections for one loaded race under a raw weight dict."""
    wn = normalize_weights(raw_weights, has_odds=bool(race["odds_finish"]), has_prac=False)
    rows, _, _ = compute_projections(
        return_signal_details=True, drivers=race["drivers"], field_size=race["field_size"],
        wn=wn, th_data=race["th_data"], tt_data=race["tt_data"], qual_pos=race["qual_pos"],
        practice_data={}, odds_finish=race["odds_finish"], odds_display=race["odds_display"],
        team_signal=race["team_signal"], mfr_adjustment={}, team_adj_data=race["team_adj"],
        dnf_data={}, race_laps=race.get("race_laps", 200), track_name=race["track_name"], track_type=race["track_type"],
        series_id=race["series_id"], calibration=race["calibration"], cross_th_lookup={})
    return {r["driver"]: r["proj_dk"] for r in rows}


def spearman(pairs):
    """pairs: list of (proj, actual). Returns Spearman rho."""
    if len(pairs) < 4:
        return None
    proj = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    act = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    rp = {i: r for r, i in enumerate(proj)}; ra = {i: r for r, i in enumerate(act)}
    n = len(pairs); dsq = sum((rp[i]-ra[i])**2 for i in range(n))
    return 1 - 6*dsq/(n*(n*n-1))


def evaluate(weight_scheme, label):
    """weight_scheme: fn(series_id, parent_type) -> raw weight dict."""
    races = backtestable_races()
    conn = sqlite3.connect(str(DB_PATH))
    by_cell = {}      # (series, parent) -> list of (overall_rho, deep_rho)
    pooled_overall, pooled_deep = [], []
    for rid, sid, season, rdate, track in races:
        race = load_race(conn, rid, sid, track, rdate)
        if not race:
            continue
        proj_dk = project_race(race, weight_scheme(sid, race["parent"]))
        overall = [(proj_dk[d], race["actual_dk"][d]) for d in race["drivers"] if d in proj_dk]
        deep = [(proj_dk[d], race["actual_dk"][d]) for d in race["drivers"]
                if d in proj_dk and race["start_pos"][d] >= 20]
        ro, rd = spearman(overall), spearman(deep)
        by_cell.setdefault((SERIES_NAME[sid], race["parent"]), []).append((ro, rd))
        if ro is not None:
            pooled_overall.append(ro)
        if rd is not None:
            pooled_deep.append(rd)
    conn.close()
    print(f"\n===== {label} =====")
    print(f"{'cell':<26}{'races':>6}{'rho_all':>9}{'rho_deep':>10}")
    for cell in sorted(by_cell):
        vals = by_cell[cell]
        ao = [v[0] for v in vals if v[0] is not None]
        ad = [v[1] for v in vals if v[1] is not None]
        mo = sum(ao)/len(ao) if ao else float('nan')
        md = sum(ad)/len(ad) if ad else float('nan')
        print(f"{cell[0]+' '+cell[1]:<26}{len(vals):>6}{mo:>9.3f}{md:>10.3f}")
    po = sum(pooled_overall)/len(pooled_overall) if pooled_overall else float('nan')
    pdp = sum(pooled_deep)/len(pooled_deep) if pooled_deep else float('nan')
    print(f"{'POOLED (mean over races)':<26}{len(pooled_overall):>6}{po:>9.3f}{pdp:>10.3f}")
    return po, pdp


def current_weights(series_id, parent_type):
    return dict(TRACK_TYPE_WEIGHT_DEFAULTS.get(parent_type, TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"]))


if __name__ == "__main__":
    evaluate(current_weights, "CURRENT weights (baseline)")
