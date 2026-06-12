"""Profit simulator — replays past races and asks the only question that
matters: would this model's lineups have MADE MONEY?

For each completed race with saved odds + results + salaries:
  1. Rebuild projections PRE-RACE-FAITHFULLY (history before_date-filtered,
     practice from NASCAR's archived lap-averages, actual grid as start).
  2. Build lineups the way the optimizer would: one max-projection CASH
     lineup, and N jittered-optimal GPP lineups (projection noise ~ the
     model's own uncertainty, then re-optimize — a standard diversification
     technique).
  3. Score every lineup on ACTUAL fantasy points (DNFs included — contests
     settle on actuals).
  4. Build a FIELD PROXY: Monte-Carlo sample of salary-legal, value-leaning
     public lineups, scored on actuals. Cash line = median field lineup;
     GPP min-cash = 80th percentile (large GPPs pay ~top 20-25%).

The cash line / min-cash thresholds are PROXIES, not real contest data —
they assume the field builds chalk-leaning random-legal lineups. Real lines
run a few points hotter than a pure-random field, so the proxy is graded
with that bias in mind (we report margins, not just booleans).
"""
import math
import random
import sqlite3

from src.config import (DB_PATH, SALARY_CAP, ROSTER_SIZE, FD_SALARY_CAP,
                        FD_ROSTER_SIZE, TRACK_TYPE_WEIGHT_DEFAULTS)
from src.utils import calc_fd_points, fuzzy_match_name, normalize_driver_name

GPP_LINEUPS = 20
FIELD_SAMPLES = 800
JITTER_SD = 0.11          # ~11% lognormal noise on projections for GPP builds
MIN_SALARIED = 25         # need at least this many salaried drivers to sim


def sim_eligible_races(series_id: int, platform: str = "DraftKings",
                       limit: int = 20) -> list:
    """Races with odds + results + salaries for `platform`, newest first.

    Returns [(db_id, api_id, season, race_date, track, race_name), ...]
    """
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute('''
        SELECT r.id, r.api_race_id, r.season, r.race_date, t.name, r.race_name
        FROM races r JOIN tracks t ON t.id = r.track_id
        WHERE r.series_id = ? AND r.api_race_id IS NOT NULL
          AND (SELECT COUNT(*) FROM odds o WHERE o.race_id = r.id) > 0
          AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id) > 0
          AND (SELECT COUNT(*) FROM salaries s
               WHERE s.race_id = r.id AND s.platform = ?) >= ?
        ORDER BY r.race_date DESC LIMIT ?
    ''', (series_id, platform, MIN_SALARIED, limit)).fetchall()
    conn.close()
    return rows


def _load_salaries(conn, db_race_id, platform):
    rows = conn.execute('''
        SELECT d.full_name, s.salary FROM salaries s
        JOIN drivers d ON d.id = s.driver_id
        WHERE s.race_id = ? AND s.platform = ?
    ''', (db_race_id, platform)).fetchall()
    return {n: s for n, s in rows if s}


def _load_fd_actuals(conn, db_race_id):
    rows = conn.execute('''
        SELECT d.full_name, rr.finish_pos, rr.start_pos, rr.laps_led,
               rr.laps_completed
        FROM race_results rr JOIN drivers d ON d.id = rr.driver_id
        WHERE rr.race_id = ? AND rr.finish_pos IS NOT NULL
    ''', (db_race_id,)).fetchall()
    return {n: calc_fd_points(f, s or f, ll or 0, lc or 0)
            for n, f, s, ll, lc in rows}


def _optimal_lineup(pool, cap, roster, obj_key, timeout_ms=600):
    """Branch-and-bound knapsack via the optimizer's solver (pure function)."""
    from tabs.tab_optimizer import _solve_optimal
    return _solve_optimal(pool, cap, roster, timeout_ms=timeout_ms,
                          objective_col=obj_key)


def _sample_field_lineup(rng, names, salaries, weights, cap, roster):
    """One salary-legal public-style lineup (weighted toward value chalk)."""
    for _ in range(60):
        picks = []
        avail = list(names)
        w = [weights[n] for n in avail]
        for _ in range(roster):
            tot = sum(w)
            if tot <= 0 or not avail:
                break
            x = rng.random() * tot
            acc = 0.0
            for i, wi in enumerate(w):
                acc += wi
                if acc >= x:
                    picks.append(avail.pop(i))
                    w.pop(i)
                    break
        if len(picks) != roster:
            continue
        sal = sum(salaries[n] for n in picks)
        if sal <= cap and sal >= cap * 0.90:
            return picks
    return None


def simulate_race(db_id, api_id, season, race_date, track_name, race_name,
                  series_id, platform="DraftKings"):
    """Run the full sim for one race. Returns a result dict or None."""
    # Reuse the pre-race-faithful projection assembly from the backtest harness
    from scripts.backtest_weights import load_race, normalize_weights
    from scripts.backtest_practice_weight import fetch_practice
    from src.projections import compute_projections

    conn = sqlite3.connect(str(DB_PATH))
    try:
        race = load_race(conn, db_id, series_id, track_name, race_date)
        if not race:
            return None
        salaries = _load_salaries(conn, db_id, platform)
        if len(salaries) < MIN_SALARIED:
            return None
        actual = (race["actual_dk"] if platform == "DraftKings"
                  else _load_fd_actuals(conn, db_id))
    finally:
        conn.close()

    # Practice (archived lap-averages), remapped to result-driver spellings
    prac = fetch_practice(season, series_id, api_id) or {}
    if prac:
        norm = {normalize_driver_name(d): d for d in race["drivers"]}
        remapped = {}
        for k, v in prac.items():
            if k in race["drivers"]:
                remapped[k] = v
                continue
            nk = normalize_driver_name(k)
            if nk in norm:
                remapped[norm[nk]] = v
            else:
                m = fuzzy_match_name(k, race["drivers"])
                if m:
                    remapped[m] = v
        prac = remapped

    base = TRACK_TYPE_WEIGHT_DEFAULTS.get(
        race["parent"], TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
    wn = normalize_weights(base, has_odds=bool(race["odds_finish"]),
                           has_prac=bool(prac))
    rows, _, _ = compute_projections(
        return_signal_details=True, drivers=race["drivers"],
        field_size=race["field_size"], wn=wn, th_data=race["th_data"],
        tt_data=race["tt_data"], qual_pos=race["qual_pos"],
        practice_data=prac, odds_finish=race["odds_finish"],
        odds_display=race["odds_display"], team_signal=race["team_signal"],
        mfr_adjustment={}, team_adj_data=race["team_adj"], dnf_data={},
        race_laps=200, track_name=track_name, track_type=race["track_type"],
        series_id=series_id, calibration=race["calibration"], cross_th_lookup={})
    proj = {r["driver"]: r["proj_dk"] for r in rows}

    # Salary-matched pool: project + salary + actual all resolvable
    sal_names = list(salaries.keys())
    pool = []
    for d, p in proj.items():
        s = salaries.get(d)
        if s is None:
            m = fuzzy_match_name(d, sal_names)
            s = salaries.get(m) if m else None
        if s and d in actual:
            pool.append({"Driver": d, "DK Salary": int(s),
                         "Proj Score": float(p), "Actual": float(actual[d])})
    if len(pool) < MIN_SALARIED:
        return None

    cap = FD_SALARY_CAP if platform == "FanDuel" else SALARY_CAP
    roster = FD_ROSTER_SIZE if platform == "FanDuel" else ROSTER_SIZE
    rng = random.Random(api_id * 7919 + (1 if platform == "FanDuel" else 0))

    def lineup_actual(lu):
        return round(sum(d["Actual"] for d in lu), 1)

    # CASH: the straight max-projection lineup
    cash_lu = _optimal_lineup(pool, cap, roster, "Proj Score", timeout_ms=800)
    if not cash_lu or len(cash_lu) < roster:
        return None
    cash_score = lineup_actual(cash_lu)

    # GPP: jitter projections by the model's own uncertainty, re-optimize
    gpp_scores, seen = [], set()
    for i in range(GPP_LINEUPS):
        jpool = []
        for d in pool:
            noise = math.exp(rng.gauss(0, JITTER_SD))
            jpool.append({**d, "Jit": d["Proj Score"] * noise})
        lu = _optimal_lineup(jpool, cap, roster, "Jit", timeout_ms=300)
        if not lu or len(lu) < roster:
            continue
        key = tuple(sorted(x["Driver"] for x in lu))
        if key in seen:
            continue
        seen.add(key)
        gpp_scores.append(lineup_actual(lu))
    if not gpp_scores:
        return None

    # FIELD PROXY: chalk-leaning random-legal lineups scored on actuals
    sal_map = {d["Driver"]: d["DK Salary"] for d in pool}
    act_map = {d["Driver"]: d["Actual"] for d in pool}
    # Public weighting: value^2 (points per $ drives chalk), floor for studs
    weights = {}
    for d in pool:
        val = d["Proj Score"] / max(d["DK Salary"] / 1000.0, 1.0)
        weights[d["Driver"]] = max(0.05, val) ** 2
    names = [d["Driver"] for d in pool]
    field = []
    for _ in range(FIELD_SAMPLES):
        picks = _sample_field_lineup(rng, names, sal_map, weights, cap, roster)
        if picks:
            field.append(round(sum(act_map[n] for n in picks), 1))
    if len(field) < 200:
        return None
    field.sort()

    def pct_of(score):
        import bisect
        return round(100.0 * bisect.bisect_left(field, score) / len(field), 1)

    cash_line = field[len(field) // 2]                  # median field lineup
    mincash_line = field[int(len(field) * 0.80)]        # top ~20% pays

    return {
        "race": race_name, "track": track_name, "date": race_date,
        "platform": platform,
        "cash_score": cash_score, "cash_line": cash_line,
        "cash_margin": round(cash_score - cash_line, 1),
        "beat_cash": cash_score > cash_line,
        "cash_pctile": pct_of(cash_score),
        "n_gpp": len(gpp_scores),
        "gpp_mincash_line": mincash_line,
        "gpp_cashed_pct": round(100.0 * sum(1 for s in gpp_scores
                                            if s > mincash_line) / len(gpp_scores), 1),
        "gpp_best": max(gpp_scores),
        "gpp_best_pctile": pct_of(max(gpp_scores)),
        "n_pool": len(pool),
    }
