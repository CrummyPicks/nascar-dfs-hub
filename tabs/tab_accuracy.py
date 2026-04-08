"""Tab 7: Accuracy — Projections vs Actuals backtesting and weight optimization."""

import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import os
from datetime import datetime

from src.config import (
    SERIES_OPTIONS, TRACK_TYPE_MAP, TRACK_TYPE_PARENT, DK_FINISH_POINTS,
)
from src.data import (
    fetch_race_list, fetch_weekend_feed, fetch_lap_times,
    extract_race_results, compute_fastest_laps,
    filter_point_races, query_salaries, load_race_odds,
)
from src.utils import calc_dk_points, safe_fillna, format_display_df, short_name_series, fuzzy_match_name

PROJ_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")


# ── DB helpers ───────────────────────────────────────────────────────────────

def _api_race_id_to_db(api_race_id):
    """Resolve a NASCAR API race_id to the internal DB race_id."""
    if not api_race_id or not os.path.exists(PROJ_DB):
        return None
    try:
        conn = sqlite3.connect(PROJ_DB)
        row = conn.execute(
            "SELECT id FROM races WHERE api_race_id = ?", (api_race_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _ensure_saved_projections_table():
    """Create the saved_projections table if it doesn't exist."""
    if not os.path.exists(PROJ_DB):
        return
    conn = sqlite3.connect(PROJ_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS saved_projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER,
            race_name TEXT,
            track_name TEXT,
            series_id INTEGER,
            season INTEGER,
            driver TEXT,
            proj_dk REAL,
            proj_finish REAL,
            proj_laps_led REAL,
            proj_fast_laps REAL,
            proj_diff_pts REAL,
            qual_pos INTEGER,
            dk_salary INTEGER,
            w_odds REAL,
            w_track REAL,
            w_practice REAL,
            w_qualifying REAL,
            w_track_type REAL,
            saved_at TEXT DEFAULT (datetime('now')),
            UNIQUE(race_id, driver, series_id)
        )
    ''')
    conn.commit()
    conn.close()


def save_projections_to_db(proj_df, race_id, race_name, track_name,
                            series_id, season, weights):
    """Save current projections to DB for future accuracy comparison."""
    _ensure_saved_projections_table()
    if not os.path.exists(PROJ_DB):
        return 0

    conn = sqlite3.connect(PROJ_DB)
    count = 0
    for _, row in proj_df.iterrows():
        try:
            conn.execute('''
                INSERT OR REPLACE INTO saved_projections
                (race_id, race_name, track_name, series_id, season,
                 driver, proj_dk, proj_finish, proj_laps_led, proj_fast_laps,
                 proj_diff_pts, qual_pos, dk_salary,
                 w_odds, w_track, w_practice, w_qualifying, w_track_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_id, race_name, track_name, series_id, season,
                row.get("Driver", ""),
                row.get("Proj DK", 0),
                row.get("Proj Finish", 0),
                row.get("Proj Laps Led", 0),
                row.get("Proj Fast Laps", 0),
                row.get("Diff Pts", 0),
                row.get("Qual Pos") or row.get("Start"),
                row.get("DK Salary"),
                weights.get("odds", 0),
                weights.get("track", 0),
                weights.get("practice", 0),
                weights.get("qual", 0),
                weights.get("track_type", 0),
            ))
            count += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return count


def load_saved_projections(series_id=None, season=None, race_id=None):
    """Load saved projections from DB."""
    _ensure_saved_projections_table()
    if not os.path.exists(PROJ_DB):
        return pd.DataFrame()

    conn = sqlite3.connect(PROJ_DB)
    query = "SELECT * FROM saved_projections WHERE 1=1"
    params = []
    if series_id is not None:
        query += " AND series_id = ?"
        params.append(series_id)
    if season is not None:
        query += " AND season = ?"
        params.append(season)
    if race_id is not None:
        query += " AND race_id = ?"
        params.append(race_id)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_saved_race_list(series_id=None):
    """Get list of races that have saved projections."""
    _ensure_saved_projections_table()
    if not os.path.exists(PROJ_DB):
        return pd.DataFrame()

    conn = sqlite3.connect(PROJ_DB)
    query = """
        SELECT DISTINCT race_id, race_name, track_name, series_id, season,
               COUNT(driver) as driver_count,
               MAX(saved_at) as saved_at
        FROM saved_projections
    """
    params = []
    if series_id is not None:
        query += " WHERE series_id = ?"
        params.append(series_id)
    query += " GROUP BY race_id, series_id ORDER BY season DESC, race_name"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def _get_race_year(race):
    """Extract year from race date string."""
    rd = race.get("race_date", "")
    try:
        return int(rd[:4])
    except Exception:
        return 2026


def _load_actual_results(race, series_id):
    """Load actual race results for a completed race."""
    rc_id = race.get("race_id")
    yr = _get_race_year(race)
    rc_feed = fetch_weekend_feed(series_id, rc_id, yr)
    rc_laps = fetch_lap_times(series_id, rc_id, yr)
    if not rc_feed:
        return pd.DataFrame()

    results = extract_race_results(rc_feed)
    if results.empty:
        return results

    fl = compute_fastest_laps(rc_laps) if rc_laps else {}
    results["Fastest Laps"] = results["Driver"].map(lambda d: fl.get(d, 0)).astype("Int64")
    results["DK Pts"] = results.apply(
        lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                 r["Laps Led"], r["Fastest Laps"]), axis=1)
    return results


# ── DB-backed track stats (fast, no API calls) ──────────────────────────────

def _query_driver_track_stats(track_name, series_id, exclude_race_id=None,
                               before_date=None):
    """Query per-driver stats at a specific track from race_results table.

    Args:
        exclude_race_id: DB race ID to exclude (prevents data leakage in backtests)
        before_date: Only include races before this date (YYYY-MM-DD) to prevent
                     using future data when backtesting historical races

    Returns dict of {driver_name: {avg_finish, avg_start, laps_led_per_race,
    fastest_laps_per_race, races, dnf_rate, crash_rate, speed_score}}
    """
    if not os.path.exists(PROJ_DB):
        return {}

    conn = sqlite3.connect(PROJ_DB)
    where = "WHERE t.name LIKE ? AND r.series_id = ?"
    params = [f"%{track_name}%", series_id]
    if exclude_race_id:
        where += " AND r.id != ?"
        params.append(exclude_race_id)
    if before_date:
        where += " AND r.race_date < ?"
        params.append(before_date)

    rows = conn.execute(f'''
        SELECT d.full_name,
               COUNT(*) as races,
               AVG(rr.finish_pos) as avg_finish,
               AVG(rr.start_pos) as avg_start,
               SUM(rr.laps_led) as total_laps_led,
               SUM(rr.fastest_laps) as total_fastest_laps,
               SUM(CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END) as dnfs,
               SUM(CASE WHEN LOWER(rr.status) IN ('accident','crash','damage') THEN 1 ELSE 0 END) as crashes
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        {where}
        GROUP BY d.id
    ''', params).fetchall()
    conn.close()

    result = {}
    for r in rows:
        name, races, avg_f, avg_s, ll, fl, dnfs, crashes = r
        if races and races > 0:
            ll_per = (ll or 0) / races
            fl_per = (fl or 0) / races
            result[name] = {
                "avg_finish": avg_f or 20,
                "avg_start": avg_s or 20,
                "laps_led_per_race": ll_per,
                "fastest_laps_per_race": fl_per,
                "races": races,
                "dnf_rate": (dnfs or 0) / races,
                "crash_rate": (crashes or 0) / races,
                "speed_score": ll_per + fl_per,
            }
    return result


def _query_driver_track_type_stats(track_type, series_id, exclude_track=None,
                                    exclude_race_id=None, before_date=None):
    """Query per-driver stats across all tracks of a given type.

    Args:
        exclude_track: Track name to exclude (for track-type cross-validation)
        exclude_race_id: DB race ID to exclude (prevents data leakage)
        before_date: Only include races before this date (YYYY-MM-DD)

    Returns dict of {driver_name: {avg_finish, laps_led_per_race, speed_score, ...}}
    """
    if not os.path.exists(PROJ_DB):
        return {}

    from src.config import TRACK_TYPE_MAP as _TTM
    matching_tracks = [t for t, tt in _TTM.items()
                       if tt == track_type
                       or TRACK_TYPE_PARENT.get(tt, tt) == track_type]
    if exclude_track:
        matching_tracks = [t for t in matching_tracks if exclude_track not in t]
    if not matching_tracks:
        return {}

    conn = sqlite3.connect(PROJ_DB)
    placeholders = ",".join("?" for _ in matching_tracks)
    where_extra = ""
    params = matching_tracks + [series_id]
    if exclude_race_id:
        where_extra += " AND r.id != ?"
        params.append(exclude_race_id)
    if before_date:
        where_extra += " AND r.race_date < ?"
        params.append(before_date)

    query = f'''
        SELECT d.full_name,
               COUNT(*) as races,
               AVG(rr.finish_pos) as avg_finish,
               AVG(rr.start_pos) as avg_start,
               SUM(rr.laps_led) as total_laps_led,
               SUM(rr.fastest_laps) as total_fastest_laps
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE t.name IN ({placeholders}) AND r.series_id = ?{where_extra}
        GROUP BY d.id
    '''

    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = {}
    for r in rows:
        name, races, avg_f, avg_s, ll, fl = r
        if races and races > 0:
            ll_per = (ll or 0) / races
            fl_per = (fl or 0) / races
            result[name] = {
                "avg_finish": avg_f or 20,
                "laps_led_per_race": ll_per,
                "speed_score": ll_per + fl_per,
                "races": races,
            }
    return result


def _query_driver_career_dnf(series_id, before_date=None):
    """Query career DNF and crash rates for all drivers.

    Args:
        before_date: Only include races before this date (YYYY-MM-DD)

    Returns dict of {driver_name: {dnf_rate, crash_rate, speed_score, races}}
    """
    if not os.path.exists(PROJ_DB):
        return {}

    conn = sqlite3.connect(PROJ_DB)
    where = "WHERE r.series_id = ?"
    params = [series_id]
    if before_date:
        where += " AND r.race_date < ?"
        params.append(before_date)

    rows = conn.execute(f'''
        SELECT d.full_name,
               COUNT(*) as races,
               SUM(CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END) as dnfs,
               SUM(CASE WHEN LOWER(rr.status) IN ('accident','crash','damage') THEN 1 ELSE 0 END) as crashes,
               1.0 * SUM(rr.laps_led) / COUNT(*) as ll_per_race,
               1.0 * SUM(rr.fastest_laps) / COUNT(*) as fl_per_race
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        {where}
        GROUP BY d.id
        HAVING races >= 5
    ''', params).fetchall()
    conn.close()

    result = {}
    for r in rows:
        name, races, dnfs, crashes, ll_per, fl_per = r
        if races and races > 0:
            speed = (ll_per or 0) + (fl_per or 0)
            result[name] = {
                "dnf_rate": (dnfs or 0) / races,
                "crash_rate": (crashes or 0) / races,
                "speed_score": speed,
                "races": races,
            }
    return result


DEFAULT_WEIGHTS = {
    "track": 0.25, "track_type": 0.20, "practice": 0.20,
    "odds": 0.15, "qual": 0.12,
}


def _generate_race_projections(race, series_id, weights=None):
    """Generate projections for a completed race using the backtest engine.

    Returns (proj_dict, actuals_df, race_data_dict) or (None, None, None).
    proj_dict: {driver: proj_dk_points}
    actuals_df: DataFrame with actual results
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    race_id = race.get("race_id")
    track_name = race.get("track_name", "")
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    yr = _get_race_year(race)

    actuals = _load_actual_results(race, series_id)
    if actuals.empty:
        return None, None, None

    drivers = actuals["Driver"].unique().tolist()
    field_size = len(drivers)

    # Exclude this race AND all future races to prevent data leakage
    db_rid = _api_race_id_to_db(race_id)
    race_date = race.get("race_date", "")[:10] if race.get("race_date") else None
    th_data = _query_driver_track_stats(track_name, series_id,
                                         exclude_race_id=db_rid,
                                         before_date=race_date)
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    tt_data = _query_driver_track_type_stats(parent_type, series_id,
                                              exclude_track=track_name,
                                              exclude_race_id=db_rid,
                                              before_date=race_date)

    start_positions = {}
    for _, row in actuals.iterrows():
        if pd.notna(row.get("Start")):
            start_positions[row["Driver"]] = int(row["Start"])

    # Load odds
    saved_odds = load_race_odds(race_id)
    odds_finish = {}
    if saved_odds:
        clean_odds = {k: v for k, v in saved_odds.items()
                      if v is not None and str(v).strip() not in ("", "None", "null")}
        odds_probs = {}
        for name, odds_str in clean_odds.items():
            try:
                odds_val = int(str(odds_str).replace("+", ""))
                if odds_val > 0:
                    prob = 100 / (odds_val + 100)
                elif odds_val < 0:
                    prob = abs(odds_val) / (abs(odds_val) + 100)
                else:
                    continue
                odds_probs[name] = prob
            except (ValueError, TypeError):
                continue
        if len(odds_probs) >= field_size * 0.3:
            ranked = sorted(odds_probs.items(), key=lambda x: x[1], reverse=True)
            for rank, (name, prob) in enumerate(ranked, 1):
                matched = fuzzy_match_name(name, drivers)
                if matched:
                    odds_finish[matched] = rank * (field_size / len(ranked))

    race_laps = race.get("scheduled_laps", 0)
    try:
        race_laps = int(race_laps) if race_laps else 0
    except (ValueError, TypeError):
        race_laps = 0

    dnf_data = _query_driver_career_dnf(series_id, before_date=race_date)

    proj_dk = _project_race_backtest(
        drivers, field_size, weights, th_data, tt_data,
        start_positions, odds_finish, dnf_data,
        race_laps=race_laps, track_type=track_type,
    )

    return proj_dk, actuals, {
        "has_odds": bool(odds_finish),
        "has_track": bool(th_data),
        "has_qual": bool(start_positions),
    }


# ── Full projection engine for backtesting ───────────────────────────────────

def _project_race_backtest(drivers, field_size, wn, th_data, tt_data,
                           start_positions, odds_finish, dnf_data,
                           race_laps=0, track_type="intermediate"):
    """Run the full projection model for a single race.

    Matches the live projection engine: finish pts + diff pts + laps led + fastest laps.
    Uses all 5 signals with weight redistribution when data is missing.
    Returns dict of {driver: proj_dk_points}

    Args:
        drivers: list of driver names
        field_size: number of drivers in field
        wn: normalized weights dict {track, track_type, qual, practice, odds}
        th_data: {driver: {avg_finish, speed_score, laps_led_per_race, ...}}
        tt_data: {driver: {avg_finish, laps_led_per_race, ...}}
        start_positions: {driver: start_pos} (qualifying proxy)
        odds_finish: {driver: implied_finish} from odds
        dnf_data: {driver: {dnf_rate, crash_rate, speed_score}} career DNF info
        race_laps: scheduled laps for the race (needed for LL/FL allocation)
        track_type: track type string for concentration lookup
    """
    driver_raw_scores = {}
    dom_raw_scores = {}
    fl_raw_scores = {}

    for d in drivers:
        th = th_data.get(d)
        tt = tt_data.get(d)
        sp = start_positions.get(d)
        od = odds_finish.get(d)

        # ── Finish composite (same as live engine) ──
        finish_signals = []
        signal_weights = []
        has_history = bool(th or tt)

        if th and wn.get("track", 0) > 0:
            finish_signals.append(th["avg_finish"])
            signal_weights.append(wn["track"])

        if tt and wn.get("track_type", 0) > 0:
            finish_signals.append(tt["avg_finish"])
            signal_weights.append(wn["track_type"])

        # Qualifying is NOT a finish signal — it only determines start position.
        # Place differential = start - proj_finish, so qualifying must stay
        # independent to allow meaningful position gain/loss projections.

        # Practice proxy: use speed_score from track history OR track type
        speed_source = th if (th and th.get("speed_score", 0) > 0) else (
            tt if (tt and tt.get("speed_score", 0) > 0) else None)
        if speed_source and wn.get("practice", 0) > 0:
            all_speeds = [t.get("speed_score", 0) for t in th_data.values()]
            all_speeds += [t.get("speed_score", 0) for t in tt_data.values()]
            max_speed = max(all_speeds) if all_speeds else 1
            if max_speed > 0:
                speed_pct = speed_source["speed_score"] / max_speed
                prac_finish = field_size * (1 - speed_pct * 0.8)
                finish_signals.append(max(1, prac_finish))
                signal_weights.append(wn["practice"])

        if od and wn.get("odds", 0) > 0:
            finish_signals.append(od)
            signal_weights.append(wn["odds"])

        if finish_signals and sum(signal_weights) > 0:
            total_w = sum(signal_weights)
            raw_finish = sum(f * w for f, w in zip(finish_signals, signal_weights)) / total_w
        else:
            raw_finish = field_size * 0.65

        # DNF risk adjustment
        dnf = dnf_data.get(d)
        if dnf and dnf["races"] >= 10:
            crash_rate = dnf["crash_rate"]
            speed = dnf["speed_score"]
            max_speed_all = max((v["speed_score"] for v in dnf_data.values()), default=1)
            speed_factor = speed / max(max_speed_all, 1)
            penalty_weight = max(0.05, 0.3 - speed_factor * 0.2)
            mech_rate = dnf["dnf_rate"] - crash_rate
            risk_penalty = crash_rate * penalty_weight + mech_rate * (penalty_weight * 0.3)
            raw_finish = raw_finish + risk_penalty * 10

        driver_raw_scores[d] = raw_finish

        # ── Dominator score (laps led potential) — weight-aware ──
        # Dom signal weights reflect the user's projection weights so that
        # changing odds weight also changes who is projected to lead laps
        dom_score = 0.0
        if race_laps > 0:
            dom_signals = []
            dom_w = []

            if th and th.get("laps_led_per_race", 0) > 0:
                dom_signals.append(th["laps_led_per_race"])
                dom_w.append(wn.get("track", 0.20))

            if tt and tt.get("laps_led_per_race", 0) > 0:
                dom_signals.append(tt["laps_led_per_race"])
                dom_w.append(wn.get("track_type", 0.10))

            if sp and sp <= field_size:
                qual_dom = max(0, (field_size + 1 - sp) / field_size) ** 1.5 * 30
                dom_signals.append(qual_dom)
                dom_w.append(wn.get("qual", 0.15))

            if od and wn.get("odds", 0) > 0:
                odds_dom = max(0, (field_size + 1 - od) / field_size) ** 1.3 * 35
                dom_signals.append(odds_dom)
                dom_w.append(wn.get("odds", 0.15))

            if dom_signals:
                total_dw = sum(dom_w)
                dom_score = sum(s * w for s, w in zip(dom_signals, dom_w)) / total_dw
            else:
                dom_score = max(0, (field_size - raw_finish) / field_size) * 5

        dom_raw_scores[d] = dom_score

        # ── Fastest laps score — weight-aware ──
        fl_score = 0.0
        if race_laps > 0:
            fl_signals = []
            fl_w = []

            if dom_score > 0:
                fl_signals.append(dom_score * 0.5)
                fl_w.append(0.25)

            if sp and sp <= field_size:
                qual_fl = max(0, (field_size + 1 - sp) / field_size) * 15
                fl_signals.append(qual_fl)
                fl_w.append(wn.get("qual", 0.15))

            if od and wn.get("odds", 0) > 0:
                odds_fl = max(0, (field_size + 1 - od) / field_size) * 12
                fl_signals.append(odds_fl)
                fl_w.append(wn.get("odds", 0.15))

            finish_fl = max(0, (field_size - raw_finish) / field_size) * 10
            fl_signals.append(finish_fl)
            fl_w.append(0.10)

            if fl_signals:
                total_fw = sum(fl_w)
                fl_score = sum(s * w for s, w in zip(fl_signals, fl_w)) / total_fw

        fl_raw_scores[d] = fl_score

    # ── Rank-order finish spreading ──
    sorted_drivers = sorted(driver_raw_scores.items(), key=lambda x: x[1])
    n = len(sorted_drivers)

    driver_proj_finish = {}
    for rank_idx, (d, raw_score) in enumerate(sorted_drivers):
        if n > 1:
            t = rank_idx / (n - 1)
            proj_finish = 1 + (field_size - 1) * (t ** 0.85)
        else:
            proj_finish = field_size * 0.5
        driver_proj_finish[d] = max(1, min(field_size, proj_finish))

    # ── Allocate laps led (zero-sum with cutoff) ──
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    LEADER_FRAC = {
        "superspeedway": 0.60, "road": 0.22, "short": 0.18,
        "short_concrete": 0.16, "intermediate": 0.22, "intermediate_worn": 0.20,
    }
    ll_frac = LEADER_FRAC.get(track_type, LEADER_FRAC.get(parent, 0.22))
    n_leaders = max(3, int(field_size * ll_frac))
    CONC = {
        "superspeedway": 0.6, "road": 1.0, "short": 2.0,
        "short_concrete": 2.2, "intermediate": 1.5, "intermediate_worn": 1.6,
    }
    ll_conc = CONC.get(track_type, CONC.get(parent, 1.5))

    allocated_ll = {}
    if race_laps > 0 and dom_raw_scores:
        top_dom = sorted(dom_raw_scores.items(), key=lambda x: x[1], reverse=True)[:n_leaders]
        ll_scores = {d: max(0.01, s) ** ll_conc for d, s in top_dom}
        ll_total = sum(ll_scores.values())
        if ll_total > 0:
            allocated_ll = {d: (s / ll_total) * race_laps for d, s in ll_scores.items()}

    # ── Allocate fastest laps (zero-sum with cutoff) ──
    FL_FRAC = {
        "superspeedway": 0.85, "road": 0.55, "short": 0.55,
        "short_concrete": 0.50, "intermediate": 0.65, "intermediate_worn": 0.60,
    }
    fl_frac = FL_FRAC.get(track_type, FL_FRAC.get(parent, 0.55))
    n_with_fl = max(5, int(field_size * fl_frac))
    fl_conc = max(0.5, ll_conc * 0.7)

    allocated_fl = {}
    if race_laps > 0 and fl_raw_scores:
        top_fl = sorted(fl_raw_scores.items(), key=lambda x: x[1], reverse=True)[:n_with_fl]
        fl_scores = {d: max(0.01, s) ** fl_conc for d, s in top_fl}
        fl_total = sum(fl_scores.values())
        if fl_total > 0:
            allocated_fl = {d: (s / fl_total) * race_laps for d, s in fl_scores.items()}

    # ── Compute full DK points: finish + diff + laps led + fastest laps ──
    proj_dk = {}
    for d in drivers:
        proj_finish = driver_proj_finish[d]
        pf_int = round(proj_finish)  # DK uses integer positions
        finish_pts = DK_FINISH_POINTS.get(max(1, min(40, pf_int)), 0)

        sp = start_positions.get(d)
        start = sp if sp else pf_int
        diff_pts = int(start - pf_int)

        led_pts = round(allocated_ll.get(d, 0)) * 0.25
        fl_pts = round(allocated_fl.get(d, 0)) * 0.45

        proj_dk[d] = finish_pts + diff_pts + led_pts + fl_pts

    return proj_dk


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, completed_races, series_id, selected_year, series_name="Cup"):
    """Render the Accuracy tab."""
    st.markdown("### Projection Accuracy")
    st.caption("Compare projections vs actual results to improve future weight tuning")

    mode = st.radio("Mode",
                    ["Race Comparison", "Accuracy Dashboard", "Weight Optimizer"],
                    horizontal=True, label_visibility="collapsed",
                    key="acc_mode")

    if mode == "Race Comparison":
        _render_race_comparison(completed_races, series_id, selected_year)
    elif mode == "Accuracy Dashboard":
        _render_accuracy_dashboard(series_id, selected_year, series_name)
    elif mode == "Weight Optimizer":
        _render_weight_optimizer(completed_races, series_id, selected_year, series_name)


# ── Race Comparison ──────────────────────────────────────────────────────────

def _render_race_comparison(completed_races, series_id, selected_year):
    """Compare projections vs actuals for a single race.

    Works for ANY completed race — auto-generates projections using default
    weights when no saved projections exist.
    """
    if not completed_races:
        st.info("No completed races available for this series/year.")
        return

    # Build race dropdown from all completed races
    race_labels = []
    race_map = {}
    for _, race in completed_races:
        track = race.get("track_name", "")
        name = race.get("race_name", "")
        date = (race.get("race_date", "") or "")[:10]
        lbl = f"{date} — {track}: {name}"
        race_labels.append(lbl)
        race_map[lbl] = race

    selected_label = st.selectbox("Select Race", race_labels,
                                   index=len(race_labels) - 1,
                                   key="acc_race_pick")
    actual_race = race_map[selected_label]
    race_id = actual_race.get("race_id")

    # Check for saved projections
    saved_races = load_saved_race_list(series_id)
    has_saved = False
    if not saved_races.empty:
        has_saved = race_id in saved_races["race_id"].values

    # Weight source selector
    weight_source = "Default Weights"
    if has_saved:
        weight_source = st.radio("Projection Source",
                                  ["Default Weights", "Saved Projections"],
                                  horizontal=True, key="acc_weight_src")

    with st.spinner("Generating projections and loading results..."):
        if weight_source == "Saved Projections" and has_saved:
            # Use saved projections
            proj_df = load_saved_projections(series_id=series_id, race_id=race_id)
            actuals = _load_actual_results(actual_race, series_id)
            if proj_df.empty or actuals.empty:
                st.warning("Could not load saved projections or actual results.")
                return

            merged = proj_df.merge(
                actuals[["Driver", "Finish Position", "Start", "Laps Led",
                         "Fastest Laps", "DK Pts"]],
                left_on="driver", right_on="Driver", how="inner"
            )
            if merged.empty:
                st.warning("Could not match projected drivers to actual results.")
                return

            comp = pd.DataFrame({
                "Driver": merged["Driver"],
                "Start": merged["Start"],
                "Proj DK": merged["proj_dk"].round(1),
                "Actual DK": merged["DK Pts"].round(1),
                "DK Error": (merged["proj_dk"] - merged["DK Pts"]).round(1),
                "Proj Finish": merged["proj_finish"].round(1),
                "Actual Finish": merged["Finish Position"],
                "Finish Error": (merged["proj_finish"] - merged["Finish Position"]).round(1),
            })

            w_row = proj_df.iloc[0]
            weights_str = (
                f"Odds {w_row.get('w_odds', 0):.0%} | "
                f"Track {w_row.get('w_track', 0):.0%} | "
                f"Practice {w_row.get('w_practice', 0):.0%} | "
                f"Track Type {w_row.get('w_track_type', 0):.0%}"
            )
        else:
            # Auto-generate projections using default weights
            proj_dk, actuals, meta = _generate_race_projections(
                actual_race, series_id, DEFAULT_WEIGHTS
            )
            if proj_dk is None or actuals is None:
                st.warning("Could not generate projections — race results may not be available.")
                return

            rows = []
            for _, row in actuals.iterrows():
                d = row["Driver"]
                proj = proj_dk.get(d, 0)
                actual_dk = row["DK Pts"]
                actual_finish = row["Finish Position"]
                start_pos = row.get("Start")
                # Estimate projected finish from rank order of proj_dk
                sorted_proj = sorted(proj_dk.items(), key=lambda x: x[1], reverse=True)
                proj_finish = next((i+1 for i, (n, _) in enumerate(sorted_proj) if n == d),
                                   len(sorted_proj))
                rows.append({
                    "Driver": d,
                    "Start": start_pos,
                    "Proj DK": round(proj, 1),
                    "Actual DK": round(actual_dk, 1),
                    "DK Error": round(proj - actual_dk, 1),
                    "Proj Finish": proj_finish,
                    "Actual Finish": actual_finish,
                    "Finish Error": round(proj_finish - actual_finish, 1),
                })
            comp = pd.DataFrame(rows)

            w = DEFAULT_WEIGHTS
            weights_str = (
                f"Odds {w['odds']:.0%} | Track {w['track']:.0%} | "
                f"Practice {w['practice']:.0%} | Qual {w['qual']:.0%} | "
                f"Track Type {w['track_type']:.0%}"
            )

            if meta and not meta.get("has_odds"):
                st.caption("⚠️ No odds data available for this race — odds signal excluded")

    comp = comp.sort_values("Actual DK", ascending=False).reset_index(drop=True)
    comp.index = comp.index + 1
    comp.index.name = "Rank"

    # Accuracy metrics
    mae_dk = comp["DK Error"].abs().mean()
    mae_finish = comp["Finish Error"].abs().mean()
    corr_dk = comp["Proj DK"].corr(comp["Actual DK"])
    corr_finish = comp["Proj Finish"].corr(comp["Actual Finish"])
    rank_corr = comp["Proj DK"].rank(ascending=False).corr(
        comp["Actual DK"].rank(ascending=False))

    m_cols = st.columns(5)
    m_cols[0].metric("DK Pts MAE", f"{mae_dk:.1f}")
    m_cols[1].metric("Finish MAE", f"{mae_finish:.1f}")
    m_cols[2].metric("DK Pts Correlation", f"{corr_dk:.3f}")
    m_cols[3].metric("Finish Correlation", f"{corr_finish:.3f}")
    m_cols[4].metric("Rank Correlation", f"{rank_corr:.3f}")

    st.caption(
        "**MAE** = Mean Absolute Error (lower is better) | "
        "**Correlation** = how well projected order matches actual (1.0 = perfect) | "
        "**Rank Correlation** = Spearman rank correlation of projected vs actual DK points"
    )
    st.caption(f"Weights: {weights_str}")

    st.dataframe(safe_fillna(format_display_df(comp)), use_container_width=True,
                 hide_index=False, height=500)

    # Scatter: Projected vs Actual DK Points
    import plotly.graph_objects as go
    from src.charts import DARK_LAYOUT

    fig = go.Figure()
    min_val = min(comp["Proj DK"].min(), comp["Actual DK"].min()) - 5
    max_val = max(comp["Proj DK"].max(), comp["Actual DK"].max()) + 5
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color="#444", dash="dash", width=1),
        showlegend=False,
    ))
    colors = np.where(comp["DK Error"] > 0, "#ef4444", "#22c55e")
    fig.add_trace(go.Scatter(
        x=comp["Actual DK"], y=comp["Proj DK"],
        mode="markers+text",
        text=short_name_series(comp["Driver"].tolist()),
        textposition="top right",
        textfont=dict(size=8, color="#8892a4"),
        marker=dict(size=9, color=colors, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Projected: %{y:.1f}<br>Actual: %{x:.1f}<br>"
            "Error: %{customdata[1]:+.1f}<extra></extra>"
        ),
        customdata=np.column_stack([comp["Driver"], comp["DK Error"]]),
        showlegend=False,
    ))
    fig.update_layout(**DARK_LAYOUT, height=500,
                      title="Projected vs Actual DK Points",
                      xaxis_title="Actual DK Points",
                      yaxis_title="Projected DK Points")
    st.plotly_chart(fig, use_container_width=True, key="acc_scatter_dk")

    # Scatter: Projected vs Actual Finish
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[0, 40], y=[0, 40], mode="lines",
        line=dict(color="#444", dash="dash", width=1), showlegend=False,
    ))
    colors2 = np.where(comp["Finish Error"] > 0, "#ef4444", "#22c55e")
    fig2.add_trace(go.Scatter(
        x=comp["Actual Finish"], y=comp["Proj Finish"],
        mode="markers+text",
        text=short_name_series(comp["Driver"].tolist()),
        textposition="top right",
        textfont=dict(size=8, color="#8892a4"),
        marker=dict(size=9, color=colors2, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Projected: %{y:.1f}<br>Actual: %{x}<br>"
            "Error: %{customdata[1]:+.1f}<extra></extra>"
        ),
        customdata=np.column_stack([comp["Driver"], comp["Finish Error"]]),
        showlegend=False,
    ))
    fig2.update_layout(**DARK_LAYOUT, height=450,
                       title="Projected vs Actual Finish Position",
                       xaxis_title="Actual Finish", yaxis_title="Projected Finish",
                       xaxis=dict(autorange="reversed"),
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True, key="acc_scatter_finish")

    # Error distribution
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=comp["DK Error"], nbinsx=20,
                                marker_color="#4a7dfc", opacity=0.8))
    fig3.add_vline(x=0, line_dash="dash", line_color="#888")
    fig3.update_layout(**DARK_LAYOUT, height=300,
                       title="DK Points Error Distribution (Projected - Actual)",
                       xaxis_title="Error (+ = over-projected, - = under-projected)",
                       yaxis_title="Count")
    st.plotly_chart(fig3, use_container_width=True, key="acc_error_dist")

    csv = comp.to_csv(index=True).encode("utf-8")
    st.download_button("Export Comparison CSV", csv,
                       f"accuracy_{race_id}.csv", "text/csv", key="acc_export")


# ── Accuracy Dashboard ───────────────────────────────────────────────────────

def _render_accuracy_dashboard(series_id, selected_year, series_name):
    """Cross-race accuracy metrics — auto-generates projections for all completed races."""
    # Get all completed races for this series/year
    races = fetch_race_list(series_id, selected_year)
    point_races = filter_point_races(races) if races else []
    now = __import__("datetime").datetime.now()
    completed = []
    for race in point_races:
        date_str = race.get("race_date", "")
        try:
            rd = __import__("datetime").datetime.fromisoformat(
                date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
            if rd.date() <= now.date():
                completed.append(race)
        except Exception:
            pass

    if not completed:
        st.info("No completed races available for accuracy tracking.")
        return

    st.caption(f"Auto-generating projections for **{len(completed)} completed races** ({series_name} {selected_year})")

    all_comparisons = []
    race_metrics = []
    progress = st.progress(0)

    for idx, race in enumerate(completed):
        progress.progress((idx + 1) / len(completed),
                          text=f"Processing {race.get('track_name', '')}...")
        race_id = race.get("race_id")
        track_name = race.get("track_name", "")

        proj_dk, actuals, meta = _generate_race_projections(race, series_id)
        if proj_dk is None or actuals is None:
            continue

        proj_list = []
        actual_list = []
        for _, row in actuals.iterrows():
            d = row["Driver"]
            if d in proj_dk:
                proj_list.append(proj_dk[d])
                actual_list.append(row["DK Pts"])

        if len(proj_list) < 5:
            continue

        proj_s = pd.Series(proj_list)
        actual_s = pd.Series(actual_list)
        dk_errors = proj_s - actual_s

        proj_finish_rank = proj_s.rank(ascending=False)
        actual_finish_rank = actual_s.rank(ascending=False)

        # Compute projected finish positions from DK rank and actual finishes
        proj_finish_list = []
        actual_finish_list = []
        sorted_proj = sorted(proj_dk.items(), key=lambda x: x[1], reverse=True)
        proj_finish_map = {d: i + 1 for i, (d, _) in enumerate(sorted_proj)}
        for _, row in actuals.iterrows():
            d = row["Driver"]
            if d in proj_dk and pd.notna(row.get("Finish Position")):
                proj_finish_list.append(proj_finish_map.get(d, len(sorted_proj)))
                actual_finish_list.append(int(row["Finish Position"]))

        finish_mae = 0.0
        finish_corr = 0.0
        if proj_finish_list:
            pf = pd.Series(proj_finish_list)
            af = pd.Series(actual_finish_list)
            finish_mae = (pf - af).abs().mean()
            finish_corr = pf.corr(af) if len(pf) > 2 else 0.0

        race_metrics.append({
            "Race": race.get("race_name", ""),
            "Track": track_name,
            "Season": selected_year,
            "Track Type": TRACK_TYPE_MAP.get(track_name, "intermediate"),
            "Drivers": len(proj_list),
            "DK MAE": dk_errors.abs().mean(),
            "Finish MAE": finish_mae,
            "DK Corr": proj_s.corr(actual_s),
            "Finish Corr": finish_corr,
            "Rank Corr": proj_finish_rank.corr(actual_finish_rank),
            "Avg Bias": dk_errors.mean(),
        })

        for _, row in actuals.iterrows():
            d = row["Driver"]
            if d in proj_dk:
                all_comparisons.append({
                    "Race": race.get("race_name", ""),
                    "Track": track_name,
                    "Track Type": TRACK_TYPE_MAP.get(track_name, "intermediate"),
                    "Driver": d,
                    "Proj DK": proj_dk[d],
                    "Actual DK": row["DK Pts"],
                    "Error": proj_dk[d] - row["DK Pts"],
                })

    progress.progress(1.0, text="Complete!")

    if not race_metrics:
        st.info("Could not generate projections for any completed races.")
        return

    metrics_df = pd.DataFrame(race_metrics)
    all_comp_df = pd.DataFrame(all_comparisons)

    overall_mae = all_comp_df["Error"].abs().mean()
    overall_corr = all_comp_df["Proj DK"].corr(all_comp_df["Actual DK"])
    overall_rank_corr = metrics_df["Rank Corr"].mean()
    overall_bias = all_comp_df["Error"].mean()

    m_cols = st.columns(4)
    m_cols[0].metric("Overall DK MAE", f"{overall_mae:.1f}")
    m_cols[1].metric("Overall DK Correlation", f"{overall_corr:.3f}")
    m_cols[2].metric("Avg Rank Correlation", f"{overall_rank_corr:.3f}")
    m_cols[3].metric("Avg Bias", f"{overall_bias:+.1f}",
                     help="Positive = over-projecting on average, Negative = under-projecting")

    st.markdown("**Race-by-Race Accuracy**")
    race_disp = metrics_df.copy()
    for col in ["DK MAE", "Finish MAE", "DK Corr", "Finish Corr", "Rank Corr", "Avg Bias"]:
        race_disp[col] = race_disp[col].round(2)
    st.dataframe(safe_fillna(race_disp), use_container_width=True, hide_index=True, height=300)

    st.markdown("**Accuracy by Track Type**")
    type_agg = all_comp_df.groupby("Track Type").agg(
        Races=("Race", lambda x: x.nunique()),
        Drivers=("Driver", "count"),
        MAE=("Error", lambda x: x.abs().mean()),
        Bias=("Error", "mean"),
    ).round(2).sort_values("MAE")
    st.dataframe(type_agg, use_container_width=True)

    st.caption(
        "Track your projection accuracy over time. Lower MAE and higher correlation = better model. "
        "Positive bias means you're over-projecting on average."
    )

    # Flag superspeedway accuracy caveat
    if not all_comp_df.empty and "Track Type" in all_comp_df.columns:
        ss_count = all_comp_df[all_comp_df["Track Type"] == "superspeedway"]["Race"].nunique()
        if ss_count > 0:
            st.caption(
                f"⚠️ Superspeedway races ({ss_count} included) are inherently unpredictable "
                "due to pack racing — expect higher MAE and lower correlation at Daytona, "
                "Talladega, Atlanta, and Indianapolis."
            )


# ── Weight Optimizer ─────────────────────────────────────────────────────────

def _render_weight_optimizer(completed_races, series_id, selected_year, series_name):
    """Find optimal weights by backtesting against completed races."""
    st.markdown(f"**Backtest Weight Combinations — {series_name} Series**")
    st.caption(
        "Runs the full 5-signal projection model with different weight combinations "
        "against completed races using DB-stored track history, track type stats, "
        "qualifying positions, speed ratings, and saved odds."
    )

    if not completed_races:
        st.info("No completed races available for backtesting.")
        return

    max_races = min(len(completed_races), 20)
    f_cols = st.columns(3)
    with f_cols[0]:
        n_races = st.slider("Races to backtest", 1, max_races,
                             min(5, max_races), key="acc_n_races")
    with f_cols[1]:
        # Show parent types for filtering (group subtypes together)
        parent_types = sorted(set(TRACK_TYPE_PARENT.get(v, v)
                                   for v in TRACK_TYPE_MAP.values()))
        type_opts = ["All Types"] + parent_types
        track_type_filter = st.selectbox("Track Type Filter", type_opts,
                                          key="acc_tt_filter")
    with f_cols[2]:
        include_dnf = st.checkbox("Include DNF risk adjustment", value=True,
                                   key="acc_dnf_toggle")

    test_races = list(completed_races)
    if track_type_filter != "All Types":
        test_races = [
            (i, r) for i, r in test_races
            if TRACK_TYPE_PARENT.get(
                TRACK_TYPE_MAP.get(r.get("track_name", ""), "intermediate"),
                "intermediate"
            ) == track_type_filter
        ]
    test_races = test_races[-n_races:]

    if not test_races:
        st.info("No completed races match the selected filters.")
        return

    race_names = [f"{r.get('track_name', '')}" for _, r in test_races]
    st.caption(f"Tracks: {', '.join(race_names)}")

    btn_cols = st.columns([1, 1, 4])
    with btn_cols[0]:
        run_clicked = st.button("Run Weight Optimization", type="primary",
                                 key="acc_run_opt")
    with btn_cols[1]:
        cancel_clicked = st.button("Cancel", key="acc_cancel_opt",
                                    type="secondary")

    if cancel_clicked:
        st.session_state["acc_cancel"] = True
        st.info("Cancelling...")
        return

    if run_clicked:
        st.session_state["acc_cancel"] = False
        _run_backtest(test_races, series_id, selected_year, series_name,
                      include_dnf)
    elif "acc_opt_results" in st.session_state:
        # Show previous results without re-running
        _display_backtest_results(
            st.session_state["acc_opt_results"],
            st.session_state.get("acc_opt_series", series_name),
        )


def _run_backtest(test_races, series_id, selected_year, series_name,
                  include_dnf=True, grid_step=5):
    """Run full-signal backtest across weight combinations."""
    from src.utils import fuzzy_match_name

    # ── Per-signal weight constraints ──────────────────────────────────────
    # These define the realistic min/max for each signal.
    # Rationale:
    #   track_history (15-45%): strongest predictor, actual results at this track
    #   odds (15-45%): Vegas pricing is very efficient, reflects overall form
    #   practice (10-35%): noisy — teams sandbag, run diff programs
    #   track_type (5-30%): useful supplement but less predictive than track-specific
    #   qualifying: NOT a finish signal — only used for start position / place diff
    SIGNAL_RANGES = {
        "odds":       (15, 45),
        "track":      (15, 45),
        "practice":   (10, 35),
        "track_type": (5, 30),
    }

    weight_combos = []
    for odds in range(SIGNAL_RANGES["odds"][0], SIGNAL_RANGES["odds"][1] + 1, grid_step):
        for track in range(SIGNAL_RANGES["track"][0], SIGNAL_RANGES["track"][1] + 1, grid_step):
            for prac in range(SIGNAL_RANGES["practice"][0], SIGNAL_RANGES["practice"][1] + 1, grid_step):
                tt = 100 - odds - track - prac
                if SIGNAL_RANGES["track_type"][0] <= tt <= SIGNAL_RANGES["track_type"][1]:
                    weight_combos.append({
                        "odds": odds, "track": track,
                        "qual": 0, "practice": prac,
                        "track_type": tt,
                    })

    ranges_str = " | ".join(f"{k}: {v[0]}-{v[1]}%" for k, v in SIGNAL_RANGES.items())
    st.caption(f"Testing **{len(weight_combos)}** weight combinations across "
               f"{len(test_races)} races ({series_name})")
    st.caption(f"Signal constraints: {ranges_str}")
    progress = st.progress(0)

    # DNF data loaded per-race (with before_date) to prevent future data leakage

    # Pre-load all race data from DB (fast — no API calls for track stats)
    race_data = []
    total_load_steps = len(test_races)

    for idx, (_, race) in enumerate(test_races):
        progress.progress(idx / (total_load_steps + 1),
                          text=f"Loading race {idx + 1}/{total_load_steps}...")
        race_id = race.get("race_id")
        track_name = race.get("track_name", "")
        track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
        yr = race.get("race_date", "")[:4]
        try:
            yr = int(yr)
        except Exception:
            yr = selected_year

        # Load actual results (still needs API for race results)
        feed = fetch_weekend_feed(series_id, race_id, yr)
        laps = fetch_lap_times(series_id, race_id, yr)
        if not feed:
            continue

        results = extract_race_results(feed)
        if results.empty:
            continue

        fl = compute_fastest_laps(laps) if laps else {}
        results["Fastest Laps"] = results["Driver"].map(lambda d: fl.get(d, 0))
        results["DK Pts"] = results.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                     r["Laps Led"], r["Fastest Laps"]), axis=1)

        drivers = results["Driver"].unique().tolist()

        # Exclude this race AND all future races to prevent data leakage
        db_rid = _api_race_id_to_db(race_id)
        race_date = race.get("race_date", "")[:10] if race.get("race_date") else None
        th_data = _query_driver_track_stats(track_name, series_id,
                                             exclude_race_id=db_rid,
                                             before_date=race_date)

        # DB-backed track type history — use parent type for broader data
        parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
        tt_data = _query_driver_track_type_stats(parent_type, series_id,
                                                  exclude_track=track_name,
                                                  exclude_race_id=db_rid,
                                                  before_date=race_date)

        # DNF data — also time-bounded
        dnf_data = _query_driver_career_dnf(series_id, before_date=race_date) if include_dnf else {}

        # Start positions from actual results (qualifying proxy)
        start_positions = {}
        for _, row in results.iterrows():
            if pd.notna(row.get("Start")):
                start_positions[row["Driver"]] = int(row["Start"])

        # Load saved odds for this race from DB
        saved_odds = load_race_odds(race_id)
        odds_finish = {}
        if saved_odds:
            # Filter null/empty odds
            clean_odds = {k: v for k, v in saved_odds.items()
                          if v is not None and str(v).strip() not in ("", "None", "null")}
            odds_probs = {}
            for name, odds_str in clean_odds.items():
                try:
                    odds_val = int(str(odds_str).replace("+", ""))
                    if odds_val > 0:
                        prob = 100 / (odds_val + 100)
                    elif odds_val < 0:
                        prob = abs(odds_val) / (abs(odds_val) + 100)
                    else:
                        continue
                    odds_probs[name] = prob
                except (ValueError, TypeError):
                    continue
            # Only use odds if meaningful coverage
            field_size = len(drivers)
            if len(odds_probs) >= field_size * 0.3:
                ranked = sorted(odds_probs.items(), key=lambda x: x[1], reverse=True)
                for rank, (name, prob) in enumerate(ranked, 1):
                    matched = fuzzy_match_name(name, drivers)
                    if matched:
                        odds_finish[matched] = rank * (field_size / len(ranked))

        # Track which signals are available for this race
        has_signals = {
            "track": bool(th_data),
            "track_type": bool(tt_data),
            "qual": bool(start_positions),
            "practice": bool(th_data),  # uses speed_score from track history
            "odds": bool(odds_finish),
        }

        # Get scheduled laps from race object
        race_laps = race.get("scheduled_laps", 0)
        try:
            race_laps = int(race_laps) if race_laps else 0
        except (ValueError, TypeError):
            race_laps = 0

        # Pre-index actual DK pts per driver for fast lookup in combo loop
        actual_dk = {}
        for _, row in results.iterrows():
            actual_dk[row["Driver"]] = row["DK Pts"]

        race_data.append({
            "race": race,
            "results": results,
            "drivers": drivers,
            "field_size": len(drivers),
            "th_data": th_data,
            "tt_data": tt_data,
            "start_positions": start_positions,
            "odds_finish": odds_finish,
            "has_signals": has_signals,
            "race_laps": race_laps,
            "track_type": track_type,
            "actual_dk": actual_dk,
            "dnf_data": dnf_data,
        })

    if not race_data:
        st.warning("Could not load any race results for backtesting.")
        return

    # Show signal availability summary
    signal_summary = {"track": 0, "track_type": 0, "qual": 0, "practice": 0, "odds": 0}
    for rd in race_data:
        for sig, available in rd["has_signals"].items():
            if available:
                signal_summary[sig] += 1
    sig_str = " | ".join(f"{k}: {v}/{len(race_data)}" for k, v in signal_summary.items())
    st.caption(f"Signal availability across races: {sig_str}")
    if signal_summary["odds"] < len(race_data) * 0.5:
        missing = len(race_data) - signal_summary["odds"]
        st.caption(f"⚠️ Odds missing for {missing}/{len(race_data)} races — "
                   f"run `python refresh_data.py --odds` before each race to save odds for backtesting")

    # Test each weight combination
    combo_results = []
    total_combos = len(weight_combos)
    cancelled = False

    for c_idx, combo in enumerate(weight_combos):
        # Check for cancel every 50 combos
        if c_idx % 50 == 0:
            if st.session_state.get("acc_cancel", False):
                cancelled = True
                break
            progress.progress(
                (total_load_steps + c_idx / total_combos) / (total_load_steps + 1),
                text=f"Testing weight combo {c_idx + 1}/{total_combos}..."
            )

        total_w = sum(combo.values())
        if total_w <= 0:
            continue
        nominal_wn = {k: v / total_w for k, v in combo.items()}

        all_errors = []
        all_rank_corrs = []

        for rd in race_data:
            drivers = rd["drivers"]
            field_size = rd["field_size"]
            results = rd["results"]
            has = rd["has_signals"]

            # Per-race weight redistribution: skip unavailable signals
            effective = {}
            for sig in ["track", "track_type", "qual", "practice", "odds"]:
                effective[sig] = nominal_wn[sig] if has[sig] else 0
            eff_total = sum(effective.values())
            if eff_total <= 0:
                continue
            wn = {k: v / eff_total for k, v in effective.items()}

            # Run full projection (including laps led + fastest laps)
            proj_dk = _project_race_backtest(
                drivers, field_size, wn,
                rd["th_data"], rd["tt_data"],
                rd["start_positions"], rd["odds_finish"],
                rd.get("dnf_data", {}),
                race_laps=rd.get("race_laps", 0),
                track_type=rd.get("track_type", "intermediate"),
            )

            # Compute errors using pre-indexed actual_dk dict
            actual_dk = rd["actual_dk"]
            proj_vals = []
            actual_vals = []
            for d in drivers:
                if d in proj_dk and d in actual_dk:
                    all_errors.append(abs(proj_dk[d] - actual_dk[d]))
                    proj_vals.append(proj_dk[d])
                    actual_vals.append(actual_dk[d])

            # Rank correlation (vectorized)
            if len(proj_vals) > 5:
                proj_s = pd.Series(proj_vals)
                actual_s = pd.Series(actual_vals)
                rc = proj_s.rank().corr(actual_s.rank())
                if pd.notna(rc):
                    all_rank_corrs.append(rc)

        if all_errors:
            combo_results.append({
                "Odds": combo["odds"],
                "Track": combo["track"],
                "Practice": combo["practice"],
                "Track Type": combo["track_type"],
                "MAE": np.mean(all_errors),
                "Rank Corr": np.mean(all_rank_corrs) if all_rank_corrs else 0,
            })

    if cancelled:
        progress.progress(1.0, text="Cancelled!")
        st.warning(f"Cancelled after testing {len(combo_results)} of {total_combos} combinations.")
        st.session_state["acc_cancel"] = False
        if not combo_results:
            return
    else:
        progress.progress(1.0, text="Complete!")

    if not combo_results:
        st.warning("No valid results from backtesting.")
        return

    results_df = pd.DataFrame(combo_results)

    # Sort by composite score (low MAE + high rank corr)
    mae_range = results_df["MAE"].max() - results_df["MAE"].min()
    rc_range = results_df["Rank Corr"].max() - results_df["Rank Corr"].min()
    if mae_range > 0 and rc_range > 0:
        results_df["Score"] = (
            -(results_df["MAE"] - results_df["MAE"].min()) / mae_range * 0.6 +
            (results_df["Rank Corr"] - results_df["Rank Corr"].min()) / rc_range * 0.4
        )
    else:
        results_df["Score"] = -results_df["MAE"]
    results_df = results_df.sort_values("Score", ascending=False)

    # Store results + race data in session state for drill-down
    st.session_state["acc_opt_results"] = results_df
    st.session_state["acc_opt_series"] = series_name
    st.session_state["acc_opt_race_data"] = race_data
    # dnf_data is now per-race inside race_data dicts

    _display_backtest_results(results_df, series_name)


def _display_backtest_results(results_df, series_name):
    """Display backtest results (separated so export doesn't re-run)."""
    # Show top 15
    st.markdown(f"**Top 15 Weight Combinations — {series_name}**")
    top = results_df.head(15).copy()
    top["MAE"] = top["MAE"].round(1)
    top["Rank Corr"] = top["Rank Corr"].round(3)
    if "Score" in top.columns:
        top = top.drop(columns=["Score"])
    top.index = range(1, len(top) + 1)
    top.index.name = "Rank"

    st.dataframe(top, use_container_width=True, hide_index=False)

    best = results_df.iloc[0]
    st.success(
        f"**Best weights ({series_name}):** Odds **{int(best['Odds'])}%** | "
        f"Track **{int(best['Track'])}%** | "
        f"Practice **{int(best['Practice'])}%** | "
        f"Track Type **{int(best['Track Type'])}%** | "
        f"MAE: {best['MAE']:.1f} | Rank Corr: {best['Rank Corr']:.3f}"
    )

    # Show current vs optimal comparison
    current_weights = {
        "Odds": st.session_state.get("pw_odds", 30),
        "Track": st.session_state.get("pw_track", 30),
        "Practice": st.session_state.get("pw_prac", 25),
        "Track Type": st.session_state.get("pw_type", 15),
    }

    current_match = results_df[
        (results_df["Odds"] == current_weights["Odds"]) &
        (results_df["Track"] == current_weights["Track"]) &
        (results_df["Practice"] == current_weights["Practice"]) &
        (results_df["Track Type"] == current_weights["Track Type"])
    ]

    if not current_match.empty:
        curr = current_match.iloc[0]
        rank_of_current = (results_df["Score"] > curr["Score"]).sum() + 1
        st.info(
            f"Your current weights rank **#{rank_of_current}** out of "
            f"{len(results_df)} tested — MAE: {curr['MAE']:.1f}, "
            f"Rank Corr: {curr['Rank Corr']:.3f}"
        )
    else:
        st.caption("Your current weights weren't in the test grid (some are below the 5% minimum floor).")

    # Export (does NOT re-run backtest)
    export_df = results_df.drop(columns=["Score"], errors="ignore").copy()
    export_df.insert(0, "Series", series_name)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export All Results CSV", csv,
                       f"weight_optimization_{series_name}.csv", "text/csv",
                       key="acc_opt_export")

    # ── Drill-down: select a weight combo to see driver-level detail ──
    st.markdown("---")
    st.markdown("**Drill Into Weight Combo**")
    st.caption("Select a weight combination to see projected vs actual driver details")

    race_data = st.session_state.get("acc_opt_race_data", [])

    if race_data and len(results_df) > 0:
        # Build selectable labels from top 15
        top15 = results_df.head(15)
        combo_labels = []
        for idx, row in top15.iterrows():
            lbl = (f"#{len(combo_labels) + 1}: "
                   f"Odds {int(row['Odds'])}% | Track {int(row['Track'])}% | "
                   f"Prac {int(row['Practice'])}% | "
                   f"Type {int(row['Track Type'])}% — "
                   f"MAE {row['MAE']:.1f}, Rank Corr {row['Rank Corr']:.3f}")
            combo_labels.append((lbl, row))

        selected_lbl = st.selectbox(
            "Weight combo", [c[0] for c in combo_labels],
            key="acc_drill_combo"
        )

        # Find the selected combo
        selected_row = None
        for lbl, row in combo_labels:
            if lbl == selected_lbl:
                selected_row = row
                break

        if selected_row is not None:
            # Also let user pick which race to view
            race_labels = [f"{rd['race'].get('track_name', '')} — {rd['race'].get('race_name', '')}"
                           for rd in race_data]
            race_labels.insert(0, "All Races (combined)")
            selected_race_lbl = st.selectbox("Race", race_labels, key="acc_drill_race")

            # Build the normalized weights
            combo = {
                "odds": int(selected_row["Odds"]),
                "track": int(selected_row["Track"]),
                "practice": int(selected_row["Practice"]),
                "qual": 0,
                "track_type": int(selected_row["Track Type"]),
            }
            total_w = sum(combo.values())
            nominal_wn = {k: v / total_w for k, v in combo.items()}

            # Determine which races to show
            if selected_race_lbl == "All Races (combined)":
                show_races = race_data
            else:
                race_idx = race_labels.index(selected_race_lbl) - 1  # offset for "All" option
                show_races = [race_data[race_idx]]

            all_rows = []
            for rd in show_races:
                drivers = rd["drivers"]
                field_size = rd["field_size"]
                results = rd["results"]
                has = rd["has_signals"]
                race_name = rd["race"].get("race_name", "")
                track_name = rd["race"].get("track_name", "")

                # Per-race weight redistribution
                effective = {}
                for sig in ["track", "track_type", "qual", "practice", "odds"]:
                    effective[sig] = nominal_wn[sig] if has[sig] else 0
                eff_total = sum(effective.values())
                if eff_total <= 0:
                    continue
                wn = {k: v / eff_total for k, v in effective.items()}

                # Run the full projection with laps led + fastest laps
                # We need detailed output, so we'll inline the key parts
                proj_dk_totals = _project_race_backtest(
                    drivers, field_size, wn,
                    rd["th_data"], rd["tt_data"],
                    rd["start_positions"], rd["odds_finish"],
                    rd.get("dnf_data", {}),
                    race_laps=rd.get("race_laps", 0),
                    track_type=rd.get("track_type", "intermediate"),
                )

                # Re-derive per-driver details for display
                # (Re-run core logic to get finish/LL/FL breakdowns)
                raw_scores = {}
                dom_scores = {}
                fl_scores_dict = {}
                for d in drivers:
                    th = rd["th_data"].get(d)
                    tt = rd["tt_data"].get(d)
                    sp = rd["start_positions"].get(d)
                    od = rd["odds_finish"].get(d)

                    finish_signals = []
                    signal_weights = []
                    if th and wn.get("track", 0) > 0:
                        finish_signals.append(th["avg_finish"])
                        signal_weights.append(wn["track"])
                    if tt and wn.get("track_type", 0) > 0:
                        finish_signals.append(tt["avg_finish"])
                        signal_weights.append(wn["track_type"])
                    # Qualifying excluded from finish signals — only used for start pos
                    if th and th.get("speed_score", 0) > 0 and wn.get("practice", 0) > 0:
                        max_spd = max((t.get("speed_score", 0) for t in rd["th_data"].values()), default=1)
                        if max_spd > 0:
                            finish_signals.append(max(1, field_size * (1 - th["speed_score"] / max_spd * 0.8)))
                            signal_weights.append(wn["practice"])
                    if od and wn.get("odds", 0) > 0:
                        finish_signals.append(od)
                        signal_weights.append(wn["odds"])
                    if finish_signals:
                        raw_scores[d] = sum(f * w for f, w in zip(finish_signals, signal_weights)) / sum(signal_weights)
                    else:
                        raw_scores[d] = field_size * 0.65

                    # Dom score — uses projection weights
                    dom_s = []
                    dom_dw = []
                    if th and th.get("laps_led_per_race", 0) > 0:
                        dom_s.append(th["laps_led_per_race"]); dom_dw.append(wn.get("track", 0.20))
                    if tt and tt.get("laps_led_per_race", 0) > 0:
                        dom_s.append(tt["laps_led_per_race"]); dom_dw.append(wn.get("track_type", 0.10))
                    if sp and sp <= field_size:
                        dom_s.append(max(0, (field_size + 1 - sp) / field_size) ** 1.5 * 30)
                        dom_dw.append(wn.get("qual", 0.15))
                    if od and wn.get("odds", 0) > 0:
                        dom_s.append(max(0, (field_size + 1 - od) / field_size) ** 1.3 * 35)
                        dom_dw.append(wn.get("odds", 0.15))
                    dom_scores[d] = sum(s * w for s, w in zip(dom_s, dom_dw)) / sum(dom_dw) if dom_s else 0

                    # FL score — uses projection weights
                    fl_s_list = []
                    fl_w_list = []
                    if dom_scores[d] > 0:
                        fl_s_list.append(dom_scores[d] * 0.5); fl_w_list.append(0.25)
                    if sp and sp <= field_size:
                        fl_s_list.append(max(0, (field_size + 1 - sp) / field_size) * 15)
                        fl_w_list.append(wn.get("qual", 0.15))
                    if od and wn.get("odds", 0) > 0:
                        fl_s_list.append(max(0, (field_size + 1 - od) / field_size) * 12)
                        fl_w_list.append(wn.get("odds", 0.15))
                    fl_s_list.append(max(0, (field_size - raw_scores.get(d, 20)) / field_size) * 10)
                    fl_w_list.append(0.10)
                    fl_scores_dict[d] = sum(s * w for s, w in zip(fl_s_list, fl_w_list)) / sum(fl_w_list) if fl_s_list else 0

                # Rank-order finish spreading
                sorted_d = sorted(raw_scores.items(), key=lambda x: x[1])
                n = len(sorted_d)
                proj_finishes = {}
                for rank_idx, (d, rs) in enumerate(sorted_d):
                    if n > 1:
                        t = rank_idx / (n - 1)
                        proj_finishes[d] = max(1, min(field_size, 1 + (field_size - 1) * (t ** 0.85)))
                    else:
                        proj_finishes[d] = field_size * 0.5

                # Allocate LL/FL using same logic as backtest
                race_laps = rd.get("race_laps", 0)
                tt_key = rd.get("track_type", "intermediate")
                parent = TRACK_TYPE_PARENT.get(tt_key, tt_key)
                LEADER_FRAC = {
                    "superspeedway": 0.60, "road": 0.22, "short": 0.18,
                    "short_concrete": 0.16, "intermediate": 0.22, "intermediate_worn": 0.20,
                }
                CONC = {
                    "superspeedway": 0.6, "road": 1.0, "short": 2.0,
                    "short_concrete": 2.2, "intermediate": 1.5, "intermediate_worn": 1.6,
                }
                ll_conc = CONC.get(tt_key, CONC.get(parent, 1.5))

                allocated_ll = {}
                if race_laps > 0 and dom_scores:
                    ll_frac = LEADER_FRAC.get(tt_key, LEADER_FRAC.get(parent, 0.22))
                    n_leaders = max(3, int(field_size * ll_frac))
                    top_dom = sorted(dom_scores.items(), key=lambda x: x[1], reverse=True)[:n_leaders]
                    ll_s = {d: max(0.01, s) ** ll_conc for d, s in top_dom}
                    ll_t = sum(ll_s.values())
                    if ll_t > 0:
                        allocated_ll = {d: (s / ll_t) * race_laps for d, s in ll_s.items()}

                FL_FRAC = {
                    "superspeedway": 0.85, "road": 0.55, "short": 0.55,
                    "short_concrete": 0.50, "intermediate": 0.65, "intermediate_worn": 0.60,
                }
                fl_conc = max(0.5, ll_conc * 0.7)
                allocated_fl = {}
                if race_laps > 0 and fl_scores_dict:
                    fl_frac = FL_FRAC.get(tt_key, FL_FRAC.get(parent, 0.55))
                    n_fl = max(5, int(field_size * fl_frac))
                    top_fl = sorted(fl_scores_dict.items(), key=lambda x: x[1], reverse=True)[:n_fl]
                    fl_s = {d: max(0.01, s) ** fl_conc for d, s in top_fl}
                    fl_t = sum(fl_s.values())
                    if fl_t > 0:
                        allocated_fl = {d: (s / fl_t) * race_laps for d, s in fl_s.items()}

                # Build output rows
                for d in drivers:
                    pf = proj_finishes.get(d, 20)
                    sp = rd["start_positions"].get(d)
                    start = sp if sp else round(pf)
                    p_ll = allocated_ll.get(d, 0)
                    p_fl = allocated_fl.get(d, 0)

                    pf_int = round(pf)
                    finish_pts = DK_FINISH_POINTS.get(max(1, min(40, pf_int)), 0)
                    diff_pts = int(start - pf_int)
                    led_pts = p_ll * 0.25
                    fl_pts = p_fl * 0.45
                    proj_total = finish_pts + diff_pts + led_pts + fl_pts

                    # Actual results
                    act_row = results[results["Driver"] == d]
                    actual_dk = act_row["DK Pts"].values[0] if len(act_row) > 0 else None
                    actual_finish = act_row["Finish Position"].values[0] if len(act_row) > 0 else None
                    actual_ll = act_row["Laps Led"].values[0] if len(act_row) > 0 else 0
                    actual_fl = act_row["Fastest Laps"].values[0] if len(act_row) > 0 else 0

                    row_data = {
                        "Driver": d,
                        "Start": start,
                        "Proj Finish": round(pf, 1),
                        "Actual Finish": actual_finish,
                        "Proj LL": round(p_ll),
                        "Actual LL": actual_ll,
                        "Proj FL": round(p_fl),
                        "Actual FL": actual_fl,
                        "Proj DK": round(proj_total, 1),
                        "Actual DK": round(actual_dk, 1) if actual_dk is not None else None,
                        "Error": round(proj_total - actual_dk, 1) if actual_dk is not None else None,
                    }
                    if len(show_races) > 1:
                        row_data["Race"] = f"{track_name}"
                    all_rows.append(row_data)

            if all_rows:
                detail_df = pd.DataFrame(all_rows)
                detail_df = detail_df.sort_values("Actual DK", ascending=False, na_position="last")
                detail_df.index = range(1, len(detail_df) + 1)
                detail_df.index.name = "Rank"

                st.dataframe(safe_fillna(format_display_df(detail_df)),
                             use_container_width=True, hide_index=False, height=500)

                # Summary metrics for this combo
                valid = detail_df.dropna(subset=["Actual DK"])
                if len(valid) > 5:
                    mc = st.columns(4)
                    mc[0].metric("MAE", f"{valid['Error'].abs().mean():.1f}")
                    mc[1].metric("Avg Error", f"{valid['Error'].mean():+.1f}")
                    mc[2].metric("LL MAE", f"{(valid['Proj LL'] - valid['Actual LL']).abs().mean():.1f}")
                    mc[3].metric("FL MAE", f"{(valid['Proj FL'] - valid['Actual FL']).abs().mean():.1f}")

                    # Scatter: Projected vs Actual DK Points with trend line
                    import plotly.graph_objects as go
                    from src.charts import DARK_LAYOUT

                    fig = go.Figure()
                    min_val = min(valid["Proj DK"].min(), valid["Actual DK"].min()) - 5
                    max_val = max(valid["Proj DK"].max(), valid["Actual DK"].max()) + 5

                    # Perfect prediction line
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode="lines", line=dict(color="#444", dash="dash", width=1),
                        showlegend=False, name="Perfect",
                    ))

                    # Data points
                    colors = np.where(valid["Error"] > 0, "#ef4444", "#22c55e")
                    fig.add_trace(go.Scatter(
                        x=valid["Actual DK"], y=valid["Proj DK"],
                        mode="markers",
                        marker=dict(size=8, color=colors, opacity=0.7),
                        hovertemplate=(
                            "<b>%{customdata[0]}</b><br>"
                            "Projected: %{y:.1f}<br>Actual: %{x:.1f}<br>"
                            "Error: %{customdata[1]:+.1f}<extra></extra>"
                        ),
                        customdata=np.column_stack([valid["Driver"], valid["Error"]]),
                        showlegend=False,
                    ))

                    # Trend line (linear regression)
                    z = np.polyfit(valid["Actual DK"].values, valid["Proj DK"].values, 1)
                    trend_x = np.linspace(min_val, max_val, 50)
                    trend_y = z[0] * trend_x + z[1]
                    corr = valid["Proj DK"].corr(valid["Actual DK"])
                    fig.add_trace(go.Scatter(
                        x=trend_x, y=trend_y,
                        mode="lines", line=dict(color="#4a7dfc", width=2),
                        name=f"Trend (r={corr:.3f})",
                    ))

                    fig.update_layout(
                        **DARK_LAYOUT, height=450,
                        title=f"Projected vs Actual DK Points — {selected_lbl.split(':')[0]}",
                        xaxis_title="Actual DK Points",
                        yaxis_title="Projected DK Points",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="acc_drill_scatter")

                csv_detail = detail_df.to_csv(index=True).encode("utf-8")
                st.download_button("Export Detail CSV", csv_detail,
                                   f"weight_detail_{series_name}.csv", "text/csv",
                                   key="acc_drill_export")

    # Clear saved projections button (separate from export)
    st.markdown("---")
    if st.button("Clear Saved Projections", key="acc_clear_proj",
                  type="secondary",
                  help="Delete all saved projection data from the database"):
        if os.path.exists(PROJ_DB):
            conn = sqlite3.connect(PROJ_DB)
            conn.execute("DELETE FROM saved_projections")
            conn.commit()
            conn.close()
            st.success("Cleared all saved projections.")
        else:
            st.warning("No database found.")
