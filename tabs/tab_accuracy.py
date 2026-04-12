"""Tab 7: Accuracy — Projections vs Actuals backtesting and weight optimization."""

import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import os
from datetime import datetime

from src.config import (
    SERIES_OPTIONS, TRACK_TYPE_MAP, TRACK_TYPE_PARENT, TRACK_TYPE_DISPLAY,
    TRACK_TYPE_WEIGHT_DEFAULTS,
)
from src.components import section_header
from src.data import (
    fetch_race_list, fetch_weekend_feed, fetch_lap_times,
    extract_race_results, compute_fastest_laps,
    filter_point_races, query_salaries, load_race_odds,
    query_driver_career_dnf,
)
from src.utils import (
    calc_dk_points, safe_fillna, format_display_df, short_name_series,
    fuzzy_match_name, fuzzy_get, build_norm_lookup,
)

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
    """Create the saved_projections table if it doesn't exist.

    Also clears any pre-existing projections that were contaminated by
    data leakage (saved before the date-filtering fix was applied).
    """
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
    # One-time migration: clear projections saved before 2026-04-08
    # (contaminated by data leakage — track history included future results)
    try:
        conn.execute(
            "DELETE FROM saved_projections WHERE saved_at < '2026-04-08'"
        )
    except Exception:
        pass
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
    fl_norm = build_norm_lookup(fl)
    results["Fastest Laps"] = results["Driver"].map(
        lambda d: fuzzy_get(d, fl, fl_norm) or 0).astype("Int64")
    results["DK Pts"] = results.apply(
        lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                 r["Laps Led"], r["Fastest Laps"]), axis=1)
    return results



# _query_driver_track_stats and _query_driver_track_type_stats removed —
# accuracy tab now uses _query_db_track_history and _query_db_track_type_history
# from the projections tab for identical data loading.



# _hybrid_track_stats and _hybrid_track_type_stats removed — accuracy tab
# now uses the same DB queries as the projections tab via compute_projections().


def _get_default_weights(track_type="intermediate"):
    """Get normalized default weights for a track type from shared config.

    Returns dict with keys: track, track_type, qual, practice, odds, team (all 0-1 floats).
    """
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    raw = TRACK_TYPE_WEIGHT_DEFAULTS.get(parent, TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
    total = raw["odds"] + raw["track"] + raw["ttype"] + raw["prac"] + raw.get("team", 0) + raw.get("qual", 0)
    return {
        "track": raw["track"] / total,
        "track_type": raw["ttype"] / total,
        "odds": raw["odds"] / total,
        "practice": raw["prac"] / total,
        "qual": raw.get("qual", 0) / total,
        "team": raw.get("team", 0) / total,
    }


DEFAULT_WEIGHTS = _get_default_weights("intermediate")


def _generate_race_projections(race, series_id, weights=None):
    """Generate projections for a completed race using the shared engine.

    Uses the SAME DB queries and compute_projections() engine as the live
    Projections tab to guarantee identical results.

    Returns (proj_dict, proj_detail, actuals_df, meta_dict) or (None,)*4.
    proj_dict: {driver: proj_dk_points}
    """
    from tabs.tab_projections import (
        _query_db_track_history, _query_db_track_type_history,
        _resolve_db_race_id, _get_track_dominator_calibration,
    )
    from src.data import (
        query_team_stats, query_manufacturer_stats,
        compute_team_adjusted_track_history, round_odds,
        query_driver_dk_points_at_track,
    )
    from src.config import CROSS_SERIES_HIERARCHY
    from src.projections import compute_projections
    import math

    race_id = race.get("race_id")
    track_name = race.get("track_name", "")
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")

    if weights is None:
        weights = _get_default_weights(track_type)

    actuals = _load_actual_results(race, series_id)
    if actuals.empty:
        return None, None, None, None

    drivers = actuals["Driver"].unique().tolist()
    field_size = len(drivers)
    race_date = race.get("race_date", "")[:10] if race.get("race_date") else None
    mid_field = field_size * 0.5

    # ── Resolve DB race ID for exclusion ──
    db_race_id = _resolve_db_race_id(race_id, series_id) if race_id else None

    # ── Cross-series hierarchy ──
    cross_ids = CROSS_SERIES_HIERARCHY.get(series_id, [])

    # ── 1. Track History (same DB query as projections tab) ──
    th_data = {}
    th_result = _query_db_track_history(
        track_name, series_id,
        exclude_race_id=db_race_id,
        before_date=race_date,
        cross_series_ids=cross_ids if cross_ids else None,
    )
    if cross_ids:
        th_df, cross_th_df = th_result
    else:
        th_df = th_result
        cross_th_df = pd.DataFrame()

    dk_history = query_driver_dk_points_at_track(
        track_name, series_id, min_season=2022, before_date=race_date,
    )

    # Build cross-series track history lookup
    cross_th_lookup = {}
    if not cross_th_df.empty:
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races",
                     "Avg Rating", "Wins", "Top 5", "Top 10", "DNF"]:
            if col in cross_th_df.columns:
                cross_th_df[col] = pd.to_numeric(cross_th_df[col], errors="coerce")
        cross_idx = cross_th_df.drop_duplicates("Driver").set_index("Driver")
        for d in drivers:
            matched = d if d in cross_idx.index else fuzzy_match_name(d, cross_idx.index.tolist())
            if matched and matched in cross_idx.index:
                row = cross_idx.loc[matched]
                races = row.get("Races", 1) if pd.notna(row.get("Races")) and row.get("Races") > 0 else 1
                arp = row.get("Avg Run Pos") if pd.notna(row.get("Avg Run Pos", None)) else None
                cross_th_lookup[d] = {
                    "avg_finish": row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20,
                    "avg_running_pos": arp,
                    "races": races,
                }

    if not th_df.empty:
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races",
                     "Avg Rating", "Wins", "Top 5", "Top 10", "DNF"]:
            if col in th_df.columns:
                th_df[col] = pd.to_numeric(th_df[col], errors="coerce")
        th_idx = th_df.drop_duplicates("Driver").set_index("Driver")
        th_names = th_idx.index.tolist()
        for d in drivers:
            matched = d if d in th_idx.index else fuzzy_match_name(d, th_names)
            if matched and matched in th_idx.index:
                row = th_idx.loc[matched]
                races = row.get("Races", 1) if pd.notna(row.get("Races")) and row.get("Races") > 0 else 1
                laps_led = row.get("Laps Led", 0) if pd.notna(row.get("Laps Led")) else 0
                fastest_laps = row.get("Fastest Laps", 0) if pd.notna(row.get("Fastest Laps")) else 0
                dk_hist_entry = dk_history.get(matched) or dk_history.get(d) or {}

                arp = row.get("Avg Run Pos") if pd.notna(row.get("Avg Run Pos", None)) else None
                af = row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20

                # Cross-series blending: only blend if cross-series is BETTER
                cross = cross_th_lookup.get(d)
                if cross and cross["races"] >= 2:
                    c_af = cross["avg_finish"]
                    c_arp = cross["avg_running_pos"]
                    if c_af < af:
                        af = af * 0.70 + c_af * 0.30
                        if arp is not None and c_arp is not None and c_arp < arp:
                            arp = arp * 0.70 + c_arp * 0.30
                        elif c_arp is not None and c_arp < (arp or 99):
                            arp = c_arp

                th_data[d] = {
                    "avg_finish": af,
                    "avg_start": row.get("Avg Start", 20) if pd.notna(row.get("Avg Start")) else 20,
                    "avg_running_pos": arp,
                    "laps_led": laps_led,
                    "fastest_laps": fastest_laps,
                    "laps_led_per_race": laps_led / races,
                    "fastest_laps_per_race": fastest_laps / races,
                    "races": races,
                    "avg_dk": dk_hist_entry.get("avg_dk"),
                    "wins": row.get("Wins", 0) if pd.notna(row.get("Wins")) else 0,
                    "top5": row.get("Top 5", 0) if pd.notna(row.get("Top 5")) else 0,
                    "dnf": row.get("DNF", 0) if pd.notna(row.get("DNF")) else 0,
                }

    # Drivers with NO current-series track history but WITH cross-series data
    SERIES_FIELD_SIZE = {1: 36, 2: 38, 3: 36}
    for d in drivers:
        if d not in th_data and d in cross_th_lookup:
            cross = cross_th_lookup[d]
            if cross["races"] >= 2:
                source_field = max(SERIES_FIELD_SIZE.get(cross_ids[0], 36) if cross_ids else 36, 1)
                c_af = cross["avg_finish"]
                c_arp = cross["avg_running_pos"]
                pct = c_af / source_field
                scaled_af = pct * field_size
                scaled_arp = (c_arp / source_field) * field_size if c_arp else None
                th_data[d] = {
                    "avg_finish": scaled_af, "avg_start": 20,
                    "avg_running_pos": scaled_arp,
                    "laps_led": 0, "fastest_laps": 0,
                    "laps_led_per_race": 0, "fastest_laps_per_race": 0,
                    "races": cross["races"], "avg_dk": None,
                    "wins": 0, "top5": 0, "dnf": 0,
                    "_cross_series_only": True,
                }

    # ── 2. Track Type (same DB query as projections tab) ──
    tt_data = {}
    if weights.get("track_type", 0) > 0:
        tt_result = _query_db_track_type_history(
            track_type, series_id,
            exclude_track=track_name,
            exclude_race_id=db_race_id,
            before_date=race_date,
            cross_series_ids=cross_ids if cross_ids else None,
        )
        if cross_ids:
            tt_raw, tt_cross_raw = tt_result
        else:
            tt_raw = tt_result
            tt_cross_raw = {}

        if tt_cross_raw:
            source_field = max(SERIES_FIELD_SIZE.get(cross_ids[0], 36) if cross_ids else 36, 1)
            for name, cdata in tt_cross_raw.items():
                if name in tt_raw and cdata.get("races", 0) >= 2:
                    cur = tt_raw[name]
                    c_af = cdata["avg_finish"]
                    if c_af < cur["avg_finish"]:
                        cur["avg_finish"] = cur["avg_finish"] * 0.70 + c_af * 0.30
                        c_arp = cdata.get("avg_running_pos")
                        if cur.get("avg_running_pos") and c_arp and c_arp < cur["avg_running_pos"]:
                            cur["avg_running_pos"] = cur["avg_running_pos"] * 0.70 + c_arp * 0.30
                elif name not in tt_raw and cdata.get("races", 0) >= 3:
                    scaled = dict(cdata)
                    scaled["avg_finish"] = (cdata["avg_finish"] / source_field) * field_size
                    if cdata.get("avg_running_pos"):
                        scaled["avg_running_pos"] = (cdata["avg_running_pos"] / source_field) * field_size
                    scaled["_cross_series_only"] = True
                    tt_raw[name] = scaled

        if tt_raw:
            tt_names = list(tt_raw.keys())
            for d in drivers:
                matched = d if d in tt_raw else fuzzy_match_name(d, tt_names)
                if matched and matched in tt_raw:
                    tt_data[d] = tt_raw[matched]

    # ── 3. Team Quality Signal ──
    # For completed races, get driver→team mapping from DB race_results
    team_signal = {}
    team_adj_data = {}
    driver_team_map = {}
    if weights.get("team", 0) > 0:
        try:
            conn = sqlite3.connect(PROJ_DB)
            # Get team from the most recent race before this date for each driver
            team_rows = conn.execute('''
                SELECT d.full_name, rr.team
                FROM race_results rr
                JOIN drivers d ON d.id = rr.driver_id
                JOIN races r ON r.id = rr.race_id
                WHERE r.series_id = ? AND rr.team IS NOT NULL AND rr.team != ''
                  AND r.race_date < ?
                GROUP BY d.id
                HAVING r.race_date = MAX(r.race_date)
            ''', (series_id, race_date)).fetchall()
            conn.close()
            for name, team in team_rows:
                driver_team_map[name] = team
        except Exception:
            pass

        team_stats = query_team_stats(
            series_id, track_type=track_type, min_season=2022,
            before_date=race_date,
        )
        if team_stats:
            ts_names = list(team_stats.keys())
            for d in drivers:
                team_name = driver_team_map.get(d)
                if not team_name:
                    matched_d = fuzzy_match_name(d, list(driver_team_map.keys()))
                    team_name = driver_team_map.get(matched_d) if matched_d else None
                if not team_name:
                    continue
                matched_team = team_name if team_name in team_stats else fuzzy_match_name(team_name, ts_names)
                if matched_team and matched_team in team_stats:
                    ts = team_stats[matched_team]
                    ts_arp = ts.get("avg_arp")
                    ts_af = ts["avg_finish"]
                    team_finish = ts_arp * 0.65 + ts_af * 0.35 if ts_arp is not None else ts_af
                    trust = min(1.0, ts["races"] / 10)
                    team_signal[d] = team_finish * trust + mid_field * (1 - trust)

        # Team-adjusted track history
        if driver_team_map:
            # Map drivers list to their teams
            dtm_for_adj = {}
            for d in drivers:
                tm = driver_team_map.get(d)
                if not tm:
                    matched_d = fuzzy_match_name(d, list(driver_team_map.keys()))
                    tm = driver_team_map.get(matched_d) if matched_d else None
                if tm:
                    dtm_for_adj[d] = tm
            if dtm_for_adj:
                team_adj_data = compute_team_adjusted_track_history(
                    track_name, series_id, dtm_for_adj,
                    before_date=race_date,
                )

    # ── 4. Manufacturer Adjustment ──
    mfr_adjustment = {}
    mfr_stats = query_manufacturer_stats(
        series_id, track_type=track_type, min_season=2022,
        before_date=race_date,
    )
    if mfr_stats:
        total_races = sum(m["races"] for m in mfr_stats.values())
        field_avg_finish = (sum(m["avg_finish"] * m["races"] for m in mfr_stats.values()) / total_races
                           if total_races > 0 else mid_field)

        # Get driver→manufacturer from DB
        driver_mfr_map = {}
        try:
            conn = sqlite3.connect(PROJ_DB)
            mfr_rows = conn.execute('''
                SELECT d.full_name, rr.manufacturer
                FROM race_results rr
                JOIN drivers d ON d.id = rr.driver_id
                JOIN races r ON r.id = rr.race_id
                WHERE r.series_id = ? AND rr.manufacturer IS NOT NULL AND rr.manufacturer != ''
                  AND r.race_date < ?
                GROUP BY d.id
                HAVING r.race_date = MAX(r.race_date)
            ''', (series_id, race_date)).fetchall()
            conn.close()
            for name, mfr in mfr_rows:
                driver_mfr_map[name] = mfr
        except Exception:
            pass

        mfr_names = list(mfr_stats.keys())
        for d in drivers:
            mfr = driver_mfr_map.get(d)
            if not mfr:
                matched_d = fuzzy_match_name(d, list(driver_mfr_map.keys()))
                mfr = driver_mfr_map.get(matched_d) if matched_d else None
            if not mfr:
                continue
            matched_mfr = mfr if mfr in mfr_stats else fuzzy_match_name(mfr, mfr_names)
            if matched_mfr and matched_mfr in mfr_stats:
                delta = mfr_stats[matched_mfr]["avg_finish"] - field_avg_finish
                adj = max(-1.5, min(1.5, delta * 0.5))
                mfr_adjustment[d] = adj

    # ── 5. Qualifying (from actual start positions) ──
    qual_pos = {}
    for _, row in actuals.iterrows():
        if pd.notna(row.get("Start")):
            qual_pos[row["Driver"]] = int(row["Start"])

    # ── 6. Practice — load from NASCAR API (same source as projections tab) ──
    practice_data = {}
    try:
        from src.data import fetch_lap_averages
        yr = _get_race_year(race)
        lap_avg_df = fetch_lap_averages(series_id, race_id, yr)
        if not lap_avg_df.empty and "Overall Rank" in lap_avg_df.columns:
            for _, prow in lap_avg_df.iterrows():
                pdriver = prow.get("Driver")
                prank = prow.get("Overall Rank")
                if pdriver and prank and not pd.isna(prank):
                    matched = fuzzy_match_name(pdriver, drivers)
                    if matched:
                        practice_data[matched] = int(prank)
    except Exception:
        pass

    # ── 7. Odds Signal ──
    odds_finish = {}
    driver_odds_display = {}
    saved_odds = load_race_odds(race_id, series_id)
    if saved_odds and weights.get("odds", 0) > 0:
        clean_odds = {k: v for k, v in saved_odds.items()
                      if v is not None and str(v).strip() not in ("", "None", "null", "N/A")}

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
            log_probs = {name: math.log(prob) for name, prob in ranked}
            max_lp = max(log_probs.values())
            min_lp = min(log_probs.values())
            lp_range = max_lp - min_lp
            for name, prob in ranked:
                matched = fuzzy_match_name(name, drivers)
                if matched:
                    if lp_range > 0:
                        t = 1 - (log_probs[name] - min_lp) / lp_range
                        odds_finish[matched] = 1 + (field_size - 1) * t
                    else:
                        odds_finish[matched] = mid_field

        # Build odds display dict (for dominator scoring)
        for name, odds_str in clean_odds.items():
            try:
                oval = int(str(odds_str).replace("+", ""))
                if oval > 0:
                    impl = 100 / (oval + 100) * 100
                elif oval < 0:
                    impl = abs(oval) / (abs(oval) + 100) * 100
                else:
                    continue
                matched = fuzzy_match_name(name, drivers)
                if matched:
                    rounded = round_odds(oval)
                    driver_odds_display[matched] = {
                        "odds_str": rounded,
                        "impl_pct": round(impl, 1),
                    }
            except (ValueError, TypeError):
                continue

    # ── 8. DNF risk data ──
    dnf_data = query_driver_career_dnf(series_id, before_date=race_date)

    # ── 9. Race laps + calibration ──
    race_laps = race.get("scheduled_laps", 0)
    try:
        race_laps = int(race_laps) if race_laps else 0
    except (ValueError, TypeError):
        race_laps = 0

    calibration = _get_track_dominator_calibration(track_name, track_type, series_id)

    # ── 10. Run shared projection engine ──
    proj_rows, proj_detail = compute_projections(
        drivers=drivers,
        field_size=field_size,
        wn=weights,
        th_data=th_data,
        tt_data=tt_data,
        qual_pos=qual_pos,
        practice_data=practice_data,
        odds_finish=odds_finish,
        odds_display=driver_odds_display,
        team_signal=team_signal,
        mfr_adjustment=mfr_adjustment,
        team_adj_data=team_adj_data,
        dnf_data=dnf_data,
        race_laps=race_laps,
        track_name=track_name,
        track_type=track_type,
        series_id=series_id,
        calibration=calibration,
        cross_th_lookup=cross_th_lookup,
    )

    # Convert proj_rows list to {driver: dk_points} dict
    proj_dk = {row["driver"]: row["proj_dk"] for row in proj_rows}

    return proj_dk, proj_detail, actuals, {
        "has_odds": bool(odds_finish),
        "has_track": bool(th_data),
        "has_qual": bool(qual_pos),
    }



# _project_race_backtest removed — accuracy tab now uses compute_projections()
# from src/projections.py, the same engine as the live Projections tab.


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, completed_races, series_id, selected_year, series_name="Cup"):
    """Render the Accuracy tab."""
    section_header("Projection Accuracy")
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

    with st.spinner("Generating projections and loading results..."):
        # Always run live projection engine to stay in sync with projections tab
        proj_dk, proj_detail, actuals, meta = _generate_race_projections(
            actual_race, series_id
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
            det = proj_detail.get(d, {}) if proj_detail else {}
            proj_finish = det.get("proj_finish") if det else None
            if proj_finish is None:
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
                "Proj LL": det.get("laps_led", 0),
                "Actual LL": int(row.get("Laps Led", 0) or 0),
                "Proj FL": det.get("fast_laps", 0),
                "Actual FL": int(row.get("Fastest Laps", 0) or 0),
            })
        comp = pd.DataFrame(rows)

        track_name_acc = actual_race.get("track_name", "")
        track_type_acc = TRACK_TYPE_MAP.get(track_name_acc, "intermediate")
        w = _get_default_weights(track_type_acc)
        w_parts = []
        for lbl, key in [("Odds", "odds"), ("Track", "track"), ("Track Type", "track_type"),
                          ("Practice", "practice"), ("Team", "team"), ("Qual", "qual")]:
            v = w.get(key, 0)
            if v > 0:
                w_parts.append(f"{lbl} {v:.0%}")
        weights_str = " | ".join(w_parts) if w_parts else "Default"

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

    st.dataframe(safe_fillna(format_display_df(comp)), width="stretch",
                 hide_index=False, height=500)

    # Scatter: Projected vs Actual DK Points
    import plotly.graph_objects as go
    from src.charts import DARK_LAYOUT, apply_dark_theme

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
    apply_dark_theme(fig)
    st.plotly_chart(fig, width="stretch", key="acc_scatter_dk")

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
    apply_dark_theme(fig2)
    st.plotly_chart(fig2, width="stretch", key="acc_scatter_finish")

    # Error distribution
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=comp["DK Error"], nbinsx=20,
                                marker_color="#0ea5e9", opacity=0.8))
    fig3.add_vline(x=0, line_dash="dash", line_color="#888")
    fig3.update_layout(**DARK_LAYOUT, height=300,
                       title="DK Points Error Distribution (Projected - Actual)",
                       xaxis_title="Error (+ = over-projected, - = under-projected)",
                       yaxis_title="Count")
    apply_dark_theme(fig3)
    st.plotly_chart(fig3, width="stretch", key="acc_error_dist")

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

        proj_dk, proj_detail_agg, actuals, meta = _generate_race_projections(race, series_id)
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

        # Compute projected finish positions and actual finishes
        proj_finish_list = []
        actual_finish_list = []
        for _, row in actuals.iterrows():
            d = row["Driver"]
            det = proj_detail_agg.get(d, {}) if proj_detail_agg else {}
            pf = det.get("proj_finish")
            if pf is not None and pd.notna(row.get("Finish Position")):
                proj_finish_list.append(pf)
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
    st.dataframe(safe_fillna(race_disp), width="stretch", hide_index=True, height=300)

    st.markdown("**Accuracy by Track Type**")
    type_agg = all_comp_df.groupby("Track Type").agg(
        Races=("Race", lambda x: x.nunique()),
        Drivers=("Driver", "count"),
        MAE=("Error", lambda x: x.abs().mean()),
        Bias=("Error", "mean"),
    ).round(2).sort_values("MAE")
    st.dataframe(type_agg, width="stretch")

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

def _query_db_completed_races(series_ids, track_type=None, track_name=None):
    """Query completed races from DB matching filters.

    Returns list of (index, race_dict) tuples compatible with _run_backtest.
    race_dict keys: race_id, track_name, race_date, scheduled_laps, race_name, series_id
    """
    if not os.path.exists(PROJ_DB):
        return []

    conn = sqlite3.connect(PROJ_DB)

    placeholders = ",".join("?" for _ in series_ids)
    params = list(series_ids)

    # Only races that have results (completed)
    query = f'''
        SELECT DISTINCT r.api_race_id, t.name, r.race_date, r.laps,
               r.race_name, r.series_id
        FROM races r
        JOIN tracks t ON t.id = r.track_id
        JOIN race_results rr ON rr.race_id = r.id
        WHERE r.series_id IN ({placeholders})
          AND r.api_race_id IS NOT NULL
          AND r.race_date < date('now')
    '''

    if track_name:
        query += " AND t.name = ?"
        params.append(track_name)

    query += " GROUP BY r.id HAVING COUNT(rr.id) >= 10"
    query += " ORDER BY r.race_date DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = []
    for i, row in enumerate(rows):
        api_race_id, t_name, race_date, laps, race_name, sid = row
        tt = TRACK_TYPE_MAP.get(t_name, "intermediate")
        parent = TRACK_TYPE_PARENT.get(tt, tt)

        # Filter by track type if specified
        if track_type and parent != track_type:
            continue

        result.append((i, {
            "race_id": api_race_id,
            "track_name": t_name,
            "race_date": race_date,
            "scheduled_laps": laps or 0,
            "race_name": race_name,
            "series_id": sid,
        }))

    return result


def _render_weight_optimizer(completed_races, series_id, selected_year, series_name):
    """Find optimal weights by backtesting against completed races."""
    from src.config import SERIES_OPTIONS, SERIES_LABELS

    st.markdown("**Weight Optimizer**")
    st.caption(
        "Find optimal signal weights by backtesting the projection model "
        "against completed races. Search by track type or specific track."
    )

    # ── Search mode + series controls ──
    ctrl_cols = st.columns([1.5, 1.5, 1.5, 1])
    with ctrl_cols[0]:
        search_mode = st.radio("Search By", ["Track Type", "Specific Track"],
                               horizontal=True, key="acc_search_mode")
    with ctrl_cols[1]:
        series_opts = list(SERIES_OPTIONS.keys())
        default_idx = series_opts.index(series_name) if series_name in series_opts else 0
        primary_series = st.selectbox("Primary Series", series_opts,
                                       index=default_idx, key="acc_opt_series_sel")
        primary_sid = SERIES_OPTIONS[primary_series]
    with ctrl_cols[2]:
        cross_series = st.checkbox("Include other series", value=False,
                                    key="acc_cross_series",
                                    help="Add races from other series for more sample size")
    with ctrl_cols[3]:
        include_dnf = st.checkbox("DNF adjustment", value=True,
                                   key="acc_dnf_toggle")

    # Build series list
    if cross_series:
        query_series = list(SERIES_OPTIONS.values())
    else:
        query_series = [primary_sid]

    # ── Track type or track selection ──
    if search_mode == "Track Type":
        parent_types = sorted(set(TRACK_TYPE_PARENT.get(v, v)
                                   for v in TRACK_TYPE_MAP.values()))
        # Default to current track type if available
        current_tt = st.session_state.get("acc_current_track_type")
        default_tt = parent_types.index(current_tt) if current_tt in parent_types else 0
        selected_type = st.selectbox("Track Type", parent_types,
                                      index=default_tt, key="acc_tt_select",
                                      format_func=lambda x: TRACK_TYPE_DISPLAY.get(x, x.title()))

        all_races = _query_db_completed_races(query_series, track_type=selected_type)
        context_label = f"{selected_type.replace('_', ' ').title()}"
        current_defaults = TRACK_TYPE_WEIGHT_DEFAULTS.get(selected_type,
                            TRACK_TYPE_WEIGHT_DEFAULTS.get("intermediate", {}))
    else:
        # Build list of tracks that have race data
        all_db_races = _query_db_completed_races(query_series)
        track_counts = {}
        for _, r in all_db_races:
            tn = r["track_name"]
            track_counts[tn] = track_counts.get(tn, 0) + 1
        # Sort by race count descending
        track_list = sorted(track_counts.keys(), key=lambda t: track_counts[t], reverse=True)
        if not track_list:
            st.info("No completed races found in the database.")
            return

        selected_track = st.selectbox(
            "Track", track_list, key="acc_track_select",
            format_func=lambda t: f"{t} ({track_counts[t]} races)")

        all_races = _query_db_completed_races(query_series, track_name=selected_track)
        tt = TRACK_TYPE_MAP.get(selected_track, "intermediate")
        parent_tt = TRACK_TYPE_PARENT.get(tt, tt)
        context_label = selected_track
        current_defaults = TRACK_TYPE_WEIGHT_DEFAULTS.get(parent_tt,
                            TRACK_TYPE_WEIGHT_DEFAULTS.get("intermediate", {}))

    if not all_races:
        st.info("No completed races match the selected filters.")
        return

    # ── Race count slider ──
    max_races = min(len(all_races), 30)
    r_cols = st.columns([2, 4])
    with r_cols[0]:
        n_races = st.slider("Races to test", 3, max_races,
                             min(max_races, 15), key="acc_n_races")

    test_races = all_races[:n_races]  # already sorted by date desc

    # Show which races will be tested
    series_breakdown = {}
    track_names = []
    for _, r in test_races:
        sid = r.get("series_id", primary_sid)
        sname = SERIES_LABELS.get(sid, str(sid))
        series_breakdown[sname] = series_breakdown.get(sname, 0) + 1
        track_names.append(r.get("track_name", ""))

    breakdown_str = ", ".join(f"{v} {k}" for k, v in sorted(series_breakdown.items()))
    unique_tracks = sorted(set(track_names))
    if len(unique_tracks) <= 8:
        st.caption(f"**{n_races} races** ({breakdown_str}): {', '.join(unique_tracks)}")
    else:
        st.caption(f"**{n_races} races** ({breakdown_str}) across {len(unique_tracks)} tracks")

    # Show current defaults for this context
    st.caption(
        f"Current defaults ({context_label}): "
        f"Odds {current_defaults.get('odds', 25)}% | "
        f"Track {current_defaults.get('track', 20)}% | "
        f"TType {current_defaults.get('ttype', 15)}% | "
        f"Prac {current_defaults.get('prac', 10)}% | "
        f"Team {current_defaults.get('team', 15)}% | "
        f"Qual {current_defaults.get('qual', 15)}%"
    )

    btn_cols = st.columns([1, 1, 4])
    with btn_cols[0]:
        run_clicked = st.button("Run Optimization", type="primary",
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
        _run_backtest(test_races, primary_sid, selected_year,
                      context_label, include_dnf)
    elif "acc_opt_results" in st.session_state:
        _display_backtest_results(
            st.session_state["acc_opt_results"],
            st.session_state.get("acc_opt_series", context_label),
        )


def _run_backtest(test_races, series_id, selected_year, context_label,
                  include_dnf=True, grid_step=5):
    """Run full-signal backtest across weight combinations.

    Args:
        test_races: list of (index, race_dict) — each race_dict has series_id
        series_id: primary series (used as fallback)
        context_label: display label for results (e.g. "Short Track" or "Bristol")
        include_dnf: include DNF risk adjustment
        grid_step: weight grid step size
    """
    from src.utils import fuzzy_match_name

    # ── Per-signal weight constraints ──────────────────────────────────────
    SIGNAL_RANGES = {
        "odds":       (10, 40),
        "track":      (10, 35),
        "ttype":      (5, 25),
        "practice":   (0, 25),
        "team":       (5, 20),
        "qual":       (5, 25),
    }

    weight_combos = []
    for odds in range(SIGNAL_RANGES["odds"][0], SIGNAL_RANGES["odds"][1] + 1, grid_step):
        for track in range(SIGNAL_RANGES["track"][0], SIGNAL_RANGES["track"][1] + 1, grid_step):
            for ttype in range(SIGNAL_RANGES["ttype"][0], SIGNAL_RANGES["ttype"][1] + 1, grid_step):
                for team in range(SIGNAL_RANGES["team"][0], SIGNAL_RANGES["team"][1] + 1, grid_step):
                    for qual in range(SIGNAL_RANGES["qual"][0], SIGNAL_RANGES["qual"][1] + 1, grid_step):
                        prac = 100 - odds - track - ttype - team - qual
                        if SIGNAL_RANGES["practice"][0] <= prac <= SIGNAL_RANGES["practice"][1]:
                            weight_combos.append({
                                "odds": odds, "track": track,
                                "track_type": ttype, "practice": prac,
                                "team": team, "qual": qual,
                            })

    range_labels = {"odds": "Odds", "track": "Track", "ttype": "Track Type",
                    "practice": "Practice", "team": "Team", "qual": "Qualifying"}
    ranges_str = " | ".join(f"{range_labels.get(k,k)}: {v[0]}-{v[1]}%" for k, v in SIGNAL_RANGES.items())
    st.caption(f"Testing **{len(weight_combos)}** weight combinations across "
               f"{len(test_races)} races ({context_label})")
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

        # Each race may be from a different series
        race_sid = race.get("series_id", series_id)

        # Load actual results (still needs API for race results)
        feed = fetch_weekend_feed(race_sid, race_id, yr)
        laps = fetch_lap_times(race_sid, race_id, yr)
        if not feed:
            continue

        results = extract_race_results(feed)
        if results.empty:
            continue

        fl = compute_fastest_laps(laps) if laps else {}
        _fl_norm = build_norm_lookup(fl)
        results["Fastest Laps"] = results["Driver"].map(
            lambda d: fuzzy_get(d, fl, _fl_norm) or 0)
        results["DK Pts"] = results.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                     r["Laps Led"], r["Fastest Laps"]), axis=1)

        drivers = results["Driver"].unique().tolist()
        field_size = len(drivers)
        mid_field = field_size * 0.5

        race_date = race.get("race_date", "")[:10] if race.get("race_date") else None

        # ── Use same DB queries as projections tab ──
        from tabs.tab_projections import (
            _query_db_track_history, _query_db_track_type_history,
            _resolve_db_race_id, _get_track_dominator_calibration,
        )
        from src.data import (
            query_team_stats as _qts, query_manufacturer_stats as _qms,
            compute_team_adjusted_track_history as _ctath,
            query_driver_dk_points_at_track as _qdkpt, round_odds,
        )
        from src.config import CROSS_SERIES_HIERARCHY
        import math

        db_race_id = _resolve_db_race_id(race_id, race_sid) if race_id else None
        cross_ids = CROSS_SERIES_HIERARCHY.get(race_sid, [])

        # Track history
        th_data = {}
        th_result = _query_db_track_history(
            track_name, race_sid, exclude_race_id=db_race_id,
            before_date=race_date,
            cross_series_ids=cross_ids if cross_ids else None,
        )
        if cross_ids:
            th_df, cross_th_df = th_result
        else:
            th_df = th_result
            cross_th_df = pd.DataFrame()

        dk_history = _qdkpt(track_name, race_sid, min_season=2022, before_date=race_date)

        cross_th_lookup = {}
        if not cross_th_df.empty:
            for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races",
                         "Avg Rating", "Wins", "Top 5", "Top 10", "DNF"]:
                if col in cross_th_df.columns:
                    cross_th_df[col] = pd.to_numeric(cross_th_df[col], errors="coerce")
            cross_idx = cross_th_df.drop_duplicates("Driver").set_index("Driver")
            for d in drivers:
                matched = d if d in cross_idx.index else fuzzy_match_name(d, cross_idx.index.tolist())
                if matched and matched in cross_idx.index:
                    row = cross_idx.loc[matched]
                    races = row.get("Races", 1) if pd.notna(row.get("Races")) and row.get("Races") > 0 else 1
                    arp = row.get("Avg Run Pos") if pd.notna(row.get("Avg Run Pos", None)) else None
                    cross_th_lookup[d] = {
                        "avg_finish": row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20,
                        "avg_running_pos": arp, "races": races,
                    }

        if not th_df.empty:
            for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races",
                         "Avg Rating", "Wins", "Top 5", "Top 10", "DNF"]:
                if col in th_df.columns:
                    th_df[col] = pd.to_numeric(th_df[col], errors="coerce")
            th_idx = th_df.drop_duplicates("Driver").set_index("Driver")
            th_names = th_idx.index.tolist()
            for d in drivers:
                matched = d if d in th_idx.index else fuzzy_match_name(d, th_names)
                if matched and matched in th_idx.index:
                    row = th_idx.loc[matched]
                    races = row.get("Races", 1) if pd.notna(row.get("Races")) and row.get("Races") > 0 else 1
                    laps_led = row.get("Laps Led", 0) if pd.notna(row.get("Laps Led")) else 0
                    fastest_laps = row.get("Fastest Laps", 0) if pd.notna(row.get("Fastest Laps")) else 0
                    dk_hist_entry = dk_history.get(matched) or dk_history.get(d) or {}
                    arp = row.get("Avg Run Pos") if pd.notna(row.get("Avg Run Pos", None)) else None
                    af = row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20
                    cross = cross_th_lookup.get(d)
                    if cross and cross["races"] >= 2:
                        c_af = cross["avg_finish"]
                        c_arp = cross["avg_running_pos"]
                        if c_af < af:
                            af = af * 0.70 + c_af * 0.30
                            if arp is not None and c_arp is not None and c_arp < arp:
                                arp = arp * 0.70 + c_arp * 0.30
                            elif c_arp is not None and c_arp < (arp or 99):
                                arp = c_arp
                    th_data[d] = {
                        "avg_finish": af,
                        "avg_start": row.get("Avg Start", 20) if pd.notna(row.get("Avg Start")) else 20,
                        "avg_running_pos": arp,
                        "laps_led": laps_led, "fastest_laps": fastest_laps,
                        "laps_led_per_race": laps_led / races,
                        "fastest_laps_per_race": fastest_laps / races,
                        "races": races, "avg_dk": dk_hist_entry.get("avg_dk"),
                        "wins": row.get("Wins", 0) if pd.notna(row.get("Wins")) else 0,
                        "top5": row.get("Top 5", 0) if pd.notna(row.get("Top 5")) else 0,
                        "dnf": row.get("DNF", 0) if pd.notna(row.get("DNF")) else 0,
                    }

        SERIES_FIELD_SIZE = {1: 36, 2: 38, 3: 36}
        for d in drivers:
            if d not in th_data and d in cross_th_lookup:
                cross = cross_th_lookup[d]
                if cross["races"] >= 2:
                    source_field = max(SERIES_FIELD_SIZE.get(cross_ids[0], 36) if cross_ids else 36, 1)
                    c_af = cross["avg_finish"]
                    c_arp = cross["avg_running_pos"]
                    pct = c_af / source_field
                    th_data[d] = {
                        "avg_finish": pct * field_size, "avg_start": 20,
                        "avg_running_pos": (c_arp / source_field) * field_size if c_arp else None,
                        "laps_led": 0, "fastest_laps": 0,
                        "laps_led_per_race": 0, "fastest_laps_per_race": 0,
                        "races": cross["races"], "avg_dk": None,
                        "wins": 0, "top5": 0, "dnf": 0,
                        "_cross_series_only": True,
                    }

        # Track type
        tt_data = {}
        tt_result = _query_db_track_type_history(
            track_type, race_sid, exclude_track=track_name,
            exclude_race_id=db_race_id, before_date=race_date,
            cross_series_ids=cross_ids if cross_ids else None,
        )
        if cross_ids:
            tt_raw, tt_cross_raw = tt_result
        else:
            tt_raw = tt_result
            tt_cross_raw = {}
        if tt_cross_raw:
            source_field = max(SERIES_FIELD_SIZE.get(cross_ids[0], 36) if cross_ids else 36, 1)
            for name, cdata in tt_cross_raw.items():
                if name in tt_raw and cdata.get("races", 0) >= 2:
                    cur = tt_raw[name]
                    if cdata["avg_finish"] < cur["avg_finish"]:
                        cur["avg_finish"] = cur["avg_finish"] * 0.70 + cdata["avg_finish"] * 0.30
                        c_arp = cdata.get("avg_running_pos")
                        if cur.get("avg_running_pos") and c_arp and c_arp < cur["avg_running_pos"]:
                            cur["avg_running_pos"] = cur["avg_running_pos"] * 0.70 + c_arp * 0.30
                elif name not in tt_raw and cdata.get("races", 0) >= 3:
                    scaled = dict(cdata)
                    scaled["avg_finish"] = (cdata["avg_finish"] / source_field) * field_size
                    if cdata.get("avg_running_pos"):
                        scaled["avg_running_pos"] = (cdata["avg_running_pos"] / source_field) * field_size
                    scaled["_cross_series_only"] = True
                    tt_raw[name] = scaled
        if tt_raw:
            tt_names = list(tt_raw.keys())
            for d in drivers:
                matched = d if d in tt_raw else fuzzy_match_name(d, tt_names)
                if matched and matched in tt_raw:
                    tt_data[d] = tt_raw[matched]

        # Team signal + team adjustment
        team_signal = {}
        team_adj_data = {}
        driver_team_map = {}
        try:
            conn = sqlite3.connect(PROJ_DB)
            team_rows = conn.execute('''
                SELECT d.full_name, rr.team
                FROM race_results rr
                JOIN drivers d ON d.id = rr.driver_id
                JOIN races r ON r.id = rr.race_id
                WHERE r.series_id = ? AND rr.team IS NOT NULL AND rr.team != ''
                  AND r.race_date < ?
                GROUP BY d.id
                HAVING r.race_date = MAX(r.race_date)
            ''', (race_sid, race_date)).fetchall()
            conn.close()
            for name, team in team_rows:
                driver_team_map[name] = team
        except Exception:
            pass

        team_stats = _qts(race_sid, track_type=track_type, min_season=2022, before_date=race_date)
        if team_stats:
            ts_names = list(team_stats.keys())
            for d in drivers:
                team_name = driver_team_map.get(d)
                if not team_name:
                    matched_d = fuzzy_match_name(d, list(driver_team_map.keys()))
                    team_name = driver_team_map.get(matched_d) if matched_d else None
                if not team_name:
                    continue
                matched_team = team_name if team_name in team_stats else fuzzy_match_name(team_name, ts_names)
                if matched_team and matched_team in team_stats:
                    ts = team_stats[matched_team]
                    ts_arp = ts.get("avg_arp")
                    ts_af = ts["avg_finish"]
                    team_finish = ts_arp * 0.65 + ts_af * 0.35 if ts_arp is not None else ts_af
                    trust = min(1.0, ts["races"] / 10)
                    team_signal[d] = team_finish * trust + mid_field * (1 - trust)
        if driver_team_map:
            dtm_for_adj = {}
            for d in drivers:
                tm = driver_team_map.get(d)
                if not tm:
                    matched_d = fuzzy_match_name(d, list(driver_team_map.keys()))
                    tm = driver_team_map.get(matched_d) if matched_d else None
                if tm:
                    dtm_for_adj[d] = tm
            if dtm_for_adj:
                team_adj_data = _ctath(track_name, race_sid, dtm_for_adj, before_date=race_date)

        # Manufacturer adjustment
        mfr_adjustment = {}
        mfr_stats = _qms(race_sid, track_type=track_type, min_season=2022, before_date=race_date)
        if mfr_stats:
            total_races = sum(m["races"] for m in mfr_stats.values())
            field_avg_finish = (sum(m["avg_finish"] * m["races"] for m in mfr_stats.values()) / total_races
                               if total_races > 0 else mid_field)
            driver_mfr_map = {}
            try:
                conn = sqlite3.connect(PROJ_DB)
                mfr_rows = conn.execute('''
                    SELECT d.full_name, rr.manufacturer
                    FROM race_results rr
                    JOIN drivers d ON d.id = rr.driver_id
                    JOIN races r ON r.id = rr.race_id
                    WHERE r.series_id = ? AND rr.manufacturer IS NOT NULL AND rr.manufacturer != ''
                      AND r.race_date < ?
                    GROUP BY d.id
                    HAVING r.race_date = MAX(r.race_date)
                ''', (race_sid, race_date)).fetchall()
                conn.close()
                for name, mfr in mfr_rows:
                    driver_mfr_map[name] = mfr
            except Exception:
                pass
            mfr_names = list(mfr_stats.keys())
            for d in drivers:
                mfr = driver_mfr_map.get(d)
                if not mfr:
                    matched_d = fuzzy_match_name(d, list(driver_mfr_map.keys()))
                    mfr = driver_mfr_map.get(matched_d) if matched_d else None
                if not mfr:
                    continue
                matched_mfr = mfr if mfr in mfr_stats else fuzzy_match_name(mfr, mfr_names)
                if matched_mfr and matched_mfr in mfr_stats:
                    delta = mfr_stats[matched_mfr]["avg_finish"] - field_avg_finish
                    mfr_adjustment[d] = max(-1.5, min(1.5, delta * 0.5))

        # DNF data
        dnf_data = query_driver_career_dnf(race_sid, before_date=race_date) if include_dnf else {}

        # Qualifying from actual start positions
        start_positions = {}
        for _, row in results.iterrows():
            if pd.notna(row.get("Start")):
                start_positions[row["Driver"]] = int(row["Start"])

        # Odds
        saved_odds = load_race_odds(race_id, race_sid)
        odds_finish = {}
        driver_odds_display = {}
        if saved_odds:
            clean_odds = {k: v for k, v in saved_odds.items()
                          if v is not None and str(v).strip() not in ("", "None", "null", "N/A")}
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
                log_probs = {name: math.log(prob) for name, prob in ranked}
                max_lp = max(log_probs.values())
                min_lp = min(log_probs.values())
                lp_range = max_lp - min_lp
                for name, prob in ranked:
                    matched = fuzzy_match_name(name, drivers)
                    if matched:
                        if lp_range > 0:
                            t = 1 - (log_probs[name] - min_lp) / lp_range
                            odds_finish[matched] = 1 + (field_size - 1) * t
                        else:
                            odds_finish[matched] = mid_field
            for name, odds_str in clean_odds.items():
                try:
                    oval = int(str(odds_str).replace("+", ""))
                    if oval > 0:
                        impl = 100 / (oval + 100) * 100
                    elif oval < 0:
                        impl = abs(oval) / (abs(oval) + 100) * 100
                    else:
                        continue
                    matched = fuzzy_match_name(name, drivers)
                    if matched:
                        rounded = round_odds(oval)
                        driver_odds_display[matched] = {"odds_str": rounded, "impl_pct": round(impl, 1)}
                except (ValueError, TypeError):
                    continue

        # Load practice data from NASCAR API
        practice_data = {}
        try:
            from src.data import fetch_lap_averages
            lap_avg_df = fetch_lap_averages(race_sid, race_id, yr)
            if not lap_avg_df.empty and "Overall Rank" in lap_avg_df.columns:
                for _, prow in lap_avg_df.iterrows():
                    pdriver = prow.get("Driver")
                    prank = prow.get("Overall Rank")
                    if pdriver and prank and not pd.isna(prank):
                        matched = fuzzy_match_name(pdriver, drivers)
                        if matched:
                            practice_data[matched] = int(prank)
        except Exception:
            pass

        # Track which signals are available for this race
        has_signals = {
            "track": bool(th_data),
            "track_type": bool(tt_data),
            "qual": bool(start_positions),
            "practice": bool(practice_data),
            "odds": bool(odds_finish),
            "team": bool(team_signal),
        }

        race_laps = race.get("scheduled_laps", 0)
        try:
            race_laps = int(race_laps) if race_laps else 0
        except (ValueError, TypeError):
            race_laps = 0

        calibration = _get_track_dominator_calibration(track_name, track_type, race_sid)

        # Pre-index actual DK pts per driver for fast lookup in combo loop
        actual_dk = {}
        for _, row in results.iterrows():
            actual_dk[row["Driver"]] = row["DK Pts"]

        race_data.append({
            "race": race,
            "results": results,
            "drivers": drivers,
            "field_size": field_size,
            "th_data": th_data,
            "tt_data": tt_data,
            "start_positions": start_positions,
            "odds_finish": odds_finish,
            "odds_display": driver_odds_display,
            "team_signal": team_signal,
            "team_adj_data": team_adj_data,
            "mfr_adjustment": mfr_adjustment,
            "cross_th_lookup": cross_th_lookup,
            "practice_data": practice_data,
            "has_signals": has_signals,
            "race_laps": race_laps,
            "track_type": track_type,
            "track_name": race.get("track_name", ""),
            "series_id": race_sid,
            "actual_dk": actual_dk,
            "dnf_data": dnf_data,
            "calibration": calibration,
        })

    if not race_data:
        st.warning("Could not load any race results for backtesting.")
        return

    # Show signal availability summary
    signal_summary = {"track": 0, "qual": 0, "practice": 0, "odds": 0, "team": 0}
    for rd in race_data:
        for sig, available in rd["has_signals"].items():
            if available:
                signal_summary[sig] = signal_summary.get(sig, 0) + 1
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
            for sig in ["track", "track_type", "qual", "practice", "odds", "team"]:
                effective[sig] = nominal_wn.get(sig, 0) if has.get(sig, False) else 0
            eff_total = sum(effective.values())
            if eff_total <= 0:
                continue
            wn = {k: v / eff_total for k, v in effective.items()}

            # Run shared projection engine
            from src.projections import compute_projections
            proj_rows, _ = compute_projections(
                drivers=drivers, field_size=field_size, wn=wn,
                th_data=rd["th_data"], tt_data=rd["tt_data"],
                qual_pos=rd["start_positions"],
                practice_data={},
                odds_finish=rd["odds_finish"],
                odds_display=rd.get("odds_display", {}),
                team_signal=rd.get("team_signal", {}),
                mfr_adjustment=rd.get("mfr_adjustment", {}),
                team_adj_data=rd.get("team_adj_data", {}),
                dnf_data=rd.get("dnf_data", {}),
                race_laps=rd.get("race_laps", 0),
                track_name=rd.get("track_name", ""),
                track_type=rd.get("track_type", "intermediate"),
                series_id=rd.get("series_id"),
                calibration=rd.get("calibration", {}),
                cross_th_lookup=rd.get("cross_th_lookup", {}),
            )
            proj_dk = {r["driver"]: r["proj_dk"] for r in proj_rows}

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
                "Track Type": combo.get("track_type", 0),
                "Practice": combo.get("practice", 0),
                "Team": combo.get("team", 0),
                "Qualifying": combo.get("qual", 0),
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
    st.session_state["acc_opt_series"] = context_label
    st.session_state["acc_opt_race_data"] = race_data

    _display_backtest_results(results_df, context_label)


def _display_backtest_results(results_df, context_label):
    """Display backtest results (separated so export doesn't re-run)."""
    # Show top 15
    st.markdown(f"**Top 15 Weight Combinations — {context_label}**")
    top = results_df.head(15).copy()
    top["MAE"] = top["MAE"].round(1)
    top["Rank Corr"] = top["Rank Corr"].round(3)
    if "Score" in top.columns:
        top = top.drop(columns=["Score"])
    top.index = range(1, len(top) + 1)
    top.index.name = "Rank"

    st.dataframe(top, width="stretch", hide_index=False)

    best = results_df.iloc[0]
    st.success(
        f"**Best weights ({context_label}):** Odds **{int(best['Odds'])}%** | "
        f"Track **{int(best['Track'])}%** | "
        f"Track Type **{int(best.get('Track Type', 0))}%** | "
        f"Practice **{int(best.get('Practice', 0))}%** | "
        f"Team **{int(best.get('Team', 0))}%** | "
        f"Qual **{int(best.get('Qualifying', 0))}%** | "
        f"MAE: {best['MAE']:.1f} | Rank Corr: {best['Rank Corr']:.3f}"
    )

    # "Apply to Projections" button — sets session state weights
    apply_cols = st.columns([1.5, 4.5])
    with apply_cols[0]:
        if st.button("Apply Best to Projections", key="acc_apply_best",
                      type="primary",
                      help="Set the projections tab sliders to these optimal weights"):
            st.session_state["pw_odds"] = int(best["Odds"])
            st.session_state["pw_track"] = int(best["Track"])
            st.session_state["pw_ttype"] = int(best.get("Track Type", 0))
            st.session_state["pw_prac"] = int(best.get("Practice", 0))
            st.session_state["pw_team"] = int(best.get("Team", 0))
            st.session_state["pw_qual"] = int(best.get("Qualifying", 0))
            st.success("Applied! Switch to the Projections tab and re-run to use these weights.")

    # Show current vs optimal comparison — read from projections tab session state
    int_defaults = TRACK_TYPE_WEIGHT_DEFAULTS.get("intermediate", {})
    current_weights = {
        "Odds": st.session_state.get("pw_odds", int_defaults.get("odds", 25)),
        "Track": st.session_state.get("pw_track", int_defaults.get("track", 20)),
        "Track Type": st.session_state.get("pw_ttype", int_defaults.get("ttype", 15)),
        "Practice": st.session_state.get("pw_prac", int_defaults.get("prac", 10)),
        "Team": st.session_state.get("pw_team", int_defaults.get("team", 15)),
        "Qualifying": st.session_state.get("pw_qual", int_defaults.get("qual", 15)),
    }

    match_mask = (
        (results_df["Odds"] == current_weights["Odds"]) &
        (results_df["Track"] == current_weights["Track"]) &
        (results_df["Track Type"] == current_weights.get("Track Type", 0)) &
        (results_df["Practice"] == current_weights.get("Practice", 0))
    )
    if "Team" in results_df.columns:
        match_mask = match_mask & (results_df["Team"] == current_weights.get("Team", 0))
    if "Qualifying" in results_df.columns:
        match_mask = match_mask & (results_df["Qualifying"] == current_weights.get("Qualifying", 0))
    current_match = results_df[match_mask]

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

    # Export
    export_df = results_df.drop(columns=["Score"], errors="ignore").copy()
    export_df.insert(0, "Context", context_label)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export All Results CSV", csv,
                       f"weight_optimization_{context_label}.csv", "text/csv",
                       key="acc_opt_export")

    # ── Drill-down: select a weight combo to see driver-level detail ──
    st.divider()
    st.markdown("**Drill Into Weight Combo**")
    st.caption("Select a weight combination to see projected vs actual driver details")

    race_data = st.session_state.get("acc_opt_race_data", [])

    if race_data and len(results_df) > 0:
        # Build selectable labels from top 15
        top15 = results_df.head(15)
        combo_labels = []
        for idx, row in top15.iterrows():
            lbl = (f"#{len(combo_labels) + 1}: "
                   f"O{int(row['Odds'])} T{int(row['Track'])} "
                   f"TT{int(row.get('Track Type', 0))} P{int(row['Practice'])} "
                   f"Tm{int(row.get('Team', 0))} Q{int(row.get('Qualifying', 0))} — "
                   f"MAE {row['MAE']:.1f}, r={row['Rank Corr']:.3f}")
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
                "track_type": int(selected_row.get("Track Type", 0)),
                "practice": int(selected_row["Practice"]),
                "qual": 0,
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
                for sig in ["track", "track_type", "qual", "practice", "odds", "team"]:
                    effective[sig] = nominal_wn.get(sig, 0) if has.get(sig, False) else 0
                eff_total = sum(effective.values())
                if eff_total <= 0:
                    continue
                wn = {k: v / eff_total for k, v in effective.items()}

                # Run shared projection engine with detailed output
                from src.projections import compute_projections
                proj_rows_det, proj_details = compute_projections(
                    drivers=drivers, field_size=field_size, wn=wn,
                    th_data=rd["th_data"], tt_data=rd["tt_data"],
                    qual_pos=rd["start_positions"],
                    practice_data=rd.get("practice_data", {}),
                    odds_finish=rd["odds_finish"],
                    odds_display=rd.get("odds_display", {}),
                    team_signal=rd.get("team_signal", {}),
                    mfr_adjustment=rd.get("mfr_adjustment", {}),
                    team_adj_data=rd.get("team_adj_data", {}),
                    dnf_data=rd.get("dnf_data", {}),
                    race_laps=rd.get("race_laps", 0),
                    track_name=rd.get("track_name", ""),
                    track_type=rd.get("track_type", "intermediate"),
                    series_id=rd.get("series_id"),
                    calibration=rd.get("calibration", {}),
                    cross_th_lookup=rd.get("cross_th_lookup", {}),
                )
                proj_dk_totals = {r["driver"]: r["proj_dk"] for r in proj_rows_det}

                # Build output rows from detailed projection data
                for d in drivers:
                    det = proj_details.get(d, {})
                    pf = det.get("proj_finish", 20)
                    start = det.get("start", round(pf))
                    p_ll = det.get("laps_led", 0)
                    p_fl = det.get("fast_laps", 0)
                    proj_total = proj_dk_totals.get(d, 0)

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
                             width="stretch", hide_index=False, height=500)

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
                    from src.charts import DARK_LAYOUT, apply_dark_theme

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
                        mode="lines", line=dict(color="#0ea5e9", width=2),
                        name=f"Trend (r={corr:.3f})",
                    ))

                    fig.update_layout(
                        **DARK_LAYOUT, height=450,
                        title=f"Projected vs Actual DK Points — {selected_lbl.split(':')[0]}",
                        xaxis_title="Actual DK Points",
                        yaxis_title="Projected DK Points",
                    )
                    apply_dark_theme(fig)
                    st.plotly_chart(fig, width="stretch", key="acc_drill_scatter")

                csv_detail = detail_df.to_csv(index=True).encode("utf-8")
                st.download_button("Export Detail CSV", csv_detail,
                                   f"weight_detail_{context_label}.csv", "text/csv",
                                   key="acc_drill_export")

    # Clear saved projections button (separate from export)
    st.divider()
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
