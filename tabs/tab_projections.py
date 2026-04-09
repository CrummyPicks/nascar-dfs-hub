"""Tab 5: Projections — DFS-Optimized Projection Engine.  # v3 — track type removed

Projects actual DraftKings points by estimating each scoring component:
  - Finish position points (from DK_FINISH_POINTS table)
  - Place differential points (start - finish) * 1.0
  - Laps led points (laps_led * 0.25)
  - Fastest laps points (fastest_laps * 0.45)

Uses weighted signals: track history, track type, qualifying, practice, odds.
Incorporates dominator potential (who can lead laps and earn fastest laps).
"""

import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import os

from src.config import (
    DEFAULT_PROJECTION_WEIGHTS, DB_PATH, TRACK_TYPE_MAP,
    TRACK_TYPE_PARENT, DK_FINISH_POINTS,
)
from src.data import (
    query_projections, scrape_track_history, query_driver_dk_points_at_track,
    query_driver_career_dnf,
)
# projection_bar no longer used — replaced with inline stacked bar
from src.utils import safe_fillna, format_display_df, calc_dk_points, fuzzy_match_name, fuzzy_merge

PROJ_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_db_race_id(series_id, race_name, track_name):
    """Try to find a matching race_id in the database."""
    if not os.path.exists(PROJ_DB):
        return None
    try:
        conn = sqlite3.connect(PROJ_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id FROM races WHERE series_id = ? AND race_name = ? ORDER BY season DESC LIMIT 1",
            (series_id, race_name)
        ).fetchone()
        if row:
            conn.close()
            return row["id"]
        row = conn.execute("""
            SELECT r.id FROM races r JOIN tracks t ON t.id = r.track_id
            WHERE r.series_id = ? AND t.name LIKE ?
            ORDER BY r.season DESC, r.race_num DESC LIMIT 1
        """, (series_id, f"%{track_name}%")).fetchone()
        conn.close()
        return row["id"] if row else None
    except Exception:
        return None


def _run_projection_engine(race_id, platform, weights):
    """Run the real 6-component projection engine."""
    try:
        from projections import get_conn, project_race
        conn = get_conn()
        projections = project_race(conn, race_id, platform, weights)
        conn.close()
        return projections
    except Exception as e:
        st.warning(f"Projection engine error: {e}")
        return []


def _get_race_laps(feed):
    """Extract total laps for the race from weekend feed."""
    if not feed:
        return 0
    races = feed.get("weekend_race", [])
    if races:
        return races[0].get("number_of_laps") or races[0].get("laps") or 0
    return 0


def _expected_finish_from_avg(avg_finish, field_size=38):
    """Convert a projected finish to expected DK finish points.

    Rounds to nearest integer position since DK awards discrete finish points.
    """
    ef = max(1, min(40, round(avg_finish)))
    return DK_FINISH_POINTS.get(ef, 0)


def _dominator_share(laps_led_history, total_laps_history, races):
    """Estimate what fraction of laps a driver leads per race."""
    if not races or races == 0 or not total_laps_history:
        return 0.0
    return (laps_led_history / races) / max(total_laps_history / races, 1)


# Fallback track-type concentration exponents
# Higher = more concentrated (top drivers get more of the pie)
# Lower = more distributed (laps spread across many drivers)
TRACK_TYPE_CONCENTRATION = {
    "superspeedway": 0.6,   # Very distributed — pack racing
    "road": 1.0,            # Moderate
    "dirt": 1.2,            # Moderate-high
    "intermediate": 1.5,    # Concentrated
    "intermediate_worn": 1.6,  # Darlington/Homestead — high wear, slightly more concentrated
    "short": 2.0,           # Very concentrated — single driver can dominate
    "short_concrete": 2.2,  # Bristol/Dover concrete — even more concentrated
}

# Fallback dominator ceilings by track type (max laps led, max fastest laps)
TRACK_TYPE_DOM_DEFAULTS = {
    "superspeedway":        {"max_ll": 70,  "max_fl": 40},
    "road":                 {"max_ll": 60,  "max_fl": 30},
    "dirt":                 {"max_ll": 100, "max_fl": 50},
    "intermediate":         {"max_ll": 200, "max_fl": 80},
    "intermediate_worn":    {"max_ll": 200, "max_fl": 80},
    "short":                {"max_ll": 350, "max_fl": 120},
    "short_concrete":       {"max_ll": 400, "max_fl": 130},
}


def _get_track_dominator_calibration(track_name: str, track_type: str) -> dict:
    """Pull historical domination stats from DB for this track.

    Returns dict with:
        avg_top_leader: average laps led by the race leader per race
        max_laps_led: highest single-race laps led at this track
        max_fastest_laps: highest single-race fastest laps at this track
        concentration: exponent for score distribution
    """
    type_defaults = TRACK_TYPE_DOM_DEFAULTS.get(track_type, {"max_ll": 150, "max_fl": 60})
    defaults = {
        "avg_top_leader": 80,
        "max_laps_led": type_defaults["max_ll"],
        "max_fastest_laps": type_defaults["max_fl"],
        "concentration": TRACK_TYPE_CONCENTRATION.get(track_type, 1.5),
    }
    if not os.path.exists(PROJ_DB):
        return defaults

    try:
        conn = sqlite3.connect(PROJ_DB)
        # Get max laps led per race at this track
        ll_rows = conn.execute('''
            SELECT MAX(rr.laps_led) as top_led
            FROM race_results rr
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE t.name LIKE ?
            GROUP BY r.id
        ''', (f"%{track_name}%",)).fetchall()

        # Get max fastest laps per race at this track
        fl_rows = conn.execute('''
            SELECT MAX(rr.fastest_laps) as top_fl
            FROM race_results rr
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE t.name LIKE ?
            GROUP BY r.id
        ''', (f"%{track_name}%",)).fetchall()
        conn.close()

        result = dict(defaults)  # start with defaults

        if ll_rows and len(ll_rows) >= 1:
            top_leaders = [r[0] for r in ll_rows if r[0] and r[0] > 0]
            if top_leaders:
                result["avg_top_leader"] = np.mean(top_leaders)
                result["max_laps_led"] = max(top_leaders)

        if fl_rows and len(fl_rows) >= 1:
            top_fl = [r[0] for r in fl_rows if r[0] and r[0] > 0]
            if top_fl:
                result["max_fastest_laps"] = max(top_fl)

        return result

    except Exception:
        pass

    return defaults


def _allocate_laps_led(driver_scores: dict, race_laps: int, track_name: str,
                        track_type: str) -> dict:
    """Allocate projected laps led across the field using DB-calibrated data.

    Uses historical data to determine how concentrated laps led should be.
    At short tracks, the top driver can lead 200+ laps (50%+).
    At superspeedways, the top leader averages only ~35 laps.

    Applies a realistic cutoff: historically only ~15-25% of the field leads
    any laps (except superspeedways ~60%).  Drivers below the cutoff get 0.
    """
    if not driver_scores or race_laps <= 0:
        return {}

    calibration = _get_track_dominator_calibration(track_name, track_type)
    concentration = calibration["concentration"]

    # What fraction of the field typically leads laps at this track type
    LEADER_FRAC = {
        "superspeedway": 0.60,
        "road": 0.22,
        "dirt": 0.20,
        "intermediate": 0.22,
        "intermediate_worn": 0.20,
        "short": 0.18,
        "short_concrete": 0.16,
    }
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    frac = LEADER_FRAC.get(track_type, LEADER_FRAC.get(parent, 0.22))
    n_leaders = max(3, int(len(driver_scores) * frac))

    # Rank drivers by raw score, only top N get any laps led
    sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
    top_drivers = dict(sorted_drivers[:n_leaders])

    # Apply concentration exponent — higher exponent = more to the top scorer
    scores = {}
    for d, s in top_drivers.items():
        scores[d] = max(0.01, s) ** concentration

    total = sum(scores.values())
    if total <= 0:
        return {}

    return {d: (s / total) * race_laps for d, s in scores.items()}


def _allocate_fastest_laps(driver_fl_scores: dict, race_laps: int,
                            track_type: str) -> dict:
    """Allocate projected fastest laps across the field (zero-sum).

    Uses a lower concentration than laps led — fastest laps are more
    distributed since even mid-pack drivers can post a fastest lap.

    Applies a realistic cutoff: historically ~40-60% of the field earns at
    least one fastest lap (superspeedways ~90%, championship races ~40%).
    Drivers below the cutoff get 0.
    """
    if not driver_fl_scores or race_laps <= 0:
        return {}

    # What fraction of the field typically gets fastest laps
    FL_FRAC = {
        "superspeedway": 0.85,
        "road": 0.55,
        "dirt": 0.50,
        "intermediate": 0.65,
        "intermediate_worn": 0.60,
        "short": 0.55,
        "short_concrete": 0.50,
    }
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    frac = FL_FRAC.get(track_type, FL_FRAC.get(parent, 0.55))
    n_with_fl = max(5, int(len(driver_fl_scores) * frac))

    # Rank drivers by raw FL score, only top N get any fastest laps
    sorted_drivers = sorted(driver_fl_scores.items(), key=lambda x: x[1], reverse=True)
    top_drivers = dict(sorted_drivers[:n_with_fl])

    # Fastest laps are less concentrated than laps led
    concentration = max(0.5, TRACK_TYPE_CONCENTRATION.get(track_type, 1.5) * 0.7)

    scores = {}
    for d, s in top_drivers.items():
        scores[d] = max(0.01, s) ** concentration

    total = sum(scores.values())
    if total <= 0:
        return {}

    return {d: (s / total) * race_laps for d, s in scores.items()}


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           is_prerace, race_name, race_id, track_name, series_id, dk_df,
           odds_data=None, scheduled_laps=0, race_date="", season=2026):
    """Render the Projections tab."""
    st.markdown(f"### Projections — {race_name}")

    if not is_prerace:
        st.caption("Race completed — projections shown for review")

    if entry_list_df.empty and dk_df.empty:
        st.warning("Entry list not available for this race.")
        return

    # Get race laps for dominator calculations
    race_laps = scheduled_laps or 0
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")

    # Track-type-specific default weights — different tracks have different
    # predictive signals. Superspeedways are chaotic (odds matter most),
    # short tracks reward specialists (track history matters most),
    # road courses depend on setup (practice matters most).
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    TRACK_TYPE_DEFAULTS = {
        "superspeedway": {"odds": 45, "track": 20, "ttype": 25, "prac": 10},
        "short":         {"odds": 30, "track": 35, "ttype": 15, "prac": 20},
        "road":          {"odds": 25, "track": 25, "ttype": 20, "prac": 30},
        "intermediate":  {"odds": 35, "track": 30, "ttype": 20, "prac": 15},
    }
    defaults = TRACK_TYPE_DEFAULTS.get(parent_type, TRACK_TYPE_DEFAULTS["intermediate"])

    # Weight sliders in collapsible expander
    with st.expander("Projection Weights", expanded=False):
        st.caption(f"Defaults tuned for **{parent_type}** tracks. "
                   "Adjust weights — auto-normalizes to 100%. "
                   "Qualifying contributes a fixed 15% as a finish signal.")
        w_cols = st.columns(4)
        w_odds = w_cols[0].number_input("Odds", 0, 100, defaults["odds"], 5, key="pw_odds")
        w_track = w_cols[1].number_input("Track History", 0, 100, defaults["track"], 5, key="pw_track")
        w_ttype = w_cols[2].number_input("Track Type", 0, 100, defaults["ttype"], 5, key="pw_ttype")
        w_prac = w_cols[3].number_input("Practice", 0, 100, defaults["prac"], 5, key="pw_prac")

    # Smart weight handling: drop unavailable signals, redistribute
    has_odds = bool(odds_data)
    has_practice = bool(practice_data)
    effective_odds = w_odds if has_odds else 0
    effective_prac = w_prac if has_practice else 0

    raw_total = w_track + w_ttype + effective_prac + effective_odds
    if raw_total > 0:
        wn = {
            "track": w_track / raw_total,
            "track_type": w_ttype / raw_total,
            "qual": 0,
            "practice": effective_prac / raw_total,
            "odds": effective_odds / raw_total,
        }
    else:
        # Fallback: 60% track, 40% track type
        wn = {"track": 0.60, "track_type": 0.40, "qual": 0, "practice": 0, "odds": 0}

    redist_msgs = []
    if not has_odds:
        redist_msgs.append("No odds data")
    if not has_practice:
        redist_msgs.append("No practice data")
    if redist_msgs:
        st.caption(f"⚠️ {' | '.join(redist_msgs)} — weight redistributed to available signals.")

    # ── Dynamic dominator ceiling from DB ────────────────────────────────────
    calibration = _get_track_dominator_calibration(track_name, track_type)

    if race_laps > 0:
        # Historical max at THIS track — the realistic dominator ceiling
        hist_max_ll = calibration["max_laps_led"]
        hist_max_fl = calibration["max_fastest_laps"]
        dom_ceiling = hist_max_ll * 0.25 + hist_max_fl * 0.45

        info_cols = st.columns(4)
        info_cols[0].metric("Race Laps", f"{race_laps}")
        info_cols[1].metric("Max Laps Led Pts", f"{race_laps * 0.25:.1f}")
        info_cols[2].metric("Max Fastest Lap Pts", f"{race_laps * 0.45:.1f}")
        info_cols[3].metric("Dominator Ceiling", f"{dom_ceiling:.1f}")
        st.caption(
            f"Laps led = 0.25 pts/lap | Fastest laps = 0.45 pts/lap | "
            f"Place diff = ±1.0 pts/pos | {race_laps} total laps | "
            f"Historical max at {track_name}: {hist_max_ll} laps led, "
            f"{hist_max_fl} fastest laps"
        )

    # Weight info — display BEFORE projections table, using the SAME wn dict
    active = [(k, v) for k, v in wn.items() if v > 0]
    weight_str = " | ".join(f"{k.replace('_', ' ').title()} {v:.0%}" for k, v in active)
    st.caption(f"Weights: {weight_str}")

    # Build projections
    _build_dfs_projections(
        entry_list_df, qualifying_df, lap_averages_df,
        practice_data, wn, track_name, series_id, dk_df, race_laps,
        odds_data=odds_data or {}, calibration=calibration,
        race_id=race_id, race_name=race_name, is_prerace=is_prerace,
        race_date=race_date, season=season,
    )


def _query_db_track_history(track_name, series_id, exclude_race_id=None,
                             before_date=None):
    """Query per-driver track history from DB with date filtering.

    Used for completed races to prevent data leakage — the scraped
    driveraverages.com data always includes the current race's results.
    """
    if not os.path.exists(PROJ_DB):
        return pd.DataFrame()

    conn = sqlite3.connect(PROJ_DB)
    where = "WHERE t.name LIKE ? AND r.series_id = ?"
    params = [f"%{track_name}%", series_id]
    if exclude_race_id:
        where += " AND r.id != ?"
        params.append(exclude_race_id)
    if before_date:
        where += " AND r.race_date < ?"
        params.append(before_date)

    query = f'''
        SELECT d.full_name as Driver,
               COUNT(*) as Races,
               ROUND(AVG(rr.finish_pos), 1) as "Avg Finish",
               ROUND(AVG(rr.start_pos), 1) as "Avg Start",
               SUM(rr.laps_led) as "Laps Led",
               SUM(rr.fastest_laps) as "Fastest Laps",
               ROUND(AVG(rr.avg_running_position), 1) as "Avg Run Pos",
               SUM(CASE WHEN rr.finish_pos = 1 THEN 1 ELSE 0 END) as Wins,
               SUM(CASE WHEN rr.finish_pos <= 5 THEN 1 ELSE 0 END) as "Top 5",
               SUM(CASE WHEN rr.finish_pos <= 10 THEN 1 ELSE 0 END) as "Top 10",
               SUM(CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END) as DNF
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        {where}
        GROUP BY d.id
        HAVING COUNT(*) >= 1
    '''
    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.warning(f"Track history query error: {e}")
        df = pd.DataFrame()
    conn.close()
    return df


def _query_db_track_type_history(track_type, series_id, exclude_track=None,
                                  exclude_race_id=None, before_date=None):
    """Query per-driver stats across all tracks of a given type from DB.

    Used for completed races to prevent data leakage.
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
               AVG(rr.avg_running_position) as avg_running_pos,
               SUM(rr.laps_led) as total_laps_led
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE t.name IN ({placeholders})
          AND r.series_id = ?
          {where_extra}
        GROUP BY d.id
    '''
    rows = conn.execute(query, params).fetchall()
    conn.close()

    result = {}
    for row in rows:
        name, races, avg_f, avg_arp, ll = row
        if races and races > 0:
            result[name] = {
                "avg_finish": avg_f or 20,
                "avg_running_pos": avg_arp,  # None if no ARP data yet
                "laps_led_per_race": (ll or 0) / races,
                "races": races,
            }
    return result


def _query_races_to_subtract(track_name, series_id, race_date, db_race_id=None):
    """Query DB for per-driver results at this track on or after race_date.

    Returns dict: {driver_name: [{finish_pos, start_pos, laps_led, ...}, ...]}
    These results need to be subtracted from the scraped driveraverages.com
    aggregates to prevent data leakage in completed race projections.

    Deduplicates by (driver, date) to handle duplicate DB race entries
    for the same real-world race (e.g. two entries for same date).
    """
    if not os.path.exists(PROJ_DB):
        return {}

    conn = sqlite3.connect(PROJ_DB)
    # Use SUBSTR to truncate datetime strings to just the date portion
    # and pick one result per driver per date to avoid double-counting
    # from duplicate DB race entries.
    where = "WHERE t.name LIKE ? AND r.series_id = ? AND SUBSTR(r.race_date, 1, 10) >= ?"
    params = [f"%{track_name}%", series_id, race_date[:10]]

    rows = conn.execute(f'''
        SELECT d.full_name, rr.finish_pos, rr.start_pos, rr.laps_led,
               CASE WHEN LOWER(rr.status) NOT IN ('running','') THEN 1 ELSE 0 END as is_dnf,
               SUBSTR(r.race_date, 1, 10) as race_dt
        FROM race_results rr
        JOIN drivers d ON d.id = rr.driver_id
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        {where}
        GROUP BY d.full_name, SUBSTR(r.race_date, 1, 10)
    ''', params).fetchall()
    conn.close()

    result = {}
    for name, fp, sp, ll, dnf, dt in rows:
        result.setdefault(name, []).append({
            "finish_pos": fp, "start_pos": sp,
            "laps_led": ll or 0, "is_dnf": dnf,
        })
    return result


def _subtract_races_from_scraped(th_df, races_to_remove):
    """Subtract specific race results from scraped aggregate data.

    driveraverages.com gives us: avg_finish over N races, total laps_led, etc.
    We need to remove specific races to get a clean historical baseline.

    Formula: new_avg = (old_avg * old_races - sum(removed_finishes)) / (old_races - n_removed)
    """
    th_df = th_df.copy()
    for col in ["Avg Finish", "Avg Start", "Laps Led", "Races", "Wins",
                 "Top 5", "Top 10", "DNF"]:
        if col in th_df.columns:
            th_df[col] = pd.to_numeric(th_df[col], errors="coerce")

    rows_to_drop = []
    for idx, row in th_df.iterrows():
        driver = row.get("Driver", "")
        removals = races_to_remove.get(driver)
        if not removals:
            # Also try fuzzy match
            matched = fuzzy_match_name(driver, list(races_to_remove.keys()))
            removals = races_to_remove.get(matched) if matched else None
        if not removals:
            continue

        n_remove = len(removals)
        old_races = row.get("Races", 0)
        if pd.isna(old_races) or old_races <= n_remove:
            # Removing all races — drop this driver entirely
            rows_to_drop.append(idx)
            continue

        new_races = old_races - n_remove

        # Adjust avg finish
        old_avg_f = row.get("Avg Finish")
        if pd.notna(old_avg_f):
            removed_f = sum(r["finish_pos"] for r in removals if r["finish_pos"])
            new_avg_f = (old_avg_f * old_races - removed_f) / new_races
            th_df.at[idx, "Avg Finish"] = round(new_avg_f, 1)

        # Adjust avg start
        old_avg_s = row.get("Avg Start")
        if pd.notna(old_avg_s):
            removed_s = sum(r["start_pos"] for r in removals if r["start_pos"])
            new_avg_s = (old_avg_s * old_races - removed_s) / new_races
            th_df.at[idx, "Avg Start"] = round(new_avg_s, 1)

        # Adjust laps led (total, not average)
        old_ll = row.get("Laps Led", 0)
        if pd.notna(old_ll):
            removed_ll = sum(r["laps_led"] for r in removals)
            th_df.at[idx, "Laps Led"] = max(0, old_ll - removed_ll)

        # Adjust DNF count
        old_dnf = row.get("DNF", 0)
        if pd.notna(old_dnf):
            removed_dnf = sum(r["is_dnf"] for r in removals)
            th_df.at[idx, "DNF"] = max(0, old_dnf - removed_dnf)

        # Adjust wins (finish_pos == 1)
        old_wins = row.get("Wins", 0)
        if pd.notna(old_wins):
            removed_wins = sum(1 for r in removals if r["finish_pos"] == 1)
            th_df.at[idx, "Wins"] = max(0, old_wins - removed_wins)

        # Adjust top 5/10
        for col_name, threshold in [("Top 5", 5), ("Top 10", 10)]:
            old_val = row.get(col_name, 0)
            if pd.notna(old_val):
                removed_val = sum(1 for r in removals
                                  if r["finish_pos"] and r["finish_pos"] <= threshold)
                th_df.at[idx, col_name] = max(0, old_val - removed_val)

        th_df.at[idx, "Races"] = new_races

    if rows_to_drop:
        th_df = th_df.drop(rows_to_drop)

    return th_df


def _resolve_db_race_id(api_race_id, series_id):
    """Resolve NASCAR API race_id to internal DB race_id."""
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


def _build_dfs_projections(entry_df, qualifying_df, lap_averages_df,
                            practice_data, wn, track_name, series_id, dk_df,
                            race_laps, odds_data=None, calibration=None,
                            race_id=None, race_name="", is_prerace=True,
                            race_date="", season=2026):
    """Build DFS-aware projections that estimate actual DK point components."""
    if odds_data is None:
        odds_data = {}
    if calibration is None:
        track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
        calibration = _get_track_dominator_calibration(track_name, track_type)

    # Use entry list or salary list as driver pool
    if not entry_df.empty:
        drivers = entry_df["Driver"].dropna().unique().tolist()
        base_df = entry_df[["Driver"] + [c for c in ["Car"] if c in entry_df.columns]].drop_duplicates("Driver")
    elif not dk_df.empty:
        drivers = dk_df["Driver"].dropna().unique().tolist()
        base_df = dk_df[["Driver"]].drop_duplicates("Driver")
    else:
        st.warning("No driver list available.")
        return

    field_size = len(drivers)
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")

    # ── 1. Track History Signal ──────────────────────────────────────────────
    # Hybrid approach: scrape driveraverages.com for rich baseline data, then
    # for completed races, subtract the current race + future races using DB
    # to prevent data leakage. This gives us full historical depth without
    # contamination from races that haven't happened yet (from this race's POV).
    th_data = {}  # driver -> {avg_finish, avg_start, laps_led, races, avg_rating}

    # Resolve DB race ID for exclusion
    db_race_id = _resolve_db_race_id(race_id, series_id) if race_id else None

    # Try scraping driveraverages.com as the baseline (rich historical depth).
    # Falls back to DB-only queries for series where the scraper returns empty
    # (e.g. O'Reilly/Xfinity, Trucks if the site doesn't serve that series).
    with st.spinner("Loading track history..."):
        th_df = scrape_track_history(track_name, series_id)

    if th_df.empty:
        # Scrape failed — fall back to DB-only track history
        db_th = _query_db_track_history(
            track_name, series_id,
            exclude_race_id=db_race_id,
            before_date=race_date if not is_prerace else None,
        )
        if not db_th.empty:
            th_df = db_th

    if not is_prerace and race_date and not th_df.empty:
        # Completed race — subtract current race + future races from scraped data.
        # Only needed for scraped data; DB queries already filter via before_date.
        # Check if this DF came from scraping (has "Avg Rating" column) vs DB.
        is_scraped = "Avg Rating" in th_df.columns
        if is_scraped:
            races_to_remove = _query_races_to_subtract(
                track_name, series_id, race_date, db_race_id
            )
            if races_to_remove:
                th_df = _subtract_races_from_scraped(th_df, races_to_remove)

    # ── Historical DK points at this track (for display + th_data enrichment) ──
    dk_history = query_driver_dk_points_at_track(
        track_name, series_id, min_season=2022,
        before_date=race_date if not is_prerace else None,
    )
    dk_hist_names = list(dk_history.keys())

    if not th_df.empty:
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races",
                     "Avg Rating", "Wins", "Top 5", "Top 10", "DNF"]:
            if col in th_df.columns:
                th_df[col] = pd.to_numeric(th_df[col], errors="coerce")
        th_idx = th_df.drop_duplicates("Driver").set_index("Driver")
        th_names = th_idx.index.tolist()
        for d in drivers:
            # Try exact match first, then fuzzy match
            matched = d if d in th_idx.index else fuzzy_match_name(d, th_names)
            if matched and matched in th_idx.index:
                row = th_idx.loc[matched]
                races = row.get("Races", 1) if pd.notna(row.get("Races")) and row.get("Races") > 0 else 1
                laps_led = row.get("Laps Led", 0) if pd.notna(row.get("Laps Led")) else 0
                fastest_laps = row.get("Fastest Laps", 0) if pd.notna(row.get("Fastest Laps")) else 0
                # Look up avg DK points for this driver at this track
                dk_hist_entry = dk_history.get(matched) or dk_history.get(d) or {}
                th_data[d] = {
                    "avg_finish": row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20,
                    "avg_start": row.get("Avg Start", 20) if pd.notna(row.get("Avg Start")) else 20,
                    "avg_running_pos": row.get("Avg Run Pos") if pd.notna(row.get("Avg Run Pos", None)) else None,
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

    # ── 2. Track Type Signal — DB-backed aggregation across similar tracks ──
    # Uses database race_results for clean, date-filtered data at tracks
    # of the same type (e.g., all intermediates for a Charlotte projection).
    # Excludes the specific track (covered by track history signal above)
    # and any races on/after the current race date for completed races.
    tt_data = {}
    if wn.get("track_type", 0) > 0:
        tt_raw = _query_db_track_type_history(
            track_type, series_id,
            exclude_track=track_name,
            exclude_race_id=db_race_id,
            before_date=race_date if not is_prerace else None,
        )
        if tt_raw:
            tt_names = list(tt_raw.keys())
            for d in drivers:
                matched = d if d in tt_raw else fuzzy_match_name(d, tt_names)
                if matched and matched in tt_raw:
                    tt_data[d] = tt_raw[matched]

    # ── 3. Qualifying Signal ─────────────────────────────────────────────────
    qual_pos = {}
    if not qualifying_df.empty and "Qualifying Position" in qualifying_df.columns:
        qclean = qualifying_df.dropna(subset=["Driver"]).copy()
        qclean["Qualifying Position"] = pd.to_numeric(qclean["Qualifying Position"], errors="coerce")
        qidx = qclean.drop_duplicates("Driver").set_index("Driver")["Qualifying Position"]
        for d in drivers:
            if d in qidx.index and pd.notna(qidx[d]):
                qual_pos[d] = int(qidx[d])

    # ── 4. Practice Signal ───────────────────────────────────────────────────
    prac_rank = {}
    if practice_data:
        max_p = max(practice_data.values()) if practice_data else field_size
        for d in drivers:
            if d in practice_data:
                prac_rank[d] = practice_data[d]

    # ── 5. Odds Signal — convert American odds to implied finish position ────
    odds_finish = {}
    if odds_data and wn.get("odds", 0) > 0:
        # Filter out null/empty odds before processing
        clean_odds = {k: v for k, v in odds_data.items()
                      if v is not None and str(v).strip() not in ("", "None", "null", "N/A")}

        # Convert odds to implied probability, then rank → finish estimate
        odds_probs = {}
        for name, odds_str in clean_odds.items():
            try:
                odds_val = int(str(odds_str).replace("+", ""))
                if odds_val > 0:
                    prob = 100 / (odds_val + 100)
                elif odds_val < 0:
                    prob = abs(odds_val) / (abs(odds_val) + 100)
                else:
                    continue  # odds_val == 0 is invalid
                odds_probs[name] = prob
            except (ValueError, TypeError):
                continue

        # Only use odds if we have data for a meaningful fraction of the field
        if len(odds_probs) >= field_size * 0.3:
            ranked = sorted(odds_probs.items(), key=lambda x: x[1], reverse=True)
            for rank, (name, prob) in enumerate(ranked, 1):
                matched = fuzzy_match_name(name, drivers)
                if matched:
                    odds_finish[matched] = rank * (field_size / len(ranked))
        elif odds_probs:
            pct = len(odds_probs) / field_size * 100
            st.caption(f"⚠️ Odds cover only {len(odds_probs)}/{field_size} drivers "
                       f"({pct:.0f}%) — need ≥30% to use as signal. "
                       f"Odds weight redistributed to other signals.")

    # ── Build odds display lookup for the projections table ──────────────────
    from src.data import round_odds
    driver_odds_display = {}  # d -> {"odds_str": "+350", "impl_pct": 22.2}
    if odds_data:
        for name, odds_str in odds_data.items():
            if odds_str is None or str(odds_str).strip() in ("", "None", "null", "N/A"):
                continue
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
                        "odds_str": rounded,  # numeric for sorting
                        "impl_pct": round(impl, 1),
                    }
            except (ValueError, TypeError):
                continue

    # ── DNF risk data (career crash/mechanical rates) ──────────────────────
    dnf_data = query_driver_career_dnf(series_id, before_date=race_date if not is_prerace else None)

    # ── PROJECT EACH DRIVER — Raw composite finish score ─────────────────────
    driver_raw_scores = {}  # d -> raw weighted score (higher = better driver)
    dom_raw_scores = {}     # d -> raw dominator score (for allocation)
    fl_raw_scores = {}      # d -> raw fastest lap score (for allocation)

    for d in drivers:
        th = th_data.get(d)
        tt = tt_data.get(d)
        qp = qual_pos.get(d)
        pr = prac_rank.get(d)
        od = odds_finish.get(d)

        # --- Compute raw weighted finish estimate (avg-of-averages approach) ---
        finish_signals = []
        signal_weights = []

        mid_field = field_size * 0.5
        MIN_RACES_FULL_TRUST = 5

        if th:
            races = th.get("races", 1)
            trust = min(1.0, races / MIN_RACES_FULL_TRUST)

            # Track finish signal: ARP 65% + Avg Finish 35%
            # ARP filters wreck luck (driver who ran 5th but wrecked to 30th
            # still projects as a ~5th-place driver next time).
            # Avg finish captures actual outcomes.
            # Dominator value (laps led, fastest laps) handled separately
            # in Stage 2 allocation — NOT included here to avoid double-counting.
            arp = th.get("avg_running_pos")
            af = th["avg_finish"]

            if arp is not None:
                base_finish = arp * 0.65 + af * 0.35
            else:
                base_finish = af  # no ARP data — use finish only

            regressed = base_finish * trust + mid_field * (1 - trust)
            finish_signals.append(regressed)
            signal_weights.append(wn["track"])

        # Track type signal — if no track history, absorb the track weight too
        tt_weight = wn.get("track_type", 0)
        if not th and tt and wn.get("track", 0) > 0:
            # No track-specific data: give track type the combined weight
            tt_weight = wn.get("track_type", 0) + wn.get("track", 0)

        if tt and tt_weight > 0:
            tt_races = tt.get("races", 1) if isinstance(tt, dict) and "races" in tt else 3
            tt_trust = min(1.0, tt_races / MIN_RACES_FULL_TRUST)
            tt_arp = tt.get("avg_running_pos") if isinstance(tt, dict) else None
            tt_af = tt.get("avg_finish", mid_field) if isinstance(tt, dict) else mid_field
            if tt_arp is not None:
                tt_avg = tt_arp * 0.65 + tt_af * 0.35
            else:
                tt_avg = tt_af
            tt_regressed = tt_avg * tt_trust + mid_field * (1 - tt_trust)
            finish_signals.append(tt_regressed)
            signal_weights.append(tt_weight)

        # Qualifying position — a meaningful finish signal. Drivers who
        # qualify well tend to finish well (especially at short tracks).
        # Uses a fixed 15% weight to anchor projections toward start position
        # and prevent absurd negative place differentials for fast qualifiers
        # with mediocre track history.
        if qp and qp <= field_size:
            qual_finish = qp * 0.80 + mid_field * 0.20  # regress toward mid
            finish_signals.append(qual_finish)
            signal_weights.append(0.15)  # fixed 15% weight

        if pr:
            # Practice rank: regress toward mid-field (noisiest signal)
            regress = 0.30
            prac_finish = pr * (1 - regress) + (field_size * 0.5) * regress
            finish_signals.append(prac_finish)
            signal_weights.append(wn["practice"])

        if od and wn.get("odds", 0) > 0:
            finish_signals.append(od)
            signal_weights.append(wn["odds"])

        if finish_signals:
            total_w = sum(signal_weights)
            raw_finish = sum(f * w for f, w in zip(finish_signals, signal_weights)) / total_w
        else:
            raw_finish = field_size * 0.75  # default: well below mid-field

        # DNF risk adjustment — penalize drivers with high crash/mechanical failure rates
        dnf = dnf_data.get(d)
        if not dnf:
            dnf_matched = fuzzy_match_name(d, list(dnf_data.keys())) if dnf_data else None
            dnf = dnf_data.get(dnf_matched) if dnf_matched else None
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

        # --- Build raw dominator score (laps led potential) — weight-aware ---
        # Dom signal weights reflect user's projection weights so changing
        # odds weight also changes who is projected to lead laps
        dom_score = 0.0
        if race_laps > 0:
            dom_signals = []
            dom_weights_list = []

            if th and th["races"] >= 1 and th["laps_led"] > 0:
                ll_per_race = th["laps_led"] / th["races"]
                dom_signals.append(ll_per_race)
                dom_weights_list.append(wn.get("track", 0.20))

            if tt and isinstance(tt, dict) and tt.get("laps_led_per_race", 0) > 0:
                dom_signals.append(tt["laps_led_per_race"])
                dom_weights_list.append(wn.get("track_type", 0.15))

            if qp and qp <= field_size:
                qual_dom = max(0, (field_size + 1 - qp) / field_size) ** 1.5 * 30
                dom_signals.append(qual_dom)
                dom_weights_list.append(0.15)  # fixed weight — qual speed predicts domination

            if od and wn.get("odds", 0) > 0:
                odds_dom = max(0, (field_size + 1 - od) / field_size) ** 1.3 * 35
                dom_signals.append(odds_dom)
                dom_weights_list.append(wn.get("odds", 0.15))

            if pr:
                max_p_val = max(practice_data.values()) if practice_data else field_size
                prac_dom = max(0, (max_p_val + 1 - pr) / max_p_val) * 15
                dom_signals.append(prac_dom)
                dom_weights_list.append(wn.get("practice", 0.10))

            if dom_signals:
                total_dw = sum(dom_weights_list)
                dom_score = sum(s * w for s, w in zip(dom_signals, dom_weights_list)) / total_dw
            else:
                dom_score = max(0, (field_size - raw_finish) / field_size) * 5

        # --- Build raw fastest laps score — weight-aware ---
        fl_score = 0.0
        if race_laps > 0:
            fl_signals = []
            fl_signal_weights = []

            if dom_score > 0:
                fl_signals.append(dom_score * 0.5)
                fl_signal_weights.append(0.25)

            if qp and qp <= field_size:
                qual_fl = max(0, (field_size + 1 - qp) / field_size) * 15
                fl_signals.append(qual_fl)
                fl_signal_weights.append(0.15)  # fixed weight — qual speed predicts fast laps

            if pr:
                max_p_val = max(practice_data.values()) if practice_data else field_size
                prac_fl = max(0, (max_p_val + 1 - pr) / max_p_val) * 12
                fl_signals.append(prac_fl)
                fl_signal_weights.append(wn.get("practice", 0.10))

            if od and wn.get("odds", 0) > 0:
                odds_fl = max(0, (field_size + 1 - od) / field_size) * 12
                fl_signals.append(odds_fl)
                fl_signal_weights.append(wn.get("odds", 0.15))

            finish_fl = max(0, (field_size - raw_finish) / field_size) * 10
            fl_signals.append(finish_fl)
            fl_signal_weights.append(0.10)

            if fl_signals:
                total_fw = sum(fl_signal_weights)
                fl_score = sum(s * w for s, w in zip(fl_signals, fl_signal_weights)) / total_fw

        dom_raw_scores[d] = dom_score
        fl_raw_scores[d] = fl_score

    # ── RANK-ORDER FINISH SPREADING ──────────────────────────────────────────
    # The raw scores cluster everyone around 8-15 because we're averaging
    # averages. Instead, rank-order drivers by raw score and map each rank
    # to a realistic projected finish position that spans 1 → field_size.
    #
    # We use a slightly concave mapping so the top drivers separate more
    # (difference between 1st and 5th is bigger than between 25th and 30th).
    sorted_drivers = sorted(driver_raw_scores.items(), key=lambda x: x[1])
    n = len(sorted_drivers)

    driver_proj_finish = {}
    for rank_idx, (d, raw_score) in enumerate(sorted_drivers):
        if n > 1:
            t = rank_idx / (n - 1)  # 0.0 (best) to 1.0 (worst)
            # Power curve: top drivers spread out, back of field compresses
            proj_finish = 1 + (field_size - 1) * (t ** 0.85)
        else:
            proj_finish = field_size * 0.5

        driver_proj_finish[d] = round(max(1, min(field_size, proj_finish)), 1)

    # ── PASS 2: Allocate laps led and fastest laps (zero-sum, DB-calibrated) ──
    allocated_ll = _allocate_laps_led(dom_raw_scores, race_laps, track_name, track_type) if race_laps > 0 else {}
    allocated_fl = _allocate_fastest_laps(fl_raw_scores, race_laps, track_type) if race_laps > 0 else {}

    # ── PASS 3: Compute final DK points ─────────────────────────────────────
    rows = []
    for d in drivers:
        proj_finish = driver_proj_finish[d]
        proj_laps_led = allocated_ll.get(d, 0)
        proj_fastest = allocated_fl.get(d, 0)

        # Start position = qualifying pos if known, else historical avg start, else projected finish
        th_for_start = th_data.get(d)
        historical_avg_start = th_for_start.get("avg_start") if th_for_start else None
        start_pos = qual_pos.get(d) or (round(historical_avg_start) if historical_avg_start else round(proj_finish))
        proj_finish_int = round(proj_finish)  # DK uses integer positions

        # DK scoring components (all based on integer positions)
        finish_pts = _expected_finish_from_avg(proj_finish_int)
        diff_pts = int(start_pos - proj_finish_int)  # DK: ±1 per position, always integer
        led_pts = round(proj_laps_led) * 0.25
        fl_pts = round(proj_fastest) * 0.45
        proj_dk = round(finish_pts + diff_pts + led_pts + fl_pts, 1)

        # Historical DK points at this track (actual avg/best/worst)
        dk_hist = dk_history.get(d)
        if not dk_hist:
            matched_dk = fuzzy_match_name(d, dk_hist_names) if dk_hist_names else None
            dk_hist = dk_history.get(matched_dk) if matched_dk else None

        avg_dk_track = dk_hist["avg_dk"] if dk_hist else None
        best_dk_track = dk_hist["best_dk"] if dk_hist else None
        worst_dk_track = dk_hist["worst_dk"] if dk_hist else None

        odds_info = driver_odds_display.get(d, {})
        odds_val = odds_info.get("odds_str", None)
        # odds_str is now numeric from round_odds()
        odds_numeric = odds_val if isinstance(odds_val, (int, float)) else None

        rows.append({
            "Driver": d,
            "Proj DK": proj_dk,
            "Proj Finish": proj_finish,
            "Win Odds": odds_numeric,
            "Impl %": odds_info.get("impl_pct", None),
            "Finish Pts": round(finish_pts, 1),
            "Diff Pts": round(diff_pts, 1),
            "Led Pts": round(led_pts, 1),
            "FL Pts": round(fl_pts, 1),
            "Proj Laps Led": round(proj_laps_led),
            "Proj Fast Laps": round(proj_fastest),
            "Avg DK": avg_dk_track,
            "Best DK": best_dk_track,
            "Worst DK": worst_dk_track,
            "Start": start_pos,
        })

    proj = pd.DataFrame(rows)

    # Merge car number if available
    if "Car" in base_df.columns:
        proj = proj.merge(base_df[["Driver", "Car"]].drop_duplicates("Driver"),
                          on="Driver", how="left")

    # Merge salary
    if not dk_df.empty:
        proj = fuzzy_merge(proj, dk_df, on="Driver", how="left",
                           right_cols=["DK Salary"])
        proj["Value"] = np.where(
            proj["DK Salary"].notna() & (proj["DK Salary"] > 0),
            (proj["Proj DK"] / (proj["DK Salary"] / 1000)).round(2),
            np.nan
        )

    proj = proj.sort_values("Proj DK", ascending=False).reset_index(drop=True)
    proj.index = proj.index + 1
    proj.index.name = "Rank"

    # Share Proj DK with optimizer tab via session state
    st.session_state["proj_dk_map"] = dict(zip(proj["Driver"], proj["Proj DK"]))

    # Rename Start to Qual Pos for clarity in projections context
    if "Start" in proj.columns:
        proj = proj.rename(columns={"Start": "Qual Pos"})

    # Display columns
    display_cols = ["Driver"]
    if "Car" in proj.columns:
        display_cols.append("Car")
    if "Win Odds" in proj.columns and proj["Win Odds"].notna().any():
        display_cols.append("Win Odds")
    if "Impl %" in proj.columns and proj["Impl %"].notna().any():
        display_cols.append("Impl %")
    if "DK Salary" in proj.columns:
        display_cols.append("DK Salary")
    if "Qual Pos" in proj.columns:
        display_cols.append("Qual Pos")
    display_cols.extend(["Proj DK", "Proj Finish", "Finish Pts", "Diff Pts",
                         "Led Pts", "FL Pts", "Proj Laps Led", "Proj Fast Laps",
                         "Avg DK", "Best DK", "Worst DK"])
    if "Value" in proj.columns:
        display_cols.append("Value")
    avail = [c for c in display_cols if c in proj.columns]

    # Auto-save pre-race projections for historical record
    if is_prerace and race_id:
        _auto_save_key = f"proj_autosaved_{race_id}"
        if _auto_save_key not in st.session_state:
            try:
                from tabs.tab_accuracy import save_projections_to_db
                from datetime import datetime
                season = datetime.now().year
                save_projections_to_db(
                    proj, race_id, race_name, track_name,
                    series_id, season, wn
                )
                st.session_state[_auto_save_key] = True
            except Exception:
                pass

    # Export and Save — above the table for easy access
    exp_cols = st.columns([1, 1, 3])
    with exp_cols[0]:
        csv = proj[avail].to_csv(index=True).encode("utf-8")
        st.download_button("Export CSV", csv,
                           "projections.csv", "text/csv", key="proj_export")
    with exp_cols[1]:
        if st.button("Save Projections", key="proj_save", type="secondary",
                      help="Save to DB for accuracy tracking in the Accuracy tab"):
            from tabs.tab_accuracy import save_projections_to_db
            from datetime import datetime
            season = datetime.now().year
            count = save_projections_to_db(
                proj, race_id, race_name, track_name,
                series_id, season, wn
            )
            if count > 0:
                st.success(f"Saved {count} projections for accuracy tracking")
            else:
                st.warning("Could not save projections")

    disp = format_display_df(proj[avail].copy())
    st.dataframe(safe_fillna(disp), use_container_width=True, hide_index=False, height=550)

    # Chart — all drivers, stacked bar with component breakdown on hover
    import plotly.graph_objects as go
    chart_df = proj.copy()
    chart_df = chart_df.sort_values("Proj DK", ascending=True)  # horizontal bar: bottom = best

    # Build stacked horizontal bar with DK scoring components
    component_cols = {
        "Finish Pts": "#4a7dfc",
        "Diff Pts": "#36b37e",
        "Led Pts": "#ff9f43",
        "FL Pts": "#f5365c",
    }

    from src.utils import short_name_series
    chart_df["Short"] = short_name_series(chart_df["Driver"].tolist())

    fig = go.Figure()
    for comp, color in component_cols.items():
        if comp in chart_df.columns:
            fig.add_trace(go.Bar(
                y=chart_df["Short"],
                x=chart_df[comp].clip(lower=0),
                name=comp,
                orientation="h",
                marker_color=color,
                hovertemplate=(
                    "%{y}<br>"
                    + comp + ": %{x:.1f}<br>"
                    + "<extra></extra>"
                ),
            ))

    # Custom hover with all components
    custom_hover = []
    for _, row in chart_df.iterrows():
        text = (
            f"<b>{row['Driver']}</b><br>"
            f"Proj DK: {row.get('Proj DK', 0):.1f}<br>"
            f"Proj Finish: {row.get('Proj Finish', 0):.1f}<br>"
            f"Finish Pts: {row.get('Finish Pts', 0):.1f}<br>"
            f"Diff Pts: {row.get('Diff Pts', 0):+.1f}<br>"
            f"Led Pts: {row.get('Led Pts', 0):.1f}<br>"
            f"FL Pts: {row.get('FL Pts', 0):.1f}<br>"
            f"Laps Led: {row.get('Proj Laps Led', 0):.0f}<br>"
            f"Fast Laps: {row.get('Proj Fast Laps', 0):.0f}<br>"
            f"Track: {pd.to_numeric(row.get('Track', 0), errors='coerce') or 0:.1f}"
        )
        custom_hover.append(text)

    # Add invisible scatter trace for rich hover
    fig.add_trace(go.Scatter(
        y=chart_df["Short"],
        x=chart_df["Proj DK"],
        mode="markers",
        marker=dict(size=1, opacity=0),
        hovertext=custom_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    n_drivers = len(chart_df)
    fig.update_layout(
        barmode="stack",
        title="All Drivers — Projected DK Points Breakdown",
        xaxis_title="Projected DK Points",
        yaxis_title="",
        height=max(400, n_drivers * 22),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#c9d1d9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=30),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)
