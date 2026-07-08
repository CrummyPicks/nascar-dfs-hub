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
import math

from src.config import (
    DEFAULT_PROJECTION_WEIGHTS, DB_PATH, TRACK_TYPE_MAP,
    TRACK_TYPE_PARENT, DK_FINISH_POINTS, TRACK_TYPE_WEIGHT_DEFAULTS,
    CROSS_SERIES_HIERARCHY, PROJECTION_RECENCY_DECAY_STEP,
    DK_PTS_LAP_LED, DK_PTS_FASTEST_LAP, DK_PTS_PLACE_DIFF,
)


def _recency_weight_sql(rank_col: str = "rn") -> str:
    """SQL fragment: per-race recency weight given a newest-first rank column.

    weight = MAX(0, 1 - (rank-1)*DECAY_STEP). Must be used inside a query that
    supplies `rank_col` via ROW_NUMBER() OVER (PARTITION BY driver/team
    ORDER BY race_date DESC), so the most recent races dominate the aggregate
    and a hot recent streak isn't drowned by the volume of older races.
    Shared by track history, track-type form, and team quality.
    """
    return f"MAX(0.0, 1.0 - ({rank_col} - 1) * {PROJECTION_RECENCY_DECAY_STEP})"
from src.data import (
    scrape_track_history, query_driver_dk_points_at_track,
    query_driver_career_dnf, query_team_stats, query_manufacturer_stats,
    compute_team_adjusted_track_history,
)
# projection_bar no longer used — replaced with inline stacked bar
from src.utils import safe_fillna, format_display_df, calc_dk_points, fuzzy_match_name, fuzzy_merge, arp_finish_blend

# Dominator allocators + helpers now live in the engine layer
# (src/projections.py). Re-exported here so existing
# `from tabs.tab_projections import ...` callers keep working unchanged.
from src.projections import (
    TRACK_TYPE_CONCENTRATION, _interp_curve, _soft_rank_shares,
    _start_avail, _apply_start_gate, _allocate_laps_led,
    _allocate_fastest_laps, _cap_and_redistribute,
    _LL_SOFT_TAU, _FL_SOFT_TAU, _LL_SOFT_TOPK, _FL_SOFT_TOPK,
    _LL_AVAIL, _LL_AVAIL_DEFAULT,
)

# Projection-engine DB queries now live in the data layer (src/data.py).
# Re-exported so existing `from tabs.tab_projections import ...` callers
# (tab_accuracy, tab_cautions, tab_race_lab, scripts/backtest_weights)
# keep working unchanged. _resolve_db_race_id intentionally stays here:
# src/data.py already has a DIFFERENT _resolve_db_race_id (series-filtered).
from src.data import (
    TRACK_TYPE_DOM_DEFAULTS, _get_track_dominator_calibration,
    _query_db_track_history, _query_db_track_type_history,
)

PROJ_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")


# ── Helpers ──────────────────────────────────────────────────────────────────

# Removed: _find_db_race_id() had an unsafe race_name-only fallback that
# could resolve a recurring race (e.g. "Kansas Lottery 300") to the wrong
# year. Use src.data._resolve_db_race_id(api_race_id, series_id) instead,
# which is uniqueness-safe.


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


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           is_prerace, race_name, race_id, track_name, series_id, dk_df,
           odds_data=None, scheduled_laps=0, race_date="", season=None,
           fd_df=None, platform="DraftKings"):
    """Render the Projections tab.

    platform: "DraftKings", "FanDuel", or "Both" — controls which salary /
    projected-points / value columns are shown. The engine itself is
    platform-agnostic (projected finish, laps led, fastest laps); DK and FD
    points are both derived from those same components.
    """
    if fd_df is None:
        fd_df = pd.DataFrame()
    if season is None:
        from datetime import datetime as _dt
        _t = _dt.now()
        season = _t.year + 1 if _t.month >= 10 else _t.year
    from src.components import section_header
    section_header("Projections", race_name)

    if not is_prerace:
        # For completed races, the engine applies before_date filters to all
        # history queries so no post-race data leaks into the projection.
        st.caption(
            f"Race completed — projections computed using only data available "
            f"before {race_date}."
        )

    if entry_list_df.empty and dk_df.empty:
        st.warning("Entry list not available for this race.")
        return

    # Get race laps for dominator calculations
    race_laps = scheduled_laps or 0
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")

    # Track-type-specific default weights (from shared config)
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    defaults = TRACK_TYPE_WEIGHT_DEFAULTS.get(parent_type, TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])

    # Weight slider widget keys. Namespaced by track type so switching races
    # with a different track type starts from that type's defaults rather than
    # carrying over the last race's manual edits.
    _wkeys = {
        "odds":  f"pw_odds_{parent_type}",
        "track": f"pw_track_{parent_type}",
        "ttype": f"pw_ttype_{parent_type}",
        "prac":  f"pw_prac_{parent_type}",
        "team":  f"pw_team_{parent_type}",
        "qual":  f"pw_qual_{parent_type}",
    }
    # Initialize / reset the weight widgets' session-state. We explicitly SET
    # each key to its track-type default — on first use for this track type, or
    # when the Reset button requested it on the previous run. The OLD approach
    # popped the keys and relied on each number_input's `value` arg to restore
    # the default, but that didn't reliably take effect (the manual values
    # stuck). The number_inputs below read their value from session_state via
    # `key`, so setting it here is exactly what they display — no `value` arg,
    # which also avoids Streamlit's "value set via both arg and Session State"
    # warning.
    _reset = st.session_state.pop(f"pw_reset_{parent_type}", False)
    for _sig, _key in _wkeys.items():
        if _reset or _key not in st.session_state:
            st.session_state[_key] = defaults[_sig]

    # Weight controls — open by default so users discover they can tune the
    # projection, and know the Optimizer follows along.
    with st.expander("Projection Weights", expanded=True):
        st.caption(f"Defaults tuned for **{parent_type}** tracks. "
                   "Adjust weights (in 5s) — auto-normalizes to 100%. "
                   "The **Optimizer** page uses these same weights for its player pool.")
        if st.button("Reset to defaults", key=f"pw_reset_btn_{parent_type}",
                     help=f"Restore the tuned default weights for {parent_type} tracks"):
            st.session_state[f"pw_reset_{parent_type}"] = True
            st.rerun()
        w_cols = st.columns(6)
        w_odds = w_cols[0].number_input("Odds", 0, 100, step=5, key=_wkeys["odds"])
        w_track = w_cols[1].number_input("Track History", 0, 100, step=5, key=_wkeys["track"])
        w_ttype = w_cols[2].number_input("Track Type", 0, 100, step=5, key=_wkeys["ttype"])
        w_prac = w_cols[3].number_input("Practice", 0, 100, step=5, key=_wkeys["prac"])
        w_team = w_cols[4].number_input("Team", 0, 100, step=5, key=_wkeys["team"])
        w_qual = w_cols[5].number_input("Qualifying", 0, 100, step=5, key=_wkeys["qual"])

    # Smart weight handling: drop unavailable signals, redistribute
    has_odds = bool(odds_data)
    has_practice = bool(practice_data)
    effective_odds = w_odds if has_odds else 0
    effective_prac = w_prac if has_practice else 0

    raw_total = w_track + w_ttype + effective_prac + effective_odds + w_team + w_qual
    if raw_total > 0:
        wn = {
            "track": w_track / raw_total,
            "track_type": w_ttype / raw_total,
            "qual": w_qual / raw_total,
            "practice": effective_prac / raw_total,
            "odds": effective_odds / raw_total,
            "team": w_team / raw_total,
        }
    else:
        wn = {"track": 0.60, "track_type": 0.40, "qual": 0, "practice": 0, "odds": 0, "team": 0}

    redist_msgs = []
    if not has_odds:
        redist_msgs.append("No odds data")
    if not has_practice:
        redist_msgs.append("No practice data")
    if redist_msgs:
        st.caption(f"⚠️ {' | '.join(redist_msgs)} — weight redistributed to available signals.")

    # ── Dynamic dominator ceiling from DB ────────────────────────────────────
    calibration = _get_track_dominator_calibration(track_name, track_type, series_id)

    if race_laps > 0:
        # Historical stats at THIS track
        hist_max_ll = calibration["max_laps_led"]
        hist_max_fl = calibration["max_fastest_laps"]
        avg_top = calibration.get("avg_top_leader", hist_max_ll)
        avg_leaders = calibration.get("avg_n_leaders", 6)
        # Cap historical maxima at this week's race lap count for display.
        # When per-track data is sparse, calibration falls back to the
        # series's track-type aggregate (all Truck road races, etc.) and
        # those numbers can come from longer races. Showing "max: 99" for
        # a 72-lap race is logically impossible from the user's perspective.
        display_max_ll = min(hist_max_ll, race_laps)
        display_avg_top = min(avg_top, race_laps)
        dom_ceiling = (display_avg_top * DK_PTS_LAP_LED
                       + min(hist_max_fl, race_laps) * DK_PTS_FASTEST_LAP)

        # Fastest-laps history (parallel to laps-led): the per-race FL leader's
        # average + the all-time single-driver max. Capped at race_laps for display.
        avg_fl_leader = calibration.get("avg_fl_leader", hist_max_fl * 0.85)
        display_avg_fl = min(avg_fl_leader, race_laps)
        display_max_fl = min(hist_max_fl, race_laps)

        # Scoring summary cards + caption follow the platform picker. FanDuel
        # scores differently (0.1/lap led, 0.1/lap COMPLETED, NO fastest-lap
        # points, 0.5/pos) so the DK numbers would be flat wrong in FD mode.
        _show_dk_score = platform in ("DraftKings", "Both")
        _show_fd_score = platform in ("FanDuel", "Both")
        # Dominator ceiling per site: DK = avg-leader laps\u00d70.25 + max FL\u00d70.45;
        # FD = avg-leader laps\u00d70.1 (FD has no fastest-lap points).
        dom_ceiling_fd = display_avg_top * 0.10

        if platform == "FanDuel":
            info_cols = st.columns(4)
            info_cols[0].metric("Race Laps", f"{race_laps}")
            info_cols[1].metric("Max Laps Led Pts", f"{race_laps * 0.10:.1f}")
            info_cols[2].metric("Max Laps-Completed Pts", f"{race_laps * 0.10:.1f}")
            info_cols[3].metric("Dominator Ceiling (FD)", f"{dom_ceiling_fd:.1f}")
        elif platform == "Both":
            info_cols = st.columns(5)
            info_cols[0].metric("Race Laps", f"{race_laps}")
            info_cols[1].metric("DK Max Led Pts", f"{race_laps * DK_PTS_LAP_LED:.1f}")
            info_cols[2].metric("DK Max Fast Pts", f"{race_laps * DK_PTS_FASTEST_LAP:.1f}")
            info_cols[3].metric("DK Dom Ceiling", f"{dom_ceiling:.1f}")
            info_cols[4].metric("FD Dom Ceiling", f"{dom_ceiling_fd:.1f}")
        else:  # DraftKings
            info_cols = st.columns(4)
            info_cols[0].metric("Race Laps", f"{race_laps}")
            info_cols[1].metric("Max Laps Led Pts", f"{race_laps * DK_PTS_LAP_LED:.1f}")
            info_cols[2].metric("Max Fastest Lap Pts", f"{race_laps * DK_PTS_FASTEST_LAP:.1f}")
            info_cols[3].metric("Dominator Ceiling", f"{dom_ceiling:.1f}")

        _score_lines = []
        if _show_dk_score:
            _score_lines.append(
                f"<b>DK</b>: laps led {DK_PTS_LAP_LED}/lap &nbsp;|&nbsp; fastest laps "
                f"{DK_PTS_FASTEST_LAP}/lap &nbsp;|&nbsp; place diff \u00b1{DK_PTS_PLACE_DIFF}/pos")
        if _show_fd_score:
            _score_lines.append(
                "<b>FD</b>: laps led 0.1/lap &nbsp;|&nbsp; laps completed "
                "0.1/lap &nbsp;|&nbsp; place diff \u00b10.5/pos &nbsp;|&nbsp; "
                "<i>no fastest-lap points</i>")
        _score_lines.append(f"{race_laps} total laps")
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.82rem;font-weight:600;margin:0.3rem 0;">'
            + "<br>".join(_score_lines) + "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.82rem;font-weight:600;margin:0.3rem 0;">'
            f"Avg {avg_leaders:.0f} leaders &nbsp;|&nbsp; "
            f"Avg Race Lap Leader leads {display_avg_top:.0f} laps (max: {display_max_ll:.0f}) &nbsp;|&nbsp; "
            f"Avg Fastest-Lap Leader gets {display_avg_fl:.0f} fast laps (max: {display_max_fl:.0f})</p>",
            unsafe_allow_html=True,
        )

        # Note when qualifying hasn't happened — start positions shown are
        # the engine's estimate from historical avg start at this track/type,
        # not actual qualifying results.
        _qualifying_available = bool(qualifying_df is not None
                                      and not qualifying_df.empty
                                      and "Qualifying Position" in qualifying_df.columns
                                      and qualifying_df["Qualifying Position"].notna().any())
        if not _qualifying_available:
            st.markdown(
                '<p style="color:#fbbf24;font-size:0.80rem;font-style:italic;margin:0.2rem 0;">'
                "\u2139\ufe0f Projected Qualifying Position shown \u2014 based on historical "
                "Avg Starting Position for this track/type. Will be replaced with actual "
                "qualifying results once available.</p>",
                unsafe_allow_html=True,
            )

    # Weight info — display BEFORE projections table, using the SAME wn dict
    active = [(k, v) for k, v in wn.items() if v > 0]
    weight_str = " | ".join(f"{k.replace('_', ' ').title()} {v:.0%}" for k, v in active)
    st.markdown(
        f'<p style="color:#94a3b8;font-size:0.82rem;font-weight:600;margin:0.2rem 0;">'
        f"Weights: {weight_str}</p>",
        unsafe_allow_html=True,
    )

    # Build projections
    _build_dfs_projections(
        entry_list_df, qualifying_df, lap_averages_df,
        practice_data, wn, track_name, series_id, dk_df, race_laps,
        odds_data=odds_data or {}, calibration=calibration,
        race_id=race_id, race_name=race_name, is_prerace=is_prerace,
        race_date=race_date, season=season,
        fd_df=fd_df, platform=platform,
    )


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
    where = "WHERE t.name = ? AND r.series_id = ? AND SUBSTR(r.race_date, 1, 10) >= ?"
    params = [track_name, series_id, race_date[:10]]

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
                            race_date="", season=None, fd_df=None,
                            platform="DraftKings"):
    """Build DFS-aware projections that estimate actual DK point components."""
    if fd_df is None:
        fd_df = pd.DataFrame()
    if season is None:
        from datetime import datetime as _dt
        _t = _dt.now()
        season = _t.year + 1 if _t.month >= 10 else _t.year
    if odds_data is None:
        odds_data = {}
    if calibration is None:
        track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
        calibration = _get_track_dominator_calibration(track_name, track_type, series_id)

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
    # Always use DB for projections — cleaner data, has ARP, fastest laps,
    # and filtered to Next Gen era (2022+). The scraper is only used for
    # the Track History display tab.
    th_data = {}

    # Resolve DB race ID for exclusion
    db_race_id = _resolve_db_race_id(race_id, series_id) if race_id else None

    # Cross-series hierarchy: Cup stats flow DOWN to O'Reilly/Truck
    cross_ids = CROSS_SERIES_HIERARCHY.get(series_id, [])

    # At superspeedways we trim each driver's WORST finish from the
    # aggregate to stop a single unavoidable pileup from poisoning a
    # driver's average beyond their race-pace reality (the
    # "Hocevar problem": 4 races of 6/6/14/17 finishes averaged 10.75,
    # but adding one P35 wreck from a different team pushed avg to 15.6).
    _trim_worst = (track_type == "superspeedway")
    with st.spinner("Loading track history..."):
        th_result = _query_db_track_history(
            track_name, series_id,
            exclude_race_id=db_race_id,
            before_date=race_date if not is_prerace else None,
            cross_series_ids=cross_ids if cross_ids else None,
            trim_worst=_trim_worst,
        )
        if cross_ids:
            th_df, cross_th_df = th_result
        else:
            th_df = th_result
            cross_th_df = pd.DataFrame()

    # ── Historical DK points at this track (for display + th_data enrichment) ──
    dk_history = query_driver_dk_points_at_track(
        track_name, series_id, min_season=2022,
        before_date=race_date if not is_prerace else None,
    )
    dk_hist_names = list(dk_history.keys())

    # ── Current-season DK points at SIMILAR tracks (track type, excluding this one) ──
    # This captures who's in form right now at this type of racing. E.g. at
    # Kansas we pull in Vegas, Texas, Charlotte, Homestead, etc. from 2026.
    from src.data import query_driver_dk_points_by_track_type as _query_tt_dk
    similar_tt_dk = _query_tt_dk(
        track_type=track_type,
        series_id=series_id,
        season=season,
        before_date=race_date if not is_prerace else None,
        exclude_track=track_name,
    )
    similar_tt_names = list(similar_tt_dk.keys())

    # Build cross-series track history lookup
    cross_th_lookup = {}
    if not cross_th_df.empty:
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races", "Wins", "Top 5", "Top 10", "DNF"]:
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
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races", "Wins", "Top 5", "Top 10", "DNF", "Avg Rating"]:
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
                dk_hist_entry = dk_history.get(matched) or dk_history.get(d) or {}

                arp = row.get("Avg Run Pos") if pd.notna(row.get("Avg Run Pos", None)) else None
                af = row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20

                # Cross-series blending: only blend if cross-series is BETTER
                # A Cup mid-packer in a bad car may dominate trucks with good equipment,
                # so cross-series should only help (lower avg finish), never hurt.
                cross = cross_th_lookup.get(d)
                if cross and cross["races"] >= 2:
                    c_af = cross["avg_finish"]
                    c_arp = cross["avg_running_pos"]
                    if c_af < af:  # cross-series is better → blend at 30%
                        af = af * 0.70 + c_af * 0.30
                        if arp is not None and c_arp is not None and c_arp < arp:
                            arp = arp * 0.70 + c_arp * 0.30
                        elif c_arp is not None and c_arp < (arp or 99):
                            arp = c_arp

                th_data[d] = {
                    "avg_finish": af,
                    "avg_start": row.get("Avg Start", 20) if pd.notna(row.get("Avg Start")) else 20,
                    "avg_running_pos": arp,
                    "th_rating": row.get("Avg Rating") if pd.notna(row.get("Avg Rating", None)) else None,
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
    # Use percentile-based scaling: Cup P5 out of 36 (top 14%) → Truck P5.2 out of 37
    # This accounts for different field sizes and competition levels
    SERIES_FIELD_SIZE = {1: 36, 2: 38, 3: 36}  # typical field sizes
    for d in drivers:
        if d not in th_data and d in cross_th_lookup:
            cross = cross_th_lookup[d]
            if cross["races"] >= 2:
                # Percentile-scale: convert position to percentile in source series,
                # then map to position in current series field
                source_field = max(SERIES_FIELD_SIZE.get(cross_ids[0], 36) if cross_ids else 36, 1)
                c_af = cross["avg_finish"]
                c_arp = cross["avg_running_pos"]
                pct = c_af / source_field  # percentile (0=best, 1=worst)
                scaled_af = pct * field_size
                scaled_arp = (c_arp / source_field) * field_size if c_arp else None

                th_data[d] = {
                    "avg_finish": scaled_af,
                    "avg_start": 20,
                    "avg_running_pos": scaled_arp,
                    "laps_led": 0, "fastest_laps": 0,
                    "laps_led_per_race": 0, "fastest_laps_per_race": 0,
                    "races": cross["races"],
                    "avg_dk": None, "wins": 0, "top5": 0, "dnf": 0,
                    "_cross_series_only": True,  # flag for trust discount
                }

    # ── 2. Track Type Signal — DB-backed aggregation across similar tracks ──
    tt_data = {}
    if wn.get("track_type", 0) > 0:
        tt_result = _query_db_track_type_history(
            track_type, series_id,
            exclude_track=track_name,
            exclude_race_id=db_race_id,
            before_date=race_date if not is_prerace else None,
            cross_series_ids=cross_ids if cross_ids else None,
        )
        if cross_ids:
            tt_raw, tt_cross_raw = tt_result
        else:
            tt_raw = tt_result
            tt_cross_raw = {}

        # Blend cross-series track type data (only when it helps)
        if tt_cross_raw:
            source_field = max(SERIES_FIELD_SIZE.get(cross_ids[0], 36) if cross_ids else 36, 1)
            for name, cdata in tt_cross_raw.items():
                if name in tt_raw and cdata.get("races", 0) >= 2:
                    cur = tt_raw[name]
                    c_af = cdata["avg_finish"]
                    # Only blend if cross-series is better
                    if c_af < cur["avg_finish"]:
                        cur["avg_finish"] = cur["avg_finish"] * 0.70 + c_af * 0.30
                        c_arp = cdata.get("avg_running_pos")
                        if cur.get("avg_running_pos") and c_arp and c_arp < cur["avg_running_pos"]:
                            cur["avg_running_pos"] = cur["avg_running_pos"] * 0.70 + c_arp * 0.30
                elif name not in tt_raw and cdata.get("races", 0) >= 3:
                    # Cross-series only: percentile-scale to current field size
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

    # ── 2b. Team Quality Signal ─────────────────────────────────────────────
    team_signal = {}  # {driver: projected_finish from team quality}
    if wn.get("team", 0) > 0:
        team_stats = query_team_stats(
            series_id, track_type=track_type, min_season=2022,
            before_date=race_date if not is_prerace else None,
        )
        if team_stats:
            # Build driver→team mapping from entry list
            driver_team_map = {}
            if not entry_df.empty and "Team" in entry_df.columns:
                for _, row in entry_df.iterrows():
                    if pd.notna(row.get("Driver")) and pd.notna(row.get("Team")):
                        driver_team_map[row["Driver"]] = row["Team"]
            ts_names = list(team_stats.keys())
            for d in drivers:
                team_name = driver_team_map.get(d)
                if not team_name:
                    continue
                matched_team = team_name if team_name in team_stats else fuzzy_match_name(team_name, ts_names)
                if matched_team and matched_team in team_stats:
                    ts = team_stats[matched_team]
                    ts_arp = ts.get("avg_arp")
                    ts_af = ts["avg_finish"]
                    if ts_arp is not None:
                        team_finish = arp_finish_blend(ts_arp, ts_af, track_type)
                    else:
                        team_finish = ts_af
                    # Regress toward mid-field (team is a broad signal)
                    trust = min(1.0, ts["races"] / 10)
                    team_signal[d] = team_finish * trust + (field_size * 0.5) * (1 - trust)

    # ── 2c. Team-Adjusted Track History ──────────────────────────────────────
    # If a driver changed teams, adjust their historical avg_finish at this
    # track by the difference in team quality (e.g., moving to Hendrick from
    # a mid-pack team should lower their expected finish).
    team_adj_data = {}  # {driver: {"team_adj": float, ...}}
    if wn.get("team", 0) > 0:
        _driver_team_map = {}
        if not entry_df.empty and "Team" in entry_df.columns:
            for _, row in entry_df.iterrows():
                if pd.notna(row.get("Driver")) and pd.notna(row.get("Team")):
                    _driver_team_map[row["Driver"]] = row["Team"]
        if _driver_team_map:
            team_adj_data = compute_team_adjusted_track_history(
                track_name, series_id, _driver_team_map,
                before_date=race_date if not is_prerace else None,
                track_type=track_type,
            )

    # ── 2c-bis. ROOKIE TEAM-FALLBACK: no personal history at this track ->
    # inherit the team's record here as a soft prior (races=2 -> partial
    # trust). Backtest-validated: affected-driver finish MAE 7.36 -> 6.49.
    try:
        from src.data import apply_team_track_fallback
        _tf_map = {}
        if not entry_df.empty and "Team" in entry_df.columns:
            for _, _tr in entry_df.iterrows():
                if pd.notna(_tr.get("Driver")) and pd.notna(_tr.get("Team")):
                    _tf_map[_tr["Driver"]] = _tr["Team"]
        if _tf_map:
            th_data = apply_team_track_fallback(
                th_data, drivers, _tf_map, track_name, series_id,
                before_date=race_date if not is_prerace else None)
    except Exception:
        pass

    # ── 2d. Manufacturer Adjustment (track-type specific) ────────────────────
    mfr_adjustment = {}  # {driver: position adjustment (+/- 1.5 max)}
    mfr_stats = query_manufacturer_stats(
        series_id, track_type=track_type, min_season=2022,
        before_date=race_date if not is_prerace else None,
    )
    if mfr_stats:
        # Compute field-average finish across all manufacturers
        total_races = sum(m["races"] for m in mfr_stats.values())
        if total_races > 0:
            field_avg_finish = sum(m["avg_finish"] * m["races"] for m in mfr_stats.values()) / total_races
        else:
            field_avg_finish = field_size * 0.5

        # Build driver→manufacturer mapping
        driver_mfr_map = {}
        if not entry_df.empty and "Manufacturer" in entry_df.columns:
            for _, row in entry_df.iterrows():
                if pd.notna(row.get("Driver")) and pd.notna(row.get("Manufacturer")):
                    driver_mfr_map[row["Driver"]] = row["Manufacturer"]

        # Also try race_results for manufacturer info if entry list lacks it
        if not driver_mfr_map:
            try:
                conn = sqlite3.connect(PROJ_DB)
                mfr_rows = conn.execute('''
                    SELECT d.full_name, rr.manufacturer
                    FROM race_results rr
                    JOIN drivers d ON d.id = rr.driver_id
                    JOIN races r ON r.id = rr.race_id
                    WHERE r.series_id = ? AND rr.manufacturer IS NOT NULL AND rr.manufacturer != ''
                    GROUP BY d.id
                    ORDER BY r.season DESC, r.race_num DESC
                ''', (series_id,)).fetchall()
                conn.close()
                for name, mfr in mfr_rows:
                    if name not in driver_mfr_map:
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
                # Scale by 0.5, cap at +/- 1.5 positions
                adj = max(-1.5, min(1.5, delta * 0.5))
                mfr_adjustment[d] = adj

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
            # Convert implied probability to finish position using LOG scale.
            # Linear probability compresses midfield when a heavy favorite exists
            # (e.g. -100 = 53% leaves everyone else in 26-38 range).
            # Log scale preserves the favorite's advantage while differentiating
            # the rest of the field meaningfully.
            #
            # IMPORTANT — scale to a realistic EXPECTED-FINISH range, not the
            # full 1..field_size. Win odds do not imply the favorite finishes
            # 1st: even a strong favorite averages ~5th-8th over a season
            # because anyone can wreck. Mapping the favorite to 1.0 (a) is
            # unrealistic and (b) gave the odds signal ~3x the spread of the
            # track/track-type signals (which clamp to their natural ~8-20
            # range), so odds dominated the weighted average far beyond its
            # nominal weight.
            #
            # Bradley-Terry pairwise mapping (2026-07): the old linear
            # log-odds squash into [0.13n, 0.58n] could never say "worse
            # than ~22nd" AND couldn't separate a +5000 value play from a
            # +250000 no-hoper at any anchor setting (0.82 was tried and
            # buried them together at the wall). Strength ratios give a
            # sharp front, a genuinely deep tail, and intra-tail
            # discrimination. See odds_expected_finish for the evidence.
            from src.projections import odds_expected_finish
            _bt = odds_expected_finish(odds_probs, field_size)
            for name, ef in _bt.items():
                matched = fuzzy_match_name(name, drivers)
                if matched:
                    odds_finish[matched] = ef
        elif odds_probs:
            pct = len(odds_probs) / field_size * 100
            st.caption(f"⚠️ Odds cover only {len(odds_probs)}/{field_size} drivers "
                       f"({pct:.0f}%) — need ≥30% to use as signal. "
                       f"Odds weight redistributed to other signals.")

    # ── Build odds display lookup for the projections table ──────────────────
    # Preserve raw odds values — sportsbooks already post clean rounded numbers
    # and further bucketing corrupts the user's paste (e.g. +550 → +600).
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
                    driver_odds_display[matched] = {
                        "odds_str": oval,  # raw numeric for sorting
                        "impl_pct": round(impl, 1),
                    }
            except (ValueError, TypeError):
                continue

    # ── DNF risk data: prefer track-specific over career ──────────────────
    # Per-track DNF rate is a much better signal at superspeedways than a
    # career-wide average. A driver who wrecks 30% of the time at Talladega
    # but 5% overall should get the higher rate applied HERE at Talladega.
    # We blend: track-specific if >= 3 races at this track, else career.
    from src.data import query_driver_track_dnf
    _before_date = race_date if not is_prerace else None
    track_dnf = query_driver_track_dnf(track_name, series_id,
                                        before_date=_before_date, min_races=3)
    career_dnf = query_driver_career_dnf(series_id, before_date=_before_date)
    # Merge: track-specific takes priority when available
    dnf_data = dict(career_dnf)
    for drv, track_stats in track_dnf.items():
        # Blend: 70% track-specific, 30% career (smooth small samples)
        career_stats = career_dnf.get(drv, {})
        if career_stats:
            blended = {
                "dnf_rate": track_stats["dnf_rate"] * 0.7 + career_stats.get("dnf_rate", 0) * 0.3,
                "crash_rate": track_stats["crash_rate"] * 0.7 + career_stats.get("crash_rate", 0) * 0.3,
                "speed_score": career_stats.get("speed_score", 0),  # speed always from career
                "races": career_stats.get("races", 0),  # use career for trust threshold
            }
            dnf_data[drv] = blended
        else:
            dnf_data[drv] = track_stats

    # ── Rear-of-field starting penalties (manual) ─────────────────────────────
    # DK scores place differential off the QUALIFYING grid, but a driver sent to
    # the rear (unapproved adjustments / backup car / failed inspection) can't
    # lead early laps or run up front. There is NO reliable pre-race API source:
    # the drops happen at the green flag and only appear in NASCAR's race-morning
    # penalty report (the live-feed's starting_position field is unreliable
    # pre-race), so flag them manually. Flagged drivers feed a rear start into
    # dominator + fast-lap scoring; qualifying position still drives place
    # differential and the displayed start, as DK scores it.
    grid_start, rear_drivers = {}, set()
    if is_prerace:
        with st.expander("⬇ Starting at the rear (penalties)"):
            st.caption("Flag drivers moved to the back (unapproved adjustments, "
                       "backup car, failed inspection — from NASCAR's race-morning "
                       "report). They lose laps-led / fast-laps upside; DK still "
                       "scores their place differential from the qualifying grid.")
            picked = st.multiselect("Drivers starting at the rear", options=sorted(drivers),
                                    default=[], key=f"rear_{series_id}_{race_id}",
                                    label_visibility="collapsed")
        rear_drivers = set(picked)
        for d in rear_drivers:
            grid_start[d] = field_size   # line them up at the back

    # ── Run shared projection engine ──────────────────────────────────────────
    # Both the Projections tab and Accuracy tab call this same function to
    # guarantee identical math.  Data loading above is tab-specific, but the
    # signal processing, normalization, dominator scoring, finish assignment,
    # and DK point computation all live in compute_projections().
    from src.projections import compute_projections

    proj_rows, _proj_detail, driver_signal_details = compute_projections(
        drivers=drivers,
        field_size=field_size,
        wn=wn,
        th_data=th_data,
        tt_data=tt_data,
        qual_pos=qual_pos,
        practice_data=prac_rank,
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
        return_signal_details=True,
        grid_start=grid_start,
    )

    # ── Build display rows from projection results ──────────────────────────
    rows = []
    for pr in proj_rows:
        d = pr["driver"]

        # Historical DK points at this track (actual avg/best/worst)
        dk_hist = dk_history.get(d)
        if not dk_hist:
            matched_dk = fuzzy_match_name(d, dk_hist_names) if dk_hist_names else None
            dk_hist = dk_history.get(matched_dk) if matched_dk else None

        avg_dk_track = dk_hist["avg_dk"] if dk_hist else None
        best_dk_track = dk_hist["best_dk"] if dk_hist else None
        worst_dk_track = dk_hist["worst_dk"] if dk_hist else None

        # Current-season DK performance at similar tracks (track type)
        sim = similar_tt_dk.get(d)
        if not sim:
            matched_sim = fuzzy_match_name(d, similar_tt_names) if similar_tt_names else None
            sim = similar_tt_dk.get(matched_sim) if matched_sim else None
        sim_avg_dk = sim["avg_dk"] if sim else None
        sim_races = sim["races"] if sim else 0

        odds_info = driver_odds_display.get(d, {})
        odds_val = odds_info.get("odds_str", None)
        odds_numeric = odds_val if isinstance(odds_val, (int, float)) else None

        sig = driver_signal_details.get(d, {})
        rows.append({
            "Driver": d,
            "Proj DK": pr["proj_dk"],
            "Floor": pr.get("proj_floor"),
            "Ceiling": pr.get("proj_ceiling"),
            "Proj Finish": pr["proj_finish"],
            "Win Odds": odds_numeric,
            "Impl %": odds_info.get("impl_pct", None),
            "Finish Pts": round(pr["finish_pts"], 1),
            "Diff Pts": round(pr["diff_pts"], 1),
            "Led Pts": round(pr["led_pts"], 1),
            "FL Pts": round(pr["fl_pts"], 1),
            "Proj Laps Led": pr["laps_led"],
            "Proj Fast Laps": pr["fast_laps"],
            "Avg DK": avg_dk_track,
            "Best DK": best_dk_track,
            "Worst DK": worst_dk_track,
            "Avg DK (Similar)": sim_avg_dk,
            "Similar Races": sim_races,
            "Start": pr["start"],
            "Sig Odds": sig.get("Odds"),
            "Sig Track": sig.get("Track"),
            "Sig TType": sig.get("TType"),
            "Sig Qual": sig.get("Qual"),
            "Sig Team": sig.get("Team"),
            "Sig Prac": sig.get("Prac"),
            "Net Sig": sig.get("Net Sig"),
            "Mfr Adj": sig.get("Mfr"),
            "Team Adj": sig.get("Team Adj"),
        })

    proj = pd.DataFrame(rows)

    # ── Projected FanDuel points — derived from the SAME engine outputs ──
    # FD scoring: finish curve (43/40/38, -1/spot to 40th=1) + 0.5/pos diff
    # + 0.1/lap led + 0.1/lap completed. No fastest-lap points. The DK
    # diff_pts column is (start - finish) × 1.0, so FD diff = diff_pts × 0.5.
    #
    # Laps completed is RELIABILITY-WEIGHTED per driver, not flat: FanDuel
    # pays for every lap completed, so a DNF is a day-ender (lost finish
    # points AND lost lap points) and chronically laps-down equipment leaks
    # points weekly. query_expected_laps_fraction blends each driver's
    # recency-weighted history of (laps completed / winner laps) toward the
    # field median.
    from src.config import (FD_FINISH_POINTS, FD_PTS_LAPS_LED,
                            FD_PTS_LAPS_COMPLETED, FD_PTS_PLACE_DIFF)
    from src.data import query_expected_laps_fraction

    def _fd_finish_pts(pf):
        """Interpolate the FD finish curve at a fractional projected finish."""
        try:
            pf = float(pf)
        except (TypeError, ValueError):
            return 0.0
        lo = max(1, min(40, int(pf)))
        hi = max(1, min(40, lo + 1))
        frac = max(0.0, min(1.0, pf - lo))
        return FD_FINISH_POINTS.get(lo, 0) * (1 - frac) + FD_FINISH_POINTS.get(hi, 0) * frac

    _laps_rates = query_expected_laps_fraction(
        series_id, before_date=(None if is_prerace else race_date) or None)
    _rate_median = (_laps_rates.get("_field_median", {}) or {}).get("frac", 0.95)
    _rate_names = [k for k in _laps_rates if k != "_field_median"]

    def _driver_rate(d):
        v = _laps_rates.get(d)
        if v is None and _rate_names:
            m = fuzzy_match_name(d, _rate_names)
            v = _laps_rates.get(m) if m else None
        return v or {"frac": _rate_median, "races": 0, "dnf_rate": 0.0}

    _rl = race_laps or 0
    _rates_by_driver = {d: _driver_rate(d) for d in proj["Driver"]}
    proj["Proj Laps"] = proj["Driver"].map(
        lambda d: round(_rl * _rates_by_driver[d]["frac"], 1))
    proj["DNF Risk %"] = proj["Driver"].map(
        lambda d: round(_rates_by_driver[d]["dnf_rate"] * 100, 1))
    proj["Proj FD"] = proj.apply(
        lambda r: round(_fd_finish_pts(r["Proj Finish"])
                        + r["Diff Pts"] * FD_PTS_PLACE_DIFF
                        + r["Proj Laps Led"] * FD_PTS_LAPS_LED
                        + r["Proj Laps"] * FD_PTS_LAPS_COMPLETED, 1), axis=1)

    # FD floor / ceiling — DNF is modeled EXPLICITLY because on FanDuel it
    # zeroes the laps-completed stream too (DK's floor handles this inside
    # the engine; FD needs its own).
    #   Floor   = P(DNF) × (crash-out ~5 laps before half distance, finish
    #             near last) + (1-P) × (bad-but-running day: finish +7 spots,
    #             full laps)
    #   Ceiling = strong day: finish -6 spots, full distance, 1.5× laps led
    def _fd_floor_ceil(r):
        d = r["Driver"]
        rate = _rates_by_driver[d]
        p_dnf = min(0.35, rate["dnf_rate"])
        start = r.get("Start")
        try:
            start = float(start)
        except (TypeError, ValueError):
            start = r["Proj Finish"]
        dnf_finish = max(1.0, field_size - 2)
        dnf_laps = 0.45 * _rl
        fd_dnf = (_fd_finish_pts(dnf_finish)
                  + (start - dnf_finish) * FD_PTS_PLACE_DIFF
                  + dnf_laps * FD_PTS_LAPS_COMPLETED)
        bad_finish = min(field_size, r["Proj Finish"] + 7)
        fd_bad = (_fd_finish_pts(bad_finish)
                  + (start - bad_finish) * FD_PTS_PLACE_DIFF
                  + _rl * FD_PTS_LAPS_COMPLETED)
        floor = p_dnf * fd_dnf + (1 - p_dnf) * fd_bad
        good_finish = max(1.0, r["Proj Finish"] - 6)
        ceil = (_fd_finish_pts(good_finish)
                + (start - good_finish) * FD_PTS_PLACE_DIFF
                + r["Proj Laps Led"] * 1.5 * FD_PTS_LAPS_LED
                + _rl * FD_PTS_LAPS_COMPLETED)
        return pd.Series({"FD Floor": round(floor, 1),
                          "FD Ceiling": round(max(ceil, r["Proj FD"]), 1)})

    proj[["FD Floor", "FD Ceiling"]] = proj.apply(_fd_floor_ceil, axis=1)

    # PD Upside: categorize each driver by their projected place-differential
    # potential. This is display-only — it does not modify projections.
    # - High:     start >= P25 AND projected finish <= P20 (bounce-back)
    # - Low:      start <= P8  AND projected finish >= P18 (fade risk)
    # - Neutral:  within +/- 5 positions of start-finish delta
    def _pd_upside_tier(start, proj_finish):
        try:
            s = int(start)
            f = float(proj_finish)
        except (TypeError, ValueError):
            return ""
        delta = s - f  # positive = gain positions (good)
        if s >= 25 and f <= 20:
            return "High"
        if s <= 8 and f >= 18:
            return "Low"
        if abs(delta) <= 5:
            return "Neutral"
        if delta > 5:
            return "Mild+"
        return "Mild-"
    proj["PD Upside"] = proj.apply(
        lambda r: _pd_upside_tier(r.get("Start"), r.get("Proj Finish")), axis=1
    )

    # Merge car number if available — use fuzzy_merge for name-variation
    # safety even though proj and base_df should share driver spellings
    # (both derived from the entry list).
    if "Car" in base_df.columns:
        proj = fuzzy_merge(proj, base_df[["Driver", "Car"]].drop_duplicates("Driver"),
                           on="Driver", how="left")

    # Merge salaries (both platforms — display filtering happens later)
    if not dk_df.empty:
        proj = fuzzy_merge(proj, dk_df, on="Driver", how="left",
                           right_cols=["DK Salary"])
        proj["Value"] = np.where(
            proj["DK Salary"].notna() & (proj["DK Salary"] > 0),
            (proj["Proj DK"] / (proj["DK Salary"] / 1000)).round(2),
            np.nan
        )
    if not fd_df.empty:
        proj = fuzzy_merge(proj, fd_df, on="Driver", how="left",
                           right_cols=["FD Salary"])
        proj["FD Value"] = np.where(
            proj["FD Salary"].notna() & (proj["FD Salary"] > 0),
            (proj["Proj FD"] / (proj["FD Salary"] / 1000)).round(2),
            np.nan
        )

    _rank_col = "Proj FD" if platform == "FanDuel" else "Proj DK"
    proj = proj.sort_values(_rank_col, ascending=False).reset_index(drop=True)
    proj.index = proj.index + 1
    proj.index.name = "Rank"

    # ── Projected ownership (heuristic) ──
    try:
        from src.ownership import project_ownership, compute_leverage
        _proj_dk_dict = dict(zip(proj["Driver"], proj["Proj DK"]))
        _sal_dict = dict(zip(proj["Driver"], proj["DK Salary"])) if "DK Salary" in proj.columns else {}
        _proj_finish_dict = dict(zip(proj["Driver"], proj["Proj Finish"])) if "Proj Finish" in proj.columns else {}
        # win_odds: odds_data is keyed by driver display name already
        _own_kwargs = dict(
            drivers=proj["Driver"].tolist(),
            proj_dk=_proj_dk_dict,
            salary=_sal_dict,
            win_odds=odds_data,
            qual_pos=qual_pos,
            proj_finish=_proj_finish_dict,
            track_type=track_type,
            field_size=field_size,
            roster_size=6,  # DK NASCAR contests are 6-driver across all series
        )
        # Cash ownership runs hotter on chalk than GPP (cash converges on the
        # safe plays; GPP spreads for leverage). Project both.
        _own_gpp = project_ownership(contest_type="gpp", **_own_kwargs)
        _own_cash = project_ownership(contest_type="cash", **_own_kwargs)
        proj["GPP Own%"] = proj["Driver"].map(_own_gpp).round(1)
        proj["Cash Own%"] = proj["Driver"].map(_own_cash).round(1)
        # Leverage = points per ownership point (a GPP concept — use GPP ownership)
        _own_map = _own_gpp
        _lev_map = compute_leverage(_proj_dk_dict, _own_gpp)
        proj["Leverage"] = proj["Driver"].map(_lev_map).round(2)

        # ── FanDuel ownership — its own model, not a copy of DK's ──
        # Different inputs change the chalk: FD salaries/points (5-man,
        # different value curve) and a DNF-risk fade (FD players avoid
        # blow-up risk because a DNF zeroes the laps-completed stream;
        # cash fades hard, GPP partially chases the discount).
        if "FD Salary" in proj.columns and proj["FD Salary"].notna().any():
            _fd_sal_dict = {d: s for d, s in zip(proj["Driver"], proj["FD Salary"])
                            if pd.notna(s)}
            _proj_fd_dict = dict(zip(proj["Driver"], proj["Proj FD"]))
            _dnf_dict = {d: _rates_by_driver[d]["dnf_rate"]
                         for d in proj["Driver"]}
            _own_kwargs_fd = dict(
                drivers=[d for d in proj["Driver"] if d in _fd_sal_dict],
                proj_dk=_proj_fd_dict,
                salary=_fd_sal_dict,
                win_odds=odds_data,
                qual_pos=qual_pos,
                proj_finish=_proj_finish_dict,
                track_type=track_type,
                field_size=field_size,
                roster_size=5,   # FanDuel NASCAR rosters 5 drivers
                dnf_risk=_dnf_dict,
            )
            _own_gpp_fd = project_ownership(contest_type="gpp", **_own_kwargs_fd)
            _own_cash_fd = project_ownership(contest_type="cash", **_own_kwargs_fd)
            _lev_fd = compute_leverage(_proj_fd_dict, _own_gpp_fd)
            proj["FD GPP Own%"] = proj["Driver"].map(_own_gpp_fd).round(1)
            proj["FD Cash Own%"] = proj["Driver"].map(_own_cash_fd).round(1)
            proj["FD Leverage"] = proj["Driver"].map(_lev_fd).round(2)
            st.session_state["proj_own_map_fd"] = _own_gpp_fd
            st.session_state["proj_leverage_map_fd"] = _lev_fd
    except Exception as _ow_err:
        # Non-fatal — ownership is additive info only
        pass

    # FD floor/ceiling for the optimizer's FanDuel Cash/GPP modes
    st.session_state["proj_floor_map_fd"] = dict(zip(proj["Driver"], proj["FD Floor"]))
    st.session_state["proj_ceiling_map_fd"] = dict(zip(proj["Driver"], proj["FD Ceiling"]))

    # Share Proj DK / Proj FD with optimizer tab via session state.
    # proj_maps_key stamps WHICH race these maps belong to — without it the
    # optimizer fuzzy-matched a different race's (even a different SERIES')
    # stale map onto the current field: crossover names like Custer/Hill/
    # Zilisch matched the Cup map while everyone else projected 0.
    st.session_state["proj_maps_key"] = f"{series_id}_{race_id}"
    st.session_state["proj_dk_map"] = dict(zip(proj["Driver"], proj["Proj DK"]))
    st.session_state["proj_fd_map"] = dict(zip(proj["Driver"], proj["Proj FD"]))
    # Share per-driver detail so optimizer can identify projected dominators
    st.session_state["proj_detail_map"] = _proj_detail
    # Floor / ceiling DK for cash vs GPP optimization
    st.session_state["proj_floor_map"] = {d: v.get("proj_floor")
                                          for d, v in _proj_detail.items()}
    st.session_state["proj_ceiling_map"] = {d: v.get("proj_ceiling")
                                            for d, v in _proj_detail.items()}
    # Share ownership + leverage for optimizer leverage scoring
    if "GPP Own%" in proj.columns:
        # Optimizer leverage uses GPP ownership (leverage is a GPP concept).
        st.session_state["proj_own_map"] = dict(zip(proj["Driver"], proj["GPP Own%"]))
    if "Leverage" in proj.columns:
        st.session_state["proj_leverage_map"] = dict(zip(proj["Driver"], proj["Leverage"]))

    # Rename Start column — "Qual Pos" if qualifying happened, "Proj Qual Pos" if not
    if "Start" in proj.columns:
        has_qual = bool(qual_pos)
        proj = proj.rename(columns={"Start": "Qual Pos" if has_qual else "Proj Qual Pos"})

    # Display columns
    display_cols = ["Driver"]
    if "Car" in proj.columns:
        display_cols.append("Car")
    # Label odds column — "Win Odds" for real odds, "Est. Odds" for salary-estimated
    odds_col_name = "Win Odds"
    if st.session_state.get("odds_source") == "salary_estimate":
        proj = proj.rename(columns={"Win Odds": "Est. Odds", "Impl %": "Est. Impl %"})
        odds_col_name = "Est. Odds"
    if odds_col_name in proj.columns and proj[odds_col_name].notna().any():
        display_cols.append(odds_col_name)
    impl_col = "Est. Impl %" if odds_col_name == "Est. Odds" else "Impl %"
    if impl_col in proj.columns and proj[impl_col].notna().any():
        display_cols.append(impl_col)
    # Platform-specific salary / points / value columns
    _show_dk = platform in ("DraftKings", "Both")
    _show_fd = platform in ("FanDuel", "Both")
    if _show_dk and "DK Salary" in proj.columns:
        display_cols.append("DK Salary")
    if _show_fd and "FD Salary" in proj.columns:
        display_cols.append("FD Salary")
    if _show_dk:
        display_cols.append("Proj DK")
    if _show_fd and "Proj FD" in proj.columns:
        display_cols.append("Proj FD")
    if _show_dk and "Floor" in proj.columns:
        display_cols.append("Floor")
    if _show_dk and "Ceiling" in proj.columns:
        display_cols.append("Ceiling")
    if _show_fd and "FD Floor" in proj.columns:
        display_cols.append("FD Floor")
    if _show_fd and "FD Ceiling" in proj.columns:
        display_cols.append("FD Ceiling")
    if _show_dk and "Value" in proj.columns:
        display_cols.append("Value")
    if _show_fd and "FD Value" in proj.columns:
        display_cols.append("FD Value")
    # Reliability columns — DNF risk matters on both sites; Proj Laps is the
    # FanDuel laps-completed stream made visible.
    if "DNF Risk %" in proj.columns:
        display_cols.append("DNF Risk %")
    if _show_fd and "Proj Laps" in proj.columns:
        display_cols.append("Proj Laps")
    # Ownership: DK-model columns in DK/Both mode, FD-model columns when
    # FanDuel salaries exist and FD is shown.
    if _show_dk:
        for c in ("GPP Own%", "Cash Own%", "Leverage"):
            if c in proj.columns:
                display_cols.append(c)
    if _show_fd:
        for c in ("FD GPP Own%", "FD Cash Own%", "FD Leverage"):
            if c in proj.columns:
                display_cols.append(c)
    qual_col = "Qual Pos" if "Qual Pos" in proj.columns else "Proj Qual Pos"
    if qual_col in proj.columns:
        display_cols.append(qual_col)
    display_cols.extend(["Proj Finish", "PD Upside"])
    # The engine's point-component breakdown is DK-SCALED (Finish Pts uses the
    # DK curve, Led Pts = 0.25/lap, FL Pts = 0.45/lap) — hide it in
    # FanDuel-only mode so the table doesn't mix scoring systems. Same for
    # the historical DK-points columns.
    if _show_dk:
        display_cols.extend(["Finish Pts", "Diff Pts", "Led Pts", "FL Pts"])
    display_cols.extend(["Proj Laps Led", "Proj Fast Laps"])
    if _show_dk:
        display_cols.extend(["Avg DK", "Best DK", "Worst DK",
                             "Avg DK (Similar)", "Similar Races"])
    # Signal detail columns at the end — practice before qualifying
    # Net Sig is the weighted average of all normalized signals — the value
    # the engine actually rank-orders on to assign Proj Finish. Useful for
    # verifying that the aggregation reflects the underlying signals.
    display_cols.extend(["Sig Odds", "Sig Track", "Sig TType",
                         "Sig Prac", "Sig Qual", "Sig Team",
                         "Net Sig", "Team Adj", "Mfr Adj"])
    avail = [c for c in display_cols if c in proj.columns]

    # Auto-save pre-race projections for the Accuracy tab's historical
    # tracking. Only runs for upcoming races — for completed races, the
    # engine already uses before_date filters so the display is already
    # snapshot-correct without needing a DB row.
    if is_prerace and race_id:
        _auto_save_key = f"proj_autosaved_{race_id}_{series_id}"
        if _auto_save_key not in st.session_state:
            try:
                from tabs.tab_accuracy import save_projections_to_db
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

    from src.components import (build_projection_column_config,
                                interactive_drill_down_dataframe, apply_car_badges,
                                CAR_BADGE_ROW_HEIGHT)
    disp = proj[avail].copy()
    col_config = build_projection_column_config(disp)
    # Show the styled car-number badge art instead of the plain text number.
    disp, _badge_cfg = apply_car_badges(disp, series_id)
    _row_h = {}
    if _badge_cfg is not None:
        col_config["Car"] = _badge_cfg
        _row_h["row_height"] = CAR_BADGE_ROW_HEIGHT
    cap = "Click any driver row for race-by-race history at this track"
    if rear_drivers:
        cap += "  ·  🟠 amber = starting at the rear (dominator upside removed, PD still off qualifying)"
    st.caption(cap)
    _show = safe_fillna(disp)
    if rear_drivers and "Driver" in _show.columns:
        # Amber the names of drivers sent to the rear (text only — leaves the
        # Driver value intact so the click-to-drill popup still resolves).
        def _hl_rear(col):
            if col.name != "Driver":
                return ["" for _ in col]
            return ["color:#f59e0b;font-weight:700" if str(v) in rear_drivers else ""
                    for v in col]
        _show = _show.style.apply(_hl_rear, axis=0)
    interactive_drill_down_dataframe(
        _show,
        key=f"proj_main_{series_id}_{race_id}",
        series_id=series_id, track_name=track_name,
        width="stretch", hide_index=False, height=550,
        column_config=col_config, **_row_h,
    )

    # Chart — all drivers, stacked bar with component breakdown on hover.
    # Follows the platform picker: FanDuel mode rebuilds the components in
    # FD scoring (FD finish curve, 0.5x diff, 0.1x led, 0.1x laps completed)
    # instead of showing DK-scale bars under an FD sort.
    import plotly.graph_objects as go
    _chart_fd = (platform == "FanDuel")
    _tag = "FD" if _chart_fd else "DK"
    chart_df = proj.copy()
    if _chart_fd:
        chart_df["Finish Pts"] = chart_df["Proj Finish"].map(_fd_finish_pts).round(1)
        chart_df["Diff Pts"] = (chart_df["Diff Pts"] * FD_PTS_PLACE_DIFF).round(1)
        chart_df["Led Pts"] = (chart_df["Proj Laps Led"] * FD_PTS_LAPS_LED).round(1)
        chart_df["Laps Pts"] = (chart_df["Proj Laps"] * FD_PTS_LAPS_COMPLETED).round(1)
    _rank_pts = "Proj FD" if _chart_fd else "Proj DK"
    chart_df = chart_df.sort_values(_rank_pts, ascending=True)  # horizontal bar: bottom = best

    # Build stacked horizontal bar with scoring components
    # Positive components stack right, negative Diff Pts stacks left
    if _chart_fd:
        pos_components = {
            "Finish Pts": "#0ea5e9",
            "Led Pts": "#fb923c",
            "Laps Pts": "#f472b6",
        }
    else:
        pos_components = {
            "Finish Pts": "#0ea5e9",
            "Led Pts": "#fb923c",
            "FL Pts": "#f472b6",
        }

    from src.utils import short_name_series
    chart_df["Short"] = short_name_series(chart_df["Driver"].tolist())

    fig = go.Figure()

    # Positive Diff Pts (gained positions)
    if "Diff Pts" in chart_df.columns:
        fig.add_trace(go.Bar(
            y=chart_df["Short"],
            x=chart_df["Diff Pts"].clip(lower=0),
            name="Diff Pts (+)",
            orientation="h",
            marker_color="#4ade80",
            hovertemplate="%{y}<br>Diff Pts: +%{x:.1f}<extra></extra>",
        ))

    for comp, color in pos_components.items():
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

    # Negative Diff Pts (lost positions) — separate trace going left
    if "Diff Pts" in chart_df.columns and (chart_df["Diff Pts"] < 0).any():
        fig.add_trace(go.Bar(
            y=chart_df["Short"],
            x=chart_df["Diff Pts"].clip(upper=0),
            name="Diff Pts (-)",
            orientation="h",
            marker_color="#ef4444",
            hovertemplate="%{y}<br>Diff Pts: %{x:.1f}<extra></extra>",
        ))

    # Custom hover with all components
    custom_hover = []
    for _, row in chart_df.iterrows():
        if _chart_fd:
            _last_lines = (
                f"Laps Pts: {row.get('Laps Pts', 0):.1f}<br>"
                f"Laps Led: {row.get('Proj Laps Led', 0):.0f}<br>"
                f"Proj Laps: {row.get('Proj Laps', 0):.0f}<br>"
            )
        else:
            _last_lines = (
                f"FL Pts: {row.get('FL Pts', 0):.1f}<br>"
                f"Laps Led: {row.get('Proj Laps Led', 0):.0f}<br>"
                f"Fast Laps: {row.get('Proj Fast Laps', 0):.0f}<br>"
            )
        text = (
            f"<b>{row['Driver']}</b><br>"
            f"Proj {_tag}: {row.get(_rank_pts, 0):.1f}<br>"
            f"Proj Finish: {row.get('Proj Finish', 0):.1f}<br>"
            f"Finish Pts: {row.get('Finish Pts', 0):.1f}<br>"
            f"Diff Pts: {row.get('Diff Pts', 0):+.1f}<br>"
            f"Led Pts: {row.get('Led Pts', 0):.1f}<br>"
            f"{_last_lines}"
            f"Track: {pd.to_numeric(row.get('Track', 0), errors='coerce') or 0:.1f}"
        )
        custom_hover.append(text)

    # Add invisible scatter trace for rich hover
    fig.add_trace(go.Scatter(
        y=chart_df["Short"],
        x=chart_df[_rank_pts],
        mode="markers",
        marker=dict(size=1, opacity=0),
        hovertext=custom_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    from src.charts import DARK_LAYOUT, apply_dark_theme
    n_drivers = len(chart_df)
    fig.update_layout(
        **DARK_LAYOUT,
        barmode="relative",
        title=f"All Drivers — Projected {_tag} Points Breakdown",
        xaxis_title=f"Projected {_tag} Points",
        yaxis_title="",
        height=max(400, n_drivers * 22),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickfont=dict(size=10)),
    )
    apply_dark_theme(fig)
    st.plotly_chart(fig, width="stretch")

    # ── Floor → Ceiling range (cash vs GPP volatility at a glance) ──
    from src.charts import floor_ceiling_range, ownership_leverage_scatter
    if _chart_fd:
        fc_fig = floor_ceiling_range(proj, pts_col="Proj FD",
                                     floor_col="FD Floor", ceil_col="FD Ceiling")
    else:
        fc_fig = floor_ceiling_range(proj, pts_col="Proj DK",
                                     floor_col="Floor", ceil_col="Ceiling")
    if fc_fig:
        st.divider()
        st.plotly_chart(fc_fig, width="stretch", key="proj_fc_range")

    # ── GPP leverage map (ownership vs points) ──
    if _chart_fd and "FD GPP Own%" in proj.columns:
        lev_fig = ownership_leverage_scatter(proj, pts_col="Proj FD",
                                             own_col="FD GPP Own%", series_id=series_id)
    else:
        lev_fig = ownership_leverage_scatter(proj, pts_col="Proj DK",
                                             own_col="GPP Own%", series_id=series_id)
    if lev_fig:
        st.divider()
        st.plotly_chart(lev_fig, width="stretch", key="proj_leverage_map")
