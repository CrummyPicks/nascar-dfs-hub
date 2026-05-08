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
    TRACK_TYPE_PARENT, DK_FINISH_POINTS, TRACK_TYPE_WEIGHT_DEFAULTS,
    CROSS_SERIES_HIERARCHY,
)
from src.data import (
    scrape_track_history, query_driver_dk_points_at_track,
    query_driver_career_dnf, query_team_stats, query_manufacturer_stats,
    compute_team_adjusted_track_history,
)
# projection_bar no longer used — replaced with inline stacked bar
from src.utils import safe_fillna, format_display_df, calc_dk_points, fuzzy_match_name, fuzzy_merge, arp_finish_blend

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


def _get_track_dominator_calibration(track_name: str, track_type: str,
                                      series_id: int = None) -> dict:
    """Pull historical domination stats from DB for this track.

    Returns dict with:
        avg_top_leader: average laps led by the race leader per race
        max_laps_led: highest single-race laps led at this track
        max_fastest_laps: highest single-race fastest laps at this track
        avg_n_leaders: average number of drivers who lead laps per race
        avg_n_fl_leaders: average number of drivers who get fastest laps per race
        concentration: exponent for score distribution

    Calibration source priority:
        1. Per-track history (if races >= MIN_RACES_TRACK_CAL at this track)
        2. Track-type history for the SAME series (e.g. for Watkins Glen
           Truck without enough race history, query all Truck road races)
        3. Hardcoded track-type defaults (Cup-scaled fallback)

    Without #2, sparse per-track Truck/O'Reilly data fell through to the
    Cup-scaled defaults — producing nonsense like "avg leader leads 80 laps"
    for a 72-lap Truck road race.
    """
    MIN_RACES_TRACK_CAL = 3  # below this, per-track numbers are too noisy

    # Pull track-type fallbacks for dominator-leader counts. These align with
    # FALLBACK_FL in _allocate_fastest_laps and are empirically calibrated
    # from Cup 2022+ data across track types.
    FALLBACK_FL_CAL = {"superspeedway": 30, "road": 12, "intermediate": 15,
                       "intermediate_worn": 18, "short": 18, "short_concrete": 18}
    FALLBACK_N_LEADERS = {"superspeedway": 12, "road": 4, "intermediate": 6,
                          "intermediate_worn": 7, "short": 6, "short_concrete": 5}

    type_defaults = TRACK_TYPE_DOM_DEFAULTS.get(track_type, {"max_ll": 150, "max_fl": 60})
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    # avg_top_leader hardcoded default scales by track type so a missing
    # series-specific fallback at least uses a number tied to the track type
    # rather than the same 80 (Cup intermediate-ish) for everything.
    AVG_TOP_LEADER_DEFAULT = {
        "superspeedway": 35, "road": 25, "intermediate": 80,
        "intermediate_worn": 80, "short": 100, "short_concrete": 110,
    }
    defaults = {
        "avg_top_leader": AVG_TOP_LEADER_DEFAULT.get(track_type,
                            AVG_TOP_LEADER_DEFAULT.get(parent, 60)),
        "max_laps_led": type_defaults["max_ll"],
        "max_fastest_laps": type_defaults["max_fl"],
        "avg_n_leaders": FALLBACK_N_LEADERS.get(track_type,
                                                  FALLBACK_N_LEADERS.get(parent, 6)),
        "avg_n_fl_leaders": FALLBACK_FL_CAL.get(track_type,
                                                  FALLBACK_FL_CAL.get(parent, 15)),
        "concentration": TRACK_TYPE_CONCENTRATION.get(track_type, 1.5),
    }
    if not os.path.exists(PROJ_DB):
        return defaults

    # Resolve the set of tracks to consider for the track-type fallback —
    # use the same family-folding logic as query_team_stats so e.g. road
    # courses pull in road, short_concrete pulls in short, etc.
    type_family = {track_type, parent}
    for tt, p in TRACK_TYPE_PARENT.items():
        if tt == track_type or p == parent:
            type_family.add(tt)
    if track_type == "short_concrete":
        type_family.add("short")
    family_tracks = [t for t, tt in TRACK_TYPE_MAP.items() if tt in type_family]

    def _summarize(rows):
        """Reduce a list of (top_led, n_leaders, top_fl, n_fl) per-race rows
        to a calibration-style summary."""
        if not rows:
            return None
        top_leaders = [r[0] for r in rows if r[0] and r[0] > 0]
        n_leaders_list = [r[1] for r in rows if r[1] and r[1] > 0]
        top_fl = [r[2] for r in rows if r[2] and r[2] > 0]
        n_fl_list = [r[3] for r in rows if r[3] and r[3] > 0]
        out = {}
        if top_leaders:
            out["avg_top_leader"] = float(np.mean(top_leaders))
            out["max_laps_led"] = int(max(top_leaders))
        if n_leaders_list:
            out["avg_n_leaders"] = float(np.mean(n_leaders_list))
        if top_fl:
            out["max_fastest_laps"] = int(max(top_fl))
        if n_fl_list:
            out["avg_n_fl_leaders"] = float(np.mean(n_fl_list))
        return out

    try:
        conn = sqlite3.connect(PROJ_DB)

        # Step 1: per-track history
        series_filter = "AND r.series_id = ?" if series_id else ""
        params = [f"%{track_name}%"]
        if series_id:
            params.append(series_id)
        per_track_rows = conn.execute(f'''
            SELECT MAX(rr.laps_led) as top_led,
                   COUNT(CASE WHEN rr.laps_led > 0 THEN 1 END) as n_leaders,
                   MAX(rr.fastest_laps) as top_fl,
                   COUNT(CASE WHEN rr.fastest_laps > 0 THEN 1 END) as n_fl
            FROM race_results rr
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE t.name LIKE ?
            {series_filter}
            GROUP BY r.id
        ''', params).fetchall()

        result = dict(defaults)
        per_track_summary = _summarize(per_track_rows) if per_track_rows else None

        # Step 2: track-type history for the same series (used as fallback when
        # per-track is too sparse, AND as the source for any fields the per-track
        # query couldn't fill in)
        type_summary = None
        if family_tracks and series_id:
            placeholders = ",".join("?" for _ in family_tracks)
            type_rows = conn.execute(f'''
                SELECT MAX(rr.laps_led) as top_led,
                       COUNT(CASE WHEN rr.laps_led > 0 THEN 1 END) as n_leaders,
                       MAX(rr.fastest_laps) as top_fl,
                       COUNT(CASE WHEN rr.fastest_laps > 0 THEN 1 END) as n_fl
                FROM race_results rr
                JOIN races r ON r.id = rr.race_id
                JOIN tracks t ON t.id = r.track_id
                WHERE t.name IN ({placeholders})
                  AND r.series_id = ?
                GROUP BY r.id
            ''', list(family_tracks) + [series_id]).fetchall()
            type_summary = _summarize(type_rows)

        conn.close()

        # Use per-track when it has enough races; otherwise fall through to
        # track-type-for-series; otherwise stay on the (track-type-scaled) defaults.
        n_per_track = len(per_track_rows) if per_track_rows else 0
        use_per_track = (n_per_track >= MIN_RACES_TRACK_CAL) and per_track_summary

        if use_per_track:
            result.update(per_track_summary)
        elif type_summary:
            result.update(type_summary)
        # else: result stays on defaults

        return result

    except Exception:
        pass

    return defaults


def _allocate_laps_led(driver_scores: dict, race_laps: int, track_name: str,
                        track_type: str, calibration: dict = None,
                        odds_display: dict = None) -> dict:
    """Allocate projected laps led across the field.

    Uses historical data to determine:
    - How many drivers lead laps (avg_n_leaders from DB)
    - Per-driver cap (avg_top_leader, NOT the extreme max)
    - Power curve from track-type concentration
    - Odds-gap boost: heavy favorites get higher concentration
    """
    if not driver_scores or race_laps <= 0:
        return {}

    cal = calibration or {}
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)

    # Number of leaders: use historical average, fall back to track-type defaults
    FALLBACK_LEADERS = {"superspeedway": 15, "road": 8, "intermediate": 8,
                        "intermediate_worn": 7, "short": 7, "short_concrete": 6}
    n_leaders = int(cal.get("avg_n_leaders",
                            FALLBACK_LEADERS.get(track_type, FALLBACK_LEADERS.get(parent, 8))))
    n_leaders = max(4, min(n_leaders, len(driver_scores)))

    # Rank drivers by raw score, only top N get any laps led
    sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
    top_drivers = dict(sorted_drivers[:n_leaders])

    # Power curve from track-type concentration (lower = more spread)
    exponent = cal.get("concentration", TRACK_TYPE_CONCENTRATION.get(track_type, 1.5))

    # Odds-gap boost: when the favorite has a massive implied probability lead,
    # increase concentration so they get a proportionally larger share of laps.
    # e.g., -115 (53.5%) vs +500 (16.7%) = 3.2x ratio → boost exponent
    if odds_display:
        impl_pcts = [v.get("impl_pct", 0) for v in odds_display.values() if v.get("impl_pct")]
        if len(impl_pcts) >= 2:
            impl_pcts_sorted = sorted(impl_pcts, reverse=True)
            top_impl = impl_pcts_sorted[0]
            second_impl = impl_pcts_sorted[1]
            if second_impl > 0:
                odds_ratio = top_impl / second_impl  # how dominant the favorite is
                # ratio 2.0+ = strong favorite, 3.0+ = heavy favorite
                if odds_ratio >= 2.0:
                    boost = min(0.8, (odds_ratio - 2.0) * 0.4)  # +0.4 per ratio point, cap +0.8
                    exponent = exponent + boost

    # Clamp to reasonable range
    exponent = max(1.0, min(exponent, 3.5))

    scores = {d: max(0.01, s) for d, s in top_drivers.items()}
    powered = {d: s ** exponent for d, s in scores.items()}
    total = sum(powered.values())
    if total <= 0:
        return {}

    # Every lap has a leader — distribute ALL race laps
    result = {d: (s / total) * race_laps for d, s in powered.items()}

    # Per-driver cap: use average top leader with generous headroom
    # A dominant favorite (-100) routinely exceeds the average leader's laps
    avg_top = cal.get("avg_top_leader", race_laps * 0.40)
    max_laps = min(race_laps * 0.75, avg_top * 1.40)  # 40% headroom over average

    # Iteratively cap and redistribute until all laps are allocated
    for _ in range(10):
        deficit = race_laps - sum(result.values())
        if abs(deficit) < 0.5:
            break
        for d in result:
            if result[d] > max_laps:
                result[d] = max_laps
        deficit = race_laps - sum(result.values())
        if deficit > 0.5:
            uncapped = {d: s for d, s in result.items() if s < max_laps}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for d in uncapped:
                    result[d] += deficit * (uncapped[d] / uncapped_total)

    return result


def _allocate_fastest_laps(driver_fl_scores: dict, race_laps: int,
                            track_type: str, calibration: dict = None,
                            odds_display: dict = None) -> dict:
    """Allocate projected fastest laps across the field (zero-sum).

    Uses historical data for number of drivers and per-driver cap.
    Fastest laps are more distributed than laps led, but still
    concentrate toward the front-runners.
    """
    if not driver_fl_scores or race_laps <= 0:
        return {}

    cal = calibration or {}
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)

    # Number of drivers with fastest laps — calibrated from historical Cup
    # data (2022+). Captures ~90-97% of real FL distribution at each
    # track type while avoiding over-dilution to drivers who realistically
    # get <2% of FL points.
    #
    # Superspeedway is the biggest outlier: the draft spreads FL across
    # 30+ drivers per race. Capping at 20 (old behavior) missed 18%+ of
    # the real distribution, which significantly under-projected FL points
    # for mid-pack speedway drivers.
    #
    # Historical Top-N share by track type (Cup 2022+):
    #    superspeedway:    Top 20 = 82%, Top 30 = 97%  → use 30
    #    road:             Top 10 = 82%, Top 15 = 92%  → use 12
    #    intermediate:     Top 10 = 80%, Top 15 = 91%  → use 15
    #    intermediate_worn: Top 15 = 86%, Top 20 = 93% → use 18
    #    short:            Top 15 = 87%, Top 20 = 93%  → use 18
    #    short_concrete:   Top 15 = 85%, Top 20 = 92%  → use 18
    FALLBACK_FL = {"superspeedway": 30, "road": 12, "intermediate": 15,
                   "intermediate_worn": 18, "short": 18, "short_concrete": 18}
    n_with_fl = int(cal.get("avg_n_fl_leaders",
                            FALLBACK_FL.get(track_type, FALLBACK_FL.get(parent, 15))))
    # Cap to track-type-appropriate ceiling (35 = max observed across all
    # track types) and the actual field size. The old hardcoded 20 cap was
    # the bug that limited Talladega to 20 even when calibration said 35+.
    max_fl_drivers = 35 if parent == "superspeedway" else 25
    n_with_fl = max(5, min(n_with_fl, max_fl_drivers, len(driver_fl_scores)))

    # Rank drivers by raw FL score, only top N get any fastest laps
    sorted_drivers = sorted(driver_fl_scores.items(), key=lambda x: x[1], reverse=True)
    top_drivers = dict(sorted_drivers[:n_with_fl])

    # Fastest laps concentration exponent — empirically tuned per track
    # type to reproduce historical Top-N FL share (Cup 2022+). Higher =
    # more concentrated on leaders, lower = more evenly spread.
    #
    # Historical targets (Top5 / Top10 / Top15 / Top20 share of total FL):
    #   superspeedway:    36% / 56% / 71% / 82%   -> exp 1.3
    #   road:             62% / 82% / 92% / 98%   -> exp 2.2
    #   intermediate:     57% / 80% / 91% / 96%   -> exp 2.4
    #   intermediate_worn: 55% / 75% / 86% / 93%  -> exp 2.2
    #   short:            54% / 76% / 87% / 93%   -> exp 2.2
    #   short_concrete:   52% / 73% / 85% / 92%   -> exp 2.0
    FL_EXPONENT_MAP = {
        "superspeedway":     1.3,
        "road":              2.2,
        "intermediate":      2.4,
        "intermediate_worn": 2.2,
        "short":             2.2,
        "short_concrete":    2.0,
    }
    fl_exponent = FL_EXPONENT_MAP.get(track_type,
                                        FL_EXPONENT_MAP.get(parent, 2.0))

    # Odds-gap boost (same logic as laps led)
    if odds_display:
        impl_pcts = [v.get("impl_pct", 0) for v in odds_display.values() if v.get("impl_pct")]
        if len(impl_pcts) >= 2:
            impl_pcts_sorted = sorted(impl_pcts, reverse=True)
            top_impl = impl_pcts_sorted[0]
            second_impl = impl_pcts_sorted[1]
            if second_impl > 0:
                odds_ratio = top_impl / second_impl
                if odds_ratio >= 2.0:
                    boost = min(0.5, (odds_ratio - 2.0) * 0.25)  # smaller boost than LL
                    fl_exponent = fl_exponent + boost

    fl_exponent = max(1.2, min(fl_exponent, 2.5))

    scores = {d: max(0.01, s) for d, s in top_drivers.items()}
    powered = {d: s ** fl_exponent for d, s in scores.items()}
    total = sum(powered.values())
    if total <= 0:
        return {}

    # Every lap has a fastest lap — distribute ALL race laps
    result = {d: (s / total) * race_laps for d, s in powered.items()}

    # Per-driver cap based on historical max fastest laps (with headroom)
    hist_max_fl = cal.get("max_fastest_laps", race_laps * 0.25)
    max_fl = min(race_laps * 0.40, hist_max_fl * 1.25)

    # Iteratively cap and redistribute
    for _ in range(10):
        deficit = race_laps - sum(result.values())
        if abs(deficit) < 0.5:
            break
        for d in result:
            if result[d] > max_fl:
                result[d] = max_fl
        deficit = race_laps - sum(result.values())
        if deficit > 0.5:
            uncapped = {d: s for d, s in result.items() if s < max_fl}
            uncapped_total = sum(uncapped.values())
            if uncapped_total > 0:
                for d in uncapped:
                    result[d] += deficit * (uncapped[d] / uncapped_total)

    return result


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           is_prerace, race_name, race_id, track_name, series_id, dk_df,
           odds_data=None, scheduled_laps=0, race_date="", season=None):
    """Render the Projections tab."""
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

    # Weight sliders in collapsible expander
    with st.expander("Projection Weights", expanded=False):
        st.caption(f"Defaults tuned for **{parent_type}** tracks. "
                   "Adjust weights — auto-normalizes to 100%.")
        w_cols = st.columns(6)
        w_odds = w_cols[0].number_input("Odds", 0, 100, defaults["odds"], 5, key="pw_odds")
        w_track = w_cols[1].number_input("Track History", 0, 100, defaults["track"], 5, key="pw_track")
        w_ttype = w_cols[2].number_input("Track Type", 0, 100, defaults["ttype"], 5, key="pw_ttype")
        w_prac = w_cols[3].number_input("Practice", 0, 100, defaults["prac"], 5, key="pw_prac")
        w_team = w_cols[4].number_input("Team", 0, 100, defaults["team"], 5, key="pw_team")
        w_qual = w_cols[5].number_input("Qualifying", 0, 100, defaults["qual"], 5, key="pw_qual")

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
        dom_ceiling = display_avg_top * 0.25 + min(hist_max_fl, race_laps) * 0.45

        info_cols = st.columns(4)
        info_cols[0].metric("Race Laps", f"{race_laps}")
        info_cols[1].metric("Max Laps Led Pts", f"{race_laps * 0.25:.1f}")
        info_cols[2].metric("Max Fastest Lap Pts", f"{race_laps * 0.45:.1f}")
        info_cols[3].metric("Dominator Ceiling", f"{dom_ceiling:.1f}")
        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.82rem;font-weight:600;margin:0.3rem 0;">'
            f"Laps led = 0.25 pts/lap | Fastest laps = 0.45 pts/lap | "
            f"Place diff = \u00b11.0 pts/pos | {race_laps} total laps | "
            f"Avg {avg_leaders:.0f} leaders, Avg Race Lap Leader leads {display_avg_top:.0f} laps "
            f"(max: {display_max_ll:.0f})</p>",
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
    )


def _query_db_track_history(track_name, series_id, exclude_race_id=None,
                             before_date=None, cross_series_ids=None,
                             trim_worst: bool = False):
    """Query per-driver track history from DB with date filtering.

    Args:
        cross_series_ids: list of higher-series IDs for cross-series blending.
            When provided, returns a tuple (current_df, cross_df).
            When None, returns just current_df.
        trim_worst: when True, each driver's WORST finish is excluded from
            the aggregates (provided they have 4+ races at this track).
            Used at superspeedways where one unavoidable pileup shouldn't
            poison a driver's pace-reflective avg.
    """
    if not os.path.exists(PROJ_DB):
        return (pd.DataFrame(), pd.DataFrame()) if cross_series_ids else pd.DataFrame()

    def _run_query(conn, sid_list):
        if len(sid_list) == 1:
            where = "WHERE t.name LIKE ? AND r.series_id = ?"
            params = [f"%{track_name}%", sid_list[0]]
        else:
            placeholders = ",".join("?" for _ in sid_list)
            where = f"WHERE t.name LIKE ? AND r.series_id IN ({placeholders})"
            params = [f"%{track_name}%"] + sid_list
        if exclude_race_id:
            where += " AND r.id != ?"
            params.append(exclude_race_id)
        if before_date:
            where += " AND r.race_date < ?"
            params.append(before_date)

        # Base aggregate query (always used)
        base_query = f'''
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
            df = pd.read_sql_query(base_query, conn, params=params)
        except Exception:
            return pd.DataFrame()

        # Optional trimmed-mean pass: for drivers with 4+ races, recompute
        # Avg Finish and Avg Run Pos excluding their worst finish at this track.
        # This is the "drop one unavoidable wreck" adjustment for supers.
        if trim_worst and not df.empty:
            trim_query = f'''
                WITH ranked AS (
                    SELECT d.id as did, d.full_name as Driver,
                           rr.finish_pos, rr.avg_running_position,
                           ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY rr.finish_pos DESC, rr.id) as rn,
                           COUNT(*) OVER (PARTITION BY d.id) as n
                    FROM race_results rr
                    JOIN drivers d ON d.id = rr.driver_id
                    JOIN races r ON r.id = rr.race_id
                    JOIN tracks t ON t.id = r.track_id
                    {where}
                )
                SELECT did, Driver,
                       ROUND(AVG(finish_pos), 1) as trimmed_finish,
                       ROUND(AVG(avg_running_position), 1) as trimmed_arp
                FROM ranked
                WHERE (n >= 4 AND rn > 1) OR n < 4
                GROUP BY did
            '''
            try:
                tdf = pd.read_sql_query(trim_query, conn, params=params)
                # Replace Avg Finish / Avg Run Pos with trimmed versions for
                # drivers with 4+ races (where trimming actually happened)
                base_idx = df.set_index("Driver")
                for _, trow in tdf.iterrows():
                    drv = trow["Driver"]
                    if drv in base_idx.index and base_idx.loc[drv, "Races"] >= 4:
                        df.loc[df["Driver"] == drv, "Avg Finish"] = trow["trimmed_finish"]
                        if pd.notna(trow.get("trimmed_arp")):
                            df.loc[df["Driver"] == drv, "Avg Run Pos"] = trow["trimmed_arp"]
            except Exception:
                # If the trim query fails, just use the un-trimmed aggregate
                pass
        return df

    conn = sqlite3.connect(PROJ_DB)
    current_df = _run_query(conn, [series_id])

    if cross_series_ids:
        cross_df = _run_query(conn, cross_series_ids)
        conn.close()
        return current_df, cross_df

    conn.close()
    return current_df


def _query_db_track_type_history(track_type, series_id, exclude_track=None,
                                  exclude_race_id=None, before_date=None,
                                  cross_series_ids=None):
    """Query per-driver stats across all tracks of a given type from DB.

    Used for completed races to prevent data leakage.
    When cross_series_ids provided, returns (current_dict, cross_dict).
    """
    if not os.path.exists(PROJ_DB):
        return ({}, {}) if cross_series_ids else {}

    from src.config import TRACK_TYPE_MAP as _TTM
    # Match by parent type so subtypes include siblings
    # e.g. Bristol (short_concrete) includes all "short" tracks
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    matching_tracks = [t for t, tt in _TTM.items()
                       if TRACK_TYPE_PARENT.get(tt, tt) == parent]
    if exclude_track:
        matching_tracks = [t for t in matching_tracks if exclude_track not in t]
    if not matching_tracks:
        return ({}, {}) if cross_series_ids else {}

    def _run_tt_query(conn, sid_list):
        placeholders_t = ",".join("?" for _ in matching_tracks)
        if len(sid_list) == 1:
            series_clause = "AND r.series_id = ?"
            params = matching_tracks + [sid_list[0]]
        else:
            series_ph = ",".join("?" for _ in sid_list)
            series_clause = f"AND r.series_id IN ({series_ph})"
            params = matching_tracks + sid_list
        where_extra = ""
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
            WHERE t.name IN ({placeholders_t})
              {series_clause}
              {where_extra}
            GROUP BY d.id
        '''
        rows = conn.execute(query, params).fetchall()
        result = {}
        for row in rows:
            name, races, avg_f, avg_arp, ll = row
            if races and races > 0:
                result[name] = {
                    "avg_finish": avg_f or 20,
                    "avg_running_pos": avg_arp,
                    "laps_led_per_race": (ll or 0) / races,
                    "races": races,
                }
        return result

    conn = sqlite3.connect(PROJ_DB)
    current = _run_tt_query(conn, [series_id])

    if cross_series_ids:
        cross = _run_tt_query(conn, cross_series_ids)
        conn.close()
        return current, cross

    conn.close()
    return current


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
                            race_date="", season=None):
    """Build DFS-aware projections that estimate actual DK point components."""
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
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Fastest Laps", "Races", "Wins", "Top 5", "Top 10", "DNF"]:
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
            import math
            ranked = sorted(odds_probs.items(), key=lambda x: x[1], reverse=True)
            log_probs = {name: math.log(prob) for name, prob in ranked}
            max_lp = max(log_probs.values())
            min_lp = min(log_probs.values())
            lp_range = max_lp - min_lp
            for name, prob in ranked:
                matched = fuzzy_match_name(name, drivers)
                if matched:
                    if lp_range > 0:
                        t = 1 - (log_probs[name] - min_lp) / lp_range  # 0=best, 1=worst
                        odds_finish[matched] = 1 + (field_size - 1) * t
                    else:
                        odds_finish[matched] = mid_field
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

    # ── Projected ownership (heuristic) ──
    try:
        from src.ownership import project_ownership, compute_leverage
        _proj_dk_dict = dict(zip(proj["Driver"], proj["Proj DK"]))
        _sal_dict = dict(zip(proj["Driver"], proj["DK Salary"])) if "DK Salary" in proj.columns else {}
        _proj_finish_dict = dict(zip(proj["Driver"], proj["Proj Finish"])) if "Proj Finish" in proj.columns else {}
        # win_odds: odds_data is keyed by driver display name already
        _own_map = project_ownership(
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
        proj["Proj Own %"] = proj["Driver"].map(_own_map).round(1)
        # Leverage = points per ownership point (higher = better GPP play)
        _lev_map = compute_leverage(_proj_dk_dict, _own_map)
        proj["Leverage"] = proj["Driver"].map(_lev_map).round(2)
    except Exception as _ow_err:
        # Non-fatal — ownership is additive info only
        pass

    # Share Proj DK with optimizer tab via session state
    st.session_state["proj_dk_map"] = dict(zip(proj["Driver"], proj["Proj DK"]))
    # Share per-driver detail so optimizer can identify projected dominators
    st.session_state["proj_detail_map"] = _proj_detail
    # Share ownership + leverage for optimizer leverage scoring
    if "Proj Own %" in proj.columns:
        st.session_state["proj_own_map"] = dict(zip(proj["Driver"], proj["Proj Own %"]))
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
    if "DK Salary" in proj.columns:
        display_cols.append("DK Salary")
    display_cols.append("Proj DK")
    if "Value" in proj.columns:
        display_cols.append("Value")
    if "Proj Own %" in proj.columns:
        display_cols.append("Proj Own %")
    if "Leverage" in proj.columns:
        display_cols.append("Leverage")
    qual_col = "Qual Pos" if "Qual Pos" in proj.columns else "Proj Qual Pos"
    if qual_col in proj.columns:
        display_cols.append(qual_col)
    display_cols.extend(["Proj Finish", "PD Upside",
                         "Finish Pts", "Diff Pts",
                         "Led Pts", "FL Pts", "Proj Laps Led", "Proj Fast Laps",
                         "Avg DK", "Best DK", "Worst DK",
                         "Avg DK (Similar)", "Similar Races"]
                        )
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

    from src.components import build_projection_column_config, interactive_drill_down_dataframe
    disp = proj[avail].copy()
    col_config = build_projection_column_config(disp)
    st.caption("Click any driver row for race-by-race history at this track")
    interactive_drill_down_dataframe(
        safe_fillna(disp),
        key=f"proj_main_{series_id}_{race_id}",
        series_id=series_id, track_name=track_name,
        width="stretch", hide_index=False, height=550,
        column_config=col_config,
    )

    # Chart — all drivers, stacked bar with component breakdown on hover
    import plotly.graph_objects as go
    chart_df = proj.copy()
    chart_df = chart_df.sort_values("Proj DK", ascending=True)  # horizontal bar: bottom = best

    # Build stacked horizontal bar with DK scoring components
    # Positive components stack right, negative Diff Pts stacks left
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

    from src.charts import DARK_LAYOUT, apply_dark_theme
    n_drivers = len(chart_df)
    fig.update_layout(
        **DARK_LAYOUT,
        barmode="relative",
        title="All Drivers — Projected DK Points Breakdown",
        xaxis_title="Projected DK Points",
        yaxis_title="",
        height=max(400, n_drivers * 22),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickfont=dict(size=10)),
    )
    apply_dark_theme(fig)
    st.plotly_chart(fig, width="stretch")
