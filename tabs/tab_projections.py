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
    # Default avg fastest-laps for the per-race FL leader, scaled by track type
    # (≈55-60% of the max). Used as the FL projection ceiling anchor.
    AVG_FL_LEADER_DEFAULT = {
        "superspeedway": 18, "road": 14, "intermediate": 45,
        "intermediate_worn": 40, "short": 55, "short_concrete": 55,
    }
    defaults = {
        "avg_top_leader": AVG_TOP_LEADER_DEFAULT.get(track_type,
                            AVG_TOP_LEADER_DEFAULT.get(parent, 60)),
        "max_laps_led": type_defaults["max_ll"],
        "max_fastest_laps": type_defaults["max_fl"],
        "avg_fl_leader": AVG_FL_LEADER_DEFAULT.get(track_type,
                            AVG_FL_LEADER_DEFAULT.get(parent, 40)),
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
            out["avg_fl_leader"] = float(np.mean(top_fl))
        if n_fl_list:
            out["avg_n_fl_leaders"] = float(np.mean(n_fl_list))
        return out

    def _rank_curve(conn, where_sql, where_params, value_col):
        """Empirical {value_col}-by-rank distribution from history.

        For each race matching the filter, sort drivers by `value_col`
        (fastest_laps OR laps_led) descending and record each rank's
        fractional share of that race's total. Average the fractions across
        races, normalize to sum to 1. Returns [frac_rank1, frac_rank2, ...]
        or None if too few races.

        Using FRACTIONS (not absolute counts) makes the shape robust to
        differing race lengths / caution-shortened totals between years.
        Only races with a meaningful total are included so empty future rows
        and tiny exhibition heats don't distort the curve.

        Critically for LAPS LED: real intermediate races concentrate laps on
        one dominator (~46% to the leader, steep dropoff), so the empirical
        curve reproduces that shape instead of the old parametric power curve,
        which spread laps too evenly across the field.
        """
        rows = conn.execute(f'''
            SELECT r.id, rr.{value_col}
            FROM race_results rr
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE {where_sql} AND rr.{value_col} IS NOT NULL
            ORDER BY r.id, rr.{value_col} DESC
        ''', where_params).fetchall()
        if not rows:
            return None
        per_race = {}
        for rid, v in rows:
            per_race.setdefault(rid, []).append(v or 0)
        rank_fracs = {}
        n_races = 0
        for rid, vals in per_race.items():
            total = sum(vals)
            if total < 30:   # skip races with negligible data (future/heats)
                continue
            n_races += 1
            for rank, v in enumerate(vals):
                rank_fracs.setdefault(rank, []).append(v / total)
        if n_races < 2:
            return None
        # Average the fraction at each rank, then normalize to sum to 1.0
        curve = []
        for rank in sorted(rank_fracs.keys()):
            curve.append(float(np.mean(rank_fracs[rank])))
        s = sum(curve)
        if s <= 0:
            return None
        curve = [c / s for c in curve]

        # Sample-size-aware temperature smoothing. With few historical races
        # the empirical curve can be extremely top-heavy by random luck — e.g.
        # Nashville Trucks (5 races, dominated 1-2 cars each) yields rank 5 =
        # 0.5% of laps, so a real contender stuck at dom-rank 5 by chance
        # would only get ~1 lap in a 150-lap race. Raising each share to a
        # sub-1 power flattens the curve while preserving rank ordering (a
        # principled "shrinkage to prior" for small samples). Full trust at
        # n ≥ 10 races; T floor 0.75 protects against extreme cliffs.
        T = max(0.75, min(1.0, n_races / 10.0))
        if T < 1.0:
            curve = [c ** T for c in curve]
            s2 = sum(curve)
            if s2 > 0:
                curve = [c / s2 for c in curve]
        return curve

    try:
        conn = sqlite3.connect(PROJ_DB)

        # Step 1: per-track history
        series_filter = "AND r.series_id = ?" if series_id else ""
        params = [track_name]
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
            WHERE t.name = ?
            {series_filter}
            GROUP BY r.id
        ''', params).fetchall()

        result = dict(defaults)
        per_track_summary = _summarize(per_track_rows) if per_track_rows else None

        # Step 2: track-type history for the same series (used as fallback when
        # per-track is too sparse, AND as the source for any fields the per-track
        # query couldn't fill in)
        type_summary = None
        type_placeholders = None
        if family_tracks and series_id:
            type_placeholders = ",".join("?" for _ in family_tracks)
            type_rows = conn.execute(f'''
                SELECT MAX(rr.laps_led) as top_led,
                       COUNT(CASE WHEN rr.laps_led > 0 THEN 1 END) as n_leaders,
                       MAX(rr.fastest_laps) as top_fl,
                       COUNT(CASE WHEN rr.fastest_laps > 0 THEN 1 END) as n_fl
                FROM race_results rr
                JOIN races r ON r.id = rr.race_id
                JOIN tracks t ON t.id = r.track_id
                WHERE t.name IN ({type_placeholders})
                  AND r.series_id = ?
                GROUP BY r.id
            ''', list(family_tracks) + [series_id]).fetchall()
            type_summary = _summarize(type_rows)

        # Empirical FL- and LL-by-rank distributions: prefer per-track, fall
        # back to track-type-for-series. Stored on the calibration so the
        # allocators map projected order onto the REAL historical shape — this
        # is what makes laps led concentrate on the dominator (~46% to the
        # leader at Charlotte) instead of the old too-flat power curve.
        for _col, _key in [("fastest_laps", "fl_rank_distribution"),
                           ("laps_led", "ll_rank_distribution")]:
            _dist = _rank_curve(conn, f"t.name = ? {series_filter}", params, _col)
            if _dist is None and type_placeholders:
                _dist = _rank_curve(
                    conn, f"t.name IN ({type_placeholders}) AND r.series_id = ?",
                    list(family_tracks) + [series_id], _col)
            if _dist:
                result[_key] = _dist

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
        # (fl_rank_distribution was set above regardless and survives .update)

        return result

    except Exception:
        pass

    return defaults


# ── Soft-rank smoothing of the empirical by-rank curves ──────────────────────
# The empirical laps-led / fastest-laps curves describe the SHAPE of a typical
# race (the leader takes ~46% of laps at Charlotte, steep dropoff). The old
# mapping assigned that shape by HARD rank — driver_k gets curve[k] — which is a
# step function of the dominator score: a razor-thin score change flips a driver
# between adjacent rungs, and because the laps-led curve is steep that swung
# drivers like Byron between 87 and 18 projected laps on a tiny weight-slider
# change. `_soft_rank_shares` replaces the hard rank with an EXPECTED (soft)
# rank so the allocation is continuous in the scores: near-tied contenders SHARE
# the dominator laps instead of one taking the whole top rung, while a clear
# runaway favorite still collapses to the full leader share.
_LL_SOFT_TAU = 5.0       # logistic temperature for laps led (0-100 score units)
_FL_SOFT_TAU = 7.0       # fastest laps spread wider, so smooth a touch softer
_LL_SOFT_TOPK = 14       # only realistic leaders compete for laps-led shares
_FL_SOFT_TOPK = 28       # fastest laps legitimately reach far more drivers


def _interp_curve(curve: list, x: float) -> float:
    """Linear interpolation of `curve` (indexed 0,1,2,...) at fractional x."""
    if not curve:
        return 0.0
    if x <= 0:
        return curve[0]
    if x >= len(curve) - 1:
        return curve[-1]
    lo = int(math.floor(x))
    frac = x - lo
    return curve[lo] * (1.0 - frac) + curve[lo + 1] * frac


def _soft_rank_shares(scores: dict, dist: list, tau: float, top_k: int) -> dict:
    """Map dominator scores onto an empirical by-rank share curve via EXPECTED
    ranks (Bradley-Terry), instead of snapping each driver to one hard rung.

    For each contender i, the expected rank is the expected number of drivers
    who outrank it:  E[rank_i] = Σ_j σ((s_j - s_i)/τ).  The empirical curve is
    then read at that fractional rank by interpolation. Properties:
      • Continuous in the scores → no step jumps; a small weight change moves a
        driver's laps smoothly instead of flipping 87 ↔ 18.
      • Separated limit (clear favorite, big score gap) → E[rank] → integer
        rank → recovers the exact hard-rank curve (leader still gets ~46%).
      • Tie limit (cluster of equal dominators) → they SHARE the top rungs
        evenly rather than one winning all the laps on a hair-thin edge.

    Scores are normalized so the field leader = 100, making τ a stable unit
    regardless of track / weight scale. Only the top_k by score compete (laps
    led realistically go to a handful of cars; restricting the pool also stops
    a long tail of backmarkers from inflating every contender's expected rank).
    Shares are renormalized to the curve's total mass so all laps stay allocated.
    """
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = {d: 0.0 for d, _ in items}
    if not items or not dist:
        return out
    smax = items[0][1]
    if smax <= 0:
        return out
    k = min(len(items), max(1, top_k))
    top = [(d, 100.0 * max(0.0, s) / smax) for d, s in items[:k]]
    curve = list(dist) + [0.0] * max(0, k - len(dist))

    raw = {}
    for d_i, ns_i in top:
        er = 0.0
        for d_j, ns_j in top:
            if d_j == d_i:
                continue
            er += 1.0 / (1.0 + math.exp(-(ns_j - ns_i) / tau))
        raw[d_i] = max(0.0, _interp_curve(curve, er))

    target = sum(dist)
    tot = sum(raw.values())
    if tot > 0:
        for d, v in raw.items():
            out[d] = v * target / tot
    return out


# ── Start-position availability gate for laps led ────────────────────────────
# The soft-rank allocator maps PACE rank onto the empirical laps-led-by-rank
# curve, but that curve is start-BLIND — it only encodes how concentrated laps
# are among whoever led them. So a fast car starting deep inherits the leader's
# ~40% share even though, empirically, a P18+ starter leads ~0.6% of laps on
# average (Cup 2022+, 90th pctl ~0%). This gate multiplies each driver's curve
# share by an availability factor in [floor, 1.0] that is ~1.0 through P10 and
# decays for deeper starts, then RE-NORMALIZES so the freed laps flow to the
# front-runners and the total stays == race_laps. It is continuous in start_pos
# (logistic), preserving the soft-rank allocator's no-step-jump property.
#
# Decay is track-type aware: steep on concrete/short (track position is king,
# dirty air makes advancing slow), moderate on intermediate/road (long green
# runs and tire falloff let a fast car work forward), near-flat on superspeedway
# (the pack cycles the lead through the whole field regardless of start).
#   (floor, p50): p50 = the start at which availability is halfway from 1.0 to
#   floor; the transition spans ~13 positions either side of p50.
_LL_AVAIL = {
    "superspeedway":     (0.82, 22),
    "road":              (0.45, 16),
    "intermediate":      (0.45, 16),
    "intermediate_worn": (0.45, 16),
    "short":             (0.30, 15),
    "short_concrete":    (0.25, 15),
}
_LL_AVAIL_DEFAULT = (0.45, 16)


def _start_avail(start_pos, floor: float, p50: float) -> float:
    """Laps-led availability multiplier in [floor, 1.0] as a function of start.

    Flat ~1.0 through ~P10, smooth logistic decay deeper, asymptote `floor`.
    Continuous in start_pos so it never reintroduces rank step-jumps."""
    if start_pos is None or start_pos <= 10:
        return 1.0
    decay = 1.0 / (1.0 + math.exp((start_pos - p50) / 6.0))
    return floor + (1.0 - floor) * decay


def _apply_start_gate(alloc: dict, start_positions: dict,
                      track_type: str, parent: str) -> dict:
    """Damp each driver's laps-led allocation by start availability, then
    renormalize so the total is unchanged (the laps a deep starter loses flow to
    the front-runners). No-op when start_positions is empty. Operates on either
    fractional shares or absolute lap counts — it preserves the input's total."""
    if not start_positions:
        return alloc
    floor, p50 = _LL_AVAIL.get(track_type, _LL_AVAIL.get(parent, _LL_AVAIL_DEFAULT))
    gated = {d: v * _start_avail(start_positions.get(d), floor, p50)
             for d, v in alloc.items()}
    target = sum(alloc.values())
    tot = sum(gated.values())
    if tot > 0 and target > 0:
        return {d: v * target / tot for d, v in gated.items()}
    return alloc


def _allocate_laps_led(driver_scores: dict, race_laps: int, track_name: str,
                        track_type: str, calibration: dict = None,
                        odds_display: dict = None,
                        start_positions: dict = None) -> dict:
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

    # ── Preferred path: map our projected dominator order onto the EMPIRICAL
    # historical laps-led-by-rank shape (same approach as fastest laps). Our
    # model decides WHO dominates; the real historical curve decides HOW MANY
    # laps each rank leads. This reproduces the true concentration of
    # intermediate races — the leader takes ~46% of laps at Charlotte with a
    # steep dropoff — instead of the parametric power curve below, which
    # spread laps too evenly (top ~60 across ~11 drivers). ──
    ll_dist = cal.get("ll_rank_distribution")
    if ll_dist:
        shares = _soft_rank_shares(driver_scores, ll_dist,
                                   tau=_LL_SOFT_TAU, top_k=_LL_SOFT_TOPK)
        shares = _apply_start_gate(shares, start_positions, track_type, parent)
        return {d: race_laps * frac for d, frac in shares.items()}

    # ── Fallback path (no historical LL shape for this track/type): parametric
    # power-curve concentration. ──
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

    # Start-position gate (same as the empirical path): pull deep starters' laps
    # toward the front-runners BEFORE capping, so cap/redistribute still
    # guarantees the total stays == race_laps and respects the per-driver ceiling.
    result = _apply_start_gate(result, start_positions, track_type, parent)

    # Per-driver cap: use average top leader with generous headroom
    # A dominant favorite (-100) routinely exceeds the average leader's laps
    avg_top = cal.get("avg_top_leader", race_laps * 0.40)
    max_laps = min(race_laps * 0.75, avg_top * 1.40)  # 40% headroom over average

    result = _cap_and_redistribute(result, max_laps, race_laps)
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

    # ── Preferred path: map our projected FL order onto the EMPIRICAL
    # historical FL-by-rank shape (track -> track-type fallback, computed in
    # the calibration). Our model only decides WHO is fastest; the real
    # historical curve decides HOW MANY fast laps each rank gets. This fixes
    # the over-concentration the parametric exponent produced (e.g. 70/70/47
    # at Charlotte vs the real 59/40/31 shape). ──
    fl_dist = cal.get("fl_rank_distribution")
    if fl_dist:
        shares = _soft_rank_shares(driver_fl_scores, fl_dist,
                                   tau=_FL_SOFT_TAU, top_k=_FL_SOFT_TOPK)
        return {d: race_laps * frac for d, frac in shares.items()}

    # ── Fallback path (no historical FL data for this track or track type):
    # parametric exponent-based concentration. ──
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

    # Per-driver cap. Anchor on the historical FL LEADER's average (the
    # typical most-fastest-laps driver) with modest headroom, not the
    # all-time max — even a dominant car rarely exceeds the leader average
    # by much (Charlotte: leader avg ~59, all-time max 70). Hard ceiling at
    # the all-time max so we never project above what's ever happened.
    avg_fl_leader = cal.get("avg_fl_leader")
    hist_max_fl = cal.get("max_fastest_laps", race_laps * 0.25)
    if avg_fl_leader:
        max_fl = min(hist_max_fl, avg_fl_leader * 1.30)
    else:
        max_fl = min(race_laps * 0.30, hist_max_fl)

    result = _cap_and_redistribute(result, max_fl, race_laps)
    return result


def _cap_and_redistribute(result: dict, max_per_driver: float,
                          total_to_allocate: float) -> dict:
    """Cap any driver above max_per_driver and redistribute the excess to
    uncapped drivers, repeating until stable.

    Replaces the previous in-line loops which were dead code: they checked
    `deficit = total - sum(result)` first and broke immediately because the
    initial allocation already summed to the total — so the per-driver cap
    never ran and top drivers could be allocated above the historical max.
    """
    result = dict(result)
    for _ in range(25):
        excess = 0.0
        for d in result:
            if result[d] > max_per_driver:
                excess += result[d] - max_per_driver
                result[d] = max_per_driver
        if excess < 0.5:
            break
        uncapped = {d: s for d, s in result.items() if s < max_per_driver - 1e-9}
        uncapped_total = sum(uncapped.values())
        if uncapped_total <= 0:
            break  # everyone is at the cap — can't redistribute further
        for d in uncapped:
            result[d] += excess * (uncapped[d] / uncapped_total)
    return result


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
        dom_ceiling = display_avg_top * 0.25 + min(hist_max_fl, race_laps) * 0.45

        # Fastest-laps history (parallel to laps-led): the per-race FL leader's
        # average + the all-time single-driver max. Capped at race_laps for display.
        avg_fl_leader = calibration.get("avg_fl_leader", hist_max_fl * 0.85)
        display_avg_fl = min(avg_fl_leader, race_laps)
        display_max_fl = min(hist_max_fl, race_laps)

        info_cols = st.columns(4)
        info_cols[0].metric("Race Laps", f"{race_laps}")
        info_cols[1].metric("Max Laps Led Pts", f"{race_laps * 0.25:.1f}")
        info_cols[2].metric("Max Fastest Lap Pts", f"{race_laps * 0.45:.1f}")
        info_cols[3].metric("Dominator Ceiling", f"{dom_ceiling:.1f}")
        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.82rem;font-weight:600;margin:0.3rem 0;">'
            f"Laps led = 0.25 pts/lap | Fastest laps = 0.45 pts/lap | "
            f"Place diff = \u00b11.0 pts/pos | {race_laps} total laps</p>",
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


def _query_db_track_history(track_name, series_id, exclude_race_id=None,
                             before_date=None, cross_series_ids=None,
                             trim_worst: bool = False):
    """Query per-driver track history from DB with date filtering.

    Args:
        cross_series_ids: list of higher-series IDs for cross-series blending.
            When provided, returns a tuple (current_df, cross_df).
            When None, returns just current_df.
        trim_worst: superspeedway flag. When True, crash/mechanical DNFs are
            KEPT in the finish average (wrecks at supers are frequent and
            partly a driver skill, so excluding them overstates pace). When
            False, DNFs are excluded from the finish average (their bad finish
            is luck; crash rate is priced separately by the DNF penalty).
            Aggregates are recency-weighted by race order either way.
    """
    if not os.path.exists(PROJ_DB):
        return (pd.DataFrame(), pd.DataFrame()) if cross_series_ids else pd.DataFrame()

    def _run_query(conn, sid_list):
        if len(sid_list) == 1:
            where = "WHERE t.name = ? AND r.series_id = ?"
            params = [track_name, sid_list[0]]
        else:
            placeholders = ",".join("?" for _ in sid_list)
            where = f"WHERE t.name = ? AND r.series_id IN ({placeholders})"
            params = [track_name] + sid_list
        if exclude_race_id:
            where += " AND r.id != ?"
            params.append(exclude_race_id)
        if before_date:
            where += " AND r.race_date < ?"
            params.append(before_date)

        # Recency by RACE ORDER: rank each driver's races newest-first and
        # weight w = max(0, 1-(rn-1)*STEP) so the last ~10-14 races dominate
        # (current form isn't drowned by the volume of older races).
        w = _recency_weight_sql("rn")
        # Accident-aware finish average: exclude crash/mechanical-DNF races
        # (the bad finish is luck; crash rate is priced by the DNF penalty),
        # raw fallback when every race was a DNF. Avg Run Pos keeps ALL races
        # (running position reflects the pace shown). Superspeedways
        # (trim_worst) keep DNFs in the finish average.
        _clean = "LOWER(COALESCE(status,'running')) IN ('running','')"
        if trim_worst:
            finish_num, finish_den = f"SUM(finish_pos*{w})", f"SUM({w})"
        else:
            finish_num = f"SUM(CASE WHEN {_clean} THEN finish_pos*{w} END)"
            finish_den = f"SUM(CASE WHEN {_clean} THEN {w} END)"

        query = f'''
            WITH ranked AS (
                SELECT d.id AS did, d.full_name AS Driver,
                       rr.finish_pos, rr.start_pos, rr.laps_led, rr.fastest_laps,
                       rr.avg_running_position AS arp, rr.rating AS rating, rr.status,
                       ROW_NUMBER() OVER (PARTITION BY d.id
                                          ORDER BY r.race_date DESC, r.id DESC) AS rn
                FROM race_results rr
                JOIN drivers d ON d.id = rr.driver_id
                JOIN races r ON r.id = rr.race_id
                JOIN tracks t ON t.id = r.track_id
                {where}
            )
            SELECT Driver,
                   COUNT(*) as Races,
                   ROUND(COALESCE({finish_num}/NULLIF({finish_den},0), AVG(finish_pos)), 1) as "Avg Finish",
                   ROUND(AVG(start_pos), 1) as "Avg Start",
                   SUM(laps_led) as "Laps Led",
                   SUM(fastest_laps) as "Fastest Laps",
                   ROUND(SUM(CASE WHEN arp IS NOT NULL THEN arp*{w} END)/
                         NULLIF(SUM(CASE WHEN arp IS NOT NULL THEN {w} END),0), 1) as "Avg Run Pos",
                   ROUND(SUM(CASE WHEN rating IS NOT NULL THEN rating*{w} END)/
                         NULLIF(SUM(CASE WHEN rating IS NOT NULL THEN {w} END),0), 1) as "Avg Rating",
                   SUM(CASE WHEN finish_pos = 1 THEN 1 ELSE 0 END) as Wins,
                   SUM(CASE WHEN finish_pos <= 5 THEN 1 ELSE 0 END) as "Top 5",
                   SUM(CASE WHEN finish_pos <= 10 THEN 1 ELSE 0 END) as "Top 10",
                   SUM(CASE WHEN LOWER(COALESCE(status,'running')) NOT IN ('running','')
                        THEN 1 ELSE 0 END) as DNF
            FROM ranked
            GROUP BY did
            HAVING COUNT(*) >= 1
        '''
        try:
            return pd.read_sql_query(query, conn, params=params)
        except Exception:
            return pd.DataFrame()

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

        # Recency by RACE ORDER (see _query_db_track_history): rank each
        # driver's track-type races newest-first; weight w=max(0,1-(rn-1)*STEP)
        # so recent form dominates the volume of older races. Accident-aware
        # finish average (exclude crash/mechanical DNFs, raw fallback); ARP
        # keeps all races; superspeedways keep DNFs in the finish average.
        w = _recency_weight_sql("rn")
        if parent == "superspeedway":
            fnum, fden = f"SUM(finish_pos*{w})", f"SUM({w})"
        else:
            _clean = "LOWER(COALESCE(status,'running')) IN ('running','')"
            fnum = f"SUM(CASE WHEN {_clean} THEN finish_pos*{w} END)"
            fden = f"SUM(CASE WHEN {_clean} THEN {w} END)"

        query = f'''
            WITH ranked AS (
                SELECT d.id AS did, d.full_name AS name,
                       rr.finish_pos, rr.avg_running_position AS arp,
                       rr.rating AS rating, rr.laps_led, rr.status,
                       ROW_NUMBER() OVER (PARTITION BY d.id
                                          ORDER BY r.race_date DESC, r.id DESC) AS rn
                FROM race_results rr
                JOIN drivers d ON d.id = rr.driver_id
                JOIN races r ON r.id = rr.race_id
                JOIN tracks t ON t.id = r.track_id
                WHERE t.name IN ({placeholders_t})
                  {series_clause}
                  {where_extra}
            )
            SELECT name,
                   COUNT(*) as races,
                   COALESCE({fnum}/NULLIF({fden},0), AVG(finish_pos)) as avg_finish,
                   SUM(CASE WHEN arp IS NOT NULL THEN arp*{w} END)/
                     NULLIF(SUM(CASE WHEN arp IS NOT NULL THEN {w} END),0) as avg_running_pos,
                   SUM(CASE WHEN rating IS NOT NULL THEN rating*{w} END)/
                     NULLIF(SUM(CASE WHEN rating IS NOT NULL THEN {w} END),0) as avg_rating,
                   SUM(laps_led) as total_laps_led
            FROM ranked
            GROUP BY did
        '''
        rows = conn.execute(query, params).fetchall()
        result = {}
        for row in rows:
            name, races, avg_f, avg_arp, avg_rating, ll = row
            if races and races > 0:
                result[name] = {
                    "avg_finish": avg_f or 20,
                    "avg_running_pos": avg_arp,
                    "tt_rating": avg_rating,
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
            # The WORST anchor is the subtle one: win odds barely discriminate
            # the back half of the field (everyone past a few favorites is a
            # tiny, noisy win%), yet a 38-car field has ONE winner and 37
            # non-winners who AVERAGE ~mid-field — not the wall. Anchoring the
            # longest shot to field*0.82 (~31st) was systematically burying
            # value plays: a driver with ~0% win equity but solid race pace
            # (good track/track-type form) was getting dragged toward 31st by
            # the heaviest single weight (odds = 33% on intermediates). Pulling
            # the worst anchor in to field*0.58 (~22nd) lets odds stay sharp
            # where they're reliable (separating the genuine contenders at the
            # front) while letting track/track-type/qual/DNF — not win odds —
            # decide who runs 22nd vs 35th.
            import math
            best_anchor = max(2.0, field_size * 0.13)
            worst_anchor = field_size * 0.58
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
                        odds_finish[matched] = best_anchor + (worst_anchor - best_anchor) * t
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
    # Laps completed is estimated at the full race distance for everyone —
    # a flat constant that doesn't change driver ordering, but keeps the
    # absolute totals comparable to real FD scores.
    from src.config import (FD_FINISH_POINTS, FD_PTS_LAPS_LED,
                            FD_PTS_LAPS_COMPLETED, FD_PTS_PLACE_DIFF)

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

    _fd_comp_pts = (race_laps or 0) * FD_PTS_LAPS_COMPLETED
    proj["Proj FD"] = proj.apply(
        lambda r: round(_fd_finish_pts(r["Proj Finish"])
                        + r["Diff Pts"] * FD_PTS_PLACE_DIFF
                        + r["Proj Laps Led"] * FD_PTS_LAPS_LED
                        + _fd_comp_pts, 1), axis=1)

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
    except Exception as _ow_err:
        # Non-fatal — ownership is additive info only
        pass

    # Share Proj DK / Proj FD with optimizer tab via session state
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
    if "Floor" in proj.columns:
        display_cols.append("Floor")
    if "Ceiling" in proj.columns:
        display_cols.append("Ceiling")
    if _show_dk and "Value" in proj.columns:
        display_cols.append("Value")
    if _show_fd and "FD Value" in proj.columns:
        display_cols.append("FD Value")
    if "GPP Own%" in proj.columns:
        display_cols.append("GPP Own%")
    if "Cash Own%" in proj.columns:
        display_cols.append("Cash Own%")
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
