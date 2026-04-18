"""Dominator lineup recommendations.

Analyzes historical race_results to recommend how many "dominator" drivers a
DFS lineup should target at a given track.

A "dominator" is a driver who accumulates significant DK points from laps-led
and fastest-laps bonuses. In DFS lineup construction, you want to roster as
many dominators as the track historically produces — no more (over-concentration
risk), no fewer (leaving bonus points on the table).

We use a 20 DK-points threshold (LL*0.25 + FL*0.5 >= 20) as "true dominator"
— drivers who meaningfully shaped the race. Lower thresholds count too many
drivers who had brief stints; higher thresholds miss typical intermediate-track
dominators.
"""
from __future__ import annotations
import sqlite3
from collections import Counter
from pathlib import Path

DOM_THRESHOLD = 20.0  # DK pts from LL+FL to count as a dominator
MIN_RACES_FOR_CONFIDENCE = 3  # need this many races to trust track-specific stat


def _compute_dominator_stats(conn, series_id, track_name=None, track_type=None, min_season=2022):
    """Query race_results and compute dominator counts per race.

    Returns list of dominator counts, one per race matching the filters.
    """
    params = [series_id, min_season]
    where = "r.series_id = ? AND r.season >= ?"
    if track_name:
        where += " AND t.name = ?"
        params.append(track_name)
    elif track_type:
        where += " AND t.track_type = ?"
        params.append(track_type)

    q = f'''
        SELECT r.id, rr.laps_led, rr.fastest_laps
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE {where}
    '''
    rows = conn.execute(q, params).fetchall()

    # Group by race and count dominators per race
    per_race = {}
    for rid, ll, fl in rows:
        pts = (ll or 0) * 0.25 + (fl or 0) * 0.5
        per_race.setdefault(rid, []).append(pts)

    counts = []
    for pts_list in per_race.values():
        n_doms = sum(1 for p in pts_list if p >= DOM_THRESHOLD)
        counts.append(n_doms)
    return counts


def get_dominator_recommendation(
    db_path: str | Path,
    series_id: int,
    track_name: str | None = None,
    track_type: str | None = None,
) -> dict:
    """Return dominator recommendation for a given track.

    Prefers track-specific stats when the track has enough history. Falls back
    to track-type aggregate otherwise.

    Returns:
        {
            "recommended": int (mode),
            "range": (min, max),
            "avg": float,
            "races_analyzed": int,
            "scope": "track" | "track_type" | "unknown",
            "track_type": str,
            "rationale": str (human-readable explanation),
        }
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return _default_recommendation(track_type)

    conn = sqlite3.connect(str(db_path))
    try:
        # Try track-specific first
        if track_name:
            counts = _compute_dominator_stats(conn, series_id, track_name=track_name)
            if len(counts) >= MIN_RACES_FOR_CONFIDENCE:
                return _build_recommendation(counts, "track", track_type, track_name)

        # Fall back to track type
        if track_type:
            counts = _compute_dominator_stats(conn, series_id, track_type=track_type)
            if len(counts) >= MIN_RACES_FOR_CONFIDENCE:
                return _build_recommendation(counts, "track_type", track_type, track_name)

        return _default_recommendation(track_type)
    finally:
        conn.close()


def _build_recommendation(counts, scope, track_type, track_name):
    """Build recommendation dict from a list of per-race dominator counts."""
    avg = sum(counts) / len(counts)
    mode = Counter(counts).most_common(1)[0][0]
    mn, mx = min(counts), max(counts)

    # Recommended count: use mode, but widen if mode and avg disagree a lot
    recommended_low = max(0, mode - 1) if avg < mode else mode
    recommended_high = mode + 1 if avg > mode else mode

    if recommended_low == recommended_high:
        rec_str = str(recommended_low)
    else:
        rec_str = f"{recommended_low}-{recommended_high}"

    # Rationale text
    if scope == "track":
        rationale = (
            f"Based on {len(counts)} prior races at {track_name}: "
            f"typical lineup has {rec_str} dominator{'s' if recommended_high != 1 else ''}. "
            f"(avg {avg:.1f}, range {mn}-{mx})"
        )
    else:
        tt_pretty = (track_type or "unknown").replace("_", " ").title()
        rationale = (
            f"Based on {len(counts)} prior {tt_pretty} races "
            f"(track-specific history not available): target {rec_str} "
            f"dominator{'s' if recommended_high != 1 else ''}. "
            f"(avg {avg:.1f}, range {mn}-{mx})"
        )

    return {
        "recommended": mode,
        "recommended_low": recommended_low,
        "recommended_high": recommended_high,
        "range": (mn, mx),
        "avg": avg,
        "races_analyzed": len(counts),
        "scope": scope,
        "track_type": track_type,
        "rationale": rationale,
    }


def _default_recommendation(track_type):
    """Fallback recommendation when DB lookup fails or has insufficient data.

    Based on broad DFS folk wisdom — used only when empirical data is unavailable.
    """
    defaults = {
        "superspeedway": (0, 0, 1, "Drafting races spread LL/FL wide; punt plays preferred."),
        "road": (1, 0, 1, "Short races with fewer total laps; track position matters more."),
        "intermediate": (2, 1, 3, "Typical intermediate: 1-2 teams dominate stages."),
        "intermediate_worn": (2, 2, 3, "Tire-wear tracks: 2-3 drivers can dominate."),
        "short": (3, 2, 4, "Short tracks concentrate laps led in 2-4 drivers."),
        "short_concrete": (3, 3, 4, "Bristol/Dover: most dominator-heavy tracks."),
    }
    mode, low, high, text = defaults.get(track_type, (2, 1, 3, "Default recommendation."))
    return {
        "recommended": mode,
        "recommended_low": low,
        "recommended_high": high,
        "range": (low, high),
        "avg": float(mode),
        "races_analyzed": 0,
        "scope": "default",
        "track_type": track_type,
        "rationale": f"No historical data available. Default guidance: {text}",
    }


def identify_dominators_in_projection(proj_detail: dict, threshold: float = DOM_THRESHOLD) -> set:
    """Given per-driver projection detail with laps_led + fast_laps, return the
    set of driver names projected to be dominators (LL*0.25 + FL*0.5 >= threshold).
    """
    doms = set()
    for driver, det in (proj_detail or {}).items():
        ll = det.get("laps_led", 0) or 0
        fl = det.get("fast_laps", 0) or 0
        if ll * 0.25 + fl * 0.5 >= threshold:
            doms.add(driver)
    return doms
