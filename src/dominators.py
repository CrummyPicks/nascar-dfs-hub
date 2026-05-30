"""Dominator lineup recommendations.

Analyzes historical race_results to recommend how many "dominator" drivers a
DFS lineup should target at a given track.

A "dominator" is a driver who accumulates significant DK points from laps-led
and fastest-laps bonuses. The threshold is track-type-aware because short
tracks (400+ laps) produce far more LL/FL points per driver than superspeedways
(200 laps with constantly-shuffling leaders). A flat threshold would under-
count dominators at Bristol and over-count them at Daytona.

Thresholds were chosen empirically so that ~2-3 "true dominators" emerge per
race at tracks where that's the DFS ideal, and ~0-1 at tracks where dominator
points are too distributed to matter.
"""
from __future__ import annotations
import sqlite3
from collections import Counter
from pathlib import Path

# DK pts (LL*0.25 + FL*0.45) required to count as a dominator, per track type.
# Based on Cup 2022-26 historical distribution analysis.
DOM_THRESHOLDS = {
    "superspeedway":    20.0,  # ~0-1 dominators typical (pack racing disperses LL)
    "road":             15.0,  # Short races, fewer total laps available
    "intermediate":     20.0,  # ~2 dominators typical (Kansas, Vegas, etc.)
    "intermediate_worn": 20.0,  # ~2-3 (Darlington, Homestead)
    "short":            25.0,  # ~3 dominators (Martinsville, Phoenix, Richmond)
    "short_concrete":   30.0,  # ~3 dominators (Bristol, Dover)
}
DOM_THRESHOLD_DEFAULT = 20.0

MIN_RACES_FOR_CONFIDENCE = 3  # need this many races to trust track-specific stat


def threshold_for_track_type(track_type: str | None) -> float:
    """Return the dominator threshold (DK pts from LL+FL) for a given track type."""
    if not track_type:
        return DOM_THRESHOLD_DEFAULT
    return DOM_THRESHOLDS.get(track_type, DOM_THRESHOLD_DEFAULT)


def _compute_dominator_stats(conn, series_id, track_name=None, track_type=None, min_season=2022):
    """Query race_results and compute dominator counts per race.

    The threshold used depends on track_type (short tracks produce more LL/FL
    points so require a higher bar). When querying by track_name, the threshold
    is looked up via the track's DB track_type.

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
        SELECT r.id, t.track_type, rr.laps_led, rr.fastest_laps
        FROM race_results rr
        JOIN races r ON r.id = rr.race_id
        JOIN tracks t ON t.id = r.track_id
        WHERE {where}
    '''
    rows = conn.execute(q, params).fetchall()

    # Group by race and count dominators per race — use each race's own
    # track-type threshold so a mixed query (by track_type only) still counts
    # correctly if there were edge cases.
    per_race = {}
    race_ttype = {}
    for rid, ttype, ll, fl in rows:
        pts = (ll or 0) * 0.25 + (fl or 0) * 0.45
        per_race.setdefault(rid, []).append(pts)
        race_ttype[rid] = ttype

    counts = []
    for rid, pts_list in per_race.items():
        thr = threshold_for_track_type(race_ttype.get(rid))
        n_doms = sum(1 for p in pts_list if p >= thr)
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


def identify_dominators_in_projection(
    proj_detail: dict, track_type: str | None = None, threshold: float | None = None
) -> set:
    """Given per-driver projection detail with laps_led + fast_laps, return the
    set of driver names projected to be dominators (LL*0.25 + FL*0.5 >= threshold).

    The threshold defaults to the track-type-appropriate value if not supplied.
    """
    if threshold is None:
        threshold = threshold_for_track_type(track_type)
    doms = set()
    for driver, det in (proj_detail or {}).items():
        ll = det.get("laps_led", 0) or 0
        fl = det.get("fast_laps", 0) or 0
        if ll * 0.25 + fl * 0.45 >= threshold:
            doms.add(driver)
    return doms
