"""Projected ownership heuristic for NASCAR DFS contests.

Ownership % = the percentage of total entered lineups that roster a given driver.
In a GPP (tournament), winning lineups differ from the field on a few key
high-projection drivers who are low-owned. This module estimates ownership so
the optimizer can compute leverage (proj_dk / proj_own).

The model is a heuristic (not ML-trained) because we don't have historical
contest ownership data stored. Key features:

1. Value = projected points / salary (strongest value plays get chalky)
2. Salary percentile (mid-salary "value zone" gets most ownership)
3. Vegas implied win probability (chalk loves favorites — but not decisive)
4. PD narrative, track-type-aware (value chalk at intermediate/speedway,
   less weight at short tracks where dominators matter more)
5. Qualifying position (high qualifiers get SOME bump, but it's the narrative
   of starting deep + being cheap that really drives ownership)

Total ownership sums to roster_size * 100 (e.g. 600% for DK Cup — 6 drivers ×
100 lineups). A driver projected at 40% ownership means 4 out of every 10
lineups have him.
"""
from __future__ import annotations
import math
from typing import Iterable


def _american_to_prob(odds_str) -> float:
    """Convert American odds string/int to implied probability [0, 1].

    Accepts EVEN / EV / PK in addition to numeric forms (e.g. +350, -150).
    """
    from src.utils import parse_american_odds
    v = parse_american_odds(odds_str)
    if v is None:
        return 0.0
    if v > 0:
        return 100.0 / (v + 100.0)
    if v < 0:
        return -v / (-v + 100.0)
    return 0.0


def _pd_multiplier(delta: float, salary_pctile: float, track_type: str | None) -> float:
    """Track-type-aware place-differential multiplier for ownership.

    Rationale for directionality:
      - At intermediate / superspeedway / road: PD contributes meaningfully to
        DK score because LL+FL points are distributed. A cheap driver starting
        deep with projected gains is the classic "value chalk" narrative —
        casuals and every DFS content source flag him. Ownership goes UP.
      - At short tracks / short_concrete: dominators own most of the point pool.
        A driver starting deep won't catch the lead pack. PD matters less,
        and casual ownership on these value plays is more muted.
      - Expensive drivers (top 25% salary) with PD upside: casuals see them
        because they're expensive names, so they're MORE owned than a leverage
        model would suggest. Exception: at short tracks, a top salary fading
        from the front is real fade risk — ownership drops slightly.

    Args:
        delta: qual_pos - proj_finish (positive = gain positions = PD upside)
        salary_pctile: 0.0-1.0, percentile rank within this race's salaries
        track_type: "intermediate" / "short" / "superspeedway" / etc.

    Returns:
        Multiplier typically in [0.85, 1.20].
    """
    # Track sensitivity — how much PD drives ownership narrative
    if track_type in ("short", "short_concrete"):
        pd_sens = 0.5  # dampened; dominators dominate attention
    elif track_type == "intermediate_worn":
        pd_sens = 0.8
    else:  # intermediate, superspeedway, road, unknown
        pd_sens = 1.0

    is_cheap = salary_pctile <= 0.5
    is_mid   = 0.5 < salary_pctile <= 0.75
    is_top   = salary_pctile > 0.75

    # Big PD upside (delta >= 10 positions)
    if delta >= 10:
        if is_cheap:
            # Classic value chalk narrative — casuals pile on
            return 1.0 + 0.15 * pd_sens
        if is_mid:
            # Known mid-price driver starting deep — still attractive narrative
            return 1.0 + 0.08 * pd_sens
        # Top tier with PD upside — depends on track
        if track_type in ("short", "short_concrete"):
            # Genuinely under-owned leverage (casuals focused on obvious doms)
            return 0.92
        # At intermediate/speedway, casuals notice top names with PD stories
        return 1.0 + 0.04 * pd_sens

    # Moderate PD upside (delta 5-9)
    if delta >= 5:
        if is_cheap:
            return 1.0 + 0.06 * pd_sens
        if is_mid:
            return 1.0 + 0.03 * pd_sens
        return 1.0  # expensive + modest PD gain = no narrative pull

    # Big fade risk (delta <= -10 — starts front, projects deep)
    if delta <= -10:
        if is_top:
            # Casuals still roster front-row names even if we project a fade
            return 1.05
        if is_cheap:
            # Cheap + starting up front + projected fade = casuals skip entirely
            return 0.90
        return 0.98

    # Mild fade (-5 to -9)
    if delta <= -5:
        if is_top:
            return 1.02
        return 0.98

    return 1.0


def project_ownership(
    drivers: Iterable[str],
    proj_dk: dict,
    salary: dict,
    win_odds: dict | None = None,
    qual_pos: dict | None = None,
    proj_finish: dict | None = None,
    track_type: str | None = None,
    field_size: int = 37,
    roster_size: int = 6,
    gpp_dispersion: float = 1.0,
) -> dict:
    """Return projected ownership % per driver.

    Args:
        drivers: list of driver names in the pool
        proj_dk: {driver: projected DK points}
        salary: {driver: DK salary}
        win_odds: {driver: American odds string} (optional)
        qual_pos: {driver: qualifying/start position} (optional)
        proj_finish: {driver: projected finish position} (optional — used for
            PD-narrative ownership adjustment)
        track_type: track type (e.g. "intermediate") — tunes PD multiplier
        field_size: total drivers in the race
        roster_size: number of roster slots (6 for Cup DK, 5 for Truck)
        gpp_dispersion: 0.5-1.5 range. <1.0 flattens (less chalk), >1.0
            concentrates (more chalk). 1.0 is neutral baseline.

    Returns:
        {driver: ownership_pct_0_to_100}
        Total ownership = roster_size * 100 (so Cup DK sums to ~600%)
    """
    drivers = [d for d in drivers if d in proj_dk and d in salary]
    if not drivers:
        return {}

    # ── Component 1: Value (proj_dk / salary_k) ──
    # Computed within-tier so a value play beats similarly-priced drivers,
    # not only the expensive ones.
    values = {}
    for d in drivers:
        sal = salary.get(d, 0) or 0
        pts = proj_dk.get(d, 0) or 0
        if sal >= 3000:
            values[d] = pts / (sal / 1000.0)
        else:
            values[d] = 0.0
    max_val = max(values.values()) if values else 1.0
    min_val = min(values.values()) if values else 0.0
    val_range = max(max_val - min_val, 0.01)
    value_score = {d: (v - min_val) / val_range for d, v in values.items()}

    # ── Component 2: Vegas implied win probability ──
    # Don't zero this out for drivers without a quote — that unfairly punishes
    # value plays whom Vegas simply doesn't price for wins. Use the median
    # probability as a neutral default so their wp_score sits mid-pack.
    win_probs = {}
    neutral_wp = 0.0
    has_any_odds = bool(win_odds)
    if has_any_odds:
        raw = {d: _american_to_prob(win_odds.get(d)) for d in drivers}
        quoted = [p for p in raw.values() if p > 0]
        total_p = sum(quoted)
        if total_p > 0:
            # Remove overround — normalize to sum to 1.0 across quoted drivers
            win_probs = {d: (p / total_p if p > 0 else 0) for d, p in raw.items()}
            # For non-quoted drivers, impute the median of quoted probs divided
            # by 2 (below median — they're longshots Vegas wouldn't touch — but
            # not zero, which would wrongly wipe their component)
            if win_probs:
                quoted_vals = sorted([v for v in win_probs.values() if v > 0])
                if quoted_vals:
                    median = quoted_vals[len(quoted_vals) // 2]
                    neutral_wp = median * 0.5
                    for d in drivers:
                        if win_probs.get(d, 0) == 0:
                            win_probs[d] = neutral_wp
    max_wp = max(win_probs.values()) if win_probs else 1.0
    wp_score = {d: (win_probs.get(d, 0) / max_wp) if max_wp > 0 else 0 for d in drivers}

    # ── Component 3: Salary percentile (sweet spot = middle) ──
    # Ownership peaks at mid-salary where value plays live. Both ends underowned.
    salaries_sorted = sorted([salary[d] for d in drivers])
    n = len(salaries_sorted)
    sal_pctile = {s: i / max(n - 1, 1) for i, s in enumerate(salaries_sorted)}
    salary_score = {}
    for d in drivers:
        pct = sal_pctile.get(salary[d], 0.5)
        # Peak at ~0.55-0.65 percentile (slightly above median)
        salary_score[d] = math.exp(-((pct - 0.6) ** 2) / 0.08)

    # ── Component 4: Qualifying position ──
    qual_score = {}
    if qual_pos:
        for d in drivers:
            qp = qual_pos.get(d)
            if qp and qp > 0:
                # Front-row drivers get SOME bump (name + position chalk),
                # but not the dominant factor
                qual_score[d] = max(0.0, (field_size - qp) / field_size)
            else:
                qual_score[d] = 0.0

    # ── Combine ──
    # Tuned for "which drivers casuals pick up" — value and salary sweet-spot
    # drive chalk more than Vegas win odds do (since nobody bets $5k drivers
    # to win but plenty roster them).
    W_VALUE = 0.40  # was 0.35 — value is king for DFS ownership
    W_WIN   = 0.15  # was 0.25 — win-prob matters less than casuals think
    W_SAL   = 0.30  # was 0.25 — mid-salary zone is the chalk magnet
    W_QUAL  = 0.15  # unchanged

    has_qual = bool(qual_score)
    if not has_any_odds:
        # No Vegas data at all — shift its weight to value/salary
        W_VALUE += 0.10
        W_SAL   += 0.05
        W_WIN = 0.0
    if not has_qual:
        W_VALUE += W_QUAL * 0.5
        W_SAL   += W_QUAL * 0.5
        W_QUAL = 0.0

    # ── PD-upside ownership adjustment ──
    pd_multiplier = {}
    if proj_finish:
        for d in drivers:
            qp = qual_pos.get(d) if qual_pos else None
            pf = proj_finish.get(d)
            if qp is None or pf is None:
                pd_multiplier[d] = 1.0
                continue
            delta = qp - pf
            pct = sal_pctile.get(salary[d], 0.5)
            pd_multiplier[d] = _pd_multiplier(delta, pct, track_type)

    raw = {}
    for d in drivers:
        score = (
            value_score.get(d, 0) * W_VALUE
            + wp_score.get(d, 0) * W_WIN
            + salary_score.get(d, 0) * W_SAL
            + qual_score.get(d, 0) * W_QUAL
        )
        score *= pd_multiplier.get(d, 1.0)
        raw[d] = score ** gpp_dispersion

    # ── Normalize to sum to roster_size * 100 ──
    total_raw = sum(raw.values())
    if total_raw <= 0:
        return {d: 0.0 for d in drivers}

    target_sum = roster_size * 100.0
    own = {d: (raw[d] / total_raw) * target_sum for d in drivers}

    # Cap max ownership at 70% (realistic ceiling — no one is 100% owned).
    # Iterate cap/redistribute until stable.
    CAP = 70.0
    for _ in range(10):
        capped = {d: own[d] for d in drivers if own[d] > CAP}
        if not capped:
            break
        overflow = sum(v - CAP for v in capped.values())
        for d in capped:
            own[d] = CAP
        uncapped = [d for d in drivers if d not in capped]
        uncapped_sum = sum(own[d] for d in uncapped)
        if uncapped_sum <= 0:
            break
        for d in uncapped:
            own[d] += overflow * (own[d] / uncapped_sum)

    return {d: round(own[d], 1) for d in drivers}


def compute_leverage(proj_dk: dict, ownership: dict) -> dict:
    """Compute leverage = proj_dk / ownership_pct for each driver.

    High leverage = lots of points relative to ownership (GPP-optimal).
    Low leverage = chalk that everyone has.
    """
    out = {}
    for d, pts in proj_dk.items():
        own = ownership.get(d, 0)
        if own and own > 0.5:  # avoid divide-by-noise for 0.1% owned
            out[d] = round(pts / own, 2)
        else:
            out[d] = 0.0
    return out
