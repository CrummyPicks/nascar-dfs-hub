"""Projected ownership heuristic for NASCAR DFS contests.

Ownership % = the percentage of total entered lineups that roster a given driver.
In a GPP (tournament), winning lineups differ from the field on a few key
high-projection drivers who are low-owned. This module estimates ownership so
the optimizer can compute leverage (proj_dk / proj_own).

The model is a heuristic (not ML-trained) because we don't have historical
contest ownership data stored. Key features:

1. Implied win probability from Vegas odds (dominant signal — chalk tracks odds)
2. Value = projected points / salary (strongest value plays get chalky)
3. Salary percentile (mid-salary drivers in the "value zone" get most ownership)
4. Qualifying position (top-5 qualifier gets a bump, especially at short tracks)

Total ownership sums to roster_size * 100 (e.g. 600% for DK Cup — 6 drivers ×
100 lineups). A driver projected at 40% ownership means 4 out of every 10
lineups have him.
"""
from __future__ import annotations
import math
from typing import Iterable


def _american_to_prob(odds_str) -> float:
    """Convert American odds string/int to implied probability [0, 1]."""
    if odds_str is None:
        return 0.0
    try:
        v = int(str(odds_str).replace("+", ""))
    except (ValueError, TypeError):
        return 0.0
    if v > 0:
        return 100.0 / (v + 100.0)
    if v < 0:
        return -v / (-v + 100.0)
    return 0.0


def project_ownership(
    drivers: Iterable[str],
    proj_dk: dict,
    salary: dict,
    win_odds: dict | None = None,
    qual_pos: dict | None = None,
    proj_finish: dict | None = None,
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
        proj_finish: {driver: projected finish position} (optional — used to
            reduce ownership for drivers with strong PD upside, which the
            casual field tends to under-roster)
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
    # Normalize per-driver value into a "chalk score"
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

    # ── Component 2: Win probability from Vegas ──
    win_probs = {}
    if win_odds:
        raw = {d: _american_to_prob(win_odds.get(d)) for d in drivers}
        total_p = sum(raw.values())
        if total_p > 0:
            # Remove overround — normalize to sum to 1.0
            win_probs = {d: p / total_p for d, p in raw.items()}
    max_wp = max(win_probs.values()) if win_probs else 1.0
    wp_score = {d: (win_probs.get(d, 0) / max_wp) if max_wp > 0 else 0 for d in drivers}

    # ── Component 3: Salary percentile (sweet spot = middle $7-9k on DK) ──
    # Ownership peaks at mid-salary where value plays live. Both ends underowned.
    salaries_sorted = sorted([salary[d] for d in drivers])
    n = len(salaries_sorted)
    sal_rank = {s: i / max(n - 1, 1) for i, s in enumerate(salaries_sorted)}
    salary_score = {}
    for d in drivers:
        pct = sal_rank.get(salary[d], 0.5)
        # Peak at ~0.55-0.65 percentile (slightly above median)
        salary_score[d] = math.exp(-((pct - 0.6) ** 2) / 0.08)

    # ── Component 4: Qualifying position ──
    qual_score = {}
    if qual_pos:
        for d in drivers:
            qp = qual_pos.get(d)
            if qp and qp > 0:
                # Front-row drivers get ownership bump
                qual_score[d] = max(0.0, (field_size - qp) / field_size)
            else:
                qual_score[d] = 0.0

    # ── Combine ──
    # Weights chosen to match DFS intuition:
    #   - Value and Vegas odds dominate (60% combined)
    #   - Salary percentile captures "this driver is in the value zone" (25%)
    #   - Qualifying nudges short-track chalk (15%)
    W_VALUE = 0.35
    W_WIN = 0.25
    W_SAL = 0.25
    W_QUAL = 0.15

    has_odds = bool(win_probs)
    has_qual = bool(qual_score)
    if not has_odds:
        W_VALUE += 0.15
        W_SAL += 0.10
        W_WIN = 0.0
    if not has_qual:
        W_VALUE += W_QUAL * 0.5
        W_SAL += W_QUAL * 0.5
        W_QUAL = 0.0

    # ── PD-upside ownership adjustment ──
    # Casual DFS players don't model place differential. Drivers with strong
    # PD upside (projected to gain 10+ positions) tend to be *under-owned*
    # relative to their projected points. Apply a small dampener for these
    # drivers — they're the low-owned GPP plays that win tournaments.
    # Conversely, fade-risk drivers (start near the front, project back) are
    # *over-owned* because casuals see "top qualifier" and lock them in.
    pd_multiplier = {}
    if proj_finish:
        for d in drivers:
            qp = qual_pos.get(d) if qual_pos else None
            pf = proj_finish.get(d)
            if qp is None or pf is None:
                pd_multiplier[d] = 1.0
                continue
            delta = qp - pf  # positive = gain positions = PD upside
            if delta >= 10:
                # Big PD upside — field under-rosters these
                pd_multiplier[d] = 0.80
            elif delta >= 5:
                pd_multiplier[d] = 0.92
            elif delta <= -10:
                # Big fade risk — field over-rosters the front-row that fades
                pd_multiplier[d] = 1.15
            elif delta <= -5:
                pd_multiplier[d] = 1.05
            else:
                pd_multiplier[d] = 1.0

    raw = {}
    for d in drivers:
        score = (
            value_score.get(d, 0) * W_VALUE
            + wp_score.get(d, 0) * W_WIN
            + salary_score.get(d, 0) * W_SAL
            + qual_score.get(d, 0) * W_QUAL
        )
        # Apply PD-upside multiplier before dispersion
        score *= pd_multiplier.get(d, 1.0)
        # Apply GPP dispersion — raises score to a power. Higher power
        # concentrates ownership on top plays (chalk leaders go from 40%→50%).
        raw[d] = score ** gpp_dispersion

    # ── Normalize to sum to roster_size * 100 ──
    total_raw = sum(raw.values())
    if total_raw <= 0:
        return {d: 0.0 for d in drivers}

    target_sum = roster_size * 100.0
    own = {d: (raw[d] / total_raw) * target_sum for d in drivers}

    # Cap max ownership at 70% (realistic ceiling — no one is 100% owned).
    # Iterate the cap/redistribute process until stable — a single pass can
    # push other drivers over the cap when overflow is large relative to
    # uncapped population.
    CAP = 70.0
    for _ in range(10):  # bounded iteration
        capped = {d: own[d] for d in drivers if own[d] > CAP}
        if not capped:
            break
        overflow = sum(v - CAP for v in capped.values())
        for d in capped:
            own[d] = CAP
        uncapped = [d for d in drivers if d not in capped]
        uncapped_sum = sum(own[d] for d in uncapped)
        if uncapped_sum <= 0:
            # All drivers maxed — leave as-is
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
