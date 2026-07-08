"""Shared projection engine used by both Projections and Accuracy tabs.

This module contains the core projection logic that converts signal data
into DK point projections.  Both tabs call compute_projections() with
their own data-loading layer, guaranteeing identical math.
"""

import math
import numpy as np
from src.config import (
    DK_FINISH_POINTS, TRACK_TYPE_PARENT,
    is_concrete_track, CONCRETE_GATE_PROFILE,
)

# ── Deep-start dominator dampener, BY TRACK TYPE ──────────────────────────────
# Starting deep hurts your chance to LEAD LAPS far more at some tracks than
# others, so the dampener on a deep starter's dominator score is track-type
# aware instead of one-size-fits-all:
#   • short / short_concrete — track position is king and passing is hard, so a
#     deep start is nearly a death penalty for leading laps (steep, low floor).
#   • superspeedway — pack/draft racing cycles the lead through the whole field,
#     so where you start barely predicts who leads (nearly flat).
#   • intermediate / road — long green-flag runs, tire falloff and pit cycles
#     let a fast car work forward, so a moderate, gentle dampener.
# Values are (floor, slope): for a start beyond P10 the multiplier is
# max(floor, 1 - (start-10)*slope). P1-P3 get a small boost, P4-P10 are neutral.
# NOTE: As of the start-aware laps-led gate (_start_avail in this module),
# the laps-led MAGNITUDE suppression for deep starters lives in the allocator.
# This score-level multiplier is now SOFT — it only nudges dominator rank ORDER
# (which feeds dominator identification), so a deep starter isn't penalized twice
# (once in the score, once in the allocation). Floors raised toward 1.0 and slopes
# roughly halved vs the pre-gate values.
_DOM_START_PENALTY = {
    "superspeedway":     (0.96, 0.002),
    "road":              (0.88, 0.005),
    "intermediate":      (0.88, 0.005),
    "intermediate_worn": (0.88, 0.005),
    "short":             (0.85, 0.010),
    "short_concrete":    (0.85, 0.011),
}
_DOM_START_PENALTY_DEFAULT = (0.88, 0.005)
# Longer races give a deep starter more time to recover track position, so the
# per-position cut is softened as race length grows past this reference — e.g.
# the Charlotte 600 (400 laps) gets a gentler dampener than a 267-lap
# intermediate. NOT applied to short tracks, where passing stays hard no matter
# how long the race is (a 500-lap Bristol is still a track-position race).
_DOM_START_LONGRACE_REF = 320

# NASCAR Driver Rating blend into the history finish signals (track + track-type).
# When a driver has a rating, the projected finish from history is an explicit
# 40/30/30 weighted blend of: rating-pseudo-finish / avg running position / avg
# finish. Rating (0-150) is the richest single race-quality measure NASCAR
# publishes (folds in quality passes, top-15 laps, lead-lap finishes, fast laps)
# and is less wreck-distorted than raw finish, so it gets the plurality weight.
# Empirically finish ≈ 40.05 − 0.304·rating (Cup 2022+, r=−0.78).
_ARP_W    = 0.40   # average running position (strongest backtested predictor)
_RATING_W = 0.20   # NASCAR Driver Rating (as a pseudo-finish)
_AF_W     = 0.40   # average finish
# Note: rating is ~0.92 collinear with ARP, so heavier rating weight does NOT
# improve accuracy. On a DNF-filtered backtest (grading only running finishers,
# which removes wreck-luck noise) 40/20/40 tested best on deep starters
# (rho_deep 0.485 vs 0.477 at 30% rating). Rating kept at 20% as a genuine
# co-signal — it folds in quality passes / top-15 laps / lead-lap finishes ARP
# alone misses — with ARP + finish carrying the plurality. Revisit as more
# rated races accumulate (rating may decouple from ARP with more data).


def _rating_pseudo_finish(rating, field_size):
    """Convert a 0-150 NASCAR Driver Rating to a projected finish position
    (1..field_size, lower = better) via the empirical Cup 2022+ regression."""
    return max(1.0, min(float(field_size), 40.05 - 0.304 * rating))


def _history_finish(arp, avg_finish, rating, field_size, track_type):
    """Projected finish from history, blending rating / ARP / avg-finish.

    With a rating present: explicit 40/30/30 (rating / ARP / finish). When ARP
    is missing, its 30% is reallocated to rating (the better anchor) so we never
    blend toward a phantom mid-field. With NO rating (≈1% of rows: DNQs, the two
    unrated historical races): fall back to the prior track-type-aware ARP/finish
    blend (arp_finish_blend), leaving those drivers' projections unchanged."""
    from src.utils import arp_finish_blend
    if rating is None:
        return arp_finish_blend(arp, avg_finish, track_type)
    pseudo = _rating_pseudo_finish(rating, field_size)
    if arp is None:
        # No ARP — give its weight to rating (keep finish at its 30% share).
        rw, fw = _RATING_W + _ARP_W, _AF_W
        return (pseudo * rw + avg_finish * fw) / (rw + fw)
    return pseudo * _RATING_W + arp * _ARP_W + avg_finish * _AF_W


def _dom_start_multiplier(qp, race_laps, track_type):
    """Track-type-aware start-position multiplier on the dominator score."""
    parent = TRACK_TYPE_PARENT.get(track_type, track_type)
    floor, slope = _DOM_START_PENALTY.get(
        track_type, _DOM_START_PENALTY.get(parent, _DOM_START_PENALTY_DEFAULT))
    if qp <= 3:
        return 1.10 - (qp - 1) * 0.03            # 1.10 / 1.07 / 1.04
    if qp <= 10:
        return 1.0
    if parent not in ("short", "short_concrete") and race_laps and race_laps > _DOM_START_LONGRACE_REF:
        slope *= max(0.6, _DOM_START_LONGRACE_REF / race_laps)
    return max(floor, 1.0 - (qp - 10) * slope)


def _expected_finish_pts(pos):
    """DK finish points for integer position."""
    return DK_FINISH_POINTS.get(max(1, min(40, int(pos))), 0)


# ── Expected-finish DISTRIBUTION (replaces the strict integer running order) ───
# Each driver's continuous expected finish (raw_finish) is spread over a Gaussian
# of finishing positions, then the field's position-probability matrix is
# Sinkhorn-normalized so it stays coherent: every driver's row sums to 1 (they
# finish somewhere) AND every position's column sums to 1 (exactly one car per
# position in expectation — so wins / top finishes are conserved and nobody's
# ceiling vanishes). Projected finish POINTS and place differential then use the
# EXPECTED value over that distribution instead of one forced integer. This
# removes the front-over / back-under bias the strict order created.
#
# The spread is ASYMMETRIC ("ramp"), keyed to how strong the driver projects:
# tight at the FRONT (an elite car reliably converts, so its outcomes cluster →
# it projects up front with a small, realistic place differential), widening to
# full width by ~the top third and STAYING wide through the back (slow cars
# genuinely get vaulted up by attrition/cautions, so they keep that upside).
# Tightening the back too (a symmetric tent) measurably re-broke back calibration
# (bias −2.1 → −4.0), so only the front is tightened. Validated in
# experiment_hetero_sigma.py: favorite 10.0 → ~7.4, its PD −5.9 → −3.3, per-tier
# bias −0.1/+0.6/−2.1 (≈ flat-σ calibration), MAE +0.25. Laps-led / fast-laps
# scoring is untouched.
_FINISH_SIGMA_FRONT = 4.0    # spread for the projected race winner (tight)
_FINISH_SIGMA_WIDE  = 11.0   # full spread, reached by the front third onward
_FINISH_RAMP_KNEE   = 0.4    # fraction of the field by which σ reaches full width


def _ramp_sigma(center, n):
    """Per-driver finishing-position spread: narrow at the front, ramping to full
    width by the front ~40% of the field and flat (wide) thereafter."""
    if n <= 1:
        return _FINISH_SIGMA_WIDE
    frac = (min(max(center, 1.0), float(n)) - 1.0) / (n - 1)   # 0=front .. 1=back
    t = min(1.0, frac / _FINISH_RAMP_KNEE)
    return _FINISH_SIGMA_FRONT + (_FINISH_SIGMA_WIDE - _FINISH_SIGMA_FRONT) * t


# Explicit attrition lift (2026-07). The old kernel simply truncated each
# driver's Gaussian at the field edge, which dragged deep centers toward
# mid-field (center 34, sigma 11 in a 38-car field -> mean ~27.6). That
# truncation artifact WAS the back-half compression (drivers who actually
# finished 23rd/30th/38th all projected ~22) — but it was also accidentally
# simulating a real effect: slow cars DO get vaulted forward by others'
# attrition (a past experiment that tightened back sigma re-broke back
# calibration for exactly this reason). Fix = separate the two: the kernel
# below is MEAN-PRESERVING (spreading no longer moves the center), and the
# attrition lift is THIS explicit, tunable pull toward mid-field applied to
# back-half raw scores. Tuned via scripts/calibration_study.py.
_BACK_ATTRITION_PULL = 0.25


def _mean_preserving_center(target, sigma, n):
    """Center c* such that a Gaussian(c*, sigma) truncated to positions 1..n
    has MEAN == target. Plain truncation biases the mean toward mid-field at
    both edges; recentering removes that artifact so deep raw scores survive
    the spreading step. Monotone in c* -> binary search."""
    lo, hi = target - 6.0 * sigma, target + 6.0 * sigma
    ks = range(1, n + 1)
    for _ in range(36):
        c = (lo + hi) / 2.0
        wsum = msum = 0.0
        for k in ks:
            w = math.exp(-0.5 * ((k - c) / sigma) ** 2)
            wsum += w
            msum += k * w
        mean = msum / wsum if wsum > 0 else c
        if mean < target:
            lo = c
        else:
            hi = c
    return (lo + hi) / 2.0


def _finish_dist_expectations(raw_scores, drivers, field_size):
    """Return {driver: (expected_finish, expected_finish_points)} from each
    driver's continuous raw_finish, via a Sinkhorn-normalized Gaussian finish
    distribution over positions 1..field_size with a front-tight 'ramp' spread.

    The kernel is mean-preserving (see _mean_preserving_center) and the
    attrition upside of slow cars is an explicit raw-space pull
    (_BACK_ATTRITION_PULL) instead of a truncation accident."""
    order = list(drivers)
    n = max(1, int(field_size))
    if n == 1 or not order:
        return {d: (1.0, _expected_finish_pts(1)) for d in order}
    mid = (n + 1) / 2.0
    # Row = driver, col = finishing position (1..n); Gaussian recentered so
    # its truncated mean equals the (attrition-adjusted) raw expected finish.
    mat = []
    for d in order:
        c = max(1.0, min(float(n), raw_scores.get(d, n * 0.75)))
        if c > mid:
            # BACK half: explicit attrition lift, then a mean-preserving
            # kernel so deep raw scores survive the spreading step (the old
            # truncation dragged center-34 to ~27.6 — the measured blob).
            c = c - _BACK_ATTRITION_PULL * (c - mid)
            sigma = _ramp_sigma(c, n)
            c_star = _mean_preserving_center(c, sigma, n)
        else:
            # FRONT half: keep the original truncated kernel. Its mean-pull
            # toward mid was accidentally CALIBRATING favorites (validated by
            # past tuning; recentering the front measurably worsened front
            # points bias +3.5 -> +5.8 in the calibration study).
            sigma = _ramp_sigma(c, n)
            c_star = c
        row = [math.exp(-0.5 * ((k - c_star) / sigma) ** 2) for k in range(1, n + 1)]
        s = sum(row) or 1.0
        mat.append([x / s for x in row])
    # Sinkhorn: alternately normalize rows → 1 and columns → 1.
    R = len(mat)
    for _ in range(20):
        for i in range(R):
            s = sum(mat[i]) or 1.0
            mat[i] = [x / s for x in mat[i]]
        for j in range(n):
            cs = sum(mat[i][j] for i in range(R)) or 1.0
            for i in range(R):
                mat[i][j] /= cs
    pos_pts = [_expected_finish_pts(k) for k in range(1, n + 1)]
    out = {}
    for i, d in enumerate(order):
        s = sum(mat[i]) or 1.0
        w = [x / s for x in mat[i]]          # final row-normalize → proper pmf
        e_finish = sum((k + 1) * w[k] for k in range(n))
        e_pts = sum(pos_pts[k] * w[k] for k in range(n))
        out[d] = (e_finish, e_pts)
    return out


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


def odds_expected_finish(odds_probs: dict, field_size: int = None) -> dict:
    """Win odds → expected finish via Bradley-Terry pairwise strengths.

    E[rank_i] = 1 + Σ_j P(j beats i), with P(j beats i) = w_j/(w_i+w_j) and
    w = implied win probability. Replaces the old linear log-odds squash into
    [0.13n, 0.58n], which had two unfixable flaws the anchors couldn't tune
    away: (1) it could never say "worse than ~22nd" in a 38-car field — a
    +250,000 no-hoper projected the same as a decent 25th-place car (measured:
    drivers who actually finished 23rd/30th/38th all carried ~22 projections);
    (2) any DEEPER anchor (0.82 was tried) pinned every longshot at the wall
    together, burying +5000 value plays next to +250000 no-hopers — which is
    why it was pulled in to 0.58 in the first place. Strength RATIOS fix both:
    sharp separation among favorites, a genuinely deep tail, and intra-tail
    discrimination (a 50x odds ratio between two longshots = many positions).

    Validated on 37 replayed races: Spearman vs actual finish 0.495 (old map
    0.468), monotone calibration through the 30s where the old map's deepest
    bucket held drivers averaging 28th.

    Only quoted drivers get a value; ranks are within the QUOTED pool (when
    the book quotes fewer cars than the field, the unquoted sit behind — the
    engine's absent-from-Vegas penalty handles them). field_size is accepted
    for signature clarity/future scaling but not required by the math.
    """
    if not odds_probs:
        return {}
    names = list(odds_probs)
    w = [max(float(odds_probs[n] or 0.0), 1e-6) for n in names]
    out = {}
    for i, nm in enumerate(names):
        wi = w[i]
        s = 0.0
        for j, wj in enumerate(w):
            if j != i:
                s += wj / (wi + wj)
        out[nm] = 1.0 + s
    return out


def compute_projections(
    drivers, field_size, wn,
    th_data, tt_data, qual_pos, practice_data,
    odds_finish, odds_display, team_signal,
    mfr_adjustment, team_adj_data, dnf_data,
    race_laps, track_name, track_type, series_id,
    calibration, cross_th_lookup=None,
    return_signal_details=False,
    grid_start=None,
):
    """Run the full projection engine.

    Returns (proj_rows, proj_detail) where:
      proj_rows: list of dicts with per-driver projection data
      proj_detail: {driver: {proj_finish, start, laps_led, fast_laps}}

    If return_signal_details=True, returns (proj_rows, proj_detail, signal_details)
    where signal_details = {driver: {"Track": val, "Odds": val, ...}} with
    normalized signal values and adjustment info for display.
    """
    from src.utils import fuzzy_match_name, arp_finish_blend

    if cross_th_lookup is None:
        cross_th_lookup = {}

    mid_field = field_size * 0.5

    # Concrete surface (Nashville/Dover/Bristol) concentrates laps led and makes
    # advancing from deep starts slow — like a short concrete track regardless of
    # the track's SIZE. So for the laps-led concentration, the deep-start score
    # penalty and the start-availability gate, a concrete track uses the
    # short_concrete profile even though its track_type stays "intermediate"
    # (Nashville) for finish projection and All-Intermediate history. This is the
    # lever that makes the model race Nashville like concrete, not like Kansas.
    gate_track_type = (CONCRETE_GATE_PROFILE if is_concrete_track(track_name)
                       else track_type)

    # Track history reaches full trust at 1 race. Most ovals run only once a
    # year, so requiring 5 races meant "5 years" before a driver's own track
    # results were trusted — far too punitive. A driver with 0 track races
    # gets no track signal at all (track-type takes over); with 1+ they're
    # trusted on their actual history.
    MIN_RACES_TRACK = 1
    # Track type is a broader signal (aggregates many tracks) and usually has
    # plenty of races, so keep a small regression for genuinely thin samples.
    MIN_RACES_TTYPE = 3
    # Back-compat alias (still referenced by dominator/laps-led code below).
    MIN_RACES_FULL_TRUST = MIN_RACES_TRACK

    # Count how many drivers in the field have Vegas odds quoted. Used for the
    # "absent from Vegas" penalty: if Vegas quoted 15+ drivers but skipped this
    # one, that's informative (Vegas won't even price him) — apply extra penalty.
    _odds_quoted_count = sum(1 for _d in drivers if odds_finish.get(_d))

    # ── Pass 1: Gather raw signal values per driver ──
    raw_signals = {}
    signal_weight_map = {}
    sig_extras = {}  # per-driver display extras (Team Adj, etc.)

    SIG_DISPLAY = {"track": "Track", "ttype": "TType", "qual": "Qual",
                   "team": "Team", "prac": "Prac", "odds": "Odds"}

    for d in drivers:
        th = th_data.get(d)
        tt = tt_data.get(d)
        qp = qual_pos.get(d)
        pr = practice_data.get(d) if practice_data else None
        od = odds_finish.get(d)

        sigs = {}
        sig_w = {}
        extras = {}

        # Track history signal
        if th and wn.get("track", 0) > 0:
            races = th.get("races", 1)
            cross = cross_th_lookup.get(d)
            cross_races = cross["races"] if cross and cross.get("races") else 0
            effective_races = races + cross_races * 0.5
            trust = min(1.0, effective_races / MIN_RACES_TRACK)
            if th.get("_cross_series_only"):
                trust *= 0.8

            arp = th.get("avg_running_pos")
            af = th["avg_finish"]
            # Projected finish from THIS-TRACK history: 40/30/30 blend of NASCAR
            # Driver Rating / avg running position / avg finish (see _history_finish).
            base_finish = _history_finish(arp, af, th.get("th_rating"),
                                          field_size, track_type)

            t_adj = team_adj_data.get(d) if team_adj_data else None
            if t_adj and t_adj.get("team_adj", 0) != 0:
                base_finish = base_finish + t_adj["team_adj"]
                extras["Team Adj"] = round(t_adj["team_adj"], 1)

            regressed = base_finish * trust + mid_field * (1 - trust)
            sigs["track"] = regressed
            sig_w["track"] = wn["track"]

        # Track type — absorb track weight if no track history
        tt_weight = wn.get("track_type", 0)
        if not th and tt and wn.get("track", 0) > 0:
            tt_weight = wn.get("track_type", 0) + wn.get("track", 0)

        if tt and tt_weight > 0:
            tt_races = tt.get("races", 1) if isinstance(tt, dict) and "races" in tt else 3
            tt_trust = min(1.0, tt_races / MIN_RACES_TTYPE)
            if isinstance(tt, dict) and tt.get("_cross_series_only"):
                tt_trust *= 0.8
            tt_arp = tt.get("avg_running_pos") if isinstance(tt, dict) else None
            tt_af = tt.get("avg_finish", mid_field) if isinstance(tt, dict) else mid_field
            tt_rating = tt.get("tt_rating") if isinstance(tt, dict) else None
            # Same 40/30/30 rating/ARP/finish blend as the single-track signal.
            tt_avg = _history_finish(tt_arp, tt_af, tt_rating, field_size, track_type)
            # Apply the SAME team-change adjustment used for track history:
            # a driver's track-type form was also earned in past equipment, so
            # moving to a better/worse team should shift it too (their team
            # progression is the same across tracks of a type). Previously only
            # the single-track signal got this, so a team-changer's broader
            # track-type form was left un-adjusted.
            tt_adj = team_adj_data.get(d) if team_adj_data else None
            if tt_adj and tt_adj.get("team_adj", 0) != 0:
                tt_avg = tt_avg + tt_adj["team_adj"]
                extras.setdefault("Team Adj", round(tt_adj["team_adj"], 1))
            tt_regressed = tt_avg * tt_trust + mid_field * (1 - tt_trust)
            sigs["ttype"] = tt_regressed
            sig_w["ttype"] = tt_weight

        # Qualifying signal.
        # When we have history: regress slightly toward mid-field (qual tends to
        # overstate race pace). When we have NO history, we should NOT regress
        # toward mid-field — that would inflate inexperienced drivers' projections
        # by blending them with ghost-average performers. Instead, anchor on the
        # qualifying position and regress slightly toward the BACK of the field,
        # since rookies with no track data typically fade in race conditions.
        has_real_history = (th and th.get("races", 0) >= 2) or \
                           (tt and isinstance(tt, dict) and tt.get("races", 0) >= 3)
        if qp and qp <= field_size and wn.get("qual", 0) > 0:
            if has_real_history:
                qual_finish = qp * 0.80 + mid_field * 0.20
            else:
                # No experience anchor — project slightly worse than start,
                # never better. Weight toward back of field (field_size * 0.85).
                back_field = field_size * 0.85
                qual_finish = qp * 0.70 + back_field * 0.30
            sigs["qual"] = qual_finish
            sig_w["qual"] = wn["qual"]

        # Team signal — auto-scaled by driver's track-history sample size.
        # Rationale: track_history already bakes in the equipment the driver
        # has been racing, so for a veteran on the same team the team
        # signal is mostly a redundant echo (double-counting). But for
        # rookies / team-changers with thin track history, team quality
        # is a vital proxy for what their car can do.
        #
        # Scale factor by race count at this specific track:
        #   0 races       -> 1.3x  (no track data — team signal is primary)
        #   1-3 races     -> 0.9x  (small sample — still want team input)
        #   4-7 races     -> 0.6x
        #   8+ races      -> 0.3x  (rich history — team info is already in it)
        #
        # Redistribution: when team weight is scaled DOWN, the savings are
        # added directly to the odds weight (Vegas is the most reliable
        # independent signal). When scaled UP for rookies, the extra comes
        # from normal redistribution (no track history means track/ttype
        # aren't contributing much anyway).
        team_savings_for_odds = 0.0
        tm = team_signal.get(d) if team_signal else None
        if tm is not None and wn.get("team", 0) > 0:
            track_races = (th.get("races", 0) if th else 0)
            if track_races >= 8:
                team_scale = 0.30
            elif track_races >= 4:
                team_scale = 0.60
            elif track_races >= 1:
                team_scale = 0.90
            else:
                team_scale = 1.30
            sigs["team"] = tm
            sig_w["team"] = wn["team"] * team_scale
            # Capture the savings (if any) to be routed to odds below
            if team_scale < 1.0:
                team_savings_for_odds = wn["team"] * (1.0 - team_scale)

        # Practice signal
        if pr:
            prac_finish = pr * 0.70 + mid_field * 0.30
            sigs["prac"] = prac_finish
            sig_w["prac"] = wn.get("practice", 0)

        # Odds signal. Absorbs any team-scale savings so reduced team
        # weight becomes increased Vegas weight (most reliable alternative).
        # When driver has no history, don't blend toward mid-field
        # (which helps longshots too much). Let odds stand on their own.
        if od and wn.get("odds", 0) > 0:
            odds_val = od
            sigs["odds"] = odds_val
            sig_w["odds"] = wn["odds"] + team_savings_for_odds
        elif team_savings_for_odds > 0 and wn.get("track", 0) > 0 and "track" in sig_w:
            # Fallback: if no odds for this driver, route team savings to
            # track history instead (next best independent signal)
            sig_w["track"] = sig_w["track"] + team_savings_for_odds

        raw_signals[d] = sigs
        signal_weight_map[d] = sig_w
        sig_extras[d] = extras

    # ── Pass 2: Normalize each signal to [1, field_size] ──
    # All "position-like" signals (track, ttype, odds, qual, prac) are
    # already in finish-position units and get CLAMPed to the valid range.
    # Previously we MINMAX-stretched track/ttype/odds to span the full field
    # which over-amplified narrow-spread signals — e.g., at Talladega
    # where track history naturally clusters 11-29 (spread 18), MINMAX
    # stretched that to 1-37 and doubled the discriminating power.
    # Clamp-only is more honest: if a signal has naturally narrow spread,
    # it should contribute less to the weighted combination, not be
    # artificially boosted.
    #
    # Team signal is still rank-normalized because team avg_finish values
    # (typically clustering 14-22) don't span the full field naturally
    # and rank-based makes a good team quality proxy.
    signal_names = set()
    for sigs in raw_signals.values():
        signal_names.update(sigs.keys())

    RANK_SIGNALS = {"team"}  # team quality → rank-based

    normalized_signals = {d: {} for d in drivers}
    for sig_name in signal_names:
        sig_vals = [(d, raw_signals[d][sig_name]) for d in drivers if sig_name in raw_signals[d]]
        if not sig_vals:
            continue

        if sig_name in RANK_SIGNALS:
            sig_vals.sort(key=lambda x: x[1])
            n_with_sig = len(sig_vals)
            for rank_idx, (d, _) in enumerate(sig_vals):
                if n_with_sig > 1:
                    normalized_signals[d][sig_name] = 1 + (field_size - 1) * (rank_idx / (n_with_sig - 1))
                else:
                    normalized_signals[d][sig_name] = mid_field
        else:
            # Position-like signal — clamp to valid range, preserve magnitude
            for d, val in sig_vals:
                normalized_signals[d][sig_name] = max(1, min(field_size, val))

    # ── Pass 3: Weighted average + adjustments ──
    driver_raw_scores = {}
    dom_raw_scores = {}
    fl_raw_scores = {}
    driver_signal_details = {}
    dnf_prob_map = {}   # driver -> P(DNF this race), for the DNF-aware floor

    ll_ref = calibration.get("avg_top_leader", race_laps * 0.35) if calibration else race_laps * 0.35

    # Max positions qualifying can differ from the driver's race-pace signals
    # before we pull it back toward pace. This is TRACK-TYPE AWARE: qualifying
    # predicts the finish only as well as track position holds in the race.
    #   • Intermediate / superspeedway: cars pass easily, deep starters
    #     routinely race up to their true pace — so a bad (or fluke-good) qual
    #     shouldn't drag the projection far. Tighter cap (5).
    #   • Short / road: track position sticks (hard to pass), so qualifying is
    #     genuinely predictive and allowed to stand further from pace (8).
    # This directly serves the cheap "starts deep, decent race pace" value play
    # on intermediates — its bad qual gets pulled toward its real pace instead
    # of pinning the projection near the back.
    _QUAL_CAP_BY_TYPE = {
        "intermediate": 5, "intermediate_worn": 5, "superspeedway": 5,
        "short": 8, "short_concrete": 8, "road": 8,
    }
    QUAL_DAMPEN_CAP = _QUAL_CAP_BY_TYPE.get(track_type, 7)

    for d in drivers:
        norm = normalized_signals[d]
        weights = signal_weight_map[d]
        sig_detail = dict(sig_extras.get(d, {}))

        # BIDIRECTIONAL qual dampener: qualifying is a single-lap signal —
        # when it's far from the driver's race-pace signals (track, ttype,
        # practice, odds), it's likely noise and should be pulled toward pace.
        #
        # Too much BETTER than pace:  fluke fast lap, not real race speed.
        # Too much WORSE than pace:   bad qualifying, race pace will reassert.
        #   (This is the "Hamlin starts P25 with avg finish P8" case — his
        #    P10 race pace shouldn't be dragged down to P18 by the qual.)
        #
        # We only trust the dampener when the driver has enough race-pace
        # signals to form a reliable anchor. Team alone isn't enough (it's
        # context, not driver-specific), so require track or ttype or prac
        # or odds to be present.
        RACE_PACE_SIGNALS = {"track", "ttype", "prac", "odds"}
        pace_sigs = [(norm[s], weights.get(s, 0))
                     for s in norm if s in RACE_PACE_SIGNALS]
        if "qual" in norm and pace_sigs:
            pace_total_w = sum(w for _, w in pace_sigs)
            if pace_total_w > 0:
                pace_avg = sum(v * w for v, w in pace_sigs) / pace_total_w
                qual_val = norm["qual"]
                gap = pace_avg - qual_val  # positive = qual is better (smaller pos)
                if gap > QUAL_DAMPEN_CAP:
                    # Qual too optimistic — pull toward pace
                    norm["qual"] = max(1, pace_avg - QUAL_DAMPEN_CAP)
                elif gap < -QUAL_DAMPEN_CAP:
                    # Qual too pessimistic — pull toward pace
                    norm["qual"] = min(field_size, pace_avg + QUAL_DAMPEN_CAP)

        finish_signals = []
        signal_weights = []
        raw_signals_for_d = raw_signals.get(d, {})
        for sig_name in norm:
            finish_signals.append(norm[sig_name])
            signal_weights.append(weights.get(sig_name, 0))
            # Display: for "team" we show the RAW team_signal value (in finish-position
            # units, comparable to other Sig columns) — not the rank-normalized value
            # which would be unintelligible mixed with the others. The rank-normalized
            # value is still what feeds into the weighted average internally.
            if sig_name == "team" and sig_name in raw_signals_for_d:
                sig_detail[SIG_DISPLAY.get(sig_name, sig_name)] = round(raw_signals_for_d[sig_name], 1)
            else:
                sig_detail[SIG_DISPLAY.get(sig_name, sig_name)] = round(norm[sig_name], 1)

        if finish_signals and sum(signal_weights) > 0:
            total_w = sum(signal_weights)
            raw_finish = sum(f * w for f, w in zip(finish_signals, signal_weights)) / total_w
        else:
            raw_finish = field_size * 0.75

        # Net Sig: the weighted-average finish position that drives proj_finish
        # rank-ordering. This is what ALL the signal weights net out to before
        # any low-info penalty / mfr / DNF adjustments. Lower = better projected
        # finish. Useful for sanity-checking that the aggregation isn't being
        # skewed by any one signal — e.g. if Sig Track says P5 and Sig Odds
        # says P5 but Net Sig comes out at P15, something is wrong.
        sig_detail["Net Sig"] = round(raw_finish, 1)

        # Low-information penalty: when a driver has few quality *driver-specific*
        # signals, blend the projection toward back-of-field proportional to how
        # little data we have. Without this, a driver with only a qualifying
        # position (no history, no odds, no practice) gets that one signal at
        # 100% confidence — inflating inexperienced drivers' projections.
        #
        # BUG FIX: this previously referenced `sigs` from Pass 1, which held
        # whatever the LAST driver in that loop had — unpredictable. Now uses
        # `norm` (the current driver's normalized signal dict).
        #
        # Weighting rule:
        #   - track, track_type, practice, odds = "driver-specific" (count fully)
        #   - a well-sampled track_type (4+ races) counts as 1.5 — a real body
        #     of form at this type of track is a reliable signal, so a value
        #     play with it shouldn't be branded "low info" merely for lacking
        #     THIS track's history or a Vegas quote (the Lavar Scott case).
        #   - qual = counts as 0.5 (single lap, doesn't prove race pace)
        #   - team = counts as 0.5 (tells us about equipment, not the driver)
        driver_specific = {"track", "prac", "odds"}  # ttype handled separately
        partial_signals = {"qual", "team"}
        tt_info = tt_data.get(d) if tt_data else None
        tt_races_ct = tt_info.get("races", 0) if isinstance(tt_info, dict) else 0
        if "ttype" in norm:
            ttype_credit = 1.5 if tt_races_ct >= 4 else 1.0
        else:
            ttype_credit = 0.0
        signal_weight_score = (
            ttype_credit
            + sum(1.0 for s in norm if s in driver_specific)
            + sum(0.5 for s in norm if s in partial_signals)
        )
        # Absent-from-Vegas penalty: when most of the field has odds but this
        # driver doesn't, Vegas is mildly signaling they don't rate him. But
        # books routinely skip the cheapest cars regardless of merit, so this
        # is a soft nudge (−0.25), not a hammer — otherwise legit value plays
        # get double-penalized (no odds AND a back-field anchor).
        vegas_skipped = (_odds_quoted_count >= 15 and not odds_finish.get(d))
        if vegas_skipped:
            signal_weight_score = max(0.0, signal_weight_score - 0.25)

        # Full trust only at 3.0+ signal-weight (e.g. track + ttype + odds,
        # OR track + ttype + prac, OR odds + prac + qual + team).
        confidence = min(1.0, signal_weight_score / 3.0)
        if confidence < 1.0:
            # Anchor the missing-information mass at field*0.80 (a back-half
            # car), not field*0.90 (near-DFL). A thin-data driver is more
            # likely a mid-to-back runner than a guaranteed tail-ender.
            back_field_anchor = field_size * 0.80
            raw_finish = raw_finish * confidence + back_field_anchor * (1 - confidence)
            sig_detail["LowInfo"] = f"{signal_weight_score:.1f}sig"
            if vegas_skipped:
                sig_detail["LowInfo"] += " (no-odds)"

        # Manufacturer adjustment
        mfr_adj = mfr_adjustment.get(d, 0) if mfr_adjustment else 0
        raw_finish = raw_finish + mfr_adj
        if mfr_adj != 0:
            sig_detail["Mfr"] = round(mfr_adj, 1)

        # DNF risk adjustment
        dnf = dnf_data.get(d) if dnf_data else None
        if not dnf and dnf_data:
            dnf_matched = fuzzy_match_name(d, list(dnf_data.keys()))
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
            # P(DNF this race) for the DNF-aware floor (capped — even wreck-prone
            # drivers finish most weeks). Used to drag the FLOOR down so cash mode
            # avoids blow-up risk; the median is left alone.
            dnf_prob_map[d] = min(0.35, dnf["dnf_rate"])

        driver_raw_scores[d] = raw_finish
        driver_signal_details[d] = sig_detail

        # ── Dominator scoring ──
        th = th_data.get(d)
        tt = tt_data.get(d)
        # Dominator / fast-lap scoring keys off the ACTUAL race-day grid: a driver
        # sent to the rear (unapproved adjustments, backup car, failed inspection)
        # won't lead early laps or run up front, no matter where he qualified. The
        # qualifying position still drives the FINISH qual signal (pace proxy) and
        # place differential below — DK scores PD off the qualified grid.
        qp = (grid_start.get(d) if grid_start else None) or qual_pos.get(d)
        pr = practice_data.get(d) if practice_data else None
        od = odds_finish.get(d)

        dom_score = 0.0
        if race_laps > 0:
            dom_signals = []
            dom_weights_list = []

            # Per-signal rule: if the user's SLIDER for a signal is > 0, that
            # signal's WEIGHT always goes into the denominator; the VALUE is
            # the real signal when data is available and 0 ("no evidence of
            # dominance from this signal") when it isn't. This matters because
            # the old "skip both signal and weight on missing data" pattern
            # silently INFLATED data-poor drivers: a contender with no track
            # LL history and no ttype LL history had a smaller weight sum, so
            # the same odds + contender contribution produced a higher dom_score
            # than a driver with real LL history. The fix gives the missing-
            # data driver a true zero on those signals (not a free pass) without
            # reintroducing the old 5.0 backmarker floor.
            #
            # Laps-led HISTORY is the most direct "does this driver dominate
            # HERE" signal, so it gets a 1.5x boost vs its finish-side weight.
            _DOM_LL_HISTORY_BOOST = 1.5
            if wn.get("track", 0) > 0:
                if th and th.get("races", 0) >= 1 and th.get("laps_led", 0) > 0:
                    ll_per_race = th["laps_led"] / th["races"]
                    ll_norm = min(100, (ll_per_race / max(ll_ref, 1)) * 100)
                else:
                    ll_norm = 0.0
                dom_signals.append(ll_norm)
                dom_weights_list.append(wn["track"] * _DOM_LL_HISTORY_BOOST)

            # Track type laps led — same per-signal rule.
            if wn.get("track_type", 0) > 0:
                if tt and isinstance(tt, dict) and tt.get("laps_led_per_race", 0) > 0:
                    tt_ll = tt["laps_led_per_race"]
                    tt_ll_norm = min(100, (tt_ll / max(ll_ref, 1)) * 100)
                else:
                    tt_ll_norm = 0.0
                dom_signals.append(tt_ll_norm)
                dom_weights_list.append(wn["track_type"] * _DOM_LL_HISTORY_BOOST)

            # Qualifying — a WEAK predictor of laps led. Real dominators come
            # from all over the grid (e.g. at Charlotte, Chastain led 153 from
            # P22, Blaney 163 from P8), so qual gets HALF its finish weight in
            # the dominator and the start multiplier below is gentle. Laps-led
            # HISTORY + pace (odds) should decide who dominates — not the grid.
            if wn.get("qual", 0) > 0:
                if qp and qp <= field_size:
                    qual_dom = max(0, (field_size + 1 - qp) / field_size) ** 1.5 * 100
                else:
                    qual_dom = 0.0
                dom_signals.append(qual_dom)
                dom_weights_list.append(wn["qual"] * 0.5)

            # Odds → leading laps correlates with WIN probability, not expected
            # finish. Use implied win % directly. CRITICAL: impl_pct legitimately
            # rounds to 0.0 for longshots, so test `is not None`, NOT truthiness.
            # The old truthy check sent every longshot into the odds_finish
            # fallback below, and since odds_finish is compressed toward mid-
            # field (worst anchor ~0.58*field), that handed +250000 longshots
            # ~34% dominator credit — the P37 car projected to lead 10+ laps.
            if wn.get("odds", 0) > 0:
                if od:
                    odds_info = odds_display.get(d) if odds_display else None
                    impl = odds_info.get("impl_pct") if odds_info else None
                    if impl is not None:
                        max_impl = max((v.get("impl_pct", 0) for v in odds_display.values()), default=1)
                        odds_dom = min(100, (impl / max(max_impl, 1)) * 100)
                    else:
                        odds_dom = max(0, (field_size + 1 - od) / field_size) ** 1.3 * 100
                else:
                    odds_dom = 0.0
                dom_signals.append(odds_dom)
                dom_weights_list.append(wn["odds"])

            # Practice
            if wn.get("practice", 0) > 0:
                if pr and practice_data:
                    max_p_val = max(practice_data.values()) if practice_data else field_size
                    prac_dom = max(0, (max_p_val + 1 - pr) / max_p_val) * 100
                else:
                    prac_dom = 0.0
                dom_signals.append(prac_dom)
                dom_weights_list.append(wn["practice"])

            # Contender signal — derived from raw_finish (which already
            # integrates ALL the finish-side inputs: history avg-finish/ARP,
            # odds, team, recency, qual, practice). A driver projected to
            # finish well is by definition expected to spend time near the
            # front, where laps led happen. Without this, top-finish contenders
            # whose specific TRACK LL HISTORY was thin (a newer driver, or a
            # rare-on-the-calendar venue) collapsed to dom-rank 5+ on the
            # empirical curve and got <1 lap led — while drivers projected
            # 20th could float up to dom-rank 1 on an outlier LL signal. This
            # connects the finish projection to the laps-led projection so
            # they tell a consistent story.
            _DOM_CONTENDER_WEIGHT = 0.30
            contender_base = max(0.0, min(1.0, (field_size + 1 - raw_finish) / field_size))
            contender_dom = (contender_base ** 1.5) * 100
            dom_signals.append(contender_dom)
            dom_weights_list.append(_DOM_CONTENDER_WEIGHT)

            if dom_signals:
                total_dw = sum(dom_weights_list)
                dom_score = sum(s * w for s, w in zip(dom_signals, dom_weights_list)) / total_dw
            else:
                dom_score = max(0, (field_size - raw_finish) / field_size) * 5

        # ── Fastest laps scoring ──
        fl_score = 0.0
        if race_laps > 0:
            fl_signals = []
            fl_signal_weights = []

            if dom_score > 0:
                fl_signals.append(dom_score * 0.5)
                fl_signal_weights.append(0.25)

            if qp and qp <= field_size and wn.get("qual", 0) > 0:
                qual_fl = max(0, (field_size + 1 - qp) / field_size) * 100
                fl_signals.append(qual_fl)
                fl_signal_weights.append(wn.get("qual", 0.15))

            if pr and practice_data:
                max_p_val = max(practice_data.values()) if practice_data else field_size
                prac_fl = max(0, (max_p_val + 1 - pr) / max_p_val) * 100
                fl_signals.append(prac_fl)
                fl_signal_weights.append(wn.get("practice", 0.10))

            # Same win-probability fix as the dominator odds signal: use
            # impl_pct via `is not None` so longshots (impl rounds to 0.0)
            # don't fall into the compressed-odds_finish fallback and get
            # spurious fast-lap credit.
            if od and wn.get("odds", 0) > 0:
                odds_info = odds_display.get(d) if odds_display else None
                impl = odds_info.get("impl_pct") if odds_info else None
                if impl is not None:
                    max_impl = max((v.get("impl_pct", 0) for v in odds_display.values()), default=1)
                    odds_fl = min(100, (impl / max(max_impl, 1)) * 100)
                else:
                    odds_fl = max(0, (field_size + 1 - od) / field_size) * 100
                fl_signals.append(odds_fl)
                fl_signal_weights.append(wn.get("odds", 0.15))

            finish_fl = max(0, (field_size - raw_finish) / field_size) * 100
            fl_signals.append(finish_fl)
            fl_signal_weights.append(0.10)

            if fl_signals:
                total_fw = sum(fl_signal_weights)
                fl_score = sum(s * w for s, w in zip(fl_signals, fl_signal_weights)) / total_fw

        # Start-position multiplier on the dominator score — TRACK-TYPE AWARE
        # (see _dom_start_multiplier). A deep start is nearly a death penalty
        # for leading laps at short tracks, barely matters at superspeedways,
        # and is a gentle dampener at intermediates/road courses (softened
        # further for long races like the Charlotte 600). This keeps a deep-
        # starting track ace a real lap-leader threat at a long intermediate
        # while correctly burying one at a short track. Pairs with the soft-
        # rank allocator so the resulting laps are also stable to small weight
        # changes.
        if qp and qp <= field_size and dom_score > 0:
            dom_score = dom_score * _dom_start_multiplier(qp, race_laps, gate_track_type)

        dom_raw_scores[d] = dom_score
        fl_raw_scores[d] = fl_score

    # ── Finish position: EXPECTED value over a distribution ──
    # (Previously every driver was forced to a UNIQUE integer 1..field_size by
    # raw_finish rank, then points were read off that single position. That
    # manufactured spread for clustered drivers and biased the front tier up /
    # back tier down — see _finish_dist_expectations.) The expectations are
    # computed AFTER the allocators below, just before DK assembly. The low-info
    # penalty is already baked into driver_raw_scores, so it still pulls thin
    # drivers toward the back via their distribution center.

    # ── Resolve each driver's projected START for the laps-led start gate ──
    # Same fallback chain used below for the displayed start (qual → track-history
    # avg start → track-type avg start → odds → mid-field), lifted here so the
    # allocator can damp a deep starter's laps led and the displayed start matches
    # the start that actually gated those laps.
    def _resolve_start(d):
        if qual_pos.get(d):
            return qual_pos[d]
        th_s = th_data.get(d)
        if th_s and th_s.get("avg_start"):
            return round(th_s["avg_start"])
        tt_s = tt_data.get(d)
        if isinstance(tt_s, dict) and tt_s.get("avg_start"):
            return round(tt_s["avg_start"])
        if odds_finish.get(d):
            return round(odds_finish[d])
        return round(field_size * 0.5)
    # pd_start = qualifying position (drives place differential + the displayed
    # start — DK scores PD off the qualified grid). start_for_alloc = the ACTUAL
    # race-day grid (rear for penalty drivers) so the laps-led gate damps a car
    # sent to the back. They're identical for everyone NOT moved to the rear.
    pd_start = {d: _resolve_start(d) for d in drivers}
    start_for_alloc = {
        d: ((grid_start.get(d) if grid_start else None) or pd_start[d])
        for d in drivers
    }

    # ── Allocate laps led and fastest laps ──
    allocated_ll = _allocate_laps_led(
        dom_raw_scores, race_laps, track_name, gate_track_type,
        calibration=calibration,
        odds_display=odds_display,
        start_positions=start_for_alloc,
    ) if race_laps > 0 else {}
    allocated_fl = _allocate_fastest_laps(
        fl_raw_scores, race_laps, track_type,
        calibration=calibration,
        odds_display=odds_display,
    ) if race_laps > 0 else {}

    # Expected finish + expected finish-points per driver (distribution-based).
    finish_exp = _finish_dist_expectations(driver_raw_scores, drivers, field_size)

    # ── Compute DK points ──
    proj_rows = []
    proj_detail = {}
    for d in drivers:
        e_finish, e_pts = finish_exp[d]
        proj_finish = round(e_finish, 1)        # fractional expected finish
        proj_laps_led = allocated_ll.get(d, 0)
        proj_fastest = allocated_fl.get(d, 0)

        # Displayed start + place differential use the QUALIFYING position (DK
        # scores PD off the qualified grid, not the penalty box). The rear grid
        # start only drives laps-led / fast-laps scoring above.
        start_pos = pd_start[d]

        # Finish points + place differential are the EXPECTED value over the
        # finish distribution (not read off one forced integer position).
        finish_pts = round(e_pts, 2)
        diff_pts = round(start_pos - e_finish, 2)
        led_pts = round(proj_laps_led) * 0.25
        fl_pts = round(proj_fastest) * 0.45
        proj_dk = round(finish_pts + diff_pts + led_pts + fl_pts, 1)

        # ── Floor / ceiling DK (for cash vs GPP optimization) ──
        # Spread the finish ~1 sigma each way (same ramp spread the median uses):
        #   floor = a bad-but-running race — worse finish, barely leads.
        #   ceiling = a strong race — better finish + dominator upside realized.
        # Steady mid-pack finishers get a HIGH floor (cash); drivers who can lead
        # laps / win get a high ceiling (GPP). DNF risk isn't modeled here.
        _c = max(1.0, min(float(field_size), driver_raw_scores[d]))
        _sig = _ramp_sigma(_c, field_size)
        _floor_fin = min(float(field_size), _c + _sig)
        _ceil_fin = max(1.0, _c - _sig)
        proj_floor = max(0.0,
            _expected_finish_pts(_floor_fin) + (start_pos - _floor_fin)
            + led_pts * 0.10 + fl_pts * 0.20)
        # DNF-aware floor: blend in the chance the race ends in a DNF (~last,
        # no laps led). Drags wreck-prone drivers' floors down so cash mode
        # avoids them; doesn't touch the median or ceiling.
        _pdnf = dnf_prob_map.get(d, 0.0)
        if _pdnf > 0:
            _dnf_score = max(0.0, _expected_finish_pts(field_size)
                             + (start_pos - field_size))
            proj_floor = (1.0 - _pdnf) * proj_floor + _pdnf * _dnf_score
        proj_floor = round(proj_floor, 1)
        _ceil_led = min(race_laps * 0.25, led_pts * 2.0) if race_laps > 0 else 0.0
        proj_ceiling = round(
            _expected_finish_pts(_ceil_fin) + (start_pos - _ceil_fin)
            + _ceil_led + fl_pts * 1.8, 1)

        proj_rows.append({
            "driver": d,
            "proj_dk": proj_dk,
            "proj_floor": proj_floor,
            "proj_ceiling": proj_ceiling,
            "proj_finish": proj_finish,
            "raw_finish": round(driver_raw_scores[d], 2),
            "start": start_pos,
            "finish_pts": finish_pts,
            "diff_pts": diff_pts,
            "led_pts": led_pts,
            "fl_pts": fl_pts,
            "laps_led": round(proj_laps_led),
            "fast_laps": round(proj_fastest),
        })
        proj_detail[d] = {
            "proj_finish": proj_finish,
            "proj_floor": proj_floor,
            "proj_ceiling": proj_ceiling,
            "raw_finish": round(driver_raw_scores[d], 2),
            "start": start_pos,
            "laps_led": round(proj_laps_led),
            "fast_laps": round(proj_fastest),
        }

    if return_signal_details:
        return proj_rows, proj_detail, driver_signal_details
    return proj_rows, proj_detail
