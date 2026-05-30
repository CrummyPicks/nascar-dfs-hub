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
# NOTE: As of the start-aware laps-led gate (_start_avail in tab_projections.py),
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


def compute_projections(
    drivers, field_size, wn,
    th_data, tt_data, qual_pos, practice_data,
    odds_finish, odds_display, team_signal,
    mfr_adjustment, team_adj_data, dnf_data,
    race_laps, track_name, track_type, series_id,
    calibration, cross_th_lookup=None,
    return_signal_details=False,
):
    """Run the full projection engine.

    Returns (proj_rows, proj_detail) where:
      proj_rows: list of dicts with per-driver projection data
      proj_detail: {driver: {proj_finish, start, laps_led, fast_laps}}

    If return_signal_details=True, returns (proj_rows, proj_detail, signal_details)
    where signal_details = {driver: {"Track": val, "Odds": val, ...}} with
    normalized signal values and adjustment info for display.
    """
    from tabs.tab_projections import (
        _allocate_laps_led, _allocate_fastest_laps,
    )
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
            base_finish = arp_finish_blend(arp, af, track_type)

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
            tt_avg = arp_finish_blend(tt_arp, tt_af, track_type)
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

        driver_raw_scores[d] = raw_finish
        driver_signal_details[d] = sig_detail

        # ── Dominator scoring ──
        th = th_data.get(d)
        tt = tt_data.get(d)
        qp = qual_pos.get(d)
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

    # ── Finish position assignment ──
    # Every driver gets a UNIQUE integer position 1..field_size, assigned by
    # ranking drivers on their raw_finish score (lowest = P1). Real races
    # have unique finishing positions, so projections should too — this
    # makes DK finish-point allocation exact (only one driver gets P1's 44
    # points, one gets P2's 40, etc.).
    #
    # The low-info penalty still has its intended effect because it's
    # already been applied to raw_finish BEFORE ranking. A driver with the
    # penalty pushed to raw=33 will rank behind a driver with raw=25, so
    # they naturally fall to the back of the field after rank assignment.
    sorted_drivers = sorted(driver_raw_scores.items(), key=lambda x: x[1])
    driver_proj_finish = {}
    for rank_idx, (d, _) in enumerate(sorted_drivers):
        driver_proj_finish[d] = rank_idx + 1

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
    start_for_alloc = {d: _resolve_start(d) for d in drivers}

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

    # ── Compute DK points ──
    proj_rows = []
    proj_detail = {}
    for d in drivers:
        proj_finish = driver_proj_finish[d]
        proj_laps_led = allocated_ll.get(d, 0)
        proj_fastest = allocated_fl.get(d, 0)

        # Start position — reuse the resolution computed for the laps-led start
        # gate so the displayed start matches the start that gated the laps.
        start_pos = start_for_alloc[d]

        finish_pts = _expected_finish_pts(proj_finish)
        diff_pts = start_pos - proj_finish
        led_pts = round(proj_laps_led) * 0.25
        fl_pts = round(proj_fastest) * 0.45
        proj_dk = round(finish_pts + diff_pts + led_pts + fl_pts, 1)

        proj_rows.append({
            "driver": d,
            "proj_dk": proj_dk,
            "proj_finish": proj_finish,
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
            "start": start_pos,
            "laps_led": round(proj_laps_led),
            "fast_laps": round(proj_fastest),
        }

    if return_signal_details:
        return proj_rows, proj_detail, driver_signal_details
    return proj_rows, proj_detail
