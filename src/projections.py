"""Shared projection engine used by both Projections and Accuracy tabs.

This module contains the core projection logic that converts signal data
into DK point projections.  Both tabs call compute_projections() with
their own data-loading layer, guaranteeing identical math.
"""

import math
import numpy as np
from src.config import DK_FINISH_POINTS


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
    from src.utils import fuzzy_match_name

    if cross_th_lookup is None:
        cross_th_lookup = {}

    mid_field = field_size * 0.5
    MIN_RACES_FULL_TRUST = 5

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
            trust = min(1.0, effective_races / MIN_RACES_FULL_TRUST)
            if th.get("_cross_series_only"):
                trust *= 0.8

            arp = th.get("avg_running_pos")
            af = th["avg_finish"]
            base_finish = arp * 0.65 + af * 0.35 if arp is not None else af

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
            tt_trust = min(1.0, tt_races / MIN_RACES_FULL_TRUST)
            if isinstance(tt, dict) and tt.get("_cross_series_only"):
                tt_trust *= 0.8
            tt_arp = tt.get("avg_running_pos") if isinstance(tt, dict) else None
            tt_af = tt.get("avg_finish", mid_field) if isinstance(tt, dict) else mid_field
            tt_avg = tt_arp * 0.65 + tt_af * 0.35 if tt_arp is not None else tt_af
            tt_regressed = tt_avg * tt_trust + mid_field * (1 - tt_trust)
            sigs["ttype"] = tt_regressed
            sig_w["ttype"] = tt_weight

        # Qualifying signal
        if qp and qp <= field_size and wn.get("qual", 0) > 0:
            has_history = bool(th or tt)
            qual_finish = qp * 0.80 + mid_field * 0.20 if has_history else qp * 0.40 + mid_field * 0.60
            sigs["qual"] = qual_finish
            sig_w["qual"] = wn["qual"]

        # Team signal
        tm = team_signal.get(d) if team_signal else None
        if tm is not None and wn.get("team", 0) > 0:
            sigs["team"] = tm
            sig_w["team"] = wn["team"]

        # Practice signal
        if pr:
            prac_finish = pr * 0.70 + mid_field * 0.30
            sigs["prac"] = prac_finish
            sig_w["prac"] = wn.get("practice", 0)

        # Odds signal
        if od and wn.get("odds", 0) > 0:
            has_history = bool(th or tt)
            odds_val = od if has_history else od * 0.60 + mid_field * 0.40
            sigs["odds"] = odds_val
            sig_w["odds"] = wn["odds"]

        raw_signals[d] = sigs
        signal_weight_map[d] = sig_w
        sig_extras[d] = extras

    # ── Pass 2: Normalize each signal to 1→field_size ──
    signal_names = set()
    for sigs in raw_signals.values():
        signal_names.update(sigs.keys())

    MINMAX_SIGNALS = {"odds", "track", "ttype"}
    RANK_SIGNALS = {"team"}

    normalized_signals = {d: {} for d in drivers}
    for sig_name in signal_names:
        sig_vals = [(d, raw_signals[d][sig_name]) for d in drivers if sig_name in raw_signals[d]]
        if not sig_vals:
            continue

        if sig_name in MINMAX_SIGNALS:
            vals_only = [v for _, v in sig_vals]
            raw_min, raw_max = min(vals_only), max(vals_only)
            raw_range = raw_max - raw_min
            for d, val in sig_vals:
                if raw_range > 0:
                    t = (val - raw_min) / raw_range
                    normalized_signals[d][sig_name] = 1 + (field_size - 1) * t
                else:
                    normalized_signals[d][sig_name] = mid_field

        elif sig_name in RANK_SIGNALS:
            sig_vals.sort(key=lambda x: x[1])
            n_with_sig = len(sig_vals)
            for rank_idx, (d, _) in enumerate(sig_vals):
                if n_with_sig > 1:
                    normalized_signals[d][sig_name] = 1 + (field_size - 1) * (rank_idx / (n_with_sig - 1))
                else:
                    normalized_signals[d][sig_name] = mid_field
        else:
            for d, val in sig_vals:
                normalized_signals[d][sig_name] = max(1, min(field_size, val))

    # ── Pass 3: Weighted average + adjustments ──
    driver_raw_scores = {}
    dom_raw_scores = {}
    fl_raw_scores = {}
    driver_signal_details = {}

    ll_ref = calibration.get("avg_top_leader", race_laps * 0.35) if calibration else race_laps * 0.35

    for d in drivers:
        norm = normalized_signals[d]
        weights = signal_weight_map[d]
        sig_detail = dict(sig_extras.get(d, {}))

        finish_signals = []
        signal_weights = []
        for sig_name in norm:
            finish_signals.append(norm[sig_name])
            signal_weights.append(weights.get(sig_name, 0))
            sig_detail[SIG_DISPLAY.get(sig_name, sig_name)] = round(norm[sig_name], 1)

        if finish_signals and sum(signal_weights) > 0:
            total_w = sum(signal_weights)
            raw_finish = sum(f * w for f, w in zip(finish_signals, signal_weights)) / total_w
        else:
            raw_finish = field_size * 0.75

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

            # Track history laps led
            if th and th.get("races", 0) >= 1 and th.get("laps_led", 0) > 0:
                ll_per_race = th["laps_led"] / th["races"]
                ll_norm = min(100, (ll_per_race / max(ll_ref, 1)) * 100)
                dom_signals.append(ll_norm)
            else:
                dom_signals.append(5.0)
            dom_weights_list.append(wn.get("track", 0.20))

            # Track type laps led
            if tt and isinstance(tt, dict) and tt.get("laps_led_per_race", 0) > 0:
                tt_ll = tt["laps_led_per_race"]
                tt_ll_norm = min(100, (tt_ll / max(ll_ref, 1)) * 100)
                dom_signals.append(tt_ll_norm)
            else:
                dom_signals.append(5.0)
            dom_weights_list.append(wn.get("track_type", 0.15))

            # Qualifying
            if qp and qp <= field_size and wn.get("qual", 0) > 0:
                qual_dom = max(0, (field_size + 1 - qp) / field_size) ** 1.5 * 100
                dom_signals.append(qual_dom)
                dom_weights_list.append(wn.get("qual", 0.15))

            # Odds (implied probability)
            if od and wn.get("odds", 0) > 0:
                odds_info = odds_display.get(d) if odds_display else None
                if odds_info and odds_info.get("impl_pct"):
                    max_impl = max((v.get("impl_pct", 0) for v in odds_display.values()), default=1)
                    odds_dom = min(100, (odds_info["impl_pct"] / max(max_impl, 1)) * 100)
                else:
                    odds_dom = max(0, (field_size + 1 - od) / field_size) ** 1.3 * 100
                dom_signals.append(odds_dom)
                dom_weights_list.append(wn.get("odds", 0.15))

            # Practice
            if pr and practice_data:
                max_p_val = max(practice_data.values()) if practice_data else field_size
                prac_dom = max(0, (max_p_val + 1 - pr) / max_p_val) * 100
                dom_signals.append(prac_dom)
                dom_weights_list.append(wn.get("practice", 0.10))

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

            if od and wn.get("odds", 0) > 0:
                odds_info = odds_display.get(d) if odds_display else None
                if odds_info and odds_info.get("impl_pct"):
                    max_impl = max((v.get("impl_pct", 0) for v in odds_display.values()), default=1)
                    odds_fl = min(100, (odds_info["impl_pct"] / max(max_impl, 1)) * 100)
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

        # Qualifying start position multiplier on dominator score
        if qp and qp <= field_size and dom_score > 0:
            if qp <= 3:
                start_mult = 1.15 - (qp - 1) * 0.05
            elif qp <= 10:
                start_mult = 1.0
            else:
                start_mult = max(0.70, 1.0 - (qp - 10) * 0.02)
            dom_score = dom_score * start_mult

        dom_raw_scores[d] = dom_score
        fl_raw_scores[d] = fl_score

    # ── Rank-order finish assignment ──
    sorted_drivers = sorted(driver_raw_scores.items(), key=lambda x: x[1])
    driver_proj_finish = {}
    for rank_idx, (d, _) in enumerate(sorted_drivers):
        driver_proj_finish[d] = rank_idx + 1

    # ── Allocate laps led and fastest laps ──
    allocated_ll = _allocate_laps_led(
        dom_raw_scores, race_laps, track_name, track_type,
        calibration=calibration,
        odds_display=odds_display,
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

        # Start position fallback
        th_s = th_data.get(d)
        tt_s = tt_data.get(d)
        hist_start = th_s.get("avg_start") if th_s else None
        tt_start = tt_s.get("avg_start") if isinstance(tt_s, dict) else None
        odds_start = odds_finish.get(d)
        if qual_pos.get(d):
            start_pos = qual_pos[d]
        elif hist_start:
            start_pos = round(hist_start)
        elif tt_start:
            start_pos = round(tt_start)
        elif odds_start:
            start_pos = round(odds_start)
        else:
            start_pos = round(field_size * 0.5)

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
