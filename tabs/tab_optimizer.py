"""Tab 6: Lineup Optimizer — FantasyPros-style with lock/exclude/swap."""

import random
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
import os

from src.config import SALARY_CAP, ROSTER_SIZE, TRACK_TYPE_MAP, TRACK_TYPE_PARENT
from src.utils import safe_fillna, format_display_df, fuzzy_match_name, fuzzy_get, build_norm_lookup


PROJ_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")


# ── Projection Pool Builder (unified with Projections tab weights) ──────────

def _get_projection_pool(entry_list_df, qualifying_df, lap_averages_df,
                          practice_data, race_name, track_name, series_id,
                          dk_df, odds_data=None):
    """Build a driver pool with projection scores for optimization.

    Uses the same weight system as the Projections tab — reads weights from
    session state if user has adjusted them, otherwise uses defaults.
    Returns (DataFrame, engine_label) where DataFrame has Driver, DK Salary,
    Proj Score, Value columns.
    """
    if dk_df.empty:
        return pd.DataFrame(), "no salary data"

    # Read weights from session state (set by Projections tab sliders)
    # Defaults match track-type-specific values from projections tab
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    _tt_defaults = {
        "superspeedway": (20, 25, 45, 10),
        "short": (35, 15, 30, 20),
        "road": (25, 20, 25, 30),
        "intermediate": (30, 20, 35, 15),
    }
    _dt, _dtt, _do, _dp = _tt_defaults.get(parent_type, (30, 20, 35, 15))
    w_track = st.session_state.get("pw_track", _dt)
    w_ttype = st.session_state.get("pw_ttype", _dtt)
    w_odds  = st.session_state.get("pw_odds", _do)
    w_prac  = st.session_state.get("pw_prac", _dp)

    # Smart weight handling: drop unavailable signals, redistribute
    has_odds = bool(odds_data)
    has_practice = bool(practice_data)
    effective_odds = w_odds if has_odds else 0
    effective_prac = w_prac if has_practice else 0

    raw_total = w_track + w_ttype + effective_prac + effective_odds
    if raw_total > 0:
        wn = {
            "track": w_track / raw_total,
            "track_type": w_ttype / raw_total,
            "qual": 0,
            "practice": effective_prac / raw_total,
            "odds": effective_odds / raw_total,
        }
    else:
        # Fallback: 60% track, 40% track type
        wn = {"track": 0.60, "track_type": 0.40, "qual": 0, "practice": 0, "odds": 0}

    pool = dk_df.drop_duplicates("Driver").copy()
    field_size = len(pool)
    drivers = pool["Driver"].tolist()

    # ── 1. Track History Score ──────────────────────────────────────────────
    from src.data import scrape_track_history
    th_df = scrape_track_history(track_name, series_id)
    th_scores = {}
    if not th_df.empty and "Avg Finish" in th_df.columns:
        for col in ["Avg Finish", "Avg Start", "Avg Rating"]:
            if col in th_df.columns:
                th_df[col] = pd.to_numeric(th_df[col], errors="coerce")
        th_idx = th_df.drop_duplicates("Driver").set_index("Driver")
        th_names = th_idx.index.tolist()
        for d in drivers:
            matched = d if d in th_idx.index else fuzzy_match_name(d, th_names)
            if matched and matched in th_idx.index:
                row = th_idx.loc[matched]
                af = row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20
                ar = row.get("Avg Rating", 80) if pd.notna(row.get("Avg Rating")) else 80
                finish_s = max(0, (40 - af) / 39 * 100) * 0.5
                rating_s = min(100, max(0, ar / 1.5)) * 0.5
                th_scores[d] = max(5, min(95, finish_s + rating_s))
            else:
                th_scores[d] = 35
    pool["TrackScore"] = pool["Driver"].map(lambda d: th_scores.get(d, 35))

    # ── 2. Track Type Score ─────────────────────────────────────────────────
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    same_type_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                        if tt == track_type and t != track_name]
    if len(same_type_tracks) < 2:
        same_type_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                            if TRACK_TYPE_PARENT.get(tt, tt) == parent_type
                            and t != track_name]
    tt_scores = {}
    if same_type_tracks:
        type_finishes = {}
        for sim_track in same_type_tracks[:4]:
            sim_th = scrape_track_history(sim_track, series_id)
            if sim_th.empty:
                continue
            for col in ["Avg Finish"]:
                if col in sim_th.columns:
                    sim_th[col] = pd.to_numeric(sim_th[col], errors="coerce")
            for _, r in sim_th.iterrows():
                d = r.get("Driver")
                af = r.get("Avg Finish")
                if d and pd.notna(af):
                    type_finishes.setdefault(d, []).append(af)
        tf_norm = build_norm_lookup(type_finishes)
        for d in drivers:
            tf_val = fuzzy_get(d, type_finishes, tf_norm)
            if tf_val is not None:
                avg_f = np.mean(tf_val)
                tt_scores[d] = max(5, min(95, (40 - avg_f) / 39 * 100))
            else:
                tt_scores[d] = 35
    pool["TypeScore"] = pool["Driver"].map(lambda d: tt_scores.get(d, 35))

    # ── 3. Qualifying Score ─────────────────────────────────────────────────
    qual_scores = {}
    if not qualifying_df.empty and "Qualifying Position" in qualifying_df.columns:
        qclean = qualifying_df.dropna(subset=["Driver"]).copy()
        qclean["Qualifying Position"] = pd.to_numeric(qclean["Qualifying Position"], errors="coerce")
        qidx = qclean.drop_duplicates("Driver").set_index("Driver")["Qualifying Position"]
        q_names = qidx.index.tolist()
        max_q = qidx.max() if not qidx.empty else field_size
        for d in drivers:
            matched = d if d in qidx.index else fuzzy_match_name(d, q_names)
            if matched and matched in qidx.index:
                qp = qidx[matched]
                qual_scores[d] = max(5, 100 - (qp - 1) / max(max_q - 1, 1) * 95)
            else:
                qual_scores[d] = 35
    pool["QualScore"] = pool["Driver"].map(lambda d: qual_scores.get(d, 50))

    # ── 4. Practice Score ───────────────────────────────────────────────────
    prac_scores = {}
    if practice_data:
        prac_norm = build_norm_lookup(practice_data)
        max_p = max(practice_data.values()) if practice_data else field_size
        for d in drivers:
            pval = fuzzy_get(d, practice_data, prac_norm)
            if pval is not None:
                prac_scores[d] = max(5, 100 - (pval - 1) / max(max_p - 1, 1) * 95)
            else:
                prac_scores[d] = 35
    pool["PracScore"] = pool["Driver"].map(lambda d: prac_scores.get(d, 50))

    # ── 5. Odds Score ───────────────────────────────────────────────────────
    odds_scores = {}
    if has_odds:
        odds_values = {}
        for d in drivers:
            matched = fuzzy_match_name(d, list(odds_data.keys()))
            if matched:
                try:
                    odds_val = int(str(odds_data[matched]).replace("+", ""))
                    if odds_val > 0:
                        prob = 100 / (odds_val + 100)
                    else:
                        prob = abs(odds_val) / (abs(odds_val) + 100)
                    odds_values[d] = prob
                except (ValueError, TypeError):
                    pass
        if odds_values:
            max_prob = max(odds_values.values())
            for d in drivers:
                if d in odds_values:
                    odds_scores[d] = min(95, max(5, (odds_values[d] / max_prob) * 95))
                else:
                    odds_scores[d] = 20
    pool["OddsScore"] = pool["Driver"].map(lambda d: odds_scores.get(d, 35))

    # ── Use Proj DK from Projections tab when available ──────────────────────
    proj_dk_map = st.session_state.get("proj_dk_map", {})
    if proj_dk_map:
        proj_dk_norm = build_norm_lookup(proj_dk_map)
        pool["Proj Score"] = pool["Driver"].map(
            lambda d: fuzzy_get(d, proj_dk_map, proj_dk_norm) or 0).round(1)
    else:
        # Fallback: weighted composite from individual signals
        pool["Proj Score"] = (
            pool["TrackScore"] * wn["track"] +
            pool["TypeScore"]  * wn["track_type"] +
            pool["QualScore"]  * wn["qual"] +
            pool["PracScore"]  * wn["practice"] +
            pool["OddsScore"]  * wn["odds"]
        ).round(1)

    pool["Value"] = np.where(
        pool["DK Salary"] > 0,
        (pool["Proj Score"] / (pool["DK Salary"] / 1000)).round(2),
        0
    )

    # Build signal label
    signals_used = []
    if proj_dk_map: signals_used.append("proj_dk")
    if th_scores: signals_used.append("track")
    if tt_scores: signals_used.append("type")
    if qual_scores: signals_used.append("qual")
    if prac_scores: signals_used.append("practice")
    if odds_scores: signals_used.append("odds")
    engine_label = f"{len(signals_used)}-signal model ({', '.join(signals_used)})"

    return pool[["Driver", "DK Salary", "Proj Score", "Value"]].sort_values(
        "Proj Score", ascending=False).reset_index(drop=True), engine_label


# ── Lineup Generation ───────────────────────────────────────────────────────

def _build_optimal_lineup(pool_df, salary_cap, roster_size, locked=None, excluded=None):
    """Build the single best lineup using the full optimizer logic."""
    results = _generate_lineups_greedy(
        pool_df, salary_cap, roster_size, 1, 100, "cash",
        locked=locked, excluded=excluded
    )
    return results[0] if results else []


def _get_swap_candidates(pool_df, current_lineup, swap_driver, salary_cap, roster_size):
    """Get ranked replacement candidates for a driver being swapped out.

    Returns DataFrame of eligible replacements sorted by Proj Score.
    """
    lineup_drivers = {d["Driver"] for d in current_lineup}
    lineup_salary = sum(d["DK Salary"] for d in current_lineup)
    swap_salary = next((d["DK Salary"] for d in current_lineup
                        if d["Driver"] == swap_driver), 0)

    # Budget for replacement = cap - (lineup salary - swapped driver salary)
    budget = salary_cap - (lineup_salary - swap_salary)

    candidates = pool_df[
        (~pool_df["Driver"].isin(lineup_drivers)) &
        (pool_df["DK Salary"] <= budget)
    ].copy()

    candidates = candidates.sort_values("Proj Score", ascending=False)
    return candidates


def _generate_lineups_greedy(pool_df, salary_cap, roster_size, num_lineups,
                              max_exposure, mode="gpp", locked=None, excluded=None):
    """Generate optimized lineups using systematic exploration.

    Strategy:
    1. Build the best possible lineup (greedy by value)
    2. Generate variations by swapping 1-2 players with alternatives
    3. Also try "stars + value" combos (expensive core + cheap fill)
    4. Score all valid lineups and keep the best unique ones
    """
    locked = locked or []
    excluded = excluded or set()

    available = pool_df[~pool_df["Driver"].isin(excluded)].copy()
    # Ensure no NaN values in scoring columns
    available["Proj Score"] = available["Proj Score"].fillna(0)
    available["Value"] = available["Value"].fillna(0)
    available["DK Salary"] = available["DK Salary"].fillna(0)
    if available.empty or len(available) < roster_size:
        return []

    # Separate locked and variable pools
    locked_data = []
    lock_salary = 0
    for driver in locked:
        match = available[available["Driver"] == driver]
        if not match.empty:
            locked_data.append(match.iloc[0].to_dict())
            lock_salary += match.iloc[0]["DK Salary"]

    variable_pool = available[~available["Driver"].isin(locked)].copy()
    remaining_cap = salary_cap - lock_salary
    remaining_slots = roster_size - len(locked_data)

    if remaining_slots <= 0:
        return [locked_data] * min(num_lineups, 1)

    # Sort by projection and value
    variable_pool = variable_pool.sort_values("Proj Score", ascending=False)
    candidates = variable_pool.to_dict("records")
    all_lineups = []  # (total_score, lineup)
    seen_keys = set()

    def _try_lineup(lineup):
        """Score and add a lineup if valid and unique."""
        if len(lineup) != remaining_slots:
            return
        total_sal = sum(d["DK Salary"] for d in lineup)
        if total_sal > remaining_cap:
            return
        key = tuple(sorted(d["Driver"] for d in lineup))
        if key in seen_keys:
            return
        seen_keys.add(key)
        total_pts = sum(d["Proj Score"] for d in lineup)
        all_lineups.append((total_pts, locked_data + lineup))

    def _greedy_fill(must_include, pool, cap):
        """Greedy fill remaining slots from pool within cap.

        Uses a balanced approach: sort by projection but ensure salary fits.
        Tries projection-first, then falls back to value-first if salary tight.
        """
        lineup = list(must_include)
        used = {d["Driver"] for d in lineup}
        rem = cap - sum(d["DK Salary"] for d in lineup)
        need = remaining_slots - len(lineup)

        remaining = [d for d in pool if d["Driver"] not in used]

        # Sort by projection (best players first, salary permitting)
        remaining.sort(key=lambda d: d.get("Proj Score", 0), reverse=True)
        for d in remaining:
            if len(lineup) >= remaining_slots:
                break
            if d["DK Salary"] <= rem:
                # Check if adding this driver leaves enough budget for remaining slots
                slots_after = remaining_slots - len(lineup) - 1
                min_salary_needed = slots_after * min(
                    (r["DK Salary"] for r in pool if r["Driver"] not in used
                     and r["Driver"] != d["Driver"]),
                    default=0
                )
                if rem - d["DK Salary"] >= min_salary_needed:
                    lineup.append(d)
                    used.add(d["Driver"])
                    rem -= d["DK Salary"]
        return lineup

    # === Strategy 1: Pure greedy by value ===
    lineup = _greedy_fill([], candidates, remaining_cap)
    _try_lineup(lineup)

    # === Strategy 2: Pure greedy by projection (top projected, fill with value) ===
    for top_n in range(1, min(remaining_slots, 5)):
        top_picks = candidates[:top_n]
        if sum(d["DK Salary"] for d in top_picks) <= remaining_cap:
            lineup = _greedy_fill(top_picks, candidates, remaining_cap)
            _try_lineup(lineup)

    # === Strategy 3: Every pair of top-15 drivers as anchors ===
    top_15 = candidates[:min(15, len(candidates))]
    for i in range(len(top_15)):
        for j in range(i + 1, len(top_15)):
            pair = [top_15[i], top_15[j]]
            pair_sal = sum(d["DK Salary"] for d in pair)
            if pair_sal <= remaining_cap:
                lineup = _greedy_fill(pair, candidates, remaining_cap)
                _try_lineup(lineup)

    # === Strategy 4: Every triple of top-15 drivers as anchors ===
    for i in range(len(top_15)):
        for j in range(i + 1, len(top_15)):
            for k in range(j + 1, len(top_15)):
                triple = [top_15[i], top_15[j], top_15[k]]
                triple_sal = sum(d["DK Salary"] for d in triple)
                if triple_sal <= remaining_cap:
                    lineup = _greedy_fill(triple, candidates, remaining_cap)
                    _try_lineup(lineup)

    # === Strategy 4b: Every quad of top-12 drivers as anchors ===
    top_12 = candidates[:min(12, len(candidates))]
    for i in range(len(top_12)):
        for j in range(i + 1, len(top_12)):
            for k in range(j + 1, len(top_12)):
                for m in range(k + 1, len(top_12)):
                    quad = [top_12[i], top_12[j], top_12[k], top_12[m]]
                    quad_sal = sum(d["DK Salary"] for d in quad)
                    if quad_sal <= remaining_cap:
                        lineup = _greedy_fill(quad, candidates, remaining_cap)
                        _try_lineup(lineup)

    # === Strategy 5: Swap-based variations from top lineups ===
    # Take the best 10 lineups so far, swap each player with alternatives
    all_lineups.sort(key=lambda x: x[0], reverse=True)
    base_lineups = [lu for _, lu in all_lineups[:10]]
    for base in base_lineups:
        variable_part = [d for d in base if d["Driver"] not in [ld["Driver"] for ld in locked_data]]
        for swap_idx in range(len(variable_part)):
            swapped_out = variable_part[swap_idx]
            remaining = [d for d in variable_part if d["Driver"] != swapped_out["Driver"]]
            rem_sal = remaining_cap - sum(d["DK Salary"] for d in remaining)
            # Try each candidate as replacement
            used = {d["Driver"] for d in remaining}
            for replacement in candidates:
                if replacement["Driver"] in used:
                    continue
                if replacement["DK Salary"] <= rem_sal:
                    new_lineup = remaining + [replacement]
                    _try_lineup(new_lineup)

    # === Strategy 6: Random exploration for diversity (GPP mode) ===
    if mode == "gpp":
        for _ in range(num_lineups * 20):
            # Pick 2 random anchors from top half, fill with value
            top_half = candidates[:max(6, len(candidates) // 2)]
            if len(top_half) < 2:
                break
            anchors = random.sample(top_half, min(2, remaining_slots - 1))
            anchor_sal = sum(d["DK Salary"] for d in anchors)
            if anchor_sal <= remaining_cap:
                lineup = _greedy_fill(anchors, candidates, remaining_cap)
                _try_lineup(lineup)

    # Sort by total projected points and apply exposure limits
    all_lineups.sort(key=lambda x: x[0], reverse=True)

    if max_exposure >= 100:
        return [lu for _, lu in all_lineups[:num_lineups]]

    # Apply exposure filter
    final = []
    exposure_count = {}
    max_exp = max(1, int(num_lineups * max_exposure / 100))
    for score, lu in all_lineups:
        ok = True
        for d in lu:
            if exposure_count.get(d["Driver"], 0) >= max_exp:
                ok = False
                break
        if ok:
            final.append(lu)
            for d in lu:
                exposure_count[d["Driver"]] = exposure_count.get(d["Driver"], 0) + 1
            if len(final) >= num_lineups:
                break

    return final


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           is_prerace, race_name, race_id, track_name, series_id, dk_df,
           odds_data=None):
    """Render the Optimizer tab."""
    st.markdown(f"### Lineup Optimizer — {race_name}")

    if dk_df.empty:
        # Try loading salaries from DB for completed races
        from src.data import query_salaries
        db_sal = query_salaries(race_id=race_id, platform="DraftKings")
        if not db_sal.empty and "Salary" in db_sal.columns:
            dk_df = db_sal.rename(columns={"Salary": "DK Salary"})[["Driver", "DK Salary"]].copy()
        if dk_df.empty:
            if not is_prerace:
                st.info("No salary data for this completed race. "
                        "Salaries expire after the race — upload a CSV or use an upcoming race.")
            else:
                st.info("No salary data available. Upload a DK CSV or wait for DK API salaries to enable the optimizer.")
            return

    # Initialize session state — clear when race/series changes
    race_key = f"{series_id}_{race_id}"
    if st.session_state.get("opt_race_key") != race_key:
        st.session_state.opt_race_key = race_key
        st.session_state.opt_lineup = []
        st.session_state.opt_locked = set()
        st.session_state.opt_excluded = set()
        st.session_state.opt_multi_lineups = []
    if "opt_lineup" not in st.session_state:
        st.session_state.opt_lineup = []
    if "opt_locked" not in st.session_state:
        st.session_state.opt_locked = set()
    if "opt_excluded" not in st.session_state:
        st.session_state.opt_excluded = set()
    if "opt_multi_lineups" not in st.session_state:
        st.session_state.opt_multi_lineups = []

    # --- Settings bar ---
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        salary_cap = st.number_input("Salary Cap", 40000, 60000, SALARY_CAP, 1000, key="opt_cap")
    with s2:
        roster_size = st.number_input("Roster Size", 4, 8, ROSTER_SIZE, key="opt_roster")
    with s3:
        mode = st.selectbox("Mode", ["GPP", "Cash"], key="opt_mode",
                            help="GPP: diverse lineups. Cash: high-floor consistency.")
    with s4:
        max_exposure = st.slider("Max Exposure %", 10, 100,
                                 60 if mode == "GPP" else 100, 5, key="opt_exposure",
                                 help="Max % of lineups a driver can appear in (multi-lineup)")

    # Build projection pool (uses same weights as Projections tab)
    with st.spinner("Building projections..."):
        pool, engine_label = _get_projection_pool(
            entry_list_df, qualifying_df, lap_averages_df,
            practice_data, race_name, track_name, series_id,
            dk_df, odds_data=odds_data)

    if pool.empty:
        st.warning("Could not build projection pool. Check salary data.")
        return

    # ─── PROJECTION OVERRIDES ──────────────────────────────────────────────
    # Allow manual override of projected points for specific drivers
    if "opt_overrides" not in st.session_state:
        st.session_state.opt_overrides = {}

    with st.expander("Projection Overrides", expanded=False):
        st.caption("Override projected points for specific drivers. "
                   "Lineups will optimize using your custom values.")
        ov_cols = st.columns([2, 1, 1])
        with ov_cols[0]:
            all_drivers = sorted(pool["Driver"].tolist())
            ov_driver = st.selectbox("Driver", [""] + all_drivers,
                                      key="opt_ov_driver", label_visibility="collapsed")
        with ov_cols[1]:
            current = pool[pool["Driver"] == ov_driver]["Proj Score"].values[0] if ov_driver else 0
            ov_pts = st.number_input("Proj Pts", 0.0, 300.0, float(current), 1.0,
                                      key="opt_ov_pts", label_visibility="collapsed")
        with ov_cols[2]:
            if st.button("Set Override", key="opt_ov_set") and ov_driver:
                st.session_state.opt_overrides[ov_driver] = ov_pts
                st.rerun()

        # Show active overrides
        if st.session_state.opt_overrides:
            st.markdown("**Active overrides:**")
            for drv, pts in sorted(st.session_state.opt_overrides.items()):
                orig = pool[pool["Driver"] == drv]["Proj Score"].values[0] if drv in pool["Driver"].values else 0
                st.caption(f"  {drv}: {orig:.1f} → **{pts:.1f}**")
            if st.button("Clear All Overrides", key="opt_ov_clear"):
                st.session_state.opt_overrides = {}
                st.rerun()

    # Apply overrides to pool
    for drv, pts in st.session_state.opt_overrides.items():
        mask = pool["Driver"] == drv
        if mask.any():
            pool.loc[mask, "Proj Score"] = pts
            sal = pool.loc[mask, "DK Salary"].values[0]
            pool.loc[mask, "Value"] = round(pts / (sal / 1000), 2) if sal > 0 else 0

    proj_source = engine_label

    # ─── OPTIMAL LINEUP PANEL ───────────────────────────────────────────────
    st.markdown("---")

    # Auto-build optimal lineup if none exists yet
    locked = list(st.session_state.opt_locked)
    excluded = st.session_state.opt_excluded

    if not st.session_state.opt_lineup:
        st.session_state.opt_lineup = _build_optimal_lineup(
            pool, salary_cap, roster_size, locked=locked, excluded=excluded)

    lineup = st.session_state.opt_lineup

    # Lineup header
    if lineup:
        total_sal = sum(d["DK Salary"] for d in lineup)
        total_pts = sum(d["Proj Score"] for d in lineup)
        remaining = salary_cap - total_sal

        hdr = st.columns([2, 1, 1, 1, 1])
        hdr[0].markdown(f"**Optimal Lineup** — *{proj_source}*")
        hdr[1].metric("Projected", f"{total_pts:.1f}")
        hdr[2].metric("Salary", f"${total_sal:,}")
        hdr[3].metric("Remaining", f"${remaining:,}")
        with hdr[4]:
            if st.button("Re-Optimize", key="opt_reoptimize", type="primary"):
                st.session_state.opt_lineup = _build_optimal_lineup(
                    pool, salary_cap, roster_size,
                    locked=locked, excluded=excluded)
                st.rerun()

        # Lineup table with swap controls
        for slot_idx, driver_data in enumerate(
                sorted(lineup, key=lambda x: x["Proj Score"], reverse=True)):
            driver = driver_data["Driver"]
            is_locked = driver in st.session_state.opt_locked

            row_cols = st.columns([0.4, 0.4, 2.5, 1, 1, 1, 2])

            # Lock toggle
            with row_cols[0]:
                lock_key = f"lock_{slot_idx}"
                if st.checkbox("🔒", value=is_locked, key=lock_key, label_visibility="collapsed"):
                    st.session_state.opt_locked.add(driver)
                elif driver in st.session_state.opt_locked:
                    st.session_state.opt_locked.discard(driver)

            # Exclude (remove from lineup)
            with row_cols[1]:
                if st.button("✕", key=f"excl_{slot_idx}"):
                    st.session_state.opt_excluded.add(driver)
                    st.session_state.opt_locked.discard(driver)
                    # Rebuild lineup without this driver
                    st.session_state.opt_lineup = _build_optimal_lineup(
                        pool, salary_cap, roster_size,
                        locked=list(st.session_state.opt_locked),
                        excluded=st.session_state.opt_excluded)
                    st.rerun()

            # Driver info
            row_cols[2].markdown(f"**{driver}**" if is_locked else driver)
            row_cols[3].markdown(f"${driver_data['DK Salary']:,}")
            row_cols[4].markdown(f"{driver_data['Proj Score']:.1f}")
            row_cols[5].markdown(f"{driver_data.get('Value', 0):.2f}x")

            # Swap dropdown
            with row_cols[6]:
                swap_candidates = _get_swap_candidates(
                    pool, lineup, driver, salary_cap, roster_size)
                if not swap_candidates.empty:
                    swap_options = ["Swap..."] + [
                        f"{r['Driver']} (${r['DK Salary']:,} | {r['Proj Score']:.1f})"
                        for _, r in swap_candidates.head(10).iterrows()
                    ]
                    swap_pick = st.selectbox(
                        "swap", swap_options, key=f"swap_{slot_idx}",
                        label_visibility="collapsed")
                    if swap_pick != "Swap...":
                        # Extract driver name from selection
                        swap_name = swap_pick.split(" ($")[0]
                        new_driver = swap_candidates[
                            swap_candidates["Driver"] == swap_name].iloc[0].to_dict()
                        # Replace in lineup
                        new_lineup = [d for d in st.session_state.opt_lineup
                                      if d["Driver"] != driver]
                        new_lineup.append(new_driver)
                        st.session_state.opt_lineup = new_lineup
                        st.session_state.opt_locked.discard(driver)
                        st.rerun()

    else:
        st.warning("Could not build a valid lineup within the salary cap.")

    # ─── PLAYER POOL ────────────────────────────────────────────────────────
    st.markdown("---")

    with st.expander("Player Pool", expanded=True):
        pool_display = pool.copy()

        # Mark status
        lineup_drivers = {d["Driver"] for d in lineup} if lineup else set()
        pool_display["Status"] = pool_display["Driver"].apply(
            lambda d: "🔒 Locked" if d in st.session_state.opt_locked
            else ("✕ Excluded" if d in st.session_state.opt_excluded
                  else ("In Lineup" if d in lineup_drivers else "")))

        pool_display["Rank"] = range(1, len(pool_display) + 1)
        show_cols = ["Rank", "Status", "Driver", "DK Salary", "Proj Score", "Value"]
        avail = [c for c in show_cols if c in pool_display.columns]

        # Search
        search = st.text_input("Search players...", "", key="pool_search",
                                label_visibility="collapsed",
                                placeholder="Search drivers...")
        if search:
            pool_display = pool_display[
                pool_display["Driver"].str.contains(search, case=False, na=False)]

        disp = format_display_df(pool_display[avail].copy())
        st.dataframe(safe_fillna(disp), width="stretch", hide_index=True, height=400)

        # Quick lock/exclude controls
        lk_col, ex_col, clr_col = st.columns([2, 2, 1])
        with lk_col:
            all_drivers = sorted(pool["Driver"].dropna().unique())
            to_lock = st.multiselect("Lock drivers", all_drivers,
                                      default=list(st.session_state.opt_locked),
                                      key="opt_lock_multi")
            if set(to_lock) != st.session_state.opt_locked:
                st.session_state.opt_locked = set(to_lock)
                st.session_state.opt_lineup = _build_optimal_lineup(
                    pool, salary_cap, roster_size,
                    locked=list(st.session_state.opt_locked),
                    excluded=st.session_state.opt_excluded)
                st.rerun()

        with ex_col:
            to_exclude = st.multiselect("Exclude drivers",
                                         [d for d in all_drivers
                                          if d not in st.session_state.opt_locked],
                                         default=list(st.session_state.opt_excluded),
                                         key="opt_exclude_multi")
            if set(to_exclude) != st.session_state.opt_excluded:
                st.session_state.opt_excluded = set(to_exclude)
                st.session_state.opt_lineup = _build_optimal_lineup(
                    pool, salary_cap, roster_size,
                    locked=list(st.session_state.opt_locked),
                    excluded=st.session_state.opt_excluded)
                st.rerun()

        with clr_col:
            if st.button("Clear All", key="opt_clear"):
                st.session_state.opt_locked = set()
                st.session_state.opt_excluded = set()
                st.session_state.opt_lineup = []
                st.session_state.opt_multi_lineups = []
                st.rerun()

    # ─── MULTI-LINEUP GENERATION ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Multi-Lineup Generator**")

    ml_cols = st.columns([1, 1, 2])
    with ml_cols[0]:
        num_lineups = st.number_input("# Lineups", 1, 150, 20, 5, key="opt_num_lineups")
    with ml_cols[1]:
        if st.button("Generate Lineups", type="primary", key="opt_gen_multi"):
            with st.spinner(f"Generating {num_lineups} {mode} lineups..."):
                try:
                    multi = _generate_lineups_greedy(
                        pool, salary_cap, roster_size, num_lineups,
                        max_exposure, mode.lower(),
                        locked=list(st.session_state.opt_locked),
                        excluded=st.session_state.opt_excluded)
                    st.session_state.opt_multi_lineups = multi
                    if not multi:
                        st.warning("No valid lineups found within salary cap.")
                except Exception as e:
                    st.error(f"Lineup generation failed: {e}")
                    st.session_state.opt_multi_lineups = []

    multi_lineups = st.session_state.opt_multi_lineups

    if multi_lineups:
        ml_header = st.columns([3, 1])
        with ml_header[0]:
            st.success(f"{len(multi_lineups)} lineups generated")
        with ml_header[1]:
            if st.button("Clear Lineups", key="opt_clear_multi"):
                st.session_state.opt_multi_lineups = []
                st.rerun()

        # Exposure summary
        exposure = {}
        for lu in multi_lineups:
            for d in lu:
                exposure[d["Driver"]] = exposure.get(d["Driver"], 0) + 1

        with st.expander("Exposure Summary", expanded=True):
            exp_rows = []
            for driver, count in sorted(exposure.items(), key=lambda x: x[1], reverse=True):
                match = pool[pool["Driver"] == driver]
                exp_rows.append({
                    "Driver": driver,
                    "Count": count,
                    "Exposure": f"{count / len(multi_lineups) * 100:.0f}%",
                    "Proj Score": match.iloc[0]["Proj Score"] if not match.empty else 0,
                    "DK Salary": match.iloc[0]["DK Salary"] if not match.empty else 0,
                })
            exp_df = format_display_df(pd.DataFrame(exp_rows))
            st.dataframe(safe_fillna(exp_df), width="stretch", hide_index=True, height=350)

        # Lineup cards
        for i, lu in enumerate(multi_lineups):
            total_sal = sum(d["DK Salary"] for d in lu)
            total_pts = sum(d["Proj Score"] for d in lu)
            remaining = salary_cap - total_sal

            with st.expander(
                f"Lineup {i + 1} — {total_pts:.1f} pts | ${total_sal:,} | ${remaining:,} left",
                expanded=(i < 3)):
                lu_df = pd.DataFrame([{
                    "Driver": d["Driver"],
                    "DK Salary": f"${d['DK Salary']:,}",
                    "Proj Score": round(d["Proj Score"], 1),
                    "Value": round(d.get("Value", 0), 2),
                } for d in sorted(lu, key=lambda x: x["Proj Score"], reverse=True)])
                st.dataframe(lu_df, width="stretch", hide_index=True)

        # Export
        st.markdown("---")
        exp_cols = st.columns(2)
        with exp_cols[0]:
            dk_rows = []
            for lu in multi_lineups:
                row = {}
                for j, d in enumerate(sorted(lu, key=lambda x: x["Proj Score"], reverse=True)):
                    row[f"D{j + 1}"] = d["Driver"]
                row["Salary"] = sum(d["DK Salary"] for d in lu)
                row["Projected"] = round(sum(d["Proj Score"] for d in lu), 1)
                dk_rows.append(row)
            dk_csv = pd.DataFrame(dk_rows).to_csv(index=False).encode("utf-8")
            st.download_button("Export DK Import CSV", dk_csv,
                               f"{race_name.replace(' ', '_')}_lineups.csv",
                               "text/csv", key="opt_export_dk")
        with exp_cols[1]:
            detail_rows = []
            for i, lu in enumerate(multi_lineups):
                for d in lu:
                    detail_rows.append({
                        "Lineup": i + 1, "Driver": d["Driver"],
                        "DK Salary": d["DK Salary"],
                        "Proj Score": round(d["Proj Score"], 1),
                        "Value": round(d.get("Value", 0), 2),
                    })
            detail_csv = pd.DataFrame(detail_rows).to_csv(index=False).encode("utf-8")
            st.download_button("Export Detailed CSV", detail_csv,
                               f"{race_name.replace(' ', '_')}_lineups_detail.csv",
                               "text/csv", key="opt_export_detail")

    # ─── SALARY TIERS (always visible) ──────────────────────────────────────
    st.markdown("---")
    st.markdown("**Salary Tiers**")
    tiers = pool.copy()
    tiers["Tier"] = pd.cut(tiers["DK Salary"],
                           bins=[0, 6000, 7500, 9000, 11000, 20000],
                           labels=["$5k-6k", "$6k-7.5k", "$7.5k-9k", "$9k-11k", "$11k+"])
    tier_summary = tiers.groupby("Tier", observed=True).agg(
        Count=("Driver", "count"),
        Avg_Proj=("Proj Score", "mean"),
        Avg_Value=("Value", "mean"),
    ).round(1)
    st.dataframe(tier_summary, width="stretch")
