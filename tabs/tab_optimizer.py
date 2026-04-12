"""Tab 6: Lineup Optimizer — with player pool lock/exclude and multi-lineup generator."""

import random
import time
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
import os

from src.config import SALARY_CAP, ROSTER_SIZE, TRACK_TYPE_MAP, TRACK_TYPE_PARENT, TRACK_TYPE_WEIGHT_DEFAULTS
from src.utils import safe_fillna, format_display_df, fuzzy_match_name, fuzzy_get, build_norm_lookup
from src.charts import salary_vs_projection_scatter


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
    # Defaults from shared config — 6 signals: odds, track, ttype, prac, team, qual
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    defaults = TRACK_TYPE_WEIGHT_DEFAULTS.get(parent_type, TRACK_TYPE_WEIGHT_DEFAULTS["intermediate"])
    w_track = st.session_state.get("pw_track", defaults["track"])
    w_ttype = st.session_state.get("pw_ttype", defaults["ttype"])
    w_odds  = st.session_state.get("pw_odds", defaults["odds"])
    w_prac  = st.session_state.get("pw_prac", defaults["prac"])
    w_team  = st.session_state.get("pw_team", defaults["team"])
    w_qual  = st.session_state.get("pw_qual", defaults["qual"])

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

def _solve_optimal(drivers, salary_cap, roster_size, timeout_ms=3000):
    """Find the mathematically optimal lineup using branch-and-bound.

    Maximizes total Proj Score subject to salary cap and roster size.
    drivers: list of dicts with Driver, DK Salary, Proj Score.
    timeout_ms: max milliseconds before returning best solution found so far.
    Returns list of driver dicts for the best lineup.
    """
    # Sort by projection descending for better pruning
    drivers = sorted(drivers, key=lambda d: d["Proj Score"], reverse=True)
    n = len(drivers)
    projs = [d["Proj Score"] for d in drivers]
    sals = [d["DK Salary"] for d in drivers]

    best_score = [0.0]
    best_lineup = [[]]
    start_time = time.time()
    deadline = start_time + timeout_ms / 1000.0
    timed_out = [False]

    def branch_and_bound(idx, chosen, total_proj, total_sal, slots_left):
        if timed_out[0]:
            return
        if slots_left == 0:
            if total_proj > best_score[0]:
                best_score[0] = total_proj
                best_lineup[0] = list(chosen)
            return
        if idx >= n:
            return
        remaining = n - idx
        if remaining < slots_left:
            return

        # Check timeout periodically (every 10000 nodes)
        if (idx + len(chosen)) % 50 == 0 and time.time() > deadline:
            timed_out[0] = True
            return

        # Upper bound: current total + best possible from remaining slots
        upper = total_proj + sum(projs[idx:idx + slots_left])
        if upper <= best_score[0]:
            return

        # Branch: include driver[idx]
        new_sal = total_sal + sals[idx]
        if new_sal <= salary_cap:
            chosen.append(idx)
            branch_and_bound(idx + 1, chosen, total_proj + projs[idx],
                             new_sal, slots_left - 1)
            chosen.pop()

        # Branch: exclude driver[idx]
        branch_and_bound(idx + 1, chosen, total_proj, total_sal, slots_left)

    branch_and_bound(0, [], 0.0, 0, roster_size)
    return [drivers[i] for i in best_lineup[0]]


def _build_optimal_lineup(pool_df, salary_cap, roster_size, locked=None, excluded=None):
    """Build the single best lineup using branch-and-bound solver."""
    locked = locked or []
    excluded = excluded or set()

    available = pool_df[~pool_df["Driver"].isin(excluded)].copy()
    available["Proj Score"] = available["Proj Score"].fillna(0)
    available["DK Salary"] = available["DK Salary"].fillna(0)

    # Handle locked drivers
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
        return locked_data

    candidates = variable_pool.to_dict("records")
    optimal = _solve_optimal(candidates, remaining_cap, remaining_slots)
    return locked_data + optimal


def _get_swap_candidates(pool_df, current_lineup, swap_driver, salary_cap, roster_size):
    """Get ranked replacement candidates for a driver being swapped out."""
    lineup_drivers = {d["Driver"] for d in current_lineup}
    lineup_salary = sum(d["DK Salary"] for d in current_lineup)
    swap_salary = next((d["DK Salary"] for d in current_lineup
                        if d["Driver"] == swap_driver), 0)
    budget = salary_cap - (lineup_salary - swap_salary)
    candidates = pool_df[
        (~pool_df["Driver"].isin(lineup_drivers)) &
        (pool_df["DK Salary"] <= budget)
    ].copy()
    candidates = candidates.sort_values("Proj Score", ascending=False)
    return candidates


def _generate_lineups(pool_df, salary_cap, roster_size, num_lineups,
                       max_exposure, mode="gpp", locked=None, excluded=None):
    """Generate multiple optimized lineups with exposure limits.

    Locked drivers bypass max exposure (always included).
    Uses a faster approach: solve once, then swap drivers to create diversity.
    """
    locked = locked or []
    excluded = excluded or set()
    locked_set = set(locked)

    available = pool_df[~pool_df["Driver"].isin(excluded)].copy()
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

    candidates = variable_pool.to_dict("records")
    all_lineups = []
    seen_keys = set()
    exposure_count = {}
    # Locked drivers get unlimited exposure
    max_exp = max(1, int(num_lineups * max_exposure / 100)) if max_exposure < 100 else num_lineups

    def _add_lineup(lineup):
        key = tuple(sorted(d["Driver"] for d in lineup))
        if key in seen_keys:
            return False
        # Check exposure limits (locked drivers exempt)
        for d in lineup:
            if d["Driver"] not in locked_set:
                if exposure_count.get(d["Driver"], 0) >= max_exp:
                    return False
        seen_keys.add(key)
        total_pts = sum(d["Proj Score"] for d in lineup)
        all_lineups.append((total_pts, lineup))
        for d in lineup:
            exposure_count[d["Driver"]] = exposure_count.get(d["Driver"], 0) + 1
        return True

    # Lineup 1: true optimal via branch-and-bound (generous timeout)
    optimal = _solve_optimal(candidates, remaining_cap, remaining_slots, timeout_ms=5000)
    if optimal:
        _add_lineup(locked_data + optimal)

    # Generate diverse lineups by excluding 1-2 drivers from previous lineups
    # Use shorter timeout for subsequent solves since they're generating variety
    attempts = 0
    max_attempts = num_lineups * 8

    while len(all_lineups) < num_lineups and attempts < max_attempts:
        attempts += 1

        if not all_lineups:
            break

        # Pick a random previous lineup and exclude 1-2 of its variable drivers
        base_idx = random.randint(0, len(all_lineups) - 1)
        base = all_lineups[base_idx][1]
        variable_drivers = [d for d in base if d["Driver"] not in locked_set]

        if not variable_drivers:
            continue

        # Exclude 1 or 2 drivers to force diversity
        n_exclude = random.randint(1, min(2, len(variable_drivers)))
        to_exclude = random.sample(variable_drivers, n_exclude)
        exclude_names = {d["Driver"] for d in to_exclude}

        # Also filter out over-exposed drivers
        reduced = [d for d in candidates
                   if d["Driver"] not in exclude_names
                   and exposure_count.get(d["Driver"], 0) < max_exp]
        if len(reduced) < remaining_slots:
            continue

        alt = _solve_optimal(reduced, remaining_cap, remaining_slots, timeout_ms=1500)
        if alt:
            _add_lineup(locked_data + alt)

    # Sort by total projected points
    all_lineups.sort(key=lambda x: x[0], reverse=True)
    return [lu for _, lu in all_lineups[:num_lineups]]


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           is_prerace, race_name, race_id, track_name, series_id, dk_df,
           odds_data=None):
    """Render the Optimizer tab."""
    from src.components import section_header
    section_header("Lineup Optimizer", race_name)

    if dk_df.empty:
        # Try loading salaries from DB for completed races
        from src.data import query_salaries
        db_sal = query_salaries(race_id=race_id, platform="DraftKings")
        if not db_sal.empty and "Salary" in db_sal.columns:
            dk_df = db_sal.rename(columns={"Salary": "DK Salary"})[["Driver", "DK Salary"]].copy()
        if dk_df.empty:
            if not is_prerace:
                st.warning("No saved salary data for this race. Upload a DK CSV below to add it.")
                dk_upload = st.file_uploader(
                    "Upload DK Salary CSV for this race",
                    type=["csv"], key=f"opt_dk_upload_{race_id}")
                if dk_upload:
                    from src.data import parse_dk_csv, sync_dk_salaries_to_db
                    dk_df = parse_dk_csv(dk_upload)
                    if not dk_df.empty:
                        count = sync_dk_salaries_to_db(dk_df, race_id, series_id, race_name)
                        st.success(f"Saved {count} DK salaries to DB for {race_name}")
                        st.rerun()
                    else:
                        st.error("Could not parse DK CSV")
                return
            else:
                st.info("No salary data available. Upload a DK CSV in Settings to enable the optimizer.")
                return

    # Initialize session state — clear when race/series changes
    race_key = f"{series_id}_{race_id}"
    if st.session_state.get("opt_race_key") != race_key:
        st.session_state.opt_race_key = race_key
        st.session_state.opt_lineup = []
        st.session_state.opt_locked = set()
        st.session_state.opt_excluded = set()
        st.session_state.opt_multi_lineups = []
        st.session_state.opt_overrides = {}
        st.session_state.opt_pool_expanded = False
    if "opt_lineup" not in st.session_state:
        st.session_state.opt_lineup = []
    if "opt_locked" not in st.session_state:
        st.session_state.opt_locked = set()
    if "opt_excluded" not in st.session_state:
        st.session_state.opt_excluded = set()
    if "opt_multi_lineups" not in st.session_state:
        st.session_state.opt_multi_lineups = []
    if "opt_overrides" not in st.session_state:
        st.session_state.opt_overrides = {}

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
                                 help="Max % of lineups a non-locked driver can appear in")

    # Build projection pool (uses same weights as Projections tab)
    with st.spinner("Building projections..."):
        pool, engine_label = _get_projection_pool(
            entry_list_df, qualifying_df, lap_averages_df,
            practice_data, race_name, track_name, series_id,
            dk_df, odds_data=odds_data)

    if pool.empty:
        st.warning("Could not build projection pool. Check salary data.")
        return

    # Apply overrides to pool BEFORE anything else uses it
    for drv, pts in st.session_state.opt_overrides.items():
        mask = pool["Driver"] == drv
        if mask.any():
            pool.loc[mask, "Proj Score"] = pts
            sal = pool.loc[mask, "DK Salary"].values[0]
            pool.loc[mask, "Value"] = round(pts / (sal / 1000), 2) if sal > 0 else 0

    proj_source = engine_label

    # ─── PLAYER POOL WITH LOCK/EXCLUDE/OVERRIDES ──────────────────────────
    st.divider()
    n_locked = len(st.session_state.opt_locked)
    n_excluded = len(st.session_state.opt_excluded)
    n_overrides = len(st.session_state.opt_overrides)
    pool_label = "Player Pool"
    status_parts = []
    if n_locked:
        status_parts.append(f"{n_locked} locked")
    if n_excluded:
        status_parts.append(f"{n_excluded} excluded")
    if n_overrides:
        status_parts.append(f"{n_overrides} overrides")
    if status_parts:
        pool_label += f"  ({', '.join(status_parts)})"

    # Keep expander open if user was interacting with player pool
    pool_expanded = st.session_state.pop("opt_pool_expanded", False)
    with st.expander(pool_label, expanded=pool_expanded):
        pool_display = pool.copy()
        lineup_drivers = {d["Driver"] for d in st.session_state.opt_lineup} if st.session_state.opt_lineup else set()

        # Build status column
        pool_display["In Lineup"] = pool_display["Driver"].apply(
            lambda d: "Yes" if d in lineup_drivers else "")
        pool_display["Rank"] = range(1, len(pool_display) + 1)

        # Search filter
        search = st.text_input("Search players...", "", key="pool_search",
                                label_visibility="collapsed",
                                placeholder="Search drivers...")
        if search:
            pool_display = pool_display[
                pool_display["Driver"].str.contains(search, case=False, na=False)]

        # Reset All button — above the driver list
        if st.button("Reset All (Locks, Excludes, Overrides)", key="opt_reset_all",
                      type="secondary"):
            st.session_state.opt_locked = set()
            st.session_state.opt_excluded = set()
            st.session_state.opt_overrides = {}
            st.session_state.opt_lineup = []
            st.session_state.opt_multi_lineups = []
            st.session_state.opt_pool_expanded = False
            # Clear all widget keys so checkboxes/overrides reset on next render
            for k in list(st.session_state.keys()):
                if k.startswith(("pp_lock_", "pp_excl_", "pp_ov_", "lu_lock_", "lu_excl_")):
                    del st.session_state[k]
            st.rerun()

        # Lock/Exclude/Override in a table-like layout
        # Header row
        hdr_cols = st.columns([0.35, 0.35, 2.2, 0.9, 0.9, 0.9, 0.9, 0.7])
        hdr_cols[0].markdown("**L**")
        hdr_cols[1].markdown("**X**")
        hdr_cols[2].markdown("**Driver**")
        hdr_cols[3].markdown("**Salary**")
        hdr_cols[4].markdown("**Proj**")
        hdr_cols[5].markdown("**Override**")
        hdr_cols[6].markdown("**Value**")
        hdr_cols[7].markdown("**Status**")

        # Scrollable player rows
        pool_rows = pool_display.head(50).to_dict("records")
        needs_rerun = False

        for i, row in enumerate(pool_rows):
            driver = row["Driver"]
            is_locked = driver in st.session_state.opt_locked
            is_excluded = driver in st.session_state.opt_excluded
            in_lineup = driver in lineup_drivers
            has_override = driver in st.session_state.opt_overrides

            r = st.columns([0.35, 0.35, 2.2, 0.9, 0.9, 0.9, 0.9, 0.7])

            with r[0]:
                new_lock = st.checkbox("L", value=is_locked, key=f"pp_lock_{i}",
                                        label_visibility="collapsed")
                if new_lock != is_locked:
                    if new_lock:
                        st.session_state.opt_locked.add(driver)
                        st.session_state.opt_excluded.discard(driver)
                    else:
                        st.session_state.opt_locked.discard(driver)
                    needs_rerun = True

            with r[1]:
                new_excl = st.checkbox("X", value=is_excluded, key=f"pp_excl_{i}",
                                        label_visibility="collapsed")
                if new_excl != is_excluded:
                    if new_excl:
                        st.session_state.opt_excluded.add(driver)
                        st.session_state.opt_locked.discard(driver)
                    else:
                        st.session_state.opt_excluded.discard(driver)
                    needs_rerun = True

            # Driver name: bold if locked, strikethrough if excluded
            name_style = f"**{driver}**" if is_locked else (f"~~{driver}~~" if is_excluded else driver)
            r[2].markdown(name_style)
            r[3].markdown(f"${row['DK Salary']:,.0f}")
            r[4].markdown(f"{row['Proj Score']:.1f}")

            # Override input — small number field
            with r[5]:
                override_val = st.number_input(
                    "ov", min_value=0.0, max_value=300.0,
                    value=float(st.session_state.opt_overrides.get(driver, 0)),
                    step=1.0, key=f"pp_ov_{i}",
                    label_visibility="collapsed",
                    format="%.0f",
                )
                # 0 means no override; any positive value is an override
                if override_val > 0 and override_val != st.session_state.opt_overrides.get(driver):
                    st.session_state.opt_overrides[driver] = override_val
                    needs_rerun = True
                elif override_val == 0 and driver in st.session_state.opt_overrides:
                    del st.session_state.opt_overrides[driver]
                    needs_rerun = True

            r[6].markdown(f"{row['Value']:.2f}x")

            status = ""
            if is_locked:
                status = "Locked"
            elif is_excluded:
                status = "Out"
            elif in_lineup:
                status = "In"
            r[7].markdown(status)

        if needs_rerun:
            # Re-apply overrides to pool before re-optimizing
            for drv, pts in st.session_state.opt_overrides.items():
                mask = pool["Driver"] == drv
                if mask.any():
                    pool.loc[mask, "Proj Score"] = pts
                    sal = pool.loc[mask, "DK Salary"].values[0]
                    pool.loc[mask, "Value"] = round(pts / (sal / 1000), 2) if sal > 0 else 0
            st.session_state.opt_lineup = _build_optimal_lineup(
                pool, salary_cap, roster_size,
                locked=list(st.session_state.opt_locked),
                excluded=st.session_state.opt_excluded)
            # Keep player pool open so user can continue making changes
            st.session_state.opt_pool_expanded = True
            st.rerun()

    # Salary vs Projection scatter
    sal_fig = salary_vs_projection_scatter(pool)
    if sal_fig:
        st.plotly_chart(sal_fig, use_container_width=True)

    # ─── OPTIMAL LINEUP PANEL ───────────────────────────────────────────────
    st.divider()

    # Auto-build optimal lineup if none exists yet
    locked = list(st.session_state.opt_locked)
    excluded = st.session_state.opt_excluded

    if not st.session_state.opt_lineup:
        st.session_state.opt_lineup = _build_optimal_lineup(
            pool, salary_cap, roster_size, locked=locked, excluded=excluded)

    # Sync lineup scores with current pool (overrides may have changed)
    lineup = st.session_state.opt_lineup
    if lineup:
        pool_scores = pool.set_index("Driver")["Proj Score"].to_dict()
        pool_values = pool.set_index("Driver")["Value"].to_dict()
        for d in lineup:
            if d["Driver"] in pool_scores:
                d["Proj Score"] = pool_scores[d["Driver"]]
                d["Value"] = pool_values.get(d["Driver"], d.get("Value", 0))

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
                # Clear swap keys so dropdowns reset
                for k in list(st.session_state.keys()):
                    if k.startswith("swap_"):
                        del st.session_state[k]
                st.session_state.opt_lineup = _build_optimal_lineup(
                    pool, salary_cap, roster_size,
                    locked=locked, excluded=excluded)
                st.rerun()

        # Lineup table with swap controls
        sorted_lineup = sorted(lineup, key=lambda x: x["Proj Score"], reverse=True)
        for slot_idx, driver_data in enumerate(sorted_lineup):
            driver = driver_data["Driver"]
            is_locked = driver in st.session_state.opt_locked

            row_cols = st.columns([0.35, 0.35, 2.2, 0.9, 0.9, 0.9, 2.2])

            # Lock toggle
            with row_cols[0]:
                new_lock = st.checkbox("L", value=is_locked,
                                        key=f"lu_lock_{slot_idx}",
                                        label_visibility="collapsed")
                if new_lock and not is_locked:
                    st.session_state.opt_locked.add(driver)
                elif not new_lock and is_locked:
                    st.session_state.opt_locked.discard(driver)

            # Exclude (remove from lineup and add to excluded set)
            with row_cols[1]:
                if st.button("X", key=f"lu_excl_{slot_idx}"):
                    st.session_state.opt_excluded.add(driver)
                    st.session_state.opt_locked.discard(driver)
                    # Clear swap keys so dropdowns reset
                    for k in list(st.session_state.keys()):
                        if k.startswith("swap_"):
                            del st.session_state[k]
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

            # Swap dropdown — use on_change callback pattern
            with row_cols[6]:
                swap_candidates = _get_swap_candidates(
                    pool, lineup, driver, salary_cap, roster_size)
                if not swap_candidates.empty:
                    swap_options = ["Swap..."] + [
                        f"{r['Driver']} (${r['DK Salary']:,} | {r['Proj Score']:.1f})"
                        for _, r in swap_candidates.head(10).iterrows()
                    ]
                    swap_key = f"swap_{slot_idx}"
                    swap_pick = st.selectbox(
                        "swap", swap_options, key=swap_key,
                        label_visibility="collapsed")
                    if swap_pick != "Swap...":
                        swap_name = swap_pick.split(" ($")[0]
                        new_driver = swap_candidates[
                            swap_candidates["Driver"] == swap_name].iloc[0].to_dict()
                        # Replace in lineup
                        new_lineup = [d for d in st.session_state.opt_lineup
                                      if d["Driver"] != driver]
                        new_lineup.append(new_driver)
                        st.session_state.opt_lineup = new_lineup
                        # Remove lock from swapped-out driver
                        st.session_state.opt_locked.discard(driver)
                        # Clear ALL swap keys so dropdowns reset to "Swap..."
                        for k in list(st.session_state.keys()):
                            if k.startswith("swap_"):
                                del st.session_state[k]
                        st.rerun()

    else:
        st.warning("Could not build a valid lineup within the salary cap.")

    # ─── MULTI-LINEUP GENERATION ────────────────────────────────────────────
    st.divider()
    st.markdown("**Multi-Lineup Generator**")

    ml_cols = st.columns([1, 1, 2])
    with ml_cols[0]:
        num_lineups = st.number_input("# Lineups", 1, 150, 20, 5, key="opt_num_lineups")
    with ml_cols[1]:
        if st.button("Generate Lineups", type="primary", key="opt_gen_multi"):
            with st.spinner(f"Generating {num_lineups} {mode} lineups..."):
                try:
                    multi = _generate_lineups(
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
                is_lk = driver in st.session_state.opt_locked
                exp_rows.append({
                    "Driver": driver,
                    "Count": count,
                    "Exposure": f"{count / len(multi_lineups) * 100:.0f}%",
                    "Locked": "Yes" if is_lk else "",
                    "Proj Score": match.iloc[0]["Proj Score"] if not match.empty else 0,
                    "DK Salary": match.iloc[0]["DK Salary"] if not match.empty else 0,
                })
            exp_df = format_display_df(pd.DataFrame(exp_rows))
            st.dataframe(safe_fillna(exp_df), use_container_width=True, hide_index=True, height=350)

        # Lineup cards
        for i, lu in enumerate(multi_lineups):
            total_sal = sum(d["DK Salary"] for d in lu)
            total_pts = sum(d["Proj Score"] for d in lu)
            remaining = salary_cap - total_sal

            with st.expander(
                f"Lineup {i + 1} -- {total_pts:.1f} pts | ${total_sal:,} | ${remaining:,} left",
                expanded=(i < 3)):
                lu_df = pd.DataFrame([{
                    "Driver": d["Driver"],
                    "DK Salary": f"${d['DK Salary']:,}",
                    "Proj Score": round(d["Proj Score"], 1),
                    "Value": round(d.get("Value", 0), 2),
                } for d in sorted(lu, key=lambda x: x["Proj Score"], reverse=True)])
                st.dataframe(lu_df, use_container_width=True, hide_index=True)

        # Export
        st.divider()
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
    st.divider()
    st.markdown("**Salary Tiers**")
    tiers = pool.copy()
    tiers["Tier"] = pd.cut(tiers["DK Salary"],
                           bins=[0, 5000, 6000, 7500, 9000, 11000, 20000],
                           labels=["$4k-5k", "$5k-6k", "$6k-7.5k", "$7.5k-9k", "$9k-11k", "$11k+"])
    tier_summary = tiers.groupby("Tier", observed=True).agg(
        Count=("Driver", "count"),
        Avg_Proj=("Proj Score", "mean"),
        Avg_Value=("Value", "mean"),
    ).round(1)
    st.dataframe(tier_summary, use_container_width=True)
