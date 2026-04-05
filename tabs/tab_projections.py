"""Tab 5: Projections — DFS-Optimized Projection Engine.

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

from src.config import (
    DEFAULT_PROJECTION_WEIGHTS, DB_PATH, TRACK_TYPE_MAP,
    DK_FINISH_POINTS,
)
from src.data import (
    query_projections, scrape_track_history,
    fetch_weekend_feed,
)
from src.charts import projection_bar
from src.utils import safe_fillna, format_display_df, calc_dk_points

PROJ_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_db_race_id(series_id, race_name, track_name):
    """Try to find a matching race_id in the database."""
    if not os.path.exists(PROJ_DB):
        return None
    try:
        conn = sqlite3.connect(PROJ_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id FROM races WHERE series_id = ? AND race_name = ? ORDER BY season DESC LIMIT 1",
            (series_id, race_name)
        ).fetchone()
        if row:
            conn.close()
            return row["id"]
        row = conn.execute("""
            SELECT r.id FROM races r JOIN tracks t ON t.id = r.track_id
            WHERE r.series_id = ? AND t.name LIKE ?
            ORDER BY r.season DESC, r.race_num DESC LIMIT 1
        """, (series_id, f"%{track_name}%")).fetchone()
        conn.close()
        return row["id"] if row else None
    except Exception:
        return None


def _run_projection_engine(race_id, platform, weights):
    """Run the real 6-component projection engine."""
    try:
        from projections import get_conn, project_race
        conn = get_conn()
        projections = project_race(conn, race_id, platform, weights)
        conn.close()
        return projections
    except Exception as e:
        st.warning(f"Projection engine error: {e}")
        return []


def _get_race_laps(feed):
    """Extract total laps for the race from weekend feed."""
    if not feed:
        return 0
    races = feed.get("weekend_race", [])
    if races:
        return races[0].get("number_of_laps") or races[0].get("laps") or 0
    return 0


def _expected_finish_from_avg(avg_finish, field_size=38):
    """Convert an average finish to expected DK finish points."""
    # Clamp to 1-40 range
    ef = max(1, min(40, round(avg_finish)))
    return DK_FINISH_POINTS.get(ef, 0)


def _dominator_share(laps_led_history, total_laps_history, races):
    """Estimate what fraction of laps a driver leads per race."""
    if not races or races == 0 or not total_laps_history:
        return 0.0
    return (laps_led_history / races) / max(total_laps_history / races, 1)


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           is_prerace, race_name, race_id, track_name, series_id, dk_df,
           odds_data=None):
    """Render the Projections tab."""
    st.markdown(f"### Projections — {race_name}")

    if not is_prerace:
        st.caption("Race completed — projections shown for review")

    if entry_list_df.empty and dk_df.empty:
        st.warning("Entry list not available for this race.")
        return

    # Get race laps for dominator calculations
    feed = fetch_weekend_feed(series_id, race_id)
    race_laps = _get_race_laps(feed)

    # Weight sliders in collapsible expander
    with st.expander("Projection Weights", expanded=False):
        st.caption("Adjust signal weights — auto-normalizes to 100%")
        w_cols = st.columns(5)
        w_track = w_cols[0].number_input("Track History", 0, 100, 30, 5, key="pw_track")
        w_type = w_cols[1].number_input("Track Type", 0, 100, 20, 5, key="pw_type")
        w_qual = w_cols[2].number_input("Qualifying", 0, 100, 15, 5, key="pw_qual")
        w_prac = w_cols[3].number_input("Practice", 0, 100, 20, 5, key="pw_prac")
        w_odds = w_cols[4].number_input("Odds", 0, 100, 15, 5, key="pw_odds")

    # Smart weight handling: if odds not available, redistribute that weight
    has_odds = bool(odds_data)
    effective_odds = w_odds if has_odds else 0
    raw_total = w_track + w_type + w_qual + w_prac + effective_odds
    if raw_total > 0:
        wn = {
            "track": w_track / raw_total,
            "track_type": w_type / raw_total,
            "qual": w_qual / raw_total,
            "practice": w_prac / raw_total,
            "odds": effective_odds / raw_total,
        }
    else:
        wn = {"track": 0.30, "track_type": 0.20, "qual": 0.15, "practice": 0.20, "odds": 0.15}

    if not has_odds:
        st.caption("Odds data not available — odds weight redistributed to other signals")

    # Race info bar
    if race_laps > 0:
        max_dom_laps_led = race_laps * 0.25  # top dominator leads ~25% of laps
        max_dom_fastest = race_laps * 0.15   # top dominator wins ~15% of fastest laps
        max_dom_pts = max_dom_laps_led * 0.25 + max_dom_fastest * 0.45

        info_cols = st.columns(4)
        info_cols[0].metric("Race Laps", f"{race_laps}")
        info_cols[1].metric("Max Laps Led Pts", f"{race_laps * 0.25:.1f}")
        info_cols[2].metric("Max Fastest Lap Pts", f"{race_laps * 0.45:.1f}")
        info_cols[3].metric("Dominator Ceiling", f"{max_dom_pts:.1f}")
        st.caption(f"Laps led = 0.25 pts/lap | Fastest laps = 0.45 pts/lap | "
                   f"Place diff = 1.0 pts/pos | {race_laps} total laps available")

    # Build projections
    _build_dfs_projections(
        entry_list_df, qualifying_df, lap_averages_df,
        practice_data, wn, track_name, series_id, dk_df, race_laps,
        odds_data=odds_data or {},
    )


def _build_dfs_projections(entry_df, qualifying_df, lap_averages_df,
                            practice_data, wn, track_name, series_id, dk_df,
                            race_laps, odds_data=None):
    """Build DFS-aware projections that estimate actual DK point components."""
    if odds_data is None:
        odds_data = {}

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

    # ── 1. Track History Signal ──────────────────────────────────────────────
    th_data = {}  # driver -> {avg_finish, avg_start, laps_led, races, avg_rating}
    with st.spinner("Loading track history..."):
        th_df = scrape_track_history(track_name, series_id)
    if not th_df.empty:
        for col in ["Avg Finish", "Avg Start", "Laps Led", "Races", "Avg Rating",
                     "Wins", "Top 5", "Top 10", "DNF"]:
            if col in th_df.columns:
                th_df[col] = pd.to_numeric(th_df[col], errors="coerce")
        th_idx = th_df.drop_duplicates("Driver").set_index("Driver")
        for d in drivers:
            if d in th_idx.index:
                row = th_idx.loc[d]
                th_data[d] = {
                    "avg_finish": row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20,
                    "avg_start": row.get("Avg Start", 20) if pd.notna(row.get("Avg Start")) else 20,
                    "laps_led": row.get("Laps Led", 0) if pd.notna(row.get("Laps Led")) else 0,
                    "races": row.get("Races", 1) if pd.notna(row.get("Races")) and row.get("Races") > 0 else 1,
                    "avg_rating": row.get("Avg Rating", 80) if pd.notna(row.get("Avg Rating")) else 80,
                    "wins": row.get("Wins", 0) if pd.notna(row.get("Wins")) else 0,
                    "top5": row.get("Top 5", 0) if pd.notna(row.get("Top 5")) else 0,
                    "dnf": row.get("DNF", 0) if pd.notna(row.get("DNF")) else 0,
                }

    # ── 2. Track Type Signal ─────────────────────────────────────────────────
    tt_data = {}
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    same_type_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                        if tt == track_type and t != track_name]
    if same_type_tracks:
        with st.spinner(f"Loading {track_type} track type data..."):
            type_finishes = {}
            type_laps_led = {}
            for sim_track in same_type_tracks[:4]:
                sim_th = scrape_track_history(sim_track, series_id)
                if sim_th.empty:
                    continue
                for col in ["Avg Finish", "Laps Led", "Races"]:
                    if col in sim_th.columns:
                        sim_th[col] = pd.to_numeric(sim_th[col], errors="coerce")
                for _, r in sim_th.iterrows():
                    d = r.get("Driver")
                    if not d:
                        continue
                    af = r.get("Avg Finish")
                    ll = r.get("Laps Led", 0)
                    races = r.get("Races", 1)
                    if pd.notna(af):
                        type_finishes.setdefault(d, []).append(af)
                    if pd.notna(ll) and pd.notna(races) and races > 0:
                        type_laps_led.setdefault(d, []).append(ll / races)

            for d in drivers:
                if d in type_finishes:
                    tt_data[d] = {
                        "avg_finish": np.mean(type_finishes[d]),
                        "laps_led_per_race": np.mean(type_laps_led.get(d, [0])),
                    }

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
        from src.utils import fuzzy_match_name
        # Convert odds to implied probability, then rank → finish estimate
        odds_probs = {}
        for name, odds_str in odds_data.items():
            try:
                odds_val = int(str(odds_str).replace("+", ""))
                if odds_val > 0:
                    prob = 100 / (odds_val + 100)
                else:
                    prob = abs(odds_val) / (abs(odds_val) + 100)
                odds_probs[name] = prob
            except (ValueError, TypeError):
                continue

        # Rank by probability (higher prob = better expected finish)
        ranked = sorted(odds_probs.items(), key=lambda x: x[1], reverse=True)
        for rank, (name, prob) in enumerate(ranked, 1):
            # Match odds driver name to entry list driver name
            matched = fuzzy_match_name(name, drivers)
            if matched:
                # Convert rank to projected finish (spread across field)
                odds_finish[matched] = rank * (field_size / len(ranked))

    # ── PROJECT EACH DRIVER ──────────────────────────────────────────────────
    rows = []
    for d in drivers:
        th = th_data.get(d)
        tt = tt_data.get(d)
        qp = qual_pos.get(d)
        pr = prac_rank.get(d)
        od = odds_finish.get(d)

        # --- Project expected finish position ---
        finish_signals = []
        signal_weights = []

        if th:
            finish_signals.append(th["avg_finish"])
            signal_weights.append(wn["track"])
        if tt:
            finish_signals.append(tt["avg_finish"])
            signal_weights.append(wn["track_type"])
        if qp:
            # Qualifying position is a strong predictor of finish
            finish_signals.append(qp * 0.85 + field_size * 0.15)  # regress slightly
            signal_weights.append(wn["qual"])
        if pr:
            # Practice rank as a weak finish predictor
            finish_signals.append(pr * 0.7 + field_size * 0.3)
            signal_weights.append(wn["practice"] * 0.5)
        if od and wn.get("odds", 0) > 0:
            finish_signals.append(od)
            signal_weights.append(wn["odds"])

        if finish_signals:
            total_w = sum(signal_weights)
            proj_finish = sum(f * w for f, w in zip(finish_signals, signal_weights)) / total_w
        else:
            proj_finish = field_size * 0.6  # default to mid-field

        proj_finish = max(1, min(40, proj_finish))

        # --- Project place differential ---
        start_pos = qp if qp else (pr if pr else round(proj_finish))
        proj_diff = start_pos - proj_finish

        # --- Project dominator stats (laps led + fastest laps) ---
        proj_laps_led = 0.0
        proj_fastest = 0.0

        if race_laps > 0:
            # Dominator score from track history
            dom_signals = []
            dom_weights_list = []

            if th and th["races"] >= 2:
                ll_per_race = th["laps_led"] / th["races"]
                # Scale to this race's lap count (assume ~250 avg historical)
                race_scale = race_laps / 250
                dom_signals.append(ll_per_race * race_scale)
                dom_weights_list.append(wn["track"] + wn["track_type"] * 0.5)

            if tt and tt.get("laps_led_per_race", 0) > 0:
                race_scale = race_laps / 250
                dom_signals.append(tt["laps_led_per_race"] * race_scale)
                dom_weights_list.append(wn["track_type"])

            if dom_signals:
                total_dw = sum(dom_weights_list)
                proj_laps_led = sum(s * w for s, w in zip(dom_signals, dom_weights_list)) / total_dw
            elif proj_finish <= 5:
                # Top finishers who have no history still get some dominator credit
                proj_laps_led = race_laps * 0.02 * (6 - proj_finish)

            # Fastest laps correlate with laps led but also with overall speed
            if proj_laps_led > 0:
                proj_fastest = proj_laps_led * 0.6  # ~60% of laps led are also fastest
            elif proj_finish <= 10:
                proj_fastest = race_laps * 0.01 * (11 - proj_finish)

            # Cap dominator projections
            proj_laps_led = min(proj_laps_led, race_laps * 0.5)
            proj_fastest = min(proj_fastest, race_laps * 0.3)

        # --- Compute projected DK points ---
        finish_pts = _expected_finish_from_avg(proj_finish)
        diff_pts = proj_diff * 1.0
        led_pts = proj_laps_led * 0.25
        fl_pts = proj_fastest * 0.45
        proj_dk = round(finish_pts + diff_pts + led_pts + fl_pts, 1)

        # --- Track Score (for display) ---
        track_score = 0
        if th:
            track_score = round(max(0, (40 - th["avg_finish"]) / 39 * 100 * 0.5 +
                                min(100, th["avg_rating"] / 1.5) * 0.3 +
                                min(30, th["laps_led"] / max(th["races"], 1) * 0.5) * 0.2), 1)

        tt_score = 0
        if tt:
            tt_score = round(max(0, (40 - tt["avg_finish"]) / 39 * 100), 1)

        rows.append({
            "Driver": d,
            "Proj DK": proj_dk,
            "Proj Finish": round(proj_finish, 1),
            "Finish Pts": round(finish_pts, 1),
            "Diff Pts": round(diff_pts, 1),
            "Led Pts": round(led_pts, 1),
            "FL Pts": round(fl_pts, 1),
            "Proj Laps Led": round(proj_laps_led),
            "Proj Fast Laps": round(proj_fastest),
            "Track": track_score,
            "Track Type": tt_score,
            "Start": start_pos,
        })

    proj = pd.DataFrame(rows)

    # Merge car number if available
    if "Car" in base_df.columns:
        proj = proj.merge(base_df[["Driver", "Car"]].drop_duplicates("Driver"),
                          on="Driver", how="left")

    # Merge salary
    if not dk_df.empty:
        proj = proj.merge(dk_df.drop_duplicates("Driver")[["Driver", "DK Salary"]],
                          on="Driver", how="left")
        proj["Value"] = np.where(
            proj["DK Salary"].notna() & (proj["DK Salary"] > 0),
            (proj["Proj DK"] / (proj["DK Salary"] / 1000)).round(2),
            np.nan
        )

    proj = proj.sort_values("Proj DK", ascending=False).reset_index(drop=True)
    proj.index = proj.index + 1
    proj.index.name = "Rank"

    # Weight info
    active = [(k, v) for k, v in wn.items() if v > 0]
    weight_str = " | ".join(f"{k.replace('_', ' ').title()} {v:.0%}" for k, v in active)
    st.caption(f"Weights: {weight_str}")

    # Display columns
    display_cols = ["Driver"]
    if "Car" in proj.columns:
        display_cols.append("Car")
    if "DK Salary" in proj.columns:
        display_cols.append("DK Salary")
    display_cols.extend(["Proj DK", "Proj Finish", "Finish Pts", "Diff Pts",
                         "Led Pts", "FL Pts", "Track", "Track Type"])
    if "Value" in proj.columns:
        display_cols.append("Value")
    avail = [c for c in display_cols if c in proj.columns]

    disp = format_display_df(proj[avail].copy())
    st.dataframe(safe_fillna(disp), use_container_width=True, hide_index=False, height=550)

    # Chart
    chart_df = proj.head(20).copy()
    if "Proj DK" in chart_df.columns:
        chart_df = chart_df.rename(columns={"Proj DK": "Proj Score"})
    fig = projection_bar(chart_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Export
    csv = proj[avail].to_csv(index=True).encode("utf-8")
    st.download_button("Export Projections CSV", csv,
                       "projections.csv", "text/csv", key="proj_export")
