"""Tab 7: Accuracy — Projections vs Actuals backtesting and weight optimization."""

import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import os
from datetime import datetime
from itertools import product

from src.config import (
    SERIES_OPTIONS, TRACK_TYPE_MAP, DK_FINISH_POINTS,
)
from src.data import (
    fetch_race_list, fetch_weekend_feed, fetch_lap_times,
    extract_race_results, compute_fastest_laps,
    filter_point_races, query_salaries,
)
from src.utils import calc_dk_points, safe_fillna, format_display_df

PROJ_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nascar.db")


# ── DB helpers ───────────────────────────────────────────────────────────────

def _ensure_saved_projections_table():
    """Create the saved_projections table if it doesn't exist."""
    if not os.path.exists(PROJ_DB):
        return
    conn = sqlite3.connect(PROJ_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS saved_projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER,
            race_name TEXT,
            track_name TEXT,
            series_id INTEGER,
            season INTEGER,
            driver TEXT,
            proj_dk REAL,
            proj_finish REAL,
            proj_laps_led REAL,
            proj_fast_laps REAL,
            proj_diff_pts REAL,
            qual_pos INTEGER,
            dk_salary INTEGER,
            w_odds REAL,
            w_track REAL,
            w_practice REAL,
            w_qualifying REAL,
            w_track_type REAL,
            saved_at TEXT DEFAULT (datetime('now')),
            UNIQUE(race_id, driver, series_id)
        )
    ''')
    conn.commit()
    conn.close()


def save_projections_to_db(proj_df, race_id, race_name, track_name,
                            series_id, season, weights):
    """Save current projections to DB for future accuracy comparison."""
    _ensure_saved_projections_table()
    if not os.path.exists(PROJ_DB):
        return 0

    conn = sqlite3.connect(PROJ_DB)
    count = 0
    for _, row in proj_df.iterrows():
        try:
            conn.execute('''
                INSERT OR REPLACE INTO saved_projections
                (race_id, race_name, track_name, series_id, season,
                 driver, proj_dk, proj_finish, proj_laps_led, proj_fast_laps,
                 proj_diff_pts, qual_pos, dk_salary,
                 w_odds, w_track, w_practice, w_qualifying, w_track_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                race_id, race_name, track_name, series_id, season,
                row.get("Driver", ""),
                row.get("Proj DK", 0),
                row.get("Proj Finish", 0),
                row.get("Proj Laps Led", 0),
                row.get("Proj Fast Laps", 0),
                row.get("Diff Pts", 0),
                row.get("Qual Pos") or row.get("Start"),
                row.get("DK Salary"),
                weights.get("odds", 0),
                weights.get("track", 0),
                weights.get("practice", 0),
                weights.get("qual", 0),
                weights.get("track_type", 0),
            ))
            count += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return count


def load_saved_projections(series_id=None, season=None, race_id=None):
    """Load saved projections from DB."""
    _ensure_saved_projections_table()
    if not os.path.exists(PROJ_DB):
        return pd.DataFrame()

    conn = sqlite3.connect(PROJ_DB)
    query = "SELECT * FROM saved_projections WHERE 1=1"
    params = []
    if series_id is not None:
        query += " AND series_id = ?"
        params.append(series_id)
    if season is not None:
        query += " AND season = ?"
        params.append(season)
    if race_id is not None:
        query += " AND race_id = ?"
        params.append(race_id)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_saved_race_list(series_id=None):
    """Get list of races that have saved projections."""
    _ensure_saved_projections_table()
    if not os.path.exists(PROJ_DB):
        return pd.DataFrame()

    conn = sqlite3.connect(PROJ_DB)
    query = """
        SELECT DISTINCT race_id, race_name, track_name, series_id, season,
               COUNT(driver) as driver_count,
               MAX(saved_at) as saved_at
        FROM saved_projections
    """
    params = []
    if series_id is not None:
        query += " WHERE series_id = ?"
        params.append(series_id)
    query += " GROUP BY race_id, series_id ORDER BY season DESC, race_name"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def _get_race_year(race):
    """Extract year from race date string."""
    rd = race.get("race_date", "")
    try:
        return int(rd[:4])
    except Exception:
        return 2026


def _load_actual_results(race, series_id):
    """Load actual race results for a completed race."""
    rc_id = race.get("race_id")
    yr = _get_race_year(race)
    rc_feed = fetch_weekend_feed(series_id, rc_id, yr)
    rc_laps = fetch_lap_times(series_id, rc_id, yr)
    if not rc_feed:
        return pd.DataFrame()

    results = extract_race_results(rc_feed)
    if results.empty:
        return results

    fl = compute_fastest_laps(rc_laps) if rc_laps else {}
    results["Fastest Laps"] = results["Driver"].map(lambda d: fl.get(d, 0)).astype("Int64")
    results["DK Pts"] = results.apply(
        lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                 r["Laps Led"], r["Fastest Laps"]), axis=1)
    return results


# ── Main Render ──────────────────────────────────────────────────────────────

def render(*, completed_races, series_id, selected_year, series_name="Cup"):
    """Render the Accuracy tab."""
    st.markdown("### Projection Accuracy")
    st.caption("Compare projections vs actual results to improve future weight tuning")

    mode = st.radio("Mode",
                    ["Race Comparison", "Accuracy Dashboard", "Weight Optimizer"],
                    horizontal=True, label_visibility="collapsed",
                    key="acc_mode")

    if mode == "Race Comparison":
        _render_race_comparison(completed_races, series_id, selected_year)
    elif mode == "Accuracy Dashboard":
        _render_accuracy_dashboard(series_id, selected_year, series_name)
    elif mode == "Weight Optimizer":
        _render_weight_optimizer(completed_races, series_id, selected_year, series_name)


# ── Race Comparison ──────────────────────────────────────────────────────────

def _render_race_comparison(completed_races, series_id, selected_year):
    """Compare projections vs actuals for a single race."""
    # Check for saved projections
    saved_races = load_saved_race_list(series_id)

    if saved_races.empty:
        st.info(
            "No saved projections yet. Go to the **Projections** tab and click "
            "**Save Projections** for upcoming races. After the race completes, "
            "come back here to compare projected vs actual results."
        )
        st.markdown("---")
        st.markdown("**Quick Start: Backtest a Completed Race**")
        st.caption(
            "You can also generate projections for a completed race and instantly "
            "compare them to actual results — select a completed race in the top "
            "bar, go to Projections, save them, then return here."
        )
        return

    # Race picker from saved projections
    race_labels = []
    race_map = {}
    for _, row in saved_races.iterrows():
        lbl = f"{row['season']} — {row['track_name']}: {row['race_name']} ({row['driver_count']} drivers)"
        race_labels.append(lbl)
        race_map[lbl] = row

    selected_label = st.selectbox("Select Race", race_labels,
                                   index=len(race_labels) - 1,
                                   key="acc_race_pick")
    saved_race = race_map[selected_label]
    race_id = int(saved_race["race_id"])

    # Load saved projections
    proj_df = load_saved_projections(series_id=series_id, race_id=race_id)
    if proj_df.empty:
        st.warning("No projection data found for this race.")
        return

    # Find matching completed race to load actuals
    actual_race = None
    for _, rc in completed_races:
        if rc.get("race_id") == race_id:
            actual_race = rc
            break

    if actual_race is None:
        # Try to find it by fetching the race list
        races = fetch_race_list(series_id, int(saved_race["season"]))
        point_races = filter_point_races(races) if races else []
        for rc in point_races:
            if rc.get("race_id") == race_id:
                actual_race = rc
                break

    if actual_race is None:
        st.warning("Could not find actual race results. The race may not have completed yet.")
        st.caption("Projections are saved — come back after the race finishes.")
        return

    with st.spinner("Loading actual results..."):
        actuals = _load_actual_results(actual_race, series_id)

    if actuals.empty:
        st.warning("Race results not available yet.")
        return

    # Merge projections with actuals
    merged = proj_df.merge(
        actuals[["Driver", "Finish Position", "Start", "Laps Led",
                 "Fastest Laps", "DK Pts"]],
        left_on="driver", right_on="Driver", how="inner"
    )

    if merged.empty:
        st.warning("Could not match projected drivers to actual results.")
        return

    # Build comparison table
    comp = pd.DataFrame({
        "Driver": merged["Driver"],
        "Proj DK": merged["proj_dk"].round(1),
        "Actual DK": merged["DK Pts"].round(1),
        "DK Error": (merged["proj_dk"] - merged["DK Pts"]).round(1),
        "Proj Finish": merged["proj_finish"].round(1),
        "Actual Finish": merged["Finish Position"],
        "Finish Error": (merged["proj_finish"] - merged["Finish Position"]).round(1),
        "Proj Laps Led": merged["proj_laps_led"].round(0).astype(int),
        "Actual Laps Led": merged["Laps Led"],
        "Proj Fast Laps": merged["proj_fast_laps"].round(0).astype(int),
        "Actual Fast Laps": merged["Fastest Laps"],
    })

    if "dk_salary" in merged.columns:
        comp["DK Salary"] = merged["dk_salary"]

    comp = comp.sort_values("Actual DK", ascending=False).reset_index(drop=True)
    comp.index = comp.index + 1
    comp.index.name = "Rank"

    # ── Accuracy metrics ──
    mae_dk = comp["DK Error"].abs().mean()
    mae_finish = comp["Finish Error"].abs().mean()
    corr_dk = comp["Proj DK"].corr(comp["Actual DK"])
    corr_finish = comp["Proj Finish"].corr(comp["Actual Finish"])

    # Rank correlation (Spearman) — did we get the ORDER right?
    rank_corr = comp["Proj DK"].rank(ascending=False).corr(
        comp["Actual DK"].rank(ascending=False))

    m_cols = st.columns(5)
    m_cols[0].metric("DK Pts MAE", f"{mae_dk:.1f}")
    m_cols[1].metric("Finish MAE", f"{mae_finish:.1f}")
    m_cols[2].metric("DK Pts Correlation", f"{corr_dk:.3f}")
    m_cols[3].metric("Finish Correlation", f"{corr_finish:.3f}")
    m_cols[4].metric("Rank Correlation", f"{rank_corr:.3f}")

    st.caption(
        "**MAE** = Mean Absolute Error (lower is better) | "
        "**Correlation** = how well projected order matches actual (1.0 = perfect) | "
        "**Rank Correlation** = Spearman rank correlation of projected vs actual DK points"
    )

    # Weights used
    w_row = proj_df.iloc[0]
    st.caption(
        f"Weights used: Odds {w_row.get('w_odds', 0):.0%} | "
        f"Track {w_row.get('w_track', 0):.0%} | "
        f"Practice {w_row.get('w_practice', 0):.0%} | "
        f"Qual {w_row.get('w_qualifying', 0):.0%} | "
        f"Track Type {w_row.get('w_track_type', 0):.0%}"
    )

    # Table
    st.dataframe(safe_fillna(format_display_df(comp)), use_container_width=True,
                 hide_index=False, height=500)

    # ── Scatter: Projected vs Actual DK Points ──
    import plotly.graph_objects as go
    from src.charts import DARK_LAYOUT

    fig = go.Figure()

    # Perfect prediction line
    min_val = min(comp["Proj DK"].min(), comp["Actual DK"].min()) - 5
    max_val = max(comp["Proj DK"].max(), comp["Actual DK"].max()) + 5
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect",
        line=dict(color="#444", dash="dash", width=1),
        showlegend=False,
    ))

    # Color by error: green = under-projected (good surprise), red = over-projected
    colors = np.where(comp["DK Error"] > 0, "#ef4444", "#22c55e")

    fig.add_trace(go.Scatter(
        x=comp["Actual DK"], y=comp["Proj DK"],
        mode="markers+text",
        text=comp["Driver"].apply(lambda d: d.split()[-1]),
        textposition="top right",
        textfont=dict(size=8, color="#8892a4"),
        marker=dict(size=9, color=colors, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Projected: %{y:.1f}<br>"
            "Actual: %{x:.1f}<br>"
            "Error: %{customdata[1]:+.1f}<extra></extra>"
        ),
        customdata=np.column_stack([comp["Driver"], comp["DK Error"]]),
        showlegend=False,
    ))

    fig.update_layout(
        **DARK_LAYOUT, height=500,
        title="Projected vs Actual DK Points",
        xaxis_title="Actual DK Points",
        yaxis_title="Projected DK Points",
    )
    st.plotly_chart(fig, use_container_width=True, key="acc_scatter_dk")

    # ── Scatter: Projected vs Actual Finish ──
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=[0, 40], y=[0, 40],
        mode="lines", name="Perfect",
        line=dict(color="#444", dash="dash", width=1),
        showlegend=False,
    ))

    colors2 = np.where(comp["Finish Error"] > 0, "#ef4444", "#22c55e")

    fig2.add_trace(go.Scatter(
        x=comp["Actual Finish"], y=comp["Proj Finish"],
        mode="markers+text",
        text=comp["Driver"].apply(lambda d: d.split()[-1]),
        textposition="top right",
        textfont=dict(size=8, color="#8892a4"),
        marker=dict(size=9, color=colors2, opacity=0.8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Projected: %{y:.1f}<br>"
            "Actual: %{x}<br>"
            "Error: %{customdata[1]:+.1f}<extra></extra>"
        ),
        customdata=np.column_stack([comp["Driver"], comp["Finish Error"]]),
        showlegend=False,
    ))

    fig2.update_layout(
        **DARK_LAYOUT, height=450,
        title="Projected vs Actual Finish Position",
        xaxis_title="Actual Finish",
        yaxis_title="Projected Finish",
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig2, use_container_width=True, key="acc_scatter_finish")

    # ── Error distribution ──
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=comp["DK Error"], nbinsx=20,
        marker_color="#4a7dfc", opacity=0.8,
        name="DK Error Distribution",
    ))
    fig3.add_vline(x=0, line_dash="dash", line_color="#888")
    fig3.update_layout(
        **DARK_LAYOUT, height=300,
        title="DK Points Error Distribution (Projected - Actual)",
        xaxis_title="Error (+ = over-projected, - = under-projected)",
        yaxis_title="Count",
    )
    st.plotly_chart(fig3, use_container_width=True, key="acc_error_dist")

    # Export
    csv = comp.to_csv(index=True).encode("utf-8")
    st.download_button("Export Comparison CSV", csv,
                       f"accuracy_{race_id}.csv", "text/csv",
                       key="acc_export")


# ── Accuracy Dashboard ───────────────────────────────────────────────────────

def _render_accuracy_dashboard(series_id, selected_year, series_name):
    """Cross-race accuracy metrics across all saved projections."""
    saved_races = load_saved_race_list(series_id)

    if saved_races.empty:
        st.info("No saved projections yet. Save projections from the Projections tab to start tracking accuracy.")
        return

    st.caption(f"**{len(saved_races)} races** with saved projections for {series_name}")

    # Load all projections and their actuals
    all_comparisons = []
    race_metrics = []

    for _, race_row in saved_races.iterrows():
        race_id = int(race_row["race_id"])
        season = int(race_row["season"])

        # Load projections
        proj_df = load_saved_projections(series_id=series_id, race_id=race_id)
        if proj_df.empty:
            continue

        # Find the actual race
        races = fetch_race_list(series_id, season)
        point_races = filter_point_races(races) if races else []
        actual_race = None
        for rc in point_races:
            if rc.get("race_id") == race_id:
                actual_race = rc
                break

        if actual_race is None:
            continue

        actuals = _load_actual_results(actual_race, series_id)
        if actuals.empty:
            continue

        # Merge
        merged = proj_df.merge(
            actuals[["Driver", "Finish Position", "Start", "Laps Led",
                     "Fastest Laps", "DK Pts"]],
            left_on="driver", right_on="Driver", how="inner"
        )

        if merged.empty:
            continue

        dk_errors = (merged["proj_dk"] - merged["DK Pts"])
        finish_errors = (merged["proj_finish"] - merged["Finish Position"])

        race_metrics.append({
            "Race": race_row["race_name"],
            "Track": race_row["track_name"],
            "Season": season,
            "Track Type": TRACK_TYPE_MAP.get(race_row["track_name"], "intermediate"),
            "Drivers": len(merged),
            "DK MAE": dk_errors.abs().mean(),
            "Finish MAE": finish_errors.abs().mean(),
            "DK Corr": merged["proj_dk"].corr(merged["DK Pts"]),
            "Finish Corr": merged["proj_finish"].corr(merged["Finish Position"]),
            "Rank Corr": merged["proj_dk"].rank(ascending=False).corr(
                merged["DK Pts"].rank(ascending=False)),
            "Avg Bias": dk_errors.mean(),  # positive = systematically over-projecting
        })

        for _, row in merged.iterrows():
            all_comparisons.append({
                "Race": race_row["race_name"],
                "Track": race_row["track_name"],
                "Track Type": TRACK_TYPE_MAP.get(race_row["track_name"], "intermediate"),
                "Driver": row["Driver"],
                "Proj DK": row["proj_dk"],
                "Actual DK": row["DK Pts"],
                "Error": row["proj_dk"] - row["DK Pts"],
            })

    if not race_metrics:
        st.info("No completed races with saved projections found.")
        return

    metrics_df = pd.DataFrame(race_metrics)
    all_comp_df = pd.DataFrame(all_comparisons)

    # ── Overall metrics ──
    overall_mae = all_comp_df["Error"].abs().mean()
    overall_corr = all_comp_df["Proj DK"].corr(all_comp_df["Actual DK"])
    overall_rank_corr = metrics_df["Rank Corr"].mean()
    overall_bias = all_comp_df["Error"].mean()

    m_cols = st.columns(4)
    m_cols[0].metric("Overall DK MAE", f"{overall_mae:.1f}")
    m_cols[1].metric("Overall DK Correlation", f"{overall_corr:.3f}")
    m_cols[2].metric("Avg Rank Correlation", f"{overall_rank_corr:.3f}")
    m_cols[3].metric("Avg Bias", f"{overall_bias:+.1f}",
                     help="Positive = over-projecting on average, Negative = under-projecting")

    # ── Per-race metrics table ──
    st.markdown("**Race-by-Race Accuracy**")
    race_disp = metrics_df.copy()
    for col in ["DK MAE", "Finish MAE", "DK Corr", "Finish Corr", "Rank Corr", "Avg Bias"]:
        race_disp[col] = race_disp[col].round(2)
    st.dataframe(safe_fillna(race_disp), use_container_width=True, hide_index=True, height=300)

    # ── By track type breakdown ──
    st.markdown("**Accuracy by Track Type**")
    type_agg = all_comp_df.groupby("Track Type").agg(
        Races=("Race", lambda x: x.nunique()),
        Drivers=("Driver", "count"),
        MAE=("Error", lambda x: x.abs().mean()),
        Bias=("Error", "mean"),
    ).round(2).sort_values("MAE")
    st.dataframe(type_agg, use_container_width=True)

    st.caption(
        "Track your projection accuracy over time. Lower MAE and higher correlation = better model. "
        "Positive bias means you're over-projecting on average."
    )


# ── Weight Optimizer ─────────────────────────────────────────────────────────

def _render_weight_optimizer(completed_races, series_id, selected_year, series_name):
    """Find optimal weights by backtesting against completed races with actual results."""
    st.markdown("**Backtest Weight Combinations**")
    st.caption(
        "Runs the projection model with different weight combinations against "
        "completed races to find which weights produce the lowest error. "
        "Uses actual race results from the database."
    )

    if not completed_races:
        st.info("No completed races available for backtesting.")
        return

    # Let user pick how many races to backtest
    max_races = min(len(completed_races), 20)
    n_races = st.slider("Races to backtest", 1, max_races,
                         min(5, max_races), key="acc_n_races")

    # Track type filter
    type_opts = ["All Types"] + sorted(set(TRACK_TYPE_MAP.values()))
    track_type_filter = st.selectbox("Track Type Filter", type_opts,
                                      key="acc_tt_filter")

    # Filter races by track type
    test_races = list(completed_races)
    if track_type_filter != "All Types":
        test_races = [
            (i, r) for i, r in test_races
            if TRACK_TYPE_MAP.get(r.get("track_name", ""), "intermediate") == track_type_filter
        ]

    test_races = test_races[-n_races:]  # most recent N

    if not test_races:
        st.info("No completed races match the selected filters.")
        return

    race_names = [f"{r.get('track_name', '')}: {r.get('race_name', '')}" for _, r in test_races]
    st.caption(f"Testing against: {', '.join(race_names)}")

    if st.button("Run Weight Optimization", type="primary", key="acc_run_opt"):
        _run_backtest(test_races, series_id, selected_year)


def _run_backtest(test_races, series_id, selected_year):
    """Run backtest across weight combinations."""
    from src.data import scrape_track_history
    from src.utils import fuzzy_match_name

    # Weight grid — test combinations in steps of 10%
    # Each weight: 0, 10, 20, 30, 40, 50
    weight_options = [0, 10, 20, 30, 40, 50]
    weight_combos = []
    for odds in weight_options:
        for track in weight_options:
            for qual in weight_options:
                for prac in weight_options:
                    # Track type gets whatever is left to make it sum to 100
                    tt = 100 - odds - track - qual - prac
                    if 0 <= tt <= 50:
                        weight_combos.append({
                            "odds": odds, "track": track,
                            "qual": qual, "practice": prac,
                            "track_type": tt,
                        })

    st.caption(f"Testing {len(weight_combos)} weight combinations across {len(test_races)} races...")
    progress = st.progress(0)

    # Pre-load all race data
    race_data = []
    for idx, (_, race) in enumerate(test_races):
        progress.progress(idx / (len(test_races) + 1),
                          text=f"Loading race {idx + 1}/{len(test_races)}...")
        race_id = race.get("race_id")
        track_name = race.get("track_name", "")
        yr = race.get("race_date", "")[:4]
        try:
            yr = int(yr)
        except Exception:
            yr = selected_year

        feed = fetch_weekend_feed(series_id, race_id, yr)
        laps = fetch_lap_times(series_id, race_id, yr)
        if not feed:
            continue

        results = extract_race_results(feed)
        if results.empty:
            continue

        fl = compute_fastest_laps(laps) if laps else {}
        results["Fastest Laps"] = results["Driver"].map(lambda d: fl.get(d, 0))
        results["DK Pts"] = results.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                     r["Laps Led"], r["Fastest Laps"]), axis=1)

        # Load track history for this track
        th_df = scrape_track_history(track_name, series_id)
        th_data = {}
        if not th_df.empty:
            for col in ["Avg Finish", "Avg Rating", "Laps Led", "Races"]:
                if col in th_df.columns:
                    th_df[col] = pd.to_numeric(th_df[col], errors="coerce")
            th_idx = th_df.drop_duplicates("Driver").set_index("Driver")
            for d in results["Driver"].unique():
                if d in th_idx.index:
                    row = th_idx.loc[d]
                    af = row.get("Avg Finish", 20) if pd.notna(row.get("Avg Finish")) else 20
                    th_data[d] = af

        race_data.append({
            "race": race,
            "results": results,
            "th_data": th_data,
            "drivers": results["Driver"].unique().tolist(),
        })

    if not race_data:
        st.warning("Could not load any race results for backtesting.")
        return

    # Test each weight combination
    combo_results = []
    total_combos = len(weight_combos)

    for c_idx, combo in enumerate(weight_combos):
        if c_idx % 50 == 0:
            progress.progress(
                (len(test_races) + c_idx / total_combos) / (len(test_races) + 1),
                text=f"Testing weight combo {c_idx + 1}/{total_combos}..."
            )

        # Normalize weights
        total_w = sum(combo.values())
        if total_w <= 0:
            continue
        wn = {k: v / total_w for k, v in combo.items()}

        all_errors = []
        all_rank_corrs = []

        for rd in race_data:
            results = rd["results"]
            th_data = rd["th_data"]
            drivers = rd["drivers"]
            field_size = len(drivers)

            # Simple projection using available signals
            driver_scores = {}
            for d in drivers:
                signals = []
                weights = []

                # Track history signal
                if d in th_data:
                    signals.append(th_data[d])
                    weights.append(wn["track"])

                # Qualifying signal (use actual start as proxy for qual)
                start = results[results["Driver"] == d]["Start"].values
                if len(start) > 0 and pd.notna(start[0]):
                    qual_finish = start[0] * 0.85 + field_size * 0.5 * 0.15
                    signals.append(qual_finish)
                    weights.append(wn["qual"])

                if signals and sum(weights) > 0:
                    raw_score = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
                else:
                    raw_score = field_size * 0.5

                driver_scores[d] = raw_score

            # Rank-order spreading
            sorted_d = sorted(driver_scores.items(), key=lambda x: x[1])
            proj_dk = {}
            for rank_idx, (d, _) in enumerate(sorted_d):
                t = rank_idx / max(len(sorted_d) - 1, 1)
                proj_finish = 1 + (field_size - 1) * (t ** 0.85)
                proj_finish = max(1, min(40, proj_finish))
                finish_pts = DK_FINISH_POINTS.get(round(proj_finish), 0)
                actual_start = results[results["Driver"] == d]["Start"].values
                s = actual_start[0] if len(actual_start) > 0 and pd.notna(actual_start[0]) else proj_finish
                diff_pts = (s - proj_finish) * 1.0
                proj_dk[d] = finish_pts + diff_pts

            # Compute errors
            for d in drivers:
                actual = results[results["Driver"] == d]["DK Pts"].values
                if len(actual) > 0 and d in proj_dk:
                    all_errors.append(abs(proj_dk[d] - actual[0]))

            # Rank correlation
            proj_series = pd.Series({d: proj_dk.get(d, 0) for d in drivers})
            actual_series = pd.Series({d: results[results["Driver"] == d]["DK Pts"].values[0]
                                       for d in drivers
                                       if len(results[results["Driver"] == d]["DK Pts"].values) > 0})
            common = proj_series.index.intersection(actual_series.index)
            if len(common) > 5:
                rc = proj_series[common].rank().corr(actual_series[common].rank())
                if pd.notna(rc):
                    all_rank_corrs.append(rc)

        if all_errors:
            combo_results.append({
                "Odds": combo["odds"],
                "Track": combo["track"],
                "Practice": combo["practice"],
                "Qualifying": combo["qual"],
                "Track Type": combo["track_type"],
                "MAE": np.mean(all_errors),
                "Rank Corr": np.mean(all_rank_corrs) if all_rank_corrs else 0,
            })

    progress.progress(1.0, text="Complete!")

    if not combo_results:
        st.warning("No valid results from backtesting.")
        return

    results_df = pd.DataFrame(combo_results)

    # Sort by best composite score (low MAE + high rank corr)
    results_df["Score"] = (
        -results_df["MAE"] / results_df["MAE"].max() * 0.6 +
        results_df["Rank Corr"] / max(results_df["Rank Corr"].max(), 0.001) * 0.4
    )
    results_df = results_df.sort_values("Score", ascending=False)

    # Show top 10
    st.markdown("**Top 10 Weight Combinations**")
    top10 = results_df.head(10).copy()
    top10["MAE"] = top10["MAE"].round(1)
    top10["Rank Corr"] = top10["Rank Corr"].round(3)
    top10 = top10.drop(columns=["Score"])
    top10.index = range(1, len(top10) + 1)
    top10.index.name = "Rank"

    st.dataframe(top10, use_container_width=True, hide_index=False)

    # Highlight best
    best = results_df.iloc[0]
    st.success(
        f"Best weights: Odds **{int(best['Odds'])}%** | "
        f"Track **{int(best['Track'])}%** | "
        f"Practice **{int(best['Practice'])}%** | "
        f"Qualifying **{int(best['Qualifying'])}%** | "
        f"Track Type **{int(best['Track Type'])}%** | "
        f"MAE: {best['MAE']:.1f} | Rank Corr: {best['Rank Corr']:.3f}"
    )

    # Show current vs optimal comparison
    current_weights = {
        "Odds": st.session_state.get("pw_odds", 25),
        "Track": st.session_state.get("pw_track", 20),
        "Practice": st.session_state.get("pw_prac", 20),
        "Qualifying": st.session_state.get("pw_qual", 15),
        "Track Type": st.session_state.get("pw_type", 10),
    }

    current_match = results_df[
        (results_df["Odds"] == current_weights["Odds"]) &
        (results_df["Track"] == current_weights["Track"]) &
        (results_df["Practice"] == current_weights["Practice"]) &
        (results_df["Qualifying"] == current_weights["Qualifying"]) &
        (results_df["Track Type"] == current_weights["Track Type"])
    ]

    if not current_match.empty:
        curr = current_match.iloc[0]
        rank_of_current = results_df.index.get_loc(current_match.index[0]) + 1
        st.info(
            f"Your current weights rank **#{rank_of_current}** out of "
            f"{len(results_df)} tested — MAE: {curr['MAE']:.1f}, "
            f"Rank Corr: {curr['Rank Corr']:.3f}"
        )

    # Export
    csv = results_df.drop(columns=["Score"]).to_csv(index=False).encode("utf-8")
    st.download_button("Export All Results CSV", csv,
                       "weight_optimization.csv", "text/csv",
                       key="acc_opt_export")
