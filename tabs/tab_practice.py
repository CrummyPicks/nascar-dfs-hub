"""Tab 2: Practice Deep-Dive (Heatmap + Lap Chart)."""

import numpy as np
import pandas as pd
import streamlit as st

from src.components import render_practice_heatmap, section_header
from src.charts import practice_lap_chart, practice_bar_chart
from src.data import (
    extract_practice_laps, extract_practice_lap_counts,
    fetch_all_practice_sessions,
)
from src.utils import format_display_df, safe_fillna


def render(*, lap_averages_df, feed, race_name, series_id, race_id, selected_year,
           track_name=None):
    """Render the Practice tab."""
    section_header("Practice", race_name)

    if lap_averages_df.empty:
        st.info("Practice data not yet available for this race.")
        return

    # NASCAR's lap-averages feed sometimes misspells names vs the entry list
    # (e.g. "Carson Kvapili" vs "Carson Kvapil"). Remap practice driver names
    # to the entry-list spelling so the heatmap reads right AND the drill-down
    # dialog resolves their history. Mirrors the projection-path remap.
    if feed:
        from src.data import extract_entry_list
        from src.utils import normalize_driver_name, fuzzy_match_name
        _entry = extract_entry_list(feed)
        _entry_drivers = (_entry["Driver"].dropna().tolist()
                          if _entry is not None and not _entry.empty else [])
        if _entry_drivers and "Driver" in lap_averages_df.columns:
            _norm_entry = {normalize_driver_name(d): d for d in _entry_drivers}

            def _canon(d):
                if d in _entry_drivers:
                    return d
                nk = normalize_driver_name(str(d))
                if nk in _norm_entry:
                    return _norm_entry[nk]
                m = fuzzy_match_name(str(d), _entry_drivers, threshold=0.82)
                return m or d
            lap_averages_df = lap_averages_df.copy()
            lap_averages_df["Driver"] = lap_averages_df["Driver"].map(_canon)

    # Fetch all practice sessions for group filtering
    all_sessions = fetch_all_practice_sessions(series_id, race_id, selected_year)

    # Merge practice lap counts from weekend-feed into lap_averages_df
    if feed:
        lap_counts = extract_practice_lap_counts(feed)
        if lap_counts and "Laps" not in lap_averages_df.columns:
            from src.utils import fuzzy_match_name
            lap_averages_df = lap_averages_df.copy()
            lap_averages_df["Laps"] = lap_averages_df["Driver"].apply(
                lambda d: lap_counts.get(d) or lap_counts.get(
                    fuzzy_match_name(d, list(lap_counts.keys())) or "", None))

    # Session/group filter
    active_df = lap_averages_df
    if len(all_sessions) > 1:
        session_labels = ["All (Combined)"] + [label for label, _ in all_sessions]
        selected_session = st.selectbox(
            "Practice Group", session_labels,
            key="prac_session_filter",
            help="Filter by practice group to compare drivers within the same session")
        if selected_session != "All (Combined)":
            for label, sdf in all_sessions:
                if label == selected_session:
                    active_df = sdf
                    break
            # Re-rank within the selected group
            if "Overall Rank" in active_df.columns:
                active_df = active_df.copy()
                active_df["Overall Rank"] = range(1, len(active_df) + 1)

    st.caption(f"{len(active_df)} drivers  •  Source: NASCAR API lap-averages"
               + (f"  •  Session: {selected_session}" if len(all_sessions) > 1 and selected_session != "All (Combined)" else ""))

    # Check if practice lap-by-lap data exists before offering Lap Chart option
    practice_laps = extract_practice_laps(feed) if feed else []
    display_options = ["Rankings (Heatmap)", "Lap Times"]
    if practice_laps:
        display_options.append("Lap Chart")

    prac_mode = st.radio("Display", display_options,
                         horizontal=True, label_visibility="collapsed", key="prac_display")

    if prac_mode == "Rankings (Heatmap)":
        show_heatmap = st.checkbox("Show heatmap colors", value=True, key="heatmap_toggle")
        if track_name:
            st.caption("Click any driver row for race-by-race history at this track")
        render_practice_heatmap(
            active_df, show_heatmap=show_heatmap,
            series_id=series_id, track_name=track_name,
        )

    elif prac_mode == "Lap Times":
        time_cols = ["Driver", "Car", "Laps", "Overall Avg", "Best Lap",
                     "5 Lap", "10 Lap", "15 Lap", "20 Lap", "25 Lap", "30 Lap"]
        avail = [c for c in time_cols if c in active_df.columns]
        disp = active_df[avail].copy()
        disp = format_display_df(disp)
        if track_name:
            from src.components import interactive_drill_down_dataframe
            st.caption("Click any driver row for race-by-race history at this track")
            interactive_drill_down_dataframe(
                safe_fillna(disp),
                key=f"prac_laps_{series_id}_{race_id}",
                series_id=series_id, track_name=track_name,
                width="stretch", hide_index=True, height=560,
            )
        else:
            st.dataframe(safe_fillna(disp), width="stretch", hide_index=True, height=560)

    elif prac_mode == "Lap Chart":
        _render_lap_chart_with_data(practice_laps, active_df)

    # Bar chart (always show below for non-chart views)
    if not active_df.empty and prac_mode != "Lap Chart":
        # Interval selector for bar chart
        interval_options = ["Overall Avg"]
        for col in ["Best Lap", "5 Lap", "10 Lap", "15 Lap", "20 Lap", "25 Lap", "30 Lap"]:
            if col in active_df.columns and active_df[col].notna().sum() > 3:
                interval_options.append(col)

        bar_cols = st.columns([2, 4])
        with bar_cols[0]:
            bar_interval = st.selectbox("Lap interval", interval_options,
                                        key="prac_bar_interval", label_visibility="collapsed")

        fig = practice_bar_chart(active_df, metric_col=bar_interval)
        if fig:
            st.plotly_chart(fig, width="stretch")

    # Export
    csv = active_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Practice CSV", csv,
                       f"{race_name.replace(' ', '_')}_practice.csv", "text/csv")


def _render_lap_chart_with_data(practice_laps, lap_averages_df):
    """Render practice lap chart with pre-fetched data."""
    if not practice_laps:
        st.info("Lap-by-lap practice data not available for this race.")
        return

    # --- Controls ---
    ctrl_cols = st.columns([2, 1])

    # Driver selection
    all_drivers = sorted([entry["driver"] for entry in practice_laps])

    # Default to top 10 by overall average rank if available
    if not lap_averages_df.empty and "Driver" in lap_averages_df.columns:
        top_drivers = lap_averages_df.head(10)["Driver"].tolist()
        default_drivers = [d for d in top_drivers if d in all_drivers]
    else:
        default_drivers = all_drivers[:10]

    with ctrl_cols[0]:
        selected_drivers = st.multiselect("Select drivers", all_drivers,
                                          default=default_drivers, key="lap_chart_drivers")

    # Outlier threshold
    with ctrl_cols[1]:
        outlier_threshold = st.slider("Outlier filter (x median)", 1.05, 1.40, 1.15, 0.05,
                                      key="outlier_thresh",
                                      help="Remove laps slower than this multiple of the driver's median")

    if not selected_drivers:
        st.info("Select at least one driver.")
        return

    # Filter to selected drivers and remove outliers
    filtered_laps = []
    for entry in practice_laps:
        if entry["driver"] not in selected_drivers:
            continue

        laps = entry["laps"]
        if not laps:
            continue

        # Compute median lap time for this driver
        times = [l["lap_time"] for l in laps]
        median_time = np.median(times)
        cutoff = median_time * outlier_threshold

        # Filter outliers
        clean_laps = [l for l in laps if l["lap_time"] <= cutoff]

        if clean_laps:
            filtered_laps.append({"driver": entry["driver"], "laps": clean_laps})

    if filtered_laps:
        fig = practice_lap_chart(filtered_laps)
        if fig:
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Could not build lap chart.")
    else:
        st.info("No laps remaining after filtering. Try increasing the outlier threshold.")
