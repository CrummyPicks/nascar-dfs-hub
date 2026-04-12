"""Tab 2: Practice Deep-Dive (Heatmap + Lap Chart)."""

import numpy as np
import pandas as pd
import streamlit as st

from src.components import render_practice_heatmap
from src.charts import practice_lap_chart, practice_bar_chart
from src.data import extract_practice_laps, extract_practice_lap_counts
from src.utils import format_display_df, safe_fillna


def render(*, lap_averages_df, feed, race_name, series_id, race_id, selected_year):
    """Render the Practice tab."""
    st.markdown(f"### Practice — {race_name}")

    if lap_averages_df.empty:
        st.info("Practice data not yet available for this race.")
        return

    # Merge practice lap counts from weekend-feed into lap_averages_df
    if feed:
        lap_counts = extract_practice_lap_counts(feed)
        if lap_counts and "Laps" not in lap_averages_df.columns:
            from src.utils import fuzzy_match_name
            lap_averages_df = lap_averages_df.copy()
            lap_averages_df["Laps"] = lap_averages_df["Driver"].apply(
                lambda d: lap_counts.get(d) or lap_counts.get(
                    fuzzy_match_name(d, list(lap_counts.keys())) or "", None))

    st.caption(f"{len(lap_averages_df)} drivers  •  Source: NASCAR API lap-averages")

    # Check if practice lap-by-lap data exists before offering Lap Chart option
    practice_laps = extract_practice_laps(feed) if feed else []
    display_options = ["Rankings (Heatmap)", "Lap Times"]
    if practice_laps:
        display_options.append("Lap Chart")

    prac_mode = st.radio("Display", display_options,
                         horizontal=True, label_visibility="collapsed", key="prac_display")

    if prac_mode == "Rankings (Heatmap)":
        show_heatmap = st.checkbox("Show heatmap colors", value=True, key="heatmap_toggle")
        render_practice_heatmap(lap_averages_df, show_heatmap=show_heatmap)

    elif prac_mode == "Lap Times":
        time_cols = ["Driver", "Car", "Laps", "Overall Avg", "Best Lap",
                     "5 Lap", "10 Lap", "15 Lap", "20 Lap", "25 Lap", "30 Lap"]
        avail = [c for c in time_cols if c in lap_averages_df.columns]
        disp = lap_averages_df[avail].copy()
        disp = format_display_df(disp)
        st.dataframe(safe_fillna(disp), width="stretch", hide_index=True, height=560)

    elif prac_mode == "Lap Chart":
        _render_lap_chart_with_data(practice_laps, lap_averages_df)

    # Bar chart (always show below for non-chart views)
    if not lap_averages_df.empty and prac_mode != "Lap Chart":
        # Interval selector for bar chart
        interval_options = ["Overall Avg"]
        for col in ["Best Lap", "5 Lap", "10 Lap", "15 Lap", "20 Lap", "25 Lap", "30 Lap"]:
            if col in lap_averages_df.columns and lap_averages_df[col].notna().sum() > 3:
                interval_options.append(col)

        bar_cols = st.columns([2, 4])
        with bar_cols[0]:
            bar_interval = st.selectbox("Lap interval", interval_options,
                                        key="prac_bar_interval", label_visibility="collapsed")

        fig = practice_bar_chart(lap_averages_df, metric_col=bar_interval)
        if fig:
            st.plotly_chart(fig, width="stretch")

    # Export
    csv = lap_averages_df.to_csv(index=False).encode("utf-8")
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
