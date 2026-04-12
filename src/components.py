"""NASCAR DFS Hub — Reusable UI Components."""

import pandas as pd
import numpy as np
import streamlit as st


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header for tab pages."""
    sub_html = f'<span style="color:#475569; font-size:0.78rem; margin-left:0.6rem;">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""<div style='
        padding: 0.4rem 0; margin-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
    '>
        <span style='color:#e2e8f0; font-size:1.15rem; font-weight:700; letter-spacing:0.3px;'>{title}</span>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def build_projection_column_config(df, max_proj_dk=None):
    """Build st.column_config for the projections table."""
    config = {}
    if max_proj_dk is None:
        max_proj_dk = df["Proj DK"].max() if "Proj DK" in df.columns else 100

    if "DK Salary" in df.columns:
        config["DK Salary"] = st.column_config.NumberColumn(
            "DK Salary", format="$%d")
    if "Proj DK" in df.columns:
        config["Proj DK"] = st.column_config.ProgressColumn(
            "Proj DK", min_value=0, max_value=float(max_proj_dk * 1.1),
            format="%.1f")
    if "Value" in df.columns:
        config["Value"] = st.column_config.NumberColumn("Value", format="%.2f")
    if "Proj Finish" in df.columns:
        config["Proj Finish"] = st.column_config.NumberColumn("Proj Finish", format="%.0f")
    for col in ["Win Odds", "Est. Odds"]:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(col, format="%+d")
    for col in ["Impl %", "Est. Impl %"]:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.1f%%")
    for col in ["Finish Pts", "Diff Pts", "Led Pts", "FL Pts"]:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.1f")
    for col in ["Avg DK", "Best DK", "Worst DK"]:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.1f")
    for col in ["Proj Laps Led", "Proj Fast Laps"]:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(col, format="%d")
    # Signal columns
    for col in ["Sig Odds", "Sig Track", "Sig TType", "Sig Prac", "Sig Qual",
                 "Sig Team", "Team Adj", "Mfr Adj"]:
        if col in df.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.1f")
    return config


def build_optimizer_column_config(df):
    """Build st.column_config for optimizer pool/lineup tables."""
    config = {}
    if "DK Salary" in df.columns:
        config["DK Salary"] = st.column_config.NumberColumn("Salary", format="$%d")
    if "Proj Score" in df.columns:
        config["Proj Score"] = st.column_config.NumberColumn("Proj", format="%.1f")
    if "Value" in df.columns:
        config["Value"] = st.column_config.NumberColumn("Value", format="%.2f")
    return config


def _rank_color(val, max_rank=40):
    """Return background color for a rank value (1=green, high=red)."""
    if pd.isna(val) or val == "-":
        return ""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    # Normalize: 1 = 0.0 (best), max_rank = 1.0 (worst)
    ratio = min(1.0, max(0.0, (v - 1) / max(max_rank - 1, 1)))
    if ratio <= 0.08:
        return "background-color: #1a7a3d; color: white"     # deep green
    elif ratio <= 0.18:
        return "background-color: #2d9a50; color: white"     # green
    elif ratio <= 0.30:
        return "background-color: #52b86a; color: #111"      # medium green
    elif ratio <= 0.42:
        return "background-color: #8fcf7e; color: #111"      # light green
    elif ratio <= 0.54:
        return "background-color: #e8d44d; color: #111"      # gold
    elif ratio <= 0.66:
        return "background-color: #e8a735; color: #111"      # amber
    elif ratio <= 0.78:
        return "background-color: #e07830; color: white"     # orange
    elif ratio <= 0.90:
        return "background-color: #d14a3a; color: white"     # red
    else:
        return "background-color: #9c2a2a; color: white"     # dark red


def style_heatmap(df: pd.DataFrame, rank_columns: list, max_rank: int = None) -> pd.io.formats.style.Styler:
    """Apply green-to-red heatmap coloring on rank columns.

    Args:
        df: DataFrame to style
        rank_columns: list of column names to apply heatmap to
        max_rank: maximum rank value (defaults to field size)

    Returns:
        Styled DataFrame ready for st.dataframe()
    """
    if max_rank is None:
        max_rank = len(df) if len(df) > 5 else 40

    def apply_colors(col):
        if col.name in rank_columns:
            return [_rank_color(v, max_rank) for v in col]
        return [""] * len(col)

    return df.style.apply(apply_colors)


def render_practice_heatmap(lap_averages_df: pd.DataFrame, show_heatmap: bool = True):
    """Render the Practice Summary heatmap table.

    Shows rankings with conditional coloring (green=best, red=worst).
    Includes computed Short Run, Long Run, and Average columns.
    """
    if lap_averages_df.empty:
        st.info("Practice data not yet available.")
        return

    df = lap_averages_df.copy()

    # Build rank-only view — use "R:" prefix to avoid clashing with lap time columns
    rank_cols_map = {
        "1 Lap Rank": "1 Lap",
        "5 Lap Rank": "5 Lap",
        "10 Lap Rank": "10 Lap",
        "15 Lap Rank": "15 Lap",
        "20 Lap Rank": "20 Lap",
        "25 Lap Rank": "25 Lap",
        "30 Lap Rank": "30 Lap",
    }

    display_cols = ["Driver"]
    if "Laps" in df.columns:
        df["Laps"] = pd.to_numeric(df["Laps"], errors="coerce").astype("Int64")
        display_cols.append("Laps")
    avail_rank_cols = []
    for rc, label in rank_cols_map.items():
        if rc in df.columns:
            # Use the rank column directly, renamed for display
            col_name = f"_r_{label}"
            df[col_name] = pd.to_numeric(df[rc], errors="coerce").astype("Int64")
            # We'll rename at the end for clean display
            display_cols.append(col_name)
            avail_rank_cols.append(col_name)

    if "Overall Rank" in df.columns:
        df["_r_Lap Avg"] = pd.to_numeric(df["Overall Rank"], errors="coerce").astype("Int64")
        display_cols.append("_r_Lap Avg")
        avail_rank_cols.append("_r_Lap Avg")

    # Compute Short Run = best of (1 Lap, 5 Lap, 10 Lap) ranks
    short_src = [c for c in ["_r_1 Lap", "_r_5 Lap", "_r_10 Lap"] if c in df.columns]
    if short_src:
        df["_r_Sh. Run"] = df[short_src].min(axis=1).astype("Int64")
        display_cols.append("_r_Sh. Run")
        avail_rank_cols.append("_r_Sh. Run")

    # Compute Long Run = best of (20 Lap, 25 Lap, 30 Lap) ranks
    long_src = [c for c in ["_r_20 Lap", "_r_25 Lap", "_r_30 Lap"] if c in df.columns]
    if long_src:
        df["_r_Lo. Run"] = df[long_src].min(axis=1).astype("Int64")
        display_cols.append("_r_Lo. Run")
        avail_rank_cols.append("_r_Lo. Run")

    # Compute Average = mean of all base rank columns
    base_src = [c for c in ["_r_1 Lap", "_r_5 Lap", "_r_10 Lap", "_r_15 Lap",
                             "_r_20 Lap", "_r_25 Lap", "_r_30 Lap"] if c in df.columns]
    if base_src:
        df["_r_Average"] = df[base_src].mean(axis=1).round(0).astype("Int64")
        display_cols.append("_r_Average")
        avail_rank_cols.append("_r_Average")

    # Sort by Average or Lap Avg
    sort_col = "_r_Average" if "_r_Average" in df.columns else ("_r_Lap Avg" if "_r_Lap Avg" in df.columns else None)
    if sort_col:
        df = df.sort_values(sort_col, na_position="last")

    disp = df[display_cols].copy()

    # Rename columns for clean display (strip _r_ prefix)
    clean_names = {c: c.replace("_r_", "") for c in disp.columns if c.startswith("_r_")}
    disp = disp.rename(columns=clean_names)
    avail_rank_display = [c.replace("_r_", "") for c in avail_rank_cols]

    # Apply heatmap styling or plain display
    if show_heatmap:
        styled = style_heatmap(disp, avail_rank_display, max_rank=len(disp))
        st.dataframe(styled, width="stretch", hide_index=True, height=560)
    else:
        from src.utils import safe_fillna
        st.dataframe(safe_fillna(disp), width="stretch", hide_index=True, height=560)


def render_driver_race_log(driver_name: str, race_data: list):
    """Render a driver's race-by-race log in an expander.

    Args:
        driver_name: Name to display
        race_data: list of dicts with race-by-race results
    """
    if not race_data:
        st.info(f"No race data found for {driver_name}")
        return

    df = pd.DataFrame(race_data)

    # Format display
    for col in ["Finish", "Start", "Laps Led", "Fast Laps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    st.dataframe(df.fillna("-"), width="stretch", hide_index=True)

    # Season summary metrics
    num_df = df.copy()
    for col in ["Finish", "Start", "DK Pts", "Avg Run"]:
        if col in num_df.columns:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce")

    cols = st.columns(4)
    if "Finish" in num_df.columns:
        cols[0].metric("Avg Finish", f"{num_df['Finish'].mean():.1f}")
    if "Start" in num_df.columns:
        cols[1].metric("Avg Start", f"{num_df['Start'].mean():.1f}")
    if "DK Pts" in num_df.columns:
        cols[2].metric("Avg DK Pts", f"{num_df['DK Pts'].mean():.1f}")
    cols[3].metric("Races", len(race_data))
