"""NASCAR DFS Hub — Reusable UI Components."""

import pandas as pd
import numpy as np
import streamlit as st


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header for tab pages."""
    sub_html = f'<span style="color:#475569; font-size:0.78rem; margin-left:0.6rem;">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div style="padding:0.4rem 0;margin-bottom:0.5rem;border-bottom:1px solid #1e293b;">'
        f'<span style="color:#e2e8f0;font-size:1.15rem;font-weight:700;letter-spacing:0.3px;">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


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
                 "Sig Team", "Net Sig", "Team Adj", "Mfr Adj"]:
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


def render_practice_heatmap(lap_averages_df: pd.DataFrame, show_heatmap: bool = True,
                              series_id: int = None, track_name: str = None):
    """Render the Practice Summary heatmap table.

    Shows rankings with conditional coloring (green=best, red=worst).
    Includes computed Short Run, Long Run, and Average columns.

    If series_id and track_name are provided, the table becomes drill-down
    enabled — clicking a driver row opens the per-race history popup.
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

    # Apply heatmap styling or plain display.
    # If we have a track context (series + track), make the table drill-down
    # enabled so the user can click any driver for race-by-race history.
    drill_args = {}
    if series_id is not None and track_name:
        drill_args = dict(key=f"prac_heat_{series_id}_{track_name}",
                          series_id=series_id, track_name=track_name)

    if show_heatmap:
        styled = style_heatmap(disp, avail_rank_display, max_rank=len(disp))
        if drill_args:
            interactive_drill_down_dataframe(
                styled, **drill_args,
                width="stretch", hide_index=True, height=560,
            )
        else:
            st.dataframe(styled, width="stretch", hide_index=True, height=560)
    else:
        from src.utils import safe_fillna
        if drill_args:
            interactive_drill_down_dataframe(
                safe_fillna(disp), **drill_args,
                width="stretch", hide_index=True, height=560,
            )
        else:
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


# ============================================================
# DRIVER HISTORY POPUP (@st.dialog)
# ------------------------------------------------------------
# Click any driver name in a drill-down-enabled table to open this
# modal showing their per-race history at the current track or track type.
# ============================================================

_TRACK_TYPE_LABELS = {
    "intermediate": "Intermediate",
    "intermediate_worn": "Intermediate (incl. worn)",
    "short": "Short Tracks",
    "short_concrete": "Short (incl. concrete)",
    "superspeedway": "Superspeedways",
    "road": "Road Courses",
}


def _render_driver_history_scope(driver_name, series_id, *, track_name=None,
                                  track_type=None, season=None, all_tracks=False,
                                  show_track_col=False, show_series_col=False):
    """Render one scope's worth of a driver's history: summary metrics + the
    per-race table with the finish-position heatmap. Used by each tab of the
    driver-history dialog."""
    from src.data import query_driver_race_log

    log = query_driver_race_log(
        driver_name=driver_name, series_id=series_id,
        track_name=track_name, track_type=track_type, season=season,
        all_tracks=all_tracks,
    )
    if not log:
        st.info("No race data for this view.")
        return

    df = pd.DataFrame(log)
    finishes = pd.to_numeric(df["Finish"], errors="coerce")
    starts = pd.to_numeric(df["Start"], errors="coerce")
    laps_led = pd.to_numeric(df.get("Laps Led", 0), errors="coerce").fillna(0)
    fast_laps = pd.to_numeric(df.get("Fast Laps", 0), errors="coerce").fillna(0)

    avg_run = pd.to_numeric(df.get("Avg Run"), errors="coerce").mean() if "Avg Run" in df.columns else None
    avg_fin, avg_st = finishes.mean(), starts.mean()
    best, worst = finishes.min(), finishes.max()
    n_dnf = int(df["Status"].astype(str).str.lower().isin(
        ["accident", "engine", "crash", "dnf", "mechanical", "rear gear",
         "transmission", "suspension", "overheating", "brakes", "electrical",
         "fuel pump", "ignition", "vibration"]
    ).sum()) if "Status" in df.columns else 0

    row1 = st.columns(6)
    row1[0].metric("Avg Run Pos", f"{avg_run:.1f}" if avg_run is not None and pd.notna(avg_run) else "—")
    row1[1].metric("Avg Finish", f"{avg_fin:.1f}" if pd.notna(avg_fin) else "—")
    row1[2].metric("Avg Start", f"{avg_st:.1f}" if pd.notna(avg_st) else "—")
    row1[3].metric("Best", f"{int(best)}" if pd.notna(best) else "—")
    row1[4].metric("Worst", f"{int(worst)}" if pd.notna(worst) else "—")
    row1[5].metric("DNFs", n_dnf)

    row2 = st.columns(6)
    row2[0].metric("Wins", int((finishes == 1).sum()))
    row2[1].metric("Top 5", int((finishes <= 5).sum()))
    row2[2].metric("Top 10", int((finishes <= 10).sum()))
    row2[3].metric("Top 20", int((finishes <= 20).sum()))
    row2[4].metric("Laps Led", int(laps_led.sum()))
    row2[5].metric("Fast Laps", int(fast_laps.sum()))
    st.markdown("")

    # Per-race table. Show the Track column for any multi-track view, and the
    # Series column when results span multiple series (All Series).
    show_cols = ["Date", "Race"]
    if show_track_col:
        show_cols.append("Track")
    if show_series_col:
        show_cols.append("Series")
    show_cols.extend(["Car", "Team", "Start", "Finish", "Laps Led",
                      "Fast Laps", "Avg Run", "DK Pts", "Status"])
    show_cols = [c for c in show_cols if c in df.columns]
    disp = df[show_cols].copy()
    for c in ["Start", "Finish", "Laps Led", "Fast Laps"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").astype("Int64")
    for c in ["Avg Run", "DK Pts"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce")
    fmt_map = {}
    if "Avg Run" in disp.columns:
        fmt_map["Avg Run"] = "{:.1f}"
    if "DK Pts" in disp.columns:
        fmt_map["DK Pts"] = "{:.2f}"

    finish_col = "Finish" if "Finish" in disp.columns else None
    if finish_col:
        styled = disp.style.apply(
            lambda col: [_rank_color(v, max_rank=40) if col.name == finish_col
                         else "" for v in col], axis=0)
    else:
        styled = disp.style
    if fmt_map:
        styled = styled.format(fmt_map, na_rep="—")
    st.dataframe(styled, width="stretch", hide_index=True, height=380)


@st.dialog("Driver History", width="large")
def render_driver_history_dialog(driver_name: str, series_id: int,
                                  track_name: str = None,
                                  track_type: str = None,
                                  season: int = None):
    """Modal dialog: a driver's race history with TABS for different scopes.

    The caller passes whatever scope it knows (the current track, a track type,
    or a season). The dialog derives the rest (track_type from track_name, the
    current season) and presents tabs so the user can drill into any view for
    that driver from one place:
        [This Track] [Track Type] [YYYY Season] [Pick a Track] [All-Time]
    The first tab is whichever scope the caller considered primary.
    """
    from src.config import TRACK_TYPE_MAP
    from src.data import query_driver_tracks_raced

    # Derive missing context
    if track_name and not track_type:
        track_type = TRACK_TYPE_MAP.get(track_name)
    if season is None:
        from datetime import datetime as _dt
        _t = _dt.now()
        season = _t.year + 1 if _t.month >= 10 else _t.year

    st.markdown(
        f'<div style="margin:-0.4rem 0 0.4rem 0;">'
        f'<span style="color:#e2e8f0;font-size:1.05rem;font-weight:700;">{driver_name}</span>'
        f' &nbsp;<span style="color:#64748b;font-size:0.82rem;">— race history</span>'
        f'</div>', unsafe_allow_html=True)

    # Series selector — defaults to ALL series so a cross-series driver's full
    # history shows even when the popup is opened from a race in a series where
    # they have little/no data (e.g. a Cup regular in a one-off Truck race —
    # the original bug where Stenhouse's "Intermediate" tab was empty).
    _SERIES_OPTS = {"All Series": None, "Cup": 1, "O'Reilly": 2, "Truck": 3}
    sel = st.radio("Series", list(_SERIES_OPTS.keys()), horizontal=True,
                   key=f"drvhist_series_{driver_name}", label_visibility="collapsed")
    eff_series = _SERIES_OPTS[sel]
    show_series_col = (eff_series is None)

    # Build tab specs: (label, render-callable). Primary scope first.
    def _abbrev(tn):
        for suf in [" International Speedway", " Motor Speedway", " Superspeedway",
                    " Speedway", " Raceway", " Course"]:
            if tn and tn.endswith(suf):
                return tn[:-len(suf)]
        return tn or "Track"

    specs = []  # list of (label, kwargs_for_scope)
    if track_name:
        specs.append((_abbrev(track_name), dict(track_name=track_name)))
    if track_type:
        specs.append((_TRACK_TYPE_LABELS.get(track_type, track_type.replace("_", " ").title()),
                      dict(track_type=track_type, show_track_col=True)))
    specs.append((f"{season} Season", dict(season=season, show_track_col=True)))
    specs.append(("Pick a Track", dict(_picker=True)))
    specs.append(("All-Time", dict(_alltime=True, show_track_col=True)))

    # De-duplicate labels (e.g. picker-less edge cases) while preserving order
    seen, uniq = set(), []
    for label, kw in specs:
        if label in seen:
            continue
        seen.add(label)
        uniq.append((label, kw))

    tabs = st.tabs([label for label, _ in uniq])
    for tab, (label, kw) in zip(tabs, uniq):
        with tab:
            if kw.get("_picker"):
                tracks = query_driver_tracks_raced(driver_name, eff_series)
                if not tracks:
                    st.info("No tracks on record for this driver.")
                    continue
                pick = st.selectbox("Track", tracks,
                                    key=f"drvhist_pick_{sel}_{driver_name}")
                if pick:
                    _render_driver_history_scope(driver_name, eff_series, track_name=pick,
                                                 show_series_col=show_series_col)
            elif kw.get("_alltime"):
                # All races (min_season floor inside the query)
                _render_driver_history_scope(driver_name, eff_series,
                                             all_tracks=True, show_track_col=True,
                                             show_series_col=show_series_col)
            else:
                _render_driver_history_scope(driver_name, eff_series,
                                             show_series_col=show_series_col, **kw)


def interactive_drill_down_dataframe(df, *, key, series_id,
                                      track_name=None, track_type=None,
                                      season=None,
                                      driver_col="Driver",
                                      **dataframe_kwargs):
    """Wrapper for st.dataframe that adds click-to-view-driver-history.

    Click a row → opens a modal dialog with that driver's per-race history at
    the current track (track_name), track type (track_type), or season.
    Pass exactly one scope.

    Args:
        df:                  DataFrame OR pandas Styler to display
        key:                 unique session-state key for this table
        series_id:           1=Cup, 2=O'Reilly, 3=Truck (passed to dialog)
        track_name:          single-track scope for the dialog
        track_type:          track-type scope for the dialog
        season:              season scope for the dialog (e.g. 2026)
        driver_col:          column containing driver names (default "Driver")
        dataframe_kwargs:    forwarded to st.dataframe (height, column_config, etc.)

    Returns:
        The DataFrameSelectionState from st.dataframe (so callers can reuse it
        if they were already using selection state).
    """
    # Resolve underlying frame so we can iloc into it after a click
    if hasattr(df, "data"):              # pandas Styler
        underlying_df = df.data
    else:
        underlying_df = df

    # Locate the driver column. For MultiIndex columns (Race Data tab uses
    # ("Driver Info", "Driver")) we match on the LAST level of the tuple.
    resolved_driver_col = None
    if isinstance(underlying_df.columns, pd.MultiIndex):
        for c in underlying_df.columns:
            if isinstance(c, tuple) and c and c[-1] == driver_col:
                resolved_driver_col = c
                break
    elif driver_col in underlying_df.columns:
        resolved_driver_col = driver_col

    if resolved_driver_col is None:
        # Nothing clickable — fall back to plain dataframe
        return st.dataframe(df, **dataframe_kwargs)

    event = st.dataframe(
        df,
        selection_mode="single-row",
        on_select="rerun",
        key=key,
        **dataframe_kwargs,
    )

    # Defensive: event shape varies by streamlit version
    selection = getattr(event, "selection", None) if event is not None else None
    selected_rows = []
    if selection is not None:
        selected_rows = getattr(selection, "rows", None) or selection.get("rows", []) \
            if hasattr(selection, "get") else (selection.rows if hasattr(selection, "rows") else [])

    state_key = f"{key}__last_drilldown_idx"

    if selected_rows:
        idx = selected_rows[0]
        if 0 <= idx < len(underlying_df):
            # Only fire the dialog when a NEW row is clicked. Without this,
            # a closed dialog would re-open on every script rerun while the
            # row remains visually "selected".
            last_idx = st.session_state.get(state_key)
            if last_idx != idx:
                st.session_state[state_key] = idx
                driver = underlying_df.iloc[idx][resolved_driver_col]
                if pd.notna(driver) and str(driver).strip() and _claim_dialog_slot():
                    # _claim_dialog_slot() ensures only ONE driver-history
                    # dialog opens per script run. Streamlit derives a
                    # dialog's element id from its title, so opening a second
                    # "Driver History" dialog in the same run (st.tabs renders
                    # every tab's tables each run) raises
                    # StreamlitDuplicateElementId. The guard lets the first
                    # newly-clicked table win and silently skips the rest;
                    # their selection state is already recorded above so they
                    # won't re-fire next run.
                    render_driver_history_dialog(
                        driver_name=str(driver),
                        series_id=series_id,
                        track_name=track_name,
                        track_type=track_type,
                        season=season,
                    )

    return event


_DIALOG_CLAIM_KEY = "_drv_dialog_claimed_this_run"


def reset_driver_dialog_guard():
    """Clear the per-run driver-history dialog guard.

    Call once at the very top of the main app script (every rerun) so the
    first drill-down table that has a newly-clicked row can open the dialog.
    """
    st.session_state[_DIALOG_CLAIM_KEY] = False


def _claim_dialog_slot() -> bool:
    """Reserve the single per-run driver-history dialog slot.

    Returns True for the first caller in a given script run and False for any
    subsequent caller, so we never open two @st.dialog elements (which share
    an element id by title) in the same run → StreamlitDuplicateElementId.

    If the guard was never initialized (reset_driver_dialog_guard not wired
    up), the .get() default of False still lets the first caller through and
    blocks the rest, which is the desired behavior.
    """
    if st.session_state.get(_DIALOG_CLAIM_KEY, False):
        return False
    st.session_state[_DIALOG_CLAIM_KEY] = True
    return True
