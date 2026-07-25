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


def _render_composite(active_df, series_id, track_name):
    """Composite practice grade — Peak / Consistency / Fade / Shape /
    Confidence blended from NASCAR's consecutive-lap-average windows."""
    from src.utils import compute_practice_composite
    from src.components import interactive_drill_down_dataframe

    st.caption(
        "**Composite** blends five reads from NASCAR's best-N-consecutive-lap "
        "averages (no lap-by-lap data exists): **Peak** (top-end speed), "
        "**Consistency** (repeatable short run), **Fade** (tire falloff over a "
        "long run), **Shape** (sustained race pace), **Confidence** (how much "
        "usable run). 0–100, higher = better. Fade/Shape need a long run — "
        "neutral 50 when a driver only turned short bursts."
    )
    with st.expander("Composite weights", expanded=False):
        wc = st.columns(5)
        weights = {
            "peak": wc[0].slider("Peak", 0.0, 1.0, 0.30, 0.02, key="pc_peak"),
            "consistency": wc[1].slider("Consistency", 0.0, 1.0, 0.22, 0.02, key="pc_cons"),
            "fade": wc[2].slider("Fade", 0.0, 1.0, 0.18, 0.02, key="pc_fade"),
            "shape": wc[3].slider("Shape", 0.0, 1.0, 0.18, 0.02, key="pc_shape"),
            "confidence": wc[4].slider("Confidence", 0.0, 1.0, 0.12, 0.02, key="pc_conf"),
        }
    comp = compute_practice_composite(active_df, weights=weights)
    if comp.empty:
        st.info("Not enough practice data to grade.")
        return

    # Heatmap the sub-scores green(best)→red so strengths/weaknesses pop
    score_cols = ["Composite", "Peak", "Consistency", "Fade", "Shape", "Confidence"]

    def _color(col):
        if col.name not in score_cols:
            return ["" for _ in col]
        out = []
        for v in col:
            try:
                t = max(0.0, min(1.0, float(v) / 100.0))
            except (TypeError, ValueError):
                out.append(""); continue
            r = int(239 + (34 - 239) * t); g = int(68 + (197 - 68) * t)
            b = int(68 + (94 - 68) * t)
            out.append(f"background-color: rgba({r},{g},{b},0.18)")
        return out

    styled = comp.style.apply(_color).format(
        {c: "{:.1f}" for c in score_cols}, na_rep="—")
    if track_name:
        st.caption("Click any driver row for race-by-race history at this track")
        interactive_drill_down_dataframe(
            styled, key=f"prac_composite_{series_id}_{track_name}",
            series_id=series_id, track_name=track_name,
            width="stretch", hide_index=True, height=600)
    else:
        st.dataframe(styled, width="stretch", hide_index=True, height=600)


def render(*, lap_averages_df, feed, race_name, series_id, race_id, selected_year,
           track_name=None, dk_df=None, odds_data=None):
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

    # Per-driver GROUP label — only when NASCAR actually splits practice into
    # separate group sessions (each driver then appears in exactly one block).
    # NASCAR's public API does NOT publish group assignments for a single
    # combined practice (verified across every weekend-feed field), so there
    # is nothing to show for those — and everyone ran in the same conditions
    # anyway, so the comparison is already apples-to-apples.
    from src.utils import normalize_driver_name
    group_of = {}
    if len(all_sessions) > 1:
        for label, sdf in all_sessions:
            if "Driver" in sdf.columns:
                for d in sdf["Driver"].dropna():
                    group_of[normalize_driver_name(str(d))] = label
    if group_of:
        lap_averages_df = lap_averages_df.copy()
        lap_averages_df["Group"] = lap_averages_df["Driver"].map(
            lambda d: group_of.get(normalize_driver_name(str(d)), ""))

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

    # Attach DK salary + odds so pace reads next to price on every display.
    # Odds values are NOT always strings — the auto-fetch and salary-estimate
    # paths yield numbers, and a mixed str/float object column makes pyarrow
    # throw ArrowTypeError (a TypeError) when st.dataframe serializes it,
    # which took the cloud app down. Normalize hard: DK Sal numeric, Odds
    # string-or-None.
    def _odds_str(v):
        if v is None:
            return None
        if isinstance(v, dict):
            v = v.get("odds") or v.get("american") or v.get("value")
            if v is None:
                return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, (int, float)):
            n = int(round(v))
            return f"+{n}" if n > 0 else str(n)
        return str(v)

    def _attach_dfs_context(pdf):
        try:
            if pdf is None or pdf.empty or "Driver" not in pdf.columns:
                return pdf
            from src.utils import fuzzy_match_name
            pdf = pdf.copy()
            if dk_df is not None and not dk_df.empty and "Driver" in dk_df.columns:
                sal_col = next((c for c in dk_df.columns if "Salary" in c), None)
                if sal_col:
                    smap = dict(zip(dk_df["Driver"], dk_df[sal_col]))
                    pdf["DK Sal"] = pd.to_numeric(pdf["Driver"].map(
                        lambda d: smap.get(d) if d in smap else smap.get(
                            fuzzy_match_name(str(d), list(smap)) or "")),
                        errors="coerce")
            if odds_data:
                pdf["Odds"] = [
                    _odds_str(odds_data.get(d) if d in odds_data else odds_data.get(
                        fuzzy_match_name(str(d), list(odds_data)) or ""))
                    for d in pdf["Driver"]]
            return pdf
        except Exception:
            # Context columns are nice-to-have — never let them kill the page
            return pdf

    active_df = _attach_dfs_context(active_df)

    st.caption(f"{len(active_df)} drivers  •  Source: NASCAR API lap-averages"
               + (f"  •  Session: {selected_session}" if len(all_sessions) > 1 and selected_session != "All (Combined)" else ""))

    # Check if practice lap-by-lap data exists before offering Lap Chart option
    practice_laps = extract_practice_laps(feed) if feed else []

    # Live-captured lap-by-lap (scripts/capture_practice_laps.py). NASCAR
    # never archives per-lap practice times, so for most weekends this is
    # the ONLY lap-by-lap source — feeds both Lap Chart and Best X Laps.
    from src.data import fetch_captured_practice_laps, best_lap_averages_from_captured
    cap_df = fetch_captured_practice_laps(race_id)
    if not practice_laps and not cap_df.empty:
        practice_laps = [
            {"driver": d, "laps": [{"lap_num": int(r.lap_number),
                                    "lap_time": float(r.lap_time)}
                                   for r in g.itertuples()
                                   if r.lap_time and r.lap_time > 0]}
            for d, g in cap_df.groupby("driver")]
        practice_laps = [e for e in practice_laps if e["laps"]]
    best_x_df = _attach_dfs_context(best_lap_averages_from_captured(cap_df))

    display_options = ["Rankings (Heatmap)", "Composite Score", "Lap Times"]
    if not best_x_df.empty:
        display_options.append("Best X Laps")
    if practice_laps:
        display_options.append("Lap Chart")

    prac_mode = st.radio("Display", display_options,
                         horizontal=True, label_visibility="collapsed", key="prac_display")

    if prac_mode == "Composite Score":
        _render_composite(active_df, series_id, track_name)

    elif prac_mode == "Rankings (Heatmap)":
        show_heatmap = st.checkbox("Show heatmap colors", value=True, key="heatmap_toggle")
        if track_name:
            st.caption("Click any driver row for race-by-race history at this track")
        render_practice_heatmap(
            active_df, show_heatmap=show_heatmap,
            series_id=series_id, track_name=track_name,
        )

    elif prac_mode == "Lap Times":
        time_cols = ["Driver", "Car", "DK Sal", "Odds", "Laps", "Overall Avg",
                     "Best Lap", "5 Lap", "10 Lap", "15 Lap", "20 Lap",
                     "25 Lap", "30 Lap"]
        # Only columns with at least one value — an all-empty window (nobody
        # ran 20+ consecutive laps) is noise
        avail = [c for c in time_cols if c in active_df.columns
                 and active_df[c].notna().any()]
        disp = active_df[avail].copy()
        if "DK Sal" in disp.columns:
            disp["DK Sal"] = pd.to_numeric(disp["DK Sal"], errors="coerce").map(
                lambda v: f"${v:,.0f}" if pd.notna(v) else None)
        disp = format_display_df(disp)
        from src.components import (apply_car_badges, CAR_BADGE_ROW_HEIGHT,
                                    practice_context_col_config)
        disp, _badge_cfg = apply_car_badges(disp, series_id)
        _col_cfg = practice_context_col_config(disp)
        if _badge_cfg:
            _col_cfg["Car"] = _badge_cfg
        _badge_kw = {"column_config": _col_cfg}
        if _badge_cfg:
            _badge_kw["row_height"] = CAR_BADGE_ROW_HEIGHT
        if track_name:
            from src.components import interactive_drill_down_dataframe
            st.caption("Click any driver row for race-by-race history at this track")
            interactive_drill_down_dataframe(
                safe_fillna(disp),
                key=f"prac_laps_{series_id}_{race_id}",
                series_id=series_id, track_name=track_name,
                width="stretch", hide_index=True, height=560,
                **_badge_kw,
            )
        else:
            st.dataframe(safe_fillna(disp), width="stretch", hide_index=True,
                         height=560, **_badge_kw)

    elif prac_mode == "Best X Laps":
        # Captured laps can span several sessions (practice AND qualifying)
        # — let the user scope the view, and always SAY what's included so
        # single-lap qualifying data doesn't read as broken practice data.
        cap_sessions = sorted(cap_df["run_name"].dropna().unique().tolist())
        if len(cap_sessions) > 1:
            sess_pick = st.selectbox("Session", ["All Sessions"] + cap_sessions,
                                     key="bestx_session")
            if sess_pick != "All Sessions":
                best_x_df = _attach_dfs_context(best_lap_averages_from_captured(
                    cap_df[cap_df["run_name"] == sess_pick]))
            sess_label = sess_pick
        else:
            sess_label = cap_sessions[0] if cap_sessions else "unknown session"
        n_cap = int(pd.to_numeric(best_x_df["Laps"], errors="coerce").sum())
        st.caption(
            f"**Best X laps overall** (not consecutive) — {n_cap} "
            f"live-captured laps from **{sess_label}**. A driver whose laps "
            "came in short bursts never posts a 10-lap *consecutive* average "
            "(NASCAR's windows), but their best 10 laps overall still show "
            "pace. Lap Avg = clean-lap average (within 115% of driver "
            "median). Single-car qualifying shows 1 lap per driver — that's "
            "the session, not missing data.")
        render_practice_heatmap(best_x_df, show_heatmap=True,
                                series_id=series_id, track_name=track_name)
        with st.expander("Best-X lap times (seconds)"):
            tcols = [c for c in ["Driver", "DK Sal", "Odds", "Laps",
                                 "Overall Avg", "Best Lap", "5 Lap", "10 Lap",
                                 "15 Lap", "20 Lap", "25 Lap", "30 Lap"]
                     if c in best_x_df.columns and best_x_df[c].notna().any()]
            tdisp = best_x_df[tcols].copy()
            if "DK Sal" in tdisp.columns:
                tdisp["DK Sal"] = pd.to_numeric(tdisp["DK Sal"], errors="coerce").map(
                    lambda v: f"${v:,.0f}" if pd.notna(v) else None)
            from src.components import practice_context_col_config
            st.dataframe(safe_fillna(format_display_df(tdisp)),
                         width="stretch", hide_index=True, height=560,
                         column_config=practice_context_col_config(tdisp))

    elif prac_mode == "Lap Chart":
        _render_lap_chart_with_data(practice_laps, active_df)

    # Bar chart (always show below for non-chart views)
    if not active_df.empty and prac_mode != "Lap Chart":
        # In Best X mode, chart the captured best-X averages, not the
        # consecutive-window frame
        chart_src = best_x_df if prac_mode == "Best X Laps" else active_df
        # Interval selector for bar chart
        interval_options = ["Overall Avg"]
        for col in ["Best Lap", "5 Lap", "10 Lap", "15 Lap", "20 Lap", "25 Lap", "30 Lap"]:
            if col in chart_src.columns and chart_src[col].notna().sum() > 3:
                interval_options.append(col)

        bar_cols = st.columns([2, 4])
        with bar_cols[0]:
            bar_interval = st.selectbox("Lap interval", interval_options,
                                        key="prac_bar_interval", label_visibility="collapsed")

        fig = practice_bar_chart(chart_src, metric_col=bar_interval)
        if fig:
            st.plotly_chart(fig, width="stretch")

    # Export
    csv = active_df.to_csv(index=False).encode("utf-8")
    st.download_button("Export Practice CSV", csv,
                       f"{race_name.replace(' ', '_')}_practice.csv", "text/csv")

    # ── Live lap-by-lap capture (PC only — writes to local nascar.db) ──────
    import os as _os
    if not (_os.path.exists("/mount/src") or _os.name != "nt"):
        with st.expander("Live lap-by-lap capture", expanded=False):
            st.markdown(
                "NASCAR only publishes consecutive-lap averages after a "
                "session — per-lap times exist **only while practice is "
                "live**. Start the recorder before or during a session and "
                "leave it running; it polls every 3s, stores each lap, and "
                "stops itself ~20 min after the session ends. Captured laps "
                "unlock the **Best X Laps** and **Lap Chart** displays "
                "(and sync to the cloud app with the DB).")
            if not cap_df.empty:
                st.caption(f"{len(cap_df)} laps / "
                           f"{cap_df['vehicle_number'].nunique()} cars "
                           f"captured for this race so far.")
            if st.button("Start recorder (background)", key="start_lap_capture"):
                import subprocess, sys as _sys
                root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                script = _os.path.join(root, "scripts", "capture_practice_laps.py")
                _os.makedirs(_os.path.join(root, "cache"), exist_ok=True)
                log = open(_os.path.join(root, "cache", f"capture_{race_id}.log"), "a")
                # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — keeps running
                # if the app is closed
                subprocess.Popen(
                    [_sys.executable, script, "--series-id", str(series_id),
                     "--race-id", str(race_id), "--year", str(selected_year)],
                    stdout=log, stderr=subprocess.STDOUT, cwd=root,
                    creationflags=0x00000208)
                st.success("Recorder started in the background. Revisit this "
                           "tab during practice to watch laps come in "
                           f"(log: cache/capture_{race_id}.log).")


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
