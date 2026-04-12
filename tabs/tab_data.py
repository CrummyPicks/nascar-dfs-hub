"""Tab 1: Consolidated Data Hub — unified view for all races."""

import pandas as pd
import streamlit as st

from src.config import SERIES_LABELS
from src.components import section_header
from src.data import (
    scrape_track_history, query_db_track_history, query_driver_dk_points_at_track,
    compute_fastest_laps, compute_avg_running_position, load_arp_from_db,
)
from src.utils import (
    calc_dk_points, calc_fd_points, safe_fillna, format_display_df,
    fuzzy_match_name, fuzzy_merge, fuzzy_get, build_norm_lookup,
)
from src.charts import (
    dfs_histogram, start_vs_finish_scatter, race_scatter, race_lap_chart,
    season_trend_line,
)


def render(*, feed, lap_data, lap_averages_df, entry_list_df, qualifying_df,
           results_df, is_prerace, series_id, race_name, track_name, track_type,
           dk_df, fd_df, completed_races, selected_year, fl_counts, odds_data=None,
           prop_odds=None, race_id=None):
    """Render the consolidated Data tab — same wide table for pre and post race."""

    series_name = SERIES_LABELS.get(series_id, "Cup")
    section_header(f"NASCAR — {series_name} Data")

    # Toggle: Table | Charts
    view_mode = st.radio("View", ["Table", "Charts"], horizontal=True,
                         label_visibility="collapsed", key="data_view_mode")

    if view_mode == "Charts":
        _render_charts_view(completed_races, series_id, selected_year, results_df,
                            lap_data, fl_counts, is_prerace)
        return

    # --- Determine base driver list ---
    # For completed races, use results as base (has finish data)
    # For upcoming races, use entry list or lap averages
    if not is_prerace and not results_df.empty:
        avg_run_pos = compute_avg_running_position(lap_data) if lap_data else {}
        # Fall back to DB-stored ARP when live lap data unavailable
        if not avg_run_pos and race_id:
            avg_run_pos = load_arp_from_db(race_id)
        res = results_df.copy()
        res = res.sort_values("Finish Position", na_position="last").reset_index(drop=True)
        fl_norm = build_norm_lookup(fl_counts)
        arp_norm = build_norm_lookup(avg_run_pos)
        res["Fastest Laps"] = res["Driver"].map(
            lambda d: fuzzy_get(d, fl_counts, fl_norm) or 0).astype("Int64")
        res["Avg Run"] = res["Driver"].map(
            lambda d: fuzzy_get(d, avg_run_pos, arp_norm))
        res["DK Pts"] = res.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"], r["Laps Led"], r["Fastest Laps"]), axis=1)
        res["FD Pts"] = res.apply(
            lambda r: calc_fd_points(r["Finish Position"], r["Start"], r["Laps Led"], r["Fastest Laps"]), axis=1)
        res["Pos Diff"] = (res["Start"] - res["Finish Position"]).astype("Int64")
        want = ["Driver", "Finish Position", "Start", "Car", "Team", "Manufacturer",
                "Crew Chief", "Laps Led", "Fastest Laps", "Avg Run",
                "DK Pts", "FD Pts", "Pos Diff", "Status"]
        master = res[[c for c in want if c in res.columns]].copy()
    elif not entry_list_df.empty:
        master = entry_list_df.copy()
    elif not lap_averages_df.empty:
        master = lap_averages_df[["Driver", "Car"]].copy()
    elif not qualifying_df.empty:
        master = qualifying_df[["Driver"]].copy()
    else:
        st.info("Race weekend data not yet available. Check back closer to race day.")
        return

    # --- Merge additional data sources (dedup right side to prevent row multiplication) ---

    # DK/FD Salary
    if not dk_df.empty:
        master = fuzzy_merge(master, dk_df, on="Driver", how="left")
    if not fd_df.empty:
        master = fuzzy_merge(master, fd_df, on="Driver", how="left")

    # Betting odds (store as numeric for proper sorting, rounded for display)
    if odds_data:
        from src.data import round_odds
        def _parse_odds(v):
            if v is None or str(v).strip() in ("", "None", "null"):
                return None
            try:
                raw = int(float(str(v).replace("+", "")))
                return round_odds(raw)
            except (ValueError, TypeError):
                return None
        # Use fuzzy matching to handle name format differences (Jr. vs Jr, etc.)
        odds_keys = list(odds_data.keys())
        def _match_odds(driver):
            if driver in odds_data:
                return odds_data[driver]
            matched = fuzzy_match_name(driver, odds_keys)
            return odds_data.get(matched) if matched else None
        master["Win Odds"] = master["Driver"].map(_match_odds).map(_parse_odds)

    # Top 3 / Top 5 / Top 10 finish odds (informational only, from sportsbook)
    if prop_odds is None:
        prop_odds = {"top3": {}, "top5": {}, "top10": {}}
    for label, key in [("Top 3 Odds", "top3"), ("Top 5 Odds", "top5"), ("Top 10 Odds", "top10")]:
        pdata = prop_odds.get(key, {})
        if pdata:
            pkeys = list(pdata.keys())
            def _match_prop(driver, _d=pdata, _k=pkeys):
                if driver in _d:
                    return _d[driver]
                matched = fuzzy_match_name(driver, _k)
                return _d.get(matched) if matched else None
            master[label] = master["Driver"].map(_match_prop)

    # Qualifying
    if not qualifying_df.empty and "Qualifying Position" not in master.columns:
        qual_want = ["Driver", "Qualifying Position", "Best Lap Speed"]
        qual_cols = qualifying_df[[c for c in qual_want if c in qualifying_df.columns]].copy()
        qual_cols = qual_cols.drop_duplicates("Driver")
        qual_cols = qual_cols.rename(columns={"Qualifying Position": "Qual", "Best Lap Speed": "Qual Speed"})
        master = master.merge(qual_cols, on="Driver", how="left")

    # Sponsor from lap averages
    if not lap_averages_df.empty and "Sponsor" in lap_averages_df.columns and "Sponsor" not in master.columns:
        sponsor_map = lap_averages_df.drop_duplicates("Driver").set_index("Driver")["Sponsor"].to_dict()
        sponsor_norm = build_norm_lookup(sponsor_map)
        master["Sponsor"] = master["Driver"].map(
            lambda d: fuzzy_get(d, sponsor_map, sponsor_norm))

    # Practice lap average ranks
    if not lap_averages_df.empty:
        prac_cols = ["Driver"]
        for col in ["Overall Avg", "Overall Rank", "Best Lap",
                     "1 Lap Rank", "5 Lap Rank", "10 Lap Rank", "15 Lap Rank",
                     "20 Lap Rank", "25 Lap Rank", "30 Lap Rank"]:
            if col in lap_averages_df.columns:
                prac_cols.append(col)
        master = master.merge(lap_averages_df[prac_cols].drop_duplicates("Driver"), on="Driver", how="left")

    # Track History — DB-backed (Next Gen era 2022+), cleaner than scraper
    with st.spinner(f"Loading {track_name} history..."):
        th_df = query_db_track_history(track_name, series_id, min_season=2022)
    if not th_df.empty:
        th_rename = {
            "Races": "TH_Races", "Avg Finish": "TH_Avg Finish", "Avg Start": "TH_Avg Start",
            "Avg Run Pos": "TH_Avg Run Pos", "Wins": "TH_Wins", "Top 5": "TH_T5",
            "Top 10": "TH_T10", "Laps Led": "TH_Laps Led",
            "DNF": "TH_DNF",
        }
        th_merge = th_df.rename(columns=th_rename)
        th_cols = ["Driver"] + [v for v in th_rename.values() if v in th_merge.columns]
        th_dedup = th_merge[th_cols].drop_duplicates("Driver")
        # Build fuzzy name mapping: master Driver → track history Driver
        th_names = th_dedup["Driver"].tolist()
        name_map = {}
        for drv in master["Driver"].unique():
            if drv in th_names:
                name_map[drv] = drv
            else:
                matched = fuzzy_match_name(drv, th_names)
                if matched:
                    name_map[drv] = matched
        # Map master drivers to TH driver names, merge, then restore original names
        master["_th_key"] = master["Driver"].map(name_map)
        th_dedup = th_dedup.rename(columns={"Driver": "_th_key"})
        master = master.merge(th_dedup, on="_th_key", how="left")
        master = master.drop(columns=["_th_key"])

    # Historical DK points at this track
    dk_hist = query_driver_dk_points_at_track(track_name, series_id, min_season=2022)
    if dk_hist:
        dk_hist_names = list(dk_hist.keys())
        for col, key in [("TH_Avg DK", "avg_dk"), ("TH_Best DK", "best_dk"), ("TH_Worst DK", "worst_dk")]:
            def _get_dk(d, _key=key):
                h = dk_hist.get(d)
                if not h:
                    m = fuzzy_match_name(d, dk_hist_names)
                    h = dk_hist.get(m) if m else None
                return h[_key] if h else None
            master[col] = master["Driver"].map(_get_dk)

    # Search
    search = st.text_input("Search driver / team / make...", "", placeholder="Type to filter...",
                           label_visibility="collapsed", key="data_search")

    # --- Build column groups for display ---
    driver_info = ["Driver"]
    for c in ["DK Salary", "FD Salary", "Win Odds", "Top 3 Odds", "Top 5 Odds", "Top 10 Odds"]:
        if c in master.columns:
            driver_info.append(c)

    # Results columns (postrace only)
    results_group = []
    if not is_prerace:
        for c in ["Finish Position", "Start", "Laps Led", "Fastest Laps",
                   "DK Pts", "FD Pts", "Avg Run", "Pos Diff", "Status"]:
            if c in master.columns:
                results_group.append(c)

    qualifying = []
    for c in ["Qual", "Qual Speed"]:
        if c in master.columns:
            qualifying.append(c)

    car_info = []
    for c in ["Car", "Manufacturer", "Team", "Sponsor", "Crew Chief"]:
        if c in master.columns:
            car_info.append(c)

    practice = []
    for c in ["Overall Avg", "Overall Rank", "1 Lap Rank",
              "5 Lap Rank", "10 Lap Rank", "15 Lap Rank",
              "20 Lap Rank", "25 Lap Rank", "30 Lap Rank"]:
        if c in master.columns:
            practice.append(c)

    track_history = []
    for c in ["TH_Races", "TH_Avg DK", "TH_Best DK", "TH_Worst DK",
              "TH_Avg Finish", "TH_Avg Start", "TH_Avg Run Pos",
              "TH_Wins", "TH_T5", "TH_T10", "TH_Laps Led", "TH_DNF"]:
        if c in master.columns:
            track_history.append(c)

    # Flat column order
    col_order = driver_info + results_group + qualifying + car_info + practice + track_history
    avail = [c for c in col_order if c in master.columns]
    display_df = master[avail].copy()

    # Apply smart formatting
    display_df = format_display_df(display_df)

    if search:
        mask = display_df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        display_df = display_df[mask]

    # Track history: large sentinel for "lower is better" cols, 0 for count/rating cols
    th_high_sentinel = {"TH_Avg Finish": 99, "TH_Avg Start": 99, "TH_DNF": 0}
    th_zero_fill = ["TH_Races", "TH_Rating", "TH_Wins", "TH_T5", "TH_T10", "TH_T20", "TH_Laps Led"]
    for col, val in th_high_sentinel.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].fillna(val)
    for col in th_zero_fill:
        if col in display_df.columns:
            display_df[col] = display_df[col].fillna(0)

    # Sort (before replacing null odds with dash, so numeric sort works)
    if not is_prerace and "Finish Position" in display_df.columns:
        display_df = display_df.sort_values("Finish Position", na_position="last")
    elif is_prerace and "Win Odds" in display_df.columns and display_df["Win Odds"].notna().any():
        display_df = display_df.sort_values("Win Odds", na_position="last")
    elif is_prerace and "TH_Avg Finish" in display_df.columns:
        display_df = display_df.sort_values("TH_Avg Finish", na_position="last")
    elif "Qual" in display_df.columns:
        display_df = display_df.sort_values("Qual", na_position="last")

    # Replace null odds with dash for display (after sorting)
    for odds_col in ["Win Odds", "Top 3 Odds", "Top 5 Odds", "Top 10 Odds",
                      "Est. Odds", "Est. Impl %"]:
        if odds_col in display_df.columns:
            display_df[odds_col] = display_df[odds_col].astype(object).fillna("—")

    # Build MultiIndex columns for grouped headers
    group_map = {}
    for c in driver_info:
        group_map[c] = "Driver Info"
    for c in results_group:
        group_map[c] = "Results"
    for c in qualifying:
        group_map[c] = "Qualifying"
    for c in car_info:
        group_map[c] = "Car Info"
    for c in practice:
        group_map[c] = "Practice"
    for c in track_history:
        group_map[c] = "Track History"

    # Build display names — strip "TH_" prefix for Track History columns
    multi_tuples = []
    for c in display_df.columns:
        group = group_map.get(c, "")
        display_name = c.replace("TH_", "") if c.startswith("TH_") else c
        multi_tuples.append((group, display_name))
    display_df.columns = pd.MultiIndex.from_tuples(multi_tuples)

    field_count = len(display_df)
    status = "Post-race" if not is_prerace else "Pre-race"
    st.caption(f"{field_count} drivers  •  {status}")

    st.dataframe(safe_fillna(display_df), width="stretch", hide_index=True, height=600)

    # Export
    if not master.empty:
        csv_data = master[avail].to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", csv_data,
                           f"{race_name.replace(' ', '_')}_data.csv", "text/csv", key="export_data")


def _render_charts_view(completed_races, series_id, selected_year,
                        results_df, lap_data, fl_counts, is_prerace):
    """Render the Charts view."""

    # If postrace, show DFS histogram and start vs finish (full width, stacked)
    if not is_prerace and not results_df.empty:
        avg_run_pos = compute_avg_running_position(lap_data) if lap_data else {}
        res = results_df.copy()
        res["Fastest Laps"] = res["Driver"].map(lambda d: fl_counts.get(d, 0)).astype("Int64")
        res["DFS Points"] = res.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"], r["Laps Led"], r["Fastest Laps"]), axis=1)

        fig = dfs_histogram(res)
        st.plotly_chart(fig, width="stretch")

        fig = start_vs_finish_scatter(res)
        st.plotly_chart(fig, width="stretch")

    # Season trend line
    trend_fig = season_trend_line(series_id, selected_year)
    if trend_fig:
        st.divider()
        st.plotly_chart(trend_fig, width="stretch")

    # Race lap-by-lap chart (from race lap-times data)
    if not is_prerace and lap_data:
        import numpy as np
        st.divider()
        st.caption("Race lap-by-lap times")

        # Build driver list from lap data
        all_drivers = sorted([d.get("FullName", d.get("NickName", "")) for d in lap_data.get("laps", [])])
        default_top = all_drivers[:10] if len(all_drivers) > 10 else all_drivers

        ctrl1, ctrl2 = st.columns([3, 1])
        with ctrl1:
            sel_drivers = st.multiselect("Select drivers", all_drivers, default=default_top, key="race_lap_drivers")
        with ctrl2:
            outlier_pct = st.slider("Outlier filter (x median)", 1.05, 1.40, 1.15, 0.05, key="race_lap_outlier",
                                    help="Hide laps slower than this multiple of driver's median (pit stops, cautions)")

        if sel_drivers:
            # Filter outliers from lap data before charting
            filtered_data = {"laps": []}
            for d in lap_data.get("laps", []):
                driver = d.get("FullName", d.get("NickName", ""))
                if driver not in sel_drivers:
                    continue
                laps = d.get("Laps", [])
                times = [l["LapTime"] for l in laps if l.get("LapTime") and l["LapTime"] > 0]
                if not times:
                    continue
                median_t = np.median(times)
                cutoff = median_t * outlier_pct
                clean_laps = [l for l in laps if l.get("LapTime") and 0 < l["LapTime"] <= cutoff]
                if clean_laps:
                    filtered_data["laps"].append({**d, "Laps": clean_laps})

            fig = race_lap_chart(filtered_data, sel_drivers)
            if fig:
                st.plotly_chart(fig, width="stretch")

    # Single race scatter — Avg Running Pos vs DK Points for THIS race
    if not is_prerace and not results_df.empty and lap_data:
        st.divider()
        avg_run_data = compute_avg_running_position(lap_data)
        race_res = results_df.copy()
        race_res["Fastest Laps"] = race_res["Driver"].map(lambda d: fl_counts.get(d, 0)).astype("Int64")
        race_res["DK Pts"] = race_res.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"], r["Laps Led"], r["Fastest Laps"]), axis=1)
        race_res["Avg Run"] = race_res["Driver"].map(lambda d: avg_run_data.get(d))

        fig = race_scatter(race_res)
        if fig:
            st.plotly_chart(fig, width="stretch", key="data_race_scatter")
