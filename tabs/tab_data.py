"""Tab 1: Consolidated Data Hub — unified view for all races."""

import pandas as pd
import streamlit as st

from src.config import SERIES_LABELS
from src.components import section_header, interactive_drill_down_dataframe
from src.data import (
    scrape_track_history, query_db_track_history, query_driver_dk_points_at_track,
    compute_fastest_laps, compute_avg_running_position, load_arp_from_db,
    query_driver_finishes_by_track_type, fetch_lap_times,
)
from src.utils import (
    calc_dk_points, calc_fd_points, safe_fillna, format_display_df,
    fuzzy_match_name, fuzzy_merge, fuzzy_get, build_norm_lookup,
)
from src.charts import (
    dfs_histogram, start_vs_finish_scatter, race_scatter, race_lap_chart,
    season_trend_line, arp_vs_finish_scatter,
    finish_distribution_box, fantasy_vs_arp_scatter, race_speed_chart,
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
                            lap_data, fl_counts, is_prerace, track_name)
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

    # Betting odds (preserve raw values — sportsbooks already post clean
    # rounded odds; re-bucketing here was corrupting the user's paste,
    # e.g. +550 → +600)
    if odds_data:
        from src.utils import normalize_driver_name, parse_american_odds as _parse_odds
        # Use normalized + fuzzy matching to handle name format differences
        # (Jr. vs Jr, Suárez vs Suarez, A.J. vs AJ, etc.)
        odds_keys = list(odds_data.keys())
        # Pre-build normalized lookup for fast matching
        _norm_odds = {normalize_driver_name(k): v for k, v in odds_data.items()}
        def _match_odds(driver):
            # Direct match
            if driver in odds_data:
                return odds_data[driver]
            # Normalized match (handles periods, accents, suffixes)
            norm = normalize_driver_name(driver)
            if norm in _norm_odds:
                return _norm_odds[norm]
            # Fuzzy fallback
            matched = fuzzy_match_name(driver, odds_keys)
            return odds_data.get(matched) if matched else None
        master["Win Odds"] = master["Driver"].map(_match_odds).map(_parse_odds)

        # Diagnostic: show how many odds matched vs. total
        _matched = master["Win Odds"].notna().sum()
        _total_drivers = len(master)
        if _matched == 0 and len(odds_data) > 0:
            # No matches at all — show debugging info to help identify the mismatch
            _sample_odds = list(odds_data.keys())[:3]
            _sample_drivers = master["Driver"].head(3).tolist()
            st.warning(
                f"Odds loaded ({len(odds_data)}) but 0 matched to entry list. "
                f"Sample odds names: {_sample_odds}  •  Sample driver names: {_sample_drivers}"
            )

    # Top 3 / Top 5 / Top 10 finish odds (informational only, from sportsbook)
    if prop_odds is None:
        prop_odds = {"top3": {}, "top5": {}, "top10": {}}
    for label, key in [("Top 3 Odds", "top3"), ("Top 5 Odds", "top5"), ("Top 10 Odds", "top10")]:
        pdata = prop_odds.get(key, {})
        if pdata:
            pkeys = list(pdata.keys())
            # Match direct -> normalized (fast, handles periods/accents) -> fuzzy
            _pnorm = {normalize_driver_name(k): v for k, v in pdata.items()}
            def _match_prop(driver, _d=pdata, _k=pkeys, _norm=_pnorm):
                if driver in _d:
                    return _d[driver]
                nk = normalize_driver_name(driver)
                if nk in _norm:
                    return _norm[nk]
                matched = fuzzy_match_name(driver, _k)
                return _d.get(matched) if matched else None
            master[label] = master["Driver"].map(_match_prop)

    # Qualifying
    if not qualifying_df.empty and "Qualifying Position" not in master.columns:
        qual_want = ["Driver", "Qualifying Position", "Best Lap Speed"]
        qual_cols = qualifying_df[[c for c in qual_want if c in qualifying_df.columns]].copy()
        qual_cols = qual_cols.drop_duplicates("Driver")
        qual_cols = qual_cols.rename(columns={"Qualifying Position": "Qual", "Best Lap Speed": "Qual Speed"})
        # Use fuzzy_merge for name-variation safety (e.g. Suárez/Suarez)
        master = fuzzy_merge(master, qual_cols, on="Driver", how="left")

    # Sponsor from lap averages
    if not lap_averages_df.empty and "Sponsor" in lap_averages_df.columns and "Sponsor" not in master.columns:
        sponsor_map = lap_averages_df.drop_duplicates("Driver").set_index("Driver")["Sponsor"].to_dict()
        sponsor_norm = build_norm_lookup(sponsor_map)
        master["Sponsor"] = master["Driver"].map(
            lambda d: fuzzy_get(d, sponsor_map, sponsor_norm))

    # Practice lap average ranks — use fuzzy_merge so name variations
    # from the lap-averages feed (e.g. "John Hunter Nemechek" vs entry-list
    # "John H. Nemechek", "Daniel Suarez" vs "Daniel Suárez") still join.
    # Plain pandas merge requires exact string match and silently produces
    # NaN for mismatched driver names.
    if not lap_averages_df.empty:
        prac_cols = ["Driver"]
        for col in ["Overall Avg", "Overall Rank", "Best Lap",
                     "1 Lap Rank", "5 Lap Rank", "10 Lap Rank", "15 Lap Rank",
                     "20 Lap Rank", "25 Lap Rank", "30 Lap Rank"]:
            if col in lap_averages_df.columns:
                prac_cols.append(col)
        master = fuzzy_merge(master, lap_averages_df[prac_cols].drop_duplicates("Driver"),
                              on="Driver", how="left")

    # Track History — DB-backed (Next Gen era 2022+), cleaner than scraper
    with st.spinner(f"Loading {track_name} history..."):
        th_df = query_db_track_history(track_name, series_id, min_season=2022)
    if not th_df.empty:
        th_rename = {
            "Races": "TH_Races", "Avg Finish": "TH_Avg Finish", "Avg Start": "TH_Avg Start",
            "Avg Run Pos": "TH_Avg Run Pos", "Avg Rating": "TH_Rating",
            "Wins": "TH_Wins", "Top 5": "TH_T5",
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
    for c in ["TH_Races", "TH_Rating", "TH_Avg DK", "TH_Best DK", "TH_Worst DK",
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

    # Replace null odds with dash for display (after sorting). Cast the WHOLE
    # column to clean strings: a numeric column with .fillna("—") becomes a MIX
    # of ints and the "—" string, which Streamlit/pyarrow cannot serialize
    # ("Could not convert '—' ... tried to convert to int64"). Stringifying the
    # whole column keeps Arrow happy and the dash visible.
    for odds_col in ["Win Odds", "Top 3 Odds", "Top 5 Odds", "Top 10 Odds",
                      "Est. Odds", "Est. Impl %"]:
        if odds_col in display_df.columns:
            display_df[odds_col] = [
                "—" if pd.isna(v) else (str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v))
                for v in display_df[odds_col]
            ]

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
    st.caption(f"{field_count} drivers  •  {status}  •  Click any driver row for race-by-race history")

    interactive_drill_down_dataframe(
        safe_fillna(display_df),
        key=f"data_main_{series_id}_{race_id}",
        series_id=series_id, track_name=track_name,
        width="stretch", hide_index=True, height=600,
    )

    # Export
    if not master.empty:
        csv_data = master[avail].to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", csv_data,
                           f"{race_name.replace(' ', '_')}_data.csv", "text/csv", key="export_data")

    # ── Track-Type Recent History (per-driver finish-position grid) ──
    # Each row is a driver, each column is one of the most recent races at
    # this track type. Cells show finish position. Newest race on the LEFT.
    # Folding rules (handled in query):
    #   - short_concrete includes regular short tracks
    #   - intermediate / intermediate_worn include each other
    if track_type and not master.empty:
        _render_track_type_history(
            master_df=master, track_type=track_type, series_id=series_id,
            track_name=track_name,
        )


def _render_track_type_history(master_df, track_type, series_id, track_name=""):
    """Render the 'Recent finishes at <track_type>' grid table."""
    drivers = master_df["Driver"].dropna().unique().tolist()
    if not drivers:
        return

    # Pull recent races at this track type for these drivers. We over-fetch
    # per-driver (last 15) so we have enough fill density when we cap columns
    # at the 10 most recent races across the whole field.
    meta, finishes = query_driver_finishes_by_track_type(
        track_type, series_id=series_id, drivers=drivers, last_n=15,
    )
    if not meta or not finishes:
        return

    # Cap at the 10 most-recent races across the field, then reverse so newest
    # is on the LEFT (most-recent-first as user requested).
    meta_newest_first = list(reversed(meta))[:10]

    # Build the display DataFrame: rows = drivers in master order, cols = race labels
    rows = []
    for d in drivers:
        f_map = finishes.get(d, {})
        if not f_map:
            continue
        row = {"Driver": d}
        for m in meta_newest_first:
            row[m["label"]] = f_map.get(m["race_id"])
        rows.append(row)

    if not rows:
        return

    grid_df = pd.DataFrame(rows)

    # Drop columns that have NO data for any driver in the field (some races
    # might exist in DB but no included driver entered them — rare but possible)
    race_cols = [c for c in grid_df.columns if c != "Driver"]
    keep_cols = ["Driver"] + [c for c in race_cols
                              if grid_df[c].notna().any()]
    grid_df = grid_df[keep_cols]

    # Sort: drivers with the most recent-track-type races first, then by
    # average finish across the visible cells (best to worst)
    race_only = grid_df.drop(columns=["Driver"])
    grid_df["_n"] = race_only.notna().sum(axis=1)
    grid_df["_avg"] = race_only.mean(axis=1, skipna=True)
    grid_df = grid_df.sort_values(["_n", "_avg"],
                                   ascending=[False, True],
                                   na_position="last")
    grid_df = grid_df.drop(columns=["_n", "_avg"]).reset_index(drop=True)

    # Heatmap-color the finish-position cells (1 = green, 40 = red)
    from src.components import style_heatmap
    rank_cols = [c for c in grid_df.columns if c != "Driver"]
    # Cast to nullable int for clean display ("23" not "23.0")
    for c in rank_cols:
        grid_df[c] = pd.to_numeric(grid_df[c], errors="coerce").astype("Int64")

    type_label = {
        "intermediate": "Intermediate",
        "intermediate_worn": "Intermediate (incl. worn)",
        "short": "Short Track",
        "short_concrete": "Short Track (incl. concrete)",
        "superspeedway": "Superspeedway",
        "road": "Road Course",
    }.get(track_type, track_type.title())

    st.markdown("---")
    section_header(
        f"Recent Finishes at {type_label}",
        f"Last {len(rank_cols)} races per driver  •  newest left → oldest right  •  Click any driver row for full history",
    )

    styled = style_heatmap(grid_df, rank_cols, max_rank=40)
    # Plus-one each finish cell rendered in compact format
    fmt_map = {c: "{:.0f}" for c in rank_cols}
    styled = styled.format(fmt_map, na_rep="—")
    interactive_drill_down_dataframe(
        styled,
        key=f"data_tt_recent_{series_id}_{track_type}",
        series_id=series_id, track_type=track_type,
        width="stretch", hide_index=True, height=560,
    )


def _render_charts_view(completed_races, series_id, selected_year,
                        results_df, lap_data, fl_counts, is_prerace,
                        track_name=""):
    """Render the Charts view."""

    # If postrace, show DFS histogram and start vs finish (full width, stacked)
    if not is_prerace and not results_df.empty:
        avg_run_pos = compute_avg_running_position(lap_data) if lap_data else {}
        res = results_df.copy()
        _fl_norm = build_norm_lookup(fl_counts)
        res["Fastest Laps"] = res["Driver"].map(
            lambda d: fuzzy_get(d, fl_counts, _fl_norm) or 0).astype("Int64")
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

    # ── Track history charts (ARP vs Finish, Rating vs Finish, Finish Distribution) ──
    if track_name:
        th_rows = query_db_track_history(track_name, series_id)
        if not th_rows.empty:
            hist_df = pd.DataFrame(th_rows)
            # Standardize column names for chart functions
            col_map = {}
            for c in hist_df.columns:
                cl = c.lower()
                if cl in ("driver", "full_name"):
                    col_map[c] = "Driver"
                elif cl in ("avg_finish", "avg_finish_pos"):
                    col_map[c] = "Avg Finish"
                elif cl in ("avg_run_pos", "avg_running_pos", "avg_running_position"):
                    col_map[c] = "Avg Run Pos"
            if col_map:
                hist_df = hist_df.rename(columns=col_map)

            # ARP vs Finish scatter — shows wreck luck
            if "Avg Run Pos" in hist_df.columns and "Avg Finish" in hist_df.columns:
                st.divider()
                fig = arp_vs_finish_scatter(hist_df, track_name)
                if fig:
                    st.plotly_chart(fig, width="stretch", key="data_arp_vs_finish")

        # Avg Fantasy Points vs Avg Running Position
        st.divider()
        fig = fantasy_vs_arp_scatter(track_name, series_id)
        if fig:
            st.plotly_chart(fig, width="stretch", key="data_fantasy_vs_arp")

        # Finish distribution box plot
        st.divider()
        fig = finish_distribution_box(track_name, series_id)
        if fig:
            st.plotly_chart(fig, width="stretch", key="data_finish_dist")

    # ── Speed Over Time — driver-selectable lap time / speed overlay ──────────
    st.divider()
    _render_speed_over_time(completed_races, series_id, selected_year,
                            lap_data, results_df, is_prerace)

    # Single race scatter — Avg Running Pos vs DK Points for THIS race
    if not is_prerace and not results_df.empty and lap_data:
        st.divider()
        avg_run_data = compute_avg_running_position(lap_data)
        race_res = results_df.copy()
        _fl_norm2 = build_norm_lookup(fl_counts)
        _arp_norm = build_norm_lookup(avg_run_data)
        race_res["Fastest Laps"] = race_res["Driver"].map(
            lambda d: fuzzy_get(d, fl_counts, _fl_norm2) or 0).astype("Int64")
        race_res["DK Pts"] = race_res.apply(
            lambda r: calc_dk_points(r["Finish Position"], r["Start"], r["Laps Led"], r["Fastest Laps"]), axis=1)
        race_res["Avg Run"] = race_res["Driver"].map(
            lambda d: fuzzy_get(d, avg_run_data, _arp_norm))

        fig = race_scatter(race_res)
        if fig:
            st.plotly_chart(fig, width="stretch", key="data_race_scatter")


def _render_speed_over_time(completed_races, series_id, selected_year, lap_data,
                            results_df, is_prerace):
    """Driver-selectable overlay of lap TIME (default) or speed across a race.

    Defaults to the currently-selected race (its lap data is already loaded) but
    lets you pick any completed race. Moved here from Race Lab."""
    st.markdown("**Speed Over Time** — lap-by-lap overlay")

    race_labels, label_to_race = [], {}
    for _, race in (completed_races or []):
        track = race.get("track_name", "")
        name = race.get("race_name", "")
        date = (race.get("race_date", "") or "")[:10]
        lbl = f"{date} — {track}: {name}"
        race_labels.append(lbl); label_to_race[lbl] = race
    race_labels = list(reversed(race_labels))

    CURRENT = "Current race (selected above)"
    has_current = bool(lap_data and not is_prerace and lap_data.get("laps"))
    options = ([CURRENT] + race_labels) if has_current else race_labels
    if not options:
        st.caption("No race lap-by-lap data available for this series/year.")
        return

    pick = st.selectbox("Race", options, index=0, key="data_speed_race")
    if pick == CURRENT:
        ld = lap_data
    else:
        race = label_to_race[pick]
        yr = (race.get("race_date", "") or "")[:4]
        try:
            yr = int(yr)
        except (TypeError, ValueError):
            yr = selected_year
        with st.spinner("Loading lap-by-lap data..."):
            ld = fetch_lap_times(series_id, race.get("race_id"), yr)
    if not ld or "laps" not in ld:
        st.info("Lap-by-lap data isn't available for this race.")
        return

    all_drivers = sorted(d.get("FullName") for d in ld["laps"] if d.get("FullName"))
    if not all_drivers:
        st.info("No lap data to plot for this race.")
        return

    # Default to the top-5 finishers when we have results for the current race,
    # else the first five drivers.
    default = all_drivers[:5]
    if pick == CURRENT and not results_df.empty and "Finish Position" in results_df.columns:
        try:
            top = results_df.sort_values("Finish Position")["Driver"].tolist()
            default = [d for d in top if d in all_drivers][:5] or all_drivers[:5]
        except Exception:
            pass

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        picks = st.multiselect("Drivers to overlay", all_drivers, default=default,
                               key="data_speed_drivers")
    with c2:
        metric_label = st.radio("Metric", ["Lap Time", "Speed"], horizontal=True,
                                key="data_speed_metric")
    with c3:
        green_only = st.toggle("Green-flag laps only", value=True,
                               key="data_speed_greenonly",
                               help="Drop caution AND green-flag pit laps so only "
                                    "clean racing pace shows.")
    if not picks:
        st.info("Select at least one driver.")
        return

    metric = "speed" if metric_label == "Speed" else "time"
    fig = race_speed_chart(ld, selected_drivers=picks, green_only=green_only,
                           metric=metric)
    if fig is None:
        st.info("No lap data to plot for the selected drivers.")
        return
    st.plotly_chart(fig, width="stretch")
    _unit = ("lap time (s, faster = higher — axis inverted)" if metric == "time"
             else "lap speed (mph)")
    if green_only:
        st.caption(f"Each line is a driver's {_unit}. Caution laps and green-flag "
                   "pit laps are removed, so this is clean racing pace.")
    else:
        st.caption(f"Each line is a driver's {_unit}. Shaded bands are caution "
                   "periods; the deep dips are pit stops. Toggle 'green-flag laps "
                   "only' for clean racing pace.")
