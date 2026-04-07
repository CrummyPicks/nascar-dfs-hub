"""Tab 4: Race Analyzer — Cross-race analysis (season stats, driver comparison, driver lookup)."""

import pandas as pd
import streamlit as st
from datetime import datetime

from src.config import SERIES_OPTIONS, TRACK_TYPE_MAP, TRACK_TYPE_PARENT
from src.data import (
    fetch_race_list, fetch_weekend_feed, fetch_lap_times,
    extract_race_results, compute_fastest_laps, compute_avg_running_position,
    filter_point_races, query_salaries,
)
from src.utils import calc_dk_points, calc_fd_points, safe_fillna, format_display_df
from src.components import render_driver_race_log
from src.charts import race_scatter


def _format_track_type_label(t: str) -> str:
    """Human-readable label for track type dropdown."""
    if t == "All Types":
        return t
    if t.startswith("All "):
        return t
    return t.replace("_", " — ").title() if "_" in t else t.title()


def _get_tracks_for_type(track_type: str) -> list:
    """Get list of track names that belong to a given type or parent group."""
    if track_type.startswith("All "):
        parent = track_type.replace("All ", "").lower()
        return sorted(t for t, tt in TRACK_TYPE_MAP.items()
                      if TRACK_TYPE_PARENT.get(tt, tt) == parent)
    return sorted(t for t, tt in TRACK_TYPE_MAP.items() if tt == track_type)


def render(*, completed_races, series_id, selected_year, series_name="Cup"):
    """Render the Race Analyzer tab."""
    st.markdown("### Race Analyzer")
    st.caption(f"Analyzing **{series_name}** — {selected_year} (change series/year in the top bar)")

    # --- Mode selector ---
    mode = st.radio("Analysis Mode",
                    ["Single Race", "Season Summary", "By Track Type", "Driver Lookup", "Driver Comparison"],
                    horizontal=True, label_visibility="collapsed", key="ra_mode")

    # Use global series/year from top bar. Only offer track-level filters here.
    ra_series_id = series_id
    ra_series_name = series_name

    # Sync year filter with global selector when it changes
    if "ra_year_synced_from" not in st.session_state:
        st.session_state["ra_year_synced_from"] = selected_year
    if st.session_state.get("ra_year_synced_from") != selected_year:
        st.session_state["ra_year"] = selected_year
        st.session_state["ra_year_synced_from"] = selected_year

    with st.expander("Filters", expanded=False):
        f_cols = st.columns(4)
        with f_cols[0]:
            year_options = ["All Years", 2026, 2025, 2024, 2023, 2022]
            default_year_idx = year_options.index(selected_year) if selected_year in year_options else 1
            ra_year_selection = st.selectbox("Year", year_options,
                                            index=default_year_idx, key="ra_year")
        with f_cols[1]:
            # Track type filter
            subtypes = sorted(set(TRACK_TYPE_MAP.values()))
            parent_groups = sorted(set(f"All {p.title()}" for p in set(TRACK_TYPE_PARENT.values())))
            type_options = ["All Types"] + parent_groups + subtypes
            ra_track_type = st.selectbox("Track Type", type_options, key="ra_track_type",
                                         format_func=_format_track_type_label)
        with f_cols[2]:
            # Build track list for filtering
            sample_year = selected_year if ra_year_selection == "All Years" else ra_year_selection
            ra_races = fetch_race_list(ra_series_id, sample_year)
            ra_point_races = filter_point_races(ra_races) if ra_races else []
            tracks = sorted(set(r.get("track_name", "") for r in ra_point_races if r.get("track_name")))
            # Filter track list by track type if selected
            if ra_track_type != "All Types":
                if ra_track_type.startswith("All "):
                    parent = ra_track_type.replace("All ", "").lower()
                    tracks = [t for t in tracks
                              if TRACK_TYPE_PARENT.get(TRACK_TYPE_MAP.get(t, "intermediate"), "intermediate") == parent]
                else:
                    tracks = [t for t in tracks if TRACK_TYPE_MAP.get(t, "intermediate") == ra_track_type]
            ra_track = st.selectbox("Track Filter", ["All Tracks"] + tracks, key="ra_track")
        with f_cols[3]:
            # Show track type description
            if ra_track_type != "All Types":
                desc_tracks = _get_tracks_for_type(ra_track_type)
                if desc_tracks:
                    st.caption(f"**{_format_track_type_label(ra_track_type)}**: {', '.join(desc_tracks)}")

    # Build completed races based on filters
    if ra_year_selection == "All Years":
        years_to_fetch = [2022, 2023, 2024, 2025, 2026]
    else:
        years_to_fetch = [ra_year_selection]

    # Build completed races for each year
    ra_completed = []
    for yr in years_to_fetch:
        if ra_series_id == series_id and yr == selected_year:
            yr_completed = completed_races
        else:
            yr_races = fetch_race_list(ra_series_id, yr)
            yr_point = filter_point_races(yr_races) if yr_races else []
            now = datetime.now()
            yr_completed = []
            for i, race in enumerate(yr_point):
                rd = race.get("race_date", "")
                try:
                    d = datetime.fromisoformat(rd.replace("Z", "+00:00").split("+")[0].split("T")[0])
                    if d.date() <= now.date():
                        yr_completed.append((i, race))
                except Exception:
                    pass
        ra_completed.extend(yr_completed)

    # Filter by track type if selected
    if ra_track_type != "All Types":
        if ra_track_type.startswith("All "):
            parent = ra_track_type.replace("All ", "").lower()
            ra_completed = [(i, r) for i, r in ra_completed
                            if TRACK_TYPE_PARENT.get(
                                TRACK_TYPE_MAP.get(r.get("track_name", ""), "intermediate"),
                                "intermediate"
                            ) == parent]
        else:
            ra_completed = [(i, r) for i, r in ra_completed
                            if TRACK_TYPE_MAP.get(r.get("track_name", ""), "intermediate") == ra_track_type]

    # Filter by specific track if selected
    if ra_track != "All Tracks":
        ra_completed = [(i, r) for i, r in ra_completed if r.get("track_name") == ra_track]

    year_label = "All Years" if ra_year_selection == "All Years" else str(ra_year_selection)

    if mode == "Single Race":
        _render_single_race(ra_completed, ra_series_id, years_to_fetch)
    elif mode == "Season Summary":
        _render_season_summary(ra_completed, ra_series_id, year_label, years_to_fetch)
    elif mode == "By Track Type":
        _render_by_track_type(ra_completed, ra_series_id, year_label, years_to_fetch, ra_track_type)
    elif mode == "Driver Lookup":
        _render_driver_lookup(ra_completed, ra_series_id, year_label, years_to_fetch)
    elif mode == "Driver Comparison":
        _render_driver_comparison(ra_completed, ra_series_id, year_label, years_to_fetch)


def _render_single_race(completed_races, series_id, years_to_fetch):
    """Single race detailed view."""
    if not completed_races:
        st.info("No completed races match the selected filters.")
        return

    # Race picker
    race_labels = []
    race_map = {}
    for rni, rc in completed_races:
        rd_str = rc.get("race_date", "")[:10]
        track = rc.get("track_name", "")
        lbl = f"{rd_str} @ {track}: {rc.get('race_name', 'Unknown')}"
        race_labels.append(lbl)
        race_map[lbl] = rc

    selected = st.selectbox("Select Race", race_labels, index=len(race_labels) - 1,
                            key="ra_single_race")
    race = race_map[selected]
    yr = _get_race_year(race)

    with st.spinner("Loading race data..."):
        results, fl, avg_run = _load_race_results(race, series_id, yr)

    if results.empty:
        st.warning("No results available for this race.")
        return

    results["Fastest Laps"] = results["Driver"].map(lambda d: fl.get(d, 0)).astype("Int64")
    results["Avg Run"] = results["Driver"].map(lambda d: avg_run.get(d))
    results["DK Pts"] = results.apply(
        lambda r: calc_dk_points(r["Finish Position"], r["Start"],
                                 r["Laps Led"], r["Fastest Laps"]), axis=1)
    results["FD Pts"] = results.apply(
        lambda r: calc_fd_points(r["Finish Position"], r["Start"],
                                 r["Laps Led"], r["Fastest Laps"]), axis=1)
    results["Pos Diff"] = (results["Start"] - results["Finish Position"]).astype("Int64")

    # Merge DK Salary from database
    race_id = race.get("race_id")
    sal_df = query_salaries(race_id=race_id, platform="DraftKings")
    if not sal_df.empty:
        sal_df = sal_df.rename(columns={"Salary": "DK Salary"})[["Driver", "DK Salary"]]
        results = results.merge(sal_df, on="Driver", how="left")

    field_size = len(results)
    results["Rating"] = results.apply(
        lambda r: round(
            max(0, (field_size - r["Finish Position"]) / field_size * 100 +
                r["Laps Led"] * 0.1 + r["Fastest Laps"] * 0.5 +
                (r["Start"] - r["Finish Position"]) * 0.8), 1), axis=1)

    # Search
    search = st.text_input("Search", "", placeholder="Filter drivers...",
                           label_visibility="collapsed", key="ra_single_search")

    show = ["Driver", "DK Salary", "Finish Position", "Start", "Laps Led", "Fastest Laps",
            "DK Pts", "FD Pts", "Avg Run", "Rating", "Pos Diff",
            "Car", "Team", "Manufacturer", "Status"]
    avail = [c for c in show if c in results.columns]
    disp = results[avail].copy().sort_values("Finish Position")
    disp = format_display_df(disp)

    if search:
        mask = disp.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        disp = disp[mask]

    st.caption(f"Track: **{race.get('track_name', '')}** | {len(results)} drivers")
    st.dataframe(safe_fillna(disp), use_container_width=True, hide_index=True, height=520)

    # Chart
    fig = race_scatter(results)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="ra_single_scatter")

    # Export
    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button("Export Race CSV", csv,
                       f"race_{race.get('race_id', '')}.csv", "text/csv", key="ra_single_export")


def _load_race_results(race, series_id, year=None):
    """Load full results for a single completed race."""
    rc_id = race.get("race_id")
    # Determine year from race date
    if year is None:
        rd = race.get("race_date", "")
        try:
            year = int(rd[:4])
        except Exception:
            year = 2026
    rc_feed = fetch_weekend_feed(series_id, rc_id, year)
    rc_laps = fetch_lap_times(series_id, rc_id, year)
    if not rc_feed:
        return pd.DataFrame(), {}, {}
    results = extract_race_results(rc_feed)
    fl = compute_fastest_laps(rc_laps) if rc_laps else {}
    avg_run = compute_avg_running_position(rc_laps) if rc_laps else {}
    return results, fl, avg_run


def _get_race_year(race):
    """Extract year from race date string."""
    rd = race.get("race_date", "")
    try:
        return int(rd[:4])
    except Exception:
        return 2026


def _build_season_data(completed_races, series_id, years_to_fetch):
    """Build season-level per-driver aggregation from completed races."""
    all_rows = []
    for rni, rc in completed_races:
        yr = _get_race_year(rc)
        results, fl, avg_run = _load_race_results(rc, series_id, yr)
        if results.empty:
            continue
        # Try to get salaries for this race
        rc_id = rc.get("race_id")
        sal_df = query_salaries(race_id=rc_id, platform="DraftKings")
        sal_map = {}
        if not sal_df.empty:
            sal_map = dict(zip(sal_df["Driver"], sal_df["Salary"]))
        for _, row in results.iterrows():
            driver = row["Driver"]
            fp = row["Finish Position"]
            start = row["Start"]
            ll = row["Laps Led"]
            fl_count = fl.get(driver, 0)
            dk = calc_dk_points(fp, start, ll, fl_count)
            fd = calc_fd_points(fp, start, ll, fl_count)
            all_rows.append({
                "Driver": driver,
                "Car": str(row.get("Car", "")),
                "Race": rc.get("race_name", ""),
                "Track": rc.get("track_name", ""),
                "Date": rc.get("race_date", "")[:10],
                "Year": yr,
                "Finish": fp,
                "Start": start,
                "Laps Led": ll,
                "Fastest Laps": fl_count,
                "Avg Run": avg_run.get(driver),
                "DK Pts": dk,
                "FD Pts": fd,
                "DK Salary": sal_map.get(driver),
                "Status": row.get("Status", ""),
            })
    return pd.DataFrame(all_rows)


def _render_season_summary(completed_races, series_id, year_label, years_to_fetch):
    """Season summary — aggregated stats across all completed races."""
    if not completed_races:
        st.info("No completed races found for the selected filters.")
        return

    with st.spinner(f"Loading {len(completed_races)} races..."):
        season_df = _build_season_data(completed_races, series_id, years_to_fetch)

    if season_df.empty:
        st.info("No race data available.")
        return

    # Aggregate by driver
    agg_dict = {
        "Races": ("Finish", "count"),
        "Avg_Finish": ("Finish", "mean"),
        "Avg_Start": ("Start", "mean"),
        "Best_Finish": ("Finish", "min"),
        "Wins": ("Finish", lambda x: (x == 1).sum()),
        "Top_5": ("Finish", lambda x: (x <= 5).sum()),
        "Top_10": ("Finish", lambda x: (x <= 10).sum()),
        "Total_Laps_Led": ("Laps Led", "sum"),
        "Total_Fast_Laps": ("Fastest Laps", "sum"),
        "Avg_DK": ("DK Pts", "mean"),
        "Avg_FD": ("FD Pts", "mean"),
    }
    col_names = ["Driver", "Car", "Races", "Avg Finish", "Avg Start",
                 "Best Finish", "Wins", "T5", "T10", "Laps Led",
                 "Fast Laps", "Avg DK", "Avg FD"]
    # Include salary if available
    if "DK Salary" in season_df.columns and season_df["DK Salary"].notna().any():
        agg_dict["Avg_Salary"] = ("DK Salary", "mean")
        col_names.append("Avg Salary")

    agg = season_df.groupby(["Driver", "Car"]).agg(**agg_dict).reset_index()
    agg.columns = col_names

    for col in ["Avg Finish", "Avg Start", "Avg DK", "Avg FD"]:
        agg[col] = agg[col].round(1)
    for col in ["Wins", "T5", "T10", "Laps Led", "Fast Laps", "Best Finish"]:
        agg[col] = agg[col].astype(int)
    if "Avg Salary" in agg.columns:
        agg["Avg Salary"] = agg["Avg Salary"].round(0).astype("Int64")

    agg = agg.sort_values("Avg DK", ascending=False).reset_index(drop=True)
    agg.index = agg.index + 1
    agg.index.name = "Rank"

    search = st.text_input("Search...", "", placeholder="Filter drivers...",
                           label_visibility="collapsed", key="ra_season_search")

    st.caption(f"{len(agg)} drivers across {len(completed_races)} races — {year_label}")

    disp = format_display_df(agg.copy())
    if search:
        mask = disp.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        disp = disp[mask]

    st.dataframe(safe_fillna(disp), use_container_width=True, hide_index=False, height=550)

    # Season charts
    import plotly.graph_objects as go
    from src.charts import DARK_LAYOUT, season_scatter
    from src.utils import short_name_series

    # Top 25 Avg DK Points bar chart
    top_dk = agg.head(25).copy()
    if not top_dk.empty:
        top_dk = top_dk.sort_values("Avg DK", ascending=True)
        top_dk["Short"] = short_name_series(top_dk["Driver"].tolist())
        fig_dk = go.Figure(go.Bar(
            y=top_dk["Short"],
            x=top_dk["Avg DK"],
            orientation="h",
            marker=dict(color=top_dk["Avg DK"], colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Avg DK")),
            hovertemplate="<b>%{customdata[0]}</b><br>Avg DK: %{x:.1f}<extra></extra>",
            customdata=top_dk[["Driver"]].values,
        ))
        fig_dk.update_layout(**DARK_LAYOUT, height=max(400, len(top_dk) * 18),
                             title=f"Top 25 Avg DK Points — {year_label}",
                             xaxis_title="Avg DK Points", yaxis_title="")
        st.plotly_chart(fig_dk, use_container_width=True, key="ra_season_dk_bar")

    # Top 25 Avg Finish bar chart
    top_finish = agg.nsmallest(25, "Avg Finish").copy()
    if not top_finish.empty:
        top_finish = top_finish.sort_values("Avg Finish", ascending=False)
        top_finish["Short"] = short_name_series(top_finish["Driver"].tolist())
        fig_fin = go.Figure(go.Bar(
            y=top_finish["Short"],
            x=top_finish["Avg Finish"],
            orientation="h",
            marker=dict(color=top_finish["Avg Finish"], colorscale="RdYlGn_r",
                        showscale=True, colorbar=dict(title="Avg Finish")),
            hovertemplate="<b>%{customdata[0]}</b><br>Avg Finish: %{x:.1f}<extra></extra>",
            customdata=top_finish[["Driver"]].values,
        ))
        fig_fin.update_layout(**DARK_LAYOUT, height=max(400, len(top_finish) * 18),
                              title=f"Top 25 Avg Finish — {year_label}",
                              xaxis_title="Avg Finish Position", yaxis_title="")
        st.plotly_chart(fig_fin, use_container_width=True, key="ra_season_fin_bar")

    # Season scatter
    if not season_df.empty and "Avg Run" in season_df.columns:
        avg_data = season_df.dropna(subset=["Avg Run"]).groupby(["Driver", "Car"]).agg(
            **{"Avg Running Pos": ("Avg Run", "mean"),
               "Avg Driver Rating": ("DK Pts", "mean")}
        ).round(1).reset_index()

        fig = season_scatter(avg_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    csv = agg.to_csv(index=True).encode("utf-8")
    st.download_button("Export Season CSV", csv,
                       f"season_{series_id}_{year_label}.csv", "text/csv", key="ra_season_export")


def _render_by_track_type(completed_races, series_id, year_label, years_to_fetch,
                          selected_track_type="All Types"):
    """Aggregate driver stats across all tracks of a selected type."""
    # Track type selector — subtypes + parent groups
    subtypes = sorted(set(TRACK_TYPE_MAP.values()))
    parent_groups = sorted(set(f"All {p.title()}" for p in set(TRACK_TYPE_PARENT.values())))
    track_types = parent_groups + subtypes
    type_badges = {
        "superspeedway": "🔴", "intermediate": "🟡",
        "short": "🟢", "road": "🔵", "dirt": "🟤",
    }

    def _format_type(t):
        if t.startswith("All "):
            parent = t.replace("All ", "").lower()
            badge = type_badges.get(parent, "")
            return f"{badge} {t}"
        badge = type_badges.get(TRACK_TYPE_PARENT.get(t, t), "")
        return f"{badge} {t.replace('_', ' ').title()}"

    # Default to filter selection if a specific type was already chosen
    default_idx = 0
    if selected_track_type != "All Types" and selected_track_type in track_types:
        default_idx = track_types.index(selected_track_type)

    chosen_type = st.selectbox(
        "Track Type",
        track_types,
        index=default_idx,
        format_func=_format_type,
        key="ra_tt_type_select",
    )

    # Filter completed races to matching track type (parent or subtype)
    if chosen_type.startswith("All "):
        parent = chosen_type.replace("All ", "").lower()
        type_races = [
            (i, r) for i, r in completed_races
            if TRACK_TYPE_PARENT.get(
                TRACK_TYPE_MAP.get(r.get("track_name", ""), "intermediate"),
                "intermediate"
            ) == parent
        ]
    else:
        type_races = [
            (i, r) for i, r in completed_races
            if TRACK_TYPE_MAP.get(r.get("track_name", ""), "intermediate") == chosen_type
        ]

    # Show which tracks are included
    included_tracks = sorted(set(r.get("track_name", "") for _, r in type_races))
    display_name = chosen_type.replace("_", " ").title()
    if included_tracks:
        st.caption(f"**{display_name}** tracks: {', '.join(included_tracks)}")
    else:
        st.info(f"No completed races at {display_name} tracks for the selected filters.")
        return

    if not type_races:
        st.info(f"No completed races at {chosen_type} tracks.")
        return

    with st.spinner(f"Loading {len(type_races)} {chosen_type} races..."):
        season_df = _build_season_data(type_races, series_id, years_to_fetch)

    if season_df.empty:
        st.info("No race data available.")
        return

    # Aggregate by driver
    agg = season_df.groupby(["Driver", "Car"]).agg(
        Races=("Finish", "count"),
        **{"Avg Finish": ("Finish", "mean")},
        **{"Avg Start": ("Start", "mean")},
        **{"Best Finish": ("Finish", "min")},
        Wins=("Finish", lambda x: (x == 1).sum()),
        T5=("Finish", lambda x: (x <= 5).sum()),
        T10=("Finish", lambda x: (x <= 10).sum()),
        **{"Laps Led": ("Laps Led", "sum")},
        **{"Fast Laps": ("Fastest Laps", "sum")},
        **{"Avg DK": ("DK Pts", "mean")},
        **{"Avg FD": ("FD Pts", "mean")},
    ).reset_index()

    # Add avg running position if available
    if "Avg Run" in season_df.columns and season_df["Avg Run"].notna().any():
        avg_run_agg = season_df.dropna(subset=["Avg Run"]).groupby("Driver").agg(
            **{"Avg Run": ("Avg Run", "mean")}
        ).round(1)
        agg = agg.merge(avg_run_agg, on="Driver", how="left")

    # Compute driver rating per-driver
    field_avg = len(season_df["Driver"].unique())
    agg["Rating"] = agg.apply(
        lambda r: round(
            max(0, (40 - r["Avg Finish"]) / 39 * 60 +
                r["Laps Led"] / max(r["Races"], 1) * 0.3 +
                r["Fast Laps"] / max(r["Races"], 1) * 0.5 +
                (r["Avg Start"] - r["Avg Finish"]) * 1.5 +
                r["Wins"] / max(r["Races"], 1) * 20 +
                r["T5"] / max(r["Races"], 1) * 8), 1), axis=1)

    # Add avg salary if available
    if "DK Salary" in season_df.columns and season_df["DK Salary"].notna().any():
        sal_agg = season_df.dropna(subset=["DK Salary"]).groupby("Driver").agg(
            **{"Avg Salary": ("DK Salary", "mean")}
        ).round(0)
        agg = agg.merge(sal_agg, on="Driver", how="left")
        if "Avg Salary" in agg.columns:
            agg["Avg Salary"] = agg["Avg Salary"].astype("Int64")

    # Format numeric columns
    for col in ["Avg Finish", "Avg Start", "Avg DK", "Avg FD"]:
        if col in agg.columns:
            agg[col] = agg[col].round(1)
    for col in ["Wins", "T5", "T10", "Laps Led", "Fast Laps", "Best Finish"]:
        if col in agg.columns:
            agg[col] = agg[col].astype(int)

    agg = agg.sort_values("Avg DK", ascending=False).reset_index(drop=True)
    agg.index = agg.index + 1
    agg.index.name = "Rank"

    # Search
    search = st.text_input("Search...", "", placeholder="Filter drivers...",
                           label_visibility="collapsed", key="ra_tt_search")

    st.caption(f"{len(agg)} drivers across {len(type_races)} {chosen_type} races — {year_label}")

    disp = format_display_df(agg.copy())
    if search:
        mask = disp.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        disp = disp[mask]

    st.dataframe(safe_fillna(disp), use_container_width=True, hide_index=False, height=550)

    # Avg Running Pos vs Rating scatter chart
    if "Avg Run" in agg.columns and agg["Avg Run"].notna().any():
        scatter_data = agg.dropna(subset=["Avg Run"])[["Driver", "Car", "Avg Run", "Rating"]].copy()
        scatter_data = scatter_data.rename(columns={
            "Avg Run": "Avg Running Pos",
            "Rating": "Avg Driver Rating",
        })
        from src.charts import season_scatter
        fig = season_scatter(scatter_data)
        if fig:
            fig.update_layout(title=f"Avg Running Pos vs Rating — {chosen_type.title()} Tracks")
            st.plotly_chart(fig, use_container_width=True, key="ra_tt_scatter")
    else:
        # Fallback: plot Avg Finish vs Avg DK
        import plotly.graph_objects as go
        from src.charts import DARK_LAYOUT
        plot_df = agg[agg["Races"] >= 2].head(40).copy()
        if not plot_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df["Avg Finish"], y=plot_df["Avg DK"],
                mode="markers+text", text=plot_df["Car"],
                textposition="top right", textfont=dict(size=9, color="#8892a4"),
                marker=dict(size=10, color="#4a7dfc", opacity=0.8),
                hovertemplate="<b>%{customdata[0]}</b><br>Avg Finish: %{x:.1f}<br>Avg DK: %{y:.1f}",
                customdata=plot_df[["Driver"]].values,
            ))
            fig.update_layout(
                **DARK_LAYOUT, height=450,
                title=f"Avg Finish vs Avg DK — {chosen_type.title()} Tracks",
                xaxis_title="Avg Finish Position",
                yaxis_title="Avg DK Points",
                xaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True, key="ra_tt_fallback_scatter")

    # Export
    csv = agg.to_csv(index=True).encode("utf-8")
    st.download_button("Export Track Type CSV", csv,
                       f"track_type_{chosen_type}_{year_label}.csv", "text/csv",
                       key="ra_tt_export")


def _render_driver_lookup(completed_races, series_id, year_label, years_to_fetch):
    """Driver lookup — race-by-race log with summary stats and charts."""
    if not completed_races:
        st.info("No completed races found.")
        return

    driver_search = st.text_input("Driver Name", "", placeholder="e.g. Denny Hamlin",
                                  key="ra_driver_search")
    if not driver_search:
        st.caption("Enter a driver name to see their race-by-race results")
        return

    with st.spinner(f"Loading race data for '{driver_search}'..."):
        season_df = _build_season_data(completed_races, series_id, years_to_fetch)

    if season_df.empty:
        st.info("No race data available.")
        return

    matches = season_df[season_df["Driver"].str.contains(driver_search, case=False, na=False)]
    if matches.empty:
        st.info(f"No results found for '{driver_search}'")
        return

    matched_drivers = matches["Driver"].unique()
    if len(matched_drivers) > 1:
        driver_pick = st.selectbox("Multiple matches — select driver",
                                    sorted(matched_drivers), key="ra_driver_pick")
        matches = matches[matches["Driver"] == driver_pick]
    else:
        driver_pick = matched_drivers[0]

    matches = matches.sort_values("Date")

    # ── Summary Stats Row ──────────────────────────────────────────────────
    m = matches
    m_cols = st.columns(6)
    m_cols[0].metric("Races", len(m))
    m_cols[1].metric("Avg Finish", f"{m['Finish'].mean():.1f}" if not m.empty else "—")
    m_cols[2].metric("Avg DK Pts", f"{m['DK Pts'].mean():.1f}" if not m.empty else "—")
    m_cols[3].metric("Wins", int((m["Finish"] == 1).sum()))
    m_cols[4].metric("Top 5s", int((m["Finish"] <= 5).sum()))
    m_cols[5].metric("Top 10s", int((m["Finish"] <= 10).sum()))

    m2_cols = st.columns(6)
    m2_cols[0].metric("Avg Start", f"{m['Start'].mean():.1f}" if not m.empty else "—")
    m2_cols[1].metric("Best Finish", int(m["Finish"].min()) if not m.empty else "—")
    m2_cols[2].metric("Total Laps Led", int(m["Laps Led"].sum()))
    m2_cols[3].metric("Total Fast Laps", int(m["Fastest Laps"].sum()))
    if "DK Salary" in m.columns and m["DK Salary"].notna().any():
        m2_cols[4].metric("Avg Salary", f"${m['DK Salary'].mean():,.0f}")
    if m["DK Pts"].notna().any():
        m2_cols[5].metric("Best DK", f"{m['DK Pts'].max():.1f}")

    # ── Track Type Breakdown ───────────────────────────────────────────────
    if "Track" in m.columns:
        m_with_type = m.copy()
        m_with_type["Track Type"] = m_with_type["Track"].map(
            lambda t: TRACK_TYPE_MAP.get(t, "intermediate"))
        m_with_type["Parent Type"] = m_with_type["Track Type"].map(
            lambda t: TRACK_TYPE_PARENT.get(t, t))
        type_agg = m_with_type.groupby("Parent Type").agg(
            Races=("Finish", "count"),
            **{"Avg Finish": ("Finish", "mean")},
            **{"Avg DK": ("DK Pts", "mean")},
        ).round(1).sort_values("Avg DK", ascending=False)
        if len(type_agg) > 1:
            with st.expander("Performance by Track Type", expanded=False):
                st.dataframe(safe_fillna(format_display_df(type_agg)),
                             use_container_width=True, hide_index=False)

    # ── Race Log Table ─────────────────────────────────────────────────────
    st.markdown(f"#### {driver_pick} — Race Log")
    show_cols = ["Date", "Race", "Track", "Start", "Finish", "Laps Led",
                 "Fastest Laps", "DK Pts", "FD Pts", "Status"]
    if "DK Salary" in m.columns:
        show_cols.insert(6, "DK Salary")
    avail = [c for c in show_cols if c in m.columns]
    disp = format_display_df(m[avail].copy())
    st.dataframe(safe_fillna(disp), use_container_width=True, hide_index=True, height=400)

    # ── Charts ─────────────────────────────────────────────────────────────
    import plotly.graph_objects as go
    from src.charts import DARK_LAYOUT

    # DK Points trend line
    if len(m) >= 2 and "DK Pts" in m.columns:
        fig_dk = go.Figure()
        fig_dk.add_trace(go.Scatter(
            x=m["Date"], y=m["DK Pts"],
            mode="lines+markers",
            marker=dict(size=8, color="#4a7dfc"),
            line=dict(color="#4a7dfc", width=2),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>DK Pts: %{y:.1f}<extra></extra>",
            customdata=m[["Race", "Track"]].values,
        ))
        # Add rolling average if enough races
        if len(m) >= 4:
            rolling = m["DK Pts"].rolling(3, min_periods=2).mean()
            fig_dk.add_trace(go.Scatter(
                x=m["Date"], y=rolling,
                mode="lines", name="3-Race Avg",
                line=dict(color="#ff9f43", width=2, dash="dash"),
            ))
        fig_dk.update_layout(**DARK_LAYOUT, height=350,
                             title=f"{driver_pick} — DK Points Trend",
                             xaxis_title="", yaxis_title="DK Points",
                             showlegend=len(m) >= 4,
                             xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
        st.plotly_chart(fig_dk, use_container_width=True, key="ra_lookup_dk_trend")

    # Finish Position trend
    if len(m) >= 2 and "Finish" in m.columns:
        fig_fin = go.Figure()
        fig_fin.add_trace(go.Scatter(
            x=m["Date"], y=m["Finish"],
            mode="lines+markers",
            marker=dict(size=8, color="#36b37e"),
            line=dict(color="#36b37e", width=2),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Finish: %{y}<extra></extra>",
            customdata=m[["Race", "Track"]].values,
        ))
        fig_fin.update_layout(**DARK_LAYOUT, height=300,
                              title=f"{driver_pick} — Finish Position Trend",
                              xaxis_title="", yaxis_title="Finish Position",
                              yaxis=dict(autorange="reversed"),
                              xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
        st.plotly_chart(fig_fin, use_container_width=True, key="ra_lookup_fin_trend")

    # Export
    csv = m[avail].to_csv(index=False).encode("utf-8")
    st.download_button("Export Driver CSV", csv,
                       f"driver_{driver_pick.replace(' ', '_')}.csv", "text/csv",
                       key="ra_lookup_export")


def _render_driver_comparison(completed_races, series_id, year_label, years_to_fetch):
    """Side-by-side driver comparison across the season."""
    if not completed_races:
        st.info("No completed races found.")
        return

    with st.spinner("Loading season data..."):
        season_df = _build_season_data(completed_races, series_id, years_to_fetch)

    if season_df.empty:
        st.info("No race data available.")
        return

    all_drivers = sorted(season_df["Driver"].unique())

    comp_cols = st.columns(2)
    with comp_cols[0]:
        driver_a = st.selectbox("Driver A", all_drivers, index=0, key="ra_comp_a")
    with comp_cols[1]:
        default_b = min(1, len(all_drivers) - 1)
        driver_b = st.selectbox("Driver B", all_drivers, index=default_b, key="ra_comp_b")

    if driver_a == driver_b:
        st.warning("Select two different drivers to compare.")
        return

    a_data = season_df[season_df["Driver"] == driver_a]
    b_data = season_df[season_df["Driver"] == driver_b]

    st.markdown(f"### {driver_a} vs {driver_b}")

    def _driver_summary(df):
        return {
            "Races": len(df),
            "Avg Finish": round(df["Finish"].mean(), 1) if not df.empty else 0,
            "Avg Start": round(df["Start"].mean(), 1) if not df.empty else 0,
            "Best Finish": int(df["Finish"].min()) if not df.empty else 0,
            "Wins": int((df["Finish"] == 1).sum()),
            "Top 5": int((df["Finish"] <= 5).sum()),
            "Top 10": int((df["Finish"] <= 10).sum()),
            "Avg DK Pts": round(df["DK Pts"].mean(), 1) if not df.empty else 0,
            "Laps Led": int(df["Laps Led"].sum()),
            "Fast Laps": int(df["Fastest Laps"].sum()),
        }

    a_stats = _driver_summary(a_data)
    b_stats = _driver_summary(b_data)

    comp_rows = []
    for stat in a_stats:
        a_val = a_stats[stat]
        b_val = b_stats[stat]
        if stat in ["Avg Finish", "Avg Start", "Best Finish"]:
            better = "A" if a_val < b_val else ("B" if b_val < a_val else "Tie")
        else:
            better = "A" if a_val > b_val else ("B" if b_val > a_val else "Tie")
        comp_rows.append({
            "Stat": stat,
            driver_a: a_val,
            driver_b: b_val,
            "Edge": driver_a if better == "A" else (driver_b if better == "B" else "—"),
        })

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True, height=420)

    # Race-by-race charts
    import plotly.graph_objects as go
    from src.charts import DARK_LAYOUT

    a_races = a_data[["Race", "Date", "Finish", "DK Pts"]].rename(
        columns={"Finish": f"{driver_a} Finish", "DK Pts": f"{driver_a} DK Pts"})
    b_races = b_data[["Race", "Date", "Finish", "DK Pts"]].rename(
        columns={"Finish": f"{driver_b} Finish", "DK Pts": f"{driver_b} DK Pts"})
    merged = a_races.merge(b_races, on=["Race", "Date"], how="outer").sort_values("Date")

    if not merged.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=merged["Race"], y=merged[f"{driver_a} DK Pts"],
            name=driver_a, marker_color="#3b82f6"))
        fig.add_trace(go.Bar(
            x=merged["Race"], y=merged[f"{driver_b} DK Pts"],
            name=driver_b, marker_color="#ef4444"))
        fig.update_layout(**DARK_LAYOUT, height=350, barmode="group",
                          title="DK Points by Race",
                          xaxis_title="", yaxis_title="DK Points",
                          xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=merged["Race"], y=merged[f"{driver_a} Finish"],
            mode="lines+markers", name=driver_a, marker_color="#3b82f6"))
        fig2.add_trace(go.Scatter(
            x=merged["Race"], y=merged[f"{driver_b} Finish"],
            mode="lines+markers", name=driver_b, marker_color="#ef4444"))
        fig2.update_layout(**DARK_LAYOUT, height=350,
                           title="Finish Position by Race",
                           xaxis_title="", yaxis_title="Finish Position",
                           yaxis=dict(autorange="reversed"),
                           xaxis=dict(tickangle=-45, tickfont=dict(size=9)))
        st.plotly_chart(fig2, use_container_width=True)
