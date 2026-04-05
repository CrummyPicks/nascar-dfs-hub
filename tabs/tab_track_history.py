"""Tab 3: Track History."""

import streamlit as st

from src.config import TRACK_TYPE_MAP
from src.data import (
    scrape_track_history, scrape_track_history_alltime,
    query_track_type_stats, query_season_stats,
)
from src.charts import track_history_bar, rating_vs_finish_scatter
from src.utils import format_display_df, safe_fillna


# Track type badge colors
TRACK_TYPE_BADGES = {
    "superspeedway": "🔴",
    "intermediate": "🟡",
    "short": "🟢",
    "road": "🔵",
    "dirt": "🟤",
}


def render(*, track_name, track_type, series_id):
    """Render the Track History tab."""
    badge = TRACK_TYPE_BADGES.get(track_type, "")
    st.markdown(f"### {track_name} — Driver History")
    st.caption(f"Track type: {badge} **{track_type.title()}**")

    # Track type filter
    filter_cols = st.columns([2, 3])
    with filter_cols[0]:
        type_options = ["This Track"] + sorted(set(TRACK_TYPE_MAP.values()))
        type_filter = st.selectbox("Track Type Filter", type_options,
                                    key="th_type_filter", label_visibility="collapsed",
                                    help="Filter to show stats for a specific track type")

    hist_view = st.radio("View", ["Recent Races", "All-Time", "By Track Type", "2026 Season"],
                         horizontal=True, label_visibility="collapsed")

    if type_filter != "This Track":
        # Show aggregated data for selected track type across all views
        _render_track_type_filtered(type_filter, hist_view, series_id)
        return

    if hist_view == "Recent Races":
        with st.spinner(f"Loading recent history at {track_name}..."):
            hist_df = scrape_track_history(track_name, series_id)
        if not hist_df.empty:
            st.caption(f"Source: driveraverages.com — Recent races at {track_name}")
            display = format_display_df(hist_df)
            st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)

            fig = track_history_bar(hist_df, track_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            fig2 = rating_vs_finish_scatter(hist_df, track_name)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"No data found for {track_name}")

    elif hist_view == "All-Time":
        with st.spinner(f"Loading all-time history at {track_name}..."):
            alltime_df = scrape_track_history_alltime(track_name, series_id)
        if not alltime_df.empty:
            st.caption(f"Source: driveraverages.com — All-time at {track_name}")
            display = format_display_df(alltime_df)
            st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)
        else:
            st.info(f"No all-time data found for {track_name}")

    elif hist_view == "By Track Type":
        st.caption(f"Track type: **{track_type}** — stats from database")
        tt_df = query_track_type_stats(track_type)
        if not tt_df.empty:
            display = format_display_df(tt_df)
            st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)
        else:
            st.info("No track-type data available.")

    elif hist_view == "2026 Season":
        st.caption("Aggregated from all 2026 races in database")
        season_df = query_season_stats()
        if not season_df.empty:
            display = format_display_df(season_df)
            st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)
        else:
            st.info("No season data available.")


def _render_track_type_filtered(track_type_filter, hist_view, series_id):
    """Render filtered view for a specific track type across all tracks."""
    # Get all tracks of this type
    type_tracks = [t for t, tt in TRACK_TYPE_MAP.items() if tt == track_type_filter]

    if hist_view in ("Recent Races", "All-Time"):
        import pandas as pd
        all_data = []
        with st.spinner(f"Loading {track_type_filter} track data..."):
            for t in type_tracks[:8]:  # Limit to avoid too many requests
                fetch_fn = scrape_track_history if hist_view == "Recent Races" else scrape_track_history_alltime
                df = fetch_fn(t, series_id)
                if not df.empty:
                    df["Track"] = t
                    all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            st.caption(f"Source: driveraverages.com — {track_type_filter.title()} tracks ({len(all_data)} tracks)")

            # Aggregate by driver across tracks
            numeric_cols = []
            for col in ["Avg Finish", "Avg Start", "Avg Rating", "Races", "Wins",
                         "Top 5", "Top 10", "Laps Led", "DNF"]:
                if col in combined.columns:
                    combined[col] = pd.to_numeric(combined[col], errors="coerce")
                    numeric_cols.append(col)

            if "Driver" in combined.columns and numeric_cols:
                agg_dict = {}
                for col in numeric_cols:
                    if col in ("Avg Finish", "Avg Start", "Avg Rating"):
                        agg_dict[col] = "mean"
                    else:
                        agg_dict[col] = "sum"
                agg = combined.groupby("Driver").agg(agg_dict).reset_index()
                agg = agg.sort_values("Avg Finish", na_position="last")
                display = format_display_df(agg)
                st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)
            else:
                st.dataframe(safe_fillna(format_display_df(combined)),
                             use_container_width=True, hide_index=True, height=550)
        else:
            st.info(f"No data found for {track_type_filter} tracks.")

    elif hist_view == "By Track Type":
        tt_df = query_track_type_stats(track_type_filter)
        if not tt_df.empty:
            display = format_display_df(tt_df)
            st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)
        else:
            st.info(f"No {track_type_filter} track-type data available.")

    elif hist_view == "2026 Season":
        st.caption(f"Season data filtered to {track_type_filter} tracks")
        season_df = query_season_stats()
        if not season_df.empty:
            display = format_display_df(season_df)
            st.dataframe(safe_fillna(display), use_container_width=True, hide_index=True, height=550)
        else:
            st.info("No season data available.")
