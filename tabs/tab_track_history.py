"""Tab 3: Track History."""

import streamlit as st

from src.data import (
    scrape_track_history, scrape_track_history_alltime,
    query_track_type_stats, query_season_stats,
)
from src.charts import track_history_bar, rating_vs_finish_scatter
from src.utils import format_display_df, safe_fillna


def render(*, track_name, track_type, series_id):
    """Render the Track History tab."""
    st.markdown(f"### {track_name} — Driver History")

    hist_view = st.radio("View", ["Recent Races", "All-Time", "By Track Type", "2026 Season"],
                         horizontal=True, label_visibility="collapsed")

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
