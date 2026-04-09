"""Tab 3: Track History."""

import streamlit as st

from src.config import TRACK_TYPE_MAP, TRACK_TYPE_PARENT
from src.data import (
    scrape_track_history, scrape_track_history_alltime,
    query_track_type_stats, query_season_stats, query_db_track_history,
)
from src.charts import track_history_bar, rating_vs_finish_scatter, arp_vs_finish_scatter
from src.utils import format_display_df, safe_fillna


# Track type badge colors
TRACK_TYPE_BADGES = {
    "superspeedway": "🔴",
    "intermediate": "🟡",
    "intermediate_worn": "🟠",
    "short": "🟢",
    "short_concrete": "🟣",
    "road": "🔵",
    "dirt": "🟤",
}

# Human-readable descriptions for track types
TRACK_TYPE_DESCRIPTIONS = {
    "superspeedway": "Superspeedway — Daytona, Atlanta, Talladega, Indianapolis",
    "road": "Road Course — COTA, Sonoma, Watkins Glen, Chicago, Charlotte Roval, etc.",
    "short": "Short Track — Phoenix, Martinsville, Richmond, Iowa, New Hampshire, etc.",
    "short_concrete": "Short Concrete — Bristol, Dover",
    "intermediate": "Intermediate — Las Vegas, Kansas, Charlotte, Texas, Nashville, Michigan, Pocono, etc.",
    "intermediate_worn": "Intermediate Worn — Darlington, Homestead (high tire wear)",
}


def _format_type_label(t):
    """Human-readable label for track type options."""
    if t == "This Track":
        return t
    if t.startswith("All "):
        return t
    # Clean display name
    label = t.replace("_", " ").title()
    # Add track list for subtypes
    tracks = _tracks_for_type(t)
    if tracks and len(tracks) <= 4:
        track_shorts = [tn.split(" Motor")[0].split(" International")[0].split(" Raceway")[0].split(" Speedway")[0]
                        for tn in tracks]
        return f"{label} ({', '.join(track_shorts)})"
    return label


def _tracks_for_type(track_type: str) -> list:
    """Get track names belonging to a type or parent group."""
    if track_type.startswith("All "):
        parent = track_type.replace("All ", "").lower()
        return sorted(t for t, tt in TRACK_TYPE_MAP.items()
                      if TRACK_TYPE_PARENT.get(tt, tt) == parent)
    return sorted(t for t, tt in TRACK_TYPE_MAP.items() if tt == track_type)


def render(*, track_name, track_type, series_id):
    """Render the Track History tab."""
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    badge = TRACK_TYPE_BADGES.get(track_type, TRACK_TYPE_BADGES.get(parent_type, ""))
    display_type = track_type.replace("_", " ").title()
    st.markdown(f"### {track_name} — Driver History")
    st.caption(f"Track type: {badge} **{display_type}**")

    # Track type filter — show all types + parent groups (only if parent has subtypes)
    filter_cols = st.columns([2, 3])
    with filter_cols[0]:
        all_types = sorted(set(TRACK_TYPE_MAP.values()))
        # Only add "All X" parent groups if they actually have subtypes
        parent_groups = sorted(set(
            f"All {p.title()}" for p in set(TRACK_TYPE_PARENT.values())
            if sum(1 for v in TRACK_TYPE_PARENT.values() if v == p) > 1
        ))
        type_options = ["This Track"] + all_types + parent_groups
        type_filter = st.selectbox("Track Type Filter", type_options,
                                    key="th_type_filter", label_visibility="collapsed",
                                    format_func=_format_type_label,
                                    help="Filter to show stats for a specific track type")
    with filter_cols[1]:
        # Show which tracks are in the selected type
        if type_filter != "This Track":
            desc_tracks = _tracks_for_type(type_filter)
            if desc_tracks:
                st.caption(f"**{_format_type_label(type_filter)}**: {', '.join(desc_tracks)}")

    hist_view = st.radio("View", ["Recent Races", "All-Time", "By Track Type", "2026 Season"],
                         horizontal=True, label_visibility="collapsed")

    if type_filter != "This Track":
        # Show aggregated data for selected track type across all views
        _render_track_type_filtered(type_filter, hist_view, series_id)
        return

    if hist_view == "Recent Races":
        # Use DB data (Next Gen 2022+) for clean track history with ARP
        with st.spinner(f"Loading recent history at {track_name}..."):
            hist_df = query_db_track_history(track_name, series_id, min_season=2022)
        if hist_df.empty:
            # Fall back to scraper if DB has no data
            hist_df = scrape_track_history(track_name, series_id)
        if not hist_df.empty:
            st.caption(f"Next Gen era (2022+) — {track_name}")
            display = format_display_df(hist_df)
            st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)

            fig = track_history_bar(hist_df, track_name)
            if fig:
                st.plotly_chart(fig, width="stretch")

            # ARP vs Avg Finish scatter — shows wreck luck
            arp_fig = arp_vs_finish_scatter(hist_df, track_name)
            if arp_fig:
                st.plotly_chart(arp_fig, width="stretch")

            fig2 = rating_vs_finish_scatter(hist_df, track_name)
            if fig2:
                st.plotly_chart(fig2, width="stretch")
        else:
            st.info(f"No data found for {track_name}")

    elif hist_view == "All-Time":
        with st.spinner(f"Loading all-time history at {track_name}..."):
            alltime_df = scrape_track_history_alltime(track_name, series_id)
        if not alltime_df.empty:
            st.caption(f"Source: driveraverages.com — All-time at {track_name}")
            display = format_display_df(alltime_df)
            st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
        else:
            st.info(f"No all-time data found for {track_name}")

    elif hist_view == "By Track Type":
        # Show stats for this track's type from database
        type_tracks = _tracks_for_type(track_type)
        parent_tracks = _tracks_for_type(f"All {parent_type.title()}")
        display_type = track_type.replace("_", " ").title()
        st.caption(f"Track type: **{display_type}** — stats from database")
        if type_tracks:
            st.caption(f"Includes: {', '.join(type_tracks)}")
        tt_df = query_track_type_stats(track_type)
        if not tt_df.empty:
            display = format_display_df(tt_df)
            st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
        else:
            # Try parent type as fallback
            if track_type != parent_type:
                st.caption(f"No data for subtype '{display_type}' — showing parent type '{parent_type.title()}'")
                parent_df = query_track_type_stats(parent_type)
                if not parent_df.empty:
                    display = format_display_df(parent_df)
                    st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
                else:
                    st.info(f"No track-type data in database. Run `python refresh_data.py` to populate race data.")
            else:
                st.info(f"No track-type data in database. Run `python refresh_data.py` to populate race data.")

    elif hist_view == "2026 Season":
        st.caption(f"Aggregated from 2026 races at {track_name}")
        season_df = query_season_stats(track_name=track_name, season=2026,
                                       series_id=series_id)
        if not season_df.empty:
            display = format_display_df(season_df)
            st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
        else:
            st.info(f"No 2026 data at {track_name}. Data appears after races are completed and synced.")


def _render_track_type_filtered(track_type_filter, hist_view, series_id):
    """Render filtered view for a specific track type across all tracks."""
    # Get all tracks of this type
    # Handle "All Short", "All Intermediate" etc. parent type filters
    if track_type_filter.startswith("All "):
        parent = track_type_filter.replace("All ", "").lower()
        type_tracks = [t for t, tt in TRACK_TYPE_MAP.items()
                        if TRACK_TYPE_PARENT.get(tt, tt) == parent]
    else:
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
                st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
            else:
                st.dataframe(safe_fillna(format_display_df(combined)),
                             width="stretch", hide_index=True, height=550)
        else:
            st.info(f"No data found for {track_type_filter} tracks.")

    elif hist_view == "By Track Type":
        desc_tracks = _tracks_for_type(track_type_filter)
        if desc_tracks:
            st.caption(f"**{_format_type_label(track_type_filter)}**: {', '.join(desc_tracks)}")
        tt_df = query_track_type_stats(track_type_filter)
        if not tt_df.empty:
            display = format_display_df(tt_df)
            st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
        else:
            st.info(f"No data for {_format_type_label(track_type_filter)} in database. "
                    f"Run `python refresh_data.py` to populate race data.")

    elif hist_view == "2026 Season":
        st.caption(f"Season data filtered to {_format_type_label(track_type_filter)} tracks (2026 only)")
        # Use track type query to filter season data to this type AND 2026 only
        season_df = query_track_type_stats(track_type_filter, season=2026)
        if not season_df.empty:
            display = format_display_df(season_df)
            st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=550)
        else:
            st.info(f"No season data for {_format_type_label(track_type_filter)}. "
                    f"Run `python refresh_data.py` to populate race data.")
