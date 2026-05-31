"""Tab: Season Points Standings (Driver, Manufacturer, Owner)."""

import pandas as pd
import streamlit as st

from src.components import section_header, interactive_drill_down_dataframe
from src.data import fetch_season_standings
from src.utils import format_display_df, safe_fillna


def render(*, series_id, series_name, selected_year):
    """Render the Season Standings tab."""
    section_header("Season Standings", f"{selected_year} {series_name}")

    with st.spinner("Loading season standings..."):
        standings = fetch_season_standings(series_id, selected_year)

    driver_df = standings.get("driver", pd.DataFrame())
    mfr_df = standings.get("manufacturer", pd.DataFrame())
    owner_df = standings.get("owner", pd.DataFrame())
    races_df = standings.get("races", pd.DataFrame())

    if driver_df.empty:
        st.info(f"No completed points races found for {selected_year}.")
        return

    n_races = races_df["Race"].nunique() if not races_df.empty else 0
    st.caption(f"{n_races} races completed")

    view = st.radio("Standings", ["Driver", "Manufacturer", "Owner"],
                    horizontal=True, label_visibility="collapsed", key="standings_view")

    if view == "Driver":
        _render_driver_standings(driver_df, races_df,
                                  series_id=series_id, season=selected_year)
    elif view == "Manufacturer":
        _render_manufacturer_standings(mfr_df)
    elif view == "Owner":
        _render_owner_standings(owner_df)


def _render_driver_standings(driver_df, races_df, series_id=None, season=None):
    """Render driver points standings with race-by-race detail."""
    if driver_df.empty:
        st.info("No driver standings data available.")
        return

    display = format_display_df(driver_df)
    if series_id is not None and season is not None:
        st.caption(f"Click any driver row for race-by-race history across the {season} season")
        interactive_drill_down_dataframe(
            safe_fillna(display),
            key=f"stand_drv_{series_id}_{season}",
            series_id=series_id, season=season,
            width="stretch", hide_index=True, height=600,
        )
    else:
        st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=600)

    # Points progression chart
    if not races_df.empty:
        _render_points_progression(races_df, driver_df)

    # Race-by-race breakdown for selected driver
    if not races_df.empty:
        st.divider()
        drivers = driver_df["Driver"].tolist()
        selected = st.selectbox("Driver Detail", drivers, key="standings_driver_detail")
        if selected:
            drv_races = races_df[races_df["Driver"] == selected].copy()
            if not drv_races.empty:
                detail_cols = ["Race", "Track", "Start", "Finish", "Points",
                               "Stage Pts", "Playoff Pts", "Laps Led", "Status"]
                avail = [c for c in detail_cols if c in drv_races.columns]
                detail = format_display_df(drv_races[avail])
                st.dataframe(safe_fillna(detail), width="stretch", hide_index=True)


def _render_manufacturer_standings(mfr_df):
    """Render manufacturer points standings."""
    if mfr_df.empty:
        st.info("No manufacturer standings data available.")
        return

    # Manufacturer points are sum of all driver points per make — show aggregate
    display = format_display_df(mfr_df)
    st.dataframe(safe_fillna(display), width="stretch", hide_index=True)

    # Bar chart
    if len(mfr_df) > 1:
        import plotly.express as px
        fig = px.bar(
            mfr_df.sort_values("Points", ascending=True),
            x="Points", y="Manufacturer", orientation="h",
            color="Manufacturer",
            color_discrete_map={"Toyota": "#D32F2F", "Ford": "#1565C0",
                                "Chevrolet": "#F9A825", "Honda": "#E0E0E0"},
            text="Points",
        )
        fig.update_layout(
            height=250, showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="", xaxis_title="Total Points",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")


def _render_owner_standings(owner_df):
    """Render owner points standings."""
    if owner_df.empty:
        st.info("No owner standings data available.")
        return

    display = format_display_df(owner_df)
    st.dataframe(safe_fillna(display), width="stretch", hide_index=True, height=600)


def _render_points_progression(races_df, driver_df):
    """Line chart showing cumulative points by race for top drivers."""
    import plotly.graph_objects as go
    from src.utils import short_name

    # Get top 10 drivers by points
    top_drivers = driver_df.head(10)["Driver"].tolist()

    # Build cumulative points per race
    race_order = races_df.drop_duplicates("Race")["Race"].tolist()
    # Deduplicate race order (preserve first occurrence order)
    seen = set()
    unique_races = []
    for r in race_order:
        if r not in seen:
            seen.add(r)
            unique_races.append(r)

    fig = go.Figure()
    all_names = top_drivers
    for driver in top_drivers:
        drv_data = races_df[races_df["Driver"] == driver]
        cum_pts = []
        running = 0
        for race in unique_races:
            race_pts = drv_data[drv_data["Race"] == race]["Points"].sum()
            running += race_pts
            cum_pts.append(running)

        # Shortened race names for x-axis
        short_races = [r.split(" at ")[0].split(" presented")[0][:20] for r in unique_races]

        fig.add_trace(go.Scatter(
            x=short_races, y=cum_pts,
            mode="lines+markers",
            name=short_name(driver, all_names),
            hovertemplate=f"{driver}<br>%{{x}}<br>Points: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Points Progression (Top 10)",
        xaxis_title="", yaxis_title="Cumulative Points",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
