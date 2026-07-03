"""Tab 3: Track History."""

from datetime import datetime as _dt

import pandas as pd
import streamlit as st

from src.config import (
    TRACK_TYPE_MAP, TRACK_TYPE_PARENT, CONCRETE_GROUP_LABEL, resolve_track_group,
)
from src.data import (
    query_track_type_stats, query_season_stats, query_db_track_history,
)


def _current_season() -> int:
    """Active season label for the 'YYYY Season' view tab.

    Pre-October: stay on the current calendar year (we're mid-season).
    October+: start surfacing next year (NASCAR posts the new schedule).
    """
    _t = _dt.now()
    return _t.year + 1 if _t.month >= 10 else _t.year
from src.charts import track_history_bar, arp_vs_finish_scatter, finish_distribution_box
from src.utils import format_display_df, safe_fillna
from src.components import section_header, interactive_drill_down_dataframe


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
    """Get track names belonging to a type or parent group (incl. All Concrete)."""
    return resolve_track_group(track_type)


def render(*, track_name, track_type, series_id, entry_list_df=None):
    """Render the Track History tab."""
    parent_type = TRACK_TYPE_PARENT.get(track_type, track_type)
    badge = TRACK_TYPE_BADGES.get(track_type, TRACK_TYPE_BADGES.get(parent_type, ""))
    display_type = track_type.replace("_", " ").title()
    section_header(f"{track_name}", "Driver History")
    st.caption(f"Track type: {badge} **{display_type}**")

    # Build active-driver set from the entry list. Used to highlight (or
    # filter to) drivers actually in the upcoming race.
    active_drivers = set()
    if entry_list_df is not None and not entry_list_df.empty and "Driver" in entry_list_df.columns:
        active_drivers = set(entry_list_df["Driver"].dropna().astype(str).str.strip().tolist())

    # Track type filter — show all types + parent groups (only if parent has subtypes)
    filter_cols = st.columns([2, 3])
    with filter_cols[0]:
        all_types = sorted(set(TRACK_TYPE_MAP.values()))
        # Only add "All X" parent groups if they actually have subtypes
        parent_groups = sorted(set(
            f"All {p.title()}" for p in set(TRACK_TYPE_PARENT.values())
            if sum(1 for v in TRACK_TYPE_PARENT.values() if v == p) > 1
        ))
        type_options = ["This Track"] + all_types + parent_groups + [CONCRETE_GROUP_LABEL]
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

    _cy = _current_season()
    _season_view_label = f"{_cy} Season"
    # Averages (Next Gen era / current season) or a SINGLE race's full-field
    # results. The single-race picker spans every season in the DB — the
    # global race selector only covers one year, which made navigating to
    # "the 2024 race here" painful.
    hist_view = st.radio("View", ["Next Gen (2022+)", _season_view_label,
                                  "Single Race"],
                         horizontal=True, label_visibility="collapsed")

    if hist_view == "Single Race":
        _scope_tracks = ([track_name] if type_filter == "This Track"
                         else _tracks_for_type(type_filter))
        _render_single_race_results(_scope_tracks, series_id)
        return

    # Active-driver controls — only visible when we have an entry list and
    # an aggregated (multi-driver) view is being shown.
    show_active_only = False
    highlight_active = True
    if active_drivers:
        ctrl_cols = st.columns([1, 1, 4])
        with ctrl_cols[0]:
            show_active_only = st.checkbox(
                "Active drivers only", value=True, key="th_active_only",
                help="Filter the table to drivers entered in the current race "
                     "(on by default — untick to include retired/part-time drivers)",
            )
        with ctrl_cols[1]:
            highlight_active = st.checkbox(
                "Highlight active", value=True, key="th_highlight_active",
                help="Bold + colored row background for drivers in the current race",
            )

    if type_filter != "This Track":
        # Show aggregated data for selected track type across all views
        _render_track_type_filtered(
            type_filter, hist_view, series_id,
            active_drivers=active_drivers,
            show_active_only=show_active_only,
            highlight_active=highlight_active,
        )
        return

    if hist_view == "Next Gen (2022+)":
        # Use DB data (Next Gen 2022+) for clean track history with ARP
        with st.spinner(f"Loading history at {track_name}..."):
            hist_df = query_db_track_history(track_name, series_id, min_season=2022)
        if not hist_df.empty:
            type_tracks = _tracks_for_type(track_type)
            st.caption(f"Track type: **{display_type}** — stats from database (current series)  •  Click any driver row for race-by-race history")
            if type_tracks:
                st.caption(f"Includes: {', '.join(type_tracks)}")
            display_df = _filter_and_sort_history(
                hist_df, active_drivers=active_drivers,
                show_active_only=show_active_only,
            )
            display = format_display_df(display_df)
            _render_history_table(
                display, key=f"th_nextgen_{series_id}_{track_name}",
                series_id=series_id, track_name=track_name,
                active_drivers=active_drivers if highlight_active else set(),
            )

            # Charts respect the same Active-drivers toggle as the table —
            # they used to chart the raw (unfiltered) history regardless.
            chart_df = display_df if show_active_only else hist_df
            fig = track_history_bar(chart_df, track_name)
            if fig:
                st.plotly_chart(fig, width="stretch")

            # ARP vs Avg Finish scatter — shows wreck luck
            arp_fig = arp_vs_finish_scatter(chart_df, track_name,
                                            series_id=series_id)
            if arp_fig:
                st.plotly_chart(arp_fig, width="stretch")

            # Finish distribution box plot — shows consistency vs boom/bust
            box_fig = finish_distribution_box(
                track_name, series_id,
                active_drivers=active_drivers if show_active_only else None)
            if box_fig:
                st.plotly_chart(box_fig, width="stretch")
        else:
            st.info(f"No data found for {track_name}. Run `python scripts/refresh_data.py --all` to populate.")

    elif hist_view == _season_view_label:
        st.caption(f"Aggregated from {_cy} races at {track_name}  •  Click any driver row for race-by-race history")
        season_df = query_season_stats(track_name=track_name, season=_cy,
                                       series_id=series_id)
        if not season_df.empty:
            display_df = _filter_and_sort_history(
                season_df, active_drivers=active_drivers,
                show_active_only=show_active_only,
            )
            display = format_display_df(display_df)
            _render_history_table(
                display, key=f"th_season_{_cy}_{series_id}_{track_name}",
                series_id=series_id, track_name=track_name,
                active_drivers=active_drivers if highlight_active else set(),
            )
        else:
            st.info(f"No {_cy} data at {track_name}. Data appears after races are completed and synced.")


def render_concrete_tab(*, series_id, entry_list_df=None):
    """Dedicated 'Concrete' tab, shown only on concrete race weeks.

    Aggregates the All-Concrete surface group (Nashville + Dover + Bristol) by
    reusing the same renderer the track-type filter uses. Concrete is a SURFACE
    group independent of track size, so this complements (does not replace) the
    intermediate / short-concrete views available elsewhere.
    """
    from src.config import CONCRETE_GROUP_LABEL
    section_header("Concrete Tracks", "Nashville · Dover · Bristol")
    st.caption(
        "Concrete is a racing **surface** (not a track size). Dirty air punishes "
        "following cars and laps led concentrate in the front-runners — so these "
        "tracks race alike regardless of length. Nashville still appears under "
        "**All Intermediate** too."
    )
    active_drivers = set()
    if entry_list_df is not None and not entry_list_df.empty and "Driver" in entry_list_df.columns:
        active_drivers = set(entry_list_df["Driver"].dropna().astype(str).str.strip().tolist())

    _cy = _current_season()
    hist_view = st.radio("View", ["Next Gen (2022+)", f"{_cy} Season",
                                  "Single Race"],
                         horizontal=True, label_visibility="collapsed",
                         key="concrete_hist_view")
    if hist_view == "Single Race":
        _render_single_race_results(
            _tracks_for_type(CONCRETE_GROUP_LABEL), series_id)
        return
    _render_track_type_filtered(
        CONCRETE_GROUP_LABEL, hist_view, series_id,
        active_drivers=active_drivers, highlight_active=bool(active_drivers),
    )


def _render_single_race_results(track_names, series_id):
    """Full-field results for ONE past race at this track (or track group).

    The picker lists every completed race in the DB across all seasons —
    no need to change the app's global year selector to look backward.
    """
    from src.data import query_track_race_list, query_race_field_results

    track_names = [t for t in (track_names or []) if t]
    if not track_names:
        st.info("No track selected.")
        return
    races = query_track_race_list(track_names, series_id)
    if not races:
        st.info("No completed races stored for this scope yet. "
                "Results appear after a race runs and the data refresh syncs it.")
        return

    # Year selector first, then races within that year — keeps the race list
    # short instead of one giant all-seasons dropdown.
    seasons = sorted({r["season"] for r in races}, reverse=True)
    pick_cols = st.columns([1, 3, 2])
    with pick_cols[0]:
        year = st.selectbox("Year", seasons, index=0, key="th_single_race_year")
    year_races = [r for r in races if r["season"] == year]

    # Label by TRACK, not the sponsor-laden race name. Tracks that race more
    # than once a year (Talladega, Kansas, ...) get a date suffix so both
    # visits stay distinguishable.
    from collections import Counter

    def _exhibition_tag(race_name):
        """Short tag for non-points races so they read as what they are."""
        n = (race_name or "").lower()
        if "duel" in n:
            return "Duel"
        if "clash" in n:
            return "Clash"
        if "all-star" in n or "all star" in n or "nascar open" in n:
            return "All-Star"
        return ""

    _track_counts = Counter(r["track"] for r in year_races)
    labels = []
    _seen = Counter()
    for r in year_races:
        lbl = r["track"]
        _tag = _exhibition_tag(r["race_name"])
        if _tag:
            lbl += f" — {_tag}"
        if _track_counts[r["track"]] > 1:
            _mmdd = r["race_date"][5:] if len(r["race_date"]) >= 10 else r["race_date"]
            lbl += f" ({_mmdd})"
        # Same track, tag AND date (the two Daytona Duels): number them so
        # both stay selectable — duplicate labels make .index() unreachable.
        _seen[lbl] += 1
        if _seen[lbl] > 1:
            lbl += f" · {_seen[lbl]}"
        labels.append(lbl)
    with pick_cols[1]:
        # Key includes the year so the widget resets cleanly on year change.
        sel = st.selectbox("Race", labels, index=0,
                           key=f"th_single_race_pick_{year}")
    race = year_races[labels.index(sel)]

    data = query_race_field_results(race["db_id"])
    rows = data.get("rows") if isinstance(data, dict) else None
    if not rows:
        st.info("No stored results for this race.")
        return

    _exh_note = ("  •  ⚠️ Exhibition race — excluded from all history averages"
                 if _exhibition_tag(data["race_name"]) else "")
    st.caption(f"**{data['race_name']}** — {data['track']}, {data['date']}  •  "
               f"{len(rows)} finishers  •  Click any driver row for race-by-race history"
               f"{_exh_note}")
    df = pd.DataFrame(rows)
    show = [c for c in ["Finish", "Driver", "Car", "Team", "Mfr", "Start",
                        "Laps Led", "Fast Laps", "Avg Run", "DK Pts", "Status"]
            if c in df.columns]
    disp = format_display_df(df[show])

    from src.components import apply_car_badges, CAR_BADGE_ROW_HEIGHT
    disp, _badge_cfg = apply_car_badges(disp, series_id)
    _badge_kw = ({"column_config": {"Car": _badge_cfg},
                  "row_height": CAR_BADGE_ROW_HEIGHT} if _badge_cfg else {})
    interactive_drill_down_dataframe(
        safe_fillna(disp),
        key=f"th_single_{series_id}_{race['db_id']}",
        series_id=series_id, track_name=data["track"],
        width="stretch", hide_index=True, height=560,
        **_badge_kw,
    )


def _filter_and_sort_history(df: pd.DataFrame, *, active_drivers: set,
                              show_active_only: bool) -> pd.DataFrame:
    """Optionally filter to active drivers, then sort by best Avg Finish."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if show_active_only and active_drivers and "Driver" in out.columns:
        out = out[out["Driver"].astype(str).str.strip().isin(active_drivers)]
    if "Avg Finish" in out.columns:
        out["Avg Finish"] = pd.to_numeric(out["Avg Finish"], errors="coerce")
        out = out.sort_values("Avg Finish", na_position="last").reset_index(drop=True)
    return out


def _build_history_format_map(df: pd.DataFrame) -> dict:
    """Per-column format strings for the track-history table.

    Used when wrapping the DataFrame in a Styler — Streamlit's plain
    st.dataframe auto-formats floats sensibly, but a Styler renders them
    with full precision (6 decimals) unless we explicitly format. This
    mirrors the rules used by src.utils.format_display_df.
    """
    INT_COLS = {"Races", "Wins", "Top 5", "Top 10", "Top 20",
                "Laps Led", "Fast Laps", "Fastest Laps", "DNF"}
    ONE_DEC_COLS = {"Avg Finish", "Avg Start", "Avg Run Pos", "Avg Rating",
                    "Avg Green Rank",
                    "Avg DFS", "Best DFS", "Worst DFS",
                    "Avg Laps Led", "Avg Fastest Laps", "Avg Fast Laps",
                    "Avg DK", "Best DK", "Worst DK"}
    TWO_DEC_COLS = {"Value", "DFS Value"}

    fmt = {}
    for c in df.columns:
        col_name = c[-1] if isinstance(c, tuple) else c
        if col_name in INT_COLS:
            fmt[c] = "{:.0f}"
        elif col_name in ONE_DEC_COLS:
            fmt[c] = "{:.1f}"
        elif col_name in TWO_DEC_COLS:
            fmt[c] = "{:.2f}"
    return fmt


def _render_history_table(display_df: pd.DataFrame, *, key: str, series_id: int,
                           track_name: str = None, track_type: str = None,
                           active_drivers: set):
    """Render a track-history table with optional active-driver highlighting.

    `active_drivers` should be empty when highlighting is disabled.
    """
    fillna_df = safe_fillna(display_df)
    if not active_drivers or "Driver" not in fillna_df.columns:
        interactive_drill_down_dataframe(
            fillna_df, key=key,
            series_id=series_id, track_name=track_name, track_type=track_type,
            width="stretch", hide_index=True, height=550,
        )
        return

    # Apply a row-level highlight for active drivers (subtle blue tint).
    def _row_style(row):
        drv = str(row.get("Driver", "")).strip()
        if drv in active_drivers:
            return ["background-color: rgba(56, 189, 248, 0.10); "
                    "color: #e0f2fe; font-weight: 600"] * len(row)
        return [""] * len(row)
    styled = fillna_df.style.apply(_row_style, axis=1)
    # Streamlit doesn't auto-format floats inside a Styler — without this
    # every numeric cell renders with 6 trailing zeros (5.000000).
    fmt_map = _build_history_format_map(fillna_df)
    if fmt_map:
        styled = styled.format(fmt_map, na_rep="")
    interactive_drill_down_dataframe(
        styled, key=key,
        series_id=series_id, track_name=track_name, track_type=track_type,
        width="stretch", hide_index=True, height=550,
    )


def _render_track_type_filtered(track_type_filter, hist_view, series_id, *,
                                 active_drivers=None, show_active_only=False,
                                 highlight_active=True):
    """Render filtered view when the user has picked a track-type filter."""
    _cy = _current_season()
    _season_view_label = f"{_cy} Season"
    active_drivers = active_drivers or set()
    # Resolve a drill-down track type — strip "All " prefix for the popup.
    _drill_type = (track_type_filter.replace("All ", "").lower()
                   if track_type_filter.startswith("All ")
                   else track_type_filter)

    desc_tracks = _tracks_for_type(track_type_filter)
    if desc_tracks:
        st.caption(f"**{_format_type_label(track_type_filter)}**: {', '.join(desc_tracks)}")

    if hist_view == "Next Gen (2022+)":
        with st.spinner(f"Loading {track_type_filter} track data..."):
            tt_df = query_track_type_stats(track_type_filter, series_id=series_id)
        if not tt_df.empty:
            st.caption(f"Next Gen era (2022+) — {_format_type_label(track_type_filter)} tracks  •  Click any driver row for race-by-race history")
            display_df = _filter_and_sort_history(
                tt_df, active_drivers=active_drivers,
                show_active_only=show_active_only,
            )
            display = format_display_df(display_df)
            _render_history_table(
                display, key=f"th_filt_{series_id}_{track_type_filter}",
                series_id=series_id, track_type=_drill_type,
                active_drivers=active_drivers if highlight_active else set(),
            )
        else:
            st.info(f"No data found for {_format_type_label(track_type_filter)}. "
                    f"Run `python scripts/refresh_data.py --all` to populate.")

    elif hist_view == _season_view_label:
        st.caption(f"Season data filtered to {_format_type_label(track_type_filter)} tracks ({_cy} only)")
        season_df = query_track_type_stats(track_type_filter, season=_cy, series_id=series_id)
        if not season_df.empty:
            display_df = _filter_and_sort_history(
                season_df, active_drivers=active_drivers,
                show_active_only=show_active_only,
            )
            display = format_display_df(display_df)
            _render_history_table(
                display, key=f"th_filtseason_{_cy}_{series_id}_{track_type_filter}",
                series_id=series_id, track_type=_drill_type,
                active_drivers=active_drivers if highlight_active else set(),
            )
        else:
            st.info(f"No season data for {_format_type_label(track_type_filter)}. "
                    f"Run `python scripts/refresh_data.py` to populate race data.")
