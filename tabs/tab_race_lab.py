"""Tab: Race Lab — green-flag pace & per-stage breakdown, single race OR
aggregated across every race at a track / track-type.

Scope toggle:
  • This Track       — averages across all stored races at the selected track
  • This Track Type  — averages across all races of that track-type group
  • Single Race      — one race's full detail (the original view)

Aggregate views use RANKS (comparable across tracks); single-track / single-race
views also show actual lap seconds. The lap-by-lap "Speed Over Time" overlay now
lives in the Data tab's Charts section.
"""
import pandas as pd
import streamlit as st

from src.components import section_header, interactive_drill_down_dataframe
from src.data import (query_race_stage_breakdown, query_race_field_results,
                      query_race_run_pace, query_track_stage_aggregate,
                      query_track_run_pace_aggregate)
from src.utils import safe_fillna, format_display_df


def _resolve_db_id(api_race_id, series_id):
    from tabs.tab_projections import _resolve_db_race_id
    try:
        return _resolve_db_race_id(api_race_id, series_id)
    except Exception:
        return None


def _df_with_drill(styled, *, key, series_id, track_name=None, track_type=None,
                   height=620):
    """Render a styled table with click-a-row → driver-history popup. The popup
    opens in the table's scope (this track, or this track-type) and ranks the
    clicked driver against the other drivers shown in the table."""
    interactive_drill_down_dataframe(
        styled, key=key, series_id=series_id,
        track_name=track_name, track_type=track_type,
        width="stretch", hide_index=True, height=height)


def render(*, completed_races, series_id, selected_year, series_name="Cup",
           track_name=None, track_type=None, selected_race=None):
    """Render the Race Lab tab."""
    section_header("Race Lab", "Green-flag pace & per-stage breakdown")

    # Scope: default to the selected/upcoming track so the tab opens on something
    # immediately useful (all results at this track), with a track-type and a
    # single-race option.
    scope = st.radio("Scope", ["This Track", "This Track Type", "Single Race"],
                     horizontal=True, key="racelab_scope")

    if scope == "Single Race":
        _render_single_race(completed_races, series_id, selected_year)
    else:
        _render_aggregate(scope, track_name, track_type, series_id)


# ───────────────────────── aggregate (multi-race) ─────────────────────────────
def _render_aggregate(scope, track_name, track_type, series_id):
    is_track = (scope == "This Track")
    if is_track and not track_name:
        st.info("No track is selected for the current race.")
        return
    if (not is_track) and not track_type:
        st.info("No track type is available for the current race.")
        return

    tkw = dict(track_name=track_name) if is_track else dict(track_type=track_type)
    scope_label = track_name if is_track else track_type.replace("_", " ").title()

    # First pass (all years) to discover the available seasons + base data.
    stage_all = query_track_stage_aggregate(series_id, **tkw)
    run_all = query_track_run_pace_aggregate(series_id, **tkw)
    years_avail = sorted(set(stage_all.get("years", []) + run_all.get("years", [])),
                         reverse=True)
    if not years_avail:
        st.info(f"No stage / run-pace data stored for **{scope_label}** yet. "
                "Run `python scripts/refresh_data.py` to populate it.")
        return

    sel_years = st.multiselect("Seasons", years_avail, default=years_avail,
                               key="racelab_agg_years")
    if not sel_years:
        st.info("Select at least one season.")
        return
    if set(sel_years) != set(years_avail):
        stage_agg = query_track_stage_aggregate(series_id, years=sel_years, **tkw)
        run_agg = query_track_run_pace_aggregate(series_id, years=sel_years, **tkw)
    else:
        stage_agg, run_agg = stage_all, run_all

    n = max(stage_agg.get("n_races", 0), run_agg.get("n_races", 0))
    note = ("Raw lap seconds shown (single track)." if is_track else
            "Ranks only — lap seconds aren't comparable across different tracks.")
    st.caption(
        f"**{scope_label}** — averaged across **{n} race(s)** "
        f"({', '.join(str(y) for y in sorted(sel_years))}). Values are per-driver "
        f"averages; ranks are 1 = best in field. {note}")

    view = st.radio("View", ["Per-Stage Breakdown", "Green Speed Summary",
                             "Long Run & Restart"],
                    horizontal=True, label_visibility="collapsed",
                    key="racelab_agg_view")

    pop_track = track_name if is_track else None
    pop_ttype = None if is_track else track_type
    if view == "Per-Stage Breakdown":
        if stage_agg["rows"]:
            _render_stage_table(stage_agg["rows"], stage_agg["stages"], agg=True,
                                series_id=series_id, track_name=pop_track,
                                track_type=pop_ttype, key="rl_stage_agg")
        else:
            st.info("No per-stage data stored for this scope.")
    elif view == "Green Speed Summary":
        if stage_agg["rows"]:
            _render_green_summary(stage_agg["rows"], stage_agg["stages"], agg=True,
                                  series_id=series_id, track_name=pop_track,
                                  track_type=pop_ttype, key="rl_green_agg")
        else:
            st.info("No per-stage data stored for this scope.")
    else:
        _render_run_pace_agg(run_agg, single_track=is_track, series_id=series_id,
                             track_name=pop_track, track_type=pop_ttype)


# ───────────────────────── single race (full detail) ─────────────────────────
def _render_single_race(completed_races, series_id, selected_year):
    if not completed_races:
        st.info("No completed races available for this series/year yet — "
                "single-race breakdowns appear after races run. Use the "
                "'This Track' or 'This Track Type' scope for pre-race research.")
        return

    # Race picker (most recent first).
    labels, label_to_race = [], {}
    for _, race in completed_races:
        track = race.get("track_name", "")
        name = race.get("race_name", "")
        date = (race.get("race_date", "") or "")[:10]
        lbl = f"{date} — {track}: {name}"
        labels.append(lbl)
        label_to_race[lbl] = race
    labels = list(reversed(labels))

    selected = st.selectbox("Race", labels, index=0, key="racelab_pick")
    race = label_to_race[selected]
    db_id = _resolve_db_id(race.get("race_id"), series_id)
    if db_id is None:
        st.info("This race isn't in the database yet. Run a data refresh to populate it.")
        return

    breakdown = query_race_stage_breakdown(db_id)
    stages = breakdown.get("stages", [])
    rows = breakdown.get("rows", [])

    _track = race.get("track_name")
    if not rows:
        st.warning(
            "No per-stage data for this race yet. Stage metrics are computed from "
            "the lap-times feed and backfilled — run `python scripts/refresh_data.py` to "
            "populate recent races."
        )
        _render_field_fallback(db_id, series_id=series_id, track_name=_track)
        return

    st.caption(
        f"**{len(rows)} drivers · {len(stages)} stages.**  "
        "Green Speed = median green-flag lap speed (mph); Rank = 1 is fastest in "
        "that stage. AvgPos = average running position. Chg = positions gained "
        "(＋) or lost (−) within the stage. Pts = NASCAR stage points."
    )

    view = st.radio("View",
                    ["Per-Stage Breakdown", "Green Speed Summary",
                     "Long Run & Restart"],
                    horizontal=True, label_visibility="collapsed",
                    key="racelab_view")

    if view == "Per-Stage Breakdown":
        _render_stage_table(rows, stages, series_id=series_id, track_name=_track,
                            key="rl_stage_single")
    elif view == "Green Speed Summary":
        _render_green_summary(rows, stages, series_id=series_id, track_name=_track,
                              key="rl_green_single")
    else:
        _render_run_pace(race, series_id, selected_year)


# ───────────────────────────── shared helpers ────────────────────────────────
def _heat_low_good(v, lo=1.0, hi=36.0):
    """Background color for a 'lower is better' value (green=good, red=bad),
    scaled between lo (best) and hi (worst). Returns a CSS string or ''."""
    if pd.isna(v):
        return ""
    r = max(0.0, min(1.0, (float(v) - lo) / max(hi - lo, 1e-9)))
    rr = int(34 + r * (239 - 34)); gg = int(197 - r * (197 - 68))
    return f"background-color: rgba({rr},{gg},68,0.22)"


def _render_stage_table(rows, stages, agg=False, *, series_id=None,
                        track_name=None, track_type=None, key="rl_stage"):
    """Wide table: one row per driver, grouped columns per stage. `agg` switches
    to averaged columns (no Speed/Pts/Car; a Races count; 1-decimal numbers)."""
    df = pd.DataFrame(rows)
    id_cols = ["Finish", "Driver", "Races"] if agg else ["Finish", "Driver", "Car"]
    cols = list(id_cols)
    for s in stages:
        block = ([f"S{s} Rank", f"S{s} AvgPos", f"S{s} Chg"] if agg
                 else [f"S{s} Speed", f"S{s} Rank", f"S{s} AvgPos", f"S{s} Chg", f"S{s} Pts"])
        for c in block:
            if c in df.columns and df[c].notna().any():
                cols.append(c)
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    fmt = {}
    for c in disp.columns:
        if c in ("Driver", "Car"):
            continue
        if c == "Races":
            fmt[c] = "{:.0f}"
        elif agg:
            fmt[c] = "{:.1f}"   # every aggregate numeric is an average
        elif c.endswith("Speed") or c.endswith("AvgPos"):
            fmt[c] = "{:.1f}"
        else:
            fmt[c] = "{:.0f}"   # integer-valued (Finish, Rank, Chg, Pts)

    def _cell_style(col):
        name = col.name
        if name.endswith("Chg"):
            out = []
            for v in col:
                if pd.isna(v):
                    out.append("")
                elif v > 0:
                    out.append("color:#22c55e; font-weight:600")
                elif v < 0:
                    out.append("color:#ef4444; font-weight:600")
                else:
                    out.append("color:#94a3b8")
            return out
        if name == "Finish" or name.endswith("Rank") or name.endswith("AvgPos"):
            return [_heat_low_good(v) for v in col]
        if name.endswith("Pts"):
            return ["background-color: rgba(34,197,94,0.18)" if pd.notna(v) and v > 0
                    else "" for v in col]
        return ["" for _ in col]

    styled = disp.style.apply(_cell_style, axis=0).format(fmt, na_rep="—")
    _df_with_drill(styled, key=key, series_id=series_id,
                   track_name=track_name, track_type=track_type)


def _render_green_summary(rows, stages, agg=False, *, series_id=None,
                          track_name=None, track_type=None, key="rl_green"):
    """Compact table: each driver's overall + per-stage green-speed RANK, to see
    pace trajectory across the race(s) at a glance (lower = faster)."""
    df = pd.DataFrame(rows)
    id_cols = ["Finish", "Driver", "Races"] if agg else ["Finish", "Driver", "Car"]
    cols = id_cols + [f"S{s} Rank" for s in stages if f"S{s} Rank" in df.columns]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()
    disp = disp.rename(columns={f"S{s} Rank": f"Stage {s}" for s in stages})

    rank_cols = [f"Stage {s}" for s in stages if f"Stage {s}" in disp.columns]

    def _rank_heat(col):
        if col.name not in rank_cols and col.name != "Finish":
            return ["" for _ in col]
        return [_heat_low_good(v) for v in col]

    num_fmt = {}
    for c in disp.columns:
        if c in ("Driver", "Car"):
            continue
        num_fmt[c] = "{:.0f}" if (c == "Races" or not agg) else "{:.1f}"
    styled = disp.style.apply(_rank_heat, axis=0).format(num_fmt, na_rep="—")
    _df_with_drill(styled, key=key, series_id=series_id,
                   track_name=track_name, track_type=track_type)
    st.caption("Per-stage green-flag speed RANK (1 = fastest). Read left→right to "
               "see whether a driver got faster or faded as the race(s) went on. "
               "Click a row for that driver's history.")


def _render_run_pace(race, series_id, selected_year):
    """Single-race long-run (sustained green pace) + restart pace, per driver."""
    api_id = race.get("race_id")
    yr = (race.get("race_date", "") or "")[:4]
    try:
        yr = int(yr)
    except (TypeError, ValueError):
        yr = selected_year

    with st.spinner("Computing long-run & restart pace..."):
        data = query_race_run_pace(series_id, api_id, yr,
                                   _resolve_db_id(api_id, series_id))
    rows = data.get("rows", [])
    if not rows:
        st.info("Lap-by-lap data isn't available for this race, so long-run / "
                "restart pace can't be computed.")
        return

    st.caption(
        "**Long Run** = median green-flag lap time over runs of 10+ consecutive "
        "green laps (sustained pace / tire management). **Restart** = avg of the "
        "first 5 green laps after each restart (short-run speed). Pit laps "
        "excluded. Rank 1 = fastest in the field; lower time = faster."
    )

    df = pd.DataFrame(rows)
    cols = ["Driver", "Long Run (s)", "Long Run Rank", "LR Laps",
            "Restart (s)", "Restart Rank", "Restart Laps"]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    def _rank_heat(col):
        if not col.name.endswith("Rank"):
            return ["" for _ in col]
        return [_heat_low_good(v) for v in col]

    fmt = {}
    for c in disp.columns:
        if c.endswith("(s)"):
            fmt[c] = "{:.2f}"
        elif c != "Driver":
            fmt[c] = "{:.0f}"
    styled = disp.style.apply(_rank_heat, axis=0).format(fmt, na_rep="—")
    _df_with_drill(styled, key="rl_runpace_single", series_id=series_id,
                   track_name=race.get("track_name"))
    st.caption("Tip: a driver who's strong on Long Run but weak on Restart "
               "(or vice-versa) is a setup/strategy read for DFS. Click a row for "
               "that driver's history.")


def _render_run_pace_agg(run_agg, single_track, *, series_id=None,
                         track_name=None, track_type=None):
    """Aggregated long-run + restart pace across many races. Ranks are the
    average field rank; seconds shown only for a single track (comparable)."""
    rows = run_agg.get("rows", [])
    if not rows:
        st.info("No long-run / restart data stored for this scope yet. "
                "Run `python scripts/refresh_data.py` to populate it.")
        return
    df = pd.DataFrame(rows)
    if single_track:
        cols = ["Driver", "Long Run Rank", "Long Run (s)",
                "Restart Rank", "Restart (s)", "Races"]
    else:
        cols = ["Driver", "Long Run Rank", "Restart Rank", "Races"]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    def _rank_heat(col):
        if not col.name.endswith("Rank"):
            return ["" for _ in col]
        return [_heat_low_good(v) for v in col]

    fmt = {}
    for c in disp.columns:
        if c.endswith("(s)"):
            fmt[c] = "{:.2f}"
        elif c == "Races":
            fmt[c] = "{:.0f}"
        elif c.endswith("Rank"):
            fmt[c] = "{:.1f}"
    styled = disp.style.apply(_rank_heat, axis=0).format(fmt, na_rep="—")
    _df_with_drill(styled, key="rl_runpace_agg", series_id=series_id,
                   track_name=track_name, track_type=track_type)
    cap = ("Avg field rank across the selected races (1 = fastest). Seconds are "
           "the average of each race's median long-run / restart lap time."
           if single_track else
           "Avg field rank across the selected races (1 = fastest). A driver "
           "strong on Long Run but weak on Restart (or vice-versa) is a "
           "setup/strategy read for DFS.")
    st.caption(cap + " Click a row for that driver's history.")


def _render_field_fallback(db_id, *, series_id=None, track_name=None):
    """When stage data is missing, at least show the full-field results."""
    data = query_race_field_results(db_id)
    rows = data.get("rows") if isinstance(data, dict) else None
    if not rows:
        return
    df = pd.DataFrame(rows)
    show = [c for c in ["Finish", "Driver", "Car", "Team", "Start",
                        "Laps Led", "Fast Laps", "Avg Run", "Rating", "DK Pts"]
            if c in df.columns]
    _df_with_drill(safe_fillna(format_display_df(df[show])),
                   key="rl_field_fallback", series_id=series_id,
                   track_name=track_name, height=560)
