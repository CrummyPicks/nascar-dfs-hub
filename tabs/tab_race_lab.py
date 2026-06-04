"""Tab: Race Lab — deep per-race breakdown (green-flag pace + per-stage arc).

Surfaces the data that was previously buried in the driver popup: for any
completed race, the full field with green-flag speed and a stage-by-stage
breakdown (speed, avg running position, positions gained/lost, stage points) —
so you can see how the race developed and who was fast vs. who hung on.
"""
import pandas as pd
import streamlit as st

from src.components import section_header
from src.data import (query_race_stage_breakdown, query_race_field_results,
                      fetch_lap_times, query_race_run_pace)
from src.charts import race_speed_chart
from src.utils import safe_fillna, format_display_df


def _resolve_db_id(api_race_id, series_id):
    from tabs.tab_projections import _resolve_db_race_id
    try:
        return _resolve_db_race_id(api_race_id, series_id)
    except Exception:
        return None


def render(*, completed_races, series_id, selected_year, series_name="Cup"):
    """Render the Race Lab tab."""
    section_header("Race Lab", "Green-flag pace & per-stage breakdown")

    if not completed_races:
        st.info("No completed races available for this series/year.")
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

    if not rows:
        st.warning(
            "No per-stage data for this race yet. Stage metrics are computed from "
            "the lap-times feed and backfilled — run `python refresh_data.py` to "
            "populate recent races."
        )
        # Still show the full-field results + season-long green rank as a fallback.
        _render_field_fallback(db_id)
        return

    st.caption(
        f"**{len(rows)} drivers · {len(stages)} stages.**  "
        "Green Speed = median green-flag lap speed (mph); Rank = 1 is fastest in "
        "that stage. AvgPos = average running position. Chg = positions gained "
        "(＋) or lost (−) within the stage. Pts = NASCAR stage points."
    )

    view = st.radio("View",
                    ["Per-Stage Breakdown", "Green Speed Summary",
                     "Long Run & Restart", "Speed Over Time"],
                    horizontal=True, label_visibility="collapsed", key="racelab_view")

    if view == "Per-Stage Breakdown":
        _render_stage_table(rows, stages)
    elif view == "Green Speed Summary":
        _render_green_summary(rows, stages)
    elif view == "Long Run & Restart":
        _render_run_pace(race, series_id, selected_year)
    else:
        _render_speed_over_time(race, series_id, selected_year, rows)


def _heat_low_good(v, lo=1.0, hi=36.0):
    """Background color for a 'lower is better' value (green=good, red=bad),
    scaled between lo (best) and hi (worst). Returns a CSS string or ''."""
    if pd.isna(v):
        return ""
    r = max(0.0, min(1.0, (float(v) - lo) / max(hi - lo, 1e-9)))
    rr = int(34 + r * (239 - 34)); gg = int(197 - r * (197 - 68))
    return f"background-color: rgba({rr},{gg},68,0.22)"


def _render_stage_table(rows, stages):
    """Wide table: one row per driver, grouped columns per stage."""
    df = pd.DataFrame(rows)
    # Column order: identity, then each stage's block. Drop any all-empty
    # column so e.g. "S3 Pts" (no stage points in a final stage) disappears
    # instead of showing a column of dashes.
    cols = ["Finish", "Driver", "Car"]
    for s in stages:
        for c in [f"S{s} Speed", f"S{s} Rank", f"S{s} AvgPos", f"S{s} Chg", f"S{s} Pts"]:
            if c in df.columns and df[c].notna().any():
                cols.append(c)
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    # Per-column number formats: speeds/avg-pos 1-decimal; everything else that's
    # integer-valued (Finish, Rank, Chg, Pts) gets {:.0f} so it renders clean
    # ("7" not "7.000000" — those cols are float64 only because NaN forces it).
    fmt = {}
    for c in disp.columns:
        if c in ("Driver", "Car"):
            continue
        if c.endswith("Speed") or c.endswith("AvgPos"):
            fmt[c] = "{:.1f}"   # genuinely decimal
        else:
            fmt[c] = "{:.0f}"   # integer-valued (Finish, Rank, Chg, Pts)

    def _cell_style(col):
        name = col.name
        # Position change: green text for gained, red for lost.
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
        # Finish + per-stage Rank + AvgPos are all "lower is better" → heatmap.
        if name == "Finish" or name.endswith("Rank") or name.endswith("AvgPos"):
            return [_heat_low_good(v) for v in col]
        # Stage points: higher is better → light green when present.
        if name.endswith("Pts"):
            return ["background-color: rgba(34,197,94,0.18)" if pd.notna(v) and v > 0
                    else "" for v in col]
        return ["" for _ in col]

    styled = disp.style.apply(_cell_style, axis=0).format(fmt, na_rep="—")
    st.dataframe(styled, width="stretch", hide_index=True, height=620)


def _render_green_summary(rows, stages):
    """Compact table: each driver's overall + per-stage green-speed RANK, to see
    pace trajectory across the race at a glance (lower = faster)."""
    df = pd.DataFrame(rows)
    cols = ["Finish", "Driver", "Car"] + [f"S{s} Rank" for s in stages if f"S{s} Rank" in df.columns]
    disp = df[cols].copy()
    disp = disp.rename(columns={f"S{s} Rank": f"Stage {s}" for s in stages})

    rank_cols = [f"Stage {s}" for s in stages if f"Stage {s}" in disp.columns]

    def _rank_heat(col):
        # Rank + Finish are "lower is better" → shared heatmap helper.
        if col.name not in rank_cols and col.name != "Finish":
            return ["" for _ in col]
        return [_heat_low_good(v) for v in col]

    # Integer-format the rank + Finish columns (float64 only because of NaN).
    int_fmt = {c: "{:.0f}" for c in disp.columns if c not in ("Driver", "Car")}
    styled = disp.style.apply(_rank_heat, axis=0).format(int_fmt, na_rep="—")
    st.dataframe(styled, width="stretch", hide_index=True, height=620)
    st.caption("Per-stage green-flag speed RANK (1 = fastest). Read left→right to "
               "see whether a driver got faster or faded as the race went on.")


def _render_run_pace(race, series_id, selected_year):
    """Long-run (sustained green pace) + restart pace, per driver, ranked."""
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
    st.dataframe(styled, width="stretch", hide_index=True, height=620)
    st.caption("Tip: a driver who's strong on Long Run but weak on Restart "
               "(or vice-versa) is a setup/strategy read for DFS.")


def _render_speed_over_time(race, series_id, selected_year, rows):
    """Driver-selectable overlay of lap speed across the race (from lap-times)."""
    api_id = race.get("race_id")
    yr = (race.get("race_date", "") or "")[:4]
    try:
        yr = int(yr)
    except (TypeError, ValueError):
        yr = selected_year

    with st.spinner("Loading lap-by-lap data..."):
        lap_data = fetch_lap_times(series_id, api_id, yr)
    if not lap_data or "laps" not in lap_data:
        st.info("Lap-by-lap data isn't available for this race.")
        return

    all_drivers = sorted(d.get("FullName") for d in lap_data["laps"] if d.get("FullName"))
    # Default to the top 5 finishers (from the stage breakdown, already finish-sorted).
    top5 = [r["Driver"] for r in rows[:5] if r.get("Driver") in all_drivers]
    default = top5 or all_drivers[:5]

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        picks = st.multiselect("Drivers to overlay", all_drivers, default=default,
                               key="racelab_speed_drivers")
    with c2:
        metric_label = st.radio("Metric", ["Lap Time", "Speed"], horizontal=True,
                                key="racelab_speed_metric")
    with c3:
        green_only = st.toggle("Green-flag laps only", value=True,
                               key="racelab_speed_greenonly",
                               help="Drop caution AND green-flag pit laps so only "
                                    "clean racing pace shows.")
    if not picks:
        st.info("Select at least one driver.")
        return

    metric = "speed" if metric_label == "Speed" else "time"
    fig = race_speed_chart(lap_data, selected_drivers=picks,
                           green_only=green_only, metric=metric)
    if fig is None:
        st.info("No lap data to plot for the selected drivers.")
        return
    st.plotly_chart(fig, width="stretch")
    _unit = "lap time (s, faster = higher — axis inverted)" if metric == "time" else "lap speed (mph)"
    if green_only:
        st.caption(f"Each line is a driver's {_unit}. Caution laps and green-flag "
                   "pit laps are removed, so this is clean racing pace.")
    else:
        st.caption(f"Each line is a driver's {_unit}. Shaded bands are caution "
                   "periods; the deep dips are pit stops. Toggle 'green-flag laps "
                   "only' for clean racing pace.")


def _render_field_fallback(db_id):
    """When stage data is missing, at least show the full-field results."""
    data = query_race_field_results(db_id)
    rows = data.get("rows") if isinstance(data, dict) else None
    if not rows:
        return
    df = pd.DataFrame(rows)
    show = [c for c in ["Finish", "Driver", "Car", "Team", "Start",
                        "Laps Led", "Fast Laps", "Avg Run", "Rating", "DK Pts"]
            if c in df.columns]
    st.dataframe(safe_fillna(format_display_df(df[show])),
                 width="stretch", hide_index=True, height=560)
