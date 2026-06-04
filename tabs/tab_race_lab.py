"""Tab: Race Lab — deep per-race breakdown (green-flag pace + per-stage arc).

Surfaces the data that was previously buried in the driver popup: for any
completed race, the full field with green-flag speed and a stage-by-stage
breakdown (speed, avg running position, positions gained/lost, stage points) —
so you can see how the race developed and who was fast vs. who hung on.
"""
import pandas as pd
import streamlit as st

from src.components import section_header
from src.data import query_race_stage_breakdown, query_race_field_results
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

    view = st.radio("View", ["Per-Stage Breakdown", "Green Speed Summary"],
                    horizontal=True, label_visibility="collapsed", key="racelab_view")

    if view == "Per-Stage Breakdown":
        _render_stage_table(rows, stages)
    else:
        _render_green_summary(rows, stages)


def _render_stage_table(rows, stages):
    """Wide table: one row per driver, grouped columns per stage."""
    df = pd.DataFrame(rows)
    # Column order: identity, then each stage's block.
    cols = ["Finish", "Driver", "Car"]
    for s in stages:
        cols += [f"S{s} Speed", f"S{s} Rank", f"S{s} AvgPos", f"S{s} Chg", f"S{s} Pts"]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    # Format: speeds 1-dec, the rest clean ints where possible.
    fmt = {}
    for s in stages:
        for c, f in [(f"S{s} Speed", "{:.1f}"), (f"S{s} AvgPos", "{:.1f}")]:
            if c in disp.columns:
                fmt[c] = f

    def _chg_style(col):
        # Color the per-stage position change: green gained, red lost.
        if not col.name.endswith("Chg"):
            return ["" for _ in col]
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

    styled = disp.style.apply(_chg_style, axis=0)
    if fmt:
        styled = styled.format(fmt, na_rep="—")
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
        if col.name not in rank_cols:
            return ["" for _ in col]
        out = []
        for v in col:
            if pd.isna(v):
                out.append(""); continue
            # green (fast, rank 1) -> red (slow). Scale to ~40-car field.
            r = max(0.0, min(1.0, (float(v) - 1) / 35.0))
            rr = int(34 + r * (239 - 34)); gg = int(197 - r * (197 - 68))
            out.append(f"background-color: rgba({rr},{gg},68,0.25)")
        return out

    styled = disp.style.apply(_rank_heat, axis=0)
    st.dataframe(styled, width="stretch", hide_index=True, height=620)
    st.caption("Per-stage green-flag speed RANK (1 = fastest). Read left→right to "
               "see whether a driver got faster or faded as the race went on.")


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
