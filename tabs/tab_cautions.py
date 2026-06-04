"""Tab: Cautions & Penalties — what actually happened during a race.

Caution timeline (lap range, reason, cars involved, lucky dog) + penalties
(pre-race from race_comments, and — once pit data is wired — likely in-race pit
penalties derived from pit anomalies). Display/story data, not a projection input.
"""
import pandas as pd
import streamlit as st

from src.components import section_header
from src.data import query_race_cautions


def _get_year(race):
    d = (race.get("race_date", "") or "")[:4]
    try:
        return int(d)
    except (TypeError, ValueError):
        return None


def render(*, completed_races, series_id, selected_year, series_name="Cup"):
    """Render the Cautions & Penalties tab."""
    section_header("Cautions & Penalties", "What happened during the race")

    if not completed_races:
        st.info("No completed races available for this series/year.")
        return

    labels, label_to_race = [], {}
    for _, race in completed_races:
        track = race.get("track_name", "")
        name = race.get("race_name", "")
        date = (race.get("race_date", "") or "")[:10]
        lbl = f"{date} — {track}: {name}"
        labels.append(lbl)
        label_to_race[lbl] = race
    labels = list(reversed(labels))

    selected = st.selectbox("Race", labels, index=0, key="cautions_pick")
    race = label_to_race[selected]
    api_id = race.get("race_id")
    yr = _get_year(race) or selected_year

    from tabs.tab_projections import _resolve_db_race_id
    db_id = _resolve_db_race_id(api_id, series_id)

    data = query_race_cautions(series_id, api_id, yr, db_id)
    cautions = data.get("cautions", [])
    penalties = data.get("prerace_penalties", [])
    summ = data.get("summary", {})

    if not cautions and not penalties:
        st.info("No caution or penalty data available for this race yet.")
        return

    # Summary strip
    cols = st.columns(4)
    cols[0].metric("Cautions", summ.get("n_cautions") if summ.get("n_cautions") is not None else len(cautions))
    cols[1].metric("Caution Laps", summ.get("caution_laps") if summ.get("caution_laps") is not None else "—")
    cols[2].metric("Lead Changes", summ.get("lead_changes") if summ.get("lead_changes") is not None else "—")
    cols[3].metric("Pre-Race Penalties", len(penalties))
    st.divider()

    # ── Penalties ──
    st.markdown("**Penalties**")
    if penalties:
        prows = []
        for p in penalties:
            prows.append({
                "When": "Pre-race",
                "Drivers": ", ".join(p["Drivers"]),
                "Cars": p["Cars"],
                "Reason": p["Reason"],
            })
        st.dataframe(pd.DataFrame(prows), width="stretch", hide_index=True)
    else:
        st.caption("No pre-race penalties reported for this race.")
    st.caption("In-race pit penalties aren't published in a clean NASCAR field; "
               "once pit data is wired in, likely pit-road penalties will appear "
               "here (flagged as derived).")
    st.divider()

    # ── Caution timeline ──
    st.markdown("**Caution Timeline**")
    if not cautions:
        st.caption("No cautions in this race.")
        return
    crows = []
    for c in cautions:
        crows.append({
            "#": c["Caution"],
            "Laps": c["Laps"],
            "Caution Laps": c["Laps#"],
            "Reason": c["Reason"],
            "Cars Involved": ", ".join(c["Involved"]) if c["Involved"] else "—",
            "Lucky Dog": c["Lucky Dog"],
        })
    cdf = pd.DataFrame(crows)
    fmt = {"Caution Laps": "{:.0f}"} if "Caution Laps" in cdf.columns else {}
    styled = cdf.style.format(fmt, na_rep="—")
    st.dataframe(styled, width="stretch", hide_index=True,
                 height=min(560, 60 + len(cdf) * 36))

    # Most-involved drivers (incident frequency in this race).
    involved_counts = {}
    for c in cautions:
        # Only count incident-type cautions (skip scheduled/competition/stage).
        if c["Reason"] in ("Accident", "Spin", "Incident", "Crash"):
            for nm in c["Involved"]:
                involved_counts[nm] = involved_counts.get(nm, 0) + 1
    if involved_counts:
        st.markdown("**Most Caution-Involved (this race)**")
        ic = sorted(involved_counts.items(), key=lambda x: -x[1])
        ic_df = pd.DataFrame([{"Driver": n, "Cautions Involved": k} for n, k in ic if k >= 1])
        st.dataframe(ic_df, width="stretch", hide_index=True,
                     height=min(300, 60 + len(ic_df) * 36))
