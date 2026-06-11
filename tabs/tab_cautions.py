"""Tab: Cautions & Penalties — what actually happened during a race.

Caution timeline (lap range, reason, cars involved, lucky dog) + penalties
(pre-race from race_comments, and — once pit data is wired — likely in-race pit
penalties derived from pit anomalies). Display/story data, not a projection input.
"""
import pandas as pd
import streamlit as st

from src.components import section_header
from src.data import query_race_cautions, query_race_pit_summary


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

    pit = (query_race_pit_summary(db_id, cautions=cautions) if db_id
           else {"rows": [], "likely_penalties": [], "repair_stops": []})
    pit_rows = pit.get("rows", [])
    likely_pen = pit.get("likely_penalties", [])
    repair_stops = pit.get("repair_stops", [])

    if not cautions and not penalties and not pit_rows:
        st.info("No caution, penalty or pit data available for this race yet — "
                "this page fills in after the race runs.")
        return

    # Summary strip
    cols = st.columns(4)
    cols[0].metric("Cautions", summ.get("n_cautions") if summ.get("n_cautions") is not None else len(cautions))
    cols[1].metric("Caution Laps", summ.get("caution_laps") if summ.get("caution_laps") is not None else "—")
    cols[2].metric("Lead Changes", summ.get("lead_changes") if summ.get("lead_changes") is not None else "—")
    cols[3].metric("Pre-Race Penalties", len(penalties))
    st.divider()

    # ── Penalties (pre-race confirmed + derived likely pit penalties) ──
    st.markdown("**Penalties**")
    prows = []
    for p in penalties:
        prows.append({
            "When": "Pre-race",
            "Driver": ", ".join(p["Drivers"]),
            "Reason": p["Reason"],
            "Confidence": "Confirmed",
        })
    for lp in likely_pen:
        prows.append({
            "When": f"Lap {lp['Lap']}",
            "Driver": lp["Driver"],
            "Reason": lp["Why"],
            "Confidence": "Likely (derived)",
        })
    if prows:
        st.dataframe(pd.DataFrame(prows), width="stretch", hide_index=True)
    else:
        st.caption("No pre-race penalties reported, and no unexplained pit anomalies detected.")
    st.caption("Pre-race penalties are parsed from official race comments "
               "(confirmed). NASCAR doesn't publish in-race penalties in a clean "
               "field, so 'Likely' rows are derived from pit anomalies — but only "
               "when nothing else explains the slow stop. Slow stops attributed "
               "to crash damage or repairs are listed separately below.")

    # Slow stops explained by a real event (wreck involvement, DNF status,
    # repair-length stop) — informational, NOT penalties.
    if repair_stops:
        with st.expander(f"Damage / repair stops ({len(repair_stops)}) — slow stops "
                         "explained by an incident, not penalties", expanded=False):
            rrows = [{"When": f"Lap {r['Lap']}", "Driver": r["Driver"],
                      "Box (s)": r["Box (s)"], "Attribution": r["Why"]}
                     for r in repair_stops]
            st.dataframe(pd.DataFrame(rrows), width="stretch", hide_index=True,
                         height=min(420, 60 + len(rrows) * 36))
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

    # ── Pit Road ──
    st.divider()
    st.markdown("**Pit Road** — crew execution (sorted by avg box time)")
    if not pit_rows:
        st.caption("No pit-road data available for this race (NASCAR doesn't "
                   "publish pit timing for every event — most common for some "
                   "Truck/Xfinity races).")
    if pit_rows:
        st.caption("Avg/Best Box = stationary time on 4-tire stops; Best 2T = "
                   "fastest 2-tire stop. Fuel-only and unmeasured stops are "
                   "excluded. NASCAR's feed sometimes mislabels 2-tire stops as "
                   "4-tire — any '4-tire' stop under 7.5s (physically impossible) "
                   "is counted as 2-tire so crew averages stay honest. Green "
                   "Stops = stops taken under green.")
        cols = [c for c in ["Driver", "Stops", "4-Tire Stops", "Avg Box (s)",
                            "Best Box (s)", "2-Tire Stops", "Best 2T (s)",
                            "Green Stops"] if c in pit_rows[0]]
        pdf = pd.DataFrame(pit_rows)[cols]
        styled_p = pdf.style.format(
            {"Avg Box (s)": "{:.1f}", "Best Box (s)": "{:.1f}",
             "Best 2T (s)": "{:.1f}", "Stops": "{:.0f}",
             "4-Tire Stops": "{:.0f}", "2-Tire Stops": "{:.0f}",
             "Green Stops": "{:.0f}"},
            na_rep="—")
        st.dataframe(styled_p, width="stretch", hide_index=True,
                     height=min(620, 60 + len(pdf) * 35))
