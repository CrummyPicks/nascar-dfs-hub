"""Track Data — a DFS-analytics profile for any track.

Behavioral + scoring metrics computed from race_results: track-position
dependency, dominator/fast-lap concentration, chaos, the DK/FD scoring
distribution, and start-bucket finish rates. This is the "what kind of race
is this" reference that informs weight choices and lineup construction.
"""
import pandas as pd
import streamlit as st

from src.components import section_header
from src.config import (TRACK_TYPE_MAP, TRACK_TYPE_DISPLAY, TRACK_TYPE_COLORS,
                        similar_tracks_for, track_specs)
from src.data import query_track_profile


def _card(label, value, sub="", color="#38bdf8"):
    return (f'<div style="background:linear-gradient(135deg,#111827,#0f172a);'
            f'border:1px solid #1e293b;border-left:3px solid {color};'
            f'border-radius:10px;padding:10px 14px;min-width:120px;">'
            f'<div style="color:#64748b;font-size:0.62rem;text-transform:uppercase;'
            f'letter-spacing:0.8px;font-weight:600;">{label}</div>'
            f'<div style="font-family:Rajdhani,sans-serif;color:#f1f5f9;'
            f'font-size:1.5rem;font-weight:700;line-height:1.1;">{value}</div>'
            f'<div style="color:#475569;font-size:0.68rem;">{sub}</div></div>')


def _row(cards):
    return ('<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.3rem 0;">'
            + "".join(cards) + '</div>')


def _fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "—"


def _render_reference_only(pick):
    """Reference view for a track with no completed races in our data (a
    returning venue like Chicagoland, or a brand-new one): physical specs,
    track type, and comparable tracks from config — everything we know that
    doesn't require race history."""
    specs = track_specs(pick)
    tt = TRACK_TYPE_MAP.get(pick, "intermediate")
    tt_color = TRACK_TYPE_COLORS.get(tt, "#3b82f6")
    tt_label = TRACK_TYPE_DISPLAY.get(tt, tt.title())
    _sim = similar_tracks_for(pick) or {}
    comps = list(_sim.get("primary") or []) + list(_sim.get("secondary") or [])
    best_comp = comps[0] if comps else "—"
    if _sim.get("profile"):
        st.caption(f"**{pick}** — {_sim['profile']}")

    st.markdown(_row([
        _card("Track Type", tt_label,
              f"{specs['length']:g} mi" if specs.get("length") else "", tt_color),
        _card("Best Comp", best_comp, "similar track", "#2dd4bf"),
    ]), unsafe_allow_html=True)
    st.info("No completed races in our data for this track yet — showing its "
            "physical profile and comparable tracks. Behavioral & scoring "
            "metrics will populate once it runs. Use the similar tracks below "
            "as a proxy in the meantime.")

    if specs:
        st.markdown("**Track Profile** (physical specs)")
        spec_rows = [
            ("Length", f"{specs['length']:g} mi" if specs.get("length") else "—"),
            ("Banking", specs.get("banking", "—")),
            ("Surface", specs.get("surface", "—")),
            ("Shape", specs.get("shape", "—")),
            ("Size Group", tt_label),
        ]
        st.dataframe(pd.DataFrame(spec_rows, columns=["Spec", "Value"]),
                     width="stretch", hide_index=True)
    else:
        st.caption("Physical specs not on file for this track — add it to "
                   "TRACK_SPECS in config.")

    if comps:
        st.divider()
        st.caption(f"**Similar tracks** (use as a proxy pre-race): {', '.join(comps)}")


def render(*, series_id, series_name, track_name=None, selected_year=None):
    section_header("Track Data", "DFS behavioral & scoring profile")

    # Track picker — every track with stored races for this series, plus the
    # current race's track preselected.
    import sqlite3
    from src.config import DB_PATH
    tracks = []
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(str(DB_PATH))
            # Every track on the 2022+ schedule for this series — NOT just those
            # with stored results, so a venue returning to the calendar (e.g.
            # Chicagoland in 2026) still appears with its physical profile.
            tracks = [r[0] for r in conn.execute('''
                SELECT DISTINCT t.name FROM races r JOIN tracks t ON t.id = r.track_id
                WHERE r.series_id = ? AND r.season >= 2022
                ORDER BY t.name
            ''', (series_id,)).fetchall()]
            conn.close()
        except Exception:
            tracks = []
    if not tracks:
        st.info("No track data available for this series yet.")
        return

    default_idx = tracks.index(track_name) if track_name in tracks else 0
    pick = st.selectbox("Track", tracks, index=default_idx, key="trackdata_pick")

    prof = query_track_profile(pick, series_id)
    if prof:
        _sim0 = similar_tracks_for(pick) or {}
        if _sim0.get("profile"):
            st.caption(f"**{pick}** — {_sim0['profile']}")
    if not prof:
        # No completed races in our data (a track returning to the schedule,
        # like Chicagoland, or a brand-new venue). Still show the physical
        # profile, track type, and comparable tracks so the page is useful
        # pre-race — behavioral/scoring metrics populate once it runs.
        _render_reference_only(pick)
        return

    tt = prof["track_type"]
    tt_color = TRACK_TYPE_COLORS.get(tt, "#3b82f6")
    tt_label = TRACK_TYPE_DISPLAY.get(tt, tt.title())
    _sim = similar_tracks_for(pick) or {}
    _primary = _sim.get("primary") or []
    _secondary = _sim.get("secondary") or []
    comps = list(_primary) + list(_secondary)
    best_comp = _primary[0] if _primary else (comps[0] if comps else "—")
    _profile_note = _sim.get("profile", "")

    specs = track_specs(pick)

    # ── Headline cards ──
    _tt_sub = " · ".join(filter(None, [
        "Concrete" if prof["concrete"] else "",
        f"{specs['length']:g} mi" if specs.get("length") else "",
        f"{prof['races']} races"]))
    st.markdown(_row([
        _card("Track Type", tt_label, _tt_sub, tt_color),
        _card("Position Dependency", _fmt(prof["pos_dependency"]),
              "start↔finish corr", "#a78bfa"),
        _card("Chaos Score", _fmt(prof["chaos"]),
              "1 = unpredictable", "#fb923c"),
        _card("Dominator Conc.", _fmt(prof["dominator_concentration"]),
              "top-5 laps-led share", "#f472b6"),
        _card("Best Comp", best_comp, "similar track", "#2dd4bf"),
        _card("Avg DK", _fmt(prof["avg_dk"]), f"max {_fmt(prof['best_dk'])}",
              "#4ade80"),
    ]), unsafe_allow_html=True)

    st.divider()

    # ── Physical specs (static reference) ──
    if specs:
        st.markdown("**Track Profile** (physical specs)")
        spec_rows = [
            ("Length", f"{specs['length']:g} mi" if specs.get("length") else "—"),
            ("Banking", specs.get("banking", "—")),
            ("Surface", specs.get("surface", "—")),
            ("Shape", specs.get("shape", "—")),
            ("Size Group", tt_label),
        ]
        st.dataframe(pd.DataFrame(spec_rows, columns=["Spec", "Value"]),
                     width="stretch", hide_index=True)
    else:
        st.caption("Physical specs not on file for this track — add it to "
                   "TRACK_SPECS in config.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Behavior Profile**")
        beh = [
            ("Races Used", prof["races"]),
            ("Driver Rows", prof["rows"]),
            ("Start / Finish Corr", prof["start_finish_corr"]),
            ("Winner from Top 5", _fmt(prof["winner_from_top5_pct"], "%")),
            ("Dominator Concentration", prof["dominator_concentration"]),
            ("Fast-Lap Concentration", prof["fast_lap_concentration"]),
            ("Finish Volatility", prof["finish_volatility"]),
            ("Avg |Place Diff|", prof["avg_abs_pd"]),
            ("DNF Rate", _fmt(prof["dnf_rate"], "%")),
            ("Avg Rating", prof["avg_rating"]),
            ("Avg Quality Passes", prof["avg_quality_passes"]),
        ]
        st.dataframe(pd.DataFrame(beh, columns=["Metric", "Value"]),
                     width="stretch", hide_index=True, height=420)

    with col_b:
        st.markdown("**DFS Scoring Profile**")
        plat_rows = [
            ("Avg DK", prof["avg_dk"], "Avg FD", prof["avg_fd"]),
            ("Median DK", prof["median_dk"], "Median FD", prof["median_fd"]),
            ("DK Std Dev", prof["dk_std"], "FD Std Dev", prof["fd_std"]),
            ("Best DK", prof["best_dk"], "Best FD", prof["best_fd"]),
        ]
        st.dataframe(
            pd.DataFrame([(a, b, c, d) for a, b, c, d in plat_rows],
                         columns=["DraftKings", "Value", "FanDuel", "Value "]),
            width="stretch", hide_index=True)

        st.markdown("**Top-5 Finish Rate by Starting Bucket**")
        bk = prof["bucket_top5"]
        bdf = pd.DataFrame(
            [(b, _fmt(bk.get(b), "%")) for b in ["P1-5", "P6-10", "P11-20", "P21+"]],
            columns=["Started", "Finished Top 5"])
        st.dataframe(bdf, width="stretch", hide_index=True)
        st.caption("How often a driver starting in each range finished top 5 — "
                   "a read on how much qualifying position matters here.")

    if comps:
        st.divider()
        st.caption(f"**Similar tracks** (for cross-track reads): {', '.join(comps)}")
    _spec_note = ("Physical specs are a hand-maintained reference; the behavioral "
                  "+ scoring metrics are computed from 2022+ results."
                  if specs else
                  "Physical specs not on file for this track yet — the behavioral "
                  "+ scoring metrics are computed from 2022+ results.")
    st.caption(_spec_note)
