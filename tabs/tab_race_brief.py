"""Race Brief — the one-page pre-lock summary.

Assembles the top-line reads the user otherwise clicks 6-8 pages for:
track snapshot, Vegas board, top projected + value plays, leverage (when
ownership has been computed), practice standouts, and risk flags. Pure
assembly — every number comes from the same engine/queries the dedicated
pages use, so the brief can never disagree with them.
"""

import pandas as pd
import streamlit as st

from src.components import (section_header, interactive_drill_down_dataframe,
                            stat_card, card_row, fmt_dash)
from src.config import (TRACK_TYPE_MAP, TRACK_TYPE_DISPLAY, TRACK_TYPE_COLORS,
                        similar_tracks_for, track_specs,
                        DK_PTS_LAP_LED, DK_PTS_FASTEST_LAP, DK_PTS_PLACE_DIFF)
from src.utils import (format_display_df, safe_fillna, parse_american_odds,
                       build_norm_lookup, fuzzy_get)

_card = stat_card
_row = card_row
_fmt = fmt_dash


def _self_serve_projections(*, series_id, race_id, race_name, track_name,
                            race_date, scheduled_laps, entry_list_df,
                            qualifying_df, practice_data, dk_df):
    """Projections for the current race — session maps when fresh, otherwise
    run the engine headlessly and cache back (same pattern as the Optimizer,
    so the brief never requires a Projections-page visit first)."""
    if st.session_state.get("proj_maps_key") == f"{series_id}_{race_id}":
        return (st.session_state.get("proj_dk_map", {}),
                st.session_state.get("proj_fd_map", {}),
                st.session_state.get("proj_detail_map", {}))
    if not race_id:
        return {}, {}, {}
    try:
        from tabs.tab_accuracy import _generate_race_projections
        drv = None
        if entry_list_df is not None and not entry_list_df.empty:
            drv = entry_list_df["Driver"].dropna().tolist()
        elif dk_df is not None and not dk_df.empty:
            drv = dk_df["Driver"].dropna().tolist()
        qual_ov = {}
        if (qualifying_df is not None and not qualifying_df.empty
                and "Qualifying Position" in qualifying_df.columns):
            for _, qr in qualifying_df.iterrows():
                qp = qr.get("Qualifying Position")
                if qr.get("Driver") and pd.notna(qp):
                    qual_ov[qr["Driver"]] = int(qp)
        race_dict = {"race_id": race_id, "race_name": race_name,
                     "track_name": track_name, "race_date": race_date,
                     "scheduled_laps": scheduled_laps}
        pdk, pdetail, _, meta = _generate_race_projections(
            race_dict, series_id, drivers_override=drv,
            qual_pos_override=qual_ov or None,
            practice_override=practice_data or None)
        if pdk:
            st.session_state["proj_maps_key"] = f"{series_id}_{race_id}"
            st.session_state["proj_dk_map"] = pdk
            st.session_state["proj_fd_map"] = (meta or {}).get("proj_fd", {})
            st.session_state["proj_detail_map"] = pdetail or {}
            st.session_state["proj_floor_map"] = {
                d: v.get("proj_floor") for d, v in (pdetail or {}).items()}
            st.session_state["proj_ceiling_map"] = {
                d: v.get("proj_ceiling") for d, v in (pdetail or {}).items()}
            return pdk, st.session_state["proj_fd_map"], pdetail or {}
    except Exception:
        pass
    return {}, {}, {}


def _salary_map(df):
    """{driver: salary} from a DK/FD salary DataFrame (first *Salary column)."""
    if df is None or df.empty or "Driver" not in df.columns:
        return {}
    sal_col = next((c for c in df.columns if "Salary" in c), None)
    if not sal_col:
        return {}
    out = {}
    for _, r in df.iterrows():
        if r.get("Driver") and pd.notna(r.get(sal_col)):
            try:
                out[r["Driver"]] = int(r[sal_col])
            except (ValueError, TypeError):
                continue
    return out


def render(*, entry_list_df, qualifying_df, lap_averages_df, practice_data,
           race_name, race_id, track_name, series_id, dk_df, fd_df,
           odds_data, scheduled_laps, race_date, platform, is_prerace=True):
    """Render the Race Brief page."""
    section_header("Race Brief", race_name)

    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    tt_label = TRACK_TYPE_DISPLAY.get(track_type, track_type.title())
    tt_color = TRACK_TYPE_COLORS.get(track_type, "#3b82f6")

    use_fd = platform == "FanDuel"
    _tag = "FD" if use_fd else "DK"
    sal_map = _salary_map(fd_df if use_fd else dk_df)
    sal_norm = build_norm_lookup(sal_map) if sal_map else {}

    # ── Data status line ─────────────────────────────────────────────
    n_entry = len(entry_list_df) if entry_list_df is not None else 0
    n_qual = (int(qualifying_df["Qualifying Position"].notna().sum())
              if qualifying_df is not None and not qualifying_df.empty
              and "Qualifying Position" in qualifying_df.columns else 0)
    n_prac = len(lap_averages_df) if lap_averages_df is not None else 0
    missing = []
    if not odds_data:
        missing.append("odds")
    if not sal_map:
        missing.append(f"{_tag} salaries")
    if not n_prac:
        missing.append("practice")
    if not n_qual:
        missing.append("qualifying")
    st.caption(f"{n_entry} entered  •  {n_qual} qualified  •  "
               f"{n_prac} with practice laps  •  {len(odds_data or {})} with odds"
               + (f"  •  ⚠️ missing: {', '.join(missing)}" if missing else "  •  ✅ all data loaded"))

    # ── Projections (self-serve, cached in session) ──────────────────
    with st.spinner("Building projections..."):
        proj_dk, proj_fd, proj_detail = _self_serve_projections(
            series_id=series_id, race_id=race_id, race_name=race_name,
            track_name=track_name, race_date=race_date,
            scheduled_laps=scheduled_laps, entry_list_df=entry_list_df,
            qualifying_df=qualifying_df, practice_data=practice_data,
            dk_df=dk_df)
    proj_pts = proj_fd if use_fd else proj_dk

    # ── Track snapshot ────────────────────────────────────────────────
    from src.data import query_track_profile
    prof = query_track_profile(track_name, series_id) or {}
    specs = track_specs(track_name) or {}
    _sim = similar_tracks_for(track_name) or {}
    comps = list(_sim.get("primary") or []) + list(_sim.get("secondary") or [])
    _len_sub = f"{specs['length']:g} mi" if specs.get("length") else ""
    cards = [_card("Track Type", tt_label, _len_sub, tt_color)]
    if prof:
        cards += [
            _card("Chaos", _fmt(prof.get("chaos")), "1 = unpredictable", "#fb923c"),
            _card("Pos Dependency", _fmt(prof.get("pos_dependency")),
                  "start↔finish corr", "#a78bfa"),
            _card("Dominator Conc.", _fmt(prof.get("dominator_concentration")),
                  "top-5 laps-led share", "#f472b6"),
            _card("DNF Rate", _fmt(prof.get("dnf_rate"), "%"), "field avg", "#ef4444"),
        ]
    if comps:
        cards.append(_card("Best Comp", comps[0], "similar track", "#2dd4bf"))
    st.markdown(_row(cards), unsafe_allow_html=True)
    if not prof:
        st.caption("No completed races at this track in our data — behavioral "
                   "metrics unavailable. Lean on the comp tracks: "
                   + (", ".join(comps) if comps else "—"))
    if _sim.get("profile"):
        st.caption(f"**{track_name}** — {_sim['profile']}")

    st.divider()
    col_l, col_r = st.columns(2)

    # ── Vegas board (top 8) ───────────────────────────────────────────
    with col_l:
        st.markdown("**Vegas Board** (win odds)")
        if odds_data:
            board = []
            for d, o in odds_data.items():
                val = parse_american_odds(o)
                if val is None:
                    continue
                imp = (100 / (val + 100) if val > 0
                       else abs(val) / (abs(val) + 100))
                board.append({"Driver": d, "Win Odds": f"+{val}" if val > 0 else str(val),
                              "Implied %": round(imp * 100, 1)})
            board.sort(key=lambda x: -x["Implied %"])
            st.dataframe(pd.DataFrame(board[:8]), width="stretch",
                         hide_index=True)
        else:
            st.info("No odds loaded for this race yet.")

    # ── Practice standouts (top 5 by the ENGINE's practice signal) ────
    # NOT NASCAR's overall lap average: that metric averages every lap a
    # driver turns, so short-burst profiles (fresh tires each run) rank
    # high while honest race-sim long runs get dragged down by worn-tire
    # laps — e.g. a driver 4th "overall" with only the 25th-best single
    # lap and mid-pack in every sustained window. The engine's signal
    # (coverage-weighted consecutive-lap window ranks) is what actually
    # feeds projections, so the brief must agree with it.
    with col_r:
        st.markdown("**Practice Standouts**")
        la = (lap_averages_df if lap_averages_df is not None
              else pd.DataFrame())
        if practice_data:
            top = sorted(practice_data.items(), key=lambda x: x[1])[:5]
            rows = []
            for d, sig in top:
                r = (la[la["Driver"] == d]
                     if not la.empty and "Driver" in la.columns
                     else pd.DataFrame())

                def _g(col):
                    if r.empty or col not in r.columns:
                        return None
                    v = r.iloc[0][col]
                    return v if pd.notna(v) else None
                rows.append({"Driver": d, "Sig Rank": round(float(sig), 1),
                             "10L Rk": _g("10 Lap Rank"),
                             "15L Rk": _g("15 Lap Rank"),
                             "Best Rk": _g("1 Lap Rank"),
                             "Laps": _g("Laps")})
            st.dataframe(safe_fillna(format_display_df(pd.DataFrame(rows))),
                         width="stretch", hide_index=True)
            st.caption("Ranked by the projection engine's practice signal "
                       "(long-run-weighted window ranks) — NASCAR's raw "
                       "'overall average' is not used; it flatters "
                       "short-burst run profiles.")
        elif not la.empty:
            # Signal unavailable (shouldn't happen when laps exist) — fall
            # back to sustained-run windows, never the overall average.
            pcols = [c for c in ["Driver", "15 Lap Rank", "10 Lap Rank",
                                 "Laps"] if c in la.columns]
            pr = la[pcols].copy()
            _sort = next((c for c in ["15 Lap Rank", "10 Lap Rank"]
                          if c in pr.columns), None)
            if _sort:
                pr = pr.sort_values(_sort, na_position="last")
            st.dataframe(safe_fillna(format_display_df(pr.head(5))),
                         width="stretch", hide_index=True)
        else:
            st.info("No practice data yet.")

    st.divider()

    # ── Top projected ─────────────────────────────────────────────────
    if proj_pts:
        st.markdown(f"**Top Projected ({_tag})**")
        rows = []
        for d, pts in sorted(proj_pts.items(), key=lambda x: -x[1]):
            det = proj_detail.get(d, {})
            sal = fuzzy_get(d, sal_map, sal_norm) if sal_map else None
            rows.append({
                "Driver": d,
                "Salary": sal,
                f"Proj {_tag}": round(pts, 1),
                "Floor": det.get("proj_floor"),
                "Ceiling": det.get("proj_ceiling"),
                "Proj Finish": det.get("proj_finish"),
                "Start": det.get("start"),
                "Value": (round(pts / (sal / 1000), 2) if sal else None),
            })
        top_df = pd.DataFrame(rows[:10])
        interactive_drill_down_dataframe(
            safe_fillna(format_display_df(top_df)),
            key=f"brief_top_{series_id}_{race_id}",
            series_id=series_id, track_name=track_name,
            width="stretch", hide_index=True,
        )

        # ── Value + leverage plays side by side ───────────────────────
        v_col, l_col = st.columns(2)
        with v_col:
            st.markdown("**Value Plays** (pts per $1k)")
            vals = [r for r in rows if r.get("Value")]
            vals.sort(key=lambda x: -x["Value"])
            if vals:
                st.dataframe(safe_fillna(format_display_df(pd.DataFrame(
                    [{"Driver": v["Driver"], "Salary": v["Salary"],
                      f"Proj {_tag}": v[f"Proj {_tag}"], "Value": v["Value"]}
                     for v in vals[:5]]))), width="stretch", hide_index=True)
            else:
                st.info(f"Load {_tag} salaries to see value plays.")

        with l_col:
            st.markdown("**Leverage Plays** (proj vs projected ownership)")
            own_map = st.session_state.get(
                "proj_own_map_fd" if use_fd else "proj_own_map", {})
            _own_fresh = (st.session_state.get("proj_maps_key")
                          == f"{series_id}_{race_id}") and own_map
            if _own_fresh:
                own_norm = build_norm_lookup(own_map)
                lev = []
                for r in rows[:20]:            # only among rosterable projections
                    own = fuzzy_get(r["Driver"], own_map, own_norm)
                    if own and own > 0:
                        lev.append({"Driver": r["Driver"],
                                    f"Proj {_tag}": r[f"Proj {_tag}"],
                                    "GPP Own%": round(own, 1),
                                    "Leverage": round(r[f"Proj {_tag}"] / own, 2)})
                lev.sort(key=lambda x: -x["Leverage"])
                if lev:
                    st.dataframe(safe_fillna(format_display_df(
                        pd.DataFrame(lev[:5]))), width="stretch", hide_index=True)
            else:
                st.info("Ownership projections are computed on the Projections "
                        "page — visit it once and the leverage reads appear here.")

        # ── Risk flags: thin floors among the top-15 projected ────────
        st.markdown("**Thin Floors** (biggest proj → floor drop in the top 15 — "
                    "wreck/DNF-risk priced in)")
        risky = []
        for r in rows[:15]:
            if r.get("Floor") is not None:
                gap = round(r[f"Proj {_tag}"] - r["Floor"], 1)
                risky.append({"Driver": r["Driver"],
                              f"Proj {_tag}": r[f"Proj {_tag}"],
                              "Floor": r["Floor"], "Drop": gap})
        risky.sort(key=lambda x: -x["Drop"])
        if risky:
            st.dataframe(safe_fillna(format_display_df(pd.DataFrame(risky[:5]))),
                         width="stretch", hide_index=True)
    else:
        st.warning("Could not build projections for this race — check that the "
                   "entry list has loaded (Data & Settings page).")

    # ── Roster rules footer ───────────────────────────────────────────
    st.divider()
    if use_fd:
        st.caption("**FanDuel**: 5 drivers / $50,000  •  0.1/lap led  •  "
                   "0.1/lap completed  •  ±0.5/place differential (off qualifying)  •  "
                   "no fastest-lap points")
    else:
        st.caption(f"**DraftKings**: 6 drivers / $50,000  •  {DK_PTS_LAP_LED}/lap led  •  "
                   f"{DK_PTS_FASTEST_LAP}/fastest lap  •  ±{DK_PTS_PLACE_DIFF}/place differential (off qualifying)")
