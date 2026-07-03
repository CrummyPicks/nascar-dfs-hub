"""My Contests — real-money ROI tracking from DraftKings entry history.

Import the 'Download Entry History' CSV from draftkings.com/mycontests
(auto-detected in Downloads, or drag-and-drop) and see where the bankroll
actually grows: ROI by contest style, series, race, and month, plus a
cumulative profit line over race days. This is the ground truth the
projections/optimizer work is ultimately graded against.

Entries are stored in a local, gitignored contests.db — private financial
data never leaves this machine.
"""

import os

import pandas as pd
import streamlit as st

from src.components import section_header
from src.contests import (attach_races, find_entry_history_csvs,
                          import_entries, load_entries,
                          parse_dk_entry_history)
from src.utils import safe_fillna


def _pl_color(v):
    return "#4ade80" if v > 0 else ("#ef4444" if v < 0 else "#94a3b8")


def _card(label, value, sub="", color="#38bdf8"):
    return (f'<div style="background:linear-gradient(135deg,#111827,#0f172a);'
            f'border:1px solid #1e293b;border-left:3px solid {color};'
            f'border-radius:10px;padding:10px 14px;min-width:130px;">'
            f'<div style="color:#64748b;font-size:0.62rem;text-transform:uppercase;'
            f'letter-spacing:0.8px;font-weight:600;">{label}</div>'
            f'<div style="font-family:Rajdhani,sans-serif;color:#f1f5f9;'
            f'font-size:1.5rem;font-weight:700;line-height:1.1;">{value}</div>'
            f'<div style="color:#475569;font-size:0.68rem;">{sub}</div></div>')


def _row(cards):
    return ('<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.3rem 0;">'
            + "".join(cards) + '</div>')


def _breakdown(df, by):
    """Aggregate P/L by a column → display frame sorted by net."""
    g = df.groupby(by, dropna=False).agg(
        Entries=("entry_key", "count"),
        Fees=("entry_fee", "sum"),
        Winnings=("winnings", "sum"),
    ).reset_index()
    g[by] = g[by].fillna("?").replace("", "?")
    g["Net"] = (g["Winnings"] - g["Fees"]).round(2)
    g["ROI %"] = (100 * g["Net"] / g["Fees"].replace(0, pd.NA)).round(1)
    g["Fees"] = g["Fees"].round(2)
    g["Winnings"] = g["Winnings"].round(2)
    return g.sort_values("Net", ascending=False)


def _style_money(df, cols):
    """Green/red the money columns so profit pockets pop."""
    def _c(col):
        if col.name not in cols:
            return [""] * len(col)
        out = []
        for v in col:
            try:
                v = float(v)
            except (TypeError, ValueError):
                out.append("")
                continue
            out.append(f"color: {_pl_color(v)}; font-weight: 600")
        return out
    return df.style.apply(_c).format(
        {c: "${:,.2f}" for c in ["Fees", "Winnings", "Net"] if c in df.columns}
        | ({"ROI %": "{:+.1f}%"} if "ROI %" in df.columns else {}), na_rep="—")


def _insights(view):
    """Auto-generated verdicts: where the bankroll grows and leaks."""
    notes = []
    for by, label in [("style", "contest style"), ("series", "series")]:
        g = _breakdown(view, by)
        g = g[g["Fees"] >= 50]        # ignore tiny-sample buckets
        if len(g) < 2:
            continue
        best, worst = g.iloc[0], g.iloc[-1]
        # NB: escape dollars — bare $...$ pairs render as LaTeX in st.markdown.
        if best["Net"] > 0:
            notes.append(
                f"🟢 Best {label}: **{best[by]}** "
                f"{'+' if best['Net'] > 0 else ''}\\${best['Net']:,.2f} "
                f"({best['ROI %']:+.1f}% ROI on \\${best['Fees']:,.0f})")
        if worst["Net"] < 0:
            notes.append(
                f"🔴 Biggest leak: **{worst[by]}** "
                f"-\\${abs(worst['Net']):,.2f} "
                f"({worst['ROI %']:+.1f}% ROI on \\${worst['Fees']:,.0f})")
    return notes


def _admin_gate() -> bool:
    """Password gate using the ADMIN_PASSWORD Streamlit secret.

    Personal financial data — locked whenever the secret is configured
    (locally in .streamlit/secrets.toml, or in the cloud app's secrets).
    Unlock persists for the browser session. When no secret is configured,
    access stays open with a setup hint (contests.db is local-only anyway).
    """
    try:
        key = st.secrets.get("ADMIN_PASSWORD", None)
    except Exception:
        key = None
    if not key:
        st.caption("🔓 No ADMIN_PASSWORD secret configured — page is open. "
                   "Add one in .streamlit/secrets.toml (and the cloud app's "
                   "secrets) to lock it down.")
        return True
    if st.session_state.get("contests_admin_ok"):
        return True
    st.markdown("#### 🔒 Personal results")
    with st.form("contests_admin_gate"):
        pw = st.text_input("Admin password", type="password",
                           label_visibility="collapsed",
                           placeholder="Admin password...")
        if st.form_submit_button("Unlock"):
            if pw == str(key):
                st.session_state["contests_admin_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    return False


def _model_vs_me(view):
    """What the app's optimal lineup would have scored in the user's
    contests — the model graded against the user's real entries.

    Conservative winnings floor: within each contest the user's own entries
    are (score -> winnings) calibration points; the model lineup 'wins' the
    payout of the best user entry it outscores. When it outscores all of
    them it still only gets the best entry's payout, and when it outscores
    none it gets $0 — both floors, never optimistic.
    """
    from src.contests import race_day_index
    from src.profit_sim import simulate_race

    st.markdown("**Model vs Me** — the app's max-projection lineup replayed "
                "into your real contests")
    idx = race_day_index()
    if idx.empty:
        st.info("No race index available.")
        return
    v = view.dropna(subset=["Race"]).copy()
    v["_date"] = v["contest_date"].astype(str).str.slice(0, 10)
    lut = {(r["date"], r["series"]): r for _, r in idx.iterrows()}
    day_keys = sorted({(r["_date"], r["series"]) for _, r in v.iterrows()
                       if (r["_date"], r["series"]) in lut}, reverse=True)
    if not day_keys:
        st.info("No linked race days to analyze yet.")
        return

    n_days = st.slider("Race days to analyze (newest first — each is a full "
                       "engine replay, so more = slower first run)",
                       2, min(20, len(day_keys)), min(6, len(day_keys)),
                       key="mvm_days")
    rows = []
    prog = st.progress(0.0, text="Replaying model lineups...")
    for i, dk in enumerate(day_keys[:n_days]):
        race = lut[dk]
        ck = f"mvm2_{race['db_id']}"          # v2: results include the field
        if ck not in st.session_state:
            try:
                st.session_state[ck] = simulate_race(
                    int(race["db_id"]), int(race["api_id"]),
                    int(race["season"]), race["date"], race["track"],
                    race["race_name"],
                    {"Cup": 1, "O'Reilly": 2, "Truck": 3}[race["series"]],
                    platform="DraftKings", return_field=True)
            except Exception:
                st.session_state[ck] = None
        sim = st.session_state[ck]
        prog.progress((i + 1) / n_days, text=f"Replaying {race['track']}...")

        mine = v[(v["_date"] == dk[0]) & (v["series"] == dk[1])]
        my_fees = float(mine["entry_fee"].sum())
        my_win = float(mine["winnings"].sum())
        my_best = float(mine["points"].max()) if mine["points"].notna().any() else None
        row = {
            "Race Day": f"{dk[0]} — {race['track']} ({dk[1]})",
            "My Entries": len(mine),
            "My Best": my_best, "My %ile": None,
            "Model FPTS": None, "Model %ile": None,
            "Model GPP Best": None, "GPP %ile": None,
            "My Net": round(my_win - my_fees, 2),
            "Model Net*": None, "_model_fees": None,
            "Model > Me": None,
        }
        if sim:
            model_score = sim["cash_score"]
            row["Model FPTS"] = model_score
            row["Model GPP Best"] = sim["gpp_best"]
            # Field percentiles: everyone ranked against the SAME simulated
            # 1,000-lineup public field — independent of my entries entirely.
            row["Model %ile"] = sim.get("cash_pctile")
            row["GPP %ile"] = sim.get("gpp_best_pctile")
            _field = sim.get("field") or []
            if _field and my_best is not None:
                import bisect
                row["My %ile"] = round(
                    100.0 * bisect.bisect_left(_field, my_best) / len(_field), 1)
            # ONE model entry per unique contest — multi-entering the same
            # lineup isn't real play, and scaling by the user's entry count
            # would weight the model's result by the user's bullet count.
            # Payout stays the conservative same-contest calibration floor.
            est_win = est_fees = 0.0
            for _, grp in mine.groupby("contest_key"):
                est_fees += float(grp["entry_fee"].iloc[0])
                pts = grp.dropna(subset=["points"])
                beaten = pts[pts["points"] <= model_score]
                if not beaten.empty:
                    est_win += float(beaten["winnings"].max())
            row["Model Net*"] = round(est_win - est_fees, 2)
            row["_model_fees"] = round(est_fees, 2)
            row["Model > Me"] = ("✅" if my_best is not None
                                 and model_score > my_best else "—")
        rows.append(row)
    prog.empty()

    mdf = pd.DataFrame(rows)
    st.dataframe(_style_money(
        mdf.drop(columns=["_model_fees"]), {"My Net", "Model Net*"}).format(
        {"My Best": "{:.1f}", "Model FPTS": "{:.1f}",
         "Model GPP Best": "{:.1f}", "My %ile": "{:.1f}",
         "Model %ile": "{:.1f}", "GPP %ile": "{:.1f}"}, na_rep="—"),
        width="stretch", hide_index=True)

    done = mdf.dropna(subset=["Model Net*"])
    if not done.empty:
        my_total = done["My Net"].sum()
        model_total = done["Model Net*"].sum()
        beat = (done["Model > Me"] == "✅").sum()
        # Fee bases differ (I multi-enter; the model plays each contest once),
        # so dollars aren't directly comparable — ROI on each side's own fees is.
        _v2 = view[view["contest_date"].astype(str).str.slice(0, 10).isin(
            {r["Race Day"][:10] for _, r in done.iterrows()})]
        my_fees_total = float(_v2["entry_fee"].sum()) or 1.0
        model_fees_total = float(done["_model_fees"].sum()) or 1.0
        my_roi = 100 * my_total / my_fees_total
        model_roi = 100 * model_total / model_fees_total
        _pct_cards = []
        _my_p = done["My %ile"].dropna()
        _mo_p = done["Model %ile"].dropna()
        if not _my_p.empty and not _mo_p.empty:
            _pct_cards.append(_card(
                "Avg Field %ile", f"{_mo_p.mean():.0f} vs {_my_p.mean():.0f}",
                "model vs my best — same simulated field", "#a78bfa"))
        st.markdown(_row([
            _card("My Net", f"${my_total:+,.2f}",
                  f"{my_roi:+.1f}% ROI on ${my_fees_total:,.0f} "
                  f"· {len(done)} days", _pl_color(my_total)),
            _card("Model Net (floor)", f"${model_total:+,.2f}",
                  f"{model_roi:+.1f}% ROI on ${model_fees_total:,.0f} "
                  "· 1 entry per contest", _pl_color(model_total)),
            _card("Model beat my best", f"{beat}/{len(done)}",
                  "days the model outscored me", "#2dd4bf"),
        ] + _pct_cards), unsafe_allow_html=True)
    st.caption("*Model Net assumes ONE model entry per unique contest you "
               "entered (multi-entering an identical lineup isn't real play), "
               "and is a conservative FLOOR: within each contest your own "
               "entries are the score→payout calibration; the model lineup "
               "only claims a payout it provably beat, and claims $0 when it "
               "outscored none of your entries. Compare the ROI figures — the "
               "dollar bases differ since you multi-enter. **%ile columns** "
               "rank scores against the same simulated 1,000-lineup public "
               "field (higher = better), independent of your entries — the "
               "cleanest my-best vs model read. Days showing — couldn't be "
               "replayed (missing archived salaries/odds). Model lineup = "
               "the engine's max-projection build from pre-race data only.")


def render(*, series_name="Cup"):
    section_header("My Contests", "Real-money ROI from DraftKings entry history")
    if not _admin_gate():
        return

    # ── Import ─────────────────────────────────────────────────────────
    with st.expander("Import entry history", expanded=False):
        st.caption(
            "On [draftkings.com/mycontests](https://www.draftkings.com/mycontests) "
            "→ **History** → **Download Entry History**. The export contains "
            "your FULL account history regardless of the page's 30-day view "
            "filter. NASCAR rows only are kept; re-imports are deduped, so "
            "download fresh anytime. Stored in a local private ledger "
            "(contests.db) — never committed to the repo."
        )
        candidates = find_entry_history_csvs()
        if candidates:
            pick = st.selectbox(
                "Found in Downloads/Desktop", candidates[:5],
                format_func=lambda p: os.path.basename(p), key="contest_csv_pick")
            if st.button("Import selected file", key="contest_import_btn"):
                try:
                    parsed = parse_dk_entry_history(pick)
                    added, skipped = import_entries(parsed)
                    st.success(f"Imported {added} new entries "
                               f"({skipped} already stored).")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        up = st.file_uploader("…or drop the CSV here", type=["csv"],
                              key="contest_csv_upload")
        if up is not None:
            try:
                parsed = parse_dk_entry_history(up)
                added, skipped = import_entries(parsed)
                st.success(f"Imported {added} new entries ({skipped} already stored).")
            except ValueError as e:
                st.error(str(e))

    df = load_entries()
    if df.empty:
        st.info("No contest entries stored yet — import your DraftKings entry "
                "history above and the ROI dashboard appears here.")
        return
    df = attach_races(df)
    df["_dt"] = pd.to_datetime(df["contest_date"], errors="coerce")

    # ── Filters: range + style + series ────────────────────────────────
    fcols = st.columns([1.2, 1, 1, 2])
    with fcols[0]:
        range_pick = st.selectbox(
            "Range", ["All Time", "Last 30 Days", "Last 90 Days",
                      "This Year", "Last Year"], key="contest_range")
    with fcols[1]:
        style_pick = st.selectbox("Style", ["All", "Cash", "GPP", "Qualifier"],
                                  key="contest_style_filter")
    with fcols[2]:
        _series_opts = ["All"] + sorted(
            {s for s in df["series"].fillna("") if s} | {"?"})
        series_pick = st.selectbox("Series", _series_opts,
                                   key="contest_series_filter")

    view = df.copy()
    now = pd.Timestamp.now()
    if range_pick == "Last 30 Days":
        view = view[view["_dt"] >= now - pd.Timedelta(days=30)]
    elif range_pick == "Last 90 Days":
        view = view[view["_dt"] >= now - pd.Timedelta(days=90)]
    elif range_pick == "This Year":
        view = view[view["_dt"].dt.year == now.year]
    elif range_pick == "Last Year":
        view = view[view["_dt"].dt.year == now.year - 1]
    if style_pick != "All":
        view = view[view["style"] == style_pick]
    if series_pick != "All":
        if series_pick == "?":
            view = view[view["series"].fillna("").eq("")]
        else:
            view = view[view["series"] == series_pick]
    if view.empty:
        st.info("No entries match the current filters.")
        return

    # ── Headline cards ─────────────────────────────────────────────────
    fees = float(view["entry_fee"].sum())
    winnings = float(view["winnings"].sum())
    net = round(winnings - fees, 2)
    roi = (100 * net / fees) if fees else 0.0
    cash_rate = (100 * (view["winnings"] > 0).mean()) if len(view) else 0.0
    n_days = view["_dt"].dt.date.nunique()
    st.markdown(_row([
        _card("Entries", f"{len(view):,}", f"{n_days} race days"),
        _card("Fees", f"${fees:,.2f}", "total buy-ins", "#fb923c"),
        _card("Winnings", f"${winnings:,.2f}", "incl. ticket value", "#a78bfa"),
        _card("Net P/L", f"${net:+,.2f}", "winnings − fees", _pl_color(net)),
        _card("ROI", f"{roi:+.1f}%", "net / fees", _pl_color(net)),
        _card("Cash Rate", f"{cash_rate:.0f}%", "entries that won", "#2dd4bf"),
    ]), unsafe_allow_html=True)

    # ── Insights ───────────────────────────────────────────────────────
    notes = _insights(view)
    if notes:
        st.markdown("  \n".join(notes))

    # ── Cumulative P/L over race days (event days only) ────────────────
    ts = view.dropna(subset=["_dt"]).sort_values("_dt")
    if len(ts) >= 2:
        gcols = st.columns([1, 4])
        with gcols[0]:
            gran = st.selectbox("Group by", ["Day", "Week", "Month"],
                                key="contest_granularity")
        if gran == "Week":
            ts["_p"] = ts["_dt"].dt.to_period("W").dt.start_time.dt.strftime("wk %Y-%m-%d")
        elif gran == "Month":
            ts["_p"] = ts["_dt"].dt.strftime("%Y-%m")
        else:
            ts["_p"] = ts["_dt"].dt.strftime("%Y-%m-%d")
        per = ts.groupby("_p", sort=True).apply(
            lambda g: g["winnings"].sum() - g["entry_fee"].sum(),
            include_groups=False)
        cum = per.cumsum()

        import plotly.graph_objects as go
        from src.charts import DARK_LAYOUT, apply_dark_theme
        fig = go.Figure()
        fig.add_bar(x=list(per.index), y=per.values, name=f"Net per {gran.lower()}",
                    marker_color=["#22c55e" if v >= 0 else "#ef4444"
                                  for v in per.values], opacity=0.55,
                    hovertemplate="%{x}<br>Net: $%{y:,.2f}<extra></extra>")
        fig.add_scatter(x=list(cum.index), y=cum.values, mode="lines+markers",
                        name="Cumulative",
                        line=dict(color="#38bdf8", width=2), marker=dict(size=5),
                        hovertemplate="%{x}<br>Cumulative: $%{y:,.2f}<extra></extra>")
        fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
        fig.update_layout(**DARK_LAYOUT, height=340,
                          title="Profit / Loss over race days",
                          yaxis_title="Net $",
                          xaxis=dict(type="category", tickangle=-45),
                          legend=dict(orientation="h", y=1.08))
        apply_dark_theme(fig)
        st.plotly_chart(fig, width="stretch", key="contest_cum_pl")

    # ── Breakdowns ─────────────────────────────────────────────────────
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**By Contest Style**")
        st.dataframe(_style_money(
            _breakdown(view, "style").rename(columns={"style": "Style"}),
            {"Net", "ROI %"}), width="stretch", hide_index=True)
        st.markdown("**By Series**")
        st.dataframe(_style_money(
            _breakdown(view, "series").rename(columns={"series": "Series"}),
            {"Net", "ROI %"}), width="stretch", hide_index=True)
    with b2:
        st.markdown("**By Track Type**")
        from src.config import TRACK_TYPE_MAP, TRACK_TYPE_DISPLAY
        vt = view.copy()
        vt["Track Type"] = vt["Track"].map(
            lambda t: TRACK_TYPE_DISPLAY.get(TRACK_TYPE_MAP.get(t, ""),
                                             TRACK_TYPE_MAP.get(t, "?"))
            if t else "?")
        st.dataframe(_style_money(
            _breakdown(vt[vt["Track Type"] != "?"], "Track Type"),
            {"Net", "ROI %"}), width="stretch", hide_index=True)
        st.markdown("**By Month**")
        vm = view.copy()
        vm["Month"] = vm["_dt"].dt.strftime("%Y-%m")
        st.dataframe(_style_money(
            _breakdown(vm.dropna(subset=["Month"]), "Month")
            .sort_values("Month", ascending=False),
            {"Net", "ROI %"}), width="stretch", hide_index=True)

    # ── By Race (entries linked to nascar.db by date + series) ─────────
    st.markdown("**By Race** — your real results per event")
    vr = view.copy()
    vr["_race_lbl"] = vr.apply(
        lambda r: (f"{str(r['contest_date'])[:10]} — "
                   f"{r['Track'] or '?'} ({r['series'] or '?'})"), axis=1)
    g = vr.groupby("_race_lbl").agg(
        Entries=("entry_key", "count"),
        Fees=("entry_fee", "sum"),
        Winnings=("winnings", "sum"),
        BestFPTS=("points", "max"),
    ).reset_index().rename(columns={"_race_lbl": "Race Day"})
    g["Net"] = (g["Winnings"] - g["Fees"]).round(2)
    g["ROI %"] = (100 * g["Net"] / g["Fees"].replace(0, pd.NA)).round(1)
    g["Fees"] = g["Fees"].round(2)
    g["Winnings"] = g["Winnings"].round(2)
    g["BestFPTS"] = g["BestFPTS"].round(1)
    g = g.sort_values("Race Day", ascending=False)
    st.dataframe(_style_money(g, {"Net", "ROI %"}).format(
        {"BestFPTS": "{:.1f}"}, na_rep="—"), width="stretch",
        hide_index=True, height=350)
    st.caption("Grade the model's projections for any of these races on the "
               "**Accuracy** page (Race Comparison) — this table is your real-"
               "money result; that page is why it happened.")

    # ── Model vs Me ────────────────────────────────────────────────────
    st.divider()
    with st.expander("Model vs Me — replay the app's lineups into my contests",
                     expanded=False):
        _model_vs_me(view)

    # ── Entry log ──────────────────────────────────────────────────────
    with st.expander(f"Entry log ({len(view):,} entries)", expanded=False):
        show = view[["contest_date", "contest_name", "Track", "series", "style",
                     "place", "field_entries", "points", "entry_fee",
                     "winnings"]].copy()
        show["Net"] = (show["winnings"] - show["entry_fee"]).round(2)
        # Finish percentile: 1 = won the contest, 100 = dead last.
        show["Finish %ile"] = (100 * show["place"] /
                               show["field_entries"].replace(0, pd.NA)).round(1)
        show = show.rename(columns={
            "contest_date": "Date", "contest_name": "Contest",
            "series": "Series", "style": "Style", "place": "Place",
            "field_entries": "Field", "points": "FPTS", "entry_fee": "Fee",
            "winnings": "Won"})
        show["Date"] = show["Date"].astype(str).str.slice(0, 10)
        st.dataframe(safe_fillna(show), width="stretch", hide_index=True,
                     height=420)

    st.caption("Data source: DraftKings 'Download Entry History' export — "
               "DK has no public API, so this CSV is the sanctioned pipe. "
               "Winnings include ticket value (qualifiers pay tickets).")
