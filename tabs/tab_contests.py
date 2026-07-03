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


def render(*, series_name="Cup"):
    section_header("My Contests", "Real-money ROI from DraftKings entry history")

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
    st.dataframe(_style_money(g, {"Net", "ROI %"}), width="stretch",
                 hide_index=True, height=350)
    st.caption("Grade the model's projections for any of these races on the "
               "**Accuracy** page (Race Comparison) — this table is your real-"
               "money result; that page is why it happened.")

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
