"""My Contests — real-money ROI tracking from DraftKings entry history.

Import the 'Download Entry History' CSV from draftkings.com/mycontests
(auto-detected in Downloads, or drag-and-drop) and see where the bankroll
actually grows: ROI by contest style, series, and month, plus a cumulative
profit line. This is the ground truth the projections/optimizer work is
ultimately graded against.
"""

import os

import pandas as pd
import streamlit as st

from src.components import section_header
from src.contests import (find_entry_history_csvs, import_entries,
                          load_entries, parse_dk_entry_history)
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


def render(*, series_name="Cup"):
    section_header("My Contests", "Real-money ROI from DraftKings entry history")

    # ── Import ─────────────────────────────────────────────────────────
    with st.expander("Import entry history", expanded=False):
        st.caption(
            "On [draftkings.com/mycontests](https://www.draftkings.com/mycontests) "
            "→ **History** → **Download Entry History**. The CSV lands in "
            "Downloads and is auto-detected below (NASCAR rows only are kept; "
            "re-imports are deduped automatically, so download fresh anytime)."
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

    # ── Filters ────────────────────────────────────────────────────────
    fcols = st.columns([1, 1, 2])
    with fcols[0]:
        style_pick = st.selectbox("Style", ["All", "Cash", "GPP", "Qualifier"],
                                  key="contest_style_filter")
    with fcols[1]:
        _series_opts = ["All"] + sorted(
            {s for s in df["series"].fillna("") if s} | {"?"})
        series_pick = st.selectbox("Series", _series_opts,
                                   key="contest_series_filter")
    view = df.copy()
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
    st.markdown(_row([
        _card("Entries", f"{len(view):,}",
              f"{view['contest_name'].nunique()} contests"),
        _card("Fees", f"${fees:,.2f}", "total buy-ins", "#fb923c"),
        _card("Winnings", f"${winnings:,.2f}", "total returned", "#a78bfa"),
        _card("Net P/L", f"${net:+,.2f}", "winnings − fees", _pl_color(net)),
        _card("ROI", f"{roi:+.1f}%", "net / fees", _pl_color(net)),
        _card("Cash Rate", f"{cash_rate:.0f}%", "entries that won $", "#2dd4bf"),
    ]), unsafe_allow_html=True)

    # ── Cumulative P/L over time ───────────────────────────────────────
    ts = view.dropna(subset=["contest_date"]).copy()
    ts["_d"] = pd.to_datetime(ts["contest_date"], errors="coerce")
    ts = ts.dropna(subset=["_d"]).sort_values("_d")
    if len(ts) >= 2:
        daily = ts.groupby(ts["_d"].dt.date).apply(
            lambda g: g["winnings"].sum() - g["entry_fee"].sum(),
            include_groups=False).cumsum()
        import plotly.graph_objects as go
        from src.charts import DARK_LAYOUT, apply_dark_theme
        fig = go.Figure(go.Scatter(
            x=list(daily.index), y=daily.values, mode="lines+markers",
            line=dict(color="#38bdf8", width=2), marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
            hovertemplate="%{x}<br>Cumulative: $%{y:,.2f}<extra></extra>"))
        fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
        fig.update_layout(**DARK_LAYOUT, height=300,
                          title="Cumulative Profit / Loss",
                          yaxis_title="Net $", xaxis_title=None)
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
        st.markdown("**By Month**")
        vm = view.copy()
        vm["Month"] = pd.to_datetime(
            vm["contest_date"], errors="coerce").dt.strftime("%Y-%m")
        st.dataframe(_style_money(
            _breakdown(vm.dropna(subset=["Month"]), "Month")
            .sort_values("Month", ascending=False),
            {"Net", "ROI %"}), width="stretch", hide_index=True)

    # ── Entry log ──────────────────────────────────────────────────────
    st.markdown("**Entries** (newest first)")
    show = view[["contest_date", "contest_name", "series", "style", "place",
                 "field_entries", "points", "entry_fee", "winnings"]].copy()
    show["Net"] = (show["winnings"] - show["entry_fee"]).round(2)
    show = show.rename(columns={
        "contest_date": "Date", "contest_name": "Contest", "series": "Series",
        "style": "Style", "place": "Place", "field_entries": "Field",
        "points": "FPTS", "entry_fee": "Fee", "winnings": "Won"})
    show["Date"] = show["Date"].astype(str).str.slice(0, 10)
    st.dataframe(safe_fillna(show), width="stretch", hide_index=True, height=420)

    st.caption("Data source: DraftKings 'Download Entry History' export — "
               "DK has no public API, so this CSV is the sanctioned pipe. "
               "FanDuel import can be added the same way (their history page "
               "has a similar export).")
