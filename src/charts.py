"""NASCAR DFS Hub — Chart Builders (Plotly)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import short_name_series

_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}

def _last_name(full_name: str) -> str:
    """Extract last name for chart labels, handling suffixes like Jr./Sr."""
    parts = full_name.split()
    if len(parts) >= 2 and parts[-1].lower().rstrip(".") in {"jr", "sr", "ii", "iii", "iv"}:
        return parts[-2]
    return parts[-1] if parts else full_name


DARK_LAYOUT = dict(
    template="plotly_dark",
    margin=dict(l=30, r=20, t=40, b=30),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.95)",
    font=dict(color="#e2e8f0", family="system-ui, -apple-system, sans-serif"),
    colorway=["#0ea5e9", "#38bdf8", "#7dd3fc", "#22d3ee", "#2dd4bf",
              "#4ade80", "#a78bfa", "#f472b6", "#fb923c", "#facc15"],
)

_GRID_STYLE = dict(gridcolor="#1e293b", zerolinecolor="#334155")


def apply_dark_theme(fig):
    """Apply DARK_LAYOUT + grid colors to a figure. Call after update_layout."""
    fig.update_xaxes(**_GRID_STYLE)
    fig.update_yaxes(**_GRID_STYLE)
    return fig


def dfs_histogram(results_df: pd.DataFrame, height: int = 400) -> go.Figure:
    """DFS points bar chart — each driver sorted by points with score breakdown on hover."""
    from src.utils import calc_dk_points
    from src.config import DK_FINISH_POINTS

    df = results_df.dropna(subset=["DFS Points"]).sort_values("DFS Points", ascending=True).copy()
    if df.empty:
        return go.Figure()

    # Compute DK point components for hover breakdown
    hover_texts = []
    for _, row in df.iterrows():
        fp = int(row.get("Finish Position", 0))
        start = int(row.get("Start", 0))
        ll = int(row.get("Laps Led", 0))
        fl = int(row.get("Fastest Laps", 0))
        finish_pts = DK_FINISH_POINTS.get(fp, 0)
        diff_pts = (start - fp) * 1.0
        led_pts = ll * 0.25
        fl_pts = fl * 0.45
        total = finish_pts + diff_pts + led_pts + fl_pts
        hover_texts.append(
            f"<b>{row['Driver']}</b><br>"
            f"DK Pts: {total:.1f}<br>"
            f"─────────────<br>"
            f"Finish: P{fp} ({finish_pts} pts)<br>"
            f"Start: P{start} → P{fp} ({diff_pts:+.1f} pts)<br>"
            f"Laps Led: {ll} ({led_pts:.1f} pts)<br>"
            f"Fastest Laps: {fl} ({fl_pts:.1f} pts)"
        )

    short_names = short_name_series(df["Driver"].tolist())

    fig = go.Figure(go.Bar(
        y=short_names,
        x=df["DFS Points"],
        orientation="h",
        marker=dict(
            color=df["DFS Points"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="DK Pts"),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=max(height, len(df) * 16),
        title="DFS Points by Driver",
        xaxis_title="DraftKings Points",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=10)),
    )
    return apply_dark_theme(fig)


def start_vs_finish_scatter(results_df: pd.DataFrame, height: int = 500) -> go.Figure:
    """Start vs Finish Position scatter plot with smart labeling."""
    df = results_df.copy()
    df = df.dropna(subset=["Start", "Finish Position", "DFS Points"])
    if df.empty:
        return go.Figure()

    # Label top 15 and bottom 5 DFS scorers; leave rest as hover-only
    sorted_by_pts = df.sort_values("DFS Points", ascending=False)
    labeled_drivers = set(
        sorted_by_pts.head(15)["Driver"].tolist() +
        sorted_by_pts.tail(5)["Driver"].tolist()
    )
    from src.utils import short_name
    all_names = df["Driver"].tolist()
    df["Label"] = df["Driver"].apply(
        lambda d: short_name(d, all_names) if d in labeled_drivers else ""
    )

    fig = px.scatter(df, x="Start", y="Finish Position",
                     text="Label", color="DFS Points",
                     color_continuous_scale="Viridis",
                     hover_data={"Driver": True, "DFS Points": ":.1f",
                                 "Start": True, "Finish Position": True, "Label": False},
                     title="Start vs Finish Position")
    fig.add_trace(go.Scatter(x=[1, 40], y=[1, 40], mode="lines",
                             line=dict(dash="dash", color="gray"), showlegend=False))
    fig.update_layout(**DARK_LAYOUT, height=height,
                      yaxis=dict(autorange="reversed"))
    fig.update_traces(textposition="top right", textfont_size=10)
    return apply_dark_theme(fig)


def practice_bar_chart(lap_averages_df: pd.DataFrame, metric_col: str = "Overall Avg",
                       height: int = 400) -> go.Figure:
    """Bar chart showing gap-to-fastest for ALL drivers at a given lap interval.

    Args:
        lap_averages_df: Practice data with driver names and lap time columns
        metric_col: Which column to chart (e.g. "Overall Avg", "5 Lap", "10 Lap")
    """
    if metric_col not in lap_averages_df.columns:
        return None

    chart_df = lap_averages_df[["Driver", metric_col]].copy()
    chart_df[metric_col] = pd.to_numeric(chart_df[metric_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[metric_col])
    if chart_df.empty:
        return None

    # Sort by time (fastest first)
    chart_df = chart_df.sort_values(metric_col).reset_index(drop=True)

    # Compute gap from fastest
    best_time = chart_df[metric_col].min()
    chart_df["Delta"] = chart_df[metric_col] - best_time

    # Dynamic height based on number of drivers
    bar_height = max(height, len(chart_df) * 18)

    chart_df["Short"] = short_name_series(chart_df["Driver"].tolist())

    fig = go.Figure(go.Bar(
        y=chart_df["Short"],
        x=chart_df["Delta"],
        orientation="h",
        marker=dict(
            color=chart_df[metric_col],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Time (s)"),
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{metric_col}: " + "%{customdata[0]:.3f}s<br>"
            "Gap: +%{x:.3f}s<extra></extra>"
        ),
        customdata=chart_df[[metric_col]].values,
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=bar_height,
        title=f"{metric_col} — Gap to Fastest (Lower = Better)",
        xaxis_title="Seconds Behind Fastest",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    return apply_dark_theme(fig)


def track_history_bar(hist_df: pd.DataFrame, track_name: str,
                      top_n: int = 25, height: int = 250) -> go.Figure:
    """Bar chart of top drivers by avg finish at a track."""
    if "Avg Finish" not in hist_df.columns:
        return None
    # Use Avg Run Pos for color if available, else just Avg Finish
    color_col = "Avg Run Pos" if "Avg Run Pos" in hist_df.columns and hist_df["Avg Run Pos"].notna().any() else "Avg Finish"
    fig = px.bar(hist_df.head(top_n), x="Driver", y="Avg Finish",
                 color=color_col, color_continuous_scale="RdYlGn_r",
                 title=f"Top {top_n} by Avg Finish at {track_name}")
    fig.update_layout(**DARK_LAYOUT, height=height)
    return apply_dark_theme(fig)


def projection_bar(proj_df: pd.DataFrame, top_n: int = 20, height: int = 280) -> go.Figure:
    """Bar chart of top projected scores."""
    fig = px.bar(proj_df.head(top_n), x="Driver", y="Proj Score",
                 color="Proj Score", color_continuous_scale="Viridis",
                 title=f"Top {top_n} by Projected Score")
    fig.update_layout(**DARK_LAYOUT, height=height)
    return apply_dark_theme(fig)


def practice_lap_chart(practice_laps: list, height: int = 400) -> go.Figure:
    """Line chart showing each driver's practice lap times.

    Args:
        practice_laps: list of {driver: str, laps: [{lap_num, lap_time}]}
    """
    if not practice_laps:
        return None

    fig = go.Figure()
    all_times = []

    from src.utils import short_name
    all_driver_names = [e["driver"] for e in practice_laps]

    for entry in practice_laps:
        driver = entry["driver"]
        short = short_name(driver, all_driver_names)
        laps = sorted(entry["laps"], key=lambda x: x["lap_num"])
        lap_nums = [l["lap_num"] for l in laps]
        lap_times = [l["lap_time"] for l in laps]
        all_times.extend(lap_times)

        avg_time = np.mean(lap_times)
        best_time = min(lap_times)
        n_laps = len(laps)

        fig.add_trace(go.Scatter(
            x=lap_nums, y=lap_times,
            mode="lines",
            name=f"{short} — avg {avg_time:.2f} • best {best_time:.2f} • {n_laps} laps",
            hovertemplate=f"{driver}<br>Lap %{{x}}: %{{y:.3f}}s<extra></extra>",
        ))

    # Field average line
    if all_times:
        field_avg = np.mean(all_times)
        fig.add_hline(y=field_avg, line_dash="dash", line_color="gray",
                      annotation_text=f"Avg {field_avg:.2f}s",
                      annotation_position="right")

    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
        title="Practice Lap Times",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        legend=dict(
            orientation="h", yanchor="top", y=-0.15,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    return apply_dark_theme(fig)


def season_scatter(season_data: pd.DataFrame, height: int = 500) -> go.Figure:
    """Avg Running Position vs Avg DK Points scatter with car numbers.

    Expects columns: Driver, Car, Avg Running Pos, Avg DK Pts
    """
    if season_data.empty:
        return None

    y_col = "Avg DK Pts" if "Avg DK Pts" in season_data.columns else None
    if y_col is None:
        return None

    fig = go.Figure()

    # Compute quadrant lines
    x_med = season_data["Avg Running Pos"].median()
    y_med = season_data[y_col].median()

    fig.add_trace(go.Scatter(
        x=season_data["Avg Running Pos"],
        y=season_data[y_col],
        mode="markers+text",
        text=season_data["Car"].astype(str),
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        marker=dict(size=28, color="#1e293b", line=dict(width=1.5, color="#64748b")),
        hovertemplate="%{customdata[0]}<br>Avg Run: %{x:.1f}<br>DK Pts: %{y:.1f}<extra></extra>",
        customdata=season_data[["Driver"]].values,
    ))

    # Quadrant lines
    fig.add_hline(y=y_med, line_dash="dash", line_color="#334155", line_width=1)
    fig.add_vline(x=x_med, line_dash="dash", line_color="#334155", line_width=1)

    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
        title="Season Overview: Avg Running Position vs DK Points",
        xaxis_title="Avg Running Pos",
        yaxis_title="Avg DK Pts",
        xaxis=dict(autorange="reversed"),
    )
    return apply_dark_theme(fig)


def race_scatter(results_df: pd.DataFrame, height: int = 350) -> go.Figure:
    """Avg Running Position vs DK Points scatter for a single race."""
    if "Avg Run" not in results_df.columns or "DK Pts" not in results_df.columns:
        return None

    clean = results_df.dropna(subset=["Avg Run", "DK Pts"]).copy()
    if clean.empty:
        return None

    fig = go.Scatter(
        x=clean["Avg Run"], y=clean["DK Pts"],
        mode="markers+text",
        text=clean["Car"].astype(str) if "Car" in clean.columns else clean["Driver"],
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=12, color=clean["DK Pts"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="DK Pts")),
        hovertemplate="%{customdata[0]}<br>Avg Run: %{x:.1f}<br>DK Pts: %{y:.1f}<extra></extra>",
        customdata=clean[["Driver"]].values,
    )

    fig_obj = go.Figure(data=[fig])
    fig_obj.update_layout(
        **DARK_LAYOUT,
        height=height,
        title="Avg Running Position vs DK Points",
        xaxis_title="Avg Running Pos",
        yaxis_title="DK Points",
        xaxis=dict(autorange="reversed"),
    )
    return apply_dark_theme(fig_obj)


def arp_vs_finish_scatter(hist_df: pd.DataFrame, track_name: str = "",
                           height: int = 450) -> go.Figure:
    """Avg Running Position vs Avg Finish scatter — shows wreck luck."""
    if "Avg Run Pos" not in hist_df.columns or "Avg Finish" not in hist_df.columns:
        return None

    clean = hist_df.dropna(subset=["Avg Run Pos", "Avg Finish"]).copy()
    if clean.empty or len(clean) < 3:
        return None

    # Color by how much ARP differs from finish (luck factor)
    clean["Luck"] = clean["Avg Run Pos"] - clean["Avg Finish"]

    fig = go.Figure(go.Scatter(
        x=clean["Avg Finish"],
        y=clean["Avg Run Pos"],
        mode="markers+text",
        text=clean["Driver"],
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(
            size=14,
            color=clean["Luck"],
            colorscale="RdYlGn",
            cmid=0,
            showscale=True,
            colorbar=dict(title="Luck<br>(ARP-Finish)"),
            line=dict(width=1, color="#334155"),
        ),
        hovertemplate="%{text}<br>Avg Finish: %{x:.1f}<br>Avg Run Pos: %{y:.1f}"
                      "<br>Luck: %{marker.color:.1f}<extra></extra>",
    ))

    # Add diagonal reference line (ARP = Finish)
    max_val = max(clean["Avg Finish"].max(), clean["Avg Run Pos"].max()) + 2
    fig.add_trace(go.Scatter(
        x=[1, max_val], y=[1, max_val],
        mode="lines", line=dict(dash="dash", color="#475569", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
        title=f"Avg Running Position vs Avg Finish{' — ' + track_name if track_name else ''}",
        xaxis_title="Avg Finish Position",
        yaxis_title="Avg Running Position",
        xaxis=dict(autorange="reversed"),
        yaxis=dict(autorange="reversed"),
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text="Above line = unlucky (ran better than finished)<br>"
             "Below line = lucky (finished better than ran)",
        showarrow=False, font=dict(size=10, color="#64748b"),
        align="left",
    )
    return apply_dark_theme(fig)


def salary_vs_projection_scatter(pool_df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Salary vs Projected Points scatter — highlights value plays."""
    if "DK Salary" not in pool_df.columns or "Proj Score" not in pool_df.columns:
        return None
    clean = pool_df.dropna(subset=["DK Salary", "Proj Score"]).copy()
    if clean.empty:
        return None

    # Value line: avg pts per $1k across the field
    avg_value = clean["Proj Score"].sum() / (clean["DK Salary"].sum() / 1000)

    fig = go.Figure()

    # Value line (no legend, no hover)
    sal_range = [clean["DK Salary"].min(), clean["DK Salary"].max()]
    fig.add_shape(type="line",
        x0=sal_range[0], y0=sal_range[0] / 1000 * avg_value,
        x1=sal_range[1], y1=sal_range[1] / 1000 * avg_value,
        line=dict(dash="dash", color="#475569", width=1),
    )

    # Driver dots
    value_col = clean["Value"] if "Value" in clean.columns else clean["Proj Score"]
    fig.add_trace(go.Scatter(
        x=clean["DK Salary"], y=clean["Proj Score"],
        mode="markers+text",
        name="",
        text=clean["Driver"].apply(_last_name),
        textposition="top center",
        textfont=dict(size=8, color="#94a3b8"),
        marker=dict(
            size=12,
            color=value_col,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Value"),
            line=dict(width=1, color="#334155"),
        ),
        hovertemplate="%{text}<br>Salary: $%{x:,}<br>Proj: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT, height=height,
        title="Salary vs Projected Points (above line = good value)",
        xaxis_title="DK Salary", yaxis_title="Projected Points",
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text="Above line = better value than average",
        showarrow=False, font=dict(size=9, color="#64748b"),
    )
    return apply_dark_theme(fig)


def finish_distribution_box(track_name: str, series_id: int = 1,
                             top_n: int = 20, height: int = 400) -> go.Figure:
    """Box plot of finish positions at a track — shows consistency vs boom/bust."""
    import sqlite3
    from src.config import DB_PATH
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query('''
            SELECT d.full_name as Driver, rr.finish_pos as Finish
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE t.name = ? AND r.series_id = ? AND r.season >= 2022
        ''', conn, params=[track_name, series_id])
        conn.close()
    except Exception:
        return None

    if df.empty or len(df) < 5:
        return None

    # Only show drivers with 3+ races
    counts = df.groupby("Driver").size()
    regulars = counts[counts >= 3].index
    df = df[df["Driver"].isin(regulars)]

    # Sort by median finish (best first)
    medians = df.groupby("Driver")["Finish"].median().sort_values()
    top_drivers = medians.head(top_n).index.tolist()
    df = df[df["Driver"].isin(top_drivers)]

    # Order by median
    df["Driver"] = pd.Categorical(df["Driver"], categories=top_drivers, ordered=True)

    fig = px.box(df, x="Driver", y="Finish",
                 title=f"Finish Distribution at {track_name} (2022+)")
    fig.update_layout(**DARK_LAYOUT, height=height,
                      yaxis=dict(autorange="reversed"),
                      xaxis_tickangle=-45)
    return apply_dark_theme(fig)


def fantasy_vs_arp_scatter(track_name: str, series_id: int = 1,
                            height: int = 450) -> go.Figure:
    """Avg DK Fantasy Points vs Avg Running Position at a track — shows value."""
    import sqlite3
    from src.config import DB_PATH
    from src.utils import calc_dk_points
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute('''
            SELECT d.full_name, rr.finish_pos, rr.start_pos,
                   rr.laps_led, rr.fastest_laps, rr.avg_running_position
            FROM race_results rr
            JOIN drivers d ON d.id = rr.driver_id
            JOIN races r ON r.id = rr.race_id
            JOIN tracks t ON t.id = r.track_id
            WHERE t.name = ? AND r.series_id = ? AND r.season >= 2022
              AND rr.finish_pos IS NOT NULL
        ''', [track_name, series_id]).fetchall()
        conn.close()
    except Exception:
        return None

    if not rows or len(rows) < 5:
        return None

    # Compute DK points per race-driver, then aggregate
    from collections import defaultdict
    driver_pts = defaultdict(list)
    driver_arp = defaultdict(list)
    for name, fp, sp, ll, fl, arp in rows:
        dk = calc_dk_points(fp, sp or fp, ll or 0, fl or 0)
        driver_pts[name].append(dk)
        if arp and arp > 0:
            driver_arp[name].append(arp)

    records = []
    for name in driver_pts:
        if len(driver_pts[name]) >= 2 and driver_arp.get(name):
            records.append({
                "Driver": name,
                "Avg DK Pts": round(np.mean(driver_pts[name]), 1),
                "Avg Run Pos": round(np.mean(driver_arp[name]), 1),
                "Races": len(driver_pts[name]),
            })

    if len(records) < 3:
        return None

    df = pd.DataFrame(records)

    fig = go.Figure(go.Scatter(
        x=df["Avg Run Pos"], y=df["Avg DK Pts"],
        mode="markers+text",
        text=df["Driver"].apply(_last_name),
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(
            size=df["Races"].clip(upper=12) + 6,
            color=df["Avg DK Pts"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Avg DK"),
            line=dict(width=1, color="#334155"),
        ),
        hovertemplate="%{customdata[0]}<br>Avg Run Pos: %{x:.1f}<br>"
                      "Avg DK Pts: %{y:.1f}<br>Races: %{customdata[1]}<extra></extra>",
        customdata=df[["Driver", "Races"]].values,
    ))

    fig.update_layout(
        **DARK_LAYOUT, height=height,
        title=f"Avg Fantasy Points vs Avg Running Position — {track_name}",
        xaxis_title="Avg Running Position",
        yaxis_title="Avg DK Points",
        xaxis=dict(autorange="reversed"),
    )
    fig.add_annotation(
        x=0.02, y=0.02, xref="paper", yref="paper",
        text="Bubble size = number of races",
        showarrow=False, font=dict(size=9, color="#64748b"),
    )
    return apply_dark_theme(fig)


def season_trend_line(series_id: int = 1, season: int = None,
                       drivers: list = None, top_n: int = 10,
                       last_n_races: int = 5,
                       height: int = 400) -> go.Figure:
    """Line chart of DK points across the most recent N races — shows form trends.

    Uses a categorical x-axis with "Track Name (Date)" labels so the chart
    renders correctly regardless of how the race_date column is stored
    (avoids plotly's datetime-axis collapse when dates are sparse/identical).
    """
    import sqlite3
    from datetime import datetime
    from src.config import DB_PATH
    if season is None:
        _t = datetime.now()
        season = _t.year + 1 if _t.month >= 10 else _t.year
    if not DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query('''
            SELECT d.full_name as Driver,
                   r.id          as RaceId,
                   r.race_name   as Race,
                   r.track_name  as Track,
                   r.race_date   as Date,
                   dp.dfs_score  as DK_Pts
            FROM dfs_points dp
            JOIN drivers d ON d.id = dp.driver_id
            JOIN races   r ON r.id = dp.race_id
            WHERE r.series_id = ? AND r.season = ? AND dp.platform = 'DraftKings'
            ORDER BY r.race_date, r.id
        ''', conn, params=[series_id, season])
        conn.close()
    except Exception:
        return None

    if df.empty:
        return None

    # Identify the most recent N races (by race_date, with race_id as tiebreaker
    # for same-day double-headers).
    race_order = (df[["RaceId", "Race", "Track", "Date"]]
                  .drop_duplicates("RaceId")
                  .sort_values(["Date", "RaceId"]))
    recent_races = race_order.tail(last_n_races).reset_index(drop=True)
    if recent_races.empty:
        return None

    df = df[df["RaceId"].isin(recent_races["RaceId"])]
    if df.empty:
        return None

    # Build "Track (M/D)" label per race, ordered chronologically.
    def _fmt_date(d):
        try:
            return pd.to_datetime(d).strftime("%-m/%-d") if d else ""
        except Exception:
            try:
                return pd.to_datetime(d).strftime("%#m/%#d") if d else ""
            except Exception:
                return str(d) if d else ""

    recent_races["Label"] = recent_races.apply(
        lambda r: f"{(r['Track'] or r['Race'] or 'Race').split(' (')[0]}"
                  + (f" ({_fmt_date(r['Date'])})" if r['Date'] else ""),
        axis=1,
    )
    label_order = recent_races["Label"].tolist()
    label_map = dict(zip(recent_races["RaceId"], recent_races["Label"]))
    df = df.assign(Label=df["RaceId"].map(label_map))

    # Pick top drivers by avg DK pts (over the recent window) if none specified.
    if not drivers:
        avg_pts = df.groupby("Driver")["DK_Pts"].mean().sort_values(ascending=False)
        drivers = avg_pts.head(top_n).index.tolist()
    df = df[df["Driver"].isin(drivers)]
    if df.empty:
        return None

    # Sort so each driver's line connects in chronological race order.
    df["_x_idx"] = df["Label"].apply(lambda lbl: label_order.index(lbl) if lbl in label_order else -1)
    df = df.sort_values(["Driver", "_x_idx"])

    n_races = len(label_order)
    title_suffix = f" — Last {n_races} Race{'s' if n_races != 1 else ''}"
    fig = px.line(df, x="Label", y="DK_Pts", color="Driver",
                  markers=True,
                  title=f"{season} Season DK Points Trend{title_suffix}",
                  hover_data={"Race": True, "Label": False, "_x_idx": False})
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=label_order)
    fig.update_layout(**DARK_LAYOUT, height=height,
                      xaxis_title="", yaxis_title="DK Points",
                      legend=dict(font=dict(size=9)))
    return apply_dark_theme(fig)


def race_speed_chart(lap_data: dict, selected_drivers: list = None,
                     height: int = 520, green_only: bool = False,
                     metric: str = "time") -> go.Figure:
    """Overlay each selected driver's lap TIME (default) or SPEED across the race.

    Args:
        lap_data: raw lap-times.json dict ("laps" + "flags").
        selected_drivers: driver full names to plot (None/empty = none).
        green_only: if True, drop caution laps AND green-flag PIT laps so only
            clean racing pace remains. A pit lap is a green lap far slower than
            the driver's own green-lap median (they stopped on that lap).
        metric: "time" -> lap time (s, lower=faster) or "speed" -> mph.
    """
    if not lap_data or "laps" not in lap_data or not selected_drivers:
        return None
    use_time = metric != "speed"
    field = "LapTime" if use_time else "LapSpeed"

    change = {f["LapsCompleted"]: f["FlagState"]
              for f in (lap_data.get("flags") or [])
              if "LapsCompleted" in f and "FlagState" in f}

    fig = go.Figure()
    plotted = False
    for d in lap_data["laps"]:
        driver = d.get("FullName", d.get("NickName", "Unknown"))
        if driver not in selected_drivers:
            continue
        # First pass: collect (lap, value, flagstate).
        seq = []
        cur = 0
        for lap in d.get("Laps", []):
            ln = lap.get("Lap", 0)
            if ln in change:
                cur = change[ln]
            val = lap.get(field)
            if not ln or not val:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            seq.append((ln, val, cur))
        if not seq:
            continue
        # Pit-lap filter for green-only: drop green laps that are clearly a pit
        # lap (slower-than-median pace). For TIME, pit laps are LARGER; for
        # SPEED, smaller. Use the driver's green-lap median as the reference.
        if green_only:
            greens = [v for _, v, c in seq if c == 1]
            ref = None
            if greens:
                gs = sorted(greens); n = len(gs)
                ref = gs[n // 2] if n % 2 else (gs[n // 2 - 1] + gs[n // 2]) / 2
            # A green-flag pit stop is the obvious slow lap (the pit-lane lap, far
            # off pace) PLUS the in-lap (slowing to enter) and out-lap (getting
            # back up to speed) on either side. Those neighbours are only ~5-10%
            # off median, so a single threshold misses them and they show up as
            # sharp V-spikes. Detect the obvious pit lap, then drop it and its
            # immediate neighbours; a tighter backstop catches any stray out-lap.
            pit_lns = set()
            if ref is not None:
                for ln, v, c in seq:
                    if c != 1:
                        continue
                    if use_time and v > ref * 1.10:
                        pit_lns.add(ln)
                    elif (not use_time) and v < ref * 0.90:
                        pit_lns.add(ln)
            drop_lns = set()
            for ln in pit_lns:
                drop_lns.update((ln - 1, ln, ln + 1))
            kept = []
            for ln, v, c in seq:
                if c != 1:
                    continue  # caution lap
                if ln in drop_lns:
                    continue  # pit lap or its in-/out-lap
                if ref is not None:  # backstop: any remaining off-pace lap
                    if use_time and v > ref * 1.05:
                        continue
                    if (not use_time) and v < ref * 0.95:
                        continue
                kept.append((ln, v))
            seq = kept
        else:
            seq = [(ln, v) for ln, v, c in seq]
        if not seq:
            continue
        plotted = True
        xs = [s[0] for s in seq]
        ys = [s[1] for s in seq]
        unit = "s" if use_time else "mph"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=_last_name(driver),
            hovertemplate=f"{driver}<br>Lap %{{x}}: %{{y:.2f}} {unit}<extra></extra>",
            connectgaps=True,
        ))
    if not plotted:
        return None

    # Shade caution segments (only meaningful when not green-only).
    if not green_only and change:
        pts = sorted(change.items())
        for i, (lap, state) in enumerate(pts):
            if state == 2:
                end = pts[i + 1][0] if i + 1 < len(pts) else lap + 1
                fig.add_vrect(x0=lap, x1=end, fillcolor="#fbbf24",
                              opacity=0.10, line_width=0, layer="below")

    suffix = " (green-flag racing laps only)" if green_only else ""
    fig.update_layout(
        **DARK_LAYOUT, height=height,
        title=("Lap Time" if use_time else "Lap Speed") + " Over the Race" + suffix,
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)" if use_time else "Lap Speed (mph)",
        legend=dict(orientation="h", yanchor="top", y=-0.12, font=dict(size=10)),
        hovermode="x unified",
    )
    # For lap TIME, lower is faster — invert the y-axis so "up = faster" reads
    # intuitively like the speed chart.
    if use_time:
        fig.update_yaxes(autorange="reversed")
    return apply_dark_theme(fig)


def race_lap_chart(lap_data: dict, selected_drivers: list = None,
                   height: int = 500) -> go.Figure:
    """Line chart showing each driver's race lap times (from lap-times.json).

    Args:
        lap_data: raw lap-times.json dict with "laps" key
        selected_drivers: list of driver names to include (None = all)
    """
    if not lap_data or "laps" not in lap_data:
        return None

    fig = go.Figure()
    all_times = []

    for d in lap_data["laps"]:
        driver = d.get("FullName", d.get("NickName", "Unknown"))
        if selected_drivers and driver not in selected_drivers:
            continue

        laps = []
        for lap in d.get("Laps", []):
            ln = lap.get("Lap", 0)
            lt = lap.get("LapTime", 0)
            if ln > 0 and lt and lt > 0:
                laps.append((ln, lt))

        if not laps:
            continue

        laps.sort(key=lambda x: x[0])
        lap_nums = [l[0] for l in laps]
        lap_times = [l[1] for l in laps]
        all_times.extend(lap_times)

        avg_time = np.mean(lap_times)
        best_time = min(lap_times)

        fig.add_trace(go.Scatter(
            x=lap_nums, y=lap_times,
            mode="lines",
            name=f"{driver} — avg {avg_time:.2f} • best {best_time:.2f}",
            hovertemplate=f"{driver}<br>Lap %{{x}}: %{{y:.3f}}s<extra></extra>",
        ))

    if not fig.data:
        return None

    if all_times:
        field_avg = np.mean(all_times)
        fig.add_hline(y=field_avg, line_dash="dash", line_color="gray",
                      annotation_text=f"Avg {field_avg:.2f}s",
                      annotation_position="right")

    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
        title="Race Lap Times",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        legend=dict(
            orientation="h", yanchor="top", y=-0.1,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )
    return apply_dark_theme(fig)
