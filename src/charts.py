"""NASCAR DFS Hub — Chart Builders (Plotly)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DARK_LAYOUT = dict(
    template="plotly_dark",
    margin=dict(l=30, r=20, t=40, b=30),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,38,0.9)",
    font=dict(color="#c9d1d9"),
)


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

    fig = go.Figure(go.Bar(
        y=df["Driver"],
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
    return fig


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
    df["Label"] = df["Driver"].where(df["Driver"].isin(labeled_drivers), "")

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
    return fig


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

    fig = go.Figure(go.Bar(
        y=chart_df["Driver"],
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
    return fig


def track_history_bar(hist_df: pd.DataFrame, track_name: str,
                      top_n: int = 25, height: int = 250) -> go.Figure:
    """Bar chart of top drivers by avg finish at a track."""
    if "Avg Finish" not in hist_df.columns:
        return None
    fig = px.bar(hist_df.head(top_n), x="Driver", y="Avg Finish",
                 color="Avg Rating", color_continuous_scale="RdYlGn",
                 title=f"Top {top_n} by Avg Finish at {track_name}")
    fig.update_layout(**DARK_LAYOUT, height=height)
    return fig


def projection_bar(proj_df: pd.DataFrame, top_n: int = 20, height: int = 280) -> go.Figure:
    """Bar chart of top projected scores."""
    fig = px.bar(proj_df.head(top_n), x="Driver", y="Proj Score",
                 color="Proj Score", color_continuous_scale="Viridis",
                 title=f"Top {top_n} by Projected Score")
    fig.update_layout(**DARK_LAYOUT, height=height)
    return fig


def practice_lap_chart(practice_laps: list, height: int = 400) -> go.Figure:
    """Line chart showing each driver's practice lap times.

    Args:
        practice_laps: list of {driver: str, laps: [{lap_num, lap_time}]}
    """
    if not practice_laps:
        return None

    fig = go.Figure()
    all_times = []

    for entry in practice_laps:
        driver = entry["driver"]
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
            name=f"{driver} — avg {avg_time:.2f} • best {best_time:.2f} • {n_laps} laps",
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
    return fig


def season_scatter(season_data: pd.DataFrame, height: int = 500) -> go.Figure:
    """Avg Running Position vs Avg Driver Rating scatter with car numbers.

    Expects columns: Driver, Car, Avg Running Pos, Avg Driver Rating
    """
    if season_data.empty:
        return None

    fig = go.Figure()

    # Compute quadrant lines
    x_med = season_data["Avg Running Pos"].median()
    y_med = season_data["Avg Driver Rating"].median()

    fig.add_trace(go.Scatter(
        x=season_data["Avg Running Pos"],
        y=season_data["Avg Driver Rating"],
        mode="markers+text",
        text=season_data["Car"].astype(str),
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        marker=dict(size=28, color="#1e293b", line=dict(width=1.5, color="#64748b")),
        hovertemplate="%{customdata[0]}<br>Avg Run: %{x:.1f}<br>Rating: %{y:.1f}<extra></extra>",
        customdata=season_data[["Driver"]].values,
    ))

    # Quadrant lines
    fig.add_hline(y=y_med, line_dash="dash", line_color="#334155", line_width=1)
    fig.add_vline(x=x_med, line_dash="dash", line_color="#334155", line_width=1)

    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
        title="Season Overview: Avg Running Position vs Driver Rating",
        xaxis_title="Avg Running Pos",
        yaxis_title="Avg Driver Rating",
        xaxis=dict(autorange="reversed"),  # Lower running pos = better = right side
    )
    return fig


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
    return fig_obj


def rating_vs_finish_scatter(hist_df: pd.DataFrame, track_name: str = "",
                              height: int = 450) -> go.Figure:
    """Avg Driver Rating vs Avg Finish scatter for track history."""
    if "Avg Rating" not in hist_df.columns or "Avg Finish" not in hist_df.columns:
        return None

    clean = hist_df.dropna(subset=["Avg Rating", "Avg Finish"]).copy()
    if clean.empty:
        return None

    fig = go.Figure(go.Scatter(
        x=clean["Avg Finish"],
        y=clean["Avg Rating"],
        mode="markers+text",
        text=clean["Driver"],
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(
            size=14,
            color=clean["Avg Rating"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Rating"),
            line=dict(width=1, color="#334155"),
        ),
        hovertemplate="%{text}<br>Avg Finish: %{x:.1f}<br>Rating: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        height=height,
        title=f"Avg Driver Rating vs Avg Finish{' — ' + track_name if track_name else ''}",
        xaxis_title="Avg Finish Position",
        yaxis_title="Avg Driver Rating",
        xaxis=dict(autorange="reversed"),  # Lower finish = better = right side
    )
    return fig


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
    return fig
