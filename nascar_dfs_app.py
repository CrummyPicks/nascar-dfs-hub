# ============================================================
# NASCAR DFS DATA HUB v5.1
# Top-bar navigation, modular architecture
# ============================================================

from datetime import datetime
import streamlit as st
import pandas as pd

try:
    from src.config import (
        SERIES_OPTIONS, SERIES_LABELS, TRACK_TYPE_MAP, TRACK_TYPE_COLORS,
    )
    from src.data import (
        fetch_race_list, fetch_weekend_feed, fetch_lap_times, fetch_lap_averages,
        extract_entry_list, extract_qualifying, extract_race_results,
        compute_fastest_laps, detect_prerace, filter_point_races,
        parse_dk_csv, parse_fd_csv, fetch_dk_salaries_live,
        sync_dk_salaries_to_db, fetch_nascar_odds, save_odds_to_db,
        estimate_odds_from_salaries, _clean_api_name,
    )
except ImportError as e:
    import streamlit as st
    st.error(f"Import failed: {e}")
    st.stop()


# ============================================================
# PAGE CONFIG & CSS
# ============================================================
st.set_page_config(page_title="NASCAR DFS Hub", page_icon="🏁", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""<style>
/* ── Base layout ── */
.block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; max-width: 1600px; }
[data-testid="collapsedControl"] { display: none; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: #1a1f2e; border: 1px solid #2d3548; border-radius: 8px; padding: 8px 12px;
}
div[data-testid="stMetric"] label { color: #8892a4 !important; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #c9d1d9 !important; font-size: 1.1rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #161b26; border-radius: 8px; padding: 3px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; padding: 6px 16px;
    font-weight: 500; color: #7c8599; font-size: 0.85rem; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: #252d3d; color: #e6edf3; }

/* ── Expanders ── */
.streamlit-expanderHeader { border-radius: 6px; font-size: 0.85rem; }
div[data-testid="stExpander"] { margin-bottom: 0.3rem; }

/* ── Spacing ── */
.element-container { margin-bottom: 0.2rem; }
div[data-baseweb="select"] ul { max-height: 300px !important; overflow-y: auto !important; }
div[data-baseweb="popover"] { max-height: 350px !important; overflow-y: auto !important; }

/* ── Remove underlines everywhere ── */
.stMarkdown a, .stCaption a { text-decoration: none !important; }
h1, h2, h3, h4, h5 { text-decoration: none !important; border-bottom: none !important; }
.stMarkdown h3 { border-bottom: none !important; }
[data-testid="stMarkdownContainer"] h3 { border-bottom: none !important; text-decoration: none !important; }
[data-testid="stMarkdownContainer"] { text-decoration: none !important; }
hr { border-color: #2d3548 !important; }
/* ── Accent color overrides (supplements .streamlit/config.toml primaryColor) ── */
[data-baseweb="tab-highlight"] { background-color: #4a7dfc !important; }
[data-baseweb="input"] [data-baseweb="base-input"] { border-color: #2d3548 !important; }
[data-baseweb="input"]:focus-within [data-baseweb="base-input"] { border-color: #4a7dfc !important; }
.stDownloadButton > button { border-color: #4a7dfc !important; color: #4a7dfc !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .block-container { padding-top: 0.5rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 4px 8px; font-size: 0.7rem; }
    div[data-testid="stMetric"] { padding: 4px 6px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 0.9rem; }
    h1 { font-size: 1.1rem !important; }
}
@media (max-width: 480px) {
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { padding: 3px 6px; font-size: 0.65rem; }
}
</style>""", unsafe_allow_html=True)

# Header
st.markdown("""<div style='text-align:center; padding:0.6rem 1rem; border-bottom:2px solid #4a7dfc;
  margin-bottom:0.5rem; background: linear-gradient(135deg, #1a1f2e, #161b26);'>
  <h1 style='color:#c9d1d9; margin:0; font-size:1.4rem; line-height:1.3; letter-spacing:1px;
  font-weight:600;'>NASCAR DFS Hub</h1>
</div>""", unsafe_allow_html=True)


# ============================================================
# TOP NAVIGATION BAR (replaces sidebar)
# ============================================================

nav_cols = st.columns([1, 1, 4])

with nav_cols[0]:
    series_name = st.selectbox("Series", list(SERIES_OPTIONS.keys()), key="series_select",
                               label_visibility="collapsed")
    series_id = SERIES_OPTIONS[series_name]

with nav_cols[1]:
    selected_year = st.selectbox("Season", [2026, 2025, 2024, 2023, 2022], key="year_select",
                                 label_visibility="collapsed")

# Fetch race list
races = fetch_race_list(series_id, selected_year)

if races:
    point_races = filter_point_races(races)

    # Classify completed vs upcoming (for default selection and data flow)
    now = datetime.now()
    completed_races = []
    upcoming_races = []
    for race_num_idx, race in enumerate(point_races):
        race_date_str = race.get("race_date", "")
        try:
            race_date = datetime.fromisoformat(
                race_date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
            if race_date.date() <= now.date():
                completed_races.append((race_num_idx, race))
            else:
                upcoming_races.append((race_num_idx, race))
        except Exception:
            upcoming_races.append((race_num_idx, race))

    # Build ALL race labels (no completed/upcoming split)
    race_map = {}
    labels = []
    for race_num_idx, race in enumerate(point_races):
        race_date_str = race.get("race_date", "")
        date_label = ""
        if race_date_str:
            try:
                rd = datetime.fromisoformat(
                    race_date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
                date_label = f" — {rd.strftime('%m/%d')}"
            except Exception:
                pass
        track_short = race.get("track_name", "")
        label = f"R{race_num_idx + 1}{date_label} @ {track_short}: {race.get('race_name', 'Unknown')}"
        labels.append(label)
        race_map[label] = (race_num_idx, race)

    # Default to most recent completed race, or first upcoming
    if completed_races:
        default_idx = completed_races[-1][0]
    elif upcoming_races:
        default_idx = upcoming_races[0][0]
    else:
        default_idx = 0

    with nav_cols[2]:
        selected_label = st.selectbox("Race", labels, index=default_idx,
                                      label_visibility="collapsed")
        race_idx, selected_race = race_map[selected_label]

    race_id = selected_race.get("race_id")
    race_name = selected_race.get("race_name", "Unknown Race")
    track_name = selected_race.get("track_name", "Unknown Track")
    track_type = TRACK_TYPE_MAP.get(track_name, "intermediate")
    scheduled_laps = selected_race.get("scheduled_laps", 0) or 0
else:
    st.warning("Could not fetch race list from API")
    race_id, race_name = 5596, "Daytona 500"
    track_name = "Daytona International Speedway"
    track_type = "superspeedway"
    scheduled_laps = 200
    completed_races = []
    upcoming_races = []

# ============================================================
# SETTINGS EXPANDER (DK/FD upload, weights)
# ============================================================
with st.expander("Settings & Data Upload", expanded=False):
    # ── Auto-fetch status row ──────────────────────────────────────────────
    auto_odds = fetch_nascar_odds()
    dk_auto = fetch_dk_salaries_live()

    # Persist last good odds — never lose data from a failed refresh
    if auto_odds:
        st.session_state["last_good_odds"] = auto_odds
    elif "last_good_odds" in st.session_state:
        auto_odds = st.session_state["last_good_odds"]

    # Status summary at top
    status_parts = []
    if not dk_auto.empty:
        status_parts.append(f"✅ DK Salary: {len(dk_auto)} drivers")
    else:
        status_parts.append("⚠️ DK Salary: unavailable")
    if auto_odds:
        status_parts.append(f"✅ Odds: {len(auto_odds)} drivers")
    else:
        status_parts.append("⚠️ Odds: unavailable")
    st.caption(" | ".join(status_parts))

    # Refresh All button
    ref_cols = st.columns([1, 1, 4])
    with ref_cols[0]:
        if st.button("Refresh All Data", key="refresh_all_btn", type="primary"):
            fetch_nascar_odds.clear()
            fetch_dk_salaries_live.clear()
            fresh_odds = fetch_nascar_odds()
            fresh_dk = fetch_dk_salaries_live()
            msgs = []
            if fresh_odds:
                auto_odds = fresh_odds
                st.session_state["last_good_odds"] = fresh_odds
                msgs.append(f"Odds: {len(fresh_odds)} drivers")
            else:
                msgs.append("Odds: failed")
            if not fresh_dk.empty:
                dk_auto = fresh_dk
                msgs.append(f"DK Salary: {len(fresh_dk)} drivers")
            else:
                msgs.append("DK Salary: failed")
            st.success(f"Refreshed — {' | '.join(msgs)}")

    s_cols = st.columns([1, 1, 1, 1])
    with s_cols[0]:
        st.markdown("**DK Salary**")
        dk_file = st.file_uploader("DK CSV", type=["csv"], label_visibility="collapsed", key="dk_upload")
        if not dk_auto.empty:
            st.caption(f"Auto: {len(dk_auto)} drivers from DK API")
    with s_cols[1]:
        st.markdown("**FD Salary CSV**")
        fd_file = st.file_uploader("FD", type=["csv"], label_visibility="collapsed", key="fd_upload")
    with s_cols[2]:
        st.markdown("**Manual Practice**")
        practice_text = st.text_area("Practice", placeholder="Chase Elliott, 3\nDenny Hamlin, 5",
                                     height=80, label_visibility="collapsed")
        practice_data = {}
        if practice_text.strip():
            for line in practice_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        practice_data[parts[0]] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
    with s_cols[3]:
        st.markdown("**Betting Odds**")
        # Action Network only has Cup odds — don't show for other series
        is_cup = (series_id == 1)
        if is_cup and auto_odds:
            st.caption(f"✅ Auto: {len(auto_odds)} drivers from Action Network")
        elif not is_cup:
            st.caption(f"⚠️ Auto odds not available for {series_name} series (Cup only)")
        else:
            st.caption("⚠️ No odds — Action Network may be down, or no upcoming race listed")
        odds_text = st.text_area("Odds", placeholder="Chase Elliott, +1200\nDenny Hamlin, +800",
                                 height=80, label_visibility="collapsed",
                                 help="Paste to override auto-fetched odds (American format)")
        odds_data = {}
        odds_source = ""
        # Manual text overrides auto if provided
        if odds_text.strip():
            for line in odds_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        odds_data[parts[0]] = parts[1]
                    except (ValueError, IndexError):
                        pass
            odds_source = "manual"
        elif is_cup and auto_odds:
            # Only use auto-fetched odds for Cup series
            odds_data = auto_odds
            odds_source = "action_network"
        # Fallback: estimate odds from DK salary when no real odds available
        if not odds_data and not dk_auto.empty:
            odds_data = estimate_odds_from_salaries(dk_auto)
            if odds_data:
                odds_source = "salary_estimate"
                st.caption("📊 Using salary-estimated odds (Action Network unavailable)")

    # Clean odds keys to match driver names from API (Jr. -> Jr, etc.)
    if odds_data:
        odds_data = {_clean_api_name(k): v for k, v in odds_data.items()}

    # Persist odds to DB for historical backtesting
    if odds_data and race_id:
        save_odds_to_db(odds_data, race_id)


# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Loading data..."):
    feed = fetch_weekend_feed(series_id, race_id, selected_year)
    lap_data = fetch_lap_times(series_id, race_id, selected_year)
    lap_averages_df = fetch_lap_averages(series_id, race_id, selected_year)

is_prerace = detect_prerace(feed)

results_df = extract_race_results(feed) if feed and not is_prerace else pd.DataFrame()
fl_counts = compute_fastest_laps(lap_data) if lap_data and not is_prerace else {}

qualifying_df = extract_qualifying(feed) if feed else pd.DataFrame()
entry_list_df = extract_entry_list(feed) if feed else pd.DataFrame()

# Auto-pull practice data from lap averages
if not lap_averages_df.empty and "Overall Rank" in lap_averages_df.columns and not practice_data:
    for _, row in lap_averages_df.iterrows():
        driver = row.get("Driver")
        rank = row.get("Overall Rank")
        if driver and rank and not pd.isna(rank):
            practice_data[driver] = int(rank)

# Parse salary CSVs (CSV upload overrides auto-fetch)
if dk_file:
    dk_df = parse_dk_csv(dk_file)
elif not dk_auto.empty:
    dk_df = dk_auto[dk_auto["Status"] != "Out"][["Driver", "DK Salary"]].copy()
else:
    dk_df = pd.DataFrame()
fd_df = parse_fd_csv(fd_file) if fd_file else pd.DataFrame()

# Auto-sync DK salaries to DB for projection engine and historical tracking
if not dk_df.empty:
    synced = sync_dk_salaries_to_db(dk_auto if not dk_auto.empty else dk_df,
                                     race_id, series_id, race_name)


# ============================================================
# TABS
# ============================================================
tab_data, tab_practice, tab_history, tab_race_analyzer, tab_proj, tab_optimizer, tab_acc = st.tabs([
    "Race Data", "Practice", "Track History", "Race Analyzer", "Projections", "Optimizer", "Accuracy"
])

from tabs import tab_data as td
from tabs import tab_practice as tp
from tabs import tab_track_history as tth
from tabs import tab_race_analyzer as tra
from tabs import tab_projections as tproj
from tabs import tab_optimizer as topt
from tabs import tab_accuracy as tacc

with tab_data:
    td.render(
        feed=feed, lap_data=lap_data, lap_averages_df=lap_averages_df,
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        results_df=results_df, is_prerace=is_prerace,
        series_id=series_id, race_name=race_name, track_name=track_name,
        track_type=track_type, dk_df=dk_df, fd_df=fd_df,
        completed_races=completed_races, selected_year=selected_year,
        fl_counts=fl_counts, odds_data=odds_data,
    )

with tab_practice:
    tp.render(
        lap_averages_df=lap_averages_df, feed=feed,
        race_name=race_name, series_id=series_id,
        race_id=race_id, selected_year=selected_year,
    )

with tab_history:
    tth.render(
        track_name=track_name, track_type=track_type, series_id=series_id,
    )

with tab_race_analyzer:
    tra.render(
        completed_races=completed_races, series_id=series_id,
        selected_year=selected_year, series_name=series_name,
    )

with tab_proj:
    tproj.render(
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        lap_averages_df=lap_averages_df, practice_data=practice_data,
        is_prerace=is_prerace, race_name=race_name, race_id=race_id,
        track_name=track_name, series_id=series_id, dk_df=dk_df,
        odds_data=odds_data, scheduled_laps=scheduled_laps,
    )

with tab_optimizer:
    topt.render(
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        lap_averages_df=lap_averages_df, practice_data=practice_data,
        is_prerace=is_prerace, race_name=race_name, race_id=race_id,
        track_name=track_name, series_id=series_id, dk_df=dk_df,
        odds_data=odds_data,
    )

with tab_acc:
    tacc.render(
        completed_races=completed_races, series_id=series_id,
        selected_year=selected_year, series_name=series_name,
    )
