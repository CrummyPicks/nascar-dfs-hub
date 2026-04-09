# ============================================================
# NASCAR DFS DATA HUB v5.1
# Top-bar navigation, modular architecture
# ============================================================

from datetime import datetime
import streamlit as st
import pandas as pd

try:
    from src.config import (
        SERIES_OPTIONS, SERIES_LABELS, TRACK_TYPE_MAP, TRACK_TYPE_COLORS, DB_PATH,
    )
    from src.data import (
        fetch_race_list, fetch_weekend_feed, fetch_lap_times, fetch_lap_averages,
        extract_entry_list, extract_qualifying, extract_race_results,
        compute_fastest_laps, detect_prerace, filter_point_races,
        parse_dk_csv, parse_fd_csv, fetch_dk_salaries_live,
        sync_dk_salaries_to_db, sync_fd_salaries_to_db,
        fetch_nascar_odds, save_odds_to_db,
        estimate_odds_from_salaries, _clean_api_name,
        fetch_nascar_prop_odds, load_race_prop_odds, load_race_odds,
        _fetch_all_nascar_odds, query_salaries,
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
    race_date_raw = (selected_race.get("race_date") or "")[:10]  # YYYY-MM-DD
else:
    st.warning("Could not fetch race list from API")
    race_id, race_name = 5596, "Daytona 500"
    track_name = "Daytona International Speedway"
    track_type = "superspeedway"
    scheduled_laps = 200
    race_date_raw = ""
    completed_races = []
    upcoming_races = []

# ============================================================
# SETTINGS EXPANDER (DK/FD upload, weights)
# ============================================================
with st.expander("Settings & Data Upload", expanded=False):
    # ── Admin authentication ──────────────────────────────────────────────
    _admin_pw = st.secrets.get("ADMIN_PASSWORD", "") if hasattr(st, "secrets") else ""
    is_admin = False
    if _admin_pw:
        pw_input = st.text_input("Admin Password", type="password", key="admin_pw",
                                 placeholder="Enter password to enable uploads")
        is_admin = (pw_input == _admin_pw)
        if pw_input and not is_admin:
            st.error("Incorrect password")
    else:
        # No password configured — allow everything (local dev)
        is_admin = True

    # ── Auto-fetch status row ──────────────────────────────────────────────
    auto_odds = fetch_nascar_odds()
    dk_auto = fetch_dk_salaries_live(series_id=series_id)

    # Persist last good odds — never lose data from a failed refresh
    if auto_odds:
        st.session_state["last_good_odds"] = auto_odds
    elif "last_good_odds" in st.session_state:
        auto_odds = st.session_state["last_good_odds"]

    # Status summary at top
    status_parts = []
    if not dk_auto.empty:
        status_parts.append(f"DK Salary: {len(dk_auto)} drivers")
    else:
        status_parts.append("DK Salary: unavailable")
    if auto_odds:
        status_parts.append(f"Odds: {len(auto_odds)} drivers")
    else:
        status_parts.append("Odds: unavailable")
    st.caption(" | ".join(status_parts))

    # Check for saved salaries in DB for this race
    db_dk_df = query_salaries(race_id=race_id, platform="DraftKings")
    db_fd_df = query_salaries(race_id=race_id, platform="FanDuel")
    has_saved_dk = not db_dk_df.empty
    has_saved_fd = not db_fd_df.empty

    # ── Admin-only controls (upload, refresh, clear) ──────────────────────
    dk_file = None
    fd_file = None
    practice_data = {}
    odds_data = {}
    odds_source = ""
    is_cup = (series_id == 1)

    if is_admin:
        # Refresh All button
        ref_cols = st.columns([1, 1, 4])
        with ref_cols[0]:
            if st.button("Refresh All Data", key="refresh_all_btn", type="primary"):
                _fetch_all_nascar_odds.clear()
                fetch_dk_salaries_live.clear()
                fresh_odds = fetch_nascar_odds()
                fresh_dk = fetch_dk_salaries_live(series_id=series_id)
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
            if has_saved_dk:
                st.caption(f"Saved: {len(db_dk_df)} drivers in DB for this race")
            dk_file = st.file_uploader("DK CSV", type=["csv"], label_visibility="collapsed",
                                       key=f"dk_upload_{race_id}")
            if dk_file:
                st.caption("CSV uploaded — will save to DB")
            elif not dk_auto.empty and not has_saved_dk:
                st.caption(f"Auto: {len(dk_auto)} drivers from DK API")
            # Clear button for saved salaries
            if has_saved_dk:
                if st.button("Clear DK Salaries", key=f"clear_dk_{race_id}"):
                    try:
                        import sqlite3 as _sql
                        _conn = _sql.connect(str(DB_PATH))
                        _db_race = _conn.execute(
                            "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
                        ).fetchone()
                        if _db_race:
                            _conn.execute(
                                "DELETE FROM salaries WHERE race_id = ? AND platform = 'DraftKings'",
                                (_db_race[0],))
                            _conn.commit()
                        _conn.close()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear salaries: {e}")
        with s_cols[1]:
            st.markdown("**FD Salary CSV**")
            if has_saved_fd:
                st.caption(f"Saved: {len(db_fd_df)} drivers in DB for this race")
            fd_file = st.file_uploader("FD", type=["csv"], label_visibility="collapsed",
                                       key=f"fd_upload_{race_id}")
            if fd_file:
                st.caption("CSV uploaded — will save to DB")
            if has_saved_fd:
                if st.button("Clear FD Salaries", key=f"clear_fd_{race_id}"):
                    try:
                        import sqlite3 as _sql
                        _conn = _sql.connect(str(DB_PATH))
                        _db_race = _conn.execute(
                            "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
                        ).fetchone()
                        if _db_race:
                            _conn.execute(
                                "DELETE FROM salaries WHERE race_id = ? AND platform = 'FanDuel'",
                                (_db_race[0],))
                            _conn.commit()
                        _conn.close()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear salaries: {e}")
        with s_cols[2]:
            st.markdown("**Manual Practice**")
            practice_text = st.text_area("Practice", placeholder="Chase Elliott, 3\nDenny Hamlin, 5",
                                         height=80, label_visibility="collapsed")
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
            if is_cup and auto_odds:
                st.caption(f"Auto: {len(auto_odds)} drivers from Action Network")
            elif not is_cup:
                st.caption(f"Auto odds not available for {series_name} series (Cup only)")
            else:
                st.caption("No odds — Action Network may be down, or no upcoming race listed")
            odds_text = st.text_area("Odds", placeholder="Chase Elliott, +1200\nDenny Hamlin, +800",
                                     height=80, label_visibility="collapsed",
                                     help="Paste to override auto-fetched odds (American format)")
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
    else:
        # Read-only view for non-admin users
        st.caption("Read-only mode — enter admin password to upload data")
        if has_saved_dk:
            st.caption(f"DK Salary: {len(db_dk_df)} drivers saved for this race")
        if has_saved_fd:
            st.caption(f"FD Salary: {len(db_fd_df)} drivers saved for this race")

    # Auto odds (available to everyone, read-only)
    if not odds_data and is_cup and auto_odds:
        odds_data = auto_odds
        odds_source = "action_network"

    # Fallback: estimate odds from DK salary when no real odds available
    _salary_for_odds = dk_auto if not dk_auto.empty else (
        db_dk_df.rename(columns={"Salary": "DK Salary"})[["Driver", "DK Salary"]]
        if has_saved_dk else pd.DataFrame()
    )
    if not odds_data and not _salary_for_odds.empty:
        odds_data = estimate_odds_from_salaries(_salary_for_odds)
        if odds_data:
            odds_source = "salary_estimate"
            reason = "Action Network unavailable" if is_cup else f"no odds source for {series_name} series"
            st.caption(f"Using salary-estimated odds ({reason})")

    # Clean odds keys to match driver names from API (Jr. -> Jr, etc.)
    if odds_data:
        odds_data = {_clean_api_name(k): v for k, v in odds_data.items()}



# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Loading data..."):
    feed = fetch_weekend_feed(series_id, race_id, selected_year)
    lap_data = fetch_lap_times(series_id, race_id, selected_year)
    lap_averages_df = fetch_lap_averages(series_id, race_id, selected_year)

is_prerace = detect_prerace(feed)

# For completed races, ONLY use saved odds from DB — never show upcoming race odds
if not is_prerace and race_id:
    saved_odds = load_race_odds(race_id)
    if saved_odds:
        odds_data = saved_odds
        odds_source = "saved"
    else:
        # No saved odds for this historical race — clear auto-fetched odds
        # (they're for the upcoming race, not this one)
        odds_data = {}
        odds_source = ""

# Persist odds to DB — only for prerace (auto odds match this race) or manual entry
if is_admin and odds_data and race_id:
    should_save_odds = (is_prerace and odds_source in ("action_network", "salary_estimate")) or \
                       odds_source == "manual"
    if should_save_odds:
        prop_odds = fetch_nascar_prop_odds()
        save_odds_to_db(odds_data, race_id,
                        top3_data=prop_odds.get("top3"),
                        top5_data=prop_odds.get("top5"),
                        top10_data=prop_odds.get("top10"))

results_df = extract_race_results(feed) if feed and not is_prerace else pd.DataFrame()
fl_counts = compute_fastest_laps(lap_data) if lap_data and not is_prerace else {}

# Auto-persist ARP to DB when viewing a completed race with lap data
if not is_prerace and lap_data and race_id:
    from src.data import compute_avg_running_position as _carp, save_arp_to_db
    _arp = _carp(lap_data)
    if _arp:
        save_arp_to_db(_arp, race_id)

qualifying_df = extract_qualifying(feed) if feed else pd.DataFrame()
entry_list_df = extract_entry_list(feed) if feed else pd.DataFrame()

# Auto-pull practice data from lap averages
if not lap_averages_df.empty and "Overall Rank" in lap_averages_df.columns and not practice_data:
    for _, row in lap_averages_df.iterrows():
        driver = row.get("Driver")
        rank = row.get("Overall Rank")
        if driver and rank and not pd.isna(rank):
            practice_data[driver] = int(rank)

# Parse salary CSVs — priority: CSV upload > auto-fetch > saved DB
if dk_file:
    dk_df = parse_dk_csv(dk_file)
elif not dk_auto.empty:
    dk_df = dk_auto[dk_auto["Status"] != "Out"][["Driver", "DK Salary"]].copy()
elif has_saved_dk:
    dk_df = db_dk_df.rename(columns={"Salary": "DK Salary"})[["Driver", "DK Salary"]].copy()
else:
    dk_df = pd.DataFrame()

if fd_file:
    fd_df = parse_fd_csv(fd_file)
elif has_saved_fd:
    fd_df = db_fd_df.rename(columns={"Salary": "FD Salary"})[["Driver", "FD Salary"]].copy()
else:
    fd_df = pd.DataFrame()

# Sync salaries to DB:
#   CSV upload → always sync (explicit intent for this race)
#   Auto-fetch → only sync for prerace (auto-fetch is for the upcoming race, not historical)
if is_admin:
    if dk_file and not dk_df.empty:
        sync_dk_salaries_to_db(dk_df, race_id, series_id, race_name)
    elif is_prerace and not has_saved_dk and not dk_auto.empty:
        sync_dk_salaries_to_db(dk_auto, race_id, series_id, race_name)

    if fd_file and not fd_df.empty:
        sync_fd_salaries_to_db(fd_df, race_id, series_id, race_name)


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
    # Load prop odds (top5/top10) from DB — always available even if live fetch fails
    _prop_odds = load_race_prop_odds(race_id) if race_id else {"top3": {}, "top5": {}, "top10": {}}
    td.render(
        feed=feed, lap_data=lap_data, lap_averages_df=lap_averages_df,
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        results_df=results_df, is_prerace=is_prerace,
        series_id=series_id, race_name=race_name, track_name=track_name,
        track_type=track_type, dk_df=dk_df, fd_df=fd_df,
        completed_races=completed_races, selected_year=selected_year,
        fl_counts=fl_counts, odds_data=odds_data, prop_odds=_prop_odds,
        race_id=race_id,
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
        race_date=race_date_raw, season=selected_year,
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
