# ============================================================
# NASCAR DFS DATA HUB v6.0
# Grouped multipage navigation (st.navigation) — only the active
# page renders, replacing the old 11-tab render-everything layout.
# ============================================================

from datetime import datetime
import logging
import streamlit as st
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

try:
    from src.config import (
        SERIES_OPTIONS, SERIES_LABELS, TRACK_TYPE_MAP, TRACK_TYPE_COLORS, DB_PATH,
        is_concrete_track,
    )
    from src.data import (
        fetch_race_list, fetch_weekend_feed, fetch_lap_times, fetch_lap_averages,
        extract_entry_list, extract_qualifying, extract_race_results,
        compute_fastest_laps, detect_prerace, filter_point_races,
        fetch_nascar_odds, save_odds_to_db,
        estimate_odds_from_salaries, _clean_api_name,
        fetch_nascar_prop_odds, load_race_prop_odds, load_race_odds,
        query_salaries,
    )
except ImportError as e:
    import streamlit as st
    st.error(f"Import failed: {e}")
    st.stop()
except KeyError as e:
    # Transient during a Streamlit Cloud redeploy: the file watcher can rerun the
    # script mid-`git pull`, interrupting an in-progress import so CPython raises
    # KeyError(<module>) from importlib. It self-heals on the next clean rerun —
    # show a friendly "updating" message instead of a scary traceback.
    import streamlit as st
    st.info("🔄 App is updating — refreshing in a moment…")
    st.stop()


# ============================================================
# PAGE CONFIG & CSS
# ============================================================
st.set_page_config(page_title="NASCAR DFS Hub", page_icon="🏁", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap');
/* ══════════════════════════════════════════════════════════════
   NASCAR DFS Hub — Global Theme
   Palette: slate-950 bg, sky-500 accent (#0ea5e9)
   Display font: Rajdhani (motorsport timing-screen look) for the
   brand header + top navigation.
   ══════════════════════════════════════════════════════════════ */

/* ── Base layout ── */
.block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 1600px; }
[data-testid="collapsedControl"] { display: none; }
.main .block-container { padding-top: 0.5rem; }

/* ── Top navigation (st.navigation position="top") ──
   Streamlit's default top-nav items are tiny on desktop — size them up so
   Build / Research / Review / Data read as real navigation. */
[data-testid="stTopNav"] {
    background: #0f172a; border-bottom: 1px solid #1e293b;
    gap: 0.4rem; min-height: 3.2rem; padding-left: 0.6rem;
}
[data-testid="stTopNavSection"] {
    padding: 8px 18px !important; border-radius: 8px;
    position: relative;
    transition: background 0.15s ease;
}
[data-testid="stTopNavSection"]:hover { background: #1e293b66; }
/* Accent bar that slides in under the section label on hover */
[data-testid="stTopNavSection"]::after {
    content: ""; position: absolute; left: 18px; right: 18px; bottom: 4px;
    height: 2px; border-radius: 2px;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.18s ease;
}
[data-testid="stTopNavSection"]:hover::after { transform: scaleX(1); }
[data-testid="stTopNavSection"] p,
[data-testid="stTopNavSection"] span,
[data-testid="stTopNavSection"] [data-testid="stMarkdownContainer"] {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
    font-size: 1.25rem !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 2.5px !important;
    color: #e2e8f0 !important;
}
[data-testid="stTopNavSection"]:hover p,
[data-testid="stTopNavSection"]:hover span { color: #38bdf8 !important; }
[data-testid="stTopNavDropdownLink"] {
    padding: 10px 18px !important;
}
[data-testid="stTopNavDropdownLink"] p,
[data-testid="stTopNavDropdownLink"] span {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    letter-spacing: 1px !important;
}
[data-testid="stTopNavDropdownLink"]:hover p,
[data-testid="stTopNavDropdownLink"]:hover span { color: #38bdf8 !important; }
@media (max-width: 768px) {
    [data-testid="stTopNavSection"] { padding: 6px 10px !important; }
    [data-testid="stTopNavSection"] p,
    [data-testid="stTopNavSection"] span {
        font-size: 1rem !important; letter-spacing: 1.5px !important;
    }
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid #1e293b;
    border-left: 3px solid #0ea5e9;
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"] label {
    color: #64748b !important; font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.8px; font-weight: 600;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif;
    color: #f1f5f9 !important; font-size: 1.45rem; font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Tabs (still used INSIDE some pages) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #0f172a; border-radius: 10px; padding: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; padding: 8px 18px;
    font-weight: 600; color: #64748b; font-size: 0.82rem;
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8; background: #1e293b40; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #1e293b; color: #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
[data-baseweb="tab-highlight"] { background-color: #0ea5e9 !important; }

/* ── Expanders ── */
div[data-testid="stExpander"] {
    background: #111827; border: 1px solid #1e293b; border-radius: 10px;
    margin-bottom: 0.4rem; overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-size: 0.85rem; font-weight: 600; color: #cbd5e1;
}
div[data-testid="stExpander"] summary:hover { color: #e2e8f0; }

/* ── Buttons ── */
.stButton > button {
    border: 1px solid #1e293b; border-radius: 8px; font-weight: 600;
    font-size: 0.8rem; transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #0ea5e9; color: #0ea5e9;
    box-shadow: 0 0 12px rgba(14,165,233,0.15);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    border: none !important; color: #0f172a !important;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
    box-shadow: 0 0 16px rgba(14,165,233,0.3) !important;
}
.stDownloadButton > button {
    border-color: #0ea5e9 !important; color: #0ea5e9 !important;
    border-radius: 8px;
}
.stDownloadButton > button:hover {
    background: #0ea5e910 !important;
    box-shadow: 0 0 12px rgba(14,165,233,0.15);
}

/* ── Inputs & Selects ── */
[data-baseweb="input"] [data-baseweb="base-input"] { border-color: #1e293b !important; border-radius: 8px; }
[data-baseweb="input"]:focus-within [data-baseweb="base-input"] { border-color: #0ea5e9 !important; }
[data-baseweb="select"] > div { border-color: #1e293b !important; border-radius: 8px !important; }

/* ── Text areas ── */
[data-baseweb="textarea"] textarea {
    border: 1px solid #1e293b !important; border-radius: 8px !important;
    background: #0f172a !important;
}
[data-baseweb="textarea"]:focus-within textarea { border-color: #0ea5e9 !important; }
div[data-baseweb="select"] ul { max-height: 300px !important; overflow-y: auto !important; }
div[data-baseweb="popover"] { max-height: 350px !important; overflow-y: auto !important; }
div[data-baseweb="popover"] > div { border-radius: 10px !important; border: 1px solid #1e293b !important; }

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b; border-radius: 10px; overflow: hidden;
}

/* ── Spacing ── */
.element-container { margin-bottom: 0.2rem; }

/* ── Dividers ── */
hr { border-color: #1e293b !important; margin: 0.6rem 0 !important; }

/* ── Remove underlines everywhere ── */
.stMarkdown a, .stCaption a { text-decoration: none !important; }
h1, h2, h3, h4, h5 { text-decoration: none !important; border-bottom: none !important; }
.stMarkdown h3, [data-testid="stMarkdownContainer"] h3 { border-bottom: none !important; text-decoration: none !important; }
[data-testid="stMarkdownContainer"] { text-decoration: none !important; }

/* ── Captions ── */
.stCaption, [data-testid="stCaptionContainer"] { color: #475569 !important; }

/* ── Radio buttons ── */
div[data-testid="stRadio"] > div { gap: 0.3rem; }
div[data-testid="stRadio"] label {
    background: #111827; border: 1px solid #1e293b; border-radius: 8px;
    padding: 4px 12px; font-size: 0.8rem; transition: all 0.15s ease;
}
div[data-testid="stRadio"] label:has(input:checked) {
    border-color: #0ea5e9; color: #e2e8f0; background: #0ea5e910;
}

/* ── Checkboxes ── */
[data-testid="stCheckbox"] label span { font-size: 0.82rem; }

/* ── Wide-layout toggle — never wrap the label vertically (st.toggle
   renders as stCheckbox; .st-key-layout_wide is its widget-key class) ── */
.st-key-layout_wide p {
    white-space: nowrap !important; word-break: normal !important;
    font-size: 0.82rem;
}
.st-key-layout_wide label { flex-wrap: nowrap !important; }

/* ── Number inputs ── */
[data-testid="stNumberInput"] input { border-radius: 6px; }

/* ── Multiselect ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #1e293b !important; border: 1px solid #334155 !important;
    border-radius: 6px !important;
}

/* ── Status strip — page chip + data-readiness chips ── */
.status-strip {
    display: flex; flex-wrap: wrap; gap: 0.45rem 0.55rem; align-items: center;
    background: #0f172a; border: 1px solid #1e293b; border-radius: 10px;
    padding: 7px 12px; margin: 0.25rem 0 0.4rem;
}
.status-strip .page-chip {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif;
    font-size: 1.15rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #38bdf8;
    padding: 0 10px 0 2px; border-right: 2px solid #1e293b;
    margin-right: 0.3rem; line-height: 1.4;
}
.status-strip .chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: #111827; border: 1px solid #1e293b; border-radius: 999px;
    padding: 3px 12px;
}
.status-strip .chip .dot {
    width: 7px; height: 7px; border-radius: 50%; flex: none;
}
.status-strip .chip.ok .dot   { background: #4ade80; box-shadow: 0 0 6px #4ade8088; }
.status-strip .chip.miss .dot { background: #64748b; }
.status-strip .chip .lbl {
    font-family: 'Rajdhani', 'Segoe UI', sans-serif;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #94a3b8;
}
.status-strip .chip .val { font-size: 0.8rem; font-weight: 600; color: #e2e8f0; }
.status-strip .chip.miss .val { color: #64748b; }
.status-strip .badge {
    padding: 2px 12px; border-radius: 999px; font-size: 0.78rem;
    font-family: 'Rajdhani', 'Segoe UI', sans-serif;
    letter-spacing: 1.5px; font-weight: 700; text-transform: uppercase;
}
.status-strip .badge.upcoming  { background: #0ea5e922; color: #38bdf8; border: 1px solid #0ea5e955; }
.status-strip .badge.completed { background: #4ade8022; color: #4ade80; border: 1px solid #4ade8055; }

/* ── Active-page highlight in the top nav: the section holding the current
   page keeps its accent bar lit + sky label; the active dropdown link is
   tinted so it's obvious which page you're on when the menu opens. ── */
[data-testid="stTopNavSection"]:has(a[aria-current="page"])::after { transform: scaleX(1); }
[data-testid="stTopNavSection"]:has(a[aria-current="page"]) p,
[data-testid="stTopNavSection"]:has(a[aria-current="page"]) span {
    color: #38bdf8 !important;
}
a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: #0ea5e91a !important; border-radius: 8px;
}
a[data-testid="stSidebarNavLink"][aria-current="page"] p,
a[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color: #38bdf8 !important;
}

/* ── Mobile ── */
@media (max-width: 768px) {
    .block-container { padding-top: 0.3rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 5px 10px; font-size: 0.7rem; }
    div[data-testid="stMetric"] { padding: 6px 8px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 0.9rem; }
    div[data-testid="stMetric"] label { font-size: 0.6rem !important; }
    h1 { font-size: 1.1rem !important; }
    .nascar-header { padding: 0.5rem 0.8rem !important; }
    .nascar-header h1 { font-size: 1.1rem !important; }
    .nascar-header p { font-size: 0.7rem !important; }
    div[data-testid="stExpander"] summary { font-size: 0.78rem; }
    div[data-testid="stExpander"] { margin-bottom: 0.3rem; }
    div[data-testid="stRadio"] label { padding: 3px 8px; font-size: 0.72rem; }
    [data-testid="stDataFrame"] { font-size: 0.75rem; }
    .stButton > button { font-size: 0.72rem; padding: 4px 10px; }
    .stDownloadButton > button { font-size: 0.72rem; }
    [data-testid="stNumberInput"] label { font-size: 0.72rem !important; }
    .status-strip { font-size: 0.68rem; gap: 0.3rem 0.7rem; padding: 5px 10px; }
}
@media (max-width: 480px) {
    .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; gap: 1px; }
    .stTabs [data-baseweb="tab"] { padding: 4px 7px; font-size: 0.65rem; }
    div[data-testid="stRadio"] > div { flex-wrap: wrap; }
    div[data-testid="stMetric"] { padding: 4px 6px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 0.8rem; }
}
</style>""", unsafe_allow_html=True)

# Header — Rajdhani display type + gradient title to match the top nav
st.markdown("""<div class="nascar-header" style='
  text-align: center;
  padding: 0.8rem 1.2rem;
  margin-bottom: 0.6rem;
  background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #0f172a 100%);
  border-bottom: 2px solid #0ea5e9;
  border-radius: 0 0 12px 12px;
  box-shadow: 0 4px 20px rgba(14,165,233,0.08);
'>
  <h1 style='
    margin: 0; font-size: 1.9rem; line-height: 1.15;
    font-family: "Rajdhani", "Segoe UI", sans-serif;
    text-transform: uppercase; letter-spacing: 5px; font-weight: 700;
    background: linear-gradient(90deg, #f1f5f9 0%, #38bdf8 50%, #f1f5f9 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: #f1f5f9;
  '>NASCAR DFS Hub</h1>
  <p style='
    color: #64748b; margin: 0.1rem 0 0; font-size: 0.82rem;
    font-family: "Rajdhani", "Segoe UI", sans-serif;
    text-transform: uppercase; letter-spacing: 3px; font-weight: 600;
  '>
    Projections &middot; Optimizer &middot; Analytics
  </p>
</div>""", unsafe_allow_html=True)


# ============================================================
# TOP CONTEXT BAR — series / season / race pickers
# (shared by every page; rendered before navigation)
# ============================================================

nav_cols = st.columns([1, 1, 3.0, 1.2, 0.55])

with nav_cols[0]:
    series_name = st.selectbox("Series", list(SERIES_OPTIONS.keys()), key="series_select",
                               label_visibility="collapsed")
    series_id = SERIES_OPTIONS[series_name]

with nav_cols[3]:
    # Platform filter: which site's salaries/projections to show app-wide.
    platform = st.selectbox("Platform", ["DraftKings", "FanDuel", "Both"],
                            key="platform_select", label_visibility="collapsed",
                            help="Which DFS site's salaries, projected points and "
                                 "value columns to show across the app")

with nav_cols[4]:
    # Layout: Wide uses the full browser width (default — best on desktop
    # monitors, kills the dead side gutters). Off = centered compact column.
    # Phones are handled by responsive CSS either way.
    wide_mode = st.toggle("Wide", value=True, key="layout_wide",
                          help="Use the full browser width (desktop). "
                               "Turn off for a centered, narrower layout. "
                               "Phones auto-adapt regardless.")

if wide_mode:
    st.markdown("""<style>
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    @media (max-width: 768px) {
        .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    }
    </style>""", unsafe_allow_html=True)

with nav_cols[1]:
    # Build the year list dynamically so the app rolls over to a new season
    # automatically. We always include the current year as default; we also
    # include next year because NASCAR posts the upcoming schedule in
    # Oct/Nov of the prior year. History goes back to 2022 (Next Gen era).
    _today = datetime.now()
    _current_year = _today.year
    _next_year = _current_year + 1 if _today.month >= 10 else _current_year
    _year_options = list(range(_next_year, 2021, -1))   # newest first
    selected_year = st.selectbox("Season", _year_options, key="year_select",
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

    # Default race: show upcoming race if last completed race was 3+ days ago
    if completed_races and upcoming_races:
        last_completed_race = completed_races[-1][1]
        try:
            last_race_date = datetime.fromisoformat(
                last_completed_race.get("race_date", "").replace("Z", "+00:00").split("+")[0].split("T")[0])
            days_since = (now.date() - last_race_date.date()).days
        except Exception:
            days_since = 0
        if days_since >= 3:
            default_idx = upcoming_races[0][0]
        else:
            default_idx = completed_races[-1][0]
    elif upcoming_races:
        default_idx = upcoming_races[0][0]
    elif completed_races:
        default_idx = completed_races[-1][0]
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
    # Distinguish "API down" from "no data": a valid season always has races,
    # so an empty list here means the NASCAR API was unreachable.
    st.error("Couldn't reach the NASCAR API — the race list is unavailable. "
             "The app will show the Daytona 500 as a fallback; refresh in a "
             "minute to try again.")
    race_id, race_name = 5596, "Daytona 500"
    track_name = "Daytona International Speedway"
    track_type = "superspeedway"
    scheduled_laps = 200
    race_date_raw = ""
    completed_races = []
    upcoming_races = []
    selected_race = {}

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Loading data..."):
    feed = fetch_weekend_feed(series_id, race_id, selected_year)
    lap_data = fetch_lap_times(series_id, race_id, selected_year)
    lap_averages_df = fetch_lap_averages(series_id, race_id, selected_year)

is_prerace = detect_prerace(feed)

if races and not feed:
    st.warning("The NASCAR API didn't return weekend data for this race — "
               "entry list, qualifying, and results pages may be empty. "
               "This is usually temporary; refresh in a minute.")

# ============================================================
# SALARIES (from DB — uploads happen on the Data & Settings page)
# ============================================================
db_dk_df = query_salaries(race_id=race_id, platform="DraftKings")
db_fd_df = query_salaries(race_id=race_id, platform="FanDuel")
has_saved_dk = not db_dk_df.empty
has_saved_fd = not db_fd_df.empty

dk_df = (db_dk_df.rename(columns={"Salary": "DK Salary"})[["Driver", "DK Salary"]].copy()
         if has_saved_dk else pd.DataFrame())
fd_df = (db_fd_df.rename(columns={"Salary": "FD Salary"})[["Driver", "FD Salary"]].copy()
         if has_saved_fd else pd.DataFrame())

# ============================================================
# ODDS RESOLUTION (no UI — managed on the Data & Settings page)
# Priority: saved DB odds (incl. manual pastes) > live auto-fetch
# (pre-race only) > salary estimate (pre-race only). Completed races
# never fall back to live odds — those belong to the NEXT race.
# ============================================================
odds_data = load_race_odds(race_id, series_id) if race_id else {}
odds_source = "saved" if odds_data else ""

if not odds_data and is_prerace:
    auto_odds = fetch_nascar_odds(series_id)
    if auto_odds:
        odds_data = auto_odds
        odds_source = "auto"
    elif not dk_df.empty:
        est_odds = estimate_odds_from_salaries(dk_df)
        if est_odds:
            odds_data = est_odds
            odds_source = "salary_estimate"

    # Persist once per session so odds survive into the completed-race view.
    if odds_data and race_id:
        _odds_save_key = f"odds_autosaved_{series_id}_{race_id}_{odds_source}"
        if _odds_save_key not in st.session_state:
            try:
                prop_odds = fetch_nascar_prop_odds(series_id)
                save_odds_to_db(odds_data, race_id, sportsbook=odds_source,
                                top3_data=prop_odds.get("top3"),
                                top5_data=prop_odds.get("top5"),
                                top10_data=prop_odds.get("top10"),
                                series_id=series_id)
            except Exception:
                logging.getLogger("nascar_dfs").warning(
                    "Failed to persist %s odds for race %s", odds_source, race_id,
                    exc_info=True)
            st.session_state[_odds_save_key] = True

# Clean odds keys to match driver names from API (Jr. -> Jr, etc.)
if odds_data:
    odds_data = {_clean_api_name(k): v for k, v in odds_data.items()}

# Store odds source for downstream pages (Projections labels salary estimates)
st.session_state["odds_source"] = odds_source

# ============================================================
# PRACTICE DATA (manual paste from Data & Settings page wins;
# otherwise auto-computed from the lap-averages feed below)
# ============================================================
from tabs.tab_settings import manual_practice_key
practice_data = dict(st.session_state.get(manual_practice_key(series_id, race_id), {}))

results_df = extract_race_results(feed) if feed and not is_prerace else pd.DataFrame()
fl_counts = compute_fastest_laps(lap_data) if lap_data and not is_prerace else {}

# Auto-persist ARP to DB when viewing a completed race with lap data
if not is_prerace and lap_data and race_id:
    from src.data import compute_avg_running_position as _carp, save_arp_to_db
    _arp = _carp(lap_data)
    if _arp:
        save_arp_to_db(_arp, race_id)

# Auto-persist race_results to DB when viewing a completed race whose results
# haven't been stored yet. Without this, completed races are only populated in
# the DB when the user explicitly runs refresh_data.py — causing DB Health to
# correctly (but confusingly) flag "missing results" for races that already
# finished. Only run once per session per race to avoid repeated work.
if not is_prerace and feed and race_id:
    _results_save_key = f"results_autosaved_{series_id}_{race_id}"
    if _results_save_key not in st.session_state:
        try:
            import sqlite3 as _sql
            _conn = _sql.connect(str(DB_PATH))
            _db_race = _conn.execute(
                "SELECT id FROM races WHERE api_race_id = ? AND series_id = ?",
                (race_id, series_id)
            ).fetchone()
            _needs_save = False
            if _db_race:
                _n = _conn.execute(
                    "SELECT COUNT(*) FROM race_results WHERE race_id = ?",
                    (_db_race[0],)
                ).fetchone()[0]
                _needs_save = (_n == 0)
            _conn.close()
            if _needs_save:
                from src.data import fetch_and_store_race
                fetch_and_store_race(series_id, race_id, selected_year)
            st.session_state[_results_save_key] = True
        except Exception:
            pass  # never block the app if auto-save fails


# Once-per-session-per-(series, season) backfill of completed races whose
# results never made it into the DB. The single-race auto-save above only
# fires for the race currently being viewed — so if the user never clicks
# into St. Petersburg, its results stay missing and that whole road race
# is invisible to track-type aggregations / Track History / projections.
# We scan the schedule and ingest any race that's: completed (date <=
# today), has an api_race_id, and has 0 race_results rows. Cap at 25 to
# bound the wait on first load, and gate by session_state.
_season_backfill_key = f"season_backfilled_{series_id}_{selected_year}"
if _season_backfill_key not in st.session_state:
    try:
        import sqlite3 as _sql
        _today_iso = datetime.now().date().isoformat()
        _conn = _sql.connect(str(DB_PATH))
        _gap_rows = _conn.execute("""
            SELECT r.api_race_id, r.race_name
            FROM races r
            LEFT JOIN race_results rr ON rr.race_id = r.id
            WHERE r.series_id = ?
              AND r.season = ?
              AND r.race_date IS NOT NULL
              AND substr(r.race_date, 1, 10) <= ?
              AND r.api_race_id IS NOT NULL
            GROUP BY r.id
            HAVING COUNT(rr.id) = 0
            ORDER BY r.race_date
            LIMIT 25
        """, (series_id, selected_year, _today_iso)).fetchall()
        _conn.close()
        if _gap_rows:
            from src.data import fetch_and_store_race as _fetch_store
            _fn = getattr(_fetch_store, "__wrapped__", _fetch_store)
            with st.spinner(f"Backfilling {len(_gap_rows)} completed race(s) from API..."):
                for _api_rid, _rname in _gap_rows:
                    try:
                        _fn(series_id, _api_rid, selected_year)
                    except Exception:
                        pass  # individual failures shouldn't kill the loop
        st.session_state[_season_backfill_key] = True
    except Exception:
        # Never block app load on backfill problems
        st.session_state[_season_backfill_key] = True

qualifying_df = extract_qualifying(feed) if feed else pd.DataFrame()
entry_list_df = extract_entry_list(feed) if feed else pd.DataFrame()

# Remap odds keys to match entry list driver names (handles Suárez/Suarez, Jr./Jr, etc.)
if odds_data and not entry_list_df.empty:
    from src.utils import normalize_driver_name, fuzzy_match_name
    entry_drivers = entry_list_df["Driver"].tolist()
    _norm_entry = {normalize_driver_name(d): d for d in entry_drivers}
    remapped = {}
    for odds_name, odds_val in odds_data.items():
        if odds_name in entry_drivers:
            remapped[odds_name] = odds_val
            continue
        # Try normalized match
        norm_key = normalize_driver_name(odds_name)
        if norm_key in _norm_entry:
            remapped[_norm_entry[norm_key]] = odds_val
        else:
            # Fuzzy fallback
            matched = fuzzy_match_name(odds_name, entry_drivers)
            remapped[matched if matched else odds_name] = odds_val
    odds_data = remapped

# Auto-pull practice data from lap averages, remapping names to match the
# entry list so downstream lookups succeed. NASCAR's lap-averages feed can
# use different spellings than the entry list (e.g. "John H. Nemechek" vs
# "John Hunter Nemechek"), so we normalize + fuzzy-match each key to the
# canonical entry-list name before storing. Mirrors the odds remap above.
#
# Signal = simple mean of the lap-window ranks the driver actually ran
# (1L, 5L, 10L, 15L, 20L, 25L, 30L). This matches the "Average" column
# shown in the in-app practice heatmap. We deliberately do NOT use
# NASCAR's "Overall Rank" — it averages lap *times*, which rewards
# drivers who ran 3 fast laps and parked it over drivers who completed
# a real long run with falloff.
if not lap_averages_df.empty and not practice_data:
    _has_rank_cols = any(c in lap_averages_df.columns
                         for c in ["1 Lap Rank", "5 Lap Rank", "10 Lap Rank",
                                   "15 Lap Rank", "20 Lap Rank", "25 Lap Rank",
                                   "30 Lap Rank"])
    if _has_rank_cols:
        from src.utils import (normalize_driver_name, fuzzy_match_name,
                               compute_practice_signals)
        _entry_drivers = entry_list_df["Driver"].tolist() if not entry_list_df.empty else []
        _norm_entry = {normalize_driver_name(d): d for d in _entry_drivers}
        # Coverage-weighted, long-run-weighted practice signal computed across the
        # WHOLE field at once (so each lap-window is scored by participation —
        # sparse buckets like a 3-driver 30-lap run don't inflate those runners).
        _prac_signals = compute_practice_signals(lap_averages_df,
                                                 field_size=len(lap_averages_df))
        for driver, signal in _prac_signals.items():
            if not driver or signal is None:
                continue
            key = driver
            if _entry_drivers and driver not in _entry_drivers:
                norm_key = normalize_driver_name(driver)
                if norm_key in _norm_entry:
                    key = _norm_entry[norm_key]
                else:
                    matched = fuzzy_match_name(driver, _entry_drivers)
                    if matched:
                        key = matched
            practice_data[key] = signal

# ============================================================
# STATUS STRIP — current page + race state + data readiness.
# Built here (data in scope), RENDERED after st.navigation() below so the
# strip can lead with the active page name.
# ============================================================
_badge = ('<span class="badge upcoming">Upcoming</span>' if is_prerace
          else '<span class="badge completed">Completed</span>')
_date_part = f'<span style="color:#64748b;font-size:0.78rem;"> {race_date_raw}</span>' \
             if race_date_raw else ''


def _chip(label, ok, detail_ok, detail_miss="not loaded"):
    cls = "ok" if ok else "miss"
    detail = detail_ok if ok else detail_miss
    return (f'<span class="chip {cls}"><span class="dot"></span>'
            f'<span class="lbl">{label}</span>'
            f'<span class="val">{detail}</span></span>')


_odds_label = {"saved": "DB", "auto": "Action Network",
               "salary_estimate": "salary estimate"}.get(odds_source, "")
_strip_items = [
    f'{_badge}{_date_part}',
    _chip("DK", has_saved_dk, f"{len(db_dk_df)} drivers"),
    _chip("FD", has_saved_fd, f"{len(db_fd_df)} drivers"),
    _chip("Odds", bool(odds_data), f"{len(odds_data)} · {_odds_label}"),
    _chip("Practice", bool(practice_data), f"{len(practice_data)} drivers",
          "none yet"),
]
if not (has_saved_dk and odds_data):
    _strip_items.append('<span style="color:#64748b;font-size:0.76rem;">'
                        '→ load data on the <b>Data &amp; Settings</b> page</span>')

# ============================================================
# PAGES — grouped navigation; only the active page renders
# ============================================================
# Keep projection-weight widget state alive across page switches. Streamlit
# garbage-collects a widget's session-state key on any run where the widget
# doesn't render — under the old single-page st.tabs layout everything
# rendered every run, but with st.navigation only the active page does. The
# Optimizer reads the Projections sliders' pw_* keys, so re-assert them here
# (the documented keep-alive pattern). Button keys can't be set — skip them.
for _k in list(st.session_state.keys()):
    if _k.startswith("pw_") and "btn" not in _k:
        st.session_state[_k] = st.session_state[_k]

# A per-run guard for the driver-history dialog (Streamlit allows only one
# @st.dialog per run). With pages only one page renders per run, but pages
# can still contain multiple drill-down tables — keep the guard.
from src.components import reset_driver_dialog_guard
reset_driver_dialog_guard()

from tabs import tab_data as td
from tabs import tab_practice as tp
from tabs import tab_track_history as tth
from tabs import tab_race_analyzer as tra
from tabs import tab_projections as tproj
from tabs import tab_optimizer as topt
from tabs import tab_race_lab as trl
from tabs import tab_cautions as tcau
from tabs import tab_accuracy as tacc
from tabs import tab_standings as tstand
from tabs import tab_db_health as tdbh
from tabs import tab_settings as tset


def _page_projections():
    tproj.render(
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        lap_averages_df=lap_averages_df, practice_data=practice_data,
        is_prerace=is_prerace, race_name=race_name, race_id=race_id,
        track_name=track_name, series_id=series_id, dk_df=dk_df,
        odds_data=odds_data, scheduled_laps=scheduled_laps,
        race_date=race_date_raw, season=selected_year,
        fd_df=fd_df, platform=platform,
    )


def _page_optimizer():
    topt.render(
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        lap_averages_df=lap_averages_df, practice_data=practice_data,
        is_prerace=is_prerace, race_name=race_name, race_id=race_id,
        track_name=track_name, series_id=series_id, dk_df=dk_df,
        odds_data=odds_data, fd_df=fd_df, platform=platform,
    )


def _page_race_data():
    # Load prop odds (top5/top10) from DB — always available even if live fetch fails
    _prop_odds = load_race_prop_odds(race_id, series_id) if race_id else {"top3": {}, "top5": {}, "top10": {}}
    td.render(
        feed=feed, lap_data=lap_data, lap_averages_df=lap_averages_df,
        entry_list_df=entry_list_df, qualifying_df=qualifying_df,
        results_df=results_df, is_prerace=is_prerace,
        series_id=series_id, race_name=race_name, track_name=track_name,
        track_type=track_type, dk_df=dk_df, fd_df=fd_df,
        completed_races=completed_races, selected_year=selected_year,
        fl_counts=fl_counts, odds_data=odds_data, prop_odds=_prop_odds,
        race_id=race_id, platform=platform,
    )


def _page_practice():
    tp.render(
        lap_averages_df=lap_averages_df, feed=feed,
        race_name=race_name, series_id=series_id,
        race_id=race_id, selected_year=selected_year,
        track_name=track_name,
    )


def _page_track_history():
    tth.render(
        track_name=track_name, track_type=track_type, series_id=series_id,
        entry_list_df=entry_list_df,
    )


def _page_concrete():
    tth.render_concrete_tab(series_id=series_id, entry_list_df=entry_list_df)


def _page_race_analyzer():
    tra.render(
        completed_races=completed_races, series_id=series_id,
        selected_year=selected_year, series_name=series_name,
        platform=platform,
    )


def _page_race_lab():
    trl.render(
        completed_races=completed_races, series_id=series_id,
        selected_year=selected_year, series_name=series_name,
        track_name=track_name, track_type=track_type,
        selected_race=selected_race,
    )


def _page_cautions():
    tcau.render(
        completed_races=completed_races, series_id=series_id,
        selected_year=selected_year, series_name=series_name,
    )


def _page_accuracy():
    tacc.render(
        completed_races=completed_races, series_id=series_id,
        selected_year=selected_year, series_name=series_name,
    )


def _page_standings():
    tstand.render(
        series_id=series_id, series_name=series_name,
        selected_year=selected_year,
    )


def _page_db_health():
    # Quick-summary banner (silent when clean) above the full diagnostics.
    # This used to render in the shell on every page — moved here so data
    # warnings live with the rest of the DB health info.
    try:
        tdbh.render_health_banner()
    except Exception:
        pass  # never block the page for a health-check error
    tdbh.render(series_id=series_id, selected_year=selected_year)


def _page_settings():
    tset.render(race_id=race_id, series_id=series_id,
                race_name=race_name, is_prerace=is_prerace)


# A dedicated "Concrete" page appears ONLY on concrete race weeks (Nashville,
# Dover, Bristol) — surfacing the All-Concrete surface group when it's
# relevant, without cluttering asphalt weeks.
_research_pages = [
    st.Page(_page_race_data, title="Race Data", icon="📋", url_path="race-data",
            default=True),
    st.Page(_page_practice, title="Practice", icon="⏱️", url_path="practice"),
    st.Page(_page_track_history, title="Track History", icon="🏟️", url_path="track-history"),
    st.Page(_page_race_analyzer, title="Race Analyzer", icon="🔬", url_path="race-analyzer"),
    st.Page(_page_race_lab, title="Race Lab", icon="🧪", url_path="race-lab"),
]
if is_concrete_track(track_name):
    _research_pages.insert(3, st.Page(_page_concrete, title="Concrete", icon="🧱",
                                      url_path="concrete"))

_nav = st.navigation(
    {
        "Build": [
            st.Page(_page_projections, title="Projections", icon="📈",
                    url_path="projections"),
            st.Page(_page_optimizer, title="Optimizer", icon="🧮", url_path="optimizer"),
        ],
        "Research": _research_pages,
        "Review": [
            st.Page(_page_accuracy, title="Accuracy", icon="🎯", url_path="accuracy"),
            st.Page(_page_cautions, title="Cautions", icon="🚧", url_path="cautions"),
            st.Page(_page_standings, title="Standings", icon="🏆", url_path="standings"),
        ],
        "Data": [
            st.Page(_page_settings, title="Data & Settings", icon="⚙️", url_path="settings"),
            st.Page(_page_db_health, title="DB Health", icon="🩺", url_path="db-health"),
        ],
    },
    position="top",
)

# Render the status strip now that the active page is known — leading with a
# big "you are here" chip so the current page is unmissable.
_page_chip = f'<span class="page-chip">{_nav.icon} {_nav.title}</span>'
st.markdown(
    f'<div class="status-strip">{_page_chip}'
    + "".join(_strip_items) + '</div>',
    unsafe_allow_html=True,
)

_nav.run()
