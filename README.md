# NASCAR DFS Hub

A comprehensive NASCAR Daily Fantasy Sports (DFS) analysis tool built with Streamlit. Combines live NASCAR API data, practice speeds, track history, betting odds, and DraftKings salaries into one dashboard for building optimal DFS lineups.

**Live app**: Hosted on [Streamlit Cloud](https://streamlit.io/cloud)

---

## Quick Start (Local Development)

### 1. Install Dependencies
```bash
pip install streamlit pandas numpy plotly requests beautifulsoup4 lxml
```

### 2. Initialize the Database
```bash
python setup_db.py
```

### 3. Populate Historical Data
```bash
python refresh_data.py --all        # Fetch ALL series, ALL years (2022-2026)
```

### 4. Backfill Average Running Position
```bash
python scrapers/backfill_arp.py     # Fetches lap-times from NASCAR API, computes ARP
```

### 5. Launch the App
```bash
streamlit run nascar_dfs_app.py
```

---

## Weekly Workflow

### Importing DraftKings Salaries

DK salaries must be imported manually (DK API is unreliable). Double-click **IMPORT_SALARIES.bat** or run:

```bash
python import_salaries.py
```

**The workflow:**
1. Download DKSalaries CSV from DraftKings for each series (Cup, Xfinity, Trucks)
2. Run `IMPORT_SALARIES.bat` (or the Python script above)
3. Select series (1=Cup, 2=Xfinity, 3=Truck)
4. Pick the CSV file (auto-finds recent DKSalaries*.csv in Downloads)
5. Select the race (shows last 3 completed + next 2 upcoming)
6. Repeat for each series, then the script commits and pushes to git

Salaries are stored in the database per race and persist across deploys via git.

### After Each Race

Race results are automatically fetched from the NASCAR API when you view a completed race. To bulk-refresh:

```bash
python refresh_data.py              # Fetch new Cup 2026 races
python refresh_data.py --all        # All series, all years
```

---

## Features by Tab

### Race Data
- Consolidated driver data: salary, results, qualifying, practice, track history
- Track history from DB (Next Gen 2022+ era) with Avg DK Points, Avg Running Position
- Fuzzy name matching handles DraftKings vs NASCAR API name differences
- Charts: DFS points bar, Start vs Finish scatter, lap-by-lap analysis

### Practice
- Rankings heatmap with color-coded practice speed rankings
- Lap times view with all intervals
- Gap-to-fastest bar chart with lap interval selector

### Track History
- Recent races at the track (DB-backed, 2022+ Next Gen era)
- ARP vs Avg Finish scatter chart (shows wreck luck factor)
- Track type filtering and aggregation
- By Track Type and Season views

### Race Analyzer
- Single race detailed results with DK/FD points
- Season summary with driver rankings
- Driver lookup and comparison modes

### Projections
- **3-stage projection engine:**
  - **Stage 1 — Projected Finish**: Weighted blend of 4 signals (Track History, Track Type, Odds, Practice) + fixed 15% qualifying signal. All signals normalized to finish position scale (1-37).
  - **Stage 2 — Dominator Allocation**: Historical laps led/fastest laps distributed across field using track-type concentration curves
  - **Stage 3 — DK Points**: `finish_pts + diff_pts + laps_led × 0.25 + fastest_laps × 0.45`
- **Track History signal**: ARP 65% + Avg Finish 35% (filters wreck luck)
- **Smart weight handling**: Missing signals auto-redistribute; no track history → track type absorbs the weight
- **Historical DK points**: Avg DK, Best DK, Worst DK at this track (display only, not in projection math to avoid double-counting dominator value)
- Adjustable weights via sliders (auto-normalizes to 100%)

### Optimizer
- FantasyPros-style lineup builder with Lock, Exclude, and Swap
- Uses Proj DK values from Projections tab for consistency
- Multi-lineup generator with GPP (exposure limits) and Cash modes
- Budget-aware greedy algorithm ($50,000 salary cap)

### Accuracy
- Projections vs Actuals backtesting for any completed race
- Uses same projection engine as live projections
- Track type signal included for consistency

---

## Admin Features

The Settings panel is password-protected (`ADMIN_PASSWORD` in Streamlit secrets):

| Feature | Description |
|---------|-------------|
| DK Salary CSV | Upload DraftKings salary CSV — saves to DB for this race |
| FD Salary CSV | Upload FanDuel salary CSV — saves to DB |
| Clear Salaries | Remove saved salary data for a race |
| Manual Practice | Paste driver practice rankings |
| Betting Odds | Auto-fetched from Action Network; paste to override |
| Refresh Odds | Re-fetch odds from Action Network |

Non-admin users see read-only mode with all saved data visible.

---

## Data Sources

| Source | Data | How |
|--------|------|-----|
| NASCAR API | Race results, qualifying, entry lists, lap times, practice | Automatic on page load |
| Action Network | Win odds (Cup series only) | Automatic (cached 30 min) |
| Database (nascar.db) | Historical results, ARP, DFS points, salaries, odds | `refresh_data.py` + `import_salaries.py` |
| Manual upload | DK/FD salary CSVs | Admin settings panel |

---

## Project Structure

```
NASCAR DFS/
├── nascar_dfs_app.py        # Main Streamlit app
├── setup_db.py              # Database schema initialization
├── refresh_data.py          # Local data refresh CLI
├── import_salaries.py       # DK/FD salary CSV import + git push
├── projections.py           # Backend projection engine (DB-backed)
├── nascar.db                # SQLite database (committed to git)
│
├── src/                     # Core modules
│   ├── config.py            # Constants, track maps, DFS scoring tables
│   ├── data.py              # API fetching, parsing, DB queries, name matching
│   ├── utils.py             # DFS calculations, fuzzy matching, formatting
│   ├── charts.py            # Plotly chart builders (ARP scatter, bar charts)
│   └── components.py        # Reusable Streamlit UI components
│
├── tabs/                    # Tab page modules
│   ├── tab_data.py          # Race Data tab
│   ├── tab_practice.py      # Practice tab
│   ├── tab_track_history.py # Track History tab
│   ├── tab_race_analyzer.py # Race Analyzer tab
│   ├── tab_projections.py   # Projections tab (3-stage engine)
│   ├── tab_optimizer.py     # Optimizer tab
│   └── tab_accuracy.py      # Accuracy backtesting tab
│
├── scrapers/                # Data collection scripts
│   ├── racing_reference.py  # Historical results scraper
│   ├── salaries.py          # DK/FD salary scraper
│   ├── backfill_arp.py      # ARP backfill from NASCAR API lap-times
│   └── frcspro.py           # Alternative data source
│
├── IMPORT_SALARIES.bat      # Double-click salary import (Windows)
├── REFRESH_DATA.bat          # Double-click data refresh (Windows)
├── START_APP.bat             # Double-click app launcher (Windows)
├── .streamlit/config.toml   # Theme configuration
└── README.md                # This file
```

---

## Database Schema

`setup_db.py` creates the SQLite database with these tables:

| Table | Purpose |
|-------|---------|
| `series` | Cup, Xfinity, Trucks identifiers |
| `tracks` | Track names and IDs |
| `drivers` | Driver names (canonical, used for matching) |
| `races` | Race metadata (date, track, laps, API race ID) |
| `race_results` | Per-driver results (finish, start, laps led, fastest laps, **avg_running_position**) |
| `dfs_points` | Calculated DFS scores per race/driver |
| `practice_results` | Practice session data |
| `qualifying_results` | Qualifying positions and speeds |
| `odds` | Saved betting odds per race (win, top 3/5/10) |
| `salaries` | DK/FD salary data per race |
| `projections` | Saved projection outputs |

The database is committed to git so Streamlit Cloud always has data. Run `setup_db.py` only once to create the schema; `refresh_data.py` populates it.

---

## DraftKings Scoring Reference

| Category | Points |
|----------|--------|
| 1st Place | 45 pts |
| 2nd Place | 42 pts |
| 3rd-40th | Descending (41 → 1) |
| Place Differential | +1.0 per position gained |
| Laps Led | +0.25 per lap |
| Fastest Laps | +0.45 per lap |

**Salary Cap**: $50,000 | **Roster Size**: 6 drivers

---

## Deployment (Streamlit Cloud)

1. Push repo to GitHub (including `nascar.db`)
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set main file to `nascar_dfs_app.py`
4. Add `ADMIN_PASSWORD` to Streamlit secrets for upload access
5. Salary imports done locally via `IMPORT_SALARIES.bat`, then pushed to git
