# NASCAR DFS Hub

A comprehensive NASCAR Daily Fantasy Sports (DFS) analysis tool built with Streamlit. Combines live NASCAR API data, practice speeds, track history, betting odds, and DraftKings salaries into one dashboard for building optimal DFS lineups.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit pandas numpy plotly requests beautifulsoup4
```

### 2. Initialize the Database
```bash
python setup_db.py
```

### 3. Populate Historical Data
```bash
python refresh_data.py              # Fetch all new Cup 2026 races
python refresh_data.py --all        # Fetch ALL series, ALL years (2022-2026)
```
Or double-click **REFRESH_DATA.bat** on Windows.

### 4. Launch the App
```bash
streamlit run nascar_dfs_app.py
```
Or double-click **START_HUB.bat** on Windows.

---

## Data Refresh (Local Script)

The database is populated using `refresh_data.py`, which runs locally on your machine. This keeps API keys and write access off the public web app.

```bash
# Fetch all completed Cup 2026 races (default)
python refresh_data.py

# Fetch a specific series
python refresh_data.py --series xfinity
python refresh_data.py --series truck

# Fetch a specific year
python refresh_data.py --year 2025

# Fetch all series and all years (2022-2026)
python refresh_data.py --all

# Fetch a single race by ID
python refresh_data.py --race 5596
```

Run this weekly (or after each race) to keep the database current. The script skips races already in the database, so re-running is safe and fast.

---

## Features by Tab

### Race Data
- **Table view**: Consolidated driver data — salary, results, qualifying, practice rankings, and track history in one table
- **Charts view**: DFS Points bar chart (hover for score breakdown), Start vs Finish scatter, race lap-by-lap analysis
- **Search**: Filter any driver, team, or manufacturer
- **Export**: Download data as CSV

### Practice
- **Rankings Heatmap**: Color-coded practice speed rankings across all lap averages (toggle colors on/off)
- **Lap Times view**: Raw practice lap time data with all intervals
- **Lap Chart**: Interactive line chart of individual practice laps with outlier filtering and driver selection
- **Gap-to-Fastest Bar Chart**: Horizontal bar chart showing delta from fastest driver, with lap interval selector (Overall Avg, 5 Lap, 10 Lap, etc.)

### Track History
- **Recent Races**: Driver performance at the selected track (from driveraverages.com)
- **All-Time**: Complete historical stats at the track
- **By Track Type**: Aggregate stats across similar track types (short, intermediate, superspeedway, road course)
- **2026 Season**: Current season aggregate from database

### Race Analyzer
Four analysis modes with independent filters for Series, Year, and Track:
- **Single Race**: Detailed results for any completed race with DK/FD points, salary, driver rating, and scatter chart
- **Season Summary**: Aggregated stats across all completed races with driver rankings
- **Driver Lookup**: Race-by-race log for any driver
- **Driver Comparison**: Side-by-side stat comparison with race-by-race charts

### Projections
- **DFS-aware projection engine** that projects actual DraftKings point components:
  - Finish Points (from projected finish position)
  - Place Differential Points (start - finish)
  - Laps Led Points (dominator projection)
  - Fastest Laps Points
- **Five weighted signals**: Track History, Track Type, Qualifying, Practice, Betting Odds
- Adjustable weights via sliders (auto-normalizes to 100%)
- Smart handling: if odds aren't available, weight redistributes automatically
- DK Salary and Value (pts per $1K) when salary data is available

### Optimizer
- **FantasyPros-style lineup builder** with Lock, Exclude, and Swap per driver
- Auto-generates optimal lineup on load
- Budget-aware greedy algorithm respects the $50,000 salary cap
- Multi-lineup generator with GPP (max exposure limits) and Cash modes
- Player pool table with status indicators

---

## Data Sources

| Source | Data | How |
|--------|------|-----|
| NASCAR API | Race results, qualifying, entry lists, lap times, practice | Automatic on page load |
| DraftKings API | Driver salaries (upcoming races only) | Automatic on page load |
| Action Network | Win odds (American format) | Automatic on page load (cached 30 min) |
| driveraverages.com | Track history stats | Automatic when Track History loads |
| Local refresh script | Historical race results, fastest laps, DFS points | `python refresh_data.py` |
| Manual upload | DK/FD salary CSVs | Settings panel file upload |

---

## Settings Panel

Click **Settings & Data Upload** to expand:

| Setting | Description |
|---------|-------------|
| DK Salary | Upload DraftKings salary CSV (overrides auto-fetch) |
| FD Salary CSV | Upload FanDuel salary CSV |
| Manual Practice | Paste driver practice rankings (format: `Driver Name, Rank`) |
| Betting Odds | Auto-fetched from Action Network; paste to override (format: `Driver Name, 1200`) |

---

## Project Structure

```
NASCAR DFS/
├── nascar_dfs_app.py       # Main Streamlit app
├── projections.py          # 6-component DFS projection engine
├── setup_db.py             # Database schema initialization
├── refresh_data.py         # Local data refresh CLI script
├── nascar.db               # SQLite database (auto-created)
│
├── src/                    # Core modules
│   ├── config.py           # Constants, track maps, DFS scoring tables
│   ├── data.py             # API fetching, parsing, DB queries
│   ├── utils.py            # Point calculations, name matching, formatting
│   ├── charts.py           # Plotly chart builders
│   └── components.py       # Reusable Streamlit UI components
│
├── tabs/                   # Tab page modules
│   ├── tab_data.py         # Race Data tab
│   ├── tab_practice.py     # Practice tab
│   ├── tab_track_history.py# Track History tab
│   ├── tab_race_analyzer.py# Race Analyzer tab
│   ├── tab_projections.py  # Projections tab
│   └── tab_optimizer.py    # Optimizer tab
│
├── .streamlit/config.toml  # Theme configuration
├── legacy/                 # Archived files from earlier iterations
├── dev/                    # Development/seed scripts
├── exports/                # Generated projection CSVs
├── START_APP.bat           # Windows launcher (app)
├── SETUP.bat               # First-time dependency installer
├── REFRESH_DATA.bat        # Windows launcher (data refresh)
└── README.md               # This file
```

---

## DraftKings Scoring Reference

| Category | Points |
|----------|--------|
| 1st Place | 45 pts |
| 2nd Place | 42 pts |
| 3rd Place | 40 pts |
| 4th-5th | 38-39 pts |
| ... down to 40th | 1 pt |
| Place Differential | +1.0 per position gained |
| Laps Led | +0.25 per lap |
| Fastest Laps | +0.45 per lap |

**Salary Cap**: $50,000 | **Roster Size**: 6 drivers

---

## Supported Series

- **Cup Series** (Series ID: 1)
- **O'Reilly Xfinity Series** (Series ID: 2)
- **Craftsman Truck Series** (Series ID: 3)

---

## Deployment

The app can be deployed on Streamlit Community Cloud, a VPS, or any platform that supports Python.

1. Push the repo to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set the main file to `nascar_dfs_app.py`
4. Run `refresh_data.py` locally to populate the database before deploying
5. Upload the `nascar.db` file with the repo (or use a remote DB)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No DK salaries showing | DK only publishes salaries for upcoming races. Check closer to race day. |
| No odds available | Odds appear once sportsbooks post lines for the next race (usually mid-week). |
| Track history empty | The driveraverages.com source may be temporarily unavailable. |
| Database empty | Run `python refresh_data.py --all` to populate all historical data. |
| Projections show 0 for Led/FL pts | Need track history data in the DB; run `python refresh_data.py --all` for historical races. |
| Port 8501 in use | Run `streamlit run nascar_dfs_app.py --server.port 8502` |
