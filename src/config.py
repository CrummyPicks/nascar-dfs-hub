"""NASCAR DFS Hub — Configuration & Constants."""

import os
from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
APP_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = APP_DIR / "nascar.db"

# ----------------------------
# NASCAR SERIES
# ----------------------------
SERIES_OPTIONS = {"Cup": 1, "O'Reilly": 2, "Truck": 3}
SERIES_LABELS = {1: "Cup", 2: "O'Reilly", 3: "Truck"}

# ----------------------------
# TRACK TYPE CLASSIFICATION
# ----------------------------
TRACK_TYPE_MAP = {
    # Superspeedway — high-speed pack racing (2.5+ mi ovals)
    "Daytona International Speedway": "superspeedway",
    "Atlanta Motor Speedway": "superspeedway",
    "Talladega Superspeedway": "superspeedway",
    "Indianapolis Motor Speedway": "superspeedway",
    # Road courses
    "Circuit of the Americas": "road",
    "Circuit of The Americas": "road",
    "Sonoma Raceway": "road",
    "Watkins Glen International": "road",
    "Chicago Street Course": "road",
    "Charlotte Roval": "road",
    "Grand Prix of St. Petersburg": "road",
    "San Diego Street Course": "road",
    "Portland International Raceway": "road",
    # Short tracks (< 1 mile ovals)
    "Phoenix Raceway": "short",
    "Martinsville Speedway": "short",
    "Richmond Raceway": "short",
    "Iowa Speedway": "short",
    "North Wilkesboro Speedway": "short",
    "Rockingham Speedway": "short",
    "Bowman Gray Stadium": "short",
    "New Hampshire Motor Speedway": "short",
    # Short concrete — high-banked concrete surface (Bristol/Dover)
    "Bristol Motor Speedway": "short_concrete",
    "Dover Motor Speedway": "short_concrete",
    # Intermediate — standard 1.5-mile ovals
    "Las Vegas Motor Speedway": "intermediate",
    "Kansas Speedway": "intermediate",
    "Charlotte Motor Speedway": "intermediate",
    "Texas Motor Speedway": "intermediate",
    "Nashville Superspeedway": "intermediate",
    "World Wide Technology Raceway": "intermediate",
    "Chicagoland Speedway": "intermediate",
    "Michigan International Speedway": "intermediate",
    "Pocono Raceway": "intermediate",
    # Intermediate worn — high tire wear/abrasive surface
    "Darlington Raceway": "intermediate_worn",
    "Homestead-Miami Speedway": "intermediate_worn",
}

# Map subtypes back to parent type for broader comparisons
TRACK_TYPE_PARENT = {
    "superspeedway": "superspeedway",
    "road": "road",
    "short": "short",
    "short_concrete": "short",
    "intermediate": "intermediate",
    "intermediate_worn": "intermediate",
}

TRACK_TYPE_COLORS = {
    "superspeedway": "#ef4444",
    "road": "#f59e0b",
    "short": "#8b5cf6",
    "short_concrete": "#a78bfa",
    "intermediate": "#3b82f6",
    "intermediate_worn": "#93c5fd",
}

TRACK_TYPE_DISPLAY = {
    "superspeedway": "Superspeedway",
    "road": "Road Course",
    "short": "Short Track",
    "short_concrete": "Short (Concrete)",
    "intermediate": "Intermediate",
    "intermediate_worn": "Intermediate (Worn)",
}

# ----------------------------
# DRIVERAVERAGES.COM TRACK IDS
# ----------------------------
DA_TRACK_IDS = {
    "atlanta": 1, "bristol": 2, "darlington": 5, "daytona": 6,
    "dover": 7, "homestead": 8, "indianapolis": 9, "sonoma": 10,
    "kansas": 11, "las vegas": 12, "charlotte": 13, "martinsville": 14,
    "michigan": 15, "new hampshire": 16, "phoenix": 17, "pocono": 18,
    "richmond": 19, "talladega": 20, "texas": 21, "watkins glen": 22,
    "north wilkesboro": 24, "rockingham": 23, "iowa": 55, "nashville": 57,
    "gateway": 61, "world wide technology": 61,
    "st. petersburg": 216, "circuit of the americas": 211,
    "cota": 211, "chicago": 215, "charlotte roval": 206,
}

# ----------------------------
# DRIVER NAME ALIASES (normalized key → normalized canonical name)
# Keys/values are lowercase, ASCII-folded, period-stripped, suffix-stripped
# ----------------------------
DRIVER_ALIASES = {
    # Middle-name / initial variants
    "john h nemechek": "john hunter nemechek",
    "john nemechek": "john hunter nemechek",
    "jh nemechek": "john hunter nemechek",
    # Abbreviation variants
    "a j allmendinger": "aj allmendinger",
    "christopher buescher": "chris buescher",
    "alexander bowman": "alex bowman",
    # Nickname variants
    "willy b": "william byron",
    # CJ vs C.J. (periods stripped → "cj")
    "c j mclaughlin": "cj mclaughlin",
    "j j yeley": "jj yeley",
    # Stage name / legal name variants
    "cleetus mcfarland": "garrett mitchell",
    "cleetus mitchell": "garrett mitchell",
}

# ----------------------------
# DFS SCORING
# ----------------------------
DK_FINISH_POINTS = {
    1: 45, 2: 42, 3: 41, 4: 40, 5: 39, 6: 38, 7: 37, 8: 36, 9: 35, 10: 34,
    11: 32, 12: 31, 13: 30, 14: 29, 15: 28, 16: 27, 17: 26, 18: 25, 19: 24, 20: 23,
    21: 21, 22: 20, 23: 19, 24: 18, 25: 17, 26: 16, 27: 15, 28: 14, 29: 13, 30: 12,
    31: 10, 32: 9, 33: 8, 34: 7, 35: 6, 36: 5, 37: 4, 38: 3, 39: 2, 40: 1,
}

FD_FINISH_POINTS = {
    1: 43, 2: 40, 3: 38, 4: 36, 5: 34, 6: 32, 7: 30, 8: 28, 9: 26, 10: 24,
    11: 22, 12: 20, 13: 19, 14: 18, 15: 17, 16: 16, 17: 15, 18: 14, 19: 13, 20: 12,
    21: 11, 22: 10, 23: 9, 24: 8, 25: 7, 26: 6, 27: 5, 28: 4, 29: 3, 30: 3,
}

SALARY_CAP = 50000
ROSTER_SIZE = 6

# ----------------------------
# EXHIBITION RACE FILTER KEYWORDS
# ----------------------------
EXHIBITION_KEYWORDS = ["clash", "duel", "all-star", "all star", "exhibition", "open"]

# ----------------------------
# PROJECTION DEFAULTS
# ----------------------------
DEFAULT_PROJECTION_WEIGHTS = {
    "odds": 0.30,
    "track_history": 0.30,
    "practice": 0.25,
    "qualifying": 0.0,  # qualifying only used for start position, not finish prediction
    "track_type": 0.15,
    "recent_form": 0.0,
}

# Track-type-specific default weights (single source of truth for all tabs).
# Values are raw integers that get normalized to 100%.
# Superspeedways: chaotic → odds matter most.
# Short tracks: specialists → track history matters most.
# Road courses: setup-dependent → practice matters most.
# Intermediate: balanced.
TRACK_TYPE_WEIGHT_DEFAULTS = {
    "superspeedway": {"odds": 45, "track": 20, "ttype": 25, "prac": 10},
    "short":         {"odds": 30, "track": 35, "ttype": 15, "prac": 20},
    "road":          {"odds": 25, "track": 25, "ttype": 20, "prac": 30},
    "intermediate":  {"odds": 35, "track": 30, "ttype": 20, "prac": 15},
}

# ----------------------------
# API
# ----------------------------
NASCAR_API_BASE = "https://cf.nascar.com/cacher"
def _get_odds_api_key():
    """Get The Odds API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        return st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))
    except Exception:
        return os.getenv("ODDS_API_KEY", "")

ODDS_API_KEY = _get_odds_api_key()
