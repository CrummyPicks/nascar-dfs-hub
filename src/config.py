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
    "Daytona International Speedway": "superspeedway",
    "Atlanta Motor Speedway": "superspeedway",
    "Talladega Superspeedway": "superspeedway",
    "Circuit of the Americas": "road",
    "Circuit of The Americas": "road",
    "Sonoma Raceway": "road",
    "Watkins Glen International": "road",
    "Chicago Street Course": "road",
    "Charlotte Roval": "road",
    "Grand Prix of St. Petersburg": "road",
    "San Diego Street Course": "road",
    "Portland International Raceway": "road",
    "Phoenix Raceway": "short",
    "Martinsville Speedway": "short",
    "Bristol Motor Speedway": "short",
    "Richmond Raceway": "short",
    "Dover Motor Speedway": "short",
    "New Hampshire Motor Speedway": "short",
    "Iowa Speedway": "short",
    "North Wilkesboro Speedway": "short",
    "Rockingham Speedway": "short",
    "Bowman Gray Stadium": "short",
    "Las Vegas Motor Speedway": "intermediate",
    "Darlington Raceway": "intermediate",
    "Kansas Speedway": "intermediate",
    "Charlotte Motor Speedway": "intermediate",
    "Texas Motor Speedway": "intermediate",
    "Nashville Superspeedway": "intermediate",
    "Michigan International Speedway": "intermediate",
    "Pocono Raceway": "intermediate",
    "Homestead-Miami Speedway": "intermediate",
    "Indianapolis Motor Speedway": "intermediate",
    "World Wide Technology Raceway": "intermediate",
    "Chicagoland Speedway": "intermediate",
}

TRACK_TYPE_COLORS = {
    "superspeedway": "#ef4444",
    "road": "#f59e0b",
    "short": "#8b5cf6",
    "intermediate": "#3b82f6",
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
    "gateway": 61, "st. petersburg": 216, "circuit of the americas": 211,
    "cota": 211, "chicago": 215, "charlotte roval": 206,
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
    "odds": 0.25,
    "track_history": 0.20,
    "practice": 0.20,
    "qualifying": 0.15,
    "track_type": 0.10,
    "recent_form": 0.10,
}

# ----------------------------
# API
# ----------------------------
NASCAR_API_BASE = "https://cf.nascar.com/cacher"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
