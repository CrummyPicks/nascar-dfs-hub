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
    "Charlotte Motor Speedway Road Course": "road",
    "Indianapolis Motor Speedway Road Course": "road",
    "Chicago Street Race": "road",
    "Mid-Ohio Sports Car Course": "road",
    "Road America": "road",
    "Lime Rock Park": "road",
    "Aut\u00f3dromo Hermanos Rodr\u00edguez": "road",
    # Short tracks (< 1 mile ovals)
    "Phoenix Raceway": "short",
    "Martinsville Speedway": "short",
    "Richmond Raceway": "short",
    "Iowa Speedway": "short",
    "North Wilkesboro Speedway": "short",
    "Rockingham Speedway": "short",
    "Bowman Gray Stadium": "short",
    "New Hampshire Motor Speedway": "short",
    "Lucas Oil Indianapolis Raceway Park": "short",
    "Milwaukee Mile Speedway": "short",
    "The Milwaukee Mile": "short",
    "Knoxville Raceway": "short",
    "Bristol Motor Speedway Dirt": "short",
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
    "Auto Club Speedway": "intermediate",
    # Intermediate worn — high tire wear/abrasive surface
    "Darlington Raceway": "intermediate_worn",
    "Homestead-Miami Speedway": "intermediate_worn",
}

# Map subtypes back to parent type for broader comparisons
TRACK_TYPE_PARENT = {
    "superspeedway": "superspeedway",
    "road": "road",
    "short": "short",
    "short_concrete": "short_concrete",
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
    # Middle-name / initial variants — most are now handled systematically
    # by stripped_middle_key() and fuzzy_match_name(). These remain for
    # non-pattern edge cases like stage/legal names.
    "john h nemechek": "john hunter nemechek",
    "john nemechek": "john hunter nemechek",
    "jh nemechek": "john hunter nemechek",
    # Justin S Carroll: results feed tacks on a middle "S" the lap feed drops.
    # There is NO separate "Justin Carroll" driver record, so it's confirmed
    # the same person — but the middle-aware matcher won't merge a
    # middle-initial difference (correct, by rule), so alias it explicitly.
    # (Jason M White vs Jason White are deliberately NOT aliased — confirmed
    # different people, with separate driver records.)
    "justin carroll": "justin s carroll",
    # Stage name / legal name variants (cannot be handled by rules)
    "cleetus mcfarland": "garrett mitchell",
    "cleetus mitchell": "garrett mitchell",
    # "Willy B" nickname — not in NICKNAME_MAP because "willy" is uncommon
    "willy b": "william byron",
}

# ----------------------------
# COMMON FIRST-NAME NICKNAMES (bidirectional — both forms map to the canonical)
# Used by nickname_canonical() in src/utils.py. Entries are lowercase.
# Keys are alternate forms, values are the canonical longer form.
# Applied during matching so "Nick Sanchez" and "Nicholas Sanchez" resolve
# to the same driver without needing a hardcoded DRIVER_ALIASES entry.
# ----------------------------
NICKNAME_MAP = {
    "nick":   "nicholas",
    "nicky":  "nicholas",
    "rob":    "robert",
    "bob":    "robert",
    "bobby":  "robert",
    "robbie": "robert",
    "dan":    "daniel",
    "danny":  "daniel",
    "mike":   "michael",
    "mikey":  "michael",
    "mick":   "michael",
    "tom":    "thomas",
    "tommy":  "thomas",
    "chris":  "christopher",
    "matt":   "matthew",
    "matty":  "matthew",
    "jim":    "james",
    "jimmy":  "james",
    "jamie":  "james",
    "will":   "william",
    "bill":   "william",
    "billy":  "william",
    "willie": "william",
    "ted":    "theodore",
    "teddy":  "theodore",
    "tony":   "anthony",
    "alex":   "alexander",
    "xander": "alexander",
    "ken":    "kenneth",
    "kenny":  "kenneth",
    "joe":    "joseph",
    "joey":   "joseph",
    "gabe":   "gabriel",
    "sam":    "samuel",
    "sammy":  "samuel",
    "ben":    "benjamin",
    "benny":  "benjamin",
    "nate":   "nathaniel",
    "ed":     "edward",
    "eddie":  "edward",
    "eddy":   "edward",
    "rick":   "richard",
    "ricky":  "richard",
    "dick":   "richard",
    "rich":   "richard",
    "fred":   "frederick",
    "freddy": "frederick",
    "greg":   "gregory",
    "pat":    "patrick",
    "patty":  "patrick",
    "andy":   "andrew",
    "drew":   "andrew",
    "jake":   "jacob",
    "johnny": "john",
    "stan":   "stanley",
    "dave":   "david",
    "davey":  "david",
    "zach":   "zachary",
    "zack":   "zachary",
    "cam":    "cameron",
    "vic":    "victor",
    "vince":  "vincent",
    "vinnie": "vincent",
    "gio":    "giovanni",
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
# Race-name fragments that should be excluded from the race picker /
# refresh script. The All-Star RACE is intentionally NOT in this list —
# it's a valid DK DFS event and we want it surfaced systematically every
# year. The All-Star OPEN (the heat race that feeds into it) is still
# excluded via the generic "open" keyword. The Standings tab filters
# separately on race_type_id == 1, so the All-Star Race won't pollute
# points standings even though it now flows through filter_point_races.
EXHIBITION_KEYWORDS = ["clash", "duel", "exhibition", "open"]

# ----------------------------
# PROJECTION DEFAULTS
# ----------------------------
# Recency weighting for historical signals — by RACE ORDER, not season.
# Each driver's (or team's) races are ranked newest-first and weighted
# w = max(0, 1 - (rank-1)*DECAY_STEP). At STEP=0.07 the most recent race gets
# full weight and weight tapers linearly to zero by ~the 15th-oldest race, so
# roughly the last ~10-14 races carry the signal and CURRENT form dominates a
# stale multi-year average — without a hard season cutoff.
#
# Why race-order, not season-linear: a hot 3-race streak is drowned by the
# VOLUME of older races under any season weighting (a driver with 36 races
# has too many old ones). Decaying by race ORDER makes the most recent races
# dominate regardless of how many old races exist. This is the lever that
# actually moves a driver's projected FINISH (a rank), because it selectively
# lifts the recently-good drivers rather than shifting the whole field.
# The 2022 Next-Gen floor still bounds the data. Set STEP=0 to disable
# (weight all races equally).
PROJECTION_RECENCY_DECAY_STEP = 0.07

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
# Tuned from 2025-2026 backtesting correlation analysis:
#   Superspeedways: chaotic — odds dominate, qual/TH nearly uncorrelated with finish.
#   Short tracks: qual most predictive (r=0.47), track history strongest (r=0.36).
#   Short concrete (Bristol/Dover): moderate qual/TH, high position churn.
#   Road courses: qual strong (r=0.45), track-type specialists matter, practice key.
#   Intermediate: balanced signals.
#
# Conventions (enforced):
#   - Every weight is a MULTIPLE OF 5 so the +/- controls (step 5) stay on-grid
#     and a clean "reset to defaults" always lands on round numbers.
#   - ODDS is capped at 25 everywhere. Win odds mostly encode where a driver
#     STARTS / whether they'll WIN, so over-weighting them punishes
#     back-of-pack value plays who can still finish mid-pack. History + pace
#     carry more of the projection instead.
#   - Each row sums to 100.
#   - These apply across ALL series (Cup/Xfinity/Truck) — per-series tuning
#     isn't supported by the data yet (see backtest_weights.py).
# Per-track-type emphasis: superspeedway = draft/equipment + (capped) odds,
# qual irrelevant; short / short_concrete = qualifying + track history;
# road = practice + qual (specialists); intermediate = balanced, history-leaning.
# (Team has a per-driver 0.30x-1.30x scale applied later, so a base of 10
# becomes ~3% for a veteran and ~13% for a rookie.)
TRACK_TYPE_WEIGHT_DEFAULTS = {
    "superspeedway":  {"odds": 25, "track": 20, "ttype": 30, "prac": 5,  "team": 15, "qual": 5},
    "short":          {"odds": 20, "track": 25, "ttype": 10, "prac": 10, "team": 10, "qual": 25},
    "short_concrete": {"odds": 20, "track": 30, "ttype": 5,  "prac": 10, "team": 10, "qual": 25},
    "road":           {"odds": 20, "track": 15, "ttype": 15, "prac": 25, "team": 10, "qual": 15},
    "intermediate":   {"odds": 20, "track": 25, "ttype": 20, "prac": 10, "team": 10, "qual": 15},
}

# ----------------------------
# CROSS-SERIES HIERARCHY
# ----------------------------
# Cross-series track history: supplement current-series data with other series.
# Cup stays isolated. O'Reilly and Truck share data bidirectionally since
# drivers frequently move between them (e.g. Caruth's 7 Truck Bristol races
# are highly relevant when he races O'Reilly at Bristol).
CROSS_SERIES_HIERARCHY = {
    1: [],        # Cup: no cross-series
    2: [1, 3],    # O'Reilly: supplement with Cup and Truck
    3: [1, 2],    # Truck: supplement with Cup and O'Reilly
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
