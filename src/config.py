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
# SIMILAR-TRACK GUIDE (hand-curated, separate from TRACK_TYPE_MAP)
# ----------------------------
# A nuanced "study these tracks to prep for this one" map, distinct from the
# mechanical track_type grouping the projection uses. Keys + the names inside
# primary/secondary are CANONICAL DB track names (so they join cleanly). Tracks
# marked "unique" have no primary comps — lead with the note, show secondary as
# a stretch comparison. Descriptive comps the guide gives without a clean track
# list (e.g. "high-speed 1.5-mile tracks") are preserved in the note text.
# Edit freely — this is a curated guide, not derived data.
SIMILAR_TRACKS = {
    "Atlanta Motor Speedway": {
        "profile": "1.5-mile narrow superspeedway",
        "unique": True,
        "primary": [],
        "secondary": ["Daytona International Speedway", "Talladega Superspeedway"],
        "note": "Narrow superspeedway — races more like Daytona than Talladega, "
                "but ultimately view it as unique.",
    },
    "Bowman Gray Stadium": {
        "profile": "0.25-mile super short track",
        "unique": True,
        "primary": [],
        "secondary": ["Martinsville Speedway"],
        "note": "A quarter mile of mayhem. Martinsville is small, but Bowman Gray "
                "is only half its size.",
    },
    "Bristol Motor Speedway": {
        "profile": "0.533-mile concrete short track",
        "unique": True,
        "primary": [],
        "secondary": ["Dover Motor Speedway"],
        "note": "Study Bristol as a unique track.",
    },
    "Charlotte Motor Speedway": {
        "profile": "1.5-mile intermediate (D-shaped, low-moderate wear)",
        "primary": ["Kansas Speedway", "Las Vegas Motor Speedway", "Texas Motor Speedway"],
        "secondary": ["Homestead-Miami Speedway", "Chicagoland Speedway", "Michigan International Speedway"],
        "note": "D-shaped oval with low to moderate wear. Emphasize other "
                "low-to-moderate-wear 1.5-mile tracks.",
    },
    "Charlotte Motor Speedway Road Course": {
        "profile": "2.28-mile road course (Roval)",
        "unique": True,
        "primary": [],
        "secondary": ["Sonoma Raceway", "Watkins Glen International",
                      "Circuit of The Americas", "San Diego Street Course"],
        "note": "Study Roval track history first, then overall recent road-course prowess.",
    },
    "Chicagoland Speedway": {
        "profile": "1.5-mile high-tire-wear intermediate",
        "primary": ["Homestead-Miami Speedway", "Kansas Speedway"],
        "secondary": ["Darlington Raceway", "Las Vegas Motor Speedway",
                      "Charlotte Motor Speedway", "Texas Motor Speedway",
                      "Michigan International Speedway"],
        "note": "High-tire-wear 1.5-mile track (returning 2026). Like Homestead, "
                "the driver can be a difference-maker.",
    },
    "Circuit of The Americas": {
        "profile": "3.4-mile road course",
        "primary": ["Sonoma Raceway", "Watkins Glen International"],
        "secondary": ["Charlotte Motor Speedway Road Course", "San Diego Street Course"],
        "note": "Study drivers' overall road-course racing prowess.",
    },
    "Darlington Raceway": {
        "profile": "1.366-mile high-wear egg-shaped oval",
        "unique": True,
        "primary": [],
        "secondary": ["Homestead-Miami Speedway", "Chicagoland Speedway", "Dover Motor Speedway"],
        "note": "Egg-shaped, high-tire-wear skill track — study as unique.",
    },
    "Daytona International Speedway": {
        "profile": "2.5-mile big superspeedway",
        "primary": ["Talladega Superspeedway"],
        "secondary": ["Atlanta Motor Speedway"],
        "note": "Daytona and Talladega are the two big superspeedways but race "
                "differently — Daytona is much narrower, Talladega much wider.",
    },
    "Dover Motor Speedway": {
        "profile": "1.0-mile concrete skill intermediate",
        "unique": True,
        "primary": [],
        "secondary": ["Darlington Raceway", "Nashville Superspeedway", "Bristol Motor Speedway"],
        "note": "Study Dover as unique. A stretch: similar to Darlington/Bristol. "
                "Nashville is also concrete but quite different.",
    },
    "World Wide Technology Raceway": {
        "profile": "1.25-mile shorter-flat track (Gateway)",
        "primary": ["New Hampshire Motor Speedway", "Phoenix Raceway"],
        "secondary": ["Iowa Speedway", "Richmond Raceway"],
        "note": "A 'big' shorter-flat track (1.25 mi); the others are 1.0 mi or less.",
    },
    "Homestead-Miami Speedway": {
        "profile": "1.5-mile high-tire-wear intermediate",
        "primary": ["Darlington Raceway", "Chicagoland Speedway"],
        "secondary": ["Charlotte Motor Speedway", "Kansas Speedway",
                      "Texas Motor Speedway", "Las Vegas Motor Speedway"],
        "note": "High-tire-wear 1.5-mile track — tire management separates drivers.",
    },
    "Indianapolis Motor Speedway": {
        "profile": "2.5-mile big flat track",
        "primary": ["Pocono Raceway"],
        "secondary": ["Michigan International Speedway"],
        "note": "Big flat rectangle — horsepower and track position are king. "
                "Pocono is the best comp.",
    },
    "Iowa Speedway": {
        "profile": "0.875-mile short track",
        "primary": ["Richmond Raceway"],
        "secondary": ["New Hampshire Motor Speedway", "Phoenix Raceway",
                      "World Wide Technology Raceway"],
        "note": "Approach as any other shorter-flat track.",
    },
    "Kansas Speedway": {
        "profile": "1.5-mile intermediate",
        "primary": ["Las Vegas Motor Speedway"],
        "secondary": ["Michigan International Speedway", "Charlotte Motor Speedway",
                      "Chicagoland Speedway", "Texas Motor Speedway"],
        "note": "Las Vegas is the 'sister track'. Also a mini-Michigan, though "
                "tire wear is much higher now.",
    },
    "Las Vegas Motor Speedway": {
        "profile": "1.5-mile intermediate",
        "primary": ["Kansas Speedway"],
        "secondary": ["Michigan International Speedway", "Charlotte Motor Speedway",
                      "Texas Motor Speedway"],
        "note": "Historically lower tire wear than peers, but it's increasing.",
    },
    "Martinsville Speedway": {
        "profile": "0.526-mile short track",
        "unique": True,
        "primary": [],
        "secondary": ["New Hampshire Motor Speedway", "Bowman Gray Stadium",
                      "North Wilkesboro Speedway"],
        "note": "Largely focus on Martinsville as a unique track.",
    },
    "Michigan International Speedway": {
        "profile": "2.0-mile intermediate",
        "primary": ["Kansas Speedway"],
        "secondary": ["Las Vegas Motor Speedway", "Charlotte Motor Speedway",
                      "Homestead-Miami Speedway", "Texas Motor Speedway"],
        "note": "To prep, study Kansas (a 'mini-Michigan', though tire wear is "
                "higher there now).",
    },
    "Nashville Superspeedway": {
        "profile": "1.33-mile concrete intermediate",
        "unique": True,
        "primary": [],
        "secondary": ["Dover Motor Speedway"],
        "note": "Unique 1.33-mi concrete track — almost a mix of Dover and "
                "high-speed 1.5-mile tracks, but really a hybrid of neither.",
    },
    "New Hampshire Motor Speedway": {
        "profile": "1.058-mile shorter-flat track",
        "primary": ["Richmond Raceway", "World Wide Technology Raceway", "Phoenix Raceway"],
        "secondary": ["Iowa Speedway", "North Wilkesboro Speedway", "Martinsville Speedway"],
        "note": "Study Richmond, Gateway and Phoenix.",
    },
    "North Wilkesboro Speedway": {
        "profile": "0.625-mile short track (repaved 2023)",
        "unique": True,
        "primary": [],
        "secondary": ["Phoenix Raceway", "Richmond Raceway",
                      "World Wide Technology Raceway", "Martinsville Speedway"],
        "note": "Newest surface (repaved after the 2023 All-Star Race). A points-"
                "paying night race in 2026.",
    },
    "Phoenix Raceway": {
        "profile": "1.0-mile shorter-flat track",
        "primary": ["World Wide Technology Raceway", "New Hampshire Motor Speedway"],
        "secondary": ["Richmond Raceway", "North Wilkesboro Speedway", "Iowa Speedway"],
        "note": "Of the five shorter-flat tracks, most focus on Gateway and New Hampshire.",
    },
    "Pocono Raceway": {
        "profile": "2.5-mile big flat triangle",
        "primary": ["Indianapolis Motor Speedway"],
        "secondary": ["Michigan International Speedway"],
        "note": "Run well here and the driver tends to factor at Indy a few weeks later.",
    },
    "Richmond Raceway": {
        "profile": "0.75-mile shorter-flat / short track",
        "primary": ["New Hampshire Motor Speedway", "World Wide Technology Raceway",
                    "Phoenix Raceway", "North Wilkesboro Speedway", "Iowa Speedway"],
        "secondary": ["Martinsville Speedway"],
        "note": "Tire wear is higher here than the other shorter-flat tracks.",
    },
    "San Diego Street Course": {
        "profile": "3.1-mile street road course (2026 debut)",
        "primary": ["Chicago Street Course"],
        "secondary": ["Sonoma Raceway", "Circuit of The Americas",
                      "Watkins Glen International", "Charlotte Motor Speedway Road Course"],
        "note": "Temporary street course — think Chicago Street (a track not "
                "purpose-built for racing).",
    },
    "Sonoma Raceway": {
        "profile": "1.99-mile road course",
        "primary": ["Watkins Glen International", "Circuit of The Americas"],
        "secondary": ["Charlotte Motor Speedway Road Course", "San Diego Street Course"],
        "note": "Very technical — don't neglect overall road-course racing prowess.",
    },
    "Talladega Superspeedway": {
        "profile": "2.66-mile big superspeedway",
        "primary": ["Daytona International Speedway"],
        "secondary": ["Atlanta Motor Speedway"],
        "note": "Daytona and Talladega race differently — Daytona much narrower, "
                "Talladega much wider.",
    },
    "Texas Motor Speedway": {
        "profile": "1.5-mile intermediate (high attrition)",
        "primary": ["Charlotte Motor Speedway", "Kansas Speedway", "Las Vegas Motor Speedway"],
        "secondary": ["Michigan International Speedway", "Chicagoland Speedway",
                      "Homestead-Miami Speedway"],
        "note": "Treacherous 1.5-mile track; attrition has spiked to superspeedway levels.",
    },
    "Watkins Glen International": {
        "profile": "2.45-mile road course",
        "primary": ["Circuit of The Americas", "Autódromo Hermanos Rodríguez",
                    "Sonoma Raceway"],
        "secondary": ["Charlotte Motor Speedway Road Course", "Chicago Street Course"],
        "note": "Largely unique, but study overall road-course racing prowess.",
    },
}


def similar_tracks_for(track_name: str) -> dict:
    """Return the curated similar-track entry for a track, or None.

    Resolves a couple of canonical-name variants the DB carries two ways
    (Circuit of The/the Americas, Chicago Street Course/Race) so the lookup
    doesn't miss. Track names inside the entry are already canonical DB names.
    """
    if not track_name:
        return None
    entry = SIMILAR_TRACKS.get(track_name)
    if entry:
        return entry
    # Variant fallbacks
    _variants = {
        "Circuit of the Americas": "Circuit of The Americas",
        "Chicago Street Race": "Chicago Street Course",
    }
    return SIMILAR_TRACKS.get(_variants.get(track_name, ""), None)

# ----------------------------
# CONCRETE SURFACE (second axis, overlaid on track_type)
# ----------------------------
# Surface is independent of track SIZE. These tracks share a concrete surface:
# dirty air punishes following cars, advancing is slow, and laps led concentrate
# more in the top few than on comparable asphalt. They keep their normal
# track_type (Nashville stays "intermediate", Bristol/Dover "short_concrete") so
# Nashville still appears in "All Intermediate" and its driver popup still shows
# intermediate history — concrete is an ADDITIONAL grouping, not a replacement.
CONCRETE_TRACKS = {
    "Nashville Superspeedway",   # 1.33mi concrete intermediate
    "Dover Motor Speedway",      # 1.0mi concrete (short_concrete)
    "Bristol Motor Speedway",    # 0.533mi concrete (short_concrete)
}
# Virtual group label used in the track-type filter dropdowns / drill-downs.
CONCRETE_GROUP_LABEL = "All Concrete"
# For dominator calibration & the laps-led start gate, concrete tracks behave
# like short_concrete (steep concentration, hard to advance) regardless of size.
CONCRETE_GATE_PROFILE = "short_concrete"


def is_concrete_track(track_name: str) -> bool:
    """True if the track has a concrete racing surface."""
    return track_name in CONCRETE_TRACKS


def resolve_track_group(group: str) -> list:
    """Resolve a track-type / 'All X' group string to a sorted list of track names.

    Single source of truth for every filter dropdown, drill-down popup, and DB
    family-folding query. Handles:
      • the 'All Concrete' surface group  -> CONCRETE_TRACKS
      • 'All <Parent>' parent groups      -> all tracks folding to that parent
      • a plain subtype (e.g. short_concrete) or base type -> exact matches
    """
    if group == CONCRETE_GROUP_LABEL or group.lower() == "concrete":
        return sorted(CONCRETE_TRACKS)
    if group.startswith("All "):
        parent = group.replace("All ", "").lower()
        return sorted(t for t, tt in TRACK_TYPE_MAP.items()
                      if TRACK_TYPE_PARENT.get(tt, tt) == parent)
    return sorted(t for t, tt in TRACK_TYPE_MAP.items() if tt == group)

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

# Official FanDuel NASCAR finish points (verified against FanDuel's scoring
# table 2026-06): 1st=43, 2nd=40, 3rd=38, then -1 per position to 40th=1.
FD_FINISH_POINTS = {1: 43, 2: 40, 3: 38}
FD_FINISH_POINTS.update({pos: 41 - pos for pos in range(4, 41)})  # 4th=37 ... 40th=1

# FanDuel per-lap/diff scoring (DK differs: 0.25 led / 0.45 fastest / 1.0 diff;
# FD has NO fastest-laps points but DOES pay laps completed).
FD_PTS_LAPS_LED = 0.1
FD_PTS_LAPS_COMPLETED = 0.1
FD_PTS_PLACE_DIFF = 0.5

SALARY_CAP = 50000          # DraftKings cap
ROSTER_SIZE = 6             # DraftKings roster
FD_SALARY_CAP = 50000       # FanDuel cap (same $50k, but only 5 drivers)
FD_ROSTER_SIZE = 5          # FanDuel roster

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
# Weights are BACKTEST-TUNED (scripts/backtest_practice_weight.py and
# scripts/backtest_grid_search.py — dual grading: clean rho primary,
# DNF-included sanity, split-half stability required).
#
# 2026-06 practice test (28 races, 22 w/ practice): prac +5 / ttype -5 helped
# intermediates and roads, hurt short tracks; +10 overshoots everywhere.
#
# 2026-06 grid search (445 combos, 15 intermediate races): the old trk=25 was
# consistently too high — 7 of the top 8 combos used trk 15, redistributing
# to team/track-type. Shipped the top ROBUST combo (beats old on both date
# halves AND DNF-included grading; clean rho .544 -> .559): trk 25->15,
# ttype 15->20, team 10->15. Practice 15 confirmed optimal (the top combos
# do NOT push it higher). Road/short/concrete/superspeedway have <8
# backtestable races — too few to tune; rerun the grid as odds accumulate.
TRACK_TYPE_WEIGHT_DEFAULTS = {
    "superspeedway":  {"odds": 25, "track": 20, "ttype": 30, "prac": 5,  "team": 15, "qual": 5},
    "short":          {"odds": 20, "track": 25, "ttype": 10, "prac": 10, "team": 10, "qual": 25},
    "short_concrete": {"odds": 20, "track": 30, "ttype": 5,  "prac": 10, "team": 10, "qual": 25},
    "road":           {"odds": 20, "track": 20, "ttype": 15, "prac": 20, "team": 10, "qual": 15},
    "intermediate":   {"odds": 20, "track": 15, "ttype": 20, "prac": 15, "team": 15, "qual": 15},
}

# ----------------------------
# PHYSICAL TRACK SPECS (static reference)
# ----------------------------
# NASCAR's API exposes no physical specs, so this is a hand-maintained table
# of public facts: length (miles), banking (turn degrees), surface, and shape.
# Used by the Track Data page to characterize a venue beyond its behavioral
# metrics. Keyed by DB track name; matched accent/case-insensitively via
# track_specs(). Add a row when a new venue joins the schedule (DB Health
# flags unmapped tracks).
TRACK_SPECS = {
    # ── Superspeedways ──
    "Daytona International Speedway":  {"length": 2.5,  "banking": "31°", "surface": "Asphalt", "shape": "Tri-oval"},
    "Talladega Superspeedway":        {"length": 2.66, "banking": "33°", "surface": "Asphalt", "shape": "Tri-oval"},
    "Atlanta Motor Speedway":         {"length": 1.54, "banking": "28°", "surface": "Asphalt", "shape": "Quad-oval"},
    # ── Intermediates ──
    "Las Vegas Motor Speedway":       {"length": 1.5,  "banking": "20°", "surface": "Asphalt", "shape": "Tri-oval"},
    "Kansas Speedway":                {"length": 1.5,  "banking": "17–20°", "surface": "Asphalt", "shape": "Tri-oval"},
    "Charlotte Motor Speedway":       {"length": 1.5,  "banking": "24°", "surface": "Asphalt", "shape": "Quad-oval"},
    "Texas Motor Speedway":           {"length": 1.5,  "banking": "20°", "surface": "Asphalt", "shape": "Quad-oval"},
    "Homestead-Miami Speedway":       {"length": 1.5,  "banking": "18–20° (progressive)", "surface": "Asphalt", "shape": "Oval"},
    "Chicagoland Speedway":           {"length": 1.5,  "banking": "18°", "surface": "Asphalt", "shape": "Tri-oval"},
    "Michigan International Speedway": {"length": 2.0,  "banking": "18°", "surface": "Asphalt", "shape": "D-shaped oval"},
    "Auto Club Speedway":             {"length": 2.0,  "banking": "14°", "surface": "Asphalt", "shape": "D-shaped oval"},
    "Pocono Raceway":                 {"length": 2.5,  "banking": "14°/8°/6°", "surface": "Asphalt", "shape": "Triangle"},
    "Darlington Raceway":             {"length": 1.366,"banking": "23–25°", "surface": "Asphalt", "shape": "Egg-shaped oval"},
    "Nashville Superspeedway":        {"length": 1.33, "banking": "14°", "surface": "Concrete", "shape": "Oval"},
    "Rockingham Speedway":            {"length": 1.017,"banking": "22–25°", "surface": "Asphalt", "shape": "Oval"},
    "World Wide Technology Raceway":  {"length": 1.25, "banking": "11°/9°", "surface": "Asphalt", "shape": "Egg-shaped oval"},
    # ── Short tracks ──
    "Bristol Motor Speedway":         {"length": 0.533,"banking": "24–30°", "surface": "Concrete", "shape": "Bullring oval"},
    "Martinsville Speedway":          {"length": 0.526,"banking": "12°", "surface": "Asphalt/Concrete", "shape": "Paperclip"},
    "Richmond Raceway":               {"length": 0.75, "banking": "14°", "surface": "Asphalt", "shape": "D-shaped oval"},
    "Phoenix Raceway":                {"length": 1.0,  "banking": "8–11°", "surface": "Asphalt", "shape": "Dogleg oval"},
    "New Hampshire Motor Speedway":   {"length": 1.058,"banking": "7°", "surface": "Asphalt", "shape": "Oval (flat)"},
    "Dover Motor Speedway":           {"length": 1.0,  "banking": "24°", "surface": "Concrete", "shape": "Oval"},
    "Iowa Speedway":                  {"length": 0.875,"banking": "12–14°", "surface": "Asphalt", "shape": "Oval"},
    "North Wilkesboro Speedway":      {"length": 0.625,"banking": "14°", "surface": "Asphalt", "shape": "Oval"},
    "Milwaukee Mile Speedway":        {"length": 1.0,  "banking": "9°", "surface": "Asphalt", "shape": "Oval (flat)"},
    "The Milwaukee Mile":             {"length": 1.0,  "banking": "9°", "surface": "Asphalt", "shape": "Oval (flat)"},
    "Bowman Gray Stadium":            {"length": 0.25, "banking": "Flat", "surface": "Asphalt", "shape": "Short oval"},
    "Lucas Oil Indianapolis Raceway Park": {"length": 0.686, "banking": "9°", "surface": "Asphalt", "shape": "Oval"},
    "Indianapolis Motor Speedway":    {"length": 2.5,  "banking": "9°", "surface": "Asphalt", "shape": "Rectangular oval (flat)"},
    # ── Road / street courses ──
    "Circuit of The Americas":        {"length": 3.41, "banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Watkins Glen International":      {"length": 2.45, "banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Sonoma Raceway":                 {"length": 1.99, "banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Road America":                   {"length": 4.048,"banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Charlotte Motor Speedway Road Course": {"length": 2.28, "banking": "Road", "surface": "Asphalt", "shape": "Roval"},
    "Indianapolis Motor Speedway Road Course": {"length": 2.439, "banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Mid-Ohio Sports Car Course":     {"length": 2.258,"banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Portland International Raceway":  {"length": 1.964,"banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Lime Rock Park":                 {"length": 1.5,  "banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    "Chicago Street Race":            {"length": 2.2,  "banking": "Street", "surface": "Asphalt", "shape": "Street course"},
    "Grand Prix of St. Petersburg":   {"length": 1.8,  "banking": "Street", "surface": "Asphalt/Concrete", "shape": "Street course"},
    "San Diego Street Course":        {"length": 2.0,  "banking": "Street", "surface": "Asphalt", "shape": "Street course"},
    "Autódromo Hermanos Rodríguez":   {"length": 2.674,"banking": "Road", "surface": "Asphalt", "shape": "Road course"},
    # ── Dirt ──
    "Bristol Motor Speedway Dirt":    {"length": 0.533,"banking": "19°", "surface": "Dirt", "shape": "Oval"},
    "Knoxville Raceway":              {"length": 0.5,  "banking": "12°", "surface": "Dirt", "shape": "Oval"},
}


def track_specs(track_name: str) -> dict:
    """Physical specs for a track (accent/case-insensitive), or {}."""
    if not track_name:
        return {}
    if track_name in TRACK_SPECS:
        return TRACK_SPECS[track_name]
    import unicodedata
    def _norm(s):
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c)).lower().strip()
    nk = _norm(track_name)
    for k, v in TRACK_SPECS.items():
        if _norm(k) == nk:
            return v
    return {}


# ----------------------------
# TEAM LINEAGE (year-over-year continuity)
# ----------------------------
# NASCAR orgs rename/rebrand across seasons (charters move, sponsors buy in).
# Our team signals group by the raw team STRING from the feed, so without
# this map a renamed org's history orphans — the "new" team starts from zero
# even though it's the same shop, people and equipment.
#
# Maintenance: when an org renames, add  "Old Name": "New Name"  here. The
# DB keeps raw names; canonicalization happens at query time, so the map is
# retroactive and reversible. Chains are supported (A->B, B->C) but flatten
# them when convenient.
TEAM_LINEAGE = {
    "Stewart-Haas Racing": "Haas Factory Team",     # 2025 rebrand
    "Petty GMS Motorsports": "Legacy Motor Club",   # 2023 rebrand
}


def canonical_team(name: str) -> str:
    """Resolve a historical team name to its current org name (chain-safe)."""
    seen = set()
    while name in TEAM_LINEAGE and name not in seen:
        seen.add(name)
        name = TEAM_LINEAGE[name]
    return name


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
