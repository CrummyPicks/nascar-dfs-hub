"""
NASCAR DFS Projections Engine
==============================
Generates per-driver DFS score projections for an upcoming race using a
six-component weighted model:

    1. Track-type history (default 25%) — avg DFS at SIMILAR tracks (intermediates,
                                          supers, short, road, dirt) — the primary signal
    2. Track-specific     (default 20%) — avg DFS at THIS exact track (last 5 visits)
    3. Practice           (default 20%) — session rank + long-run pace vs field
    4. Odds               (default 15%) — sportsbook win/top5/top10 implied score
    5. Qual position      (default 12%) — start position advantage → place-diff pts
    6. Recent form        (default  8%) — last 10 races, exponentially decayed

All six component scores are stored individually in the DB so the web UI can
apply custom weight sliders and recompute totals client-side in real time.

Missing data degrades gracefully — never silently zeros a driver out.

Usage
-----
  python projections.py                              # auto-detect race, both platforms
  python projections.py --race-id 42                # explicit race
  python projections.py --platform DraftKings        # one platform
  python projections.py --series cup                 # one series
  python projections.py --form-races 5               # shorter form window
  python projections.py --weights 25 20 20 15 12 8   # TRKTYPE TRK PRAC ODDS QUAL FORM
  python projections.py --dry-run                    # preview without saving
"""

import sqlite3
import csv
import os
import sys
import argparse
import math
from datetime import datetime

DB_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nascar.db")
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")

# ── Scoring rates (for converting raw metrics → DFS-point adjustments) ───────
# DraftKings NASCAR Classic
DK_PLACE_DIFF_RATE = 1.00     # pts per position gained/lost
DK_FINISH_PTS = {
    1: 40, 2: 35, 3: 33, 4: 31, 5: 29, 6: 27, 7: 25, 8: 23, 9: 21, 10: 20,
    11: 19, 12: 18, 13: 17, 14: 16, 15: 15, 16: 14, 17: 13, 18: 12, 19: 11,
    20: 10, 21: 9,  22: 8,  23: 7,  24: 6,  25: 5,  26: 4,  27: 3,  28: 2,
    29: 1,  30: 0,
}

# FanDuel NASCAR
FD_PLACE_DIFF_RATE = 1.00     # pts per position gained/lost
FD_FINISH_PTS = {
    1: 48, 2: 40, 3: 38, 4: 36, 5: 34, 6: 32, 7: 30, 8: 28, 9: 26, 10: 24,
    11: 22, 12: 21, 13: 20, 14: 19, 15: 18, 16: 17, 17: 16, 18: 15, 19: 14,
    20: 13, 21: 12, 22: 11, 23: 10, 24: 9,  25: 8,  26: 7,  27: 6,  28: 5,
    29: 4,  30: 3,
}

PLACE_DIFF_RATE = {"DraftKings": DK_PLACE_DIFF_RATE, "FanDuel": FD_PLACE_DIFF_RATE}
FINISH_PTS      = {"DraftKings": DK_FINISH_PTS,      "FanDuel": FD_FINISH_PTS}

# ── Default weights — must sum to 100 (treated as %) ─────────────────────────
# Order matters for CLI --weights positional args:
#   track_type  track  practice  odds  qual  form
DEFAULT_WEIGHTS = {
    "track":      25,   # this specific track history  ← primary signal
    "track_type": 20,   # similar-track-type history
    "practice":   20,   # practice speed / long-run pace
    "odds":       15,   # sportsbook implied probability
    "qual":       12,   # qualifying position delta
    "form":        8,   # recent form (last N races)
}

# Keys in the order expected by --weights positional args
WEIGHT_KEYS = ["track", "track_type", "practice", "odds", "qual", "form"]

# Exponential decay for recent-form window (index 0 = most recent race)
# 10-race window; weights auto-normalize
FORM_DECAY = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30]

# Track-history recency multiplier by seasons-ago
# (0 seasons ago = same season, 1 = last year, etc.)
TRACK_DECAY = {0: 1.00, 1: 0.90, 2: 0.78, 3: 0.65, 4: 0.52, 5: 0.40}

# Minimum track visits to trust track_score alone (otherwise blend w/ track-type)
MIN_TRACK_VISITS = 3

# Max DFS adjustment from practice (caps noise from outlier sessions)
MAX_PRACTICE_ADJ = 3.5

# Max DFS adjustment from odds
MAX_ODDS_ADJ = 4.0


# ── DB connection + schema migration ─────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply any schema additions that may not exist in older DB files."""
    # ── practice_results additions ────────────────────────────────────────────
    pr_cols = {r[1] for r in conn.execute("PRAGMA table_info(practice_results)")}
    for col, typ in [("long_run_avg", "REAL"), ("avg_lap", "REAL")]:
        if col not in pr_cols:
            conn.execute(f"ALTER TABLE practice_results ADD COLUMN {col} {typ}")

    # ── salaries: status column (Available / Out / Questionable / Probable) ──
    sal_cols = {r[1] for r in conn.execute("PRAGMA table_info(salaries)")}
    if "status" not in sal_cols:
        conn.execute(
            "ALTER TABLE salaries ADD COLUMN status TEXT NOT NULL DEFAULT 'Available'"
        )

    # ── Projections table (IF NOT EXISTS = idempotent) ────────────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projections (
            id               INTEGER PRIMARY KEY,
            race_id          INTEGER NOT NULL REFERENCES races(id),
            driver_id        INTEGER NOT NULL REFERENCES drivers(id),
            platform         TEXT NOT NULL,
            proj_score       REAL,
            salary           INTEGER,
            value            REAL,
            track_score      REAL,
            track_type_score REAL,
            form_score       REAL,
            qual_adj         REAL,
            practice_adj     REAL,
            odds_adj         REAL,
            track_races      INTEGER,
            form_races       INTEGER,
            track_type_used  INTEGER DEFAULT 0,
            generated_at     TEXT DEFAULT (datetime('now')),
            UNIQUE(race_id, driver_id, platform)
        );
        CREATE INDEX IF NOT EXISTS idx_proj_race   ON projections(race_id);
        CREATE INDEX IF NOT EXISTS idx_proj_driver ON projections(driver_id);
    """)

    # ── projections: track_type_score column (older DBs may be missing it) ───
    proj_cols = {r[1] for r in conn.execute("PRAGMA table_info(projections)")}
    if "track_type_score" not in proj_cols:
        conn.execute("ALTER TABLE projections ADD COLUMN track_type_score REAL")

    # ── Seed known tracks (always INSERT OR IGNORE so new tracks get added) ────
    conn.executescript("""
        INSERT OR IGNORE INTO tracks(name,short_name,city,state,track_type,length_miles) VALUES
            ('Daytona International Speedway','Daytona','Daytona Beach','FL','superspeedway',2.5),
            ('Talladega Superspeedway','Talladega','Talladega','AL','superspeedway',2.66),
            ('Atlanta Motor Speedway','Atlanta','Hampton','GA','intermediate',1.54),
            ('Charlotte Motor Speedway','Charlotte','Concord','NC','intermediate',1.5),
            ('Texas Motor Speedway','Texas','Fort Worth','TX','intermediate',1.5),
            ('Kansas Speedway','Kansas','Kansas City','KS','intermediate',1.5),
            ('Las Vegas Motor Speedway','Las Vegas','Las Vegas','NV','intermediate',1.5),
            ('Michigan International Speedway','Michigan','Brooklyn','MI','intermediate',2.0),
            ('Homestead-Miami Speedway','Homestead','Homestead','FL','intermediate',1.5),
            ('Nashville Superspeedway','Nashville','Lebanon','TN','intermediate',1.33),
            ('Iowa Speedway','Iowa','Newton','IA','short',0.875),
            ('Bristol Motor Speedway','Bristol','Bristol','TN','short',0.533),
            ('Martinsville Speedway','Martinsville','Martinsville','VA','short',0.526),
            ('Richmond Raceway','Richmond','Richmond','VA','short',0.75),
            ('New Hampshire Motor Speedway','New Hampshire','Loudon','NH','short',1.058),
            ('Phoenix Raceway','Phoenix','Avondale','AZ','short',1.0),
            ('Dover Motor Speedway','Dover','Dover','DE','short',1.0),
            ('North Wilkesboro Speedway','North Wilkesboro','North Wilkesboro','NC','short',0.625),
            ('Rockingham Speedway','Rockingham','Rockingham','NC','short',0.94),
            ('Pocono Raceway','Pocono','Long Pond','PA','intermediate',2.5),
            ('Indianapolis Motor Speedway','Indianapolis','Indianapolis','IN','intermediate',2.5),
            ('World Wide Technology Raceway','Gateway','Madison','IL','intermediate',1.25),
            ('Sonoma Raceway','Sonoma','Sonoma','CA','road',1.99),
            ('Watkins Glen International','Watkins Glen','Watkins Glen','NY','road',2.45),
            ('Road America','Road America','Elkhart Lake','WI','road',4.048),
            ('Circuit of the Americas','COTA','Austin','TX','road',3.41),
            ('Chicago Street Course','Chicago','Chicago','IL','road',2.2),
            ('Knoxville Raceway','Knoxville','Knoxville','IA','dirt',0.5),
            ('Bristol Motor Speedway (Dirt)','Bristol Dirt','Bristol','TN','dirt',0.533);
    """)

    conn.commit()


# ── Scoring helpers ───────────────────────────────────────────────────────────

def finish_pts(position: int, platform: str) -> float:
    tbl = FINISH_PTS[platform]
    return tbl.get(position, max(0.0, 3.0 - (position - 30) * 0.5))


def implied_prob(american_odds: float) -> float:
    """Convert American odds to implied win probability (0–1)."""
    if american_odds >= 100:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def odds_to_expected_finish(win_prob: float, top5_prob: float,
                             top10_prob: float, field_size: int = 36) -> float:
    """
    Estimate expected finishing position from win/top5/top10 probabilities.
    Uses a simple weighted blend calibrated to typical NASCAR field sizes.
    """
    # Probability of each finish band:
    p_win   = win_prob
    p_top5  = max(top5_prob  - win_prob,   0)
    p_top10 = max(top10_prob - top5_prob,  0)
    p_rest  = max(1.0        - top10_prob, 0)

    # Expected position within each band
    e_top5  = 3.0          # avg of 2–5
    e_top10 = 7.5          # avg of 6–10
    e_rest  = (10 + field_size) / 2  # avg of 11–N

    return (p_win   * 1.0 +
            p_top5  * e_top5  +
            p_top10 * e_top10 +
            p_rest  * e_rest)


# ── Component 1: Track-type history (similar tracks) ─────────────────────────

def calc_track_type_score(conn, driver_id: int, track_id: int, series_code: str,
                          platform: str, current_season: int
                          ) -> tuple[float | None, int]:
    """
    Returns (score, n_races_used).
    Weighted average DFS score across ALL tracks of the same type
    (e.g., all intermediates, all superspeedways), excluding this specific track
    so there's no overlap with calc_track_score.
    Uses recency decay within the 5-year window.
    """
    type_row = conn.execute(
        "SELECT track_type FROM tracks WHERE id=?", (track_id,)
    ).fetchone()
    if not type_row or not type_row["track_type"]:
        return None, 0

    track_type = type_row["track_type"]

    rows = conn.execute("""
        SELECT dp.dfs_score,
               ABS(? - r.season) AS seasons_ago
        FROM   dfs_points dp
        JOIN   races r  ON r.id  = dp.race_id
        JOIN   tracks t ON t.id  = r.track_id
        JOIN   series s ON s.id  = r.series_id
        WHERE  dp.driver_id      = ?
          AND  dp.platform       = ?
          AND  s.code            = ?
          AND  t.track_type      = ?
          AND  r.track_id       != ?
          AND  dp.dfs_score      IS NOT NULL
          AND  r.season          < ?
          AND  ABS(? - r.season) <= 5
        ORDER  BY r.race_date DESC
        LIMIT  20
    """, (current_season,
          driver_id, platform, series_code,
          track_type, track_id,
          current_season, current_season)).fetchall()

    if not rows:
        return None, 0

    total_w = total_ws = 0.0
    for row in rows:
        w = TRACK_DECAY.get(int(row["seasons_ago"]), 0.30)
        total_ws += row["dfs_score"] * w
        total_w  += w

    return (total_ws / total_w, len(rows)) if total_w else (None, 0)


# ── Component 2: Track-specific history ──────────────────────────────────────

def calc_track_score(conn, driver_id: int, track_id: int, series_code: str,
                     platform: str, current_season: int
                     ) -> tuple[float | None, int]:
    """
    Returns (score, n_races_used).
    Weighted average DFS score at THIS exact track over the last 5 seasons.
    No longer falls back to track-type (that's its own component now).
    Returns (None, 0) if the driver has never run here.
    """
    rows = conn.execute("""
        SELECT dp.dfs_score,
               ABS(? - r.season) AS seasons_ago
        FROM   dfs_points dp
        JOIN   races r  ON r.id  = dp.race_id
        JOIN   series s ON s.id  = r.series_id
        WHERE  dp.driver_id = ?
          AND  dp.platform  = ?
          AND  s.code       = ?
          AND  r.track_id   = ?
          AND  dp.dfs_score IS NOT NULL
          AND  r.season     < ?
          AND  ABS(? - r.season) <= 5
        ORDER  BY r.race_date DESC
        LIMIT  6
    """, (current_season, driver_id, platform, series_code,
          track_id, current_season, current_season)).fetchall()

    if not rows:
        return None, 0

    total_w = total_ws = 0.0
    for row in rows:
        w = TRACK_DECAY.get(int(row["seasons_ago"]), 0.30)
        total_ws += row["dfs_score"] * w
        total_w  += w

    return (total_ws / total_w, len(rows)) if total_w else (None, 0)


# ── Component 2: Recent form score ────────────────────────────────────────────

def calc_form_score(conn, driver_id: int, series_code: str, platform: str,
                    before_race_id: int, n: int = 10
                    ) -> tuple[float | None, int]:
    """
    Returns (score, n_races_used).
    Exponentially decays recent DFS scores; most recent race = highest weight.
    Uses race_date comparison so insertion order doesn't affect results.
    """
    # Get the date of the target race so we can find all prior races by date
    race_date_row = conn.execute(
        "SELECT race_date FROM races WHERE id=?", (before_race_id,)
    ).fetchone()
    if not race_date_row or not race_date_row["race_date"]:
        return None, 0

    target_date = race_date_row["race_date"]

    rows = conn.execute("""
        SELECT dp.dfs_score
        FROM   dfs_points dp
        JOIN   races r  ON r.id = dp.race_id
        JOIN   series s ON s.id = r.series_id
        WHERE  dp.driver_id    = ?
          AND  dp.platform     = ?
          AND  s.code          = ?
          AND  r.race_date     < ?
          AND  dp.dfs_score    IS NOT NULL
        ORDER  BY r.race_date DESC
        LIMIT  ?
    """, (driver_id, platform, series_code, target_date, n)).fetchall()

    if not rows:
        return None, 0

    decay  = FORM_DECAY[:len(rows)]
    total_ws = sum(r["dfs_score"] * w for r, w in zip(rows, decay))
    total_w  = sum(decay)
    return total_ws / total_w, len(rows)


# ── Component 3: Qualifying position adjustment ───────────────────────────────

def calc_qual_adj(conn, driver_id: int, race_id: int, track_id: int,
                  series_code: str, platform: str, current_season: int
                  ) -> float:
    """
    Returns a point delta based on whether the driver qualifies better or
    worse than their historical average start at this track.
    Positive = qualified better → gains place-diff points.
    Returns 0.0 if no qualifying data available.
    """
    # This week's qualifying position
    qual_row = conn.execute(
        "SELECT q_position FROM qualifying_results WHERE race_id=? AND driver_id=?",
        (race_id, driver_id)
    ).fetchone()
    if not qual_row or qual_row["q_position"] is None:
        return 0.0

    this_qual = qual_row["q_position"]

    # Historical average START position at this track
    avg_start = conn.execute("""
        SELECT AVG(rr.start_pos)
        FROM   race_results rr
        JOIN   races r  ON r.id = rr.race_id
        JOIN   series s ON s.id = r.series_id
        WHERE  rr.driver_id = ?
          AND  r.track_id   = ?
          AND  s.code       = ?
          AND  rr.start_pos IS NOT NULL
          AND  r.season     < ?
    """, (driver_id, track_id, series_code, current_season)).fetchone()[0]

    if avg_start is None:
        # Fall back to overall series average start position
        avg_start = conn.execute("""
            SELECT AVG(rr.start_pos)
            FROM   race_results rr
            JOIN   races r  ON r.id = rr.race_id
            JOIN   series s ON s.id = r.series_id
            WHERE  rr.driver_id = ?
              AND  s.code       = ?
              AND  rr.start_pos IS NOT NULL
              AND  r.season     < ?
        """, (driver_id, series_code, current_season)).fetchone()[0]

    if avg_start is None:
        return 0.0

    # Delta: positive = starting further up front than usual → more pts
    rate = PLACE_DIFF_RATE[platform]
    return round((avg_start - this_qual) * rate, 2)


# ── Component 4: Practice speed adjustment ────────────────────────────────────

def calc_practice_adj(conn, driver_id: int, race_id: int,
                      series_code: str, current_season: int) -> float:
    """
    Returns a point delta based on practice rank relative to field.
    Uses long_run_avg (if available) as the primary signal, falls back to
    best_speed rank.  Long-run avg captures pace falloff better than single-lap.

    Positive = faster than expected (better rank) → car likely to run well.
    Capped at MAX_PRACTICE_ADJ to prevent outlier sessions distorting projections.
    """
    # This week's practice result (best available session)
    prac_row = conn.execute("""
        SELECT rank, laps_run, long_run_avg, best_speed,
               (SELECT COUNT(*) FROM practice_results WHERE race_id=? AND session=pr.session)
               AS field_size
        FROM   practice_results pr
        WHERE  pr.race_id   = ?
          AND  pr.driver_id = ?
        ORDER  BY
          CASE WHEN long_run_avg IS NOT NULL THEN 0 ELSE 1 END,
          pr.laps_run DESC,
          pr.session  DESC
        LIMIT  1
    """, (race_id, race_id, driver_id)).fetchone()

    if not prac_row or prac_row["rank"] is None:
        return 0.0

    rank       = prac_row["rank"]
    field_size = max(int(prac_row["field_size"] or 0) or 36, 2)  # min 2 to avoid /0
    laps_run   = int(prac_row["laps_run"] or 0)

    # Percentile rank (0 = worst, 1 = best)
    pct = 1.0 - (rank - 1) / (field_size - 1) if field_size > 1 else 0.5

    # Long-run vs best-lap falloff signal (if available)
    falloff_penalty = 0.0
    if prac_row["long_run_avg"] and prac_row["best_speed"]:
        # long_run_avg < best_speed means pace dropped off during long run
        falloff_ratio = prac_row["long_run_avg"] / prac_row["best_speed"]
        # ratio close to 1.0 = minimal falloff (good); lower = falls off
        if falloff_ratio < 0.995:
            falloff_penalty = (0.995 - falloff_ratio) * 40  # ~0.2 pts per 0.5% falloff

    # Weight by how many laps were run (more laps = more reliable signal)
    lap_confidence = min(laps_run / 20, 1.0)  # full confidence at 20+ laps

    # Center at 0.5 (median rank): positive above, negative below
    raw_adj = (pct - 0.5) * MAX_PRACTICE_ADJ * 2 * lap_confidence - falloff_penalty

    return round(max(-MAX_PRACTICE_ADJ, min(MAX_PRACTICE_ADJ, raw_adj)), 2)


# ── Component 5: Odds adjustment ──────────────────────────────────────────────

def calc_odds_adj(conn, driver_id: int, race_id: int,
                  platform: str, baseline_proj: float, field_size: int = 36
                  ) -> float:
    """
    Returns a point delta that nudges the baseline toward the sportsbook's view.
    Uses American odds from the odds table (win / top5 / top10).
    Capped at MAX_ODDS_ADJ.  Returns 0.0 if no odds data.
    """
    odds_row = conn.execute("""
        SELECT win_odds, top5_odds, top10_odds
        FROM   odds
        WHERE  race_id   = ?
          AND  driver_id = ?
        ORDER  BY scraped_at DESC
        LIMIT  1
    """, (race_id, driver_id)).fetchone()

    if not odds_row or all(
        odds_row[k] is None for k in ("win_odds", "top5_odds", "top10_odds")
    ):
        return 0.0

    # Convert available odds to probabilities (fallback to rough estimates)
    def safe_prob(field, fallback):
        return implied_prob(odds_row[field]) if odds_row[field] else fallback

    p_win   = safe_prob("win_odds",  0.028)   # ~1/36 base
    p_top5  = safe_prob("top5_odds", 0.139)   # ~5/36 base
    p_top10 = safe_prob("top10_odds",0.278)   # ~10/36 base

    # Odds-implied expected finish
    impl_finish = odds_to_expected_finish(p_win, p_top5, p_top10, field_size)

    # Odds-implied DFS score from finish pts alone (rough but directionally right)
    impl_pts   = finish_pts(round(impl_finish), platform)

    # Compare to baseline
    delta = (impl_pts - baseline_proj) * 0.25   # blend 25% toward odds view
    return round(max(-MAX_ODDS_ADJ, min(MAX_ODDS_ADJ, delta)), 2)


# ── Field average helper ──────────────────────────────────────────────────────

def field_avg_score(conn, race_id: int, platform: str) -> float:
    """Average DFS score across all drivers with salary in this race."""
    # Use series + season avg as field proxy when we have no race-specific data
    race_meta = conn.execute("""
        SELECT s.code AS series, r.season
        FROM   races r JOIN series s ON s.id = r.series_id
        WHERE  r.id = ?
    """, (race_id,)).fetchone()
    if not race_meta:
        return 32.0   # overall sensible default

    row = conn.execute("""
        SELECT AVG(dp.dfs_score)
        FROM   dfs_points dp
        JOIN   races r  ON r.id = dp.race_id
        JOIN   series s ON s.id = r.series_id
        WHERE  s.code      = ?
          AND  dp.platform = ?
          AND  r.season    < ?
          AND  dp.dfs_score IS NOT NULL
    """, (race_meta["series"], platform, race_meta["season"])).fetchone()

    return row[0] if row and row[0] else 32.0


# ── Main projection builder ───────────────────────────────────────────────────

def project_race(conn, race_id: int, platform: str,
                 weights: dict, form_n: int = 10) -> list[dict]:
    """
    Project every driver who has a salary entry for race_id + platform.
    Returns a list of projection dicts, sorted by proj_score descending.
    """
    # Race metadata
    race = conn.execute("""
        SELECT r.id, r.season, r.race_name, r.race_date,
               r.track_id,    r.laps,
               s.code         AS series,
               t.name         AS track_name,
               t.track_type
        FROM   races r
        JOIN   series s ON s.id = r.series_id
        LEFT JOIN tracks t ON t.id = r.track_id
        WHERE  r.id = ?
    """, (race_id,)).fetchone()

    if not race:
        print(f"[ERROR] Race {race_id} not found.")
        return []

    season  = race["season"]
    series  = race["series"]
    track_id = race["track_id"]
    w_total  = sum(weights.values())

    f_avg = field_avg_score(conn, race_id, platform)

    # Estimate field size for this race
    field_size = conn.execute(
        "SELECT COUNT(DISTINCT driver_id) FROM salaries WHERE race_id=? AND platform=?",
        (race_id, platform)
    ).fetchone()[0] or 36

    # Drivers with salaries for this race — exclude anyone marked Out
    # (scratched / DNQ / withdrawal confirmed before race day)
    salary_rows = conn.execute("""
        SELECT s.driver_id, s.salary, d.full_name
        FROM   salaries s
        JOIN   drivers  d ON d.id = s.driver_id
        WHERE  s.race_id  = ?
          AND  s.platform = ?
          AND  COALESCE(s.status, 'Available') != 'Out'
        ORDER  BY s.salary DESC
    """, (race_id, platform)).fetchall()

    if not salary_rows:
        print(f"  [WARN] No salary data for race {race_id} / {platform}. "
              f"Run the salary scraper first.")
        return []

    projections = []
    w = weights

    for row in salary_rows:
        driver_id  = row["driver_id"]
        salary     = row["salary"]
        name       = row["full_name"]

        # ── Component 1: Track-type history (similar tracks) ──────────────────
        track_type_score, track_type_n = calc_track_type_score(
            conn, driver_id, track_id, series, platform, season
        ) if track_id else (None, 0)

        # ── Component 2: Track-specific history ───────────────────────────────
        track_score, track_n = calc_track_score(
            conn, driver_id, track_id, series, platform, season
        ) if track_id else (None, 0)

        # ── Component 3: Recent form ──────────────────────────────────────────
        form_score, form_n_used = calc_form_score(
            conn, driver_id, series, platform, race_id, n=form_n
        )

        # ── Component 4: Practice ─────────────────────────────────────────────
        practice_adj = calc_practice_adj(conn, driver_id, race_id, series, season)

        # ── Component 5: Odds ─────────────────────────────────────────────────
        # Odds produce an absolute score, not a delta — treat like a baseline signal
        odds_adj = calc_odds_adj(conn, driver_id, race_id, platform,
                                 f_avg, field_size)

        # ── Component 6: Qual position ────────────────────────────────────────
        qual_adj = calc_qual_adj(
            conn, driver_id, race_id, track_id, series, platform, season
        ) if track_id else 0.0

        # ── Weighted blend ────────────────────────────────────────────────────
        # Baseline components (all on DFS-point scale): track_type, track, form
        # Adjustment components (point deltas): practice, odds, qual
        # We weight-average the baselines, then add scaled adjustments.

        baselines  = []
        b_weights  = []
        for score, key in [(track_type_score, "track_type"),
                           (track_score,      "track"),
                           (form_score,       "form")]:
            if score is not None:
                baselines.append(score)
                b_weights.append(w[key])

        baseline = (
            sum(s * wt for s, wt in zip(baselines, b_weights)) / sum(b_weights)
            if baselines else f_avg
        )

        # Scale adjustments by their weight share so heavier weights mean
        # more influence on the final number
        adj_total_w = w["practice"] + w["odds"] + w["qual"]
        proj = (
            baseline
            + practice_adj * (w["practice"] / adj_total_w)
            + odds_adj     * (w["odds"]     / adj_total_w)
            + qual_adj     * (w["qual"]     / adj_total_w)
        )

        proj  = round(proj, 2)
        value = round(proj / salary * 1000, 2) if salary else None

        projections.append({
            "driver_id":        driver_id,
            "name":             name,
            "salary":           salary,
            "proj_score":       proj,
            "value":            value,
            "track_type_score": round(track_type_score, 2) if track_type_score is not None else None,
            "track_score":      round(track_score,      2) if track_score      is not None else None,
            "form_score":       round(form_score,       2) if form_score       is not None else None,
            "qual_adj":         qual_adj,
            "practice_adj":     practice_adj,
            "odds_adj":         odds_adj,
            "track_races":      track_n,
            "track_type_races": track_type_n,
            "form_races":       form_n_used,
            "track_type_used":  0,   # no longer a fallback flag; always its own signal
        })

    projections.sort(key=lambda x: x["proj_score"], reverse=True)
    return projections


# ── DB write ──────────────────────────────────────────────────────────────────

def save_projections(conn, race_id: int, platform: str,
                     projections: list[dict]) -> int:
    """Upsert projections into DB. Returns count saved."""
    ts = datetime.now().isoformat(timespec="seconds")
    saved = 0
    for p in projections:
        conn.execute("""
            INSERT INTO projections
                (race_id, driver_id, platform,
                 proj_score, salary, value,
                 track_score, track_type_score, form_score,
                 qual_adj, practice_adj, odds_adj,
                 track_races, form_races, track_type_used, generated_at)
            VALUES (?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?,?)
            ON CONFLICT(race_id, driver_id, platform)
            DO UPDATE SET
                proj_score       = excluded.proj_score,
                salary           = excluded.salary,
                value            = excluded.value,
                track_score      = excluded.track_score,
                track_type_score = excluded.track_type_score,
                form_score       = excluded.form_score,
                qual_adj         = excluded.qual_adj,
                practice_adj     = excluded.practice_adj,
                odds_adj         = excluded.odds_adj,
                track_races      = excluded.track_races,
                form_races       = excluded.form_races,
                track_type_used  = excluded.track_type_used,
                generated_at     = excluded.generated_at
        """, (
            race_id, p["driver_id"], platform,
            p["proj_score"], p["salary"], p["value"],
            p["track_score"], p["track_type_score"], p["form_score"],
            p["qual_adj"], p["practice_adj"], p["odds_adj"],
            p["track_races"], p["form_races"], p["track_type_used"], ts,
        ))
        saved += 1
    conn.commit()
    return saved


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(projections: list[dict], race_meta: sqlite3.Row,
               platform: str) -> str:
    """Write projections to exports/ folder. Returns file path."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    date_str  = (race_meta["race_date"] or datetime.now().strftime("%Y-%m-%d"))
    track_str = (race_meta["track_name"] or "unknown").replace(" ", "_")
    plat_tag  = "DK" if platform == "DraftKings" else "FD"
    filename  = f"projections_{plat_tag}_{date_str}_{track_str}.csv"
    filepath  = os.path.join(EXPORT_DIR, filename)

    cols = [
        ("Rank",             lambda i, p: i + 1),
        ("Driver",           lambda i, p: p["name"]),
        ("Salary",           lambda i, p: p["salary"]),
        ("Proj Score",       lambda i, p: p["proj_score"]),
        ("Value (pts/K)",    lambda i, p: p["value"]),
        ("Track Avg",        lambda i, p: p["track_score"]      if p["track_score"]      is not None else "—"),
        ("Track Races",      lambda i, p: p["track_races"]),
        ("TrackType Avg",    lambda i, p: p["track_type_score"] if p["track_type_score"] is not None else "—"),
        ("TrackType Races",  lambda i, p: p["track_type_races"]),
        ("Recent Form",      lambda i, p: p["form_score"]       if p["form_score"]       is not None else "—"),
        ("Form Races",       lambda i, p: p["form_races"]),
        ("Qual Adj",         lambda i, p: p["qual_adj"]),
        ("Practice Adj",     lambda i, p: p["practice_adj"]),
        ("Odds Adj",         lambda i, p: p["odds_adj"]),
    ]

    def _fmt(v):
        """Convert None to empty string for cleaner CSV output."""
        return "" if v is None else v

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([c[0] for c in cols])
        for i, p in enumerate(projections):
            writer.writerow([_fmt(fn(i, p)) for _, fn in cols])

    return filepath


# ── Print table ───────────────────────────────────────────────────────────────

def print_table(projections: list[dict], platform: str, race_meta) -> None:
    race_str = (f"{race_meta['race_name'] or 'Race'} | "
                f"{race_meta['race_date']} | {race_meta['track_name'] or '?'}")
    plat_tag = "DK" if platform == "DraftKings" else "FD"

    print(f"\n{'='*90}")
    print(f"  {plat_tag} PROJECTIONS  —  {race_str}")
    print(f"{'='*90}")
    print(f"  {'#':>2}  {'Driver':<24} {'Salary':>7}  {'Proj':>5}  {'Val':>5}  "
          f"{'Trk':>5}  {'TrkTyp':>6}  {'Form':>5}  {'Q':>5}  {'P':>5}  {'O':>5}")
    print(f"  {'─'*86}")

    for i, p in enumerate(projections[:40]):
        trk  = f"{p['track_score']:.1f}"      if p["track_score"]      is not None else "   —"
        ttyp = f"{p['track_type_score']:.1f}" if p["track_type_score"] is not None else "    —"
        frm  = f"{p['form_score']:.1f}"       if p["form_score"]       is not None else "   —"
        val  = f"{p['value']:.2f}"            if p["value"]            is not None else "   —"
        sal  = f"${p['salary']:>6,}" if p["salary"] else "      ?"
        print(
            f"  {i+1:>2}  {p['name']:<24} "
            f"{sal}  "
            f"{p['proj_score']:>5.1f}  "
            f"{val:>5}  "
            f"{trk:>5}  {ttyp:>6}  {frm:>5}  "
            f"{p['qual_adj']:>+5.1f}  "
            f"{p['practice_adj']:>+5.1f}  "
            f"{p['odds_adj']:>+5.1f}"
        )


# ── Race finder ───────────────────────────────────────────────────────────────

def find_target_race(conn, series: str, race_id_override: int | None):
    """Return race row for the race to project."""
    if race_id_override:
        return conn.execute("""
            SELECT r.id, r.season, r.race_name, r.race_date, r.track_id, r.laps,
                   s.code AS series, t.name AS track_name, t.track_type
            FROM   races r
            JOIN   series s ON s.id = r.series_id
            LEFT JOIN tracks t ON t.id = r.track_id
            WHERE  r.id = ?
        """, (race_id_override,)).fetchone()

    # Auto-detect: most recent race that has salary data
    return conn.execute("""
        SELECT r.id, r.season, r.race_name, r.race_date, r.track_id, r.laps,
               s.code AS series, t.name AS track_name, t.track_type
        FROM   races r
        JOIN   series s ON s.id = r.series_id
        LEFT JOIN tracks t ON t.id = r.track_id
        WHERE  s.code = ?
          AND  r.id IN (SELECT DISTINCT race_id FROM salaries)
        ORDER  BY r.race_date DESC
        LIMIT  1
    """, (series,)).fetchone()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NASCAR DFS projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
weights are percentages that auto-normalize if they don't sum to 100:
  --weights TRACK TRACK_TYPE PRACTICE ODDS QUAL FORM
  e.g.  --weights 25 20 20 15 12  8   (default)
        --weights 30 25 20 15  5  5   (even heavier on track history)
        --weights 20 15 25 20 10 10   (lean harder on practice + odds)
        --weights 35 20 15 15  5 10   (super track-specific focus)
        """,
    )
    parser.add_argument("--race-id",  type=int, default=None,
                        help="Race ID to project (default: latest with salary data)")
    parser.add_argument("--series",   default="cup",
                        choices=["cup", "xfinity", "trucks"])
    parser.add_argument("--platform", nargs="+",
                        default=["DraftKings", "FanDuel"],
                        choices=["DraftKings", "FanDuel"])
    parser.add_argument("--form-races", type=int, default=10,
                        help="Recent races to include in form score (default: 10)")
    parser.add_argument("--weights", nargs=6, type=float, metavar="W",
                        default=None,
                        help="Weights: TRACK TRACK_TYPE PRACTICE ODDS QUAL FORM  (must sum to ~100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print projections but don't save to DB or CSV")
    args = parser.parse_args()

    # Build weights dict
    if args.weights:
        keys = ["track", "track_type", "practice", "odds", "qual", "form"]
        weights = dict(zip(keys, args.weights))
    else:
        weights = dict(DEFAULT_WEIGHTS)

    # Normalize
    total = sum(weights.values())
    weights = {k: v / total * 100 for k, v in weights.items()}

    conn = get_conn()

    race = find_target_race(conn, args.series, args.race_id)
    if not race:
        print(f"[ERROR] No race found for series='{args.series}'.")
        print("  Make sure salary data has been imported (run RUN_SCRAPER.bat).")
        conn.close()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  NASCAR DFS Projections Engine")
    print(f"  Race   : {race['race_name'] or '(unnamed)'}")
    print(f"  Date   : {race['race_date']}")
    print(f"  Track  : {race['track_name'] or '?'} ({race['track_type'] or '?'})")
    print(f"  Series : {race['series'].upper()}")
    if args.dry_run:
        print(f"  Mode   : DRY RUN")
    print(f"\n  Weights:")
    for k, v in weights.items():
        print(f"    {k.capitalize():<10}: {v:.0f}%")
    print(f"{'='*60}\n")

    for platform in args.platform:
        print(f"\n── {platform} ──────────────────────────────────────────")
        projs = project_race(conn, race["id"], platform, weights, args.form_races)

        if not projs:
            continue

        print_table(projs, platform, race)

        if not args.dry_run:
            n = save_projections(conn, race["id"], platform, projs)
            csv_path = export_csv(projs, race, platform)
            print(f"\n  [✓] {n} projections saved to DB")
            print(f"  [✓] CSV  →  {csv_path}")

    conn.close()
    print()


if __name__ == "__main__":
    main()
