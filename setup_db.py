"""
NASCAR DFS Optimizer - Database Setup
Creates nascar.db with all required tables.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nascar.db")


def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;

    -- ── SERIES ──────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS series (
        id          INTEGER PRIMARY KEY,
        code        TEXT UNIQUE NOT NULL,   -- 'cup','xfinity','trucks'
        name        TEXT NOT NULL
    );
    INSERT OR IGNORE INTO series(code,name) VALUES
        ('cup',     'NASCAR Cup Series'),
        ('xfinity', 'NASCAR O''Reilly Auto Parts Series'),
        ('trucks',  'NASCAR Craftsman Truck Series');

    -- ── TRACKS ──────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS tracks (
        id           INTEGER PRIMARY KEY,
        name         TEXT NOT NULL,
        short_name   TEXT,
        city         TEXT,
        state        TEXT,
        track_type   TEXT,   -- 'superspeedway','intermediate','short','road','dirt'
        length_miles REAL,
        UNIQUE(name)
    );

    -- ── KNOWN TRACKS (seed data) ────────────────────────────────
    INSERT OR IGNORE INTO tracks(name,short_name,city,state,track_type,length_miles) VALUES
        ('Daytona International Speedway',   'Daytona',      'Daytona Beach', 'FL', 'superspeedway', 2.5),
        ('Talladega Superspeedway',          'Talladega',    'Talladega',     'AL', 'superspeedway', 2.66),
        ('Atlanta Motor Speedway',           'Atlanta',      'Hampton',       'GA', 'superspeedway', 1.54),
        ('Charlotte Motor Speedway',         'Charlotte',    'Concord',       'NC', 'intermediate',  1.5),
        ('Texas Motor Speedway',             'Texas',        'Fort Worth',    'TX', 'intermediate',  1.5),
        ('Kansas Speedway',                  'Kansas',       'Kansas City',   'KS', 'intermediate',  1.5),
        ('Las Vegas Motor Speedway',         'Las Vegas',    'Las Vegas',     'NV', 'intermediate',  1.5),
        ('Michigan International Speedway',  'Michigan',     'Brooklyn',      'MI', 'intermediate',  2.0),
        ('Homestead-Miami Speedway',         'Homestead',    'Homestead',     'FL', 'intermediate',  1.5),
        ('Nashville Superspeedway',          'Nashville',    'Lebanon',       'TN', 'intermediate',  1.33),
        ('Iowa Speedway',                    'Iowa',         'Newton',        'IA', 'short',         0.875),
        ('Bristol Motor Speedway',           'Bristol',      'Bristol',       'TN', 'short',         0.533),
        ('Martinsville Speedway',            'Martinsville', 'Martinsville',  'VA', 'short',         0.526),
        ('Richmond Raceway',                 'Richmond',     'Richmond',      'VA', 'short',         0.75),
        ('New Hampshire Motor Speedway',     'New Hampshire','Loudon',        'NH', 'short',         1.058),
        ('Phoenix Raceway',                  'Phoenix',      'Avondale',      'AZ', 'short',         1.0),
        ('Dover Motor Speedway',             'Dover',        'Dover',         'DE', 'short',         1.0),
        ('North Wilkesboro Speedway',        'North Wilkesboro','North Wilkesboro','NC','short',     0.625),
        ('Rockingham Speedway',              'Rockingham',   'Rockingham',    'NC', 'short',         0.94),
        ('Pocono Raceway',                   'Pocono',       'Long Pond',     'PA', 'intermediate',  2.5),
        ('Indianapolis Motor Speedway',      'Indianapolis', 'Indianapolis',  'IN', 'intermediate',  2.5),
        ('World Wide Technology Raceway',    'Gateway',      'Madison',       'IL', 'intermediate',  1.25),
        ('Darlington Raceway',               'Darlington',   'Darlington',    'SC', 'intermediate',  1.366),
        ('Sonoma Raceway',                   'Sonoma',       'Sonoma',        'CA', 'road',          1.99),
        ('Watkins Glen International',       'Watkins Glen', 'Watkins Glen',  'NY', 'road',          2.45),
        ('Road America',                     'Road America', 'Elkhart Lake',  'WI', 'road',          4.048),
        ('Circuit of the Americas',          'COTA',         'Austin',        'TX', 'road',          3.41),
        ('Chicago Street Course',            'Chicago',      'Chicago',       'IL', 'road',          2.2),
        ('Knoxville Raceway',                'Knoxville',    'Knoxville',     'IA', 'dirt',          0.5),
        ('Bristol Motor Speedway (Dirt)',    'Bristol Dirt', 'Bristol',       'TN', 'dirt',          0.533);

    -- ── DRIVERS ─────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS drivers (
        id         INTEGER PRIMARY KEY,
        full_name  TEXT NOT NULL,
        first_name TEXT,
        last_name  TEXT,
        UNIQUE(full_name)
    );

    -- ── RACES ───────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS races (
        id            INTEGER PRIMARY KEY,
        series_id     INTEGER NOT NULL REFERENCES series(id),
        track_id      INTEGER REFERENCES tracks(id),
        season        INTEGER NOT NULL,
        race_num      INTEGER NOT NULL,
        race_name     TEXT,
        race_date     TEXT,   -- ISO-8601
        laps          INTEGER,
        miles         REAL,
        api_race_id   INTEGER,   -- NASCAR API race_id for cross-reference
        UNIQUE(series_id, season, race_num)
    );

    -- ── RACE RESULTS ────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS race_results (
        id             INTEGER PRIMARY KEY,
        race_id        INTEGER NOT NULL REFERENCES races(id),
        driver_id      INTEGER NOT NULL REFERENCES drivers(id),
        car_number     TEXT,
        team           TEXT,
        manufacturer   TEXT,
        start_pos      INTEGER,
        finish_pos     INTEGER,
        laps_completed INTEGER,
        laps_led       INTEGER,
        fastest_laps   INTEGER DEFAULT 0,
        status         TEXT,   -- 'Running','Accident','Engine', etc.
        points         INTEGER,
        money          REAL,
        avg_running_position REAL,  -- avg position across all race laps
        UNIQUE(race_id, driver_id)
    );

    -- ── DFS POINTS ──────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS dfs_points (
        id          INTEGER PRIMARY KEY,
        race_id     INTEGER NOT NULL REFERENCES races(id),
        driver_id   INTEGER NOT NULL REFERENCES drivers(id),
        platform    TEXT NOT NULL,   -- 'DraftKings','FanDuel'
        dfs_score   REAL,
        -- score breakdown
        place_pts       REAL,
        place_diff_pts  REAL,
        laps_led_pts    REAL,
        fastest_laps_pts REAL,
        UNIQUE(race_id, driver_id, platform)
    );

    -- ── PRACTICE RESULTS ────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS practice_results (
        id            INTEGER PRIMARY KEY,
        race_id       INTEGER NOT NULL REFERENCES races(id),
        driver_id     INTEGER NOT NULL REFERENCES drivers(id),
        session       INTEGER DEFAULT 1,
        best_speed    REAL,       -- single-lap best (mph)
        best_lap      REAL,       -- single-lap best (seconds)
        laps_run      INTEGER,    -- total laps completed in session
        rank          INTEGER,    -- position in session results
        long_run_avg  REAL,       -- avg speed over 10+ consecutive laps (long-run pace)
        avg_lap       REAL,       -- average lap time across full session (falloff indicator)
        UNIQUE(race_id, driver_id, session)
    );

    -- ── QUALIFYING RESULTS ──────────────────────────────────────
    CREATE TABLE IF NOT EXISTS qualifying_results (
        id          INTEGER PRIMARY KEY,
        race_id     INTEGER NOT NULL REFERENCES races(id),
        driver_id   INTEGER NOT NULL REFERENCES drivers(id),
        q_position  INTEGER,
        q_speed     REAL,
        q_time      REAL,
        UNIQUE(race_id, driver_id)
    );

    -- ── ODDS ────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS odds (
        id           INTEGER PRIMARY KEY,
        race_id      INTEGER NOT NULL REFERENCES races(id),
        driver_id    INTEGER NOT NULL REFERENCES drivers(id),
        sportsbook   TEXT,
        win_odds     REAL,
        top3_odds    REAL,
        top5_odds    REAL,
        top10_odds   REAL,
        scraped_at   TEXT,
        UNIQUE(race_id, driver_id, sportsbook)
    );

    -- ── SALARIES ────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS salaries (
        id          INTEGER PRIMARY KEY,
        race_id     INTEGER NOT NULL REFERENCES races(id),
        driver_id   INTEGER NOT NULL REFERENCES drivers(id),
        platform    TEXT NOT NULL,   -- 'DraftKings','FanDuel'
        salary      INTEGER,
        status      TEXT NOT NULL DEFAULT 'Available',  -- 'Available','Out','Questionable','Probable'
        UNIQUE(race_id, driver_id, platform)
    );

    -- ── PROJECTIONS ─────────────────────────────────────────────
    -- Regenerated each week by projections.py.
    -- Stores final projected DFS score + all components so the UI
    -- can show exactly how each number was built.
    CREATE TABLE IF NOT EXISTS projections (
        id              INTEGER PRIMARY KEY,
        race_id         INTEGER NOT NULL REFERENCES races(id),
        driver_id       INTEGER NOT NULL REFERENCES drivers(id),
        platform        TEXT NOT NULL,         -- 'DraftKings' | 'FanDuel'
        -- Final numbers
        proj_score      REAL,                  -- projected DFS points
        salary          INTEGER,               -- from salaries table (denorm)
        value           REAL,                  -- proj_score / salary * 1000
        -- Component scores (all in DFS-point units, stored separately so the
        -- web UI can apply its own weight sliders without re-running Python)
        track_score      REAL,                 -- weighted avg at THIS specific track
        track_type_score REAL,                 -- weighted avg at similar track types
        form_score       REAL,                 -- recent-form weighted avg
        qual_adj         REAL,                 -- qual-position point delta
        practice_adj     REAL,                 -- practice-speed point delta
        odds_adj         REAL,                 -- odds-implied point delta
        -- Confidence metadata
        track_races     INTEGER,               -- # track visits used
        form_races      INTEGER,               -- # recent races used
        track_type_used INTEGER DEFAULT 0,     -- 1 if fell back to track-type avg
        -- Timestamp
        generated_at    TEXT DEFAULT (datetime('now')),
        UNIQUE(race_id, driver_id, platform)
    );

    -- ── INDEXES ─────────────────────────────────────────────────
    CREATE INDEX IF NOT EXISTS idx_race_results_race   ON race_results(race_id);
    CREATE INDEX IF NOT EXISTS idx_race_results_driver ON race_results(driver_id);
    CREATE INDEX IF NOT EXISTS idx_dfs_race            ON dfs_points(race_id);
    CREATE INDEX IF NOT EXISTS idx_dfs_driver          ON dfs_points(driver_id);
    CREATE INDEX IF NOT EXISTS idx_races_season        ON races(season);
    CREATE INDEX IF NOT EXISTS idx_salaries_race       ON salaries(race_id);
    CREATE INDEX IF NOT EXISTS idx_proj_race           ON projections(race_id);
    CREATE INDEX IF NOT EXISTS idx_proj_driver         ON projections(driver_id);
    """)
    conn.commit()
    conn.close()
    print(f"[OK] Database created at {DB_PATH}")


if __name__ == "__main__":
    create_database()
