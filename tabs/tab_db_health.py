"""DB Health tab.

Surfaces data-quality anomalies so bugs like the Kansas salary bleed-through
are caught automatically instead of by accident. Everything is read-only —
no writes happen here. Findings are grouped by severity.
"""
from __future__ import annotations
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DB_PATH
from src.utils import normalize_driver_name


def _file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024 / 1024
    except Exception:
        return 0.0


def _row_counts(conn) -> dict:
    tables = [
        "races", "drivers", "tracks", "race_results",
        "salaries", "odds", "saved_projections",
    ]
    counts = {}
    for t in tables:
        try:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            counts[t] = None
    # Rename for clearer display
    counts["projections"] = counts.pop("saved_projections", 0)
    return counts


def _projections_coverage(conn) -> dict:
    """How many races have saved projections vs total races?

    A "covered" race is one with at least one row in saved_projections.
    Useful for tracking how well we're capturing snapshot-in-time projections
    as a % of races the DB knows about.
    """
    try:
        total = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        covered = conn.execute(
            "SELECT COUNT(DISTINCT race_id) FROM saved_projections"
        ).fetchone()[0]
        return {"total": total, "covered": covered,
                "pct": (covered / total * 100) if total else 0.0}
    except Exception:
        return {"total": 0, "covered": 0, "pct": 0.0}


def _find_salary_fingerprint_collisions(conn) -> pd.DataFrame:
    """Detect races that share the exact same DK salary set (same drivers + prices).

    This catches the Kansas-style bug where salaries get written to the wrong
    year's race_id.
    """
    races_with_sal = conn.execute('''
        SELECT DISTINCT s.race_id, r.race_name, r.race_date, r.series_id
        FROM salaries s JOIN races r ON r.id = s.race_id
        WHERE s.platform = 'DraftKings'
    ''').fetchall()

    fp_map = defaultdict(list)
    for rid, rname, rdate, sid in races_with_sal:
        rows = conn.execute('''
            SELECT driver_id, salary FROM salaries
            WHERE race_id = ? AND platform = 'DraftKings'
            ORDER BY driver_id
        ''', (rid,)).fetchall()
        if not rows:
            continue
        fp = tuple(rows)
        fp_map[fp].append((rid, rname, rdate, sid))

    collisions = []
    for fp, races in fp_map.items():
        if len(races) > 1:
            for rid, rname, rdate, sid in races:
                collisions.append({
                    "Race ID": rid,
                    "Series": {1: "Cup", 2: "O'Reilly", 3: "Truck"}.get(sid, str(sid)),
                    "Race": rname,
                    "Date": (rdate or "")[:10],
                    "Group Size": len(races),
                })
    return pd.DataFrame(collisions).sort_values(
        ["Group Size", "Race"], ascending=[False, True]
    ) if collisions else pd.DataFrame()


def _find_races_missing_data(conn, series_id: int | None = None, season: int | None = None) -> pd.DataFrame:
    """Find races missing critical data (salaries / odds / ARP)."""
    params = []
    where = "1=1"
    if series_id:
        where += " AND r.series_id = ?"
        params.append(series_id)
    if season:
        where += " AND r.season = ?"
        params.append(season)

    q = f'''
        SELECT
            r.id, r.race_name, r.race_date, r.series_id, r.season,
            (SELECT COUNT(*) FROM salaries s WHERE s.race_id = r.id AND s.platform = 'DraftKings') as n_sal,
            (SELECT COUNT(*) FROM odds o WHERE o.race_id = r.id) as n_odds,
            (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id) as n_results,
            (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id = r.id AND rr.avg_running_position IS NOT NULL) as n_arp
        FROM races r
        WHERE {where}
    '''
    rows = conn.execute(q, params).fetchall()

    today = datetime.now().strftime("%Y-%m-%d")
    series_names = {1: "Cup", 2: "O'Reilly", 3: "Truck"}
    out = []
    for rid, rname, rdate, sid, season_year, n_sal, n_odds, n_results, n_arp in rows:
        is_past = (rdate or "") < today
        # Report what's missing
        missing = []
        if n_sal == 0:
            missing.append("Salaries")
        if n_odds == 0:
            missing.append("Odds")
        if is_past:
            if n_results == 0:
                missing.append("Results")
            elif n_arp == 0:
                missing.append("ARP")
        if not missing:
            continue
        out.append({
            "Season": season_year,
            "Series": series_names.get(sid, str(sid)),
            "Date": (rdate or "")[:10],
            "Race": rname,
            "Missing": ", ".join(missing),
            "Past": is_past,
        })
    return pd.DataFrame(out)


def _find_orphaned_rows(conn) -> dict:
    """Find salary/odds rows referencing races that don't exist."""
    orphan_sal = conn.execute('''
        SELECT COUNT(*) FROM salaries s
        WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = s.race_id)
    ''').fetchone()[0]
    orphan_odds = conn.execute('''
        SELECT COUNT(*) FROM odds o
        WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = o.race_id)
    ''').fetchone()[0]
    orphan_results = conn.execute('''
        SELECT COUNT(*) FROM race_results rr
        WHERE NOT EXISTS (SELECT 1 FROM races r WHERE r.id = rr.race_id)
    ''').fetchone()[0]
    return {
        "salaries": orphan_sal,
        "odds": orphan_odds,
        "race_results": orphan_results,
    }


def _find_duplicate_drivers(conn) -> pd.DataFrame:
    """Find driver rows that normalize to the same key AND aren't provably
    distinct (no shared races).

    Two drivers that BOTH appear in the same race can't be the same person
    (one driver, one car per race), so we exclude those pairs. This avoids
    false positives like "Austin Hill" vs "Austin J Hill" — two different
    drivers who've raced each other and happen to share a last name.
    """
    rows = conn.execute("SELECT id, full_name FROM drivers").fetchall()
    norm_map = defaultdict(list)
    for did, name in rows:
        nn = normalize_driver_name(name)
        norm_map[nn].append((did, name))

    def _shares_race(a_id, b_id):
        r = conn.execute(
            "SELECT 1 FROM race_results ra "
            "JOIN race_results rb ON rb.race_id = ra.race_id "
            "WHERE ra.driver_id = ? AND rb.driver_id = ? LIMIT 1",
            (a_id, b_id)
        ).fetchone()
        return r is not None

    out = []
    for nn, entries in norm_map.items():
        if len(entries) <= 1:
            continue
        # Skip if any pair in the group has raced the same event (distinct drivers)
        distinct = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if _shares_race(entries[i][0], entries[j][0]):
                    distinct = True
                    break
            if distinct:
                break
        if distinct:
            continue
        out.append({
            "Normalized Key": nn,
            "Duplicate Names": " | ".join(f"{n} (id={i})" for i, n in entries),
            "Count": len(entries),
        })
    return pd.DataFrame(out).sort_values("Count", ascending=False) if out else pd.DataFrame()


def _find_duplicate_races(conn) -> pd.DataFrame:
    """Find races with the same (series_id, season, race_date) — indicates
    either a genuine doubleheader (rare) or a stale schedule entry that
    wasn't cleaned up when the API's race list changed.

    Excludes Daytona Duels (pre-season qualifying races, legitimately
    scheduled on the same day).
    """
    rows = conn.execute('''
        SELECT r1.id, r1.series_id, r1.season, r1.race_date, r1.race_name,
               r1.api_race_id,
               (SELECT COUNT(*) FROM race_results WHERE race_id = r1.id) as nr,
               (SELECT COUNT(*) FROM salaries WHERE race_id = r1.id) as ns,
               (SELECT COUNT(*) FROM odds WHERE race_id = r1.id) as no
        FROM races r1
        WHERE EXISTS (
            SELECT 1 FROM races r2
            WHERE r2.series_id = r1.series_id
              AND r2.season = r1.season
              AND r2.race_date = r1.race_date
              AND r2.id != r1.id
        )
        AND r1.race_name NOT LIKE '%Duel%'
        ORDER BY r1.race_date, r1.series_id
    ''').fetchall()
    series_names = {1: "Cup", 2: "O'Reilly", 3: "Truck"}
    return pd.DataFrame([{
        "DB ID": r[0],
        "Series": series_names.get(r[1], str(r[1])),
        "Season": r[2],
        "Date": (r[3] or "")[:10],
        "Race": r[4],
        "API ID": r[5] if r[5] else "—",
        "Results": r[6],
        "Salaries": r[7],
        "Odds": r[8],
    } for r in rows])


def _find_races_without_api_race_id(conn) -> pd.DataFrame:
    """Races without an api_race_id can't be resolved by the NASCAR API."""
    rows = conn.execute('''
        SELECT id, race_name, race_date, series_id, season
        FROM races
        WHERE api_race_id IS NULL
        ORDER BY race_date DESC LIMIT 50
    ''').fetchall()
    series_names = {1: "Cup", 2: "O'Reilly", 3: "Truck"}
    return pd.DataFrame([{
        "ID": r[0],
        "Race": r[1],
        "Date": (r[2] or "")[:10],
        "Series": series_names.get(r[3], str(r[3])),
        "Season": r[4],
    } for r in rows])


def quick_health_check() -> dict:
    """Lightweight DB anomaly scan for boot-time display.

    Runs the same underlying queries as the full DB Health tab but returns
    only anomaly counts, not DataFrames. Designed to be cheap enough to
    run on every app load. Returns:
        {
            "ok": bool,
            "anomalies": [
                {"kind": str, "count": int, "severity": "warn"|"error"},
                ...
            ],
        }
    """
    result = {"ok": True, "anomalies": []}
    if not DB_PATH.exists():
        result["ok"] = False
        result["anomalies"].append({"kind": "DB file missing", "count": 0, "severity": "error"})
        return result
    try:
        conn = sqlite3.connect(str(DB_PATH))

        # Salary fingerprint collisions — high severity (silent data corruption)
        fp_df = _find_salary_fingerprint_collisions(conn)
        if not fp_df.empty:
            result["anomalies"].append({
                "kind": "Salary fingerprint collisions",
                "count": len(fp_df),
                "severity": "error",
            })

        # Duplicate driver entries — high severity (breaks joins)
        dup_df = _find_duplicate_drivers(conn)
        if not dup_df.empty:
            result["anomalies"].append({
                "kind": "Duplicate driver entries",
                "count": len(dup_df),
                "severity": "error",
            })

        # Duplicate race entries — medium severity
        dup_race_df = _find_duplicate_races(conn)
        if not dup_race_df.empty:
            result["anomalies"].append({
                "kind": "Same-date race duplicates",
                "count": len(dup_race_df),
                "severity": "warn",
            })

        # Orphaned rows — medium severity
        orphans = _find_orphaned_rows(conn)
        total_orphans = sum(orphans.values())
        if total_orphans > 0:
            result["anomalies"].append({
                "kind": "Orphaned rows (salaries/odds/results)",
                "count": total_orphans,
                "severity": "warn",
            })

        # Unmapped tracks — the #1 silent year-over-year failure. A new
        # venue absent from TRACK_TYPE_MAP defaults to "intermediate"
        # everywhere (wrong weights, wrong dominator profile, wrong
        # similar-track groups) without any visible error. Surface it loudly
        # so schedule changes each season get a one-line config fix.
        try:
            from src.config import TRACK_TYPE_MAP
            from datetime import datetime as _dt
            _season = _dt.now().year
            tr_rows = conn.execute('''
                SELECT DISTINCT t.name FROM races r
                JOIN tracks t ON t.id = r.track_id
                WHERE r.season >= ?
            ''', (_season - 1,)).fetchall()
            unmapped = sorted(t for (t,) in tr_rows
                              if t and t not in TRACK_TYPE_MAP)
            if unmapped:
                result["anomalies"].append({
                    "kind": ("Tracks missing from TRACK_TYPE_MAP "
                             f"({', '.join(unmapped[:4])}"
                             f"{'…' if len(unmapped) > 4 else ''})"),
                    "count": len(unmapped),
                    "severity": "warn",
                })
        except Exception:
            pass

        conn.close()
    except Exception as e:
        result["ok"] = False
        result["anomalies"].append({
            "kind": f"Health check error: {e}",
            "count": 0, "severity": "error",
        })

    result["ok"] = len(result["anomalies"]) == 0
    return result


def render_health_banner():
    """Render a one-line banner at the top of the app if anomalies exist.

    Silent when everything is clean. Uses an expander so it doesn't take much
    vertical space when collapsed. Caches for 5 minutes to avoid running the
    checks on every rerun.
    """
    # Cache the check results for 5 minutes so reruns don't hammer the DB
    import time
    CACHE_KEY = "_dbhealth_cache"
    CACHE_TTL = 300  # seconds
    now = time.time()
    cached = st.session_state.get(CACHE_KEY)
    if cached and now - cached["t"] < CACHE_TTL:
        check = cached["check"]
    else:
        check = quick_health_check()
        st.session_state[CACHE_KEY] = {"t": now, "check": check}

    if check["ok"]:
        return  # silent when clean — no banner needed

    # Build a concise one-line message
    has_error = any(a["severity"] == "error" for a in check["anomalies"])
    summary = " • ".join(
        f"{a['count']} {a['kind']}" if a["count"] > 0 else a["kind"]
        for a in check["anomalies"]
    )
    if has_error:
        st.error(f"⚠️ DB Health: {summary} — open the **DB Health** tab for details.")
    else:
        st.warning(f"⚠️ DB Health: {summary} — open the **DB Health** tab for details.")


def render(*, series_id: int = None, selected_year: int = None):
    """Render the DB Health tab."""
    from src.components import section_header
    section_header("Database Health", "Data-quality diagnostics and anomaly detection")

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))

    # ── Overview ──
    st.markdown("### Overview")
    counts = _row_counts(conn)
    size_mb = _file_size_mb(DB_PATH)
    cols = st.columns(5)
    cols[0].metric("DB Size", f"{size_mb:.1f} MB")
    cols[1].metric("Races", f"{counts.get('races', 0):,}")
    cols[2].metric("Drivers", f"{counts.get('drivers', 0):,}")
    cols[3].metric("Results", f"{counts.get('race_results', 0):,}")
    cols[4].metric("Odds rows", f"{counts.get('odds', 0):,}")

    cols2 = st.columns(4)
    cols2[0].metric("Salaries", f"{counts.get('salaries', 0):,}")
    cols2[1].metric("Projections", f"{counts.get('projections', 0):,}")
    cols2[2].metric("Tracks", f"{counts.get('tracks', 0):,}")
    cov = _projections_coverage(conn)
    cols2[3].metric(
        "Proj Coverage",
        f"{cov['covered']}/{cov['total']} races",
        f"{cov['pct']:.0f}%"
    )

    st.divider()

    # ── Salary fingerprint collisions (the Kansas bug detector) ──
    st.markdown("### 🔍 Salary fingerprint collisions")
    st.caption(
        "Detects races that share identical DK salary sets (same drivers + prices). "
        "Collisions often mean salaries got written to the wrong year's race_id. "
        "Expected: 0 collisions."
    )
    fp_df = _find_salary_fingerprint_collisions(conn)
    if fp_df.empty:
        st.success("✅ No salary fingerprint collisions detected.")
    else:
        st.error(f"⚠️ Found {len(fp_df)} races with duplicate salary fingerprints:")
        st.dataframe(fp_df, width="stretch", hide_index=True)
        st.caption(
            "**Action:** Identify which race should have these salaries. "
            "Delete from the others using: "
            "`DELETE FROM salaries WHERE race_id = <bad_race_id>`"
        )

    st.divider()

    # ── Missing data ──
    st.markdown("### 📋 Races missing critical data")
    filter_cols = st.columns(3)
    with filter_cols[0]:
        scope = st.selectbox(
            "Scope",
            ["This series + season", "All series, current season",
             "All series, all seasons"],
            key="dbh_scope",
        )
    _sid = series_id if scope == "This series + season" else None
    _season = selected_year if scope != "All series, all seasons" else None
    missing_df = _find_races_missing_data(conn, series_id=_sid, season=_season)
    if missing_df.empty:
        st.success("✅ All races have required data.")
    else:
        # Past races: most-recent first (you care about what JUST happened).
        # Upcoming races: soonest first (next races on the calendar).
        past_df = (missing_df[missing_df["Past"] == True]
                   .drop(columns=["Past"])
                   .sort_values("Date", ascending=False))
        upc_df = (missing_df[missing_df["Past"] == False]
                  .drop(columns=["Past"])
                  .sort_values("Date", ascending=True))
        if not upc_df.empty:
            st.info(f"ℹ️ {len(upc_df)} upcoming races missing data (soonest first):")
            st.dataframe(upc_df, width="stretch", hide_index=True)
        if not past_df.empty:
            st.warning(f"⚠️ {len(past_df)} past races missing data (most recent first):")
            st.dataframe(past_df, width="stretch", hide_index=True)

    st.divider()

    # ── Duplicate drivers ──
    st.markdown("### 👥 Duplicate driver entries")
    st.caption(
        "Drivers whose names normalize to the same key but have separate driver_ids. "
        "These indicate name-matching failures where the same driver got split into "
        "multiple rows. Expected: 0 duplicates."
    )
    dup_df = _find_duplicate_drivers(conn)
    if dup_df.empty:
        st.success("✅ No duplicate driver entries detected.")
    else:
        st.error(f"⚠️ Found {len(dup_df)} normalized keys with multiple driver_ids:")
        st.dataframe(dup_df, width="stretch", hide_index=True)
        st.caption(
            "**Action:** Merge duplicates. Pick one canonical driver_id, update "
            "foreign keys in race_results/salaries/odds to point to it, then delete "
            "the redundant driver row."
        )

    st.divider()

    # ── Orphaned rows ──
    st.markdown("### 🔗 Orphaned rows")
    orphans = _find_orphaned_rows(conn)
    total_orphans = sum(orphans.values())
    if total_orphans == 0:
        st.success("✅ No orphaned rows.")
    else:
        st.warning(
            f"⚠️ Found {total_orphans} rows referencing non-existent races: "
            f"salaries={orphans['salaries']}, odds={orphans['odds']}, "
            f"race_results={orphans['race_results']}"
        )

    st.divider()

    # ── Duplicate races (same date, same series) ──
    st.markdown("### 📅 Duplicate races on the same date")
    st.caption(
        "Races with the same (series, season, date) but different DB IDs. "
        "Usually stale schedule entries from pre-season that weren't cleaned "
        "up when the NASCAR API updated the schedule. Pre-season Daytona "
        "Duels are excluded automatically. "
        "**Safe to delete:** rows with no api_race_id, 0 results, 0 salaries, 0 odds."
    )
    dup_race_df = _find_duplicate_races(conn)
    if dup_race_df.empty:
        st.success("✅ No duplicate race-date entries detected.")
    else:
        st.warning(f"⚠️ Found {len(dup_race_df)} races sharing dates with another race:")
        st.dataframe(dup_race_df, width="stretch", hide_index=True)

    st.divider()

    # ── Races without api_race_id ──
    st.markdown("### 🔖 Races without `api_race_id`")
    st.caption(
        "Races that lack a NASCAR API ID can't be auto-resolved. Historical races "
        "often lack this and that's fine. Recent/upcoming races should all have one."
    )
    no_api_df = _find_races_without_api_race_id(conn)
    if no_api_df.empty:
        st.success("✅ All races have api_race_id set.")
    else:
        st.info(f"ℹ️ {len(no_api_df)} races missing api_race_id (showing 50 most recent):")
        st.dataframe(no_api_df, width="stretch", hide_index=True)

    conn.close()
