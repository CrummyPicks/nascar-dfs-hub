"""Data & Settings page — salary uploads, odds management, manual practice.

This used to live in a collapsed "Settings & Data Upload" expander at the top
of the single-page app. As a dedicated page, everything saves straight to the
DB (salaries, odds) or session state (manual practice), and the rest of the
app reads from those stores — so nothing here needs to stay rendered for the
data to keep flowing.
"""

import re
import sqlite3

import streamlit as st

from src.config import DB_PATH
from src.data import (
    fetch_nascar_odds, save_odds_to_db, fetch_nascar_prop_odds,
    load_race_odds, query_salaries, parse_dk_csv, parse_fd_csv,
    sync_dk_salaries_to_db, sync_fd_salaries_to_db, _fetch_all_nascar_odds,
)
from src.utils import parse_american_odds


def manual_practice_key(series_id: int, race_id: int) -> str:
    """Session-state key for manually pasted practice data (shared with shell)."""
    return f"manual_practice_{series_id}_{race_id}"


def parse_odds_text(odds_text: str) -> dict:
    """Parse pasted sportsbook win odds into {driver: "+350"} form.

    Supports comma-separated ("Kyle Larson, -115"), trailing ("Chase Elliott
    +1200"), and glued ("CoreyHeim+300") formats, plus EVEN/EV/PK. Header
    lines (race name, dates, times, section labels) are auto-skipped.
    """
    odds_data = {}
    if not odds_text.strip():
        return odds_data

    skip_patterns = re.compile(
        r'^(outright|futures?|top\s*\d|moneyline|head.to.head'
        r'|\d{1,2}/\d{1,2}/\d{2,4}'     # dates like 4/10/26
        r'|\d{1,2}:\d{2}\s*(am|pm)?'    # times like 2:30 PM
        r')$', re.IGNORECASE
    )
    # Odds tail: a signed/unsigned integer OR EVEN/EV/PK (case-insensitive)
    ODDS_RE = r'(?:[+-]?\d+|even|evens|ev|pk|pick(?:\'?em)?)'
    has_odds_re = re.compile(ODDS_RE, re.IGNORECASE)
    csv_odds_re = re.compile(rf'^{ODDS_RE}$', re.IGNORECASE)
    trail_odds_re = re.compile(rf'^(.+?)\s*({ODDS_RE})$', re.IGNORECASE)

    def _store(name: str, raw_odds: str) -> bool:
        val = parse_american_odds(raw_odds)
        if val is None:
            return False
        odds_data[name] = f"+{val}" if val >= 0 else str(val)
        return True

    for line in odds_text.strip().split("\n"):
        line = line.strip()
        if not line or skip_patterns.match(line):
            continue
        if not has_odds_re.search(line):
            continue
        # Format 1: comma-separated "Driver Name, +350" (or ", EVEN")
        if "," in line:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) == 2 and parts[0] and csv_odds_re.match(parts[1]):
                if _store(parts[0], parts[1]):
                    continue
        # Format 2: trailing odds — "Driver Name +350", "DriverName+300",
        # "Connor Zilisch EVEN"
        m = trail_odds_re.match(line)
        if m:
            name = m.group(1).strip().rstrip(",")
            if name:
                _store(name, m.group(2))
    return odds_data


def _clear_salaries(race_id: int, platform: str):
    """Delete saved salaries for a race/platform from the DB."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        db_race = conn.execute(
            "SELECT id FROM races WHERE api_race_id = ?", (race_id,)
        ).fetchone()
        if db_race:
            conn.execute(
                "DELETE FROM salaries WHERE race_id = ? AND platform = ?",
                (db_race[0], platform))
            conn.commit()
    finally:
        conn.close()


def render(race_id: int, series_id: int, race_name: str, is_prerace: bool):
    st.markdown(f"#### Data & Settings — {race_name}")
    st.caption(
        "Load this week's data here. Everything saves to the database, so you "
        "only need to do it once per race — the Projections and Optimizer "
        "pages pick it up automatically."
    )

    # ════════════════════════════════════════════════════════════
    # 1. SALARIES
    # ════════════════════════════════════════════════════════════
    st.markdown("##### 1 · Salaries")
    db_dk_df = query_salaries(race_id=race_id, platform="DraftKings")
    db_fd_df = query_salaries(race_id=race_id, platform="FanDuel")

    s_cols = st.columns(2)
    with s_cols[0]:
        st.markdown("**DraftKings CSV**")
        if not db_dk_df.empty:
            st.success(f"Saved: {len(db_dk_df)} drivers in DB for this race")
        dk_file = st.file_uploader("DK CSV", type=["csv"], label_visibility="collapsed",
                                   key=f"dk_upload_{race_id}")
        if dk_file:
            dk_df = parse_dk_csv(dk_file)
            if not dk_df.empty:
                sync_dk_salaries_to_db(dk_df, race_id, series_id, race_name)
                st.success(f"Saved {len(dk_df)} DK salaries to DB")
            else:
                st.error("Couldn't parse that CSV — is it the DKSalaries export?")
        if not db_dk_df.empty:
            if st.button("Clear DK Salaries", key=f"clear_dk_{race_id}"):
                try:
                    _clear_salaries(race_id, "DraftKings")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear salaries: {e}")

    with s_cols[1]:
        st.markdown("**FanDuel CSV**")
        if not db_fd_df.empty:
            st.success(f"Saved: {len(db_fd_df)} drivers in DB for this race")
        fd_file = st.file_uploader("FD CSV", type=["csv"], label_visibility="collapsed",
                                   key=f"fd_upload_{race_id}")
        if fd_file:
            fd_df = parse_fd_csv(fd_file)
            if not fd_df.empty:
                sync_fd_salaries_to_db(fd_df, race_id, series_id, race_name)
                st.success(f"Saved {len(fd_df)} FD salaries to DB")
            else:
                st.error("Couldn't parse that CSV — is it the FanDuel export?")
        if not db_fd_df.empty:
            if st.button("Clear FD Salaries", key=f"clear_fd_{race_id}"):
                try:
                    _clear_salaries(race_id, "FanDuel")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear salaries: {e}")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # 2. BETTING ODDS
    # ════════════════════════════════════════════════════════════
    st.markdown("##### 2 · Betting Odds")

    saved_odds = load_race_odds(race_id, series_id) if race_id else {}
    if saved_odds:
        st.success(f"Saved: odds for {len(saved_odds)} drivers in DB for this race")
    else:
        st.info("No odds saved for this race yet — auto-fetch or paste below.")

    o_cols = st.columns([1, 2])
    with o_cols[0]:
        st.markdown("**Auto-fetch**")
        if st.button("Auto-Fetch Odds", key="refresh_all_btn", type="primary",
                     help="Fetches win/top5/top10 odds from Action Network (Cup only). "
                          "May not always be available — paste manually if it fails."):
            _fetch_all_nascar_odds.clear()
            fresh_odds = fetch_nascar_odds(series_id)
            if fresh_odds and is_prerace:
                prop_odds = fetch_nascar_prop_odds(series_id)
                save_odds_to_db(fresh_odds, race_id, sportsbook="auto",
                                top3_data=prop_odds.get("top3"),
                                top5_data=prop_odds.get("top5"),
                                top10_data=prop_odds.get("top10"),
                                series_id=series_id)
                st.success(f"Fetched and saved odds for {len(fresh_odds)} drivers")
            elif fresh_odds:
                st.warning("Odds fetched, but this race is already completed — "
                           "not saving upcoming-race odds over a finished race.")
            else:
                st.warning("Auto-fetch unavailable — paste odds manually")
        st.caption("Source: Action Network (Cup series only).")

    with o_cols[1]:
        st.markdown("**Manual paste** (overrides everything else)")
        odds_text = st.text_area(
            "Odds", height=140, label_visibility="collapsed",
            placeholder="Paste win odds:\nCorey Heim+300\nKyle Busch+450\n\n"
                        "Or comma/space separated:\nKyle Larson, -115\nChase Elliott +1200",
            help="Paste win odds from any sportsbook. Header lines "
                 "(race name, date, 'Outright') are auto-skipped.",
            key=f"odds_paste_{series_id}_{race_id}",
        )
        if odds_text.strip():
            odds_data = parse_odds_text(odds_text)
            if odds_data:
                st.success(f"Parsed {len(odds_data)} drivers from pasted odds")
                if st.button(f"Save {len(odds_data)} odds to DB", type="primary",
                             key=f"save_odds_{series_id}_{race_id}"):
                    prop_odds = fetch_nascar_prop_odds(series_id) if is_prerace else {}
                    save_odds_to_db(odds_data, race_id, sportsbook="manual",
                                    top3_data=prop_odds.get("top3"),
                                    top5_data=prop_odds.get("top5"),
                                    top10_data=prop_odds.get("top10"),
                                    series_id=series_id)
                    st.success("Saved — all pages now use these odds")
                    st.rerun()
            else:
                st.warning("Couldn't parse any odds from that text")

    st.divider()

    # ════════════════════════════════════════════════════════════
    # 3. MANUAL PRACTICE (rarely needed)
    # ════════════════════════════════════════════════════════════
    st.markdown("##### 3 · Manual Practice Ranks")
    st.caption(
        "Practice speeds load automatically from NASCAR's lap-averages feed. "
        "Only paste here if that feed is missing/wrong — manual entries "
        "override the automatic signal for this session."
    )
    _prac_key = manual_practice_key(series_id, race_id)
    practice_text = st.text_area(
        "Practice", placeholder="Chase Elliott, 3\nDenny Hamlin, 5",
        height=100, label_visibility="collapsed",
        key=f"practice_paste_{series_id}_{race_id}",
    )
    practice_data = {}
    if practice_text.strip():
        for line in practice_text.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    practice_data[parts[0]] = float(parts[1])
                except (ValueError, IndexError):
                    pass
    if practice_data:
        st.session_state[_prac_key] = practice_data
        st.success(f"Using manual practice ranks for {len(practice_data)} drivers")
    elif st.session_state.get(_prac_key):
        # Text cleared (or page revisited with stale state) — keep showing
        # what's active and offer a reset.
        n = len(st.session_state[_prac_key])
        st.info(f"Manual practice ranks active for {n} drivers (from an earlier paste)")
        if st.button("Clear manual practice", key=f"clear_prac_{series_id}_{race_id}"):
            del st.session_state[_prac_key]
            st.rerun()

    st.divider()
    _render_actual_ownership_section(race_id, series_id)


def _render_actual_ownership_section(race_id: int, series_id: int):
    """Post-race paste of ACTUAL contest ownership — feeds the Accuracy
    page's ownership-projection grading. Without ground truth, the GPP
    leverage model is unvalidated guesswork."""
    from src.data import save_actual_ownership, load_actual_ownership

    st.markdown("##### 4 · Actual Contest Ownership (post-race)")
    st.caption(
        "After a contest settles, paste the field's actual ownership from "
        "your DK/FD results page. The Accuracy page grades our projected "
        "ownership against it — that's what validates the GPP leverage model."
    )
    existing = load_actual_ownership(race_id, series_id)
    if existing:
        st.success("Saved: " + " · ".join(
            f"{p}/{c}: {len(d)} drivers" for (p, c), d in sorted(existing.items())))

    o_cols = st.columns([1, 1, 3])
    with o_cols[0]:
        own_platform = st.selectbox("Site", ["DraftKings", "FanDuel"],
                                    key=f"own_plat_{race_id}")
    with o_cols[1]:
        own_contest = st.selectbox("Contest", ["gpp", "cash"],
                                   format_func=lambda c: c.upper(),
                                   key=f"own_ct_{race_id}")
    with o_cols[2]:
        own_text = st.text_area(
            "Ownership", height=110, label_visibility="collapsed",
            placeholder="Kyle Larson, 42.3\nDenny Hamlin 38%\nRyan Preece, 6.1",
            key=f"own_paste_{own_platform}_{own_contest}_{race_id}",
            help="One driver per line: 'Name, 42.3' or 'Name 42.3%'",
        )
    own_data = {}
    if own_text and own_text.strip():
        import re as _re
        for line in own_text.strip().split("\n"):
            m = _re.match(r'^(.+?)[,\s]+([\d.]+)\s*%?\s*$', line.strip())
            if m:
                try:
                    own_data[m.group(1).strip()] = float(m.group(2))
                except ValueError:
                    pass
    if own_data:
        st.caption(f"Parsed {len(own_data)} drivers")
        if st.button(f"Save {own_platform} {own_contest.upper()} ownership",
                     type="primary", key=f"own_save_{race_id}"):
            n = save_actual_ownership(race_id, series_id, own_platform,
                                      own_contest, own_data)
            if n:
                st.success(f"Saved actual ownership for {n} drivers")
                st.rerun()
            else:
                st.error("Could not save — race not resolvable in DB")
