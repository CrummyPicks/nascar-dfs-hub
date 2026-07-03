"""Contest ROI tracking — import DraftKings entry-history CSVs, store, analyze.

DraftKings has no public API and its internal endpoints sit behind bot
protection + login, so the sanctioned pipe is the "Download Entry History"
CSV on draftkings.com/mycontests. This module parses that export (NASCAR
rows only), dedupes on DK's Entry_Key, and answers the question the app
exists for: where am I actually making money?
"""

import glob
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd

from src.config import DB_PATH


# ── classification helpers ─────────────────────────────────────────────

def classify_style(contest_name: str) -> str:
    """Cash / GPP / Qualifier from the contest name (DK convention)."""
    n = (contest_name or "").lower()
    if any(k in n for k in ("double up", "50/50", "fifty", "head to head",
                            "h2h", "1v1")):
        return "Cash"
    if any(k in n for k in ("satellite", "qualifier", "ticket")):
        return "Qualifier"
    return "GPP"


def guess_series(contest_name: str) -> str:
    """Cup / Xfinity / Truck from the contest name; '' when unknown."""
    n = (contest_name or "").lower()
    if "truck" in n:
        return "Truck"
    if "xfinity" in n or "o'reilly" in n or "oreilly" in n:
        return "Xfinity"
    if "(cup)" in n or " cup" in n:
        return "Cup"
    return ""


def _money(v) -> float:
    """Parse '$1.50', '(1.00)', '1,234.00' → float; blank/None → 0."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    s = str(v).strip().replace("$", "").replace(",", "")
    if not s or s.lower() in ("nan", "none", "-"):
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        val = float(s)
    except ValueError:
        return 0.0
    return -val if neg else val


def _int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(float(str(v).replace(",", "")))
    except (ValueError, TypeError):
        return None


# ── storage ────────────────────────────────────────────────────────────

def _ensure_table(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS contest_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_key TEXT UNIQUE,
            platform TEXT DEFAULT 'DraftKings',
            sport TEXT,
            contest_key TEXT,
            contest_name TEXT,
            contest_date TEXT,
            place INTEGER,
            points REAL,
            winnings REAL,
            entry_fee REAL,
            field_entries INTEGER,
            places_paid INTEGER,
            prize_pool REAL,
            series TEXT,
            style TEXT,
            imported_at TEXT
        )
    ''')


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def parse_dk_entry_history(source) -> pd.DataFrame:
    """Parse a DK 'Download Entry History' CSV (path or file-like).

    Returns a normalized DataFrame of NASCAR rows only — one row per entry
    with entry_key/contest fields/fee/winnings/derived series+style. Raises
    ValueError when the file doesn't look like a DK entry-history export.
    """
    df = pd.read_csv(source)
    cols = {_norm_col(c): c for c in df.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_entry = col("entrykey", "entryid")
    c_name = col("contestname", "entry", "contest")
    c_sport = col("sport")
    if not c_entry or not c_name:
        raise ValueError(
            "This doesn't look like a DraftKings entry-history CSV "
            "(expected Entry_Key and Entry/Contest_Name columns). Use "
            "'Download Entry History' on draftkings.com/mycontests.")

    c_date = col("contestdateest", "contestdate", "date")
    c_place = col("place", "rank")
    c_points = col("points", "fpts")
    c_fee = col("entryfee", "fee")
    c_win = col("winningsnonticket", "winnings")
    c_win_tkt = col("winningsticket")
    c_key = col("contestkey", "contestid")
    c_field = col("contestentries", "entries")
    c_paid = col("placespaid")
    c_pool = col("prizepool", "totalprizes")

    rows = []
    for _, r in df.iterrows():
        sport = str(r.get(c_sport, "") or "").strip().upper() if c_sport else ""
        name = str(r.get(c_name, "") or "").strip()
        # NASCAR rows only: DK marks the sport "NAS"; fall back to the "NAS "
        # contest-name prefix when the Sport column is absent.
        if c_sport:
            if not sport.startswith("NAS"):
                continue
        elif not name.upper().startswith("NAS"):
            continue
        winnings = _money(r.get(c_win)) + (_money(r.get(c_win_tkt)) if c_win_tkt else 0.0)
        rows.append({
            "entry_key": str(r.get(c_entry)).strip(),
            "sport": sport or "NAS",
            "contest_key": str(r.get(c_key, "") or "").strip() if c_key else "",
            "contest_name": name,
            "contest_date": str(r.get(c_date, "") or "").strip() if c_date else "",
            "place": _int(r.get(c_place)) if c_place else None,
            "points": _money(r.get(c_points)) if c_points else None,
            "winnings": round(winnings, 2),
            "entry_fee": _money(r.get(c_fee)) if c_fee else 0.0,
            "field_entries": _int(r.get(c_field)) if c_field else None,
            "places_paid": _int(r.get(c_paid)) if c_paid else None,
            "prize_pool": _money(r.get(c_pool)) if c_pool else None,
            "series": guess_series(name),
            "style": classify_style(name),
        })
    return pd.DataFrame(rows)


def import_entries(parsed: pd.DataFrame) -> tuple:
    """Insert parsed entries; dedupe on entry_key. Returns (added, skipped)."""
    if parsed is None or parsed.empty or not DB_PATH.exists():
        return 0, 0
    conn = sqlite3.connect(str(DB_PATH))
    _ensure_table(conn)
    added = skipped = 0
    now = datetime.now().isoformat(timespec="seconds")
    for _, r in parsed.iterrows():
        try:
            cur = conn.execute('''
                INSERT OR IGNORE INTO contest_entries
                (entry_key, platform, sport, contest_key, contest_name,
                 contest_date, place, points, winnings, entry_fee,
                 field_entries, places_paid, prize_pool, series, style,
                 imported_at)
                VALUES (?, 'DraftKings', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (r["entry_key"], r["sport"], r["contest_key"],
                  r["contest_name"], r["contest_date"], r["place"],
                  r["points"], r["winnings"], r["entry_fee"],
                  r["field_entries"], r["places_paid"], r["prize_pool"],
                  r["series"], r["style"], now))
            if cur.rowcount:
                added += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    conn.commit()
    conn.close()
    return added, skipped


def load_entries() -> pd.DataFrame:
    """All stored contest entries, newest first."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    _ensure_table(conn)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM contest_entries ORDER BY contest_date DESC, id DESC",
            conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def find_entry_history_csvs() -> list:
    """Candidate DK entry-history CSVs in Downloads/Desktop, newest first.

    DK names the export like 'draftkings-contest-entry-history.csv'; users
    also end up with 'entry-history (1).csv' style duplicates — match loosely.
    """
    search_dirs = [os.path.expanduser("~/Downloads"),
                   os.path.expanduser("~/Desktop")]
    out, seen = [], set()
    for d in search_dirs:
        for pattern in ["*entry*history*.csv", "*Entry*History*.csv",
                        "*contest-entry*.csv", "*EntryHistory*.csv"]:
            for f in glob.glob(os.path.join(d, pattern)):
                real = os.path.realpath(f)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    out.append((real, os.path.getmtime(real)))
                except OSError:
                    continue
    out.sort(key=lambda x: -x[1])
    return [f for f, _ in out]
