"""Contest ROI tracking — import DraftKings entry-history CSVs, store, analyze.

DraftKings has no public API and its internal endpoints sit behind bot
protection + login, so the sanctioned pipe is the "Download Entry History"
CSV on draftkings.com/mycontests (despite the page's 30-day view filter,
the export contains FULL account history). This module parses that export
(NASCAR rows only), dedupes on DK's Entry_Key, links entries to races in
nascar.db by date + series, and answers the question the app exists for:
where am I actually making money?

STORAGE: entries live in a SEPARATE, GITIGNORED contests.db — NOT nascar.db.
Two reasons: (1) nascar.db is committed to a public repo and overwritten
daily by the auto-refresh job, which would expose and/or wipe personal
financial data; (2) contest history is the user's private ledger, not
shared app data.
"""

import glob
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd

from src.config import DB_PATH

# Personal ledger — sits next to nascar.db but is gitignored and never
# touched by git operations or the daily refresh job.
CONTESTS_DB = DB_PATH.parent / "contests.db"


# ── classification helpers (tuned against the user's real 2025-26 history) ──

def classify_style(contest_name: str) -> str:
    """Cash / GPP / Qualifier from the contest name.

    Real-history conventions: cash games are 'Double Up' / '50/50' branded;
    qualifiers ('...Championship Qualifier', 'Satellite', ticket runs) pay
    tickets, not cash. Everything else (Engine Block, Happy Hour, Piston,
    Pit Stop, ...) is a GPP.
    """
    n = (contest_name or "").lower()
    if any(k in n for k in ("50/50", "double up", "fifty", "head to head",
                            "h2h", "1v1")):
        return "Cash"
    if any(k in n for k in ("satellite", "qualifier", "ticket")):
        return "Qualifier"
    return "GPP"


def guess_series(contest_name: str) -> str:
    """Cup / O'Reilly / Truck from DK's naming conventions.

    DK prefixes NASCAR contests by series: NAS + (Cup); NOS + (ORLY);
    NTS + (Trucks); NXS + (XFIN) (older Xfinity branding). Series 2 is
    labeled "O'Reilly" to match the rest of the app.
    """
    n = (contest_name or "").lower()
    if "(truck" in n or "truck" in n or n.startswith("nts "):
        return "Truck"
    if ("(orly" in n or "(xfin" in n or "o'reilly" in n or "oreilly" in n
            or "xfinity" in n or n.startswith("nos ") or n.startswith("nxs ")):
        return "O'Reilly"
    if "(cup" in n or n.startswith("nas "):
        return "Cup"
    return ""


SERIES_TO_ID = {"Cup": 1, "O'Reilly": 2, "Truck": 3}


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

def _conn():
    conn = sqlite3.connect(str(CONTESTS_DB))
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
            winnings_ticket REAL DEFAULT 0,
            entry_fee REAL,
            field_entries INTEGER,
            places_paid INTEGER,
            prize_pool REAL,
            series TEXT,
            style TEXT,
            imported_at TEXT
        )
    ''')
    return conn


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
    c_name = col("entry", "contestname", "contest")
    c_sport = col("sport")
    if not c_entry or not c_name:
        raise ValueError(
            "This doesn't look like a DraftKings entry-history CSV "
            "(expected Entry_Key and Entry columns). Use 'Download Entry "
            "History' on draftkings.com/mycontests.")

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
        # NASCAR rows only: DK's Sport column is "NAS" for every NASCAR
        # series; contest-name prefixes (NAS/NOS/NTS/NXS) are the fallback.
        if c_sport:
            if not sport.startswith("NAS"):
                continue
        elif not name.upper().startswith(("NAS", "NOS", "NTS", "NXS")):
            continue
        win_cash = _money(r.get(c_win))
        win_tkt = _money(r.get(c_win_tkt)) if c_win_tkt else 0.0
        rows.append({
            "entry_key": str(r.get(c_entry)).strip(),
            "sport": sport or "NAS",
            "contest_key": str(r.get(c_key, "") or "").strip() if c_key else "",
            "contest_name": name,
            "contest_date": str(r.get(c_date, "") or "").strip() if c_date else "",
            "place": _int(r.get(c_place)) if c_place else None,
            "points": _money(r.get(c_points)) if c_points else None,
            "winnings": round(win_cash + win_tkt, 2),
            "winnings_ticket": round(win_tkt, 2),
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
    if parsed is None or parsed.empty:
        return 0, 0
    conn = _conn()
    added = skipped = 0
    now = datetime.now().isoformat(timespec="seconds")
    for _, r in parsed.iterrows():
        try:
            cur = conn.execute('''
                INSERT OR IGNORE INTO contest_entries
                (entry_key, platform, sport, contest_key, contest_name,
                 contest_date, place, points, winnings, winnings_ticket,
                 entry_fee, field_entries, places_paid, prize_pool, series,
                 style, imported_at)
                VALUES (?, 'DraftKings', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (r["entry_key"], r["sport"], r["contest_key"],
                  r["contest_name"], r["contest_date"], r["place"],
                  r["points"], r["winnings"], r["winnings_ticket"],
                  r["entry_fee"], r["field_entries"], r["places_paid"],
                  r["prize_pool"], r["series"], r["style"], now))
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
    conn = _conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM contest_entries ORDER BY contest_date DESC, id DESC",
            conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def attach_races(df: pd.DataFrame) -> pd.DataFrame:
    """Add Race / Track columns by matching entry date + series to nascar.db.

    Computed at load time (not import) so races synced AFTER an import still
    link up. Match: same calendar date + same series (points races only).
    """
    if df is None or df.empty or not DB_PATH.exists():
        return df
    out = df.copy()
    out["_date"] = out["contest_date"].astype(str).str.slice(0, 10)
    try:
        conn = sqlite3.connect(str(DB_PATH))
        races = pd.read_sql_query('''
            SELECT r.series_id, substr(r.race_date, 1, 10) as d,
                   r.race_name, t.name as track
            FROM races r JOIN tracks t ON t.id = r.track_id
            WHERE COALESCE(r.is_exhibition, 0) = 0
        ''', conn)
        conn.close()
    except Exception:
        out.drop(columns=["_date"], inplace=True)
        return out
    lut = {(int(r["series_id"]), r["d"]): (r["race_name"], r["track"])
           for _, r in races.iterrows()}
    # Fallback for entries with no series tag (private leagues): when exactly
    # ONE points race ran that calendar date across all series, it's that one.
    by_date = {}
    for _, r in races.iterrows():
        by_date.setdefault(r["d"], []).append(
            (int(r["series_id"]), r["race_name"], r["track"]))
    _id_to_series = {v: k for k, v in SERIES_TO_ID.items()}

    def _match(row):
        sid = SERIES_TO_ID.get(row.get("series"))
        if sid:
            hit = lut.get((sid, row["_date"]))
            if hit:
                return hit + (row.get("series"),)
            return None
        candidates = by_date.get(row["_date"], [])
        if len(candidates) == 1:
            c = candidates[0]
            return (c[1], c[2], _id_to_series.get(c[0], ""))
        return None

    m = out.apply(_match, axis=1)
    out["Race"] = m.map(lambda v: v[0] if v else None)
    out["Track"] = m.map(lambda v: v[1] if v else None)
    # Backfill the series guess from the linked race (private leagues).
    _linked_series = m.map(lambda v: v[2] if v else None)
    out["series"] = out["series"].mask(
        out["series"].fillna("").eq("") & _linked_series.notna(),
        _linked_series)
    out.drop(columns=["_date"], inplace=True)
    return out


def race_day_index() -> pd.DataFrame:
    """Every points race with ids, keyed by (date, series label).

    Columns: date (YYYY-MM-DD), series, db_id, api_id, season, race_name,
    track. Used by Model-vs-Me to run the profit-sim engine against the
    races the user actually entered contests for.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query('''
            SELECT substr(r.race_date, 1, 10) as date, r.series_id,
                   r.id as db_id, r.api_race_id as api_id, r.season,
                   r.race_name, t.name as track
            FROM races r JOIN tracks t ON t.id = r.track_id
            WHERE COALESCE(r.is_exhibition, 0) = 0
              AND EXISTS (SELECT 1 FROM race_results rr WHERE rr.race_id = r.id)
        ''', conn)
        conn.close()
    except Exception:
        return pd.DataFrame()
    _id_to_series = {v: k for k, v in SERIES_TO_ID.items()}
    df["series"] = df["series_id"].map(_id_to_series)
    return df.dropna(subset=["series"])


# ── universal ingest: drop ANY DraftKings export, we route it ──────────

def detect_csv_type(df: pd.DataFrame) -> str:
    """'entry_history' | 'standings' | 'unknown' from the header shape."""
    cols = {_norm_col(c) for c in df.columns}
    if "entrykey" in cols:
        return "entry_history"
    if "player" in cols and any("drafted" in c for c in cols) and "rank" in cols:
        return "standings"
    return "unknown"


def parse_dk_standings(df: pd.DataFrame) -> dict:
    """Parse a DK contest-standings ('Export Lineups') CSV.

    Right-hand block: Player / Roster Position / %Drafted / FPTS — the
    field's ACTUAL ownership. Left block: Rank / Points — every entrant's
    score, which yields the exact cash line at the last paid place.
    Returns {ownership, fpts, positions, scores}.
    """
    cols = {_norm_col(c): c for c in df.columns}
    p_col, d_col = cols.get("player"), None
    for k, c in cols.items():
        if "drafted" in k:
            d_col = c
    pos_col = cols.get("rosterposition") or cols.get("position")
    f_col = cols.get("fpts")
    own, fpts, positions = {}, {}, set()
    if p_col and d_col:
        for _, r in df[[c for c in [p_col, d_col, pos_col, f_col] if c]].dropna(
                subset=[p_col]).iterrows():
            nm = str(r[p_col]).strip()
            if not nm or nm.lower() == "nan":
                continue
            try:
                own[nm] = float(str(r[d_col]).replace("%", "").strip())
            except (ValueError, TypeError):
                continue
            if pos_col and pd.notna(r.get(pos_col)):
                positions.add(str(r[pos_col]).strip().upper())
            if f_col and pd.notna(r.get(f_col)):
                try:
                    fpts[nm] = float(r[f_col])
                except (ValueError, TypeError):
                    pass
    scores = []
    r_col, pts_col = cols.get("rank"), cols.get("points")
    if r_col and pts_col:
        s = pd.to_numeric(df[pts_col], errors="coerce").dropna()
        scores = sorted(s.tolist(), reverse=True)
    return {"ownership": own, "fpts": fpts, "positions": positions,
            "scores": scores}


def ingest_file(source, filename: str = "") -> dict:
    """Route ANY dropped DraftKings CSV to the right parser + storage.

    Handles mixed-sport content automatically: entry-history rows are
    filtered to NASCAR; standings files from other sports (roster positions
    other than D) are skipped with a clear reason. Returns
    {type, status: ok|skipped|error, msg}.
    """
    try:
        df = pd.read_csv(source)
    except Exception as e:
        return {"type": "unknown", "status": "error",
                "msg": f"could not read CSV ({e})"}
    kind = detect_csv_type(df)

    if kind == "entry_history":
        try:
            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            parsed = parse_dk_entry_history(buf)
        except ValueError as e:
            return {"type": kind, "status": "error", "msg": str(e)}
        added, skipped = import_entries(parsed)
        return {"type": kind, "status": "ok",
                "msg": (f"entry history — {added} new NASCAR entries imported "
                        f"({skipped} already stored; other sports ignored)")}

    if kind == "standings":
        parsed = parse_dk_standings(df)
        own = parsed["ownership"]
        if not own:
            return {"type": kind, "status": "error",
                    "msg": "standings file, but no Player/%Drafted rows parsed"}
        # NASCAR standings roster positions are all 'D' — anything else
        # (QB/RB/PG/...) is another sport; skip it quietly.
        pos = parsed["positions"]
        if pos and not pos <= {"D", "CPT"}:
            return {"type": kind, "status": "skipped",
                    "msg": f"not a NASCAR contest (positions: {', '.join(sorted(pos)[:4])})"}

        # Resolve which race this contest was: the filename carries DK's
        # contest key, which joins to the imported entry history.
        m = re.search(r"(\d{6,})", filename or "")
        ck = m.group(1) if m else None
        ledger = load_entries()
        rows = (ledger[ledger["contest_key"] == ck]
                if ck and not ledger.empty else pd.DataFrame())
        if rows.empty:
            return {"type": kind, "status": "skipped",
                    "msg": ("NASCAR standings parsed, but the contest key "
                            f"({ck or 'none in filename'}) isn't in your "
                            "imported entry history — import that first so "
                            "the file can be tied to a race")}
        date = str(rows.iloc[0]["contest_date"])[:10]
        series = rows.iloc[0]["series"] or ""
        style = (rows.iloc[0]["style"] or "GPP").lower()
        style = "cash" if style == "cash" else "gpp"

        idx = race_day_index()
        hit = idx[(idx["date"] == date) & (idx["series"] == series)] if series \
            else idx[idx["date"] == date]
        if len(hit) != 1:
            return {"type": kind, "status": "skipped",
                    "msg": f"couldn't uniquely match a race for {date} ({series or '?'})"}
        race = hit.iloc[0]

        from src.data import save_actual_ownership, save_contest_lines
        n = save_actual_ownership(int(race["api_id"]), int(race["series_id"]),
                                  "DraftKings", style, own)
        msgs = [f"{race['track']} {date} ({series or race['series']})",
                f"actual {style.upper()} ownership saved for {n} drivers"]

        # Exact contest line: the score at the last paid place. Feeds the
        # profit sim's real-lines override (beats the simulated-field proxy).
        pp = rows["places_paid"].dropna()
        scores = parsed["scores"]
        if not pp.empty and scores:
            pp = int(pp.iloc[0])
            if 0 < pp <= len(scores):
                line = float(scores[pp - 1])
                save_contest_lines(
                    int(race["api_id"]), int(race["series_id"]), "DraftKings",
                    cash_line=line if style == "cash" else None,
                    gpp_mincash=line if style == "gpp" else None)
                msgs.append(f"real {'cash' if style == 'cash' else 'GPP min-cash'} "
                            f"line saved: {line:.1f}")
        return {"type": kind, "status": "ok", "msg": " · ".join(msgs)}

    return {"type": kind, "status": "skipped",
            "msg": "not a recognized DraftKings export (need the entry-history "
                   "or contest-standings CSV)"}


def find_dk_export_csvs() -> list:
    """All candidate DK exports in Downloads/Desktop (entry history AND
    contest standings), newest first."""
    search_dirs = [os.path.expanduser("~/Downloads"),
                   os.path.expanduser("~/Desktop")]
    out, seen = [], set()
    for d in search_dirs:
        for pattern in ["*entry*history*.csv", "*Entry*History*.csv",
                        "*contest-entry*.csv", "*EntryHistory*.csv",
                        "*contest-standings*.csv", "*Contest*Standings*.csv"]:
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


def find_entry_history_csvs() -> list:
    """Candidate DK entry-history CSVs in Downloads/Desktop, newest first."""
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
