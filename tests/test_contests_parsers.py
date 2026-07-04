"""Contest-ledger CSV parsing (src/contests.py) against DK's real layouts.

Pure-parse functions only — ingest_file/import_entries touch the real
contests.db and are deliberately NOT exercised here.
"""
import io

import pandas as pd
import pytest

from src.contests import (
    _money,
    classify_style,
    detect_csv_type,
    guess_series,
    parse_dk_entry_history,
    parse_dk_standings,
)

# ── synthetic fixtures: DK's real column layouts ─────────────────────────

ENTRY_HISTORY_CSV = """\
Sport,Game_Type,Entry_Key,Entry,Contest_Key,Contest_Date_EST,Place,Points,Winnings_Non_Ticket,Winnings_Ticket,Contest_Entries,Entry_Fee,Prize_Pool,Places_Paid
NAS,Classic,4001,"NAS $4 Happy Hour [20 Entry Max] (Cup)",9001,2026-06-01 19:00:00,12,88.5,$1.50,$0.00,"1,500",$4.00,"$5,000.00",350
NAS,Classic,4002,"NOS $2 Double Up (ORLY)",9002,2026-06-06 19:30:00,40,75,$0.00,$5.00,100,$2.00,$180.00,45
NFL,Classic,4003,"NFL $20 Sunday Million (Main)",9003,2026-06-07 13:00:00,1,150,$50.00,$0.00,10,$20.00,$180.00,3
NBA,Showdown,4004,"NBA $5 Shot (Fri)",9004,2026-06-05 19:00:00,2,300,$8.00,$0.00,20,$5.00,$90.00,6
NAS,Classic,4005,"NTS Championship Qualifier (Trucks)",9005,2026-06-05 20:00:00,1,110,"$1,234.00",$25.00,500,$5.00,"$3,000.00",10
NAS,Classic,4006,"NAS Satellite (Cup)",9006,2026-06-01 19:00:00,3,90,$0.00,$10.00,20,$1.00,$200.00,5
"""

# DK "Export Lineups" standings: left block = every entrant's rank/score,
# EMPTY spacer column, right block = per-driver ownership. The right block
# has fewer rows than the left, so its trailing cells are blank.
STANDINGS_CSV = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,111,sharkfan (1/20),0,120.5,D Kyle Larson D Chase Elliott,,Kyle Larson,D,45.2%,55.5
2,112,userB,0,98.25,D Chase Elliott D Ross Chastain,,Chase Elliott,D,12.5%,40
3,113,userC,0,95,D Kyle Larson D Ross Chastain,,Daniel Suárez,D,8%,33.25
4,114,userD,0,60.75,D Ross Chastain D Kyle Larson,,Ross Chastain,D,3.1%,21
5,115,userE,0,44,D Chase Elliott D Kyle Larson,,,,,
6,116,userF,0,12.5,D Kyle Larson D Chase Elliott,,,,,
"""

NFL_STANDINGS_CSV = """\
Rank,EntryId,EntryName,TimeRemaining,Points,Lineup,,Player,Roster Position,%Drafted,FPTS
1,211,userA,0,180.4,QB Someone RB Other,,Josh Allen,QB,30.5%,28.4
2,212,userB,0,150.2,QB Someone RB Other,,Bijan Robinson,RB,22.1%,19.7
"""


def _df(text):
    return pd.read_csv(io.StringIO(text))


# ── detect_csv_type ──────────────────────────────────────────────────────

def test_detect_entry_history():
    assert detect_csv_type(_df(ENTRY_HISTORY_CSV)) == "entry_history"


def test_detect_standings():
    assert detect_csv_type(_df(STANDINGS_CSV)) == "standings"


def test_detect_unknown():
    junk = pd.DataFrame({"Driver": ["Kyle Larson"], "Salary": [10500]})
    assert detect_csv_type(junk) == "unknown"


# ── parse_dk_entry_history ───────────────────────────────────────────────

def test_entry_history_filters_to_nascar_rows_only():
    parsed = parse_dk_entry_history(io.StringIO(ENTRY_HISTORY_CSV))
    assert len(parsed) == 4  # NFL + NBA rows dropped
    assert set(parsed["entry_key"]) == {"4001", "4002", "4005", "4006"}
    assert all(s.startswith("NAS") for s in parsed["sport"])


def test_entry_history_parses_money_and_counts():
    parsed = parse_dk_entry_history(io.StringIO(ENTRY_HISTORY_CSV))
    row = parsed[parsed["entry_key"] == "4001"].iloc[0]
    assert row["entry_fee"] == 4.0            # "$4.00"
    assert row["winnings"] == 1.5             # "$1.50" cash, no ticket
    assert row["winnings_ticket"] == 0.0
    assert row["place"] == 12
    assert row["points"] == 88.5
    assert row["field_entries"] == 1500       # "1,500"
    assert row["prize_pool"] == 5000.0        # "$5,000.00"
    assert row["places_paid"] == 350
    assert row["contest_key"] == "9001"
    assert row["contest_date"] == "2026-06-01 19:00:00"


def test_entry_history_sums_ticket_and_cash_winnings():
    parsed = parse_dk_entry_history(io.StringIO(ENTRY_HISTORY_CSV))
    qual = parsed[parsed["entry_key"] == "4005"].iloc[0]
    assert qual["winnings"] == 1259.0         # $1,234.00 cash + $25.00 ticket
    assert qual["winnings_ticket"] == 25.0
    ticket_only = parsed[parsed["entry_key"] == "4002"].iloc[0]
    assert ticket_only["winnings"] == 5.0
    assert ticket_only["winnings_ticket"] == 5.0


def test_entry_history_derives_series_and_style():
    parsed = parse_dk_entry_history(io.StringIO(ENTRY_HISTORY_CSV)).set_index("entry_key")
    assert parsed.loc["4001", "series"] == "Cup"
    assert parsed.loc["4001", "style"] == "GPP"        # Happy Hour = GPP
    assert parsed.loc["4002", "series"] == "O'Reilly"
    assert parsed.loc["4002", "style"] == "Cash"       # Double Up
    assert parsed.loc["4005", "series"] == "Truck"
    assert parsed.loc["4005", "style"] == "Qualifier"
    assert parsed.loc["4006", "style"] == "Qualifier"  # Satellite


def test_entry_history_without_sport_column_falls_back_to_name_prefix():
    csv = (
        "Entry_Key,Entry,Contest_Date_EST,Place,Entry_Fee,Winnings_Non_Ticket\n"
        '5001,"NAS $1 Pit Stop (Cup)",2026-06-01 19:00:00,5,$1.00,$0.00\n'
        '5002,"NFL $9 Flea Flicker",2026-06-07 13:00:00,9,$9.00,$0.00\n'
    )
    parsed = parse_dk_entry_history(io.StringIO(csv))
    assert list(parsed["entry_key"]) == ["5001"]


def test_entry_history_rejects_non_entry_history_csv():
    with pytest.raises(ValueError):
        parse_dk_entry_history(io.StringIO(STANDINGS_CSV))


# ── parse_dk_standings ───────────────────────────────────────────────────

def test_standings_ownership_percent_stripped():
    parsed = parse_dk_standings(_df(STANDINGS_CSV))
    assert parsed["ownership"] == {
        "Kyle Larson": 45.2,
        "Chase Elliott": 12.5,
        "Daniel Suárez": 8.0,
        "Ross Chastain": 3.1,
    }


def test_standings_positions_and_fpts():
    parsed = parse_dk_standings(_df(STANDINGS_CSV))
    assert parsed["positions"] == {"D"}       # NASCAR: every slot is D
    assert parsed["fpts"]["Kyle Larson"] == 55.5
    assert parsed["fpts"]["Ross Chastain"] == 21.0


def test_standings_scores_sorted_descending_from_points_column():
    parsed = parse_dk_standings(_df(STANDINGS_CSV))
    assert parsed["scores"] == [120.5, 98.25, 95.0, 60.75, 44.0, 12.5]


def test_nfl_standings_positions_reveal_other_sport():
    parsed = parse_dk_standings(_df(NFL_STANDINGS_CSV))
    assert parsed["positions"] == {"QB", "RB"}


# ── classify_style / guess_series / _money ───────────────────────────────

@pytest.mark.parametrize("name,style", [
    ("NAS $10 50/50 (Cup)", "Cash"),
    ("NOS $2 Double Up (ORLY)", "Cash"),
    ("NAS Championship Qualifier (Cup)", "Qualifier"),
    ("NTS Satellite (Trucks)", "Qualifier"),
    ("NAS $4 Happy Hour [20 Entry Max] (Cup)", "GPP"),
    ("NOS $1 Engine Block (ORLY)", "GPP"),
])
def test_classify_style(name, style):
    assert classify_style(name) == style


@pytest.mark.parametrize("name,series", [
    ("NAS $4 Happy Hour (Cup)", "Cup"),
    ("NAS Piston Special", "Cup"),            # "NAS " prefix fallback
    ("NOS $2 Piston (ORLY)", "O'Reilly"),
    ("NOS Big One", "O'Reilly"),              # "NOS " prefix fallback
    ("NXS $5 Throwback (XFIN)", "O'Reilly"),  # old Xfinity branding
    ("NXS Dash", "O'Reilly"),
    ("NTS $3 Hauler (Trucks)", "Truck"),
    ("NTS Tailgate", "Truck"),
    ("Some Random Contest", ""),
])
def test_guess_series(name, series):
    assert guess_series(name) == series


@pytest.mark.parametrize("raw,expected", [
    ("$1.50", 1.5),
    ("$1,234.00", 1234.0),
    ("(1.00)", -1.0),           # accounting negative
    ("1234.5", 1234.5),
    (None, 0.0),
    ("", 0.0),
    ("-", 0.0),
    ("nan", 0.0),
    ("garbage", 0.0),
])
def test_money_parsing(raw, expected):
    assert _money(raw) == expected
