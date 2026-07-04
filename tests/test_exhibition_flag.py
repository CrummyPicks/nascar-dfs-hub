"""Exhibition-race flagging (scripts/refresh_data.py flag_exhibition_races).

Runs the REAL function against a temp sqlite DB by monkeypatching
src.config.DB_PATH — the function does `from src.config import DB_PATH`
inside its own body, so the patched attribute is what it reads at call time.
"""
import sqlite3

import src.config
from scripts.refresh_data import flag_exhibition_races

EXHIBITIONS = [
    "America 250 Florida Duel at DAYTONA",   # duel
    "Cook Out Clash at Bowman Gray",         # clash
    "NASCAR All-Star Race",                  # all-star
    "NASCAR Open",                           # nascar open
]
POINTS_RACES = [
    "Coca-Cola 600",
    "Daytona 500",
    "eero 400",
]


def _make_db(path, with_flag_column=True):
    conn = sqlite3.connect(str(path))
    if with_flag_column:
        conn.execute(
            "CREATE TABLE races (id INTEGER PRIMARY KEY, race_name TEXT, "
            "is_exhibition INTEGER DEFAULT 0)")
    else:
        conn.execute("CREATE TABLE races (id INTEGER PRIMARY KEY, race_name TEXT)")
    conn.executemany("INSERT INTO races (race_name) VALUES (?)",
                     [(n,) for n in EXHIBITIONS + POINTS_RACES])
    conn.commit()
    conn.close()


def _flagged_names(path):
    conn = sqlite3.connect(str(path))
    flagged = {r[0] for r in conn.execute(
        "SELECT race_name FROM races WHERE is_exhibition = 1")}
    total = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    conn.close()
    return flagged, total


def test_flags_exactly_the_four_exhibitions(tmp_path, monkeypatch):
    db = tmp_path / "nascar_test.db"
    _make_db(db)
    monkeypatch.setattr(src.config, "DB_PATH", db)

    flag_exhibition_races()

    flagged, total = _flagged_names(db)
    assert flagged == set(EXHIBITIONS)
    assert total == len(EXHIBITIONS) + len(POINTS_RACES)  # nothing added/removed


def test_points_races_left_untouched(tmp_path, monkeypatch):
    db = tmp_path / "nascar_test.db"
    _make_db(db)
    monkeypatch.setattr(src.config, "DB_PATH", db)

    flag_exhibition_races()

    conn = sqlite3.connect(str(db))
    rows = dict(conn.execute("SELECT race_name, is_exhibition FROM races"))
    conn.close()
    for name in POINTS_RACES:
        assert rows[name] == 0, f"{name} wrongly flagged as exhibition"


def test_adds_missing_is_exhibition_column(tmp_path, monkeypatch):
    # Older DBs predate the column; the function ALTERs it in, then flags.
    db = tmp_path / "nascar_legacy.db"
    _make_db(db, with_flag_column=False)
    monkeypatch.setattr(src.config, "DB_PATH", db)

    flag_exhibition_races()

    flagged, _ = _flagged_names(db)
    assert flagged == set(EXHIBITIONS)


def test_reflag_is_idempotent(tmp_path, monkeypatch):
    db = tmp_path / "nascar_test.db"
    _make_db(db)
    monkeypatch.setattr(src.config, "DB_PATH", db)

    flag_exhibition_races()
    flag_exhibition_races()  # second refresh run

    flagged, total = _flagged_names(db)
    assert flagged == set(EXHIBITIONS)
    assert total == len(EXHIBITIONS) + len(POINTS_RACES)


def test_missing_db_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(src.config, "DB_PATH", tmp_path / "does_not_exist.db")
    flag_exhibition_races()  # must not raise or create the file
    assert not (tmp_path / "does_not_exist.db").exists()
