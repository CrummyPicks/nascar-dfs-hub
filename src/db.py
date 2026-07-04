"""Shared sqlite connection helper.

New code should use this instead of hand-rolling
``sqlite3.connect(str(DB_PATH))`` / ``conn.close()`` pairs — the context
manager guarantees the connection is closed even when a query raises.
Existing call sites migrate opportunistically (Phase 3 of the refactor).

Usage::

    from src.db import db

    with db() as conn:
        rows = conn.execute("SELECT ...").fetchall()

    with db() as conn:              # writes commit explicitly
        conn.execute("UPDATE ...")
        conn.commit()
"""

from contextlib import contextmanager
import sqlite3

from src.config import DB_PATH


@contextmanager
def db(path=None):
    """Read-mostly sqlite connection that always closes; commit explicitly."""
    conn = sqlite3.connect(str(path or DB_PATH))
    try:
        yield conn
    finally:
        conn.close()
