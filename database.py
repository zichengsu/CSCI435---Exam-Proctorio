"""
database.py -- SQLite + Pandas data layer
=========================================

Stores proctoring sessions and every violation event in a local SQLite file
(`proctor.db`). The Streamlit app writes here; the "Session History" tab reads
it back as pandas DataFrames.

Two tables:
    sessions(id, source_type, source_name, started_at, ended_at,
             total_frames, avg_fps, total_violations)
    violations(id, session_id, frame_number, timestamp, violation_type,
               overall_status, risk_score, risk_level, details)
"""

import sqlite3
from datetime import datetime
import pandas as pd

DB_PATH = "proctor.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create tables if they do not exist. Safe to call on every app start."""
    con = _connect()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type      TEXT,
            source_name      TEXT,
            started_at       TEXT,
            ended_at         TEXT,
            total_frames     INTEGER DEFAULT 0,
            avg_fps          REAL    DEFAULT 0,
            total_violations INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER,
            frame_number   INTEGER,
            timestamp      TEXT,
            violation_type TEXT,
            overall_status TEXT,
            risk_score     INTEGER,
            risk_level     TEXT,
            details        TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
        """
    )
    con.commit()
    con.close()


def start_session(source_type, source_name):
    """Create a new session row and return its id."""
    con = _connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO sessions (source_type, source_name, started_at) VALUES (?, ?, ?)",
        (source_type, source_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    con.commit()
    session_id = cur.lastrowid
    con.close()
    return session_id


def log_violation(session_id, frame_number, violation_type, overall_status,
                  risk_score, risk_level, details=""):
    """Insert one violation event."""
    con = _connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO violations
            (session_id, frame_number, timestamp, violation_type,
             overall_status, risk_score, risk_level, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id, frame_number,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            violation_type, overall_status, risk_score, risk_level, details,
        ),
    )
    con.commit()
    con.close()


def end_session(session_id, total_frames, avg_fps, total_violations):
    """Fill in the closing stats for a session."""
    con = _connect()
    cur = con.cursor()
    cur.execute(
        """
        UPDATE sessions
        SET ended_at = ?, total_frames = ?, avg_fps = ?, total_violations = ?
        WHERE id = ?
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_frames, round(avg_fps, 2), total_violations, session_id,
        ),
    )
    con.commit()
    con.close()


def get_sessions_df():
    con = _connect()
    df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", con)
    con.close()
    return df


def get_violations_df(session_id=None):
    con = _connect()
    if session_id is None:
        df = pd.read_sql_query(
            "SELECT * FROM violations ORDER BY id DESC", con
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM violations WHERE session_id = ? ORDER BY id DESC",
            con, params=(session_id,),
        )
    con.close()
    return df


def clear_all():
    """Wipe all sessions and violations (handy for a clean demo)."""
    con = _connect()
    cur = con.cursor()
    cur.execute("DELETE FROM violations")
    cur.execute("DELETE FROM sessions")
    con.commit()
    con.close()
