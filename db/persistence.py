"""
SQLite persistence — stores tick prices and VaR computations across sessions.
"""
import sqlite3, os, json, time
from pathlib import Path

DB_PATH = Path(__file__).parent / "risk_dashboard.db"


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS price_ticks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL    NOT NULL,
                ticker    TEXT    NOT NULL,
                price     REAL    NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS var_snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL    NOT NULL,
                total_nav  REAL,
                var_95     REAL,
                cvar_95    REAL,
                confidence REAL,
                beta       REAL,
                sharpe     REAL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_ticks_ts ON price_ticks(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_var_ts   ON var_snapshots(ts)")


def save_prices(prices: dict):
    ts = time.time()
    with _conn() as c:
        c.executemany(
            "INSERT INTO price_ticks (ts, ticker, price) VALUES (?, ?, ?)",
            [(ts, ticker, price) for ticker, price in prices.items()]
        )


def save_var_snapshot(risk: dict, confidence: float):
    with _conn() as c:
        c.execute("""
            INSERT INTO var_snapshots (ts, total_nav, var_95, cvar_95, confidence, beta, sharpe)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            risk.get("total_nav"),
            risk.get("var_95"),
            risk.get("cvar_95"),
            confidence,
            risk.get("beta"),
            risk.get("sharpe"),
        ))


def load_var_history(limit: int = 500):
    """Returns list of (ts, var_95, total_nav) most recent rows."""
    import pandas as pd
    with _conn() as c:
        df = pd.read_sql_query(
            f"SELECT ts, var_95, cvar_95, total_nav, beta, sharpe "
            f"FROM var_snapshots ORDER BY ts DESC LIMIT {limit}",
            c
        )
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df.sort_values("ts")


def load_price_history(ticker: str, limit: int = 500):
    import pandas as pd
    with _conn() as c:
        df = pd.read_sql_query(
            "SELECT ts, price FROM price_ticks WHERE ticker=? "
            "ORDER BY ts DESC LIMIT ?",
            c, params=(ticker, limit)
        )
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df.sort_values("ts")


# Initialise on import
init_db()
