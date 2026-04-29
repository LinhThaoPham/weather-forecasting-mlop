"""Database layer — Cloud SQL (PostgreSQL) on GCP, SQLite for local dev.

Environment:
    DB_TYPE:                "postgres" or "sqlite" (default: sqlite)
    DB_USER, DB_PASS, DB_NAME: PostgreSQL credentials
    INSTANCE_CONNECTION_NAME: Cloud SQL instance (project:region:instance)
    DB_PATH:                SQLite file path (fallback)
"""
import os
import sqlite3
from contextlib import contextmanager

from src.config.settings import DB_PATH

DB_TYPE = os.getenv("DB_TYPE", "sqlite")

# ── Schema (compatible with both SQLite and PostgreSQL) ──

SCHEMA_SQL_SQLITE = """
CREATE TABLE IF NOT EXISTS weather_historical (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id         TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    temperature     REAL,
    humidity        REAL,
    cloud_cover     REAL,
    apparent_temp   REAL,
    precipitation   REAL,
    rain            REAL,
    weather_code    INTEGER,
    pressure        REAL,
    wind_speed      REAL,
    wind_direction  REAL,
    wind_gusts      REAL,
    dewpoint        REAL,
    fetched_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(city_id, timestamp)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id         TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    temperature     REAL,
    humidity        REAL,
    cloud_cover     REAL,
    apparent_temp   REAL,
    precipitation   REAL,
    rain            REAL,
    weather_code    INTEGER,
    pressure        REAL,
    wind_speed      REAL,
    wind_direction  REAL,
    wind_gusts      REAL,
    dewpoint        REAL,
    forecast_days   INTEGER DEFAULT 3,
    fetched_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(city_id, timestamp, fetched_at)
);

CREATE TABLE IF NOT EXISTS weather_current (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id      TEXT    NOT NULL,
    temperature  REAL,
    feels_like   REAL,
    humidity     REAL,
    pressure     REAL,
    cloud_cover  REAL,
    wind_speed   REAL,
    weather_code INTEGER,
    weather_desc TEXT,
    visibility   REAL,
    fetched_at   TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_hist_city_ts
    ON weather_historical(city_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_fc_city_ts
    ON weather_forecast(city_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cur_city
    ON weather_current(city_id, fetched_at);
"""

SCHEMA_SQL_POSTGRES = """
CREATE TABLE IF NOT EXISTS weather_historical (
    id              SERIAL PRIMARY KEY,
    city_id         VARCHAR(50)  NOT NULL,
    timestamp       TIMESTAMP    NOT NULL,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    cloud_cover     DOUBLE PRECISION,
    apparent_temp   DOUBLE PRECISION,
    precipitation   DOUBLE PRECISION,
    rain            DOUBLE PRECISION,
    weather_code    INTEGER,
    pressure        DOUBLE PRECISION,
    wind_speed      DOUBLE PRECISION,
    wind_direction  DOUBLE PRECISION,
    wind_gusts      DOUBLE PRECISION,
    dewpoint        DOUBLE PRECISION,
    fetched_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(city_id, timestamp)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
    id              SERIAL PRIMARY KEY,
    city_id         VARCHAR(50)  NOT NULL,
    timestamp       TIMESTAMP    NOT NULL,
    temperature     DOUBLE PRECISION,
    humidity        DOUBLE PRECISION,
    cloud_cover     DOUBLE PRECISION,
    apparent_temp   DOUBLE PRECISION,
    precipitation   DOUBLE PRECISION,
    rain            DOUBLE PRECISION,
    weather_code    INTEGER,
    pressure        DOUBLE PRECISION,
    wind_speed      DOUBLE PRECISION,
    wind_direction  DOUBLE PRECISION,
    wind_gusts      DOUBLE PRECISION,
    dewpoint        DOUBLE PRECISION,
    forecast_days   INTEGER DEFAULT 3,
    fetched_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(city_id, timestamp, fetched_at)
);

CREATE TABLE IF NOT EXISTS weather_current (
    id           SERIAL PRIMARY KEY,
    city_id      VARCHAR(50)  NOT NULL,
    temperature  DOUBLE PRECISION,
    feels_like   DOUBLE PRECISION,
    humidity     DOUBLE PRECISION,
    pressure     DOUBLE PRECISION,
    cloud_cover  DOUBLE PRECISION,
    wind_speed   DOUBLE PRECISION,
    weather_code INTEGER,
    weather_desc VARCHAR(200),
    visibility   DOUBLE PRECISION,
    fetched_at   TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hist_city_ts
    ON weather_historical(city_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_fc_city_ts
    ON weather_forecast(city_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cur_city
    ON weather_current(city_id, fetched_at);
"""


# ── PostgreSQL Connection (Cloud SQL) ──

_pg_pool = None


def _get_pg_pool():
    """Lazy-init PostgreSQL connection pool via Cloud SQL Connector."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    instance_name = os.getenv("INSTANCE_CONNECTION_NAME", "")

    if instance_name:
        # Cloud SQL Connector (recommended for Cloud Run)
        from google.cloud.sql.connector import Connector
        import pg8000

        connector = Connector()

        def getconn():
            return connector.connect(
                instance_name,
                "pg8000",
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASS", ""),
                db=os.getenv("DB_NAME", "weather"),
            )

        import pg8000.dbapi
        _pg_pool = getconn
    else:
        # Direct TCP connection (local dev with Cloud SQL Proxy)
        import pg8000

        def getconn():
            return pg8000.connect(
                host=os.getenv("DB_HOST", "127.0.0.1"),
                port=int(os.getenv("DB_PORT", "5432")),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASS", ""),
                database=os.getenv("DB_NAME", "weather"),
            )

        _pg_pool = getconn

    return _pg_pool


class PgRowProxy:
    """Make pg8000 rows behave like sqlite3.Row (dict-like access)."""

    def __init__(self, columns, values):
        self._data = dict(zip(columns, values))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data[key]

    def keys(self):
        return self._data.keys()


@contextmanager
def _get_pg_connection():
    """Yield a PostgreSQL connection with dict-like row access."""
    pool_fn = _get_pg_pool()
    conn = pool_fn()
    cursor = conn.cursor()

    class PgConnWrapper:
        """Wrapper to make pg8000 API similar to sqlite3."""

        def execute(self, sql, params=None):
            # Convert SQLite-style ? params to pg8000 %s params
            pg_sql = sql.replace("?", "%s")
            cursor.execute(pg_sql, params or ())
            return self

        def executescript(self, sql):
            for statement in sql.split(";"):
                stmt = statement.strip()
                if stmt:
                    cursor.execute(stmt)

        def fetchone(self):
            row = cursor.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cursor.description]
            return PgRowProxy(cols, row)

        def fetchall(self):
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            return [PgRowProxy(cols, r) for r in rows]

    wrapper = PgConnWrapper()
    try:
        yield wrapper
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


# ── SQLite Connection (Local Dev) ──

@contextmanager
def _get_sqlite_connection():
    """Yield a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Public API (auto-selects backend) ──

@contextmanager
def get_connection():
    """Get database connection. Auto-selects PostgreSQL or SQLite."""
    if DB_TYPE == "postgres":
        with _get_pg_connection() as conn:
            yield conn
    else:
        with _get_sqlite_connection() as conn:
            yield conn


def init_db():
    """Create tables + indexes if they don't exist."""
    if DB_TYPE == "postgres":
        with _get_pg_connection() as conn:
            conn.executescript(SCHEMA_SQL_POSTGRES)
        print("✓ Database initialized (Cloud SQL PostgreSQL)")
    else:
        with _get_sqlite_connection() as conn:
            conn.executescript(SCHEMA_SQL_SQLITE)
        print(f"✓ Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
