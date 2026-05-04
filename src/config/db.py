"""SQLite database layer — connection, schema init, CRUD helpers."""
import sqlite3
from contextlib import contextmanager

from src.config.settings import DB_PATH

SCHEMA_SQL = """
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

CREATE TABLE IF NOT EXISTS weather_ai_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    city_id         TEXT    NOT NULL,
    target_time     TEXT    NOT NULL,
    predicted_temp  REAL,
    predicted_humidity   REAL,
    predicted_wind_speed REAL,
    predicted_cloud_cover REAL,
    model_version   TEXT,
    created_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(city_id, target_time, model_version)
);
CREATE INDEX IF NOT EXISTS idx_ai_pred_city_time
    ON weather_ai_predictions(city_id, target_time);
"""


@contextmanager
def get_connection():
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


def init_db():
    """Create tables + indexes if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"✓ Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()
