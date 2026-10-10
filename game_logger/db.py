"""SQLite connection, schema and backups for the game logger."""
import os
import shutil
import sqlite3
import datetime

from . import config

SCHEMA_VERSION = 2

SCHEMA_V1 = """
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE teams (
    team_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    kit_colours TEXT NOT NULL DEFAULT '[]',      -- JSON: [{"rgb":[r,g,b],"tolerance":[h,s,v]}, ...] as the team classifier uses them
    is_active   INTEGER NOT NULL DEFAULT 1,
    notes       TEXT
);

CREATE TABLE games (
    game_id             INTEGER PRIMARY KEY,     -- same ids as the old spreadsheet; new games get max+1
    date                TEXT,                    -- 'YYYY.MM.DD' (the format the spreadsheet used)
    game_type           TEXT,
    game_format         TEXT,
    title               TEXT NOT NULL,
    description         TEXT,
    team_a_id           INTEGER REFERENCES teams(team_id),
    team_b_id           INTEGER REFERENCES teams(team_id),
    score_a             INTEGER,
    score_b             INTEGER,
    source_video        TEXT,                    -- the ANALYSIS video (the pipeline's input): as typed, relative to the application folder or absolute
    output_video        TEXT,                    -- the ANNOTATED video (detections drawn on it)
    youtube_360_video   TEXT,
    running_speed_km_h  REAL,
    game_start_s        INTEGER,                 -- seconds into the video
    half_time_start_s   INTEGER,
    half_time_end_s     INTEGER,
    game_end_s          INTEGER,
    notes               TEXT,
    archived            INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT,
    updated_at          TEXT
);

CREATE TABLE game_exclusions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id  INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    start_s  INTEGER NOT NULL,
    end_s    INTEGER NOT NULL,
    reason   TEXT
);

CREATE TABLE pitch_calibrations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id          INTEGER NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    created_at       TEXT NOT NULL,
    source           TEXT NOT NULL,              -- 'excel-import' | 'calibrator' | 'manual'
    label            TEXT,
    is_active        INTEGER NOT NULL DEFAULT 0,
    vertices_json    TEXT NOT NULL,              -- {"left_bottom":[x,y], ... six outline points}
    goal_left_json   TEXT,                       -- [[x,y] x4] or NULL
    goal_right_json  TEXT,
    calibrator_json  TEXT,                       -- the Pitch Calibrator's full JSON, exactly as exported
    notes            TEXT
);
CREATE UNIQUE INDEX one_active_calibration ON pitch_calibrations(game_id) WHERE is_active = 1;
CREATE INDEX idx_calibrations_game ON pitch_calibrations(game_id);
CREATE INDEX idx_exclusions_game ON game_exclusions(game_id);
"""


# Each step takes the database from version N to N+1.
MIGRATIONS = {
    1: ["ALTER TABLE games ADD COLUMN edited_video TEXT"],      # v2: the edited, TV-style video
}


def now_iso():
    return datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")


def connect(path=None):
    """Open (creating if needed) the game logger database."""
    path = path or config.DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, check_same_thread=False)   # FastAPI may open and close a request's connection on different worker threads; each connection serves one request at a time
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA_V1)
        conn.execute("PRAGMA user_version = 1")
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES ('running_speed_km_h', ?)",
                     (str(config.DEFAULT_RUNNING_SPEED_KM_H),))
        conn.commit()
        version = 1
    elif version > SCHEMA_VERSION:
        raise RuntimeError(f"{path} was made by a newer game_logger (schema {version}); this one only knows {SCHEMA_VERSION}")
    elif version < SCHEMA_VERSION:
        conn.commit()
        backup("before-upgrade", path=path)                      # a copy of the database as it was, just in case
    while version < SCHEMA_VERSION:
        for stmt in MIGRATIONS[version]:
            conn.execute(stmt)
        version += 1
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    return conn


def backup(reason="daily", path=None, keep=40):
    """Copy the database into the backups folder (SQLite's online backup, safe while the app is running).
    'daily' makes at most one copy per day. Returns the backup path, or None if nothing was done."""
    path = path or config.DB_PATH
    if not os.path.exists(path):
        return None
    os.makedirs(config.BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now()
    if reason == "daily":
        name = f"game-logger-{stamp:%Y%m%d}.db"
        dest = os.path.join(config.BACKUP_DIR, name)
        if os.path.exists(dest):
            return None
    else:
        name = f"game-logger-{stamp:%Y%m%d-%H%M%S}-{reason}.db"
        dest = os.path.join(config.BACKUP_DIR, name)
    src = sqlite3.connect(path)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    files = sorted(f for f in os.listdir(config.BACKUP_DIR) if f.startswith("game-logger-") and f.endswith(".db"))
    for old in files[:-keep]:
        try:
            os.remove(os.path.join(config.BACKUP_DIR, old))
        except OSError:
            pass
    return dest
