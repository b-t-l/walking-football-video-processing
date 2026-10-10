"""Paths and settings. Override the database location with the GAME_LOGGER_DB environment variable."""
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the application folder (contains main.py)
VIDEO_PROCESSING_DIR = os.path.dirname(APP_DIR)                           # parent of it (contains output/)

DB_PATH = os.environ.get("GAME_LOGGER_DB") or os.path.join(VIDEO_PROCESSING_DIR, "game-logger.db")
BACKUP_DIR = os.path.join(os.path.dirname(DB_PATH), "game-logger-backups")
EXCEL_PATH_DEFAULT = os.path.join(APP_DIR, "games-logger.xlsx")
TEAMS_PY = os.path.join(APP_DIR, "team_assigner", "teams.py")

HOST = "127.0.0.1"          # local only: nothing outside this computer can reach the app
PORT = int(os.environ.get("GAME_LOGGER_PORT", "8765"))

DEFAULT_RUNNING_SPEED_KM_H = 15
GAME_TYPES = ["league", "cup", "friendly", "training", "n/a"]
GAME_FORMATS = ["mens", "womens", "mixed", "n/a"]
PITCH_POINT_NAMES = ["left_bottom", "left_top", "right_top", "right_bottom", "centre_top", "centre_bottom"]

# The published Pitch Calibrator (a Claude artifact). Change it on the Settings page if it ever moves.
CALIBRATOR_URL = "https://claude.ai/artifact/KgxwynwZoS3vxKkQnwsBBo"
