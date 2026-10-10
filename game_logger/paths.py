"""Where a game's files live: one data folder (normally on the external drive) with a folder per game inside it.

    <data folder>/
        games/0022_2026-10-02_mens-league_polis-v-aphrodite-wanderers/
            videos/    0022_analysis.mp4  0022_edited.mp4  annotated/0022_annotated_<time>.mp4
            data/      0022_detections.db  object-images/
            reports/   0022_match-report_<time>.pdf  0022_detection-report_<time>.pdf
            _work/     throw-away files made while a run is going (safe to delete)
        _app-data/game-logger-backups/   a copy of every database backup
        _library/  _inbox/

A game's folder is found by the 4-digit game number at the start of its name, so the rest of the name can be changed
freely. Nothing in here needs the web framework, so the pipeline (main.py) can use it too."""
import os
import re
import shutil
import sqlite3
import time

from . import config, db

ROOT_SETTING = "data_root"
ENV_ROOT = "WALKING_FOOTBALL_ROOT"
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
SUBFOLDERS = ("videos", "videos/annotated", "data", "reports", "_work")
SLOT_FILES = {"analysis": "analysis", "edited": "edited"}

_cache = {}


class GameFolderError(Exception):
    """Something about the data folder or a game's folder is wrong; the message says what to do."""


def _cached(key, ttl, fn):
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def forget():
    _cache.clear()


def prefix(game_id):
    try:
        return f"{int(game_id):04d}"
    except (TypeError, ValueError):
        return str(game_id)


# ------------------------------------------------------------------------------------------ the data folder

def configured_root():
    """The data folder as set (environment variable first, then the Settings page), whether or not it is connected."""
    env = (os.environ.get(ENV_ROOT) or "").strip()
    if env:
        return os.path.expanduser(env)

    def read():
        # a plain read-only look at the settings table: it must not run db.connect(), which can be what is calling us
        # (a database upgrade makes a backup, and the backup looks for the drive)
        try:
            conn = sqlite3.connect("file:" + config.DB_PATH + "?mode=ro", uri=True, timeout=5)
            try:
                row = conn.execute("SELECT value FROM settings WHERE key = ?", (ROOT_SETTING,)).fetchone()
            finally:
                conn.close()
        except Exception:                                   # noqa: BLE001 - no database or settings yet
            return None
        v = (row[0] if row else None) or ""
        return os.path.expanduser(v.strip()) if v.strip() else None
    return _cached("root", 2.0, read)


def root_info():
    p = configured_root()
    return {"path": p, "set": bool(p), "exists": bool(p and os.path.isdir(p))}


def active_root():
    """The data folder if it is set AND connected, else None."""
    r = root_info()
    return r["path"] if r["exists"] else None


def set_root(path):
    """Save the data folder setting ('' clears it). The folder must exist."""
    path = (path or "").strip()
    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            raise GameFolderError(f"That folder was not found: {path}")
    conn = db.connect()
    try:
        conn.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (ROOT_SETTING, path))
        conn.commit()
    finally:
        conn.close()
    forget()
    return path


def games_dir(root=None):
    root = root or active_root()
    return os.path.join(root, "games") if root else None


# ------------------------------------------------------------------------------------------ game folders

def _slug(text):
    t = (text or "").lower().replace(" vs ", " v ")
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "game"


def folder_name(g):
    """0022_2026-10-02_mens-league_polis-v-aphrodite-wanderers"""
    date = (g.get("date") or "").replace(".", "-") or "undated"
    kind = "-".join(_slug(x) for x in (g.get("game_format"), g.get("game_type")) if x)
    parts = [prefix(g["game_id"]), date] + ([kind] if kind else []) + [_slug(g.get("title"))]
    return "_".join(parts)[:120]


def _index(root):
    def build():
        gd = os.path.join(root, "games")
        out = {}
        try:
            for name in sorted(os.listdir(gd)):
                m = re.match(r"^(\d{4,})_", name)
                if m and os.path.isdir(os.path.join(gd, name)):
                    out.setdefault(int(m.group(1)), os.path.join(gd, name))
        except OSError:
            pass
        return out
    return _cached(("index", root), 3.0, build)


def game_folder(game_id):
    """Absolute path of the game's folder, or None (no data folder, drive not connected, or not made yet)."""
    root = active_root()
    if not root:
        return None
    try:
        return _index(root).get(int(game_id))
    except (TypeError, ValueError):
        return None


def create_game_folder(g):
    """Make (or find) the game's folder with all its sub-folders and a notes file. Returns (path, created)."""
    root = active_root()
    if not root:
        info = root_info()
        if info["set"]:
            raise GameFolderError(f"The data folder {info['path']} is not connected. Connect the drive and try again.")
        raise GameFolderError("No data folder is set yet. Choose it on the Settings page first.")
    existing = game_folder(g["game_id"])
    created = existing is None
    folder = existing or os.path.join(root, "games", folder_name(g))
    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    notes = os.path.join(folder, "notes.txt")
    if created and not os.path.exists(notes):
        with open(notes, "w", encoding="utf-8") as f:
            f.write(f"Game {g['game_id']}: {g.get('title') or ''}\n{g.get('date') or ''}  {g.get('game_format') or ''} {g.get('game_type') or ''}\n\n"
                    "Notes (e.g. the camera's original file name):\n")
    forget()
    return folder, created


def subfolder(folder, name):
    return os.path.join(folder, *name.split("/"))


def _videos_in(folder):
    try:
        return sorted((os.path.join(folder, n) for n in os.listdir(folder)
                       if os.path.splitext(n)[1].lower() in VIDEO_EXT and not n.startswith(".")), key=lambda p: os.path.basename(p).lower())
    except OSError:
        return []


def conventional_video(game_id, slot):
    """The analysis / edited video found in the game's videos folder by name (0022_analysis.mp4, 0022_edited.mp4), or None."""
    folder = game_folder(game_id)
    if not folder:
        return None
    base = SLOT_FILES[slot]
    want = (f"{prefix(game_id)}_{base}", base)
    for p in _videos_in(os.path.join(folder, "videos")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in want or stem.startswith(base + "_") or stem.startswith(base + "-"):
            return p
    return None


def annotated_files(game_id):
    """Annotated videos for the game, newest first: videos/annotated/* plus any 'annotated*' video loose in videos/."""
    folder = game_folder(game_id)
    if not folder:
        return []
    found = _videos_in(os.path.join(folder, "videos", "annotated"))
    found += [p for p in _videos_in(os.path.join(folder, "videos")) if os.path.basename(p).lower().startswith("annotated")]
    return sorted(found, key=os.path.getmtime, reverse=True)


def detections_db(game_id):
    folder = game_folder(game_id)
    return os.path.join(folder, "data", f"{prefix(game_id)}_detections.db") if folder else None


# ------------------------------------------------------------------------------------------ for the pipeline

def pipeline_paths(game_id):
    """Where main.py should put everything for this game. None when no data folder is set (the old shared output
    folder is then used). Raises GameFolderError, with what to do, when the folder is set but not connected."""
    info = root_info()
    if not info["set"]:
        return None
    if not info["exists"]:
        raise GameFolderError(f"The data folder {info['path']} is not connected. Connect the drive (or clear the data folder "
                              "on the game logger's Settings page to use the old output folder).")
    folder = game_folder(game_id)
    if folder is None:
        from . import repo
        conn = db.connect()
        try:
            g = repo.get_game(conn, int(game_id), with_detail=False)
        finally:
            conn.close()
        if g is None:
            raise GameFolderError(f"Game {game_id} is not in the game logger.")
        folder, _ = create_game_folder(g)
        print(f"Made the folder for game {game_id}: {folder}")
    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    data = os.path.join(folder, "data")
    return {
        "folder": folder,
        "db": os.path.join(data, f"{prefix(game_id)}_detections.db"),
        "object_images": os.path.join(data, "object-images"),
        "reports": os.path.join(folder, "reports"),
        "annotated": os.path.join(folder, "videos", "annotated"),
        "work": os.path.join(folder, "_work"),
    }


# ------------------------------------------------------------------------------------------ stored paths

def resolve_stored(path):
    """A path as stored in a game -> the real path. Absolute stays; relative is looked for in the data folder
    first, then in the application folder (how older games were saved)."""
    p = os.path.expanduser(str(path).strip())
    if os.path.isabs(p):
        return p
    root = active_root()
    if root:
        cand = os.path.normpath(os.path.join(root, p))
        if os.path.exists(cand):
            return cand
    return os.path.normpath(os.path.join(config.APP_DIR, p))


def stored_form(path):
    """What to save for a chosen file: relative to the data folder when it is inside it, else relative to the
    application folder when inside that, else the full path."""
    root = configured_root()
    if root:
        rel = os.path.relpath(path, root)
        if not rel.startswith(".."):
            return rel
    rel = os.path.relpath(path, config.APP_DIR)
    return path if rel.startswith("..") else rel


def shown(path):
    """A short form for display: relative to the data folder when inside it."""
    if not path:
        return None
    root = configured_root()
    if root:
        rel = os.path.relpath(path, root)
        if not rel.startswith(".."):
            return rel
    return path


# ------------------------------------------------------------------------------------------ backups on the drive

def copy_backup_to_drive(local_backup, keep=200):
    """Put a copy of a database backup in <data folder>/_app-data/game-logger-backups when the drive is connected.
    Returns the copy's path, or None."""
    root = active_root()
    if not root or not local_backup or not os.path.exists(local_backup):
        return None
    try:
        dest_dir = os.path.join(root, "_app-data", "game-logger-backups")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(local_backup))
        if not os.path.exists(dest):
            shutil.copy2(local_backup, dest + ".part")
            os.replace(dest + ".part", dest)
        files = sorted(f for f in os.listdir(dest_dir) if f.startswith("game-logger-") and f.endswith(".db"))
        for old in files[:-keep]:
            try:
                os.remove(os.path.join(dest_dir, old))
            except OSError:
                pass
        return dest
    except OSError:
        return None
