"""A folder browser for the 'Browse...' buttons. The app runs on your computer, so it can list your real folders
(a web page normally cannot see file paths). Read-only: it only lists names; it never opens, moves or changes files."""
import os

from fastapi import APIRouter, Depends

from . import config, db, repo

router = APIRouter()

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".webm", ".mpg", ".mpeg", ".insv"}
MAX_ENTRIES = 3000


def stored_form(path):
    """What gets saved in the game: relative to the application folder when the file is inside it, otherwise the full path."""
    rel = os.path.relpath(path, config.APP_DIR)
    return path if rel.startswith("..") else rel


def _get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _shortcuts():
    items = []

    def add(label, path):
        if path and os.path.isdir(path) and all(path != i["path"] for i in items):
            items.append({"label": label, "path": path})

    add("input_videos", os.path.join(config.APP_DIR, "input_videos"))
    add("Application folder", config.APP_DIR)
    if os.path.isdir("/Volumes"):
        try:
            for name in sorted(os.listdir("/Volumes")):
                if not name.startswith("."):
                    add(name, os.path.join("/Volumes", name))
        except OSError:
            pass
    home = os.path.expanduser("~")
    add("Movies", os.path.join(home, "Movies"))
    add("Home", home)
    return items


def _start_folder(conn, current):
    cur = repo.resolve_video_path(current) if current else None
    if cur and os.path.isfile(cur):
        return os.path.dirname(cur)
    if cur and os.path.isdir(os.path.dirname(cur)):
        return os.path.dirname(cur)
    last = repo.get_setting(conn, "last_video_folder")
    if last and os.path.isdir(last):
        return last
    vids = os.path.join(config.APP_DIR, "input_videos")
    return vids if os.path.isdir(vids) else config.APP_DIR


@router.get("/api/browse")
def browse(path: str = "", all: bool = False, current: str = "", conn=Depends(_get_conn)):
    note = None
    folder = os.path.abspath(os.path.expanduser(path)) if path.strip() else _start_folder(conn, current)
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    while not os.path.isdir(folder) and os.path.dirname(folder) != folder:
        note = "That folder does not exist, so the nearest one that does is shown."
        folder = os.path.dirname(folder)
    entries = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    is_dir = e.is_dir()
                    if is_dir:
                        entries.append({"name": e.name, "type": "dir", "path": e.path})
                    else:
                        ext = os.path.splitext(e.name)[1].lower()
                        if all or ext in VIDEO_EXT:
                            entries.append({"name": e.name, "type": "video" if ext in VIDEO_EXT else "file", "path": e.path,
                                            "size": e.stat().st_size, "stored": stored_form(e.path)})
                except OSError:
                    continue
                if len(entries) >= MAX_ENTRIES:
                    note = "Only the first 3000 items are shown."
                    break
    except PermissionError:
        note = "This computer did not let the app open that folder. If it is on the Desktop, Documents or an external drive, allow access when macOS asks, or allow your terminal under System Settings > Privacy & Security > Files and Folders."
    except OSError as err:
        note = f"Could not read that folder ({err.strerror})."
    entries.sort(key=lambda x: (x["type"] != "dir", x["name"].casefold()))
    parent = os.path.dirname(folder)
    return {"path": folder, "parent": parent if parent != folder else None, "entries": entries, "note": note,
            "shortcuts": _shortcuts()}


@router.put("/api/video-folder")
def remember_folder(body: dict, conn=Depends(_get_conn)):
    p = body.get("path") or ""
    if os.path.isdir(p):
        repo.set_setting(conn, "last_video_folder", p)
    return {"ok": True}
