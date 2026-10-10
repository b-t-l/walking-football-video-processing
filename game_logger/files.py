"""A folder browser for the 'Browse...' buttons. The app runs on your computer, so it can list your real folders
(a web page normally cannot see file paths). Read-only: it only lists names; it never opens, moves or changes files."""
import os

from fastapi import APIRouter, Depends

from . import config, db, paths, repo

router = APIRouter()

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".webm", ".mpg", ".mpeg", ".insv"}
MAX_ENTRIES = 3000


def stored_form(path):
    """What gets saved in the game: relative to the data folder (the external drive) when the file is inside it, else
    relative to the application folder when inside that, otherwise the full path."""
    return paths.stored_form(path)


def _get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _shortcuts(game_id=None):
    items = []

    def add(label, path):
        if path and os.path.isdir(path) and all(path != i["path"] for i in items):
            items.append({"label": label, "path": path})

    if game_id is not None:
        gf = paths.game_folder(game_id)
        if gf:
            add("This game's folder", gf)
            add("Its videos", os.path.join(gf, "videos"))
    root = paths.active_root()
    if root:
        add("Data folder", root)
        add("All games", os.path.join(root, "games"))
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


def _start_folder(conn, current, game_id=None):
    cur = repo.resolve_video_path(current) if current else None
    if not cur and game_id is not None:                         # no file chosen yet: start in this game's videos folder
        gf = paths.game_folder(game_id)
        if gf and os.path.isdir(os.path.join(gf, "videos")):
            return os.path.join(gf, "videos")
    if cur and os.path.isdir(cur):
        return cur
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
def browse(path: str = "", all: bool = False, current: str = "", game: int | None = None, conn=Depends(_get_conn)):
    note = None
    folder = os.path.abspath(os.path.expanduser(path)) if path.strip() else _start_folder(conn, current, game)
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
            "shortcuts": _shortcuts(game)}


@router.put("/api/video-folder")
def remember_folder(body: dict, conn=Depends(_get_conn)):
    p = body.get("path") or ""
    if os.path.isdir(p):
        repo.set_setting(conn, "last_video_folder", p)
    return {"ok": True}
