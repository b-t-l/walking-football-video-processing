"""Creating and opening a game's folder in the data folder (the external drive)."""
import os
import subprocess
import sys

import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from . import db, paths, repo

router = APIRouter()


def _get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _need_game(conn, game_id):
    g = repo.get_game(conn, game_id, with_detail=False)
    if g is None:
        raise HTTPException(404, f"Game {game_id} does not exist")
    return g


@router.post("/api/games/{game_id}/folder")
def create_folder(game_id: int, conn=Depends(_get_conn)):
    g = _need_game(conn, game_id)
    try:
        folder, created = paths.create_game_folder(g)
    except paths.GameFolderError as e:
        raise repo.ValidationError(str(e))
    return {"path": folder, "shown": paths.shown(folder), "created": created}


@router.post("/api/games/{game_id}/folder/open")
def open_folder(game_id: int, conn=Depends(_get_conn)):
    """Show the game's folder in Finder (this app only ever runs on your own computer)."""
    _need_game(conn, game_id)
    folder = paths.game_folder(game_id)
    if not folder:
        raise repo.ValidationError("This game has no folder yet")
    cmd = ["open", folder] if sys.platform == "darwin" else ["xdg-open", folder] if sys.platform.startswith("linux") else ["explorer", folder]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        raise repo.ValidationError(f"Could not open the folder: {e}")
    return {"opened": folder}


# ---------------------------------------------------------------------------------------------- the game's reports

REPORT_KINDS = {".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8"}


def _report_dirs(game_id):
    folder = paths.game_folder(game_id)
    if not folder:
        return []
    base = os.path.join(folder, "reports")
    return [(False, base), (True, os.path.join(base, "test"))]


@router.get("/api/games/{game_id}/reports")
def list_reports(game_id: int, conn=Depends(_get_conn)):
    """The PDF and text reports in the game's reports folder (and its test folder), newest first."""
    _need_game(conn, game_id)
    out = []
    for is_test, d in _report_dirs(game_id):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            p = os.path.join(d, n)
            ext = os.path.splitext(n)[1].lower()
            if n.startswith(".") or ext not in REPORT_KINDS or not os.path.isfile(p):
                continue
            st = os.stat(p)
            out.append({"name": n, "test": is_test, "size": st.st_size, "mtime": st.st_mtime,
                        "when": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%d %b %Y %H:%M"), "kind": ext[1:]})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


@router.get("/api/games/{game_id}/reports/file")
def report_file(game_id: int, name: str, test: bool = False, conn=Depends(_get_conn)):
    """One report, shown in the browser. Only plain file names inside the reports folders are served."""
    _need_game(conn, game_id)
    ext = os.path.splitext(name)[1].lower()
    if name != os.path.basename(name) or name.startswith(".") or ext not in REPORT_KINDS:
        raise HTTPException(404, "No such report")
    for is_test, d in _report_dirs(game_id):
        p = os.path.join(d, name)
        if is_test == bool(test) and os.path.isfile(p):
            return FileResponse(p, media_type=REPORT_KINDS[ext], headers={"Content-Disposition": f'inline; filename="{name}"'})
    raise HTTPException(404, "No such report")
