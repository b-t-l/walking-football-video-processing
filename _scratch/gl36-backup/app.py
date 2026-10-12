"""The game logger web app: a small JSON API plus the static pages in game_logger/static/."""
import io
import math
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, config, db, paths, repo
from .files import router as files_router
from .calibrator_api import router as calibrator_router
from .video_api import router as video_router
from .folder_api import router as folder_router
from .run_api import router as run_router
from .record import RECORD_KEYS, build_game_record

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Game logger", version=__version__, docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")


@app.middleware("http")
async def only_local_hosts(request, call_next):
    """Refuse requests whose address is not this computer (guards against other web pages talking to the app)."""
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "testserver"):
        return JSONResponse(status_code=400, content={"errors": ["This app only answers on 127.0.0.1 / localhost"]})
    return await call_next(request)


def get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


@app.exception_handler(repo.ValidationError)
async def validation_handler(request, exc):
    return JSONResponse(status_code=422, content={"errors": exc.errors})


def _json_safe(rec):
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}


def _need_game(conn, game_id, detail=True):
    g = repo.get_game(conn, game_id, with_detail=detail)
    if g is None:
        raise HTTPException(404, f"Game {game_id} does not exist")
    return g


# ------------------------------------------------------------------------------------------ meta / settings

@app.get("/api/meta")
def meta(conn=Depends(get_conn)):
    return {"version": __version__, "db_path": config.DB_PATH, "backup_dir": config.BACKUP_DIR,
            "game_types": config.GAME_TYPES, "game_formats": config.GAME_FORMATS,
            "running_speed_km_h": repo.default_running_speed(conn), "next_game_id": repo.next_game_id(conn),
            "calibrator_url": repo.get_setting(conn, "calibrator_url", config.CALIBRATOR_URL),
            "data_root": paths.root_info(), "data_root_from_env": bool((os.environ.get(paths.ENV_ROOT) or "").strip()),
            "point_names": config.PITCH_POINT_NAMES}


@app.put("/api/settings")
def put_settings(body: dict, conn=Depends(get_conn)):
    if "running_speed_km_h" in body:
        try:
            v = float(body["running_speed_km_h"])
        except (TypeError, ValueError):
            raise repo.ValidationError("Running speed must be a number")
        if not 3 <= v <= 40:
            raise repo.ValidationError("Running speed must be between 3 and 40 km/h")
        repo.set_setting(conn, "running_speed_km_h", v)
    if "data_root" in body:
        try:
            paths.set_root(body["data_root"])
        except paths.GameFolderError as e:
            raise repo.ValidationError(str(e))
    if "calibrator_url" in body:
        repo.set_setting(conn, "calibrator_url", (body["calibrator_url"] or "").strip())
    return meta(conn)


@app.get("/api/check-video")
def check_video(path: str):
    resolved = repo.resolve_video_path(path)
    return {"resolved": resolved, "exists": bool(resolved and os.path.exists(resolved))}


# ------------------------------------------------------------------------------------------ games

@app.get("/api/games")
def games(archived: bool = False, conn=Depends(get_conn)):
    return repo.list_games(conn, include_archived=archived)


@app.post("/api/games")
def create_game(body: dict, conn=Depends(get_conn)):
    g, warnings = repo.save_game(conn, body)
    warnings = warnings + _make_folder_for_new_game(g)
    return {"game": repo.get_game(conn, g["game_id"]), "warnings": warnings}


def _make_folder_for_new_game(g):
    """A new game gets its folder in the data folder straight away. If that cannot be done (drive not connected) the
    game is still saved and the warning says what to do."""
    info = paths.root_info()
    if not info["set"]:
        return []
    try:
        paths.create_game_folder(g)
    except paths.GameFolderError as e:
        return [f"The game was saved, but its folder was not made: {e} Press \"Create game folder\" in the Video files section later."]
    except OSError as e:
        return [f"The game was saved, but its folder could not be made ({e.strerror}). Press \"Create game folder\" in the Video files section later."]
    return []


@app.get("/api/games/{game_id}")
def read_game(game_id: int, conn=Depends(get_conn)):
    return _need_game(conn, game_id)


@app.put("/api/games/{game_id}")
def update_game(game_id: int, body: dict, conn=Depends(get_conn)):
    _need_game(conn, game_id, detail=False)
    g, warnings = repo.save_game(conn, body, game_id)
    return {"game": g, "warnings": warnings}


@app.delete("/api/games/{game_id}")
def remove_game(game_id: int, conn=Depends(get_conn)):
    _need_game(conn, game_id, detail=False)
    db.backup("before-delete")
    repo.delete_game(conn, game_id)
    return {"deleted": game_id}


@app.post("/api/games/{game_id}/archive")
def archive_game(game_id: int, body: dict, conn=Depends(get_conn)):
    _need_game(conn, game_id, detail=False)
    repo.set_archived(conn, game_id, bool(body.get("archived", True)))
    return repo.get_game(conn, game_id)


@app.post("/api/games/{game_id}/duplicate")
def copy_game(game_id: int, conn=Depends(get_conn)):
    _need_game(conn, game_id, detail=False)
    return repo.duplicate_game(conn, game_id)


@app.get("/api/games/{game_id}/record")
def pipeline_record(game_id: int, conn=Depends(get_conn)):
    """Exactly what the pipeline is given for this game."""
    _need_game(conn, game_id, detail=False)
    return _json_safe(build_game_record(conn, game_id))


# ------------------------------------------------------------------------------------------ calibrations

@app.post("/api/games/{game_id}/calibrations")
def add_calibration(game_id: int, body: dict, conn=Depends(get_conn)):
    _need_game(conn, game_id, detail=False)
    cal = repo.add_calibration_from_json(conn, game_id, body.get("json"), label=(body.get("label") or None),
                                         notes=body.get("notes"), make_active=bool(body.get("make_active", True)))
    return {"calibration": cal, "game": repo.get_game(conn, game_id)}


@app.get("/api/calibrations/{cal_id}")
def read_calibration(cal_id: int, conn=Depends(get_conn)):
    cal = repo.get_calibration(conn, cal_id)
    if not cal:
        raise HTTPException(404, "No such calibration")
    return cal


@app.get("/api/calibrations/{cal_id}/download")
def download_calibration(cal_id: int, conn=Depends(get_conn)):
    cal = repo.get_calibration(conn, cal_id)
    if not cal or not cal["calibrator_json"]:
        raise HTTPException(404, "That calibration has no calibrator JSON to download")
    return Response(cal["calibrator_json"], media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="game-{cal["game_id"]}-calibration-{cal_id}.json"'})


@app.post("/api/games/{game_id}/calibrations/{cal_id}/activate")
def activate_calibration(game_id: int, cal_id: int, conn=Depends(get_conn)):
    repo.set_active_calibration(conn, game_id, cal_id)
    return repo.get_game(conn, game_id)


@app.delete("/api/games/{game_id}/calibrations/{cal_id}")
def remove_calibration(game_id: int, cal_id: int, conn=Depends(get_conn)):
    repo.delete_calibration(conn, game_id, cal_id)
    return repo.get_game(conn, game_id)


# ------------------------------------------------------------------------------------------ teams

@app.get("/api/teams")
def teams(conn=Depends(get_conn)):
    return repo.list_teams(conn)


@app.post("/api/teams")
def create_team(body: dict, conn=Depends(get_conn)):
    return repo.save_team(conn, body)


@app.put("/api/teams/{team_id}")
def update_team(team_id: int, body: dict, conn=Depends(get_conn)):
    return repo.save_team(conn, body, team_id)


@app.delete("/api/teams/{team_id}")
def remove_team(team_id: int, clear_from_games: bool = False, conn=Depends(get_conn)):
    if clear_from_games:
        db.backup("before-team-delete")
    repo.delete_team(conn, team_id, clear_from_games)
    return {"deleted": team_id}


# ------------------------------------------------------------------------------------------ export

@app.get("/api/export.xlsx")
def export_excel(conn=Depends(get_conn)):
    """The whole database as a spreadsheet in the old games-logger.xlsx layout (a safety net, not a way to edit)."""
    import pandas as pd
    rows = [build_game_record(conn, g["game_id"]) for g in sorted(repo.list_games(conn, include_archived=True), key=lambda g: g["game_id"])]
    df = pd.DataFrame(rows, columns=RECORD_KEYS)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="games")
    return Response(buf.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": 'attachment; filename="games-logger-export.xlsx"'})


# ------------------------------------------------------------------------------------------ pages

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-store"})


app.include_router(files_router)
app.include_router(calibrator_router)
app.include_router(video_router)
app.include_router(folder_router)
app.include_router(run_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
