"""Run the pipeline from the browser: start, stop, status, and the console log."""
import os

from fastapi import APIRouter, Depends, HTTPException, Request

from . import db, paths, repo, runner

router = APIRouter()


def _get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _only_from_the_app(request: Request):
    """Starting or stopping a run needs this header, which only the game logger's own page sends (another web page
    cannot add it to a request to this computer without the browser asking first)."""
    if request.headers.get("x-gl-app") != "1":
        raise HTTPException(403, "Runs can only be started from the game logger page")


def _public(meta):
    if not meta:
        return None
    out = {k: meta.get(k) for k in ("id", "game_id", "title", "steps", "duration_s", "test", "started", "ended", "exit_code", "status", "notes", "note")}
    out["log_shown"] = paths.shown(meta.get("log_file"))
    return out


@router.get("/api/run/steps")
def steps():
    return [{"id": i, "label": l, "help": h} for i, l, h in runner.STEPS]


@router.get("/api/run")
def run_status():
    return {"run": _public(runner.status())}


@router.post("/api/games/{game_id}/run")
def start_run(game_id: int, body: dict, request: Request, conn=Depends(_get_conn)):
    _only_from_the_app(request)
    try:
        meta = runner.start(conn, game_id, body.get("steps") or [], body.get("duration_s"))
    except paths.GameFolderError as e:
        raise repo.ValidationError(str(e))
    return {"run": _public(meta)}


@router.post("/api/run/stop")
def stop_run(request: Request):
    _only_from_the_app(request)
    return {"run": _public(runner.stop())}


@router.get("/api/run/log")
def run_log(offset: int = 0, tail: bool = False):
    out = runner.current_log(offset, tail)
    return out


@router.get("/api/games/{game_id}/runs")
def game_runs(game_id: int):
    return runner.list_runs(game_id)


@router.get("/api/games/{game_id}/runs/{run_id}/log")
def old_log(game_id: int, run_id: str, offset: int = 0, tail: bool = True):
    p = runner._safe_log_path(game_id, run_id)
    if not p:
        raise HTTPException(404, "That log was not found")
    return runner.read_log(p, offset, tail)
