"""Identity page APIs: detections boxes at a time + assign Polis roster names to track ids."""
from fastapi import APIRouter, Depends, HTTPException

from . import db, identity, repo

router = APIRouter()


def _get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _need_game(conn, game_id):
    g = repo.get_game(conn, game_id, with_detail=True)
    if g is None:
        raise HTTPException(404, f"Game {game_id} does not exist")
    return g


@router.get("/api/games/{game_id}/identity")
def identity_overview(game_id: int, conn=Depends(_get_conn)):
    g = _need_game(conn, game_id)
    try:
        meta = identity.detections_meta(game_id)
        has_detections = True
        error = None
    except repo.ValidationError as e:
        meta = None
        has_detections = False
        error = e.errors[0] if e.errors else str(e)
    roster = identity.roster_for_game(conn, g)
    polis_ids = identity.polis_team_ids(conn, g)
    return {
        "game_id": game_id,
        "title": g.get("title"),
        "game_format": g.get("game_format"),
        "team_a": g.get("team_a"),
        "team_b": g.get("team_b"),
        "team_a_id": g.get("team_a_id"),
        "team_b_id": g.get("team_b_id"),
        "polis_team_ids": polis_ids,
        "has_detections": has_detections,
        "detections_error": error,
        "detections": meta,
        "roster": roster,
        "assignments": identity.list_assignments(conn, game_id) if has_detections or True else [],
        "videos": g.get("videos"),
        "game_start_s": g.get("game_start_s"),
        "game_end_s": g.get("game_end_s"),
    }


@router.get("/api/games/{game_id}/identity/boxes")
def identity_boxes(game_id: int, t: float = 0.0, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    try:
        return identity.boxes_at_time(conn, game_id, max(0.0, float(t)))
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)


@router.get("/api/games/{game_id}/identity/assignments")
def identity_assignments(game_id: int, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    return identity.list_assignments(conn, game_id)


@router.get("/api/games/{game_id}/identity/breaks")
def identity_breaks(game_id: int, conn=Depends(_get_conn)):
    """Open named tracks that disappear — jump points for re-assigning after a lost id."""
    _need_game(conn, game_id)
    try:
        return identity.track_breaks_for_game(conn, game_id)
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)


@router.post("/api/games/{game_id}/identity/skip-break")
def identity_skip_break(game_id: int, body: dict, conn=Depends(_get_conn)):
    """Skip the current break for a roster player (off-camera / not detected, or ignore a gap)."""
    _need_game(conn, game_id)
    try:
        rid = int(body.get("roster_player_id"))
    except (TypeError, ValueError):
        raise HTTPException(422, detail=["roster_player_id is required"])
    try:
        return identity.skip_break(conn, game_id, rid)
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)


@router.post("/api/games/{game_id}/identity/assign")
def identity_assign(game_id: int, body: dict, conn=Depends(_get_conn)):
    g = _need_game(conn, game_id)
    try:
        return identity.assign(
            conn,
            g,
            roster_player_id=int(body.get("roster_player_id")),
            track_player_id=body.get("track_player_id"),
            t=float(body.get("t", 0)),
        )
    except (TypeError, ValueError):
        raise HTTPException(422, detail=["roster_player_id, track_player_id and t are required"])
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)


@router.delete("/api/games/{game_id}/identity/assignments/{assignment_id}")
def identity_delete_assignment(game_id: int, assignment_id: int, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    try:
        identity.clear_assignment(conn, game_id, assignment_id)
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)
    return {"deleted": assignment_id}


@router.post("/api/games/{game_id}/identity/clear")
def identity_clear_at_time(game_id: int, body: dict, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    try:
        return identity.clear_name_at_time(
            conn,
            game_id,
            track_player_id=body.get("track_player_id"),
            t=float(body.get("t", 0)),
        )
    except (TypeError, ValueError):
        raise HTTPException(422, detail=["track_player_id and t are required"])
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)
