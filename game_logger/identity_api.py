"""Identity page APIs: detections boxes at a time + assign Polis roster names to track ids."""
from fastapi import APIRouter, Depends, HTTPException

from . import db, game_roles, identity, identity_health, repo, run_status

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
        "assignments": identity.list_assignments(conn, game_id),
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
        out = identity.assign(
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
    try:
        run_status.record_identity_fingerprint(game_id)           # remembers which stitched ids the tags were made on (first tag only)
        out = dict(out)
        out["warnings"] = identity_health.join_warnings(conn, game_id, out)
    except Exception:                                             # noqa: BLE001 - a failed check must never lose the tag
        out = dict(out)
        out.setdefault("warnings", [])
    return out


@router.get("/api/games/{game_id}/identity/health")
def identity_health_view(game_id: int, conn=Depends(_get_conn)):
    """Quality of the tags: saved-positions state, coverage, problems with jump-to times, tags that can be tidied."""
    _need_game(conn, game_id)
    return identity_health.health(conn, game_id)


@router.post("/api/games/{game_id}/identity/build-positions")
def identity_build_positions(game_id: int, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    from . import runner
    cur = runner.status()
    if cur and cur.get("status") == "running" and cur.get("game_id") == game_id:
        raise HTTPException(422, detail=["A pipeline run is in progress for this game. Wait for it to finish."])
    if run_status.tag_count(conn, game_id) == 0:
        raise HTTPException(422, detail=["Name at least one player first."])
    started = identity_health.build_async(game_id)
    return {"started": started, "message": "Building..." if started else "A build is already running."}


@router.post("/api/games/{game_id}/identity/tidy")
def identity_tidy(game_id: int, body: dict, conn=Depends(_get_conn)):
    """Delete tags the builder marked redundant (they change nothing). dry_run (default true) only lists them."""
    _need_game(conn, game_id)
    try:
        return identity_health.tidy(conn, game_id, dry_run=bool(body.get("dry_run", True)))
    except ValueError as e:
        raise HTTPException(422, detail=[str(e)])


@router.post("/api/games/{game_id}/identity/reset")
def identity_reset(game_id: int, conn=Depends(_get_conn)):
    """Start again: delete every tag for this game. A JSON copy is saved first; if it cannot be saved, nothing is deleted.
    Positions played / formation are kept. The saved NNNN_players.db is left alone and shows as out of date."""
    _need_game(conn, game_id)
    n = run_status.tag_count(conn, game_id)
    if n == 0:
        return {"deleted": 0}
    saved = run_status.backup_tags(conn, game_id)
    if not saved:
        raise HTTPException(422, detail=["The copy of the tags could not be saved to the game's data folder (is the drive connected?), so nothing was deleted."])
    conn.execute("DELETE FROM identity_assignments WHERE game_id = ?", (game_id,))
    conn.commit()
    run_status.clear_identity_fingerprint(game_id)
    import os
    return {"deleted": n, "saved_to": os.path.basename(saved)}


@router.get("/api/games/{game_id}/identity/roles")
def identity_roles(game_id: int, conn=Depends(_get_conn)):
    """Positions played (per half) and formation (per half) for the players named in this game."""
    _need_game(conn, game_id)
    return game_roles.get(conn, game_id)


@router.put("/api/games/{game_id}/identity/roles/{roster_player_id}")
def identity_set_role(game_id: int, roster_player_id: int, body: dict, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    try:
        out = game_roles.set_position(conn, game_id, roster_player_id, body.get("half"), body.get("position"), body.get("note"))
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)
    return {"saved": True, **out}


@router.delete("/api/games/{game_id}/identity/roles/{roster_player_id}")
def identity_remove_role(game_id: int, roster_player_id: int, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    return {"removed": True, **game_roles.remove_player(conn, game_id, roster_player_id)}


@router.put("/api/games/{game_id}/identity/formation")
def identity_set_formation(game_id: int, body: dict, conn=Depends(_get_conn)):
    _need_game(conn, game_id)
    try:
        out = game_roles.set_formation(conn, game_id, body.get("half"), body.get("formation"))
    except repo.ValidationError as e:
        raise HTTPException(422, detail=e.errors)
    return {"saved": True, **out}


@router.post("/api/games/{game_id}/identity/fingerprint/accept")
def identity_fingerprint_accept(game_id: int, conn=Depends(_get_conn)):
    """'The tags are right for the current ids': re-record the fingerprint after checking."""
    _need_game(conn, game_id)
    run_status.record_identity_fingerprint(game_id, force=True)
    return run_status.identity_fingerprint(game_id)


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
