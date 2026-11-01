"""Named-player identity: read detections boxes at a time, and link stitched track ids to roster players."""
import os
import sqlite3

from . import paths, repo
from .db import now_iso
from .repo import ValidationError


def _detections_conn(game_id):
    path = paths.detections_db(game_id)
    if not path or not os.path.isfile(path):
        raise ValidationError(
            "There is no detections database for this game yet. Run Detections (and Transformations) first."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def detections_meta(game_id):
    """video_fps / detection_fps from the detections DB (needed to map video time ↔ frame_id)."""
    conn = _detections_conn(game_id)
    try:
        row = conn.execute("SELECT video_fps, detection_fps FROM game LIMIT 1").fetchone()
        if not row:
            raise ValidationError("The detections database has no game row")
        return {
            "video_fps": float(row["video_fps"] or 24),
            "detection_fps": float(row["detection_fps"] or 12),
            "db_path": paths.detections_db(game_id),
        }
    finally:
        conn.close()


def time_to_frame(t, video_fps):
    return max(0, int(round(float(t) * float(video_fps))))


def nearest_detection_frame(dconn, frame_id, window=60):
    """Snap to the closest frame that has player/goalkeeper boxes (detections are every other video frame)."""
    row = dconn.execute(
        """
        SELECT frame_id FROM detected_objects
        WHERE class_name IN ('player', 'goalkeeper')
          AND frame_id BETWEEN ? AND ?
        ORDER BY ABS(frame_id - ?) ASC, frame_id ASC
        LIMIT 1
        """,
        (max(0, frame_id - window), frame_id + window, frame_id),
    ).fetchone()
    return int(row["frame_id"]) if row else None


def polis_team_ids(conn, game):
    """Team ids on this game whose name looks like Polis (the club we name players for)."""
    out = []
    for tid in (game.get("team_a_id"), game.get("team_b_id")):
        if not tid:
            continue
        t = repo.get_team(conn, tid)
        if t and "polis" in (t.get("name") or "").lower():
            out.append(tid)
    return out


def roster_for_game(conn, game):
    """Active Polis players for this game's format (mens / womens / mixed)."""
    tids = polis_team_ids(conn, game)
    if not tids:
        return []
    fmt = (game.get("game_format") or "").strip().lower()
    players = []
    for tid in tids:
        for p in repo.list_players(conn, tid, include_inactive=False):
            sq = p.get("squad") or "mens"
            if fmt == "mens" and sq not in ("mens", "both"):
                continue
            if fmt == "womens" and sq not in ("womens", "both"):
                continue
            # mixed / n/a / blank: everyone
            p = dict(p)
            team = repo.get_team(conn, tid)
            p["team_name"] = team["name"] if team else None
            players.append(p)
    players.sort(key=lambda p: ((p.get("squad") or ""), (p.get("name") or "").lower()))
    return players


def _assignment_dict(row, name=None, squad=None):
    d = dict(row)
    if name is not None:
        d["name"] = name
    if squad is not None:
        d["squad"] = squad
    return d


def list_assignments(conn, game_id):
    rows = conn.execute(
        """
        SELECT a.*, p.name, p.squad, p.team_id
        FROM identity_assignments a
        JOIN players p ON p.player_id = a.roster_player_id
        WHERE a.game_id = ?
        ORDER BY a.from_frame ASC, a.id ASC
        """,
        (game_id,),
    ).fetchall()
    return [_assignment_dict(r, r["name"], r["squad"]) for r in rows]


def _name_at_frame(conn, game_id, track_player_id, frame_id):
    row = conn.execute(
        """
        SELECT a.*, p.name, p.squad
        FROM identity_assignments a
        JOIN players p ON p.player_id = a.roster_player_id
        WHERE a.game_id = ? AND a.track_player_id = ?
          AND a.from_frame <= ?
          AND (a.to_frame IS NULL OR a.to_frame >= ?)
        ORDER BY a.from_frame DESC, a.id DESC
        LIMIT 1
        """,
        (game_id, track_player_id, frame_id, frame_id),
    ).fetchone()
    return _assignment_dict(row, row["name"], row["squad"]) if row else None


def boxes_at_time(conn, game_id, t):
    """Player/goalkeeper boxes nearest to video time t, with any identity name attached."""
    meta = detections_meta(game_id)
    video_fps = meta["video_fps"]
    want = time_to_frame(t, video_fps)
    dconn = _detections_conn(game_id)
    try:
        frame_id = nearest_detection_frame(dconn, want)
        if frame_id is None:
            return {
                "t": float(t),
                "frame_id": None,
                "video_fps": video_fps,
                "detection_fps": meta["detection_fps"],
                "boxes": [],
            }
        rows = dconn.execute(
            """
            SELECT frame_id, tracker_id, player_id, team, team_color, class_name,
                   xmin, ymin, xmax, ymax, confidence
            FROM detected_objects
            WHERE frame_id = ? AND class_name IN ('player', 'goalkeeper')
            ORDER BY ymin ASC, xmin ASC
            """,
            (frame_id,),
        ).fetchall()
    finally:
        dconn.close()

    boxes = []
    for r in rows:
        track_id = r["player_id"]
        if track_id is None:
            track_id = int(r["tracker_id"]) if r["tracker_id"] is not None else None
        else:
            track_id = int(track_id)
        asg = _name_at_frame(conn, game_id, track_id, frame_id) if track_id is not None else None
        team = r["team"] or ""
        boxes.append({
            "track_player_id": track_id,
            "tracker_id": int(r["tracker_id"]) if r["tracker_id"] is not None else None,
            "team": team,
            "team_color": r["team_color"],
            "class_name": r["class_name"],
            "xmin": float(r["xmin"]),
            "ymin": float(r["ymin"]),
            "xmax": float(r["xmax"]),
            "ymax": float(r["ymax"]),
            "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
            "is_polis": "polis" in team.lower(),
            "name": asg["name"] if asg else None,
            "roster_player_id": asg["roster_player_id"] if asg else None,
            "assignment_id": asg["id"] if asg else None,
        })
    # A stitched id that has two boxes on one frame (usually a goalkeeper box merged into a player's track): mark both so the page can say so.
    counts = {}
    for b in boxes:
        if b["track_player_id"] is not None:
            counts[b["track_player_id"]] = counts.get(b["track_player_id"], 0) + 1
    for b in boxes:
        b["dup_track"] = counts.get(b["track_player_id"], 0) > 1
    # One person → one box on this frame: if a name appears twice, keep it on the stronger box only.
    conflicts = _dedupe_names_on_frame(boxes)
    return {
        "t": float(t),
        "frame_id": frame_id,
        "video_fps": video_fps,
        "detection_fps": meta["detection_fps"],
        "boxes": boxes,
        "identity_conflicts": conflicts,
    }


def _dedupe_names_on_frame(boxes):
    """A person can only be in one place: keep each roster name on at most one box.

    Winner = highest confidence, then largest box. Others lose the name for this frame only
    (assignments in the DB are unchanged — assign() is what repairs overlapping ranges).
    Returns a short list of conflicts for the UI.
    """
    by_roster = {}
    for i, b in enumerate(boxes):
        rid = b.get("roster_player_id")
        if rid is None:
            continue
        by_roster.setdefault(int(rid), []).append(i)
    conflicts = []
    for rid, idxs in by_roster.items():
        if len(idxs) < 2:
            continue
        def score(i):
            b = boxes[i]
            conf = b.get("confidence")
            conf = float(conf) if conf is not None else -1.0
            area = max(0.0, (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"]))
            return (conf, area)
        winner = max(idxs, key=score)
        name = boxes[winner].get("name")
        losers = []
        for i in idxs:
            if i == winner:
                continue
            losers.append(boxes[i].get("track_player_id"))
            boxes[i]["name"] = None
            boxes[i]["roster_player_id"] = None
            boxes[i]["assignment_id"] = None
            boxes[i]["identity_conflict"] = True
        boxes[winner]["identity_conflict"] = True
        conflicts.append({
            "roster_player_id": rid,
            "name": name,
            "kept_track_player_id": boxes[winner].get("track_player_id"),
            "cleared_track_player_ids": losers,
        })
    return conflicts


def _end_other_links_for_person(conn, game_id, roster_player_id, track_player_id, frame_id):
    """End every other assignment for this person that still covers frame_id.

    A player cannot be two track ids at once — truncate those ranges so they stop
    just before this frame. Cheap: one indexed UPDATE on identity_assignments.
    """
    end_at = max(0, int(frame_id) - 1)
    cur = conn.execute(
        """
        UPDATE identity_assignments
        SET to_frame = ?
        WHERE game_id = ?
          AND roster_player_id = ?
          AND track_player_id != ?
          AND from_frame <= ?
          AND (to_frame IS NULL OR to_frame >= ?)
        """,
        (end_at, game_id, roster_player_id, track_player_id, frame_id, frame_id),
    )
    return cur.rowcount


def assign(conn, game, roster_player_id, track_player_id, t):
    """Link a roster player to a stitched track id from time t onward (closes any open previous links)."""
    game_id = game["game_id"]
    player = repo.get_player(conn, roster_player_id)
    if not player or not player.get("is_active"):
        raise ValidationError("That player is not on the roster (or is inactive)")
    polis = set(polis_team_ids(conn, game))
    if polis and player["team_id"] not in polis:
        raise ValidationError("That player is not on the Polis side of this game")
    try:
        track_player_id = int(track_player_id)
    except (TypeError, ValueError):
        raise ValidationError("track_player_id must be a whole number")
    meta = detections_meta(game_id)
    frame_id = time_to_frame(t, meta["video_fps"])
    dconn = _detections_conn(game_id)
    try:
        snap = nearest_detection_frame(dconn, frame_id)
    finally:
        dconn.close()
    if snap is not None:
        frame_id = snap

    # Close open assignment for this person (they're now on a new track after a break).
    conn.execute(
        """
        UPDATE identity_assignments
        SET to_frame = ?
        WHERE game_id = ? AND roster_player_id = ? AND to_frame IS NULL AND from_frame < ?
        """,
        (max(0, frame_id - 1), game_id, roster_player_id, frame_id),
    )
    # Also end any *closed* ranges for this person that still cover this frame on another track
    # (legacy overlap / bad ranges) — one person cannot be two places at once.
    ended = _end_other_links_for_person(conn, game_id, roster_player_id, track_player_id, frame_id)
    # Close open assignment that already claimed this track for someone else.
    conn.execute(
        """
        UPDATE identity_assignments
        SET to_frame = ?
        WHERE game_id = ? AND track_player_id = ? AND to_frame IS NULL AND from_frame < ?
        """,
        (max(0, frame_id - 1), game_id, track_player_id, frame_id),
    )
    # Same person + same track already open covering this frame: nothing to do.
    existing = conn.execute(
        """
        SELECT id FROM identity_assignments
        WHERE game_id = ? AND roster_player_id = ? AND track_player_id = ?
          AND from_frame <= ? AND (to_frame IS NULL OR to_frame >= ?)
        LIMIT 1
        """,
        (game_id, roster_player_id, track_player_id, frame_id, frame_id),
    ).fetchone()
    if existing:
        conn.commit()
        out = get_assignment(conn, existing["id"])
        if ended:
            out = dict(out)
            out["ended_overlapping"] = ended
        return out

    cur = conn.execute(
        """
        INSERT INTO identity_assignments(game_id, roster_player_id, track_player_id, from_frame, to_frame, skip_until_frame, created_at)
        VALUES (?,?,?,?,NULL,NULL,?)
        """,
        (game_id, roster_player_id, track_player_id, frame_id, now_iso()),
    )
    # Tidy as we go: the new open link already covers everything from this frame on, so any later link for the same person on the
    # same track says nothing new.
    conn.execute(
        "DELETE FROM identity_assignments WHERE game_id = ? AND roster_player_id = ? AND track_player_id = ? AND id != ? AND from_frame >= ?",
        (game_id, roster_player_id, track_player_id, cur.lastrowid, frame_id),
    )
    conn.commit()
    out = get_assignment(conn, cur.lastrowid)
    if ended:
        out = dict(out)
        out["ended_overlapping"] = ended
    return out


def get_assignment(conn, assignment_id):
    row = conn.execute(
        """
        SELECT a.*, p.name, p.squad, p.team_id
        FROM identity_assignments a
        JOIN players p ON p.player_id = a.roster_player_id
        WHERE a.id = ?
        """,
        (assignment_id,),
    ).fetchone()
    return _assignment_dict(row, row["name"], row["squad"]) if row else None


def clear_assignment(conn, game_id, assignment_id):
    row = conn.execute(
        "SELECT id FROM identity_assignments WHERE id = ? AND game_id = ?",
        (assignment_id, game_id),
    ).fetchone()
    if not row:
        raise ValidationError("That assignment does not exist")
    conn.execute("DELETE FROM identity_assignments WHERE id = ?", (assignment_id,))
    conn.commit()


def clear_name_at_time(conn, game_id, track_player_id, t):
    """Remove the name covering this track at time t (delete that assignment row)."""
    meta = detections_meta(game_id)
    frame_id = time_to_frame(t, meta["video_fps"])
    asg = _name_at_frame(conn, game_id, int(track_player_id), frame_id)
    if not asg:
        raise ValidationError("That track has no name at this time")
    clear_assignment(conn, game_id, asg["id"])
    return {"cleared": asg["id"]}


def _track_frames_from(dconn, track_player_id, from_frame):
    """Ordered video frame_ids where this stitched player_id (fallback tracker_id) appears."""
    rows = dconn.execute(
        """
        SELECT DISTINCT frame_id
        FROM detected_objects
        WHERE class_name IN ('player', 'goalkeeper')
          AND frame_id >= ?
          AND COALESCE(player_id, CAST(tracker_id AS INTEGER)) = ?
        ORDER BY frame_id ASC
        """,
        (int(from_frame), int(track_player_id)),
    ).fetchall()
    return [int(r["frame_id"]) for r in rows]


def track_breaks_for_game(conn, game_id):
    """Where each *open* named track is lost — so the Identity page can jump there for re-assign.

    A break is:
      - an internal gap in that track_id after the assignment started, or
      - the track ending while the match detections continue.

    Assignments may set skip_until_frame so an earlier break the user skipped is ignored.
    """
    meta = detections_meta(game_id)
    video_fps = meta["video_fps"]
    # detections are usually every 2 video frames; treat ~0.75s without the id as "lost"
    min_gap = max(8, int(round(video_fps * 0.75)))
    end_slack = max(min_gap, int(round(video_fps * 1.0)))

    dconn = _detections_conn(game_id)
    try:
        last_row = dconn.execute(
            """
            SELECT MAX(frame_id) AS last_frame
            FROM detected_objects
            WHERE class_name IN ('player', 'goalkeeper')
            """
        ).fetchone()
        last_game_frame = int(last_row["last_frame"] or 0)
        breaks = []
        for a in list_assignments(conn, game_id):
            if a.get("to_frame") is not None:
                continue  # already closed — user moved on or cleared
            track_id = int(a["track_player_id"])
            from_frame = int(a["from_frame"])
            skip_until = a.get("skip_until_frame")
            skip_until = int(skip_until) if skip_until is not None else None
            frames = _track_frames_from(dconn, track_id, from_frame)
            lost_at = None
            resumes_at = None
            kind = None
            if not frames:
                lost_at = from_frame
                kind = "missing"
            else:
                for prev, nxt in zip(frames, frames[1:]):
                    if nxt - prev > min_gap:
                        # ignore gaps the user already skipped past
                        if skip_until is not None and prev < skip_until:
                            continue
                        lost_at = prev
                        resumes_at = nxt
                        kind = "gap"
                        break
                if lost_at is None:
                    last = frames[-1]
                    if skip_until is not None and last < skip_until:
                        pass  # still inside skipped region with no further gap
                    elif last_game_frame and last < last_game_frame - end_slack:
                        if skip_until is None or last >= skip_until:
                            lost_at = last
                            kind = "end"
            if lost_at is None:
                continue
            if skip_until is not None and lost_at < skip_until:
                continue
            lost_t = lost_at / video_fps if video_fps else 0.0
            # seek just after the loss so the old box is gone and new candidates show
            seek_t = lost_t + (0.35 if kind != "missing" else 0.0)
            breaks.append({
                "assignment_id": a["id"],
                "roster_player_id": a["roster_player_id"],
                "name": a.get("name"),
                "squad": a.get("squad"),
                "track_player_id": track_id,
                "from_frame": from_frame,
                "lost_at_frame": lost_at,
                "lost_at_t": round(lost_t, 3),
                "seek_t": round(seek_t, 3),
                "resumes_at_frame": resumes_at,
                "kind": kind,
                "last_game_frame": last_game_frame,
            })
        breaks.sort(key=lambda b: (b["lost_at_frame"], b["name"] or ""))
        return {
            "video_fps": video_fps,
            "last_game_frame": last_game_frame,
            "min_gap_frames": min_gap,
            "breaks": breaks,
        }
    finally:
        dconn.close()


def skip_break(conn, game_id, roster_player_id):
    """Skip the current break for a named player.

    - gap (same track resumes later): keep the name on that track; ignore this gap.
    - end / missing (player not detected / track dead): close the segment so you can
      move on; re-assign when you see them again.
    """
    info = track_breaks_for_game(conn, game_id)
    br = next((b for b in info["breaks"] if b["roster_player_id"] == int(roster_player_id)), None)
    if not br:
        raise ValidationError("That player has no open track break to skip")
    asg = get_assignment(conn, br["assignment_id"])
    if not asg or asg["game_id"] != game_id:
        raise ValidationError("That assignment does not exist")

    if br["kind"] == "gap" and br.get("resumes_at_frame") is not None:
        # Keep following the same track_id; don't ask about this gap again.
        conn.execute(
            "UPDATE identity_assignments SET skip_until_frame = ? WHERE id = ?",
            (int(br["resumes_at_frame"]), br["assignment_id"]),
        )
        conn.commit()
        return {
            "action": "ignored_gap",
            "assignment_id": br["assignment_id"],
            "name": br["name"],
            "skip_until_frame": int(br["resumes_at_frame"]),
            "message": f"Skipped gap for {br['name']} — still on track #{br['track_player_id']}",
        }

    # Track ended or never seen: close the open segment at the loss.
    lost = int(br["lost_at_frame"])
    from_f = int(asg["from_frame"])
    to_frame = max(from_f, lost)
    conn.execute(
        "UPDATE identity_assignments SET to_frame = ?, skip_until_frame = NULL WHERE id = ?",
        (to_frame, br["assignment_id"]),
    )
    conn.commit()
    return {
        "action": "closed",
        "assignment_id": br["assignment_id"],
        "name": br["name"],
        "to_frame": to_frame,
        "message": (
            f"Skipped break for {br['name']} — name ends at frame {to_frame}. "
            "Click them again later when they reappear."
        ),
    }
