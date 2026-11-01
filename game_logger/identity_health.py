"""Quality view for the Track Players page: what is wrong with the tags, where, and how to tidy them.

Reads the saved NNNN_players.db (built by player_positions) read-only, plus game-logger.db (tags) and, for the check made at
the moment of tagging, the detections database (read-only). Nothing here changes tags except tidy(), which only deletes rows
the builder has marked 'redundant' (verified to change nothing in the result) and saves them to a JSON file first."""
import json
import math
import os
import sqlite3
import threading
import time

from . import config, paths, run_status

JUMP_KMH = 25.0            # a join implying more than this is flagged (same rule as player_positions)
JOIN_MAX_GAP_S = 5.0
COEXIST_SEP_M = 2.0        # two boxes of "the same person" this far apart at the same moment are probably two people
MIN_COVERAGE = 60.0        # per-half coverage below this is worth a warning
LONG_GAP_S = 3.0

_build = {"running": False, "game_id": None, "error": None, "finished": None}
_lock = threading.Lock()


def _ro(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15)
    c.row_factory = sqlite3.Row
    return c


def _positions_path(game_id):
    try:
        from player_positions import players_db_path
        p = players_db_path(game_id)
        return p if os.path.isfile(p) else None
    except Exception:                                           # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------ building the saved layer

def build_state():
    with _lock:
        return dict(_build)


def build_async(game_id):
    with _lock:
        if _build["running"]:
            return False
        _build.update(running=True, game_id=game_id, error=None, finished=None)

    def work():
        err = None
        try:
            from player_positions import build_player_positions
            build_player_positions(game_id, verbose=False)
        except Exception as e:                                  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        with _lock:
            _build.update(running=False, error=err, finished=run_status.now())
    threading.Thread(target=work, daemon=True).start()
    return True


# ------------------------------------------------------------------------------------------ health

def _gaps(pconn, vfps):
    """Longest gaps per person from the saved frames: [(rid, name, t_start, length_s)]."""
    out = []
    rows = pconn.execute("SELECT roster_player_id, name, t_video_s FROM player_frames ORDER BY roster_player_id, t_video_s").fetchall()
    last = {}
    for r in rows:
        rid, t = r["roster_player_id"], r["t_video_s"]
        if rid in last and t - last[rid] >= LONG_GAP_S:
            out.append((rid, r["name"], last[rid], t - last[rid]))
        last[rid] = t
    return out


def health(conn, game_id):
    fp = run_status.identity_fingerprint(game_id)
    res = {"fingerprint": fp, "build": build_state() if build_state().get("game_id") == game_id else {"running": False},
           "positions": {"exists": False}, "players": [], "issues": [], "tidy_candidates": 0, "tags": run_status.tag_count(conn, game_id)}
    issues = res["issues"]
    if fp.get("matches") is False:
        issues.append({"severity": "bad", "kind": "ids_changed", "text": fp["message"]})
    path = _positions_path(game_id)
    if not path:
        if res["tags"]:
            issues.append({"severity": "warn", "kind": "no_positions", "text": "Player positions has not been built for this game yet. Build it to see coverage and problems."})
        return res
    try:
        from player_positions import check_current
        chk = check_current(game_id)
    except Exception as e:                                      # noqa: BLE001
        chk = {"current": False, "reasons": [str(e)], "built_at": None}
    res["positions"] = {"exists": True, "current": chk["current"], "reasons": chk["reasons"], "built_at": chk["built_at"]}
    if not chk["current"]:
        issues.append({"severity": "warn", "kind": "stale",
                       "text": "The saved player positions are out of date (" + "; ".join(chk["reasons"]) + "). The details below are from " +
                               f"{chk['built_at']}; build again to refresh."})
    p = _ro(path)
    try:
        info = dict(p.execute("SELECT key, value FROM build_info").fetchall())
        vfps = float(info.get("video_fps") or 24)
        q = p.execute("SELECT * FROM quality ORDER BY name, half").fetchall()
        by = {}
        for r in q:
            d = by.setdefault(r["roster_player_id"], {"roster_player_id": r["roster_player_id"], "name": r["name"], "halves": {}, "issues": 0})
            d["halves"][str(r["half"])] = {"coverage_pct": r["coverage_pct"], "gaps_over_2s": r["gaps_over_2s"],
                                           "longest_gap_s": r["longest_gap_s"], "tracks_used": r["tracks_used"]}
            if r["coverage_pct"] is not None and r["coverage_pct"] < MIN_COVERAGE:
                d["issues"] += 1
                issues.append({"severity": "warn", "kind": "coverage", "player": r["name"], "roster_player_id": r["roster_player_id"],
                               "text": f"{r['name']}: only {r['coverage_pct']:.0f}% of the {'first' if r['half'] == 1 else 'second'} half is followed."})
        res["players"] = list(by.values())
        for rid, name, t0, length in sorted(_gaps(p, vfps), key=lambda g: -g[3])[:12]:
            issues.append({"severity": "info", "kind": "gap", "player": name, "roster_player_id": rid, "seek_t": round(t0 + 0.2, 2),
                           "text": f"{name} is lost for {length:.0f} s from {_mmss(t0)}."})
        for r in p.execute("SELECT * FROM join_checks WHERE flagged = 1 ORDER BY from_frame").fetchall():
            issues.append({"severity": "bad", "kind": "join", "player": r["name"], "roster_player_id": r["roster_player_id"],
                           "seek_t": round(r["to_frame"] / vfps, 2),
                           "text": f"{r['name']}: track #{r['from_track']} to #{r['to_track']} jumps {r['dist_m']:.1f} m in {r['gap_s']:.1f} s ({r['implied_kmh']:.0f} km/h): check it: the tags may be swapped between two players, or these are two different people."})
        for r in p.execute("SELECT * FROM tag_collisions ORDER BY first_frame").fetchall():
            if r["kind"] == "track_two_boxes":
                text = (f"Track #{_trk(p, r['newer_id'])} holds two boxes for {r['seconds']:.0f} s about {r['median_sep_m']:.0f} m apart "
                        "(a goalkeeper box is probably merged into this player). One box is kept.")
                sev = "warn" if (r["seconds"] or 0) >= 5 else "info"
            else:
                text = f"Tags {r['older_id']} and {r['newer_id']} disagree for {r['seconds']:.0f} s ({r['kind'].replace('_', ' ')}): {r['action']}."
                sev = "info" if (r["seconds"] or 0) < 3 else "warn"
            issues.append({"severity": sev, "kind": "collision", "seek_t": round((r["first_frame"] or 0) / vfps, 2), "text": text})
        res["tidy_candidates"] = p.execute("SELECT COUNT(*) FROM identity_segments WHERE status = 'redundant'").fetchone()[0]
    finally:
        p.close()
    order = {"bad": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda i: (order.get(i["severity"], 3), i.get("seek_t") or 0))
    return res


def _trk(p, tag_id):
    r = p.execute("SELECT track_player_id FROM identity_segments WHERE id = ?", (tag_id,)).fetchone()
    return r[0] if r else "?"


def _mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ------------------------------------------------------------------------------------------ tidy

def tidy(conn, game_id, dry_run=True):
    """Delete the tags the builder marked 'redundant' (they change nothing in the result). Needs a current saved file."""
    path = _positions_path(game_id)
    if not path:
        raise ValueError("Build Player positions first: it is what says which tags are redundant.")
    from player_positions import check_current
    chk = check_current(game_id)
    if not chk["current"]:
        raise ValueError("The saved player positions are out of date (" + "; ".join(chk["reasons"]) + "). Build them again, then tidy.")
    p = _ro(path)
    try:
        segs = p.execute("SELECT id, name, track_player_id, from_frame, to_frame FROM identity_segments WHERE status = 'redundant' ORDER BY id").fetchall()
    finally:
        p.close()
    ids = [s["id"] for s in segs]
    rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM identity_assignments WHERE game_id = ? AND id IN ({','.join('?' * len(ids)) or 'NULL'})", [game_id] + ids)]
    out = {"count": len(rows), "dry_run": dry_run,
           "rows": [{"id": r["id"], "track_player_id": r["track_player_id"], "from_frame": r["from_frame"], "to_frame": r["to_frame"],
                     "name": next((s["name"] for s in segs if s["id"] == r["id"]), None)} for r in rows]}
    if dry_run or not rows:
        return out
    folder = paths.game_folder(game_id)
    saved = os.path.join(folder, "data", f"{paths.prefix(game_id)}_tags_tidied_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(saved, "w", encoding="utf-8") as f:
        json.dump({"game_id": game_id, "saved_at": run_status.now(), "deleted_assignments": rows}, f, indent=1)
    conn.execute(f"DELETE FROM identity_assignments WHERE game_id = ? AND id IN ({','.join('?' * len(ids))})", [game_id] + ids)
    conn.commit()
    out["saved_to"] = os.path.basename(saved)
    return out


# ------------------------------------------------------------------------------------------ check at the moment of tagging

def _pos(dconn, track, frame, tol=0):
    return dconn.execute(
        "SELECT frame_id, x, y, class_name FROM detected_objects WHERE player_id = ? AND frame_id BETWEEN ? AND ? "
        "AND class_name IN ('player','goalkeeper') AND x IS NOT NULL ORDER BY ABS(frame_id - ?) LIMIT 1",
        (track, frame - tol, frame + tol, frame)).fetchone()


def join_warnings(conn, game_id, assignment):
    """Plain-language warnings about a tag just made. [] when it looks fine."""
    from . import identity
    warns = []
    try:
        dbp = paths.detections_db(game_id)
        meta = identity.detections_meta(game_id)
        vfps = meta["video_fps"]
        dconn = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True, timeout=15)
        dconn.row_factory = sqlite3.Row
    except Exception:                                           # noqa: BLE001
        return warns
    try:
        f = int(assignment["from_frame"])
        trk = int(assignment["track_player_id"])
        rid = int(assignment["roster_player_id"])
        name = assignment.get("name") or "This player"
        n = dconn.execute("SELECT COUNT(*) FROM detected_objects WHERE player_id = ? AND frame_id = ? AND class_name IN ('player','goalkeeper')", (trk, f)).fetchone()[0]
        if n > 1:
            warns.append(f"Track #{trk} has {n} boxes on this frame (merged ids, usually a goalkeeper box inside a player's track). The positions step keeps one box per person.")
        prior = conn.execute(
            "SELECT track_player_id, to_frame FROM identity_assignments WHERE game_id = ? AND roster_player_id = ? AND id != ? "
            "AND from_frame < ? AND track_player_id != ? ORDER BY from_frame DESC, id DESC LIMIT 1",
            (game_id, rid, assignment["id"], f, trk)).fetchone()
        if prior:
            window = int(vfps * JOIN_MAX_GAP_S)
            last = dconn.execute(
                "SELECT frame_id, x, y FROM detected_objects WHERE player_id = ? AND frame_id BETWEEN ? AND ? AND x IS NOT NULL "
                "AND class_name IN ('player','goalkeeper') ORDER BY frame_id DESC LIMIT 1", (prior["track_player_id"], f - window, f)).fetchone()
            new = _pos(dconn, trk, f, tol=4)
            if last and new:
                gap = max((new["frame_id"] - last["frame_id"]) / vfps, 0.0)
                dist = math.hypot(new["x"] - last["x"], new["y"] - last["y"])
                if gap == 0 and dist > COEXIST_SEP_M:
                    warns.append(f"{name} was still on track #{prior['track_player_id']} at this moment, {dist:.1f} m away: these may be two different people.")
                elif gap > 0.04 and dist / gap * 3.6 > JUMP_KMH and dist > COEXIST_SEP_M:
                    warns.append(f"Check this join: {name} would have moved {dist:.1f} m in {gap:.1f} s ({dist / gap * 3.6:.0f} km/h) from track #{prior['track_player_id']}.")
    except sqlite3.DatabaseError:
        pass
    finally:
        dconn.close()
    return warns
