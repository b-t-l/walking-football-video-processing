"""All reading and writing of games, teams and calibrations. The web app and the importer both go through here,
so the rules (validation, one active calibration per game, time formats) live in one place."""
import ast
import json
import os
import re

from . import config, paths
from .db import now_iso
from .timeutil import parse_time, format_mmss


class ValidationError(Exception):
    def __init__(self, errors):
        self.errors = errors if isinstance(errors, list) else [errors]
        super().__init__("; ".join(self.errors))


# --------------------------------------------------------------------------------------------- settings

def get_setting(conn, key, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn, key, value):
    conn.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                 (key, str(value)))
    conn.commit()


def default_running_speed(conn):
    try:
        return float(get_setting(conn, "running_speed_km_h", config.DEFAULT_RUNNING_SPEED_KM_H))
    except ValueError:
        return float(config.DEFAULT_RUNNING_SPEED_KM_H)


# --------------------------------------------------------------------------------------------- teams

def _team_dict(row):
    d = dict(row)
    d["kit_colours"] = json.loads(d["kit_colours"] or "[]")
    d["is_active"] = bool(d["is_active"])
    return d


def list_teams(conn, include_inactive=True):
    sql = "SELECT * FROM teams" + ("" if include_inactive else " WHERE is_active = 1") + " ORDER BY name COLLATE NOCASE"
    teams = [_team_dict(r) for r in conn.execute(sql)]
    counts = {r["tid"]: r["n"] for r in conn.execute(
        "SELECT tid, COUNT(*) n FROM (SELECT team_a_id tid FROM games UNION ALL SELECT team_b_id FROM games) "
        "WHERE tid IS NOT NULL GROUP BY tid")}
    for t in teams:
        t["games"] = counts.get(t["team_id"], 0)
    return teams


def get_team(conn, team_id):
    row = conn.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,)).fetchone()
    return _team_dict(row) if row else None


def team_id_by_name(conn, name):
    if not name or not str(name).strip():
        return None
    row = conn.execute("SELECT team_id FROM teams WHERE name = ? COLLATE NOCASE", (str(name).strip(),)).fetchone()
    return row["team_id"] if row else None


def _clean_kit_colours(colours):
    out, errors = [], []
    for i, c in enumerate(colours or [], 1):
        try:
            rgb = [int(v) for v in c["rgb"]]
            tol = [int(v) for v in c.get("tolerance", (10, 50, 50))]
            if len(rgb) != 3 or len(tol) != 3 or any(not 0 <= v <= 255 for v in rgb) or any(v < 0 for v in tol):
                raise ValueError
            out.append({"rgb": rgb, "tolerance": tol})
        except (KeyError, TypeError, ValueError):
            errors.append(f"Kit colour {i} is not valid (needs rgb = 3 numbers 0-255 and tolerance = 3 numbers)")
    if errors:
        raise ValidationError(errors)
    return out


def save_team(conn, data, team_id=None):
    name = (data.get("name") or "").strip()
    if not name:
        raise ValidationError("A team needs a name")
    colours = _clean_kit_colours(data.get("kit_colours"))
    clash = conn.execute("SELECT team_id FROM teams WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    if clash and clash["team_id"] != team_id:
        raise ValidationError(f"There is already a team called '{name}'")
    active = 1 if data.get("is_active", True) else 0
    if team_id is None:
        cur = conn.execute("INSERT INTO teams(name, kit_colours, is_active, notes) VALUES (?,?,?,?)",
                           (name, json.dumps(colours), active, data.get("notes")))
        team_id = cur.lastrowid
    else:
        if not get_team(conn, team_id):
            raise ValidationError("That team does not exist")
        conn.execute("UPDATE teams SET name=?, kit_colours=?, is_active=?, notes=? WHERE team_id=?",
                     (name, json.dumps(colours), active, data.get("notes"), team_id))
    conn.commit()
    return get_team(conn, team_id)


def delete_team(conn, team_id, clear_from_games=False):
    """Delete a team. If games use it, that is refused unless clear_from_games is set, in which case the team is
    taken off those games (they keep everything else: scores, title, videos) and show 'Teams missing' until re-picked."""
    if not get_team(conn, team_id):
        raise ValidationError("That team does not exist")
    used = conn.execute("SELECT COUNT(*) FROM games WHERE team_a_id=? OR team_b_id=?", (team_id, team_id)).fetchone()[0]
    if used and not clear_from_games:
        raise ValidationError(f"This team is used in {used} game(s) - confirm to remove it from those games, or make it inactive instead")
    if used:
        conn.execute("UPDATE games SET team_a_id = NULL WHERE team_a_id = ?", (team_id,))
        conn.execute("UPDATE games SET team_b_id = NULL WHERE team_b_id = ?", (team_id,))
    conn.execute("DELETE FROM teams WHERE team_id = ?", (team_id,))
    conn.commit()


# --------------------------------------------------------------------------------------------- calibrations

def parse_calibrator_json(text):
    """Validate a Pitch Calibrator export. Returns (vertices, goal_left, goal_right, normalised_text)."""
    if isinstance(text, dict):
        data = text
        text = json.dumps(data, indent=2)
    else:
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as e:
            raise ValidationError(f"That is not valid JSON ({e}). Paste the whole export from the Pitch Calibrator.")
    if not isinstance(data, dict) or "pitch_vertices" not in data:
        raise ValidationError("This does not look like a Pitch Calibrator export (no 'pitch_vertices' found)")
    errors, vertices = [], {}
    for name in config.PITCH_POINT_NAMES:
        p = data["pitch_vertices"].get(name)
        try:
            vertices[name] = [int(round(float(p["x"]))), int(round(float(p["y"])))]
        except (TypeError, KeyError, ValueError):
            errors.append(f"pitch_vertices.{name} is missing or not a point")
    if "image_size" not in data or "pitch_dimensions_m" not in data:
        errors.append("The export is missing image_size or pitch_dimensions_m (the pipeline needs both)")
    if errors:
        raise ValidationError(errors)

    def goal(key):
        pts = data.get(key) or []
        if len(pts) != 4:
            return None
        try:
            return [[int(round(float(p["x"]))), int(round(float(p["y"])))] for p in pts]
        except (TypeError, KeyError, ValueError):
            return None
    return vertices, goal("goal_posts_left_vertices"), goal("goal_posts_right_vertices"), text.strip()


def parse_vertex_text(value):
    """'146,653' -> [146, 653]; raises ValueError when it is not two whole numbers."""
    parts = str(value).split(",")
    if len(parts) != 2:
        raise ValueError(f"'{value}' is not a point like 146,653")
    return [int(float(parts[0])), int(float(parts[1]))]


def parse_goal_text(value):
    """'[(739,598),(343,404),(490,266),(823,481)]' (brackets optional) -> [[739,598],...]; None when empty.
    Raises ValueError when it is not four complete points (e.g. '[(707,607),(),(),()]')."""
    if value is None or (isinstance(value, float) and value != value) or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        pts = ast.literal_eval(text if text.startswith("[") else f"[{text}]")
        if len(pts) != 4 or any(len(p) != 2 for p in pts):
            raise ValueError
        return [[int(p[0]), int(p[1])] for p in pts]
    except (ValueError, SyntaxError, TypeError):
        raise ValueError("goal posts need 4 complete points")


def _cal_dict(row, full=False):
    d = {"id": row["id"], "game_id": row["game_id"], "created_at": row["created_at"], "source": row["source"],
         "label": row["label"], "is_active": bool(row["is_active"]), "notes": row["notes"],
         "vertices": json.loads(row["vertices_json"]),
         "has_goal_posts": bool(row["goal_left_json"] or row["goal_right_json"]),
         "has_calibrator_json": bool(row["calibrator_json"])}
    if full:
        d["goal_left"] = json.loads(row["goal_left_json"]) if row["goal_left_json"] else None
        d["goal_right"] = json.loads(row["goal_right_json"]) if row["goal_right_json"] else None
        d["calibrator_json"] = row["calibrator_json"]
    return d


def list_calibrations(conn, game_id):
    rows = conn.execute("SELECT * FROM pitch_calibrations WHERE game_id = ? ORDER BY id DESC", (game_id,))
    return [_cal_dict(r) for r in rows]


def get_calibration(conn, cal_id):
    row = conn.execute("SELECT * FROM pitch_calibrations WHERE id = ?", (cal_id,)).fetchone()
    return _cal_dict(row, full=True) if row else None


def active_calibration(conn, game_id):
    row = conn.execute("SELECT * FROM pitch_calibrations WHERE game_id = ? AND is_active = 1", (game_id,)).fetchone()
    return _cal_dict(row, full=True) if row else None


def add_calibration(conn, game_id, source, vertices, goal_left=None, goal_right=None, calibrator_json=None,
                    label=None, notes=None, make_active=True, created_at=None):
    if not conn.execute("SELECT 1 FROM games WHERE game_id = ?", (game_id,)).fetchone():
        raise ValidationError(f"Game {game_id} does not exist")
    cur = conn.execute(
        "INSERT INTO pitch_calibrations(game_id, created_at, source, label, is_active, vertices_json, goal_left_json,"
        " goal_right_json, calibrator_json, notes) VALUES (?,?,?,?,0,?,?,?,?,?)",
        (game_id, created_at or now_iso(), source, label, json.dumps(vertices),
         json.dumps(goal_left) if goal_left else None, json.dumps(goal_right) if goal_right else None,
         calibrator_json, notes))
    cal_id = cur.lastrowid
    if make_active:
        _activate(conn, game_id, cal_id)
    conn.commit()
    return get_calibration(conn, cal_id)


def add_calibration_from_json(conn, game_id, text, label=None, notes=None, make_active=True, source="calibrator"):
    vertices, gl, gr, norm = parse_calibrator_json(text)
    return add_calibration(conn, game_id, source, vertices, gl, gr, norm, label, notes, make_active)


def _activate(conn, game_id, cal_id):
    conn.execute("UPDATE pitch_calibrations SET is_active = 0 WHERE game_id = ?", (game_id,))
    conn.execute("UPDATE pitch_calibrations SET is_active = 1 WHERE id = ?", (cal_id,))


def set_active_calibration(conn, game_id, cal_id):
    row = conn.execute("SELECT game_id FROM pitch_calibrations WHERE id = ?", (cal_id,)).fetchone()
    if not row or row["game_id"] != game_id:
        raise ValidationError("That calibration does not belong to this game")
    _activate(conn, game_id, cal_id)
    conn.commit()


def delete_calibration(conn, game_id, cal_id):
    row = conn.execute("SELECT game_id, is_active FROM pitch_calibrations WHERE id = ?", (cal_id,)).fetchone()
    if not row or row["game_id"] != game_id:
        raise ValidationError("That calibration does not belong to this game")
    conn.execute("DELETE FROM pitch_calibrations WHERE id = ?", (cal_id,))
    if row["is_active"]:                              # promote the newest remaining one so the game is not left uncalibrated
        nxt = conn.execute("SELECT id FROM pitch_calibrations WHERE game_id = ? ORDER BY id DESC LIMIT 1", (game_id,)).fetchone()
        if nxt:
            _activate(conn, game_id, nxt["id"])
    conn.commit()


# --------------------------------------------------------------------------------------------- games

TIME_FIELDS = [("game_start", "game_start_s"), ("half_time_start", "half_time_start_s"),
               ("half_time_end", "half_time_end_s"), ("game_end", "game_end_s")]
TIME_LABELS = {"game_start": "Game start", "half_time_start": "Half-time start", "half_time_end": "Half-time end",
               "game_end": "Game end"}


def normalise_date(value):
    """Accepts 2026-10-02, 2026.10.02, 2026/10/02 (or empty) and returns the spreadsheet's 2026.10.02 form."""
    if value is None or not str(value).strip():
        return None
    m = re.fullmatch(r"\s*(\d{4})[-./](\d{1,2})[-./](\d{1,2})\s*", str(value))
    if not m:
        raise ValueError(f"'{value}' is not a date like 2026-10-02")
    y, mo, d = (int(x) for x in m.groups())
    import datetime
    datetime.date(y, mo, d)                                       # raises ValueError for 31 Feb etc.
    return f"{y:04d}.{mo:02d}.{d:02d}"


def display_date(stored):
    return stored.replace(".", "-") if stored else ""


def resolve_video_path(path):
    """Where a stored video path really is: relative to the data folder (the external drive) when it is there,
    otherwise relative to the application folder (how older games were saved)."""
    if not path or not str(path).strip():
        return None
    return paths.resolve_stored(path)


def effective_video(g, slot):
    """The analysis / edited video for a game: the path typed on the game if there is one, otherwise the file
    found by name in the game's folder. Returns {"path", "auto", "exists"}."""
    field = {"analysis": "source_video", "edited": "edited_video"}[slot]
    explicit = g.get(field)
    if explicit and str(explicit).strip():
        p = resolve_video_path(explicit)
        if p and os.path.isfile(p):
            return {"path": p, "auto": False, "exists": True}
        alt = paths.conventional_video(g["game_id"], slot)      # the typed path is wrong or the drive moved: use the game folder's file
        if alt:
            return {"path": alt, "auto": True, "exists": True, "typed_missing": explicit}
        return {"path": p, "auto": False, "exists": False}
    p = paths.conventional_video(g["game_id"], slot)
    return {"path": p, "auto": True, "exists": bool(p)}


def output_db_path(game_id, title):
    """The game's detections database: in its folder when it has one, else the old shared output folder."""
    p = paths.detections_db(game_id)
    if p and os.path.exists(p):
        return p
    return os.path.join(config.VIDEO_PROCESSING_DIR, "output", f"game-id-{game_id}-{(title or '').replace(' ', '_')}.db")


def _opt_text(v):
    v = None if v is None else str(v).strip()
    return v or None


def _opt_int(v, label, errors, minimum=0):
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        n = int(str(v).strip())
        if n < minimum:
            raise ValueError
        return n
    except ValueError:
        errors.append(f"{label} must be a whole number")
        return None


def validate_game(conn, data, game_id=None):
    """Turn what the form sent into a clean row. Returns (row_dict, exclusions, warnings); raises ValidationError."""
    errors, warnings = [], []
    row = {}
    row["title"] = _opt_text(data.get("title"))
    row["description"] = _opt_text(data.get("description"))
    row["notes"] = _opt_text(data.get("notes"))
    row["game_type"] = _opt_text(data.get("game_type"))
    row["game_format"] = _opt_text(data.get("game_format"))
    try:
        row["date"] = normalise_date(data.get("date"))
    except ValueError as e:
        errors.append(f"Date: {e}")
        row["date"] = None

    for side in ("a", "b"):
        tid = data.get(f"team_{side}_id")
        tid = None if tid in (None, "", 0, "0") else int(tid)
        if tid is not None and not get_team(conn, tid):
            errors.append(f"Team {side.upper()} does not exist")
        row[f"team_{side}_id"] = tid
        row[f"score_{side}"] = _opt_int(data.get(f"score_{side}"), f"Score for team {side.upper()}", errors)
    if row["team_a_id"] and row["team_a_id"] == row["team_b_id"]:
        errors.append("Team A and team B are the same team")
    if (row["score_a"] is None) != (row["score_b"] is None):
        warnings.append("Only one score is filled in")

    if not row["title"] and row["team_a_id"]:                      # sensible default: 'Polis vs Aphrodite Wanderers'
        names = [get_team(conn, t)["name"] for t in (row["team_a_id"], row["team_b_id"]) if t]
        row["title"] = " vs ".join(names)
    if not row["title"]:
        errors.append("The game needs a title (or pick the teams and it is filled in for you)")

    for src, dst in (("source_video", "source_video"), ("output_video", "output_video"),
                     ("edited_video", "edited_video"), ("youtube_360_video", "youtube_360_video")):
        row[dst] = _opt_text(data.get(src))

    speed = data.get("running_speed_km_h")
    if speed in (None, ""):
        row["running_speed_km_h"] = None
    else:
        try:
            row["running_speed_km_h"] = float(speed)
            if not 3 <= row["running_speed_km_h"] <= 40:
                warnings.append("Running speed looks unusual (normally 12-18 km/h)")
        except ValueError:
            errors.append("Running speed must be a number")

    times = {}
    for field, col in TIME_FIELDS:
        try:
            row[col] = parse_time(data.get(field))
        except ValueError as e:
            errors.append(f"{TIME_LABELS[field]}: {e}")
            row[col] = None
        times[field] = row[col]
    order = [(TIME_LABELS[f], times[f]) for f, _ in TIME_FIELDS if times[f] is not None]
    for (la, a), (lb, b) in zip(order, order[1:]):
        if b <= a:
            warnings.append(f"{lb} ({format_mmss(b)}) is not after {la} ({format_mmss(a)})")
    if row["game_start_s"] is None or row["game_end_s"] is None:
        warnings.append("Game start and game end are needed before this game can be processed")

    exclusions = []
    for i, ex in enumerate(data.get("exclusions") or [], 1):
        try:
            s, e = parse_time(ex.get("start")), parse_time(ex.get("end"))
        except ValueError as err:
            errors.append(f"Exclusion {i}: {err}")
            continue
        if s is None and e is None and not (ex.get("reason") or "").strip():
            continue                                               # an untouched blank row
        if s is None or e is None or e <= s:
            errors.append(f"Exclusion {i}: needs a start time and an end time after it")
            continue
        exclusions.append({"start_s": s, "end_s": e, "reason": _opt_text(ex.get("reason"))})

    if game_id is None and data.get("game_id") not in (None, ""):
        try:
            gid = int(data["game_id"])
            if conn.execute("SELECT 1 FROM games WHERE game_id=?", (gid,)).fetchone():
                errors.append(f"Game {gid} already exists")
        except ValueError:
            errors.append("Game id must be a whole number")
    if errors:
        raise ValidationError(errors)
    return row, exclusions, warnings


def next_game_id(conn):
    return (conn.execute("SELECT COALESCE(MAX(game_id), -1) FROM games").fetchone()[0]) + 1


GAME_COLS = ["date", "game_type", "game_format", "title", "description", "team_a_id", "team_b_id", "score_a",
             "score_b", "source_video", "output_video", "edited_video", "youtube_360_video", "running_speed_km_h", "game_start_s",
             "half_time_start_s", "half_time_end_s", "game_end_s", "notes"]


def save_game(conn, data, game_id=None):
    """Create (game_id None) or update a game. Returns (game_dict, warnings)."""
    row, exclusions, warnings = validate_game(conn, data, game_id)
    stamp = now_iso()
    if game_id is None:
        game_id = int(data["game_id"]) if data.get("game_id") not in (None, "") else next_game_id(conn)
        conn.execute(f"INSERT INTO games(game_id, {', '.join(GAME_COLS)}, archived, created_at, updated_at) "
                     f"VALUES (?, {', '.join('?' * len(GAME_COLS))}, 0, ?, ?)",
                     [game_id] + [row[c] for c in GAME_COLS] + [stamp, stamp])
    else:
        if not conn.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,)).fetchone():
            raise ValidationError(f"Game {game_id} does not exist")
        conn.execute(f"UPDATE games SET {', '.join(c + '=?' for c in GAME_COLS)}, updated_at=? WHERE game_id=?",
                     [row[c] for c in GAME_COLS] + [stamp, game_id])
    conn.execute("DELETE FROM game_exclusions WHERE game_id = ?", (game_id,))
    for ex in exclusions:
        conn.execute("INSERT INTO game_exclusions(game_id, start_s, end_s, reason) VALUES (?,?,?,?)",
                     (game_id, ex["start_s"], ex["end_s"], ex["reason"]))
    conn.commit()
    return get_game(conn, game_id), warnings


def set_archived(conn, game_id, archived):
    conn.execute("UPDATE games SET archived=?, updated_at=? WHERE game_id=?", (1 if archived else 0, now_iso(), game_id))
    conn.commit()


def delete_game(conn, game_id):
    conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))     # exclusions and calibrations cascade
    conn.commit()


def duplicate_game(conn, game_id):
    """A new game with the same teams, type, format and calibration (a handy start for the next match filmed from the same spot)."""
    g = get_game(conn, game_id)
    if not g:
        raise ValidationError("That game does not exist")
    data = {k: g[k] for k in ("game_type", "game_format", "team_a_id", "team_b_id", "title", "running_speed_km_h")}
    new, _ = save_game(conn, data)
    cal = active_calibration(conn, game_id)
    if cal:
        add_calibration(conn, new["game_id"], "manual", cal["vertices"], cal["goal_left"], cal["goal_right"],
                        cal["calibrator_json"], label=f"copied from game {game_id}")
    return get_game(conn, new["game_id"])


def get_exclusions(conn, game_id):
    return [{"id": r["id"], "start_s": r["start_s"], "end_s": r["end_s"], "start": format_mmss(r["start_s"]),
             "end": format_mmss(r["end_s"]), "reason": r["reason"]}
            for r in conn.execute("SELECT * FROM game_exclusions WHERE game_id = ? ORDER BY start_s", (game_id,))]


def readiness(g, has_cal, video_found):
    """What is still missing before the pipeline can run this game."""
    missing = []
    info = paths.root_info()
    analysis = (g.get("videos") or {}).get("analysis")
    if info["set"] and not info["exists"] and not g["source_video"]:
        missing.append("Data folder not connected")
    elif analysis is not None and not analysis["path"]:
        missing.append("No analysis video chosen")
    elif video_found is False:
        missing.append("Analysis video not found")
    if not has_cal:
        missing.append("No pitch calibration")
    if g["game_start_s"] is None or g["game_end_s"] is None:
        missing.append("Start/end times missing")
    if not g["team_a_id"] or not g["team_b_id"]:
        missing.append("Teams missing")
    return missing


def _game_dict(conn, row, with_detail=True):
    g = dict(row)
    g["archived"] = bool(g["archived"])
    g["date_display"] = display_date(g["date"])
    for field, col in TIME_FIELDS:
        g[field] = format_mmss(g[col])
    teams_by_id = {r["team_id"]: r for r in conn.execute("SELECT team_id, name, kit_colours FROM teams")}
    names = {k: v["name"] for k, v in teams_by_id.items()}
    g["team_a"] = names.get(g["team_a_id"], "")
    g["team_b"] = names.get(g["team_b_id"], "")

    def _kit(team_id):                    # the team's kit colours as [[r,g,b], ...] (for the colour dots in the games list)
        t = teams_by_id.get(team_id)
        try:
            return [list(c["rgb"]) for c in json.loads((t["kit_colours"] if t else "") or "[]")][:4]
        except (ValueError, KeyError, TypeError):
            return []
    g["team_a_colours"], g["team_b_colours"] = _kit(g["team_a_id"]), _kit(g["team_b_id"])
    g["score"] = f"{g['score_a']}-{g['score_b']}" if g["score_a"] is not None and g["score_b"] is not None else ""
    cal_n = conn.execute("SELECT COUNT(*), COALESCE(SUM(is_active),0) FROM pitch_calibrations WHERE game_id=?",
                         (g["game_id"],)).fetchone()
    g["calibration_count"], g["has_active_calibration"] = cal_n[0], bool(cal_n[1])
    g["videos"] = {}
    for slot in ("edited", "analysis"):
        v = effective_video(g, slot)
        g["videos"][slot] = {"path": v["path"], "shown": paths.shown(v["path"]), "auto": v["auto"], "exists": v["exists"],
                             "typed_missing": v.get("typed_missing")}
    a = g["videos"]["analysis"]
    g["video_found"] = a["exists"] if a["path"] else None
    folder = paths.game_folder(g["game_id"])
    g["folder"] = {"path": folder, "shown": paths.shown(folder), "exists": bool(folder)}
    g["data_root"] = paths.root_info()
    g["processed"] = os.path.exists(output_db_path(g["game_id"], g["title"]))
    g["missing"] = readiness(g, g["has_active_calibration"], g["video_found"])
    g["ready"] = not g["missing"]
    if with_detail:
        g["exclusions"] = get_exclusions(conn, g["game_id"])
        g["calibrations"] = list_calibrations(conn, g["game_id"])
    return g


def get_game(conn, game_id, with_detail=True):
    row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
    return _game_dict(conn, row, with_detail) if row else None


def list_games(conn, include_archived=False):
    sql = "SELECT * FROM games" + ("" if include_archived else " WHERE archived = 0") + " ORDER BY game_id DESC"
    return [_game_dict(conn, r, with_detail=False) for r in conn.execute(sql)]
