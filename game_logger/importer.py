"""One-off (and repeatable) import of games-logger.xlsx into the game logger database.

    python -m game_logger.importer                      # imports games that are not in the database yet
    python -m game_logger.importer --overwrite          # re-imports every game, replacing what is in the database
    python -m game_logger.importer --excel other.xlsx

The spreadsheet is only ever read. A backup of the database is taken first when it already has data.
"""
import argparse
import ast
import datetime
import json
import os
import re
import sys

from . import config, db, repo
from .timeutil import normalise_excel_time, parse_time

EXCEL_TIME_COLUMNS = [("game_start_seconds", "game_start_s"), ("half_time_start_seconds", "half_time_start_s"),
                      ("half_time_end_seconds", "half_time_end_s"), ("game_end_seconds", "game_end_s")]


def _blank(v):
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


def _text(v):
    return None if _blank(v) else str(v).strip()


def kit_colours_from_teams_py(path=None):
    """Read the hard-coded kit colours out of team_assigner/teams.py without importing it."""
    path = path or config.TEAMS_PY
    if not os.path.exists(path):
        return {}
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "TEAMS" for t in node.targets):
            try:
                raw = ast.literal_eval(node.value)
            except ValueError:
                continue
            return {name: [{"rgb": list(c["rgb"]), "tolerance": list(c["tolerance"])} for c in cols]
                    for name, cols in raw.items()}
    return {}


def seed_teams(conn, warnings):
    created = 0
    for name, colours in kit_colours_from_teams_py().items():
        tid = repo.team_id_by_name(conn, name)
        if tid is None:
            repo.save_team(conn, {"name": name, "kit_colours": colours})
            created += 1
        else:
            existing = repo.get_team(conn, tid)
            if not existing["kit_colours"]:
                existing["kit_colours"] = colours
                repo.save_team(conn, existing, tid)
    return created


def _team_for(conn, name, game_id, warnings):
    name = _text(name)
    if not name:
        return None
    tid = repo.team_id_by_name(conn, name)
    if tid is None:
        tid = repo.save_team(conn, {"name": name, "kit_colours": []})["team_id"]
        warnings.append(f"game {game_id}: team '{name}' was not in the pipeline's team list, so it was added with no kit "
                        f"colours (add them on the Teams page before processing a game that includes this team)")
    return tid


def _parse_score(v, game_id, warnings):
    if _blank(v):
        return None, None
    m = re.fullmatch(r"\s*(\d+)\s*[/:\-]\s*(\d+)\s*", str(v))
    if not m:
        warnings.append(f"game {game_id}: score '{v}' was not understood, left blank")
        return None, None
    return int(m.group(1)), int(m.group(2))


def _parse_date(v, game_id, warnings):
    if _blank(v):
        return None
    if isinstance(v, (datetime.datetime, datetime.date)):
        return f"{v.year:04d}.{v.month:02d}.{v.day:02d}"
    try:
        return repo.normalise_date(v)
    except ValueError:
        warnings.append(f"game {game_id}: date '{v}' was not understood, left blank")
        return None


def _parse_exclusions(v, game_id, warnings):
    out = []
    if _blank(v):
        return out
    for part in str(v).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split("-")
            s, e = parse_time(a.strip()), parse_time(b.strip())
            if s is None or e is None or e <= s:
                raise ValueError
            out.append({"start": s, "end": e, "reason": "imported from spreadsheet"})
        except ValueError:
            warnings.append(f"game {game_id}: exclusion '{part}' was not understood, skipped")
    return out


def import_row(conn, row, overwrite, warnings):
    """Import one spreadsheet row (a dict). Returns 'added', 'replaced' or 'skipped'."""
    game_id = int(row["game_id"])
    exists = conn.execute("SELECT 1 FROM games WHERE game_id=?", (game_id,)).fetchone()
    if exists and not overwrite:
        return "skipped"
    if exists:
        conn.execute("DELETE FROM games WHERE game_id=?", (game_id,))

    ta, tb = _team_for(conn, row.get("team_a"), game_id, warnings), _team_for(conn, row.get("team_b"), game_id, warnings)
    score_a, score_b = _parse_score(row.get("score"), game_id, warnings)
    times = {}
    for col, field in EXCEL_TIME_COLUMNS:
        txt = normalise_excel_time(row.get(col))
        try:
            times[field] = parse_time(txt)
        except ValueError:
            warnings.append(f"game {game_id}: {col} '{txt}' was not understood, left blank")
            times[field] = None
    speed = row.get("running_speed_km_h")
    stamp = db.now_iso()
    vals = {
        "game_id": game_id, "date": _parse_date(row.get("date"), game_id, warnings),
        "game_type": _text(row.get("game_type")), "game_format": _text(row.get("game_format")),
        "title": _text(row.get("title")) or f"Game {game_id}", "description": _text(row.get("description")),
        "team_a_id": ta, "team_b_id": tb, "score_a": score_a, "score_b": score_b,
        "source_video": _text(row.get("statistics_source_video")), "output_video": _text(row.get("statistics_output_video")),
        "youtube_360_video": _text(row.get("youtube_360_video")),
        "running_speed_km_h": None if _blank(speed) else float(speed), **times,
        "notes": None, "archived": 0, "created_at": stamp, "updated_at": stamp,
    }
    cols = list(vals)
    conn.execute(f"INSERT INTO games({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", [vals[c] for c in cols])
    for ex in _parse_exclusions(row.get("extra_exclusions"), game_id, warnings):
        conn.execute("INSERT INTO game_exclusions(game_id, start_s, end_s, reason) VALUES (?,?,?,?)",
                     (game_id, ex["start"], ex["end"], ex["reason"]))

    # calibration: the legacy vertex columns are kept exactly (they can differ by a pixel from the calibrator JSON)
    try:
        vertices = {n: repo.parse_vertex_text(row.get(f"pitch_vertices_{n}")) for n in config.PITCH_POINT_NAMES}
    except (ValueError, TypeError):
        vertices = None
    cal_json = _text(row.get("pitch_calibrator_json"))
    goal_l = goal_r = None
    goals = []
    for side in ("left", "right"):
        try:
            goals.append(repo.parse_goal_text(row.get(f"goal_posts_{side}_vertices")))
        except ValueError:
            goals.append(None)
            warnings.append(f"game {game_id}: the {side} goal-post cell is incomplete ({row.get(f'goal_posts_{side}_vertices')}), left blank")
    goal_l, goal_r = goals
    if cal_json:
        try:
            jv, jl, jr, norm = repo.parse_calibrator_json(cal_json)
        except repo.ValidationError as e:
            warnings.append(f"game {game_id}: the pitch_calibrator_json cell is not usable ({e}); legacy columns used instead")
            cal_json = None
        else:
            vertices = vertices or jv
            goal_l, goal_r = goal_l or jl, goal_r or jr
            cal_json = norm
    if vertices:
        repo.add_calibration(conn, game_id, "excel-import", vertices, goal_l, goal_r, cal_json,
                             label="imported from games-logger.xlsx", make_active=True)
    elif any(not _blank(row.get(f"pitch_vertices_{n}")) for n in config.PITCH_POINT_NAMES):
        warnings.append(f"game {game_id}: pitch vertex columns are incomplete or not understood, no calibration imported")
    conn.commit()
    return "replaced" if exists else "added"


def import_excel(conn, excel_path, overwrite=False):
    import pandas as pd
    warnings = []
    df = pd.read_excel(excel_path)
    teams_seeded = seed_teams(conn, warnings)
    counts = {"added": 0, "replaced": 0, "skipped": 0}
    for row in df.to_dict(orient="records"):
        if _blank(row.get("game_id")):
            continue
        counts[import_row(conn, row, overwrite, warnings)] += 1
    return {"rows": len(df), "teams_seeded": teams_seeded, **counts, "warnings": warnings}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import games-logger.xlsx into the game logger database")
    ap.add_argument("--excel", default=config.EXCEL_PATH_DEFAULT)
    ap.add_argument("--overwrite", action="store_true", help="replace games already in the database")
    args = ap.parse_args(argv)
    if not os.path.exists(args.excel):
        sys.exit(f"Cannot find {args.excel}")
    if os.path.exists(config.DB_PATH):
        b = db.backup("before-import")
        print(f"Backed up the existing database to {b}")
    conn = db.connect()
    result = import_excel(conn, args.excel, args.overwrite)
    print(f"Database: {config.DB_PATH}")
    print(f"Spreadsheet rows: {result['rows']} | added {result['added']}, replaced {result['replaced']}, "
          f"already there (skipped) {result['skipped']} | teams seeded from teams.py: {result['teams_seeded']}")
    for w in result["warnings"]:
        print("  note:", w)


if __name__ == "__main__":
    main()
