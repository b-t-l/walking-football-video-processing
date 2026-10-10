"""Builds the 'game record' the pipeline reads: the same 27 keys (and value styles) GetGameData.gameRecord() used to
produce from games-logger.xlsx, so nothing downstream has to change."""
import json

from . import repo
from .timeutil import format_mmss

NAN = float("nan")

RECORD_KEYS = [
    "game_id", "date", "game_type", "game_format", "title", "description", "youtube_360_video",
    "statistics_source_video", "statistics_output_video", "score", "team_a", "team_b", "pitch_calibrator_json",
    "pitch_vertices_left_bottom", "pitch_vertices_left_top", "pitch_vertices_right_top", "pitch_vertices_right_bottom",
    "pitch_vertices_centre_top", "pitch_vertices_centre_bottom", "goal_posts_left_vertices",
    "goal_posts_right_vertices", "game_start_seconds", "half_time_start_seconds", "half_time_end_seconds",
    "game_end_seconds", "extra_exclusions", "running_speed_km_h",
]


def _or_nan(v):
    return NAN if v is None or v == "" else v


def _goal_text(pts):
    return "[" + ",".join(f"({x},{y})" for x, y in pts) + "]" if pts else NAN


def build_game_record(conn, game_id):
    """Returns the record dict, or None if there is no such game. Empty values are NaN, as pandas gave them."""
    g = repo.get_game(conn, game_id)
    if g is None:
        return None
    cal = repo.active_calibration(conn, game_id)
    v = cal["vertices"] if cal else {}

    def vtx(name):
        return f"{v[name][0]},{v[name][1]}" if name in v else NAN

    exclusions = ",".join(f"{format_mmss(e['start_s'])}-{format_mmss(e['end_s'])}" for e in g["exclusions"])
    speed = g["running_speed_km_h"] if g["running_speed_km_h"] is not None else repo.default_running_speed(conn)
    rec = {
        "game_id": g["game_id"],
        "date": _or_nan(g["date"]),
        "game_type": _or_nan(g["game_type"]),
        "game_format": _or_nan(g["game_format"]),
        "title": g["title"],
        "description": _or_nan(g["description"]),
        "youtube_360_video": _or_nan(g["youtube_360_video"]),
        "statistics_source_video": _or_nan(g["videos"]["analysis"]["path"]),     # the real path: typed on the game, else found in its folder
        "statistics_output_video": _or_nan(g["output_video"]),
        "score": f"{g['score_a']}/{g['score_b']}" if g["score_a"] is not None and g["score_b"] is not None else NAN,
        "team_a": _or_nan(g["team_a"]),
        "team_b": _or_nan(g["team_b"]),
        "pitch_calibrator_json": _or_nan(cal["calibrator_json"]) if cal else NAN,
        "pitch_vertices_left_bottom": vtx("left_bottom"),
        "pitch_vertices_left_top": vtx("left_top"),
        "pitch_vertices_right_top": vtx("right_top"),
        "pitch_vertices_right_bottom": vtx("right_bottom"),
        "pitch_vertices_centre_top": vtx("centre_top"),
        "pitch_vertices_centre_bottom": vtx("centre_bottom"),
        "goal_posts_left_vertices": _goal_text(cal["goal_left"]) if cal else NAN,
        "goal_posts_right_vertices": _goal_text(cal["goal_right"]) if cal else NAN,
        "game_start_seconds": g["game_start"],          # '' when empty, like normalise_mmss
        "half_time_start_seconds": g["half_time_start"],
        "half_time_end_seconds": g["half_time_end"],
        "game_end_seconds": g["game_end"],
        "extra_exclusions": exclusions or NAN,
        "running_speed_km_h": float(speed),
    }
    assert list(rec) == RECORD_KEYS
    return rec


def pipeline_teams(conn):
    """{team name: [{'rgb': (r,g,b), 'tolerance': (h,s,v)}, ...]} - the shape team_assigner.teams.Teams.TEAMS has.
    Teams with no kit colours are left out (the classifier cannot use them)."""
    out = {}
    for t in repo.list_teams(conn, include_inactive=True):
        if t["kit_colours"]:
            out[t["name"]] = [{"rgb": tuple(c["rgb"]), "tolerance": tuple(c["tolerance"])} for c in t["kit_colours"]]
    return out
