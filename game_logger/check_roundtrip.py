"""Proves the database gives the pipeline exactly what the spreadsheet did.

    python -m game_logger.check_roundtrip [--excel path] [--only 22]

For every spreadsheet row it builds the game record the old way (pandas + normalise_mmss) and the new way (from the
database) and lists any field that differs. Empty values are treated as equal whatever way they are written."""
import argparse
import json
import sys

from . import config, db, repo
from .record import RECORD_KEYS, build_game_record
from .timeutil import normalise_excel_time

TIME_COLS = ("game_start_seconds", "half_time_start_seconds", "half_time_end_seconds", "game_end_seconds")


def _empty(v):
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


def _norm(key, v):
    if _empty(v):
        return None
    if key == "pitch_calibrator_json":
        return json.loads(v)
    if key == "running_speed_km_h" or key == "game_id":
        return float(v)
    if key in ("goal_posts_left_vertices", "goal_posts_right_vertices"):
        try:
            return repo.parse_goal_text(v)                 # compare the points, not the bracket style
        except ValueError:
            return None                                    # incomplete in the spreadsheet: not carried over
    if key == "statistics_source_video":                    # the database record carries the real, full path
        return repo.resolve_video_path(str(v).strip())
    return str(v).strip()


def old_style_record(row):
    rec = dict(row)
    for k in TIME_COLS:
        rec[k] = normalise_excel_time(rec.get(k))
    return rec


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=config.EXCEL_PATH_DEFAULT)
    ap.add_argument("--only", type=int)
    args = ap.parse_args(argv)
    conn = db.connect()
    default_speed = float(conn.execute("SELECT value FROM settings WHERE key='running_speed_km_h'").fetchone()[0])
    rows = pd.read_excel(args.excel).to_dict(orient="records")
    bad = checked = 0
    for row in rows:
        gid = int(row["game_id"])
        if args.only is not None and gid != args.only:
            continue
        old, new = old_style_record(row), build_game_record(conn, gid)
        checked += 1
        if new is None:
            print(f"game {gid}: MISSING from the database")
            bad += 1
            continue
        diffs = []
        if list(new) != RECORD_KEYS:
            diffs.append("record keys differ")
        for key in RECORD_KEYS:
            a, b = _norm(key, old.get(key)), _norm(key, new.get(key))
            if key == "running_speed_km_h" and a is None:
                a = default_speed                      # an empty speed cell now falls back to the default setting
            if a != b:
                diffs.append(f"{key}: spreadsheet={old.get(key)!r:.60} database={new.get(key)!r:.60}")
        if diffs:
            bad += 1
            print(f"game {gid}: {len(diffs)} difference(s)")
            for d in diffs:
                print("    ", d)
    print(f"\nChecked {checked} game(s): {checked - bad} identical, {bad} with differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
