"""The position each player played in a game (per half) and the formation played (per half).

Stored in game-logger.db (tables game_player_positions and game_formations) and copied into the game's NNNN_players.db as the
tables player_game_roles and game_formations, so analysts read it from the same file as the positions. Nothing in the pipeline
uses it yet. It belongs to the PERSON in the GAME, not to a track: track ids change when detections are re-run, positions played do not.

Half 2 inherits half 1 unless it is set: only explicit rows are stored; the saved copy shows both halves (column `inherited`)."""
import os
import shutil
import sqlite3
import tempfile

import pandas as pd

from . import paths
from .db import now_iso
from .repo import ValidationError

POSITIONS = ["Goalkeeper", "Left back", "Right back", "Centre back", "Midfield", "Left wing", "Right wing", "Forward", "Rotating / mixed"]
FORMATIONS = ["2-1-2", "1-3-1", "2-2-1", "2-3-0", "Other"]


def _half(h):
    try:
        h = int(h)
    except (TypeError, ValueError):
        raise ValidationError("half must be 1 or 2")
    if h not in (1, 2):
        raise ValidationError("half must be 1 or 2")
    return h


def _tables_exist(conn):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='game_player_positions'").fetchone() is not None


def get(conn, game_id):
    """{vocabulary, formations, players:[{roster_player_id,name,tagged,h1,h2,note1,note2}], formation:{1,2}}."""
    names = {r["player_id"]: r["name"] for r in conn.execute("SELECT player_id, name FROM players")}
    tagged = {r[0] for r in conn.execute("SELECT DISTINCT roster_player_id FROM identity_assignments WHERE game_id = ?", (game_id,))}
    rows = conn.execute("SELECT roster_player_id, half, position, note FROM game_player_positions WHERE game_id = ?", (game_id,)).fetchall()
    by = {}
    for r in rows:
        d = by.setdefault(r["roster_player_id"], {})
        d[r["half"]] = (r["position"], r["note"])
    ids = sorted(tagged | set(by), key=lambda i: (names.get(i) or "").lower())
    players = []
    for i in ids:
        d = by.get(i, {})
        players.append({"roster_player_id": i, "name": names.get(i), "tagged": i in tagged,
                        "h1": (d.get(1) or (None, None))[0], "note1": (d.get(1) or (None, None))[1],
                        "h2": (d.get(2) or (None, None))[0], "note2": (d.get(2) or (None, None))[1]})
    f = {r["half"]: r["formation"] for r in conn.execute("SELECT half, formation FROM game_formations WHERE game_id = ?", (game_id,))}
    return {"positions": POSITIONS, "formations": FORMATIONS, "players": players, "formation": {"1": f.get(1), "2": f.get(2)}}


def set_position(conn, game_id, roster_player_id, half, position, note=None):
    """position None / '' removes the row (half 2: 'same as first half'; half 1: not set)."""
    half = _half(half)
    if not conn.execute("SELECT 1 FROM players WHERE player_id = ?", (roster_player_id,)).fetchone():
        raise ValidationError("That player does not exist")
    position = (position or "").strip()
    if position and position not in POSITIONS:
        raise ValidationError("Choose a position from the list")
    if not position:
        conn.execute("DELETE FROM game_player_positions WHERE game_id = ? AND roster_player_id = ? AND half = ?", (game_id, roster_player_id, half))
    else:
        conn.execute(
            "INSERT INTO game_player_positions(game_id, roster_player_id, half, position, note, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(game_id, roster_player_id, half) DO UPDATE SET position = excluded.position, note = excluded.note, updated_at = excluded.updated_at",
            (game_id, roster_player_id, half, position, (note or "").strip() or None, now_iso()))
    conn.commit()
    return sync_into_players_db(conn, game_id)


def remove_player(conn, game_id, roster_player_id):
    conn.execute("DELETE FROM game_player_positions WHERE game_id = ? AND roster_player_id = ?", (game_id, roster_player_id))
    conn.commit()
    return sync_into_players_db(conn, game_id)


def set_formation(conn, game_id, half, formation):
    half = _half(half)
    formation = (formation or "").strip()
    if formation and formation not in FORMATIONS:
        raise ValidationError("Choose a formation from the list")
    if not formation:
        conn.execute("DELETE FROM game_formations WHERE game_id = ? AND half = ?", (game_id, half))
    else:
        conn.execute("INSERT INTO game_formations(game_id, half, formation, updated_at) VALUES (?,?,?,?) "
                     "ON CONFLICT(game_id, half) DO UPDATE SET formation = excluded.formation, updated_at = excluded.updated_at",
                     (game_id, half, formation, now_iso()))
    conn.commit()
    return sync_into_players_db(conn, game_id)


# ------------------------------------------------------------------------------------------ the saved copy

def _has_half_time(conn, game_id):
    """A game with no half-time set ran as one period (no end swap): there is no second half to copy positions into."""
    r = conn.execute("SELECT half_time_start_s, half_time_end_s FROM games WHERE game_id = ?", (game_id,)).fetchone()
    return bool(r and r[0] is not None and r[1] is not None)


def resolved_frames(conn, game_id):
    """(roles DataFrame, formations DataFrame) with half 2 filled from half 1 where it was not set."""
    if not _tables_exist(conn):
        return (pd.DataFrame(columns=["roster_player_id", "name", "half", "position", "note", "inherited"]),
                pd.DataFrame(columns=["half", "formation", "inherited"]))
    g = get(conn, game_id)
    two_halves = _has_half_time(conn, game_id)
    roles = []
    for p in g["players"]:
        for half, pos, note, other, onote in ((1, p["h1"], p["note1"], None, None), (2, p["h2"], p["note2"], p["h1"], p["note1"])):
            if pos:
                roles.append((p["roster_player_id"], p["name"], half, pos, note, 0))
            elif half == 2 and other and two_halves:
                roles.append((p["roster_player_id"], p["name"], 2, other, onote, 1))
    f = g["formation"]
    forms = []
    if f["1"]:
        forms.append((1, f["1"], 0))
    if f["2"]:
        forms.append((2, f["2"], 0))
    elif f["1"] and two_halves:
        forms.append((2, f["1"], 1))
    return (pd.DataFrame(roles, columns=["roster_player_id", "name", "half", "position", "note", "inherited"]),
            pd.DataFrame(forms, columns=["half", "formation", "inherited"]))


DOCS = [("player_game_roles", "position", "position the player played in that half (declared on the Track Players page; see the Game Logger list)"),
        ("player_game_roles", "inherited", "1 = not set for this half, copied from the first half"),
        ("player_game_roles", "note", "free text from the coach/logger"),
        ("game_formations", "formation", "formation played in that half (declared)"),
        ("game_formations", "inherited", "1 = not set for this half, copied from the first half")]


def write_tables(out_conn, conn, game_id):
    """Write the two tables (and their column notes) into an open players.db connection."""
    roles, forms = resolved_frames(conn, game_id)
    out_conn.execute("DROP TABLE IF EXISTS player_game_roles")
    out_conn.execute("DROP TABLE IF EXISTS game_formations")
    roles.to_sql("player_game_roles", out_conn, index=False)
    forms.to_sql("game_formations", out_conn, index=False)
    try:
        out_conn.execute("DELETE FROM columns_doc WHERE table_name IN ('player_game_roles', 'game_formations')")
        out_conn.executemany("INSERT INTO columns_doc(table_name, column_name, description) VALUES (?,?,?)", DOCS)
    except sqlite3.DatabaseError:
        pass


def sync_into_players_db(conn, game_id):
    """Refresh just these two tables in the saved NNNN_players.db (if there is one). Returns {"synced": bool}."""
    try:
        from player_positions import players_db_path
        path = players_db_path(game_id)
    except Exception:                                           # noqa: BLE001
        return {"synced": False}
    if not os.path.isfile(path):
        return {"synced": False}
    tmpdir = tempfile.mkdtemp(prefix="roles_")
    tmp = os.path.join(tmpdir, os.path.basename(path))
    try:
        shutil.copyfile(path, tmp)
        c = sqlite3.connect(tmp)
        try:
            write_tables(c, conn, game_id)
            c.commit()
            ok = c.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            c.close()
        if ok != "ok":
            return {"synced": False, "error": "check failed"}
        shutil.copyfile(tmp, path)
        return {"synced": True}
    except (OSError, sqlite3.DatabaseError) as e:
        return {"synced": False, "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
