"""Starting, watching and stopping a pipeline run (main.py) from the game logger.

main.py runs as its own process (the same Python that runs this app), writes everything it prints to a log file in the
game's folder, and this module reads that file back for the console on the game page. The run carries on if the browser
page is closed; there is only ever one run at a time."""
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from . import config, db, paths, repo

STEPS = [
    ("detections", "Detections", "find players, goalkeepers and the ball in every frame (the long one)"),
    ("transformations", "Transformations and statistics", "overhead positions, team assignment, tracks, speed, distance, possession"),
    ("detection_report", "Detection report", "scores each detection stage and says what to fix first (PDF, plus a review sheet to check by eye)"),
    ("match_report", "Match report", "12-page PDF: possession, territory, team shape, distance, speed, running, key moments"),
    ("opposition_report", "Opposition reports", "2 scouting PDFs (one per team, written for the other team's coach): style, strong and weak points, game plan"),
    ("annotation", "Annotated video", "the video with detections drawn on it"),
]
STEP_IDS = [s[0] for s in STEPS]
NEEDS_DB = {"transformations", "detection_report", "match_report", "opposition_report", "annotation"}
NEEDS_READY = {"detections", "transformations", "annotation"}

POINTER = os.path.join(config.VIDEO_PROCESSING_DIR, "game-logger-current-run.json")
_lock = threading.Lock()
_run = {"meta": None, "proc": None}            # the current (or most recent) run


# ------------------------------------------------------------------------------------------ helpers

def _now():
    return datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _write_meta(meta):
    tmp = meta["meta_file"] + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, meta["meta_file"])


def _write_pointer(meta):
    try:
        with open(POINTER, "w", encoding="utf-8") as f:
            json.dump({"meta_file": meta["meta_file"]}, f)
    except OSError:
        pass


def _load_current():
    """The run this app knows about; after an app restart, the one the pointer file names."""
    if _run["meta"] is None:
        try:
            with open(POINTER, encoding="utf-8") as f:
                mf = json.load(f)["meta_file"]
            with open(mf, encoding="utf-8") as f:
                _run["meta"] = json.load(f)
        except (OSError, ValueError, KeyError):
            return None
    return _run["meta"]


def _refresh(meta):
    """Update a 'running' meta if its process has ended (including one started before the app was restarted)."""
    if meta and meta["status"] == "running":
        proc = _run["proc"]
        if proc is not None and proc.pid == meta["pid"]:
            return meta                                      # our own process: _watch records how it ended
        if not _pid_alive(meta["pid"]):
            meta["status"] = "finished"
            meta["ended"] = meta.get("ended") or _now()
            meta["note"] = "The run ended while the game logger was not watching, so its result is not known - see the end of the log."
            _write_meta(meta)
    return meta


def log_folder(game_id):
    info = paths.root_info()
    if info["set"]:
        folder = paths.game_folder(game_id)
        if not folder:
            conn = db.connect()
            try:
                g = repo.get_game(conn, game_id, with_detail=False)
            finally:
                conn.close()
            folder, _ = paths.create_game_folder(g)
        return os.path.join(folder, "logs")
    return os.path.join(config.VIDEO_PROCESSING_DIR, "output", "logs")


def _db_path(g):
    if paths.root_info()["set"]:
        return paths.detections_db(g["game_id"])
    return repo.output_db_path(g["game_id"], g["title"])


def _test_name(p):
    b, e = os.path.splitext(p)
    return b + "-test" + e


# ------------------------------------------------------------------------------------------ public

def status():
    with _lock:
        meta = _refresh(_load_current())
        return dict(meta) if meta else None


def start(conn, game_id, steps, duration_s=None):
    steps = [s for s in STEP_IDS if s in set(steps or [])]       # keep the pipeline's own order
    errors = []
    if not steps:
        raise repo.ValidationError("Tick at least one stage to run")
    g = repo.get_game(conn, game_id, with_detail=False)
    if g is None:
        raise repo.ValidationError(f"Game {game_id} does not exist")
    try:
        duration_s = int(duration_s) if duration_s not in (None, "", 0, "0") else 0
    except (TypeError, ValueError):
        raise repo.ValidationError("The test length must be a whole number of seconds")
    if duration_s < 0:
        raise repo.ValidationError("The test length cannot be negative")

    with _lock:
        cur = _refresh(_load_current())
        if cur and cur["status"] == "running":
            raise repo.ValidationError(f"A run for game {cur['game_id']} is already in progress. Wait for it to finish or stop it first.")

        if NEEDS_READY & set(steps) and g["missing"]:
            raise repo.ValidationError(["This game is not ready to process yet:"] + g["missing"])
        info = paths.root_info()
        if info["set"] and not info["exists"]:
            raise repo.ValidationError(f"The data folder {info['path']} is not connected. Connect the drive first.")

        notes = []
        dbp = _db_path(g)
        if dbp and duration_s:
            dbp = _test_name(dbp)
        if "detections" not in steps and NEEDS_DB & set(steps) and not (dbp and os.path.exists(dbp)):
            raise repo.ValidationError("There is no detections database for this game" + (" test run" if duration_s else "") +
                                       " yet, so tick Detections as well.")
        if "annotation" in steps and "detections" not in steps:
            notes.append("Annotated video without Detections in this run: it uses the player pictures saved by the last detections run.")

        folder = log_folder(game_id)
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = f"{paths.prefix(game_id)}_run{'-test' if duration_s else ''}_{stamp}"
        log_file, meta_file = os.path.join(folder, base + ".log"), os.path.join(folder, base + ".json")

        cmd = [sys.executable, "-u", "main.py", "--game", str(game_id), "--steps", ",".join(steps)]
        if duration_s:
            cmd += ["--duration", str(duration_s)]
        shown_cmd = " ".join(cmd)
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            cmd = ["caffeinate", "-i"] + cmd                  # the Mac stays awake while the run is going
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with open(log_file, "ab") as lf:
            header = (f"=== Game logger run: game {game_id} ({g['title']})  {_now()}\n"
                      f"=== Stages: {', '.join(steps)}" + (f"   TEST RUN: first {duration_s} seconds only (results are kept in separate test files)" if duration_s else "") + "\n"
                      f"=== {shown_cmd}\n" + "".join(f"=== Note: {n}\n" for n in notes) + "\n")
            lf.write(header.encode("utf-8"))
        logf = open(log_file, "ab")
        try:
            proc = subprocess.Popen(cmd, cwd=config.APP_DIR, env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError as e:
            logf.close()
            raise repo.ValidationError(f"Could not start the run: {e}")
        logf.close()
        meta = {"id": base, "game_id": game_id, "title": g["title"], "steps": steps, "duration_s": duration_s, "test": bool(duration_s),
                "started": _now(), "ended": None, "exit_code": None, "status": "running", "pid": proc.pid,
                "log_file": log_file, "meta_file": meta_file, "notes": notes, "stop_requested": False}
        _run["meta"], _run["proc"] = meta, proc
        _write_meta(meta)
        _write_pointer(meta)
    threading.Thread(target=_watch, args=(proc, meta), daemon=True).start()
    return dict(meta)


def _watch(proc, meta):
    code = proc.wait()
    with _lock:
        meta["ended"] = _now()
        meta["exit_code"] = code
        meta["status"] = "stopped" if meta.get("stop_requested") else ("finished" if code == 0 else "failed")
        try:
            with open(meta["log_file"], "ab") as f:
                f.write(f"\n=== Run {meta['status']} (exit code {code}) at {meta['ended']}\n".encode("utf-8"))
        except OSError:
            pass
        _write_meta(meta)


def stop():
    with _lock:
        meta = _refresh(_load_current())
        if not meta or meta["status"] != "running":
            raise repo.ValidationError("There is no run in progress")
        meta["stop_requested"] = True
        _write_meta(meta)
        pid = meta["pid"]
    try:
        os.killpg(pid, signal.SIGTERM)                       # main.py and anything it started (ffmpeg) share this group
    except (ProcessLookupError, PermissionError, OSError):
        pass

    def hard_kill():
        time.sleep(8)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    threading.Thread(target=hard_kill, daemon=True).start()
    return status()


def _safe_log_path(game_id, run_id):
    folder = log_folder(game_id)
    name = os.path.basename(run_id)
    p = os.path.join(folder, name + ".log")
    return p if os.path.isfile(p) else None


def read_log(path, offset=0, tail=False, limit=262144):
    """{'text','offset','size'}: new text since `offset` (bytes). tail=True starts near the end instead."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"text": "", "offset": 0, "size": 0}
    if tail:
        offset = max(0, size - 60000)
    offset = max(0, min(int(offset), size))
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(limit)
    if tail and offset > 0:
        nl = data.find(b"\n")
        if nl >= 0:
            offset += nl + 1
            data = data[nl + 1:]
    text = ""
    for cut in range(4):                                     # do not split a multi-byte character across two reads
        try:
            text = data[:len(data) - cut].decode("utf-8")
            offset += len(data) - cut
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", "replace")
        offset += len(data)
    return {"text": text, "offset": offset, "size": size}


def current_log(offset=0, tail=False):
    with _lock:
        meta = _refresh(_load_current())
    if not meta:
        return {"text": "", "offset": 0, "size": 0, "run": None}
    out = read_log(meta["log_file"], offset, tail)
    out["run"] = {k: meta[k] for k in ("id", "game_id", "status", "exit_code", "started", "ended", "steps", "test")}
    return out


def list_runs(game_id):
    folder = log_folder(game_id) if (not paths.root_info()["set"] or paths.game_folder(game_id)) else None
    runs = []
    if folder and os.path.isdir(folder):
        for name in sorted(os.listdir(folder), reverse=True):
            if name.endswith(".json") and name.startswith(paths.prefix(game_id) + "_run"):
                try:
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        m = json.load(f)
                except (OSError, ValueError):
                    continue
                with _lock:
                    if m.get("status") == "running":
                        cur = _refresh(_load_current())
                        if cur and cur["id"] == m["id"]:
                            m = cur
                runs.append({k: m.get(k) for k in ("id", "status", "started", "ended", "steps", "test", "exit_code")})
    return runs[:50]
