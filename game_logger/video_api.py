"""Serving a game's videos to the preview player (with seeking), listing its input/output videos, and an optional
lighter 'preview copy' for videos the browser cannot play (typically HEVC) or finds slow to scrub."""
import datetime
import hashlib
import mimetypes
import os
import shutil
import subprocess
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from . import config, db, repo
from .calibrator_api import _ffmpeg

router = APIRouter()

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
CACHE_DIR = os.path.join(config.VIDEO_PROCESSING_DIR, "preview-cache")
JOBS = {}                      # game_id -> {"state": "running"|"error", "pct": float|None, "error": str}
JOBS_LOCK = threading.Lock()


def _get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def _need_game(conn, game_id):
    g = repo.get_game(conn, game_id, with_detail=False)
    if g is None:
        raise HTTPException(404, f"Game {game_id} does not exist")
    return g


SLOTS = (("edited", "edited_video", "Edited video"),
         ("analysis", "source_video", "Analysis video"),
         ("annotated", "output_video", "Annotated video"))


def _slot_path(g, field):
    p = repo.resolve_video_path(g[field]) if g.get(field) else None
    return p if p and os.path.isfile(p) else None


def _input_path(g):
    return _slot_path(g, "source_video")


def _extra_outputs(g):
    """Other videos the pipeline has made for this game (output/game-id-<id>-*), newest first, apart from the one set as the annotated video."""
    items = []
    out_dir = os.path.join(config.VIDEO_PROCESSING_DIR, "output")
    prefix = f"game-id-{g['game_id']}-"
    annotated = _slot_path(g, "output_video")
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            full = os.path.join(out_dir, name)
            if name.startswith(prefix) and os.path.splitext(name)[1].lower() in VIDEO_EXT and os.path.isfile(full) \
                    and not (annotated and os.path.abspath(annotated) == os.path.abspath(full)):
                items.append(("out:" + name, full))
    items.sort(key=lambda kp: os.path.getmtime(kp[1]), reverse=True)
    return items


def _all_videos(g):
    """[(key, kind, label, path-or-None)] in the order shown in the droplist."""
    rows = []
    for key, field, title in SLOTS:
        p = _slot_path(g, field)
        name = os.path.basename(g[field]) if g.get(field) else None
        rows.append((key, key, f"{title}: {name}" if name else f"{title}: not set", p))
    for key, p in _extra_outputs(g):
        rows.append((key, "annotated", f"Annotated (pipeline output): {os.path.basename(p)} ({_fmt_time(os.path.getmtime(p))})", p))
    return rows


def _resolve_key(g, key):
    key = "analysis" if key == "input" else key
    for k, _kind, _label, p in _all_videos(g):
        if k == key:
            return p
    return None


def _tag(key):
    return hashlib.md5(key.encode()).hexdigest()[:6]


def proxy_path(game_id, key, src):
    st = os.stat(src)
    return os.path.join(CACHE_DIR, f"game-{game_id}-{_tag(key)}-{st.st_size}-{int(st.st_mtime)}.mp4")


def _fmt_time(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%d %b %H:%M")


def _proxy_info(game_id, key, src):
    info = {"exists": False, "state": None, "pct": None, "error": None}
    if not src:
        return info
    info["exists"] = os.path.isfile(proxy_path(game_id, key, src))
    with JOBS_LOCK:
        job = JOBS.get((game_id, key))
    if job and not info["exists"]:
        info.update(state=job["state"], pct=job["pct"], error=job.get("error"))
    return info


@router.get("/api/games/{game_id}/videos")
def list_videos(game_id: int, conn=Depends(_get_conn)):
    g = _need_game(conn, game_id)
    items = []
    for key, kind, label, path in _all_videos(g):
        items.append({"key": key, "kind": kind, "exists": bool(path), "label": label,
                      "size": os.path.getsize(path) if path else None, "proxy": _proxy_info(game_id, key, path)})
    return {"items": items, "ffmpeg": bool(_ffmpeg())}


def _range_response(path, request):
    size = os.path.getsize(path)
    ctype = mimetypes.guess_type(path)[0] or "video/mp4"
    if os.path.splitext(path)[1].lower() in (".mov", ".m4v"):
        ctype = "video/mp4"
    start, end, status = 0, size - 1, 200
    rng = request.headers.get("range", "")
    if rng.startswith("bytes="):
        first, _, last = rng[6:].split(",")[0].strip().partition("-")
        try:
            if first == "":
                start = max(0, size - int(last))
            else:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
        except ValueError:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if start > end or start >= size:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        status = 206
    length = end - start + 1

    def chunks():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(1 << 20, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(length), "Cache-Control": "no-store"}
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=status, media_type=ctype, headers=headers)


@router.get("/api/games/{game_id}/video")
def stream_video(game_id: int, request: Request, key: str = "input", proxy: bool = False, conn=Depends(_get_conn)):
    g = _need_game(conn, game_id)
    path = _resolve_key(g, key)
    if not path:
        raise HTTPException(404, "That video was not found")
    if proxy:
        pp = proxy_path(game_id, key, path)
        if os.path.isfile(pp):
            path = pp
    return _range_response(path, request)


# ------------------------------------------------------------------------------------------ preview copy

def _duration(ffmpeg, src):
    probe = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    probe = probe if os.path.exists(probe) else shutil.which("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                             capture_output=True, timeout=30).stdout.decode().strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def _make_proxy(game_id, key, src, dest):
    ffmpeg = _ffmpeg()
    tmp = dest + ".part"
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        total = _duration(ffmpeg, src)
        cmd = [ffmpeg, "-y", "-v", "error", "-progress", "pipe:1", "-nostats", "-i", src,
               "-vf", "scale=-2:540", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", "24",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "mp4", tmp]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    secs = int(line.split("=")[1]) / 1_000_000
                except ValueError:
                    continue
                if total:
                    with JOBS_LOCK:
                        JOBS[(game_id, key)]["pct"] = min(99.0, 100.0 * secs / total)
        err = proc.stderr.read()
        if proc.wait() != 0 or not os.path.exists(tmp):
            raise RuntimeError((err or "ffmpeg failed")[-300:])
        os.replace(tmp, dest)
        for name in os.listdir(CACHE_DIR):          # forget older copies of this game's video
            if name.startswith(f"game-{game_id}-{_tag(key)}-") and name != os.path.basename(dest):
                try:
                    os.remove(os.path.join(CACHE_DIR, name))
                except OSError:
                    pass
        with JOBS_LOCK:
            JOBS.pop((game_id, key), None)
    except Exception as e:                           # noqa: BLE001 - shown to the user
        with JOBS_LOCK:
            JOBS[(game_id, key)] = {"state": "error", "pct": None, "error": str(e)}
        try:
            os.remove(tmp)
        except OSError:
            pass


@router.post("/api/games/{game_id}/preview-copy")
def start_preview_copy(game_id: int, key: str = "analysis", conn=Depends(_get_conn)):
    g = _need_game(conn, game_id)
    key = "analysis" if key == "input" else key
    src = _resolve_key(g, key)
    if not src:
        raise repo.ValidationError("That video was not found")
    if not _ffmpeg():
        raise repo.ValidationError("ffmpeg is needed to make a preview copy and was not found. On a Mac: brew install ffmpeg")
    dest = proxy_path(game_id, key, src)
    if os.path.isfile(dest):
        return {"state": "done"}
    with JOBS_LOCK:
        if JOBS.get((game_id, key), {}).get("state") == "running":
            return {"state": "running"}
        JOBS[(game_id, key)] = {"state": "running", "pct": None, "error": None}
    threading.Thread(target=_make_proxy, args=(game_id, key, src, dest), daemon=True).start()
    return {"state": "running"}
