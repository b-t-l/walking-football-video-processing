"""Match metrics: everything the match report shows, worked out from the detections database of one game.

Team level only (players stay anonymous). Nothing here draws anything: `compute_metrics()` returns a plain dictionary
(numbers, lists, small arrays) that the pages draw from, and that is also saved next to the report as a .json file so
other reports (opposition report, season trends) can reuse the same numbers.

Coordinates: the database stores positions in metres on the overhead plan. x runs along the pitch length (about -18 .. +18)
and y across the width (about -10 .. +10); (0, 0) is the centre spot.
"""
import math
import sqlite3

import numpy as np
import pandas as pd

PITCH_LENGTH = 36.0
PITCH_WIDTH = 20.0
HALF_L, HALF_W = PITCH_LENGTH / 2, PITCH_WIDTH / 2

HOLD_RADIUS_M = 2.0          # a player this close to the ball (and clearly the closest) has it
CONTEST_MARGIN_M = 0.5       # if an opponent is almost as close, the ball is contested: no change of possession
MAX_HOLD_GAP_S = 5.0         # possession is kept this long after the last time the ball was seen with a player
MIN_PLAYERS_FOR_SHAPE = 3    # a team needs this many players detected in a frame to measure its shape
BLOCK_S = 120                # length of the blocks used for the over-time charts
SPEED_GLITCH_KMH = 25.0      # fitted speeds above this are measurement glitches (the pipeline already drops them)


def _num(v, default=None):
    """Seconds from the game record: a number, 'mm:ss', 'h:mm:ss' or empty/NaN."""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return default
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            out = 0.0
            for p in parts:
                out = out * 60 + p
            return out
        return float(s)
    except (TypeError, ValueError):
        return default


def mmss(seconds):
    seconds = int(max(0, round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ------------------------------------------------------------------------------------------------ loading

def resolve_labels(db_path, names):
    """The game's team names may not be spelled exactly as the detections database labelled the teams (a team renamed in the
    game logger after the detections were run, e.g. 'Polis Lions' now, 'Polis' in the database). Returns {game team name: label in
    the database}: an exact match (ignoring capitals) first, then the closest-looking label, then whichever players' labels
    are most common."""
    import difflib
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT team, COUNT(*) FROM detected_objects WHERE class_name = 'player' AND team IS NOT NULL GROUP BY team ORDER BY 2 DESC").fetchall()
    finally:
        conn.close()
    labels = [r[0] for r in rows if r[0] not in ("Goalkeeper", "Referee", "")]
    if len(labels) < 2:
        raise ValueError(f"The detections database has fewer than two team labels ({', '.join(labels) or 'none'}), so a match report cannot be made.")
    out, free = {}, list(labels)
    for n in names:                                                       # exact first
        hit = next((l for l in free if l.strip().lower() == str(n).strip().lower()), None)
        if hit:
            out[n] = hit
            free.remove(hit)

    def score(n, l):
        n, l = str(n).lower(), l.lower()
        tn, tl = set(n.split()), set(l.split())
        return difflib.SequenceMatcher(None, n, l).ratio() + (0.5 if (n in l or l in n) else 0) + 0.3 * len(tn & tl)

    for n in names:
        if n in out:
            continue
        best = max(free, key=lambda l: score(n, l))
        if score(n, best) >= 0.55 or len(free) == len([x for x in names if x not in out]):
            out[n] = best
        else:
            out[n] = free[0]                                              # most common remaining label
        free.remove(out[n])
    return out


def load(db_path, team_names):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) if not str(db_path).startswith("file:") else sqlite3.connect(db_path, uri=True)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(detected_objects)")}
        id_col = "player_id" if "player_id" in cols else "tracker_id"
        df = pd.read_sql(
            f"SELECT frame_id, class_name, team, {id_col} AS pid, confidence, x_transformed_metres AS x, y_transformed_metres AS y, "
            "speed_km_per_hour AS spd, is_inside_pitch AS ins FROM detected_objects", conn)
        g = conn.execute("SELECT video_fps, detection_fps FROM game LIMIT 1").fetchone()
    finally:
        conn.close()
    video_fps = float(g[0]) if g and g[0] else 24.0
    det_fps = float(g[1]) if g and g[1] else 12.0
    df["t"] = df["frame_id"] / video_fps
    mapping = resolve_labels(db_path, team_names)                        # database label -> the game's team name
    df["team"] = df["team"].map({lab: name for name, lab in mapping.items()}).fillna(df["team"])
    return df, video_fps, det_fps, mapping


def _clean_ball(df):
    """One real ball position per frame: inside the pitch only, with places where a 'ball' keeps turning up all game
    (a marker, a spare ball, a reflection) taken out, and the most confident detection when there are several."""
    b = df[(df["class_name"] == "ball") & (df["ins"] == 1) & df["x"].notna()].copy()
    if b.empty:
        return b
    b["cx"] = (b["x"] / 0.5).round()
    b["cy"] = (b["y"] / 0.5).round()
    cell = b.groupby(["cx", "cy"])["t"].agg(["size", "min", "max"])
    fixed = cell[(cell["size"] >= 15) & ((cell["max"] - cell["min"]) >= 120)].index
    if len(fixed):
        keep = ~pd.MultiIndex.from_frame(b[["cx", "cy"]]).isin(fixed)
        b = b[keep]
    b = b.sort_values(["frame_id", "confidence"], ascending=[True, False]).groupby("frame_id", as_index=False).first()
    return b[["frame_id", "t", "x", "y"]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ pieces

def _possession(ball, players, teams, step, t0, t1):
    """Who has the ball, moment by moment. Returns the per-step owner (0 = team A, 1 = team B, -1 = nobody/unknown)."""
    n_steps = int(math.floor((t1 - t0) / step)) + 1
    owner = np.full(n_steps, -1, dtype=np.int8)
    events = []                                    # (t, team_index)
    if ball.empty:
        return owner, events
    pf = players[players["team"].isin(teams)].copy()
    pf["ti"] = pf["team"].map({teams[0]: 0, teams[1]: 1})
    by_frame = {k: v for k, v in pf.groupby("frame_id")}
    for fid, t, bx, by in ball[["frame_id", "t", "x", "y"]].itertuples(index=False):
        grp = by_frame.get(fid)
        if grp is None:
            continue
        d = np.hypot(grp["x"].to_numpy() - bx, grp["y"].to_numpy() - by)
        ti = grp["ti"].to_numpy()
        i = int(np.argmin(d))
        if d[i] > HOLD_RADIUS_M:
            continue
        other = d[ti != ti[i]]
        if other.size and other.min() - d[i] < CONTEST_MARGIN_M:
            continue
        events.append((float(t), int(ti[i])))
    # hold possession forward from each sighting until the next one (any team) or MAX_HOLD_GAP_S
    for k, (t, ti) in enumerate(events):
        nxt = events[k + 1][0] if k + 1 < len(events) else t1 + step
        end = min(nxt, t + MAX_HOLD_GAP_S)
        a = int(round((t - t0) / step)); b = int(round((end - t0) / step))
        owner[max(a, 0):max(min(b, n_steps), 0)] = ti
    return owner, events


def _runs(owner, step, t0):
    """Spells of possession: [(team, start_t, length_s), ...] from the per-step owner array."""
    spells, cur, start = [], -2, 0
    for i, o in enumerate(list(owner) + [-2]):
        if o != cur:
            if cur in (0, 1):
                spells.append((int(cur), t0 + start * step, (i - start) * step))
            cur, start = o, i
    return spells


def _hull_area(pts):
    pts = sorted(set(map(tuple, np.round(pts, 3))))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    return abs(sum(h[i][0] * h[(i + 1) % len(h)][1] - h[(i + 1) % len(h)][0] * h[i][1] for i in range(len(h)))) / 2


def _shape(players, team, t0, t1, step):
    """Per detection step: players detected, centroid, length (along the pitch), width (across), hull area."""
    rows = []
    for fid, grp in players[players["team"] == team].groupby("frame_id"):
        if len(grp) < MIN_PLAYERS_FOR_SHAPE:
            continue
        x, y = grp["x"].to_numpy(), grp["y"].to_numpy()
        rows.append((float(grp["t"].iloc[0]), len(grp), x.mean(), y.mean(), x.max() - x.min(), y.max() - y.min(),
                     _hull_area(np.c_[x, y]), float(np.mean(np.hypot(x - x.mean(), y - y.mean())))))
    return pd.DataFrame(rows, columns=["t", "n", "cx", "cy", "length", "width", "hull", "spread"])


def _blur(a, sigma_cells):
    """Gaussian blur with numpy only (no scipy needed)."""
    r = int(max(1, round(sigma_cells * 3)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def _heat(x, y, sigma=1.5):
    H, _, _ = np.histogram2d(x, y, bins=[int(PITCH_LENGTH), int(PITCH_WIDTH)], range=[[-HALF_L, HALF_L], [-HALF_W, HALF_W]])
    H = _blur(H, sigma)
    return (H / H.sum()) if H.sum() else H


def _running_events(pl, pid_col, threshold, det_fps):
    """Spells where one player holds a speed above the threshold for at least half a second."""
    min_n = max(2, int(round(det_fps / 2)))
    events = []
    hot = pl[pl["spd"] > threshold].sort_values([pid_col, "t"])
    for (team, pid), grp in hot.groupby(["team", pid_col]):
        t = grp["t"].to_numpy(); s = grp["spd"].to_numpy()
        start = 0
        for i in range(1, len(t) + 1):
            if i == len(t) or t[i] - t[i - 1] > 1.5 / det_fps + 1e-6:
                if i - start >= min_n:
                    events.append({"team": team, "t": float(t[start]), "end": float(t[i - 1]), "dur": float(t[i - 1] - t[start]) + 1 / det_fps,
                                   "max": float(s[start:i].max())})
                start = i
    return sorted(events, key=lambda e: e["t"])


# ------------------------------------------------------------------------------------------------ main entry

def compute_metrics(db_path, record, teams, running_speed_kmh=15.0, return_raw=False):
    """teams: the two team names [team A, team B] in the order the game record lists them."""
    df, video_fps, det_fps, mapping = load(db_path, teams)
    step = float(np.median(np.diff(sorted(df["frame_id"].unique())))) / video_fps
    t_first, t_last = float(df["t"].min()), float(df["t"].max())
    start = _num(record.get("game_start_seconds"), t_first)
    end = _num(record.get("game_end_seconds"), t_last)
    ht0, ht1 = _num(record.get("half_time_start_seconds")), _num(record.get("half_time_end_seconds"))
    start, end = max(start, t_first), min(end, t_last)
    in_game = (df["t"] >= start) & (df["t"] <= end)
    if ht0 is not None and ht1 is not None and ht1 > ht0:
        in_game &= ~((df["t"] > ht0) & (df["t"] < ht1))
    df = df[in_game]
    game_s = float(in_game.sum() and (end - start))
    if ht0 is not None and ht1 is not None and ht1 > ht0:
        game_s -= max(0.0, min(ht1, end) - max(ht0, start))
    nb = int(math.ceil((end - start) / BLOCK_S))
    blocks = [(start + i * BLOCK_S, min(start + (i + 1) * BLOCK_S, end)) for i in range(nb)]

    players = df[(df["class_name"] == "player") & (df["ins"] == 1) & df["team"].isin(teams) & df["x"].notna()].copy()
    ball = _clean_ball(df)
    pid_col = "pid"
    M = {"teams": list(teams), "team_labels": mapping, "start": start, "end": end, "duration": game_s, "video_fps": video_fps, "detection_fps": det_fps,
         "step": step, "running_threshold": running_speed_kmh, "block_s": BLOCK_S, "blocks": [[a, b] for a, b in blocks],
         "half_time": [ht0, ht1] if ht0 is not None and ht1 is not None else None}

    # ---- coverage (what the numbers rest on)
    n_frames = int(df["frame_id"].nunique())
    M["coverage"] = {"frames": n_frames, "ball_frames": int(len(ball)), "ball_pct": 100 * len(ball) / max(n_frames, 1),
                     "players_per_frame": {t: float(players[players.team == t].groupby("frame_id").size().reindex(df["frame_id"].unique(), fill_value=0).mean()) for t in teams}}

    # ---- possession
    owner, events = _possession(ball, players, teams, step, start, end)
    known = owner >= 0
    M["possession"] = {"known_pct": 100 * float(known.mean()) if len(owner) else 0.0,
                       "share": [float((owner == i).sum() / max(known.sum(), 1) * 100) for i in (0, 1)],
                       "seconds": [float((owner == i).sum() * step) for i in (0, 1)]}
    spells = _runs(owner, step, start)
    M["spells"] = [{"team": s[0], "t": s[1], "len": s[2]} for s in spells]
    M["possession"]["spell_count"] = [sum(1 for s in spells if s[0] == i) for i in (0, 1)]
    M["possession"]["avg_spell_s"] = [float(np.mean([s[2] for s in spells if s[0] == i])) if any(s[0] == i for s in spells) else 0.0 for i in (0, 1)]
    M["possession"]["longest_spell"] = [max(([s for s in spells if s[0] == i]), key=lambda s: s[2], default=None) for i in (0, 1)]
    won = [0, 0]
    for a, b in zip(spells, spells[1:]):
        if b[1] - (a[1] + a[2]) <= step * 1.5 and a[0] != b[0]:
            won[b[0]] += 1                                  # team b[0] took the ball from a[0]
    M["possession"]["won_from_opponent"] = won
    # by block
    pb = []
    for a, b in blocks:
        i0, i1 = int(round((a - start) / step)), int(round((b - start) / step))
        seg = owner[i0:i1]
        k = max(int((seg >= 0).sum()), 1)
        pb.append({"share": [float((seg == i).sum() / k * 100) for i in (0, 1)], "known": float((seg >= 0).mean() * 100) if len(seg) else 0.0})
    M["possession_blocks"] = pb
    # per minute timeline for the stacked chart
    mins = int(math.ceil((end - start) / 60))
    tl = []
    for m in range(mins):
        i0, i1 = int(round(m * 60 / step)), int(round((m + 1) * 60 / step))
        seg = owner[i0:i1]
        tl.append([float((seg == 0).sum() * step), float((seg == 1).sum() * step), float((seg < 0).sum() * step)])
    M["possession_minutes"] = tl

    # ---- territory (ball by third of the pitch, and who has it there)
    thirds = [(-HALF_L, -HALF_L / 3), (-HALF_L / 3, HALF_L / 3), (HALF_L / 3, HALF_L)]
    bt = [int(((ball["x"] >= lo) & (ball["x"] < hi + (1e-6 if hi == HALF_L else 0))).sum()) for lo, hi in thirds]
    M["territory"] = {"ball_third_pct": [100 * v / max(sum(bt), 1) for v in bt], "thirds": [[lo, hi] for lo, hi in thirds]}
    # possession by third: owner at each ball-seen step
    if len(ball):
        idx = np.clip(np.round((ball["t"].to_numpy() - start) / step).astype(int), 0, len(owner) - 1)
        ow = owner[idx]; bx = ball["x"].to_numpy()
        tp = [[0, 0], [0, 0], [0, 0]]
        for k, (lo, hi) in enumerate(thirds):
            sel = (bx >= lo) & (bx <= hi)
            for ti in (0, 1):
                tp[k][ti] = int((sel & (ow == ti)).sum())
        M["territory"]["possession_by_third"] = tp
        # ball position (x) smoothed per block, for the 'where was the game played' timeline
        M["territory"]["ball_x_blocks"] = [float(ball[(ball.t >= a) & (ball.t < b)]["x"].mean()) if ((ball.t >= a) & (ball.t < b)).any() else None for a, b in blocks]
        M["territory"]["ball_xy"] = [[float(r.t), float(r.x), float(r.y)] for r in ball.itertuples()]
    M["heat_ball"] = _heat(ball["x"].to_numpy(), ball["y"].to_numpy(), 1.5).tolist() if len(ball) else None

    # ---- team shape
    shapes = {}
    M["shape"] = {}
    M["heat_team"] = {}
    for i, tm in enumerate(teams):
        sh = _shape(players, tm, start, end, step)
        shapes[tm] = sh
        pl = players[players["team"] == tm]
        M["heat_team"][tm] = _heat(pl["x"].to_numpy(), pl["y"].to_numpy(), 1.5).tolist()
        if len(sh):
            roll = sh.set_index("t")[["cx", "cy", "length", "width", "hull", "spread"]].rolling(window=int(30 * det_fps), min_periods=int(5 * det_fps), center=True).mean()
            M["shape"][tm] = {"avg": {k: float(sh[k].mean()) for k in ("cx", "cy", "length", "width", "hull", "spread")},
                              "t": roll.index.to_list()[::int(det_fps)], **{k: [None if np.isnan(v) else float(v) for v in roll[k].to_numpy()[::int(det_fps)]] for k in roll.columns},
                              "frames": int(len(sh))}
    # distance between the two team centres per step
    if all(len(shapes[t]) for t in teams):
        a = shapes[teams[0]].set_index("t"); b = shapes[teams[1]].set_index("t")
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        sep = np.hypot(j["cx_a"] - j["cx_b"], j["cy_a"] - j["cy_b"])
        M["shape"]["separation"] = {"avg": float(sep.mean()), "t": j.index.to_list()[::int(det_fps)],
                                    "v": pd.Series(sep.to_numpy()).rolling(int(30 * det_fps), min_periods=int(5 * det_fps), center=True).mean().to_numpy()[::int(det_fps)].tolist()}

    # ---- distance and speed
    sp = players[(players["spd"].notna()) & (players["spd"] <= SPEED_GLITCH_KMH)]
    M["distance"], M["speed"] = {"blocks": {}, "per_player_m": {}, "team_total_m": {}}, {"bands": {}, "avg": {}, "p95": {}, "max": {}, "blocks": {}}
    edges = [0, 2, 6, 10, max(running_speed_kmh, 11), 99]
    M["speed"]["band_edges"] = edges
    for tm in teams:
        s = sp[sp["team"] == tm]
        per_block = []
        for a, b in blocks:
            seg = s[(s["t"] >= a) & (s["t"] < b)]
            m = float(seg["spd"].mean()) if len(seg) else 0.0
            per_block.append(m / 3.6 * (b - a))
        M["distance"]["blocks"][tm] = per_block
        M["distance"]["per_player_m"][tm] = float(sum(per_block))
        M["distance"]["team_total_m"][tm] = float(sum(per_block) * 5)
        hist, _ = np.histogram(s["spd"], bins=edges)
        M["speed"]["bands"][tm] = (hist / max(hist.sum(), 1) * 100).tolist()
        M["speed"]["avg"][tm] = float(s["spd"].mean()) if len(s) else 0.0
        M["speed"]["p95"][tm] = float(s["spd"].quantile(0.95)) if len(s) else 0.0
        M["speed"]["max"][tm] = float(s["spd"].max()) if len(s) else 0.0
        M["speed"]["blocks"][tm] = [float(s[(s["t"] >= a) & (s["t"] < b)]["spd"].mean()) if ((s["t"] >= a) & (s["t"] < b)).any() else None for a, b in blocks]

    # ---- running (above the walking-football limit)
    ev = _running_events(sp, pid_col, running_speed_kmh, det_fps)
    M["running"] = {"events": ev, "count": {tm: sum(1 for e in ev if e["team"] == tm) for tm in teams},
                    "blocks": {tm: [sum(1 for e in ev if e["team"] == tm and a <= e["t"] < b) for a, b in blocks] for tm in teams}}
    if return_raw:                                  # for the opposition report, which needs the frame-by-frame data too
        return M, {"players": players, "ball": ball, "owner": owner, "step": step, "start": start, "end": end,
                   "half_time": M["half_time"], "det_fps": det_fps, "sp": sp}
    return M
