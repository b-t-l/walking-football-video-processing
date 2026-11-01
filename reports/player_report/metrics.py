"""Player-report metrics: distance, speed, and pitch positions coloured by team possession.

Subjects come from identity (named roster players) and/or stitched track ids on a team.
Coordinates match the match report: metres on the overhead plan, x along the pitch, y across.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pandas as pd

from reports.match_report import metrics as mm

SPEED_GLITCH_KMH = mm.SPEED_GLITCH_KMH
HALF_L, HALF_W = mm.HALF_L, mm.HALF_W
THIRD = HALF_L / 3.0            # 6 m: edge of final third (u >= 6) / own third (u <= -6)
MAX_DOTS = 2500                 # subsample positions so the PDF stays light
MIN_TRACK_FRAMES = 24           # ignore short anonymous tracks when reporting a whole team
MAX_HOLD_GAP_S = 1.5            # gap between ball-hold samples that still counts as one spell
NEAR_RUN_FRAC = 0.80            # "close to running" = this fraction of the running limit … up to the limit


def _load_detections(db_path, team_names):
    """Like match-report load, plus ball_posession when the column exists."""
    df, video_fps, det_fps, mapping = mm.load(db_path, team_names)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(detected_objects)")}
        if "ball_posession" not in cols:
            df["has_ball"] = False
            return df, video_fps, det_fps, mapping
        id_col = "player_id" if "player_id" in cols else "tracker_id"
        bp = pd.read_sql(
            f"SELECT frame_id, {id_col} AS pid, ball_posession AS has_ball FROM detected_objects "
            f"WHERE class_name IN ('player','goalkeeper')",
            conn,
        )
    finally:
        conn.close()
    bp["has_ball"] = bp["has_ball"].fillna(0).astype(bool)
    df = df.merge(bp, on=["frame_id", "pid"], how="left")
    df["has_ball"] = df["has_ball"].fillna(False).astype(bool)
    return df, video_fps, det_fps, mapping


def _possession_spells(g, det_fps):
    """Spells where this player is the ball holder (ball_posession).

    Returns dict: count, min_s, max_s, avg_s, total_s (empty-safe).
    """
    empty = {"count": 0, "min_s": None, "max_s": None, "avg_s": None, "total_s": 0.0}
    if g is None or g.empty or "has_ball" not in g.columns:
        return empty
    hold = g[g["has_ball"]].sort_values("t")
    if hold.empty:
        return empty
    t = hold["t"].to_numpy(dtype=float)
    gap = MAX_HOLD_GAP_S
    frame_dt = 1.0 / max(float(det_fps), 1e-6)
    spells = []
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] - t[i - 1] > gap:
            # duration: last - first + one detection tick
            dur = float(t[i - 1] - t[start]) + frame_dt
            if dur > 0:
                spells.append(dur)
            start = i
    if not spells:
        return empty
    return {
        "count": len(spells),
        "min_s": float(min(spells)),
        "max_s": float(max(spells)),
        "avg_s": float(np.mean(spells)),
        "total_s": float(sum(spells)),
    }


def _count_band_spells(t, band, target, gap):
    """Number of contiguous spells where `band == target` (gaps > gap start a new spell)."""
    spells = 0
    start = None
    for i in range(len(band)):
        if band[i] == target:
            if start is None:
                start = i
            elif i > 0 and (t[i] - t[i - 1]) > gap:
                spells += 1
                start = i
        elif start is not None:
            spells += 1
            start = None
    if start is not None:
        spells += 1
    return spells


def _speed_bands(g, running_kmh, det_fps=None):
    """How many times the player was close to / at the running limit (spell counts).

    A spell is a run of detections in that band; gaps longer than MAX_HOLD_GAP_S start a new one.
    """
    run = float(running_kmh or 15.0)
    near = max(run * NEAR_RUN_FRAC, run - 3.0)
    out = {
        "close": 0,
        "running": 0,
        "near_kmh": near,
        "run_kmh": run,
        "total": 0,
    }
    if g is None or getattr(g, "empty", True) or "spd" not in getattr(g, "columns", []):
        return out
    gg = g[(g["spd"].notna()) & (g["spd"] <= SPEED_GLITCH_KMH)].sort_values("t")
    if gg.empty:
        return out
    t = gg["t"].to_numpy(dtype=float)
    s = gg["spd"].to_numpy(dtype=float)
    # 1 = close to running, 2 = at/above running limit
    band = np.where(s >= run, 2, np.where(s >= near, 1, 0)).astype(np.int8)
    gap = MAX_HOLD_GAP_S
    close = _count_band_spells(t, band, 1, gap)
    running = _count_band_spells(t, band, 2, gap)
    out["close"] = int(close)
    out["running"] = int(running)
    out["total"] = int(close + running)
    return out


def _final_third_team_stats(ball, players_teams, owner, step, start, team_index, attack_periods, det_fps):
    """How often the ball entered the final third while this team had possession.

    Also estimates passes in the build-up: distinct ball-holder changes on this team
    between leaving midfield and entering the final third (needs ball_posession).
    """
    out = {
        "entries": 0,
        "passes_to_enter": [],       # per entry
        "passes_avg": None,
        "passes_min": None,
        "passes_max": None,
    }
    if ball is None or ball.empty or team_index is None or not len(owner):
        return out
    bx = ball["x"].to_numpy(dtype=float)
    by = ball["y"].to_numpy(dtype=float)
    bt = ball["t"].to_numpy(dtype=float)
    bf = ball["frame_id"].to_numpy()
    bu, _, _ = _to_attacking_coords(bx, by, bt, attack_periods, team_index)
    idx = np.clip(np.round((bt - start) / max(step, 1e-6)).astype(int), 0, len(owner) - 1)
    ow = owner[idx]
    in_final = bu >= THIRD
    team_has = ow == team_index

    # Holder pid per frame (for pass counting)
    holders = {}
    if "has_ball" in players_teams.columns:
        hold = players_teams[players_teams["has_ball"] & players_teams["team"].notna()]
        # restrict to this team
        # team name resolved outside — filter by ti via mapping later; pass team mask from caller
        for fid, pid in hold[["frame_id", "pid"]].itertuples(index=False):
            holders[int(fid)] = int(pid) if pid is not None and not (isinstance(pid, float) and math.isnan(pid)) else None

    entries = 0
    pass_counts = []
    prev_in = False
    build_holders = []
    for i in range(len(bu)):
        now_in = bool(in_final[i] and team_has[i])
        if team_has[i] and not in_final[i]:
            hid = holders.get(int(bf[i]))
            if hid is not None:
                if not build_holders or build_holders[-1] != hid:
                    build_holders.append(hid)
        if now_in and not prev_in and team_has[i]:
            entries += 1
            # passes ≈ holder changes during the build-up (n_holders - 1)
            n_pass = max(0, len(build_holders) - 1)
            pass_counts.append(n_pass)
            build_holders = []
        if not team_has[i]:
            build_holders = []
        prev_in = now_in

    out["entries"] = entries
    out["passes_to_enter"] = pass_counts
    if pass_counts:
        out["passes_avg"] = float(np.mean(pass_counts))
        out["passes_min"] = int(min(pass_counts))
        out["passes_max"] = int(max(pass_counts))
    return out


# ------------------------------------------------------------------------------------------------ targets

def _load_identity_assignments(game_id):
    """[(roster_player_id, name, team_id, track_player_id, from_frame, to_frame), ...] from the game logger DB."""
    try:
        from game_logger import db as gl_db
    except Exception:
        return []
    conn = gl_db.connect()
    try:
        rows = conn.execute(
            """
            SELECT a.roster_player_id, p.name, p.team_id, a.track_player_id, a.from_frame, a.to_frame
            FROM identity_assignments a
            JOIN players p ON p.player_id = a.roster_player_id
            WHERE a.game_id = ?
            ORDER BY p.name COLLATE NOCASE, a.from_frame, a.id
            """,
            (int(game_id),),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _team_name_for_id(team_id):
    try:
        from game_logger import db as gl_db, repo
        conn = gl_db.connect()
        try:
            t = repo.get_team(conn, team_id)
            return t["name"] if t else None
        finally:
            conn.close()
    except Exception:
        return None


def list_report_targets(game_id, record):
    """Choices for the Run panel / CLI: named players + whole-team options."""
    teams = [record.get("team_a"), record.get("team_b")]
    out = []
    for tm in teams:
        if tm:
            out.append({"id": f"team:{tm}", "label": f"Whole team · {tm}", "kind": "team", "team": tm})
    asg = _load_identity_assignments(game_id)
    by_roster = {}
    for a in asg:
        by_roster.setdefault(a["roster_player_id"], a["name"])
    if by_roster:
        out.append({"id": "named", "label": "All named players", "kind": "named"})
        for rid, name in sorted(by_roster.items(), key=lambda kv: (kv[1] or "").lower()):
            out.append({"id": f"roster:{rid}", "label": name, "kind": "roster", "roster_player_id": rid})
    return out


def resolve_subjects(db_path, record, target, players_df):
    """Turn a target string into a list of subjects to report on.

    target:
      team:<name>     — one subject per long-enough track on that team
      named           — one subject per named roster player (identity links)
      roster:<id>     — one named roster player
      track:<id>      — one stitched track id
    """
    target = (target or "named").strip()
    game_id = record.get("game_id")
    subjects = []

    if target.lower() == "named" or target.lower().startswith("roster:"):
        asg = _load_identity_assignments(game_id)
        if target.lower().startswith("roster:"):
            want = int(target.split(":", 1)[1])
            asg = [a for a in asg if int(a["roster_player_id"]) == want]
        by = {}
        for a in asg:
            rid = int(a["roster_player_id"])
            sub = by.setdefault(rid, {
                "key": f"roster:{rid}",
                "label": a["name"] or f"Player {rid}",
                "kind": "roster",
                "roster_player_id": rid,
                "team": _team_name_for_id(a.get("team_id")),
                "segments": [],
            })
            sub["segments"].append({
                "track_player_id": int(a["track_player_id"]),
                "from_frame": int(a["from_frame"]),
                "to_frame": int(a["to_frame"]) if a["to_frame"] is not None else None,
            })
        subjects = list(by.values())
        # Prefer the game's team name spelling when identity team matches a side
        teams = [record.get("team_a"), record.get("team_b")]
        for s in subjects:
            if not s["team"]:
                continue
            for tm in teams:
                if tm and tm.strip().lower() == s["team"].strip().lower():
                    s["team"] = tm
        if not subjects:
            raise ValueError(
                "No named players to report on. Name players on Track Players first, "
                "or use --player-report 'team:<Team Name>' for anonymous tracks."
            )
        return subjects

    if target.lower().startswith("track:"):
        tid = int(target.split(":", 1)[1])
        hit = players_df[players_df["pid"] == tid]
        team = str(hit["team"].mode().iloc[0]) if len(hit) and hit["team"].notna().any() else None
        return [{
            "key": f"track:{tid}",
            "label": f"Track #{tid}",
            "kind": "track",
            "team": team,
            "segments": [{"track_player_id": tid, "from_frame": None, "to_frame": None}],
        }]

    if target.lower().startswith("team:"):
        team_name = target.split(":", 1)[1].strip()
        pl = players_df[players_df["team"] == team_name]
        if pl.empty:
            mask = players_df["team"].astype(str).str.strip().str.lower() == team_name.lower()
            pl = players_df[mask]
            if len(pl):
                team_name = str(pl["team"].iloc[0])
        if pl.empty:
            raise ValueError(f"No player detections found for team '{team_name}'.")
        # Prefer named roster players whose linked tracks appear on this team
        asg = _load_identity_assignments(game_id)
        by = {}
        for a in asg:
            tid = int(a["track_player_id"])
            fr = int(a["from_frame"])
            to = int(a["to_frame"]) if a["to_frame"] is not None else None
            seg = pl[pl["pid"] == tid]
            if seg.empty:
                continue
            seg = seg[seg["frame_id"] >= fr]
            if to is not None:
                seg = seg[seg["frame_id"] <= to]
            if seg.empty:
                continue
            rid = int(a["roster_player_id"])
            sub = by.setdefault(rid, {
                "key": f"roster:{rid}",
                "label": a["name"] or f"Player {rid}",
                "kind": "roster",
                "roster_player_id": rid,
                "team": team_name,
                "segments": [],
            })
            sub["segments"].append({"track_player_id": tid, "from_frame": fr, "to_frame": to})
        if by:
            return list(by.values())
        # Anonymous: one subject per long track on the team
        for pid, g in pl.groupby("pid"):
            if pid is None or (isinstance(pid, float) and math.isnan(pid)):
                continue
            if len(g) < MIN_TRACK_FRAMES:
                continue
            subjects.append({
                "key": f"track:{int(pid)}",
                "label": f"Track #{int(pid)}",
                "kind": "track",
                "team": team_name,
                "segments": [{"track_player_id": int(pid), "from_frame": None, "to_frame": None}],
            })
        if not subjects:
            raise ValueError(f"No long-enough tracks for team '{team_name}'.")
        subjects.sort(key=lambda s: s["label"])
        return subjects

    raise ValueError(
        f"Unknown player-report target '{target}'. "
        "Use named, roster:<id>, team:<Team Name>, or track:<id>."
    )


# ------------------------------------------------------------------------------------------------ samples

def _mask_segments(df, segments):
    """Rows of df that fall in any (track_id, from_frame, to_frame) segment."""
    if not segments:
        return df.iloc[0:0]
    parts = []
    for seg in segments:
        tid = int(seg["track_player_id"])
        m = df["pid"] == tid
        fr, to = seg.get("from_frame"), seg.get("to_frame")
        if fr is not None:
            m &= df["frame_id"] >= int(fr)
        if to is not None:
            m &= df["frame_id"] <= int(to)
        parts.append(df[m])
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["frame_id", "pid"])


def _path_distance_m(g):
    """Metres travelled along consecutive detections, skipping glitch jumps."""
    if g.empty:
        return 0.0
    g = g.sort_values(["pid", "frame_id"])
    total = 0.0
    for _, part in g.groupby("pid"):
        part = part.sort_values("frame_id")
        dx = part["x"].to_numpy(dtype=float)
        dy = part["y"].to_numpy(dtype=float)
        tt = part["t"].to_numpy(dtype=float)
        if len(dx) < 2:
            continue
        step = np.hypot(np.diff(dx), np.diff(dy))
        dt = np.diff(tt)
        # drop jumps that imply > SPEED_GLITCH_KMH
        ok = (dt > 0) & ((step / np.maximum(dt, 1e-6)) * 3.6 <= SPEED_GLITCH_KMH)
        total += float(step[ok].sum())
    return total


def _subsample_xy(x, y, flags, n_max=MAX_DOTS):
    n = len(x)
    if n <= n_max:
        return x, y, flags
    idx = np.linspace(0, n - 1, n_max).astype(int)
    return x[idx], y[idx], flags[idx]


def _playable_seconds(start, end, ht0, ht1):
    """Seconds of play in the analysis window (half-time removed)."""
    game_s = max(0.0, float(end) - float(start))
    if ht0 is not None and ht1 is not None and ht1 > ht0:
        game_s -= max(0.0, min(ht1, end) - max(ht0, start))
    return max(game_s, 1e-6)


def _accuracy_from_coverage(coverage_pct, n_tracks):
    """Turn tracking coverage into a report-confidence score and band.

    Primary signal: share of play where this player has a detection (missing frames hurt).
    Light penalty when one named person is split across many track ids (identity breaks).
    """
    cov = float(np.clip(coverage_pct, 0.0, 100.0))
    # Soft penalty: more than 2 track fragments → up to ~12 points off
    frag = max(0, int(n_tracks) - 2)
    penalty = min(12.0, frag * 3.0)
    score = float(np.clip(cov - penalty, 0.0, 100.0))
    if score >= 70:
        band, blurb = "High", "Tracked for most of the game — figures should be reliable."
    elif score >= 45:
        band, blurb = "Fair", "Tracked for a fair share of the game — use figures with some caution."
    elif score >= 25:
        band, blurb = "Limited", "Missing for long stretches — distance and heat map are incomplete."
    else:
        band, blurb = "Low", "Rarely tracked — this report is a rough sketch only."
    return {
        "score": round(score, 1),
        "band": band,
        "blurb": blurb,
        "coverage_pct": round(cov, 1),
        "track_penalty": round(penalty, 1),
    }


def _attack_periods(db_path, players_df, teams, start, end, half_time):
    """[(t0, t1, team_a_attacks +1/-1), ...] — prefer stored direction_of_play, else estimate like the opposition report.

    If the game has no measured half-time, there is only one period: no end-switch for the whole game.
    """
    has_half = (
        half_time is not None
        and half_time[0] is not None
        and half_time[1] is not None
        and half_time[1] > half_time[0]
    )
    try:
        from direction_of_play.direction_of_play import load_direction
        stored = load_direction(db_path)
    except Exception:
        stored = None
    if stored and stored.get("periods"):
        periods = [(float(p["t_start"]), float(p["t_end"]), int(p["team_a_attacks"])) for p in stored["periods"]]
        if not has_half:
            # Game logger has no half-time → ignore any second-half flip stored earlier
            d = periods[0][2]
            return [(float(start), float(end), d)]
        return periods

    from reports.opposition_report.scout import _periods
    # No half-time → scout returns one period (no end-switch)
    return _periods(players_df, teams, start, end, half_time if has_half else None)


def _attack_dir_for_times(t, periods, team_index):
    """Per-sample attack direction for this team: +1 = toward +x on the video plan."""
    t = np.asarray(t, dtype=float)
    if not periods:
        return np.ones(len(t), dtype=float)
    # Default to first period; team B is opposite of team A
    d0 = periods[0][2] if team_index == 0 else -periods[0][2]
    out = np.full(len(t), float(d0))
    for a, b, da in periods:
        sign = da if team_index == 0 else -da
        out[(t >= a) & (t <= b)] = float(sign)
    return out


def _to_attacking_coords(x, y, t, periods, team_index):
    """Map x/y so this team attacks left→right (flip only when half-time creates a second period)."""
    if team_index is None or not len(x):
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), False
    d = _attack_dir_for_times(t, periods, team_index)
    # True when we applied any attack-orientation mapping (including a single-period whole-game map)
    return np.asarray(x, dtype=float) * d, np.asarray(y, dtype=float) * d, True


def compute_player_reports(db_path, record, teams, target="named"):
    """Build metrics for each subject selected by `target`."""
    df, video_fps, det_fps, mapping = _load_detections(db_path, teams)
    step = float(np.median(np.diff(sorted(df["frame_id"].unique())))) / video_fps if df["frame_id"].nunique() > 1 else 1.0 / max(det_fps, 1)
    t_first, t_last = float(df["t"].min()), float(df["t"].max())
    start = mm._num(record.get("game_start_seconds"), t_first)
    end = mm._num(record.get("game_end_seconds"), t_last)
    ht0, ht1 = mm._num(record.get("half_time_start_seconds")), mm._num(record.get("half_time_end_seconds"))
    start, end = max(start, t_first), min(end, t_last)
    in_game = (df["t"] >= start) & (df["t"] <= end)
    if ht0 is not None and ht1 is not None and ht1 > ht0:
        in_game &= ~((df["t"] > ht0) & (df["t"] < ht1))
    df = df[in_game].copy()
    playable_s = _playable_seconds(start, end, ht0, ht1)
    # Detection frames available in the window (what continuous tracking would look like)
    expected_frames = max(1, int(round(playable_s * float(det_fps))))
    try:
        running_kmh = float(record.get("running_speed_km_h") or 15)
    except (TypeError, ValueError):
        running_kmh = 15.0

    players = df[(df["class_name"].isin(["player", "goalkeeper"])) & (df["ins"] == 1) & df["x"].notna()].copy()
    # Keep only the two match teams for possession / team filtering
    players_teams = players[players["team"].isin(teams)].copy()
    ball = mm._clean_ball(df)
    owner, _ = mm._possession(ball, players_teams, teams, step, start, end)

    half_time = [ht0, ht1] if ht0 is not None and ht1 is not None and ht1 > ht0 else None
    attack_periods = _attack_periods(db_path, players_teams, teams, start, end, half_time)

    subjects = resolve_subjects(db_path, record, target, players_teams)
    team_index = {tm: i for i, tm in enumerate(teams)}

    reports = []
    for sub in subjects:
        g = _mask_segments(players_teams, sub["segments"])
        if g.empty:
            # named player might have segments but no inside-pitch points in range
            g = _mask_segments(players, sub["segments"])
        team = sub.get("team")
        if not team and len(g) and g["team"].notna().any():
            team = str(g["team"].mode().iloc[0])
            sub["team"] = team
        ti = team_index.get(team)
        if ti is None and team:
            # case-insensitive match to game team list
            for i, tm in enumerate(teams):
                if tm and str(tm).strip().lower() == str(team).strip().lower():
                    ti = i
                    team = tm
                    sub["team"] = tm
                    break

        # Possession flag per sample: 1 = own team has ball, 0 = opposition, -1 = unknown
        if len(g) and len(owner) and ti is not None:
            idx = np.clip(np.round((g["t"].to_numpy() - start) / max(step, 1e-6)).astype(int), 0, len(owner) - 1)
            ow = owner[idx]
            flags = np.where(ow < 0, -1, np.where(ow == ti, 1, 0)).astype(np.int8)
        else:
            flags = np.full(len(g), -1, dtype=np.int8)

        spd = g[(g["spd"].notna()) & (g["spd"] <= SPEED_GLITCH_KMH)]["spd"]
        speed_bands = _speed_bands(g, running_kmh, det_fps)
        ball_spells = _possession_spells(g, det_fps)
        # Distance uses raw path (absolute pitch metres) — flipping halves would cancel on a round trip
        dist_m = _path_distance_m(g)
        n_frames = int(g["frame_id"].nunique()) if len(g) else 0
        # Each unique detection frame ≈ 1/det_fps seconds of “seen” time
        present_s = float(n_frames) / max(float(det_fps), 1e-6)
        coverage_pct = 100.0 * min(1.0, present_s / playable_s)
        tracks = sorted({int(s["track_player_id"]) for s in sub["segments"]})
        accuracy = _accuracy_from_coverage(coverage_pct, len(tracks))
        # Span from first to last sighting (can be long even when sparse)
        tracked_span_s = float(g["t"].max() - g["t"].min()) if len(g) else 0.0
        n_in = int((flags == 1).sum())
        n_out = int((flags == 0).sum())
        n_unk = int((flags == -1).sum())

        # Pitch dots: rotate each half so this team always attacks toward +x (right)
        xs_raw = g["x"].to_numpy(dtype=float) if len(g) else np.array([])
        ys_raw = g["y"].to_numpy(dtype=float) if len(g) else np.array([])
        ts = g["t"].to_numpy(dtype=float) if len(g) else np.array([])
        fids = g["frame_id"].to_numpy() if len(g) else np.array([])
        xs, ys, flipped = _to_attacking_coords(xs_raw, ys_raw, ts, attack_periods, ti)

        # Ball location in the same attacking frame (for "ball in final third" filters)
        ball_u_by_frame = {}
        if len(ball) and ti is not None:
            bx = ball["x"].to_numpy(dtype=float)
            by = ball["y"].to_numpy(dtype=float)
            bt = ball["t"].to_numpy(dtype=float)
            bu, _, _ = _to_attacking_coords(bx, by, bt, attack_periods, ti)
            for fid, u in zip(ball["frame_id"].to_numpy(), bu):
                ball_u_by_frame[int(fid)] = float(u)

        ball_u = np.full(len(g), np.nan, dtype=float)
        for i, fid in enumerate(fids):
            u = ball_u_by_frame.get(int(fid))
            if u is not None:
                ball_u[i] = u

        def _pack(mask):
            if not len(xs) or not np.any(mask):
                return {"x": [], "y": []}
            n = int(np.sum(mask))
            xx, yy, _ = _subsample_xy(xs[mask], ys[mask], np.zeros(n, dtype=np.int8))
            return {"x": xx.tolist(), "y": yy.tolist()}

        xs_all, ys_all, fl = _subsample_xy(xs, ys, flags)
        # In possession + ball third (attacking coords: final = right, own = left)
        in_pos = flags == 1
        has_ball = np.isfinite(ball_u)
        in_final = in_pos & has_ball & (ball_u >= THIRD)
        in_middle = in_pos & has_ball & (ball_u > -THIRD) & (ball_u < THIRD)
        in_own = in_pos & has_ball & (ball_u <= -THIRD)
        # Out of possession + ball in this team's defensive third (pressure / deep defending)
        out_own = (flags == 0) & has_ball & (ball_u <= -THIRD)

        reports.append({
            "subject": sub,
            "team": team,
            "n_samples": int(len(g)),
            "n_frames": n_frames,
            "expected_frames": expected_frames,
            "present_s": present_s,
            "playable_s": playable_s,
            "tracked_s": present_s,          # time actually covered by detections
            "tracked_span_s": tracked_span_s,
            "distance_m": float(dist_m),
            "speed_avg": float(spd.mean()) if len(spd) else None,
            "speed_max": float(spd.max()) if len(spd) else None,
            "speed_p95": float(spd.quantile(0.95)) if len(spd) else None,
            "possession_in_pct": 100.0 * n_in / max(n_in + n_out, 1),
            "possession_out_pct": 100.0 * n_out / max(n_in + n_out, 1),
            "possession_known_pct": 100.0 * (n_in + n_out) / max(n_in + n_out + n_unk, 1),
            "n_in_final_third": int(in_final.sum()),
            "n_in_middle_third": int(in_middle.sum()),
            "n_in_own_third": int(in_own.sum()),
            "n_out_own_third": int(out_own.sum()),
            "accuracy": accuracy,
            "coords_attacking": flipped,
            "speed_bands": speed_bands,
            "ball_spells": ball_spells,
            "dots_in": {"x": xs_all[fl == 1].tolist(), "y": ys_all[fl == 1].tolist()},
            "dots_out": {"x": xs_all[fl == 0].tolist(), "y": ys_all[fl == 0].tolist()},
            "dots_unk": {"x": xs_all[fl < 0].tolist(), "y": ys_all[fl < 0].tolist()},
            "dots_in_final": _pack(in_final),
            "dots_in_middle": _pack(in_middle),
            "dots_in_own": _pack(in_own),
            "dots_out_own": _pack(out_own),
            "tracks": tracks,
        })

    # Team-level: ball entries into the final third while that team had possession
    final_third_by_team = {}
    for i, tm in enumerate(teams):
        # holders filtered to this team for pass estimates
        pt = players_teams[players_teams["team"] == tm] if len(players_teams) else players_teams
        final_third_by_team[tm] = _final_third_team_stats(
            ball, pt, owner, step, start, i, attack_periods, det_fps)

    return {
        "teams": list(teams),
        "team_labels": mapping,
        "start": start,
        "end": end,
        "half_time": half_time,
        "has_half_time": bool(half_time),
        "playable_s": playable_s,
        "attack_periods": [[a, b, d] for a, b, d in attack_periods],
        "running_kmh": running_kmh,
        "final_third_by_team": final_third_by_team,
        "target": target,
        "video_fps": video_fps,
        "detection_fps": det_fps,
        "step": step,
        "reports": reports,
    }
