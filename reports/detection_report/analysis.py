"""Measures how well each detection stage worked on one game, using only what is in the detections database.

There is no 'right answer' to compare with, so every figure is a PROXY: how complete the detections are (are all 10 players
found in each frame?), how steady they are (does the same player keep the same track and team?), and how plausible they are
(does anything jump across the pitch?). The review sheet (review.py) adds a human check on a sample of frames.

Positions are in metres on the 36 x 20 m pitch, origin in the centre."""
import math
import sqlite3

import numpy as np
import pandas as pd

from ..match_report import metrics
from ..match_report.metrics import HALF_L, HALF_W

EXPECTED_PER_TEAM = 5           # players per team on the pitch (five-a-side)
TARGET_BALL_COVERAGE = 0.70     # share of frames in which the ball should be found for the reports to work well
LONG_TRACK_S = 60.0             # an identity lasting this long counts as a 'steady' identity
MAX_GAP_S = 3.0                 # a player missing for longer than this is treated as gone, not as a missed detection
JUMP_KMH = 25.0                 # moving faster than this between two detections of one player = a jump (a wrong id or position)
BALL_JUMP_MS = 30.0             # the ball cannot move this fast between two sightings
NEAR_PLAYER_M = 2.0             # a real ball in play is usually this close to somebody
EDGE_M = 2.0                    # a track that ends this close to the boundary probably left the pitch
GOOD, FAIR = 85.0, 60.0         # score bands: good, fair, poor

# how much each stage matters to the numbers in our reports (adds up to 1). Each weight is the share of the report numbers
# that rest on the stage: the ball drives possession, territory, turnovers and fast breaks; the players drive shape, space and
# heat maps; tracking drives distance, speed and running; team labels touch every team figure.
STAGE_WEIGHTS = {"ball": 0.30, "players": 0.25, "tracking": 0.20, "teams": 0.15, "positions": 0.07, "goalkeepers": 0.03}
STAGE_NAMES = {"players": "Player detection", "ball": "Ball detection", "goalkeepers": "Goalkeepers", "tracking": "Tracking and identities",
               "teams": "Team assignment", "positions": "Pitch positions"}
STAGE_FEEDS = {"ball": "possession, territory, turnovers, fast breaks",
               "players": "team shape, space given, compactness, heat maps",
               "tracking": "distance, speed, running events, fade",
               "teams": "every team figure",
               "positions": "every figure in metres",
               "goalkeepers": "shape and possession near the goals"}


def grade(score):
    return "good" if score >= GOOD else "fair" if score >= FAIR else "poor"


def _load(db_path, teams):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(detected_objects)")}
        pid = "player_id" if "player_id" in cols else "tracker_id"
        raw = "team_raw" if "team_raw" in cols else "team"
        df = pd.read_sql(
            f"SELECT frame_id, class_name, confidence, tracker_id, {pid} AS pid, team, {raw} AS team_raw, team_color AS colour, "
            "x_transformed_metres AS x, y_transformed_metres AS y, speed_km_per_hour AS spd, is_inside_pitch AS ins, "
            "xmin, ymin, xmax, ymax FROM detected_objects", conn)
        g = conn.execute("SELECT video_fps, detection_fps, video FROM game LIMIT 1").fetchone()
        try:
            n_pitch = conn.execute("SELECT COUNT(DISTINCT frame_id) FROM detected_pitch").fetchone()[0]
            kp_names = [r[0] for r in conn.execute("SELECT name, COUNT(*) FROM detected_pitch GROUP BY name")]
            kp_conf = conn.execute("SELECT MIN(confidence), AVG(confidence) FROM detected_pitch").fetchone()
        except sqlite3.DatabaseError:
            n_pitch, kp_names, kp_conf = 0, [], (None, None)
    finally:
        conn.close()
    vfps = float(g[0]) if g and g[0] else 24.0
    dfps = float(g[1]) if g and g[1] else 12.0
    video = g[2] if g else None
    mapping = metrics.resolve_labels(db_path, teams)                    # game team name -> label in the database
    inv = {lab: name for name, lab in mapping.items()}
    df["team"] = df["team"].map(inv).fillna(df["team"])
    df["team_raw"] = df["team_raw"].map(inv).fillna(df["team_raw"])
    df["t"] = df["frame_id"] / vfps
    return df, vfps, dfps, video, mapping, {"frames": n_pitch, "names": kp_names, "conf": kp_conf}


def _hist2d(x, y, sigma=1.5):
    H, _, _ = np.histogram2d(x, y, bins=[int(2 * HALF_L), int(2 * HALF_W)], range=[[-HALF_L, HALF_L], [-HALF_W, HALF_W]])
    return metrics._blur(H, sigma).tolist() if H.sum() else None


def _runs_of(mask):
    """Start index and length of each run of True in a boolean array."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s, e - s))


def analyse(db_path, record, teams):
    df, vfps, dfps, video, mapping, kp = _load(db_path, teams)
    frames_all = np.sort(df["frame_id"].unique())
    stride = int(round(float(np.median(np.diff(frames_all))))) if len(frames_all) > 1 else 2
    step = stride / vfps
    t_first, t_last = float(df["t"].min()), float(df["t"].max())
    start = metrics._num(record.get("game_start_seconds"), t_first)
    end = metrics._num(record.get("game_end_seconds"), t_last)
    ht0, ht1 = metrics._num(record.get("half_time_start_seconds")), metrics._num(record.get("half_time_end_seconds"))
    start, end = max(start, t_first), min(end, t_last)
    keep = (df["t"] >= start) & (df["t"] <= end)
    if ht0 is not None and ht1 is not None and ht1 > ht0:
        keep &= ~((df["t"] > ht0) & (df["t"] < ht1))
    df = df[keep].copy()
    fr = np.sort(df["frame_id"].unique())
    ft = fr / vfps
    nF = len(fr)
    A = {"teams": list(teams), "team_labels": mapping, "video": video, "video_fps": vfps, "detection_fps": dfps, "stride": stride,
         "start": float(start), "end": float(end), "frames": int(nF), "duration": float(nF * step),
         "frame_ids": fr.tolist() if nF <= 400000 else [], "expected_per_team": EXPECTED_PER_TEAM}
    minute = ((ft - start) // 60).astype(int)
    nmin = int(minute.max()) + 1 if nF else 0
    A["minutes"] = nmin
    fmin = pd.Series(minute, index=fr)                                      # frame_id -> minute of the game window
    per_min_frames = np.bincount(minute, minlength=nmin)
    A["empty_minutes"] = [int(i) for i in np.where(per_min_frames == 0)[0]]          # minutes with no frames at all (half time was skipped)

    def per_minute(series_by_frame):
        s = series_by_frame.reindex(fr, fill_value=0).to_numpy(dtype=float)
        v = np.bincount(minute, weights=s, minlength=nmin) / np.maximum(per_min_frames, 1)
        return [None if per_min_frames[i] == 0 else float(v[i]) for i in range(nmin)]

    def blank_empty(lst):
        return [None if per_min_frames[i] == 0 else v for i, v in enumerate(lst)]

    # ------------------------------------------------------------------ players
    P = df[(df["class_name"] == "player") & df["x"].notna()]
    team_pl = P[P["team"].isin(teams)]
    ins = team_pl[team_pl["ins"] == 1]
    cnt = {t: ins[ins["team"] == t].groupby("frame_id").size().reindex(fr, fill_value=0) for t in teams}
    tot = cnt[teams[0]] + cnt[teams[1]]
    need = 2 * EXPECTED_PER_TEAM
    comp = (np.minimum(cnt[teams[0]], EXPECTED_PER_TEAM) + np.minimum(cnt[teams[1]], EXPECTED_PER_TEAM)) / need
    dist = tot.clip(upper=need + 3).value_counts().sort_index()
    conf = team_pl["confidence"].to_numpy()
    cut = math.floor(float(conf.min()) * 20) / 20 if len(conf) else 0.0       # the cut-off the detector used (lowest confidence kept)
    near = float(((conf >= cut) & (conf < cut + 0.10)).mean()) if len(conf) else 0.0
    # detections missing inside a track: a player seen before and after a short gap
    miss_xy, missing, slots = [], 0, 0
    ordered = ins.sort_values(["pid", "frame_id"])
    for pid, g in ordered.groupby("pid"):
        f = g["frame_id"].to_numpy()
        if len(f) < 2:
            slots += len(f)
            continue
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        gaps = np.diff(f)
        slots += int((f[-1] - f[0]) // stride) + 1
        for k in np.where((gaps > stride) & (gaps <= MAX_GAP_S * vfps))[0]:
            n_missing = int(gaps[k] // stride) - 1
            missing += n_missing
            for j in range(1, n_missing + 1):
                a = j / (n_missing + 1)
                miss_xy.append((x[k] + a * (x[k + 1] - x[k]), y[k] + a * (y[k + 1] - y[k])))
    miss_xy = np.array(miss_xy) if miss_xy else np.zeros((0, 2))
    A["players"] = {
        "per_frame_mean": float(tot.mean()), "per_frame_team": {t: float(cnt[t].mean()) for t in teams},
        "completeness": float(comp.mean()), "all_found_pct": float((tot >= need).mean() * 100),
        "dist": {int(k): int(v) for k, v in dist.items()}, "extra_pct": float((tot > need).mean() * 100),
        "below8_pct": float((tot < 8).mean() * 100),
        "per_minute": per_minute(tot), "per_minute_comp": per_minute(comp),
        "conf_cut": cut, "conf_near_cut_pct": near * 100, "conf_hist": np.histogram(conf, bins=np.arange(0.3, 1.0001, 0.025))[0].tolist() if len(conf) else [],
        "conf_mean": float(conf.mean()) if len(conf) else None,
        "outside_pct": float((P["ins"] == 0).mean() * 100) if len(P) else 0.0,
        "dropout_pct": float(100 * missing / max(slots, 1)), "dropout_n": int(missing),
        "dropout_heat": _hist2d(miss_xy[:, 0], miss_xy[:, 1]) if len(miss_xy) > 30 else None,
        "dropout_far_side_pct": float((miss_xy[:, 1] > HALF_W / 3).mean() * 100) if len(miss_xy) else None,
        "dropout_ends_pct": float((np.abs(miss_xy[:, 0]) > HALF_L * 2 / 3).mean() * 100) if len(miss_xy) else None,
        "heat": _hist2d(ins["x"].to_numpy(), ins["y"].to_numpy()) if len(ins) > 30 else None,
    }
    # coverage by pitch zone: where are players found least often? (detections per square metre of the zone, 3 x 3 grid)
    zx = np.digitize(ins["x"], [-HALF_L / 3, HALF_L / 3]); zy = np.digitize(ins["y"], [-HALF_W / 3, HALF_W / 3])
    A["players"]["zone_share"] = [[float(((zx == i) & (zy == j)).mean() * 100) for j in range(3)] for i in range(3)]

    # ------------------------------------------------------------------ ball
    B = df[df["class_name"] == "ball"]
    bn = B.groupby("frame_id").size()
    Bi = B[(B["ins"] == 1) & B["x"].notna()].copy()
    Bi["cx"], Bi["cy"] = (Bi["x"] / 0.5).round(), (Bi["y"] / 0.5).round()
    cell = Bi.assign(t_=Bi["t"]).groupby(["cx", "cy"])["t_"].agg(["size", "min", "max"])
    fixed = cell[(cell["size"] >= 15) & ((cell["max"] - cell["min"]) >= 120)]
    spots = [{"x": float(i[0] * 0.5), "y": float(i[1] * 0.5), "hits": int(r["size"])} for i, r in fixed.sort_values("size", ascending=False).iterrows()]
    if len(fixed):
        Bi = Bi[~pd.MultiIndex.from_frame(Bi[["cx", "cy"]]).isin(fixed.index)]
    clean = Bi.sort_values(["frame_id", "confidence"], ascending=[True, False]).groupby("frame_id", as_index=False).first()
    seen = set(clean["frame_id"])
    has_ball = pd.Series(np.isin(fr, list(seen)).astype(float), index=fr)
    # gaps between sightings
    bt = np.sort(clean["t"].to_numpy())
    gaps = np.diff(bt) if len(bt) > 1 else np.array([])
    gap_lens = gaps[gaps > 2 * step]
    long_gaps = sorted([(float(bt[i]), float(gaps[i])) for i in np.where(gaps > 5.0)[0]], key=lambda z: -z[1])[:5]
    time_in_gaps = float(np.clip(gaps[gaps > 5.0] if len(gaps) else 0, 0, None).sum()) if len(gaps) else 0.0
    # near a player?
    pl_xy = {k: g[["x", "y"]].to_numpy() for k, g in ins.groupby("frame_id")}
    near_flags = []
    for fid, bx, by in clean[["frame_id", "x", "y"]].itertuples(index=False):
        g = pl_xy.get(fid)
        if g is not None and len(g):
            near_flags.append(float(np.hypot(g[:, 0] - bx, g[:, 1] - by).min()) <= NEAR_PLAYER_M)
    # impossible jumps
    jumps = 0
    pairs = 0
    if len(clean) > 1:
        c = clean.sort_values("t")
        dt = np.diff(c["t"].to_numpy()); dd = np.hypot(np.diff(c["x"].to_numpy()), np.diff(c["y"].to_numpy()))
        ok = (dt > 0) & (dt <= 0.5)
        pairs = int(ok.sum())
        jumps = int((dd[ok] / dt[ok] > BALL_JUMP_MS).sum())
    bconf = B["confidence"].to_numpy()
    A["ball"] = {
        "raw_frames": int(bn.reindex(fr, fill_value=0).gt(0).sum()), "raw_pct": float(bn.reindex(fr, fill_value=0).gt(0).mean() * 100),
        "clean_frames": int(len(clean)), "coverage": float(has_ball.mean()), "false_spot_hits": int(sum(s["hits"] for s in spots)),
        "spots": spots[:6], "per_minute": per_minute(has_ball),
        "multi_pct": float((bn.reindex(fr, fill_value=0) > 1).sum() / max(int((bn.reindex(fr, fill_value=0) > 0).sum()), 1) * 100),
        "outside_pct": float((B["ins"] == 0).mean() * 100) if len(B) else 0.0,
        "conf_cut": math.floor(float(bconf.min()) * 20) / 20 if len(bconf) else 0.0,
        "conf_mean": float(bconf.mean()) if len(bconf) else None,
        "conf_hist": np.histogram(bconf, bins=np.arange(0.2, 1.0001, 0.05))[0].tolist() if len(bconf) else [],
        "conf_low_pct": float((bconf < 0.45).mean() * 100) if len(bconf) else 0.0,
        "gap_over5_pct": float(100 * time_in_gaps / max(A["duration"], 1)), "long_gaps": long_gaps,
        "gap_hist": np.histogram(gap_lens, bins=[2 * step, 2, 5, 10, 20, 60, 1e9])[0].tolist() if len(gap_lens) else [0] * 6,
        "near_player_pct": float(np.mean(near_flags) * 100) if near_flags else None, "jump_pct": float(100 * jumps / max(pairs, 1)),
        "xy": clean[["x", "y"]].to_numpy().tolist() if len(clean) < 6000 else clean.sample(6000, random_state=1)[["x", "y"]].to_numpy().tolist(),
        "heat": _hist2d(clean["x"].to_numpy(), clean["y"].to_numpy()) if len(clean) > 30 else None,
    }
    A["ball"]["score"] = float(min(100.0, A["ball"]["coverage"] / TARGET_BALL_COVERAGE * 100))

    # ------------------------------------------------------------------ goalkeepers and referee
    gk = df[((df["class_name"] == "goalkeeper") | (df["team"] == "Goalkeeper")) & df["x"].notna()]
    gk_in = gk[gk["ins"] == 1]
    left = gk_in[gk_in["x"] < -HALF_L / 2].groupby("frame_id").size().reindex(fr, fill_value=0)
    right = gk_in[gk_in["x"] > HALF_L / 2].groupby("frame_id").size().reindex(fr, fill_value=0)
    any_gk = gk.groupby("frame_id").size().reindex(fr, fill_value=0)
    gk_cls = df[df["class_name"] == "goalkeeper"]
    R = df[df["class_name"] == "referee"]
    A["goalkeepers"] = {
        "left_pct": float((left > 0).mean() * 100), "right_pct": float((right > 0).mean() * 100),
        "both_pct": float(((left > 0) & (right > 0)).mean() * 100), "none_pct": float(((left == 0) & (right == 0)).mean() * 100),
        "any_mean": float(any_gk.mean()), "as_player_class": int(((df["class_name"] == "player") & (df["team"] == "Goalkeeper")).sum()),
        "as_gk_class": int(len(gk_cls)), "far_from_goal_pct": float((gk_in["x"].abs() < HALF_L / 2).mean() * 100) if len(gk_in) else 0.0,
        "per_minute_both": per_minute(((left > 0) & (right > 0)).astype(float)),
        "per_minute_left": per_minute((left > 0).astype(float)), "per_minute_right": per_minute((right > 0).astype(float)),
        "conf_mean": float(gk_cls["confidence"].mean()) if len(gk_cls) else None,
        "referee_dets": int(len(R)), "referee_frames": int(R["frame_id"].nunique()),
        "referee_team_labelled": int(R["team"].isin(teams).sum()),
        "xy": gk_in[["x", "y"]].sample(min(len(gk_in), 4000), random_state=1).to_numpy().tolist() if len(gk_in) else [],
    }
    A["goalkeepers"]["score"] = float(((left > 0).mean() + (right > 0).mean()) / 2 * 100)

    # ------------------------------------------------------------------ tracking and identities
    tp = team_pl[["tracker_id", "pid", "frame_id", "t", "team", "x", "y", "ins"]].copy()
    n_tr, n_id = int(tp["tracker_id"].nunique()), int(tp["pid"].nunique())
    life = tp.groupby("pid")["t"].agg(["min", "max", "size"])
    life["dur"] = life["max"] - life["min"]
    life_tr = tp.groupby("tracker_id")["t"].agg(["min", "max", "size"])
    life_tr["dur"] = life_tr["max"] - life_tr["min"]
    long_share = float(life.loc[life["dur"] >= LONG_TRACK_S, "size"].sum() / max(life["size"].sum(), 1) * 100)
    long_share_tr = float(life_tr.loc[life_tr["dur"] >= LONG_TRACK_S, "size"].sum() / max(life_tr["size"].sum(), 1) * 100)
    first_seen = life["min"].to_numpy()
    new_ids = np.bincount(np.clip(((first_seen - start) // 60).astype(int), 0, nmin - 1), minlength=nmin).tolist() if len(first_seen) else [0] * nmin
    first_tr = life_tr["min"].to_numpy()
    new_tr = np.bincount(np.clip(((first_tr - start) // 60).astype(int), 0, nmin - 1), minlength=nmin).tolist() if len(first_tr) else [0] * nmin
    # where do raw tracks end? mid-pitch = a player lost behind someone else, edge = left the pitch / lost at the boundary
    ends = tp.sort_values("frame_id").groupby("tracker_id").tail(1)
    ends = ends[(ends["t"] < end - 5) & ends["x"].notna()]
    edge = ((ends["x"].abs() > HALF_L - EDGE_M) | (ends["y"].abs() > HALF_W - EDGE_M))
    # jumps between consecutive detections of one identity
    s = tp[tp["x"].notna()].sort_values(["pid", "frame_id"])
    dt = s.groupby("pid")["t"].diff().to_numpy()
    dd = np.hypot(s.groupby("pid")["x"].diff().to_numpy(), s.groupby("pid")["y"].diff().to_numpy())
    okm = (dt > 0) & (dt <= 1.0)
    jump_pct = float(100 * ((dd[okm] / dt[okm]) * 3.6 > JUMP_KMH).sum() / max(okm.sum(), 1))
    exp_ids = 2 * EXPECTED_PER_TEAM
    A["tracking"] = {
        "tracker_ids": n_tr, "player_ids": n_id, "expected": exp_ids, "reduction_pct": float(100 * (1 - n_id / max(n_tr, 1))),
        "long_share": long_share, "long_share_raw": long_share_tr, "median_life_s": float(life["dur"].median()) if len(life) else 0.0,
        "mean_life_s": float(life["dur"].mean()) if len(life) else 0.0,
        "life_hist": np.histogram(life["dur"], bins=[0, 2, 5, 10, 30, 60, 120, 300, 1e9])[0].tolist() if len(life) else [],
        "life_hist_raw": np.histogram(life_tr["dur"], bins=[0, 2, 5, 10, 30, 60, 120, 300, 1e9])[0].tolist() if len(life_tr) else [],
        "new_ids_per_min": blank_empty(new_ids), "new_trackers_per_min": blank_empty(new_tr),
        "ids_per_player_per_min": float(np.mean([v for v in new_ids if v is not None]) / exp_ids) if new_ids else 0.0,
        "ends_n": int(len(ends)), "ends_midpitch_pct": float(100 * (~edge).mean()) if len(ends) else 0.0,
        "ends_heat": _hist2d(ends["x"].to_numpy(), ends["y"].to_numpy()) if len(ends) > 30 else None,
        "jump_pct": jump_pct, "short_ids_pct": float(100 * (life["dur"] < 5).mean()) if len(life) else 0.0,
    }
    # score: the share of player time carried by steady identities, 60% weighted to long identities, 40% to how many ids per player
    ratio = exp_ids / max(n_id, 1)
    A["tracking"]["score"] = float(min(100.0, 0.7 * long_share + 0.3 * min(1.0, ratio * 3) * 100))

    # ------------------------------------------------------------------ team assignment
    pp = P.copy()
    in_two = pp["team"].isin(teams)
    raw_ok = pp["team_raw"].isin(teams)
    agree = (pp["team_raw"] == pp["team"]) & in_two
    noise_labels = pp.loc[~pp["team_raw"].isin(list(teams) + ["Goalkeeper", "Referee"]), "team_raw"].value_counts()
    both = pp[in_two & raw_ok]
    flips = float((both["team_raw"] != both["team"]).mean() * 100) if len(both) else 0.0
    sub_ = pp[in_two & raw_ok].copy()
    sub_["differs"] = (sub_["team_raw"] != sub_["team"]).astype(float)
    flick = sub_.groupby("tracker_id")["differs"].mean()                    # per track: share of detections labelled differently from its final team
    mixed_dets = sub_[sub_["tracker_id"].isin(flick[flick >= 0.10].index)]
    # kit colours
    def rgb_of(s):
        try:
            b, g, r = [float(v) for v in str(s).strip("()[] ").split(",")][:3]
            return (r, g, b)
        except Exception:
            return None
    col = {}
    for t in teams:
        v = [rgb_of(c) for c in team_pl.loc[team_pl["team"] == t, "colour"].dropna().sample(min(400, int((team_pl["team"] == t).sum())), random_state=1)]
        v = [c for c in v if c]
        col[t] = tuple(np.median(np.array(v), axis=0).round().astype(int).tolist()) if v else None
    kit_dist = float(np.linalg.norm(np.array(col[teams[0]]) - np.array(col[teams[1]]))) if all(col.values()) else None
    imb = (cnt[teams[0]] - cnt[teams[1]]).abs().mean()
    matrix = pd.crosstab(pp["team_raw"].fillna("none"), pp["team"].fillna("none"))
    A["teams_check"] = {
        "agree_pct": float(100 * agree.sum() / max(in_two.sum(), 1)), "flip_pct": flips,
        "noise_n": int(noise_labels.sum()), "noise_pct": float(100 * noise_labels.sum() / max(len(pp), 1)),
        "noise_labels": {str(k): int(v) for k, v in noise_labels.head(5).items()},
        "gk_labelled_players_pct": float(100 * ((pp["team"] == "Goalkeeper").sum()) / max(len(pp), 1)),
        "mixed_tracks": int((flick >= 0.10).sum()), "tracks": int(len(flick)), "mixed_dets_pct": float(100 * len(mixed_dets) / max(len(sub_), 1)),
        "kit_rgb": {t: col[t] for t in teams}, "kit_distance": kit_dist, "balance_gap": float(imb),
        "per_minute_flip": None,
        "matrix": {str(r): {str(c): int(matrix.loc[r, c]) for c in matrix.columns} for r in matrix.index},
    }
    stable = max(0.0, 100 - flips - 100 * float(A["teams_check"]["noise_pct"]) / 100 * 1.0 - A["teams_check"]["gk_labelled_players_pct"] * 0.5)
    A["teams_check"]["score"] = float(min(100.0, stable))

    # ------------------------------------------------------------------ positions and speed
    allp = df[(df["class_name"].isin(["player", "goalkeeper"])) & df["x"].notna()]
    sp = ins["spd"].dropna()
    A["positions"] = {
        "pitch_keypoint_frames": kp["frames"], "pitch_keypoints": len(kp["names"]), "pitch_conf_min": kp["conf"][0],
        "pitch_fixed": bool(kp["frames"] and kp["conf"][0] is not None and kp["conf"][0] >= 0.999 and kp["conf"][1] >= 0.999),
        "inside_pct": float((allp["ins"] == 1).mean() * 100) if len(allp) else 0.0,
        "off_pitch_xy_pct": float(((allp["x"].abs() > HALF_L + 1) | (allp["y"].abs() > HALF_W + 1)).mean() * 100) if len(allp) else 0.0,
        "speed_p50": float(sp.median()) if len(sp) else None, "speed_p95": float(sp.quantile(0.95)) if len(sp) else None,
        "speed_p99": float(sp.quantile(0.99)) if len(sp) else None, "speed_over25_pct": float((sp > JUMP_KMH).mean() * 100) if len(sp) else 0.0,
        "speed_hist": np.histogram(sp.clip(upper=24.99), bins=np.arange(0, 25.01, 1))[0].tolist() if len(sp) else [],
        "jump_pct": jump_pct, "speed_null_pct": float(ins["spd"].isna().mean() * 100) if len(ins) else 0.0,
    }
    A["positions"]["score"] = float(max(0.0, 100 - A["positions"]["jump_pct"] * 15 - A["positions"]["off_pitch_xy_pct"] * 3
                                        - (100 - A["positions"]["inside_pct"]) * 0.5))
    A["players"]["score"] = float(A["players"]["completeness"] * 100)

    # ------------------------------------------------------------------ scores and ranking
    scores = {"players": A["players"]["score"], "ball": A["ball"]["score"], "goalkeepers": A["goalkeepers"]["score"],
              "tracking": A["tracking"]["score"], "teams": A["teams_check"]["score"], "positions": A["positions"]["score"]}
    A["scores"] = scores
    A["overall"] = float(sum(STAGE_WEIGHTS[k] * scores[k] for k in scores))
    ranking = []
    for k, v in scores.items():
        gain = STAGE_WEIGHTS[k] * max(0.0, GOOD - v)
        ranking.append({"stage": k, "score": v, "grade": grade(v), "gain": float(gain)})
    ranking.sort(key=lambda r: -r["gain"])
    A["ranking"] = ranking
    A["overall_if_fixed"] = {r["stage"]: float(A["overall"] + r["gain"]) for r in ranking}
    return A
