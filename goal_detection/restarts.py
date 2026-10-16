"""
GOAL DETECTION - version 0: find the RESTARTS (kick-offs from the centre spot).

A goal in walking football is followed by a restart from the centre spot: play stops, the teams drop back into their own
halves, and the team that conceded restarts from the middle. That pattern can be read from the player positions alone, so it
still works when the ball is hardly ever seen (the ball is detected in only about half the frames, and least of all near the
goal). This step finds those restarts and, where it can, says which end the ball went to before it and who probably scored.
It is only a calculation on the finished database: run it after Transformations. Nothing is written to the database.

SIGNALS (per second of the game)
  FORMATION  the two teams are on opposite halves (their median positions at least 5 m apart, one each side of the centre
             line, almost everyone on their own side), one or two players are near the centre spot, and nobody is moving
             much. Scored 0-1; averaged over 4 seconds.            -> formation periods (score >= FORMATION_MIN)
  BALL DWELL the detected ball sits within 1.5 m of the centre spot for 1.5 s or more (gaps of up to 0.6 s allowed).
  LEAD-UP    in the 60 s before the restart the ball was seen within 4 m of one goal line. That end is where the ball went in
             (a guess: it could be a miss or a save), and the team whose players are in that half at the restart conceded.

Confidence counts the independent signals: a formation score of 0.6+ (FORMATION_STRONG) and a ball sitting at the centre
spot for 2.5 s+ (BALL_DWELL_STRONG_S). Both = strong, one = possible, neither = weak. (The lead-up is only reported: the ball
is near a goal at some point in most minutes, so it does not separate goals from other play yet.) The first kick-off of the game (and the second-half one, if the game has
a half time set) are labelled as such and are not goals.

OUTPUT  <reports>/<id>_goal-detection_<timestamp>.csv + .json, a table on the screen that compares the number of restarts with
the score entered in the Game Logger, and (when the video is given) a contact sheet of stills to check by eye.
"""
import os
import re
import json
import time
import sqlite3

import numpy as np
import pandas as pd

# --- tuning knobs -------------------------------------------------------------------------------------------------
FORMATION_MIN = 0.35         # a smoothed formation score at least this much makes a candidate period
FORMATION_STRONG = 0.60
TEAM_GAP_M = 5.0             # the two teams' median x at least this far apart (and on opposite sides of the centre line)
OWN_SIDE_MIN = 0.80          # share of a team's detections on its own side (full marks at this level)
CENTRE_PLAYERS = (0.4, 2.5)  # players near the centre spot per frame (the kick-off taker, maybe a second one)
CENTRE_RADIUS_M = 3.0
QUIET_KMH, QUIET_MIN = 3.0, 0.85
BALL_CENTRE_M = 1.5          # the ball counts as 'at the centre spot' inside this radius
BALL_DWELL_MIN_S = 1.5
BALL_DWELL_STRONG_S = 2.5
BALL_GAP_S = 0.6
MERGE_GAP_S = 3              # formation seconds closer than this belong to one period
LINK_S = 8.0                 # a ball dwell within this many seconds of a formation period belongs to it
LEADUP_S = 60.0
LEADUP_MIN_DETECTIONS = 6    # a visit to a goal end needs at least this many ball detections (0.5 s at 12 per second)
LEADUP_GAP_S = 2.0
GOAL_END_X_M = 14.0          # ball seen at least this far from the centre line (the goal lines are at +/-18 m)
KICKOFF_WINDOW_S = 25.0
BALL_BURST_FRAMES = 6   # a quick kick-off: this many frames with the ball within BALL_CENTRE_M ...
BALL_BURST_S = 2.0      # ... inside this many seconds counts as 'ball at centre' when a formation period backs it up
BALL_BURST_SPREAD_M = 1.0  # and the ball must be sitting there (a ball rolling through the middle moves more than this)
GOAL_MOUTH_X_M = 17.0    # GOAL MOUTH: ball seen at least this far from the centre line (the goal lines are at +/-18 m) ...
GOAL_MOUTH_Y_M = 3.5     # ... and within this of the middle of the goal ...
GOAL_MOUTH_CONF = 0.30   # ... (detections at least this confident) ...
GOAL_MOUTH_MIN = 3       # ... in at least this many frames (a ball resting in a corner or a spare ball gives many more)
GOAL_TO_RESTART_S = (3.0, 22.0)  # and the ball is at the centre spot this long after it (seen: 10, 10, 14 and 19 s)
NON_TEAMS = {"Goalkeeper", "Referee", "unknown", "Unknown", "", None}


def _seconds(v, default=0.0):
    """'mm:ss', 'h:mm:ss' or a number -> seconds (the game record holds either)"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return default if v != v else float(v)
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return default
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def _mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


class GoalDetection:

    def __init__(self, DATABASE_FILE, OUTPUT_PATH, GAME_RECORD, VIDEO_PATH=None, quiet=False):
        self.db = DATABASE_FILE
        self.out = OUTPUT_PATH
        self.rec = GAME_RECORD or {}
        self.video = VIDEO_PATH
        self.quiet = quiet

    # ------------------------------------------------------------------ data
    def _load(self):
        con = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        try:
            gm = con.execute("SELECT video_fps FROM game LIMIT 1").fetchone()
            fps = float(gm[0]) if gm and gm[0] else 24.0
            d = pd.read_sql("SELECT frame_id, class_name, team, x_transformed_metres AS x, y_transformed_metres AS y, "
                            "is_inside_pitch AS ins, speed_km_per_hour AS spd, confidence AS conf, xmin, ymin, xmax, ymax "
                            "FROM detected_objects WHERE class_name IN ('player','ball')", con)
        finally:
            con.close()
        d["t"] = d.frame_id / fps
        return d, fps

    # ------------------------------------------------------------------ signals
    def _formation(self, players, teams):
        rows = {}
        p = players[players.team.isin(teams) & (players.ins == 1)].copy()
        p["sec"] = p.t.astype(int)
        for s, g in p.groupby("sec"):
            med, own = [], []
            for tm in teams:
                h = g[g.team == tm]
                if len(h) < 12:
                    med.append(np.nan); own.append(np.nan)
                    continue
                m = h.x.median()
                med.append(m)
                own.append(float(((np.sign(h.x) == np.sign(m)) & (h.x.abs() > 0.5)).mean()))
            if np.isnan(med).any():
                continue
            opp = np.sign(med[0]) != np.sign(med[1]) and abs(med[0] - med[1]) >= TEAM_GAP_M
            centre = float((np.hypot(g.x, g.y) < CENTRE_RADIUS_M).sum() / max(g.frame_id.nunique(), 1))
            quiet = float((g.spd < QUIET_KMH).mean())
            score = (1.0 if opp else 0.0) * min(min(own) / OWN_SIDE_MIN, 1.0) * \
                    (1.0 if CENTRE_PLAYERS[0] <= centre <= CENTRE_PLAYERS[1] else 0.0) * min(quiet / QUIET_MIN, 1.0)
            rows[int(s)] = dict(score=score, med=med, centre=centre, quiet=quiet)
        if not rows:
            return pd.DataFrame(columns=["score", "sm", "mA", "mB", "centre", "quiet"])
        idx = range(int(min(rows)), int(max(rows)) + 1)
        df = pd.DataFrame({s: rows.get(s, dict(score=0.0, med=[np.nan, np.nan], centre=np.nan, quiet=np.nan)) for s in idx}).T
        df["mA"] = [m[0] for m in df.med]; df["mB"] = [m[1] for m in df.med]
        df["score"] = df.score.astype(float)
        df["sm"] = df.score.rolling(4, min_periods=1, center=True).mean()
        return df

    def _formation_periods(self, df):
        on = df.index[df.sm >= FORMATION_MIN].tolist()
        periods = []
        for s in on:
            if periods and s - periods[-1][1] <= MERGE_GAP_S:
                periods[-1][1] = s
            else:
                periods.append([s, s])
        return periods

    def _ball_dwells(self, ball):
        b = ball.sort_values("t")
        b = b[np.hypot(b.x, b.y) < BALL_CENTRE_M]
        t = b.t.values
        runs, s = [], 0
        for i in range(1, len(t) + 1):
            if i == len(t) or t[i] - t[i - 1] > BALL_GAP_S:
                if i > s and t[i - 1] - t[s] >= BALL_DWELL_MIN_S:
                    runs.append((float(t[s]), float(t[i - 1])))
                s = i
        return runs

    def _ball_bursts(self, ball):
        """Short, dense visits of the ball to the centre spot (BALL_BURST_FRAMES frames within BALL_BURST_S). A ball rolling
        through the middle can do this too, so these only count when a formation period backs them up."""
        b = ball[np.hypot(ball.x, ball.y) < BALL_CENTRE_M]
        f = b.groupby("frame_id").agg(t=("t", "first"), x=("x", "median"), y=("y", "median")).sort_values("t")
        t, x, y = f.t.values, f.x.values, f.y.values
        bursts = []
        n = BALL_BURST_FRAMES
        for i in range(0, len(t) - n + 1):
            j = i + n
            if t[j - 1] - t[i] <= BALL_BURST_S and np.ptp(x[i:j]) <= BALL_BURST_SPREAD_M and np.ptp(y[i:j]) <= BALL_BURST_SPREAD_M:
                if bursts and t[i] <= bursts[-1][1] + 1.0:
                    bursts[-1][1] = float(t[j - 1])
                else:
                    bursts.append([float(t[i]), float(t[j - 1])])
        return [tuple(x) for x in bursts]

    def _goal_mouth_events(self, ball_all):
        """Visits of the ball to the mouth of a goal (any detection, in or out of the pitch outline: a ball in the net is
        outside it). -> list of (start, end, frames, side)."""
        a = ball_all[(ball_all.x.abs() >= GOAL_MOUTH_X_M) & (ball_all.y.abs() < GOAL_MOUTH_Y_M) & (ball_all.conf >= GOAL_MOUTH_CONF)]
        out = []
        for side, g in (("right", a[a.x > 0]), ("left", a[a.x < 0])):
            f = g.groupby("frame_id").t.first().sort_values().values
            ev = []
            for t in f:
                if ev and t - ev[-1][1] <= 1.5:
                    ev[-1][1] = float(t); ev[-1][2] += 1
                else:
                    ev.append([float(t), float(t), 1])
            out += [(e[0], e[1], e[2], side) for e in ev if e[2] >= GOAL_MOUTH_MIN]
        return sorted(out)

    def _medians(self, players, teams, t):
        """each team's median x in the 6 s around t (None when unseen)"""
        w = players[(players.t >= t - 3) & (players.t <= t + 3) & (players.ins == 1)]
        r = []
        for tm in teams:
            h = w[w.team == tm]
            r.append(float(h.x.median()) if len(h) >= 12 else np.nan)
        return r

    def _leadup(self, ball, t0):
        """The last SUSTAINED visit of the ball to a goal end (within GOAL_END_X_M of the centre line... i.e. 4 m from the goal
        line) in the LEADUP_S before t0: at least LEADUP_MIN_DETECTIONS detections, so a few stray false 'balls' do not count.
        -> (side, time the visit started, deepest x reached) or (None, None, None)."""
        w = ball[(ball.t >= t0 - LEADUP_S) & (ball.t < t0) & (ball.x.abs() >= GOAL_END_X_M)].sort_values("t")
        best = None
        for side, g in (("right", w[w.x > 0]), ("left", w[w.x < 0])):
            t = g.t.values
            s = 0
            for i in range(1, len(t) + 1):
                if i == len(t) or t[i] - t[i - 1] > LEADUP_GAP_S:
                    if i - s >= LEADUP_MIN_DETECTIONS:
                        if best is None or t[i - 1] > best[3]:
                            best = (side, float(t[s]), float(g.x.abs().iloc[s:i].max()), float(t[i - 1]))
                    s = i
        return (best[0], best[1], best[2]) if best else (None, None, None)

    # ------------------------------------------------------------------ run
    def run(self):
        print("-------------------------------------------------")
        print("GOAL DETECTION: finding the restarts from the centre spot (version 0)")
        if not os.path.isfile(self.db):
            raise SystemExit(f"\nSTOPPED: no detections database at {self.db}. Run Detections and Transformations first.\n")
        d, fps = self._load()
        players = d[d.class_name == "player"]
        ball = d[d.class_name == "ball"].copy()
        ball = ball[ball.ins.fillna(1) == 1] if len(ball) else ball
        counts = players[players.team.notna() & ~players.team.isin(NON_TEAMS)].team.value_counts()
        teams = list(counts.index[:2])
        if len(teams) < 2 or players.x.notna().sum() == 0:
            raise SystemExit("\nSTOPPED: there are no team labels or pitch positions in this database (run Transformations).\n")
        print(f"   teams in the database: {teams[0]} / {teams[1]}")
        game_start = _seconds(self.rec.get("game_start_seconds"), 0.0)
        ht_end = _seconds(self.rec.get("half_time_end_seconds"), None)
        game_end = _seconds(self.rec.get("game_end_seconds"), float(d.t.max()))

        form = self._formation(players, teams)
        periods = self._formation_periods(form) if len(form) else []
        dwells = self._ball_dwells(ball) if len(ball) else []
        bursts = self._ball_bursts(ball) if len(ball) else []

        # --- candidates: every formation period, plus ball dwells that no formation period explains
        cands, used = [], set()
        for a, z in periods:
            w = form.loc[a:z]
            c = dict(start=float(a), end=float(z) + 1.0, formation_peak=round(float(w.sm.max()), 2),
                     mA=float(w.mA.median()), mB=float(w.mB.median()), dwell_s=0.0, dwell_start=None)
            for i, (da, dz) in enumerate(dwells):
                if da <= c["end"] + LINK_S and dz >= c["start"] - LINK_S:
                    if dz - da > c["dwell_s"]:
                        c["dwell_s"], c["dwell_start"] = round(dz - da, 1), da
                    used.add(i)
            c["burst"] = any(ba <= c["end"] + LINK_S and bz >= c["start"] - LINK_S for ba, bz in bursts)
            cands.append(c)
        for i, (da, dz) in enumerate(dwells):
            if i in used or dz - da < BALL_DWELL_STRONG_S:
                continue
            cands.append(dict(start=da, end=dz, formation_peak=0.0, mA=np.nan, mB=np.nan, dwell_s=round(dz - da, 1), dwell_start=da, burst=False))
        # --- GOAL MOUTH -> centre: the ball was in a goal mouth and shortly after it is on the centre spot
        gm_events = self._goal_mouth_events(d[d.class_name == "ball"]) if len(ball) else []
        centre_balls = sorted([(a, z) for a, z in bursts] + [(a, z) for a, z in dwells])
        for ge in gm_events:
            nxt = [(a, z) for a, z in centre_balls if GOAL_TO_RESTART_S[0] <= a - ge[1] <= GOAL_TO_RESTART_S[1]]
            if not nxt:
                continue
            ta, tz = nxt[0]
            hit = [c for c in cands if abs(min(c["start"], c["dwell_start"] if c["dwell_start"] is not None else c["start"]) - ta) <= LINK_S
                   or (c["start"] - LINK_S <= ta <= c["end"] + LINK_S)]
            if hit and "gm" not in hit[0]:
                hit[0]["gm"] = ge
                if hit[0]["dwell_start"] is None:
                    hit[0]["dwell_start"], hit[0]["dwell_s"] = ta, round(tz - ta, 1)
            elif not hit:
                cands.append(dict(start=ta, end=tz + 1.0, formation_peak=0.0, mA=np.nan, mB=np.nan,
                                  dwell_s=round(tz - ta, 1), dwell_start=ta, burst=True, gm=ge))
        cands.sort(key=lambda c: c["start"])

        # --- lead-up + labels + confidence
        out = []
        first_done = False
        for c in cands:
            t0 = min(c["start"], c["dwell_start"]) if c["dwell_start"] is not None else c["start"]
            end_side, last_t, depth = self._leadup(ball, t0)
            gm = c.get("gm")
            if gm:  # the ball seen in the goal mouth is better evidence than a visit to that end
                end_side, last_t, depth = gm[3], gm[0], 18.0
                if np.isnan(c["mA"]) or np.isnan(c["mB"]):
                    c["mA"], c["mB"] = self._medians(players, teams, t0)
            n_sig = int(c["formation_peak"] >= FORMATION_STRONG) + int(c["dwell_s"] >= BALL_DWELL_STRONG_S or (c["burst"] and c["formation_peak"] >= FORMATION_STRONG))
            pts = n_sig + (2 if gm else 0)
            kind = "restart"
            if abs(t0 - game_start) <= KICKOFF_WINDOW_S:
                kind = "kick-off"
            elif ht_end is not None and abs(t0 - ht_end) <= KICKOFF_WINDOW_S:
                kind = "second-half kick-off"
            conceded = scorer = None
            if end_side and not (np.isnan(c["mA"]) or np.isnan(c["mB"])):
                sgn = 1 if end_side == "right" else -1
                if np.sign(c["mA"]) == sgn and np.sign(c["mB"]) != sgn:
                    conceded, scorer = teams[0], teams[1]
                elif np.sign(c["mB"]) == sgn and np.sign(c["mA"]) != sgn:
                    conceded, scorer = teams[1], teams[0]
            first_done = True
            out.append(dict(time_s=round(t0, 1), time=_mmss(t0), kind=kind,
                            confidence="strong" if (n_sig == 2 or gm) else "possible" if n_sig == 1 else "weak", points=pts,
                            goal_mouth_time=_mmss(gm[0]) if gm else "", goal_mouth_s=round(gm[0], 1) if gm else None,
                            formation_peak=c["formation_peak"], ball_dwell_s=c["dwell_s"], ball_went_to=end_side or "",
                            ball_near_goal_time=_mmss(last_t) if last_t is not None else "",
                            ball_near_goal_s=round(last_t, 1) if last_t is not None else None,
                            ball_max_depth_m=round(depth, 1) if depth is not None else None,
                            conceded_guess=conceded or "", scorer_guess=scorer or ""))
        res = pd.DataFrame(out)

        # --- compare with the score typed into the game logger
        score = str(self.rec.get("score", ""))
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", score)
        expected = int(m.group(1)) + int(m.group(2)) if m else None
        goal_like = res[res.kind == "restart"] if len(res) else res
        strong = goal_like[goal_like.confidence == "strong"] if len(goal_like) else goal_like
        self._print(res, expected, goal_like, strong, game_end)
        paths = self._save(res, expected, goal_like, strong, teams)
        if self.video and len(res):
            self._contact_sheet(res, paths["stem"])
        return res

    # ------------------------------------------------------------------ output
    def _print(self, res, expected, goal_like, strong, game_end):
        print(f"   restarts found: {len(res)}  ({len(goal_like)} after the kick-off(s); {len(strong)} strong)")
        for _, r in res.iterrows():
            extra = (f"  ball went {r.ball_went_to} (near the goal from {r.ball_near_goal_time}, "
                     f"{r.ball_max_depth_m:.0f} m from the centre line)") if r.ball_went_to else ""
            who = f"  -> {r.scorer_guess} probably scored" if r.scorer_guess else ""
            if r.goal_mouth_time:
                extra = f"  BALL IN THE {r.ball_went_to.upper()} GOAL MOUTH at {r.goal_mouth_time}"
            print(f"     {r.time:>6}  {r.kind:<20} {r.confidence:<8} formation {r.formation_peak:.2f}  ball at centre {r.ball_dwell_s:.1f}s{extra}{who}")
        if expected is not None:
            flag = "matches" if len(strong) == expected else "DOES NOT MATCH"
            print(f"   Game Logger score says {expected} goal(s); strong restarts after the kick-off: {len(strong)} ({flag})")
        else:
            print("   (no score in the Game Logger, so nothing to compare with)")

    def _save(self, res, expected, goal_like, strong, teams):
        os.makedirs(self.out, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        gid = self.rec.get("game_id", "game")
        try:
            gid = f"{int(gid):04d}"
        except (TypeError, ValueError):
            pass
        stem = os.path.join(self.out, f"{gid}_goal-detection_{stamp}")
        res.to_csv(stem + ".csv", index=False)
        summary = dict(game_id=str(gid), made=time.strftime("%Y-%m-%d %H:%M"), teams=teams, expected_goals_from_score=expected,
                       restarts=len(res), restarts_after_kickoff=len(goal_like), strong=len(strong),
                       restarts_list=json.loads(res.to_json(orient="records")) if len(res) else [])
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        print(f"   saved: {stem}.csv (and .json)")
        return {"stem": stem}

    def _contact_sheet(self, res, stem):
        import cv2
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            print("   (video not found: no stills made)")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        W, H = 640, 360
        tiles = []
        for _, r in res[res.kind == "restart"].head(8).iterrows():
            stills = [("restart", r.time_s + 1.0)]
            if r.ball_near_goal_s is not None and r.ball_near_goal_s == r.ball_near_goal_s:
                stills.insert(0, ("ball near goal", float(r.ball_near_goal_s)))
            for label, t in stills:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ok, im = cap.read()
                if not ok:
                    continue
                im = cv2.resize(im, (W, H))
                cv2.putText(im, f"{_mmss(t)} {label} ({r.confidence})", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                tiles.append(im)
        cap.release()
        if not tiles:
            return
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(stem + "_stills.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"   saved: {stem}_stills.jpg  (stills at each restart, and where the ball was last near a goal)")
