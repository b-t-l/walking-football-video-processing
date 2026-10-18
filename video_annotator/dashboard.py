"""
LIVE DASHBOARD for the annotated video: the numbers shown to the right of the overhead map, worked out ONCE before the video
is drawn (so drawing a frame is just a look-up).

Every number comes from the same code as the match report (reports/match_report/metrics.py), so the dashboard and the PDF agree:

  possession      share of the time the ball has been with each team so far, and in the last 2 minutes
  distance        each team's distance covered so far (the report's way: average speed x game time x 5 players)
  running spells  spells above the running speed for at least 1 second so far
  top speed       the highest speed any of the team's players has held for a whole second so far
  score / clock   the game clock ("1st half 04:12") and the score; the score is only shown when the goals found by Goal
                  detection agree with the score typed into the Game Logger (otherwise a wrong goal could be shown)
  timeline        possession over the whole game in 15 s bars, a moving marker for now, and the goals

Possession needs the ball to be seen, so it stays '-' until there is enough to go on (POSS_MIN_KNOWN_S).
"""
import ast
import math
import os

import numpy as np

POSS_RECENT_S = 120.0       # the 'last 2 min' possession window
POSS_MIN_KNOWN_S = 60.0     # no total possession figure until the ball has been with someone this long
POSS_RECENT_MIN_KNOWN_S = 20.0
TIMELINE_BIN_S = 15.0
PLAYERS_PER_TEAM = 5        # the report scales one player's distance to the team by this


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


class LiveDashboard:

    def __init__(self, db_path, record, out_path, running_speed_kmh=15.0):
        from reports.match_report import metrics
        from youtube_description.description import YoutubeDescription
        self.rec = record or {}
        ta, tb = _clean(self.rec.get("team_a")), _clean(self.rec.get("team_b"))
        if not ta or not tb:
            raise ValueError("the game has no two teams (team A / team B) set")
        self.teams = [ta, tb]
        M, raw = metrics.compute_metrics(db_path, self.rec, self.teams, running_speed_kmh, return_raw=True)
        self.label = {name: M["team_labels"].get(name, name) for name in self.teams}   # name -> label in the database
        self.start, self.end, self.step = float(M["start"]), float(M["end"]), float(raw["step"])
        self.video_fps = float(M["video_fps"])
        self.half = M["half_time"] if M["half_time"] and M["half_time"][0] is not None else None
        self.running_kmh = running_speed_kmh
        owner = np.asarray(raw["owner"])
        self.n = len(owner)
        self.cA = np.cumsum(owner == 0)
        self.cB = np.cumsum(owner == 1)
        self.owner = owner

        # per-step speed sums for the distance, and the running top speed
        sp = raw["sp"]
        self.dist_mean = {}
        self.top = {}
        win = max(2, int(round(float(M["detection_fps"]) * 1.0)))
        for tm in self.teams:
            s = sp[sp["team"] == tm].sort_values(["pid", "frame_id"]).copy()
            # TOP SPEED = the highest speed a player held for a whole second (the lowest speed over any 1 s window). A single
            # jumpy reading cannot set it, and it follows the same 1 s idea as the running spells.
            s["sus"] = s.groupby("pid")["spd"].transform(lambda x: x.rolling(win, min_periods=win).min())
            idx = np.clip(np.round((s["t"].to_numpy() - self.start) / self.step).astype(int), 0, self.n - 1)
            ssum, scnt, smax = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
            np.add.at(ssum, idx, s["spd"].to_numpy())
            np.add.at(scnt, idx, 1)
            sus = np.nan_to_num(s["sus"].to_numpy(), nan=0.0)
            np.maximum.at(smax, idx, sus)
            self.dist_mean[tm] = (np.cumsum(ssum), np.cumsum(scnt))
            self.top[tm] = np.maximum.accumulate(smax)

        # running spells so far
        ev = M["running"]["events"]
        self.run_t = {tm: np.sort(np.array([e["t"] for e in ev if e["team"] == tm], dtype=float)) for tm in self.teams}

        # team colours (BGR) from the database labels
        self.colour = {tm: self._db_colour(db_path, self.label[tm]) for tm in self.teams}
        # a very dark kit (black) would vanish on the black panel: the timeline and goal markers use a lifted version of it
        self.colour_light = {tm: self._lift(c) for tm, c in self.colour.items()}

        # goals for the score line and the timeline
        self.goals, self.score_ok = [], False
        self._load_goals(out_path, YoutubeDescription)

        # possession timeline bins
        self.bins = []
        per = max(1, int(round(TIMELINE_BIN_S / self.step)))
        for i in range(0, self.n, per):
            seg = owner[i:i + per]
            a, b = int((seg == 0).sum()), int((seg == 1).sum())
            self.bins.append((a, b, len(seg)))
        print(f"   live dashboard ready: {self.teams[0]} / {self.teams[1]}, "
              f"{len(self.goals)} goal(s) {'shown in the score' if self.score_ok else '(score line hidden: goals not confirmed by the logged score)'}")

    @staticmethod
    def _lift(bgr):
        b, g, r = bgr
        if 0.299 * r + 0.587 * g + 0.114 * b >= 90:
            return bgr
        return tuple(int(0.45 * v + 0.55 * 205) for v in bgr)

    @staticmethod
    def _db_colour(db_path, label):
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            r = con.execute("SELECT team_color FROM detected_objects WHERE team = ? AND team_color IS NOT NULL AND team_color != 'None' LIMIT 1", (label,)).fetchone()
            con.close()
            if r:
                c = ast.literal_eval(r[0])
                return tuple(int(v) for v in c)
        except Exception:
            pass
        return (200, 200, 200)

    # ------------------------------------------------------------------ goals / score
    def _load_goals(self, out_path, YD):
        yd = YD(out_path, self.rec)
        path = yd._latest_goal_json()
        score = _clean(self.rec.get("score"))
        import re
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", score)
        if not m:
            return
        sa, sb = int(m.group(1)), int(m.group(2))
        if sa + sb == 0:
            self.score_ok = True
            return
        if path is None:
            return
        import json
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh).get("restarts_list", [])
        goals, names, expected = yd._goals(rows)
        if len(goals) != expected:
            return
        ta, tb = self.teams
        if not all(names):
            if sa == 0:
                names = [tb] * len(goals)
            elif sb == 0:
                names = [ta] * len(goals)
            else:
                return
        self.goals = [{"t": yd._estimate(g)[1], "team": n} for g, n in zip(goals, names)]
        self.score_ok = True

    # ------------------------------------------------------------------ look-ups for one video frame
    def _i(self, t):
        return int(min(max(round((t - self.start) / self.step), 0), self.n - 1))

    def at(self, frame_id):
        """everything the panel shows for this video frame"""
        t = frame_id / self.video_fps
        i = self._i(t)
        out = {"t": t, "phase": self.phase(t), "teams": []}
        a_all, b_all = int(self.cA[i]), int(self.cB[i])
        known_s = (a_all + b_all) * self.step
        j = max(i - int(POSS_RECENT_S / self.step), 0)
        a_rec, b_rec = int(self.cA[i] - self.cA[j]), int(self.cB[i] - self.cB[j])
        rec_known_s = (a_rec + b_rec) * self.step
        clock_s = self.game_seconds(t)
        for k, tm in enumerate(self.teams):
            tot = (a_all, b_all)[k]; rec = (a_rec, b_rec)[k]
            poss = 100.0 * tot / (a_all + b_all) if known_s >= POSS_MIN_KNOWN_S and (a_all + b_all) else None
            poss_rec = 100.0 * rec / (a_rec + b_rec) if rec_known_s >= POSS_RECENT_MIN_KNOWN_S and (a_rec + b_rec) else None
            ssum, scnt = self.dist_mean[tm]
            mean_spd = ssum[i] / scnt[i] if scnt[i] else 0.0
            dist_km = mean_spd / 3.6 * clock_s * PLAYERS_PER_TEAM / 1000.0
            runs = int(np.searchsorted(self.run_t[tm], t, side="right"))
            out["teams"].append({"name": tm, "label": self.label[tm], "colour": self.colour[tm], "poss": poss, "poss_recent": poss_rec,
                                 "dist_km": dist_km, "runs": runs, "top": float(self.top[tm][i])})
        if self.score_ok:
            s = [sum(1 for g in self.goals if g["team"] == tm and g["t"] <= t) for tm in self.teams]
            out["score"] = s
        else:
            out["score"] = None
        return out

    def game_seconds(self, t):
        """seconds of play so far (half-time not counted)"""
        t = min(max(t, self.start), self.end)
        s = t - self.start
        if self.half and t > self.half[0]:
            s -= min(t, self.half[1]) - self.half[0]
        return max(s, 0.0)

    def phase(self, t):
        if t < self.start:
            return "Pre-game"
        if t > self.end + 1:
            return "Full time"
        if self.half:
            if t < self.half[0]:
                label, clock = "1st half", t - self.start
            elif t < self.half[1]:
                return "Half-time"
            else:
                label, clock = "2nd half", (self.half[0] - self.start) + (t - self.half[1])
            return f"{label}  {int(clock // 60):02d}:{int(clock % 60):02d}"
        clock = t - self.start
        return f"{int(clock // 60):02d}:{int(clock % 60):02d}"
