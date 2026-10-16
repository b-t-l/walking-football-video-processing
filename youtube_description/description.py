"""
YOUTUBE DESCRIPTION - a text file to paste into the description box of the video once it is uploaded.

It holds the teams, the date and the game's description, then the main times in the game (kick-off, half-time, second half,
full time) and then GOAL ESTIMATES made by the goal detection step: for each goal a build-up time, the goal itself and the
restart from the centre.

THE TIME STAMPS. YouTube turns a time like 0:29 or 12:05 (or 1:02:30 for a long video) in the description into a link that
jumps the video to that moment, so every time stamp here is written plainly as m:ss at the START of its own line, with the
label after it. Nothing else is needed. (Chapters, the labelled segments on the progress bar, additionally need ONE list that
starts at 0:00, runs in time order, has at least three entries and keeps them 10 s apart. The game times below are such a
list on their own, but the goal list repeats the earlier part of the video, so YouTube will not make chapters from the whole
description. The links work either way.) YouTube does not allow the characters < and > in a description, so none are used.

THE TIMES ARE FROM THE ANALYSIS VIDEO. The annotated video that is uploaded is the same length and starts at the same moment,
so a time here is the same time in the YouTube video.

THE GOAL TIMES ARE ESTIMATES. Goal detection finds the restart from the centre spot. If the ball was seen in the goal mouth
that time is the goal; if not, the goal is taken as the moment the ball last moved towards that end, or, failing both, 10 s
before the restart (the usual gap seen in the games checked: 8 to 19 s). The build-up line starts 15 s before the goal so the
goal is always inside what the viewer sees. Only strong restarts are listed, and the scorer is named only when the guess agrees
with the score in the Game Logger.
"""
import os
import re
import glob
import json
import time
import datetime

NAN_TEXT = ("", "nan", "none", "NaN")

BUILD_UP_S = 15.0           # the 'build-up' line is this long before the estimated goal
RESTART_GAP_S = 10.0        # goal estimated this long before the restart when nothing better is known
MIN_BUILD_GAP_S = 3.0       # a 'last visit to that end' closer to the restart than this is not used as the goal time
MAX_BUILD_GAP_S = 40.0      # ... nor one further away than this


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def _seconds(v):
    """'mm:ss', 'h:mm:ss' or a number -> seconds, or None when empty"""
    s = _clean(v)
    if not s:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    try:
        sec = 0.0
        for p in s.split(":"):
            sec = sec * 60 + float(p)
        return sec
    except ValueError:
        return None


def stamp(t):
    """seconds -> the form YouTube makes a link of: 0:29, 12:05, 1:02:30"""
    t = int(max(t, 0))
    h, m, s = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _date_text(d):
    s = _clean(d)
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return f"{dt.day} {dt.strftime('%B %Y')}"
        except ValueError:
            pass
    return s


def _kind_text(rec):
    fmt = {"mens": "Men's", "womens": "Women's"}.get(_clean(rec.get("game_format")).lower(), _clean(rec.get("game_format")).capitalize())
    typ = _clean(rec.get("game_type"))
    return " ".join(x for x in (fmt, typ) if x)


class YoutubeDescription:

    def __init__(self, OUTPUT_PATH, GAME_RECORD, DATABASE_FILE=None, VIDEO_PATH=None):
        self.out = OUTPUT_PATH
        self.rec = GAME_RECORD or {}
        self.db = DATABASE_FILE
        self.video = VIDEO_PATH

    # ------------------------------------------------------------------ goal detection results
    def _gid(self):
        try:
            return f"{int(self.rec.get('game_id')):04d}"
        except (TypeError, ValueError):
            return "game"

    def _latest_goal_json(self):
        files = sorted(glob.glob(os.path.join(self.out, f"{self._gid()}_goal-detection_*.json")))
        return files[-1] if files else None

    def _goal_rows(self):
        path = self._latest_goal_json()
        if path is None and self.db and os.path.isfile(self.db):
            print("   no goal detection result for this game yet: running goal detection first")
            from goal_detection import GoalDetection
            GoalDetection(self.db, self.out, self.rec, self.video).run()
            path = self._latest_goal_json()
        if path is None:
            return None, None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"   goals read from {os.path.basename(path)}")
        return data.get("restarts_list", []), path

    # ------------------------------------------------------------------ goal estimates
    def _estimate(self, r):
        """-> (build-up, goal, restart) in seconds for one strong restart"""
        restart = float(r["time_s"])
        goal = None
        gm = r.get("goal_mouth_s")
        if gm not in (None, "") and gm == gm:
            goal = float(gm)
        else:
            near = r.get("ball_near_goal_s")
            if near not in (None, "") and near == near and MIN_BUILD_GAP_S <= restart - float(near) <= MAX_BUILD_GAP_S:
                goal = float(near)
        if goal is None:
            goal = restart - RESTART_GAP_S
        goal = max(min(goal, restart - 1.0), 0.0)
        return max(goal - BUILD_UP_S, 0.0), goal, restart

    def _goals(self, rows):
        goals = [r for r in rows if r.get("kind") == "restart" and r.get("confidence") == "strong"]
        goals.sort(key=lambda r: r["time_s"])
        # name the scorer only if the guesses fit the score the game logger holds
        names = [(r.get("scorer_guess") or "") for r in goals]
        score = _clean(self.rec.get("score"))
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", score)
        ta, tb = _clean(self.rec.get("team_a")), _clean(self.rec.get("team_b"))
        if m and ta and tb:
            ok = names.count(ta) <= int(m.group(1)) and names.count(tb) <= int(m.group(2)) and all(n in ("", ta, tb) for n in names)
        else:
            ok = False
        if not ok:
            names = [""] * len(goals)
        return goals, names, (int(m.group(1)) + int(m.group(2)) if m else None)

    # ------------------------------------------------------------------ text
    def _text(self, rows):
        rec = self.rec
        ta, tb = _clean(rec.get("team_a")), _clean(rec.get("team_b"))
        title = _clean(rec.get("title")) or (f"{ta} vs {tb}" if ta and tb else "Walking football")
        L = [title]
        bits = [x for x in (_kind_text(rec), _date_text(rec.get("date"))) if x]
        if bits:
            L.append(" | ".join(bits))
        desc = _clean(rec.get("description"))
        if desc:
            L += ["", desc]
        score = _clean(rec.get("score"))
        m = re.match(r"^(\d+)\s*/\s*(\d+)$", score)
        if m and ta and tb:
            L += ["", f"Final score: {ta} {m.group(1)} - {m.group(2)} {tb}"]

        # --- the main times
        start = _seconds(rec.get("game_start_seconds")) or 0.0
        h1, h2 = _seconds(rec.get("half_time_start_seconds")), _seconds(rec.get("half_time_end_seconds"))
        end = _seconds(rec.get("game_end_seconds"))
        T = []
        if start >= 10:
            T.append((0.0, "Before kick-off"))
        has_half = h1 is not None and h2 is not None
        T.append((start, "First half starts" if has_half else "Game starts"))
        if has_half:
            T += [(h1, "First half ends (half-time)"), (h2, "Second half starts")]
        if end is not None:
            T.append((end, "Game ends" if not has_half else "Game ends (full time)"))
        T = [t for i, t in enumerate(sorted(T, key=lambda x: x[0]))]
        L += ["", "GAME TIMES"] + [f"{stamp(t)} {label}" for t, label in T]

        # --- the goals
        L += ["", "GOAL ESTIMATES (found automatically from the player and ball positions, so a few seconds out is possible)"]
        if rows is None:
            L.append("Goal times could not be worked out for this game.")
        else:
            goals, names, expected = self._goals(rows)
            if not goals:
                L.append("No goals found." if expected in (0, None) else "No goal times could be worked out for this game.")
            for i, (r, who) in enumerate(zip(goals, names), 1):
                b, g, rs = self._estimate(r)
                tag = f"Goal {i}" + (f" ({who})" if who else "")
                L += [f"{stamp(b)} {tag} - build-up",
                      f"{stamp(g)} {tag} - about here",
                      f"{stamp(rs)} {tag} - restart from the centre"]
            if expected is not None and goals and len(goals) != expected:
                L.append(f"(The game was logged with {expected} goal{'s' if expected != 1 else ''}; {len(goals)} could be placed.)")
        return "\n".join(L).replace("<", "").replace(">", "") + "\n"

    # ------------------------------------------------------------------ run
    def run(self):
        print("-------------------------------------------------")
        print("YOUTUBE DESCRIPTION: writing the text for the video description")
        rows, _ = self._goal_rows()
        text = self._text(rows)
        os.makedirs(self.out, exist_ok=True)
        path = os.path.join(self.out, f"{self._gid()}_youtube-description_{time.strftime('%Y%m%d-%H%M%S')}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text)
        print(f"   saved: {path}")
        return path
