"""Ball training images: full frames chosen to teach the ball detector where it fails, ready to upload to Roboflow.

Run on a game that has been through Detections (Transformations as well gives better picks: pitch positions are used to
spot impossible jumps and fast shots). The frames are cut from the analysis video at full resolution, exactly what the
detector sees. No boxes are drawn on them and no label files are written: you label them in Roboflow.

WHICH FRAMES (N = 40 by default, shared out like this; a group with too few candidates hands its places to the others):
  lost    25%  the ball WAS being followed, vanished for a stretch (a quarter of a second up to 3 seconds, ball last seen and
               seen again inside the pitch) and came back. A frame from the middle of the gap: the ball is almost certainly
               on the pitch in it, so these are the balls the model MISSED. (Long gaps give up to 3 frames.)
  edge    15%  the ball is at the edge of the pitch (within 60 px of the outline, inside or just outside): a ball that vanished
               there (last or next seen at the edge), or a weak detection (confidence under 0.6) there. The review of game 27
               showed the model finds only about half of the balls near the lines, mostly on the far touchline.
  near    10%  a BIG ball close to the camera (18 px or more across) or one in the bottom corners of the picture: a ball that
               vanished there, or a big one detected. The review showed obvious balls missed in the bottom-left corner.
  multi   15%  frames where the model found two or more 'balls'. At most one is real: the rest teach it what is NOT a ball.
  unsure  10%  the best ball in the frame has a confidence of only 0.30-0.50.
  odd      7%  the best ball is outside the pitch, or it jumped further than a ball can travel since the last detection.
  small    5%  the best ball is tiny (7 pixels or less across): the far side of the pitch.
  fast     3%  a smooth, fast stretch (shots and clearances: motion blur).
  random  10%  random frames from the whole game, with or without a ball. Good to keep as a TEST set you never train on.

NEVER PICKED: frames within 3 seconds of a frame in a ball-review set for the same game (<game>/ball-review/*/frames.json).
Those 100 frames are the hand-marked TEST set; keeping training pictures away from them keeps the score honest.

HOW THEY ARE SPREAD: frames are at least 2 seconds apart, and frames that an earlier run for this game already exported are
never picked again, so every run gives new pictures and you can run it as often as you like.

FILE NAMES are unique across the whole dataset (Roboflow needs that):  <game>_f<frame>_<group>.jpg  e.g. 0027_f012480_lost.jpg
(the game number and the frame number in the analysis video can only ever be one picture).

Layout:   <game folder>/ball-images/<timestamp>/<pictures>.jpg   plus _index.csv and README.txt in <timestamp>/
_index.csv tells you, for each picture, why it was chosen and (for 'lost') roughly where the ball was last and next seen, in pixels.
"""
import os
import re
import time
import glob
import sqlite3

import cv2
import numpy as np
import pandas as pd

QUOTAS = [("lost", .25), ("edge", .15), ("near", .10), ("multi", .15), ("unsure", .10), ("odd", .07), ("small", .05),
          ("fast", .03), ("random", .10)]
MIN_SPACING_S = 2.0          # picked frames are at least this far apart (and from earlier exports)
LOST_MIN_STEPS = 3           # shortest gap that counts (in detection steps: 3 = 0.25 s at 12 fps)
LOST_MAX_S = 3.0             # longer gaps are probably the ball out of play
UNSURE_RANGE = (0.0, 0.50)   # best ball confidence below this = unsure (the database only holds >= 0.30)
SMALL_MAX_W = 7              # px
FAST_MS = (6.0, 25.0)        # m/s, kept for at least 3 detections in a row
EDGE_PX = 60                 # 'at the edge' = within this many pixels of the pitch outline (inside or just outside)
EDGE_OUT_MAX_PX = 60         # ...but a ball further outside than this is a spare ball / out of play: not used for edge or near
EDGE_WEAK_CONF = 0.60        # edge detections below this confidence are offered too
NEAR_MIN_W = 18              # px: a ball this wide is close to the camera
NEAR_Y_FRAC = 0.60           # the bottom 40% of the picture, in the left and right thirds-ish (see NEAR_X_FRAC)
NEAR_X_FRAC = 0.35
PROTECT_REVIEW_S = 3.0       # keep this far from frames of a ball-review (test) set
JUMP_MS = 35.0               # m/s: no ball goes this fast, so a detection that needs it is a different object


class ExtractBallImages:

    def __init__(self, db_path, video_path, out_root, game_record=None, n_frames=40, seed=None):
        self.db_path = db_path
        self.video_path = video_path
        self.out_root = out_root
        self.game_record = game_record or {}
        self.n_frames = max(1, int(n_frames))
        self.seed = seed                       # None = different frames every run
        self.poly = None                       # pitch outline in video pixels (float32 Nx1x2), set in run()
        self.W = self.H = 0

    def _prefix(self):
        try:
            return f"{int(self.game_record['game_id']):04d}"
        except (KeyError, TypeError, ValueError):
            return "game"

    def _earlier_frames(self):
        """frames that earlier runs for this game exported (read from their _index.csv files)"""
        out = set()
        for f in glob.glob(os.path.join(self.out_root, "*", "_index.csv")):
            try:
                out.update(int(v) for v in pd.read_csv(f)["frame_id"])
            except Exception:                                                # noqa
                pass
        return out

    def _review_frames(self):
        """frames of the hand-marked ball-review sets of this game: never used for training"""
        import json
        out = set()
        review_root = os.path.join(os.path.dirname(os.path.abspath(self.out_root)), "ball-review")
        for f in glob.glob(os.path.join(review_root, "*", "frames.json")):
            try:
                out.update(int(v) for v in json.load(open(f))["frames"])
            except Exception:                                                # noqa
                pass
        return out

    def _signed_edge_distance(self, cx, cy):
        """pixels from the pitch outline: positive inside, negative outside (None when there is no outline)"""
        if self.poly is None:
            return None
        return float(cv2.pointPolygonTest(self.poly, (float(cx), float(cy)), True))

    # ------------------------------------------------------------------ candidates
    def _candidates(self, frames, ball, fps, rng):
        """-> {group: [(frame_id, note dict), ...]} each list in the order the group would like them picked"""
        step_s = lambda a, b: (b - a) / fps
        best = ball.sort_values("confidence", ascending=False).drop_duplicates("frame_id").set_index("frame_id").sort_index()
        n_balls = ball.groupby("frame_id").size()
        has_ball = np.isin(frames, best.index.values)
        cand = {k: [] for k, _ in QUOTAS}

        # --- lost: runs of frames without a ball, with a ball on both sides, both seen inside the pitch
        i, n = 0, len(frames)
        gaps, gaps_any = [], []
        while i < n:
            if has_ball[i]:
                i += 1
                continue
            j = i
            while j < n and not has_ball[j]:
                j += 1
            if i > 0 and j < n:
                a, b = frames[i - 1], frames[j]
                steps = j - i
                ra, rb = best.loc[a], best.loc[b]
                inside = all(pd.isna(r.get("is_inside_pitch")) or int(r.get("is_inside_pitch")) == 1 for r in (ra, rb))
                if steps >= LOST_MIN_STEPS and step_s(a, b) <= LOST_MAX_S:
                    gaps_any.append((i, j, a, b, ra, rb))
                    if inside:
                        gaps.append((i, j, a, b, ra, rb))
            i = j
        if gaps:
            w = np.array([g[1] - g[0] for g in gaps], dtype=float)
            order = rng.choice(len(gaps), size=len(gaps), replace=False, p=w / w.sum())
            for gi in order:
                i, j, a, b, ra, rb = gaps[gi]
                marks = [(i + j - 1) // 2] if j - i < 12 else [i + (j - i) // 3, i + (j - i) // 2, i + 2 * (j - i) // 3]
                for m in marks:
                    cand["lost"].append((int(frames[m]), {
                        "gap_seconds": round(step_s(a, b), 2),
                        "last_seen_xy_px": f"{(ra.xmin + ra.xmax) / 2:.0f},{(ra.ymin + ra.ymax) / 2:.0f}",
                        "next_seen_xy_px": f"{(rb.xmin + rb.xmax) / 2:.0f},{(rb.ymin + rb.ymax) / 2:.0f}"}))

        # --- edge / near: balls at the pitch outline, and big balls close to the camera (the weak spots found by the review)
        centre = lambda r: ((r.xmin + r.xmax) / 2.0, (r.ymin + r.ymax) / 2.0)
        width = lambda r: float(r.xmax - r.xmin)
        sd_cache = {}

        def sd_of(frame_id):
            if frame_id not in sd_cache:
                sd_cache[frame_id] = self._signed_edge_distance(*centre(best.loc[frame_id]))
            return sd_cache[frame_id]

        def usable(frame_id):          # not a spare ball / out of play: no further than EDGE_OUT_MAX_PX outside the outline
            d = sd_of(frame_id)
            return d is None or d >= -EDGE_OUT_MAX_PX

        def is_edge(frame_id):
            d = sd_of(frame_id)
            return d is not None and -EDGE_PX <= d <= EDGE_PX

        def is_near(frame_id):
            r = best.loc[frame_id]
            cx, cy = centre(r)
            corner = (self.H > 0 and cy > NEAR_Y_FRAC * self.H and self.W > 0 and
                      (cx < NEAR_X_FRAC * self.W or cx > (1 - NEAR_X_FRAC) * self.W))
            return width(r) >= NEAR_MIN_W or corner

        edge_gaps, near_gaps = [], []
        for i, j, a, b, ra, rb in gaps_any:
            if not (usable(a) and usable(b)):
                continue
            mid = int(frames[(i + j - 1) // 2])
            if is_edge(a) or is_edge(b):
                edge_gaps.append((mid, {"why": "ball vanished at the pitch edge", "gap_seconds": round(step_s(a, b), 2),
                                        "last_seen_xy_px": f"{centre(ra)[0]:.0f},{centre(ra)[1]:.0f}",
                                        "next_seen_xy_px": f"{centre(rb)[0]:.0f},{centre(rb)[1]:.0f}"}))
            if is_near(a) or is_near(b):
                near_gaps.append((mid, {"why": "ball vanished close to the camera / in a bottom corner",
                                        "gap_seconds": round(step_s(a, b), 2),
                                        "last_seen_xy_px": f"{centre(ra)[0]:.0f},{centre(ra)[1]:.0f}",
                                        "next_seen_xy_px": f"{centre(rb)[0]:.0f},{centre(rb)[1]:.0f}"}))
        edge_det, near_det = [], []
        if self.poly is not None:
            for f in best.index:
                f = int(f)
                if not usable(f):
                    continue
                if is_edge(f) and float(best.loc[f, "confidence"]) < EDGE_WEAK_CONF:
                    edge_det.append((f, {"why": "weak ball detection at the pitch edge",
                                         "best_confidence": round(float(best.loc[f, "confidence"]), 2)}))
        for f in best.index:
            f = int(f)
            if usable(f) and width(best.loc[f]) >= NEAR_MIN_W:
                near_det.append((f, {"why": "big ball close to the camera", "ball_width_px": int(width(best.loc[f]))}))
        for lst in (edge_gaps, near_gaps, edge_det, near_det):
            rng.shuffle(lst)
        cand["edge"] = edge_gaps + edge_det           # balls the model MISSED first, then weak detections
        cand["near"] = near_gaps + near_det

        # --- multi / unsure / small: straight from the per-frame ball rows
        multi = [int(f) for f in n_balls[n_balls >= 2].index]
        rng.shuffle(multi)
        cand["multi"] = [(f, {"balls_found": int(n_balls[f])}) for f in multi]
        uns = [int(f) for f in best.index[(best.confidence < UNSURE_RANGE[1])]]
        rng.shuffle(uns)
        cand["unsure"] = [(f, {"best_confidence": round(float(best.loc[f, "confidence"]), 2)}) for f in uns]
        small = [int(f) for f in best.index[((best.xmax - best.xmin) <= SMALL_MAX_W)]]
        rng.shuffle(small)
        cand["small"] = [(f, {"ball_width_px": int(best.loc[f, "xmax"] - best.loc[f, "xmin"])}) for f in small]

        # --- odd / fast: need pitch positions (Transformations)
        odd, fast = [], []
        if "is_inside_pitch" in best:
            off = best.index[best.is_inside_pitch.fillna(1).astype(int) == 0]
            odd += [(int(f), {"why": "ball outside the pitch"}) for f in off]
        pos = best.dropna(subset=["x", "y"])
        if len(pos) > 3:
            fr = pos.index.values.astype(float)
            x, y = pos.x.values.astype(float), pos.y.values.astype(float)
            dt = np.diff(fr) / fps
            stride = max(float(np.median(np.diff(frames))), 1.0)
            ok = (dt > 0) & (np.diff(fr) <= 1.5 * stride)                                     # consecutive detections only
            v = np.where(dt > 0, np.hypot(np.diff(x), np.diff(y)) / np.where(dt > 0, dt, 1), 0.0)
            for k in np.where(ok & (v > JUMP_MS))[0]:
                odd.append((int(fr[k + 1]), {"why": f"jumped {v[k]:.0f} m/s from the previous detection"}))
            run = []
            for k in range(len(v)):
                if ok[k] and FAST_MS[0] <= v[k] <= FAST_MS[1]:
                    run.append(k)
                else:
                    if len(run) >= 3:
                        mid = run[len(run) // 2]
                        fast.append((int(fr[mid]), {"speed_m_s": round(float(v[mid]), 1)}))
                    run = []
            if len(run) >= 3:
                mid = run[len(run) // 2]
                fast.append((int(fr[mid]), {"speed_m_s": round(float(v[mid]), 1)}))
        rng.shuffle(odd)
        rng.shuffle(fast)
        cand["odd"], cand["fast"] = odd, fast

        # --- random: any frame
        cand["random"] = [(int(f), {}) for f in rng.permutation(frames)]
        return cand

    def _load_pitch(self):
        try:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            first = con.execute("select min(frame_id) from detected_pitch").fetchone()[0]
            pts = con.execute("select x, y from detected_pitch where frame_id=? order by rowid", (first,)).fetchall()
            con.close()
            return np.array(pts, dtype=np.float32).reshape(-1, 1, 2) if len(pts) >= 3 else None
        except Exception:                                                    # noqa
            return None

    # ------------------------------------------------------------------ run
    def run(self):
        print("-------------------------- BALL TRAINING IMAGES")
        if not os.path.isfile(self.db_path):
            raise SystemExit(f"\nSTOPPED: no detections database at {self.db_path}. Run Detections for this game first.\n")
        if not os.path.isfile(str(self.video_path)):
            raise SystemExit(f"\nSTOPPED: the analysis video was not found: {self.video_path}\n")
        con = sqlite3.connect(self.db_path)
        try:
            cols = {r[1] for r in con.execute("pragma table_info(detected_objects)")}
            frames = np.sort(pd.read_sql("select distinct frame_id from detected_objects", con).frame_id.values)
            sel = "frame_id, xmin, ymin, xmax, ymax, confidence"
            sel += ", is_inside_pitch" if "is_inside_pitch" in cols else ", NULL as is_inside_pitch"
            sel += ", x_transformed_metres as x, y_transformed_metres as y" if "x_transformed_metres" in cols else ", NULL as x, NULL as y"
            ball = pd.read_sql(f"select {sel} from detected_objects where class_name = 'ball'", con)
        finally:
            con.close()
        if len(frames) == 0:
            raise SystemExit("\nSTOPPED: there are no detections in the database for this game.\n")

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.poly = self._load_pitch()
        if self.poly is None:
            print("  (no pitch outline in the database: the 'edge' group will be mostly empty)")
        rng = np.random.default_rng(self.seed)
        cand = self._candidates(frames, ball, fps, rng) if len(ball) else {k: [] for k, _ in QUOTAS}
        if not len(ball):
            print("  no ball was detected at all in this game: only random frames can be offered")
            cand["random"] = [(int(f), {}) for f in rng.permutation(frames)]

        taken = sorted(self._earlier_frames())
        protected = sorted(self._review_frames())
        if protected:
            print(f"  keeping {PROTECT_REVIEW_S:.0f} s clear of the {len(protected)} frames in this game's ball-review test set")
        min_gap = int(MIN_SPACING_S * fps)
        protect_gap = int(PROTECT_REVIEW_S * fps)
        picked, notes = [], {}

        def free(f):
            if any(abs(f - p) < min_gap for p in picked):
                return False
            if protected:
                k = np.searchsorted(protected, f)
                if any(0 <= m < len(protected) and abs(protected[m] - f) < protect_gap for m in (k - 1, k)):
                    return False
            i = np.searchsorted(taken, f)
            return not any(0 <= k < len(taken) and abs(taken[k] - f) < min_gap for k in (i - 1, i))

        def fill(group, quota):
            got = 0
            for f, note in cand[group]:
                if got >= quota or len(picked) >= self.n_frames:
                    break
                if f not in notes and free(f):
                    picked.append(f)
                    notes[f] = (group, note)
                    got += 1
            return got

        shortfall = 0
        for group, share in QUOTAS:
            quota = int(round(self.n_frames * share))
            shortfall += quota - fill(group, quota)
        for group, _ in QUOTAS:                    # hand the unfilled places to whoever still has candidates
            if len(picked) >= self.n_frames:
                break
            fill(group, self.n_frames - len(picked))
        if not picked:
            cap.release()
            raise SystemExit("\nSTOPPED: no new frames are available (everything suitable was exported before).\n")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.out_root, stamp)
        os.makedirs(run_dir, exist_ok=True)
        prefix = self._prefix()
        rows, counts = [], {}
        for fi in sorted(picked):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read()
            if not ok:
                print(f"  frame {fi}: could not be read from the video, skipped")
                continue
            group, note = notes[fi]
            name = f"{prefix}_f{fi:06d}_{group}.jpg"
            cv2.imwrite(os.path.join(run_dir, name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            seconds = fi / fps
            counts[group] = counts.get(group, 0) + 1
            rows.append({"file": name, "frame_id": fi, "time": f"{int(seconds // 60):02d}:{int(seconds % 60):02d}",
                         "group": group, **{k: v for k, v in note.items()}})
        cap.release()
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "_index.csv"), index=False)
        with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"Ball training images - game {prefix} - made {time.strftime('%d %b %Y %H:%M')}\n\n"
                f"{len(rows)} full frames from the analysis video ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))}).\n"
                "The group is in each file name:\n"
                "  lost    the ball was followed, vanished and came back: it is on the pitch in this frame but the model missed it\n"
                "  edge    the ball is at the pitch edge / far touchline (vanished there, or a weak detection): the model's weak spot\n"
                "  near    a big ball close to the camera / in a bottom corner: the model's other weak spot\n"
                "  multi   the model found 2+ 'balls': label only the real one, leave the others unlabelled\n"
                "  unsure  the model was unsure (confidence 0.30-0.50)\n"
                "  odd     ball outside the pitch / an impossible jump: often NOT a ball, in which case leave the frame unlabelled\n"
                "  small   a tiny ball, far side of the pitch\n"
                "  fast    shots and clearances (motion blur)\n"
                "  random  any frame: keep these aside as a TEST set\n\n"
                "In Roboflow: upload the pictures, draw a 'ball' box on the real ball in each, and add frames with no ball as they are\n"
                "(an unlabelled frame is a 'null' example and teaches the model what is not a ball). Names are unique across games and\n"
                "runs. _index.csv gives the reason for each frame and, for 'lost', where the ball was last and next seen (pixels).\n")
        print(f"  {len(rows)} frames  ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})")
        if shortfall:
            print(f"  ({shortfall} places in the groups could not be filled and went to the other groups)")
        print(f"  saved in: {run_dir}")
        return run_dir
