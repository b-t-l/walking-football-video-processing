"""Team training images: pictures of the players from a few random frames, ready to correct and upload to Roboflow.

The pipeline's team model is a classifier trained on pictures of each team's players. To teach it a new team (or fix a
mistake) it needs new, correctly labelled pictures. This step makes them from a game that has already been through
Detections:

  * the game's detected frames are split into N equal stretches and one random frame is picked from each, so the pictures
    come from the whole game, not just one moment (N = 10 by default; run it again for ten different frames)
  * every player / goalkeeper / referee the pipeline kept in those frames is cut out of the analysis video at full
    resolution (exactly the box the detector drew, nothing added)
  * each picture is filed in a folder named after what the CURRENT team model said, so wrong ones are easy to spot and
    drag to the right folder. Roboflow reads folder names as the class when you upload a folder.

Layout:   <game folder>/team-images/<timestamp>/<label>/<picture>.jpg   plus _index.csv and README.txt in <timestamp>/

About the picture size: the team model squashes every crop to 192 x 192 whatever its shape, so in Roboflow set the resize
to "Stretch to 192x192" (not fit/letterbox) and the pictures will look to the model exactly as they do in the pipeline.
"""
import os
import re
import time
import sqlite3

import cv2
import numpy as np
import pandas as pd

MIN_W, MIN_H = 20, 40            # smaller boxes are too small to tell a kit from; skipped


def _safe(text):
    t = re.sub(r"[^A-Za-z0-9 _.-]+", "", str(text or "")).strip()
    return t or "Unassigned"


class ExtractTeamImages:

    def __init__(self, db_path, video_path, out_root, game_record=None, n_frames=10, min_conf=0.6, seed=None):
        self.db_path = db_path
        self.video_path = video_path
        self.out_root = out_root
        self.game_record = game_record or {}
        self.n_frames = max(1, int(n_frames))
        self.min_conf = min_conf
        self.seed = seed                      # None = different frames every run

    def _prefix(self):
        try:
            return f"{int(self.game_record['game_id']):04d}"
        except (KeyError, TypeError, ValueError):
            return "game"

    def run(self):
        print("-------------------------- TEAM TRAINING IMAGES")
        if not os.path.isfile(self.db_path):
            raise SystemExit(f"\nSTOPPED: no detections database at {self.db_path}. Run Detections for this game first.\n")
        if not os.path.isfile(str(self.video_path)):
            raise SystemExit(f"\nSTOPPED: the analysis video was not found: {self.video_path}\n")
        con = sqlite3.connect(self.db_path)
        try:
            cols = {r[1] for r in con.execute("pragma table_info(detected_objects)")}
            # team_raw (the team model's own answer) is only added once Transformations has run; before that, 'team' IS the model's answer
            raw = "team_raw" if "team_raw" in cols else "team"
            d = pd.read_sql(f"select frame_id, xmin, ymin, xmax, ymax, confidence, tracker_id, class_name, {raw} as team_raw, team "
                            "from detected_objects where class_name != 'ball' and confidence >= ?", con, params=(self.min_conf,))
        finally:
            con.close()
        if d.empty:
            raise SystemExit("\nSTOPPED: there are no player detections in the database for this game.\n")

        frames = np.sort(d.frame_id.unique())
        rng = np.random.default_rng(self.seed)
        n = min(self.n_frames, len(frames))
        picks = []
        for part in np.array_split(frames, n):                       # one random frame from each equal stretch of the game
            if len(part):
                picks.append(int(rng.choice(part)))
        picks.sort()

        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.out_root, stamp)
        os.makedirs(run_dir, exist_ok=True)
        prefix = self._prefix()
        rows, counts = [], {}
        for fi in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read()
            if not ok:
                print(f"  frame {fi}: could not be read from the video, skipped")
                continue
            H, W = im.shape[:2]
            seconds = fi / fps
            mmss = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
            for r in d[d.frame_id == fi].itertuples():
                x0, y0 = max(0, int(round(r.xmin))), max(0, int(round(r.ymin)))
                x1, y1 = min(W, int(round(r.xmax))), min(H, int(round(r.ymax)))
                if x1 - x0 < MIN_W or y1 - y0 < MIN_H:
                    continue
                label = _safe(r.team_raw)
                folder = os.path.join(run_dir, label)
                os.makedirs(folder, exist_ok=True)
                tid = "x" if pd.isna(r.tracker_id) else str(int(r.tracker_id))
                name = f"{prefix}_f{fi:06d}_id{tid}_{_safe(r.class_name)}.jpg"
                cv2.imwrite(os.path.join(folder, name), im[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                counts[label] = counts.get(label, 0) + 1
                rows.append({"file": f"{label}/{name}", "frame_id": fi, "time": mmss, "tracker_id": tid,
                             "detector_class": r.class_name, "team_model_said": r.team_raw, "after_smoothing": r.team,
                             "confidence": round(float(r.confidence), 3), "width": x1 - x0, "height": y1 - y0})
        cap.release()
        if not rows:
            raise SystemExit("\nSTOPPED: no usable player pictures could be cut out.\n")
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "_index.csv"), index=False)
        with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"Team training images - game {prefix} - made {time.strftime('%d %b %Y %H:%M')}\n\n"
                f"{len(rows)} pictures of players from {len(picks)} random frames of the analysis video "
                f"(at {', '.join(sorted({r['time'] for r in rows}))}).\n\n"
                "Each folder is named after what the CURRENT team model decided. Some will be wrong - that is the point:\n"
                "  1. Look through each folder and drag any picture that is in the wrong folder into the right one.\n"
                "     For a team the model does not know yet, make a new folder named after the team and move its players there.\n"
                "     Move pictures that are not a player of either team (referee, a person off the pitch, two players in one box)\n"
                "     to folders called Referee / Goalkeeper, or delete them.\n"
                "  2. Upload the folder to Roboflow as a classification dataset: the folder names become the class names.\n"
                "  3. In Roboflow set the resize preprocessing to 'Stretch to 192x192' (the pipeline squashes every crop to a square).\n"
                "  4. Train, download the weights, and put them in models/teams/ (then point TEAM_MODEL_PATH in data/create_database.py at them).\n\n"
                "_index.csv lists every picture: its frame, time, tracker id and what the model said before and after smoothing.\n")
        print(f"  {len(rows)} pictures from {len(picks)} frames  ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})")
        print(f"  saved in: {run_dir}")
        print("  Next: fix the folders, upload to Roboflow (see README.txt in that folder).")
        return run_dir
