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

  * every picture is also scored by the team model. Any picture it is less than 90% sure of (or where it would have chosen a
    class the pipeline is not allowed to use, e.g. Goalkeeper) goes in a single _unsure folder instead, named with its score
    (41pc_Polis-Lions_...) so the least sure sort to the top. Those are the ones most worth correcting and training on.

Layout:   <game folder>/team-images/<timestamp>/<label>/<picture>.jpg   plus _unsure/, _index.csv and README.txt in <timestamp>/

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
UNSURE_BELOW = 0.90              # a picture the team model is less than this sure of goes to the _unsure folder (0.90 = 90%)
UNSURE_FOLDER = "_unsure"


def _safe(text):
    t = re.sub(r"[^A-Za-z0-9 _.-]+", "", str(text or "")).strip()
    return t or "Unassigned"


class ExtractTeamImages:

    def __init__(self, db_path, video_path, out_root, game_record=None, n_frames=10, min_conf=0.6, seed=None,
                 team_model_path=None, unsure_below=UNSURE_BELOW):
        self.db_path = db_path
        self.video_path = video_path
        self.out_root = out_root
        self.game_record = game_record or {}
        self.n_frames = max(1, int(n_frames))
        self.min_conf = min_conf
        self.seed = seed                      # None = different frames every run
        self.team_model_path = team_model_path   # the team model, to score every picture (None = no scoring, no _unsure folder)
        self.unsure_below = unsure_below

    def _load_team_model(self):
        if not self.team_model_path:
            return None
        if not os.path.isfile(str(self.team_model_path)):
            print(f"  (team model not found at {self.team_model_path}: pictures are not scored, no {UNSURE_FOLDER} folder)")
            return None
        try:
            from ultralytics import YOLO
            return YOLO(str(self.team_model_path))
        except Exception as e:                                       # scoring is a bonus: never stop the step for it
            print(f"  (could not load the team model, pictures are not scored: {e})")
            return None

    @staticmethod
    def _score(model, crops):
        """Every class probability the team model gives each picture, one dict per picture.
        Fed exactly like the pipeline feeds it: RGB, squashed to 192 x 192, 0-1."""
        import torch
        batch = []
        for c in crops:
            rgb = cv2.cvtColor(cv2.resize(c, (192, 192)), cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0)
        out = []
        for r in model.predict(torch.stack(batch), imgsz=(192, 192), verbose=False):
            p = r.probs.data.cpu().numpy()
            out.append({r.names[i]: float(p[i]) for i in range(len(p))})
        return out

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

        model = self._load_team_model()
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
            cuts = []
            for r in d[d.frame_id == fi].itertuples():
                x0, y0 = max(0, int(round(r.xmin))), max(0, int(round(r.ymin)))
                x1, y1 = min(W, int(round(r.xmax))), min(H, int(round(r.ymax)))
                if x1 - x0 < MIN_W or y1 - y0 < MIN_H:
                    continue
                cuts.append((r, im[y0:y1, x0:x1], x1 - x0, y1 - y0))
            scores = self._score(model, [c[1] for c in cuts]) if (model is not None and cuts) else [None] * len(cuts)
            for (r, crop, w, h), probs in zip(cuts, scores):
                label = _safe(r.team_raw)
                tid = "x" if pd.isna(r.tracker_id) else str(int(r.tracker_id))
                name = f"{prefix}_f{fi:06d}_id{tid}_{_safe(r.class_name)}.jpg"
                # how sure the team model is of the folder this picture would go in, and what it would pick with no
                # restrictions (a Goalkeeper / other-team answer the pipeline is not allowed to use also counts as unsure)
                said_conf = top = top_conf = ""
                unsure = False
                if probs:
                    top = max(probs, key=probs.get)
                    top_conf = round(probs[top], 3)
                    said_conf = round(probs.get(str(r.team_raw), 0.0), 3)
                    unsure = said_conf < self.unsure_below or top != str(r.team_raw)
                if unsure:
                    folder = os.path.join(run_dir, UNSURE_FOLDER)
                    # the model's score first, so the least sure pictures sort to the top in Finder
                    fname = f"{int(round(said_conf * 100)):02d}pc_{label.replace(' ', '-')}_{name}"
                    counts[UNSURE_FOLDER] = counts.get(UNSURE_FOLDER, 0) + 1
                else:
                    folder = os.path.join(run_dir, label)
                    fname = name
                    counts[label] = counts.get(label, 0) + 1
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"file": f"{os.path.basename(folder)}/{fname}", "frame_id": fi, "time": mmss, "tracker_id": tid,
                             "detector_class": r.class_name, "team_model_said": r.team_raw, "after_smoothing": r.team,
                             "team_model_confidence": said_conf, "team_model_top_pick": top, "team_model_top_confidence": top_conf,
                             "unsure": unsure, "detector_confidence": round(float(r.confidence), 3), "width": w, "height": h})
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
                f"  START WITH {UNSURE_FOLDER}: pictures the model scored under {int(self.unsure_below * 100)}% (or would have called Goalkeeper / another team).\n"
                "  The file name starts with its score, the model's best allowed answer follows. These teach it the most.\n"
                f"  Move each one into the right team folder (or Goalkeeper / Referee / delete it), so no {UNSURE_FOLDER} folder is uploaded.\n"
                "  1. Look through each folder and drag any picture that is in the wrong folder into the right one.\n"
                "     For a team the model does not know yet, make a new folder named after the team and move its players there.\n"
                "     Move pictures that are not a player of either team (referee, a person off the pitch, two players in one box)\n"
                "     to folders called Referee / Goalkeeper, or delete them.\n"
                "  2. Upload the folder to Roboflow as a classification dataset: the folder names become the class names.\n"
                "  3. In Roboflow set the resize preprocessing to 'Stretch to 192x192' (the pipeline squashes every crop to a square).\n"
                "  4. Train, download the weights, and put them in models/teams/ (then point TEAM_MODEL_PATH in data/create_database.py at them).\n\n"
                "_index.csv lists every picture: its frame, time, tracker id and what the model said before and after smoothing.\n")
        print(f"  {len(rows)} pictures from {len(picks)} frames  ({', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))})")
        if UNSURE_FOLDER in counts:
            print(f"  {counts[UNSURE_FOLDER]} of them are in {UNSURE_FOLDER} (the team model was under {int(self.unsure_below * 100)}% sure): start with those")
        print(f"  saved in: {run_dir}")
        print("  Next: fix the folders, upload to Roboflow (see README.txt in that folder).")
        return run_dir
