"""Broadcast-video pitch training images: full frames from the edited (TV / broadcast) video for Roboflow keypoint labelling.

These pictures are for training a pitch-landmark model on the panning broadcast camera — not the fixed analysis camera.
No boxes or keypoints are drawn: you label them in Roboflow.

WHICH FRAMES (N = 40 by default):
  Spread evenly across the broadcast video so wide shots, tight pans and both ends of the pitch show up.
  Frames are at least MIN_SPACING_S seconds apart, and frames an earlier run for this game already exported are never
  picked again, so you can run the step as often as you like and always get new pictures.

FILE NAMES are unique across games (Roboflow needs that):  <game>_f<frame>.jpg  e.g. 0022_f012480.jpg
(the game number and the frame index in the broadcast video).

Layout:   <game folder>/broadcast-pitch-images/<timestamp>/<pictures>.jpg
          plus _index.csv and README.txt in <timestamp>/
"""
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd

MIN_SPACING_S = 3.0          # picked frames are at least this far apart (and from earlier exports)
DEFAULT_N_FRAMES = 40
JPEG_QUALITY = 95


class ExtractBroadcastPitchImages:

    def __init__(self, video_path, out_root, game_record=None, n_frames=DEFAULT_N_FRAMES, seed=None):
        self.video_path = video_path
        self.out_root = out_root
        self.game_record = game_record or {}
        self.n_frames = max(1, int(n_frames))
        self.seed = seed                       # None = different shuffle of near-ties every run

    def _prefix(self):
        try:
            return f"{int(self.game_record['game_id']):04d}"
        except (KeyError, TypeError, ValueError):
            return "game"

    def _earlier_frames(self):
        """Frame indices already exported by earlier runs for this game (from their _index.csv files)."""
        out = set()
        for f in glob.glob(os.path.join(self.out_root, "*", "_index.csv")):
            try:
                out.update(int(v) for v in pd.read_csv(f)["frame_id"])
            except Exception:                                                # noqa
                pass
        return out

    def _pick_frames(self, n_video_frames, fps, taken):
        """Evenly spaced candidates across the video, skipping ones too close to earlier exports or each other."""
        if n_video_frames < 1:
            return []
        min_gap = max(1, int(round(MIN_SPACING_S * fps)))
        taken_sorted = sorted(taken)
        # Over-sample along the timeline, then keep the first free slots in order (keeps spread).
        n_cand = max(self.n_frames * 4, self.n_frames + 10)
        raw = np.linspace(0, n_video_frames - 1, num=n_cand)
        # Small per-run jitter so repeats do not always land on the same linspace picks when many are free.
        rng = np.random.default_rng(self.seed)
        jitter = rng.uniform(-0.35, 0.35, size=len(raw)) * (n_video_frames / max(n_cand, 1))
        candidates = np.clip(np.round(raw + jitter).astype(int), 0, n_video_frames - 1)
        # Stable unique order along the video
        ordered = []
        seen = set()
        for f in candidates:
            fi = int(f)
            if fi not in seen:
                seen.add(fi)
                ordered.append(fi)
        ordered.sort()

        def free(f, picked):
            if any(abs(f - p) < min_gap for p in picked):
                return False
            i = np.searchsorted(taken_sorted, f)
            return not any(0 <= k < len(taken_sorted) and abs(taken_sorted[k] - f) < min_gap
                           for k in (i - 1, i))

        picked = []
        for f in ordered:
            if len(picked) >= self.n_frames:
                break
            if free(f, picked):
                picked.append(f)

        # If still short (dense earlier exports), walk every min_gap-th frame.
        if len(picked) < self.n_frames:
            for f in range(0, n_video_frames, min_gap):
                if len(picked) >= self.n_frames:
                    break
                if free(f, picked):
                    picked.append(f)
            picked.sort()
        return picked[: self.n_frames]

    def run(self):
        print("\n-------------------------------------------------")
        print("BROADCAST VIDEO PITCH TRAINING IMAGES: frames from the edited / broadcast video for Roboflow")
        if not self.video_path or not os.path.exists(str(self.video_path)):
            raise SystemExit(
                f"\nSTOPPED: the broadcast (edited) video was not found"
                + (f": {self.video_path}" if self.video_path else "")
                + "\nPut it in the game's videos folder as <game number>_edited.mp4, or choose it on the game page.\n"
            )

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise SystemExit(f"\nSTOPPED: could not open the broadcast video: {self.video_path}\n")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_video_frames < 1:
            # Some containers report 0; fall back to reading length via duration.
            duration_ms = cap.get(cv2.CAP_PROP_DURATION)  # may be 0
            if duration_ms and duration_ms > 0:
                n_video_frames = max(1, int(round((duration_ms / 1000.0) * fps)))
        if n_video_frames < 1:
            cap.release()
            raise SystemExit(f"\nSTOPPED: the broadcast video has no readable frames: {self.video_path}\n")

        taken = self._earlier_frames()
        if taken:
            print(f"  skipping {len(taken)} frames already exported in earlier runs for this game")
        picked = self._pick_frames(n_video_frames, fps, taken)
        if not picked:
            cap.release()
            raise SystemExit(
                "\nSTOPPED: no new frames are available "
                "(everything suitable was exported before, or the video is too short).\n"
            )

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.out_root, stamp)
        os.makedirs(run_dir, exist_ok=True)
        prefix = self._prefix()
        rows = []
        for fi in picked:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read()
            if not ok:
                print(f"  frame {fi}: could not be read from the video, skipped")
                continue
            name = f"{prefix}_f{fi:06d}.jpg"
            cv2.imwrite(os.path.join(run_dir, name), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            seconds = fi / fps
            rows.append({
                "file": name,
                "frame_id": fi,
                "time": f"{int(seconds // 60):02d}:{int(seconds % 60):02d}",
                "seconds": round(seconds, 3),
            })
        cap.release()

        if not rows:
            raise SystemExit("\nSTOPPED: no frames could be read from the broadcast video.\n")

        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "_index.csv"), index=False)
        with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
            f.write(
                f"Broadcast video pitch training images - game {prefix} - made {time.strftime('%d %b %Y %H:%M')}\n\n"
                f"{len(rows)} full frames from the broadcast (edited) video, spaced about {MIN_SPACING_S:.0f} s apart "
                "across the match.\n"
                "Upload these to a Roboflow keypoint project and label the visible pitch landmarks "
                "(corners, halfway-line touchline points, centre-circle samples, etc.).\n"
                "Names are unique across games and runs. _index.csv lists each file with its frame index and time.\n"
                f"\nSource video: {self.video_path}\n"
            )
        print(f"  {len(rows)} frames from the broadcast video")
        print(f"  saved in: {run_dir}")
        return run_dir
