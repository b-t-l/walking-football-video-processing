"""Detect pitch landmarks on a broadcast / edited-video frame using the Roboflow YOLO-pose weights."""
import os

import numpy as np
from ultralytics import YOLO

from .keypoints import KEYPOINT_NAMES

DEFAULT_WEIGHTS = os.path.join("models", "pitch", "weights.pt")
DEFAULT_CONF = 0.25          # box confidence
DEFAULT_KPT_CONF = 0.30      # per-keypoint visibility / confidence floor
DEFAULT_IMGSZ = 640


class DetectPitchKeypoints:
    def __init__(self, weights_path=None, conf=DEFAULT_CONF, kpt_conf=DEFAULT_KPT_CONF, imgsz=DEFAULT_IMGSZ,
                 keypoint_names=None, device=None):
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Pitch keypoint weights not found: {self.weights_path}")
        self.conf = float(conf)
        self.kpt_conf = float(kpt_conf)
        self.imgsz = int(imgsz)
        self.keypoint_names = list(keypoint_names) if keypoint_names is not None else list(KEYPOINT_NAMES)
        self.device = device
        self.model = YOLO(self.weights_path)

    def detect(self, frame_bgr):
        """Return list of keypoints for the best pitch instance:
        [{index, name, x, y, conf}, ...] — only points above kpt_conf.
        Empty list if nothing usable is found.
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []
        r = results[0]
        if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
            return []

        # Pick the highest-confidence pitch box when several are returned
        best_i = 0
        if r.boxes is not None and len(r.boxes) > 1 and r.boxes.conf is not None:
            best_i = int(r.boxes.conf.cpu().numpy().argmax())

        kpts = r.keypoints.data[best_i].cpu().numpy()  # (N, 2) or (N, 3)
        out = []
        for i, row in enumerate(kpts):
            if len(row) >= 3:
                x, y, c = float(row[0]), float(row[1]), float(row[2])
            else:
                x, y, c = float(row[0]), float(row[1]), 1.0
            if c < self.kpt_conf or not np.isfinite(x) or not np.isfinite(y):
                continue
            name = self.keypoint_names[i] if i < len(self.keypoint_names) else str(i)
            out.append({"index": i, "name": name, "x": x, "y": y, "conf": c})
        return out
