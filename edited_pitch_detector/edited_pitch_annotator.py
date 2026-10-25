"""Draw dotted thirds (and optional debug keypoints) on a broadcast / edited frame."""
import cv2
import numpy as np

from .detect_pitch_keypoints import DetectPitchKeypoints
from .edited_pitch_homography import EditedPitchHomography
from .keypoints import dims_from_game_record

THIRD_COLOUR = (0, 255, 0)       # BGR green — match analysis overlay
OUTLINE_COLOUR = (0, 200, 0)
KPT_COLOUR = (0, 165, 255)       # orange for detected points
THIRD_ALPHA = 0.55
# Base sizes at ~4K; scaled by frame height / 2160 in draw_onto
DOT_GAP = 14
DOT_RADIUS = 3
REF_FRAME_HEIGHT = 2160


def _dotted_points(pts, gap):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return []
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return []
    n = max(2, int(cum[-1] // gap) + 1)
    d = np.linspace(0.0, cum[-1], n)
    return list(zip(np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])))


class EditedPitchAnnotator:
    """Detect pitch keypoints → homography → draw thirds on one frame (or a sequence)."""

    def __init__(self, game_record=None, weights_path=None, detector=None, homography=None,
                 draw_keypoints=True, draw_outline=True, draw_status=True, hold_last_h=True):
        dims = dims_from_game_record(game_record)
        self.detector = detector or DetectPitchKeypoints(weights_path=weights_path)
        self.homography = homography or EditedPitchHomography(
            length_m=dims.get("length_m"),
            width_m=dims.get("width_m"),
            centre_radius_m=dims.get("centre_radius_m"),
            end_arc_radius_m=dims.get("end_arc_radius_m"),
            end_arc_max_depth_m=dims.get("end_arc_max_depth_m"),
        )
        self.draw_keypoints = bool(draw_keypoints)
        self.draw_outline = bool(draw_outline)
        self.draw_status = bool(draw_status)
        self.hold_last_h = bool(hold_last_h)
        self.last_detections = []
        self.last_ok = False
        self._held_h = None
        self.stats = {"frames": 0, "ok": 0, "hold": 0, "fail": 0, "rejects": {}}

    def update(self, frame_bgr):
        """Detect keypoints and estimate H from a clean video frame (call before HUD overlays)."""
        dets = self.detector.detect(frame_bgr)
        self.last_detections = dets
        H = self.homography.estimate(dets)
        self.stats["frames"] += 1
        if H is not None:
            self._held_h = H.copy()
            self.last_ok = True
            self.stats["ok"] += 1
        elif self.hold_last_h and self._held_h is not None:
            H = self._held_h
            self.homography.H = H
            self.last_ok = False  # fresh estimate failed; drawing from hold
            self.stats["hold"] += 1
        else:
            self.last_ok = False
            self.stats["fail"] += 1
            why = self.homography.last_reject or "unknown"
            self.stats["rejects"][why] = self.stats["rejects"].get(why, 0) + 1
        return H

    def stats_summary(self):
        s = self.stats
        parts = [f"frames={s['frames']} H_ok={s['ok']} hold={s['hold']} fail={s['fail']}"]
        if s["rejects"]:
            top = sorted(s["rejects"].items(), key=lambda x: -x[1])[:3]
            parts.append("rejects: " + "; ".join(f"{c}× {r}" for r, c in top))
        return "  ".join(parts)

    def draw_onto(self, frame_bgr):
        """Draw thirds (and optional debug) onto frame in place — above video, below HUD."""
        H = self.homography.H if self.homography.H is not None else self._held_h
        scale = max(0.5, float(frame_bgr.shape[0]) / REF_FRAME_HEIGHT)
        gap = max(8, int(round(DOT_GAP * scale)))
        radius = max(2, int(round(DOT_RADIUS * scale)))
        if H is not None:
            overlay = frame_bgr.copy()
            if self.draw_outline:
                outline = self.homography.outline_polyline(H=H)
                if outline is not None:
                    for cx, cy in _dotted_points(outline, gap):
                        cv2.circle(overlay, (int(round(cx)), int(round(cy))), radius, OUTLINE_COLOUR, -1, cv2.LINE_AA)
            for line in self.homography.thirds_polylines(H=H):
                for cx, cy in _dotted_points(line, gap):
                    cv2.circle(overlay, (int(round(cx)), int(round(cy))), radius, THIRD_COLOUR, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, THIRD_ALPHA, frame_bgr, 1.0 - THIRD_ALPHA, 0, frame_bgr)

        if self.draw_keypoints:
            for d in self.last_detections:
                p = (int(round(d["x"])), int(round(d["y"])))
                cv2.circle(frame_bgr, p, max(4, int(round(6 * scale))), KPT_COLOUR, -1, cv2.LINE_AA)
                label = f'{d["index"]}:{d["name"]} ({d["conf"]:.2f})'
                cv2.putText(frame_bgr, label, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_bgr, label, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if self.draw_status:
            why = self.homography.last_reject
            if self.last_ok:
                tag = "OK"
            elif H is not None:
                tag = "HOLD"
            else:
                tag = "FAIL"
            status = (
                f"kpts={len(self.last_detections)}  H={tag}  "
                f"inliers={self.homography.last_inliers}/{self.homography.last_n}"
            )
            if why and tag == "FAIL":
                status += f"  ({why})"
            cv2.putText(frame_bgr, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame_bgr, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    def annotate(self, frame_bgr):
        """Return a copy of the frame with thirds drawn when homography succeeds (or last H is held)."""
        out = frame_bgr.copy()
        self.update(frame_bgr)
        return self.draw_onto(out)
