"""Per-frame homography from detected pitch keypoints to world metres; project thirds grid."""
import cv2
import numpy as np

from .keypoints import (
    DEFAULT_LENGTH_M,
    DEFAULT_WIDTH_M,
    world_keypoints,
)

MIN_POINTS = 4
RANSAC_REPROJ_THRESH = 8.0   # pixels
# Degenerate H guard: centre-line-only detections make findHomography "succeed" but collapse
# every projected line to a single pixel (invisible thirds). Reject those.
MIN_WORLD_SPAN_X_M = 8.0
MIN_WORLD_SPAN_Y_M = 6.0
MIN_IMAGE_SPAN_PX = 80.0
MIN_OUTLINE_SPAN_PX = 200.0
MIN_THIRD_LINE_LEN_PX = 150.0


class EditedPitchHomography:
    def __init__(self, length_m=None, width_m=None, centre_radius_m=None,
                 end_arc_radius_m=None, end_arc_max_depth_m=None,
                 min_points=MIN_POINTS):
        self.length_m = float(length_m if length_m is not None else DEFAULT_LENGTH_M)
        self.width_m = float(width_m if width_m is not None else DEFAULT_WIDTH_M)
        self.world = world_keypoints(
            length_m=self.length_m,
            width_m=self.width_m,
            centre_radius_m=centre_radius_m,
            end_arc_radius_m=end_arc_radius_m,
            end_arc_max_depth_m=end_arc_max_depth_m,
        )
        self.min_points = int(min_points)
        self.H = None            # 3x3 world -> pixel (maps metres to frame)
        self.last_inliers = 0
        self.last_n = 0
        self.last_reject = None  # why estimate returned None (for debug)

    def estimate(self, detections):
        """detections: [{name, x, y, conf}, ...]. Sets and returns H (world→pixel) or None."""
        self.last_reject = None
        src, dst = [], []
        for d in detections:
            name = d.get("name")
            if name not in self.world:
                continue
            wx, wy = self.world[name]
            src.append((wx, wy))
            dst.append((d["x"], d["y"]))
        self.last_n = len(src)
        if len(src) < self.min_points:
            self.H = None
            self.last_inliers = 0
            self.last_reject = f"need>={self.min_points} pts (got {len(src)})"
            return None
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        if float(np.ptp(src[:, 0])) < MIN_WORLD_SPAN_X_M or float(np.ptp(src[:, 1])) < MIN_WORLD_SPAN_Y_M:
            self.H = None
            self.last_inliers = 0
            self.last_reject = (
                f"world span too small "
                f"(dx={np.ptp(src[:, 0]):.1f}m dy={np.ptp(src[:, 1]):.1f}m)"
            )
            return None
        if float(np.ptp(dst[:, 0])) < MIN_IMAGE_SPAN_PX or float(np.ptp(dst[:, 1])) < MIN_IMAGE_SPAN_PX:
            self.H = None
            self.last_inliers = 0
            self.last_reject = (
                f"image span too small "
                f"(dx={np.ptp(dst[:, 0]):.0f}px dy={np.ptp(dst[:, 1]):.0f}px)"
            )
            return None
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESH)
        if H is None:
            self.H = None
            self.last_inliers = 0
            self.last_reject = "findHomography failed"
            return None
        inliers = int(mask.sum()) if mask is not None else len(src)
        ok, reason = self._projection_ok(H)
        if not ok:
            self.H = None
            self.last_inliers = inliers
            self.last_reject = reason
            return None
        self.H = H
        self.last_inliers = inliers
        self.last_reject = None
        return H

    def _projection_ok(self, H):
        """Reject collapsed / sky-line projections from poorly conditioned H."""
        outline = self.outline_polyline(H=H)
        if outline is None or len(outline) < 4:
            return False, "outline empty"
        if not np.isfinite(outline).all():
            return False, "outline non-finite"
        ox, oy = float(np.ptp(outline[:, 0])), float(np.ptp(outline[:, 1]))
        if ox < MIN_OUTLINE_SPAN_PX or oy < MIN_OUTLINE_SPAN_PX:
            return False, f"outline collapsed (dx={ox:.0f} dy={oy:.0f})"
        for i, line in enumerate(self.thirds_polylines(H=H)):
            if line is None or len(line) < 2 or not np.isfinite(line).all():
                return False, f"thirds[{i}] bad"
            seg = np.hypot(np.diff(line[:, 0]), np.diff(line[:, 1]))
            if float(seg.sum()) < MIN_THIRD_LINE_LEN_PX:
                return False, f"thirds[{i}] too short ({seg.sum():.0f}px)"
        return True, None

    def project_points(self, world_pts, H=None):
        """world_pts: Nx2 metres → Nx2 pixels."""
        H = self.H if H is None else H
        if H is None:
            return None
        pts = np.asarray(world_pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, H)
        return out.reshape(-1, 2)

    def thirds_polylines(self, samples_per_line=80, H=None):
        """Four thirds-grid lines in pixel space (same geometry as analysis overlay)."""
        H = self.H if H is None else H
        if H is None:
            return []
        L, W = self.length_m, self.width_m
        lines_world = []
        for x in (-L / 6.0, L / 6.0):
            lines_world.append([(x, -W / 2.0 + W * t) for t in np.linspace(0.0, 1.0, samples_per_line)])
        for y in (-W / 6.0, W / 6.0):
            lines_world.append([(-L / 2.0 + L * t, y) for t in np.linspace(0.0, 1.0, samples_per_line)])
        out = []
        for line in lines_world:
            px = self.project_points(line, H=H)
            if px is not None:
                out.append(px)
        return out

    def outline_polyline(self, samples_per_side=60, H=None):
        """Pitch rectangle outline in pixels (for debug overlays)."""
        H = self.H if H is None else H
        if H is None:
            return None
        L, W = self.length_m, self.width_m
        corners = [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2), (-L / 2, -W / 2)]
        pts = []
        for a, b in zip(corners[:-1], corners[1:]):
            for t in np.linspace(0.0, 1.0, samples_per_side, endpoint=False):
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        return self.project_points(pts, H=H)
