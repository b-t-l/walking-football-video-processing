"""
fitted_pitch_transformer.py

Replacement for the flat cv2.getPerspectiveTransform homography used when
PITCH_DETECTION_OVERRIDE is on. A straight perspective homography assumes a flat,
undistorted (pinhole) camera. Insta360 X4 footage reframed through Insta360 Studio's
Dewarp export still carries residual lens curvature, which a flat homography cannot
remove -- it produces a skewed, off-centre "overhead" view (confirmed visually: an
off-centre, squashed centre circle).

This module fits, independently for EVERY processed video, a parametric model:

    world = Homography( undistort(pixel; k1, k2) )

 - undistort() is a 2-term radial correction about the image centre (fixed, not
   fitted -- see notes below on why the centre is fixed rather than floated).
 - Homography is a standard 8-DOF projective transform (3x3 matrix, H[2,2]=1).

Fitting uses scipy.optimize.least_squares (Levenberg-Marquardt / trust-region) against
two kinds of constraints built from the Pitch Calibrator's exported JSON:

  1. HARD constraints: the 6 pitch outline points (left_bottom, left_top, right_top,
     right_bottom, centre_top, centre_bottom) must map to their known world
     coordinates exactly (residual = predicted_world - true_world, x and y).

  2. SOFT constraints: the centre-circle and goal-end-arc points must map to a point
     at the KNOWN RADIUS from their known centre -- not a known (x,y), since we don't
     know each point's angular position on the circle, only that it's somewhere on
     it (residual = |predicted_world - known_centre| - known_radius).

This is a deliberate, independent fit per video (no cached/shared lens model reused
across games): every video gets its own full point set and its own freshly-fitted
mapping, because framing/zoom/reframing can differ video to video even with the same
physical camera and lens.

WORLD COORDINATE CONVENTION
----------------------------
Origin at the pitch centre. X runs along the length axis (goal-to-goal): negative
toward the "left" end, positive toward the "right" end, matching the calibrator's
left_* / right_* vertex naming. Y runs along the width axis (touchline-to-touchline):
positive toward the "top" side, negative toward the "bottom" side, matching the
calibrator's *_top / *_bottom naming.

    left_bottom  = (-length/2, -width/2)      left_top     = (-length/2, +width/2)
    right_top    = (+length/2, +width/2)      right_bottom = (+length/2, -width/2)
    centre_top   = (0,         +width/2)      centre_bottom= (0,         -width/2)

    centre_circle center = (0, 0),            radius = pitch_dimensions_m.centre_radius
    end_arc_left  center = (-length/2, 0),     radius = pitch_dimensions_m.end_arc_radius
    end_arc_right center = (+length/2, 0),     radius = pitch_dimensions_m.end_arc_radius

WHY A FIXED DISTORTION CENTRE, k1/k2 ONLY (NOT A FREE CENTRE, NOT MORE TERMS, NOT A
FREE-FORM SPLINE)
----------------------------------------------------------------------------------
Several alternatives were tried and rejected against this project's actual
calibration data (6 outline + up to 10 circle points):

  - A free-form thin-plate-spline warp interpolates the given points exactly, but a
    leave-one-out test showed 5-11m prediction error on a ~30-45m pitch -- with only
    16 points it has too many degrees of freedom and effectively memorises noise
    rather than generalising.
  - Floating the distortion centre as 2 extra free parameters, or adding 3rd-order
    radial + tangential (Brown-Conrady) terms, did not reduce residuals versus the
    simple fixed-centre k1/k2 model, and in leave-one-out testing was no more (and
    sometimes less) stable.

The fixed-centre k1/k2 + 8-DOF homography model is therefore the simplest model that
is still physically structured (smooth, well-conditioned, bounded extrapolation) and
empirically matches the richer alternatives on both in-sample residual and
leave-one-out stability.

KNOWN ACCURACY CAVEAT
----------------------
With the game_id 22 calibration data used during development, this model fit with
roughly 1.0-1.5m RMS error on the outline points and ~2.0-2.5m RMS error on the
circle-radius constraint (on a 30m x 45m pitch) -- worse at the frame corners (most
residual lens curvature) than near the centre. Two likely contributors, both worth
fixing at the source rather than compensating for in code:

  1. pitch_dimensions_m.width/length may still be CWFF-range placeholders rather
     than Mark's actual measured pitch, while centre_radius/end_arc_radius are
     confirmed measurements. A mismatch between a guessed outline size and a
     measured circle size pulls the fit in two directions at once.
  2. Click precision on a 1920x1080 frame: a few pixels of error at the far/corner
     regions (where perspective compresses pixels-per-metre) can translate into a
     metre or more of world error.

Supplying the real measured pitch width/length (and, if useful later, placing goal
post points with a known goal width) should directly improve every future fit.
"""

import json
import numpy as np
from scipy.optimize import least_squares


class FittedPitchTransformer:

    # how large a radial coefficient we consider physically plausible
    K_BOUND = 3.0

    def __init__(self, calibration, verbose=True):
        """
        calibration: either the parsed dict from a Pitch Calibrator export, or the
        raw JSON string (as stored in games-logger.xlsx's pitch_calibrator_json column).
        """
        if isinstance(calibration, str):
            calibration = json.loads(calibration)
        self.calibration = calibration
        self.verbose = verbose

        dims = calibration['pitch_dimensions_m']
        self.width = float(dims['width'])
        self.length = float(dims['length'])
        self.centre_radius = float(dims.get('centre_radius', 0) or 0)
        self.end_arc_radius = float(dims.get('end_arc_radius', 0) or 0)
        # End arcs are a circle of end_arc_radius centred on the goal-line midpoint, with the part
        # furthest from the goal flattened: it never reaches further than end_arc_max_depth.
        # 0 / missing = plain semicircle.
        self.end_arc_max_depth = float(dims.get('end_arc_max_depth', 0) or 0)

        self.img_w = float(calibration['image_size']['w'])
        self.img_h = float(calibration['image_size']['h'])
        self._cx = self.img_w / 2.0
        self._cy = self.img_h / 2.0
        self._scale = max(self.img_w, self.img_h) / 2.0

        self._build_constraints()
        self._fit()
        self._estimate_vertical_vp()

        if self.verbose:
            self._print_diagnostics()

    # ------------------------------------------------------------------ setup
    def _world_vertices(self):
        w, l = self.width, self.length
        return {
            'left_bottom':   (-l / 2, -w / 2),
            'left_top':      (-l / 2, +w / 2),
            'right_top':     (+l / 2, +w / 2),
            'right_bottom':  (+l / 2, -w / 2),
            'centre_top':    (0.0,    +w / 2),
            'centre_bottom': (0.0,    -w / 2),
        }

    def _build_constraints(self):
        order = ['left_bottom', 'left_top', 'right_top', 'right_bottom', 'centre_top', 'centre_bottom']
        world_vertices = self._world_vertices()
        pv = self.calibration['pitch_vertices']

        self._outline_names = order
        self._outline_pixel = np.array([(pv[n]['x'], pv[n]['y']) for n in order], dtype=np.float64)
        self._outline_world = np.array([world_vertices[n] for n in order], dtype=np.float64)

        circle_defs = [
            ('centre_circle', np.array([0.0, 0.0]), self.centre_radius),
            ('end_arc_left',  np.array([-self.length / 2, 0.0]), self.end_arc_radius),
            ('end_arc_right', np.array([+self.length / 2, 0.0]), self.end_arc_radius),
        ]
        circle_points = self.calibration.get('circle_points', {}) or {}
        pix, idx = [], []
        for i, (key, _center, radius) in enumerate(circle_defs):
            if radius <= 0:
                continue
            for p in circle_points.get(key, []) or []:
                pix.append((p['x'], p['y']))
                idx.append(i)

        self._circle_defs = circle_defs
        self._circle_pixel = np.array(pix, dtype=np.float64) if pix else np.zeros((0, 2))
        self._circle_idx = np.array(idx, dtype=int)
        self._circle_centers = (np.array([circle_defs[i][1] for i in idx], dtype=np.float64)
                                 if idx else np.zeros((0, 2)))
        self._circle_radii = (np.array([circle_defs[i][2] for i in idx], dtype=np.float64)
                               if idx else np.zeros((0,)))

    # ------------------------------------------------------------- lens model
    def _undistort(self, pix, k1, k2):
        dx = (pix[:, 0] - self._cx) / self._scale
        dy = (pix[:, 1] - self._cy) / self._scale
        r2 = dx * dx + dy * dy
        factor = 1.0 + k1 * r2 + k2 * r2 * r2
        return np.stack([dx * factor, dy * factor], axis=1)

    @staticmethod
    def _apply_homography(norm_pts, h):
        H = np.array([[h[0], h[1], h[2]],
                      [h[3], h[4], h[5]],
                      [h[6], h[7], 1.0]])
        ones = np.ones((norm_pts.shape[0], 1))
        homo = np.concatenate([norm_pts, ones], axis=1)
        proj = homo @ H.T
        return proj[:, :2] / proj[:, 2:3]

    @staticmethod
    def _dlt_homography(src, dst):
        A = []
        for (x, y), (X, Y) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
            A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        Hh = Vt[-1].reshape(3, 3)
        return Hh / Hh[2, 2]

    def _end_arc_distance(self, world, centre_x, inward_sign):
        """
        Distance (metres) from world points to the END-ARC SHAPE: the arc of a circle (radius
        end_arc_radius, centred on the goal-line midpoint) that is cut off by a straight line
        end_arc_max_depth out from the goal line. i.e. curved side pieces + a flat middle.
        """
        R = self.end_arc_radius
        D = self.end_arc_max_depth
        u = (world[:, 0] - centre_x) * inward_sign   # depth into the pitch, measured from the goal line
        v = np.abs(world[:, 1])                       # lateral offset (shape is symmetric)
        flattened = 0 < D < R
        phi0 = np.arccos(D / R) if flattened else 0.0
        psi = np.clip(np.arctan2(v, u), phi0, np.pi / 2)
        d = np.hypot(u - R * np.cos(psi), v - R * np.sin(psi))
        if flattened:
            lat = R * np.sin(phi0)
            d = np.minimum(d, np.hypot(u - D, v - np.clip(v, 0, lat)))
        return d

    def _circle_constraint_residuals(self, world):
        out = np.zeros(len(world))
        m0 = self._circle_idx == 0
        out[m0] = np.linalg.norm(world[m0] - self._circle_centers[m0], axis=1) - self._circle_radii[m0]
        for gi, cx, sign in ((1, -self.length / 2, 1.0), (2, +self.length / 2, -1.0)):
            m = self._circle_idx == gi
            if m.any():
                out[m] = self._end_arc_distance(world[m], cx, sign)
        return out

    def _residuals(self, params):
        k1, k2 = params[:2]
        h = params[2:]
        out_pred = self._apply_homography(self._undistort(self._outline_pixel, k1, k2), h)
        r_outline = (out_pred - self._outline_world).ravel()

        if len(self._circle_pixel):
            circ_pred = self._apply_homography(self._undistort(self._circle_pixel, k1, k2), h)
            r_circle = self._circle_constraint_residuals(circ_pred)
        else:
            r_circle = np.zeros((0,))

        return np.concatenate([r_outline, r_circle])

    def _fit(self):
        norm0 = self._undistort(self._outline_pixel, 0.0, 0.0)
        H0 = self._dlt_homography(norm0, self._outline_world)
        h0 = [H0[0, 0], H0[0, 1], H0[0, 2], H0[1, 0], H0[1, 1], H0[1, 2], H0[2, 0], H0[2, 1]]
        x0 = np.array([0.0, 0.0] + h0)

        lower = [-self.K_BOUND, -self.K_BOUND] + [-np.inf] * 8
        upper = [self.K_BOUND, self.K_BOUND] + [np.inf] * 8

        result = least_squares(self._residuals, x0, bounds=(lower, upper),
                                method='trf', xtol=1e-14, ftol=1e-14, max_nfev=30000)

        self._fit_success = bool(result.success)
        self.k1, self.k2 = result.x[:2]
        self._h = result.x[2:]
        self._H = np.array([[self._h[0], self._h[1], self._h[2]],
                             [self._h[3], self._h[4], self._h[5]],
                             [self._h[6], self._h[7], 1.0]])

        residuals = self._residuals(result.x)
        n_out = len(self._outline_pixel)
        self._outline_residuals = residuals[:n_out * 2].reshape(n_out, 2)
        self._circle_residuals = residuals[n_out * 2:]
        self.rms_outline_m = float(np.sqrt(np.mean(self._outline_residuals ** 2))) if n_out else 0.0
        self.rms_circle_m = (float(np.sqrt(np.mean(self._circle_residuals ** 2)))
                              if len(self._circle_residuals) else 0.0)

    # ------------------------------------------------------------- public API
    def transform_point(self, x, y):
        """Transform a single pixel (x, y) to world metres (x_world, y_world)."""
        pix = np.array([[x, y]], dtype=np.float64)
        norm = self._undistort(pix, self.k1, self.k2)
        world = self._apply_homography(norm, self._h)
        return float(world[0, 0]), float(world[0, 1])

    def transform_points(self, points):
        """Transform an (N, 2) array of pixel coordinates to (N, 2) world metres."""
        pix = np.asarray(points, dtype=np.float64)
        norm = self._undistort(pix, self.k1, self.k2)
        return self._apply_homography(norm, self._h)

    # ----------------------------------------------- vertical vanishing point + foot position
    @property
    def has_vertical_vp(self):
        return self.vertical_vp is not None

    def _estimate_vertical_vp(self):
        """
        Estimate where real-world VERTICAL lines converge in the image (the point below the camera,
        since it looks down at the pitch). Uses the goal posts clicked in the Pitch Calibrator:
        goal_posts_*_vertices = [left post base, left post top, right post top, right post base],
        i.e. two vertical lines per goal. Done in the lens-corrected (undistorted) plane, where
        straight lines are straight and verticals genuinely converge. Needs >= 2 lines.
        """
        self.vertical_vp = None
        self.vertical_line_residuals_deg = []
        lines = []
        for key in ('goal_posts_left_vertices', 'goal_posts_right_vertices'):
            pts = self.calibration.get(key) or []
            if len(pts) == 4:
                for a, b in ((0, 1), (3, 2)):          # (base, top) of each post
                    if pts[a] and pts[b]:
                        lines.append((pts[a], pts[b]))
        for vl in self.calibration.get('vertical_lines', []) or []:
            if vl.get('base') and vl.get('top'):
                lines.append((vl['base'], vl['top']))
        if len(lines) < 2:
            return

        base = self._undistort(np.array([[l[0]['x'], l[0]['y']] for l in lines], dtype=np.float64), self.k1, self.k2)
        top = self._undistort(np.array([[l[1]['x'], l[1]['y']] for l in lines], dtype=np.float64), self.k1, self.k2)

        A = np.zeros((2, 2))
        rhs = np.zeros(2)
        for p0, p1 in zip(base, top):
            d = p1 - p0
            length = np.linalg.norm(d)
            if length < 1e-9:
                continue
            d = d / length
            P = np.eye(2) - np.outer(d, d)             # projects onto the normal of this line
            A += length * P                            # longer lines count for more
            rhs += length * (P @ p0)
        try:
            vp = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return

        # sanity: every post must point TOWARD the vanishing point going from its top to its base
        residuals = []
        for p0, p1 in zip(base, top):
            toward = vp - p0
            along = p0 - p1
            if np.dot(toward, along) <= 0:
                if self.verbose:
                    print("  WARNING: goal-post lines do not converge below the camera - ignoring them "
                          "(check the base/top clicks); using the legacy foot-point rule.")
                return
            cosang = np.dot(toward, along) / (np.linalg.norm(toward) * np.linalg.norm(along))
            residuals.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        self.vertical_vp = vp
        self.vertical_line_residuals_deg = residuals
        self.n_vertical_lines = len(lines)

    def _foot_undistorted(self, xmin, ymin, xmax, ymax):
        """
        Foot position (undistorted plane) of a player's bounding box. The body's upright axis passes
        through the box centre and points at the vertical vanishing point; the feet are where that axis
        reaches the bottom of the box. This slides the foot smoothly sideways with distance from the
        vanishing point (replacing the old left/right-of-halfway-line corner rule, which jumped by a
        whole box width as a player crossed the line).
        """
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        centre, bottom = self._undistort(np.array([[cx, cy], [cx, ymax]], dtype=np.float64), self.k1, self.k2)
        if self.vertical_vp is None:
            return bottom
        dy = self.vertical_vp[1] - centre[1]
        if dy <= 1e-9:
            return bottom
        t = float(np.clip((bottom[1] - centre[1]) / dy, 0.0, 1.0))
        return centre + t * (self.vertical_vp - centre)

    def transform_foot(self, xmin, ymin, xmax, ymax):
        """World metres (x, y) of a player's feet from their bounding box (see _foot_undistorted)."""
        f = self._foot_undistorted(xmin, ymin, xmax, ymax)
        world = self._apply_homography(f.reshape(1, 2), self._h)
        return float(world[0, 0]), float(world[0, 1])

    def _distort(self, norm):
        """Inverse of the radial lens correction: undistorted normalised point -> normalised pixel offset."""
        q = np.atleast_2d(norm).astype(np.float64)
        ru = np.hypot(q[:, 0], q[:, 1])
        lo = np.zeros_like(ru)
        hi = np.maximum(ru, 1e-6)
        for _ in range(60):                                       # grow the bracket until it contains the answer
            f = hi * (1 + self.k1 * hi ** 2 + self.k2 * hi ** 4)
            short = f < ru
            if not short.any():
                break
            hi = np.where(short, hi * 2, hi)
        for _ in range(60):
            mid = (lo + hi) / 2
            f = mid * (1 + self.k1 * mid ** 2 + self.k2 * mid ** 4)
            lo = np.where(f < ru, mid, lo)
            hi = np.where(f < ru, hi, mid)
        rd = (lo + hi) / 2
        scale = np.divide(rd, ru, out=np.ones_like(ru), where=ru > 1e-12)
        return q * scale[:, None]

    def foot_pixel(self, xmin, ymin, xmax, ymax):
        """Pixel (x, y) in the original frame where transform_foot() puts the feet (for drawing)."""
        if self.vertical_vp is None:
            return None
        f = self._foot_undistorted(xmin, ymin, xmax, ymax)
        d = self._distort(f.reshape(1, 2))[0]
        return float(d[0] * self._scale + self._cx), float(d[1] * self._scale + self._cy)

    def overlay_geometry(self):
        """
        Geometry for drawing an ACCURACY-TEST overlay on the top-view inset:
          'shapes' : the ideal centre circle and goal-end arcs as world-metre polylines
                     (what the pitch SHOULD look like, from pitch_dimensions_m)
          'points' : where each clicked circle/arc calibration pixel actually lands after
                     this fitted transform. A perfect fit puts every point exactly on a shape.
        """
        shapes = []
        if self.centre_radius > 0:
            th = np.linspace(0, 2 * np.pi, 120)
            shapes.append({'name': 'centre_circle',
                           'pts': [(float(self.centre_radius * np.cos(t)), float(self.centre_radius * np.sin(t))) for t in th]})
        if self.end_arc_radius > 0:
            R, D = self.end_arc_radius, self.end_arc_max_depth
            flattened = 0 < D < R
            phi0 = float(np.arccos(D / R)) if flattened else 0.0
            for name, cx, sign in (('end_arc_left', -self.length / 2, 1.0), ('end_arc_right', self.length / 2, -1.0)):
                pts = []
                for phi in np.linspace(-np.pi / 2, -phi0, 40):          # lower curved piece, goal line -> flat
                    pts.append((cx + sign * R * np.cos(phi), R * np.sin(phi)))
                if flattened:                                           # flat middle
                    pts.append((cx + sign * D, -R * np.sin(phi0)))
                    pts.append((cx + sign * D, +R * np.sin(phi0)))
                for phi in np.linspace(phi0, np.pi / 2, 40):            # upper curved piece, flat -> goal line
                    pts.append((cx + sign * R * np.cos(phi), R * np.sin(phi)))
                shapes.append({'name': name, 'pts': [(float(a), float(b)) for a, b in pts]})
        points = []
        if len(self._circle_pixel):
            world = self.transform_points(self._circle_pixel)
            for i, (wx, wy) in zip(self._circle_idx, world):
                points.append({'group': self._circle_defs[i][0], 'x': float(wx), 'y': float(wy)})
        return {'shapes': shapes, 'points': points}

    def _print_diagnostics(self):
        print("-------------------------------------------------")
        print("FITTED PITCH TRANSFORMER: per-video calibration fit")
        print(f"  source_frame   : {self.calibration.get('source_frame')}")
        print(f"  pitch size (m) : width={self.width} length={self.length} "
              f"centre_radius={self.centre_radius} end_arc_radius={self.end_arc_radius} end_arc_max_depth={self.end_arc_max_depth}")
        print(f"  fit converged  : {self._fit_success}  k1={self.k1:.5f}  k2={self.k2:.5f}")
        print(f"  RMS outline error : {self.rms_outline_m:.3f} m "
              f"(hard constraints, {len(self._outline_pixel)} points)")
        if len(self._circle_residuals):
            print(f"  RMS circle/arc error : {self.rms_circle_m:.3f} m "
                  f"(soft constraints, {len(self._circle_residuals)} points)")
        worst_idx = int(np.argmax(np.linalg.norm(self._outline_residuals, axis=1))) if len(self._outline_pixel) else None
        if worst_idx is not None:
            worst_name = self._outline_names[worst_idx]
            worst_err = float(np.linalg.norm(self._outline_residuals[worst_idx]))
            print(f"  worst outline point : {worst_name} ({worst_err:.3f} m)")
        if self.vertical_vp is not None:
            res = self.vertical_line_residuals_deg
            print(f"  vertical vanishing point : found from {self.n_vertical_lines} post lines "
                  f"(worst line off by {max(res):.1f} deg) -> smooth foot-point correction ON")
        else:
            print("  vertical vanishing point : none (no goal posts clicked) -> legacy corner foot-point rule")
        if self.width in (30.0,) and self.length in (45.0,):
            print("  NOTE: pitch width/length look like CWFF-range placeholders, not a measured "
                  "pitch -- confirm real dimensions to improve fit accuracy.")
        print("-------------------------------------------------")
