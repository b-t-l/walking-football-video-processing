import sqlite3
import math
import time

import numpy as np
from tqdm import tqdm


class SpeedDistanceUpdater:
    """
    Calculates each player's speed (km/h) and cumulative distance (m) from the pitch positions
    (x/y_transformed_metres) already stored in the database.

    WHY IT IS NOT A SIMPLE FRAME-TO-FRAME DIFFERENCE (rewritten 2026-10):
      * Foot-point positions carry ~5-10 cm of detection jitter per frame. Dividing a 5 cm wobble by
        1/24 s reads as ~4 km/h of "speed" even for a player standing still, and summing the wobble
        roughly DOUBLES the distance covered. Speed is therefore the slope of a short, weighted
        straight-line fit through the player's recent positions (about +/-0.6 s), and distance is
        measured along that smoothed path.
      * The tracker sometimes swaps IDs or glitches, which shows up as a position that "teleports".
        If two consecutive detections of one tracker_id are further apart than a player could
        plausibly move (faster than MAX_PLAUSIBLE_SPEED_KMH and at least MIN_JUMP_METRES), the track is split there: no speed or distance is credited across
        the jump and the smoothing never mixes the two sides.
      * Detections outside the pitch get speed 0 and add no distance.
    """

    # --- tuning knobs ------------------------------------------------------------------------
    MAX_PLAUSIBLE_SPEED_KMH = 25.0   # faster than this between two detections of one track = glitch / ID swap...
    MIN_JUMP_METRES = 0.8            # ...but only if the position also moved at least this far (ordinary one-frame
                                     #    jitter can be 0.3 m = "25 km/h" over 1/24 s; a real glitch moves >0.8 m)
    SMOOTH_SIGMA_S = 0.25            # Gaussian weight width of the local straight-line fit (seconds)
    SMOOTH_HALF_WINDOW_S = 0.6       # only detections within +/- this many seconds are used (seconds)
    MIN_POINTS_IN_WINDOW = 4         # fewer detections than this in the window -> speed cannot be measured (0.0)

    ########## INITIALIZE THE SPEED/DISTANCE CALCULATOR:
    def __init__(self, DATABASE_FILE, GAME_RECORD, DURATION, batch_size=10000):

        print("-------------------------------------------------")
        print("CALCULATE SPEED/DISTANCE: from db update each object calculating speed/distance")

        """
        Initialize with database file, game record, batch size, and optional duration filter.
        duration: Tuple (start_frame, end_frame) to limit processing.
        """
        self.batch_size = batch_size
        self.GAME_RECORD = GAME_RECORD
        self.DATABASE_FILE = DATABASE_FILE
        self.conn = sqlite3.connect(self.DATABASE_FILE)
        self.cursor = self.conn.cursor()

        # Fetch game details
        self.cursor.execute('''
            SELECT _id, game_id, video, video_fps, detection_fps
            FROM game
            WHERE game_id = ?
        ''', (self.GAME_RECORD['game_id'],))
        self.DB_GAME_DETAILS = self.cursor.fetchone()
        self.DETECTION_FPS = float(self.DB_GAME_DETAILS[4])
        self.duration = (0, DURATION * self.DETECTION_FPS)  # Store the duration filter

    ########## SMOOTHING HELPERS:
    def _segment_starts(self, t, x, y):
        """Indices where a new, unbroken stretch of one track starts (a first one at 0, plus after every implausible jump)."""
        starts = [0]
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            d = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
            if dt > 0 and d > max(self.MIN_JUMP_METRES, (self.MAX_PLAUSIBLE_SPEED_KMH / 3.6) * dt):
                starts.append(i)
        return starts

    def _smooth_segment(self, t, x, y):
        """
        Gaussian-weighted local straight-line fit around every detection of one unbroken segment.
        Returns smoothed x, y (metres) and speed (km/h); NaN where fewer than MIN_POINTS_IN_WINDOW detections are in range.
        """
        n = len(t)
        sx = np.full(n, np.nan)
        sy = np.full(n, np.nan)
        speed = np.full(n, np.nan)
        lo = np.searchsorted(t, t - self.SMOOTH_HALF_WINDOW_S, side='left')
        hi = np.searchsorted(t, t + self.SMOOTH_HALF_WINDOW_S, side='right')
        for i in range(n):
            a, b = lo[i], hi[i]
            if b - a < self.MIN_POINTS_IN_WINDOW:
                continue
            dt = t[a:b] - t[i]
            w = np.exp(-0.5 * (dt / self.SMOOTH_SIGMA_S) ** 2)
            sw = w.sum()
            mean_dt = (w * dt).sum() / sw
            var_dt = (w * (dt - mean_dt) ** 2).sum()
            if var_dt < 1e-12:
                continue
            mx = (w * x[a:b]).sum() / sw
            my = (w * y[a:b]).sum() / sw
            vx = (w * (dt - mean_dt) * (x[a:b] - mx)).sum() / var_dt
            vy = (w * (dt - mean_dt) * (y[a:b] - my)).sum() / var_dt
            sx[i] = mx - vx * mean_dt   # fitted position at dt = 0
            sy[i] = my - vy * mean_dt
            speed[i] = math.hypot(vx, vy) * 3.6
        return sx, sy, speed

    ########## RUN THE SPEED/DISTANCE CALCULATOR:
    def run(self):

        start_time = time.time()  # Start the timer for overall process time

        """Fetch, calculate, and update database records in batches."""
        self.cursor.execute('''
            SELECT DISTINCT tracker_id FROM detected_objects
            WHERE class_name = "player"
        ''')
        tracker_ids = self.cursor.fetchall()

        total_splits = 0
        updates_batch = []

        with tqdm(total=len(tracker_ids), desc="🔄 Processing each player via Tracker ID", unit="tracker") as tracker_pbar:
            for tracker_id in tracker_ids:
                tracker_id = tracker_id[0]

                self.cursor.execute('''
                        SELECT frame_id, x_transformed_metres, y_transformed_metres, is_inside_pitch
                        FROM detected_objects
                        WHERE tracker_id = ? AND class_name = "player"
                        ORDER BY frame_id
                    ''', (tracker_id,))
                records = self.cursor.fetchall()

                frames = [r[0] for r in records]
                speeds = [0.0] * len(records)
                dist_inc = [0.0] * len(records)

                # only detections with a usable position can be measured; the rest keep speed 0 / distance 0
                usable = [i for i, r in enumerate(records) if r[1] is not None and r[2] is not None]
                if usable:
                    t = np.array([frames[i] for i in usable], dtype=float) / self.DETECTION_FPS
                    x = np.array([records[i][1] for i in usable], dtype=float)
                    y = np.array([records[i][2] for i in usable], dtype=float)

                    starts = self._segment_starts(t, x, y)
                    total_splits += len(starts) - 1
                    bounds = starts + [len(t)]

                    for a, b in zip(bounds[:-1], bounds[1:]):
                        sx, sy, sp = self._smooth_segment(t[a:b], x[a:b], y[a:b])
                        prev = None   # previous measurable smoothed point inside this segment
                        for k in range(b - a):
                            i = usable[a + k]
                            if np.isnan(sp[k]):
                                continue
                            inside = records[i][3] != 0
                            speeds[i] = float(sp[k]) if inside else 0.0
                            if prev is not None and inside:
                                dist_inc[i] = math.hypot(sx[k] - sx[prev], sy[k] - sy[prev])
                            prev = k

                accumulated_distance = 0.0
                for i in range(len(records)):
                    accumulated_distance += dist_inc[i]
                    updates_batch.append((speeds[i], accumulated_distance, frames[i], tracker_id))

                if len(updates_batch) >= self.batch_size:
                    self._execute_batch_update(updates_batch)
                    updates_batch = []

                tracker_pbar.update(1)

        if updates_batch:
            self._execute_batch_update(updates_batch)

        print(f"   tracks split at implausible jumps (> {self.MAX_PLAUSIBLE_SPEED_KMH:.0f} km/h between detections): {total_splits}")

        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        print("🔒 Database connection closed.")

        # TIME FOR FULL PROCESS: Calculate elapsed time for entire process:
        end_time = time.time()  # Stop the timer
        total_time = end_time - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")

    ########## EXECUTE BATCH UPDATES:
    def _execute_batch_update(self, updates_batch):
        """Execute batched updates."""
        self.cursor.executemany('''
            UPDATE detected_objects
            SET speed_km_per_hour = ?, total_distance_metres = ?
            WHERE frame_id = ? AND tracker_id = ? AND class_name = "player"
        ''', updates_batch)
        self.conn.commit()
