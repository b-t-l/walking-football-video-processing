import json
import sqlite3
import time
from collections import defaultdict

import numpy as np


class KeeperZone:
    """
    Finds the goalkeepers by WHERE they stand, not by what they wear.

    In walking football only the goalkeeper may be inside the goal-end arc (the "semi-circle"), so a
    player who spends the time in an arc is that end's goalkeeper. Kit colours change every game and
    the team model gets them wrong, but the arc never moves. Runs AFTER the view transformations
    (needs the pitch positions) and the track stitcher (needs one player_id per real player), and
    BEFORE speed/distance and possession.

    How it decides (one decision per player_id, never per frame, so a keeper who steps out for a moment
    is still the keeper and an outfield player who clips the line for a second is not):
      * a detection is "in the zone" when its pitch position is inside the end arc of either goal,
        made ZONE_MARGIN_M bigger (the view is most distorted close to the goal, so positions there
        are the least exact) and extended ZONE_BEHIND_M behind the goal line (keepers stand there);
      * a player_id is a goalkeeper when at least ZONE_SHARE of its positioned detections are in the
        zone AND there are at least ZONE_MIN_SECONDS worth of them.
    Goalkeepers get team = 'Goalkeeper' (the label every report and the annotated video already treat as
    "not one of the two teams"). team_raw (what the team model said) and class_name are left alone.
    The detector's own class (player / goalkeeper) is ignored: only the position counts.

    Needs the pitch size in the game's calibration (pitch_calibrator_json: length, end_arc_radius,
    end_arc_max_depth). Games without it are skipped with a message.

    Re-running: the track stitcher rewrites team from team_raw every time it runs, so always run this
    straight after it (that is how main.py does it).
    """

    # --- tuning knobs ----------------------------------------------------------------------------
    ZONE_MARGIN_M = 1.0        # the arc is made this much bigger on every side
    ZONE_BEHIND_M = 1.5        # ... and extended this far behind the goal line
    ZONE_SHARE = 0.60          # share of a player's positioned detections that must be in the zone
    ZONE_MIN_SECONDS = 1.5     # and at least this much time in the zone (stops short flickers)
    PROVISIONAL_BELOW = 0.6    # 'goalkeeper' boxes under this confidence are the extra low-confidence ones (see create_database.KEEPER_MIN_CONF)
    KEEPER_COLOUR = "(232, 224, 132)"   # BGR text, same as the team assigner's fixed Goalkeeper colour

    def __init__(self, DATABASE_FILE, GAME_RECORD):
        print("-------------------------------------------------")
        print("KEEPER ZONE: whoever lives in a goal-end arc is that end's goalkeeper")
        self.DATABASE_FILE = DATABASE_FILE
        self.GAME_RECORD = GAME_RECORD
        self.dims = self._pitch_dims(GAME_RECORD)

    @staticmethod
    def _pitch_dims(game_record):
        raw = game_record.get('pitch_calibrator_json') if hasattr(game_record, 'get') else None
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            d = json.loads(raw)['pitch_dimensions_m']
            dims = {'length': float(d['length']), 'width': float(d['width']),
                    'R': float(d.get('end_arc_radius', 0) or 0),
                    'D': float(d.get('end_arc_max_depth', 0) or 0)}
        except Exception:                                                       # noqa
            return None
        if dims['R'] <= 0:
            return None
        if not (0 < dims['D'] < dims['R']):
            dims['D'] = dims['R']                                               # plain semicircle
        return dims

    def in_zone(self, x, y):
        """Boolean array / which end: positions in pitch metres (centre of the pitch = 0,0) -> (in_zone, end) where end is -1 (left), +1 (right)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        L, R, D = self.dims['length'], self.dims['R'], self.dims['D']
        m, behind = self.ZONE_MARGIN_M, self.ZONE_BEHIND_M
        inside = np.zeros(x.shape, dtype=bool)
        end = np.zeros(x.shape, dtype=int)
        for sign, cx in ((1.0, -L / 2.0), (-1.0, L / 2.0)):
            u = (x - cx) * sign                    # depth into the pitch measured from that goal line
            v = np.abs(y)
            ok = (u >= -behind) & (u <= D + m) & (np.hypot(np.maximum(u, 0.0), v) <= R + m)
            inside |= ok
            end[ok] = int(-sign)
        return inside, end

    def run(self):
        start_time = time.time()
        if self.dims is None:
            print("   skipped: this game has no pitch calibration with an end-arc size (pitch_calibrator_json)")
            return
        conn = sqlite3.connect(self.DATABASE_FILE)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(detected_objects)")
        cols = {r[1] for r in cur.fetchall()}
        if "player_id" not in cols:
            print("   skipped: run the track stitcher first (no player_id column yet)")
            conn.close()
            return
        cur.execute("SELECT video_fps, detection_fps FROM game LIMIT 1")
        _, det_fps = cur.fetchone()
        min_n = max(6, int(round(self.ZONE_MIN_SECONDS * float(det_fps))))

        cur.execute('''
            SELECT _id, class_name, player_id, x_transformed_metres, y_transformed_metres
            FROM detected_objects
            WHERE class_name IN ("player", "goalkeeper") AND player_id IS NOT NULL
              AND x_transformed_metres IS NOT NULL AND y_transformed_metres IS NOT NULL
        ''')
        rows = cur.fetchall()
        if not rows:
            print("   skipped: no positioned players")
            conn.close()
            return
        ids = np.array([r[0] for r in rows])
        x = np.array([r[3] for r in rows], dtype=float)
        y = np.array([r[4] for r in rows], dtype=float)
        inside, end = self.in_zone(x, y)

        # one decision per (class, player_id): a goalkeeper-class id and a stitched player id can share a number
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[(r[1], r[2])].append(i)

        keeper_rows, found = [], []
        for (cls, pid), idx in groups.items():
            idx = np.array(idx)
            n_in = int(inside[idx].sum())
            share = n_in / len(idx)
            if n_in >= min_n and share >= self.ZONE_SHARE:
                side = np.sign(np.sum(end[idx][inside[idx]]))
                keeper_rows.extend(int(v) for v in ids[idx])
                found.append((cls, pid, len(idx), share, "left" if side < 0 else "right" if side > 0 else "both"))

        cur.executemany('UPDATE detected_objects SET team = "Goalkeeper", team_color = ? WHERE _id = ?',
                        [(self.KEEPER_COLOUR, rid) for rid in keeper_rows])
        # Detections stores 'goalkeeper' boxes from KEEPER_MIN_CONF up while everything else needs PLAYER_CONF. Those extra,
        # lower-confidence goalkeeper boxes are only wanted when they are a real goalkeeper (in an arc): drop the rest.
        keep = set(keeper_rows)
        cur.execute('SELECT _id FROM detected_objects WHERE class_name = "goalkeeper" AND confidence < ?', (self.PROVISIONAL_BELOW,))
        drop = [(r[0],) for r in cur.fetchall() if r[0] not in keep]
        cur.executemany('DELETE FROM detected_objects WHERE _id = ?', drop)
        self.dropped = len(drop)
        conn.commit()
        conn.close()

        self.found = found
        by_side = {s: sum(f[2] for f in found if f[4] == s) for s in ("left", "right", "both")}
        print(f"   pitch {self.dims['length']:g} x {self.dims['width']:g} m, arc radius {self.dims['R']:g} m "
              f"(+{self.ZONE_MARGIN_M:g} m margin); needs >= {self.ZONE_SHARE:.0%} of a player's detections in the zone "
              f"and >= {min_n} of them")
        print(f"   goalkeeper players found: {len(found)}  ({len(keeper_rows)} detections:  "
              f"left end {by_side['left']},  right end {by_side['right']})")
        if self.dropped:
            print(f"   {self.dropped} low-confidence goalkeeper boxes outside the arcs removed (not a goalkeeper)")
        for cls, pid, n, share, side in sorted(found, key=lambda f: -f[2])[:8]:
            print(f"     player_id {pid} (detector class {cls}): {n} detections, {share:.0%} in the {side} arc")
        total_time = time.time() - start_time
        print(f"✅ Processing time: {int(total_time // 60)}:{int(total_time % 60)} sec")
