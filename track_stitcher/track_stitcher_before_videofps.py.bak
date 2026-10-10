import math
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackStitcher:
    """
    Gives every player ONE stable id (player_id) for as much of the game as can be justified, by
    repairing what the frame-to-frame tracker (ByteTrack) gets wrong. Works offline on the pitch
    positions already in the database (x/y_transformed_metres), so it must run AFTER the view
    transformations and BEFORE speed/distance and possession.

    Why it is needed: ByteTrack only looks at box overlap between frames. When a player goes behind
    another player the track dies and a new tracker_id is issued; when two players cross it can swap
    their ids. Over a game that gives hundreds of tracker_ids for ~11 players.

    What it does (per game, class "player" only):
      1. FRAGMENTS: cut every tracker_id's path (a) wherever the box jumps further than a person can
         move, and (b) wherever the team label switches for good (e.g. 40 frames "Aphrodite" then 40
         frames "Polis" on one tracker_id is two different players that the tracker swapped while they
         were touching - a swap that position alone cannot reveal).
      2. LINK: decide which fragment continues which, using a global (Hungarian) assignment. A
         continuation must start after the earlier fragment ends, be reachable at walking/jogging pace,
         and be compatible in team. Cost = how far the later fragment starts from where the earlier one
         was heading + a small penalty per second of gap + a team-mismatch penalty.

    DISTANCES ARE MEASURED IN THE IMAGE, in "person-heights": the shift of the box centre divided by the
    box height, times 1.7 m. Positions in metres are NOT used for stitching because the far side of the
    pitch is so oblique that 1 pixel of box-bottom jitter is 0.1-0.25 m there, which fakes "jumps" for a
    player who has not moved. A box-centre shift scaled by box size is equally reliable everywhere.
      3. CHAINS: following the links gives chains of fragments = one real player = one player_id.
      4. TEAM: each chain takes the team that most of its detections were labelled with, instead of
         the per-frame label flicker.

    DATABASE: adds player_id (stitched id), team_raw and team_color_raw (the original per-detection
    team labels, copied ONCE so this step can be re-run safely). The team / team_color columns are then
    overwritten with the chain's majority team. tracker_id is never changed.
    """

    # --- tuning knobs ----------------------------------------------------------------------------
    PERSON_HEIGHT_M = 1.7         # converts box-height units into approximate metres
    SPLIT_SPEED_KMH = 25.0        # a jump faster than this ...
    SPLIT_MIN_JUMP_M = 0.45       # ... and at least this far (in person-height metres) = ID swap / glitch -> cut the track
    MIN_TEAM_SWITCH_RUN = 15      # a team label must hold for this many detections on each side to count as a swap...
    MIN_TEAM_SWITCH_GAIN = 15     # ...and splitting must explain this many more labels than leaving the track whole
    IGNORED_AS_TEAM = ("Goalkeeper", "Referee", "None", "")   # never treated as one of the two playing teams
    LINK_MAX_GAP_S = 6.0          # longest disappearance bridged when joining two fragments
    LINK_MAX_OVERLAP_FRAMES = 3   # ByteTrack sometimes starts a new id while the old one still reports the same person
    LINK_MAX_OVERLAP_DIST_M = 0.6 # ... accepted only if the two boxes are on (nearly) the same spot
    LINK_BASE_M = 0.6             # slack allowed on top of the distance a player could cover in the gap
    LINK_MAX_SPEED_MS = 4.0       # fastest believable speed while out of sight (4 m/s = 14.4 km/h)
    GAP_COST_PER_S = 0.3          # prefers short gaps (cost units ~ metres)
    TEAM_MISMATCH_COST = 3.0      # prefers same-team joins
    MIN_TEAM_COMPAT = 0.10        # joins between clearly different teams are not allowed at all
    MAX_LINK_COST = 2.5           # joins costing more than this are refused
    VELOCITY_POINTS = 6           # detections used to estimate a fragment's end / start velocity
    MAX_PREDICT_SPEED_MS = 2.5    # cap on the extrapolation speed
    MAX_PREDICT_S = 1.0           # extrapolate the path at most this long into a gap
    IGNORED_TEAM_LABELS = ("None", "", None)

    ########## INITIALIZE:
    def __init__(self, DATABASE_FILE, GAME_RECORD):
        print("-------------------------------------------------")
        print("TRACK STITCHER: merge broken tracker ids into one player_id per real player")
        self.DATABASE_FILE = DATABASE_FILE
        self.GAME_RECORD = GAME_RECORD
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.cursor = self.conn.cursor()
        self.cursor.execute('SELECT detection_fps FROM game WHERE game_id = ?', (GAME_RECORD['game_id'],))
        self.FPS = float(self.cursor.fetchone()[0])

    ########## DATABASE PREPARATION:
    def _prepare_columns(self):
        self.cursor.execute("PRAGMA table_info(detected_objects)")
        existing = {row[1] for row in self.cursor.fetchall()}
        for column, sql_type in (("player_id", "INTEGER"), ("team_raw", "TEXT"), ("team_color_raw", "TEXT")):
            if column not in existing:
                self.cursor.execute(f"ALTER TABLE detected_objects ADD COLUMN {column} {sql_type}")
        # keep the ORIGINAL per-detection labels once, so reruns always start from them
        self.cursor.execute('''
            UPDATE detected_objects SET team_raw = team, team_color_raw = team_color
            WHERE team_raw IS NULL
        ''')
        # everything that is not a player keeps its tracker_id as its id
        self.cursor.execute('''
            UPDATE detected_objects SET player_id = CAST(tracker_id AS INTEGER)
            WHERE class_name != "player"
        ''')
        self.conn.commit()

    def _shift_m(self, c0, h0, c1, h1):
        """Distance between two box centres in person-height metres."""
        scale = self.PERSON_HEIGHT_M / max((h0 + h1) / 2.0, 10.0)
        return math.hypot(c1[0] - c0[0], c1[1] - c0[1]) * scale

    @staticmethod
    def _box(r):
        return ((r[4] + r[6]) / 2.0, (r[5] + r[7]) / 2.0), r[7] - r[5]

    def _team_cut_indexes(self, labels):
        """
        labels: team label of each detection of one tracker_id (None where not one of the two playing teams).
        Returns indexes where a lasting switch between the two teams happens (cut BEFORE that index).
        """
        idx = [i for i, l in enumerate(labels) if l is not None]
        if len(idx) < 2 * self.MIN_TEAM_SWITCH_RUN:
            return []
        seq = [labels[i] for i in idx]
        names = sorted(set(seq))
        if len(names) < 2:
            return []
        cuts = []

        def split(lo, hi):
            # lo..hi indexes into seq; find the best single split of seq[lo:hi]
            n = hi - lo
            if n < 2 * self.MIN_TEAM_SWITCH_RUN:
                return
            cum = {name: np.concatenate([[0], np.cumsum([1 if v == name else 0 for v in seq[lo:hi]])]) for name in names}
            best_single = max(int(cum[name][n]) for name in names)
            best = (0, None)
            for k in range(self.MIN_TEAM_SWITCH_RUN, n - self.MIN_TEAM_SWITCH_RUN + 1):
                for left in names:
                    for right in names:
                        if left == right:
                            continue
                        score = int(cum[left][k]) + int(cum[right][n] - cum[right][k])
                        if score - best_single > best[0]:
                            best = (score - best_single, k)
            if best[1] is not None and best[0] >= self.MIN_TEAM_SWITCH_GAIN:
                k = lo + best[1]
                cuts.append(idx[k])
                split(lo, k)
                split(k, hi)

        split(0, len(seq))
        return sorted(cuts)

    ########## STEP 1: FRAGMENTS
    def _build_fragments(self, rows):
        """rows: (rowid, tracker_id, frame_id, team_raw, xmin, ymin, xmax, ymax) ordered by tracker_id, frame_id -> list of fragment dicts"""
        by_tracker = defaultdict(list)
        for r in rows:
            by_tracker[r[1]].append(r)

        fragments = []
        jump_ms = self.SPLIT_SPEED_KMH / 3.6
        label_counts = Counter(r[3] for r in rows if r[3] not in self.IGNORED_AS_TEAM)
        playing_teams = {name for name, _ in label_counts.most_common(2)}
        self.team_switch_cuts = 0
        for tracker_id, track in by_tracker.items():
            start = 0
            team_cuts = set(self._team_cut_indexes([r[3] if r[3] in playing_teams else None for r in track]))
            self.team_switch_cuts += len(team_cuts)
            for i in range(1, len(track) + 1):
                cut = (i == len(track)) or (i in team_cuts)
                if not cut:
                    dt = (track[i][2] - track[i - 1][2]) / self.FPS
                    c0, h0 = self._box(track[i - 1])
                    c1, h1 = self._box(track[i])
                    d = self._shift_m(c0, h0, c1, h1)
                    cut = dt > 0 and d > max(self.SPLIT_MIN_JUMP_M, jump_ms * dt)
                if cut:
                    fragments.append(self._make_fragment(track[start:i], tracker_id))
                    start = i
        return fragments

    def _end_velocity(self, t, cx, cy, h, at_end):
        """Velocity (box-height units per second -> returned in pixels/s) of the first / last few boxes by straight-line fit."""
        n = min(self.VELOCITY_POINTS, len(t))
        if n < 3:
            return 0.0, 0.0
        sl = slice(len(t) - n, len(t)) if at_end else slice(0, n)
        tt = t[sl] - t[sl].mean()
        denom = float((tt ** 2).sum())
        if denom < 1e-9:
            return 0.0, 0.0
        vx = float((tt * (cx[sl] - cx[sl].mean())).sum() / denom)
        vy = float((tt * (cy[sl] - cy[sl].mean())).sum() / denom)
        speed_m = math.hypot(vx, vy) * self.PERSON_HEIGHT_M / max(float(h[sl].mean()), 10.0)
        if speed_m > self.MAX_PREDICT_SPEED_MS:
            k = self.MAX_PREDICT_SPEED_MS / speed_m
            vx, vy = vx * k, vy * k
        return vx, vy

    def _make_fragment(self, track, tracker_id):
        frames = np.array([r[2] for r in track], dtype=float)
        t = frames / self.FPS
        boxes = np.array([[r[4], r[5], r[6], r[7]] for r in track], dtype=float)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        h = boxes[:, 3] - boxes[:, 1]
        teams = Counter(r[3] for r in track if r[3] not in self.IGNORED_TEAM_LABELS)
        total = sum(teams.values())
        team_p = {k: v / total for k, v in teams.items()} if total else {}
        k = min(3, len(track))
        return {
            'tracker_id': tracker_id, 'rowids': [r[0] for r in track], 'frames': frames,
            'start_frame': frames[0], 'end_frame': frames[-1],
            'start_c': (cx[0], cy[0]), 'end_c': (cx[-1], cy[-1]),
            'start_h': float(h[:k].mean()), 'end_h': float(h[-k:].mean()),
            'end_vel': self._end_velocity(t, cx, cy, h, True),
            'start_vel': self._end_velocity(t, cx, cy, h, False),
            'team_p': team_p, 'n': len(track),
        }

    ########## STEP 2: LINK FRAGMENTS
    def _link_cost(self, a, b):
        """Cost of 'fragment b continues fragment a' or None if impossible (all distances in person-height metres)."""
        gap_frames = b['start_frame'] - a['end_frame']
        if b['end_frame'] <= a['end_frame'] or b['start_frame'] <= a['start_frame']:
            return None
        d = self._shift_m(a['end_c'], a['end_h'], b['start_c'], b['start_h'])
        if gap_frames <= 0:
            # a short overlap = the same person reported twice for a frame or two (an id hand-over)
            if gap_frames < -self.LINK_MAX_OVERLAP_FRAMES or d > self.LINK_MAX_OVERLAP_DIST_M:
                return None
        gap_s = max(gap_frames, 0) / self.FPS
        if gap_s > self.LINK_MAX_GAP_S:
            return None
        if d > self.LINK_BASE_M + self.LINK_MAX_SPEED_MS * gap_s:
            return None
        compat = sum(p * b['team_p'].get(team, 0.0) for team, p in a['team_p'].items())
        if a['team_p'] and b['team_p'] and compat < self.MIN_TEAM_COMPAT:
            return None
        if not a['team_p'] or not b['team_p']:
            compat = 0.5
        lead = min(gap_s, self.MAX_PREDICT_S)
        pa = (a['end_c'][0] + a['end_vel'][0] * lead, a['end_c'][1] + a['end_vel'][1] * lead)
        pb = (b['start_c'][0] - b['start_vel'][0] * lead, b['start_c'][1] - b['start_vel'][1] * lead)
        fwd = self._shift_m(pa, a['end_h'], b['start_c'], b['start_h'])
        bwd = self._shift_m(a['end_c'], a['end_h'], pb, b['start_h'])
        return 0.5 * (fwd + bwd) + self.GAP_COST_PER_S * gap_s + self.TEAM_MISMATCH_COST * (1.0 - compat)

    def _link_fragments(self, fragments):
        n = len(fragments)
        BIG = 1e6
        order = sorted(range(n), key=lambda i: fragments[i]['end_frame'])
        cost = np.full((n, n), BIG, dtype=np.float32)
        ends = np.array([f['end_frame'] for f in fragments])
        starts = np.array([f['start_frame'] for f in fragments])
        max_gap_frames = self.LINK_MAX_GAP_S * self.FPS
        for i in range(n):
            # only fragments starting inside the allowed window after this one ends are candidates
            cand = np.where((starts > ends[i] - self.LINK_MAX_OVERLAP_FRAMES - 1) & (starts <= ends[i] + max_gap_frames))[0]
            for j in cand:
                c = self._link_cost(fragments[i], fragments[j])
                if c is not None and c <= self.MAX_LINK_COST:
                    cost[i, j] = c
        rows_i, cols_j = linear_sum_assignment(cost)
        nxt = {}
        for i, j in zip(rows_i, cols_j):
            if cost[i, j] < BIG:
                nxt[i] = j
        return nxt

    ########## STEP 3 + 4: CHAINS AND TEAM
    def _chains(self, fragments, nxt):
        has_prev = set(nxt.values())
        chains = []
        for i in range(len(fragments)):
            if i in has_prev:
                continue
            chain = [i]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            chains.append(chain)
        chains.sort(key=lambda c: (fragments[c[0]]['start_frame'], fragments[c[0]]['tracker_id']))
        return chains

    ########## RUN:
    def run(self):
        start_time = time.time()
        self._prepare_columns()

        self.cursor.execute('''
            SELECT _id, tracker_id, frame_id, team_raw, xmin, ymin, xmax, ymax
            FROM detected_objects
            WHERE class_name = "player" AND tracker_id IS NOT NULL
              AND xmin IS NOT NULL AND ymin IS NOT NULL AND xmax IS NOT NULL AND ymax IS NOT NULL
            ORDER BY tracker_id, frame_id
        ''')
        rows = self.cursor.fetchall()
        fragments = self._build_fragments(rows)
        nxt = self._link_fragments(fragments)
        chains = self._chains(fragments, nxt)

        # row id -> player id, plus the chain's majority team
        updates = []
        team_color_votes = defaultdict(Counter)
        self.cursor.execute('''SELECT team_raw, team_color_raw, COUNT(*) FROM detected_objects
                               WHERE class_name = "player" GROUP BY team_raw, team_color_raw''')
        for team, colour, n in self.cursor.fetchall():
            team_color_votes[team][colour] += n

        summary = []
        for player_id, chain in enumerate(chains, start=1):
            votes = Counter()
            for i in chain:
                for team, p in fragments[i]['team_p'].items():
                    votes[team] += p * fragments[i]['n']
            team = votes.most_common(1)[0][0] if votes else None
            colour = team_color_votes[team].most_common(1)[0][0] if team in team_color_votes else None
            for i in chain:
                for rid in fragments[i]['rowids']:
                    updates.append((player_id, team, colour, rid))
            summary.append((player_id, team, len(chain), sum(fragments[i]['n'] for i in chain),
                            fragments[chain[0]]['start_frame'], fragments[chain[-1]]['end_frame']))

        self.cursor.executemany('''
            UPDATE detected_objects SET player_id = ?, team = ?, team_color = ?
            WHERE _id = ?
        ''', updates)
        # player-class rows with no usable position keep their tracker_id and original label
        self.cursor.execute('''
            UPDATE detected_objects SET player_id = CAST(tracker_id AS INTEGER) + 100000
            WHERE class_name = "player" AND player_id IS NULL AND tracker_id IS NOT NULL
        ''')
        self.conn.commit()

        n_tracker_ids = len({f['tracker_id'] for f in fragments})
        print(f"   tracker_ids: {n_tracker_ids}  ->  fragments after cutting at jumps / team switches"
              f" ({self.team_switch_cuts} team switches): {len(fragments)}"
              f"  ->  stitched players: {len(chains)}")
        self.summary = summary
        self.fragments = fragments
        self.chains = chains

        self.cursor.close()
        self.conn.close()
        print("🔒 Database connection closed.")
        total_time = time.time() - start_time
        minutes, seconds = divmod(int(total_time), 60)
        print(f"✅ Processing time: {minutes}:{seconds} sec")
