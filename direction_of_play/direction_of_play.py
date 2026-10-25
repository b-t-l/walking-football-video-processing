"""Direction of play: which way each team attacks along the pitch (+x / -x).

Computed once after Transformations (needs team labels and x_transformed_metres), then stored in
the game's detections database so annotation / reports can reuse it without recomputing.

Simple model only:
  - detect attack direction from a short window at the start of play (mean player x per team)
  - if the Game Logger has a half-time window, flip that direction for the second half
  - otherwise one direction for the whole game

Convention (same as the opposition report):
  team_a_attacks = +1  →  team A attacks toward +x (right on the overhead plan)
  team_a_attacks = -1  →  team A attacks toward -x
  team B always attacks the opposite way
"""
import math
import sqlite3

import numpy as np

# How long after game start (seconds) to average team positions when picking the first-half direction.
SAMPLE_WINDOW_S = 20.0
# Need at least this many in-pitch player detections per team inside the sample window.
MIN_SAMPLES_PER_TEAM = 8

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS direction_of_play (
    period_index INTEGER PRIMARY KEY,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    team_a TEXT NOT NULL,
    team_b TEXT NOT NULL,
    team_a_attacks INTEGER NOT NULL,
    sample_t0 REAL,
    sample_t1 REAL,
    team_a_mean_x REAL,
    team_b_mean_x REAL
)
"""


def _num(v, default=None):
    """Seconds from the game record: a number, 'mm:ss', 'h:mm:ss' or empty/NaN."""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return default
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            out = 0.0
            for p in parts:
                out = out * 60 + p
            return out
        return float(s)
    except (TypeError, ValueError):
        return default


def load_direction(db_path):
    """Read stored periods from the detections DB, or None if the table is missing/empty.

    Returns dict:
      team_a, team_b,
      periods: [{period_index, t_start, t_end, team_a_attacks, sample_t0, sample_t1,
                 team_a_mean_x, team_b_mean_x}, ...]
    """
    conn = sqlite3.connect(str(db_path))
    try:
        names = {r[1] for r in conn.execute("PRAGMA table_info(direction_of_play)")}
        if not names:
            return None
        rows = conn.execute(
            "SELECT period_index, t_start, t_end, team_a, team_b, team_a_attacks, "
            "sample_t0, sample_t1, team_a_mean_x, team_b_mean_x "
            "FROM direction_of_play ORDER BY period_index"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    periods = []
    for r in rows:
        periods.append({
            "period_index": int(r[0]),
            "t_start": float(r[1]),
            "t_end": float(r[2]),
            "team_a_attacks": int(r[5]),
            "sample_t0": None if r[6] is None else float(r[6]),
            "sample_t1": None if r[7] is None else float(r[7]),
            "team_a_mean_x": None if r[8] is None else float(r[8]),
            "team_b_mean_x": None if r[9] is None else float(r[9]),
        })
    return {"team_a": rows[0][3], "team_b": rows[0][4], "periods": periods}


def direction_at(direction, t, team=None):
    """Attack direction (+1 / -1 toward +x) for `team` at time `t` seconds.

    `direction` is the dict from load_direction / DirectionOfPlay.result.
    If team is None, returns team A's direction. Team B is always the opposite of A.
    """
    if not direction or not direction.get("periods"):
        return None
    t = float(t)
    periods = direction["periods"]
    period = periods[0]
    for p in periods:
        if p["t_start"] <= t <= p["t_end"]:
            period = p
            break
        # after half-time gap: use the next period once t has reached it
        if t >= p["t_start"]:
            period = p
    d_a = int(period["team_a_attacks"])
    if team is None or team == direction["team_a"]:
        return d_a
    if team == direction["team_b"]:
        return -d_a
    return None


class DirectionOfPlay:
    """Compute direction of play from the detections DB and store it there."""

    def __init__(self, database_file, game_record, teams=None):
        print("-------------------------------------------------")
        print("DIRECTION OF PLAY: which way each team attacks (+x / -x)")
        self.DATABASE_FILE = database_file
        self.GAME_RECORD = game_record or {}
        if teams and len(teams) >= 2:
            self.team_a, self.team_b = teams[0], teams[1]
        else:
            self.team_a = game_record.get("team_a")
            self.team_b = game_record.get("team_b")
        if not self.team_a or not self.team_b:
            raise SystemExit("\nSTOPPED: direction of play needs team_a and team_b on the game record.\n")
        self.result = None

    def run(self):
        self.result = self._compute()
        self._save(self.result)
        self._print(self.result)
        return self.result

    def ensure(self):
        """Load from DB if present; otherwise compute, save, and return."""
        existing = load_direction(self.DATABASE_FILE)
        if existing and existing.get("periods"):
            # Refresh if the stored team names no longer match this game (renamed in logger).
            if existing["team_a"] == self.team_a and existing["team_b"] == self.team_b:
                self.result = existing
                print("  (already stored in the detections database — reusing)")
                self._print(existing)
                return existing
        return self.run()

    # ------------------------------------------------------------------ compute

    def _compute(self):
        from reports.match_report.metrics import resolve_labels

        conn = sqlite3.connect(self.DATABASE_FILE)
        try:
            video_fps, det_fps = conn.execute(
                "SELECT video_fps, detection_fps FROM game WHERE game_id = ?",
                (self.GAME_RECORD["game_id"],),
            ).fetchone()
            video_fps = float(video_fps or 24.0)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(detected_objects)")}
            rows = conn.execute(
                "SELECT frame_id, team, x_transformed_metres, is_inside_pitch, class_name "
                "FROM detected_objects WHERE class_name = 'player' AND x_transformed_metres IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            raise SystemExit(
                "\nSTOPPED: direction of play needs player positions in the detections database "
                "(run Detections and Transformations first).\n"
            )

        mapping = resolve_labels(self.DATABASE_FILE, [self.team_a, self.team_b])
        label_a, label_b = mapping[self.team_a], mapping[self.team_b]

        # Collect in-pitch detections for the two playing teams
        by_team = {label_a: [], label_b: []}
        t_min = t_max = None
        for frame_id, team, x, inside, _cls in rows:
            if team not in by_team or not inside or x is None:
                continue
            t = float(frame_id) / video_fps
            by_team[team].append((t, float(x)))
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)

        if not by_team[label_a] or not by_team[label_b]:
            raise SystemExit(
                "\nSTOPPED: direction of play could not find in-pitch players for both teams "
                f"({self.team_a!r} / {self.team_b!r}).\n"
            )

        start = _num(self.GAME_RECORD.get("game_start_seconds"), t_min)
        end = _num(self.GAME_RECORD.get("game_end_seconds"), t_max)
        start = max(float(start), float(t_min))
        end = min(float(end), float(t_max))
        ht0 = _num(self.GAME_RECORD.get("half_time_start_seconds"))
        ht1 = _num(self.GAME_RECORD.get("half_time_end_seconds"))
        has_half = ht0 is not None and ht1 is not None and ht1 > ht0

        sample_t0 = start
        sample_t1 = min(start + SAMPLE_WINDOW_S, ht0 if has_half else end)
        if sample_t1 <= sample_t0:
            sample_t1 = min(start + SAMPLE_WINDOW_S, end)

        xa, na = self._mean_x(by_team[label_a], sample_t0, sample_t1)
        xb, nb = self._mean_x(by_team[label_b], sample_t0, sample_t1)
        if na < MIN_SAMPLES_PER_TEAM or nb < MIN_SAMPLES_PER_TEAM:
            # Widen the window toward the end of the first period if the start is sparse.
            widen_end = ht0 if has_half else end
            sample_t1 = min(start + max(SAMPLE_WINDOW_S * 3, 60.0), widen_end if widen_end else end)
            xa, na = self._mean_x(by_team[label_a], sample_t0, sample_t1)
            xb, nb = self._mean_x(by_team[label_b], sample_t0, sample_t1)
        if na < 1 or nb < 1 or xa is None or xb is None:
            raise SystemExit(
                "\nSTOPPED: direction of play could not average team positions at the start of the game "
                "(not enough in-pitch player detections).\n"
            )

        # Team standing further right (+x) is defending that end → the other team attacks +x.
        # If A is further left (or tied), A attacks toward +x → team_a_attacks = +1.
        team_a_attacks = 1 if xa <= xb else -1

        periods = []
        if has_half:
            periods.append(self._period(0, start, float(ht0), team_a_attacks, sample_t0, sample_t1, xa, xb))
            periods.append(self._period(1, float(ht1), end, -team_a_attacks, None, None, None, None))
        else:
            periods.append(self._period(0, start, end, team_a_attacks, sample_t0, sample_t1, xa, xb))

        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "periods": periods,
            "video_fps": video_fps,
            "detection_fps": float(det_fps or 0),
        }

    @staticmethod
    def _mean_x(samples, t0, t1):
        xs = [x for t, x in samples if t0 <= t <= t1]
        if not xs:
            return None, 0
        return float(np.mean(xs)), len(xs)

    def _period(self, index, t_start, t_end, team_a_attacks, sample_t0, sample_t1, xa, xb):
        return {
            "period_index": index,
            "t_start": float(t_start),
            "t_end": float(t_end),
            "team_a_attacks": int(team_a_attacks),
            "sample_t0": None if sample_t0 is None else float(sample_t0),
            "sample_t1": None if sample_t1 is None else float(sample_t1),
            "team_a_mean_x": None if xa is None else float(xa),
            "team_b_mean_x": None if xb is None else float(xb),
        }

    # ------------------------------------------------------------------ persist

    def _save(self, result):
        conn = sqlite3.connect(self.DATABASE_FILE)
        try:
            conn.execute(TABLE_SQL)
            conn.execute("DELETE FROM direction_of_play")
            for p in result["periods"]:
                conn.execute(
                    "INSERT INTO direction_of_play "
                    "(period_index, t_start, t_end, team_a, team_b, team_a_attacks, "
                    "sample_t0, sample_t1, team_a_mean_x, team_b_mean_x) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        p["period_index"], p["t_start"], p["t_end"],
                        result["team_a"], result["team_b"], p["team_a_attacks"],
                        p["sample_t0"], p["sample_t1"], p["team_a_mean_x"], p["team_b_mean_x"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _print(self, result):
        def side(d):
            return "right (+x)" if d > 0 else "left (-x)"

        print(f"  {result['team_a']} vs {result['team_b']}")
        for p in result["periods"]:
            d = p["team_a_attacks"]
            label = "1st half" if p["period_index"] == 0 and len(result["periods"]) > 1 else (
                "2nd half" if p["period_index"] == 1 else "full game"
            )
            print(
                f"  {label}: t={p['t_start']:.1f}–{p['t_end']:.1f}s  "
                f"{result['team_a']} attacks {side(d)}, "
                f"{result['team_b']} attacks {side(-d)}"
            )
            if p["team_a_mean_x"] is not None:
                print(
                    f"    sample {p['sample_t0']:.1f}–{p['sample_t1']:.1f}s  "
                    f"mean x: {result['team_a']}={p['team_a_mean_x']:.1f} m, "
                    f"{result['team_b']}={p['team_b_mean_x']:.1f} m"
                )
        print(f"  saved in detections database: direction_of_play ({len(result['periods'])} period(s))")
