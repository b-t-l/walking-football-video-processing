"""
Smoother player movement on the overhead map (picture only - nothing is written to the database).

The pitch positions of a player (x/y_transformed_metres) wobble from one detection to the next by more than the player
actually moves, so the dots on the overhead jitter. Here every player's track is smoothed with a short centred
(Gaussian) average over the detections around each moment: no lag, because it looks both backwards and forwards.

- a track is only smoothed within an unbroken run: it is cut where detections are more than MAX_GAP_FRAMES apart or the
  player 'jumps' more than MAX_JUMP_M between two detections (an id swap / bad detection), so dots never slide across
  the pitch and a jump stays a jump
- the ball is not smoothed (it is only seen part of the time and really does change direction suddenly)
"""
import numpy as np

MAX_GAP_FRAMES = 4     # video frames between neighbouring detections above which the track is cut (detections are 2 apart at 12 fps)
MAX_JUMP_M = 2.5       # metres between two neighbouring detections above which the track is cut


def _smooth_run(f, x, y, sigma_frames):
    d = f[:, None] - f[None, :]
    w = np.exp(-(d.astype(float) ** 2) / (2.0 * sigma_frames ** 2))
    w[np.abs(d) > 3 * sigma_frames] = 0.0
    w /= w.sum(axis=1, keepdims=True)
    return w @ x, w @ y


def smoothed_positions(cursor, sigma_s, video_fps, id_expression="tracker_id"):
    """{(frame_id, player id): (x_metres, y_metres)} for every player / goalkeeper / referee on the pitch."""
    cursor.execute(f'''
        SELECT frame_id, {id_expression}, x_transformed_metres, y_transformed_metres
        FROM detected_objects
        WHERE is_inside_pitch = 1 AND class_name != 'ball'
          AND x_transformed_metres IS NOT NULL AND y_transformed_metres IS NOT NULL
          AND {id_expression} IS NOT NULL
        ORDER BY {id_expression}, frame_id
    ''')
    rows = np.array(cursor.fetchall(), dtype=float)
    out = {}
    if rows.size == 0 or sigma_s <= 0:
        return out
    sigma_frames = sigma_s * video_fps
    ids = rows[:, 1]
    starts = np.r_[0, np.where(np.diff(ids) != 0)[0] + 1, len(rows)]
    for a, b in zip(starts[:-1], starts[1:]):
        t = rows[a:b]
        f, x, y = t[:, 0], t[:, 2], t[:, 3]
        pid = int(t[0, 1])
        cut = (np.diff(f) > MAX_GAP_FRAMES) | (np.hypot(np.diff(x), np.diff(y)) > MAX_JUMP_M)
        edges = np.r_[0, np.where(cut)[0] + 1, len(f)]
        for s, e in zip(edges[:-1], edges[1:]):
            if e - s < 3:
                for i in range(s, e):
                    out[(int(f[i]), pid)] = (float(x[i]), float(y[i]))
                continue
            xs, ys = _smooth_run(f[s:e], x[s:e], y[s:e], sigma_frames)
            for i, k in enumerate(range(s, e)):
                out[(int(f[k]), pid)] = (float(xs[i]), float(ys[i]))
    return out
