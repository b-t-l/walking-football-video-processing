"""
Shared rules that keep 'running' honest. Used by the speed step (speed_and_distance_db_updater) and the match / opposition
reports (reports/match_report/metrics.py).

Why they exist (review of game 27, 2026-10): a player's pitch position comes from the bottom of their bounding box. When the
box height changes (an arm swings out, another player partly blocks the view) the foot point jumps by up to 1-2 m, most of
all on the far touchline where the picture is compressed. A jump like that reads as a burst of 15-23 km/h that lasts half a
second. Of the nine 'running' spells in game 27, seven lasted exactly the old half-second minimum and six needed an
acceleration no walker can produce (5-13 m/s2).

  FIX 1  limit_acceleration()   a player's speed cannot change faster than MAX_ACCEL_MS2.
  FIX 2  clip_short_spells()    speed counts as running (above the game's running speed) only if it is held for MIN_RUN_SECONDS.
"""
import numpy as np

MAX_ACCEL_MS2 = 4.0        # a fit sprinter reaches 6-9 m/s2 from a standing start; walkers and jogging seniors manage 1-3
MIN_RUN_SECONDS = 1.0      # a spell above the running speed must last this long to count (was 0.5 s)


def limit_acceleration(t, v_kmh, max_accel_ms2=MAX_ACCEL_MS2):
    """
    The highest speed curve that is never above the measured one and never changes faster than max_accel_ms2 (two passes,
    forward then backward). A genuine run (a gradual rise) is left untouched; a one-detection jump in position that shows up
    as a brief burst of speed is cut down to what a person could actually do from the speed on either side of it.
    t in seconds (increasing), v_kmh in km/h; returns a new array.
    """
    v = np.array(v_kmh, dtype=float)
    n = len(v)
    if n < 2:
        return v
    a = max_accel_ms2 * 3.6                               # km/h per second
    for i in range(1, n):
        v[i] = min(v[i], v[i - 1] + a * (t[i] - t[i - 1]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], v[i + 1] + a * (t[i + 1] - t[i]))
    return v


def clip_short_spells(t, v_kmh, threshold_kmh, det_fps, min_seconds=MIN_RUN_SECONDS):
    """
    Any stretch of consecutive detections above threshold_kmh that lasts less than min_seconds is cut down to just under the
    threshold, so it can never be counted (or drawn) as running. Longer stretches are untouched.
    t in seconds (increasing, one entry per detection of one player), v_kmh in km/h; returns (new array, spells clipped).
    """
    v = np.array(v_kmh, dtype=float)
    n = len(v)
    gap = 1.5 / det_fps + 1e-6                           # detections further apart than this are not one continuous spell
    cap = threshold_kmh * 0.999
    clipped = 0
    i = 0
    while i < n:
        if v[i] > threshold_kmh:
            j = i
            while j + 1 < n and v[j + 1] > threshold_kmh and t[j + 1] - t[j] <= gap:
                j += 1
            if (t[j] - t[i]) + 1.0 / det_fps < min_seconds - 1e-6:
                v[i:j + 1] = np.minimum(v[i:j + 1], cap)
                clipped += 1
            i = j + 1
        else:
            i += 1
    return v, clipped
