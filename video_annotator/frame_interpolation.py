"""
Frame interpolation for the annotated video.

Detections can be run on every 2nd (or 3rd ...) video frame to save time (DETECTION_FPS < video fps). The annotated
video is still drawn on EVERY video frame, so for the frames in between we place each box / dot half way between its
position in the detection frame before and the detection frame after (straight line, per player id).

Only the picture is interpolated. Nothing is written to the database: speed, distance, possession and team figures
stay exactly what the detections measured.

Rules
- an object (matched by class + id) in BOTH neighbouring detection frames: boxes, pitch positions, speed and
  distance move in a straight line between the two; labels (team, colour, possession ...) come from the nearer frame
  (the earlier one when exactly half way)
- an object in only ONE of the two frames (it appears / disappears): shown as it was, so nothing flickers on and off
- a jump bigger than MAX_JUMP_PX between the two detections (an id swap or a false ball) is not blended: the object
  stays where it was in the nearer frame instead of sliding across the pitch
- frames with no detection frame on one side, or neighbours further apart than allowed (half time, outside the game):
  nothing is drawn
"""
import bisect

MAX_JUMP_PX = 250          # box-centre jump (pixels at 1920 wide) above which two detections are not blended

_BLEND_FIELDS = ("xmin", "ymin", "xmax", "ymax",
                 "x_transformed_metres", "y_transformed_metres",
                 "speed_km_per_hour", "total_distance_metres", "confidence")


def _object_keys(objects):
    """key for each object: (class, id, n). Objects without an id (the ball) are numbered in order of confidence."""
    counters = {}
    key_of = {}
    for i in sorted(range(len(objects)), key=lambda i: -(objects[i].get("confidence") or 0)):
        o = objects[i]
        if o.get("tracker_id") is not None:
            key_of[i] = (o.get("class_name"), o.get("tracker_id"), 0)
        else:
            n = counters.get(o.get("class_name"), 0)
            counters[o.get("class_name")] = n + 1
            key_of[i] = (o.get("class_name"), None, n)
    return [key_of[i] for i in range(len(objects))]


def _centre(o):
    try:
        return (o["xmin"] + o["xmax"]) / 2.0, (o["ymin"] + o["ymax"]) / 2.0
    except (KeyError, TypeError):
        return None


def blend_objects(prev_objects, next_objects, alpha, max_jump_px=MAX_JUMP_PX):
    """objects between two detection frames (alpha 0 = previous frame ... 1 = next frame)."""
    prev_by_key = dict(zip(_object_keys(prev_objects), prev_objects))
    next_by_key = dict(zip(_object_keys(next_objects), next_objects))
    nearer_is_next = alpha > 0.5
    result = []
    for key in list(prev_by_key) + [k for k in next_by_key if k not in prev_by_key]:
        a = prev_by_key.get(key)
        b = next_by_key.get(key)
        if a is None or b is None:
            result.append(dict(a if a is not None else b))
            continue
        ca, cb = _centre(a), _centre(b)
        if ca is not None and cb is not None and max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1])) > max_jump_px:
            result.append(dict(b if nearer_is_next else a))
            continue
        merged = dict(b if nearer_is_next else a)          # labels from the nearer detection
        for f in _BLEND_FIELDS:
            va, vb = a.get(f), b.get(f)
            if va is None and vb is None:
                merged[f] = None
            elif va is None:
                merged[f] = vb
            elif vb is None:
                merged[f] = va
            else:
                merged[f] = va + (vb - va) * alpha
        result.append(merged)
    return result


class FrameInterpolator:
    """
    fetch(frame_id) -> (objects, pitch_points) for a frame that has detections (the annotator's database reader).
    detection_frames: every frame id that has detections.   max_gap: furthest apart (in video frames) two
    neighbouring detection frames may be and still be blended.
    """

    def __init__(self, fetch, detection_frames, max_gap, cache_size=6):
        self.fetch = fetch
        self.frames = sorted(set(detection_frames))
        self.max_gap = max_gap
        self._cache = {}
        self._cache_size = cache_size

    def _get(self, frame_id):
        if frame_id not in self._cache:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[frame_id] = self.fetch(frame_id)
        return self._cache[frame_id]

    def get(self, frame_id):
        i = bisect.bisect_left(self.frames, frame_id)
        if i < len(self.frames) and self.frames[i] == frame_id:
            return self._get(frame_id)                       # a real detection frame
        if i == 0 or i >= len(self.frames):
            return [], []                                    # before the first / after the last detection
        prev_id, next_id = self.frames[i - 1], self.frames[i]
        if next_id - prev_id > self.max_gap:
            return [], []                                    # a gap (half time, excluded time): draw nothing
        alpha = (frame_id - prev_id) / float(next_id - prev_id)
        prev_objects, prev_pitch = self._get(prev_id)
        next_objects, _ = self._get(next_id)
        return blend_objects(prev_objects, next_objects, alpha), prev_pitch
