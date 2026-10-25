"""Named pitch keypoints and their world-metre positions.

Roboflow/YOLO exports keypoints as indices 0..N-1. KEYPOINT_NAMES must match the order of
the skeleton you defined in Roboflow. If thirds look wrong, reorder this list — do not change
the world coordinates.
"""

# Default 36 x 20 m walking-football pitch (Pitch Calibrator defaults / games 22, 27).
DEFAULT_LENGTH_M = 36.0
DEFAULT_WIDTH_M = 20.0
DEFAULT_CENTRE_RADIUS_M = 3.0
DEFAULT_END_ARC_RADIUS_M = 7.5
DEFAULT_END_ARC_MAX_DEPTH_M = 6.0

# Index order from Roboflow skeleton (names from UI) + v3 label geometry.
# data.yaml does not list names; indices verified on train labels.
KEYPOINT_NAMES = [
    "left_top",           # 0
    "left_bottom",        # 1
    "right_top",          # 2
    "right_bottom",       # 3
    "centre_top",         # 4
    "centre_bottom",      # 5
    "circle_top",         # 6  halfway ∩ centre circle (+y)
    "circle_bottom",      # 7  halfway ∩ centre circle (-y)
    "arc_left_top",       # 8  left D ∩ goal line (+y)
    "arc_left_bottom",    # 9  left D ∩ goal line (-y)
    "arc_right_top",      # 10 right D ∩ goal line (+y)
    "arc_right_bottom",   # 11 right D ∩ goal line (-y)
]


def world_keypoints(length_m=None, width_m=None, centre_radius_m=None,
                    end_arc_radius_m=None, end_arc_max_depth_m=None):
    """name -> (x, y) metres. Origin at centre; +x toward right goal; +y toward top touchline."""
    L = float(length_m if length_m is not None else DEFAULT_LENGTH_M)
    W = float(width_m if width_m is not None else DEFAULT_WIDTH_M)
    Rc = float(centre_radius_m if centre_radius_m is not None else DEFAULT_CENTRE_RADIUS_M)
    D = float(end_arc_max_depth_m if end_arc_max_depth_m is not None else DEFAULT_END_ARC_MAX_DEPTH_M)
    _ = end_arc_radius_m  # reserved; apex uses max depth on this pitch
    half_l, half_w = L / 2.0, W / 2.0
    return {
        "left_bottom": (-half_l, -half_w),
        "left_top": (-half_l, +half_w),
        "right_top": (+half_l, +half_w),
        "right_bottom": (+half_l, -half_w),
        "centre_bottom": (0.0, -half_w),
        "centre_top": (0.0, +half_w),
        "centre_spot": (0.0, 0.0),
        "circle_bottom": (0.0, -Rc),
        "circle_top": (0.0, +Rc),
        "centre_circle_left": (-Rc, 0.0),
        "centre_circle_right": (+Rc, 0.0),
        "end_arc_left_apex": (-half_l + D, 0.0),
        "end_arc_right_apex": (+half_l - D, 0.0),
        "arc_left_top": (-half_l, +DEFAULT_END_ARC_RADIUS_M),
        "arc_left_bottom": (-half_l, -DEFAULT_END_ARC_RADIUS_M),
        "arc_right_top": (+half_l, +DEFAULT_END_ARC_RADIUS_M),
        "arc_right_bottom": (+half_l, -DEFAULT_END_ARC_RADIUS_M),
    }


def dims_from_game_record(game_record):
    """Pull pitch_dimensions_m from the game record calibrator JSON when present."""
    import json
    raw = (game_record or {}).get("pitch_calibrator_json")
    if not raw:
        return {}
    try:
        cal = json.loads(raw) if isinstance(raw, str) else raw
        d = cal.get("pitch_dimensions_m") or {}
        return {
            "length_m": float(d["length"]) if d.get("length") is not None else None,
            "width_m": float(d["width"]) if d.get("width") is not None else None,
            "centre_radius_m": float(d["centre_radius"]) if d.get("centre_radius") is not None else None,
            "end_arc_radius_m": float(d["end_arc_radius"]) if d.get("end_arc_radius") is not None else None,
            "end_arc_max_depth_m": float(d["end_arc_max_depth"]) if d.get("end_arc_max_depth") is not None else None,
        }
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return {}
