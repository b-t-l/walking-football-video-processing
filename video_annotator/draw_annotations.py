import os
import cv2
import numpy as np
import ast
from collections import defaultdict
from utils import *

PLAYER_COLOR = (255, 255, 0)
PLAYER_TEXT_COLOR = (0, 0, 0)
REFEREE_COLOR = (96, 136, 250)
REFEREE_TEXT_COLOR = (0,60,255)
GOALKEEPER_COLOR = (232,224,132)
GOALKEEPER_TEXT_COLOR = (173,152,37)
BALL_COLOR = (255,255,255)

STATS_BOX_WIDTH = 3820

# Bottom-panel layout was drawn for a 3840x2160 frame. Scale it onto the frame being annotated.
REF_FRAME_WIDTH = 3840
REF_FRAME_HEIGHT = 2160
REF_PANEL_TOP = 1300


HEADER_TEXT_HEIGHT_FRAC = (0.015, 0.012, 0.009)  # capital-letter height of: 'Video Analysis', teams + score, date + description (share of video height)
TOP_STRIP_FRAC = 0.12             # height of the top-left header box behind the logo and text (share of video height; 0 = none)
TOP_STRIP_FEATHER_FRAC = 0.0      # soft fade below the header box (0 = hard edge); only applies under the box, not full-width
HEADER_HEADING_GAP = 1.15          # gap between header lines, as a share of the heading's letter height (larger = more breathing room)
HEADER_MAX_WIDTH_FRAC = 0.42      # the text block is at most this wide (share of video width); the description wraps to 2 lines
HEADER_DESCRIPTION_LINES = 2
HEADER_BOX_PAD_FRAC = 0.012       # padding inside the black header box (share of video height)
HEADER_BOX_OPACITY = 0.7          # how solid the top-left header box is (1 = fully black, 0 = invisible)
BOTTOM_PANEL_OPACITY = 0.7        # how solid the black bottom stats strip is (1 = fully black, 0 = invisible)
TIMELINE_FRAC = 0.26              # share of the bottom panel height reserved at the top for the possession/goals/running strip
TOP_MASK_FEATHER_FRAC = 0.10      # the black cover above the netting fades out over this share of the video height (0 = hard edge)
_mask_cache = {}
LOGO_FILE = "polis_logo.png"      # in the images folder; set to None to leave the logo off
LOGO_HEIGHT_FRAC = 0.055          # logo height as a share of the video height (about the height of the three text lines beside it)
LOGO_MARGIN_FRAC = 0.012          # gap from the top when no header box is drawn
LOGO_TEXT_GAP_FRAC = 0.018        # gap between the logo and the header text
LOGO_LEFT_FRAC = 0.018            # gap from the left edge to the logo (the header text moves with it), as a share of the video height
_logo_cache = {}

GOAL_BALL_D = 0.40                       # diameter of the football on the goal marker, as a share of the timeline strip's height
RUN_LINE_COLOUR = (40, 40, 240)         # BGR red: dotted line on the timeline wherever a running spell started
GOAL_FLAG_W, GOAL_FLAG_H = 0.34, 0.28   # goal flag on the timeline: width and height as a share of the timeline strip's height
SHAPE_DOT_ALPHA = 0.50   # how solid the dotted white circles on the overhead map are (1 = fully solid)


def dotted_points(pts, gap):
    """Points spaced `gap` pixels apart along the polyline `pts` (N x 2), measured along the line, so a circle gets even dots."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return []
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return []
    n = max(2, int(cum[-1] // gap) + 1)
    d = np.linspace(0.0, cum[-1], n)
    return list(zip(np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])))


def frame_layout_scale(frame):
    """Scale factor that fits the 4K overlay layout inside this frame."""
    height, width = frame.shape[:2]
    scale = min(width / REF_FRAME_WIDTH, height / REF_FRAME_HEIGHT)
    return scale, width, height


# ---- look of the overlays on the video (tweak here) ----
PITCH_DOT_RADIUS = 6            # corner dots (was 15)
PITCH_DOT_ALPHA = 0.35
PITCH_LINE_THICKNESS = 2
PITCH_LINE_ALPHA = 0.30         # pitch outline, lower = more transparent (was 0.18)
PITCH_THIRD_DOT_RADIUS = 3      # dotted third-lines: smaller than corner keypoints
PITCH_THIRD_DOT_GAP = 14        # spacing along each third line (px at ~4K; scaled below)
BOX_ALPHA_NORMAL = 0.12         # ordinary player box (team colour): barely there
BOX_ALPHA_ORANGE = 0.75         # 10-15 km/h
BOX_THICKNESS_RUNNING = 3       # red (>15 km/h) is fully opaque
BRACKET_FRACTION = 0.28         # running boxes are corner brackets: each arm = this share of the box's shorter side
BRACKET_MIN_PX, BRACKET_MAX_PX = 10, 34
LABEL_FONT_SCALE = 0.6          # speed label shown only for running players
# Speed bands shared by on-field brackets and overhead dots (BGR)
SPEED_NEAR_RUN_KMH = 10.0       # above this (up to the run threshold): orange
SPEED_NEAR_RUN_COLOUR = (0, 165, 255)
SPEED_RUN_COLOUR = (0, 0, 255)  # at/above the game's running speed: red
# Overhead speed halo: large soft ring so it reads on the small inset
SPEED_RING_ALPHA = 0.75         # mostly solid, slight transparency so it glows on the pitch
SPEED_RING_RADIUS_FRAC = 3.2    # outer radius as a multiple of the player-dot radius

def draw_corner_brackets(img, x0, y0, x1, y1, color, thickness):
    arm = int(max(BRACKET_MIN_PX, min(BRACKET_MAX_PX, BRACKET_FRACTION * min(x1 - x0, y1 - y0))))
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, thickness, cv2.LINE_AA)

def draw_outlined_text(img, text, org, scale=LABEL_FONT_SCALE):
    # white text with a dark outline: readable on grass, shirts and sky alike
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def compute_bottom_layout(frame, panel_top_fraction, pitch_length_m, pitch_width_m, pitch_outline_frac=None):
    """
    Where everything in the black bottom panel goes, for a panel that starts at row panel_top
    (panel_top_fraction = how far down the frame it starts, 0-1, chosen just below the pitch so the
    whole pitch stays visible in the video above it).
    A possession/goals/running timeline strip sits across the top of the panel; the overhead inset
    and stats table share the space below it (overhead kept inside the black strip).
    Returns None when panel_top_fraction is None -> callers fall back to the legacy fixed layout.
    """
    if panel_top_fraction is None:
        return None
    _ = pitch_outline_frac  # kept on the signature for callers; overhead no longer grows into the video using it
    scale, width, height = frame_layout_scale(frame)
    panel_top = int(min(max(panel_top_fraction * height, 0.5 * height), 0.95 * height))
    margin = max(4, int(20 * scale))
    panel_h = max(40, height - panel_top)
    timeline_h = max(16, int(panel_h * TIMELINE_FRAC))
    content_top = panel_top + timeline_h + margin          # overhead + stats start below the timeline
    # Keep the overhead map inside the remaining panel space (do not grow it up into the video).
    inset_h = max(20, height - content_top - margin)
    inset_w = int(round(inset_h * pitch_length_m / pitch_width_m))
    max_w = int(0.5 * width)
    if inset_w > max_w:
        inset_w = max_w
        inset_h = int(round(inset_w * pitch_width_m / pitch_length_m))
    inset_x = margin
    inset_y = content_top
    return {
        'panel_top': panel_top, 'margin': margin,
        'timeline_h': timeline_h, 'content_top': content_top,
        'inset_x': inset_x, 'inset_y': inset_y, 'inset_w': inset_w, 'inset_h': inset_h,
        # stats sit level with the overhead inset, under the timeline
        'stats_x': inset_x + inset_w + 3 * margin, 'stats_y': content_top,
        'details_x': width - int(560 * scale), 'details_y': content_top + max(14, int(30 * scale)),
    }


def compute_top_mask_polygon(frame, top_mask_frac):
    """
    Black polygon for the top of the frame (the safety netting): everything above the line through the
    clicked points (fractions of the frame, left to right), out to the left, right and top edges.
    Returns None when fewer than 2 points are given.
    """
    if not top_mask_frac or len(top_mask_frac) < 2:
        return None
    height, width = frame.shape[:2]
    pts = sorted(((fx * width, fy * height) for fx, fy in top_mask_frac), key=lambda p: p[0])
    poly = [(0, 0), (width, 0), (width, pts[-1][1])] + pts[::-1] + [(0, pts[0][1])]
    return np.array(poly, dtype=np.int32)


def top_mask_keep(frame, polygon, feather_frac=None):
    """Per-pixel 'how much of the video to keep' (float32 H x W x 1; 0 = black) for the netting cover. Fully black inside the
    clicked polygon, then fading smoothly back to the video over feather_frac of the video height below its edge (the fade goes
    downwards only, so the line itself does not move). None = no feathering needed. Cached per frame size and polygon."""
    feather_frac = TOP_MASK_FEATHER_FRAC if feather_frac is None else feather_frac
    h, w = frame.shape[:2]
    feather = int(h * feather_frac)
    key = (h, w, polygon.tobytes(), feather)
    if key not in _mask_cache:
        if feather < 2:
            _mask_cache[key] = None
        else:
            masked = np.zeros((h, w), np.uint8)
            cv2.fillPoly(masked, [polygon], 255)
            dist = cv2.distanceTransform(255 - masked, cv2.DIST_L2, 5)      # pixels from the nearest covered pixel (0 inside)
            t = np.clip(dist / float(feather), 0.0, 1.0)
            keep = (t * t * (3.0 - 2.0 * t)).astype(np.float32)            # smoothstep: 0 at the edge, 1 at feather px away
            rows = np.where(keep.min(axis=1) < 1.0)[0]
            bottom = int(rows.max()) + 1 if len(rows) else 0
            _mask_cache[key] = (bottom, keep[:bottom, :, None].copy())
        if len(_mask_cache) > 4:
            _mask_cache.pop(next(iter(_mask_cache)))
    return _mask_cache[key]


def _blend_black(roi, opacity):
    """Darken `roi` toward black by `opacity` (1 = fully black, 0 = unchanged). Modifies and returns roi."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0:
        return roi
    if opacity >= 1:
        roi[:] = 0
        return roi
    roi[:] = (roi.astype(np.float32) * (1.0 - opacity)).astype(np.uint8)
    return roi


def apply_header_backdrop(frame, box_w, box_h, opacity=None):
    """Black box in the top-left behind the logo and header text (not a full-width bar). Soft fade only under the box."""
    if box_w < 2 or box_h < 2:
        return frame
    opacity = HEADER_BOX_OPACITY if opacity is None else opacity
    h, w = frame.shape[:2]
    box_w = min(box_w, w)
    box_h = min(box_h, h)
    feather = int(h * TOP_STRIP_FEATHER_FRAC)
    _blend_black(frame[:box_h, :box_w], opacity)
    if feather >= 2:
        rows = min(feather, h - box_h)
        if rows > 0:
            t = np.linspace(0.0, 1.0, rows, dtype=np.float32)
            # fade from the box opacity down to a clear video
            fade = (opacity * (1.0 - (t * t * (3.0 - 2.0 * t))))[:, None, None]
            band = frame[box_h:box_h + rows, :box_w].astype(np.float32)
            frame[box_h:box_h + rows, :box_w] = (band * (1.0 - fade)).astype(np.uint8)
    return frame


def apply_top_strip(frame):
    """Legacy full-width top strip. Prefer apply_header_backdrop via draw_header_block."""
    h, w = frame.shape[:2]
    strip_h = int(h * TOP_STRIP_FRAC)
    if strip_h < 2:
        return frame
    return apply_header_backdrop(frame, w, strip_h)


def apply_top_mask(frame, polygon):
    """Black out the netting above the clicked line, with a soft fade along its lower edge."""
    cached = top_mask_keep(frame, polygon)
    if cached is None:                                   # feathering off: the old hard edge
        cv2.fillPoly(frame, [polygon], (0, 0, 0))
        return frame
    bottom, keep = cached
    frame[:bottom] = (frame[:bottom].astype(np.float32) * keep).astype(np.uint8)
    return frame


############################################### DRAW ALL ANNOTATIONS: Draw all the annotations onto the video we want:
def draw_annotations_batch(frame_ids, frames, data_batch, pitch_points_batch, TEAMS, PITCH_DETECTION_OVERRIDE, DETECTION_FPS,ORIGINAL_VIDEO_FPS,GAME_RUNNING_SPEED, PITCH_LENGTH_M=46.0, PITCH_WIDTH_M=21.0, PITCH_OVERLAY=None, FOOT_PIXEL_FN=None, PANEL_TOP_FRACTION=None, PITCH_OUTLINE_FRAC=None, TOP_MASK_FRAC=None, DASHBOARD=None, HEADER=None, field_overlays=True, PITCH_THIRDS_FRAC=None, edited_pitch=None):
    """Process multiple frames in a batch for better performance.

    field_overlays=False (edited-video background): skip boxes, pitch outline and netting mask;
    still draw the overhead map, dashboard/header and top strip.

    Layer order: video → bottom panel / analysis field overlays → edited pitch thirds → HUD
    (overhead, dashboard, header). edited_pitch detects on the clean frame, then draws thirds
    after the panel so lines sit on the pitch and under the HUD.
    """
    annotation_frames = []

    frame_height, frame_width = frames[0].shape[:2]
    scale, _, _ = frame_layout_scale(frames[0])
    layout = compute_bottom_layout(frames[0], PANEL_TOP_FRACTION, PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_OUTLINE_FRAC)
    top_mask_polygon = compute_top_mask_polygon(frames[0], TOP_MASK_FRAC) if field_overlays else None
    outline_px = (np.asarray(PITCH_OUTLINE_FRAC, dtype=float) * np.array([frame_width, frame_height], dtype=float)
                  if field_overlays and PITCH_OUTLINE_FRAC is not None and len(PITCH_OUTLINE_FRAC) else None)
    thirds_px = None
    if field_overlays and PITCH_THIRDS_FRAC:
        thirds_px = [
            np.asarray(line, dtype=float) * np.array([frame_width, frame_height], dtype=float)
            for line in PITCH_THIRDS_FRAC if line
        ]
    panel_top = layout['panel_top'] if layout else min(int(REF_PANEL_TOP * scale), frame_height)

    # Process frames in batch
    for frame_id, frame, data, pitch_points in zip(frame_ids, frames, data_batch, pitch_points_batch):
        # Use numpy operations where possible
        annotation_frame = frame.copy()

        # Homography from the clean video (before panel / HUD cover any landmarks).
        if edited_pitch is not None:
            edited_pitch.update(frame)

        # Black panel under the pitch map (semi-transparent so a little of the video shows through).
        _blend_black(annotation_frame[panel_top:frame_height, 0:frame_width], BOTTOM_PANEL_OPACITY)

        # Black out the netting: everything above the line clicked along its lower edge (Pitch Calibrator).
        if top_mask_polygon is not None:
            annotation_frame = apply_top_mask(annotation_frame, top_mask_polygon)

        # Apply annotations (consider vectorizing these operations)
        if field_overlays:
            annotation_frame = draw_pitch_detection_on_field(annotation_frame, frame_id, pitch_points, outline_px, thirds_px)
            annotation_frame = draw_object_detections_on_field(annotation_frame, frame_id, data, pitch_points,GAME_RUNNING_SPEED, FOOT_PIXEL_FN)

        # Edited/broadcast thirds: above the video (+ panel), below overhead / dashboard / header.
        if edited_pitch is not None:
            edited_pitch.draw_onto(annotation_frame)

        annotation_frame = draw_pitch_top_view(annotation_frame, data, pitch_points, PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_OVERLAY, layout, GAME_RUNNING_SPEED)
        # live panel (clock/video time, team table, timeline) - or, if it cannot be drawn, the plain team boxes + the frame details on the right
        d = DASHBOARD.at(frame_id) if (DASHBOARD is not None and layout) else None
        if not (d is not None and draw_dashboard(annotation_frame, frame_id, data, DASHBOARD, layout, ORIGINAL_VIDEO_FPS, DETECTION_FPS, d)):
            annotation_frame = draw_stats(annotation_frame, data, layout=layout)
            annotation_frame = draw_video_details(annotation_frame, frame_id, data, DETECTION_FPS, ORIGINAL_VIDEO_FPS, layout)
        # top-left: black box behind logo + header text only (not a full-width bar)
        live_title = None
        if d is not None and HEADER:
            a_name, b_name = DASHBOARD.teams
            live_title = f"{a_name}  {d['score'][0]} - {d['score'][1]}  {b_name}" if d.get("score") is not None else f"{a_name} vs {b_name}"
        annotation_frame = draw_header_block(annotation_frame, HEADER, live_title)
        
        annotation_frames.append(annotation_frame)
    
    return annotation_frames

def _ensure_logo(logo_h):
    """Load/resize the club logo into _logo_cache. Returns (rgb, alpha) or None."""
    if not LOGO_FILE:
        return None
    if logo_h not in _logo_cache:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", LOGO_FILE)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim < 3 or img.shape[2] != 4:
            _logo_cache[logo_h] = None
        else:
            logo_w = max(1, int(img.shape[1] * logo_h / img.shape[0]))
            small = cv2.resize(img, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
            _logo_cache[logo_h] = (small[..., :3].astype(np.float32), small[..., 3:4].astype(np.float32) / 255.0)
    return _logo_cache[logo_h]


def _wrap_text(text, fs, th, max_w, max_lines):
    """word-wrap to max_w pixels, at most max_lines lines (the last one ends with '...' when the text is cut)"""
    words, lines, cur = text.split(), [], ""
    for i, w in enumerate(words):
        trial = (cur + " " + w).strip()
        if _text_w(trial, fs, th) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
        if len(lines) == max_lines:
            break
    else:
        lines.append(cur)
        return lines
    lines = lines[:max_lines]
    last = lines[-1]
    while last and _text_w(last + "...", fs, th) > max_w:
        last = last[:-1].rstrip()
    lines[-1] = last + "..."
    return lines


def _header_content(frame, header, live_title=None):
    """Measure the top-left header: logo, text lines, and the black box that should sit behind them."""
    h, w = frame.shape[:2]
    pad = max(4, int(h * HEADER_BOX_PAD_FRAC))
    logo_h = max(8, int(h * LOGO_HEIGHT_FRAC))
    logo = _ensure_logo(logo_h)
    lx = int(h * LOGO_LEFT_FRAC)
    logo_w = logo[0].shape[1] if logo else 0
    x0 = lx + (logo_w + int(h * LOGO_TEXT_GAP_FRAC) if logo else 0)
    max_w = max(80, int(w * HEADER_MAX_WIDTH_FRAC) - x0)
    caps = [max(6, int(h * f)) for f in HEADER_TEXT_HEIGHT_FRAC]
    detail = "  |  ".join(x for x in ((header or {}).get("date"), (header or {}).get("description")) if x) if header else ""
    gap = max(int(h * 0.008), int(caps[0] * HEADER_HEADING_GAP))  # breathing room between the three header lines
    lines = []
    if header:
        for text, colour, k, max_lines in (
            (header.get("heading"), (2, 202, 254), 0, 1),
            (live_title or header.get("title"), (255, 255, 255), 1, 1),
            (detail, (200, 200, 200), 2, HEADER_DESCRIPTION_LINES),
        ):
            if not text:
                continue
            fs = _fit(caps[k])
            th = max(1, int(round(caps[k] / 14.0)))
            for line in _wrap_text(text, fs, th, max_w, max_lines):
                lines.append((line, colour, k, fs, th))
    text_w = max((_text_w(line, fs, th) for line, _, _, fs, th in lines), default=0)
    text_h = 0
    if lines:
        text_h = caps[lines[0][2]] + sum(gap + caps[k] for _, _, k, _, _ in lines[1:])
    content_h = max(logo_h if logo else 0, text_h)
    box_h = max(int(h * TOP_STRIP_FRAC), content_h + 2 * pad) if (logo or lines) and TOP_STRIP_FRAC > 0 else 0
    box_w = (lx + max(logo_w, 0) + (int(h * LOGO_TEXT_GAP_FRAC) if logo and lines else 0) + text_w + pad) if (logo or lines) else 0
    if box_w:
        box_w = min(w, box_w + pad)
    return {
        "pad": pad, "logo": logo, "lx": lx, "logo_h": logo_h, "x0": x0,
        "caps": caps, "gap": gap, "lines": lines, "text_h": text_h,
        "box_w": box_w, "box_h": box_h, "content_h": content_h,
    }


def draw_logo(frame, layout=None):
    """Club logo in the top-left corner (images/polis_logo.png). Silently skipped if missing."""
    h, w = frame.shape[:2]
    layout = layout or _header_content(frame, None)
    cached = layout["logo"]
    if cached is None:
        return frame
    rgb, alpha = cached
    lx = layout["lx"]
    box_h = layout["box_h"] or int(h * TOP_STRIP_FRAC)
    m = int(h * LOGO_MARGIN_FRAC)
    ty = (box_h - rgb.shape[0]) // 2 if box_h >= rgb.shape[0] else m
    y1, x1 = min(ty + rgb.shape[0], h), min(lx + rgb.shape[1], w)
    rgb, alpha = rgb[:y1 - ty, :x1 - lx], alpha[:y1 - ty, :x1 - lx]
    roi = frame[ty:y1, lx:x1].astype(np.float32)
    frame[ty:y1, lx:x1] = (rgb * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return frame


def draw_header(frame, header, live_title=None, layout=None):
    """'Video Analysis', teams + score, and date + description to the right of the logo."""
    if not header:
        return frame
    h = frame.shape[0]
    layout = layout or _header_content(frame, header, live_title)
    lines, caps, gap = layout["lines"], layout["caps"], layout["gap"]
    if not lines:
        return frame
    x0 = layout["x0"]
    box_h = layout["box_h"] or int(h * TOP_STRIP_FRAC)
    m = int(h * LOGO_MARGIN_FRAC)
    top = (box_h - layout["text_h"]) // 2 if box_h >= layout["text_h"] else m
    y = top + caps[lines[0][2]]
    for i, (line, colour, k, fs, th) in enumerate(lines):
        if i:
            y += gap + caps[k]
        cv2.putText(frame, line, (x0, y), _FONT, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x0, y), _FONT, fs, colour, th + (1 if k == 0 else 0), cv2.LINE_AA)
    return frame


def draw_header_block(frame, header, live_title=None):
    """Top-left black box sized to the logo + text, then the logo and header lines on top of it."""
    layout = _header_content(frame, header, live_title)
    if layout["box_w"] > 0 and layout["box_h"] > 0:
        apply_header_backdrop(frame, layout["box_w"], layout["box_h"])
    draw_logo(frame, layout)
    draw_header(frame, header, live_title, layout)
    return frame

# Original function maintained for backward compatibility
def draw_annotations(frame_id, frame, data, pitch_points, TEAMS, PITCH_DETECTION_OVERRIDE, DETECTION_FPS):
    """Wrapper for single frame processing"""
    return draw_annotations_batch(
        [frame_id], 
        [frame], 
        [data], 
        [pitch_points], 
        TEAMS, 
        PITCH_DETECTION_OVERRIDE, 
        DETECTION_FPS
    )[0]


def is_point_left_or_right(pitch_keypoints, obj_x, obj_y):
    """
    Determine if a point (obj_x, obj_y) is left or right of the line 
    running from centre_top to centre_bottom.
    
    Args:
        pitch_keypoints (list): List of keypoints with 'name', 'x', 'y'.
        obj_x (int): X-coordinate of the object.
        obj_y (int): Y-coordinate of the object.
    
    Returns:
        str: 'left', 'right', or 'on the line'
    """
    # Extract centre_top and centre_bottom from the keypoints
    centre_top = next(kp for kp in pitch_keypoints if kp['name'] == 'centre_top')
    centre_bottom = next(kp for kp in pitch_keypoints if kp['name'] == 'centre_bottom')
    
    x1, y1 = centre_top['x'], centre_top['y']
    x2, y2 = centre_bottom['x'], centre_bottom['y']
    
    # Calculate cross product
    cross_product = (x2 - x1) * (obj_y - y1) - (y2 - y1) * (obj_x - x1)
    
    if cross_product > 0:
        return 'left'
    elif cross_product < 0:
        return 'right'
    else:
        return 'centre'
    
##################################### -- draws the predicted pitch outline on the video
def draw_pitch_detection_on_field(frame,frame_id,pitch_points,outline_px=None,thirds_px=None):
    """Corner dots + pitch outline (+ optional thirds-grid dotted lines). outline_px / thirds_px are
    dense polylines in frame pixels (follow the lens curvature); without outline_px the corner dots
    are joined by straight lines."""

    dots = frame.copy()
    lines = frame.copy()
    scale, _, _ = frame_layout_scale(frame)

    for i in range(len(pitch_points)):
        name = pitch_points[i]['name']
        point = (int(pitch_points[i]['x']), int(pitch_points[i]['y']))
        cv2.circle(dots, point, PITCH_DOT_RADIUS, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(frame, name, (point[0], point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if outline_px is None:
            next_point = (int(pitch_points[(i + 1) % len(pitch_points)]['x']),
                          int(pitch_points[(i + 1) % len(pitch_points)]['y']))
            cv2.line(lines, point, next_point, (0, 255, 0), PITCH_LINE_THICKNESS, cv2.LINE_AA)

    if outline_px is not None:
        cv2.polylines(lines, [np.asarray(outline_px, dtype=np.int32).reshape(-1, 1, 2)], True,
                      (0, 255, 0), PITCH_LINE_THICKNESS, cv2.LINE_AA)

    # Thirds grid (across + along): same green as the outline, dotted so they read as zones not edges
    if thirds_px:
        gap = max(8, int(PITCH_THIRD_DOT_GAP * scale))
        radius = max(2, int(PITCH_THIRD_DOT_RADIUS * scale))
        for line in thirds_px:
            for cx, cy in dotted_points(np.asarray(line, dtype=np.float64), gap):
                cv2.circle(lines, (int(round(cx)), int(round(cy))), radius, (0, 255, 0), -1, cv2.LINE_AA)

    cv2.addWeighted(lines, PITCH_LINE_ALPHA, frame, 1 - PITCH_LINE_ALPHA, 0, frame)
    cv2.addWeighted(dots, PITCH_DOT_ALPHA, frame, 1 - PITCH_DOT_ALPHA, 0, frame)
    return frame



def draw_pointer(frame, x, y, color, half_w, height):
    """Down-pointing triangle with its tip at (x, y): smooth edges and an outline that contrasts with
    the fill (dark outline normally, white when the fill itself is dark, e.g. a black kit)."""
    pts = np.array([(x, y), (x - half_w, y - height), (x + half_w, y - height)], np.float32)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    outline = (255, 255, 255) if lum < 70 else (0, 0, 0)
    fixed = np.round(pts * 16).astype(np.int32)                      # sub-pixel (shift=4) so the edges are smooth
    cv2.fillConvexPoly(frame, fixed, tuple(int(c) for c in color), cv2.LINE_AA, 4)
    cv2.polylines(frame, [fixed], True, outline, max(1, int(round(half_w / 9))), cv2.LINE_AA, 4)


############################################### DRAW PLAYER ANNOTATION ON FRAME VIDEO
def draw_object_detections_on_field(frame,frame_id,data,pitch_points,GAME_RUNNING_SPEED,foot_pixel_fn=None):

    # Get the centre top pitch point ( we will use to place the annoation for each player/goalkeeper/ref on left or side of centre)
    #centre_top = next((kp for kp in pitch_points if kp['name'] == 'centre_top'), None)
    # just draw out in out points:
    # Define the points

    '''
    polygon_points = [(297, 1104), (1005, 217), (1913, 217), (2868, 199), (3624, 1122), (1913, 1096)]

    # Draw red circles at each point
    radius = 10
    color = (0, 0, 255)  # Red in BGR
    thickness = -1  # Filled circle

    for point in polygon_points:
        cv2.circle(frame, point, radius, color, thickness)

    # Draw lines connecting the points
    line_color = (255, 255, 255)  # White lines
    line_thickness = 2

    for i in range(len(polygon_points)):
        # Get the current point and the next point (wrap around for the last point)
        start_point = polygon_points[i]
        end_point = polygon_points[(i + 1) % len(polygon_points)]  # Wrap around to the first point
        cv2.line(frame, start_point, end_point, line_color, line_thickness)
    '''


    ########## 1. draw the tranparent items we want:

    # Create an overlay for the object detections as we want all this dat to be slightly transparent.
    overlay = frame.copy()  # Create a copy of the frame for the overlay

    normal_boxes, orange_boxes, red_boxes, running_labels = [], [], [], []
    # Draw the transformed points of players on the video:
    for object in data:
        
        # skip doing anything if it is a ball of referee:
        if object['class_name'] == "ball" or object['team'] == "Goalkeeper" or object['team'] == "Referee":
            continue
        
        #### Setup the variables:
        x_min =  int(object['xmin'])
        y_min =  int(object['ymin'])
        x_max =  int(object['xmax'])
        y_max =  int(object['ymax'])
        tracker_id = int(object["tracker_id"])
        team = object['team']
        team_color = object['team_color']
        speed = object['speed_km_per_hour']
        if speed is None:
            speed = 0
        distance = object['total_distance_metres']

        # work out if the object is on the left/right/center of the pitch from the centre of the bbox:
        object_placement = is_point_left_or_right(pitch_points, ((x_max-x_min)/2 + x_min), ((y_max-y_min)/2 + y_min))
        
        #### color:
        color = PLAYER_COLOR
        text_color = PLAYER_TEXT_COLOR
        if object["team_color"]:
            color = ast.literal_eval(object["team_color"])
            text_color = get_contrasting_color(color)
        
        ##### draw the circle showing the foot position and team color:
        '''
        if object_placement == 'left' and object['class_name'] != "ball":
            # Draw the circle
            cv2.circle(overlay, (x_max,y_max+4), 8, color, -1)
        elif object_placement == 'right' and object['class_name'] != "ball":
            cv2.circle(overlay, (x_min,y_max+4), 8, color, -1)
        '''
        ##### draw the plus sign showing the foot position and team color:
        foot_px = foot_pixel_fn(x_min, y_min, x_max, y_max) if foot_pixel_fn else None
        if foot_px is not None:
            # smooth foot position (same point the database metres were computed from)
            center_x, center_y = int(round(foot_px[0])), int(round(foot_px[1]))
        elif object_placement == 'left' and object['class_name'] != "ball":
            # Calculate the center of the plus sign
            center_x, center_y = x_max, y_max + 4
        elif object_placement == 'right' and object['class_name'] != "ball":
            center_x, center_y = x_min, y_max + 4
        # Draw the plus sign
        line_length = 8  # Length of each line in the plus sign
        line_thickness = 2  # Thickness of the lines
        # Horizontal line
        cv2.line(overlay, (center_x - line_length, center_y), (center_x + line_length, center_y), color, line_thickness)
        # Vertical line
        cv2.line(overlay, (center_x, center_y - line_length), (center_x, center_y + line_length), color, line_thickness)


        ### Draw the line for team
        '''
        y_start_drawing = int(y_max+3)
        x_start_drawing = x_min
        line_thickness = 2
        line_length = min(((x_max - x_min) // 2),80)
        line_start = (x_start_drawing, y_start_drawing)
        line_end = (x_start_drawing + line_length, y_start_drawing)
        if object_placement == 'left':
            line_start = (x_max - line_length, y_start_drawing)
            line_end = (x_max, y_start_drawing)
        elif object_placement == 'right':
            line_start = (x_start_drawing, y_start_drawing)
            line_end = (x_start_drawing + line_length, y_start_drawing)
        cv2.line(overlay, line_start, line_end, color, line_thickness)
        '''

        ##### speed color: determine a speed based on relative to GAME_RUNNING_SPEED and show different color for the speed green,orange, red ( eg 0-5kmh = green, 5-10kmh = dark green, 10-15:orange , >15kmh = red )
        
        # we have 3 distinctions for running colors:
        game_running_segment = GAME_RUNNING_SPEED/3
        speed_color = (0, 255, 0)
        speed_line_length = 10
        max_line_length = 80
        # Determine speed color and line length based on speed relative to GAME_RUNNING_SPEED
        if speed is not None and distance is not None:
            if speed <= 0:
                speed_color = (0, 0, 0)  # Green for 0-5 km/h
                speed_line_length = max(min(max_line_length, int(0.3 * speed)), 10)  # 30% of speed, minimum 10 pixels
            elif speed <= 10:
                speed_color = (0, 0, 0)  # Dark green for 5-10 km/h
                speed_line_length = max(min(max_line_length, int(0.4 * speed)), 10)  # 40% of speed, minimum 10 pixels
            elif speed <= 15:
                speed_color = (0, 165, 255)  # Orange for 10-15 km/h
                speed_line_length = max(min(max_line_length, int(0.5 * speed)), 10)  # 50% of speed, minimum 10 pixels
            else:
                speed_color = (0, 0, 255)  # Red for >15 km/h
                speed_line_length = max(min(max_line_length, int(0.6 * speed)), 10)  # 60% of speed, minimum 10 pixels
        
            speed_line_length = max(min(max_line_length, speed_line_length), 10)  # Ensure minimum length
            #speed_line_length = 40
        else:
            speed_line_length = 10  # Set line length to 0 if speed is None

        #print(f"speed: {speed} - {speed_line_length}")
        
        ### Draw a bounding box around the player with different color for running:
        # (drawn after the main blend, each with its own opacity)
        if speed > 15:
            red_boxes.append((x_min, y_min, x_max, y_max))
            running_labels.append((x_min, y_min, f"{int(object['tracker_id'])} - {speed:.1f} km/h"))
        elif speed > 10:
            orange_boxes.append((x_min, y_min, x_max, y_max))
            running_labels.append((x_min, y_min, f"{int(object['tracker_id'])} - {speed:.1f} km/h"))
        else:
            normal_boxes.append((x_min, y_min, x_max, y_max, color))
        
        ### add the bounding box area as text for debugging:
        current_bbox_area = (x_max - x_min) * (y_max - y_min)


        ### show the tracker id + speed of player (show speed only if running )
        player_id = f"{str(int(object['tracker_id']))}"
        confidence = object['confidence']
        player_text_ypos = int(y_max) + 10
        player_text_xpos = int(x_min) + 10
        if object_placement == 'left':
            player_text_xpos = int(x_max) - 50
        ### show the confidence of the prediction:



        #### DRAW line underneath for team and line for speed they are travelling:
        #speed_y_start_drawing = int(y_max-5)
        #speed_x_start_drawing = x_min
        #speed_line_thickness = 2
        #speed_line_start = (speed_x_start_drawing, speed_y_start_drawing)
        #speed_line_end = (speed_x_start_drawing + speed_line_length, speed_y_start_drawing)

        #cv2.line(overlay, speed_line_start, speed_line_end, color, line_thickness)


    # Blend the overlay with the original frame using transparency
    alpha = 0.6  # Set the transparency level (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the original frame

    # player boxes: ordinary play almost invisible; running boxes stand out (orange fairly, red fully opaque)
    if normal_boxes:
        faint = frame.copy()
        for x0, y0, x1, y1, col in normal_boxes:
            cv2.rectangle(faint, (x0, y0), (x1, y1), col, 1)
        cv2.addWeighted(faint, BOX_ALPHA_NORMAL, frame, 1 - BOX_ALPHA_NORMAL, 0, frame)
    if orange_boxes:
        orange = frame.copy()
        for x0, y0, x1, y1 in orange_boxes:
            draw_corner_brackets(orange, x0, y0, x1, y1, (0, 165, 255), BOX_THICKNESS_RUNNING)
        cv2.addWeighted(orange, BOX_ALPHA_ORANGE, frame, 1 - BOX_ALPHA_ORANGE, 0, frame)
    for x0, y0, x1, y1 in red_boxes:
        draw_corner_brackets(frame, x0, y0, x1, y1, (0, 0, 255), BOX_THICKNESS_RUNNING)
    for lx, ly, text in running_labels:
        draw_outlined_text(frame, text, (lx, max(18, ly - 10)))


    ######################### 2: Draw fully opaque elements directly on the frame: BALL AND POSESSION
    # small down-pointing triangles with a dark outline and smooth edges, sized for the frame:
    # white = the ball, team colour = the player in possession of it
    ui = frame_layout_scale(frame)[0] * 2.0       # 1.0 on a 1920 px wide frame
    for object in data:

        x_min = int(object['xmin'])
        y_min = int(object['ymin'])
        x_max = int(object['xmax'])
        y_max = int(object['ymax'])
        x, _ = get_center_of_bbox(x_min, y_min, x_max, y_max)

        if object['class_name'] == "ball":
            draw_pointer(frame, x, y_min - int(5 * ui), BALL_COLOR, 6 * ui, 11 * ui)

        if object['ball_posession'] == True:
            color = PLAYER_COLOR
            if object.get('team_color'):
                try:
                    color = tuple(int(c) for c in ast.literal_eval(object['team_color']))
                except Exception:
                    pass
            draw_pointer(frame, x, y_min - int(8 * ui), color, 9 * ui, 15 * ui)

    return frame



##################################### -- draws the predictedtop view pitch with players/ball position
_GRASS_TILE = None
GRASS_STRIPE_PAIR_M = 7.0   # one light+dark mowing-stripe pair is about this wide on the real pitch
GRASS_BRIGHTNESS = 0.80     # dim the texture a little so the dots and dotted circles stand out
_grass_cache = {}


def grass_texture(w, h, pitch_length_m):
    """Mown-grass texture of size w x h (BGR), or None if images/pitch_grass.jpg is missing.
    The source tile (two stripes) is tiled along the goal-to-goal axis so the stripes run
    across the pitch, with one stripe pair ~GRASS_STRIPE_PAIR_M metres wide."""
    global _GRASS_TILE
    key = (int(w), int(h), round(float(pitch_length_m), 1))
    if key in _grass_cache:
        return _grass_cache[key]
    if _GRASS_TILE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "pitch_grass.jpg")
        _GRASS_TILE = cv2.imread(path)
        if _GRASS_TILE is None:
            _GRASS_TILE = False
    if _GRASS_TILE is False:
        return None
    period = max(8, int(round(w * GRASS_STRIPE_PAIR_M / max(1.0, pitch_length_m))))
    tile = cv2.resize(_GRASS_TILE, (period, int(h)), interpolation=cv2.INTER_AREA)
    reps = int(np.ceil(w / period)) + 1
    wide = np.tile(tile, (1, reps, 1))
    # centre the tiling so the halfway line falls on a stripe edge (mirror-symmetric look)
    start = (wide.shape[1] - w) // 2
    out = wide[:, start:start + int(w)]
    out = np.clip(out.astype(np.float32) * GRASS_BRIGHTNESS, 0, 255).astype(np.uint8)
    if len(_grass_cache) > 8:
        _grass_cache.clear()
    _grass_cache[key] = out
    return out


def draw_pitch_top_view(frame,data,pitch_points, pitch_length_m=46.0, pitch_width_m=21.0, overlay=None, layout=None, GAME_RUNNING_SPEED=15.0):

    # Real-world rectangle dimensions for THIS game (legacy default 46x21 kept only for
    # backward compatibility with callers that don't pass a per-game size).
    # NOTE: "pitch_width" here is actually the length axis (goal-to-goal) and
    # "pitch_height" is the width axis (touchline-to-touchline) -- naming kept as-is
    # to minimise the diff against the rest of this function.
    pitch_width = pitch_length_m  # meters
    pitch_height = pitch_width_m  # meters

    # Pitch map size and position on a 3840x2160 frame, then scaled to this frame.
    new_width = 460  # pixels
    new_height = 210  # pixels
    scaling_factor = 3.6
    scale, frame_width, frame_height = frame_layout_scale(frame)

    scaled_width = max(1, int(new_width * scaling_factor * scale))
    scaled_height = max(1, int(new_height * scaling_factor * scale))
    if overlay:
        # Calibrated game: draw the inset at the pitch's TRUE aspect ratio (width/length) so a
        # circle stays a circle. The fit-to-frame block below shrinks it uniformly if needed.
        scaled_height = max(1, int(scaled_width * pitch_height / pitch_width))
    x_offset = int(50 * scale)
    y_offset = int(1350 * scale)

    max_width = max(1, frame_width - x_offset)
    max_height = max(1, frame_height - y_offset)
    if scaled_width > max_width or scaled_height > max_height:
        fit = min(max_width / scaled_width, max_height / scaled_height)
        scaled_width = max(1, int(scaled_width * fit))
        scaled_height = max(1, int(scaled_height * fit))
    x_offset = min(x_offset, frame_width - scaled_width)
    y_offset = min(y_offset, frame_height - scaled_height)
    if layout:
        # panel layout chosen by compute_bottom_layout (inset fills the bottom panel)
        scaled_width, scaled_height = layout['inset_w'], layout['inset_h']
        x_offset, y_offset = layout['inset_x'], layout['inset_y']
    # size of dots/lines relative to the inset, so a smaller inset gets proportionally smaller markers
    ui = (scaled_width / max(1.0, new_width * scaling_factor * scale)) if layout else 1.0

    # Function to transform points with scaling
    def transform_point_to_new_rectangle(x, y, pitch_width, pitch_height, scaled_width, scaled_height, x_offset, y_offset):
        # World coords from the fitted per-video transformer are centred on the pitch
        # middle (0,0), not a corner -- shift into 0..pitch_width / 0..pitch_height
        # before scaling into the inset rectangle. (Harmless no-op under the old
        # corner-origin convention would have needed its own un-shifted values, but
        # that convention is no longer produced anywhere in this pipeline.)
        x_shifted = x + (pitch_width / 2.0)
        y_shifted = y + (pitch_height / 2.0)
        x_scaled = (x_shifted / pitch_width) * scaled_width
        y_scaled = (y_shifted / pitch_height) * scaled_height
        x_final = int(x_scaled + x_offset)
        y_final = int(y_scaled + y_offset)
        return x_final, y_final

    # Load the background image
    if overlay:
        # plain grass colour: pitch_map.jpg has its own pre-drawn 46x21 markings which would
        # confuse an accuracy test (our own ideal markings are drawn below instead)
        grass = grass_texture(scaled_width, scaled_height, pitch_length_m)
        if grass is not None:
            frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = grass
        else:
            frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = (46, 96, 46)
    else:
        background_img = cv2.imread("images/pitch_map.jpg")  # Replace with your background image
        if background_img is None:
            raise FileNotFoundError("The background image could not be loaded. Check the file path.")

        # Resize the background image to match the scaled rectangle dimensions
        background_img_resized = cv2.resize(background_img, (scaled_width, scaled_height))

        # Add the background image to the frame at the specified position
        frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = background_img_resized

    # Draw the scaled rectangle on the existing frame
    cv2.rectangle(
        frame,
        (x_offset, y_offset),
        (x_offset + scaled_width, y_offset + scaled_height),
        (255, 255, 255),
        2  # White rectangle
    )

    
    ## ACCURACY TEST OVERLAY (optional): ideal centre circle / goal-end arcs (dotted white) and where the
    ## clicked calibration circle points actually land after the fitted transform (yellow dots).
    ## If the fit were perfect every yellow dot would sit exactly on a dotted white line.
    def world_to_inset(wx, wy):
        px, py = transform_point_to_new_rectangle(wx, wy, pitch_width, pitch_height, scaled_width, scaled_height, x_offset, y_offset)
        return px, (y_offset + scaled_height) - (py - y_offset)

    if overlay:
        line_w = max(1, int(3 * scale * ui))
        # halfway line
        cv2.line(frame, world_to_inset(0, -pitch_height / 2), world_to_inset(0, pitch_height / 2), (255, 255, 255), max(1, int(2 * scale * ui)), cv2.LINE_AA)
        # centre circle and goal-end arcs: a dotted, semi-transparent white line (blended over the pitch inset only)
        roi = frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width]
        dots = roi.copy()
        dot_r = max(2, int(3.0 * scale * ui))
        dot_gap = max(7.0, 16.0 * scale * ui)
        for shape in overlay.get('shapes', []):
            shape_pts = np.array([world_to_inset(wx, wy) for wx, wy in shape['pts']], dtype=np.float64)
            shape_pts -= (x_offset, y_offset)
            for cx, cy in dotted_points(shape_pts, dot_gap):
                cv2.circle(dots, (int(round(cx)), int(round(cy))), dot_r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(dots, SHAPE_DOT_ALPHA, roi, 1.0 - SHAPE_DOT_ALPHA, 0, roi)
        for pt in overlay.get('points', []):
            c = world_to_inset(pt['x'], pt['y'])
            cv2.circle(frame, c, max(3, int(11 * scale * ui)), (0, 0, 0), -1)
            cv2.circle(frame, c, max(2, int(8 * scale * ui)), (0, 255, 255), -1)

    ## Draw the transformed pitch points and connecting lines from the pitch_points inputted on the offest and scaled map
    # Transform and draw pitch points
    transformed_pitch_points = []
    for point in pitch_points:
        transformed_point = transform_point_to_new_rectangle(
            point["x_transformed_metres"],
            point["y_transformed_metres"],
            pitch_width,
            pitch_height,
            scaled_width,
            scaled_height,
            x_offset,
            y_offset
        )
        # Invert the y-coordinate to match the pitch's orientation
        transformed_point = (transformed_point[0], (y_offset + scaled_height) - (transformed_point[1] - y_offset))
        transformed_pitch_points.append(transformed_point)

        # Draw the point on the frame
        cv2.circle(frame, transformed_point, max(1, int(8 * scale * ui)), (0, 255, 0), -1)  # Green dots for pitch points
    
    # Connect pitch points with lines
    for i in range(len(transformed_pitch_points)):
        start_point = transformed_pitch_points[i]
        end_point = transformed_pitch_points[(i + 1) % len(transformed_pitch_points)]  # Loop back to the start
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Yellow lines for pitch edges
  

    ball_positions = []
    player_circle_radius = max(3, int(25 * scale * ui))   # defined up front: the ball is drawn after the loop, which may not run
    speed_halo_radius = max(player_circle_radius + 8, int(player_circle_radius * SPEED_RING_RADIUS_FRAC))
    players_to_draw = []  # (cx, cy, team_colour, possession, run_ring) — speed halos first, then solid dots
    ## Draw the transformed points of players on the top view pitch:
    for object in data:


        # - draw the player:
        if object["x_transformed_metres"] and object["y_transformed_metres"]:

            map_position = transform_point_to_new_rectangle(float(object["x_transformed_metres"]), 
                                                                float(object["y_transformed_metres"]), 
                                                                pitch_width, 
                                                                pitch_height, 
                                                                scaled_width, 
                                                                scaled_height,
                                                                x_offset,
                                                                y_offset)
            # invert the y pos :
            final_position_y = (y_offset + scaled_height) - ( map_position[1] - y_offset )   # dot centred on the foot position (was lifted by its radius)
            # create the final position to use in drawing the circle ( include the diameter offset because of size )
            

            # draw a circle representing the player/referee/goalkeeper
            if object['class_name'] != "ball" and object['class_name'] != "referee":
                
                # deal with colors
                circle_color = PLAYER_COLOR
                if object["team_color"]:
                    circle_color = ast.literal_eval(object["team_color"])

                # same speed bands as on-field brackets: orange near-running, red at/above running speed
                speed = object.get('speed_km_per_hour')
                if speed is None:
                    speed = 0
                run_threshold = float(GAME_RUNNING_SPEED) if GAME_RUNNING_SPEED else 15.0
                run_ring = None
                if speed > run_threshold:
                    run_ring = SPEED_RUN_COLOUR
                elif speed > SPEED_NEAR_RUN_KMH:
                    run_ring = SPEED_NEAR_RUN_COLOUR

                players_to_draw.append((
                    map_position[0], final_position_y, circle_color,
                    bool(object['ball_posession']), run_ring,
                ))

            
            # the ball is collected here and drawn after every player, so it is never hidden under a player's dot
            if object['class_name'] == "ball":
                ball_positions.append((map_position[0], final_position_y))

    # Large soft speed rings first (translucent), then opaque team dots on top so the coloured
    # annulus stays thick and readable on the small overhead inset.
    if any(ring is not None for *_, ring in players_to_draw):
        speed_overlay = frame.copy()
        for cx, cy, _colour, _poss, run_ring in players_to_draw:
            if run_ring is not None:
                cv2.circle(speed_overlay, (cx, cy), speed_halo_radius, run_ring, -1, cv2.LINE_AA)
        cv2.addWeighted(speed_overlay, SPEED_RING_ALPHA, frame, 1.0 - SPEED_RING_ALPHA, 0, frame)

    for cx, cy, circle_color, possession, _run_ring in players_to_draw:
        if possession:
            cv2.circle(frame, (cx, cy), player_circle_radius + max(1, int(5 * scale * ui)), (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), player_circle_radius, circle_color, -1, cv2.LINE_AA)

    ball_radius = max(3, int(player_circle_radius * 0.55))
    for bx, by in ball_positions:
        cv2.circle(frame, (bx, by), ball_radius + max(2, int(3 * scale * ui)), (0, 0, 0), -1, cv2.LINE_AA)   # dark ring so it stands out on a player's dot
        cv2.circle(frame, (bx, by), ball_radius, BALL_COLOR, -1, cv2.LINE_AA)

    return frame


############################################### DRAW STATS NEXT TO THE PTICH TOP VIEW: TEAM/NR OF PLAYERS
def draw_stats(frame, data, padding_top=20, padding_bottom=20, font_scale=1.5, layout=None, dashboard=None, frame_id=None):

    # the live dashboard (score, possession, distance, running, top speed, timeline) replaces the plain team boxes when it is available
    if dashboard is not None and layout and frame_id is not None and draw_dashboard(frame, frame_id, data, dashboard, layout):
        return frame
    
    # Dictionary to count occurrences of teams and their colors
    team_data = defaultdict(lambda: {'team_color': None, 'nr_of_players': 0})

    # Process the data
    for entry in data:
        team = entry['team']
        team_color = entry['team_color']
        
        # Skip Goalkeeper and Referee teams, only count players
        if team not in ['Goalkeeper', 'Referee'] and entry['class_name'] == 'player':
            team_data[team]['nr_of_players'] += 1
            if team_data[team]['team_color'] is None:
                team_data[team]['team_color'] = team_color

    # Convert defaultdict to a regular list of dictionaries and sort alphabetically
    team_results = sorted(
        [{'team': team, 'team_color': info['team_color'], 'nr_of_players': info['nr_of_players']}
         for team, info in team_data.items()],
        key=lambda x: x['team']
    )
    
    # Team statistics sit to the right of the pitch map. Positions are for 3840x2160.
    scale, _, _ = frame_layout_scale(frame)
    offset_x = layout['stats_x'] if layout else int(1800 * scale)
    y_offset = layout['stats_y'] if layout else int(1400 * scale)
    padding_top = max(2, int(padding_top * scale))
    padding_bottom = max(2, int(padding_bottom * scale))
    font_scale = font_scale * scale
    text_thickness = max(1, int(4 * scale))
    box_padding = max(4, int(20 * scale))
    text_pad = max(2, int(10 * scale))
        
    for team in team_results:

        team_name = f"{team['team']}: {team['nr_of_players']} players"
        team_color = ast.literal_eval(team["team_color"])
        contrast_color = get_contrasting_color(team_color)

        # Calculate the size of the text based on the dynamic font scale
        text_size, _ = cv2.getTextSize(team_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        text_width, text_height = text_size
        
        # Adjust the height of the rectangle with padding
        rect_height = text_height + padding_top + padding_bottom
            
        # Draw background rectangle with padding
        cv2.rectangle(frame, (offset_x, y_offset), (offset_x + text_width + box_padding, y_offset + rect_height), team_color, -1)
            
        # Draw the text over the rectangle, adjusting for padding
        cv2.putText(frame, team_name, (offset_x + text_pad, y_offset + text_height + padding_top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, contrast_color, text_thickness)
            
        # Update y_offset for the next team
        y_offset += rect_height + max(4, int(20 * scale))

    return frame



############################################### LIVE DASHBOARD: right of the overhead map (numbers come from dashboard.LiveDashboard)
DASH_COLUMNS = ("Players", "Possession", "Last 2 min", "Distance", "Running", "Top speed (1 s)")
DASH_MIN_W, DASH_MIN_H = 520, 110          # (at 3840 wide) smaller than this and the plain team boxes are used instead
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_H1 = cv2.getTextSize("Hg", _FONT, 1.0, 1)[0][1]


def _fit(h):
    """font scale whose capital height is about h pixels"""
    return max(0.2, h / _TEXT_H1)


def _text_w(text, fs, th):
    return cv2.getTextSize(text, _FONT, fs, th)[0][0]


def _timeline_image(dash, w, h):
    """the possession-over-the-game strip, drawn once per video size"""
    cache = getattr(dash, "_tl_cache", None)
    if cache is not None and cache[0] == (w, h):
        return cache[1]
    img = np.full((h, w, 3), (38, 38, 38), dtype=np.uint8)
    nb = len(dash.bins)
    if nb:
        for k, (a, b, n) in enumerate(dash.bins):
            x_a, x_b = int(k * w / nb), max(int((k + 1) * w / nb), int(k * w / nb) + 1)
            if n <= 0 or a + b == 0:
                continue
            fa, fb = a / (a + b), b / (a + b)
            ha = int(round(fa * (h - 2)))
            img[1:1 + ha, x_a:x_b] = dash.colour_light[dash.teams[0]]
            img[1 + ha:h - 1, x_a:x_b] = dash.colour_light[dash.teams[1]]
            known = (a + b) / n                      # bars fade towards grey when the ball was seen only part of the time
            if known < 0.9:
                shade = np.full_like(img[1:h - 1, x_a:x_b], (38, 38, 38))
                img[1:h - 1, x_a:x_b] = cv2.addWeighted(img[1:h - 1, x_a:x_b], 0.35 + 0.65 * known, shade, 0.65 * (1 - known), 0)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    dash._tl_cache = ((w, h), img)
    return img


_ball_icon_cache = {}


def _ball_icon(diameter):
    """Small football icon (white ball, black patches, dark rim) as (BGR float32 image, alpha float32 image), drawn big and
    shrunk so the edges are smooth. Cached per size."""
    d = max(7, int(diameter))
    if d in _ball_icon_cache:
        return _ball_icon_cache[d]
    S = 8                                                    # supersampling
    N = d * S
    img = np.zeros((N, N, 3), np.uint8)
    alpha = np.zeros((N, N), np.uint8)
    c = (N // 2, N // 2)
    R = N // 2 - S                                           # leave room for the rim
    cv2.circle(alpha, c, N // 2 - 1, 255, -1, cv2.LINE_AA)
    cv2.circle(img, c, N // 2 - 1, (30, 30, 30), -1, cv2.LINE_AA)       # dark rim
    cv2.circle(img, c, R, (250, 250, 250), -1, cv2.LINE_AA)             # white ball
    import math

    def pentagon(cx, cy, r, rot):
        return np.array([(cx + r * math.cos(rot + k * 2 * math.pi / 5), cy + r * math.sin(rot + k * 2 * math.pi / 5)) for k in range(5)], np.int32)

    cv2.fillConvexPoly(img, pentagon(c[0], c[1], R * 0.36, -math.pi / 2), (30, 30, 30), cv2.LINE_AA)
    for k in range(5):                                       # the five patches round the edge, clipped by the ball
        a = -math.pi / 2 + k * 2 * math.pi / 5 + math.pi / 5
        px, py = c[0] + R * 0.92 * math.cos(a), c[1] + R * 0.92 * math.sin(a)
        patch = np.zeros((N, N), np.uint8)
        cv2.fillConvexPoly(patch, pentagon(px, py, R * 0.30, a + math.pi), 255, cv2.LINE_AA)
        inside = np.zeros((N, N), np.uint8)
        cv2.circle(inside, c, R, 255, -1)
        img[(patch > 0) & (inside > 0)] = (30, 30, 30)
    small = cv2.resize(img, (d, d), interpolation=cv2.INTER_AREA).astype(np.float32)
    a_small = cv2.resize(alpha, (d, d), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255.0
    _ball_icon_cache[d] = (small, a_small)
    return _ball_icon_cache[d]


def draw_ball_icon(frame, cx, cy, diameter):
    rgb, a = _ball_icon(diameter)
    d = rgb.shape[0]
    x1, y1 = cx - d // 2, cy - d // 2
    x2, y2 = x1 + d, y1 + d
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (rgb * a + roi * (1.0 - a)).astype(np.uint8)


def draw_goal_flag(frame, x, y_top, y_bottom, colour):
    """Goal marker on the possession timeline: a thin dotted vertical line the height of the strip with a small flag in the
    scoring team's colour at its top."""
    h = y_bottom - y_top
    step = max(4, int(h * 0.11))
    r = max(1, int(h * 0.03))
    for yy in range(y_top + 1, y_bottom, step):
        cv2.circle(frame, (x, yy), r + 1, (0, 0, 0), -1, cv2.LINE_AA)          # dark halo so the dots show on yellow and green bars alike
        cv2.circle(frame, (x, yy), r, (255, 255, 255), -1, cv2.LINE_AA)
    fw, fh = max(6, int(h * GOAL_FLAG_W)), max(5, int(h * GOAL_FLAG_H))
    flag = np.array([(x, y_top + 1), (x + fw, y_top + 1 + fh // 2), (x, y_top + 1 + fh)], np.int32)
    cv2.fillConvexPoly(frame, flag, tuple(int(c) for c in colour), cv2.LINE_AA)
    cv2.polylines(frame, [flag], True, (255, 255, 255), 1, cv2.LINE_AA)
    draw_ball_icon(frame, x, (y_top + y_bottom) // 2, int(h * GOAL_BALL_D))      # a little football in the middle of the line


def draw_run_line(frame, x, y_top, y_bottom):
    """Running marker on the possession timeline: a red dotted vertical line the height of the strip (no flag, so it
    cannot be mistaken for a goal)."""
    h = y_bottom - y_top
    step = max(5, int(h * 0.15))
    r = max(1.5, h * 0.03)               # dot radius in pixels (fractions allowed)
    for yy in range(y_top + 1, y_bottom, step):
        cv2.circle(frame, (int(x * 16), int(yy * 16)), int(r * 16), RUN_LINE_COLOUR, -1, cv2.LINE_AA, 4)    # plain red dots, no outline (sub-pixel radius)


def draw_dashboard(frame, frame_id, data, dash, layout, video_fps=None, detection_fps=None, d=None):
    """Possession/goals/running timeline across the top of the HUD, then clock + team stats below.
    Returns False (nothing drawn) when the panel is too small, so the caller can fall back."""
    scale, W, H = frame_layout_scale(frame)
    margin = layout['margin']
    x0, x1 = layout['stats_x'], W - margin
    content_y0 = layout.get('content_top', layout['panel_top'] + margin)
    y1 = H - margin
    if (x1 - x0) < DASH_MIN_W * scale or (y1 - content_y0) < DASH_MIN_H * scale:
        return False
    d = d if d is not None else dash.at(frame_id)
    tl_h = layout.get('timeline_h') or max(16, int((H - layout['panel_top']) * TIMELINE_FRAC))
    total = y1 - content_y0
    gap = max(3, int(total * 0.028))
    hl_h = int(total * 0.20)
    hdr_h = int(total * 0.12)
    row_h = max(12, (total - hl_h - hdr_h - 3 * gap) // 2)
    th = max(1, int(round((y1 - layout['panel_top']) / 90)))

    # ---- timeline across the full panel width at the top of the HUD
    ty = layout['panel_top'] + max(2, margin // 2)
    tl_x0, tl_x1 = margin, W - margin
    tw = max(2, tl_x1 - tl_x0)
    # leave a thin gap under the strip before the overhead / stats
    usable_tl = min(tl_h - max(2, margin // 2), content_y0 - ty - 2)
    usable_tl = max(12, usable_tl)
    frame[ty:ty + usable_tl, tl_x0:tl_x0 + tw] = _timeline_image(dash, tw, usable_tl)
    span = max(dash.end - dash.start, 1.0)
    for t_run in getattr(dash, "run_marks", []):                    # running first, so a goal marker sits on top if they coincide
        draw_run_line(frame, tl_x0 + int(min(max((t_run - dash.start) / span, 0), 1) * (tw - 1)), ty, ty + usable_tl)
    for g in dash.goals if d["score"] is not None else []:
        gx = tl_x0 + int(min(max((g["t"] - dash.start) / span, 0), 1) * (tw - 1))
        draw_goal_flag(frame, gx, ty, ty + usable_tl, dash.colour_light[g["team"]])
    nx = tl_x0 + int(min(max((d["t"] - dash.start) / span, 0), 1) * (tw - 1))
    cv2.line(frame, (nx, ty - 3), (nx, ty + usable_tl + 3), (255, 255, 255), max(2, th), cv2.LINE_AA)

    # ---- headline: game clock, then frame id / detection fps (same size/colour as the column headers below)
    fs_h = _fit(hdr_h * 0.55)                     # shared with Players / Possession / ...
    hdr_colour = (165, 165, 165)
    fs = _fit(hl_h * 0.66)
    base = content_y0 + int(hl_h * 0.90)
    cv2.putText(frame, d["phase"], (x0, base), _FONT, fs, (255, 255, 255), th + 1, cv2.LINE_AA)
    info_parts = [f"Frame id: {frame_id}"]
    if detection_fps:
        info_parts.append(f"Detection fps: {detection_fps}")
    info = "   |   ".join(info_parts)
    cv2.putText(frame, info, (x0 + _text_w(d["phase"], fs, th + 1) + int(60 * scale), base),
                _FONT, fs_h, hdr_colour, max(1, th - 1), cv2.LINE_AA)

    # ---- rows: team box + the numbers
    counts = {}
    for e in data:
        if e.get('class_name') == 'player' and e.get('team') not in ('Goalkeeper', 'Referee'):
            counts[e['team']] = counts.get(e['team'], 0) + 1
    # Team-name chip only; player count is its own column with the other stats
    fs_chip = _fit(row_h * 0.28)
    chip_texts = [t["name"] for t in d["teams"]]
    chip_pad_x = max(8, int(row_h * 0.28))
    chip_w = max(_text_w(c, fs_chip, th + 1) for c in chip_texts) + 2 * chip_pad_x
    cx0 = x0 + chip_w + int(40 * scale)
    col_w = max(1.0, (x1 - cx0) / len(DASH_COLUMNS))
    hdr_y = content_y0 + hl_h + gap + int(hdr_h * 0.85)
    for c, name in enumerate(DASH_COLUMNS):
        cv2.putText(frame, name, (int(cx0 + c * col_w), hdr_y), _FONT, fs_h, hdr_colour, max(1, th - 1), cv2.LINE_AA)
    fs_v = _fit(row_h * 0.44)
    for k, t in enumerate(d["teams"]):
        ry = content_y0 + hl_h + gap + hdr_h + gap + k * (row_h + gap)
        colour = t["colour"]
        cv2.rectangle(frame, (x0, ry), (x0 + chip_w, ry + row_h), colour, -1)
        if sum(colour) < 200:       # a black kit: a light edge so the box still shows on the black panel
            cv2.rectangle(frame, (x0, ry), (x0 + chip_w, ry + row_h), (150, 150, 150), 1)
        # centre the name in the chip (OpenCV baseline sits below the glyph centre)
        (_, chip_text_h), baseline = cv2.getTextSize(chip_texts[k], _FONT, fs_chip, th + 1)
        text_y = ry + (row_h + chip_text_h) // 2 - baseline // 2
        cv2.putText(frame, chip_texts[k], (x0 + chip_pad_x, text_y), _FONT, fs_chip, get_contrasting_color(colour), th + 1, cv2.LINE_AA)
        vals = [str(counts.get(t["label"], 0)),
                f"{t['poss']:.0f}%" if t["poss"] is not None else "-",
                f"{t['poss_recent']:.0f}%" if t["poss_recent"] is not None else "-",
                f"{t['dist_km']:.2f} km" if t["dist_km"] > 0.005 else "-",
                str(t["runs"]),
                f"{t['top']:.1f} km/h" if t["top"] > 0 else "-"]
        for c, v in enumerate(vals):
            cv2.putText(frame, v, (int(cx0 + c * col_w), ry + int(row_h * 0.68)), _FONT, fs_v, (255, 255, 255), th + 1, cv2.LINE_AA)
    return True


############################################### DRAW VIDEO STATS ON THE RIGHT SIDE:
def draw_video_details(frame,frame_id,data,DETECTION_FPS,ORIGINAL_VIDEO_FPS, layout=None):

    scale, _, _ = frame_layout_scale(frame)
    offset_x = layout['details_x'] if layout else int(3000 * scale)
    y_offset = layout['details_y'] if layout else int(1400 * scale)
    font_scale = 1 * scale
    thickness = max(1, int(2 * scale))
    line_gap = max(16, int(50 * scale))

    text = f"Frame id: {str(frame_id)}"
    cv2.putText(frame, text, (offset_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    detection_fps_txt = f"Detection fps: {DETECTION_FPS}"
    cv2.putText(frame, detection_fps_txt, (offset_x, y_offset + line_gap), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    video_position_time = TimeUtils.frame_to_timestamp(frame_id,ORIGINAL_VIDEO_FPS)
    cv2.putText(frame, video_position_time, (offset_x, y_offset + line_gap * 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    return frame



############################################### GENERAL REUSED FUNCTIONS
def get_contrasting_color(bgr_color):
    """
    Compute a contrasting text color (black or white) for the given BGR background color.
    
    Args:
        bgr_color (tuple): Background color in BGR format (B, G, R).
    
    Returns:
        tuple: Contrasting color in BGR format.
    """
    b, g, r = bgr_color
    # Calculate luminance (perceived brightness)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Choose black for bright backgrounds, white for dark backgrounds
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)

def get_center_of_bbox(x1,y1,x2,y2):
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(x_min,y_min,x_max,y_max):
    return x_max - x_min

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(x1,y1,x2,y2):
    return int((x1+x2)/2),int(y2)