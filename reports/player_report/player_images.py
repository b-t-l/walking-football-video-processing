"""Resolve player portrait PNGs from the data folder's `_library/player_images/`.

Files are named like the player (`amanda daughters.png`). `default.png` is used when no
match is found. Matching is case-insensitive and ignores extra spaces / underscores.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

DEFAULT_STEM = "default"
# Bake portraits at print density; imshow must not resample again (that is what blurred them)
_PORTRAIT_DPI = 300


def _library_dir():
    try:
        from game_logger import paths
        root = paths.configured_root()
    except Exception:
        root = None
    if not root:
        root = (os.environ.get("WALKING_FOOTBALL_ROOT") or "").strip()
    if not root:
        return None
    return os.path.join(os.path.expanduser(root), "_library", "player_images")


def _norm(s):
    return " ".join(str(s or "").strip().lower().replace("_", " ").split())


@lru_cache(maxsize=1)
def _index():
    """stem_normalized -> absolute path (excludes default)."""
    folder = _library_dir()
    out = {}
    if not folder or not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0]
        if _norm(stem) == DEFAULT_STEM:
            continue
        out[_norm(stem)] = os.path.join(folder, name)
    return out


def default_image_path():
    folder = _library_dir()
    if not folder:
        return None
    p = os.path.join(folder, f"{DEFAULT_STEM}.png")
    return p if os.path.isfile(p) else None


def resolve_player_image_path(player_name):
    """Best PNG path for this player name, or the default portrait, or None."""
    key = _norm(player_name)
    idx = _index()
    if key and key in idx:
        return idx[key]
    # soft match: filename contained in name or vice versa
    if key:
        for stem, path in idx.items():
            if stem in key or key in stem:
                return path
    return default_image_path()


@lru_cache(maxsize=64)
def load_player_image(player_name):
    """RGBA uint8 array (H, W, 4), or None if nothing can be loaded."""
    path = resolve_player_image_path(player_name)
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image
        im = Image.open(path).convert("RGBA")
        return np.asarray(im)
    except Exception:
        return None


def _facecolor_rgb(fig):
    """Figure face colour as 0–255 RGB ints (ignore alpha)."""
    fc = fig.get_facecolor()
    if fc is None:
        return (251, 251, 248)  # match report BG
    r, g, b = fc[0], fc[1], fc[2]
    if r <= 1 and g <= 1 and b <= 1:
        return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
    return (int(round(r)), int(round(g)), int(round(b)))


def _prepare_portrait(rgba, slot_w_in, slot_h_in, bg_rgb=None):
    """Lanczos-resize the portrait. If bg_rgb is set, bake alpha onto that colour (RGB).

    If bg_rgb is None, keep RGBA so a circle drawn behind can show through.
    """
    from PIL import Image

    tw = max(48, int(round(slot_w_in * _PORTRAIT_DPI)))
    th = max(48, int(round(slot_h_in * _PORTRAIT_DPI)))
    src = Image.fromarray(rgba, mode="RGBA")
    # Fit inside slot, preserving aspect (letterbox with transparent padding)
    src_ratio = src.width / max(src.height, 1)
    slot_ratio = tw / max(th, 1)
    if src_ratio > slot_ratio:
        nw, nh = tw, max(1, int(round(tw / src_ratio)))
    else:
        nh, nw = th, max(1, int(round(th * src_ratio)))
    fitted = src.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    canvas.paste(fitted, ((tw - nw) // 2, (th - nh) // 2), fitted)
    if bg_rgb is None:
        return np.asarray(canvas)
    base = Image.new("RGBA", (tw, th), (*bg_rgb, 255))
    return np.asarray(Image.alpha_composite(base, canvas).convert("RGB"))


def draw_portrait(fig, player_name, rect, zorder=5, keep_alpha=False):
    """Draw the player portrait (or default) in figure fraction rect (x, y, w, h).

    keep_alpha=True leaves transparency so a coloured circle behind shows as a ring.
    Returns True if an image was drawn.
    """
    img = load_player_image(player_name)
    if img is None:
        return False
    x, y, w, h = rect
    fw, fh = fig.get_size_inches()
    bg = None if keep_alpha else _facecolor_rgb(fig)
    pix = _prepare_portrait(img, fw * w, fh * h, bg)

    ax = fig.add_axes([x, y, w, h], zorder=zorder, frameon=False, facecolor="none")
    ax.patch.set_visible(False)
    # "none" = already Lanczos-sized for this slot; another filter made faces look soft
    ax.imshow(pix, interpolation="none", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, pix.shape[1] - 0.5)
    ax.set_ylim(pix.shape[0] - 0.5, -0.5)
    return True
