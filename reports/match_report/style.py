"""Look and feel of the match report: colours, fonts, the page frame (header/footer/logo) and the pitch drawing."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from .metrics import HALF_L, HALF_W

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

PAGE_W, PAGE_H = 11.69, 8.27                    # A4 landscape, inches
BG = "#fbfbf8"
INK = "#1d2822"
MUTED = "#6a766f"
FAINT = "#dfe3dd"
CARD = "#ffffff"
BRAND = "#028c38"                               # Polis green
UNKNOWN = "#c9cfc9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "axes.edgecolor": FAINT, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
})


# ------------------------------------------------------------------------------------------------ colours

def _lum(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def team_colours(kit_rgb_a, kit_rgb_b):
    """Chart colours for the two teams, taken from their kit colours but kept readable on a light page and apart from each other."""
    out = []
    for rgb in (kit_rgb_a, kit_rgb_b):
        c = tuple(v / 255 for v in rgb) if rgb else None
        if c is None:
            out.append(None)
            continue
        if _lum(c) < 0.12:
            c = (0.17, 0.18, 0.20)                   # black kit -> charcoal
        elif _lum(c) > 0.75:
            c = tuple(v * 0.55 for v in c)           # very light kit -> darker so it shows on white
        out.append(c)
    defaults = [(0.01, 0.55, 0.22), (0.17, 0.18, 0.20)]
    out = [o if o is not None else defaults[i] for i, o in enumerate(out)]
    if np.linalg.norm(np.array(out[0]) - np.array(out[1])) < 0.35:
        out[1] = (0.87, 0.45, 0.0)                   # too alike: the second team becomes orange
    return [matplotlib.colors.to_hex(c) for c in out]


def tint(hexcol, amount):
    """Mix a colour with white (amount 0 = the colour, 1 = white)."""
    c = np.array(to_rgb(hexcol))
    return matplotlib.colors.to_hex(c + (1 - c) * amount)


def heat_cmap(hexcol):
    return LinearSegmentedColormap.from_list("h", [(1, 1, 1, 0), tint(hexcol, 0.55) + "aa", hexcol + "ff"])


# ------------------------------------------------------------------------------------------------ page frame

_logos = {}


def logo(name="green"):
    if name not in _logos:
        _logos[name] = plt.imread(os.path.join(ASSETS, f"logo-{name}.png"))
    return _logos[name]


class Frame:
    """Creates pages with the same header and footer and keeps count."""

    def __init__(self, title, subtitle_line, total_pages):
        self.title, self.line, self.total, self.n = title, subtitle_line, total_pages, 0

    def new(self, heading, intro=None, cover=False):
        self.n += 1
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
        if not cover:
            # header
            fig.text(0.05, 0.935, heading, fontsize=21, fontweight="bold", color=INK, va="center")
            if intro:
                fig.text(0.05, 0.893, intro, fontsize=10.5, color=MUTED, va="center")
            lax = fig.add_axes([0.915, 0.905, 0.04, 0.075]); lax.imshow(logo("green")); lax.axis("off")
            fig.add_artist(plt.Line2D([0.05, 0.95], [0.868, 0.868], color=FAINT, lw=1))
            # footer
            fig.add_artist(plt.Line2D([0.05, 0.95], [0.052, 0.052], color=FAINT, lw=0.8))
            fig.text(0.05, 0.03, self.line, fontsize=8, color=MUTED, va="center")
            fig.text(0.95, 0.03, f"{self.n} / {self.total}", fontsize=8, color=MUTED, va="center", ha="right")
        return fig


def card(fig, rect, fc=CARD, ec=FAINT, lw=1, radius=0.008):
    """A white rounded panel drawn behind content (rect in figure fractions)."""
    x, y, w, h = rect
    p = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", transform=fig.transFigure,
                               fc=fc, ec=ec, lw=lw, zorder=0)
    fig.add_artist(p)
    return p


# ------------------------------------------------------------------------------------------------ the pitch

def draw_pitch(ax, line="#9aa59d", lw=1.0, fill=None, goal=True):
    """The 36 x 20 m pitch as an overhead plan: x along the length, far touchline at the top (as on the video)."""
    if fill:
        ax.add_patch(patches.Rectangle((-HALF_L, -HALF_W), 2 * HALF_L, 2 * HALF_W, fc=fill, ec="none", zorder=0))
    ax.add_patch(patches.Rectangle((-HALF_L, -HALF_W), 2 * HALF_L, 2 * HALF_W, fc="none", ec=line, lw=lw, zorder=3))
    ax.plot([0, 0], [-HALF_W, HALF_W], color=line, lw=lw, zorder=3)
    ax.add_patch(patches.Circle((0, 0), 3.0, fc="none", ec=line, lw=lw, zorder=3))
    ax.plot([0], [0], "o", color=line, ms=2, zorder=3)
    for s in (-1, 1):                                       # end arcs: radius 7.5 across, flattened to 6 m deep
        ax.add_patch(patches.Arc((s * HALF_L, 0), 12, 15, theta1=90, theta2=270 if s > 0 else 450, ec=line, lw=lw, zorder=3))
        if goal:
            ax.add_patch(patches.Rectangle((s * HALF_L - (0.6 if s < 0 else 0), -1.5), 0.6, 3.0, fc="none", ec=line, lw=lw, zorder=3))
    ax.set_xlim(-HALF_L - 1, HALF_L + 1)
    ax.set_ylim(-HALF_W - 1, HALF_W + 1)
    ax.set_aspect("equal")
    ax.axis("off")


def pitch_heat(ax, H, colour, vmax=None):
    """Draw an occupancy grid (36 x 20 cells, x then y) on a pitch axis."""
    H = np.array(H)
    ax.imshow(H.T, origin="lower", extent=[-HALF_L, HALF_L, -HALF_W, HALF_W], cmap=heat_cmap(colour),
              vmin=0, vmax=vmax or (H.max() or 1), interpolation="bilinear", zorder=1, aspect="equal")
