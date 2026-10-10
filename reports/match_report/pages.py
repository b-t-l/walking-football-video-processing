"""The twelve pages of the match report. Each function takes the page frame, the metrics dictionary M and a context
(colours, game details, video link) and returns a finished matplotlib figure."""
import math

import numpy as np
from matplotlib import patches

from . import style as S
from .metrics import HALF_L, HALF_W, mmss


class Ctx:
    def __init__(self, M, record, colours, link_base=None, created=""):
        self.M, self.rec, self.col, self.link_base, self.created = M, record, colours, link_base, created
        self.teams = M["teams"]
        self.cA, self.cB = colours
        self.colour = {self.teams[0]: colours[0], self.teams[1]: colours[1]}

    def link(self, seconds):
        if not self.link_base:
            return None
        sep = "&" if "?" in self.link_base else "?"
        return f"{self.link_base}{sep}t={int(max(seconds, 0))}s"


def _dotp(fig, x, y, R, fc):
    """A round dot (R = radius as a fraction of the page width; the page is not square, so the height is scaled)."""
    return patches.Ellipse((x, y), 2 * R, 2 * R * S.PAGE_W / S.PAGE_H, transform=fig.transFigure, fc=fc, ec="none")


def _nice(v, nd=0):
    return f"{v:,.{nd}f}"


def _team_legend(fig, ctx, x=0.885, y=0.935):
    """Two coloured dots with the team names, top right of a page (placed from the right, using the real text widths)."""
    r = fig.canvas.get_renderer()
    pos = x
    for i in (1, 0):
        t = fig.text(pos, y, ctx.teams[i], fontsize=10.5, color=S.INK, ha="right", va="center", fontweight="bold")
        w = t.get_window_extent(r).width / fig.bbox.width
        pos -= w + 0.012
        fig.add_artist(_dotp(fig, pos, y, 0.0058, ctx.col[i]))
        pos -= 0.030


def _legend_chips(fig, ctx, x0, y, fs=10):
    pass


def _style_axes(ax, ylabel=None, grid="y"):
    ax.set_facecolor("none")
    ax.tick_params(length=0, labelsize=8.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(S.FAINT)
    if grid:
        ax.grid(axis=grid, color=S.FAINT, lw=0.8)
        ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def _block_labels(M):
    return [mmss(a - M["start"]) for a, _ in M["blocks"]]


def _txt(fig, x, y, s, **kw):
    kw.setdefault("fontsize", 10)
    kw.setdefault("color", S.INK)
    return fig.text(x, y, s, **kw)


def _big(fig, x, y, value, label, colour=S.INK, size=30, w=0.2):
    fig.text(x, y, value, fontsize=size, fontweight="bold", color=colour, va="baseline")
    fig.text(x, y - 0.034, label, fontsize=9, color=S.MUTED, va="top")


def _split_bar(fig, rect, a, b, ca, cb, label_fmt="{:.0f}%"):
    """A horizontal bar split between the two teams. rect = (x, y, w, h) in figure fractions."""
    x, y, w, h = rect
    tot = (a + b) or 1
    wa = w * a / tot
    fig.add_artist(patches.Rectangle((x, y), wa, h, transform=fig.transFigure, fc=ca, ec="none"))
    fig.add_artist(patches.Rectangle((x + wa, y), w - wa, h, transform=fig.transFigure, fc=cb, ec="none"))
    fig.text(x + 0.006, y + h / 2, label_fmt.format(a), color="white", fontsize=9.5, va="center", fontweight="bold")
    fig.text(x + w - 0.006, y + h / 2, label_fmt.format(b), color="white", fontsize=9.5, va="center", ha="right", fontweight="bold")


# ======================================================================================== 1 cover
def cover(fr, ctx):
    M, rec = ctx.M, ctx.rec
    fig = fr.new("", cover=True)
    fig.add_artist(patches.Rectangle((0, 0), 1, 0.30, transform=fig.transFigure, fc=S.BRAND, ec="none", zorder=0))
    lax = fig.add_axes([0.06, 0.40, 0.17, 0.5]); lax.imshow(S.logo("green")); lax.axis("off")
    fig.text(0.30, 0.82, "MATCH REPORT", fontsize=13, color=S.BRAND, fontweight="bold")
    title = str(rec.get("title") or " vs ".join(ctx.teams))
    fig.text(0.30, 0.74, title, fontsize=34, fontweight="bold", color=S.INK, va="center")
    # the two teams
    y = 0.62
    for i, nm in enumerate(ctx.teams):
        fig.add_artist(_dotp(fig, 0.31, y, 0.011, ctx.col[i]))
        fig.text(0.33, y, nm, fontsize=18, va="center", color=S.INK)
        y -= 0.065
    score = rec.get("score")
    score = "" if score is None or str(score).lower() == "nan" else str(score)
    if score:
        fig.text(0.74, 0.62, score, fontsize=44, fontweight="bold", color=S.INK, va="center", ha="center")
        fig.text(0.74, 0.545, "final score", fontsize=9, color=S.MUTED, ha="center")
    meta = [str(rec.get("date") or ""), f"{rec.get('game_format', '')} · {rec.get('game_type', '')}".strip(" ·"),
            f"{mmss(M['duration'])} of play analysed"]
    fig.text(0.30, 0.43, "   |   ".join(m for m in meta if m and m.strip()), fontsize=12, color=S.MUTED)
    # headline tiles on the green band
    p = M["possession"]["share"]
    d = [M["distance"]["per_player_m"][t] for t in ctx.teams]
    tiles = [("Possession", f"{p[0]:.0f}% – {p[1]:.0f}%"),
             ("Distance per player", f"{d[0]:,.0f} m – {d[1]:,.0f} m"),
             ("Running events", f"{M['running']['count'][ctx.teams[0]]} – {M['running']['count'][ctx.teams[1]]}")]
    for k, (lab, val) in enumerate(tiles):
        x = 0.06 + k * 0.31
        fig.text(x, 0.185, val, fontsize=24, color="white", fontweight="bold")
        fig.text(x, 0.135, f"{lab}  ({ctx.teams[0].split()[0]} – {ctx.teams[1].split()[0]})", fontsize=10, color="#d8f0df")
    fig.text(0.06, 0.06, f"Report created {ctx.created}", fontsize=8.5, color="#d8f0df")
    fig.text(0.94, 0.06, "Polis Walking Football · match analysis", fontsize=8.5, color="#d8f0df", ha="right")
    return fig


# ======================================================================================== 2 summary
def summary(fr, ctx):
    M = ctx.M
    A, B = ctx.teams
    fig = fr.new("Match summary", "The game at a glance, both teams side by side.")
    _team_legend(fig, ctx)
    P = M["possession"]
    sh = M["shape"]
    rows = [
        ("Possession", P["share"][0], P["share"][1], "{:.0f}%"),
        ("Possession spells", P["spell_count"][0], P["spell_count"][1], "{:.0f}"),
        ("Average spell (seconds)", P["avg_spell_s"][0], P["avg_spell_s"][1], "{:.1f}"),
        ("Ball won from the opponent", P["won_from_opponent"][0], P["won_from_opponent"][1], "{:.0f}"),
        ("Distance per player (m)", M["distance"]["per_player_m"][A], M["distance"]["per_player_m"][B], "{:,.0f}"),
        ("Team distance, 5 players (m)", M["distance"]["team_total_m"][A], M["distance"]["team_total_m"][B], "{:,.0f}"),
        ("Average speed (km/h)", M["speed"]["avg"][A], M["speed"]["avg"][B], "{:.1f}"),
        ("Time above the running limit", M["speed"]["bands"][A][-1], M["speed"]["bands"][B][-1], "{:.1f}%"),
        ("Running events", M["running"]["count"][A], M["running"]["count"][B], "{:.0f}"),
        ("Team length (m)", sh[A]["avg"]["length"], sh[B]["avg"]["length"], "{:.1f}"),
        ("Team width (m)", sh[A]["avg"]["width"], sh[B]["avg"]["width"], "{:.1f}"),
        ("Area covered by the team (m²)", sh[A]["avg"]["hull"], sh[B]["avg"]["hull"], "{:.0f}"),
    ]
    S.card(fig, (0.05, 0.075, 0.60, 0.775))
    top, rh = 0.815, 0.0575
    for k, (lab, a, b, fmt) in enumerate(rows):
        y = top - k * rh
        fig.text(0.07, y, lab, fontsize=10.5, va="center")
        tot = (a + b) or 1
        bx, bw = 0.335, 0.29
        wa = bw * 0.5 * a / max(a, b, 1e-9); wb = bw * 0.5 * b / max(a, b, 1e-9)
        # centred diverging bars so the eye compares lengths
        mid = bx + bw / 2
        fig.add_artist(patches.Rectangle((mid - wa, y - 0.012), wa, 0.024, transform=fig.transFigure, fc=ctx.cA, ec="none"))
        fig.add_artist(patches.Rectangle((mid, y - 0.012), wb, 0.024, transform=fig.transFigure, fc=ctx.cB, ec="none"))
        fig.text(mid - wa - 0.006, y, fmt.format(a), fontsize=10, va="center", ha="right", fontweight="bold", color=ctx.cA)
        fig.text(mid + wb + 0.006, y, fmt.format(b), fontsize=10, va="center", ha="left", fontweight="bold", color=ctx.cB)
        if k < len(rows) - 1:
            fig.add_artist(plt_line(0.065, y - rh / 2, 0.635, y - rh / 2))
    # reading panel
    S.card(fig, (0.68, 0.075, 0.27, 0.775))
    fig.text(0.695, 0.82, "What stands out", fontsize=13, fontweight="bold")
    notes = _headlines(ctx)
    fig.text(0.695, 0.775, "\n\n".join(_wrap(n, 37) for n in notes), fontsize=10.5, va="top", color=S.INK, linespacing=1.4)
    return fig


def plt_line(x0, y0, x1, y1, color=S.FAINT, lw=0.8):
    import matplotlib.pyplot as plt
    return plt.Line2D([x0, x1], [y0, y1], color=color, lw=lw)


def _wrap(s, n):
    import textwrap
    return "\n".join(textwrap.wrap(s, n))


def _headlines(ctx):
    M = ctx.M
    A, B = ctx.teams
    P = M["possession"]
    out = []
    lead = 0 if P["share"][0] >= P["share"][1] else 1
    out.append(f"{ctx.teams[lead]} had more of the ball: {P['share'][lead]:.0f}% of the possession that could be measured.")
    third = M["territory"]["ball_third_pct"]
    out.append(f"The ball spent {third[1]:.0f}% of the time in the middle third, {third[0]:.0f}% in the left end and {third[2]:.0f}% in the right end.")
    d = M["distance"]["per_player_m"]
    more = A if d[A] >= d[B] else B
    out.append(f"{more} covered more ground per player ({d[more]:,.0f} m, against {d[B if more == A else A]:,.0f} m).")
    r = M["running"]["count"]
    out.append(f"Running above {M['running_threshold']:.0f} km/h was spotted {r[A]} time{'s' if r[A] != 1 else ''} for {A} and {r[B]} for {B}.")
    sh = M["shape"]
    wide = A if sh[A]["avg"]["width"] >= sh[B]["avg"]["width"] else B
    out.append(f"{wide} played the wider shape ({sh[wide]['avg']['width']:.1f} m on average).")
    return out


# ======================================================================================== 3 possession
def possession(fr, ctx):
    import matplotlib.pyplot as plt
    M = ctx.M
    A, B = ctx.teams
    fig = fr.new("Possession", "Who had the ball, from the moments the ball was seen with a player.")
    _team_legend(fig, ctx)
    P = M["possession"]
    S.card(fig, (0.05, 0.075, 0.31, 0.775))
    ax = fig.add_axes([0.07, 0.40, 0.27, 0.36])
    if sum(P["share"]) > 0:
        ax.pie(P["share"], colors=ctx.col, startangle=90, counterclock=False, wedgeprops=dict(width=0.34, edgecolor="white", linewidth=2))
    else:
        ax.pie([1], colors=[S.UNKNOWN], wedgeprops=dict(width=0.34, edgecolor="white", linewidth=2))
    ax.text(0, 0.08, "possession", ha="center", fontsize=10, color=S.MUTED)
    ax.text(0, -0.14, f"{P['share'][0]:.0f} – {P['share'][1]:.0f}", ha="center", fontsize=20, fontweight="bold")
    y = 0.33
    for i, t in enumerate(ctx.teams):
        fig.add_artist(_dotp(fig, 0.085, y, 0.0075, ctx.col[i]))
        fig.text(0.10, y, t, fontsize=11, va="center")
        fig.text(0.34, y, f"{P['share'][i]:.0f}%  ·  {mmss(P['seconds'][i])}", fontsize=11, va="center", ha="right", fontweight="bold")
        y -= 0.05
    fig.text(0.07, 0.17, _wrap(f"Based on the {P['known_pct']:.0f}% of the game where the ball could be placed with a team.", 42), fontsize=8.5, color=S.MUTED, va="top")
    # stacked minute bars
    S.card(fig, (0.38, 0.075, 0.57, 0.775))
    fig.text(0.40, 0.82, "Possession, minute by minute", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.42, 0.14, 0.51, 0.62]); _style_axes(ax, "seconds of that minute")
    tl = np.array(M["possession_minutes"])
    x = np.arange(len(tl))
    ax.bar(x, tl[:, 0], color=ctx.cA, width=0.72, label=A)
    ax.bar(x, tl[:, 1], bottom=tl[:, 0], color=ctx.cB, width=0.72, label=B)
    ax.bar(x, tl[:, 2], bottom=tl[:, 0] + tl[:, 1], color=S.UNKNOWN, width=0.72, label="ball not placed")
    ax.set_xticks(x); ax.set_xticklabels([f"{i}" for i in x]); ax.set_xlabel("minute of the game", fontsize=9)
    ax.set_ylim(0, 60)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=9)
    return fig


# ======================================================================================== 4 possession detail
def possession_detail(fr, ctx):
    M = ctx.M
    A, B = ctx.teams
    fig = fr.new("Possession in detail", "How long each team kept the ball, and how it changed hands.")
    _team_legend(fig, ctx)
    spells = M["spells"]
    # histogram of spell lengths
    S.card(fig, (0.05, 0.075, 0.44, 0.775))
    fig.text(0.07, 0.82, "How long a team kept the ball", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.09, 0.14, 0.38, 0.60]); _style_axes(ax, "number of spells")
    bins = [(0, 2, "under 2 s"), (2, 5, "2–5 s"), (5, 10, "5–10 s"), (10, 1e9, "over 10 s")]
    w = 0.36
    for i, t in enumerate(ctx.teams):
        c = [sum(1 for s in spells if s["team"] == i and lo <= s["len"] < hi) for lo, hi, _ in bins]
        xs = np.arange(len(bins)) + (i - 0.5) * w
        ax.bar(xs, c, width=w, color=ctx.col[i])
        for xv, cv in zip(xs, c):
            ax.text(xv, cv + 0.6, str(cv), ha="center", fontsize=8.5, color=S.MUTED)
    ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels([b[2] for b in bins])
    # share by block
    S.card(fig, (0.51, 0.40, 0.44, 0.45))
    fig.text(0.53, 0.82, "Share of possession through the game", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.56, 0.45, 0.37, 0.30]); _style_axes(ax, "% of ball time")
    pb = M["possession_blocks"]
    xs = np.arange(len(pb))
    for i, t in enumerate(ctx.teams):
        ax.plot(xs, [b["share"][i] for b in pb], marker="o", color=ctx.col[i], lw=2.4, ms=5)
    ax.axhline(50, color=S.FAINT, lw=1, ls="--")
    ax.set_ylim(0, 100); ax.set_xticks(xs); ax.set_xticklabels(_block_labels(M)); ax.set_xlabel(f"time into the game (every {M['block_s'] // 60} min)", fontsize=9)
    # changes of possession
    S.card(fig, (0.51, 0.075, 0.44, 0.30))
    fig.text(0.53, 0.345, "Ball changing hands", fontsize=12, fontweight="bold")
    won = M["possession"]["won_from_opponent"]; P = M["possession"]
    for i, t in enumerate(ctx.teams):
        x = 0.53 + i * 0.21
        fig.add_artist(_dotp(fig, x + 0.005, 0.30, 0.0065, ctx.col[i]))
        fig.text(x + 0.016, 0.30, t, fontsize=10, va="center", fontweight="bold")
        fig.text(x, 0.225, f"{won[i]}", fontsize=28, fontweight="bold", color=ctx.col[i])
        fig.text(x, 0.19, "times won the ball", fontsize=9, color=S.MUTED)
        ls = P["longest_spell"][i]
        fig.text(x, 0.135, f"{ls['len'] if isinstance(ls, dict) else (ls[2] if ls else 0):.0f} s", fontsize=18, fontweight="bold")
        fig.text(x, 0.105, "longest spell on the ball", fontsize=9, color=S.MUTED)
    return fig


# ======================================================================================== 5 territory
def territory(fr, ctx):
    M = ctx.M
    A, B = ctx.teams
    T = M["territory"]
    fig = fr.new("Territory", "Where the ball was, and who had it there. Left and right are as seen on the match video.")
    _team_legend(fig, ctx)
    S.card(fig, (0.05, 0.075, 0.57, 0.775))
    fig.text(0.07, 0.82, "Where the ball spent its time", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.07, 0.33, 0.53, 0.44])
    S.draw_pitch(ax, fill="#f6f8f4")
    if M["heat_ball"]:
        S.pitch_heat(ax, M["heat_ball"], "#d94a1e")
    for lo, hi in T["thirds"][:2]:
        ax.plot([hi, hi], [-HALF_W, HALF_W], color="#9aa59d", lw=0.8, ls=(0, (4, 4)), zorder=3)
    for k, (lo, hi) in enumerate(T["thirds"]):
        ax.text((lo + hi) / 2, HALF_W + 1.9, f"{T['ball_third_pct'][k]:.0f}%", ha="center", fontsize=13, fontweight="bold", color=S.INK)
    ax.set_ylim(-HALF_W - 1, HALF_W + 3)
    fig.text(0.07, 0.295, "Darker = more time. The percentages are the share of ball time in each third of the pitch.", fontsize=8.5, color=S.MUTED)
    fig.text(0.07, 0.255, "Where the game was played, through the game", fontsize=10, fontweight="bold")
    # ball position through game
    ax2 = fig.add_axes([0.10, 0.125, 0.47, 0.10]); _style_axes(ax2, None, grid="y")
    xs = np.arange(len(M["blocks"]))
    vals = [v if v is not None else np.nan for v in T.get("ball_x_blocks", [None] * len(M["blocks"]))]
    ax2.fill_between(xs, 0, vals, color="#d94a1e", alpha=0.25); ax2.plot(xs, vals, color="#d94a1e", marker="o", ms=4, lw=2)
    ax2.axhline(0, color=S.FAINT)
    ax2.set_ylim(-HALF_L / 2, HALF_L / 2); ax2.set_xticks(xs); ax2.set_xticklabels(_block_labels(M), fontsize=8)
    ax2.set_yticks([-10, 0, 10]); ax2.set_yticklabels(["left", "mid", "right"], fontsize=8)
    # who held the ball in each third
    S.card(fig, (0.64, 0.075, 0.31, 0.775))
    fig.text(0.655, 0.82, "Who had it, third by third", fontsize=12, fontweight="bold")
    tp = T.get("possession_by_third")
    names = ["Left third", "Middle third", "Right third"]
    for k in range(3):
        y = 0.70 - k * 0.19
        fig.text(0.655, y + 0.05, names[k], fontsize=10.5, fontweight="bold")
        a, b = tp[k] if tp else (0, 0)
        tot = (a + b) or 1
        S._split = None
        _split_bar(fig, (0.655, y - 0.02, 0.275, 0.05), a / tot * 100, b / tot * 100, ctx.cA, ctx.cB)
    return fig


# ======================================================================================== 6 team shape
def team_shape(fr, ctx):
    M = ctx.M
    fig = fr.new("Team shape", "Where each team's players spent their time. Darker = more time.")
    for i, t in enumerate(ctx.teams):
        x0 = 0.05 + i * 0.475
        S.card(fig, (x0, 0.075, 0.425, 0.775))
        fig.add_artist(_dotp(fig, x0 + 0.02, 0.815, 0.0075, ctx.col[i]))
        fig.text(x0 + 0.035, 0.815, t, fontsize=14, fontweight="bold", va="center")
        ax = fig.add_axes([x0 + 0.01, 0.40, 0.405, 0.38])
        S.draw_pitch(ax, fill="#f6f8f4")
        S.pitch_heat(ax, M["heat_team"][t], ctx.col[i])
        a = M["shape"][t]["avg"]
        ax.plot([a["cx"]], [a["cy"]], marker="o", ms=11, mfc="white", mec=ctx.col[i], mew=3, zorder=6)
        ax.annotate("average position of the team", (a["cx"], a["cy"]), xytext=(0, -28), textcoords="offset points", ha="center", fontsize=8, color=S.INK,
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.8), zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        stats = [("Length", f"{a['length']:.1f} m", "front to back"), ("Width", f"{a['width']:.1f} m", "side to side"),
                 ("Area", f"{a['hull']:.0f} m²", "covered by the team"), ("Spread", f"{a['spread']:.1f} m", "from the team centre")]
        for k, (lab, val, sub) in enumerate(stats):
            xx = x0 + 0.02 + (k % 2) * 0.20
            yy = 0.30 - (k // 2) * 0.12
            fig.text(xx, yy, val, fontsize=22, fontweight="bold", color=ctx.col[i])
            fig.text(xx, yy - 0.033, f"{lab} — {sub}", fontsize=8.5, color=S.MUTED, va="top")
    return fig


# ======================================================================================== 7 shape over time
def shape_over_time(fr, ctx):
    M = ctx.M
    fig = fr.new("Shape through the game", "How compact each team was, and how far apart the two teams were (30-second averages).")
    _team_legend(fig, ctx)
    panels = [("length", "Team length (m)", "front to back"), ("width", "Team width (m)", "side to side"),
              ("cx", "Position along the pitch (m)", "left end to right end"), ("sep", "Distance between the teams (m)", "centre to centre")]
    for k, (key, title, sub) in enumerate(panels):
        col, row = k % 2, k // 2
        x0, y0 = 0.05 + col * 0.475, 0.50 - row * 0.385
        S.card(fig, (x0, y0 - 0.03, 0.425, 0.37))
        fig.text(x0 + 0.015, y0 + 0.30, title, fontsize=12, fontweight="bold")
        fig.text(x0 + 0.015, y0 + 0.268, sub, fontsize=8.5, color=S.MUTED)
        ax = fig.add_axes([x0 + 0.045, y0 + 0.015, 0.36, 0.21]); _style_axes(ax)
        if key == "sep":
            sp = M["shape"].get("separation")
            if sp:
                ax.plot(np.array(sp["t"]) - M["start"], sp["v"], color=S.INK, lw=2)
        else:
            for i, t in enumerate(ctx.teams):
                s = M["shape"].get(t)
                if s:
                    ax.plot(np.array(s["t"]) - M["start"], [np.nan if v is None else v for v in s[key]], color=ctx.col[i], lw=2.2)
        ax.set_xlim(0, M["end"] - M["start"])
        ticks = np.arange(0, M["end"] - M["start"] + 1, 120)
        ax.set_xticks(ticks); ax.set_xticklabels([mmss(v) for v in ticks], fontsize=8)
        if key == "cx":
            ax.axhline(0, color=S.FAINT)
    return fig


# ======================================================================================== 8 distance
def distance(fr, ctx):
    M = ctx.M
    A, B = ctx.teams
    D = M["distance"]
    fig = fr.new("Distance covered", "Ground covered by the average player, in each stretch of the game.")
    _team_legend(fig, ctx)
    S.card(fig, (0.05, 0.075, 0.28, 0.775))
    fig.text(0.065, 0.82, "Whole game", fontsize=12, fontweight="bold")
    for i, t in enumerate(ctx.teams):
        y = 0.70 - i * 0.30
        fig.add_artist(_dotp(fig, 0.072, y + 0.045, 0.0065, ctx.col[i]))
        fig.text(0.085, y + 0.045, t, fontsize=10.5, fontweight="bold", va="center")
        fig.text(0.065, y - 0.02, f"{D['per_player_m'][t]:,.0f} m", fontsize=30, fontweight="bold", color=ctx.col[i])
        fig.text(0.065, y - 0.055, "per player", fontsize=9, color=S.MUTED)
        fig.text(0.065, y - 0.115, f"{D['team_total_m'][t] / 1000:.2f} km", fontsize=16, fontweight="bold")
        fig.text(0.065, y - 0.14, "whole team (5 players)", fontsize=9, color=S.MUTED)
    S.card(fig, (0.35, 0.075, 0.60, 0.775))
    fig.text(0.365, 0.82, "Distance per player through the game", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.40, 0.14, 0.53, 0.62]); _style_axes(ax, f"metres per {M['block_s'] // 60} minutes")
    n = len(M["blocks"]); xs = np.arange(n); w = 0.38
    for i, t in enumerate(ctx.teams):
        v = D["blocks"][t]
        ax.bar(xs + (i - 0.5) * w, v, width=w, color=ctx.col[i])
    ax.set_xticks(xs); ax.set_xticklabels(_block_labels(M)); ax.set_xlabel("time into the game", fontsize=9)
    ax.axhline(0, color=S.FAINT)
    return fig


# ======================================================================================== 9 speed
def speed(fr, ctx):
    M = ctx.M
    A, B = ctx.teams
    Sd = M["speed"]
    fig = fr.new("Speed and intensity", "How fast the players moved, from standing still to running.")
    _team_legend(fig, ctx)
    edges = Sd["band_edges"]
    names = ["Standing", "Walking", "Brisk walk", "Jogging pace", "Running"]
    ranges = [f"under {edges[1]:.0f} km/h", f"{edges[1]:.0f}–{edges[2]:.0f} km/h", f"{edges[2]:.0f}–{edges[3]:.0f} km/h", f"{edges[3]:.0f}–{edges[4]:.0f} km/h", f"over {edges[4]:.0f} km/h"]
    shades = [0.80, 0.6, 0.4, 0.2, 0.0]
    S.card(fig, (0.05, 0.31, 0.57, 0.54))
    fig.text(0.065, 0.82, "Share of time at each speed", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.08, 0.42, 0.52, 0.36]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(-0.7, 2.25)
    for i, t in enumerate(ctx.teams):
        y = 1.0 - i * 1.15
        left = 0
        for b, (v, sh) in enumerate(zip(Sd["bands"][t], shades)):
            ax.barh(y, v, left=left, height=0.8, color=S.tint(ctx.col[i], sh), ec="white", lw=1.5)
            if v >= 4:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white" if sh < 0.45 else S.INK, fontweight="bold")
            left += v
        ax.text(0, y + 0.52, t, ha="left", va="bottom", fontsize=10, fontweight="bold", color=ctx.col[i])
    ax.set_xlim(0, 100)
    # band key
    for b in range(5):
        x = 0.075 + b * 0.105
        fig.add_artist(patches.Rectangle((x, 0.355), 0.012, 0.022, transform=fig.transFigure, fc=S.tint(S.INK, shades[b] * 0.9), ec="none"))
        fig.text(x + 0.016, 0.366, f"{names[b]}\n{ranges[b]}", fontsize=7.6, color=S.MUTED, va="center", linespacing=1.25)
    # speed through the game
    S.card(fig, (0.64, 0.31, 0.31, 0.54))
    fig.text(0.655, 0.82, "Average speed (km/h)", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.68, 0.38, 0.25, 0.38]); _style_axes(ax)
    xs = np.arange(len(M["blocks"]))
    for i, t in enumerate(ctx.teams):
        ax.plot(xs, [np.nan if v is None else v for v in Sd["blocks"][t]], marker="o", color=ctx.col[i], lw=2.2, ms=4)
    ax.set_xticks(xs); ax.set_xticklabels(_block_labels(M), fontsize=7.5); ax.set_ylim(0, None)
    # top speeds
    S.card(fig, (0.05, 0.075, 0.90, 0.205))
    fig.text(0.065, 0.255, "Speed figures", fontsize=12, fontweight="bold")
    cols = [("Average speed", "avg", "{:.1f} km/h"), ("Fast end (top 5% of readings)", "p95", "{:.1f} km/h"), ("Highest speed measured", "max", "{:.1f} km/h")]
    for k, (lab, key, fmt) in enumerate(cols):
        x = 0.065 + k * 0.30
        fig.text(x, 0.222, lab, fontsize=9.5, color=S.MUTED)
        for i, t in enumerate(ctx.teams):
            fig.add_artist(_dotp(fig, x + 0.005, 0.175 - i * 0.055, 0.0065, ctx.col[i]))
            fig.text(x + 0.018, 0.175 - i * 0.055, fmt.format(Sd[key][t]), fontsize=19, fontweight="bold", color=ctx.col[i], va="center")
    return fig


# ======================================================================================== 10 running
def running(fr, ctx):
    M = ctx.M
    R = M["running"]
    fig = fr.new("Running check", f"Spells of running above {M['running_threshold']:.0f} km/h (walking football is a walking game).")
    _team_legend(fig, ctx)
    ev = R["events"][:14]
    tab_h = min(0.50, 0.15 + len(ev) * 0.0295)
    top0 = 0.075 + tab_h + 0.02
    S.card(fig, (0.05, top0, 0.90, 0.85 - top0))
    fig.text(0.065, 0.82, "When running was spotted", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.20, top0 + 0.05, 0.73, 0.85 - top0 - 0.14]); _style_axes(ax, None, grid="x")
    for i, t in enumerate(ctx.teams):
        mine = [e for e in R["events"] if e["team"] == t]
        ax.scatter([e["t"] - M["start"] for e in mine], [1 - i] * len(mine), s=[40 + (e["max"] - M["running_threshold"]) * 18 for e in mine], color=ctx.col[i], alpha=0.85, zorder=3)
    ax.set_yticks([1, 0]); ax.set_yticklabels([ctx.teams[0], ctx.teams[1]], fontsize=9); ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(0, M["end"] - M["start"]); ticks = np.arange(0, M["end"] - M["start"] + 1, 60)
    ax.set_xticks(ticks); ax.set_xticklabels([mmss(v) for v in ticks], fontsize=8)
    S.card(fig, (0.05, 0.075, 0.90, tab_h))
    ytop = 0.075 + tab_h
    fig.text(0.065, ytop - 0.03, "Every spell, with a link to that moment in the video", fontsize=12, fontweight="bold")
    hdr_y = ytop - 0.075
    for x, h in ((0.07, "Time"), (0.17, "Team"), (0.52, "Top speed"), (0.64, "Lasted")):
        fig.text(x, hdr_y, h, fontsize=9, color=S.MUTED, fontweight="bold")
    if not ev:
        fig.text(0.07, 0.45, "No running spells were detected in this game.", fontsize=11, color=S.MUTED)
    for k, e in enumerate(ev):
        y = hdr_y - 0.03 - k * 0.0295
        url = ctx.link(e["t"])
        tt = fig.text(0.07, y, mmss(e["t"] - M["start"]), fontsize=10, va="center", color=S.BRAND if url else S.INK, fontweight="bold")
        if url:
            tt.set_url(url)
        fig.add_artist(_dotp(fig, 0.178, y, 0.005, ctx.colour[e["team"]]))
        fig.text(0.19, y, e["team"], fontsize=10, va="center")
        fig.text(0.52, y, f"{e['max']:.1f} km/h", fontsize=10, va="center")
        fig.text(0.64, y, f"{e['dur']:.1f} s", fontsize=10, va="center")
    if len(R["events"]) > 14:
        fig.text(0.07, 0.09, f"+ {len(R['events']) - 14} more spells (all of them are in the YouTube description file).", fontsize=8.5, color=S.MUTED)
    # per-team counts
    for i, t in enumerate(ctx.teams):
        x = 0.74
        y = ytop - 0.10 - i * 0.11
        fig.text(x, y, f"{R['count'][t]}", fontsize=30, fontweight="bold", color=ctx.col[i], va="center")
        fig.text(x + 0.05, y, f"running spells\n{t}", fontsize=9, color=S.MUTED, va="center", linespacing=1.3)
    return fig


# ======================================================================================== 11 key moments
def key_moments(fr, ctx):
    M = ctx.M
    fig = fr.new("Key moments", "Moments from the game worth watching again" + (" — click a time to open the video." if ctx.link_base else "."))
    items = []
    for i, t in enumerate(ctx.teams):
        ls = M["possession"]["longest_spell"][i]
        if ls:
            items.append((ls[1], i, "Longest spell on the ball", f"{ls[2]:.0f} seconds of unbroken possession"))
    for i, t in enumerate(ctx.teams):
        ev = [e for e in M["running"]["events"] if e["team"] == t]
        if ev:
            e = max(ev, key=lambda z: z["max"])
            items.append((e["t"], i, "Fastest burst", f"{e['max']:.1f} km/h for {e['dur']:.1f} s"))
    pb = M["possession_blocks"]
    for i, t in enumerate(ctx.teams):
        k = int(np.argmax([b["share"][i] for b in pb]))
        if pb[k]["share"][i] > 50:
            items.append((M["blocks"][k][0], i, "Strongest spell of possession", f"{pb[k]['share'][i]:.0f}% of the ball between {mmss(M['blocks'][k][0] - M['start'])} and {mmss(M['blocks'][k][1] - M['start'])}"))
    for i, t in enumerate(ctx.teams):
        s = M["shape"].get(t)
        if s:
            vals = [v if v is not None else 1e9 for v in s["hull"]]
            k = int(np.argmin(vals))
            items.append((s["t"][k], i, "Tightest shape", f"the team squeezed into {vals[k]:.0f} m² (average {s['avg']['hull']:.0f} m²)"))
    for i, t in enumerate(ctx.teams):
        v = [x if x is not None else -1 for x in M["speed"]["blocks"][t]]
        k = int(np.argmax(v))
        items.append((M["blocks"][k][0], i, "Most energetic stretch", f"average {v[k]:.1f} km/h between {mmss(M['blocks'][k][0] - M['start'])} and {mmss(M['blocks'][k][1] - M['start'])}"))
    items.sort(key=lambda z: z[0])
    cols = 2
    per = math.ceil(len(items) / cols)
    for n, (t, i, head, desc) in enumerate(items):
        c, r = divmod(n, per)
        x = 0.05 + c * 0.465; y = 0.755 - r * 0.142
        S.card(fig, (x, y - 0.02, 0.445, 0.122))
        fig.add_artist(patches.Rectangle((x, y - 0.02), 0.007, 0.122, transform=fig.transFigure, fc=ctx.col[i], ec="none"))
        url = ctx.link(t)
        tt = fig.text(x + 0.02, y + 0.035, mmss(t - M["start"]), fontsize=20, fontweight="bold", color=S.BRAND if url else S.INK, va="center")
        if url:
            tt.set_url(url)
        fig.text(x + 0.10, y + 0.057, head, fontsize=11, fontweight="bold", va="center")
        fig.text(x + 0.10, y + 0.027, ctx.teams[i], fontsize=9.5, color=ctx.col[i], va="center", fontweight="bold")
        fig.text(x + 0.10, y + 0.0, _wrap(desc, 52), fontsize=9, color=S.MUTED, va="center")
    return fig


# ======================================================================================== 12 about
def about(fr, ctx):
    M = ctx.M
    fig = fr.new("About the numbers", "What each figure means and how it was worked out.")
    c = M["coverage"]
    left = [
        ("Possession", f"The ball counts as held by a team when it is seen within {2.0:.0f} m of one of its players and no opponent is almost as close. That team keeps it until the other side is seen with it, or for up to 5 seconds if the ball is lost from view. Percentages use only the time the ball could be placed with a team."),
        ("Territory", "Pitch thirds measured along the length of the 36 m pitch. Left and right follow the match video; they do not mean attacking or defending ends."),
        ("Team shape", "From the positions of the players detected in each frame (at least 3 per team). Length = front to back, width = side to side, area = the space inside the outline of the players."),
        ("Distance", "Average speed of the detected players multiplied by the time, for each stretch of the game. Because players are not always detected, it is shown per player; the team figure is five times that."),
    ]
    right = [
        ("Speed", f"Fitted from each player's path, in km/h. Readings above 25 km/h are treated as glitches and left out. Bands: under {M['speed']['band_edges'][1]:.0f}, {M['speed']['band_edges'][1]:.0f}–{M['speed']['band_edges'][2]:.0f}, {M['speed']['band_edges'][2]:.0f}–{M['speed']['band_edges'][3]:.0f}, {M['speed']['band_edges'][3]:.0f}–{M['speed']['band_edges'][4]:.0f} and over {M['speed']['band_edges'][4]:.0f} km/h."),
        ("Running spells", f"A player above {M['running_threshold']:.0f} km/h for at least half a second. The limit is the 'running speed' on the game page."),
        ("Players", "Players are not named: every figure is for the team as a whole."),
        ("This game", f"{c['frames']:,} frames analysed ({M['detection_fps']:.0f} per second) over {mmss(M['duration'])}. The ball was seen in {c['ball_pct']:.0f}% of them. About {c['players_per_frame'][ctx.teams[0]]:.1f} of {ctx.teams[0]}'s and {c['players_per_frame'][ctx.teams[1]]:.1f} of {ctx.teams[1]}'s 5 players were detected per frame. How reliable the detections are is covered in the separate accuracy report."),
    ]
    for col, items in enumerate((left, right)):
        x0 = 0.05 + col * 0.475
        S.card(fig, (x0, 0.075, 0.425, 0.775))
        y = 0.81
        for head, body in items:
            fig.text(x0 + 0.02, y, head, fontsize=11.5, fontweight="bold", color=S.BRAND, va="top")
            w = _wrap(body, 64)
            fig.text(x0 + 0.02, y - 0.032, w, fontsize=9.5, va="top", linespacing=1.4)
            y -= 0.032 + 0.027 * (w.count("\n") + 1) + 0.045
    return fig


PAGES = [cover, summary, possession, possession_detail, territory, team_shape, shape_over_time, distance, speed, running, key_moments, about]
