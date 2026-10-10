"""The pages of the opposition report. Each report is about one team and written for the coach of the team that plays
against it. Every page function takes the page frame, a context and the team the report is about."""
import math
import textwrap

import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt

from ..match_report import style as S
from ..match_report.metrics import HALF_L, HALF_W, mmss
from ..match_report.pages import _dotp, _style_axes, plt_line
from . import scout as SC

GOOD = "#0b7a3e"
BAD = "#c2410c"


class Ctx:
    def __init__(self, scoutd, record, colours, link_base=None, created=""):
        self.S, self.rec, self.link_base, self.created = scoutd, record, link_base, created
        self.M = scoutd["M"]
        self.teams = scoutd["teams"]
        self.colour = {self.teams[0]: colours[0], self.teams[1]: colours[1]}

    def link(self, seconds):
        if not self.link_base:
            return None
        sep = "&" if "?" in self.link_base else "?"
        return f"{self.link_base}{sep}t={int(max(seconds, 0))}s"

    def other(self, tm):
        return self.teams[1 - self.teams.index(tm)]


def _wrap(s, n):
    return "\n".join(textwrap.wrap(s, n))


def _t(ctx, tm):
    return ctx.S["per_team"][tm]


def _frame_note(fig, text, y=0.062):
    fig.text(0.05, y, text, fontsize=8, color=S.MUTED, va="center")


def _pitch_axes(fig, rect, with_arrow=True):
    ax = fig.add_axes(rect)
    S.draw_pitch(ax)
    ax.set_ylim(-HALF_W - 3.2, HALF_W + 1)
    if with_arrow:
        ax.annotate("", xy=(9, -HALF_W - 2.2), xytext=(-9, -HALF_W - 2.2), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.2))
        ax.text(0, -HALF_W - 2.2, "  attacking direction  ", ha="center", va="center", fontsize=7.5, color=S.MUTED,
                bbox=dict(fc=S.BG, ec="none", pad=1))
    for u in (-HALF_L / 3, HALF_L / 3):
        ax.plot([u, u], [-HALF_W, HALF_W], color=S.FAINT, lw=0.8, ls=(0, (3, 3)), zorder=2)
    return ax


def _stat_rows(fig, x, y0, rows, colour, ocol, rh=0.062, width=0.22):
    """rows: (label, mine, theirs, fmt). Draws label, the team's value and the other team's value."""
    chars = max(int((width - 0.115) / 0.0066), 10)
    fig.text(x + width - 0.087, y0 + 0.032, "this team", fontsize=8, color=colour, ha="center", fontweight="bold")
    fig.text(x + width - 0.017, y0 + 0.032, "opponent", fontsize=8, color=S.MUTED, ha="center")
    for k, (lab, a, b, fmt) in enumerate(rows):
        y = y0 - k * rh
        fig.text(x, y, _wrap(lab, chars), fontsize=9.5, va="center", linespacing=1.15)
        fig.text(x + width - 0.087, y, fmt.format(a) if a is not None else "n/a", fontsize=11, fontweight="bold", color=colour, ha="center", va="center")
        fig.text(x + width - 0.017, y, fmt.format(b) if b is not None else "n/a", fontsize=10, color=S.MUTED, ha="center", va="center")
        if k < len(rows) - 1:
            fig.add_artist(plt_line(x, y - rh / 2, x + width, y - rh / 2))


def _header(fr, ctx, tm, page_title, intro):
    fig = fr.new(f"{tm}: {page_title}", intro)
    return fig


# ======================================================================================== page 1: the brief
def brief(fr, ctx, tm):
    S_ = ctx.S
    opp = ctx.other(tm)
    col = ctx.colour[tm]
    fig = fr.new(f"Opposition report: {tm}", f"How {tm} play and where to beat them. Written for the {opp} coach. {ctx.rec.get('date', '')}")
    # style strip
    lines = SC.style_lines(S_, tm)
    n = max(len(lines), 1)
    w = (0.90 - 0.012 * (n - 1)) / n
    for k, (lab, head, ev) in enumerate(lines):
        x = 0.05 + k * (w + 0.012)
        S.card(fig, (x, 0.685, w, 0.168))
        fig.add_artist(patches.Rectangle((x, 0.685), 0.005, 0.168, transform=fig.transFigure, fc=col, ec="none"))
        fig.text(x + 0.016, 0.833, lab.upper(), fontsize=7.5, color=S.MUTED, fontweight="bold", va="center")
        fig.text(x + 0.016, 0.812, _wrap(head, 24), fontsize=11.5, fontweight="bold", va="top", linespacing=1.15)
        fig.text(x + 0.016, 0.742, _wrap(ev, 30), fontsize=8, color=S.MUTED, va="top", linespacing=1.25)
    items = S_["items"][tm]
    strong = [r for r in items if r["kind"] == "strong"][:4]
    weak = [r for r in items if r["kind"] == "weak"][:4]
    for x, title, colr, rows, empty in ((0.05, "Strong points: watch out for these", GOOD, strong, "Nothing clearly better than the opponent in this game."),
                                        (0.505, "Weak points: look to exploit these", BAD, weak, "Nothing clearly weaker than the opponent in this game.")):
        S.card(fig, (x, 0.285, 0.445, 0.385))
        fig.text(x + 0.018, 0.645, title, fontsize=12, fontweight="bold", color=colr, va="center")
        y = 0.605
        if not rows:
            fig.text(x + 0.018, y, empty, fontsize=9.5, color=S.MUTED, va="center")
        for r in rows:
            txt = _wrap(r["text"], 62)
            nl = txt.count("\n") + 1
            fig.add_artist(patches.Rectangle((x + 0.018, y - 0.0185 * nl + 0.002), 0.004, 0.0195 * nl, transform=fig.transFigure, fc=colr, ec="none"))
            fig.text(x + 0.032, y + 0.003, txt, fontsize=9.8, va="top", linespacing=1.25)
            y -= 0.0205 * nl + 0.030
    # game plan
    S.card(fig, (0.05, 0.075, 0.90, 0.195))
    fig.text(0.068, 0.247, f"Game plan against {tm}", fontsize=12, fontweight="bold", va="center", color=S.BRAND)
    wk, st = SC.plan(S_, tm)
    tips = [r["tip"] for r in wk] + [r["tip"] for r in st]
    if not tips:
        tips = ["The two teams were evenly matched on the measures in this report. Nothing stands out to target."]
    half = math.ceil(len(tips) / 2)
    for c, group in enumerate((tips[:half], tips[half:])):
        y = 0.213
        for tip in group:
            txt = _wrap(tip, 72)
            fig.text(0.07 + c * 0.45, y, "•", fontsize=11, color=S.BRAND, va="top")
            fig.text(0.082 + c * 0.45, y, txt, fontsize=9.3, va="top", linespacing=1.25)
            y -= 0.0185 * (txt.count("\n") + 1) + 0.020
    fig.text(0.068, 0.092, f"Strong and weak points are measured against {opp} in this one game, not against other teams. Players are not named.", fontsize=8, color=S.MUTED, va="center")
    return fig


# ======================================================================================== page 2: with the ball
def with_ball(fr, ctx, tm):
    S_ = ctx.S
    opp = ctx.other(tm)
    me, op = _t(ctx, tm), _t(ctx, opp)
    col, ocol = ctx.colour[tm], ctx.colour[opp]
    fig = fr.new(f"{tm}: with the ball", "How they use the ball: where they keep it, which side they favour, how far they get.")
    # pitch heat
    S.card(fig, (0.05, 0.075, 0.36, 0.775))
    fig.text(0.068, 0.825, "Where they have the ball", fontsize=12, fontweight="bold", va="center")
    ax = _pitch_axes(fig, [0.06, 0.40, 0.34, 0.37])
    if me["heat_ball_with"]:
        S.pitch_heat(ax, me["heat_ball_with"], col)
    else:
        ax.text(0, 0, "not enough ball sightings", ha="center", color=S.MUTED, fontsize=9)
    ax.text(-HALF_L + 0.6, HALF_W - 1.2, "own goal", fontsize=7.5, color=S.MUTED)
    ax.text(HALF_L - 0.6, HALF_W - 1.2, "goal they attack", fontsize=7.5, color=S.MUTED, ha="right")
    # left / centre / right
    fig.text(0.068, 0.335, "Which side they play down (their left and right)", fontsize=10.5, fontweight="bold", va="center")
    for k, (lab, a, b) in enumerate(zip(("Left", "Centre", "Right"), me["flank_with"] or [0, 0, 0], op["flank_with"] or [0, 0, 0])):
        y = 0.285 - k * 0.075
        fig.text(0.07, y, lab, fontsize=9.5, va="center")
        for j, (val, c, nm) in enumerate(((a, col, tm), (b, ocol, opp))):
            fig.add_artist(patches.Rectangle((0.14, y + 0.004 - j * 0.026), 0.24 * val / 60, 0.021, transform=fig.transFigure, fc=c, ec="none", alpha=1 if j == 0 else 0.55))
            fig.text(0.14 + 0.24 * val / 60 + 0.006, y + 0.0145 - j * 0.026, f"{val:.0f}%", fontsize=8.5, va="center", color=S.MUTED if j else S.INK, fontweight="normal" if j else "bold")
    # stats
    S.card(fig, (0.43, 0.075, 0.25, 0.775))
    fig.text(0.447, 0.825, "Getting the ball forward", fontsize=12, fontweight="bold", va="center")
    rows = [("Possession", me["share"], op["share"], "{:.0f}%"),
            ("Average spell on the ball", me["avg_spell"], op["avg_spell"], "{:.1f} s"),
            ("Spells that reach the final third", me["final_third_pct"], op["final_third_pct"], "{:.0f}%"),
            ("Spells that reach the opponent's half", me["half_pct"], op["half_pct"], "{:.0f}%"),
            ("Fast breaks", me["fast_breaks"], op["fast_breaks"], "{:.0f}"),
            ("Ball lost in own third", me["lost_own_third"], op["lost_own_third"], "{:.0f}"),
            ("Width when attacking", me["shape_with"]["width"], op["shape_with"]["width"], "{:.1f} m")]
    _stat_rows(fig, 0.447, 0.755, rows, col, ocol, rh=0.093, width=0.225)
    # possession by block
    S.card(fig, (0.70, 0.075, 0.25, 0.775))
    fig.text(0.717, 0.825, "Share of the ball over time", fontsize=12, fontweight="bold", va="center")
    ax2 = fig.add_axes([0.735, 0.20, 0.19, 0.53]); _style_axes(ax2, None, grid="x")
    pb = me["possession_blocks"]
    M = ctx.M
    labels = [mmss(a - M["start"]) for a, _ in M["blocks"]]
    y = np.arange(len(pb))
    ax2.barh(y, pb, color=col, height=0.7)
    ax2.axvline(50, color=S.MUTED, lw=1, ls=(0, (3, 3)))
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8); ax2.invert_yaxis(); ax2.set_xlim(0, 100)
    ax2.set_xlabel("% of the ball (2-minute blocks)", fontsize=8.5)
    fig.text(0.717, 0.135, _wrap("Bars past the dashed line mean they had more of the ball in that block.", 36), fontsize=8, color=S.MUTED, va="top")
    return fig


# ======================================================================================== page 3: without the ball
def without_ball(fr, ctx, tm):
    S_ = ctx.S
    opp = ctx.other(tm)
    me, op = _t(ctx, tm), _t(ctx, opp)
    col, ocol = ctx.colour[tm], ctx.colour[opp]
    fig = fr.new(f"{tm}: without the ball", "How they defend: where they stand, how tight they are, and where they win the ball back.")
    S.card(fig, (0.05, 0.075, 0.34, 0.775))
    fig.text(0.068, 0.825, "When the opponent has the ball", fontsize=12, fontweight="bold", va="center")
    ax = _pitch_axes(fig, [0.055, 0.45, 0.33, 0.33])
    if me["heat_without"]:
        S.pitch_heat(ax, me["heat_without"], col)
    cu = me["shape_without"]["cu"]
    if cu is not None:
        ax.plot([cu, cu], [-HALF_W, HALF_W], color=col, lw=1.6, ls="--", zorder=5)
        ax.text(cu, HALF_W + 0.2, f"centre of the team", fontsize=7.5, color=col, ha="center", va="bottom")
    ax.text(-HALF_L + 0.6, HALF_W - 1.2, "own goal", fontsize=7.5, color=S.MUTED)
    txt = "n/a"
    if cu is not None:
        txt = (f"Their block sits {HALF_L + cu:.0f} m from their own goal line on average, "
               + ("so they defend high up the pitch." if cu > 1 else "so they defend around halfway." if cu > -3 else "so they defend deep, close to their own goal."))
    fig.text(0.068, 0.425, _wrap(txt, 52), fontsize=9.5, va="top", linespacing=1.3)
    fig.text(0.068, 0.355, _wrap("The dashed line is the middle of the five players. Further right means a higher defensive line.", 56), fontsize=8, color=S.MUTED, va="top")
    hb = me.get("hull_without_blocks"); ho = op.get("hull_without_blocks")
    if hb and ho:
        fig.text(0.068, 0.285, "Area covered when defending (m²)", fontsize=9.5, fontweight="bold", va="center")
        axh = fig.add_axes([0.085, 0.11, 0.285, 0.14]); _style_axes(axh, None)
        xs = np.arange(len(hb))
        axh.bar(xs - 0.2, [0 if v is None else v for v in hb], width=0.38, color=col)
        axh.bar(xs + 0.2, [0 if v is None else v for v in ho], width=0.38, color=ocol, alpha=0.55)
        axh.set_xticks(xs); axh.set_xticklabels([mmss(a - ctx.M["start"]) for a, _ in ctx.M["blocks"]], fontsize=7)
    # where they win the ball
    S.card(fig, (0.405, 0.075, 0.285, 0.775))
    fig.text(0.422, 0.825, "Where they win the ball", fontsize=12, fontweight="bold", va="center")
    ax2 = _pitch_axes(fig, [0.41, 0.52, 0.275, 0.26])
    w = me["wins"]
    if w:
        ax2.scatter([a[1] for a in w], [a[2] for a in w], s=34, color=col, edgecolor="white", lw=0.8, zorder=6, alpha=0.9)
    own = sum(1 for a in w if a[1] <= -6); mid = sum(1 for a in w if -6 < a[1] < 6); att = sum(1 for a in w if a[1] >= 6)
    for x0, n in ((-12, own), (0, mid), (12, att)):
        ax2.text(x0, HALF_W - 1.4, str(n), ha="center", fontsize=11, fontweight="bold", color=S.INK, zorder=7,
                 bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.85))
    fig.text(0.422, 0.455, _wrap(f"{len(w)} times they took the ball off {opp}. Numbers show how many were won in their own third, the middle and the final third.", 40), fontsize=8.5, color=S.MUTED, va="top", linespacing=1.3)
    sp, sp2 = me["space"], op["space"]
    if sp is not None:
        fig.text(0.422, 0.33, _wrap(f"When {opp} have the ball, the nearest {tm} player is on average {sp:.1f} m from the ball carrier" + (f" ({opp} give theirs {sp2:.1f} m)." if sp2 is not None else "."), 40), fontsize=9.3, va="top", linespacing=1.3)
    # stats
    S.card(fig, (0.705, 0.075, 0.245, 0.775))
    fig.text(0.72, 0.825, "Defending", fontsize=12, fontweight="bold", va="center")
    rows = [("Area covered by the five", me["shape_without"]["hull"], op["shape_without"]["hull"], "{:.0f} m²"),
            ("Length of the team", me["shape_without"]["length"], op["shape_without"]["length"], "{:.1f} m"),
            ("Width of the team", me["shape_without"]["width"], op["shape_without"]["width"], "{:.1f} m"),
            ("Space given to the ball carrier", me["space"], op["space"], "{:.1f} m"),
            ("Ball won", me["won"], op["won"], "{:.0f}"),
            ("Ball won in opponent's half", me["won_opp_half"], op["won_opp_half"], "{:.0f}")]
    _stat_rows(fig, 0.72, 0.745, rows, col, ocol, rh=0.105, width=0.222)
    return fig


# ======================================================================================== page 4: energy and moments
def energy_moments(fr, ctx, tm):
    S_ = ctx.S
    M = ctx.M
    opp = ctx.other(tm)
    me, op = _t(ctx, tm), _t(ctx, opp)
    col, ocol = ctx.colour[tm], ctx.colour[opp]
    fig = fr.new(f"{tm}: energy and chances to hit them", "How hard they work through the game, and the moments when they were caught out." + (" Click a time to open the video." if ctx.link_base else ""))
    labels = [mmss(a - M["start"]) for a, _ in M["blocks"]]
    x = np.arange(len(labels))
    # distance
    S.card(fig, (0.05, 0.46, 0.44, 0.39))
    fig.text(0.068, 0.825, "Distance covered per player, by 2-minute block", fontsize=11, fontweight="bold", va="center")
    ax = fig.add_axes([0.085, 0.51, 0.38, 0.27]); _style_axes(ax, "metres")
    ax.bar(x - 0.2, me["dist_blocks"], width=0.38, color=col, label=tm)
    ax.bar(x + 0.2, op["dist_blocks"], width=0.38, color=ocol, alpha=0.55, label=opp)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    # speed + running
    S.card(fig, (0.05, 0.075, 0.44, 0.37))
    fig.text(0.068, 0.425, "Average speed, and running above the limit", fontsize=11, fontweight="bold", va="center")
    ax = fig.add_axes([0.085, 0.13, 0.38, 0.25]); _style_axes(ax, "km/h")
    for T, c, nm, a in ((me, col, tm, 1), (op, ocol, opp, 0.55)):
        v = [np.nan if q is None else q for q in T["speed_blocks"]]
        ax.plot(x, v, color=c, lw=2, alpha=a, marker="o", ms=3.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    rb = me["running_blocks"]
    for k, r in enumerate(rb):
        if r:
            ax.text(k, ax.get_ylim()[1], "▲" * min(r, 4), ha="center", va="top", fontsize=7, color=BAD)
    def _times(n):
        return "never" if n == 0 else "once" if n == 1 else f"{n} times"
    fig.text(0.07, 0.092, f"▲ marks blocks where {tm} ran above {M['running_threshold']:.0f} km/h ({_times(me['running'])} in all; {opp}: {_times(op['running'])}).", fontsize=8, color=S.MUTED, va="center")
    # moments
    S.card(fig, (0.51, 0.075, 0.44, 0.775))
    fig.text(0.528, 0.825, "When they lost the ball deep in their own half", fontsize=11.5, fontweight="bold", va="center")
    ax = _pitch_axes(fig, [0.52, 0.55, 0.41, 0.23], with_arrow=False)
    ls = me["losses"]
    if ls:
        ax.scatter([a[1] for a in ls], [a[2] for a in ls], s=24, color=col, edgecolor="white", lw=0.7, alpha=0.85, zorder=6)
    ax.text(-HALF_L + 0.6, HALF_W - 1.2, "own goal", fontsize=7.5, color=S.MUTED)
    deep = sorted([a for a in ls if a[1] < 0], key=lambda a: a[1])[:6]
    y = 0.50
    if not deep:
        fig.text(0.53, y, "No clear turnovers in their own half could be placed.", fontsize=9.5, color=S.MUTED)
    for (t, u, v) in deep:
        url = ctx.link(t)
        tt = fig.text(0.53, y, mmss(t - M["start"]), fontsize=13, fontweight="bold", color=S.BRAND if url else S.INK, va="center")
        if url:
            tt.set_url(url)
        fig.text(0.60, y, f"lost the ball {HALF_L + u:.0f} m from their own goal line", fontsize=9.5, va="center")
        y -= 0.052
    cov = M["coverage"]
    note = (f"How reliable: the ball was placed in {cov['ball_pct']:.0f}% of the frames and with a team for {M['possession']['known_pct']:.0f}% of the game. "
            f"About {cov['players_per_frame'][tm]:.1f} of {tm}'s 5 players were detected per frame. Turnovers are where the ball changed teams; treat them as pointers, then check the video.")
    fig.text(0.53, 0.185, _wrap(note, 74), fontsize=8, color=S.MUTED, va="top", linespacing=1.3)
    return fig


PAGES = [brief, with_ball, without_ball, energy_moments]
