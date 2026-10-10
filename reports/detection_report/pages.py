"""Pages of the detection report. Each page function takes the page frame and the context and returns a finished figure."""
import math
import textwrap

import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt

from ..match_report import style as S
from ..match_report.metrics import HALF_L, HALF_W, mmss
from ..match_report.pages import _dotp, _style_axes, plt_line
from . import analysis as an

C_GOOD, C_FAIR, C_POOR = "#0b7a3e", "#c98a0b", "#c2410c"
GRADE_COL = {"good": C_GOOD, "fair": C_FAIR, "poor": C_POOR}
GRADE_TXT = {"good": "Good", "fair": "Fair", "poor": "Poor"}


class Ctx:
    def __init__(self, A, record, advice, review, colours, created="", review_note=None, files=None):
        self.A, self.rec, self.advice, self.review, self.col, self.created = A, record, advice, review, colours, created
        self.review_note, self.files = review_note, files or {}


def _wrap(s, n):
    return "\n".join(textwrap.wrap(str(s), n))


def _g(score):
    return GRADE_COL[an.grade(score)]


def _card(fig, rect, title=None, sub=None):
    S.card(fig, rect)
    x, y, w, h = rect
    if title:
        fig.text(x + 0.015, y + h - 0.028, title, fontsize=11, fontweight="bold", va="center")
    if sub:
        fig.text(x + 0.015, y + h - 0.053, _wrap(sub, int(w * 150)), fontsize=8, color=S.MUTED, va="top", linespacing=1.25)


def _axes(fig, rect, ylabel=None, grid="y"):
    ax = fig.add_axes(rect)
    _style_axes(ax, ylabel, grid=grid)
    return ax


def _stats(fig, x, y0, rows, width=0.25, rh=0.047, fs=9.3):
    """rows: (label, value). A tidy two-column list."""
    for k, (lab, val) in enumerate(rows):
        y = y0 - k * rh
        fig.text(x, y, _wrap(lab, int(width * 105)), fontsize=fs, va="center", linespacing=1.1)
        fig.text(x + width, y, val, fontsize=fs + 1, fontweight="bold", va="center", ha="right")
        if k < len(rows) - 1:
            fig.add_artist(plt_line(x, y - rh / 2, x + width, y - rh / 2))


def _pitch(fig, rect):
    ax = fig.add_axes(rect)
    S.draw_pitch(ax)
    return ax


def _mins(A):
    return [f"{i}" for i in range(A["minutes"])]


def _p(v, nd=0):
    return "n/a" if v is None else f"{v:.{nd}f}%"


# ======================================================================================== cover
def cover(fr, ctx):
    A, rec = ctx.A, ctx.rec
    fig = fr.new("", cover=True)
    fig.add_artist(patches.Rectangle((0, 0), 1, 0.30, transform=fig.transFigure, fc=S.BRAND, ec="none", zorder=0))
    lax = fig.add_axes([0.06, 0.40, 0.17, 0.5]); lax.imshow(S.logo("green")); lax.axis("off")
    fig.text(0.30, 0.82, "DETECTION REPORT", fontsize=13, color=S.BRAND, fontweight="bold")
    title = str(rec.get("title") or " vs ".join(A["teams"]))
    fig.text(0.30, 0.74, title, fontsize=min(34, 34 * 30 / max(len(title), 30)), fontweight="bold", color=S.INK, va="center")
    fig.text(0.30, 0.645, _wrap("How well the computer found the players, goalkeepers and ball in this game's video, what is holding the reports back, and what to try first.", 70),
             fontsize=12, color=S.MUTED, va="center", linespacing=1.4)
    score = rec.get("score")
    meta = [str(rec.get("date") or ""), f"{rec.get('game_format', '')} · {rec.get('game_type', '')}".strip(" ·"),
            f"{mmss(A['duration'])} of play checked · {A['frames']:,} frames"]
    fig.text(0.30, 0.43, "   |   ".join(m for m in meta if m and m.strip()), fontsize=12, color=S.MUTED)
    fig.text(0.30, 0.37, "Every figure comes from the saved detections. There is no 'right answer' to compare with, so they measure completeness, steadiness and plausibility.", fontsize=9.5, color=S.MUTED)
    P, B, T = A["players"], A["ball"], A["tracking"]
    tiles = [("Overall detection health", f"{A['overall']:.0f} / 100"), ("Frames with the ball placed", f"{B['coverage'] * 100:.0f}%"),
             ("Players found per frame", f"{P['per_frame_mean']:.1f} of {2 * A['expected_per_team']}"), ("Player time on steady identities", f"{T['long_share']:.0f}%")]
    for k, (lab, val) in enumerate(tiles):
        x = 0.06 + k * 0.235
        fig.text(x, 0.185, val, fontsize=22, color="white", fontweight="bold")
        fig.text(x, 0.135, lab, fontsize=10, color="#d8f0df")
    fig.text(0.06, 0.06, f"Report created {ctx.created}", fontsize=8.5, color="#d8f0df")
    fig.text(0.94, 0.06, "Polis Walking Football · match analysis", fontsize=8.5, color="#d8f0df", ha="right")
    return fig


# ======================================================================================== scorecard
def _evidence(A, k):
    P, B, G, T, K, X = A["players"], A["ball"], A["goalkeepers"], A["tracking"], A["teams_check"], A["positions"]
    return {"players": f"{P['per_frame_mean']:.1f} of {2 * A['expected_per_team']} found per frame; all of them in {P['all_found_pct']:.0f}% of frames",
            "ball": f"placed in {B['coverage'] * 100:.0f}% of frames (aim: {an.TARGET_BALL_COVERAGE * 100:.0f}%)",
            "goalkeepers": f"found at the left end in {G['left_pct']:.0f}% and the right end in {G['right_pct']:.0f}% of frames",
            "tracking": f"{T['player_ids']} identities for {T['expected']} players; {T['long_share']:.0f}% of time on steady ones",
            "teams": f"{K['agree_pct']:.0f}% of labels agree with the final team; {K['gk_labelled_players_pct']:.1f}% wrongly 'Goalkeeper'",
            "positions": f"{X['inside_pct']:.0f}% inside the pitch; {X['jump_pct']:.1f}% of moves are jumps"}[k]


def scorecard(fr, ctx):
    A = ctx.A
    fig = fr.new("Scorecard", "One score per detection stage (100 = nothing to improve), in order of how much improving it would lift the reports.")
    S.card(fig, (0.05, 0.40, 0.90, 0.455))
    hy = 0.825
    for x, t in ((0.07, "Stage"), (0.30, "Score"), (0.50, "Grade"), (0.575, "What we found"), (0.905, "Priority")):
        fig.text(x, hy, t, fontsize=8.5, color=S.MUTED, fontweight="bold", va="center")
    fig.add_artist(plt_line(0.065, hy - 0.016, 0.935, hy - 0.016))
    rh = 0.0625
    for k, r in enumerate(A["ranking"]):
        y = hy - 0.055 - k * rh
        st = r["stage"]
        fig.text(0.07, y + 0.009, an.STAGE_NAMES[st], fontsize=10.5, fontweight="bold", va="center")
        fig.text(0.07, y - 0.014, _wrap("feeds: " + an.STAGE_FEEDS[st], 46).split("\n")[0] + ("..." if len("feeds: " + an.STAGE_FEEDS[st]) > 46 else ""), fontsize=7.2, color=S.MUTED, va="center")
        fig.add_artist(patches.Rectangle((0.30, y - 0.008), 0.15, 0.018, transform=fig.transFigure, fc=S.FAINT, ec="none"))
        fig.add_artist(patches.Rectangle((0.30, y - 0.008), 0.15 * min(r["score"], 100) / 100, 0.018, transform=fig.transFigure, fc=_g(r["score"]), ec="none"))
        fig.text(0.455, y + 0.001, f"{r['score']:.0f}", fontsize=10.5, fontweight="bold", va="center", color=_g(r["score"]))
        fig.text(0.50, y + 0.001, GRADE_TXT[r["grade"]], fontsize=10, fontweight="bold", color=_g(r["score"]), va="center")
        fig.text(0.575, y + 0.001, _wrap(_evidence(A, st), 62), fontsize=8.8, va="center", linespacing=1.2)
        if r["gain"] > 0.05:
            fig.add_artist(_dotp(fig, 0.915, y + 0.001, 0.0085, S.BRAND))
            fig.text(0.915, y + 0.001, str(sum(1 for q in A["ranking"][:k + 1] if q["gain"] > 0.05)), fontsize=9, color="white", fontweight="bold", ha="center", va="center")
        else:
            fig.text(0.915, y + 0.001, "-", fontsize=10, color=S.MUTED, ha="center", va="center")
        if k < len(A["ranking"]) - 1:
            fig.add_artist(plt_line(0.065, y - rh / 2 + 0.003, 0.935, y - rh / 2 + 0.003))
    # what to fix first
    S.card(fig, (0.05, 0.075, 0.90, 0.315))
    fig.text(0.07, 0.362, f"Overall detection health: {A['overall']:.0f} / 100", fontsize=14, fontweight="bold", color=_g(A["overall"]), va="center")
    fig.text(0.07, 0.332, "A weighted average of the six scores. The weights are how much of our report numbers rest on each stage.", fontsize=8.5, color=S.MUTED, va="center")
    fig.text(0.07, 0.298, "What to fix first", fontsize=11.5, fontweight="bold", color=S.BRAND, va="center")
    y = 0.262
    shown = [a for a in ctx.advice if a["gain"] > 0.05][:4]
    for k, a in enumerate(shown):
        up = A["overall_if_fixed"][a["stage"]]
        fig.text(0.07, y, f"{k + 1}.", fontsize=10, fontweight="bold", va="center", color=S.BRAND)
        fig.text(0.095, y, a["title"], fontsize=10, fontweight="bold", va="center")
        fig.text(0.095, y - 0.019, _wrap(f"Getting this stage to 'good' would lift overall health from {A['overall']:.0f} to {up:.0f}.", 100), fontsize=8.8, color=S.MUTED, va="center")
        y -= 0.047
    if not shown:
        fig.text(0.07, y, "Every stage is in the good band. Nothing stands out to fix.", fontsize=10, color=S.MUTED, va="center")
    return fig


# ======================================================================================== players
def players(fr, ctx):
    A = ctx.A
    P = A["players"]
    need = 2 * A["expected_per_team"]
    fig = fr.new("Player detection", f"Is every player found in every frame? Ten are expected on the pitch ({A['expected_per_team']} per team).")
    # distribution
    _card(fig, (0.05, 0.475, 0.30, 0.375), "Players found per frame", "Share of frames with each count. The dashed line is a full pitch.")
    ax = _axes(fig, [0.075, 0.51, 0.255, 0.255], "% of frames")
    keys = sorted(P["dist"])
    tot = sum(P["dist"].values()) or 1
    cols = [C_GOOD if k == need else (C_FAIR if k > need else S.BRAND if k >= need - 2 else C_POOR) for k in keys]
    ax.bar(keys, [100 * P["dist"][k] / tot for k in keys], color=cols, width=0.75)
    ax.axvline(need, color=S.MUTED, ls=(0, (3, 3)), lw=1)
    ax.set_xticks(keys)
    # per minute
    _card(fig, (0.365, 0.475, 0.30, 0.375), "Players found, minute by minute", "Average number found per frame, by minute of the game.")
    ax = _axes(fig, [0.39, 0.51, 0.255, 0.255], "players")
    x = np.arange(len(P["per_minute"]))
    pm = np.array([np.nan if v is None else v for v in P["per_minute"]], dtype=float)
    ax.plot(x, pm, color=S.BRAND, lw=2)
    ax.fill_between(x, pm, color=S.BRAND, alpha=0.12)
    ax.axhline(need, color=S.MUTED, ls=(0, (3, 3)), lw=1)
    ax.set_ylim(0, need + 1)
    ax.set_xlabel("minute of the game", fontsize=8.5)
    # confidence
    _card(fig, (0.68, 0.475, 0.27, 0.375), "How sure the model was", f"Boxes below {P['conf_cut']:.2f} are thrown away before the report sees them.")
    ax = _axes(fig, [0.70, 0.51, 0.235, 0.255], "boxes")
    edges = np.arange(0.3, 1.0001, 0.025)
    ax.bar(edges[:-1] + 0.0125, P["conf_hist"], width=0.022, color=S.BRAND)
    ax.axvline(P["conf_cut"], color=C_POOR, lw=1.4)
    ax.text(P["conf_cut"] + 0.01, ax.get_ylim()[1] * 0.92, "cut-off", color=C_POOR, fontsize=8, va="top")
    ax.set_xlim(0.5, 1.0)
    # drop-outs map
    _card(fig, (0.05, 0.075, 0.30, 0.385), "Where players go missing", "Moments inside a player's track with no detection, placed halfway between the sightings. Darker = more.")
    ax = _pitch(fig, [0.06, 0.11, 0.28, 0.2])
    if P["dropout_heat"]:
        S.pitch_heat(ax, P["dropout_heat"], C_POOR)
    ax.text(0, HALF_W + 1.6, "far touchline (top of the video)", fontsize=7, color=S.MUTED, ha="center")
    # where found
    _card(fig, (0.365, 0.075, 0.30, 0.385), "Where players are found", "All detections of the two teams inside the pitch.")
    ax = _pitch(fig, [0.375, 0.11, 0.28, 0.2])
    if P["heat"]:
        S.pitch_heat(ax, P["heat"], S.BRAND)
    # stats
    _card(fig, (0.68, 0.075, 0.27, 0.385), "In numbers")
    rows = [("Players per frame", f"{P['per_frame_mean']:.1f}"), ("Frames with all 10 found", _p(P["all_found_pct"])),
            ("Frames with fewer than 8", _p(P["below8_pct"])), ("Frames with more than 10 (extra boxes)", _p(P["extra_pct"], 1)),
            ("Moments missing inside a track", _p(P["dropout_pct"], 1)), ("Boxes within 0.10 of the cut-off", _p(P["conf_near_cut_pct"])),
            ("Player boxes outside the pitch", _p(P["outside_pct"], 1))]
    _stats(fig, 0.695, 0.385, rows, width=0.24, rh=0.0445, fs=8.8)
    return fig


# ======================================================================================== ball
def ball(fr, ctx):
    A = ctx.A
    B = A["ball"]
    fig = fr.new("Ball detection", f"Is the ball found often enough to say who has it? The reports work well when it is placed in about {an.TARGET_BALL_COVERAGE * 100:.0f}% of frames.")
    _card(fig, (0.05, 0.475, 0.30, 0.375), "Ball placed, minute by minute", "Share of frames in each minute with a believable ball.")
    ax = _axes(fig, [0.075, 0.51, 0.255, 0.255], "% of frames")
    bm = [0 if v is None else v for v in B["per_minute"]]
    x = np.arange(len(bm))
    ax.bar(x, [100 * v for v in bm], color=[(C_GOOD if v >= an.TARGET_BALL_COVERAGE else C_FAIR if v >= 0.4 else C_POOR) for v in bm], width=0.75)
    ax.axhline(an.TARGET_BALL_COVERAGE * 100, color=S.MUTED, ls=(0, (3, 3)), lw=1)
    ax.set_ylim(0, 100); ax.set_xlabel("minute of the game", fontsize=8.5)
    _card(fig, (0.365, 0.475, 0.30, 0.375), "How long the ball goes missing", "Number of stretches without a sighting, by length.")
    ax = _axes(fig, [0.39, 0.51, 0.255, 0.255], "stretches")
    labs = ["0.2-2 s", "2-5 s", "5-10 s", "10-20 s", "20-60 s", "> 1 min"]
    ax.bar(range(len(labs)), B["gap_hist"], color=[S.BRAND, S.BRAND, C_FAIR, C_FAIR, C_POOR, C_POOR], width=0.72)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=7.5)
    _card(fig, (0.68, 0.475, 0.27, 0.375), "How sure the model was", f"Sightings below {B['conf_cut']:.2f} are thrown away.")
    ax = _axes(fig, [0.70, 0.51, 0.235, 0.255], "sightings")
    edges = np.arange(0.2, 1.0001, 0.05)
    ax.bar(edges[:-1] + 0.025, B["conf_hist"], width=0.045, color=S.BRAND)
    ax.axvline(B["conf_cut"], color=C_POOR, lw=1.4)
    ax.axvline(0.45, color=S.MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0.25, 1.0)
    _card(fig, (0.05, 0.075, 0.30, 0.385), "Where the ball was placed", "Each dot is one sighting. Red rings are fixed spots that kept being reported as the ball.")
    ax = _pitch(fig, [0.06, 0.11, 0.28, 0.2])
    xy = np.array(B["xy"]) if B["xy"] else np.zeros((0, 2))
    if len(xy):
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color=S.BRAND, alpha=0.35, zorder=4)
    for sp in B["spots"][:4]:
        ax.add_patch(patches.Circle((sp["x"], sp["y"]), 1.3, fc="none", ec=C_POOR, lw=1.6, zorder=6))
    _card(fig, (0.365, 0.075, 0.30, 0.385), "Longest stretches without the ball")
    y = 0.375
    if not B["long_gaps"]:
        fig.text(0.38, y, "No gaps of 5 seconds or more.", fontsize=9.5, color=S.MUTED)
    for t0, ln in B["long_gaps"]:
        fig.text(0.38, y, mmss(t0 - A["start"]), fontsize=13, fontweight="bold", color=S.BRAND, va="center")
        fig.text(0.45, y, f"no ball for {ln:.0f} seconds", fontsize=9.5, va="center")
        y -= 0.052
    fig.text(0.38, 0.12, _wrap("Check the video at these times: is the ball really hidden, or is the model missing it?", 52), fontsize=8, color=S.MUTED, va="top")
    _card(fig, (0.68, 0.075, 0.27, 0.385), "In numbers")
    rows = [("Frames with a believable ball", _p(B["coverage"] * 100)), ("Frames with any ball detection", _p(B["raw_pct"])),
            ("Sightings at a fixed spot (removed)", f"{B['false_spot_hits']:,}"), ("Ball boxes outside the pitch", _p(B["outside_pct"])),
            ("Time with the ball missing 5 s +", _p(B["gap_over5_pct"])), ("Sightings with confidence under 0.45", _p(B["conf_low_pct"])),
            ("Sightings close to a player", _p(B["near_player_pct"])), ("Impossible jumps between sightings", _p(B["jump_pct"], 1))]
    _stats(fig, 0.695, 0.385, rows, width=0.24, rh=0.0405, fs=8.6)
    return fig


# ======================================================================================== goalkeepers
def goalkeepers(fr, ctx):
    A = ctx.A
    G = A["goalkeepers"]
    fig = fr.new("Goalkeepers and referee", "Is a goalkeeper found at each end? A goalkeeper is a player who stays close to one of the goal lines.")
    _card(fig, (0.05, 0.475, 0.30, 0.375), "Goalkeeper found, by end", "Share of frames with a goalkeeper at each end of the pitch.")
    ax = _axes(fig, [0.075, 0.51, 0.255, 0.255], "% of frames", grid="y")
    vals = [G["left_pct"], G["right_pct"], G["both_pct"], G["none_pct"]]
    ax.bar(range(4), vals, color=[S.BRAND, S.BRAND, C_GOOD, C_POOR], width=0.65)
    ax.set_xticks(range(4)); ax.set_xticklabels(["left end", "right end", "both", "neither"], fontsize=8.5)
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    _card(fig, (0.365, 0.475, 0.585, 0.375), "Goalkeeper found at each end, minute by minute", "Share of frames in each minute with a goalkeeper near that goal.")
    ax = _axes(fig, [0.39, 0.51, 0.54, 0.25], "% of the minute")
    lv = [0 if v is None else v for v in G["per_minute_left"]]; rv = [0 if v is None else v for v in G["per_minute_right"]]
    x = np.arange(len(lv))
    ax.bar(x - 0.2, [100 * v for v in lv], width=0.38, color=S.BRAND, label="left end")
    ax.bar(x + 0.2, [100 * v for v in rv], width=0.38, color="#db8225", label="right end")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 100); ax.set_xlabel("minute of the game", fontsize=8.5)
    _card(fig, (0.05, 0.075, 0.30, 0.385), "Where goalkeepers were placed", "Each dot is a goalkeeper box inside the pitch.")
    ax = _pitch(fig, [0.06, 0.11, 0.28, 0.2])
    xy = np.array(G["xy"]) if G["xy"] else np.zeros((0, 2))
    if len(xy):
        ax.scatter(xy[:, 0], xy[:, 1], s=3, color="#db8225", alpha=0.35, zorder=4)
    _card(fig, (0.365, 0.075, 0.30, 0.385), "Referee")
    fig.text(0.38, 0.385, _wrap(f"{G['referee_dets']} referee boxes were found in {G['referee_frames']} frames. {G['referee_team_labelled']} of them were labelled as a team's player. The reports do not use the referee, so this only matters if referee boxes are being counted as players.", 50),
             fontsize=9.3, va="top", linespacing=1.35)
    _card(fig, (0.68, 0.075, 0.27, 0.385), "In numbers")
    rows = [("Frames with the left-end goalkeeper", _p(G["left_pct"])), ("Frames with the right-end goalkeeper", _p(G["right_pct"])),
            ("Frames with both", _p(G["both_pct"])), ("Boxes from the goalkeeper class", f"{G['as_gk_class']:,}"),
            ("Goalkeeper-labelled boxes from the player class", f"{G['as_player_class']:,}"), ("Goalkeeper boxes away from both goals", _p(G["far_from_goal_pct"]))]
    _stats(fig, 0.695, 0.385, rows, width=0.24, rh=0.0495, fs=8.7)
    return fig


# ======================================================================================== tracking
def tracking(fr, ctx):
    A = ctx.A
    T = A["tracking"]
    fig = fr.new("Tracking and identities", "Does each player keep one identity? Broken tracks make a player look like several, which spoils distance, speed and running.")
    _card(fig, (0.05, 0.475, 0.30, 0.375), "How long identities last", "Number of identities by how long they lasted, before and after stitching.")
    ax = _axes(fig, [0.075, 0.51, 0.255, 0.255], "identities")
    labs = ["<2 s", "2-5", "5-10", "10-30", "30-60", "1-2 m", "2-5 m", "5 m+"]
    x = np.arange(len(labs))
    ax.bar(x - 0.2, T["life_hist_raw"], width=0.38, color=S.MUTED, alpha=0.6, label="tracker")
    ax.bar(x + 0.2, T["life_hist"], width=0.38, color=S.BRAND, label="after stitching")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _card(fig, (0.365, 0.475, 0.30, 0.375), "New identities, minute by minute", "A steady game would show about none after the first minute.")
    ax = _axes(fig, [0.39, 0.51, 0.255, 0.255], "new identities")
    xm = np.arange(len(T["new_ids_per_min"]))
    nt = np.array([np.nan if v is None else v for v in T["new_trackers_per_min"]], dtype=float)
    ni = np.array([np.nan if v is None else v for v in T["new_ids_per_min"]], dtype=float)
    ax.plot(xm, nt, color=S.MUTED, lw=1.6, label="tracker")
    ax.plot(xm, ni, color=S.BRAND, lw=2, label="after stitching")
    ax.legend(frameon=False, fontsize=8); ax.set_xlabel("minute of the game", fontsize=8.5); ax.set_ylim(bottom=0)
    _card(fig, (0.68, 0.475, 0.27, 0.375), "Where tracks end", "A track that ends mid-pitch is a player lost, usually behind someone.")
    ax = _pitch(fig, [0.69, 0.52, 0.25, 0.23])
    if T["ends_heat"]:
        S.pitch_heat(ax, T["ends_heat"], C_POOR)
    _card(fig, (0.05, 0.075, 0.90, 0.385), "In numbers")
    left = [("Track numbers made by the tracker", f"{T['tracker_ids']:,}"), ("Identities after stitching", f"{T['player_ids']:,}"),
            ("Players expected", f"{T['expected']}"), ("Identities removed by stitching", _p(T["reduction_pct"]))]
    mid = [("Typical (middle) identity lasts", f"{T['median_life_s']:.0f} s"), ("Player time on identities of 1 minute or more", _p(T["long_share"])),
           ("Same, before stitching", _p(T["long_share_raw"])), ("Identities lasting under 5 seconds", _p(T["short_ids_pct"]))]
    right = [("Tracks that end mid-pitch", _p(T["ends_midpitch_pct"])), ("Moves faster than 25 km/h within an identity", _p(T["jump_pct"], 1)),
             ("New identities per player per minute", f"{T['ids_per_player_per_min']:.1f}")]
    for col, rows in enumerate((left, mid, right)):
        _stats(fig, 0.07 + col * 0.30, 0.385, rows, width=0.26, rh=0.0525, fs=9.3)
    return fig


# ======================================================================================== teams
def teams(fr, ctx):
    A = ctx.A
    K = A["teams_check"]
    fig = fr.new("Team assignment", "Is each player put on the right team? The team model labels every box; the stitching step then gives each identity its majority team.")
    _card(fig, (0.05, 0.40, 0.55, 0.45), "Label from the team model against the final team", "Rows: what the model said for that box. Columns: the team the identity ended up with. Counts of player boxes.")
    labels_final = [t for t in A["teams"]] + ["Goalkeeper"]
    other = sorted({c for r in K["matrix"].values() for c in r if c not in labels_final})
    cols = labels_final + ([other[0]] if other else [])
    rows = [r for r in list(A["teams"]) + ["Goalkeeper"] if r in K["matrix"]]
    extra = sorted([r for r in K["matrix"] if r not in rows], key=lambda r: -sum(K["matrix"][r].values()))[:2]
    rows += extra
    x0, y0 = 0.07, 0.69
    cw = 0.088
    for j, c in enumerate(cols):
        fig.text(x0 + 0.155 + j * cw + cw / 2, y0 + 0.012, _wrap(c, 11), fontsize=7.5, fontweight="bold", ha="center", va="bottom", color=S.MUTED)
    for i, r in enumerate(rows):
        y = y0 - 0.025 - i * 0.055
        fig.text(x0, y, _wrap(r, 20), fontsize=8.5, fontweight="bold", va="center")
        tot = sum(K["matrix"][r].values()) or 1
        for j, c in enumerate(cols):
            v = K["matrix"][r].get(c, 0)
            good = (r == c)
            fig.add_artist(patches.Rectangle((x0 + 0.155 + j * cw + 0.004, y - 0.021), cw - 0.008, 0.042, transform=fig.transFigure,
                                             fc=S.tint(S.BRAND, 1 - min(0.8, 0.15 + v / tot)) if good else S.tint(C_POOR, 1 - min(0.8, 0.1 + 3 * v / tot)) if v else "#f3f4f1", ec="none"))
            fig.text(x0 + 0.155 + j * cw + cw / 2, y, f"{v:,}", fontsize=8.5, ha="center", va="center")
    _card(fig, (0.62, 0.40, 0.33, 0.45), "Team kit colours", "The first kit colour set for each team in the Game Logger.")
    for i, t in enumerate(A["teams"]):
        rgb = K["kit_rgb"].get(t)
        y = 0.70 - i * 0.12
        if rgb:
            fig.add_artist(patches.Rectangle((0.64, y - 0.04), 0.06, 0.08, transform=fig.transFigure, fc=tuple(v / 255 for v in rgb), ec=S.FAINT))
        fig.text(0.715, y, t, fontsize=11, va="center", fontweight="bold")
    if K["kit_distance"] is not None:
        good = K["kit_distance"] > 90
        fig.text(0.64, 0.50, _wrap(f"The two kits are {'clearly different' if good else 'close in colour'} (distance {K['kit_distance']:.0f}). " + ("Colour is not a source of mix-ups." if good else "Similar kits make team mix-ups more likely."), 38), fontsize=8.5, va="top", color=S.MUTED, linespacing=1.3)
    _card(fig, (0.05, 0.075, 0.90, 0.31), "In numbers")
    left = [("Labels that agree with the final team", _p(K["agree_pct"])), ("Labels that differ from the final team", _p(K["flip_pct"], 1)),
            ("Player boxes labelled 'Goalkeeper'", _p(K["gk_labelled_players_pct"], 1))]
    mid = [("Boxes with a label that is neither team", f"{K['noise_n']:,}"), ("Tracks with more than 1 in 10 labels differing", f"{K['mixed_tracks']} of {K['tracks']}"),
           ("Average team size gap per frame", f"{K['balance_gap']:.1f}")]
    for col, rws in enumerate((left, mid)):
        _stats(fig, 0.07 + col * 0.31, 0.305, rws, width=0.27, rh=0.055, fs=9.3)
    fig.text(0.68, 0.30, _wrap("Other labels seen: " + (", ".join(f"{k} ({v})" for k, v in K["noise_labels"].items()) if K["noise_labels"] else "none") + ". These come from teams the model was trained on; they should be mapped to the nearest of the two kits.", 45),
             fontsize=8.8, va="top", color=S.MUTED, linespacing=1.3)
    return fig


# ======================================================================================== positions
def positions(fr, ctx):
    A = ctx.A
    X = A["positions"]
    fig = fr.new("Pitch positions and speed", "Do the detections land in believable places on the pitch, and do the speeds look right?")
    _card(fig, (0.05, 0.40, 0.30, 0.45), "Pitch calibration")
    if X["pitch_fixed"]:
        txt = (f"The pitch is set from the calibrator, not detected: the same {X['pitch_keypoints']} corner and centre points are used in all {X['pitch_keypoint_frames']:,} frames. "
               "That keeps the positions steady, but it cannot show whether the camera moved. Check the annotated video if the picture looks off.")
    else:
        txt = f"{X['pitch_keypoints']} pitch points were detected in {X['pitch_keypoint_frames']:,} frames; the lowest confidence was {X['pitch_conf_min']:.2f}."
    fig.text(0.065, 0.76, _wrap(txt, 46), fontsize=9.5, va="top", linespacing=1.35)
    _card(fig, (0.365, 0.40, 0.585, 0.45), "How fast players moved", "Speed of each detection, in km/h. Dashed: the running limit; red: faster than a player can really move (a glitch).")
    ax = _axes(fig, [0.39, 0.45, 0.54, 0.31], "detections")
    ax.bar(np.arange(len(X["speed_hist"])) + 0.5, X["speed_hist"], width=0.9, color=S.BRAND)
    lim = ctx.rec.get("running_speed_km_h")
    try:
        lim = float(lim)
    except (TypeError, ValueError):
        lim = None
    if lim:
        ax.axvline(lim, color=S.MUTED, ls=(0, (3, 3)), lw=1.2)
    ax.axvline(an.JUMP_KMH, color=C_POOR, lw=1.2)
    ax.set_xlim(0, 26); ax.set_xlabel("km/h", fontsize=8.5)
    _card(fig, (0.05, 0.075, 0.90, 0.31), "In numbers")
    left = [("Detections inside the pitch", _p(X["inside_pct"], 1)), ("Positions well outside the pitch", _p(X["off_pitch_xy_pct"], 1)), ("Detections with no speed", _p(X["speed_null_pct"], 1))]
    mid = [("Typical speed", f"{X['speed_p50']:.1f} km/h" if X["speed_p50"] is not None else "n/a"), ("Fast speed (95th percentile)", f"{X['speed_p95']:.1f} km/h" if X["speed_p95"] is not None else "n/a"),
           ("Fastest 1% start at", f"{X['speed_p99']:.1f} km/h" if X["speed_p99"] is not None else "n/a")]
    right = [("Moves faster than 25 km/h", _p(X["speed_over25_pct"], 1)), ("Jumps within an identity", _p(X["jump_pct"], 1))]
    for col, rws in enumerate((left, mid, right)):
        _stats(fig, 0.07 + col * 0.30, 0.305, rws, width=0.26, rh=0.055, fs=9.3)
    return fig


# ======================================================================================== fixes
def _fix_card(fig, a, n, x, y, w, h, A):
    S.card(fig, (x, y, w, h))
    fig.add_artist(patches.Rectangle((x, y), 0.006, h, transform=fig.transFigure, fc=_g(A["scores"][a["stage"]]), ec="none"))
    top = y + h - 0.03
    fig.text(x + 0.022, top, f"{n}. {a['title']}", fontsize=13, fontweight="bold", va="center")
    fig.text(x + 0.022, top - 0.03, f"Fixing this would add about {a['gain']:.0f} to the overall score", fontsize=8.5, color=S.BRAND, fontweight="bold", va="center")
    yy = top - 0.07
    for lab, txt in (("SAW", a["saw"]), ("LIKELY CAUSE", a["why"])):
        wr = _wrap(txt, 62)
        fig.text(x + 0.022, yy, lab, fontsize=7, color=S.MUTED, fontweight="bold", va="top")
        fig.text(x + 0.022, yy - 0.018, wr, fontsize=9, va="top", linespacing=1.3)
        yy -= 0.018 + 0.0185 * (wr.count("\n") + 1) + 0.022
    fig.text(x + 0.022, yy, "TRY", fontsize=7, color=S.MUTED, fontweight="bold", va="top")
    yy -= 0.018
    for tip in a["try"]:
        wr = _wrap(tip, 60)
        fig.text(x + 0.022, yy, "•", fontsize=9, va="top", color=S.BRAND)
        fig.text(x + 0.036, yy, wr, fontsize=9, va="top", linespacing=1.3)
        yy -= 0.0185 * (wr.count("\n") + 1) + 0.012
    yy -= 0.01
    fig.text(x + 0.022, yy, "CHECK AFTER A RE-RUN", fontsize=7, color=S.MUTED, fontweight="bold", va="top")
    fig.text(x + 0.022, yy - 0.018, _wrap(a["watch"], 62), fontsize=9, va="top", linespacing=1.3)


def _fixes_page(fr, ctx, heading, items, first):
    fig = fr.new(heading, "Ordered by how much each would lift the reports. After a change, run this report again and compare the numbers named under 'Check'.")
    if not items:
        S.card(fig, (0.05, 0.075, 0.90, 0.775))
        fig.text(0.07, 0.78, "Nothing else to fix: the remaining stages are in the good band.", fontsize=12, color=S.MUTED)
        return fig
    for k, a in enumerate(items):
        _fix_card(fig, a, first + k, 0.05 + k * 0.455, 0.075, 0.445, 0.775, ctx.A)
    return fig


def fixes(fr, ctx):
    items = [a for a in ctx.advice if a["gain"] > 0.05] or ctx.advice
    return _fixes_page(fr, ctx, "What to try first", items[:2], 1)


def fixes_more(fr, ctx):
    items = [a for a in ctx.advice if a["gain"] > 0.05] or ctx.advice
    return _fixes_page(fr, ctx, "What to try next", items[2:4], 3)


# ======================================================================================== review
def review(fr, ctx):
    A, R = ctx.A, ctx.review
    fig = fr.new("Checked against the video", "A sample of frames looked at by eye shows what the automatic checks cannot: wrong boxes and players the model never reported.")
    if R and R.get("ready"):
        S.card(fig, (0.05, 0.075, 0.90, 0.775))
        fig.text(0.07, 0.815, f"Measured on {R['rows']} frames you reviewed", fontsize=13, fontweight="bold", color=S.BRAND, va="center")
        tiles = [("Players: boxes that are right (precision)", R["player_precision"] * 100 if R["player_precision"] is not None else None, True),
                 ("Players: found of those on the pitch (recall)", R["player_recall"] * 100 if R["player_recall"] is not None else None, True),
                 ("Ball found when visible", R["ball_found_of_visible_pct"], True),
                 ("Ball boxes that are the ball (precision)", R["ball_precision_pct"], True),
                 ("Player boxes on the wrong team", R["team_wrong_pct"], False)]
        for k, (lab, v, hi) in enumerate(tiles):
            x = 0.07 + (k % 3) * 0.29
            y = 0.68 - (k // 3) * 0.2
            val = "n/a" if v is None else f"{v:.0f}%"
            colr = S.INK if v is None else (_g(v if hi else 100 - 4 * v))
            fig.text(x, y, val, fontsize=30, fontweight="bold", color=colr, va="center")
            fig.text(x, y - 0.06, _wrap(lab, 36), fontsize=9.5, color=S.MUTED, va="top")
        fig.text(0.07, 0.30, _wrap(f"Of {R['players_shown']:.0f} player boxes drawn, {R['wrong']:.0f} were wrong and {R['missed']:.0f} players had no box. "
                                  f"The ball was visible in {R['ball_visible_frames']} of the frames. These figures come from {R['rows']} of {R['of']} sampled frames, so treat them as an estimate, "
                                  "and re-check the same frames after changing a detection setting to see if it helped.", 120), fontsize=10, va="top", linespacing=1.4)
    else:
        S.card(fig, (0.05, 0.075, 0.90, 0.775))
        fig.text(0.07, 0.815, "No review yet", fontsize=13, fontweight="bold", color=S.BRAND, va="center")
        n = (R or {}).get("rows", 0)
        steps = ["Open the review sheet PDF saved next to this report. It shows about 24 frames from the game with every detection drawn on.",
                 "Open the matching CSV file. For each frame (R01, R02, ...) fill in how many players are really on the pitch, how many boxes are wrong, how many players have no box, and whether the ball was visible and found.",
                 "Fill in at least 8 rows (more is better), save the CSV, and run the detection report again. This page then shows real precision and recall.",
                 "Keep the CSV file name. The same frames are used every time, so after changing a detection setting you can re-check the same frames."]
        y = 0.75
        for k, s in enumerate(steps):
            fig.text(0.07, y, f"{k + 1}.", fontsize=12, fontweight="bold", color=S.BRAND, va="center")
            w = _wrap(s, 110)
            fig.text(0.10, y + 0.008, w, fontsize=10.5, va="top", linespacing=1.4)
            y -= 0.025 * (w.count("\n") + 1) + 0.06
        if R:
            fig.text(0.07, 0.25, f"{n} of {R['of']} rows are filled in so far.", fontsize=10, color=S.MUTED)
        if ctx.review_note:
            fig.text(0.07, 0.18, _wrap(f"The review sheet could not be made this time: {ctx.review_note}.", 110), fontsize=10, color=C_POOR, va="top")
        if ctx.files.get("sheet"):
            fig.text(0.07, 0.12, f"Review sheet: {ctx.files['sheet']}", fontsize=9, color=S.MUTED)
            fig.text(0.07, 0.095, f"CSV to fill in: {ctx.files.get('csv', '')}", fontsize=9, color=S.MUTED)
    return fig


# ======================================================================================== about
def about(fr, ctx):
    A = ctx.A
    fig = fr.new("About the numbers", "What each score means and how it was worked out.")
    left = [("Overall health", "A weighted average of the six stage scores. Weights: " + ", ".join(f"{an.STAGE_NAMES[k].lower()} {int(v * 100)}%" for k, v in an.STAGE_WEIGHTS.items()) + ". They say how much of the report numbers rest on each stage; change them in analysis.py if you disagree."),
            ("Player detection", f"The average number of players found per frame inside the pitch, out of {2 * A['expected_per_team']} ({A['expected_per_team']} per team), as a share. A frame with 12 found does not score above 100%."),
            ("Ball detection", f"The share of frames with a believable ball, divided by the target of {an.TARGET_BALL_COVERAGE * 100:.0f}%. A believable ball is inside the pitch, not at a spot that keeps repeating, and the most confident one if there are several."),
            ("Goalkeepers", "The average of the share of frames with a goalkeeper near the left goal and near the right goal.")]
    right = [("Tracking", "70% the share of player time carried by identities that last a minute or more, 30% how few identities there are for the number of players."),
             ("Team assignment", "100 minus the share of labels that differ from the final team, minus a penalty for labels that are neither team or say 'Goalkeeper' on a player."),
             ("Pitch positions", "Starts at 100 and loses points for jumps within an identity, positions well outside the pitch and detections outside the pitch."),
             ("Priority", "Each stage's gain = its weight x the points it is short of 'good' (85). The order of the list is the order of those gains. It is a guide to where effort pays off most, not a promise of the result.")]
    for col, items in enumerate((left, right)):
        x0 = 0.05 + col * 0.475
        S.card(fig, (x0, 0.075, 0.425, 0.775))
        y = 0.81
        for head, body in items:
            fig.text(x0 + 0.02, y, head, fontsize=11.5, fontweight="bold", color=S.BRAND, va="top")
            w = _wrap(body, 64)
            fig.text(x0 + 0.02, y - 0.032, w, fontsize=9.5, va="top", linespacing=1.4)
            y -= 0.032 + 0.0275 * (w.count("\n") + 1) + 0.05
    return fig


PAGES = [cover, scorecard, players, ball, goalkeepers, tracking, teams, positions, fixes, fixes_more, review, about]
