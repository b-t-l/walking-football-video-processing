"""Team overview PDF built from a set of individual player-report metrics.

Written automatically when a player-report run covers two or more named/tracked
players on the same team. Complements (does not replace) the per-player PDFs.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

from reports.match_report import style as S
from reports.match_report.metrics import HALF_L, HALF_W, mmss
from reports.player_report import metrics as pm
from utils import GameFiles

COL_IN = "#1f6f43"
COL_OUT = "#c45c26"
COL_CLOSE = "#e0a106"       # close to running
COL_RUN = "#b42318"         # at / above running limit

# Stronger colours for final-third player dots
FINAL_PALETTE = [
    "#0d5c2e", "#c62828", "#1565c0", "#e65100", "#6a1b9a",
    "#00838f", "#f9a825", "#ad1457", "#2e7d32", "#4527a0",
]


def _safe_name(label):
    s = re.sub(r"[^\w\-]+", "-", str(label).strip(), flags=re.UNICODE)
    return (re.sub(r"-{2,}", "-", s).strip("-_") or "team")[:48]


def _dotp(fig, x, y, R, fc):
    return patches.Ellipse((x, y), 2 * R, 2 * R * S.PAGE_W / S.PAGE_H,
                           transform=fig.transFigure, fc=fc, ec="none")


def _mmss(seconds):
    if seconds is None:
        return "—"
    seconds = int(max(0, round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _num(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _fmt_s(v):
    if v is None:
        return "—"
    return f"{v:.1f}s"


# ------------------------------------------------------------------------------------------------ aggregate

def summarise_squad(player_reports, team_name, M):
    """Turn a list of per-player report dicts into squad-level numbers for the PDF."""
    rows = []
    for r in player_reports:
        acc = r.get("accuracy") or {}
        spells = r.get("ball_spells") or {}
        bands = r.get("speed_bands") or {}
        rows.append({
            "name": r["subject"]["label"],
            "distance_m": _num(r.get("distance_m")),
            "speed_avg": r.get("speed_avg"),
            "speed_max": r.get("speed_max"),
            "tracked_s": _num(r.get("tracked_s")),
            "coverage_pct": _num(acc.get("coverage_pct")),
            "confidence": acc.get("band") or "—",
            "confidence_score": _num(acc.get("score")),
            "possession_in_pct": _num(r.get("possession_in_pct")),
            "ball_spells": spells,
            "speed_bands": bands,
            "n_in_final": int(r.get("n_in_final_third") or 0),
            "n_samples": int(r.get("n_samples") or 0),
            "dots_in_final": r.get("dots_in_final") or {"x": [], "y": []},
            "dots_in": r.get("dots_in") or {"x": [], "y": []},
            "dots_out": r.get("dots_out") or {"x": [], "y": []},
        })
    rows.sort(key=lambda x: (-x["distance_m"], x["name"].lower()))

    def mean_of(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    total_dist = sum(r["distance_m"] for r in rows)
    n = max(len(rows), 1)
    ft = (M.get("final_third_by_team") or {}).get(team_name) or {}
    return {
        "team": team_name,
        "n_players": len(rows),
        "players": rows,
        "total_distance_m": total_dist,
        "mean_distance_m": total_dist / n,
        "mean_speed_avg": mean_of("speed_avg"),
        "max_speed_squad": max((_num(r["speed_max"]) for r in rows if r["speed_max"] is not None), default=None),
        "mean_coverage_pct": mean_of("coverage_pct"),
        "mean_confidence_score": mean_of("confidence_score"),
        "playable_s": M.get("playable_s"),
        "start": M.get("start"),
        "end": M.get("end"),
        "has_half_time": M.get("has_half_time"),
        "running_kmh": M.get("running_kmh") or 15.0,
        "final_third": ft,
        "target": M.get("target"),
    }


# ------------------------------------------------------------------------------------------------ pages

class Ctx:
    def __init__(self, squad, rec, colours, team_colour, created=""):
        self.S = squad
        self.rec = rec
        self.col = colours
        self.team_colour = team_colour
        self.created = created
        self.teams = [rec.get("team_a"), rec.get("team_b")]


def cover(fr, ctx):
    from .player_images import draw_portrait

    S_, rec = ctx.S, ctx.rec
    fig = fr.new("", cover=True)
    fig.add_artist(patches.Rectangle((0, 0), 1, 0.30, transform=fig.transFigure, fc=S.BRAND, ec="none", zorder=0))
    lax = fig.add_axes([0.06, 0.40, 0.17, 0.5], facecolor="none", frameon=False)
    S.show_logo(lax, "green")
    fig.text(0.30, 0.82, "TEAM PLAYER REPORT", fontsize=13, color=S.BRAND, fontweight="bold")
    fig.add_artist(_dotp(fig, 0.312, 0.735, 0.0125, ctx.team_colour))
    fig.text(0.335, 0.735, S_["team"] or "Team", fontsize=30, fontweight="bold", color=S.INK, va="center")
    fig.text(0.30, 0.665, f"{S_['n_players']} named players  ·  squad distance, speed, possession and tracking",
             fontsize=12, color=S.MUTED, va="center")
    title = str(rec.get("title") or " vs ".join(t for t in ctx.teams if t))
    fig.text(0.30, 0.56, title, fontsize=17, color=S.INK, va="center")
    playable = S_.get("playable_s") or 0
    meta = [str(rec.get("date") or ""),
            f"{rec.get('game_format', '')} · {rec.get('game_type', '')}".strip(" ·"),
            f"{mmss(playable)} of play analysed"]
    fig.text(0.30, 0.43, "   |   ".join(m for m in meta if m and str(m).strip()), fontsize=12, color=S.MUTED)

    # Squad portraits (up to 8) from <data>/_library/player_images/
    players = list(S_.get("players") or [])[:8]
    n = len(players)
    if n:
        pw, ph, gap = 0.055, 0.07, 0.012
        total_w = n * pw + max(0, n - 1) * gap
        x0 = 0.30
        if total_w > 0.64:
            pw = (0.64 - max(0, n - 1) * gap) / n
            ph = pw * (0.07 / 0.055)
        for i, p in enumerate(players):
            draw_portrait(fig, p.get("name"), (x0 + i * (pw + gap), 0.325, pw, ph))

    cov = S_.get("mean_coverage_pct")
    ft = S_.get("final_third") or {}
    tiles = [
        ("Squad distance", f"{S_['total_distance_m']:,.0f} m"),
        ("Final-third entries", f"{ft.get('entries', 0)}"),
        ("Avg tracking", f"{cov:.0f}%" if cov is not None else "—"),
    ]
    for k, (lab, val) in enumerate(tiles):
        x = 0.06 + k * 0.31
        fig.text(x, 0.185, val, fontsize=24, color="white", fontweight="bold")
        fig.text(x, 0.135, lab, fontsize=10, color="#d8f0df")
    fig.text(0.06, 0.06, f"Report created {ctx.created}", fontsize=8.5, color="#d8f0df")
    fig.text(0.94, 0.06, "Polis Walking Football · squad overview", fontsize=8.5, color="#d8f0df", ha="right")
    return fig


def work_rate(fr, ctx):
    """Distance bars + close/running spell counts (number of times)."""
    S_ = ctx.S
    fig = fr.new(f"{S_['team']} · Work rate",
                 "Distance covered, and how often each player was close to or at the running limit.")
    players = S_["players"]
    names = [p["name"].split()[0] if p["name"] else "?" for p in players]
    x = np.arange(len(players))
    run_kmh = S_.get("running_kmh") or 15.0
    near = max(run_kmh * pm.NEAR_RUN_FRAC, run_kmh - 3.0)

    # ---- distance (titles sit below the page header rule at ~0.868)
    ax1 = fig.add_axes([0.08, 0.38, 0.40, 0.44])
    dists = [p["distance_m"] for p in players]
    ax1.bar(x, dists, color=ctx.team_colour, width=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8, rotation=35, ha="right")
    ax1.set_ylabel("Distance (m)", fontsize=9, color=S.MUTED)
    ax1.set_facecolor("none")
    ax1.tick_params(length=0, labelsize=8)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.spines["left"].set_color(S.FAINT)
    ax1.spines["bottom"].set_color(S.FAINT)
    ax1.grid(axis="y", color=S.FAINT, lw=0.8)
    ax1.set_axisbelow(True)
    fig.text(0.08, 0.845, "Distance covered", fontsize=12, fontweight="bold")

    # ---- close / running: number of times (spells), not sample stacks
    ax2 = fig.add_axes([0.56, 0.38, 0.40, 0.44])
    close = np.array([int((p.get("speed_bands") or {}).get("close") or 0) for p in players], dtype=float)
    running = np.array([int((p.get("speed_bands") or {}).get("running") or 0) for p in players], dtype=float)
    ax2.bar(x, close, color=COL_CLOSE, width=0.7, label=f"Close to running ({near:.0f}–{run_kmh:.0f})")
    ax2.bar(x, running, bottom=close, color=COL_RUN, width=0.7, label=f"Running (≥{run_kmh:.0f})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8, rotation=35, ha="right")
    ax2.set_ylabel("Number of times", fontsize=9, color=S.MUTED)
    ax2.set_facecolor("none")
    ax2.tick_params(length=0, labelsize=8)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.spines["left"].set_color(S.FAINT)
    ax2.spines["bottom"].set_color(S.FAINT)
    ax2.grid(axis="y", color=S.FAINT, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.text(0.56, 0.845, "Speed vs running limit", fontsize=12, fontweight="bold")

    for i in range(len(players)):
        if close[i] > 0:
            ax2.text(x[i], close[i] / 2, f"{int(close[i])}",
                     ha="center", va="center", fontsize=7, color="#3a2a00", fontweight="bold")
        if running[i] > 0:
            ax2.text(x[i], close[i] + running[i] / 2, f"{int(running[i])}",
                     ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    lines = []
    for p in players:
        b = p.get("speed_bands") or {}
        lines.append(f"{p['name'].split()[0]}: close {b.get('close', 0)}, running {b.get('running', 0)}")
    fig.text(0.08, 0.22, "Close / running counts", fontsize=10, fontweight="bold")
    fig.text(0.08, 0.18, "   ·   ".join(lines[:8]), fontsize=8, color=S.MUTED)
    if len(lines) > 8:
        fig.text(0.08, 0.145, "   ·   ".join(lines[8:16]), fontsize=8, color=S.MUTED)
    fig.text(0.08, 0.08,
             f"Running limit for this game: {run_kmh:.0f} km/h. "
             f"“Close” is {near:.0f}–{run_kmh:.0f} km/h. "
             f"Each bar segment is a spell (a run of detections in that band).",
             fontsize=8.5, color=S.MUTED)
    return fig


def possession_stats(fr, ctx):
    """How often each player held the ball, and how long those spells lasted."""
    from .player_images import draw_portrait

    S_ = ctx.S
    fig = fr.new(f"{S_['team']} · Ball possession",
                 "Times each player was the ball holder, and how long those spells lasted.")
    players = sorted(S_["players"], key=lambda p: (-(p.get("ball_spells") or {}).get("count", 0), p["name"]))
    names = [p["name"].split()[0] if p["name"] else "?" for p in players]
    counts = [int((p.get("ball_spells") or {}).get("count") or 0) for p in players]
    avgs = [_num((p.get("ball_spells") or {}).get("avg_s"), 0) for p in players]
    mins = [(p.get("ball_spells") or {}).get("min_s") for p in players]
    maxs = [(p.get("ball_spells") or {}).get("max_s") for p in players]

    ax = fig.add_axes([0.10, 0.42, 0.55, 0.38])
    x = np.arange(len(players))
    ax.bar(x, counts, color=COL_IN, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=35, ha="right")
    ax.set_ylabel("Possession spells", fontsize=9, color=S.MUTED)
    ax.set_facecolor("none")
    ax.tick_params(length=0, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(S.FAINT)
    ax.spines["bottom"].set_color(S.FAINT)
    ax.grid(axis="y", color=S.FAINT, lw=0.8)
    ax.set_axisbelow(True)
    for i, c in enumerate(counts):
        if c:
            ax.text(i, c + max(counts) * 0.02, str(c), ha="center", va="bottom", fontsize=8, color=S.INK)
    fig.text(0.10, 0.845, "Number of times on the ball", fontsize=12, fontweight="bold")

    # Spell length table with small portraits
    S.card(fig, (0.70, 0.38, 0.25, 0.44))
    fig.text(0.72, 0.78, "Spell length", fontsize=11, fontweight="bold")
    fig.text(0.72, 0.735, "min – avg – max", fontsize=8, color=S.MUTED)
    yy = 0.68
    for p, mn, av, mx in zip(players, mins, avgs, maxs):
        if yy < 0.42:
            break
        draw_portrait(fig, p["name"], (0.715, yy - 0.028, 0.028, 0.036))
        fig.text(0.75, yy, (p["name"] or "?")[:12], fontsize=8.5, color=S.INK, va="top")
        fig.text(0.93, yy, f"{_fmt_s(mn)} · {_fmt_s(av)} · {_fmt_s(mx)}",
                 fontsize=8, color=S.MUTED, va="top", ha="right")
        yy -= 0.042

    # Average length bars
    ax2 = fig.add_axes([0.10, 0.10, 0.55, 0.24])
    ax2.bar(x, avgs, color="#2a6f9e", width=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8, rotation=35, ha="right")
    ax2.set_ylabel("Avg spell (s)", fontsize=9, color=S.MUTED)
    ax2.set_facecolor("none")
    ax2.tick_params(length=0, labelsize=8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.spines["left"].set_color(S.FAINT)
    ax2.spines["bottom"].set_color(S.FAINT)
    ax2.grid(axis="y", color=S.FAINT, lw=0.8)
    ax2.set_axisbelow(True)
    fig.text(0.10, 0.36, "Average time on the ball per spell", fontsize=11, fontweight="bold")
    fig.text(0.70, 0.22,
             "A spell is a run of frames where this\n"
             "player is the closest holder of the ball\n"
             "(within ~1.5 m). Gaps under 1.5 s are\n"
             "still one spell.",
             fontsize=8.5, color=S.MUTED, va="top")
    return fig


def tracking_quality(fr, ctx):
    S_ = ctx.S
    fig = fr.new(f"{S_['team']} · Tracking quality",
                 "Report confidence rests on how much of the game each player was detected.")
    players = sorted(S_["players"], key=lambda p: (-p["coverage_pct"], p["name"].lower()))
    names = [p["name"].split()[0] if p["name"] else "?" for p in players]
    cov = [p["coverage_pct"] for p in players]
    colours = []
    for c in cov:
        if c >= 70:
            colours.append(COL_IN)
        elif c >= 45:
            colours.append("#c4a035")
        else:
            colours.append(COL_OUT)
    ax = fig.add_axes([0.12, 0.22, 0.55, 0.58])
    y = np.arange(len(players))
    ax.barh(y, cov, color=colours, height=0.7)
    ax.axvline(70, color=COL_IN, ls="--", lw=0.9, alpha=0.7)
    ax.axvline(45, color="#c4a035", ls="--", lw=0.9, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Tracking coverage (% of play)", fontsize=9, color=S.MUTED)
    ax.set_facecolor("none")
    ax.tick_params(length=0, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(S.FAINT)
    ax.spines["bottom"].set_color(S.FAINT)

    mean_c = S_.get("mean_coverage_pct")
    S.card(fig, (0.72, 0.40, 0.22, 0.40))
    fig.text(0.74, 0.74, "Squad average", fontsize=10, color=S.MUTED)
    fig.text(0.74, 0.62, f"{mean_c:.0f}%" if mean_c is not None else "—",
             fontsize=28, fontweight="bold", color=S.INK)
    fig.text(0.74, 0.52, "of play tracked", fontsize=9, color=S.MUTED)
    fig.text(0.74, 0.44, "Dashed lines: High (70)\nand Fair (45) bands.", fontsize=8, color=S.MUTED, va="top")
    fig.text(0.12, 0.10,
             "Players with low coverage make the team totals less reliable — treat their distance and heat maps with caution.",
             fontsize=9, color=S.MUTED)
    return fig


def pitch_squad_final_third(fr, ctx):
    from .player_images import draw_portrait

    S_ = ctx.S
    fig = fr.new(f"{S_['team']} · Final-third shape",
                 "Named players while the team had the ball in the attacking third (shaded).")
    players = list(S_.get("players") or [])
    n = len(players)

    # Legend row: thin colour ring (circle barely larger than the image), name centred under it
    if n:
        pw, ph = 0.045, 0.055
        # Visual circle: diameter in inches = image size + modest ring
        img_d = max(pw * S.PAGE_W, ph * S.PAGE_H)
        ring_in = 0.04
        D = img_d + 2 * ring_in
        ew, eh = D / S.PAGE_W, D / S.PAGE_H
        pad_x = 0.025
        step = ew + pad_x
        span = (n - 1) * step
        cx0 = 0.5 - span / 2
        # leave clear space under the header rule (~0.868)
        y_img = 0.730
        y_name = 0.705
        for i, p in enumerate(players):
            col = FINAL_PALETTE[i % len(FINAL_PALETTE)]
            label = (p.get("name") or f"P{i + 1}").split()[0]
            cx = cx0 + i * step
            cy = y_img + ph * 0.5
            xi = cx - pw * 0.5
            fig.add_artist(patches.Ellipse(
                (cx, cy), ew, eh,
                transform=fig.transFigure, fc=col, ec="none", zorder=4,
            ))
            draw_portrait(fig, p.get("name"), (xi, y_img, pw, ph), keep_alpha=True)
            fig.text(cx, y_name, label, fontsize=8.5, color=S.INK, ha="center", va="top")

    ax = fig.add_axes([0.10, 0.08, 0.80, 0.58])
    S.draw_pitch(ax, fill="#f3f6f1")
    ax.axvspan(pm.THIRD, HALF_L, color=COL_IN, alpha=0.12, zorder=1)
    drawn = False
    for i, p in enumerate(players):
        d = p["dots_in_final"]
        if not d.get("x"):
            continue
        col = FINAL_PALETTE[i % len(FINAL_PALETTE)]
        ax.scatter(d["x"], d["y"], s=24, c=col, alpha=0.85, linewidths=0,
                   edgecolors="none", zorder=4)
        drawn = True
    if not drawn:
        ax.text(0, 0, "No final-third in-possession samples for this squad.",
                ha="center", va="center", fontsize=12, color=S.MUTED)
    ax.annotate("", xy=(HALF_L - 1.2, -HALF_W - 1.6), xytext=(-HALF_L + 1.2, -HALF_W - 1.6),
                arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.2), annotation_clip=False)
    ax.text(0, -HALF_W - 2.2, "attacking direction", ha="center", va="center", fontsize=8, color=S.MUTED,
            clip_on=False)
    return fig


def final_third_entries(fr, ctx):
    """How often the ball reached the final third in possession, and estimated passes to get there."""
    S_ = ctx.S
    ft = S_.get("final_third") or {}
    fig = fr.new(f"{S_['team']} · Reaching the final third",
                 "Times the ball entered the attacking third while this team had possession.")
    entries = int(ft.get("entries") or 0)
    passes = ft.get("passes_to_enter") or []
    p_avg, p_min, p_max = ft.get("passes_avg"), ft.get("passes_min"), ft.get("passes_max")

    # Big number tiles
    tiles = [
        ("Entries", f"{entries}"),
        ("Avg passes to enter", f"{p_avg:.1f}" if p_avg is not None else "—"),
        ("Pass range", f"{p_min}–{p_max}" if p_min is not None else "—"),
    ]
    for i, (lab, val) in enumerate(tiles):
        x = 0.08 + i * 0.30
        S.card(fig, (x, 0.62, 0.27, 0.20))
        fig.text(x + 0.02, 0.76, lab, fontsize=10, color=S.MUTED)
        fig.text(x + 0.02, 0.68, val, fontsize=28, fontweight="bold", color=S.INK, va="center")

    # Histogram of passes per entry
    ax = fig.add_axes([0.12, 0.16, 0.55, 0.38])
    if passes:
        mx = max(passes) if passes else 0
        bins = np.arange(0, mx + 2) - 0.5
        ax.hist(passes, bins=bins, color=COL_IN, edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Estimated passes in the build-up before entry", fontsize=9, color=S.MUTED)
        ax.set_ylabel("Entries", fontsize=9, color=S.MUTED)
    else:
        ax.text(0.5, 0.5, "No final-third entries measured\n(or possession / ball data missing).",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color=S.MUTED)
    ax.set_facecolor("none")
    ax.tick_params(length=0, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(S.FAINT)
    ax.spines["bottom"].set_color(S.FAINT)
    ax.grid(axis="y", color=S.FAINT, lw=0.8)
    ax.set_axisbelow(True)
    fig.text(0.12, 0.56, "Passes before each entry", fontsize=12, fontweight="bold")

    fig.text(0.72, 0.48, "How this is counted", fontsize=11, fontweight="bold")
    fig.text(0.72, 0.44,
             "An entry is when the ball crosses\n"
             "into the final third while this team\n"
             "has possession (attacking to the right).\n\n"
             "Passes are estimated from changes of\n"
             "ball holder on this team during the\n"
             "build-up before that entry. Short gaps\n"
             "and missed detections can under-count.",
             fontsize=8.5, color=S.MUTED, va="top")
    return fig


PAGES = [
    cover,
    work_rate,
    possession_stats,
    tracking_quality,
    pitch_squad_final_third,
    final_third_entries,
]


# ------------------------------------------------------------------------------------------------ writer

def write_team_overview(squad, rec, colours, team_colour, out_dir, stamp, created=None):
    """Write one team-overview PDF (+ json) into out_dir/Players. Returns the pdf path."""
    created = created or time.strftime("%d %b %Y %H:%M")
    ctx = Ctx(squad, rec, colours, team_colour, created=created)
    team = squad["team"] or "team"
    game_title = str(rec.get("title") or " vs ".join(t for t in [rec.get("team_a"), rec.get("team_b")] if t))
    frame = S.Frame(
        f"{team} · Team player report",
        f"{game_title} · {rec.get('date', '')} · {team}",
        len(PAGES),
    )
    pdf_path = GameFiles.report_file(
        out_dir, rec, f"team-players-{_safe_name(team)}", stamp, "pdf", subfolder="Players")
    with PdfPages(pdf_path, metadata={
        "Title": f"Team player report: {team} · {game_title}",
        "Author": "Polis Walking Football",
    }) as pdf:
        for fn in PAGES:
            try:
                fig = fn(frame, ctx)
            except Exception as e:
                print(f"⚠️ Team overview page '{fn.__name__}' failed: {e}")
                fig = frame.new(team, "This page could not be built.")
                fig.text(0.05, 0.75, f"{type(e).__name__}: {e}", fontsize=10, color=S.MUTED)
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            S.plt.close(fig)
    slim = {
        "team": squad["team"],
        "n_players": squad["n_players"],
        "total_distance_m": squad["total_distance_m"],
        "mean_distance_m": squad["mean_distance_m"],
        "mean_speed_avg": squad["mean_speed_avg"],
        "mean_coverage_pct": squad["mean_coverage_pct"],
        "running_kmh": squad.get("running_kmh"),
        "final_third": squad.get("final_third"),
        "players": [{
            "name": p["name"],
            "distance_m": p["distance_m"],
            "speed_avg": p["speed_avg"],
            "speed_max": p["speed_max"],
            "coverage_pct": p["coverage_pct"],
            "ball_spells": p.get("ball_spells"),
            "speed_bands": p.get("speed_bands"),
            "n_in_final": p["n_in_final"],
        } for p in squad["players"]],
    }
    with open(os.path.splitext(pdf_path)[0] + ".json", "w") as f:
        json.dump(slim, f, indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o)
    return pdf_path


def write_overviews_for_run(M, rec, colours, team_colours_by_name, out_dir, stamp, created=None):
    """Group player reports by team and write an overview for each team with 2+ players."""
    by_team = {}
    for r in M["reports"]:
        tm = r.get("team") or "Unknown"
        by_team.setdefault(tm, []).append(r)
    paths = []
    for tm, reps in by_team.items():
        if len(reps) < 2:
            continue
        squad = summarise_squad(reps, tm, M)
        tcol = team_colours_by_name.get(tm) or colours[0]
        path = write_team_overview(squad, rec, colours, tcol, out_dir, stamp, created=created)
        paths.append(path)
        print(f"  · Team overview ({tm}): {path}")
    return paths
