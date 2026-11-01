"""PDF pages for one player's report: cover, statistics, then three overhead pitch views."""
import textwrap

from matplotlib import patches

from reports.match_report import style as S
from reports.match_report.metrics import mmss as _mmss_game

# In possession / out of possession / unknown
COL_IN = "#1f6f43"
COL_OUT = "#c45c26"
COL_UNK = "#9aa3b2"


class Ctx:
    def __init__(self, M, record, colours, report, created=""):
        self.M = M
        self.rec = record
        self.col = colours
        self.report = report
        self.created = created
        self.teams = M["teams"]
        self.name = report["subject"]["label"]
        self.team = report.get("team") or "—"


def _mmss(seconds):
    seconds = int(max(0, round(seconds or 0)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _player_page(fr, ctx, title_tail, intro=None):
    """Standard inner page: portrait + '{name} · {title_tail}', then a wrapped intro."""
    from .player_images import draw_portrait

    fig = fr.new("", None)
    pw, ph = 0.038, 0.050
    draw_portrait(fig, ctx.name, (0.05, 0.935 - ph * 0.5, pw, ph))
    fig.text(0.05 + pw + 0.012, 0.935, f"{ctx.name} · {title_tail}",
             fontsize=21, fontweight="bold", color=S.INK, va="center")
    if intro:
        wrapped = textwrap.fill(str(intro), width=145)
        fig.text(0.05, 0.878, wrapped, fontsize=10.5, color=S.MUTED, va="top", linespacing=1.3)
    return fig


# ======================================================================================== 1 cover
def cover(fr, ctx):
    """Same layout language as the match / opposition covers: logo, brand strip, headline tiles."""
    from .player_images import draw_portrait

    M, rec, r = ctx.M, ctx.rec, ctx.report
    fig = fr.new("", cover=True)
    fig.add_artist(patches.Rectangle((0, 0), 1, 0.30, transform=fig.transFigure, fc=S.BRAND, ec="none", zorder=0))
    lax = fig.add_axes([0.06, 0.40, 0.17, 0.5], facecolor="none", frameon=False)
    S.show_logo(lax, "green")
    fig.text(0.30, 0.82, "PLAYER REPORT", fontsize=13, color=S.BRAND, fontweight="bold")
    # Portrait where the team-colour dot used to be; name with a little padding after it
    pw, ph = 0.055, 0.12
    name_y = 0.735
    draw_portrait(fig, ctx.name, (0.30, name_y - ph * 0.5, pw, ph))
    fig.text(0.30 + pw + 0.018, name_y, ctx.name, fontsize=34, fontweight="bold", color=S.INK, va="center")
    fig.text(0.30, 0.655, f"{ctx.team}  ·  distance, speed and where they spent the game",
             fontsize=12, color=S.MUTED, va="center")
    title = str(rec.get("title") or " vs ".join(ctx.teams))
    fig.text(0.30, 0.56, title, fontsize=17, color=S.INK, va="center")
    acc = r.get("accuracy") or {}
    # Detection confidence (right) — how reliably this player was tracked
    fig.text(0.86, 0.72, acc.get("band", "—"), fontsize=28, fontweight="bold", color=S.INK, va="center", ha="center")
    fig.text(0.86, 0.655, "Detection confidence", fontsize=11, color=S.INK, ha="center", fontweight="bold")
    fig.text(0.86, 0.615, f"score · {acc.get('score', 0):.0f}", fontsize=10, color=S.MUTED, ha="center")
    fig.text(0.86, 0.575, f"{acc.get('coverage_pct', 0):.0f}% of play tracked", fontsize=9, color=S.MUTED, ha="center")
    fig.text(0.86, 0.525,
             "Higher when the player is seen\nmore often during play.",
             fontsize=8.5, color=S.MUTED, ha="center", va="top", linespacing=1.35)
    playable = M.get("playable_s") or (M.get("end", 0) - M.get("start", 0))
    meta = [str(rec.get("date") or ""),
            f"{rec.get('game_format', '')} · {rec.get('game_type', '')}".strip(" ·"),
            f"{_mmss_game(playable)} of play analysed"]
    fig.text(0.30, 0.43, "   |   ".join(m for m in meta if m and str(m).strip()), fontsize=12, color=S.MUTED)
    if acc.get("blurb"):
        fig.text(0.30, 0.375, acc["blurb"], fontsize=10, color=S.MUTED)

    tiles = [
        ("Distance", f"{r['distance_m']:,.0f} m"),
        ("Max speed", f"{r['speed_max']:.1f} km/h" if r["speed_max"] is not None else "—"),
        ("Tracking coverage", f"{acc.get('coverage_pct', 0):.0f}%"),
    ]
    for k, (lab, val) in enumerate(tiles):
        x = 0.06 + k * 0.31
        fig.text(x, 0.185, val, fontsize=24, color="white", fontweight="bold")
        fig.text(x, 0.135, lab, fontsize=10, color="#d8f0df")
    fig.text(0.06, 0.06, f"Report created {ctx.created}", fontsize=8.5, color="#d8f0df")
    fig.text(0.94, 0.06, "Polis Walking Football · player analysis", fontsize=8.5, color="#d8f0df", ha="right")
    return fig


# ======================================================================================== 2 statistics
def statistics(fr, ctx):
    M, r = ctx.M, ctx.report
    tracks = ", ".join(f"#{t}" for t in r["tracks"]) or "—"
    fig = _player_page(fr, ctx, "Statistics", f"{ctx.team}  ·  tracks {tracks}")

    acc = r.get("accuracy") or {}
    cards = [
        ("Distance", f"{r['distance_m']:,.0f} m"),
        ("Max speed", f"{r['speed_max']:.1f} km/h" if r["speed_max"] is not None else "—"),
        ("Average speed", f"{r['speed_avg']:.1f} km/h" if r["speed_avg"] is not None else "—"),
        ("Detection confidence", f"{acc.get('band', '—')} · {acc.get('score', 0):.0f}"),
        ("Tracking coverage", f"{acc.get('coverage_pct', 0):.0f}%"),
        ("Time tracked", _mmss(r["tracked_s"])),
        ("With ball (team)", f"{r['possession_in_pct']:.0f}%"),
        ("Without ball", f"{r['possession_out_pct']:.0f}%"),
        ("Frames seen", f"{r.get('n_frames', r['n_samples']):,} / {r.get('expected_frames', '—')}"),
    ]
    for i, (lab, val) in enumerate(cards):
        col, row = i % 3, i // 3
        x = 0.06 + col * 0.30
        y = 0.66 - row * 0.14
        S.card(fig, (x, y, 0.28, 0.12), ec="none", lw=0)
        fig.text(x + 0.016, y + 0.085, lab, fontsize=10, color=S.MUTED, va="top")
        fig.text(x + 0.016, y + 0.035, val, fontsize=20, fontweight="bold", color=S.INK, va="center")

    # Brand strip above the footer (same idea as the cover) so notes never collide with it
    fig.add_artist(patches.Rectangle((0, 0.055), 1, 0.175, transform=fig.transFigure, fc=S.BRAND, ec="none", zorder=0))
    notes = [
        f"Game window {_mmss(M['start'])}–{_mmss(M['end'])} on the analysis clock"
        + (f" ({_mmss(M.get('playable_s'))} of play)." if M.get("playable_s") else "."),
        "With ball — share of this player’s tracked time while their team had possession "
        "(not how often they personally held the ball).",
        "Without ball — share of tracked time while the opposition had possession.",
        "Detection confidence mainly reflects how often this player was detected during play; "
        "missing frames lower the score. Distance follows the tracked path (jumps over 25 km/h ignored).",
    ]
    fig.text(0.06, 0.205, "Notes", fontsize=11, fontweight="bold", color="white", va="top")
    wrapped_notes = "\n".join(textwrap.fill(n, width=160) for n in notes)
    fig.text(0.06, 0.175, wrapped_notes, fontsize=8.5, color="#d8f0df", va="top", linespacing=1.35)
    return fig


# ======================================================================================== 3–5 pitch views
def _pitch_page(fr, ctx, title_tail, intro, series, empty_msg, shade_third=None):
    """shade_third: 'final' | 'middle' | 'own' — highlight that third of the attacking pitch."""
    from reports.match_report.metrics import HALF_L, HALF_W
    from reports.player_report.metrics import THIRD
    fig = _player_page(fr, ctx, title_tail, intro)
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.68])
    S.draw_pitch(ax, fill="#f3f6f1")
    if shade_third == "final":
        ax.axvspan(THIRD, HALF_L, color=COL_IN, alpha=0.08, zorder=1)
    elif shade_third == "middle":
        ax.axvspan(-THIRD, THIRD, color="#5b7c99", alpha=0.08, zorder=1)
    elif shade_third == "own":
        ax.axvspan(-HALF_L, -THIRD, color=COL_OUT, alpha=0.08, zorder=1)
    drawn = False
    for xs_ys, colour, label, size, alpha in series:
        xs, ys = xs_ys
        if not xs:
            continue
        ax.scatter(xs, ys, s=size, c=colour, alpha=alpha, linewidths=0, zorder=4, label=label)
        drawn = True
    if not drawn:
        ax.text(0, 0, empty_msg, ha="center", va="center", fontsize=12, color=S.MUTED, zorder=5)
    else:
        ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=False, edgecolor=S.FAINT)
    if ctx.report.get("coords_attacking"):
        ax.annotate("", xy=(HALF_L - 1.2, -HALF_W - 1.6), xytext=(-HALF_L + 1.2, -HALF_W - 1.6),
                    arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.2), annotation_clip=False)
        ax.text(0, -HALF_W - 2.2, "attacking direction", ha="center", va="center", fontsize=8, color=S.MUTED,
                clip_on=False)
    return fig


def _pitch_intro(ctx, extra):
    if not ctx.report.get("coords_attacking"):
        flip = "Pitch uses camera coordinates (attack direction not available). "
    elif ctx.M.get("has_half_time"):
        flip = "Pitch oriented so this team always attacks to the right (second half flipped after half-time). "
    else:
        flip = "Pitch oriented so this team attacks to the right (no half-time set, so no end-switch). "
    return flip + extra


def pitch_combined(fr, ctx):
    r = ctx.report
    di, do, du = r["dots_in"], r["dots_out"], r["dots_unk"]
    series = [
        ((di["x"], di["y"]), COL_IN, "Own team possession", 10, 0.55),
        ((do["x"], do["y"]), COL_OUT, "Opposition possession", 10, 0.55),
        ((du["x"], du["y"]), COL_UNK, "Possession unknown", 7, 0.35),
    ]
    return _pitch_page(
        fr, ctx,
        "Pitch (combined)",
        _pitch_intro(ctx, "Green = own team had the ball; orange = the other team did."),
        series,
        "No positions to show for this player.",
    )


def pitch_in_possession(fr, ctx):
    r = ctx.report
    di = r["dots_in"]
    series = [((di["x"], di["y"]), COL_IN, "Own team possession", 12, 0.65)]
    return _pitch_page(
        fr, ctx,
        "Pitch (in possession)",
        _pitch_intro(ctx, "Only moments when this player's team had the ball."),
        series,
        "No in-possession positions for this player.",
    )


def pitch_out_possession(fr, ctx):
    r = ctx.report
    do = r["dots_out"]
    series = [((do["x"], do["y"]), COL_OUT, "Opposition possession", 12, 0.65)]
    return _pitch_page(
        fr, ctx,
        "Pitch (out of possession)",
        _pitch_intro(ctx, "Only moments when the other team had the ball."),
        series,
        "No out-of-possession positions for this player.",
    )


def pitch_in_possession_final(fr, ctx):
    r = ctx.report
    d = r.get("dots_in_final") or {"x": [], "y": []}
    n = r.get("n_in_final_third", len(d["x"]))
    series = [((d["x"], d["y"]), COL_IN, f"Player ({n} samples)", 14, 0.7)]
    return _pitch_page(
        fr, ctx,
        "In possession, ball in final third",
        _pitch_intro(ctx, "Player positions only when their team had the ball and the ball was in the attacking third (shaded)."),
        series,
        "No samples with the ball in the final third while in possession.",
        shade_third="final",
    )


def pitch_in_possession_middle(fr, ctx):
    r = ctx.report
    d = r.get("dots_in_middle") or {"x": [], "y": []}
    n = r.get("n_in_middle_third", len(d["x"]))
    series = [((d["x"], d["y"]), "#3d6b8c", f"Player ({n} samples)", 14, 0.7)]
    return _pitch_page(
        fr, ctx,
        "In possession, ball in middle third",
        _pitch_intro(ctx, "Player positions when their team had the ball in the middle third (shaded)."),
        series,
        "No samples with the ball in the middle third while in possession.",
        shade_third="middle",
    )


def pitch_in_possession_own(fr, ctx):
    r = ctx.report
    d = r.get("dots_in_own") or {"x": [], "y": []}
    n = r.get("n_in_own_third", len(d["x"]))
    series = [((d["x"], d["y"]), "#6b5b95", f"Player ({n} samples)", 14, 0.7)]
    return _pitch_page(
        fr, ctx,
        "In possession, ball in own third",
        _pitch_intro(ctx, "Player positions when their team had the ball in their own defensive third (shaded)."),
        series,
        "No samples with the ball in their own third while in possession.",
        shade_third="own",
    )


def pitch_out_possession_own(fr, ctx):
    r = ctx.report
    d = r.get("dots_out_own") or {"x": [], "y": []}
    n = r.get("n_out_own_third", len(d["x"]))
    series = [((d["x"], d["y"]), COL_OUT, f"Player ({n} samples)", 14, 0.7)]
    return _pitch_page(
        fr, ctx,
        "Out of possession, ball in own third",
        _pitch_intro(ctx, "Player positions when the other team had the ball in this team’s defensive third (shaded)."),
        series,
        "No samples defending their own third out of possession.",
        shade_third="own",
    )


PAGES_PER_PLAYER = [
    cover,
    statistics,
    pitch_combined,
    pitch_in_possession,
    pitch_out_possession,
    pitch_in_possession_final,
    pitch_in_possession_middle,
    pitch_in_possession_own,
    pitch_out_possession_own,
]
