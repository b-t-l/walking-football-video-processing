"""The numbers behind the opposition report: for each team, how it plays with and without the ball, and where it is
strong or weak compared with the other team in the same game.

Everything is looked at from the team's own point of view: positions are turned round so that each team always attacks
from left to right (u = distance along the pitch towards the goal it attacks, 0 at halfway, -18 at its own goal line;
v = across the pitch, positive on the team's own left). Which way a team attacks is worked out from where the two teams
stand in each half (the team standing further right is defending the right-hand goal)."""
import math

import numpy as np
import pandas as pd

from ..match_report import metrics
from ..match_report.metrics import HALF_L, HALF_W

THIRD = HALF_L / 3.0            # 6 m: the edge of the final third (u >= 6) and of the own third (u <= -6)
MIN_PHASE_S = 2.0               # a spell shorter than this is a flicker of the ball detection, not real possession
MERGE_GAP_S = 4.0               # two spells by the same team this close together are one spell of possession
BREAK_S = 10.0                  # won the ball and reached the final third within this many seconds = a fast break


def _periods(players, teams, start, end, half_time):
    """[(t0, t1, direction of team A)] - the direction team A attacks (+1 = towards +x) in each half."""
    bounds = [(start, end)]
    if half_time and half_time[0] is not None and half_time[1] is not None and half_time[1] > half_time[0]:
        bounds = [(start, half_time[0]), (half_time[1], end)]
    out = []
    for a, b in bounds:
        seg = players[(players["t"] >= a) & (players["t"] <= b)]
        xa = seg[seg.team == teams[0]]["x"].mean()
        xb = seg[seg.team == teams[1]]["x"].mean()
        d = 1 if (np.isnan(xa) or np.isnan(xb) or xa <= xb) else -1       # the team standing further right defends the right
        out.append((a, b, d))
    return out


def _dir(t, periods, i):
    t = np.asarray(t, dtype=float)
    d = np.full(len(t), periods[0][2] if i == 0 else -periods[0][2], dtype=float)
    for a, b, da in periods:
        d[(t >= a) & (t <= b)] = da if i == 0 else -da
    return d


def _phases(spells):
    """Real spells of possession: flickers dropped, and neighbouring spells of the same team joined together."""
    real = sorted([s for s in spells if s["len"] >= MIN_PHASE_S], key=lambda s: s["t"])
    out = []
    for s in real:
        if out and out[-1]["team"] == s["team"] and s["t"] - (out[-1]["t"] + out[-1]["len"]) <= MERGE_GAP_S:
            out[-1]["len"] = s["t"] + s["len"] - out[-1]["t"]
        else:
            out.append(dict(s))
    return out


def _fmt_time(s):
    return metrics.mmss(s)


def scout(db_path, record, teams, running_speed_kmh=15.0):
    M, R = metrics.compute_metrics(db_path, record, teams, running_speed_kmh, return_raw=True)
    start, end, step, owner = R["start"], R["end"], R["step"], R["owner"]
    pl, ball = R["players"].copy(), R["ball"].copy()
    periods = _periods(pl, teams, start, end, R["half_time"])
    n_own = len(owner)
    pl["own"] = owner[np.clip(np.round((pl["t"].to_numpy() - start) / step).astype(int), 0, n_own - 1)] if n_own else -1
    if len(ball):
        ball["own"] = owner[np.clip(np.round((ball["t"].to_numpy() - start) / step).astype(int), 0, n_own - 1)]
        ball = ball.sort_values("t").reset_index(drop=True)
    spells = M["spells"]
    S = {"teams": list(teams), "periods": [[a, b, d] for a, b, d in periods], "per_team": {}}

    for i, tm in enumerate(teams):
        o = 1 - i
        opp = teams[o]
        T = {}
        # ---- turn this team's data round so it attacks left to right
        mine = pl[pl["team"] == tm].copy()
        d = _dir(mine["t"].to_numpy(), periods, i)
        mine["u"], mine["v"] = mine["x"].to_numpy() * d, mine["y"].to_numpy() * d
        if len(ball):
            bd = _dir(ball["t"].to_numpy(), periods, i)
            ball["u"], ball["v"] = ball["x"].to_numpy() * bd, ball["y"].to_numpy() * bd
        # ---- shape in and out of possession
        rows = []
        for fid, g in mine.groupby("frame_id"):
            if len(g) < metrics.MIN_PLAYERS_FOR_SHAPE:
                continue
            u, v = g["u"].to_numpy(), g["v"].to_numpy()
            rows.append((float(g["t"].iloc[0]), int(g["own"].iloc[0]), u.mean(), u.max() - u.min(), v.max() - v.min(), metrics._hull_area(np.c_[u, v])))
        sh = pd.DataFrame(rows, columns=["t", "own", "cu", "length", "width", "hull"])
        T["hull_without_blocks"] = []
        for ba, bb in M["blocks"]:
            seg = sh[(sh["own"] == o) & (sh["t"] >= ba) & (sh["t"] < bb)]
            T["hull_without_blocks"].append(float(seg["hull"].mean()) if len(seg) >= 10 else None)
        for name, who in (("with", i), ("without", o)):
            s = sh[sh["own"] == who]
            T[f"shape_{name}"] = {k: (float(s[k].mean()) if len(s) else None) for k in ("cu", "length", "width", "hull")} | {"frames": int(len(s))}
        # ---- heat maps (turned round)
        T["heat_with"] = metrics._heat(mine[mine["own"] == i]["u"].to_numpy(), mine[mine["own"] == i]["v"].to_numpy(), 1.5).tolist() if (mine["own"] == i).sum() > 20 else None
        T["heat_without"] = metrics._heat(mine[mine["own"] == o]["u"].to_numpy(), mine[mine["own"] == o]["v"].to_numpy(), 1.5).tolist() if (mine["own"] == o).sum() > 20 else None
        bw = ball[ball["own"] == i] if len(ball) else ball
        T["heat_ball_with"] = metrics._heat(bw["u"].to_numpy(), bw["v"].to_numpy(), 1.5).tolist() if len(bw) > 15 else None
        # ---- where the team plays when it has the ball: left / centre / right of the pitch, from its own point of view
        if len(bw):
            v = bw["v"].to_numpy()
            T["flank_with"] = [float((v > HALF_W / 3).mean() * 100), float((np.abs(v) <= HALF_W / 3).mean() * 100), float((v < -HALF_W / 3).mean() * 100)]
            T["ball_u_with"] = float(bw["u"].mean())
        else:
            T["flank_with"], T["ball_u_with"] = None, None
        # ---- space given to the ball carrier when the opponent has the ball
        space = None
        if len(ball):
            bo = ball[ball["own"] == o]
            if len(bo):
                by_frame = {k: g[["x", "y"]].to_numpy() for k, g in pl[pl["team"] == tm].groupby("frame_id")}
                ds = []
                for fid, bx, by in bo[["frame_id", "x", "y"]].itertuples(index=False):
                    g = by_frame.get(fid)
                    if g is not None and len(g):
                        ds.append(min(float(np.hypot(g[:, 0] - bx, g[:, 1] - by).min()), 12.0))
                space = float(np.mean(ds)) if len(ds) >= 10 else None
                T["space_n"] = len(ds)
        T["space"] = space
        # ---- spells: what happened to the ball in each one
        all_sp = _phases(spells)
        mine_sp = [s for s in all_sp if s["team"] == i]
        bt = ball["t"].to_numpy() if len(ball) else np.array([])
        n_checked = reached_final = reached_half = 0
        lost_own, lost_total, won_total, won_opp_half, won_att_third, breaks = 0, 0, 0, 0, 0, 0
        losses = []                                   # (time, u) every time the opponent took the ball
        wins = []                                     # (time, u, v) every time this team took it
        for k, s in enumerate(all_sp):
            if s["team"] != i:
                continue
            t0_, t1_ = s["t"], s["t"] + s["len"]
            a, b = np.searchsorted(bt, t0_), np.searchsorted(bt, t1_, side="right")
            seg = ball.iloc[a:b] if len(bt) else None
            if seg is not None and len(seg):
                n_checked += 1
                mu = float(seg["u"].max())
                reached_final += mu >= THIRD
                reached_half += mu >= 0
            prev_s = all_sp[k - 1] if k > 0 else None
            next_s = all_sp[k + 1] if k + 1 < len(all_sp) else None
            if prev_s is not None and prev_s["team"] == o and s["t"] - (prev_s["t"] + prev_s["len"]) <= MERGE_GAP_S and seg is not None and len(seg):
                won_total += 1
                u0 = float(seg["u"].iloc[0])
                wins.append((float(seg["t"].iloc[0]), u0, float(seg["v"].iloc[0])))
                won_opp_half += u0 > 0
                won_att_third += u0 >= THIRD
                quick = seg[(seg["t"] - t0_ <= BREAK_S) & (seg["u"] >= THIRD)]
                breaks += len(quick) > 0 and u0 < THIRD
            if next_s is not None and next_s["team"] == o and next_s["t"] - t1_ <= MERGE_GAP_S and seg is not None and len(seg):
                lost_total += 1
                ul = float(seg["u"].iloc[-1])
                losses.append((float(seg["t"].iloc[-1]), ul, float(seg["v"].iloc[-1])))
                lost_own += ul <= -THIRD
        T["spell_n"] = len(mine_sp)
        T["spells_checked"] = n_checked
        T["final_third_pct"] = 100 * reached_final / n_checked if n_checked else None
        T["half_pct"] = 100 * reached_half / n_checked if n_checked else None
        T["won"], T["won_opp_half"], T["won_att_third"], T["fast_breaks"] = won_total, int(won_opp_half), int(won_att_third), int(breaks)
        T["lost"], T["lost_own_third"] = lost_total, int(lost_own)
        T["wins"], T["losses"] = wins, losses
        T["share"] = M["possession"]["share"][i]
        T["avg_spell"] = M["possession"]["avg_spell_s"][i]
        # ---- energy
        blocks_d = M["distance"]["blocks"][tm]
        T["dist_pp"] = M["distance"]["per_player_m"][tm]
        T["speed_avg"] = M["speed"]["avg"][tm]
        T["speed_p95"] = M["speed"]["p95"][tm]
        T["speed_blocks"] = M["speed"]["blocks"][tm]
        T["dist_blocks"] = blocks_d
        T["running"] = M["running"]["count"][tm]
        T["running_blocks"] = M["running"]["blocks"][tm]
        sb = [v for v in T["speed_blocks"]]
        valid = [v for v in sb if v is not None]
        T["fade"] = None
        if len(valid) >= 6:
            k = max(2, len(valid) // 3)
            first, last = float(np.mean(valid[:k])), float(np.mean(valid[-k:]))
            T["fade"] = (last / first - 1) * 100 if first else None
        T["possession_blocks"] = [b["share"][i] for b in M["possession_blocks"]]
        S["per_team"][tm] = T

    S["M"] = M
    S["coverage"] = M["coverage"]
    S["items"] = _items(S)
    return S


# ------------------------------------------------------------------------------------------------ strengths and weaknesses

def _g(S, i, path):
    t = S["per_team"][S["teams"][i]]
    cur = t
    for p in path.split("."):
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


# key, label, value path, higher is better, relative margin, absolute margin, format, unit, texts
ITEM_DEFS = [
    ("poss", "Possession", "share", True, 0.10, 6.0, "{:.0f}%", "",
     "They keep the ball well: {a} of the possession we could measure, against {b} for {opp}.",
     "They see little of the ball: {a} of the possession we could measure, against {b} for {opp}.",
     "Expect them to dominate the ball. Stay compact, be patient, and pick the moment to press.",
     "They are happy without the ball. Win it and keep it - make them run after it."),
    ("spell", "Time on the ball per spell", "avg_spell", True, 0.20, 2.0, "{:.1f} s", "",
     "They hold the ball for longer spells: {a} on average, against {b} for {opp}.",
     "They lose the ball quickly: spells last {a} on average, against {b} for {opp}.",
     "Their spells are long, so shut passing lanes early instead of chasing the ball.",
     "Their spells are short. Press the moment they get it and they give it back."),
    ("won", "Ball won from the opponent", "won", True, 0.30, 2, "{:.0f}", "",
     "They win the ball back well: {a} turnovers won, against {b} for {opp}.",
     "They rarely win the ball from the opponent: {a} times, against {b} for {opp}.",
     "They win the ball back often, so protect it when you get it and do not dwell on it.",
     "They do not win the ball back much: you can keep the ball and be patient."),
    ("final", "Spells that reach the final third", "final_third_pct", True, 0.20, 8.0, "{:.0f}%", "",
     "They get forward: {a} of their spells reach the final third, against {b} for {opp}.",
     "They struggle to reach the final third: only {a} of their spells get there, against {b} for {opp}.",
     "They reach the final third often. Make sure the area in front of your goal is protected.",
     "They find it hard to get into the final third. Drop in, stay tight and let them pass it round the outside."),
    ("lost_own", "Ball lost in their own third", "lost_own_third", False, 0.30, 2, "{:.0f}", "",
     "They seldom lose the ball in their own third: {a} times, against {b} for {opp}.",
     "They lose the ball in their own third {a} times, against {b} for {opp}.",
     "They are careful at the back, so pressing high will not pay off often.",
     "Press them high: they give the ball away in their own third, close to their goal."),
    ("won_high", "Ball won in the opponent's half", "won_opp_half", True, 0.30, 2, "{:.0f}", "",
     "They win the ball high up the pitch: {a} times in the opponent's half, against {b} for {opp}.",
     "They hardly ever win the ball in the opponent's half: {a} times, against {b} for {opp}.",
     "They win it high up the pitch. Move the ball quickly out of your own half.",
     "They do not win the ball high up the pitch, so you can build from the back without being hassled."),
    ("fast", "Fast breaks", "fast_breaks", True, 0.40, 2, "{:.0f}", "",
     "They are dangerous on the break: {a} fast breaks (won the ball and reached the final third within 10 s), against {b} for {opp}.",
     "They are little threat on the break: {a} fast breaks, against {b} for {opp}.",
     "Do not leave yourselves short at the back when you lose the ball: they hit quickly.",
     "They are not a threat on the break, so you can commit players forward."),
    ("compact", "Area covered when defending", "shape_without.hull", False, 0.15, 5.0, "{:.0f} m²", "",
     "They are compact when defending: the five players cover {a}, against {b} for {opp}.",
     "They are stretched when defending: the five players cover {a}, against {b} for {opp}.",
     "Hard to play through the middle. Use the width and switch the ball quickly.",
     "There are gaps between their players when they defend. Passes through the middle can get through."),
    ("space", "Space given to the ball carrier", "space", False, 0.20, 0.7, "{:.1f} m", "",
     "They close the ball down quickly: the nearest defender is {a} from the ball carrier on average, against {b} for {opp}.",
     "They give the ball carrier room: the nearest defender is {a} away on average, against {b} for {opp}.",
     "They close down fast. Play one touch where you can and move the ball before they arrive.",
     "They give the player on the ball time and space. Take your time and look up."),
    ("width", "Width when attacking", "shape_with.width", True, 0.12, 1.0, "{:.1f} m", "",
     "They stretch the pitch when attacking: {a} wide on average, against {b} for {opp}.",
     "They are narrow when attacking: {a} wide on average, against {b} for {opp}.",
     "They use the full width. Do not let the wide players get isolated.",
     "They attack narrowly. Block the middle and they have little else."),
    ("dist", "Distance covered per player", "dist_pp", True, 0.05, 25, "{:,.0f} m", "",
     "They do the work: {a} covered per player, against {b} for {opp}.",
     "They cover less ground: {a} per player, against {b} for {opp}.",
     "They work hard. Play the ball quickly and let it do the running.",
     "They do less running than you, so tempo and movement will tell over time."),
    ("fade", "Intensity late in the game", "fade", True, 0.0, 8.0, "{:+.0f}%", "",
     "They keep their intensity up late on: average speed changes {a} from the start to the end, against {b} for {opp}.",
     "They tire late on: average speed changes {a} from the start to the end, against {b} for {opp}.",
     "They keep going to the end, so do not rely on them fading.",
     "They tire late on. Push the pace in the last third of the game."),
    ("running", "Running above the limit", "running", False, 0.30, 2, "{:.0f}", "",
     "They stay within walking pace: running above the limit {a}, against {b} for {opp}.",
     "They run above walking pace: running above the limit {a}, against {b} for {opp}. Likely to give away free kicks.",
     "They stay within the rules, so do not expect free kicks for running.",
     "They run more than the rules allow. Play quickly and they will chase and concede free kicks."),
    ("pace", "Top speed (95th percentile)", "speed_p95", True, 0.08, 1.0, "{:.1f} km/h", "",
     "They are quick over short distances: top speed {a}, against {b} for {opp}.",
     "They are slower over short distances: top speed {a}, against {b} for {opp}.",
     "They are quick when they move. Do not give them space to run into.",
     "They are not quick. Balls over the top and quick switches should work."),
]


def _items(S):
    out = {tm: [] for tm in S["teams"]}
    for i, tm in enumerate(S["teams"]):
        o = 1 - i
        for key, label, path, hb, rel, absm, fmt, unit, strong_t, weak_t, beware, exploit in ITEM_DEFS:
            a, b = _g(S, i, path), _g(S, o, path)
            if a is None or b is None:
                continue
            diff = (a - b) if hb else (b - a)                          # positive = better than the opponent
            base = max(abs(a), abs(b), 1e-9)
            if abs(diff) < absm or abs(diff) / base < rel:
                continue
            score = min(abs(diff) / base, 1.5)
            txt = dict(a=fmt.format(a), b=fmt.format(b), opp=S["teams"][o])
            kind = "strong" if diff > 0 else "weak"
            out[tm].append({"key": key, "label": label, "kind": kind, "score": float(score),
                            "text": (strong_t if diff > 0 else weak_t).format(**txt),
                            "tip": beware if diff > 0 else exploit, "a": a, "b": b})
        out[tm].sort(key=lambda r: -r["score"])
    return out


def style_lines(S, tm):
    """A few plain descriptors of how the team plays, compared with the other team."""
    i = S["teams"].index(tm)
    o = 1 - i
    me, op = S["per_team"][tm], S["per_team"][S["teams"][o]]
    out = []
    pd_ = me["share"] - op["share"]
    out.append(("Possession", "Mostly has the ball" if pd_ > 6 else "Lets the other team have it" if pd_ < -6 else "Shares the ball evenly",
                f"{me['share']:.0f}% of possession"))
    cu, cu2 = me["shape_without"]["cu"], op["shape_without"]["cu"]
    if cu is not None and cu2 is not None:
        out.append(("Defending", "Defends higher up the pitch" if cu - cu2 > 1.5 else "Sits deeper when defending" if cu2 - cu > 1.5 else "Defends at a similar height",
                    f"centre of the team {cu:+.1f} m from halfway"))
    w, w2 = me["shape_with"]["width"], op["shape_with"]["width"]
    if w is not None and w2 is not None:
        out.append(("Attacking shape", "Plays wide" if w - w2 > 1 else "Plays narrow" if w2 - w > 1 else "Similar width to the opponent",
                    f"{w:.1f} m wide on average"))
    fl = me["flank_with"]
    if fl:
        k = int(np.argmax(fl))
        out.append(("Favoured side", ["Attacks down their left", "Attacks through the middle", "Attacks down their right"][k],
                    f"{fl[0]:.0f}% left · {fl[1]:.0f}% centre · {fl[2]:.0f}% right of the ball time"))
    return out


def plan(S, tm, total=4):
    """Up to four pointers: mostly ways to exploit the weak points, with at least one warning about a strength."""
    items = S["items"][tm]
    weak_all = [r for r in items if r["kind"] == "weak"]
    strong_all = [r for r in items if r["kind"] == "strong"]
    n_strong = min(len(strong_all), max(1, total - len(weak_all)) if strong_all else 0, 2)
    weak = weak_all[:max(total - n_strong, 0)]
    return weak, strong_all[:n_strong]
