"""Turns the measurements into a short, ordered list of things to try. Every item says what was seen, the likely cause, what to change
and which number to watch when the report is run again. The settings named here are the ones in this project's own code."""
from . import analysis as an


def _cap(s):
    return s[:1].upper() + s[1:]


def _p(v, nd=0):
    return f"{v:.{nd}f}%"


def advise(A):
    out = []
    by_stage = {r["stage"]: r for r in A["ranking"]}
    B, P, G, T, K, X = A["ball"], A["players"], A["goalkeepers"], A["tracking"], A["teams_check"], A["positions"]

    # ------------------------------------------------------------------ ball
    if A["scores"]["ball"] < an.GOOD:
        why = []
        if B["spots"]:
            sp = B["spots"][0]
            why.append(f"a fixed spot at about ({sp['x']:.0f} m, {sp['y']:.0f} m) kept being reported as the ball ({B['false_spot_hits']} sightings in all); it is removed here but each one used up a frame's ball")
        if B["outside_pct"] > 15:
            why.append(f"{_p(B['outside_pct'])} of ball detections were outside the pitch (netting, bags, spectators)")
        if B["conf_low_pct"] > 15:
            why.append(f"{_p(B['conf_low_pct'])} of sightings are weak (confidence under 0.45), so many real ones may be falling under the cut-off of {B['conf_cut']:.2f}")
        why.append(f"for {_p(B['gap_over5_pct'])} of the game the ball was out of sight for 5 seconds or more")
        out.append({"stage": "ball", "title": "Find the ball more often",
                    "saw": f"The ball was placed in {_p(B['coverage'] * 100)} of frames (the reports work well from about {_p(an.TARGET_BALL_COVERAGE * 100)}).",
                    "why": ("; ".join(why) + ".").capitalize() if False else _cap("; ".join(why) + "."),
                    "try": ["Compare the newer ball model (models/ball/yolo26-2026-10-09-v2.pt) with the one in use (best-v2-1920x1920.pt) using compare_ball_models.py, on a few minutes of this game.",
                            "Run the ball model on overlapping tiles of the pitch instead of one resized frame; the ball is only a few pixels wide.",
                            f"Lower the ball confidence from {B['conf_cut']:.2f} to about 0.20 in data/create_database.py, together with dropping any detection outside the pitch and any spot that stays in place for minutes, so extra sightings are real ones.",
                            "Fill gaps of up to a second by joining the ball's path between sightings (needs no new detection run)."],
                    "watch": f"Share of frames with the ball (now {_p(B['coverage'] * 100)}) and the 'ball placed' figure on the match report's possession page."})
    # ------------------------------------------------------------------ tracking
    if A["scores"]["tracking"] < an.GOOD:
        out.append({"stage": "tracking", "title": "Keep each player's identity for longer",
                    "saw": f"{T['tracker_ids']} track numbers were made for about {T['expected']} players; stitching joined them into {T['player_ids']} identities. The middle identity lasts {T['median_life_s']:.0f} seconds, and {_p(T['long_share'])} of player time sits on identities of a minute or more.",
                    "why": f"{_p(T['ends_midpitch_pct'])} of tracks end in the middle of the pitch rather than at an edge, which is a player lost behind someone else; {_p(T['short_ids_pct'])} of the identities last under 5 seconds.",
                    "try": ["Raise the stitcher's reach in track_stitcher/track_stitcher.py: LINK_MAX_GAP_S from 6 to 10 and MAX_LINK_COST from 2.5 to 3.5, then check the jump rate does not rise.",
                            "Lengthen the tracker's memory: lost_track_buffer in data/create_database.py is 90 (3 seconds); try 150.",
                            "Find more of the missed players first (see player detection): a player who is detected every frame is much easier to keep."],
                    "watch": f"Identities per game (now {T['player_ids']}) and the steady-identity share (now {_p(T['long_share'])})."})
    # ------------------------------------------------------------------ goalkeepers
    if A["scores"]["goalkeepers"] < an.GOOD:
        weak, strong = ("right", "left") if G["right_pct"] < G["left_pct"] else ("left", "right")
        lm, rm = G.get("per_minute_left") or [], G.get("per_minute_right") or []
        both_m = [(l, r) for l, r in zip(lm, rm) if l is not None and r is not None]
        one_at_a_time = bool(both_m) and sum(1 for l, r in both_m if (l > 0.3 and r < 0.15) or (r > 0.3 and l < 0.15)) / len(both_m) > 0.5
        out.append({"stage": "goalkeepers", "title": "Find the goalkeepers",
                    "saw": f"A goalkeeper was found at the left end in {_p(G['left_pct'])} of frames and at the right end in {_p(G['right_pct'])}. Both were seen together in {_p(G['both_pct'])}.",
                    "why": ("Only one goalkeeper is found at a time: the same goalkeeper turns up at the left end in one half and the right end in the other, so the other team's goalkeeper is probably never recognised (often a kit that looks like an outfield player). " if one_at_a_time else "") + (f"{_p(G['far_from_goal_pct'])} of goalkeeper boxes are away from both goals (see the map), so some are probably other people at the edge of the pitch. " if G["far_from_goal_pct"] > 20 else "") + f"The {weak} end is found much less than the {strong} end. {G['as_player_class']:,} goalkeeper-labelled boxes came from the player class, so the goalkeeper class is missing many of them." if abs(G["left_pct"] - G["right_pct"]) > 15 else f"The goalkeeper class finds few goalkeepers at either end, and {G['as_player_class']:,} goalkeeper-labelled boxes came from the player class instead.",
                    "try": ["Treat the player who stays within a few metres of each goal line as that end's goalkeeper (a rule on positions, applied after detection).",
                            "Add frames from this game's goalkeepers to the training set for the objects model, then compare with compare_player_models.py.",
                            "Lower the goalkeeper-class confidence on its own, since goalkeepers are rarely mistaken for anything else."],
                    "watch": f"Goalkeeper found at each end (now {_p(G['left_pct'])} / {_p(G['right_pct'])})."})
    # ------------------------------------------------------------------ players
    if A["scores"]["players"] < an.GOOD or P["conf_near_cut_pct"] > 12:
        why = [f"players are only kept when the model is at least {P['conf_cut']:.2f} sure, and {_p(P['conf_near_cut_pct'])} of the boxes kept sit within 0.10 of that cut-off, so more are likely just below it"]
        if P["dropout_pct"] > 4:
            why.append(f"{_p(P['dropout_pct'], 1)} of the moments inside a player's track had no detection (about {P['dropout_n']:,})")
        if P["dropout_far_side_pct"] is not None and P["dropout_far_side_pct"] > 38:
            why.append(f"{_p(P['dropout_far_side_pct'])} of those misses are on the far third of the pitch, where players are smallest")
        out.append({"stage": "players", "title": "Find every player in every frame",
                    "saw": f"On average {P['per_frame_mean']:.1f} of 10 players were found per frame; all 10 in only {_p(P['all_found_pct'])} of frames.",
                    "why": _cap("; ".join(why) + "."),
                    "try": [f"Lower the player confidence in data/create_database.py from {P['conf_cut']:.2f} to about 0.45 and re-run detection on a few minutes; keep it only if the review sheet shows few extra wrong boxes.",
                            "Fill short gaps within a track by joining the positions before and after (needs no new detection run).",
                            "If the far side stays weak, run the player model on a zoomed crop of the far third as well as the whole frame."],
                    "watch": f"Players found per frame (now {P['per_frame_mean']:.1f}) and frames with all 10 (now {_p(P['all_found_pct'])})."})
    # ------------------------------------------------------------------ teams
    if A["scores"]["teams"] < an.GOOD or K["gk_labelled_players_pct"] > 2 or K["noise_pct"] > 0.5:
        bits = []
        if K["gk_labelled_players_pct"] > 2:
            bits.append(f"{_p(K['gk_labelled_players_pct'], 1)} of player boxes were labelled 'Goalkeeper' by the team model")
        if K["noise_n"]:
            bits.append(f"{K['noise_n']:,} boxes got a label that is neither team ({', '.join(list(K['noise_labels'])[:3])})")
        bits.append(f"{_p(K['flip_pct'], 1)} of labels disagree with the player's final team")
        out.append({"stage": "teams", "title": "Tidy up the team labels",
                    "saw": f"Team labels agree with the final team for {_p(K['agree_pct'])} of player boxes.",
                    "why": "; ".join(bits) + ".",
                    "try": ["Map any label that is not one of this game's two teams to whichever of the two kits it is closest to, before the stitching step.",
                            "Never accept 'Goalkeeper' for a player who is far from both goals; take the identity's majority team instead."],
                    "watch": f"Label agreement (now {_p(K['agree_pct'])}) and boxes labelled 'Goalkeeper' (now {_p(K['gk_labelled_players_pct'], 1)})."})
    # ------------------------------------------------------------------ positions
    if A["scores"]["positions"] < an.GOOD:
        out.append({"stage": "positions", "title": "Check pitch positions",
                    "saw": f"{_p(X['inside_pct'])} of detections are inside the pitch; {_p(X['jump_pct'], 1)} of moves between detections of one identity are faster than {an.JUMP_KMH:.0f} km/h.",
                    "why": "Jumps usually mean a wrong identity or a badly placed foot point; boxes outside the pitch can be people on the sideline.",
                    "try": ["Re-check the calibration points in the Pitch Calibrator against the annotated video.", "Use the box's bottom-centre as the foot point on the far side."],
                    "watch": f"Jump rate (now {_p(X['jump_pct'], 1)})."})
    rank = {r["stage"]: i for i, r in enumerate(A["ranking"])}
    out.sort(key=lambda a: rank[a["stage"]])
    for a in out:
        a["gain"] = by_stage[a["stage"]]["gain"]
    return out
