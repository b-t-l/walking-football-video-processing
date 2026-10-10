"""
compare_team_models.py  --  does the NEW team model beat the CURRENT one on a real game?

Run from the application folder (the one containing main.py), in the same python environment as main.py:

    python compare_team_models.py --game 22
    python compare_team_models.py --game 24 --frames 100
    python compare_team_models.py --game 22 --new models/teams/team-20261010-yolo26s-cls.pt

It reads a game that has already been through Detections, picks random frames (never frames whose pictures are in the
training library), cuts out every player the pipeline kept and asks BOTH models which team each one is in, exactly as the
pipeline does it (the box is squashed to 192x192). Nothing here touches your database or the pipeline.

Two ways to judge, neither needs you to label anything:
  1. IMPOSSIBLE ANSWERS. Only two teams play a game (plus a goalkeeper and a referee). Any other answer -- another team's
     name -- is certainly wrong. The model with fewer of these is the better one. Pass --teams "A,B" to say which two teams
     played, or it takes the two most common team answers.
  2. THE DISAGREEMENTS. The pictures where the two models differ are saved as a sheet (old answer on top, new answer
     below) so you can see who is right.  compare_team_models_out/game<N>_disagreements.png
Also checked: the CURRENT model re-run here is compared with what the pipeline stored in the database (team_raw). They
should agree almost completely; if they do not, this script is not showing the models the same pictures the pipeline does.

Results are also saved to compare_team_models_results.txt, and every prediction to compare_team_models_out/game<N>_predictions.csv.
"""
import os
import re
import sys
import glob
import csv
import sqlite3
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import pandas as pd

APP = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.join(os.path.dirname(APP), "_team-model-training")
LIBRARY = os.path.join(TRAIN_ROOT, "team-pictures")
OUT = os.path.join(APP, "compare_team_models_out")
RESULTS = os.path.join(APP, "compare_team_models_results.txt")
SIZE = 192
SHOW = 12                          # pictures per row on the sheet
LOG = []
ALIAS = {"Polis": "Polis Lions"}   # the old model's name -> the library's name
NOT_TEAMS = ("Goalkeeper", "Referee", "Unassigned")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def find_files(args):
    db, video = args.db, args.video
    if not db or not video:
        try:
            from game_logger import paths
            db = db or paths.detections_db(args.game)
            if not video:
                folder = paths.game_folder(args.game)
                hits = sorted(glob.glob(os.path.join(folder, "videos", "*_analysis.mp4"))) if folder else []
                video = hits[0] if hits else None
        except Exception as e:                                                   # noqa
            say("(could not use game_logger.paths:", e, ")")
    if not db or not os.path.isfile(db):
        sys.exit(f"Cannot find the detections database for game {args.game}. Pass --db /path/to/..._detections.db")
    if not video or not os.path.isfile(video):
        sys.exit("Cannot find the analysis video. Pass --video /path/to/..._analysis.mp4")
    return db, video


def newest_new_model():
    c = sorted(glob.glob(os.path.join(APP, "models", "teams", "team-*.pt")), key=os.path.getmtime)
    return c[-1] if c else None


def frames_in_library(game):
    """frame ids of this game whose pictures are already in the training library (made by the Team training images step)"""
    used = set()
    for f in glob.glob(os.path.join(LIBRARY, "*", f"{int(game):04d}_f*")):
        m = re.match(r"^\d{4}_f(\d+)_", os.path.basename(f))
        if m:
            used.add(int(m.group(1)))
    return used


def sample_crops(db, video, n_frames, seed, exclude):
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("pragma table_info(detected_objects)")}
    raw = "team_raw" if "team_raw" in cols else "team"
    d = pd.read_sql(f"select frame_id,xmin,ymin,xmax,ymax,confidence,tracker_id,class_name,{raw} as db_team from detected_objects "
                    "where class_name!='ball' and confidence>=0.6", con)
    frames = np.array([f for f in np.sort(d.frame_id.unique()) if int(f) not in exclude])
    rng = np.random.default_rng(seed)
    n = min(n_frames, len(frames))
    picks = sorted(int(x) for x in rng.choice(frames, size=n, replace=False))
    cap = cv2.VideoCapture(video)
    crops, meta = [], []
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, im = cap.read()
        if not ok:
            continue
        H, W = im.shape[:2]
        for r in d[d.frame_id == fi].itertuples():
            x0, y0, x1, y1 = max(0, int(round(r.xmin))), max(0, int(round(r.ymin))), min(W, int(round(r.xmax))), min(H, int(round(r.ymax)))
            if x1 - x0 < 12 or y1 - y0 < 24:
                continue
            crops.append(im[y0:y1, x0:x1])
            meta.append((fi, r.tracker_id, r.class_name, r.db_team, x1 - x0, y1 - y0))
    cap.release()
    M = pd.DataFrame(meta, columns="frame_id tracker_id detector_class db_team w h".split())
    return crops, M


def predict(weights, crops, device, alias):
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    squashed = [cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for c in crops]      # as the pipeline does
    out, conf = [], []
    for i in range(0, len(squashed), 64):
        for r in model.predict(squashed[i:i + 64], imgsz=SIZE, device=device, verbose=False):
            n = names[int(r.probs.top1)]
            out.append(alias.get(n, n))
            conf.append(float(r.probs.top1conf))
    return np.array(out, dtype=object), np.array(conf)


def tile_sheet(crops, top, bottom, cols=SHOW):
    w, h, cap = 48, 96, 30
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (h + cap), cols * w, 3), 30, np.uint8)
    short = lambda s: {"Aphrodite Wanderers": "Aphrod", "Goalkeeper": "GK", "Referee": "Ref", "Polis Lions": "Polis", "West Coast": "WestC"}.get(s, str(s)[:7])
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        sheet[r * (h + cap):r * (h + cap) + h, k * w:(k + 1) * w] = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, "old " + short(top[i]), (k * w + 1, r * (h + cap) + h + 11), cv2.FONT_HERSHEY_SIMPLEX, .27, (200, 200, 200), 1)
        cv2.putText(sheet, "NEW " + short(bottom[i]), (k * w + 1, r * (h + cap) + h + 24), cv2.FONT_HERSHEY_SIMPLEX, .27, (0, 255, 255), 1)
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--frames", type=int, default=60, help="how many random frames to test (default 60, about 700 players)")
    ap.add_argument("--new", help="the new weights (default: the newest models/teams/team-*.pt)")
    ap.add_argument("--old", default=os.path.join(APP, "models", "teams", "best-v7-192x192.pt"))
    ap.add_argument("--teams", help='the two teams that played, e.g. "Polis Lions,Aphrodite Wanderers"')
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device")
    ap.add_argument("--db"), ap.add_argument("--video")
    args = ap.parse_args()

    new_w = args.new or newest_new_model()
    if not new_w or not os.path.isfile(new_w):
        sys.exit("No new weights found. Train one first (train_team_model.py) or pass --new <file>")
    if not os.path.isfile(args.old):
        sys.exit(f"Current weights not found: {args.old}")
    try:
        import torch
        device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    except ImportError:
        sys.exit("Run this in the python environment you use for main.py (it needs torch and ultralytics)")

    db, video = find_files(args)
    exclude = frames_in_library(args.game)
    say(f"game {args.game}: {db}\n  new model:     {os.path.basename(new_w)}\n  current model: {os.path.basename(args.old)}   device: {device}")
    say(f"  skipping {len(exclude)} frames whose pictures are in the training library")
    crops, M = sample_crops(db, video, args.frames, args.seed, exclude)
    say(f"  {len(M)} player pictures from {M.frame_id.nunique()} random frames\n")

    old, old_c = predict(args.old, crops, device, ALIAS)
    new, new_c = predict(new_w, crops, device, {})
    M["old_model"], M["new_model"], M["old_conf"], M["new_conf"] = old, new, old_c.round(3), new_c.round(3)
    db_lab = np.array([ALIAS.get(x, x) for x in M.db_team.fillna("Unassigned")], dtype=object)

    # ---- sanity: the re-run of the current model must match what the pipeline stored
    same = float(np.mean(old == db_lab))
    say(f"SANITY: the current model, re-run here, gives the same answer as the pipeline stored for {same:.1%} of pictures")
    say("        (should be above ~95%; much lower means the pictures here are not being prepared like the pipeline does it)\n")

    # ---- the two teams
    if args.teams:
        teams = [t.strip() for t in args.teams.split(",")]
    else:
        vc = pd.Series([x for x in old if x not in NOT_TEAMS]).value_counts()
        teams = list(vc.index[:2])
    allowed = set(teams) | set(NOT_TEAMS)
    say(f"teams that played: {teams}   (any other team name as an answer is impossible)\n")

    say(f"{'answer':<24}{'current model':>15}{'NEW model':>12}")
    labels = sorted(set(old) | set(new), key=lambda x: (x not in teams, x))
    for lab in labels:
        flag = "   <- impossible" if lab not in allowed else ""
        say(f"{lab:<24}{int(np.sum(old == lab)):>15}{int(np.sum(new == lab)):>12}{flag}")
    imp_old, imp_new = int(np.sum([x not in allowed for x in old])), int(np.sum([x not in allowed for x in new]))
    say(f"\nIMPOSSIBLE answers: current model {imp_old} ({imp_old / len(M):.1%})   NEW model {imp_new} ({imp_new / len(M):.1%})")
    both_teams = (np.isin(old, teams)) & (np.isin(new, teams))
    say(f"the two models differ on {int(np.sum(old != new))} of {len(M)} pictures ({np.mean(old != new):.1%}); "
        f"between the two real teams on {int(np.sum(both_teams & (old != new)))}\n")

    say("current model -> NEW model, where they differ:")
    diff = pd.crosstab(M.old_model[old != new], M.new_model[old != new])
    say(diff.to_string() if len(diff) else "  (they never differ)")

    # ---- files
    os.makedirs(OUT, exist_ok=True)
    M.drop(columns=["w", "h"]).to_csv(os.path.join(OUT, f"game{args.game}_predictions.csv"), index=False)
    idx = np.where(old != new)[0]
    rng = np.random.default_rng(args.seed)
    if len(idx):
        sel = np.sort(rng.choice(idx, size=min(96, len(idx)), replace=False))
        cv2.imwrite(os.path.join(OUT, f"game{args.game}_disagreements.png"), tile_sheet([crops[i] for i in sel], old[sel], new[sel]))
    say(f"\nsheet of the differences: {os.path.join(OUT, f'game{args.game}_disagreements.png')}  (old answer grey, NEW answer yellow)")
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
