"""
compare_team_methods.py  --  can players be put into teams WITHOUT training a classifier for every team?

Run from the application folder (the one containing main.py), in the same python environment as main.py:

    python compare_team_methods.py --game 24                       # DINOv2 embeddings (downloads ~85 MB once)
    python compare_team_methods.py --game 22 --embed colour        # no downloads, colour-only baseline
    python compare_team_methods.py --game 24 --embed dinov2,resnet,colour --bursts 16

Idea being tested
  Your current model (models/teams) is a YOLO classifier trained on crops of the teams it knows. A new team means
  new training. The alternative: take a general-purpose image model that has never seen your teams, turn each player's
  shirt into a vector (an "embedding"), and simply group the vectors of one game into clusters -- two teams plus the
  goalkeeper/referee. No training, so a new team costs nothing.

What it does (nothing here touches the database or the pipeline)
  * samples BURSTS of consecutive detection frames from the game (so the same player is seen several times),
    cuts out each player box that the pipeline kept (conf >= 0.6) and keeps the torso (the shirt) for the embedding
  * for each method it groups the crops, matches groups to your current labels (best one-to-one match) and reports how
    often they agree, for the two teams only and for everyone
  * 'with voting' repeats that after giving every tracked player the majority answer over all his crops (this is what
    the pipeline's player_id does for the team label)
  * 'few-shot' = what you would do for a new team: take N crops per team, no training, nearest-centroid on the embedding
  * writes pictures of the crops where a method and your current classifier disagree, and of what each cluster contains

IMPORTANT: 'agreement with the current classifier' is NOT accuracy -- the classifier is not the truth either. Look at
the disagreement pictures: in each one, who is right?   Results are also saved to compare_team_methods_results.txt and the
pictures to compare_team_methods_out/.

Embedding choices (--embed, comma separated):
    dinov2   torch.hub facebookresearch/dinov2 vits14   (best for this; needs internet once)
    resnet   torchvision ResNet-50 ImageNet weights      (fallback)
    siglip   transformers google/siglip-base-patch16-224 (only if you pip install transformers)
    colour   hand-made colour statistics, numpy only     (the baseline you tried before)
"""
import os
import sys
import glob
import argparse
import sqlite3
import itertools

import numpy as np
import pandas as pd
import cv2

OUT_DIR = "compare_team_methods_out"
RESULTS = "compare_team_methods_results.txt"
MAIN_TEAMS_MIN_SHARE = 0.15          # a label counts as a "team" if it has at least this share of the crops
CROP_W, CROP_H = 48, 96              # stored whole-box crop (pictures)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ------------------------------------------------------------------------------------------------ data

def find_game_files(args):
    db, video = args.db, args.video
    if not db or not video:
        try:
            from game_logger import paths
            if not db:
                db = paths.detections_db(args.game)
            if not video:
                folder = paths.game_folder(args.game)
                hits = sorted(glob.glob(os.path.join(folder, "videos", "*_analysis.mp4"))) if folder else []
                video = hits[0] if hits else None
        except Exception as e:                                          # noqa
            say("(could not use game_logger.paths:", e, ")")
    if not db or not os.path.isfile(db):
        sys.exit(f"Cannot find the detections database for game {args.game}. Pass --db /path/to/..._detections.db")
    if not video or not os.path.isfile(video):
        try:
            v = pd.read_sql("select video from game", sqlite3.connect(db)).iloc[0, 0]
            from game_logger import paths
            video = paths.resolve_stored(v) or v
        except Exception:                                               # noqa
            pass
    if not video or not os.path.isfile(video):
        sys.exit("Cannot find the analysis video. Pass --video /path/to/..._analysis.mp4")
    return db, video


def sample_crops(db, video, bursts, burst_len, stride, seed):
    c = sqlite3.connect(db)
    d = pd.read_sql("select frame_id,xmin,ymin,xmax,ymax,confidence,tracker_id,player_id,class_name,team,team_raw "
                    "from detected_objects where class_name!='ball' and confidence>=0.6", c)
    frames = np.sort(d.frame_id.unique())
    rng = np.random.default_rng(seed)
    span = burst_len * stride
    starts = np.sort(rng.choice(np.arange(0, len(frames) - span), size=bursts, replace=False))
    cap = cv2.VideoCapture(video)
    crops, torsos, meta = [], [], []
    for b, s0 in enumerate(starts):
        for fi in frames[s0:s0 + span:stride]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = cap.read()
            if not ok:
                continue
            H, W = im.shape[:2]
            for r in d[d.frame_id == fi].itertuples():
                x0, y0, x1, y1 = [int(round(v)) for v in (r.xmin, r.ymin, r.xmax, r.ymax)]
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                w, h = x1 - x0, y1 - y0
                if w < 12 or h < 24:
                    continue
                box = im[y0:y1, x0:x1]
                crops.append(cv2.resize(box, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA))
                tx0, tx1 = int(w * .18), int(w * .82)
                ty0, ty1 = int(h * .12), int(h * .58)                    # shoulders to waist: the shirt
                t = box[ty0:ty1, tx0:tx1]
                torsos.append(cv2.resize(t, (112, 168), interpolation=cv2.INTER_AREA))
                meta.append((b, int(fi), r.tracker_id, r.player_id, r.class_name, r.team, r.team_raw, r.confidence))
    M = pd.DataFrame(meta, columns="burst frame_id tracker_id player_id class_name team team_raw conf".split())
    return np.stack(crops), np.stack(torsos), M


# ------------------------------------------------------------------------------------------------ embeddings

def emb_colour(torsos, **kw):
    out = []
    for t in torsos:
        lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(float)
        hh, ww = lab.shape[:2]
        border = np.vstack([lab[:, :4].reshape(-1, 3), lab[:, -4:].reshape(-1, 3)])
        bg = np.median(border, 0)
        px = lab.reshape(-1, 3)
        keep = px[np.linalg.norm(px - bg, axis=1) > 18]
        if len(keep) < 20:
            keep = px
        hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        hue = (hsv[:, 0] / 180 * 12).astype(int)
        dark, light = hsv[:, 2] < 70, (hsv[:, 1] < 50) & (hsv[:, 2] >= 70)
        col = ~dark & ~light
        hist = np.array([np.sum(col & (hue == b)) for b in range(12)] + [dark.sum(), light.sum()], float)
        hist /= hist.sum()
        out.append(np.r_[np.median(keep, 0) / 255, keep.mean(0) / 255, hist])
    F = np.array(out)
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


def _batches(torsos, n=64):
    for i in range(0, len(torsos), n):
        yield torsos[i:i + n]


def _to_tensor(batch, size, mean, std):
    import torch
    x = np.stack([cv2.resize(cv2.cvtColor(t, cv2.COLOR_BGR2RGB), (size[1], size[0])) for t in batch]).astype(np.float32) / 255
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def emb_dinov2(torsos, **kw):
    import torch
    dev = _device()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(dev).eval()
    out = []
    with torch.no_grad():
        for b in _batches(torsos):
            x = _to_tensor(b, (168, 112), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(dev)   # 12x8 patches of 14
            out.append(model(x).float().cpu().numpy())
    return np.vstack(out)


def emb_resnet(torsos, **kw):
    import torch
    import torchvision
    dev = _device()
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m = m.to(dev).eval()
    out = []
    with torch.no_grad():
        for b in _batches(torsos):
            x = _to_tensor(b, (168, 112), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(dev)
            out.append(m(x).float().cpu().numpy())
    return np.vstack(out)


def emb_siglip(torsos, **kw):
    import torch
    from transformers import AutoProcessor, SiglipVisionModel
    dev = _device()
    name = "google/siglip-base-patch16-224"
    proc, model = AutoProcessor.from_pretrained(name), SiglipVisionModel.from_pretrained(name).to(dev).eval()
    out = []
    with torch.no_grad():
        for b in _batches(torsos, 32):
            rgb = [cv2.cvtColor(t, cv2.COLOR_BGR2RGB) for t in b]
            inp = proc(images=rgb, return_tensors="pt").to(dev)
            out.append(model(**inp).pooler_output.float().cpu().numpy())
    return np.vstack(out)


EMBEDDERS = {"colour": emb_colour, "dinov2": emb_dinov2, "resnet": emb_resnet, "siglip": emb_siglip}


def normalise(F):
    F = F - F.mean(0)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------------------------------------ clustering helpers

def kmeans(X, k, n_init=12, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        c = [X[rng.integers(len(X))]]
        for _ in range(k - 1):                                                    # k-means++ seeding
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(2), axis=1)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
        c = np.array(c)
        for _ in range(iters):
            lab = ((X[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
            new = np.array([X[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((X - c[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = (lab, c), inertia
    return best


def best_matching(clusters, labels, cluster_ids, label_names):
    """Give each label (two teams, then 'other') its own cluster, choosing the combination that makes most crops agree.
    Clusters left over are called 'other'."""
    tab = np.array([[np.sum((clusters == ci) & (labels == ln)) for ln in label_names] for ci in cluster_ids])
    best, best_score = None, -1
    for perm in itertools.permutations(range(len(cluster_ids)), min(len(label_names), len(cluster_ids))):
        score = sum(tab[perm[j], j] for j in range(len(perm)))
        if score > best_score:
            best, best_score = perm, score
    mapping = {ci: "other" for ci in cluster_ids}
    for j, ci_pos in enumerate(best):
        mapping[cluster_ids[ci_pos]] = label_names[j]
    return mapping, tab


def vote(values, groups):
    out = np.array(values, dtype=object)
    s = pd.Series(values)
    for g, idx in s.groupby(groups).groups.items():
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        out[list(idx)] = s.loc[idx].mode().iloc[0]
    return out


def collapse(lbl, teams):
    return np.array([l if l in teams else "other" for l in lbl], dtype=object)


def score(pred, ref, teams):
    pred, ref = np.asarray(pred, object), np.asarray(ref, object)
    allm = float(np.mean(pred == ref))
    two = np.isin(ref, teams)
    twom = float(np.mean(pred[two] == ref[two])) if two.any() else float("nan")
    return allm, twom


# ------------------------------------------------------------------------------------------------ pictures

def tile(crops, caps, cols=12, scale=1.0):
    h, w = CROP_H, CROP_W
    capH = 22
    rows = int(np.ceil(len(crops) / cols))
    sheet = np.full((rows * (h + capH), cols * w, 3), 30, np.uint8)
    for i, (c, cap) in enumerate(zip(crops, caps)):
        r, k = divmod(i, cols)
        sheet[r * (h + capH):r * (h + capH) + h, k * w:(k + 1) * w] = c
        for j, line in enumerate(cap.split("|")):
            cv2.putText(sheet, line[:9], (k * w + 1, r * (h + capH) + h + 9 + 9 * j), cv2.FONT_HERSHEY_SIMPLEX, .26, (0, 255, 255), 1)
    return sheet


def short(l):
    return {"Aphrodite Wanderers": "Aphrod", "Goalkeeper": "GK", "Referee": "Ref", "Unassigned": "?"}.get(l, str(l)[:8])


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--embed", default="dinov2,colour", help="comma separated: dinov2,resnet,siglip,colour")
    ap.add_argument("--bursts", type=int, default=14, help="how many short stretches of the game to sample")
    ap.add_argument("--burst-len", type=int, default=10, help="frames per stretch")
    ap.add_argument("--stride", type=int, default=3, help="detection frames between samples in a stretch (3 = 0.25 s)")
    ap.add_argument("--k", type=int, default=4, help="clusters: two teams + the odd ones (goalkeeper, referee)")
    ap.add_argument("--shots", type=int, default=25, help="crops per team for the few-shot test")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--db"), ap.add_argument("--video")
    args = ap.parse_args()

    db, video = find_game_files(args)
    say(f"game {args.game}\n  database {db}\n  video    {video}")
    crops, torsos, M = sample_crops(db, video, args.bursts, args.burst_len, args.stride, args.seed)
    say(f"{len(M)} player crops from {args.bursts} bursts x {args.burst_len} frames")
    labels = M.team_raw.fillna("Unassigned").values.astype(object)          # the classifier BEFORE any smoothing
    share = pd.Series(labels).value_counts(normalize=True)
    teams = [l for l in share.index if share[l] >= MAIN_TEAMS_MIN_SHARE and l not in ("Goalkeeper", "Referee", "Unassigned")][:2]
    say("current classifier (raw) counts:", pd.Series(labels).value_counts().to_dict(), "-> the two teams:", teams)
    ref = collapse(labels, teams)
    ref_sm = collapse(M.team.fillna("Unassigned").values.astype(object), teams)
    say(f"the pipeline's smoothed team label agrees with its raw label on {np.mean(ref == ref_sm):.1%} of crops "
        f"({np.mean(ref[np.isin(ref, teams)] == ref_sm[np.isin(ref, teams)]):.1%} for the two teams)\n")
    groups = M.player_id.where(M.player_id.notna(), M.tracker_id + 1e6).values           # stitched id, else tracker id
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    for name in [e.strip() for e in args.embed.split(",") if e.strip()]:
        say(f"=== {name} " + "=" * 60)
        try:
            F = EMBEDDERS[name](torsos)
        except Exception as e:                                                  # noqa
            say(f"  could not run '{name}': {type(e).__name__}: {e}\n")
            continue
        X = normalise(F) if name != "colour" else F / np.sqrt(F.shape[1])
        # --- unsupervised: group the game's crops, then name the groups after the labels they overlap most
        lab, cen = kmeans(X, args.k)
        cl_ids = list(range(args.k))
        mapping, tab = best_matching(lab, ref, cl_ids, teams + ["other"])
        pred = np.array([mapping.get(l, "other") for l in lab], dtype=object)
        a_all, a_two = score(pred, ref, teams)
        pred_v = vote(pred, groups)
        v_all, v_two = score(pred_v, ref, teams)
        say(f"  clustering into {args.k} groups (no training):")
        say(f"    agreement with current classifier:  teams only {a_two:.1%}   everyone {a_all:.1%}")
        say(f"    after one vote per tracked player:  teams only {v_two:.1%}   everyone {v_all:.1%}")
        say("    what each group overlaps with (rows = group):  " + str(["->" + mapping[i] for i in cl_ids]))
        say(pd.DataFrame(tab, index=[f"group {i} -> {mapping[i]}" for i in cl_ids], columns=teams + ["other"]).to_string())
        # --- few-shot: N crops per team, nearest centroid, scored on the crops NOT used
        fs_pred = np.array(["other"] * len(M), dtype=object)
        used = np.zeros(len(M), bool)
        cents = {}
        for t in teams:
            idx = np.where((ref == t) & (M.conf.values >= 0.7))[0]
            pick = rng.choice(idx, size=min(args.shots, len(idx)), replace=False)
            used[pick] = True
            cents[t] = X[pick].mean(0)
        other_idx = np.where((ref == "other") & (M.conf.values >= 0.7))[0]
        if len(other_idx):
            pick = rng.choice(other_idx, size=min(args.shots, len(other_idx)), replace=False)
            used[pick] = True
            cents["other"] = X[pick].mean(0)
        names = list(cents)
        D = np.array([[np.linalg.norm(x - cents[n]) for n in names] for x in X])
        fs_pred = np.array([names[i] for i in D.argmin(1)], dtype=object)
        keep = ~used
        f_all, f_two = score(fs_pred[keep], ref[keep], teams)
        fv_all, fv_two = score(vote(fs_pred, groups)[keep], ref[keep], teams)
        say(f"  few-shot, {args.shots} labelled crops per group, no training (scored on the other {keep.sum()} crops):")
        say(f"    agreement with current classifier:  teams only {f_two:.1%}   everyone {f_all:.1%}")
        say(f"    after one vote per tracked player:  teams only {fv_two:.1%}   everyone {fv_all:.1%}\n")
        rows.append((name, a_two, v_two, f_two, fv_two))
        # --- pictures
        pic_a = np.where(pred != ref)[0]
        pic_a = pic_a[np.isin(ref[pic_a], teams)]
        if len(pic_a):
            sel = rng.choice(pic_a, size=min(72, len(pic_a)), replace=False)
            cv2.imwrite(os.path.join(OUT_DIR, f"game{args.game}_{name}_disagreements.png"),
                        tile(crops[sel], [f"{short(ref[i])}|{short(pred[i])}" for i in sel]))
        grid = []
        for ci in cl_ids:
            idx = np.where(lab == ci)[0]
            sel = rng.choice(idx, size=min(24, len(idx)), replace=False) if len(idx) else []
            if len(sel):
                row = tile(crops[sel], [f"g{ci}|{short(ref[i])}" for i in sel], cols=24)
                grid.append(np.pad(row, ((0, 0), (0, 24 * CROP_W - row.shape[1]), (0, 0))))
        if grid:
            cv2.imwrite(os.path.join(OUT_DIR, f"game{args.game}_{name}_clusters.png"), np.vstack(grid))

    if rows:
        say("SUMMARY (agreement with current classifier, the two teams only)")
        say(f"{'embedding':<10}{'clustering':>12}{'+ voting':>10}{'few-shot':>10}{'+ voting':>10}")
        for r in rows:
            say(f"{r[0]:<10}{r[1]:>12.1%}{r[2]:>10.1%}{r[3]:>10.1%}{r[4]:>10.1%}")
    say(f"\npictures: {OUT_DIR}/ (…_disagreements.png: label on the first line = current classifier, second = this method; "
        "…_clusters.png: what each group contains)")
    with open(RESULTS, "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
