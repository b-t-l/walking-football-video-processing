"""
train_team_model.py  --  keep one folder of team pictures, train a new team classifier from it on this Mac, and compare the
new model with the one you use now. No Roboflow needed.

Run from the application folder (the one containing main.py), in the same python environment as main.py.

THE PICTURE LIBRARY   (video-processing/_team-model-training/team-pictures/<team>/*.jpg)
    One folder per team, named exactly as the model should call it: Polis Lions, Aphrodite Wanderers, Akamas, West Coast,
    Goalkeeper, Referee ... Every picture you ever sort goes in here and stays. A new team = a new folder.

    python train_team_model.py --import-export "../_team-model-training/wtb-team-classification.v4i.folder"
                              one-off: copy a Roboflow export (train/valid/test) into the library, skipping duplicates
    python train_team_model.py --add "/Volumes/LaCie/walking-football/games/0025_*/team-images/20261010-210504"
                              copy a folder made by the "Team training images" step (after you sorted it) into the library
                              (pictures keep their full size; they are squashed to 192x192 only when training)
    (or just drag pictures into the team folders in Finder)

TRAINING
    python train_team_model.py --dry-run                 # only show the counts and how the pictures will be split
    python train_team_model.py                           # train yolo26s-cls on the Mac's GPU, then compare old and new
    python train_team_model.py --model yolo26n-cls.pt    # the smaller / faster model

What a training run does
  1. splits the library into train (75%), val (15%) and test (10%). The split is by game and frame, so pictures taken
     at the same moment never end up on both sides (neighbouring crops are near copies and would flatter the score).
     The split is fixed by the picture's name, so adding pictures later does not reshuffle the old ones.
  2. trains the chosen YOLO26 classification model at 192x192 with colour-safe augmentation: no hue shifts and no
     RandAugment (which can recolour a kit), because the kit colour is exactly what the model has to learn
  3. scores the NEW weights and your CURRENT weights on the same val and test pictures, per team, and says what each team
     was mistaken for
  4. saves the new weights as models/teams/team-<date>-<model>.pt. It never overwrites the current model, and it tells
     you the one line to change in data/create_database.py when you decide to switch.

The current model has probably seen many of these pictures in training, so its score here is flattering. The real test is a
game neither model has seen: run the pipeline with each and look at the team colours.
"""
import os
import re
import sys
import glob
import time
import shutil
import hashlib
import argparse
import datetime

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2

APP = os.path.dirname(os.path.abspath(__file__))
TRAIN_ROOT = os.path.join(os.path.dirname(APP), "_team-model-training")
LIBRARY = os.path.join(TRAIN_ROOT, "team-pictures")
IMG = (".jpg", ".jpeg", ".png", ".bmp")
SIZE = 192
VAL_PCT, TEST_PCT = 15, 10
MIN_PICTURES = 20          # a team with fewer pictures than this is left out of the training until it has more
RESULTS = os.path.join(APP, "train_team_model_results.txt")
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def images_in(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMG))


def class_folders(split_dir):
    if not os.path.isdir(split_dir):
        return {}
    return {d: os.path.join(split_dir, d) for d in sorted(os.listdir(split_dir))
            if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith((".", "_"))}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------------ the library

def clean_name(fname):
    """0025_f000598_id26_player_jpg.rf.<hash>.jpg -> 0025_f000598_id26_player.jpg  (Roboflow adds the .rf.<hash> tail)"""
    base = os.path.splitext(fname)[0]
    base = re.sub(r"\.rf\.[0-9a-fA-F]+$", "", base)
    base = re.sub(r"_(jpg|jpeg|png)$", "", base)
    return base


def library_hashes():
    seen = {}
    for cls, folder in class_folders(LIBRARY).items():
        for f in images_in(folder):
            seen[md5(f)] = cls
    return seen


def library_names():
    """pictures made by the Team training images step have unique names (game + frame + player); the same name means the
    same picture even if it was re-saved (e.g. squashed to 192x192 by Roboflow), so it must not be added twice"""
    seen = {}
    for cls, folder in class_folders(LIBRARY).items():
        for f in images_in(folder):
            n = clean_name(os.path.basename(f))
            if re.match(r"^\d{4}_f\d+_", n):
                seen[n] = cls
    return seen


def add_to_library(sources, alias, label):
    """sources: list of (class name, picture path). Copies new pictures into the library; skips ones already there."""
    seen = library_hashes()
    names = library_names()
    added, dupes, conflicts, per = 0, 0, 0, {}
    for cls0, f in sources:
        cls = alias.get(cls0, cls0)
        if cls == "Unassigned":
            continue
        h = md5(f)
        nm = clean_name(os.path.basename(f))
        new_style = bool(re.match(r"^\d{4}_f\d+_", nm))
        if new_style and nm in names and h not in seen:                                    # same picture under the same name, re-saved
            if names[nm] == cls:
                dupes += 1
            else:
                conflicts += 1
                say(f"  conflict: {os.path.basename(f)} is in the library as '{names[nm]}' but is in '{cls}' here - left as '{names[nm]}'")
            continue
        if h in seen:
            if seen[h] == cls:
                dupes += 1
            else:
                conflicts += 1
                say(f"  conflict: {os.path.basename(f)} is already in the library as '{seen[h]}' but is in '{cls}' here - left as '{seen[h]}'")
            continue
        out = os.path.join(LIBRARY, cls)
        os.makedirs(out, exist_ok=True)
        ext = os.path.splitext(f)[1].lower()
        name = clean_name(os.path.basename(f)) + ext
        if os.path.exists(os.path.join(out, name)):                          # same name, different picture
            name = f"{clean_name(os.path.basename(f))}-{h[:6]}{ext}"
        shutil.copy2(f, os.path.join(out, name))
        seen[h] = cls
        if new_style:
            names[nm] = cls
        added += 1
        per[cls] = per.get(cls, 0) + 1
    say(f"{label}: {added} pictures added, {dupes} already in the library, {conflicts} conflicts")
    if per:
        say("   added: " + ", ".join(f"{k} {v}" for k, v in sorted(per.items())))


def import_export(folder):
    folder = os.path.abspath(folder)
    srcs = []
    for split in ("train", "valid", "val", "test"):
        for cls, d in class_folders(os.path.join(folder, split)).items():
            srcs += [(cls, f) for f in images_in(d)]
    if not srcs:
        sys.exit(f"No train/valid/test team folders found in {folder}")
    add_to_library(srcs, ALIAS, "import")


def add_sorted(folder_glob):
    folders = sorted(glob.glob(folder_glob))
    if not folders:
        sys.exit(f"No folder matches {folder_glob}")
    for folder in folders:
        srcs = []
        for cls, d in class_folders(folder).items():
            srcs += [(cls, f) for f in images_in(d)]
        add_to_library(srcs, ALIAS, os.path.basename(os.path.dirname(folder)) + "/" + os.path.basename(folder))


# ------------------------------------------------------------------------------------------------ split + prepare

def group_key(fname):
    """Pictures from the same game frame belong together (made by the Team training images step: 0025_f000598_...)."""
    m = re.match(r"^(\d{4})_f(\d+)_", fname)
    return f"{m.group(1)}_{m.group(2)}" if m else clean_name(fname)


def bucket(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 100


def make_split():
    """{class: {'train': [paths], 'val': [...], 'test': [...]}}, split by group and fixed by the picture's name."""
    out = {}
    for cls, folder in class_folders(LIBRARY).items():
        groups = {}
        for f in images_in(folder):
            groups.setdefault(group_key(os.path.basename(f)), []).append(f)
        sp = {"train": [], "val": [], "test": []}
        for key, files in sorted(groups.items(), key=lambda kv: bucket(kv[0])):
            b = bucket(key)
            sp["val" if b < VAL_PCT else "test" if b < VAL_PCT + TEST_PCT else "train"] += files
        # a team with a handful of pictures could get none by chance: take whole groups from train (lowest bucket first)
        for need, share in (("val", 8), ("test", 12)):
            if not sp[need] and len(groups) >= 3:
                for key, files in sorted(groups.items(), key=lambda kv: bucket(kv[0])):
                    if all(f in sp["train"] for f in files):
                        for f in files:
                            sp["train"].remove(f)
                        sp[need] += files
                        break
        out[cls] = sp
    return out


def prepare(split, work):
    for cls, sp in split.items():
        for name, files in sp.items():
            out = os.path.join(work, name, cls)
            os.makedirs(out, exist_ok=True)
            for f in files:
                im = cv2.imread(f)
                if im is None:
                    continue
                # exactly what the pipeline does to every crop before the model sees it: squash to a square
                cv2.imwrite(os.path.join(out, os.path.splitext(os.path.basename(f))[0] + ".jpg"),
                            cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Ultralytics numbers the classes from each split's own folders: val must have every class train has
    for cls, sp in split.items():
        vdir = os.path.join(work, "val", cls)
        if not images_in(vdir) and images_in(os.path.join(work, "train", cls)):
            os.makedirs(vdir, exist_ok=True)
            for f in images_in(os.path.join(work, "train", cls))[:2]:
                shutil.copy2(f, os.path.join(vdir, os.path.basename(f)))
            say(f"  note: '{cls}' has too few pictures for its own validation set; 2 training pictures were copied in (its val score means little)")


def total_of(sp):
    return len(sp["train"]) + len(sp["val"]) + len(sp["test"])


def show_counts(split):
    say(f"{'team':<24}{'train':>7}{'val':>6}{'test':>6}{'total':>7}")
    tot = [0, 0, 0]
    for cls, sp in split.items():
        n = [len(sp["train"]), len(sp["val"]), len(sp["test"])]
        t = sum(n)
        if t < MIN_PICTURES:
            flag = "   <- no pictures yet, not used" if t == 0 else f"   <- only {t}: not used in training until it has {MIN_PICTURES}"
        else:
            tot = [a + b for a, b in zip(tot, n)]
            flag = "   <- very few: add more" if t < 60 else ""
        say(f"{cls:<24}{n[0]:>7}{n[1]:>6}{n[2]:>6}{t:>7}{flag}")
    say(f"{'used in training':<24}{tot[0]:>7}{tot[1]:>6}{tot[2]:>6}{sum(tot):>7}")


def usable(split):
    """only the teams with enough pictures take part in a training run"""
    return {c: sp for c, sp in split.items() if total_of(sp) >= MIN_PICTURES}


def sync_teams():
    """make a folder in the library for every team in the game logger (plus Goalkeeper and Referee)"""
    sys.path.insert(0, APP)
    from game_logger import db
    conn = db.connect()
    try:
        names = [r[0] for r in conn.execute("SELECT name FROM teams ORDER BY name")]
    finally:
        conn.close()
    made = []
    for n in names + ["Goalkeeper", "Referee"]:
        safe = re.sub(r'[\\/:*?"<>|]+', "-", n).strip()
        d = os.path.join(LIBRARY, safe)
        if not os.path.isdir(d):
            os.makedirs(d)
            made.append(safe)
    say(f"team folders in {LIBRARY}: {len(names)} teams from the game logger + Goalkeeper + Referee")
    say("  created: " + (", ".join(made) if made else "none (all already there)"))


# ------------------------------------------------------------------------------------------------ scoring

def evaluate(weights, split_dir, device, alias):
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    paths, truth = [], []
    for cls, folder in class_folders(split_dir).items():
        for f in images_in(folder):
            paths.append(f)
            truth.append(cls)
    if not paths:
        return None
    pred = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], imgsz=SIZE, device=device, verbose=False):
            n = names[int(r.probs.top1)]
            pred.append(alias.get(n, n))
    return truth, pred


def report(title, res):
    if res is None:
        say(f"  {title}: no pictures")
        return
    truth, pred = res
    ok = sum(t == p for t, p in zip(truth, pred))
    say(f"  {title}: {ok}/{len(truth)} right = {ok / len(truth):.1%}")
    for c in sorted(set(truth)):
        idx = [i for i, t in enumerate(truth) if t == c]
        good = sum(pred[i] == c for i in idx)
        wrong = {}
        for i in idx:
            if pred[i] != c:
                wrong[pred[i]] = wrong.get(pred[i], 0) + 1
        say(f"      {c:<22}{good:>4}/{len(idx):<4}" + ("   mistaken for: " + ", ".join(f"{k} x{v}" for k, v in sorted(wrong.items(), key=lambda kv: -kv[1])) if wrong else ""))


ALIAS = {"Polis": "Polis Lions"}


def main():
    global LIBRARY, MIN_PICTURES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=LIBRARY, help="the folder of team folders (default: _team-model-training/team-pictures)")
    ap.add_argument("--import-export", help="copy a Roboflow Folder-Structure export (train/valid/test) into the library, then stop")
    ap.add_argument("--add", action="append", default=[], help="copy a sorted team-images/<timestamp> folder into the library, then stop")
    ap.add_argument("--model", default="yolo26s-cls.pt", help="starting weights: yolo26n-cls.pt / yolo26s-cls.pt / yolo26m-cls.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", help="mps, cpu or cuda (default: mps if the Mac has it)")
    ap.add_argument("--old", default=os.path.join(APP, "models", "teams", "best-v7-192x192.pt"), help="the current team model to compare with")
    ap.add_argument("--alias", action="append", default=["Polis=Polis Lions"],
                    help="old class name = library name (the old model says 'Polis'; the library calls it 'Polis Lions')")
    ap.add_argument("--dry-run", action="store_true", help="only show the counts and the split; train nothing")
    ap.add_argument("--sync-teams", action="store_true", help="make a library folder for every team in the game logger, then stop")
    ap.add_argument("--min-pictures", type=int, default=MIN_PICTURES, help="a team needs this many pictures to be trained (default 20)")
    args = ap.parse_args()
    LIBRARY = os.path.abspath(args.library)
    MIN_PICTURES = args.min_pictures
    ALIAS.clear()
    ALIAS.update(dict(a.split("=", 1) for a in args.alias if "=" in a))

    if args.sync_teams:
        os.makedirs(LIBRARY, exist_ok=True)
        sync_teams()
        show_counts(make_split())
        return

    if args.import_export or args.add:
        os.makedirs(LIBRARY, exist_ok=True)
        if args.import_export:
            import_export(args.import_export)
        for a in args.add:
            add_sorted(a)
        say(f"\nlibrary: {LIBRARY}")
        show_counts(make_split())
        return

    if not class_folders(LIBRARY):
        sys.exit(f"The picture library {LIBRARY} is empty. First:  python train_team_model.py --import-export <Roboflow export folder>")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = f"team-{stamp}"
    work = os.path.join(TRAIN_ROOT, "_prepared", run)
    say(f"library: {LIBRARY}")
    split = make_split()
    show_counts(split)
    split = usable(split)
    if len(split) < 2:
        sys.exit("Fewer than two teams have enough pictures to train.")
    if args.dry_run:
        say("\n(dry run: nothing prepared or trained)")
        return
    prepare(split, work)
    say(f"prepared in: {work}")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"Run this in the python environment you use for main.py (it needs torch and ultralytics): {e}")
    device = args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    say(f"\ntraining {args.model} on {device}: {args.epochs} epochs at most, stops early when it stops improving\n")
    t0 = time.time()
    model = YOLO(args.model)
    model.train(data=work, imgsz=SIZE, epochs=args.epochs, batch=args.batch, device=device, project=os.path.join(TRAIN_ROOT, "runs"),
                name=run, exist_ok=True, patience=15, workers=4,
                auto_augment=None, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, scale=0.3, erasing=0.1, fliplr=0.5)
    best = str(model.trainer.best)
    say(f"\ntraining took {(time.time() - t0) / 60:.1f} minutes. best weights: {best}")

    dst = os.path.join(APP, "models", "teams", f"team-{datetime.date.today():%Y%m%d}-{os.path.splitext(os.path.basename(args.model))[0]}.pt")
    shutil.copy2(best, dst)
    say("\n================ COMPARISON on the same pictures ================")
    for label, w, al in (("NEW", dst, {}), ("CURRENT", args.old, ALIAS)):
        if not os.path.isfile(w):
            say(f"\n{label}: {w} not found, skipped")
            continue
        say(f"\n{label}  ({os.path.basename(w)})")
        for part in ("val", "test"):
            try:
                report(part, evaluate(w, os.path.join(work, part), device, al))
            except Exception as e:                                           # noqa
                say(f"  {part}: could not score ({type(e).__name__}: {e})")
    say("\nREMEMBER: the current model has probably seen many of these pictures in training, so its score is flattering.")
    say(f"\nnew weights saved as: {dst}")
    say("to use them: in data/create_database.py change the line")
    say("    self.TEAM_MODEL_PATH = os.path.join('models', 'teams', 'best-v7-192x192.pt')")
    say(f"to  self.TEAM_MODEL_PATH = os.path.join('models', 'teams', '{os.path.basename(dst)}')")
    say("(team colours come from the first kit colour of the team on the Game Logger Teams page)")
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    say(f"(this report is also in {RESULTS})")


if __name__ == "__main__":
    main()
