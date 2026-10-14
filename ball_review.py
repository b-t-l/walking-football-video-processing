"""
ball_review.py  --  measure how well the ball models really do, using YOUR eyes as the answer key.

Step 1 - make a review set (needs the same python environment as main.py; takes a minute or two):

    python ball_review.py --video "<game>/videos/0027_analysis.mp4" --pitch-db "<game>/data/0027_detections.db"

  * picks 100 frames spread evenly across the game (one random frame per slice of the game, so none are neighbours)
  * runs BOTH ball models on them (12 Oct = what the earlier games used, 14 Oct = new) and stores what each one found
  * writes <game>/ball-review/<timestamp>/ with the frames, picks.csv and review.html

Step 2 - open review.html (double-click it). For each frame, WITHOUT seeing what the models picked:
      click on the ball                     (a magnifier follows the mouse; + / - change its zoom)
      N   no ball visible at all in the picture
      O   the only ball in view is outside the pitch (a spare ball by the fence, etc.)
      U   unsure - skip this one
      B   (or the Back button) go back one frame
  When you reach the end (or any time) press "Download results" - it saves ball_review_results.csv to Downloads.
  Your progress is remembered in the browser, so you can stop and carry on later.

Step 3 - score the models:

    python ball_review.py --summary "<game>/ball-review/<timestamp>"

  (it finds ball_review_results.csv in that folder or in ~/Downloads)
  It prints, for each model at several confidence thresholds: how often the ball is really visible, how many of those
  the model finds (recall), how many it misses, how often it picks the wrong thing, and false picks when no ball is in
  play. It also saves a contact sheet of the balls each model missed, so you can see WHY.
"""
import os
import sys
import csv
import json
import glob
import time
import random
import argparse

import numpy as np

TOL_PX = 25          # a pick within this many pixels of your click counts as the ball
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]
MIN_CONF = 0.1


# --------------------------------------------------------------------------------------------- step 1: prepare
def sample_frames(total, n, start, end, seed):
    """one random frame in each of n equal slices between start and end -> sorted frame numbers"""
    rnd = random.Random(seed)
    span = max(end - start, n)
    edges = [start + int(span * k / n) for k in range(n + 1)]
    return [rnd.randrange(edges[k], max(edges[k + 1], edges[k] + 1)) for k in range(n)]


def read_one(cap, frame_no):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, f = cap.read()
    return f if ok else None


def prepare(args):
    import cv2
    from compare_ball_models import FrameStore, WEIGHTS, run
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = args.start_frame
    end = total - args.end_margin
    frames_no = sample_frames(total, args.n, start, end, args.seed)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    game_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.video)))
    out = args.out or os.path.join(game_dir, "ball-review", stamp)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)

    store = FrameStore()
    kept = []
    for fn in frames_no:
        f = read_one(cap, fn)
        if f is None:
            continue
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        store.add(rgb)
        kept.append(fn)
    cap.release()
    h, w = store.shape[:2]
    # the image Mark reviews is exactly the one the models see
    for k, fn in enumerate(kept):
        cv2.imwrite(os.path.join(out, "frames", f"f{fn:06d}.jpg"), cv2.cvtColor(store[k], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {len(kept)} frames ({w}x{h}) to {out}")

    rows = []
    poly = None
    if not args.frames_only:
        import torch
        from shapely.geometry import Polygon
        from compare_player_models import load_pitch_polygon
        from compare_ball_models import on_pitch_flags
        device, half = ("mps", True) if torch.backends.mps.is_available() else ("cpu", False)
        if args.pitch_db and os.path.exists(args.pitch_db):
            poly = Polygon(load_pitch_polygon(args.pitch_db)).buffer(40)
        else:
            print("(no pitch outline: every box counts as on the pitch)")
        for label, path in WEIGHTS:
            print(f"\n=== {label}  ({path}) ===")
            if not os.path.exists(path):
                print("   missing, skipped")
                continue
            ms, dets = run(path, store, args.imgsz, device, half, 2, MIN_CONF)
            print(f"   {ms:.1f} ms/frame")
            for fn, d in zip(kept, dets):
                flags = on_pitch_flags(d, poly)
                for (x1, y1, x2, y2), c, on in zip(d["xyxy"], d["conf"], flags):
                    rows.append([label, fn, round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1),
                                 round(float(c), 4), int(on), round(float(x2 - x1), 1), round(float(y2 - y1), 1)])
    with open(os.path.join(out, "picks.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "frame", "x", "y", "conf", "on_pitch", "w", "h"])
        wr.writerows(rows)
    with open(os.path.join(out, "frames.json"), "w") as fh:
        json.dump(dict(video=os.path.basename(args.video), width=w, height=h, frames=kept), fh)
    with open(os.path.join(out, "review.html"), "w") as fh:
        fh.write(HTML.replace("__DATA__", json.dumps(dict(video=os.path.basename(args.video), width=w, height=h,
                                                          frames=kept, key=os.path.basename(out)))))
    print(f"\nNow open this file and review the frames:\n   {os.path.join(out, 'review.html')}")


# --------------------------------------------------------------------------------------------- step 3: summary
def load_results(path):
    res = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            res[int(r["frame"])] = (r["label"], float(r["x"]) if r["x"] else None, float(r["y"]) if r["y"] else None)
    return res


def find_results(folder):
    for d in (folder, os.path.expanduser("~/Downloads")):
        c = sorted(glob.glob(os.path.join(d, "ball_review_results*.csv")), key=os.path.getmtime)
        if c:
            return c[-1]
    raise SystemExit("Could not find ball_review_results.csv in the review folder or in Downloads. "
                     "Open review.html and press 'Download results' first, or pass --results <file>.")


def contact_sheet(folder, items, outfile):
    """items: (frame, x, y, note) -> crops around the labelled ball, so you can see why it was missed"""
    import cv2
    tiles = []
    for fn, x, y, note in items[:48]:
        img = cv2.imread(os.path.join(folder, "frames", f"f{fn:06d}.jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        x0 = int(min(max(x - 60, 0), w - 120))
        y0 = int(min(max(y - 60, 0), h - 120))
        crop = cv2.resize(img[y0:y0 + 120, x0:x0 + 120], (240, 240), interpolation=cv2.INTER_CUBIC)
        cv2.circle(crop, (int((x - x0) * 2), int((y - y0) * 2)), 26, (0, 255, 255), 1)
        cv2.putText(crop, f"{fn} {note}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(crop)
    if not tiles:
        return False
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(outfile, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def summary(args):
    folder = args.summary
    res = load_results(args.results or find_results(folder))
    picks = {}
    with open(os.path.join(folder, "picks.csv")) as fh:
        for r in csv.DictReader(fh):
            picks.setdefault(r["model"], {}).setdefault(int(r["frame"]), []).append(
                (float(r["x"]), float(r["y"]), float(r["conf"]), int(r["on_pitch"])))
    n = len(res)
    ball = {f: v for f, v in res.items() if v[0] == "ball"}
    none = [f for f, v in res.items() if v[0] == "none"]
    off = [f for f, v in res.items() if v[0] == "offpitch"]
    unsure = [f for f, v in res.items() if v[0] == "unsure"]
    L = [f"BALL REVIEW  {n} frames labelled (of {os.path.basename(folder)})", "",
         f"  ball visible on the pitch : {len(ball):3d}  ({100 * len(ball) / max(n, 1):.0f}% of frames)",
         f"  no ball visible           : {len(none):3d}  ({100 * len(none) / max(n, 1):.0f}%)",
         f"  only a ball off the pitch : {len(off):3d}",
         f"  unsure (ignored)          : {len(unsure):3d}", ""]
    if not picks:
        L.append("picks.csv has no model picks (was it made with --frames-only?)")
    L += ["  Recall = of the frames where YOU could see the ball, how many the model found (top on-pitch pick within "
          f"{TOL_PX} px of your click).",
          "  'ball in lower picks' = the ball WAS among the model's boxes but another box scored higher.", ""]
    head = (f"{'conf':>5} {'model':<26}{'found':>7}{'recall':>8}{'missed':>8}{'wrong spot':>11}{'in lower picks':>15}"
            f"{'false picks*':>14}{'quiet ok*':>11}")
    L += [head, "-" * len(head)]
    missed_items = {}
    for thr in THRESHOLDS:
        for model, pf in picks.items():
            found = missed = wrong = lower = 0
            miss_list = []
            for f, (lab, x, y) in ball.items():
                boxes = [b for b in pf.get(f, []) if b[2] >= thr and b[3]]
                if not boxes:
                    missed += 1
                    miss_list.append((f, x, y, "none"))
                    continue
                top = max(boxes, key=lambda b: b[2])
                if np.hypot(top[0] - x, top[1] - y) <= TOL_PX:
                    found += 1
                else:
                    wrong += 1
                    if any(np.hypot(b[0] - x, b[1] - y) <= TOL_PX for b in boxes):
                        lower += 1
                    miss_list.append((f, x, y, f"wrong {top[2]:.2f}"))
            quiet_frames = none + off
            false_picks = sum(1 for f in quiet_frames if any(b[2] >= thr and b[3] for b in pf.get(f, [])))
            quiet_ok = len(quiet_frames) - false_picks
            rec = 100 * found / max(len(ball), 1)
            L.append(f"{thr:5.2f} {model:<26}{found:7d}{rec:7.0f}%{missed:8d}{wrong:11d}{lower:15d}"
                     f"{false_picks:14d}{quiet_ok:11d}")
            missed_items[(model, thr)] = miss_list
        L.append("")
    L += ["* false picks = frames with no in-play ball where the model still put a ball on the pitch; "
          "quiet ok = frames with no in-play ball where it stayed quiet.",
          "  'found' counts only the single top pick, as the pipeline does."]
    text = "\n".join(L)
    print(text)
    with open(os.path.join(folder, "ball_review_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    for model in picks:
        thr = 0.3
        items = sorted(missed_items.get((model, thr), []))
        tag = "".join(ch for ch in model if ch.isalnum())
        out = os.path.join(folder, f"missed_{tag}_thr{thr}.jpg")
        if contact_sheet(folder, items, out):
            print(f"saved contact sheet of the balls '{model}' missed or got wrong at {thr}: {out}")
    print(f"\n(saved to {os.path.join(folder, 'ball_review_summary.txt')})")


# --------------------------------------------------------------------------------------------- review page
HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ball review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; --bg:#14171a; --panel:#1f2428; --line:#343b41; --text:#e8ecef; --muted:#9aa5ad; --acc:#ffd23f; --ok:#4cc38a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }
  header b { font-size:15px; }
  .muted { color:var(--muted); }
  button { background:#2b3238; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 12px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--acc); }
  button.primary { background:var(--acc); color:#222; border-color:var(--acc); font-weight:600; }
  kbd { background:#2b3238; border:1px solid var(--line); border-radius:4px; padding:1px 6px; font-size:12px; }
  #wrap { position:relative; display:inline-block; margin:12px 16px; line-height:0; }
  #img { display:block; width:calc(100vw - 32px); max-width:1900px; cursor:crosshair; border:1px solid var(--line); }
  #mark { position:absolute; width:26px; height:26px; margin:-13px 0 0 -13px; border:2px solid var(--acc); border-radius:50%; pointer-events:none; display:none; }
  #lens { position:fixed; right:18px; bottom:18px; width:300px; height:300px; border:2px solid var(--acc); border-radius:8px; background:#000 no-repeat; z-index:9; pointer-events:none; box-shadow:0 4px 24px #000a; }
  #lens::after { content:""; position:absolute; left:50%; top:50%; width:14px; height:14px; margin:-7px 0 0 -7px; border:1px solid #ffd23f99; border-radius:50%; }
  #status { padding:0 16px 12px; }
  #done { display:none; margin:20px 16px; padding:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; max-width:640px; line-height:1.5; }
  .pill { display:inline-block; padding:1px 8px; border-radius:99px; background:#2b3238; border:1px solid var(--line); font-size:12px; }
</style></head>
<body>
<header>
  <b>Ball review</b>
  <span id="prog" class="muted"></span>
  <span class="muted">Click the ball &middot; <kbd>N</kbd> no ball visible &middot; <kbd>O</kbd> ball only off the pitch &middot; <kbd>U</kbd> unsure &middot; <kbd>B</kbd> back &middot; <kbd>+</kbd><kbd>-</kbd> zoom</span>
  <span style="flex:1"></span>
  <button id="back">Back</button>
  <button id="dl" class="primary">Download results</button>
</header>
<div id="wrap"><img id="img" alt="frame"><div id="mark"></div></div>
<div id="status" class="muted"></div>
<div id="lens"></div>
<div id="done">
  <b>All frames reviewed.</b><br>Press <b>Download results</b> (top right). It saves <code>ball_review_results.csv</code> to your Downloads folder.
  Then in Terminal run the summary command from the instructions (<code>python ball_review.py --summary &lt;this folder&gt;</code>).
</div>
<script>
const DATA = __DATA__;
const KEY = "ballreview:" + DATA.key;
let results = {};                 // frame -> {label,x,y}
let idx = 0, zoom = 5;
try { results = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const img = document.getElementById("img"), mark = document.getElementById("mark"), lens = document.getElementById("lens");
const first = DATA.frames.findIndex(f => !results[f]);
idx = first < 0 ? DATA.frames.length : first;

function save() { try { localStorage.setItem(KEY, JSON.stringify(results)); } catch (e) {} }
function counts() {
  let c = {ball:0, none:0, offpitch:0, unsure:0};
  for (const f in results) c[results[f].label]++;
  return c;
}
function show() {
  const n = DATA.frames.length;
  const c = counts();
  document.getElementById("prog").textContent = `frame ${Math.min(idx + 1, n)} of ${n}`;
  document.getElementById("status").innerHTML =
    `<span class="pill">ball ${c.ball}</span> <span class="pill">no ball ${c.none}</span> <span class="pill">off-pitch ${c.offpitch}</span> <span class="pill">unsure ${c.unsure}</span>`;
  const done = idx >= n;
  document.getElementById("done").style.display = done ? "block" : "none";
  document.getElementById("wrap").style.display = done ? "none" : "inline-block";
  lens.style.display = "none";
  if (done) return;
  const f = DATA.frames[idx];
  img.src = "frames/f" + String(f).padStart(6, "0") + ".jpg";
  showMark();
}
function showMark() {
  const f = DATA.frames[idx], r = results[f];
  if (r && r.label === "ball") {
    const s = img.clientWidth / DATA.width;
    mark.style.left = (r.x * s) + "px"; mark.style.top = (r.y * s) + "px"; mark.style.display = "block";
  } else mark.style.display = "none";
}
img.addEventListener("load", showMark);
window.addEventListener("resize", showMark);
function setLabel(label, x, y) {
  results[DATA.frames[idx]] = {label, x: x == null ? "" : Math.round(x * 10) / 10, y: y == null ? "" : Math.round(y * 10) / 10};
  save(); idx++; show();
}
img.addEventListener("click", ev => {
  const r = img.getBoundingClientRect(), s = DATA.width / r.width;
  setLabel("ball", (ev.clientX - r.left) * s, (ev.clientY - r.top) * s);
});
img.addEventListener("mousemove", ev => {
  const r = img.getBoundingClientRect(), s = DATA.width / r.width;
  const nx = (ev.clientX - r.left) * s, ny = (ev.clientY - r.top) * s;
  lens.style.display = "block";
  lens.style.backgroundImage = `url("${img.src}")`;
  lens.style.backgroundSize = `${DATA.width * zoom}px ${DATA.height * zoom}px`;
  lens.style.backgroundPosition = `${150 - nx * zoom}px ${150 - ny * zoom}px`;
});
img.addEventListener("mouseleave", () => { lens.style.display = "none"; });
function back() { if (idx > 0) { idx--; delete results[DATA.frames[idx]]; save(); show(); } }
document.getElementById("back").onclick = back;
document.addEventListener("keydown", ev => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const k = ev.key.toLowerCase();
  if (idx < DATA.frames.length) {
    if (k === "n") setLabel("none");
    else if (k === "o") setLabel("offpitch");
    else if (k === "u") setLabel("unsure");
  }
  if (k === "b" || k === "arrowleft") back();
  if (k === "+" || k === "=") zoom = Math.min(zoom + 1, 10);
  if (k === "-") zoom = Math.max(zoom - 1, 2);
});
document.getElementById("dl").onclick = () => {
  let t = "frame,label,x,y\n";
  for (const f of DATA.frames) { const r = results[f]; if (r) t += `${f},${r.label},${r.x},${r.y}\n`; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([t], {type: "text/csv"}));
  a.download = "ball_review_results.csv"; a.click();
};
show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="the game's analysis video (step 1)")
    ap.add_argument("--pitch-db", default=None, help="the game's detections DB, to take the pitch outline from")
    ap.add_argument("--n", type=int, default=100, help="number of frames to review")
    ap.add_argument("--start-frame", type=int, default=1500, help="skip the first frames (warm-up, before kick-off)")
    ap.add_argument("--end-margin", type=int, default=1500, help="skip the last frames")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--out", default=None, help="output folder (default <game>/ball-review/<timestamp>)")
    ap.add_argument("--frames-only", action="store_true", help="save the frames and the page but do not run the models")
    ap.add_argument("--summary", metavar="FOLDER", help="step 3: score the models against your answers")
    ap.add_argument("--results", default=None, help="path of ball_review_results.csv (default: look in the folder, then Downloads)")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    elif args.video:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        prepare(args)
    else:
        ap.error("give --video (step 1) or --summary FOLDER (step 3)")


if __name__ == "__main__":
    main()
