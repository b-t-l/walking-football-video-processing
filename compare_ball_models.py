"""
compare_ball_models.py  --  current ball model (best-v2) against the new YOLO26 ball weights, on real game footage.

Run from the application folder (the one containing main.py), in the same python environment as main.py:

    python compare_ball_models.py                                   # 80 consecutive frames of game 22 from frame 240
    python compare_ball_models.py --start-frame 2400 --frames 120
    python compare_ball_models.py --save-diff ball_diff             # also save close-up pictures where the two models differ

Each model is run ONCE at a low threshold (0.1); every table is then worked out at several thresholds (--thresholds), so
you can see where each model behaves best. The pipeline's own threshold is 0.3.

Per model and threshold it prints
  * ms per frame (MPS, FP16, 1920x1920 stretched square - the pipeline setting)
  * frames with at least one ball box, boxes per frame
  * how many of the most-confident boxes are on the pitch (centre inside the pitch outline + 40 px) - a ball seen
    off the pitch (boots, cones, spectators' bags) is a false hit
  * how steady the most-confident ball is from one frame to the next: median move in pixels and how often it
    jumps more than 100 px in a single frame (a real ball rarely does; a false hit usually does)
  * agreement with the current model on the most-confident box: both within 20 px / only the current model / only the new one

Nothing here touches the database or the pipeline. Results are also saved to compare_ball_models_results.txt.
Agreement is not accuracy: look at the pictures (--save-diff) to see who is right.
"""
import os
import time
import argparse
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from benchmark_detection_devices import to_tensor
from compare_player_models import load_pitch_polygon, DEFAULT_DB

WEIGHTS = [
    ("best-v2 (current)", os.path.join("models", "ball", "best-v2-1920x1920.pt")),
    ("yolo26 v1", os.path.join("models", "ball", "yolo26-2026-10-09.pt")),
    ("yolo26 v2 (new)", os.path.join("models", "ball", "yolo26-2026-10-09-v2.pt")),
]


def read_frames(video, start, n):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}  (run this from the folder that contains main.py)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not out:
        raise SystemExit("No frames could be read - check --video and --start-frame")
    return out


def run(path, raw, imgsz, device, half, warmup, min_conf):
    import torch
    from ultralytics import YOLO
    model = YOLO(path)
    tensors = [to_tensor(f, imgsz) for f in raw]
    kw = dict(conf=min_conf, device=device, imgsz=(imgsz, imgsz), verbose=False)
    if half:
        kw["half"] = True

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model.predict(torch.stack(tensors[:1]), **kw)
    sync()
    h, w = raw[0].shape[:2]
    times, dets = [], []
    for t in tensors:
        t0 = time.perf_counter()
        r = model.predict(torch.stack([t]), **kw)[0]
        sync()
        times.append(time.perf_counter() - t0)
        b = r.boxes
        if len(b):
            xyxy = b.xyxy.cpu().numpy() * np.array([w / imgsz, h / imgsz, w / imgsz, h / imgsz])   # back to video pixels
            conf = b.conf.cpu().numpy()
        else:
            xyxy, conf = np.zeros((0, 4)), np.zeros((0,))
        dets.append(dict(xyxy=xyxy, conf=conf))
    return 1000.0 * float(np.mean(times)), dets


def centres(xyxy):
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1) if len(xyxy) else np.zeros((0, 2))


def on_pitch_flags(det, poly):
    """True for each box whose centre is inside the pitch outline (+40 px); all True when no outline is available"""
    from shapely.geometry import Point
    c = centres(det["xyxy"])
    if poly is None:
        return np.ones(len(c), dtype=bool)
    return np.array([bool(poly.contains(Point(float(x), float(y)))) for x, y in c], dtype=bool)


def top1(det, thr, poly=None):
    """most confident box above thr that is on the pitch -> (cx, cy, conf) or None"""
    keep = (det["conf"] >= thr) & on_pitch_flags(det, poly)
    if not keep.any():
        return None
    i = int(np.argmax(np.where(keep, det["conf"], -1)))
    c = centres(det["xyxy"][i:i + 1])[0]
    return float(c[0]), float(c[1]), float(det["conf"][i])


def summarise(dets, thr, poly):
    n = len(dets)
    boxes_on = boxes_off = frames_off = 0
    for d in dets:
        k = d["conf"] >= thr
        f = on_pitch_flags(d, poly)
        boxes_on += int((k & f).sum())
        off = int((k & ~f).sum())
        boxes_off += off
        frames_off += 1 if off else 0
    tops = [top1(d, thr, poly) for d in dets]
    found = [t for t in tops if t]
    moves = []
    for a, b in zip(tops[:-1], tops[1:]):
        if a and b:
            moves.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    return dict(frames_with=len(found), n=n, boxes_on=boxes_on, boxes_off=boxes_off, frames_off=frames_off,
                med_move=float(np.median(moves)) if moves else None,
                jumps=sum(m > 100 for m in moves), pairs=len(moves), tops=tops)


def agree(ref_tops, new_tops, tol=20.0):
    both = only_ref = only_new = far = 0
    for a, b in zip(ref_tops, new_tops):
        if a and b:
            if np.hypot(a[0] - b[0], a[1] - b[1]) <= tol:
                both += 1
            else:
                far += 1
        elif a:
            only_ref += 1
        elif b:
            only_new += 1
    return both, far, only_ref, only_new


def save_diff(raw, ref_tops, new_tops, outdir, thr, max_pictures=10):
    """a close-up around each model's on-pitch pick for frames where the two differ (red = current, green = new)"""
    import cv2
    os.makedirs(outdir, exist_ok=True)
    items = []
    for k, (a, b) in enumerate(zip(ref_tops, new_tops)):
        if a and b and np.hypot(a[0] - b[0], a[1] - b[1]) <= 20:
            continue
        if a or b:
            items.append(k)
    step = max(1, len(items) // max_pictures)
    h, w = raw[0].shape[:2]
    saved = 0
    for k in items[::step][:max_pictures]:
        img = cv2.cvtColor(raw[k], cv2.COLOR_RGB2BGR)
        for name, t, col in (("current", ref_tops[k], (0, 0, 255)), ("new", new_tops[k], (0, 200, 0))):
            if not t:
                continue
            cv2.circle(img, (int(t[0]), int(t[1])), 22, col, 2)
            cv2.putText(img, f"{name} {t[2]:.2f}", (int(t[0]) + 26, int(t[1]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        for name, t in (("current", ref_tops[k]), ("new", new_tops[k])):
            if not t:
                continue
            x0 = int(min(max(t[0] - 240, 0), w - 480))
            y0 = int(min(max(t[1] - 135, 0), h - 270))
            crop = cv2.resize(img[y0:y0 + 270, x0:x0 + 480], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(outdir, f"ball_thr{thr}_frame{k:03d}_{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
    print(f"   saved {saved} close-ups to {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=os.path.join("input_videos", "VID_20181002_211402_00_004.mp4"))
    ap.add_argument("--start-frame", type=int, default=240)
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--thresholds", default="0.2,0.3,0.4,0.5")
    ap.add_argument("--save-diff", default=None, metavar="DIR")
    ap.add_argument("--diff-threshold", type=float, default=0.3, help="threshold used for the close-up pictures")
    ap.add_argument("--pitch-db", default=DEFAULT_DB)
    args = ap.parse_args()

    import torch
    import ultralytics
    from shapely.geometry import Polygon
    device, half = ("mps", True) if torch.backends.mps.is_available() else ("cpu", False)
    print(f"torch {torch.__version__} | ultralytics {ultralytics.__version__} | device {device} | fp16 {half}")
    raw = read_frames(args.video, args.start_frame, args.frames)
    print(f"read {len(raw)} frames from frame {args.start_frame}")
    thresholds = [float(x) for x in args.thresholds.split(",")]
    min_conf = min(thresholds + [args.diff_threshold, 0.1])
    min_conf = min(min_conf, 0.1)

    poly = None
    if os.path.exists(args.pitch_db):
        poly = Polygon(load_pitch_polygon(args.pitch_db)).buffer(40)
    else:
        print(f"(pitch outline not found at {args.pitch_db}: 'on pitch' column skipped)")

    global POLY
    POLY = poly
    runs = []
    for label, path in WEIGHTS:
        print(f"\n=== {label}  ({path}) ===")
        if not os.path.exists(path):
            print("   missing, skipped")
            continue
        try:
            ms, dets = run(path, raw, args.imgsz, device, half, args.warmup, min_conf)
            print(f"   {ms:6.1f} ms/frame, {sum(len(d['conf']) for d in dets)} boxes at conf >= {min_conf}")
            runs.append(dict(label=label, ms=ms, dets=dets))
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc(limit=3)

    L = ["", f"SUMMARY  {len(raw)} consecutive frames from frame {args.start_frame}, {device}{' fp16' if half else ''}, "
             f"{args.imgsz}x{args.imgsz}, ultralytics {ultralytics.__version__}", "",
         "ms/frame: " + ", ".join(f"{r['label']} {r['ms']:.1f}" for r in runs), ""]
    head = (f"{'conf':>5} {'weights':<18}{'ball on':>9}{'boxes':>7}{'boxes':>7}{'frames w/':>10}{'median':>8}{'jumps':>9}   vs current (on-pitch ball): "
            f"same / far apart / only current / only new")
    L.append(f"{'':>5} {'':<18}{'pitch in':>9}{'on':>7}{'off':>7}{'off-pitch':>10}{'move':>8}{'':>9}")
    L += [head, "-" * len(head)]
    for thr in thresholds:
        ref = None
        for r in runs:
            s = summarise(r["dets"], thr, poly)
            move_txt = "n/a" if s["med_move"] is None else "%.0fpx" % s["med_move"]
            line = "%5.2f %-18s%5d/%-3d%7d%7d%10d%8s%5d/%-3d" % (thr, r["label"], s["frames_with"], s["n"], s["boxes_on"],
                                                               s["boxes_off"], s["frames_off"], move_txt, s["jumps"], s["pairs"])
            if ref is None:
                ref = s
            else:
                line += "   %d / %d / %d / %d" % agree(ref["tops"], s["tops"])
            L.append(line)
        L.append("")
    L += ["ball on pitch in = frames with at least one ball box whose centre is inside the pitch outline (+40 px); boxes on / off = ball boxes",
          "  inside / outside the outline; frames w/ off-pitch = frames with at least one ball box outside it (a stationary one = a false ball);",
          "  median move / jumps = movement of the most-confident ON-PITCH ball between consecutive frames (jumps = over 100 px in one frame);",
          "  agreement is on that on-pitch ball, within 20 px"]
    out = "\n".join(L)
    print(out)
    with open("compare_ball_models_results.txt", "w") as fh:
        fh.write(out + "\n")
    print("\n(saved to compare_ball_models_results.txt)")
    if args.save_diff and len(runs) >= 2:
        save_diff(raw, top1_list(runs[0], args.diff_threshold), top1_list(runs[-1], args.diff_threshold), args.save_diff, args.diff_threshold)


POLY = None


def top1_list(run_, thr):
    return [top1(d, thr, POLY) for d in run_["dets"]]


if __name__ == "__main__":
    main()
