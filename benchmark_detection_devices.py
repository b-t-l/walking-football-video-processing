"""
benchmark_detection_devices.py  --  how fast can this Mac run the two YOLO models, and are the answers the same?

Run from the application folder (the one containing main.py), in the same python environment you use for main.py:

    python benchmark_detection_devices.py                  # cpu (baseline) vs mps vs mps_half vs coreml, 20 frames
    python benchmark_detection_devices.py --imgsz 1280     # same, but feed the models a smaller stretched square
    python benchmark_detection_devices.py --configs mps --batch 4

What it does
  * reads a few consecutive frames of the game video and prepares them EXACTLY like the pipeline does
    (RGB, stretched to a square, float 0-1, CHW) -- the models were trained on stretched 1920x1920 images
  * runs the player model and the ball model on those frames for each configuration and times the whole
    predict() call per frame (preprocess + inference + NMS), after a warm-up
  * the first run is always the current pipeline setting (cpu, fp32, 1920): its detections are the reference.
    Every other configuration is compared with it: recall = share of reference boxes found again (IoU >= 0.5),
    precision = share of its own boxes that match a reference box, mean IoU of the matches.
  * a configuration that crashes (for example the old MPS "Output channels > 65536" error) is reported in the
    table with its error message and the benchmark carries on.

Nothing here touches your database or your pipeline code. Results are also saved to benchmark_results.txt.
"""
import os
import sys
import time
import argparse
import traceback

# if one MPS op is unsupported, run just that op on the CPU rather than failing (set before torch is imported)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

PLAYER_MODEL = os.path.join("models", "objects", "best-v10-1920x1920.pt")
BALL_MODEL = os.path.join("models", "ball", "best-v2-1920x1920.pt")
PLAYER_CONF = 0.6      # same as data/create_database.py
BALL_CONF = 0.3
BASE_IMGSZ = 1920

CONFIGS = {
    "cpu":      dict(device="cpu", half=False, fmt="pt"),
    "mps":      dict(device="mps", half=False, fmt="pt"),
    "mps_half": dict(device="mps", half=True,  fmt="pt"),
    # exported once to a Core ML package (FP16); runs on the GPU / Neural Engine via Apple's Core ML runtime
    "coreml":   dict(device="cpu", half=False, fmt="coreml"),
}


# ----------------------------------------------------------------------------------------------- comparison maths
def iou_matrix(a, b):
    """a (n,4), b (m,4) as x1,y1,x2,y2 -> (n,m) IoU"""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_frame(ref, test, iou_thr=0.5, class_aware=True):
    """greedy one-to-one matching, best IoU first. ref/test = dict(xyxy=(n,4), cls=(n,)).
    returns (n_ref, n_test, n_matched, [ious of matches])"""
    ref_xyxy, test_xyxy = np.asarray(ref["xyxy"]).reshape(-1, 4), np.asarray(test["xyxy"]).reshape(-1, 4)
    n_ref, n_test = len(ref_xyxy), len(test_xyxy)
    if n_ref == 0 or n_test == 0:
        return n_ref, n_test, 0, []
    m = iou_matrix(ref_xyxy, test_xyxy)
    if class_aware:
        same = np.asarray(ref["cls"]).reshape(-1, 1) == np.asarray(test["cls"]).reshape(1, -1)
        m = np.where(same, m, 0.0)
    pairs = sorted(((m[i, j], i, j) for i in range(n_ref) for j in range(n_test) if m[i, j] >= iou_thr), reverse=True)
    used_r, used_t, ious = set(), set(), []
    for v, i, j in pairs:
        if i in used_r or j in used_t:
            continue
        used_r.add(i)
        used_t.add(j)
        ious.append(float(v))
    return n_ref, n_test, len(ious), ious


def compare_runs(ref_dets, test_dets, class_aware=True):
    """frame lists of dicts -> recall, precision, mean IoU (as fractions; None when there is nothing to compare)"""
    tr = tt = tm = 0
    all_ious = []
    for r, t in zip(ref_dets, test_dets):
        n_r, n_t, n_m, ious = match_frame(r, t, class_aware=class_aware)
        tr, tt, tm = tr + n_r, tt + n_t, tm + n_m
        all_ious += ious
    recall = tm / tr if tr else None
    precision = tm / tt if tt else None
    mean_iou = float(np.mean(all_ious)) if all_ious else None
    return dict(ref_boxes=tr, test_boxes=tt, matched=tm, recall=recall, precision=precision, mean_iou=mean_iou)


def fmt_pct(x):
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


# ----------------------------------------------------------------------------------------------- video + models
def load_raw_frames(video, start_frame, n):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}  (run this from the folder that contains main.py)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit("No frames could be read - check --video and --start-frame")
    return frames


def to_tensor(rgb, size):
    """same preparation as ImageUtils.create_tensor_and_resize: stretch to size x size, float 0-1, CHW"""
    import cv2
    import torch
    img = cv2.resize(rgb, (size, size))
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0


def coreml_package(pt_path, imgsz):
    """export the .pt to a Core ML package next to it (once) and return its path"""
    from ultralytics import YOLO
    package = os.path.splitext(pt_path)[0] + ".mlpackage"
    if not os.path.exists(package):
        print(f"      exporting {pt_path} to Core ML at {imgsz}x{imgsz} (FP16) - one-off, can take a few minutes ...")
        exported = YOLO(pt_path).export(format="coreml", imgsz=imgsz, half=True, nms=False)
        package = str(exported)
    return package


def run_model(pt_path, conf, cfg, imgsz, raw_frames, batch, warmup):
    """returns (ms_per_frame, detections) ; detections = list of dict(xyxy normalised 0-1, cls)"""
    import torch
    from ultralytics import YOLO

    if cfg["fmt"] == "coreml":
        model = YOLO(coreml_package(pt_path, imgsz), task="detect")
    else:
        model = YOLO(pt_path)

    tensors = [to_tensor(f, imgsz) for f in raw_frames]
    kw = dict(conf=conf, device=cfg["device"], half=cfg["half"], imgsz=(imgsz, imgsz), verbose=False)

    def sync():
        if cfg["device"] == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    for _ in range(warmup):                      # first calls compile kernels / load the model: not timed
        model.predict(torch.stack(tensors[:batch]), **kw)
    sync()

    times, dets = [], []
    for i in range(0, len(tensors), batch):
        chunk = torch.stack(tensors[i:i + batch])
        t0 = time.perf_counter()
        results = model.predict(chunk, **kw)
        sync()
        times.append((time.perf_counter() - t0) / len(chunk))
        for r in results:
            boxes = r.boxes
            xyxy = boxes.xyxy.cpu().numpy() / float(imgsz) if len(boxes) else np.zeros((0, 4))
            cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros((0,), dtype=int)
            dets.append(dict(xyxy=xyxy, cls=cls))
    return 1000.0 * float(np.mean(times)), dets


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=os.path.join("input_videos", "VID_20181002_211402_00_004.mp4"))
    ap.add_argument("--start-frame", type=int, default=240, help="first frame to read (240 = 10 s in)")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--configs", default="mps,mps_half,coreml", help="comma list from: " + ", ".join(CONFIGS))
    ap.add_argument("--imgsz", type=int, default=BASE_IMGSZ, help="square input size fed to the models (stretched, like training)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--models", default="player,ball", help="player, ball or player,ball")
    args = ap.parse_args()

    import torch
    import ultralytics
    print(f"torch {torch.__version__} | ultralytics {ultralytics.__version__} | mps available: "
          f"{torch.backends.mps.is_available()} | cwd: {os.getcwd()}")

    raw = load_raw_frames(args.video, args.start_frame, args.frames)
    print(f"read {len(raw)} frames of {args.video} starting at frame {args.start_frame}")

    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in wanted:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config '{c}' (choose from {', '.join(CONFIGS)})")
    models = []
    if "player" in args.models:
        models.append(("player", PLAYER_MODEL, PLAYER_CONF, True))     # class-aware matching
    if "ball" in args.models:
        models.append(("ball", BALL_MODEL, BALL_CONF, False))

    # (label, config name, imgsz) - the reference run first
    plan = [("cpu @1920 (reference)", "cpu", BASE_IMGSZ)]
    for c in wanted:
        if c == "cpu" and args.imgsz == BASE_IMGSZ:
            continue
        plan.append((f"{c} @{args.imgsz}" + (f" batch{args.batch}" if args.batch != 1 else ""), c, args.imgsz))

    table = []          # one row per plan item
    reference = {}      # model name -> detections from the reference run
    for label, cname, size in plan:
        print(f"\n=== {label} ===")
        row = dict(label=label)
        for mname, path, conf, class_aware in models:
            is_ref = label.endswith("(reference)")
            try:
                ms, dets = run_model(path, conf, CONFIGS[cname], size, raw, 1 if is_ref else args.batch, args.warmup)
                row[f"{mname}_ms"] = ms
                if is_ref:
                    reference[mname] = dets
                    row[f"{mname}_cmp"] = None
                else:
                    row[f"{mname}_cmp"] = compare_runs(reference[mname], dets, class_aware)
                print(f"   {mname}: {ms:7.1f} ms/frame")
            except Exception as e:                       # report and carry on with the next one
                row[f"{mname}_error"] = f"{type(e).__name__}: {str(e)[:160]}"
                print(f"   {mname}: FAILED -> {row[f'{mname}_error']}")
                traceback.print_exc(limit=2)
        table.append(row)

    # ------------------------------------------------------------------------------------------- summary
    lines = []
    ref_ms = {m[0]: table[0].get(f"{m[0]}_ms") for m in models}
    lines.append("")
    lines.append(f"SUMMARY  ({len(raw)} frames, frames {args.start_frame}-{args.start_frame + len(raw) - 1}, "
                 f"torch {torch.__version__}, ultralytics {ultralytics.__version__})")
    lines.append("ms/frame = whole predict() call; speed-up is against the cpu reference; recall/precision/IoU compare with the reference detections")
    lines.append("")
    header = f"{'configuration':<30}"
    for mname, *_ in models:
        header += f"| {mname:^6} ms  x-faster  recall  prec.  IoU  "
    lines.append(header)
    lines.append("-" * len(header))
    for row in table:
        text = f"{row['label']:<30}"
        for mname, *_ in models:
            if f"{mname}_error" in row:
                text += f"| FAILED: {row[f'{mname}_error'][:60]:<50}"
                continue
            ms = row.get(f"{mname}_ms")
            fast = (ref_ms[mname] / ms) if (ms and ref_ms.get(mname)) else None
            cmp_ = row.get(f"{mname}_cmp")
            if cmp_ is None:
                text += f"| {ms:8.1f}     1.0x        -      -     -  "
            else:
                iou = "  n/a" if cmp_["mean_iou"] is None else f"{cmp_['mean_iou']:.2f}"
                text += f"| {ms:8.1f}  {fast:6.1f}x  {fmt_pct(cmp_['recall'])} {fmt_pct(cmp_['precision'])} {iou:>5}  "
        lines.append(text)
    lines.append("")
    lines.append("reading it: recall below ~97% for the players, or any clear drop for the ball, means that setting changes the detections")
    lines.append("            (for the ball, compare the two 'n boxes' totals as well: the ball is rare and small)")
    for row in table[1:]:
        for mname, *_ in models:
            c = row.get(f"{mname}_cmp")
            if c:
                lines.append(f"  {row['label']:<28} {mname}: reference boxes {c['ref_boxes']}, this setting {c['test_boxes']}, matched {c['matched']}")
    out = "\n".join(lines)
    print(out)
    with open("benchmark_results.txt", "w") as fh:
        fh.write(out + "\n")
    print("\n(saved to benchmark_results.txt)")


if __name__ == "__main__":
    main()
