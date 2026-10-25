"""Smoke-test edited_pitch_detector on a few broadcast-pitch training frames (or the edited video).

  python _scratch/smoke_edited_pitch.py
  python _scratch/smoke_edited_pitch.py --game 22 --n 4
"""
import argparse
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from edited_pitch_detector import EditedPitchAnnotator
from game_logger import db as gl_db, paths, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=22)
    ap.add_argument("--n", type=int, default=4, help="how many frames to annotate")
    ap.add_argument("--out", default=os.path.join("_scratch", "edited_pitch_smoke"))
    args = ap.parse_args()

    conn = gl_db.connect()
    rec = record.build_game_record(conn, args.game)
    g = None
    try:
        from game_logger import repo
        g = repo.get_game(conn, args.game)
    finally:
        conn.close()

    folder = paths.game_folder(args.game)
    jpg_glob = os.path.join(folder or "", "broadcast-pitch-images", "*", "*.jpg")
    frames = sorted(glob.glob(jpg_glob))
    # spread across the export
    if len(frames) > args.n:
        idxs = [int(round(i * (len(frames) - 1) / max(args.n - 1, 1))) for i in range(args.n)]
        frames = [frames[i] for i in idxs]
    else:
        frames = frames[: args.n]

    if not frames:
        # fall back to seeking the edited video
        edited = ((g or {}).get("videos") or {}).get("edited") or {}
        path = edited.get("path")
        if not path or not os.path.exists(path):
            raise SystemExit("No broadcast-pitch-images and no edited video for this game.")
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pick = [int(round(i * (n - 1) / max(args.n - 1, 1))) for i in range(args.n)]
        frames = []
        os.makedirs(args.out, exist_ok=True)
        for fi in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, im = cap.read()
            if ok:
                p = os.path.join(args.out, f"_src_f{fi:06d}.jpg")
                cv2.imwrite(p, im)
                frames.append(p)
        cap.release()

    os.makedirs(args.out, exist_ok=True)
    ann = EditedPitchAnnotator(game_record=rec, draw_keypoints=True, draw_outline=True)
    print(f"game {args.game}: {rec.get('title')}")
    print(f"annotating {len(frames)} frame(s) → {args.out}")
    for src in frames:
        im = cv2.imread(src)
        if im is None:
            print("  skip (unreadable)", src)
            continue
        out = ann.annotate(im)
        name = os.path.splitext(os.path.basename(src))[0] + "_annotated.jpg"
        dest = os.path.join(args.out, name)
        cv2.imwrite(dest, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(
            f"  {name}: kpts={len(ann.last_detections)} "
            f"H={'OK' if ann.last_ok else 'FAIL'} "
            f"inliers={ann.homography.last_inliers}/{ann.homography.last_n}"
        )
        for d in sorted(ann.last_detections, key=lambda x: x["index"]):
            print(f"    [{d['index']:2d}] {d['name']:<22} conf={d['conf']:.2f}  xy=({d['x']:.0f},{d['y']:.0f})")
    print("done — check the orange labels: if names don't match the geometry, reorder KEYPOINT_NAMES in edited_pitch_detector/keypoints.py")


if __name__ == "__main__":
    main()
