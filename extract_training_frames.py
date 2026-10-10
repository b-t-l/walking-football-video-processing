"""
extract_training_frames.py  --  pull one frame every N seconds from the match videos, for labelling in Roboflow.

    python extract_training_frames.py "/Volumes/LaCie/walking-football-videos/mens/flattened for video analysis"
    python extract_training_frames.py <video folder> --every 10 --first 5 --out <folder for the images>

* frames are decoded straight from the video file with ffmpeg (full 1920x1080, JPEG quality ~95), not screenshots
* one frame at second 5, 15, 25 ... of each video (--first / --every); the last 3 seconds are skipped
* images go to <out>/<video name>/<video name>_t00015.jpg  (t = seconds into that video, so any image can be traced back)
  <out> defaults to <video folder>/training_frames
* a manifest.csv is written/extended next to them: video, second, frame number, file
* --budget 150 stops after 150 s so it can be run in short steps; existing images are never overwritten, so you can rerun it (for example with a different --first to add a second set)
"""
import os
import re
import csv
import sys
import json
import argparse
import time
import subprocess


def video_info(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate,width,height:format=duration", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    num, den = j["streams"][0]["r_frame_rate"].split("/")
    return float(j["format"]["duration"]), float(num) / float(den), j["streams"][0]["width"], j["streams"][0]["height"]


def clean_name(filename):
    stem = os.path.splitext(filename)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "", stem.replace("(", "_").replace(")", ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--every", type=float, default=10.0, help="seconds between frames")
    ap.add_argument("--first", type=float, default=5.0, help="second of the first frame")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (rerun to carry on); 0 = no limit")
    args = ap.parse_args()

    started = time.time()
    out_root = args.out or os.path.join(args.folder, "training_frames")
    os.makedirs(out_root, exist_ok=True)
    videos = sorted(f for f in os.listdir(args.folder) if f.lower().endswith((".mp4", ".mov", ".mkv")) and not f.startswith("."))
    manifest = os.path.join(out_root, "manifest.csv")
    new_file = not os.path.exists(manifest)
    mf = open(manifest, "a", newline="")
    w = csv.writer(mf)
    if new_file:
        w.writerow(["video", "second", "frame_number", "file"])

    for v in videos:
        path = os.path.join(args.folder, v)
        name = clean_name(v)
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        dur, fps, width, height = video_info(path)
        times = []
        t = args.first
        while t < dur - 3:
            times.append(t)
            t += args.every
        print(f"{v}: {dur:.0f} s, {fps:g} fps, {width}x{height} -> {len(times)} frames", flush=True)
        # one quick seek per frame (fast: ffmpeg jumps to the nearest key frame, then decodes only up to the requested second)
        done = 0
        for t in times:
            dst = os.path.join(out_dir, f"{name}_t{int(round(t)):05d}.jpg")
            if os.path.exists(dst):
                continue
            if args.budget and time.time() - started > args.budget:
                print(f"   time budget used up - run the same command again to carry on ({len(times) - done} frames left in this video)", flush=True)
                mf.close()
                return
            tmp = dst + ".part.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-q:v", "2", tmp], check=True)
            os.rename(tmp, dst)
            w.writerow([v, f"{t:g}", int(round(t * fps)), os.path.relpath(dst, out_root)])
            mf.flush()
            done += 1
        print(f"   done: {len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])} images in {out_dir}", flush=True)
    mf.close()
    print("finished. manifest:", manifest)


if __name__ == "__main__":
    main()
