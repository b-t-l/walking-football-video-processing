"""The review sheet: about 24 frames from the game with the detections drawn on and numbered, plus a CSV to fill in.

You look at each frame in the PDF and, in the matching row of the CSV, note how many players the boxes missed, how many boxes are
wrong, and whether the ball was visible and found. The next time the detection report is run, it reads the filled CSV and shows
real precision and recall figures. The same frames are used every time (the CSV remembers them), so the figures are comparable
after a change to the detection settings."""
import math
import os
import random

import numpy as np
import pandas as pd

from ..match_report import metrics, style as S

N_FRAMES = 24
CSV_COLUMNS = ["row", "frame_id", "time", "page", "players_shown", "ball_shown",
               "players_on_pitch", "wrong_player_boxes", "missed_players", "wrong_team_boxes", "ball_visible", "ball_box_correct", "notes"]
FILL_COLUMNS = ["players_on_pitch", "wrong_player_boxes", "missed_players", "wrong_team_boxes", "ball_visible", "ball_box_correct", "notes"]
HOW_TO = ("For each row, look at the numbered frame in the PDF. players_on_pitch = how many players you can really see on the pitch; "
          "wrong_player_boxes = boxes that are not a player (or are on the wrong team: put those in wrong_team_boxes); missed_players = players with no box; "
          "ball_visible = yes/no; ball_box_correct = yes/no (leave empty if no ball box is drawn).")


def pick_frames(A, n=N_FRAMES):
    ids = A.get("frame_ids") or []
    if not ids:
        return []
    rng = random.Random(1000 + int(A.get("seed", 0)))
    edges = np.linspace(0, len(ids), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out.append(int(ids[rng.randrange(a, b)]))
    return out


def _frame_rows(conn, frame_id):
    return pd.read_sql("SELECT class_name, confidence, team, xmin, ymin, xmax, ymax, is_inside_pitch AS ins, tracker_id "
                       "FROM detected_objects WHERE frame_id = ?", conn, params=(int(frame_id),))


def read_review(csv_path, min_rows=8):
    """Figures from a filled-in review CSV, or None if it has not been (sufficiently) filled in."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        d = pd.read_csv(csv_path)
    except Exception:
        return None
    for c in ("players_on_pitch", "wrong_player_boxes", "missed_players", "wrong_team_boxes", "players_shown", "ball_shown"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    done = d[d["players_on_pitch"].notna()].copy()
    if len(done) < min_rows:
        return {"rows": int(len(done)), "of": int(len(d)), "ready": False}
    done["wrong_player_boxes"] = done["wrong_player_boxes"].fillna(0)
    done["missed_players"] = done["missed_players"].fillna(0)
    done["wrong_team_boxes"] = done["wrong_team_boxes"].fillna(0)
    shown = float(done["players_shown"].sum())
    wrong = float(done["wrong_player_boxes"].sum())
    missed = float(done["missed_players"].sum())
    right = max(shown - wrong, 0.0)
    prec = right / shown if shown else None
    rec = right / (right + missed) if (right + missed) else None
    team_wrong = float(done["wrong_team_boxes"].sum())
    vis = done[done["ball_visible"].astype(str).str.lower().str.startswith("y")]
    found = vis[vis["ball_box_correct"].astype(str).str.lower().str.startswith("y")]
    shown_ball = done[done["ball_shown"].fillna(0) > 0]
    ball_right = shown_ball[shown_ball["ball_box_correct"].astype(str).str.lower().str.startswith("y")]
    return {"ready": True, "rows": int(len(done)), "of": int(len(d)), "players_shown": shown, "wrong": wrong, "missed": missed,
            "player_precision": prec, "player_recall": rec,
            "team_wrong_pct": 100 * team_wrong / shown if shown else None,
            "ball_visible_frames": int(len(vis)), "ball_found_of_visible_pct": 100 * len(found) / len(vis) if len(vis) else None,
            "ball_boxes": int(len(shown_ball)), "ball_precision_pct": 100 * len(ball_right) / len(shown_ball) if len(shown_ball) else None}


def make_sheet(db_path, A, pdf_path, csv_path, colours):
    """Writes the PDF and, unless it exists already, the CSV. Returns (pdf_path or None, reason if None)."""
    try:
        import cv2
    except Exception:
        return None, "OpenCV is not available"
    video = A.get("video")
    if not video or not os.path.exists(video):
        return None, f"the analysis video was not found ({video})"
    frames = None
    if os.path.exists(csv_path):
        try:
            frames = [int(v) for v in pd.read_csv(csv_path)["frame_id"].tolist()]
        except Exception:
            frames = None
    if not frames:
        frames = pick_frames(A)
    if not frames:
        return None, "no frames to sample"
    import sqlite3
    from matplotlib.backends.backend_pdf import PdfPages
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cap = cv2.VideoCapture(video)
    rows, shots = [], []
    team_col = {A["teams"][0]: colours[0], A["teams"][1]: colours[1]}
    inv = {lab: n for n, lab in A["team_labels"].items()}

    def bgr(hexcol):
        h = hexcol.lstrip("#")
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

    for k, fid in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, img = cap.read()
        if not ok:
            continue
        H0, W0 = img.shape[:2]
        sc = 1900 / W0
        img = cv2.resize(img, (1900, int(H0 * sc)))
        d = _frame_rows(conn, fid)
        d["team"] = d["team"].map(inv).fillna(d["team"])
        n_pl = 0
        n_ball = 0
        n_gk = 0
        for r in d.itertuples(index=False):
            x0, y0, x1, y1 = [int(v * sc) for v in (r.xmin, r.ymin, r.xmax, r.ymax)]
            if r.class_name == "ball":
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                cv2.circle(img, (cx, cy), max(18, (x1 - x0)), (255, 0, 255), 3)
                cv2.putText(img, "BALL", (cx + 14, cy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                n_ball += int(r.ins == 1)
                continue
            if r.ins != 1:
                cv2.rectangle(img, (x0, y0), (x1, y1), (170, 170, 170), 1)
                continue
            if r.class_name == "goalkeeper" or r.team == "Goalkeeper":
                c = (219, 130, 37); n_gk += 1
            elif r.class_name == "referee":
                c = (0, 200, 255)
            else:
                c = bgr(team_col.get(r.team, "#888888")); n_pl += 1
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 3)
            cv2.putText(img, f"{r.confidence:.2f}", (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        tsec = fid / A["video_fps"]
        rows.append({"row": f"R{k + 1:02d}", "frame_id": int(fid), "time": metrics.mmss(tsec), "page": k + 2, "players_shown": n_pl, "ball_shown": n_ball})
        shots.append((f"R{k + 1:02d}", fid, tsec, n_pl, n_gk, n_ball, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    conn.close()
    if not shots:
        return None, "no frames could be read from the video"
    import matplotlib.pyplot as plt
    with PdfPages(pdf_path, metadata={"Title": "Detection review sheet", "Author": "Polis Walking Football"}) as pdf:
        fig = plt.figure(figsize=(S.PAGE_W, S.PAGE_H), facecolor=S.BG)
        fig.text(0.05, 0.92, "Detection review sheet", fontsize=24, fontweight="bold")
        fig.text(0.05, 0.86, f"{len(shots)} frames from game {A.get('game_id', '')}, with the detections drawn on.", fontsize=12, color=S.MUTED)
        lines = ["1. Look at each numbered frame. Boxes: team colours for players, orange for goalkeepers, pink circle for the ball, grey = outside the pitch (ignored).",
                 "2. In the CSV file saved next to this PDF, fill in the row with the same label (R01, R02, ...):",
                 "     " + "\n     ".join(__import__("textwrap").wrap(HOW_TO, 105)),
                 "3. Save the CSV and run the Detection report again. It will add a page with the measured accuracy.",
                 "Fill in at least 8 rows (all 24 is better). Keep the CSV file name; the same frames are used each time, so you can re-check after changing a setting."]
        fig.text(0.05, 0.78, "\n\n".join(lines), fontsize=11, va="top", linespacing=1.3)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        for lab, fid, tsec, n_pl, n_gk, n_ball, im in shots:
            fig = plt.figure(figsize=(S.PAGE_W, S.PAGE_H), facecolor=S.BG)
            h, w = im.shape[:2]
            width = 0.92
            height = width * (S.PAGE_W / S.PAGE_H) * h / w
            if height > 0.80:
                height = 0.80
                width = height * w / h / (S.PAGE_W / S.PAGE_H)
            ax = fig.add_axes([0.04, 0.90 - height, width, height])
            ax.imshow(im); ax.axis("off")
            fig.text(0.04, 0.945, f"{lab}   frame {fid}   at {metrics.mmss(tsec)}", fontsize=15, fontweight="bold", va="center")
            fig.text(0.96, 0.945, f"boxes drawn: {n_pl} players, {n_gk} goalkeepers, ball {'found' if n_ball else 'not found'}", fontsize=11, color=S.MUTED, ha="right", va="center")
            pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
    if not os.path.exists(csv_path):
        df = pd.DataFrame(rows)
        for c in CSV_COLUMNS:
            if c not in df:
                df[c] = ""
        df = df[CSV_COLUMNS]
        df.to_csv(csv_path, index=False)
    return pdf_path, None
