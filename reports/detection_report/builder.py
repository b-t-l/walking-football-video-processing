"""CreateDetectionsReport: scores every detection stage for one game, ranks what to fix, and writes the PDF.
Also makes the review sheet (frames to check by eye) and reads back your filled-in review CSV."""
import json
import math
import os
import sqlite3
import time

from matplotlib.backends.backend_pdf import PdfPages

from ..match_report import style
from ..match_report.builder import _kit_rgb, _team_rgb
from . import advice, analysis, pages, review


class CreateDetectionsReport:
    def __init__(self, db_path, OUTPUT_PATH, TEAMS, GAME_RECORD, DETECTION_FPS, report_file=None, make_review_sheet=True):
        self.db_path, self.out, self.rec, self.fps = db_path, OUTPUT_PATH, GAME_RECORD, DETECTION_FPS
        self.teams = [GAME_RECORD.get("team_a"), GAME_RECORD.get("team_b")]
        self.TEAMS = TEAMS
        self.report_file = report_file
        self.make_review_sheet = make_review_sheet

    def _prefix(self):
        try:
            return f"{int(self.rec['game_id']):04d}"
        except (TypeError, ValueError, KeyError):
            return str(self.rec.get("game_id", "game"))

    def run(self):
        print("-" * 98)
        print("DETECTION REPORT: how well each detection stage worked, and what to fix first")
        print("-" * 98)
        t0 = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rec = self.rec
        A = analysis.analyse(self.db_path, rec, self.teams)
        A["game_id"] = rec.get("game_id")
        A["seed"] = int(rec["game_id"]) if str(rec.get("game_id", "")).isdigit() else 0
        for name, lab in A["team_labels"].items():
            if name != lab:
                print(f"Note: the game's team '{name}' is called '{lab}' in the detections database; the report uses '{name}'.")
        kits = [_team_rgb(self.db_path, self.TEAMS, t, A["team_labels"][t]) for t in self.teams]
        tc = A.get("teams_check")
        if tc is not None and all(kits):                  # the kit card shows the Game Logger colours too
            tc["kit_rgb"] = {t: k for t, k in zip(self.teams, kits)}
            tc["kit_distance"] = math.dist(kits[0], kits[1])
        tips = advice.advise(A)
        colours = style.team_colours(kits[0], kits[1])
        os.makedirs(self.out, exist_ok=True)
        prefix = self._prefix()
        csv_path = os.path.join(self.out, f"{prefix}_detection-review.csv")
        sheet_path = os.path.join(self.out, f"{prefix}_detection-review-sheet.pdf")
        # read any review you filled in BEFORE making the sheet (the sheet never overwrites your CSV)
        rv = review.read_review(csv_path)
        note, files = None, {}
        if self.make_review_sheet:
            try:
                p, why = review.make_sheet(self.db_path, A, sheet_path, csv_path, colours)
            except Exception as e:                             # the review sheet is a bonus: never lose the report for it
                p, why = None, f"{type(e).__name__}: {e}"
            if p:
                files = {"sheet": os.path.basename(p), "csv": os.path.basename(csv_path)}
                print(f"Review sheet: {p}")
                print(f"Review CSV (fill this in, then run this report again): {csv_path}")
            else:
                note = why
                print(f"Note: the review sheet was not made ({why}).")
            if rv is None and os.path.exists(csv_path):
                rv = review.read_review(csv_path)
        ctx = pages.Ctx(A, rec, tips, rv, colours, created=time.strftime("%d %b %Y %H:%M"), review_note=note, files=files)
        title = str(rec.get("title") or " vs ".join(self.teams))
        frame = style.Frame(title, f"{title} · {rec.get('date', '')} · Detection report", len(pages.PAGES))
        if self.report_file:
            pdf_path = self.report_file
        else:
            from utils import GameFiles
            pdf_path = GameFiles.report_file(self.out, rec, "detection-report", stamp, "pdf")
        with PdfPages(pdf_path, metadata={"Title": f"Detection report: {title}", "Author": "Polis Walking Football"}) as pdf:
            for fn in pages.PAGES:
                try:
                    fig = fn(frame, ctx)
                except Exception as e:
                    print(f"⚠️ The '{fn.__name__}' page could not be built: {type(e).__name__}: {e}")
                    frame.n = frame.n if frame.n else 1
                    fig = frame.new(fn.__name__.replace("_", " ").capitalize(), "This page could not be built for this game.")
                    fig.text(0.05, 0.75, f"{type(e).__name__}: {e}", fontsize=10, color=style.MUTED)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                style.plt.close(fig)
        slim = {k: v for k, v in A.items() if k not in ("frame_ids",)}
        for k in ("players", "ball", "goalkeepers", "tracking"):
            slim[k] = {a: b for a, b in A[k].items() if a not in ("heat", "dropout_heat", "ends_heat", "xy")}
        slim["advice"] = tips
        with open(os.path.splitext(pdf_path)[0] + ".json", "w") as f:
            json.dump(slim, f, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o))
        print(f"Detection report generated: {pdf_path}")
        print(f"Overall detection health {A['overall']:.0f}/100. Fix first: " + ", ".join(a["title"] for a in tips[:3]))
        print(f"✅ Processing time: {int(time.time() - t0)} sec")
        return pdf_path
