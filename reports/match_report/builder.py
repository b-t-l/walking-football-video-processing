"""CreateMatchReport: works out the match metrics from the detections database and writes the PDF (and a .json of the numbers)."""
import json
import math
import sqlite3
import time

from matplotlib.backends.backend_pdf import PdfPages

from . import metrics, pages, style


def _kit_rgb(db_path, team):
    """The team's kit colour as the pipeline stored it (BGR text like '(40, 200, 75)') -> (r, g, b)."""
    try:
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = c.execute("SELECT team_color, COUNT(*) n FROM detected_objects WHERE team = ? AND team_color IS NOT NULL GROUP BY team_color ORDER BY n DESC LIMIT 1", (team,)).fetchone()
        c.close()
        if row:
            b, g, r = [int(v) for v in row[0].strip("()").split(",")]
            return (r, g, b)
    except Exception:
        pass
    return None


class CreateMatchReport:
    def __init__(self, db_path, OUTPUT_PATH, TEAMS, GAME_RECORD, DETECTION_FPS, report_file=None):
        self.db_path, self.out, self.rec, self.fps = db_path, OUTPUT_PATH, GAME_RECORD, DETECTION_FPS
        a, b = GAME_RECORD.get("team_a"), GAME_RECORD.get("team_b")
        self.teams = [a, b]
        self.report_file = report_file

    def run(self):
        print("-" * 98)
        print("MATCH REPORT: team statistics for both teams, saved as a PDF")
        print("-" * 98)
        t0 = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rec = self.rec
        try:
            speed = float(rec.get("running_speed_km_h") or 15)
        except (TypeError, ValueError):
            speed = 15.0
        M = metrics.compute_metrics(self.db_path, rec, self.teams, speed)
        labels = M["team_labels"]
        for name, lab in labels.items():
            if name != lab:
                print(f"Note: the game's team '{name}' is called '{lab}' in the detections database; the report uses '{name}'.")
        colours = style.team_colours(_kit_rgb(self.db_path, labels[self.teams[0]]), _kit_rgb(self.db_path, labels[self.teams[1]]))
        link = rec.get("statistics_output_video")
        link = link if isinstance(link, str) and link.startswith("http") else None
        ctx = pages.Ctx(M, rec, colours, link_base=link, created=time.strftime("%d %b %Y %H:%M"))
        title = str(rec.get("title") or " vs ".join(self.teams))
        frame = style.Frame(title, f"{title} · {rec.get('date', '')} · Match report", len(pages.PAGES))
        if self.report_file:
            pdf_path = self.report_file
        else:
            from utils import GameFiles
            pdf_path = GameFiles.report_file(self.out, rec, "match-report", stamp, "pdf")
        import os
        with PdfPages(pdf_path, metadata={"Title": f"Match report: {title}", "Author": "Polis Walking Football"}) as pdf:
            for fn in pages.PAGES:
                try:
                    fig = fn(frame, ctx)
                except Exception as e:             # one page failing must not lose the whole report
                    print(f"⚠️ The '{fn.__name__}' page could not be built: {e}")
                    frame.n = frame.n if frame.n else 1
                    fig = frame.new(fn.__name__.replace("_", " ").capitalize(), "This page could not be built for this game.")
                    fig.text(0.05, 0.75, f"{type(e).__name__}: {e}", fontsize=10, color=style.MUTED)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                style.plt.close(fig)
        json_path = os.path.splitext(pdf_path)[0] + ".json"
        slim = {k: v for k, v in M.items() if k not in ("heat_ball", "heat_team")}
        slim["territory"] = {k: v for k, v in M["territory"].items() if k != "ball_xy"}
        with open(json_path, "w") as f:
            json.dump(slim, f, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o))
        print(f"Match report generated: {pdf_path}")
        print(f"✅ Processing time: {int(time.time() - t0)} sec")
        return pdf_path
