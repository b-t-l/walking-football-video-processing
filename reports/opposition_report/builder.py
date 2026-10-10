"""CreateOppositionReport: one scouting PDF for each of the two teams in a game, written for the coach of the other team."""
import json
import math
import os
import re
import time

from matplotlib.backends.backend_pdf import PdfPages

from ..match_report import style
from ..match_report.builder import _kit_rgb, _team_rgb
from . import pages, scout


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "team"


class CreateOppositionReport:
    def __init__(self, db_path, OUTPUT_PATH, TEAMS, GAME_RECORD, DETECTION_FPS, report_file=None):
        self.db_path, self.out, self.rec, self.fps = db_path, OUTPUT_PATH, GAME_RECORD, DETECTION_FPS
        self.teams = [GAME_RECORD.get("team_a"), GAME_RECORD.get("team_b")]
        self.TEAMS = TEAMS
        self.report_file = report_file          # (testing) a path to write the first team's report to; the second gets a suffix

    def run(self):
        print("-" * 98)
        print("OPPOSITION REPORTS: one scouting report for each team, written for the other team's coach")
        print("-" * 98)
        t0 = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rec = self.rec
        try:
            speed = float(rec.get("running_speed_km_h") or 15)
        except (TypeError, ValueError):
            speed = 15.0
        S = scout.scout(self.db_path, rec, self.teams, speed)
        labels = S["M"]["team_labels"]
        for name, lab in labels.items():
            if name != lab:
                print(f"Note: the game's team '{name}' is called '{lab}' in the detections database; the report uses '{name}'.")
        colours = style.team_colours(_team_rgb(self.db_path, self.TEAMS, self.teams[0], labels[self.teams[0]]), _team_rgb(self.db_path, self.TEAMS, self.teams[1], labels[self.teams[1]]))
        link = rec.get("statistics_output_video")
        link = link if isinstance(link, str) and link.startswith("http") else None
        ctx = pages.Ctx(S, rec, colours, link_base=link, created=time.strftime("%d %b %Y %H:%M"))
        title = str(rec.get("title") or " vs ".join(self.teams))
        made = []
        for tm in self.teams:
            frame = style.Frame(title, f"{title} · {rec.get('date', '')} · Opposition report: {tm}", len(pages.PAGES))
            if self.report_file:
                base, ext = os.path.splitext(self.report_file)
                pdf_path = f"{base}_{_slug(tm)}{ext}"
            else:
                from utils import GameFiles
                pdf_path = GameFiles.report_file(self.out, rec, f"opposition-report_{_slug(tm)}", stamp, "pdf")
            with PdfPages(pdf_path, metadata={"Title": f"Opposition report: {tm}", "Author": "Polis Walking Football"}) as pdf:
                for fn in pages.PAGES:
                    try:
                        fig = fn(frame, ctx, tm)
                    except Exception as e:             # one page failing must not lose the whole report
                        print(f"⚠️ The '{fn.__name__}' page for {tm} could not be built: {e}")
                        frame.n = frame.n if frame.n else 1
                        fig = frame.new(f"{tm}: {fn.__name__.replace('_', ' ')}", "This page could not be built for this game.")
                        fig.text(0.05, 0.75, f"{type(e).__name__}: {e}", fontsize=10, color=style.MUTED)
                    pdf.savefig(fig, facecolor=fig.get_facecolor())
                    style.plt.close(fig)
            made.append(pdf_path)
            print(f"Opposition report for {tm} generated: {pdf_path}")
        # the numbers behind the reports, as json
        json_path = os.path.splitext(made[0])[0] + ".json"
        slim = {"teams": S["teams"], "periods": S["periods"], "items": S["items"],
                "per_team": {tm: {k: v for k, v in T.items() if not k.startswith("heat_") and k not in ("wins", "losses")} for tm, T in S["per_team"].items()}}
        with open(json_path, "w") as f:
            json.dump(slim, f, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o))
        print(f"✅ Processing time: {int(time.time() - t0)} sec")
        return made
