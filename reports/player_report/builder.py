"""CreatePlayerReport: one PDF per player — cover, statistics, three possession pitch views."""
import json
import math
import os
import re
import time

from matplotlib.backends.backend_pdf import PdfPages

from reports.match_report import style
from reports.match_report.builder import _team_rgb
from utils import GameFiles

from . import metrics, pages, team_overview


def _safe_name(label):
    s = re.sub(r"[^\w\-]+", "-", str(label).strip(), flags=re.UNICODE)
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    return (s or "player")[:48]


class CreatePlayerReport:
    def __init__(self, db_path, OUTPUT_PATH, TEAMS, GAME_RECORD, DETECTION_FPS,
                 target="named", report_file=None):
        self.db_path = db_path
        self.out = OUTPUT_PATH
        self.rec = GAME_RECORD
        self.fps = DETECTION_FPS
        self.teams = [GAME_RECORD.get("team_a"), GAME_RECORD.get("team_b")]
        self.TEAMS = TEAMS
        self.target = target or "named"
        self.report_file = report_file  # only used when exactly one player is produced

    def run(self):
        print("-" * 98)
        print(f"PLAYER REPORT: distance, speed and possession-coloured positions (target={self.target})")
        print("-" * 98)
        t0 = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rec = self.rec
        M = metrics.compute_player_reports(self.db_path, rec, self.teams, self.target)
        if not M["reports"]:
            raise SystemExit("STOPPED: player report found no subjects to write.")
        labels = M["team_labels"]
        colours = style.team_colours(
            _team_rgb(self.db_path, self.TEAMS, self.teams[0], labels.get(self.teams[0], self.teams[0])),
            _team_rgb(self.db_path, self.TEAMS, self.teams[1], labels.get(self.teams[1], self.teams[1])),
        )
        team_colour = {
            self.teams[0]: colours[0],
            self.teams[1]: colours[1],
        }
        game_title = str(rec.get("title") or " vs ".join(self.teams))
        created = time.strftime("%d %b %Y %H:%M")
        written = []
        n_pages = len(pages.PAGES_PER_PLAYER)

        for i, report in enumerate(M["reports"]):
            name = report["subject"]["label"]
            ctx = pages.Ctx(M, rec, colours, report, created=created)
            frame = style.Frame(
                f"{name} · Player report",
                f"{game_title} · {rec.get('date', '')} · {name}",
                n_pages,
            )
            if self.report_file and len(M["reports"]) == 1:
                pdf_path = self.report_file
            else:
                pdf_path = GameFiles.report_file(
                    self.out, rec, f"player-report-{_safe_name(name)}", stamp, "pdf",
                    subfolder="Players")
            pdf_title = f"Player report: {name} · {game_title}"
            with PdfPages(pdf_path, metadata={"Title": pdf_title, "Author": "Polis Walking Football"}) as pdf:
                for fn in pages.PAGES_PER_PLAYER:
                    try:
                        fig = fn(frame, ctx)
                    except Exception as e:
                        print(f"⚠️ Player report page '{fn.__name__}' for {name} failed: {e}")
                        fig = frame.new(name, "This page could not be built for this player.")
                        fig.text(0.05, 0.75, f"{type(e).__name__}: {e}", fontsize=10, color=style.MUTED)
                    pdf.savefig(fig, facecolor=fig.get_facecolor())
                    style.plt.close(fig)
            json_path = os.path.splitext(pdf_path)[0] + ".json"
            slim = {
                "target": M["target"],
                "player": name,
                "key": report["subject"]["key"],
                "team": report["team"],
                "teams": M["teams"],
                "start": M["start"],
                "end": M["end"],
                "distance_m": report["distance_m"],
                "speed_avg": report["speed_avg"],
                "speed_max": report["speed_max"],
                "speed_p95": report["speed_p95"],
                "tracked_s": report["tracked_s"],
                "n_frames": report.get("n_frames"),
                "expected_frames": report.get("expected_frames"),
                "possession_in_pct": report["possession_in_pct"],
                "possession_out_pct": report["possession_out_pct"],
                "possession_known_pct": report["possession_known_pct"],
                "accuracy": report.get("accuracy"),
                "tracks": report["tracks"],
                "n_samples": report["n_samples"],
            }
            with open(json_path, "w") as f:
                json.dump(slim, f, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else o, indent=1)
            written.append(pdf_path)
            print(f"  · {name}: {pdf_path}")

        # Squad overview when two or more players on the same team were included
        if len(M["reports"]) >= 2:
            print("TEAM OVERVIEW: squad sheet from the player reports above")
            team_pdfs = team_overview.write_overviews_for_run(
                M, rec, colours, team_colour, self.out, stamp, created=created)
            written.extend(team_pdfs)

        print(f"Player report generated: {len(written)} PDF(s)")
        print(f"✅ Processing time: {int(time.time() - t0)} sec")
        return written[0] if len(written) == 1 else written
