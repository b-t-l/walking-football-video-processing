# Walking Football Video Processing

Video analysis for Polis Cyprus Men's Walking Football. A game video goes in; player/ball detections, statistics, PDF reports and an annotated video come out.

There are two parts:

1. **The Game Logger** – a small web app on your own Mac where you keep a record of every game (teams, scores, videos, pitch calibration, timings) and where you can start the processing and watch its progress.
2. **The processing pipeline** – `main.py` and its helper folders (`data/`, `reports/`, `video_annotator/`, `team_assigner/`, ...). It does the detections, statistics, reports and annotated video. The Game Logger runs it for you, or you can still run `python main.py` yourself.

---

## 1. Starting the Game Logger

Double-click **`Start Game Logger.command`** (make an alias on the Desktop with Cmd+Option-drag). A Terminal window opens and the app appears in your browser at <http://127.0.0.1:8765>. Close the Terminal window (or press Ctrl+C) to stop it.

Or from Terminal, in this folder: `python -m game_logger`

The app only listens on your own computer; nothing outside it can reach it. If you update the app's files, restart it and hard-refresh the browser (Cmd+Shift+R).

Needs `pip install fastapi uvicorn` in addition to the pipeline packages in `requirements.txt`.

## 2. Where everything is kept

| What | Where |
|---|---|
| Game database (all games, teams, calibrations, settings) | on the Mac: `video-processing/game-logger.db` |
| Daily backups of that database | `video-processing/game-logger-backups/` **and** a copy on the data folder (`_app-data/game-logger-backups`) |
| Videos, detections, reports, logs | the **Data folder** (the LaCie drive: `/Volumes/LaCie/walking-football`) |

Set the Data folder once in **Settings → Data folder** (or with the environment variable `WALKING_FOOTBALL_ROOT`). To move to a new/faster drive later, copy the folder across and change this one setting.

Each game gets its own folder, found by its 4-digit game number:

```
<Data folder>/games/0022_2026-10-02_mens-league_polis-v-aphrodite-wanderers/
    videos/       0022_analysis.mp4   0022_edited.mp4
        annotated/  0022_annotated_<timestamp>.mp4
    data/         0022_detections.db   object-images/
    reports/      0022_match-report_<ts>.pdf (+ .json)  0022_opposition-report_<team>_<ts>.pdf  0022_detection-report_<ts>.pdf
    _work/        temporary files while processing (tracker state, video chunks)
    logs/         one log per run from the Run panel
    notes.txt
<Data folder>/_app-data/   backups
<Data folder>/_library/    <Data folder>/_inbox/
```

The folder is **created automatically** when you save a new game. Copy the game's videos into its `videos/` folder with the names above and the app finds them by itself (or type/browse to a different file on the game page).

## 3. Using the Game Logger

**Games page** – the list of all games with search and filters (type, format, status, archived). Click a game to open it. Status chips show whether the game is ready to process.

**New game / game page** has these sections:

- **Match** – date, game type and format, teams A and B, scores, title (blank = "Team A vs Team B"; it also names the output files), description, notes, running speed.
- **Video files** – three videos per game, each also available in the video player's droplist:
  1. **Edited** – the panned, TV-style video for watching.
  2. **Analysis** – the video used for the detections.
  3. **Annotated** – the output with the detections drawn on (found automatically in the game folder).
  A separate field holds the **YouTube link** of the uploaded annotated video (used for the chapter links in the stats report) and the 360 video link. The built-in player lets you scrub the videos and set timings from the current frame.
- **Timings & exclusions** – game start/end and periods of the video to leave out of the annotation.
- **Pitch calibration** – the pitch corner/centre points that turn camera positions into metres. Click **Calibrate pitch** to open the Pitch Calibrator with this game's video, mark the points, and save them back to the game.
- **Run** – start the processing (see below).
- **What the pipeline is given** – the exact record `main.py` receives for this game, for checking.

Buttons on the game page: **Save**, **Duplicate** (same teams/type/format/calibration – handy for the next match filmed from the same spot), **Archive/Unarchive** (hide it from the list), **Delete** (backup taken first).

**Teams page** – add teams and their **kit colours** (colour + tolerance), which the team classifier uses to tell the teams apart. A team needs at least one colour before its games can be processed. *Edit → Delete team* removes a team; if games use it, it is taken off those games after a confirmation (database backed up first). Untick *Active* to hide a team from new games without deleting it.

**Settings** – default running speed, Pitch Calibrator link, Data folder, database/backup locations, and **Download everything as a spreadsheet** (a read-only copy in the old `games-logger.xlsx` layout).

## 4. Running the processing from the browser (the Run panel)

On a game page open **Run**:

1. Tick the stages you want:
   - **Detections** – detect and track players and the ball in the analysis video (the slow step). Saved to `data/0022_detections.db`.
   - **Transformations** – overhead pitch positions, teams, speed, distance, possession, statistics.
   - **Detection report** – PDF report on the detections.
   - **Match report** – 12-page PDF for the coach and team: possession, territory, team shape, distance, speed, running and key moments (neutral, both teams).
   - **Opposition reports** – two scouting PDFs, one per team, written for the other team's coach: how the team plays, strong and weak points (measured against the other team in the game) and a game plan.
   - **Annotation** – creates the annotated video.
2. Optionally tick **Test run** and give a number of minutes to process only the start of the game. Test results are kept apart (`-test` names, `reports/test`) so they never overwrite real ones.
3. Click **Run**. The console shows the live output with a progress bar; **Stop** ends it. Only one run at a time. If you restart the app mid-run it picks the run back up.
4. Finished runs are listed under **Earlier runs** (logs in the game's `logs/` folder).

Notes: stages after detections need the detections DB to exist (run detections first, or have run it before). Annotation-only reruns are quick – useful after changing the calibration or exclusions. The Mac is kept awake while a run is going (`caffeinate`).

## 5. Running the pipeline by hand

`main.py` still works from Terminal and reads the game from the Game Logger database (`GAME_SOURCE = 'database'` near the top). Edit the `GAME_ID` and the `RUN_...` switches, then `python main.py`. Or use the same options the Run panel uses:

```
python main.py --game 22 --steps detections,transformations,detection_report,match_report,annotation
python main.py --game 22 --steps annotation --duration 120      # 2-minute test run
```

Outputs go into the game's folder with the standard names. It stops with a clear message if the data drive is not connected or the analysis video is missing. With no data folder set (or `GAME_SOURCE = 'excel'`) it uses the old shared output folder and `games-logger.xlsx`.

## 6. Moving old games from the spreadsheet

`python -m game_logger.importer` imports games from `games-logger.xlsx` that are not yet in the database (`--overwrite` to re-import all; the spreadsheet is only read). `python -m game_logger.check_roundtrip` proves the database gives the pipeline exactly the same game record as the spreadsheet did. Old games are not moved into game folders automatically; their videos stay where they were until you choose to file them.

## 7. Safety net

- The database is backed up daily, before every upgrade and before deleting a game or team, to the Mac and the data drive.
- Games and teams are archived/inactive rather than lost; deletes always ask first.
- Pipeline files that were changed for the Game Logger have `*.bak` copies next to them.

## Version history

**Version 4**
- Time exclusions no longer cut the video: the whole video is processed. Exclusions are only used to skip adding annotations for parts that were not analysed.
- Audio is now saved into the annotated video file.

**Game Logger 1.0**
- Replaced `games-logger.xlsx` with a database and web app; three videos per game; per-game folders on the data drive with correct file naming; embedded pitch calibration; teams and kit colours; Run panel with live console; Run from `main.py --game/--steps/--duration`.
