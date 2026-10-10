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
    team-images/  <timestamp>/<team>/  pictures of players to correct and retrain the team model (made on demand)
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
   - **Detection report** – scores every detection stage (players, ball, goalkeepers, tracking, teams, positions) out of 100, ranks what to fix first, and lists what to try. It also saves a review sheet (about 24 frames with the detections drawn on) and a CSV: fill the CSV in, run the report again, and it shows measured accuracy.
   - **Match report** – 12-page PDF for the coach and team: possession, territory, team shape, distance, speed, running and key moments (neutral, both teams).
   - **Opposition reports** – two scouting PDFs, one per team, written for the other team's coach: how the team plays, strong and weak points (measured against the other team in the game) and a game plan.
   - **Annotation** – creates the annotated video.
   - **Team training images** – picks 10 random frames and saves every detected player as a picture in the game's `team-images/<timestamp>/` folder, in sub-folders named after what the current team model says (Polis, Aphrodite Wanderers, Goalkeeper, Referee…). Drag the wrong ones into the right folder (make a new folder for a team the model does not know), then add them to the training library and retrain (section 7). Run it again for ten different frames.
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

## 7. Improving the team classifier (training your own team model)

The team model (`models/teams/`, a small YOLO classification model working on 192x192 pictures) decides which team every player is in. It only knows the teams it was trained on. To add a team or fix mistakes you give it more pictures and retrain it **on this Mac – Roboflow is not needed**.

**The picture library** lives next to the application folder: `video-processing/_team-model-training/team-pictures/<team>/`. There is one folder per team, named exactly as the team is named in the Game Logger, plus `Goalkeeper` and `Referee`. Every picture you ever sort goes in and stays there; over time you just add more. A team needs at least 20 pictures before it is included in training (below that it is skipped and the counts say so).

All the commands below are run in Terminal from the application folder, in the same Python environment you use for `main.py`:

```
cd /Users/unilever/Desktop/Active-Projects/ai-agents/walking-football/video-processing/walking-football-video-processing-application
conda activate walking-football-2026
```

1. **Make pictures.** In the Run panel tick **Team training images** for a game (run it on several games for variety). The pictures appear in that game's `team-images/<timestamp>/` folder, sorted into folders by what the current model thinks.
2. **Sort them.** Drag every wrong picture into the right team's folder. For a team the model does not know, make a folder named after the team. Delete pictures that are no use (two players in one box, a person off the pitch).
3. **Add them to the library:**
   ```
   python train_team_model.py --add "/Volumes/LaCie/walking-football/games/0025_<...>/team-images/<timestamp>"
   ```
   Pictures that are already in the library are skipped (if you changed a picture's team, the library's version is kept). You can also simply drag pictures into the team folders in Finder.
4. **New team in the Game Logger?** `python train_team_model.py --sync-teams` makes a library folder for every team in the Game Logger (plus Goalkeeper and Referee).
5. **Check the counts** (nothing is trained): `python train_team_model.py --dry-run`
6. **Train:**
   ```
   python train_team_model.py
   python train_team_model.py --model yolo26n-cls.pt      # the smaller, faster model (default is yolo26s-cls.pt)
   ```
   It splits the library into train / validation / test (by game and frame, so near-identical pictures never sit on both sides), trains on the Mac's GPU with colour-safe augmentation (a few minutes), scores the new model and your current one on the same pictures, and saves the new weights as `models/teams/team-<date>-<model>.pt`. It never overwrites the current model. The report is also in `train_team_model_results.txt`. The first run downloads the starting weights, so it needs internet.
7. **Test it on a real game** (the validation score alone is flattering):
   ```
   python compare_team_models.py --game 22
   python compare_team_models.py --game 24 --teams "Polis Lions,Aphrodite Wanderers"
   ```
   It asks both models about players in random frames of the game and counts *impossible* answers (a third team's name in a two-team game), checks its pictures against what the pipeline stored, and saves a sheet of the pictures where the models disagree (`compare_team_models_out/`) so you can see who is right.
8. **Switch to the new model** when you are happy: in `data/create_database.py` change the `TEAM_MODEL_PATH` line to the new file name, Keep the old weights file so you can switch back. The pipeline currently uses `team-20261010-yolo26s-cls.pt` (the old model was `best-v7-192x192.pt`). Team colours (annotated video and reports) come from the **first kit colour on the Game Logger's Teams page**, so a new team just needs a kit colour there; the team's name in the library folder must match its Game Logger name. Goalkeeper and Referee have fixed colours, and any other name shows white. Colours are stored when detections run, so games processed earlier keep their old colours until you run Detections again.

Other options: `--min-pictures 30` (how many pictures a team needs), `--import-export <folder>` (one-off: copy a Roboflow "Folder Structure" export into the library).

## 8. Safety net

- The database is backed up daily, before every upgrade and before deleting a game or team, to the Mac and the data drive.
- Games and teams are archived/inactive rather than lost; deletes always ask first.
- Pipeline files that were changed for the Game Logger have `*.bak` copies next to them.

## Version history

**Version 4**
- Time exclusions no longer cut the video: the whole video is processed. Exclusions are only used to skip adding annotations for parts that were not analysed.
- Audio is now saved into the annotated video file.

**Game Logger 1.0**
- Replaced `games-logger.xlsx` with a database and web app; three videos per game; per-game folders on the data drive with correct file naming; embedded pitch calibration; teams and kit colours; Run panel with live console; Run from `main.py --game/--steps/--duration`.
