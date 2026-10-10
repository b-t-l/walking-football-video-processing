# Game logger

A small local web app that replaces `games-logger.xlsx`. It keeps your games, teams, kit colours, timings, exclusions and
pitch calibrations in one database file and hands the pipeline exactly the same record the spreadsheet used to.

## Install (once)

In the `walking-football-2026` environment:

    pip install fastapi uvicorn

(`pandas` and `openpyxl`, which you already have, are used for the spreadsheet import and export.)

## First run

From the application folder (the one that contains `main.py`):

    python -m game_logger.importer          # copies the games, calibrations and kit colours from games-logger.xlsx
    python -m game_logger.check_roundtrip   # proves the database gives the pipeline identical data (expect: all identical)
    python -m game_logger                   # starts the app and opens http://127.0.0.1:8765

The spreadsheet is only read, never changed. Re-running the importer skips games already in the database;
`--overwrite` replaces them.

## Using it with the pipeline

In `main.py` set `GAME_SOURCE = 'database'` (the default is `'excel'`). Nothing else changes: `GetGameData` and `Teams`
return the same things they did before. Switch back to `'excel'` at any time.

## Where things live

* Database: `video-processing/game-logger.db` (next to the `output` folder; override with the `GAME_LOGGER_DB` environment variable).
  Keep it out of OneDrive/iCloud-synced folders: sync can corrupt a SQLite database while it is open.
* Backups: `video-processing/game-logger-backups/` - one per day on first start, plus one before every import or game delete.
* The app only listens on this computer (127.0.0.1); nothing on your network can reach it.

## Notes

* Times are stored as whole seconds and shown as MM:SS. Scores are numbers (the pipeline still receives "0/2").
* Each game keeps every calibration you add, with exactly one marked "in use"; the pipeline uses that one.
  Imported games keep their old spreadsheet vertex columns exactly (they can differ from the calibrator JSON by a pixel).
* Teams without kit colours are left out of the team list the classifier sees; add colours on the Teams page.
* Settings -> "Download everything as a spreadsheet" gives a read-only copy in the old layout.
