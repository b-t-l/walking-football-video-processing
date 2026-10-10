"""
game_logger - a small local web app + database that replaces games-logger.xlsx.

    python -m game_logger                  # start the web app at http://127.0.0.1:8765
    python -m game_logger.importer         # one-off import of games-logger.xlsx into the database
    python -m game_logger.check_roundtrip  # check the database gives the pipeline the same game records as the Excel file

The pipeline reads games through game_logger.record.build_game_record(), which returns the same dictionary the Excel
file used to (see data/game_data_import.py, GAME_SOURCE in main.py).
"""
__version__ = "1.0.0"
