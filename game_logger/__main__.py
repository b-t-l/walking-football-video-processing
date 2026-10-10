"""python -m game_logger  - starts the game logger and opens it in your browser."""
import argparse
import threading
import webbrowser

from . import config, db


def main():
    ap = argparse.ArgumentParser(description="Game logger")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    import uvicorn
    conn = db.connect()                       # creates the database on first run
    conn.close()
    made = db.backup("daily")
    url = f"http://{config.HOST}:{args.port}"
    print(f"Game logger {url}")
    print(f"Database: {config.DB_PATH}" + (f"   (today's backup made: {made})" if made else ""))
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("game_logger.app:app", host=config.HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
