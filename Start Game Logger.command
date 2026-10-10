#!/bin/bash
# Double-click to start the game logger. Close this Terminal window (or press Ctrl+C) to stop it.
APP_DIR="$(cd "$(dirname "$(readlink "$0" || echo "$0")")" && pwd)"
PORT=8765
if lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Game logger is already running - opening it in your browser."
  open "http://127.0.0.1:$PORT"
  sleep 2
  exit 0
fi
cd "$APP_DIR" || exit 1
# run in the same kind of shell you normally use, so python / conda / your packages are found
exec "${SHELL:-/bin/zsh}" -l -i -c 'python -m game_logger || python3 -m game_logger; echo; echo "Game logger stopped. Press Return to close."; read'
