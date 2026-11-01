"""python -m player_positions --game 25 [--check] [--out FILE]"""
import argparse
import json
import sys

from .build import build_player_positions, check_current


def main():
    ap = argparse.ArgumentParser(description="Save the named-player positions of a game (resolves the Identity tags).")
    ap.add_argument("--game", type=int, required=True, help="game id, e.g. 25")
    ap.add_argument("--check", action="store_true", help="only say whether the saved file is still current (no rebuild)")
    ap.add_argument("--out", help="write the database here instead of <game folder>/data/NNNN_players.db")
    a = ap.parse_args()
    if a.check:
        r = check_current(a.game, a.out)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["current"] else 1)
    build_player_positions(a.game, a.out)


main()
