from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.agents import create_agent
from gomoku_ai.constants import BLACK, PLAYER_NAMES, WHITE
from gomoku_ai.self_play import play_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gomoku AI self-play games.")
    parser.add_argument("--games", type=int, default=20, help="Number of games to run.")
    parser.add_argument("--black", default="heuristic", help="Agent: random, heuristic, heuristic-temp:x, linear:path, cnn:path, or search:depth:width.")
    parser.add_argument("--white", default="heuristic", help="Agent: random, heuristic, heuristic-temp:x, linear:path, cnn:path, or search:depth:width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "self_play_games.jsonl"),
        help="Path to JSONL output file.",
    )
    parser.add_argument("--max-moves", type=int, default=225, help="Move limit per game.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be greater than 0")

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    black_agent = create_agent(args.black)
    white_agent = create_agent(args.white)
    stats = {
        BLACK: 0,
        WHITE: 0,
        None: 0,
        "moves": 0,
    }

    with output_path.open("w", encoding="utf-8") as file:
        for index in range(1, args.games + 1):
            result = play_game(black_agent, white_agent, max_moves=args.max_moves)
            stats[result.winner] += 1
            stats["moves"] += result.moves

            record = {
                "game_index": index,
                **result.record,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index == 1 or index == args.games or index % max(1, args.games // 10) == 0:
                winner_text = PLAYER_NAMES.get(result.winner, "draw")
                print(f"game {index:>4}/{args.games}: winner={winner_text}, moves={result.moves}")

    average_moves = stats["moves"] / args.games
    print()
    print("summary")
    print(f"  black wins: {stats[BLACK]}")
    print(f"  white wins: {stats[WHITE]}")
    print(f"  draws:      {stats[None]}")
    print(f"  avg moves:  {average_moves:.1f}")
    print(f"  output:     {output_path}")


if __name__ == "__main__":
    main()
