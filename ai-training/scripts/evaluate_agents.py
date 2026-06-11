from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate trained Gomoku agents.")
    parser.add_argument("--games", type=int, default=20, help="Games per color pairing.")
    parser.add_argument("--agent", default=f"linear:{PROJECT_ROOT / 'models' / 'linear_policy.json'}", help="Agent to evaluate.")
    parser.add_argument("--opponent", default="heuristic", help="Opponent agent.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    model_agent = create_agent(args.agent)
    opponent_agent = create_agent(args.opponent)

    stats = {
        "agent_as_black": {BLACK: 0, WHITE: 0, None: 0, "moves": 0},
        "agent_as_white": {BLACK: 0, WHITE: 0, None: 0, "moves": 0},
    }

    for _ in range(args.games):
        result = play_game(model_agent, opponent_agent)
        stats["agent_as_black"][result.winner] += 1
        stats["agent_as_black"]["moves"] += result.moves

    for _ in range(args.games):
        result = play_game(opponent_agent, model_agent)
        stats["agent_as_white"][result.winner] += 1
        stats["agent_as_white"]["moves"] += result.moves

    print("evaluation complete")
    print_block("agent as black", stats["agent_as_black"], args.games)
    print_block("agent as white", stats["agent_as_white"], args.games)


def print_block(title: str, stats: dict, games: int) -> None:
    print(title)
    print(f"  black wins: {stats[BLACK]}")
    print(f"  white wins: {stats[WHITE]}")
    print(f"  draws:      {stats[None]}")
    print(f"  avg moves:  {stats['moves'] / games:.1f}")


if __name__ == "__main__":
    main()
