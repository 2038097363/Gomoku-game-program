from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.agents import HeuristicAgent
from gomoku_ai.board import GomokuBoard, Move


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label board positions with the deterministic heuristic teacher.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "outputs" / "teacher_games_mix_400.jsonl"), help="Source game records.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "distilled_positions.jsonl"), help="Output labeled position records.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Probability of keeping each position.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    teacher = HeuristicAgent()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_path.open("r", encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)
            board = GomokuBoard(size=int(record.get("size", 15)))

            for move in record["moves"]:
                row = int(move["row"])
                col = int(move["col"])
                player = int(move["player"])
                board.current_player = player

                if random.random() <= args.sample_rate:
                    target_row, target_col = teacher.choose_move(board.clone())
                    sink.write(
                        json.dumps(
                            {
                                "size": board.size,
                                "player": player,
                                "board": board.grid,
                                "target": target_row * board.size + target_col,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

                    if args.max_examples is not None and written >= args.max_examples:
                        print(f"labeled examples: {written}")
                        print(f"output: {output_path}")
                        return

                board.grid[row][col] = player
                board.moves.append(Move(row=row, col=col, player=player))

    print(f"labeled examples: {written}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
