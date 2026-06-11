from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.board import GomokuBoard, Move


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract winner-side policy examples from game records.")
    parser.add_argument("--input", required=True, help="Source game JSONL.")
    parser.add_argument("--output", required=True, help="Output labeled position JSONL.")
    parser.add_argument("--include-draws", action="store_true", help="Keep all moves from drawn games.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with input_path.open("r", encoding="utf-8-sig") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)
            winner = record.get("winner")
            board = GomokuBoard(size=int(record.get("size", 15)))

            for move in record["moves"]:
                row = int(move["row"])
                col = int(move["col"])
                player = int(move["player"])

                if player == winner or (winner is None and args.include_draws):
                    board.current_player = player
                    sink.write(
                        json.dumps(
                            {
                                "size": board.size,
                                "player": player,
                                "board": board.grid,
                                "target": row * board.size + col,
                                "source": "winner-extract",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

                board.grid[row][col] = player
                board.moves.append(Move(row=row, col=col, player=player))

    print(f"winner examples: {written}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
