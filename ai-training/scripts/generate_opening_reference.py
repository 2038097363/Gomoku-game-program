from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.board import GomokuBoard, Move
from gomoku_ai.openings import formula_openings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight policy labels from formula opening references.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "formula_opening_positions.jsonl"))
    parser.add_argument("--repeat", type=int, default=4, help="Repeat examples to give the tiny reference set some weight.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as sink:
        for _ in range(args.repeat):
            for opening in formula_openings():
                board = GomokuBoard()
                for row, col, player in opening.moves:
                    board.current_player = player
                    sink.write(
                        json.dumps(
                            {
                                "size": board.size,
                                "player": player,
                                "board": board.grid,
                                "target": row * board.size + col,
                                "source": "formula-opening-reference",
                                "opening": opening.code,
                                "opening_name": opening.name,
                                "opening_family": opening.family,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                    board.grid[row][col] = player
                    board.moves.append(Move(row=row, col=col, player=player))

    print(f"opening examples: {written}")
    print(f"openings: {len(formula_openings())}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
