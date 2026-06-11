from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .board import GomokuBoard, Move
from .features import candidate_feature_rows


@dataclass
class TrainingExample:
    candidates: list[tuple[int, int, list[float]]]
    target: tuple[int, int]


def load_examples(path: str | Path, max_examples: int | None = None) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            board = GomokuBoard(size=int(record.get("size", 15)))

            for move in record["moves"]:
                row = int(move["row"])
                col = int(move["col"])
                player = int(move["player"])

                board.current_player = player
                candidates = candidate_feature_rows(board, player)
                if any(candidate_row == row and candidate_col == col for candidate_row, candidate_col, _ in candidates):
                    examples.append(TrainingExample(candidates=candidates, target=(row, col)))

                board.grid[row][col] = player
                board.moves.append(Move(row=row, col=col, player=player))

                if max_examples is not None and len(examples) >= max_examples:
                    return examples

    return examples
