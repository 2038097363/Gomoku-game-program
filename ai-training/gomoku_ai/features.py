from __future__ import annotations

from .agents import candidate_moves, count_line
from .board import GomokuBoard, opponent
from .constants import DIRECTIONS

FEATURE_NAMES = (
    "bias",
    "center",
    "own_five",
    "own_open_four",
    "own_closed_four",
    "own_open_three",
    "own_closed_three",
    "own_open_two",
    "block_five",
    "block_open_four",
    "block_closed_four",
    "block_open_three",
    "block_closed_three",
    "block_open_two",
)


def extract_features(board: GomokuBoard, row: int, col: int, player: int) -> list[float]:
    rival = opponent(player)
    own = shape_counts(board, row, col, player)
    block = shape_counts(board, row, col, rival)

    return [
        1.0,
        center_feature(board, row, col),
        float(own["five"]),
        float(own["open_four"]),
        float(own["closed_four"]),
        float(own["open_three"]),
        float(own["closed_three"]),
        float(own["open_two"]),
        float(block["five"]),
        float(block["open_four"]),
        float(block["closed_four"]),
        float(block["open_three"]),
        float(block["closed_three"]),
        float(block["open_two"]),
    ]


def shape_counts(board: GomokuBoard, row: int, col: int, player: int) -> dict[str, int]:
    counts = {
        "five": 0,
        "open_four": 0,
        "closed_four": 0,
        "open_three": 0,
        "closed_three": 0,
        "open_two": 0,
    }

    for dr, dc in DIRECTIONS:
        left_count, left_open = count_line(board, row, col, -dr, -dc, player)
        right_count, right_open = count_line(board, row, col, dr, dc, player)
        total = left_count + 1 + right_count
        open_ends = int(left_open) + int(right_open)

        if total >= 5:
            counts["five"] += 1
        elif total == 4 and open_ends == 2:
            counts["open_four"] += 1
        elif total == 4 and open_ends == 1:
            counts["closed_four"] += 1
        elif total == 3 and open_ends == 2:
            counts["open_three"] += 1
        elif total == 3 and open_ends == 1:
            counts["closed_three"] += 1
        elif total == 2 and open_ends == 2:
            counts["open_two"] += 1

    return counts


def center_feature(board: GomokuBoard, row: int, col: int) -> float:
    center = (board.size - 1) / 2
    distance = abs(row - center) + abs(col - center)
    return 1.0 - distance / (board.size - 1)


def candidate_feature_rows(board: GomokuBoard, player: int, radius: int = 2) -> list[tuple[int, int, list[float]]]:
    rows = []
    for row, col in candidate_moves(board, radius):
        rows.append((row, col, extract_features(board, row, col, player)))
    return rows
