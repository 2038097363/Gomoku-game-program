from __future__ import annotations

from dataclasses import dataclass

from .board import GomokuBoard
from .constants import BLACK, WHITE


@dataclass(frozen=True)
class OpeningPattern:
    code: str
    name: str
    family: str
    moves: tuple[tuple[int, int, int], ...]


CENTER = 7

# Reference coordinates are normalized on a 15x15 board with the first stone at H8
# in common human notation, represented internally as (7, 7). They are used as
# lightweight exposure to formula openings, not as a mandatory move book.
INDIRECT_THIRD_OFFSETS = (
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, 0),
    (-1, 2),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (1, -2),
)

DIRECT_THIRD_OFFSETS = (
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, -1),
    (-1, 1),
    (-1, 2),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
)

INDIRECT_NAMES = (
    "Chosei",
    "Kyogetsu",
    "Kosei",
    "Suigetsu",
    "Ryusei",
    "Ungetsu",
    "Hogetsu",
    "Rangetsu",
    "Gingetsu",
    "Myojo",
    "Shagetsu",
    "Meigetsu",
    "Suisei",
)

DIRECT_NAMES = (
    "Kansei",
    "Keigetsu",
    "Sosei",
    "Kagetsu",
    "Zangetsu",
    "Ugetsu",
    "Kinsei",
    "Shogetsu",
    "Kyugetsu",
    "Shingetsu",
    "Zuigetsu",
    "Sangetsu",
    "Yusei",
)


def formula_openings() -> list[OpeningPattern]:
    patterns = []
    patterns.extend(
        make_patterns(
            family="indirect",
            prefix="I",
            names=INDIRECT_NAMES,
            second=(CENTER - 1, CENTER + 1, WHITE),
            third_offsets=INDIRECT_THIRD_OFFSETS,
        )
    )
    patterns.extend(
        make_patterns(
            family="direct",
            prefix="D",
            names=DIRECT_NAMES,
            second=(CENTER - 1, CENTER, WHITE),
            third_offsets=DIRECT_THIRD_OFFSETS,
        )
    )
    return patterns


def make_patterns(
    family: str,
    prefix: str,
    names: tuple[str, ...],
    second: tuple[int, int, int],
    third_offsets: tuple[tuple[int, int], ...],
) -> list[OpeningPattern]:
    patterns = []
    for index, (name, offset) in enumerate(zip(names, third_offsets), start=1):
        third = (CENTER + offset[0], CENTER + offset[1], BLACK)
        patterns.append(
            OpeningPattern(
                code=f"{prefix}{index}",
                name=name,
                family=family,
                moves=((CENTER, CENTER, BLACK), second, third),
            )
        )
    return patterns


def is_formula_like_opening(board: GomokuBoard) -> bool:
    if len(board.moves) < 2 or len(board.moves) > 8:
        return False

    first = board.moves[0]
    second = board.moves[1]
    if first.player != BLACK or first.row != CENTER or first.col != CENTER:
        return False
    if second.player != WHITE:
        return False

    row_distance = abs(second.row - CENTER)
    col_distance = abs(second.col - CENTER)
    if max(row_distance, col_distance) != 1:
        return False

    if len(board.moves) >= 3:
        third = board.moves[2]
        if third.player != BLACK:
            return False
        if abs(third.row - CENTER) > 2 or abs(third.col - CENTER) > 2:
            return False

    return True
