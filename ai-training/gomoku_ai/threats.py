from __future__ import annotations

from dataclasses import dataclass

from .board import GomokuBoard, opponent
from .constants import DIRECTIONS, EMPTY


@dataclass(frozen=True)
class ThreatProfile:
    five: int = 0
    open_four: int = 0
    closed_four: int = 0
    open_three: int = 0
    closed_three: int = 0
    open_two: int = 0

    @property
    def double_four(self) -> bool:
        return self.open_four + self.closed_four >= 2

    @property
    def four_three(self) -> bool:
        return self.open_four + self.closed_four >= 1 and self.open_three >= 1

    @property
    def double_open_three(self) -> bool:
        return self.open_three >= 2

    @property
    def forcing(self) -> bool:
        return self.five > 0 or self.open_four > 0 or self.closed_four > 0


@dataclass(frozen=True)
class MoveThreat:
    row: int
    col: int
    own: ThreatProfile
    opponent: ThreatProfile

    @property
    def priority(self) -> int:
        if self.own.five:
            return 1_000_000_000
        if self.opponent.five:
            return 900_000_000
        if self.own.double_four:
            return 800_000_000
        if self.opponent.double_four:
            return 760_000_000
        if self.own.four_three:
            return 720_000_000
        if self.opponent.four_three:
            return 690_000_000
        if self.own.open_four:
            return 650_000_000
        if self.opponent.open_four:
            return 630_000_000
        if self.own.closed_four:
            return 520_000_000
        if self.opponent.closed_four:
            return 500_000_000
        if self.own.double_open_three:
            return 420_000_000
        if self.opponent.double_open_three:
            return 400_000_000
        if self.own.open_three:
            return 220_000_000 + self.own.open_three * 10_000
        if self.opponent.open_three:
            return 205_000_000 + self.opponent.open_three * 10_000
        return 0

    @property
    def tactical_score(self) -> int:
        return self.priority + profile_score(self.own) - int(profile_score(self.opponent) * 1.08)


def analyze_move(board: GomokuBoard, row: int, col: int, player: int) -> MoveThreat:
    rival = opponent(player)
    return MoveThreat(
        row=row,
        col=col,
        own=profile_for(board, row, col, player),
        opponent=profile_for(board, row, col, rival),
    )


def profile_for(board: GomokuBoard, row: int, col: int, player: int) -> ThreatProfile:
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

    return ThreatProfile(**counts)


def profile_score(profile: ThreatProfile) -> int:
    return (
        profile.five * 1_000_000
        + profile.open_four * 140_000
        + profile.closed_four * 35_000
        + profile.open_three * 8_000
        + profile.closed_three * 1_500
        + profile.open_two * 650
        + (280_000 if profile.double_four else 0)
        + (180_000 if profile.four_three else 0)
        + (95_000 if profile.double_open_three else 0)
    )


def count_line(
    board: GomokuBoard,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> tuple[int, bool]:
    count = 0
    next_row = row + dr
    next_col = col + dc

    while (
        0 <= next_row < board.size
        and 0 <= next_col < board.size
        and board.grid[next_row][next_col] == player
    ):
        count += 1
        next_row += dr
        next_col += dc

    open_end = (
        0 <= next_row < board.size
        and 0 <= next_col < board.size
        and board.grid[next_row][next_col] == EMPTY
    )
    return count, open_end
