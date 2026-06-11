from __future__ import annotations

from dataclasses import dataclass, field

from .constants import BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, WHITE


@dataclass
class Move:
    row: int
    col: int
    player: int


@dataclass
class GomokuBoard:
    size: int = BOARD_SIZE
    grid: list[list[int]] = field(default_factory=list)
    current_player: int = BLACK
    moves: list[Move] = field(default_factory=list)
    winner: int | None = None
    winning_line: list[tuple[int, int]] = field(default_factory=list)
    game_over: bool = False

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]

    def clone(self) -> "GomokuBoard":
        copied = GomokuBoard(size=self.size)
        copied.grid = [row[:] for row in self.grid]
        copied.current_player = self.current_player
        copied.moves = [Move(move.row, move.col, move.player) for move in self.moves]
        copied.winner = self.winner
        copied.winning_line = self.winning_line[:]
        copied.game_over = self.game_over
        return copied

    def legal_moves(self) -> list[tuple[int, int]]:
        if self.game_over:
            return []

        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if self.grid[row][col] == EMPTY
        ]

    def place(self, row: int, col: int) -> bool:
        if not self.can_place(row, col):
            return False

        player = self.current_player
        self.grid[row][col] = player
        self.moves.append(Move(row, col, player))

        line = self.find_winning_line(row, col, player)
        if line:
            self.game_over = True
            self.winner = player
            self.winning_line = line
        elif len(self.moves) == self.size * self.size:
            self.game_over = True
            self.winner = None
        else:
            self.current_player = opponent(player)

        return True

    def can_place(self, row: int, col: int) -> bool:
        return (
            not self.game_over
            and 0 <= row < self.size
            and 0 <= col < self.size
            and self.grid[row][col] == EMPTY
        )

    def find_winning_line(self, row: int, col: int, player: int) -> list[tuple[int, int]]:
        for dr, dc in DIRECTIONS:
            line = [(row, col)]
            line.extend(self._collect(row, col, dr, dc, player))
            line[:0] = reversed(self._collect(row, col, -dr, -dc, player))

            if len(line) >= 5:
                return line

        return []

    def _collect(self, row: int, col: int, dr: int, dc: int, player: int) -> list[tuple[int, int]]:
        line = []
        next_row = row + dr
        next_col = col + dc

        while (
            0 <= next_row < self.size
            and 0 <= next_col < self.size
            and self.grid[next_row][next_col] == player
        ):
            line.append((next_row, next_col))
            next_row += dr
            next_col += dc

        return line

    def to_training_record(self) -> dict:
        return {
            "size": self.size,
            "winner": self.winner,
            "moves": [
                {"row": move.row, "col": move.col, "player": move.player}
                for move in self.moves
            ],
        }


def opponent(player: int) -> int:
    return WHITE if player == BLACK else BLACK
