from __future__ import annotations

from dataclasses import dataclass

from .agents import Agent
from .board import GomokuBoard
from .constants import BLACK, PLAYER_NAMES, WHITE


@dataclass
class GameResult:
    winner: int | None
    moves: int
    record: dict


def play_game(black_agent: Agent, white_agent: Agent, max_moves: int | None = None) -> GameResult:
    board = GomokuBoard()
    agents = {
        BLACK: black_agent,
        WHITE: white_agent,
    }
    move_limit = max_moves or board.size * board.size

    while not board.game_over and len(board.moves) < move_limit:
        agent = agents[board.current_player]
        row, col = agent.choose_move(board.clone())

        if not board.place(row, col):
            raise RuntimeError(
                f"{agent.name} produced illegal move: row={row}, col={col}, player={board.current_player}"
            )

    if not board.game_over:
        board.game_over = True
        board.winner = None

    record = board.to_training_record()
    record["agents"] = {
        PLAYER_NAMES[BLACK]: black_agent.name,
        PLAYER_NAMES[WHITE]: white_agent.name,
    }

    return GameResult(winner=board.winner, moves=len(board.moves), record=record)
