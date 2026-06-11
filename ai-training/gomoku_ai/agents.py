from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .board import GomokuBoard, opponent
from .constants import DIRECTIONS, EMPTY


class Agent(Protocol):
    name: str

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        ...


@dataclass
class RandomAgent:
    name: str = "random"

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        return random.choice(board.legal_moves())


@dataclass
class HeuristicAgent:
    name: str = "heuristic"
    search_radius: int = 2
    temperature: float = 0.0

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        candidates = candidate_moves(board, self.search_radius)
        player = board.current_player
        rival = opponent(player)

        scored_moves: list[tuple[int, int, int]] = []

        for row, col in candidates:
            own_score = evaluate_move(board, row, col, player)
            block_score = evaluate_move(board, row, col, rival)
            center_score = center_bonus(board, row, col)
            score = own_score + int(block_score * 0.92) + center_score
            scored_moves.append((score, row, col))

        if self.temperature > 0:
            return sample_scored_move(scored_moves, self.temperature)

        best_score = max(score for score, _, _ in scored_moves)
        best_moves = [(row, col) for score, row, col in scored_moves if score == best_score]
        return random.choice(best_moves)


def create_agent(name: str) -> Agent:
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomAgent()
    if normalized == "heuristic":
        return HeuristicAgent()
    if normalized.startswith("heuristic-temp:"):
        return HeuristicAgent(
            name="heuristic-temp",
            temperature=float(normalized.split(":", 1)[1]),
        )
    if normalized.startswith("linear:"):
        from .linear_policy import LinearPolicy, LinearPolicyAgent

        return LinearPolicyAgent(model=LinearPolicy.load(normalized.split(":", 1)[1]))
    if normalized.startswith("cnn:"):
        from .torch_policy import TorchPolicyAgent

        return TorchPolicyAgent(model_path=normalized.split(":", 1)[1])
    if normalized.startswith("cnn-blend:"):
        from .torch_policy import TorchPolicyAgent

        _, model_path, blend = normalized.split(":", 2)
        return TorchPolicyAgent(model_path=model_path, name="cnn-blend", heuristic_blend=float(blend))
    if normalized.startswith("expert-cnn:"):
        from .search_agent import ExpertSearchAgent
        from .torch_policy import TorchPolicyAgent

        model_path = normalized.split(":", 1)[1]
        policy_agent = TorchPolicyAgent(model_path=model_path, name="cnn-prior", tactical_guard=False)
        return ExpertSearchAgent(name="expert-cnn", policy_prior=policy_agent.move_priors)
    if normalized == "search":
        from .search_agent import SearchAgent

        return SearchAgent(width=12)
    if normalized == "expert":
        from .search_agent import ExpertSearchAgent

        return ExpertSearchAgent()
    if normalized.startswith("search:"):
        from .search_agent import SearchAgent

        _, depth, width = normalized.split(":", 2)
        return SearchAgent(depth=int(depth), width=int(width), name=f"search-d{depth}-w{width}")
    if normalized.startswith("expert:"):
        from .search_agent import ExpertSearchAgent

        _, depth, width = normalized.split(":", 2)
        return ExpertSearchAgent(depth=int(depth), width=int(width), name=f"expert-d{depth}-w{width}")

    raise ValueError(f"Unknown agent: {name}")


def candidate_moves(board: GomokuBoard, radius: int) -> list[tuple[int, int]]:
    if not board.moves:
        center = board.size // 2
        return [(center, center)]

    candidates: set[tuple[int, int]] = set()
    for move in board.moves:
        for row in range(move.row - radius, move.row + radius + 1):
            for col in range(move.col - radius, move.col + radius + 1):
                if board.can_place(row, col):
                    candidates.add((row, col))

    return list(candidates) or board.legal_moves()


def evaluate_move(board: GomokuBoard, row: int, col: int, player: int) -> int:
    score = 0

    for dr, dc in DIRECTIONS:
        left_count, left_open = count_line(board, row, col, -dr, -dc, player)
        right_count, right_open = count_line(board, row, col, dr, dc, player)
        total = left_count + 1 + right_count
        open_ends = int(left_open) + int(right_open)

        score += shape_score(total, open_ends)

    return score


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


def shape_score(total: int, open_ends: int) -> int:
    if total >= 5:
        return 1_000_000
    if total == 4 and open_ends == 2:
        return 120_000
    if total == 4 and open_ends == 1:
        return 30_000
    if total == 3 and open_ends == 2:
        return 6_000
    if total == 3 and open_ends == 1:
        return 1_200
    if total == 2 and open_ends == 2:
        return 600
    if total == 2 and open_ends == 1:
        return 120
    if total == 1 and open_ends == 2:
        return 20
    return 1


def center_bonus(board: GomokuBoard, row: int, col: int) -> int:
    center = (board.size - 1) / 2
    distance = abs(row - center) + abs(col - center)
    return int((board.size - distance) * 2)


def sample_scored_move(scored_moves: list[tuple[int, int, int]], temperature: float) -> tuple[int, int]:
    import math

    best_score = max(score for score, _, _ in scored_moves)
    scaled = [
        (math.exp((score - best_score) / temperature), row, col)
        for score, row, col in scored_moves
    ]
    total = sum(weight for weight, _, _ in scaled)
    pick = random.random() * total
    running = 0.0

    for weight, row, col in scaled:
        running += weight
        if running >= pick:
            return row, col

    return scored_moves[-1][1], scored_moves[-1][2]
