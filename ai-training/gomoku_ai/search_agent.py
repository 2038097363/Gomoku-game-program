from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

from .agents import candidate_moves, center_bonus, evaluate_move
from .board import GomokuBoard, opponent
from .constants import DIRECTIONS, EMPTY
from .openings import is_formula_like_opening
from .threats import analyze_move
from .vct import find_vct_defense, find_vct_move
from .vcf import find_vcf_defense, find_vcf_move

WIN_SCORE = 10_000_000
POLICY_PRIOR_WEIGHT = 75_000
DOUBLE_THREAT_PENALTY = 700_000
DOUBLE_THREAT_BONUS = 1_050_000
FORCING_ATTACK_BONUS = 180_000
PolicyPrior = Callable[[GomokuBoard, int], dict[tuple[int, int], float]]


class SearchTimeout(RuntimeError):
    pass


@dataclass
class SearchAgent:
    name: str = "search"
    depth: int = 2
    width: int = 10
    radius: int = 2
    max_extra_width: int = 8
    policy_prior: PolicyPrior | None = None

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        player = board.current_player
        vcf_move = find_vcf_move(board, player)
        if vcf_move is not None:
            return vcf_move
        vcf_defense = find_vcf_defense(board, player)
        if vcf_defense is not None:
            return vcf_defense
        vct_move = find_vct_move(board, player)
        if vct_move is not None:
            return vct_move
        vct_defense = find_vct_defense(board, player)
        if vct_defense is not None:
            return vct_defense
        tactical = choose_tactical_move(board, player, self.radius)
        if tactical is not None:
            return tactical

        scored = safe_ordered_candidates(board, player, self.radius, self.width, self.max_extra_width)
        best_score = None
        best_moves: list[tuple[int, int]] = []

        for _, row, col in scored:
            child = board.clone()
            child.place(row, col)
            score = minimax(
                child,
                root_player=player,
                depth=self.depth - 1,
                alpha=-WIN_SCORE,
                beta=WIN_SCORE,
                width=self.width,
                radius=self.radius,
                max_extra_width=self.max_extra_width,
                table={},
            )

            if best_score is None or score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))

        return random.choice(best_moves)


@dataclass
class ExpertSearchAgent(SearchAgent):
    name: str = "expert"
    depth: int = 3
    width: int = 50
    radius: int = 2
    max_extra_width: int = 50
    time_limit_seconds: float = 26.0
    opening_moves: int = 8
    opening_depth: int = 2
    opening_width: int = 14
    opening_time_limit_seconds: float = 2.0

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        player = board.current_player
        if not board.moves:
            center = board.size // 2
            return center, center
        vcf_move = find_vcf_move(board, player, max_depth=7, width=10)
        if vcf_move is not None:
            return vcf_move
        vcf_defense = find_vcf_defense(board, player, max_depth=5, width=10)
        if vcf_defense is not None:
            return vcf_defense
        tactical = choose_tactical_move(board, player, self.radius)
        if tactical is not None:
            return tactical

        search_depth, search_width, search_extra_width, time_limit = self.search_profile(board)
        deadline = time.monotonic() + time_limit
        table: dict[tuple, int] = {}
        prior_scores = self.get_policy_prior(board, player)
        scored = safe_ordered_candidates(
            board,
            player,
            self.radius,
            search_width,
            search_extra_width,
            prior_scores=prior_scores,
        )
        best_score = None
        best_moves: list[tuple[int, int]] = []
        fallback = stable_best_move(board, [(row, col) for _, row, col in scored], player, prior_scores)

        for _, row, col in scored:
            if time.monotonic() >= deadline:
                break
            child = board.clone()
            child.place(row, col)
            try:
                score = minimax(
                    child,
                    root_player=player,
                    depth=search_depth - 1,
                    alpha=-WIN_SCORE,
                    beta=WIN_SCORE,
                    width=search_width,
                    radius=self.radius,
                    max_extra_width=search_extra_width,
                    table=table,
                    deadline=deadline,
                )
            except SearchTimeout:
                break

            if best_score is None or score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))

        if not best_moves:
            return fallback
        return stable_best_move(board, best_moves, player, prior_scores)

    def get_policy_prior(self, board: GomokuBoard, player: int) -> dict[tuple[int, int], float] | None:
        if self.policy_prior is None:
            return None
        return self.policy_prior(board, player)

    def search_profile(self, board: GomokuBoard) -> tuple[int, int, int, float]:
        if is_formula_like_opening(board):
            return self.depth, self.width, self.max_extra_width, min(self.time_limit_seconds, 8.0)
        if len(board.moves) <= self.opening_moves:
            return (
                self.opening_depth,
                self.opening_width,
                max(10, self.opening_width),
                self.opening_time_limit_seconds,
            )
        return self.depth, self.width, self.max_extra_width, self.time_limit_seconds


def minimax(
    board: GomokuBoard,
    root_player: int,
    depth: int,
    alpha: int,
    beta: int,
    width: int,
    radius: int,
    max_extra_width: int,
    table: dict[tuple, int],
    deadline: float | None = None,
) -> int:
    if deadline is not None and time.monotonic() >= deadline:
        raise SearchTimeout()

    key = position_key(board, root_player, depth)
    if key in table:
        return table[key]

    if board.game_over:
        if board.winner == root_player:
            return WIN_SCORE - len(board.moves)
        if board.winner is None:
            return 0
        return -WIN_SCORE + len(board.moves)

    if depth <= 0:
        value = evaluate_position(board, root_player)
        table[key] = value
        return value

    current = board.current_player
    maximizing = current == root_player
    tactical = choose_tactical_move(board, current, radius)
    if tactical is not None:
        candidates = [(0, tactical[0], tactical[1])]
    else:
        node_width, node_extra_width = node_search_limits(width, max_extra_width, depth)
        candidates = safe_ordered_candidates(board, current, radius, node_width, node_extra_width)

    if maximizing:
        value = -WIN_SCORE
        for _, row, col in candidates:
            child = board.clone()
            child.place(row, col)
            value = max(value, minimax(child, root_player, depth - 1, alpha, beta, width, radius, max_extra_width, table, deadline))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        table[key] = value
        return value

    value = WIN_SCORE
    for _, row, col in candidates:
        child = board.clone()
        child.place(row, col)
        value = min(value, minimax(child, root_player, depth - 1, alpha, beta, width, radius, max_extra_width, table, deadline))
        beta = min(beta, value)
        if alpha >= beta:
            break
    table[key] = value
    return value


def choose_tactical_move(board: GomokuBoard, player: int, radius: int) -> tuple[int, int] | None:
    rival = opponent(player)
    candidates = candidate_moves(board, radius)

    for target_player in (player, rival):
        winning = []
        for row, col in candidates:
            child = board.clone()
            child.current_player = target_player
            if child.place(row, col) and child.winner == target_player:
                score = evaluate_move(board, row, col, player) + evaluate_move(board, row, col, rival)
                winning.append((score, row, col))
        if winning:
            return best_scored_move(winning)

    forced = []
    for row, col in candidates:
        threat = analyze_move(board, row, col, player)
        broken_score = directional_threat_bonus(board, row, col, player)
        if (
            threat.own.forcing
            or threat.opponent.forcing
            or threat.own.double_open_three
            or threat.opponent.double_open_three
            or broken_score >= 165_000
        ):
            forced.append((threat.tactical_score + broken_score, row, col))

    if forced:
        return best_scored_move(forced)

    return None


def immediate_winning_moves(board: GomokuBoard, player: int, radius: int) -> list[tuple[int, int]]:
    winning = []
    for row, col in candidate_moves(board, radius):
        child = board.clone()
        child.current_player = player
        if child.place(row, col) and child.winner == player:
            winning.append((row, col))
    return winning


def ordered_candidates(
    board: GomokuBoard,
    player: int,
    radius: int,
    width: int,
    prior_scores: dict[tuple[int, int], float] | None = None,
) -> list[tuple[int, int, int]]:
    rival = opponent(player)
    scored = []

    for row, col in candidate_moves(board, radius):
        own = evaluate_move(board, row, col, player)
        block = evaluate_move(board, row, col, rival)
        score = (
            + own
            + int(block * 1.04)
            + int(threat_priority(board, row, col, player) / 10_000)
            + directional_threat_bonus(board, row, col, player)
            + policy_prior_bonus(prior_scores, row, col)
            + center_bonus(board, row, col)
        )
        scored.append((score, row, col))

    scored.sort(reverse=True)
    return scored[:width]


def safe_ordered_candidates(
    board: GomokuBoard,
    player: int,
    radius: int,
    width: int,
    max_extra_width: int,
    prior_scores: dict[tuple[int, int], float] | None = None,
) -> list[tuple[int, int, int]]:
    target_width = dynamic_width(board, player, radius, width, max_extra_width)
    expanded_width = min(target_width * 2, target_width + max_extra_width)
    scored = ordered_candidates(board, player, radius, expanded_width, prior_scores)
    safe = []
    risky = []

    for score, row, col in scored:
        child = board.clone()
        child.place(row, col)
        if child.game_over and child.winner == player:
            safe.append((score + WIN_SCORE, row, col))
            continue
        rival = opponent(player)
        if has_immediate_win(child, rival, radius):
            continue
        attack_score = attacking_followup_score(child, player, radius)
        double_threats = opponent_double_threat_count(child, rival, radius)
        if double_threats:
            risky.append((score + attack_score - DOUBLE_THREAT_PENALTY * double_threats, row, col))
        else:
            safe.append((score + attack_score, row, col))

    return (safe or risky or scored)[:target_width]


def node_search_limits(width: int, max_extra_width: int, depth: int) -> tuple[int, int]:
    if depth >= 4:
        return max(7, int(width * 0.36)), max(8, int(max_extra_width * 0.28))
    if depth == 3:
        return max(9, int(width * 0.50)), max(9, int(max_extra_width * 0.38))
    if depth == 2:
        return max(11, int(width * 0.68)), max(11, int(max_extra_width * 0.52))
    return width, max_extra_width


def dynamic_width(
    board: GomokuBoard,
    player: int,
    radius: int,
    base_width: int,
    max_extra_width: int,
) -> int:
    rival = opponent(player)
    max_priority = 0
    strong_count = 0
    medium_count = 0
    forcing_count = 0
    diagonal_threat_count = 0

    for row, col in candidate_moves(board, radius):
        own = analyze_move(board, row, col, player)
        block = analyze_move(board, row, col, rival)
        priority = max(own.priority, block.priority)
        max_priority = max(max_priority, priority)
        if priority >= 200_000_000:
            strong_count += 1
        if priority >= 50_000_000:
            medium_count += 1
        if own.own.forcing or own.opponent.forcing or block.own.forcing or block.opponent.forcing:
            forcing_count += 1
        if has_diagonal_pressure(board, row, col, player) or has_diagonal_pressure(board, row, col, rival):
            diagonal_threat_count += 1

    extra = 0
    if max_priority >= 700_000_000:
        extra += 8
    elif max_priority >= 400_000_000:
        extra += 6
    elif max_priority >= 200_000_000:
        extra += 3

    extra += min(8, strong_count // 2)
    extra += min(6, medium_count // 4)
    extra += min(4, forcing_count // 2)
    extra += min(6, diagonal_threat_count // 2)

    return base_width + min(max_extra_width, extra)


def has_immediate_win(board: GomokuBoard, player: int, radius: int) -> bool:
    for row, col in candidate_moves(board, radius):
        child = board.clone()
        child.current_player = player
        if child.place(row, col) and child.winner == player:
            return True
    return False


def opponent_double_threat_count(board: GomokuBoard, player: int, radius: int) -> int:
    max_winning_replies = 0
    for row, col in candidate_moves(board, radius):
        child = board.clone()
        child.current_player = player
        if not child.place(row, col):
            continue
        if child.game_over and child.winner == player:
            return 2
        winning_replies = len(immediate_winning_moves(child, player, radius))
        max_winning_replies = max(max_winning_replies, winning_replies)
        if max_winning_replies >= 2:
            return max_winning_replies
    return max_winning_replies


def attacking_followup_score(board: GomokuBoard, player: int, radius: int) -> int:
    winning_replies = len(immediate_winning_moves(board, player, radius))
    if winning_replies >= 2:
        return DOUBLE_THREAT_BONUS + min(4, winning_replies) * 120_000
    if winning_replies == 1:
        return FORCING_ATTACK_BONUS

    best_combo = 0
    for row, col in candidate_moves(board, radius):
        threat = analyze_move(board, row, col, player)
        combo = 0
        if threat.own.four_three:
            combo += 520_000
        if threat.own.double_open_three:
            combo += 360_000
        if threat.own.double_four:
            combo += 650_000
        if threat.own.open_four:
            combo += 220_000
        if threat.own.open_three:
            combo += threat.own.open_three * 70_000
        combo += directional_threat_bonus_for(board, row, col, player)
        best_combo = max(best_combo, combo)

    return min(best_combo, 900_000)


def evaluate_position(board: GomokuBoard, player: int) -> int:
    rival = opponent(player)
    player_score = 0
    rival_score = 0

    for row, col in candidate_moves(board, radius=2):
        player_score += (
            evaluate_move(board, row, col, player)
            + analyze_move(board, row, col, player).tactical_score
            + directional_threat_bonus_for(board, row, col, player)
        )
        rival_score += (
            evaluate_move(board, row, col, rival)
            + analyze_move(board, row, col, rival).tactical_score
            + directional_threat_bonus_for(board, row, col, rival)
        )

    player_score += attacking_followup_score(board, player, radius=2)
    rival_score += int(attacking_followup_score(board, rival, radius=2) * 0.96)

    return player_score - int(rival_score * 1.01) + initiative_bonus(board, player)


def best_scored_move(scored: list[tuple[int, int, int]]) -> tuple[int, int]:
    best_score = max(score for score, _, _ in scored)
    best = [(row, col) for score, row, col in scored if score == best_score]
    return random.choice(best)


def stable_best_move(
    board: GomokuBoard,
    moves: list[tuple[int, int]],
    player: int,
    prior_scores: dict[tuple[int, int], float] | None = None,
) -> tuple[int, int]:
    rival = opponent(player)
    scored = []
    for row, col in moves:
        score = (
            evaluate_move(board, row, col, player)
            + int(evaluate_move(board, row, col, rival) * 1.03)
            + directional_threat_bonus(board, row, col, player)
            + policy_prior_bonus(prior_scores, row, col)
            + center_bonus(board, row, col)
        )
        scored.append((score, row, col))
    return best_scored_move(scored)


def initiative_bonus(board: GomokuBoard, player: int) -> int:
    if board.current_player == player:
        return 150
    return -150


def threat_priority(board: GomokuBoard, row: int, col: int, player: int) -> int:
    return analyze_move(board, row, col, player).priority


def policy_prior_bonus(prior_scores: dict[tuple[int, int], float] | None, row: int, col: int) -> int:
    if not prior_scores:
        return 0
    return int(prior_scores.get((row, col), 0.0) * POLICY_PRIOR_WEIGHT)


def directional_threat_bonus(board: GomokuBoard, row: int, col: int, player: int) -> int:
    rival = opponent(player)
    own_bonus = directional_threat_bonus_for(board, row, col, player, diagonal_multiplier=1.18)
    block_bonus = directional_threat_bonus_for(board, row, col, rival, diagonal_multiplier=1.32)
    return own_bonus + int(block_bonus * 1.08)


def directional_threat_bonus_for(
    board: GomokuBoard,
    row: int,
    col: int,
    player: int,
    diagonal_multiplier: float = 1.25,
) -> int:
    return (
        line_pressure_bonus(board, row, col, player, diagonal_multiplier)
        + broken_pattern_bonus(board, row, col, player, diagonal_multiplier + 0.18)
    )


def has_diagonal_pressure(board: GomokuBoard, row: int, col: int, player: int) -> bool:
    for dr, dc in ((1, 1), (1, -1)):
        left_count, left_open = count_direction(board, row, col, -dr, -dc, player)
        right_count, right_open = count_direction(board, row, col, dr, dc, player)
        total = left_count + 1 + right_count
        open_ends = int(left_open) + int(right_open)
        if total >= 3 and open_ends > 0:
            return True
        if broken_window_pressure(board, row, col, dr, dc, player):
            return True
    return False


def line_pressure_bonus(
    board: GomokuBoard,
    row: int,
    col: int,
    player: int,
    diagonal_multiplier: float,
) -> int:
    bonus = 0
    for dr, dc in DIRECTIONS:
        left_count, left_open = count_direction(board, row, col, -dr, -dc, player)
        right_count, right_open = count_direction(board, row, col, dr, dc, player)
        total = left_count + 1 + right_count
        open_ends = int(left_open) + int(right_open)
        direction_bonus = pressure_shape_bonus(total, open_ends)
        if dr != 0 and dc != 0:
            direction_bonus = int(direction_bonus * diagonal_multiplier)
        bonus += direction_bonus
    return bonus


def pressure_shape_bonus(total: int, open_ends: int) -> int:
    if total >= 5:
        return 900_000
    if total == 4 and open_ends == 2:
        return 180_000
    if total == 4 and open_ends == 1:
        return 70_000
    if total == 3 and open_ends == 2:
        return 20_000
    if total == 3 and open_ends == 1:
        return 5_500
    if total == 2 and open_ends == 2:
        return 1_100
    return 0


def broken_pattern_bonus(
    board: GomokuBoard,
    row: int,
    col: int,
    player: int,
    diagonal_multiplier: float,
) -> int:
    bonus = 0
    for dr, dc in DIRECTIONS:
        direction_bonus = 0
        for start in range(-4, 1):
            window = scan_window(board, row, col, dr, dc, start, player)
            if window is None:
                continue
            stones, empties, edge_open = window
            if stones == 4 and empties == 1:
                direction_bonus = max(direction_bonus, 125_000 + edge_open * 18_000)
            elif stones == 3 and empties == 2:
                direction_bonus = max(direction_bonus, 16_000 + edge_open * 3_500)
            elif stones == 2 and empties == 3 and edge_open == 2:
                direction_bonus = max(direction_bonus, 1_800)
        if dr != 0 and dc != 0:
            direction_bonus = int(direction_bonus * diagonal_multiplier)
        bonus += direction_bonus
    return bonus


def broken_window_pressure(
    board: GomokuBoard,
    row: int,
    col: int,
    dr: int,
    dc: int,
    player: int,
) -> bool:
    for start in range(-4, 1):
        window = scan_window(board, row, col, dr, dc, start, player)
        if window is None:
            continue
        stones, empties, _ = window
        if stones >= 3 and empties > 0:
            return True
    return False


def scan_window(
    board: GomokuBoard,
    row: int,
    col: int,
    dr: int,
    dc: int,
    start: int,
    player: int,
) -> tuple[int, int, int] | None:
    rival = opponent(player)
    stones = 0
    empties = 0

    for offset in range(start, start + 5):
        next_row = row + dr * offset
        next_col = col + dc * offset
        if not (0 <= next_row < board.size and 0 <= next_col < board.size):
            return None
        value = player if offset == 0 else board.grid[next_row][next_col]
        if value == player:
            stones += 1
        elif value == EMPTY:
            empties += 1
        elif value == rival:
            return None

    before_row = row + dr * (start - 1)
    before_col = col + dc * (start - 1)
    after_row = row + dr * (start + 5)
    after_col = col + dc * (start + 5)
    edge_open = int(is_empty_cell(board, before_row, before_col)) + int(is_empty_cell(board, after_row, after_col))
    return stones, empties, edge_open


def is_empty_cell(board: GomokuBoard, row: int, col: int) -> bool:
    return 0 <= row < board.size and 0 <= col < board.size and board.grid[row][col] == EMPTY


def count_direction(
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


def position_key(board: GomokuBoard, root_player: int, depth: int) -> tuple:
    return (
        root_player,
        board.current_player,
        depth,
        tuple(tuple(row) for row in board.grid),
    )
