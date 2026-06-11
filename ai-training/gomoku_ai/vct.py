from __future__ import annotations

from dataclasses import dataclass

from .agents import candidate_moves, center_bonus, evaluate_move
from .board import GomokuBoard, opponent
from .threats import analyze_move


@dataclass(frozen=True)
class VctResult:
    row: int
    col: int
    depth: int


def find_vct_move(
    board: GomokuBoard,
    player: int,
    max_depth: int = 5,
    radius: int = 2,
    width: int = 8,
) -> tuple[int, int] | None:
    table: dict[tuple, bool] = {}
    for row, col in threat_candidates(board, player, radius, width):
        child = board.clone()
        child.place(row, col)
        if child.winner == player:
            return row, col
        if vct_attacker_wins(child, player, max_depth - 1, radius, width, table):
            return row, col
    return None


def find_vct_defense(
    board: GomokuBoard,
    player: int,
    max_depth: int = 5,
    radius: int = 2,
    width: int = 8,
) -> tuple[int, int] | None:
    rival = opponent(player)
    rival_vct = find_vct_move(board, rival, max_depth=max_depth, radius=radius, width=width)
    if rival_vct is None:
        return None

    defenses = defense_candidates(board, player, rival, radius, width + 4)
    safe = []
    for row, col in defenses:
        child = board.clone()
        child.place(row, col)
        if find_vct_move(child, rival, max_depth=max_depth - 1, radius=radius, width=width) is None:
            score = evaluate_move(board, row, col, player) + evaluate_move(board, row, col, rival)
            safe.append((score, row, col))

    if safe:
        safe.sort(reverse=True)
        return safe[0][1], safe[0][2]

    return rival_vct


def vct_attacker_wins(
    board: GomokuBoard,
    attacker: int,
    depth: int,
    radius: int,
    width: int,
    table: dict[tuple, bool],
) -> bool:
    key = vct_key(board, attacker, depth)
    if key in table:
        return table[key]

    if board.game_over:
        table[key] = board.winner == attacker
        return table[key]

    if depth <= 0:
        table[key] = False
        return False

    defender = opponent(attacker)
    if board.current_player == attacker:
        for row, col in threat_candidates(board, attacker, radius, width):
            child = board.clone()
            child.place(row, col)
            if child.winner == attacker or vct_attacker_wins(child, attacker, depth - 1, radius, width, table):
                table[key] = True
                return True
        table[key] = False
        return False

    defenses = defense_candidates(board, defender, attacker, radius, width + 4)
    if not defenses:
        table[key] = True
        return True

    for row, col in defenses:
        child = board.clone()
        child.place(row, col)
        if not vct_attacker_wins(child, attacker, depth - 1, radius, width, table):
            table[key] = False
            return False

    table[key] = True
    return True


def threat_candidates(board: GomokuBoard, player: int, radius: int, width: int) -> list[tuple[int, int]]:
    scored = []
    for row, col in candidate_moves(board, radius):
        threat = analyze_move(board, row, col, player).own
        if (
            threat.five
            or threat.open_four
            or threat.closed_four
            or threat.double_four
            or threat.four_three
            or threat.double_open_three
        ):
            score = (
                threat.five * 1_000_000
                + threat.open_four * 150_000
                + threat.closed_four * 45_000
                + (300_000 if threat.double_four else 0)
                + (220_000 if threat.four_three else 0)
                + (120_000 if threat.double_open_three else 0)
                + center_bonus(board, row, col)
            )
            scored.append((score, row, col))
    scored.sort(reverse=True)
    return [(row, col) for _, row, col in scored[:width]]


def defense_candidates(
    board: GomokuBoard,
    defender: int,
    attacker: int,
    radius: int,
    width: int,
) -> list[tuple[int, int]]:
    scored = []
    for row, col in candidate_moves(board, radius):
        attack = analyze_move(board, row, col, attacker).own
        defend = analyze_move(board, row, col, defender).own
        if (
            attack.five
            or attack.open_four
            or attack.closed_four
            or attack.double_four
            or attack.four_three
            or attack.double_open_three
            or defend.five
            or defend.open_four
            or defend.double_four
        ):
            score = (
                attack.five * 1_000_000
                + attack.open_four * 160_000
                + attack.closed_four * 50_000
                + (320_000 if attack.double_four else 0)
                + (230_000 if attack.four_three else 0)
                + (130_000 if attack.double_open_three else 0)
                + defend.five * 900_000
                + defend.open_four * 120_000
                + center_bonus(board, row, col)
            )
            scored.append((score, row, col))
    scored.sort(reverse=True)
    return [(row, col) for _, row, col in scored[:width]]


def vct_key(board: GomokuBoard, attacker: int, depth: int) -> tuple:
    return (
        attacker,
        board.current_player,
        depth,
        tuple(tuple(row) for row in board.grid),
    )
