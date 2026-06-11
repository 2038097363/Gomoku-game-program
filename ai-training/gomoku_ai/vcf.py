from __future__ import annotations

from dataclasses import dataclass

from .agents import candidate_moves, center_bonus, evaluate_move
from .board import GomokuBoard, opponent
from .threats import analyze_move


@dataclass(frozen=True)
class VcfResult:
    row: int
    col: int
    depth: int


def find_vcf_move(
    board: GomokuBoard,
    player: int,
    max_depth: int = 7,
    radius: int = 2,
    width: int = 8,
) -> tuple[int, int] | None:
    result = search_vcf(board, player, max_depth=max_depth, radius=radius, width=width)
    if result is None:
        return None
    return result.row, result.col


def find_vcf_defense(
    board: GomokuBoard,
    player: int,
    max_depth: int = 5,
    radius: int = 2,
    width: int = 8,
) -> tuple[int, int] | None:
    rival = opponent(player)
    rival_kill = find_vcf_move(board, rival, max_depth=max_depth, radius=radius, width=width)
    if rival_kill is None:
        return None

    defenses = defensive_candidates(board, player, rival, radius)
    safe_defenses = []
    for row, col in defenses:
        child = board.clone()
        child.place(row, col)
        if find_vcf_move(child, rival, max_depth=max_depth - 1, radius=radius, width=width) is None:
            score = evaluate_move(board, row, col, player) + evaluate_move(board, row, col, rival)
            safe_defenses.append((score, row, col))

    if safe_defenses:
        safe_defenses.sort(reverse=True)
        return safe_defenses[0][1], safe_defenses[0][2]

    return rival_kill


def search_vcf(
    board: GomokuBoard,
    player: int,
    max_depth: int,
    radius: int,
    width: int,
) -> VcfResult | None:
    table: dict[tuple, bool] = {}
    moves = forcing_candidates(board, player, radius, width)
    for row, col in moves:
        child = board.clone()
        child.place(row, col)
        if child.winner == player:
            return VcfResult(row=row, col=col, depth=1)
        if vcf_attacker_wins(child, player, max_depth - 1, radius, width, table):
            return VcfResult(row=row, col=col, depth=max_depth)
    return None


def vcf_attacker_wins(
    board: GomokuBoard,
    attacker: int,
    depth: int,
    radius: int,
    width: int,
    table: dict[tuple, bool],
) -> bool:
    key = vcf_key(board, attacker, depth)
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
        for row, col in forcing_candidates(board, attacker, radius, width):
            child = board.clone()
            child.place(row, col)
            if child.winner == attacker or vcf_attacker_wins(child, attacker, depth - 1, radius, width, table):
                table[key] = True
                return True
        table[key] = False
        return False

    defenses = defensive_candidates(board, defender, attacker, radius)
    if not defenses:
        table[key] = True
        return True

    for row, col in defenses:
        child = board.clone()
        child.place(row, col)
        if not vcf_attacker_wins(child, attacker, depth - 1, radius, width, table):
            table[key] = False
            return False

    table[key] = True
    return True


def forcing_candidates(board: GomokuBoard, player: int, radius: int, width: int) -> list[tuple[int, int]]:
    scored = []
    for row, col in candidate_moves(board, radius):
        threat = analyze_move(board, row, col, player).own
        if threat.five or threat.open_four or threat.closed_four or threat.double_four:
            score = (
                threat.five * 1_000_000
                + threat.open_four * 120_000
                + threat.closed_four * 40_000
                + (240_000 if threat.double_four else 0)
                + center_bonus(board, row, col)
            )
            scored.append((score, row, col))
    scored.sort(reverse=True)
    return [(row, col) for _, row, col in scored[:width]]


def defensive_candidates(
    board: GomokuBoard,
    defender: int,
    attacker: int,
    radius: int,
) -> list[tuple[int, int]]:
    scored = []
    for row, col in candidate_moves(board, radius):
        attack_threat = analyze_move(board, row, col, attacker).own
        defend_threat = analyze_move(board, row, col, defender).own
        if attack_threat.forcing or defend_threat.five or defend_threat.forcing:
            score = (
                attack_threat.five * 1_000_000
                + attack_threat.open_four * 140_000
                + attack_threat.closed_four * 40_000
                + defend_threat.five * 900_000
                + defend_threat.open_four * 100_000
                + center_bonus(board, row, col)
            )
            scored.append((score, row, col))
    scored.sort(reverse=True)
    return [(row, col) for _, row, col in scored]


def vcf_key(board: GomokuBoard, attacker: int, depth: int) -> tuple:
    return (
        attacker,
        board.current_player,
        depth,
        tuple(tuple(row) for row in board.grid),
    )
