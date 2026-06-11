from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.agents import create_agent
from gomoku_ai.board import GomokuBoard, Move
from gomoku_ai.constants import BLACK, WHITE


@dataclass(frozen=True)
class TacticalCase:
    name: str
    description: str
    current_player: int
    stones: tuple[tuple[int, int, int], ...]
    expected: frozenset[tuple[int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed Gomoku tactical positions.")
    parser.add_argument("--agent", default="expert", help="Agent name, for example expert or expert-cnn:path.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failed case.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_agent(args.agent)
    cases = tactical_cases()
    passed = 0
    total_time = 0.0

    print(f"agent: {args.agent}")
    print(f"cases: {len(cases)}")

    for case in cases:
        board = build_board(case)
        start = time.perf_counter()
        move = agent.choose_move(board)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        ok = move in case.expected
        status = "PASS" if ok else "FAIL"
        expected = " ".join(format_move(move) for move in sorted(case.expected))
        print(f"{status} {case.name}: move={format_move(move)} expected={expected} time={elapsed:.3f}s")
        if not ok:
            print(f"  {case.description}")
            if args.stop_on_fail:
                break
        else:
            passed += 1

    print(f"summary: {passed}/{len(cases)} passed, total_time={total_time:.3f}s")
    if passed != len(cases):
        raise SystemExit(1)


def build_board(case: TacticalCase) -> GomokuBoard:
    board = GomokuBoard()
    for row, col, player in case.stones:
        board.grid[row][col] = player
        board.moves.append(Move(row=row, col=col, player=player))
    board.current_player = case.current_player
    return board


def tactical_cases() -> list[TacticalCase]:
    return [
        TacticalCase(
            name="horizontal_win",
            description="Take the immediate horizontal five.",
            current_player=BLACK,
            stones=(
                (7, 4, BLACK),
                (7, 5, BLACK),
                (7, 6, BLACK),
                (7, 7, BLACK),
                (6, 6, WHITE),
                (8, 6, WHITE),
            ),
            expected=frozenset({(7, 3), (7, 8)}),
        ),
        TacticalCase(
            name="vertical_block",
            description="Block the opponent's immediate vertical five.",
            current_player=BLACK,
            stones=(
                (4, 8, WHITE),
                (5, 8, WHITE),
                (6, 8, WHITE),
                (7, 8, WHITE),
                (7, 6, BLACK),
                (8, 6, BLACK),
            ),
            expected=frozenset({(3, 8), (8, 8)}),
        ),
        TacticalCase(
            name="main_diagonal_block",
            description="Block a main-diagonal open four.",
            current_player=BLACK,
            stones=(
                (4, 4, WHITE),
                (5, 5, WHITE),
                (6, 6, WHITE),
                (7, 7, WHITE),
                (7, 5, BLACK),
                (8, 5, BLACK),
            ),
            expected=frozenset({(3, 3), (8, 8)}),
        ),
        TacticalCase(
            name="anti_diagonal_block",
            description="Block an anti-diagonal open four.",
            current_player=BLACK,
            stones=(
                (4, 10, WHITE),
                (5, 9, WHITE),
                (6, 8, WHITE),
                (7, 7, WHITE),
                (7, 5, BLACK),
                (8, 5, BLACK),
            ),
            expected=frozenset({(3, 11), (8, 6)}),
        ),
        TacticalCase(
            name="main_diagonal_broken_block",
            description="Fill the diagonal XX_X gap before it turns into a forcing attack.",
            current_player=BLACK,
            stones=(
                (5, 5, WHITE),
                (6, 6, WHITE),
                (8, 8, WHITE),
                (7, 5, BLACK),
                (8, 5, BLACK),
            ),
            expected=frozenset({(7, 7)}),
        ),
        TacticalCase(
            name="anti_diagonal_broken_block",
            description="Fill the anti-diagonal XX_X gap.",
            current_player=BLACK,
            stones=(
                (5, 9, WHITE),
                (6, 8, WHITE),
                (8, 6, WHITE),
                (7, 5, BLACK),
                (8, 5, BLACK),
            ),
            expected=frozenset({(7, 7)}),
        ),
        TacticalCase(
            name="broken_attack",
            description="Complete own diagonal X_XX attacking shape.",
            current_player=BLACK,
            stones=(
                (5, 5, BLACK),
                (6, 6, BLACK),
                (8, 8, BLACK),
                (6, 8, WHITE),
                (8, 6, WHITE),
            ),
            expected=frozenset({(7, 7)}),
        ),
        TacticalCase(
            name="win_over_block",
            description="Prefer winning immediately over blocking the opponent.",
            current_player=BLACK,
            stones=(
                (7, 4, BLACK),
                (7, 5, BLACK),
                (7, 6, BLACK),
                (7, 7, BLACK),
                (4, 10, WHITE),
                (5, 10, WHITE),
                (6, 10, WHITE),
                (7, 10, WHITE),
            ),
            expected=frozenset({(7, 3), (7, 8)}),
        ),
        TacticalCase(
            name="open_four_attack",
            description="Create or extend the strongest open-four attack.",
            current_player=BLACK,
            stones=(
                (7, 6, BLACK),
                (7, 7, BLACK),
                (7, 8, BLACK),
                (6, 6, WHITE),
                (8, 8, WHITE),
            ),
            expected=frozenset({(7, 5), (7, 9)}),
        ),
        TacticalCase(
            name="center_first",
            description="Empty board should open at center.",
            current_player=BLACK,
            stones=(),
            expected=frozenset({(7, 7)}),
        ),
        TacticalCase(
            name="recorded_game_1_cross_line_defense",
            description="Avoid allowing a horizontal plus diagonal double-kill from the selected human game.",
            current_player=WHITE,
            stones=(
                (7, 7, BLACK),
                (5, 9, WHITE),
                (8, 8, BLACK),
                (9, 9, WHITE),
                (8, 6, BLACK),
                (8, 10, WHITE),
                (9, 7, BLACK),
                (8, 7, WHITE),
                (7, 9, BLACK),
                (6, 10, WHITE),
                (7, 5, BLACK),
                (10, 8, WHITE),
                (7, 11, BLACK),
                (11, 7, WHITE),
                (12, 6, BLACK),
            ),
            expected=frozenset({(10, 6), (6, 4)}),
        ),
        TacticalCase(
            name="recorded_game_2_three_four_defense",
            description="Avoid allowing a vertical plus diagonal double-kill from the selected human game.",
            current_player=WHITE,
            stones=(
                (7, 7, BLACK),
                (5, 9, WHITE),
                (6, 6, BLACK),
                (5, 5, WHITE),
                (5, 7, BLACK),
                (4, 8, WHITE),
                (6, 8, BLACK),
                (6, 7, WHITE),
                (7, 9, BLACK),
                (4, 6, WHITE),
                (8, 10, BLACK),
                (9, 11, WHITE),
                (6, 10, BLACK),
                (4, 10, WHITE),
                (7, 10, BLACK),
            ),
            expected=frozenset({(7, 8), (9, 10)}),
        ),
        TacticalCase(
            name="cross_double_kill_attack",
            description="Create a cross-line attack with multiple immediate winning replies.",
            current_player=BLACK,
            stones=(
                (7, 5, BLACK),
                (7, 6, BLACK),
                (7, 8, BLACK),
                (5, 7, BLACK),
                (6, 7, BLACK),
                (8, 7, BLACK),
                (5, 5, WHITE),
                (9, 9, WHITE),
            ),
            expected=frozenset({(7, 7)}),
        ),
        TacticalCase(
            name="diagonal_horizontal_fork_attack",
            description="Prefer the intersection that creates horizontal and diagonal forcing pressure.",
            current_player=BLACK,
            stones=(
                (7, 5, BLACK),
                (7, 6, BLACK),
                (8, 8, BLACK),
                (9, 9, BLACK),
                (5, 8, WHITE),
                (8, 5, WHITE),
            ),
            expected=frozenset({(7, 7)}),
        ),
    ]


def format_move(move: tuple[int, int]) -> str:
    return f"({move[0]},{move[1]})"


if __name__ == "__main__":
    main()
