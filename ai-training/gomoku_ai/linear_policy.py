from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .board import GomokuBoard
from .features import FEATURE_NAMES, candidate_feature_rows


@dataclass
class LinearPolicy:
    weights: list[float]
    tactical_guard: bool = True

    @classmethod
    def fresh(cls) -> "LinearPolicy":
        return cls(weights=[0.0 for _ in FEATURE_NAMES])

    def score(self, features: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features))

    def choose_move(self, board: GomokuBoard, temperature: float = 0.0) -> tuple[int, int]:
        candidates = candidate_feature_rows(board, board.current_player)
        if not candidates:
            raise RuntimeError("No legal moves available")

        if self.tactical_guard:
            forced_move = choose_forced_tactical_move(candidates)
            if forced_move is not None:
                return forced_move

        scores = [self.score(features) for _, _, features in candidates]
        if temperature > 0:
            return sample_by_temperature(candidates, scores, temperature)

        best_score = max(scores)
        best = [
            (row, col)
            for (row, col, _), score in zip(candidates, scores)
            if score == best_score
        ]
        return random.choice(best)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "weights": self.weights,
            "tactical_guard": self.tactical_guard,
            "metadata": metadata or {},
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = tuple(payload["feature_names"])
        if feature_names != FEATURE_NAMES:
            raise ValueError("Model feature set does not match current code")
        return cls(
            weights=[float(value) for value in payload["weights"]],
            tactical_guard=bool(payload.get("tactical_guard", True)),
        )


@dataclass
class LinearPolicyAgent:
    model: LinearPolicy
    name: str = "linear"
    temperature: float = 0.0

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        return self.model.choose_move(board, temperature=self.temperature)


def sample_by_temperature(
    candidates: list[tuple[int, int, list[float]]],
    scores: list[float],
    temperature: float,
) -> tuple[int, int]:
    max_score = max(scores)
    scaled = [math.exp((score - max_score) / temperature) for score in scores]
    total = sum(scaled)
    pick = random.random() * total

    running = 0.0
    for (row, col, _), weight in zip(candidates, scaled):
        running += weight
        if running >= pick:
            return row, col

    return candidates[-1][0], candidates[-1][1]


def choose_forced_tactical_move(candidates: list[tuple[int, int, list[float]]]) -> tuple[int, int] | None:
    own_five_index = FEATURE_NAMES.index("own_five")
    block_five_index = FEATURE_NAMES.index("block_five")
    own_open_four_index = FEATURE_NAMES.index("own_open_four")
    block_open_four_index = FEATURE_NAMES.index("block_open_four")
    own_closed_four_index = FEATURE_NAMES.index("own_closed_four")
    block_closed_four_index = FEATURE_NAMES.index("block_closed_four")

    for feature_index in (
        own_five_index,
        block_five_index,
        own_open_four_index,
        block_open_four_index,
        own_closed_four_index,
        block_closed_four_index,
    ):
        tactical = [
            (row, col)
            for row, col, features in candidates
            if features[feature_index] > 0
        ]
        if tactical:
            return random.choice(tactical)

    return None
