from __future__ import annotations

import random
from dataclasses import dataclass

from .dataset import TrainingExample
from .linear_policy import LinearPolicy


@dataclass
class TrainStats:
    epochs: int
    examples: int
    accuracy: float
    updates: int


def train_linear_policy(
    examples: list[TrainingExample],
    epochs: int = 8,
    learning_rate: float = 0.08,
    negative_samples: int = 8,
    seed: int = 42,
) -> tuple[LinearPolicy, TrainStats]:
    rng = random.Random(seed)
    model = LinearPolicy.fresh()
    updates = 0

    for _ in range(epochs):
        shuffled = examples[:]
        rng.shuffle(shuffled)

        for example in shuffled:
            target_features = find_target_features(example)
            negatives = [
                features
                for row, col, features in example.candidates
                if (row, col) != example.target
            ]

            if not negatives:
                continue

            if len(negatives) > negative_samples:
                negatives = rng.sample(negatives, negative_samples)

            target_score = model.score(target_features)
            for negative_features in negatives:
                negative_score = model.score(negative_features)
                margin = 1.0 - target_score + negative_score
                if margin <= 0:
                    continue

                for index, (target_value, negative_value) in enumerate(zip(target_features, negative_features)):
                    model.weights[index] += learning_rate * (target_value - negative_value)
                updates += 1
                target_score = model.score(target_features)

    accuracy = top1_accuracy(model, examples)
    return model, TrainStats(epochs=epochs, examples=len(examples), accuracy=accuracy, updates=updates)


def find_target_features(example: TrainingExample) -> list[float]:
    for row, col, features in example.candidates:
        if (row, col) == example.target:
            return features

    raise ValueError("Target move was not found among candidates")


def top1_accuracy(model: LinearPolicy, examples: list[TrainingExample]) -> float:
    if not examples:
        return 0.0

    hits = 0
    for example in examples:
        best_score = None
        best_moves = []
        for row, col, features in example.candidates:
            score = model.score(features)
            if best_score is None or score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))

        if example.target in best_moves:
            hits += 1

    return hits / len(examples)
