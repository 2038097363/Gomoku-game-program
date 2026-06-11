from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.dataset import load_examples
from gomoku_ai.training import train_linear_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight Gomoku policy model.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "outputs" / "self_play_games.jsonl"),
        help="JSONL self-play file.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "models" / "linear_policy.json"),
        help="Output model JSON path.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Learning rate.")
    parser.add_argument("--negative-samples", type=int, default=8, help="Negative candidates per example.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit training examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input, max_examples=args.max_examples)
    if not examples:
        raise SystemExit("No training examples were loaded.")

    model, stats = train_linear_policy(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        negative_samples=args.negative_samples,
        seed=args.seed,
    )
    model.save(
        args.output,
        metadata={
            "input": args.input,
            "epochs": stats.epochs,
            "examples": stats.examples,
            "accuracy": stats.accuracy,
            "updates": stats.updates,
        },
    )

    print("training complete")
    print(f"  examples: {stats.examples}")
    print(f"  epochs:   {stats.epochs}")
    print(f"  updates:  {stats.updates}")
    print(f"  top1 acc: {stats.accuracy:.3f}")
    print(f"  output:   {args.output}")


if __name__ == "__main__":
    main()
