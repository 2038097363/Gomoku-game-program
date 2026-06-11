from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.torch_policy import build_jsonl_policy_dataset, build_policy_net, load_torch_symbols, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small CNN Gomoku policy with PyTorch.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "outputs" / "teacher_games.jsonl"), help="JSONL training data.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "cnn_policy.pt"), help="Output .pt model.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--channels", type=int, default=64, help="CNN channel width.")
    parser.add_argument("--init-model", default=None, help="Optional .pt checkpoint to continue training from.")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples for a quick smoke run.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    return parser.parse_args()


def main() -> None:
    try:
        torch, _, functional, data_loader_class, _ = load_torch_symbols()
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    args = parse_args()
    device = resolve_device(args.device)
    dataset_class = build_jsonl_policy_dataset()
    dataset = dataset_class(args.input, max_examples=args.max_examples)
    if len(dataset) == 0:
        raise SystemExit("No training examples were loaded.")

    loader = data_loader_class(dataset, batch_size=args.batch_size, shuffle=True)
    model_class = build_policy_net()
    channels = args.channels
    init_payload = None
    if args.init_model:
        init_payload = torch.load(args.init_model, map_location=device)
        channels = int(init_payload.get("channels", channels))

    model = model_class(channels=channels).to(device)
    if init_payload is not None:
        model.load_state_dict(init_payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    print(f"device: {device}")
    print(f"examples: {len(dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for inputs, targets, masks in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            logits = model(inputs)
            logits = logits.masked_fill(~masks, -1e9)
            loss = functional.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_seen += int(inputs.size(0))

        print(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={total_loss / total_seen:.4f}, acc={total_correct / total_seen:.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "channels": channels,
            "input": args.input,
            "epochs": args.epochs,
            "examples": len(dataset),
            "init_model": args.init_model,
        },
        output_path,
    )
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
