from __future__ import annotations

import json
from pathlib import Path

from .board import GomokuBoard, Move
from .constants import BLACK, BOARD_SIZE, EMPTY, WHITE
from .features import FEATURE_NAMES, candidate_feature_rows


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install it before CNN training. "
            "For your NVIDIA setup, try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ) from exc

    return torch, nn, functional, DataLoader, Dataset


torch = None
nn = None
F = None
DataLoader = None
Dataset = object


def load_torch_symbols():
    global torch, nn, F, DataLoader, Dataset
    torch, nn, F, DataLoader, Dataset = require_torch()
    return torch, nn, F, DataLoader, Dataset


class PolicyNetBase:
    pass


def build_policy_net():
    torch_mod, nn_mod, _, _, _ = load_torch_symbols()

    class PolicyNet(nn_mod.Module):
        def __init__(self, channels: int = 64) -> None:
            super().__init__()
            self.body = nn_mod.Sequential(
                nn_mod.Conv2d(4, channels, kernel_size=3, padding=1),
                nn_mod.BatchNorm2d(channels),
                nn_mod.ReLU(inplace=True),
                nn_mod.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn_mod.BatchNorm2d(channels),
                nn_mod.ReLU(inplace=True),
                nn_mod.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn_mod.BatchNorm2d(channels),
                nn_mod.ReLU(inplace=True),
            )
            self.policy_head = nn_mod.Sequential(
                nn_mod.Conv2d(channels, 2, kernel_size=1),
                nn_mod.ReLU(inplace=True),
                nn_mod.Flatten(),
                nn_mod.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            )

        def forward(self, inputs):
            return self.policy_head(self.body(inputs))

    return PolicyNet


def board_to_tensor(board: GomokuBoard, player: int):
    torch_mod, _, _, _, _ = load_torch_symbols()
    rival = WHITE if player == BLACK else BLACK
    tensor = torch_mod.zeros((4, board.size, board.size), dtype=torch_mod.float32)

    for row in range(board.size):
        for col in range(board.size):
            value = board.grid[row][col]
            if value == player:
                tensor[0, row, col] = 1.0
            elif value == rival:
                tensor[1, row, col] = 1.0
            elif value == EMPTY:
                tensor[2, row, col] = 1.0

    tensor[3, :, :] = 1.0 if player == BLACK else 0.0
    return tensor


def legal_mask(board: GomokuBoard):
    torch_mod, _, _, _, _ = load_torch_symbols()
    mask = torch_mod.zeros((board.size * board.size,), dtype=torch_mod.bool)
    for row, col in board.legal_moves():
        mask[row * board.size + col] = True
    return mask


def replay_record(record: dict):
    board = GomokuBoard(size=int(record.get("size", BOARD_SIZE)))

    for move in record["moves"]:
        row = int(move["row"])
        col = int(move["col"])
        player = int(move["player"])
        yield board.clone(), player, row * board.size + col
        board.current_player = player
        board.grid[row][col] = player
        board.moves.append(Move(row=row, col=col, player=player))


class JsonlPolicyDatasetBase:
    pass


def build_jsonl_policy_dataset():
    torch_mod, _, _, _, dataset_base = load_torch_symbols()

    class JsonlPolicyDataset(dataset_base):
        def __init__(self, path: str | Path, max_examples: int | None = None) -> None:
            self.samples = []
            input_path = Path(path)

            with input_path.open("r", encoding="utf-8-sig") as file:
                for line in file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "board" in record and "target" in record:
                        board = GomokuBoard(size=int(record.get("size", BOARD_SIZE)))
                        board.grid = [[int(value) for value in row] for row in record["board"]]
                        player = int(record["player"])
                        board.current_player = player
                        target = int(record["target"])
                        self.samples.append((board_to_tensor(board, player), target, legal_mask(board)))
                        if max_examples is not None and len(self.samples) >= max_examples:
                            return
                        continue

                    for board, player, target in replay_record(record):
                        self.samples.append((board_to_tensor(board, player), target, legal_mask(board)))
                        if max_examples is not None and len(self.samples) >= max_examples:
                            return

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            board_tensor, target, mask = self.samples[index]
            return board_tensor, torch_mod.tensor(target, dtype=torch_mod.long), mask

    return JsonlPolicyDataset


class TorchPolicyAgent:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        name: str = "cnn",
        tactical_guard: bool = True,
        heuristic_blend: float = 0.0,
    ) -> None:
        torch_mod, _, _, _, _ = load_torch_symbols()
        self.name = name
        self.device = resolve_device(device)
        self.tactical_guard = tactical_guard
        self.heuristic_blend = heuristic_blend
        payload = torch_mod.load(model_path, map_location=self.device)
        model_class = build_policy_net()
        self.model = model_class(channels=int(payload.get("channels", 64))).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def choose_move(self, board: GomokuBoard) -> tuple[int, int]:
        torch_mod, _, _, _, _ = load_torch_symbols()

        if self.tactical_guard:
            forced_move = choose_forced_tactical_move(board)
            if forced_move is not None:
                return forced_move

        with torch_mod.no_grad():
            inputs = board_to_tensor(board, board.current_player).unsqueeze(0).to(self.device)
            logits = self.model(inputs)[0].detach().cpu()
            mask = legal_mask(board)
            logits[~mask] = -1e9
            if self.heuristic_blend > 0:
                apply_heuristic_blend(board, logits, self.heuristic_blend)
            index = int(torch_mod.argmax(logits).item())
            return index // board.size, index % board.size

    def move_priors(self, board: GomokuBoard, player: int | None = None) -> dict[tuple[int, int], float]:
        torch_mod, _, functional, _, _ = load_torch_symbols()
        target_player = board.current_player if player is None else player

        with torch_mod.no_grad():
            inputs = board_to_tensor(board, target_player).unsqueeze(0).to(self.device)
            logits = self.model(inputs)[0].detach().cpu()
            mask = legal_mask(board)
            legal_indices = mask.nonzero(as_tuple=False).flatten()
            if legal_indices.numel() == 0:
                return {}

            legal_logits = logits[legal_indices]
            probabilities = functional.softmax(legal_logits, dim=0)
            max_probability = float(torch_mod.max(probabilities).item())
            if max_probability <= 0:
                return {}

            priors: dict[tuple[int, int], float] = {}
            for tensor_index, probability in zip(legal_indices, probabilities):
                index = int(tensor_index.item())
                priors[(index // board.size, index % board.size)] = float(probability.item()) / max_probability
            return priors


def resolve_device(device: str):
    torch_mod, _, _, _, _ = load_torch_symbols()
    if device == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(device)


def choose_forced_tactical_move(board: GomokuBoard) -> tuple[int, int] | None:
    candidates = candidate_feature_rows(board, board.current_player)
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
        for row, col, features in candidates:
            if features[feature_index] > 0:
                return row, col

    return None


def apply_heuristic_blend(board: GomokuBoard, logits, strength: float) -> None:
    from .agents import evaluate_move

    player = board.current_player
    rival = WHITE if player == BLACK else BLACK
    scores = []

    for row, col in board.legal_moves():
        own = evaluate_move(board, row, col, player)
        block = evaluate_move(board, row, col, rival)
        score = own + int(block * 0.92)
        scores.append((row * board.size + col, score))

    if not scores:
        return

    max_score = max(score for _, score in scores)
    if max_score <= 0:
        return

    for index, score in scores:
        logits[index] += strength * (score / max_score)
