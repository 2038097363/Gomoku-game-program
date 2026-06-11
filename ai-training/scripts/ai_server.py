from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gomoku_ai.agents import create_agent
from gomoku_ai.board import GomokuBoard, Move
from gomoku_ai.constants import BLACK, BOARD_SIZE, EMPTY, WHITE

app = Flask(__name__)
AGENT_CACHE = {}
TRAINING_GAMES_PATH = PROJECT_ROOT / "outputs" / "human_selected_games.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Gomoku AI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser.parse_args()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "models": ["server-search", "server-cnn", "server-hybrid"]})


@app.route("/api/move", methods=["OPTIONS"])
def move_options():
    return "", 204


@app.route("/api/training-game", methods=["OPTIONS"])
def training_game_options():
    return "", 204


@app.route("/api/move", methods=["POST"])
def choose_move():
    payload = request.get_json(force=True)
    board = build_board(payload)
    model = str(payload.get("model", "server-search"))

    agent = get_agent(model)
    if agent is None:
        return jsonify({"ok": False, "error": f"Unknown server model: {model}"}), 400

    row, col = agent.choose_move(board)
    return jsonify({"ok": True, "row": row, "col": col, "model": model})


@app.route("/api/training-game", methods=["POST"])
def save_training_game():
    payload = request.get_json(force=True)
    record = build_training_record(payload)

    TRAINING_GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_GAMES_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return jsonify({
        "ok": True,
        "path": str(TRAINING_GAMES_PATH),
        "moves": len(record["moves"]),
    })


def get_agent(model: str):
    if model in AGENT_CACHE:
        return AGENT_CACHE[model]

    model_path = PROJECT_ROOT / "models" / "cnn_policy_distilled.pt"
    if model == "server-search":
        agent = create_agent("expert")
    elif model == "server-cnn":
        agent = create_agent(f"cnn:{model_path}")
    elif model == "server-hybrid":
        agent = create_agent(f"expert-cnn:{model_path}")
    else:
        return None

    AGENT_CACHE[model] = agent
    return agent


def build_board(payload: dict) -> GomokuBoard:
    size = int(payload.get("size", BOARD_SIZE))
    board = GomokuBoard(size=size)
    grid = payload.get("board")
    if not isinstance(grid, list) or len(grid) != size:
        raise ValueError("Invalid board")

    board.grid = [[int(value) for value in row] for row in grid]
    board.current_player = int(payload.get("currentPlayer", BLACK))
    board.moves = [
        Move(row=int(move["row"]), col=int(move["col"]), player=int(move["player"]))
        for move in payload.get("moves", [])
    ]
    board.game_over = bool(payload.get("gameOver", False))
    winner = payload.get("winner")
    board.winner = None if winner is None else int(winner)
    return board


def build_training_record(payload: dict) -> dict:
    size = int(payload.get("size", BOARD_SIZE))
    moves = payload.get("moves")
    if not isinstance(moves, list) or not moves:
        raise ValueError("Training record needs at least one move.")

    normalized_moves = []
    for move in moves:
        normalized_moves.append({
            "row": int(move["row"]),
            "col": int(move["col"]),
            "player": int(move["player"]),
        })

    winner = payload.get("winner")
    return {
        "size": size,
        "winner": None if winner is None else int(winner),
        "moves": normalized_moves,
        "source": "frontend-selected",
        "mode": str(payload.get("mode", "")),
        "ai_model": str(payload.get("aiModel", "")),
        "human_player": int(payload.get("humanPlayer", BLACK)),
        "game_over": bool(payload.get("gameOver", False)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port)
