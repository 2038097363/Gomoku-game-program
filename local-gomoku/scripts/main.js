(function initGame(namespace) {
  const { BOARD_SIZE, BLACK, WHITE, TEXT } = namespace.config;
  const totalCells = BOARD_SIZE * BOARD_SIZE;
  const aiDelayMs = 450;
  const game = namespace.state.createGameState();
  const elements = namespace.ui.createElements();
  let aiTimer = null;

  const viewState = {
    mode: "two-player",
    humanPlayer: BLACK,
    aiModel: "heuristic",
    boardLocked: false,
    message: "",
    trainingRecording: false,
    trainingSaved: false,
  };

  const handlers = {
    onCellClick(row, col) {
      if (viewState.boardLocked || isAiTurn()) {
        return;
      }

      playMove(row, col);
    },
  };

  async function playMove(row, col) {
    if (!namespace.rules.canPlace(game, row, col)) {
      return false;
    }

    const player = game.currentPlayer;
    namespace.state.placeStone(game, row, col);
    finishMove(row, col, player);
    namespace.ui.render(game, handlers, elements, viewState);
    if (game.gameOver) {
      await finalizeTrainingRecordIfNeeded();
    } else {
      scheduleAiIfNeeded();
    }

    return true;
  }

  function finishMove(row, col, player) {
    const winningLine = namespace.rules.getWinningLine(game.board, row, col, player);

    viewState.message = "";

    if (winningLine.length >= 5) {
      game.gameOver = true;
      game.winner = player;
      game.winningLine = winningLine;
      game.scores[player] += 1;
      game.scoredWinner = player;
      viewState.boardLocked = false;
    } else if (game.moves.length === totalCells) {
      game.gameOver = true;
      game.winner = null;
      game.scoredWinner = null;
      viewState.boardLocked = false;
    } else {
      game.currentPlayer = namespace.state.switchPlayer(game.currentPlayer);
    }
  }

  function isAiTurn() {
    return viewState.mode === "ai" && game.currentPlayer !== viewState.humanPlayer && !game.gameOver;
  }

  function scheduleAiIfNeeded() {
    clearAiTimer();

    if (!isAiTurn()) {
      viewState.boardLocked = false;
      namespace.ui.render(game, handlers, elements, viewState);
      return;
    }

    viewState.boardLocked = true;
    viewState.message = getThinkingMessage();
    namespace.ui.render(game, handlers, elements, viewState);

    aiTimer = window.setTimeout(() => {
      chooseAiMove()
        .then((move) => {
          if (move) {
            return playMove(move[0], move[1]);
          }
          return null;
        })
        .catch(() => {
          viewState.message = "\u672c\u5730 AI \u670d\u52a1\u672a\u542f\u52a8\uff0c\u8bf7\u5148\u8fd0\u884c python ai-training\\scripts\\ai_server.py";
        })
        .finally(() => {
          viewState.boardLocked = false;
          aiTimer = null;
          namespace.ui.render(game, handlers, elements, viewState);
        });
    }, aiDelayMs);
  }

  function clearAiTimer() {
    if (aiTimer !== null) {
      window.clearTimeout(aiTimer);
      aiTimer = null;
    }
  }

  function resetRound() {
    clearAiTimer();
    namespace.state.resetRound(game);
    game.currentPlayer = BLACK;
    viewState.boardLocked = false;
    viewState.message = "";
    viewState.trainingRecording = false;
    viewState.trainingSaved = false;
    namespace.ui.render(game, handlers, elements, viewState);
    scheduleAiIfNeeded();
  }

  function setMode(mode) {
    viewState.mode = mode;
    resetRound();
  }

  function setHumanPlayer(player) {
    viewState.humanPlayer = player;
    resetRound();
  }

  function setAiModel(model) {
    viewState.aiModel = model;
    resetRound();
  }

  function startTrainingRecord() {
    if (game.moves.length === 0 || viewState.boardLocked) {
      return;
    }

    viewState.trainingRecording = true;
    viewState.trainingSaved = false;
    viewState.message = "\u5df2\u542f\u52a8\u8bad\u7ec3\u8bb0\u5f55\uff0c\u672c\u5c40\u5c06\u4ece\u7b2c\u4e00\u624b\u8bb0\u5230\u7ed3\u675f\u3002";
    namespace.ui.render(game, handlers, elements, viewState);
  }

  async function finalizeTrainingRecordIfNeeded() {
    if (!viewState.trainingRecording || viewState.trainingSaved || !game.gameOver) {
      namespace.ui.render(game, handlers, elements, viewState);
      return;
    }

    if (!game.winner) {
      viewState.trainingRecording = false;
      viewState.message = "\u672c\u5c40\u672a\u51fa\u73b0\u4e94\u8fde\uff0c\u672a\u7eb3\u5165\u8bad\u7ec3\u6570\u636e\u3002";
      namespace.ui.render(game, handlers, elements, viewState);
      return;
    }

    viewState.boardLocked = true;
    viewState.message = "\u5bf9\u5c40\u7ed3\u675f\uff0c\u6b63\u5728\u81ea\u52a8\u7eb3\u5165\u8bad\u7ec3...";
    namespace.ui.render(game, handlers, elements, viewState);

    try {
      const response = await fetch("http://127.0.0.1:8765/api/training-game", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          size: BOARD_SIZE,
          moves: game.moves,
          winner: game.winner,
          gameOver: game.gameOver,
          mode: viewState.mode,
          aiModel: viewState.aiModel,
          humanPlayer: viewState.humanPlayer,
        }),
      });

      if (!response.ok) {
        throw new Error("save failed");
      }

      const payload = await response.json();
      if (!payload.ok) {
        throw new Error(payload.error || "save failed");
      }

      viewState.trainingSaved = true;
      viewState.message = `\u5df2\u7eb3\u5165\u8bad\u7ec3\u6570\u636e\uff1a${payload.moves} \u624b`;
    } catch (error) {
      viewState.message = "\u81ea\u52a8\u4fdd\u5b58\u5931\u8d25\uff0c\u8bf7\u786e\u8ba4\u672c\u5730 AI \u670d\u52a1\u6b63\u5728\u8fd0\u884c\u3002";
    } finally {
      viewState.boardLocked = false;
      namespace.ui.render(game, handlers, elements, viewState);
    }
  }

  async function chooseAiMove() {
    if (viewState.aiModel === "server-search" || viewState.aiModel === "server-cnn" || viewState.aiModel === "server-hybrid") {
      return chooseServerAiMove(viewState.aiModel);
    }

    if (viewState.aiModel === "search") {
      return namespace.ai.chooseSearchMove(game, game.currentPlayer, {
        depth: 2,
        width: 10,
        radius: 2,
      });
    }

    return namespace.ai.chooseHeuristicMove(game, game.currentPlayer);
  }

  async function chooseServerAiMove(model) {
    const response = await fetch("http://127.0.0.1:8765/api/move", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        size: BOARD_SIZE,
        board: game.board,
        currentPlayer: game.currentPlayer,
        moves: game.moves,
        gameOver: game.gameOver,
        winner: game.winner,
        model,
      }),
    });

    if (!response.ok) {
      throw new Error("AI server request failed");
    }

    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload.error || "AI server failed");
    }

    return [payload.row, payload.col];
  }

  function getThinkingMessage() {
    const labels = {
      heuristic: "\u5feb\u901f\u89c4\u5219 AI",
      search: "\u6d45\u5c42\u641c\u7d22 AI",
      "server-search": "\u4e13\u5bb6\u641c\u7d22 AI",
      "server-cnn": "CNN AI",
      "server-hybrid": "\u4e13\u5bb6+CNN AI",
    };
    return `${labels[viewState.aiModel] || "\u7535\u8111"} \u6b63\u5728\u601d\u8003...`;
  }

  elements.twoPlayerModeButton.addEventListener("click", () => setMode("two-player"));
  elements.aiModeButton.addEventListener("click", () => setMode("ai"));
  elements.humanBlackButton.addEventListener("click", () => setHumanPlayer(BLACK));
  elements.humanWhiteButton.addEventListener("click", () => setHumanPlayer(WHITE));
  elements.heuristicAiButton.addEventListener("click", () => setAiModel("heuristic"));
  elements.searchAiButton.addEventListener("click", () => setAiModel("search"));
  elements.serverSearchAiButton.addEventListener("click", () => setAiModel("server-search"));
  elements.serverCnnAiButton.addEventListener("click", () => setAiModel("server-cnn"));
  elements.serverHybridAiButton.addEventListener("click", () => setAiModel("server-hybrid"));
  elements.saveTrainingButton.addEventListener("click", startTrainingRecord);

  elements.undoButton.addEventListener("click", () => {
    if (viewState.trainingRecording) {
      viewState.message = "\u8bad\u7ec3\u8bb0\u5f55\u5df2\u542f\u52a8\uff0c\u672c\u5c40\u4e0d\u5141\u8bb8\u6094\u68cb\u3002";
      namespace.ui.render(game, handlers, elements, viewState);
      return;
    }

    clearAiTimer();

    if (viewState.mode === "ai") {
      namespace.state.undoLastMove(game);
      if (game.moves.length > 0 && game.currentPlayer !== viewState.humanPlayer) {
        namespace.state.undoLastMove(game);
      }
    } else {
      namespace.state.undoLastMove(game);
    }

    viewState.boardLocked = false;
    viewState.message = "";
    viewState.trainingSaved = false;
    namespace.ui.render(game, handlers, elements, viewState);
    scheduleAiIfNeeded();
  });

  elements.restartButton.addEventListener("click", resetRound);

  viewState.message = TEXT.blackStarts;
  namespace.ui.render(game, handlers, elements, viewState);
})(window.Gomoku = window.Gomoku || {});
