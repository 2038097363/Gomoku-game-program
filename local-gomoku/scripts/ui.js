(function initUi(namespace) {
  const { BOARD_SIZE, EMPTY, BLACK, WHITE, STAR_POINTS, TEXT } = namespace.config;
  const totalCells = BOARD_SIZE * BOARD_SIZE;

  function format(template, data) {
    return template.replace(/\{(\w+)\}/g, (_, key) => data[key]);
  }

  function playerName(player) {
    return player === BLACK ? TEXT.black : TEXT.white;
  }

  function playerClass(player) {
    return player === BLACK ? "black" : "white";
  }

  function createElements() {
    return {
      board: document.querySelector("#board"),
      twoPlayerModeButton: document.querySelector("#twoPlayerModeBtn"),
      aiModeButton: document.querySelector("#aiModeBtn"),
      sidePicker: document.querySelector("#sidePicker"),
      humanBlackButton: document.querySelector("#humanBlackBtn"),
      humanWhiteButton: document.querySelector("#humanWhiteBtn"),
      aiModelPicker: document.querySelector("#aiModelPicker"),
      heuristicAiButton: document.querySelector("#heuristicAiBtn"),
      searchAiButton: document.querySelector("#searchAiBtn"),
      serverSearchAiButton: document.querySelector("#serverSearchAiBtn"),
      serverCnnAiButton: document.querySelector("#serverCnnAiBtn"),
      serverHybridAiButton: document.querySelector("#serverHybridAiBtn"),
      statusDot: document.querySelector("#statusDot"),
      headlineStatus: document.querySelector("#headlineStatus"),
      turnStone: document.querySelector("#turnStone"),
      turnText: document.querySelector("#turnText"),
      gameMessage: document.querySelector("#gameMessage"),
      moveCount: document.querySelector("#moveCount"),
      emptyCount: document.querySelector("#emptyCount"),
      blackScore: document.querySelector("#blackScore"),
      whiteScore: document.querySelector("#whiteScore"),
      undoButton: document.querySelector("#undoBtn"),
      restartButton: document.querySelector("#restartBtn"),
      saveTrainingButton: document.querySelector("#saveTrainingBtn"),
    };
  }

  function render(game, handlers, elements, viewState) {
    renderBoard(game, handlers, elements.board, viewState);
    renderPanel(game, elements, viewState);
    renderSettings(elements, viewState);
  }

  function renderBoard(game, handlers, boardElement, viewState) {
    const lastMove = game.moves[game.moves.length - 1];
    const winningKeys = new Set(game.winningLine.map((point) => `${point.row}-${point.col}`));
    const locked = Boolean(viewState && viewState.boardLocked);

    boardElement.innerHTML = "";

    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const cell = document.createElement("button");
        const value = game.board[row][col];
        const isEmpty = value === EMPTY;

        cell.type = "button";
        cell.className = `cell ${isEmpty && !locked ? "empty" : ""} ${game.gameOver || locked ? "disabled" : ""}`;
        cell.setAttribute("role", "gridcell");
        cell.setAttribute("aria-label", format(TEXT.cellLabel, { row: row + 1, col: col + 1 }));
        cell.disabled = game.gameOver || locked || !isEmpty;
        cell.addEventListener("click", () => handlers.onCellClick(row, col));

        if (!isEmpty) {
          cell.appendChild(createStone(value, row, col, lastMove, winningKeys));
        }

        boardElement.appendChild(cell);
      }
    }

    renderStars(boardElement);
  }

  function createStone(value, row, col, lastMove, winningKeys) {
    const stone = document.createElement("span");
    const isLastMove = lastMove && lastMove.row === row && lastMove.col === col;
    const isWinning = winningKeys.has(`${row}-${col}`);

    stone.className = [
      "stone",
      playerClass(value),
      isLastMove ? "last" : "",
      isWinning ? "winning" : "",
    ].join(" ");

    return stone;
  }

  function renderStars(boardElement) {
    for (const [row, col] of STAR_POINTS) {
      const star = document.createElement("span");
      star.className = "star";
      star.style.left = `${((col + 0.5) / BOARD_SIZE) * 100}%`;
      star.style.top = `${((row + 0.5) / BOARD_SIZE) * 100}%`;
      boardElement.appendChild(star);
    }
  }

  function renderPanel(game, elements, viewState) {
    const currentClass = playerClass(game.currentPlayer);
    const currentName = playerName(game.currentPlayer);
    const emptyCount = totalCells - game.moves.length;

    elements.statusDot.className = `status-dot ${currentClass}`;
    elements.turnStone.className = `mini-stone ${currentClass}`;
    elements.turnText.textContent = currentName;
    elements.moveCount.textContent = String(game.moves.length);
    elements.emptyCount.textContent = String(emptyCount);
    elements.blackScore.textContent = String(game.scores[BLACK]);
    elements.whiteScore.textContent = String(game.scores[WHITE]);
    elements.undoButton.disabled = game.moves.length === 0 || Boolean(viewState && viewState.trainingRecording);
    elements.saveTrainingButton.disabled = (
      game.moves.length === 0
      || Boolean(viewState && viewState.boardLocked)
      || Boolean(viewState && viewState.trainingRecording)
    );
    elements.saveTrainingButton.textContent = trainingButtonText(viewState);

    if (game.gameOver && game.winner) {
      const winnerName = playerName(game.winner);
      elements.headlineStatus.textContent = format(TEXT.winHeadline, { player: winnerName });
      elements.gameMessage.textContent = format(TEXT.winMessage, { player: winnerName });
      return;
    }

    if (game.gameOver) {
      elements.headlineStatus.textContent = TEXT.draw;
      elements.gameMessage.textContent = TEXT.drawMessage;
      return;
    }

    elements.headlineStatus.textContent = format(TEXT.turnHeadline, { player: currentName });
    elements.gameMessage.textContent = viewState && viewState.message
      ? viewState.message
      : format(TEXT.movePrompt, { player: currentName });
  }

  function trainingButtonText(viewState) {
    if (viewState && viewState.trainingSaved) {
      return "\u5df2\u7eb3\u5165\u8bad\u7ec3";
    }
    if (viewState && viewState.trainingRecording) {
      return "\u8bb0\u5f55\u4e2d...";
    }
    return "\u7eb3\u5165\u8bad\u7ec3";
  }

  function renderSettings(elements, viewState) {
    if (!viewState) {
      return;
    }

    elements.twoPlayerModeButton.classList.toggle("active", viewState.mode === "two-player");
    elements.aiModeButton.classList.toggle("active", viewState.mode === "ai");
    elements.sidePicker.classList.toggle("hidden", viewState.mode !== "ai");
    elements.aiModelPicker.classList.toggle("hidden", viewState.mode !== "ai");
    elements.humanBlackButton.classList.toggle("active", viewState.humanPlayer === BLACK);
    elements.humanWhiteButton.classList.toggle("active", viewState.humanPlayer === WHITE);
    elements.heuristicAiButton.classList.toggle("active", viewState.aiModel === "heuristic");
    elements.searchAiButton.classList.toggle("active", viewState.aiModel === "search");
    elements.serverSearchAiButton.classList.toggle("active", viewState.aiModel === "server-search");
    elements.serverCnnAiButton.classList.toggle("active", viewState.aiModel === "server-cnn");
    elements.serverHybridAiButton.classList.toggle("active", viewState.aiModel === "server-hybrid");
  }

  namespace.ui = {
    createElements,
    render,
  };
})(window.Gomoku = window.Gomoku || {});
