(function initRules(namespace) {
  const { BOARD_SIZE, EMPTY, DIRECTIONS } = namespace.config;

  function isInside(row, col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
  }

  function canPlace(game, row, col) {
    return !game.gameOver && isInside(row, col) && game.board[row][col] === EMPTY;
  }

  function collectDirection(board, row, col, dr, dc, player) {
    const line = [];
    let nextRow = row + dr;
    let nextCol = col + dc;

    while (isInside(nextRow, nextCol) && board[nextRow][nextCol] === player) {
      line.push({ row: nextRow, col: nextCol });
      nextRow += dr;
      nextCol += dc;
    }

    return line;
  }

  function getWinningLine(board, row, col, player) {
    for (const [dr, dc] of DIRECTIONS) {
      const line = [{ row, col }];
      line.push(...collectDirection(board, row, col, dr, dc, player));
      line.unshift(...collectDirection(board, row, col, -dr, -dc, player).reverse());

      if (line.length >= 5) {
        return line;
      }
    }

    return [];
  }

  namespace.rules = {
    canPlace,
    getWinningLine,
  };
})(window.Gomoku = window.Gomoku || {});
