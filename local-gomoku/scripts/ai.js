(function initAi(namespace) {
  const { BOARD_SIZE, EMPTY, BLACK, WHITE, DIRECTIONS } = namespace.config;
  const WIN_SCORE = 10_000_000;

  function chooseHeuristicMove(game, player) {
    const candidates = getCandidateMoves(game, 2);
    const rival = opponent(player);
    let bestScore = -Infinity;
    let bestMoves = [];

    for (const [row, col] of candidates) {
      const ownScore = evaluateMove(game.board, row, col, player);
      const blockScore = evaluateMove(game.board, row, col, rival);
      const score = ownScore + blockScore * 0.92 + centerBonus(row, col);

      if (score > bestScore) {
        bestScore = score;
        bestMoves = [[row, col]];
      } else if (score === bestScore) {
        bestMoves.push([row, col]);
      }
    }

    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  function chooseSearchMove(game, player, options = {}) {
    const depth = options.depth || 2;
    const width = options.width || 10;
    const radius = options.radius || 2;
    const tactical = chooseTacticalMove(game, player, radius);

    if (tactical) {
      return tactical;
    }

    const candidates = orderedCandidates(game, player, radius, width);
    let bestScore = -Infinity;
    let bestMoves = [];

    for (const [, row, col] of candidates) {
      const child = cloneGame(game);
      simulateMove(child, row, col);
      const score = minimax(child, player, depth - 1, -WIN_SCORE, WIN_SCORE, width, radius);

      if (score > bestScore) {
        bestScore = score;
        bestMoves = [[row, col]];
      } else if (score === bestScore) {
        bestMoves.push([row, col]);
      }
    }

    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
  }

  function minimax(game, rootPlayer, depth, alpha, beta, width, radius) {
    if (game.gameOver) {
      if (game.winner === rootPlayer) return WIN_SCORE - game.moves.length;
      if (!game.winner) return 0;
      return -WIN_SCORE + game.moves.length;
    }

    if (depth <= 0) {
      return evaluatePosition(game, rootPlayer);
    }

    const current = game.currentPlayer;
    const maximizing = current === rootPlayer;
    const tactical = chooseTacticalMove(game, current, radius);
    const candidates = tactical ? [[0, tactical[0], tactical[1]]] : orderedCandidates(game, current, radius, width);

    if (maximizing) {
      let value = -WIN_SCORE;
      for (const [, row, col] of candidates) {
        const child = cloneGame(game);
        simulateMove(child, row, col);
        value = Math.max(value, minimax(child, rootPlayer, depth - 1, alpha, beta, width, radius));
        alpha = Math.max(alpha, value);
        if (alpha >= beta) break;
      }
      return value;
    }

    let value = WIN_SCORE;
    for (const [, row, col] of candidates) {
      const child = cloneGame(game);
      simulateMove(child, row, col);
      value = Math.min(value, minimax(child, rootPlayer, depth - 1, alpha, beta, width, radius));
      beta = Math.min(beta, value);
      if (alpha >= beta) break;
    }
    return value;
  }

  function chooseTacticalMove(game, player, radius) {
    const rival = opponent(player);
    const candidates = getCandidateMoves(game, radius);

    for (const targetPlayer of [player, rival]) {
      const winning = [];
      for (const [row, col] of candidates) {
        const child = cloneGame(game);
        child.currentPlayer = targetPlayer;
        simulateMove(child, row, col);
        if (child.winner === targetPlayer) {
          const score = evaluateMove(game.board, row, col, player) + evaluateMove(game.board, row, col, rival);
          winning.push([score, row, col]);
        }
      }

      if (winning.length > 0) {
        return bestScoredMove(winning);
      }
    }

    const forced = [];
    for (const [row, col] of candidates) {
      const own = evaluateMove(game.board, row, col, player);
      const block = evaluateMove(game.board, row, col, rival);
      if (own >= 120_000 || block >= 120_000) {
        forced.push([own + block, row, col]);
      }
    }

    return forced.length > 0 ? bestScoredMove(forced) : null;
  }

  function orderedCandidates(game, player, radius, width) {
    const rival = opponent(player);
    const scored = getCandidateMoves(game, radius).map(([row, col]) => {
      const score =
        evaluateMove(game.board, row, col, player) +
        evaluateMove(game.board, row, col, rival) * 0.95 +
        centerBonus(row, col);
      return [score, row, col];
    });

    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, width);
  }

  function evaluatePosition(game, player) {
    const rival = opponent(player);
    let playerScore = 0;
    let rivalScore = 0;

    for (const [row, col] of getCandidateMoves(game, 2)) {
      playerScore += evaluateMove(game.board, row, col, player);
      rivalScore += evaluateMove(game.board, row, col, rival);
    }

    return playerScore - rivalScore * 1.05;
  }

  function cloneGame(game) {
    return {
      board: game.board.map((row) => row.slice()),
      currentPlayer: game.currentPlayer,
      moves: game.moves.map((move) => ({ ...move })),
      winner: game.winner,
      winningLine: game.winningLine ? game.winningLine.map((point) => ({ ...point })) : [],
      gameOver: game.gameOver,
    };
  }

  function simulateMove(game, row, col) {
    if (game.gameOver || game.board[row][col] !== EMPTY) {
      return false;
    }

    const player = game.currentPlayer;
    game.board[row][col] = player;
    game.moves.push({ row, col, player });

    const line = getWinningLine(game.board, row, col, player);
    if (line.length >= 5) {
      game.gameOver = true;
      game.winner = player;
      game.winningLine = line;
    } else if (game.moves.length === BOARD_SIZE * BOARD_SIZE) {
      game.gameOver = true;
      game.winner = null;
    } else {
      game.currentPlayer = opponent(player);
    }

    return true;
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

  function bestScoredMove(scored) {
    const bestScore = Math.max(...scored.map(([score]) => score));
    const best = scored.filter(([score]) => score === bestScore);
    const [, row, col] = best[Math.floor(Math.random() * best.length)];
    return [row, col];
  }

  function getCandidateMoves(game, radius) {
    if (game.moves.length === 0) {
      const center = Math.floor(BOARD_SIZE / 2);
      return [[center, center]];
    }

    const candidates = new Map();
    for (const move of game.moves) {
      for (let row = move.row - radius; row <= move.row + radius; row += 1) {
        for (let col = move.col - radius; col <= move.col + radius; col += 1) {
          if (isOpenPoint(game.board, row, col)) {
            candidates.set(`${row}-${col}`, [row, col]);
          }
        }
      }
    }

    return [...candidates.values()];
  }

  function evaluateMove(board, row, col, player) {
    let score = 0;

    for (const [dr, dc] of DIRECTIONS) {
      const left = countLine(board, row, col, -dr, -dc, player);
      const right = countLine(board, row, col, dr, dc, player);
      const total = left.count + 1 + right.count;
      const openEnds = Number(left.open) + Number(right.open);
      score += shapeScore(total, openEnds);
    }

    return score;
  }

  function countLine(board, row, col, dr, dc, player) {
    let count = 0;
    let nextRow = row + dr;
    let nextCol = col + dc;

    while (isInside(nextRow, nextCol) && board[nextRow][nextCol] === player) {
      count += 1;
      nextRow += dr;
      nextCol += dc;
    }

    return {
      count,
      open: isOpenPoint(board, nextRow, nextCol),
    };
  }

  function shapeScore(total, openEnds) {
    if (total >= 5) return 1_000_000;
    if (total === 4 && openEnds === 2) return 120_000;
    if (total === 4 && openEnds === 1) return 30_000;
    if (total === 3 && openEnds === 2) return 6_000;
    if (total === 3 && openEnds === 1) return 1_200;
    if (total === 2 && openEnds === 2) return 600;
    if (total === 2 && openEnds === 1) return 120;
    if (total === 1 && openEnds === 2) return 20;
    return 1;
  }

  function centerBonus(row, col) {
    const center = (BOARD_SIZE - 1) / 2;
    const distance = Math.abs(row - center) + Math.abs(col - center);
    return (BOARD_SIZE - distance) * 2;
  }

  function opponent(player) {
    return player === BLACK ? WHITE : BLACK;
  }

  function isInside(row, col) {
    return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
  }

  function isOpenPoint(board, row, col) {
    return isInside(row, col) && board[row][col] === EMPTY;
  }

  namespace.ai = {
    chooseHeuristicMove,
    chooseSearchMove,
  };
})(window.Gomoku = window.Gomoku || {});
