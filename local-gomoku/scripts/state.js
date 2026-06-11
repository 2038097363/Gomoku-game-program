(function initState(namespace) {
  const { BOARD_SIZE, EMPTY, BLACK, WHITE } = namespace.config;

  function createEmptyBoard() {
    return Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(EMPTY));
  }

  function createGameState() {
    return {
      board: createEmptyBoard(),
      currentPlayer: BLACK,
      moves: [],
      gameOver: false,
      winner: null,
      winningLine: [],
      scoredWinner: null,
      scores: {
        [BLACK]: 0,
        [WHITE]: 0,
      },
    };
  }

  function resetRound(game) {
    game.board = createEmptyBoard();
    game.currentPlayer = BLACK;
    game.moves = [];
    game.gameOver = false;
    game.winner = null;
    game.winningLine = [];
    game.scoredWinner = null;
  }

  function switchPlayer(player) {
    return player === BLACK ? WHITE : BLACK;
  }

  function placeStone(game, row, col) {
    game.board[row][col] = game.currentPlayer;
    game.moves.push({ row, col, player: game.currentPlayer });
  }

  function undoLastMove(game) {
    const lastMove = game.moves.pop();

    if (!lastMove) {
      return null;
    }

    if (game.gameOver && game.winner && game.scoredWinner === game.winner) {
      game.scores[game.winner] = Math.max(0, game.scores[game.winner] - 1);
    }

    game.board[lastMove.row][lastMove.col] = EMPTY;
    game.currentPlayer = lastMove.player;
    game.gameOver = false;
    game.winner = null;
    game.winningLine = [];
    game.scoredWinner = null;

    return lastMove;
  }

  namespace.state = {
    createGameState,
    resetRound,
    switchPlayer,
    placeStone,
    undoLastMove,
  };
})(window.Gomoku = window.Gomoku || {});
