(function initConfig(namespace) {
  namespace.config = {
    BOARD_SIZE: 15,
    EMPTY: 0,
    BLACK: 1,
    WHITE: 2,
    DIRECTIONS: [
      [0, 1],
      [1, 0],
      [1, 1],
      [1, -1],
    ],
    STAR_POINTS: [
      [3, 3],
      [3, 7],
      [3, 11],
      [7, 3],
      [7, 7],
      [7, 11],
      [11, 3],
      [11, 7],
      [11, 11],
    ],
    TEXT: {
      black: "\u9ed1\u68cb",
      white: "\u767d\u68cb",
      blackStarts: "\u9ed1\u68cb\u5148\u624b",
      draw: "\u672c\u5c40\u548c\u68cb",
      drawMessage: "\u68cb\u76d8\u5df2\u7ecf\u4e0b\u6ee1\uff0c\u6ca1\u6709\u73a9\u5bb6\u8fde\u6210\u4e94\u5b50\u3002",
      movePrompt: "\u8f6e\u5230{player}\u843d\u5b50\u3002",
      turnHeadline: "{player}\u56de\u5408",
      winHeadline: "{player}\u83b7\u80dc",
      winMessage: "{player}\u8fde\u6210\u4e94\u5b50\uff0c\u672c\u5c40\u7ed3\u675f\u3002\u53ef\u4ee5\u6094\u68cb\u590d\u76d8\uff0c\u6216\u91cd\u65b0\u5f00\u59cb\u4e0b\u4e00\u5c40\u3002",
      cellLabel: "\u7b2c{row}\u884c\u7b2c{col}\u5217",
    },
  };
})(window.Gomoku = window.Gomoku || {});
