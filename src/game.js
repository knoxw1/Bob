export const ROWS = 6;
export const COLS = 7;
export const BOARD_SIZE = ROWS * COLS;
export const STATE_SIZE = BOARD_SIZE * 2;

export function createBoard() {
  return new Array(BOARD_SIZE).fill(0);
}

export function playersForOpening(starter) {
  if (starter === 'human') return { human: 1, agent: 2 };
  if (starter === 'learner') return { human: 2, agent: 1 };
  throw new RangeError('Starter must be human or learner.');
}

export function getCell(board, row, col) {
  return board[row * COLS + col];
}

export function legalMoves(board) {
  const moves = [];
  for (let col = 0; col < COLS; col += 1) {
    if (getCell(board, ROWS - 1, col) === 0) moves.push(col);
  }
  return moves;
}

export function drop(board, col, player) {
  if (!Number.isInteger(col) || col < 0 || col >= COLS || getCell(board, ROWS - 1, col) !== 0) return -1;
  for (let row = 0; row < ROWS; row += 1) {
    const index = row * COLS + col;
    if (board[index] === 0) {
      board[index] = player;
      return row;
    }
  }
  return -1;
}

export function isWinningMove(board, row, col, player) {
  const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
  for (const [rowStep, colStep] of directions) {
    let length = 1;
    for (const sign of [-1, 1]) {
      let nextRow = row + rowStep * sign;
      let nextCol = col + colStep * sign;
      while (
        nextRow >= 0 && nextRow < ROWS && nextCol >= 0 && nextCol < COLS
        && getCell(board, nextRow, nextCol) === player
      ) {
        length += 1;
        nextRow += rowStep * sign;
        nextCol += colStep * sign;
      }
    }
    if (length >= 4) return true;
  }
  return false;
}

export function winningMove(board, player) {
  for (const col of legalMoves(board)) {
    const row = drop(board, col, player);
    const wins = isWinningMove(board, row, col, player);
    board[row * COLS + col] = 0;
    if (wins) return col;
  }
  return -1;
}

export function encodeState(board, player) {
  const state = new Float32Array(STATE_SIZE);
  const opponent = player === 1 ? 2 : 1;
  for (let index = 0; index < BOARD_SIZE; index += 1) {
    if (board[index] === player) state[index] = 1;
    else if (board[index] === opponent) state[BOARD_SIZE + index] = 1;
  }
  return state;
}

export function mirrorState(state) {
  const mirrored = new Float32Array(STATE_SIZE);
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const index = row * COLS + col;
      const reflected = row * COLS + (COLS - col - 1);
      mirrored[reflected] = state[index];
      mirrored[BOARD_SIZE + reflected] = state[BOARD_SIZE + index];
    }
  }
  return mirrored;
}