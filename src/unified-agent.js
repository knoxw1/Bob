import * as tf from '@tensorflow/tfjs';
import { Chess } from 'chess.js';
import { COLS, STATE_SIZE as CF_STATE_SIZE, createBoard, drop, encodeState, isWinningMove, legalMoves, mirrorState } from './game.js';
import { chessMoveIndex, encodeChessState, CHESS_STATE_SIZE } from './chess-agent.js';

export const TOTAL_INPUT_SIZE = CF_STATE_SIZE + CHESS_STATE_SIZE + 2;
export const TOTAL_OUTPUT_SIZE = COLS + 4864;
export const CF_GAME_INDEX = CF_STATE_SIZE + CHESS_STATE_SIZE;
export const CHESS_GAME_INDEX = CF_GAME_INDEX + 1;

const REPLAY_LIMIT = 8000;
const BATCH_SIZE = 32;
const GAMMA = 0.98;

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [TOTAL_INPUT_SIZE], units: 512, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.dense({ units: TOTAL_OUTPUT_SIZE, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
  return model;
}

function copyWeights(from, to) {
  const weights = from.getWeights().map((weight) => weight.clone());
  to.setWeights(weights);
  weights.forEach((weight) => weight.dispose());
}

export class UnifiedAgent {
  constructor(model = createModel(), targetModel = null) {
    this.model = model;
    this.targetModel = targetModel || createModel();
    copyWeights(this.model, this.targetModel);
    this.replay = [];
    this.steps = 0;
  }

  static async create(savedModel = null) {
    const agent = new UnifiedAgent();
    if (savedModel) await agent.loadModel(savedModel);
    return agent;
  }

  encodeConnectFour(board, player) {
    const out = new Float32Array(TOTAL_INPUT_SIZE);
    const cf = encodeState(board, player);
    out.set(cf, 0);
    out[CF_GAME_INDEX] = 1;
    return out;
  }

  encodeChess(game) {
    const out = new Float32Array(TOTAL_INPUT_SIZE);
    const chess = encodeChessState(game);
    out.set(chess, CF_STATE_SIZE);
    out[CHESS_GAME_INDEX] = 1;
    return out;
  }

  async predictConnectFour(board, player) {
    const state = this.encodeConnectFour(board, player);
    const tensor = tf.tensor2d([Array.from(state)], [1, TOTAL_INPUT_SIZE]);
    const output = this.model.predict(tensor);
    const values = await output.data();
    tensor.dispose();
    output.dispose();
    return Array.from(values).slice(0, COLS);
  }

  async predictChess(game) {
    const state = this.encodeChess(game);
    const tensor = tf.tensor2d([Array.from(state)], [1, TOTAL_INPUT_SIZE]);
    const output = this.model.predict(tensor);
    const values = await output.data();
    tensor.dispose();
    output.dispose();
    return Array.from(values).slice(COLS);
  }

  chooseConnectFourMove(board, player, epsilon = 0.1, random = Math.random) {
    const moves = legalMoves(board);
    if (!moves.length) return null;
    if (random() < epsilon) return moves[Math.floor(random() * moves.length)];
    // sync prediction not possible; callers should use async variant
    return moves[0];
  }

  async chooseConnectFourMoveAsync(board, player, epsilon = 0.1, random = Math.random) {
    const moves = legalMoves(board);
    if (!moves.length) return null;
    if (random() < epsilon) return moves[Math.floor(random() * moves.length)];
    const values = await this.predictConnectFour(board, player);
    let best = moves[0], bestVal = -Infinity;
    for (const m of moves) if (values[m] > bestVal) { bestVal = values[m]; best = m; }
    return best;
  }

  chooseChessMove(game, epsilon = 0.1, random = Math.random) {
    const moves = game.moves({ verbose: true });
    if (!moves.length) return null;
    if (random() < epsilon) return moves[Math.floor(random() * moves.length)];
    return moves[0];
  }

  async chooseChessMoveAsync(game, epsilon = 0.1, random = Math.random) {
    const moves = game.moves({ verbose: true });
    if (!moves.length) return null;
    if (random() < epsilon) return moves[Math.floor(random() * moves.length)];
    const values = await this.predictChess(game);
    let bestMove = moves[0], bestValue = -Infinity;
    for (const move of moves) {
      const index = chessMoveIndex(move);
      if (index !== null && values[index] > bestValue) { bestValue = values[index]; bestMove = move; }
    }
    return bestMove;
  }

  exportModel() {
    const weights = this.model.getWeights().map((w) => Array.from(w.dataSync()));
    const shapes = this.model.getWeights().map((w) => Array.from(w.shape));
    return { unified: { weights, shapes }, format: 'unified-v1' };
  }

  async loadModel(modelData) {
    if (!modelData?.unified) return;
    const tensors = modelData.unified.weights.map((w, i) => tf.tensor(w, modelData.unified.shapes[i]));
    this.model.setWeights(tensors);
    copyWeights(this.model, this.targetModel);
    tensors.forEach((t) => t.dispose());
  }

  dispose() {
    this.model.dispose();
    this.targetModel.dispose();
  }

  getConnectFourVisualization(state) {
    const layers = [
      { label: 'BOARD', count: TOTAL_INPUT_SIZE },
      { label: 'HIDDEN 1', count: 512 },
      { label: 'HIDDEN 2', count: 512 },
      { label: 'MOVES', count: TOTAL_OUTPUT_SIZE }
    ];
    return { layers, activeInputs: state ? Array.from(state) : null, mode: 'weights' };
  }

  getChessVisualization(game) {
    const state = game ? this.encodeChess(game) : null;
    return this.getConnectFourVisualization(state);
  }
}
