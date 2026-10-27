import * as tf from '@tensorflow/tfjs';
import { Chess } from 'chess.js';

export const CHESS_STATE_SIZE = 836;
export const CHESS_ACTION_COUNT = 4864;
const PIECE_PLANES = 12 * 64;
const CASTLING_OFFSET = PIECE_PLANES;
const EN_PASSANT_OFFSET = CASTLING_OFFSET + 4;
const PROMOTION_OFFSET = 4096;
const PROMOTIONS = ['q', 'r', 'b', 'n'];
const REPLAY_LIMIT = 2500;
const BATCH_SIZE = 8;
const GAMMA = 0.97;

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [CHESS_STATE_SIZE], units: 512, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.dense({ units: CHESS_ACTION_COUNT, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.0008), loss: 'meanSquaredError' });
  return model;
}

function copyWeights(from, to) {
  const weights = from.getWeights().map((weight) => weight.clone());
  to.setWeights(weights);
  weights.forEach((weight) => weight.dispose());
}

export function chessSquareIndex(square) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return rank * 8 + file;
}

export function chessMoveIndex(move) {
  const from = chessSquareIndex(move.from);
  const to = chessSquareIndex(move.to);
  if (!move.promotion) return from * 64 + to;
  const direction = move.to.charCodeAt(0) - move.from.charCodeAt(0) + 1;
  const promotion = PROMOTIONS.indexOf(move.promotion);
  return PROMOTION_OFFSET + from * 12 + direction * 4 + promotion;
}

export function encodeChessState(game) {
  const state = new Float32Array(CHESS_STATE_SIZE);
  const board = game.board();
  const currentColor = game.turn();
  const pieceTypes = ['p', 'n', 'b', 'r', 'q', 'k'];

  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const piece = board[row][col];
      if (!piece) continue;
      const rank = 7 - row;
      const square = rank * 8 + col;
      const relativeColor = piece.color === currentColor ? 0 : 6;
      const plane = pieceTypes.indexOf(piece.type) + relativeColor;
      state[plane * 64 + square] = 1;
    }
  }

  const ownCastling = game.getCastlingRights(currentColor);
  const otherCastling = game.getCastlingRights(currentColor === 'w' ? 'b' : 'w');
  state[CASTLING_OFFSET] = Number(ownCastling.k);
  state[CASTLING_OFFSET + 1] = Number(ownCastling.q);
  state[CASTLING_OFFSET + 2] = Number(otherCastling.k);
  state[CASTLING_OFFSET + 3] = Number(otherCastling.q);

  const enPassantSquare = game.fen().split(' ')[3];
  if (enPassantSquare !== '-') state[EN_PASSANT_OFFSET + chessSquareIndex(enPassantSquare)] = 1;
  return state;
}

function validateChessModel(savedModel) {
  if (!savedModel || savedModel.version !== 1 || !Array.isArray(savedModel.weights) || savedModel.weights.length !== 6) return false;
  const expectedShapes = [[836, 512], [512], [512, 512], [512], [512, CHESS_ACTION_COUNT], [CHESS_ACTION_COUNT]];
  return savedModel.weights.every((weight, index) => {
    const shape = expectedShapes[index];
    return weight.shape.length === shape.length
      && weight.shape.every((dimension, axis) => dimension === shape[axis])
      && weight.values.length === shape.reduce((size, dimension) => size * dimension, 1);
  });
}

const PIECE_VALUES = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };

function chessEvaluate(game) {
  let score = 0;
  const board = game.board();
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const piece = board[row][col];
      if (!piece) continue;
      const value = PIECE_VALUES[piece.type] || 0;
      const centerBonus = (3.5 - Math.abs(col - 3.5)) + (3.5 - Math.abs(row - 3.5));
      const signed = piece.color === 'w' ? value : -value;
      score += signed + (piece.type === 'p' || piece.type === 'n' ? (piece.color === 'w' ? centerBonus * 2 : -centerBonus * 2) : 0);
    }
  }
  return score;
}

function chessMinimax(game, depth, alpha, beta, maximizing) {
  if (game.isCheckmate()) return maximizing ? -100000 : 100000;
  if (game.isStalemate() || game.isDraw()) return 0;
  if (depth === 0) return chessEvaluate(game);
  const moves = game.moves({ verbose: true });
  moves.sort((left, right) => (right.captured ? 1 : 0) - (left.captured ? 1 : 0));
  if (maximizing) {
    let best = -Infinity;
    for (const move of moves) {
      game.move(move);
      const score = chessMinimax(game, depth - 1, alpha, beta, false);
      game.undo();
      best = Math.max(best, score);
      alpha = Math.max(alpha, score);
      if (beta <= alpha) break;
    }
    return best;
  }
  let best = Infinity;
  for (const move of moves) {
    game.move(move);
    const score = chessMinimax(game, depth - 1, alpha, beta, true);
    game.undo();
    best = Math.min(best, score);
    beta = Math.min(beta, score);
    if (beta <= alpha) break;
  }
  return best;
}

export class ChessAgent {
  constructor() {
    this.model = createModel();
    this.target = createModel();
    copyWeights(this.model, this.target);
    this.frozenModel = createModel();
    copyWeights(this.model, this.frozenModel);
    this.replay = [];
    this.replayCursor = 0;
    this.trainingGames = 0;
    this.observedGames = 0;
    this.trainingSteps = 0;
  }

  static async create(savedModel = null) {
    await tf.setBackend('cpu');
    await tf.ready();
    const agent = new ChessAgent();
    if (validateChessModel(savedModel)) {
      const tensors = savedModel.weights.map(({ shape, values }) => tf.tensor(values, shape));
      try {
        agent.model.setWeights(tensors);
        copyWeights(agent.model, agent.target);
        copyWeights(agent.model, agent.frozenModel);
        agent.trainingGames = Number(savedModel.trainingGames) || 0;
        agent.observedGames = Number(savedModel.observedGames) || 0;
      } finally {
        tensors.forEach((tensor) => tensor.dispose());
      }
    }
    return agent;
  }

  predict(state) {
    return tf.tidy(() => Array.from(this.model.predict(tf.tensor2d(state, [1, CHESS_STATE_SIZE])).dataSync()));
  }

  predictFrozen(state) {
    return tf.tidy(() => Array.from(this.frozenModel.predict(tf.tensor2d(state, [1, CHESS_STATE_SIZE])).dataSync()));
  }

  getVisualization(game) {
    const state = encodeChessState(game);
    const inputIndices = Array.from({ length: 96 }, (_, index) => Math.floor(index * CHESS_STATE_SIZE / 96));
    const layerWeights = this.model.layers.map((layer) => layer.getWeights().map((tensor) => tensor.clone()));
    const kernels = layerWeights.map(([kernel]) => kernel.dataSync());
    const biases = layerWeights.map(([, bias]) => bias.dataSync());
    layerWeights.flat().forEach((tensor) => tensor.dispose());
    const forward = (values, kernel, bias, inputSize, outputSize, activate) => {
      const output = new Float32Array(outputSize);
      for (let unit = 0; unit < outputSize; unit += 1) {
        let sum = bias[unit];
        for (let input = 0; input < inputSize; input += 1) sum += values[input] * kernel[input * outputSize + unit];
        output[unit] = activate ? Math.max(0, sum) : sum;
      }
      return output;
    };
    const hiddenOne = forward(state, kernels[0], biases[0], CHESS_STATE_SIZE, 512, true);
    const hiddenTwo = forward(hiddenOne, kernels[1], biases[1], 512, 512, true);
    const values = this.predict(state);
    const outputs = game.moves({ verbose: true })
      .map((move) => ({ value: values[chessMoveIndex(move)], label: move.san, action: chessMoveIndex(move) }))
      .sort((left, right) => right.value - left.value);
    const makeEdges = (fromIndices, toIndices, weightAt, outputSize, topCount) => fromIndices.flatMap((source, from) =>
      toIndices.map((target, to) => ({ from, to, weight: weightAt(source, target) }))
        .sort((left, right) => Math.abs(right.weight) - Math.abs(left.weight))
        .slice(0, topCount),
    );

    return {
      layerLabels: ['CHESS / 836', 'HIDDEN / 512', 'HIDDEN / 512', `LEGAL / ${outputs.length}`],
      inputs: inputIndices.map((index) => ({ index, label: `F${index + 1}`, value: state[index] })),
      hiddenOne: Array.from(hiddenOne),
      hiddenTwo: Array.from(hiddenTwo),
      outputs,
      edges: {
        inputHidden: makeEdges(inputIndices, Array.from({ length: 512 }, (_, index) => index),
          (source, target) => kernels[0][source * 512 + target], 512, 5),
        hiddenHidden: makeEdges(Array.from({ length: 512 }, (_, index) => index), Array.from({ length: 512 }, (_, index) => index),
          (source, target) => kernels[1][source * 512 + target], 512, 5),
        hiddenOutput: makeEdges(Array.from({ length: 512 }, (_, index) => index), outputs,
          (source, target) => kernels[2][source * CHESS_ACTION_COUNT + target.action], outputs.length, 1),
      },
    };
  }

  chooseMove(game, epsilon = 0, random = Math.random, useFrozen = false) {
    const moves = game.moves({ verbose: true });
    if (!moves.length) return null;
    if (random() < epsilon) return moves[Math.floor(random() * moves.length)];

    if (epsilon <= 0.15) {
      const maximizing = game.turn() === 'w';
      let bestMoves = [moves[0]];
      const startTime = performance.now();
      const timeLimitMs = 1500;
      for (let extraDepth = 2; extraDepth <= 6; extraDepth += 1) {
        if (performance.now() - startTime > timeLimitMs) break;
        let currentBestScore = maximizing ? -Infinity : Infinity;
        let currentBestMoves = [];
        let aborted = false;
        for (const move of moves) {
          if (performance.now() - startTime > timeLimitMs) { aborted = true; break; }
          game.move(move);
          const score = chessMinimax(game, extraDepth, -Infinity, Infinity, !maximizing);
          game.undo();
          if (maximizing) {
            if (score > currentBestScore + 1e-7) { currentBestScore = score; currentBestMoves = [move]; }
            else if (Math.abs(score - currentBestScore) <= 1e-7) currentBestMoves.push(move);
          } else {
            if (score < currentBestScore - 1e-7) { currentBestScore = score; currentBestMoves = [move]; }
            else if (Math.abs(score - currentBestScore) <= 1e-7) currentBestMoves.push(move);
          }
        }
        if (aborted) break;
        if (currentBestMoves.length) bestMoves = currentBestMoves;
      }
      return bestMoves[Math.floor(random() * bestMoves.length)];
    }

    const values = useFrozen ? this.predictFrozen(encodeChessState(game)) : this.predict(encodeChessState(game));
    let bestValue = -Infinity;
    let bestMoves = [];
    for (const move of moves) {
      const value = values[chessMoveIndex(move)];
      if (value > bestValue + 1e-7) {
        bestValue = value;
        bestMoves = [move];
      } else if (Math.abs(value - bestValue) <= 1e-7) {
        bestMoves.push(move);
      }
    }
    return bestMoves[Math.floor(random() * bestMoves.length)];
  }

  remember(experience) {
    if (this.replay.length < REPLAY_LIMIT) this.replay.push(experience);
    else {
      this.replay[this.replayCursor] = experience;
      this.replayCursor = (this.replayCursor + 1) % REPLAY_LIMIT;
    }
  }

  async trainBatch(requestedBatchSize = BATCH_SIZE) {
    if (!this.replay.length) return null;
    const batchSize = Math.min(requestedBatchSize, this.replay.length);
    const batch = Array.from({ length: batchSize }, () => this.replay[Math.floor(Math.random() * this.replay.length)]);
    const states = tf.tensor2d(batch.map((sample) => Array.from(sample.state)), [batchSize, CHESS_STATE_SIZE]);
    const nextStates = tf.tensor2d(batch.map((sample) => Array.from(sample.nextState)), [batchSize, CHESS_STATE_SIZE]);
    const currentTensor = this.model.predict(states);
    const futureTensor = this.target.predict(nextStates);
    const targets = currentTensor.arraySync();
    const future = futureTensor.arraySync();

    for (let index = 0; index < batch.length; index += 1) {
      const sample = batch[index];
      const bestFuture = sample.done || !sample.nextLegal.length
        ? 0
        : Math.max(...sample.nextLegal.map((action) => future[index][action]));
      targets[index][sample.action] = sample.reward + (sample.done ? 0 : -GAMMA * bestFuture);
    }

    const targetTensor = tf.tensor2d(targets, [batchSize, CHESS_ACTION_COUNT]);
    let loss;
    try {
      const result = await this.model.trainOnBatch(states, targetTensor);
      loss = Array.isArray(result) ? result[0] : result;
    } finally {
      states.dispose();
      nextStates.dispose();
      currentTensor.dispose();
      futureTensor.dispose();
      targetTensor.dispose();
    }

    this.trainingSteps += 1;
    if (this.trainingSteps % 80 === 0) copyWeights(this.model, this.target);
    return Number(loss);
  }

  async finishExperienceGame(history, winner, train = true) {
    for (let index = 0; index < history.length; index += 1) {
      const move = history[index];
      const outcome = winner === null ? 0 : winner === move.color ? 1 : -1;
      this.remember({
        action: move.action,
        done: true,
        nextLegal: [],
        nextState: new Float32Array(CHESS_STATE_SIZE),
        reward: outcome * (GAMMA ** (history.length - index - 1)),
        state: Float32Array.from(move.state),
      });
    }

    if (!train || !history.length) return null;
    let totalLoss = 0;
    const batches = Math.min(10, Math.max(2, Math.ceil(history.length / 16)));
    for (let batch = 0; batch < batches; batch += 1) {
      const loss = await this.trainBatch();
      if (Number.isFinite(loss)) totalLoss += loss;
    }
    return totalLoss / batches;
  }

  async trainSelfPlay(games, onProgress = () => {}, seed = 1) {
    let randomState = seed >>> 0 || 1;
    const random = () => {
      randomState = (randomState * 1664525 + 1013904223) >>> 0;
      return randomState / 0x100000000;
    };

    for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
      const useChallenger = gameIndex % 3 === 0;
      const game = new Chess();
      const history = [];
      let winner = null;
      let loss = null;
      for (let ply = 0; ply < 180 && !game.isGameOver(); ply += 1) {
        const color = game.turn();
        const isCurrentPlayer = color === 'w';
        const state = encodeChessState(game);
        const epsilon = Math.max(0.05, 0.25 - (this.trainingGames / 25000) * 0.2);
        const move = (!useChallenger || isCurrentPlayer)
          ? this.chooseMove(game, epsilon, random)
          : this.chooseMove(game, epsilon, random, true);
        if (!move) break;
        const action = chessMoveIndex(move);
        game.move(move);
        const mate = game.isCheckmate();
        if (!useChallenger || isCurrentPlayer) history.push({ action, color, state });
        if (mate) winner = game.turn() === 'w' ? 'b' : 'w';
        if (ply % 8 === 7 && history.length >= BATCH_SIZE) await this.trainBatch();
      }
      loss = await this.finishExperienceGame(history, winner);
      this.trainingGames += 1;
      if (gameIndex % 5 === 4) {
        copyWeights(this.model, this.frozenModel);
        const weights = this.frozenModel.getWeights().map((weight) => weight.clone());
        const mutated = weights.map((weight) => {
          const values = Array.from(weight.dataSync());
          const noisy = values.map((value) => value + (Math.random() - 0.5) * 0.005);
          const tensor = tf.tensor(noisy, weight.shape);
          weight.dispose();
          return tensor;
        });
        this.frozenModel.setWeights(mutated);
        mutated.forEach((tensor) => tensor.dispose());
      }
      const update = {
        completed: gameIndex + 1,
        games,
        loss,
        replaySize: this.replay.length,
        trainingGames: this.trainingGames,
        observedGames: this.observedGames,
        winner,
        visualization: this.getVisualization(game),
      };
      if ((gameIndex + 1) % 5 === 0 || gameIndex + 1 === games) update.modelData = await this.exportModel();
      await onProgress(update);
    }
  }

  async learnFromGame(history, winner, onProgress = () => {}) {
    if (!history.length) return;
    this.observedGames += 1;
    const loss = await this.finishExperienceGame(history, winner);
    await onProgress({
      completed: 1,
      games: 1,
      loss,
      replaySize: this.replay.length,
      trainingGames: this.trainingGames,
      observedGames: this.observedGames,
      visualization: this.getVisualization(new Chess()),
    });
  }

  async learnFromChessComGames(games, onProgress = () => {}) {
    const list = Array.isArray(games) ? games : games?.games || [];
    if (!list.length) return;
    this.observedGames += list.length;
    let totalLoss = 0;
    let completed = 0;
    for (const entry of list) {
      const pgn = typeof entry === 'string' ? entry : entry.pgn;
      const game = new Chess();
      if (pgn) {
        try { game.loadPgn(pgn); } catch { continue; }
      }
      const sans = game.history();
      const replayGame = new Chess();
      const history = [];
      for (const san of sans) {
        const state = encodeChessState(replayGame);
        const color = replayGame.turn();
        const verbose = replayGame.moves({ verbose: true }).find((move) => move.san === san);
        if (!verbose) break;
        const action = chessMoveIndex(verbose);
        replayGame.move(san);
        if (action !== null) history.push({ action, color, state });
        if (replayGame.isGameOver()) break;
      }
      const result = pgn?.match(/\[Result "([^"]+)"\]/)?.[1] || entry.result || '*';
      const winner = result === '1-0' ? 'w' : result === '0-1' ? 'b' : null;
      const loss = await this.finishExperienceGame(history, winner);
      if (Number.isFinite(loss)) totalLoss += loss;
      completed += 1;
      await onProgress({
        completed,
        games: list.length,
        loss: Number.isFinite(loss) ? loss : null,
        replaySize: this.replay.length,
        trainingGames: this.trainingGames,
        observedGames: this.observedGames,
        visualization: this.getVisualization(new Chess()),
      });
    }
    return totalLoss / Math.max(1, completed);
  }

  async exportModel() {
    const weights = this.model.getWeights().map((tensor) => tensor.clone());
    const savedWeights = await Promise.all(weights.map(async (weight) => {
      const values = Array.from(await weight.data());
      const shape = [...weight.shape];
      weight.dispose();
      return {
        shape,
        values,
      };
    }));
    return {
      version: 1,
      trainingGames: this.trainingGames,
      observedGames: this.observedGames,
      weights: savedWeights,
    };
  }

  dispose() {
    this.model.dispose();
    this.target.dispose();
    this.frozenModel.dispose();
  }
}
