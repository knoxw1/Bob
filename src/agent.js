import * as tf from '@tensorflow/tfjs';
import { COLS, STATE_SIZE, createBoard, drop, encodeState, isWinningMove, legalMoves, mirrorState, winningMove } from './game.js';

const REPLAY_LIMIT = 12000;
const BATCH_SIZE = 48;
const TARGET_SYNC_INTERVAL = 120;
const GAMMA = 0.98;

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [STATE_SIZE], units: 512, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.dense({ units: COLS, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
  return model;
}

function copyWeights(from, to) {
  const weights = from.getWeights().map((weight) => weight.clone());
  to.setWeights(weights);
  weights.forEach((weight) => weight.dispose());
}

function argmaxLegal(values, moves) {
  let bestValue = -Infinity;
  let bestMoves = [];
  for (const move of moves) {
    if (values[move] > bestValue + 1e-7) {
      bestValue = values[move];
      bestMoves = [move];
    } else if (Math.abs(values[move] - bestValue) <= 1e-7) {
      bestMoves.push(move);
    }
  }
  bestMoves.sort((left, right) => Math.abs(left - 3) - Math.abs(right - 3));
  const centerDistance = Math.abs(bestMoves[0] - 3);
  const central = bestMoves.filter((move) => Math.abs(move - 3) === centerDistance);
  return central[Math.floor(Math.random() * central.length)];
}

function hasFour(board, player) {
  for (let row = 0; row < 6; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      const index = row * COLS + col;
      if (board[index] !== player) continue;
      if (col + 3 < 7 && board[index + 1] === player && board[index + 2] === player && board[index + 3] === player) return true;
      if (row + 3 < 6 && board[index + COLS] === player && board[index + 2 * COLS] === player && board[index + 3 * COLS] === player) return true;
      if (row + 3 < 6 && col + 3 < 7 && board[index + COLS + 1] === player && board[index + 2 * COLS + 2] === player && board[index + 3 * COLS + 3] === player) return true;
      if (row + 3 < 6 && col - 3 >= 0 && board[index + COLS - 1] === player && board[index + 2 * COLS - 2] === player && board[index + 3 * COLS - 3] === player) return true;
    }
  }
  return false;
}

function evaluateConnectFour(board, aiPlayer) {
  const opponent = aiPlayer === 1 ? 2 : 1;
  if (hasFour(board, aiPlayer)) return 100000;
  if (hasFour(board, opponent)) return -100000;
  let score = 0;
  const centerCol = 3;
  for (let row = 0; row < 6; row += 1) if (board[row * COLS + centerCol] === aiPlayer) score += 3;
  const scoreWindow = (windowCells, player) => {
    let count = 0; let empty = 0; let opp = 0;
    for (const cell of windowCells) {
      if (cell === player) count += 1;
      else if (cell === 0) empty += 1;
      else opp += 1;
    }
    if (count === 3 && empty === 1) return 5;
    if (count === 2 && empty === 2) return 2;
    if (opp === 3 && empty === 1) return -4;
    return 0;
  };
  for (let row = 0; row < 6; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      if (col + 3 < 7) {
        const w = [board[row*COLS+col], board[row*COLS+col+1], board[row*COLS+col+2], board[row*COLS+col+3]];
        score += scoreWindow(w, aiPlayer);
      }
      if (row + 3 < 6) {
        const w = [board[row*COLS+col], board[(row+1)*COLS+col], board[(row+2)*COLS+col], board[(row+3)*COLS+col]];
        score += scoreWindow(w, aiPlayer);
      }
      if (row + 3 < 6 && col + 3 < 7) {
        const w = [board[row*COLS+col], board[(row+1)*COLS+col+1], board[(row+2)*COLS+col+2], board[(row+3)*COLS+col+3]];
        score += scoreWindow(w, aiPlayer);
      }
      if (row + 3 < 6 && col - 3 >= 0) {
        const w = [board[row*COLS+col], board[(row+1)*COLS+col-1], board[(row+2)*COLS+col-2], board[(row+3)*COLS+col-3]];
        score += scoreWindow(w, aiPlayer);
      }
    }
  }
  return score;
}

function connectFourMinimax(board, depth, alpha, beta, maximizing, aiPlayer) {
  const opponent = aiPlayer === 1 ? 2 : 1;
  if (hasFour(board, aiPlayer)) return 100000 + depth;
  if (hasFour(board, opponent)) return -100000 - depth;
  const moves = legalMoves(board);
  if (!moves.length) return 0;
  if (depth === 0) return evaluateConnectFour(board, aiPlayer);
  if (maximizing) {
    let value = -Infinity;
    for (const move of moves) {
      const row = drop(board, move, aiPlayer);
      const child = connectFourMinimax(board, depth - 1, alpha, beta, false, aiPlayer);
      board[row * COLS + move] = 0;
      value = Math.max(value, child);
      alpha = Math.max(alpha, value);
      if (alpha >= beta) break;
    }
    return value;
  }
  let value = Infinity;
  for (const move of moves) {
    const row = drop(board, move, opponent);
    const child = connectFourMinimax(board, depth - 1, alpha, beta, true, aiPlayer);
    board[row * COLS + move] = 0;
    value = Math.min(value, child);
    beta = Math.min(beta, value);
    if (alpha >= beta) break;
  }
  return value;
}

function mirrorEncodedState(encoded) {
  const output = new Float32Array(STATE_SIZE);
  for (let row = 0; row < 6; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      const source = row * 7 + col;
      const destination = row * 7 + (6 - col);
      output[destination] = encoded[source];
      output[STATE_SIZE / 2 + destination] = encoded[STATE_SIZE / 2 + source];
    }
  }
  return output;
}

export class ConnectFourAgent {
  constructor(model = createModel()) {
    this.model = model;
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
    const agent = new ConnectFourAgent();
    if (savedModel?.version === 1 && Array.isArray(savedModel.weights)) {
      const tensors = savedModel.weights.map(({ shape, values }) => tf.tensor(values, shape));
      try {
        agent.model.setWeights(tensors);
        copyWeights(agent.model, agent.target);
        copyWeights(agent.model, agent.frozenModel);
        agent.trainingGames = Number(savedModel.trainingGames) || 0;
        agent.observedGames = Number(savedModel.observedGames) || 0;
      } catch {
        agent.model.dispose();
        agent.target.dispose();
        agent.frozenModel.dispose();
        return new ConnectFourAgent();
      } finally {
        tensors.forEach((tensor) => tensor.dispose());
      }
    }
    return agent;
  }

  predict(state) {
    return tf.tidy(() => {
      const direct = Array.from(this.model.predict(tf.tensor2d(state, [1, STATE_SIZE])).dataSync());
      const reflectedState = mirrorState(state);
      const reflected = Array.from(this.model.predict(tf.tensor2d(reflectedState, [1, STATE_SIZE])).dataSync());
      return direct.map((value, column) => (value + reflected[COLS - column - 1]) / 2);
    });
  }

  predictFrozen(state) {
    return tf.tidy(() => Array.from(this.frozenModel.predict(tf.tensor2d(state, [1, STATE_SIZE])).dataSync()));
  }

  chooseActionWithFrozen(board, player, epsilon, random) {
    const moves = legalMoves(board);
    if (!moves.length) return -1;
    const win = winningMove(board, player);
    if (win !== -1) return win;
    const block = winningMove(board, player === 1 ? 2 : 1);
    if (block !== -1) return block;
    const safeMoves = moves.filter((move) => {
      const row = drop(board, move, player);
      const exposesWin = winningMove(board, player === 1 ? 2 : 1) !== -1;
      board[row * COLS + move] = 0;
      return !exposesWin;
    });
    const choices = safeMoves.length ? safeMoves : moves;
    if (random() < epsilon) return choices[Math.floor(random() * choices.length)];
    if (epsilon >= 0.25) {
      const values = this.predictFrozen(encodeState(board, player));
      return argmaxLegal(values, choices);
    }
    let bestMove = choices[0];
    let bestScore = -Infinity;
    for (const move of choices) {
      const row = drop(board, move, player);
      const score = connectFourMinimax(board, 5, -Infinity, Infinity, false, player);
      board[row * COLS + move] = 0;
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }
    return bestMove;
  }

  getVisualization(state = new Float32Array(STATE_SIZE)) {
    const inputIndices = Array.from({ length: STATE_SIZE }, (_, index) => index);
    const hiddenIndices = Array.from({ length: 512 }, (_, index) => index);
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
    const hiddenOne = forward(state, kernels[0], biases[0], STATE_SIZE, 512, true);
    const hiddenTwo = forward(hiddenOne, kernels[1], biases[1], 512, 512, true);
    const outputs = this.predict(state);
    const makeEdges = (fromIndices, toIndices, kernel, outputSize, topCount) => fromIndices.flatMap((from, source) => {
      const strongest = toIndices.map((to, target) => ({
        from: source,
        to: target,
        weight: kernel[from * outputSize + to],
      })).sort((left, right) => Math.abs(right.weight) - Math.abs(left.weight));
      return strongest.slice(0, topCount);
    });
    const inputs = inputIndices.map((index) => ({
      index,
      label: `${index >= 42 ? 'O' : 'Y'}${Math.floor((index % 42) / 7) + 1}.${(index % 7) + 1}`,
      value: state[index],
    }));
    const firstLayer = Array.from(hiddenOne);
    const secondLayer = Array.from(hiddenTwo);

    return {
      layerLabels: ['BOARD / 84', 'HIDDEN / 512', 'HIDDEN / 512', 'MOVES / 7'],
      inputs,
      hiddenOne: firstLayer,
      hiddenTwo: secondLayer,
      outputs: Array.from(outputs),
      edges: {
        inputHidden: makeEdges(inputIndices, hiddenIndices, kernels[0], 512, 5),
        hiddenHidden: makeEdges(hiddenIndices, hiddenIndices, kernels[1], 512, 5),
        hiddenOutput: makeEdges(hiddenIndices, Array.from({ length: COLS }, (_, index) => index), kernels[2], COLS, COLS),
      },
    };
  }

  chooseAction(board, player, epsilon = 0) {
    const moves = legalMoves(board);
    if (!moves.length) return -1;

    const win = winningMove(board, player);
    if (win !== -1) return win;
    const block = winningMove(board, player === 1 ? 2 : 1);
    if (block !== -1) return block;

    const safeMoves = moves.filter((move) => {
      const row = drop(board, move, player);
      const exposesWin = winningMove(board, player === 1 ? 2 : 1) !== -1;
      board[row * COLS + move] = 0;
      return !exposesWin;
    });
    const choices = safeMoves.length ? safeMoves : moves;
    if (Math.random() < epsilon) return choices[Math.floor(Math.random() * choices.length)];

    if (choices.length === 1) return choices[0];
    let bestMove = choices[0];
    let bestScore = -Infinity;
    for (const move of choices) {
      const row = drop(board, move, player);
      const score = connectFourMinimax(board, 5, -Infinity, Infinity, false, player);
      board[row * COLS + move] = 0;
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }
    return bestMove;
  }

  remember(experience) {
    if (this.replay.length < REPLAY_LIMIT) {
      this.replay.push(experience);
      return;
    }
    this.replay[this.replayCursor] = experience;
    this.replayCursor = (this.replayCursor + 1) % REPLAY_LIMIT;
  }

  async trainBatch(requestedBatchSize = BATCH_SIZE) {
    if (!this.replay.length) return null;
    const batchSize = Math.min(requestedBatchSize, this.replay.length);
    const batch = Array.from({ length: batchSize }, () => this.replay[Math.floor(Math.random() * this.replay.length)]);
    const states = tf.tensor2d(batch.map((sample) => Array.from(sample.state)), [batchSize, STATE_SIZE]);
    const nextStates = tf.tensor2d(batch.map((sample) => Array.from(sample.nextState)), [batchSize, STATE_SIZE]);
    const currentTensor = this.model.predict(states);
    const futureTensor = this.target.predict(nextStates);
    const onlineFutureTensor = this.model.predict(nextStates);
    const targets = currentTensor.arraySync();
    const future = futureTensor.arraySync();
    const onlineFuture = onlineFutureTensor.arraySync();

    for (let index = 0; index < batch.length; index += 1) {
      const sample = batch[index];
      let bestFuture = 0;
      if (!sample.done && sample.nextLegal.length) {
        let bestAction = sample.nextLegal[0];
        let bestValue = onlineFuture[index][bestAction];
        for (const move of sample.nextLegal.slice(1)) {
          if (onlineFuture[index][move] > bestValue + 1e-9) {
            bestValue = onlineFuture[index][move];
            bestAction = move;
          }
        }
        bestFuture = future[index][bestAction];
      }
      targets[index][sample.action] = sample.reward + (sample.done ? 0 : -GAMMA * bestFuture);
    }
    onlineFutureTensor.dispose();

    const targetTensor = tf.tensor2d(targets, [batchSize, COLS]);
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
    if (this.trainingSteps % TARGET_SYNC_INTERVAL === 0) copyWeights(this.model, this.target);
    return Number(loss);
  }

  async trainSelfPlay(games, onProgress = () => {}, seed = 1) {
    let randomState = seed >>> 0 || 1;
    const random = () => {
      randomState = (randomState * 1664525 + 1013904223) >>> 0;
      return randomState / 0x100000000;
    };
    let totalLoss = 0;
    let lossBatches = 0;
    let moveCount = 0;

    for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
      const useChallenger = gameIndex % 3 === 0;
      const board = createBoard();
      let player = 1;
      let done = false;
      let gameMoves = 0;

      while (!done) {
        const progress = Math.min(1, (this.trainingGames + gameIndex) / 30000);
        const epsilon = Math.max(0.05, 0.25 - progress * 0.2);
        const isCurrentPlayer = player === 1;
        const state = encodeState(board, player);
        const action = (!useChallenger || isCurrentPlayer)
          ? this.chooseActionWithRandom(board, player, epsilon, random)
          : this.chooseActionWithFrozen(board, player, epsilon, random);
        const row = drop(board, action, player);
        gameMoves += 1;
        moveCount += 1;
        const won = isWinningMove(board, row, action, player);
        const draw = !won && legalMoves(board).length === 0;
        done = won || draw;
        const nextPlayer = player === 1 ? 2 : 1;
        const nextState = done ? new Float32Array(STATE_SIZE) : encodeState(board, nextPlayer);
        const centerReward = 0.01 * (3 - Math.abs(action - 3));
        const reward = won ? 1 : draw ? 0 : centerReward;
        if (!useChallenger || isCurrentPlayer) {
          this.remember({
            action,
            done,
            nextLegal: done ? [] : legalMoves(board),
            nextState,
            reward,
            state,
          });
          const mirroredAction = COLS - 1 - action;
          const mirroredNextLegal = done ? [] : legalMoves(board).map((move) => COLS - 1 - move);
          const mirroredState = mirrorEncodedState(state);
          const mirroredNextState = done ? new Float32Array(STATE_SIZE) : mirrorEncodedState(nextState);
          this.remember({
            action: mirroredAction,
            done,
            nextLegal: mirroredNextLegal,
            nextState: mirroredNextState,
            reward,
            state: mirroredState,
          });
        }

        if (moveCount % 4 === 0 && this.replay.length >= BATCH_SIZE) {
          const loss = await this.trainBatch();
          if (Number.isFinite(loss)) {
            totalLoss += loss;
            lossBatches += 1;
          }
        }
        player = nextPlayer;
      }

      this.trainingGames += 1;
      if (gameIndex % 5 === 4) {
        copyWeights(this.model, this.frozenModel);
        const weights = this.frozenModel.getWeights().map((weight) => weight.clone());
        const mutated = weights.map((weight) => {
          const values = Array.from(weight.dataSync());
          const noisy = values.map((value) => value + (Math.random() - 0.5) * 0.01);
          const tensor = tf.tensor(noisy, weight.shape);
          weight.dispose();
          return tensor;
        });
        this.frozenModel.setWeights(mutated);
        mutated.forEach((tensor) => tensor.dispose());
      }
      {
        const progress = {
          completed: gameIndex + 1,
          games,
          loss: lossBatches ? totalLoss / lossBatches : null,
          replaySize: this.replay.length,
          trainingGames: this.trainingGames,
          observedGames: this.observedGames,
          visualization: this.getVisualization(encodeState(board, player)),
        };
        if ((gameIndex + 1) % 50 === 0 || gameIndex + 1 === games) {
          progress.modelData = await this.exportModel();
        }
        await onProgress(progress);
      }
    }

    if (this.replay.length >= BATCH_SIZE) {
      for (let batch = 0; batch < 8; batch += 1) await this.trainBatch();
    }
  }

  async learnFromGame(moves, winner, onProgress = () => {}) {
    if (!Array.isArray(moves) || moves.length === 0) return;
    this.observedGames += 1;
    for (let index = 0; index < moves.length; index += 1) {
      const move = moves[index];
      const outcome = winner === 0 ? 0 : winner === move.player ? 1 : -1;
      this.remember({
        action: move.action,
        done: true,
        nextLegal: [],
        nextState: new Float32Array(STATE_SIZE),
        reward: outcome * (GAMMA ** (moves.length - index - 1)),
        state: Float32Array.from(move.state),
      });
    }

    const batches = Math.min(12, Math.max(4, Math.ceil(moves.length / 6)));
    let totalLoss = 0;
    for (let batch = 0; batch < batches; batch += 1) {
      const loss = await this.trainBatch(Math.min(BATCH_SIZE, this.replay.length));
      if (Number.isFinite(loss)) totalLoss += loss;
      if (batch % 3 === 2 || batch + 1 === batches) {
        await onProgress({
          completed: batch + 1,
          games: batches,
          loss: totalLoss / (batch + 1),
          replaySize: this.replay.length,
          trainingGames: this.trainingGames,
          observedGames: this.observedGames,
          visualization: this.getVisualization(Float32Array.from(moves.at(-1).state)),
        });
      }
    }
  }

  chooseActionWithRandom(board, player, epsilon, random) {
    const moves = legalMoves(board);
    if (!moves.length) return -1;
    const win = winningMove(board, player);
    if (win !== -1) return win;
    const block = winningMove(board, player === 1 ? 2 : 1);
    if (block !== -1) return block;
    const safeMoves = moves.filter((move) => {
      const row = drop(board, move, player);
      const exposesWin = winningMove(board, player === 1 ? 2 : 1) !== -1;
      board[row * COLS + move] = 0;
      return !exposesWin;
    });
    const choices = safeMoves.length ? safeMoves : moves;
    if (random() < epsilon) return choices[Math.floor(random() * choices.length)];
    if (epsilon >= 0.25) {
      const values = this.predict(encodeState(board, player));
      return argmaxLegal(values, choices);
    }
    let bestMove = choices[0];
    let bestScore = -Infinity;
    for (const move of choices) {
      const row = drop(board, move, player);
      const score = connectFourMinimax(board, 5, -Infinity, Infinity, false, player);
      board[row * COLS + move] = 0;
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }
    return bestMove;
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