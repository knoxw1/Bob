import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { ConnectFourAgent } from '../src/agent.js';
import { COLS, createBoard, drop, encodeState, isWinningMove, legalMoves, playersForOpening, winningMove } from '../src/game.js';

test('either player can open while roles map to distinct board pieces', () => {
  assert.deepEqual(playersForOpening('human'), { human: 1, agent: 2 });
  assert.deepEqual(playersForOpening('learner'), { human: 2, agent: 1 });
  assert.throws(() => playersForOpening('nobody'), /Starter must be human or learner/);
});

test('pieces fall to the lowest empty row and full columns reject moves', () => {
  const board = createBoard();
  assert.equal(drop(board, 4, 1), 0);
  assert.equal(drop(board, 4, 2), 1);
  for (let row = 2; row < 6; row += 1) assert.equal(drop(board, 4, 1), row);
  assert.equal(drop(board, 4, 2), -1);
  assert.ok(!legalMoves(board).includes(4));
});

test('win detection covers horizontal, vertical, and both diagonals', () => {
  const lines = [
    [[0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 2], [1, 2], [2, 2], [3, 2]],
    [[0, 0], [1, 1], [2, 2], [3, 3]],
    [[3, 0], [2, 1], [1, 2], [0, 3]],
  ];
  for (const line of lines) {
    const board = createBoard();
    for (const [row, col] of line) board[row * COLS + col] = 1;
    const [row, col] = line.at(-1);
    assert.ok(isWinningMove(board, row, col, 1));
  }
});

test('winningMove finds a playable winning column without changing the board', () => {
  const board = createBoard();
  for (const col of [0, 1, 2]) drop(board, col, 1);
  const before = board.slice();
  assert.equal(winningMove(board, 1), 3);
  assert.deepEqual(board, before);
});

test('state encoding is a fixed-size perspective-aware neural input', () => {
  const board = createBoard();
  drop(board, 3, 1);
  drop(board, 2, 2);
  const state = encodeState(board, 1);
  assert.equal(state.length, 84);
  assert.equal(state[3], 1);
  assert.equal(state[42 + 2], 1);
});

test('neural network trains locally and exports reloadable weights', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = await ConnectFourAgent.create();
  const before = await agent.exportModel();
  const updates = [];
  await agent.trainSelfPlay(12, (update) => updates.push(update), 77);
  const after = await agent.exportModel();
  const changed = before.weights.some((weight, index) => weight.values.some((value, offset) => value !== after.weights[index].values[offset]));
  assert.ok(changed, 'training should update neural network weights');
  assert.equal(after.trainingGames, 12);
  assert.equal(updates.length, 12, 'the network UI should receive a snapshot for every self-play game');
  assert.ok(updates.some((update) => update.visualization && update.modelData), 'final training progress should include a live snapshot and checkpoint');
  const board = createBoard();
  const move = agent.chooseAction(board, 2);
  assert.ok(legalMoves(board).includes(move));
  assert.ok(agent.predict(encodeState(board, 2)).every(Number.isFinite));

  const visualization = agent.getVisualization(encodeState(board, 2));
  assert.equal(visualization.inputs.length, 84);
  assert.equal(visualization.hiddenOne.length, 512);
  assert.equal(visualization.hiddenTwo.length, 512);
  assert.equal(visualization.outputs.length, 7);
  assert.equal(visualization.edges.inputHidden.length, 420);
  assert.equal(visualization.edges.hiddenHidden.length, 2560);

  const beforeMatch = await agent.exportModel();
  await agent.learnFromGame([
    { state: Array.from(encodeState(board, 1)), action: 3, player: 1 },
    { state: Array.from(encodeState(board, 2)), action: 2, player: 2 },
  ], 1);
  const afterMatch = await agent.exportModel();
  assert.equal(afterMatch.observedGames, 1);
  assert.ok(beforeMatch.weights.some((weight, index) => weight.values.some((value, offset) => value !== afterMatch.weights[index].values[offset])));
  agent.dispose();
});

test('a fresh network updates from a short completed match', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = await ConnectFourAgent.create();
  try {
    const board = createBoard();
    const moves = [];
    for (let index = 0; index < 6; index += 1) {
      const player = index % 2 === 0 ? 1 : 2;
      const state = Array.from(encodeState(board, player));
      const action = index;
      moves.push({ state, action, player });
      drop(board, action, player);
    }
    const before = await agent.exportModel();
    await agent.learnFromGame(moves, 1);
    const after = await agent.exportModel();
    assert.equal(after.observedGames, 1);
    assert.ok(before.weights.some((weight, layer) => weight.values.some((value, index) => value !== after.weights[layer].values[index])));
  } finally {
    agent.dispose();
  }
});