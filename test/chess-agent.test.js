import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { Chess } from 'chess.js';
import { ChessAgent, CHESS_ACTION_COUNT, CHESS_STATE_SIZE, chessMoveIndex, encodeChessState } from '../src/chess-agent.js';

test('chess encoding includes board, castling, and en-passant state', () => {
  const game = new Chess();
  const state = encodeChessState(game);
  assert.equal(state.length, CHESS_STATE_SIZE);
  assert.equal(state[8], 1);
  assert.equal(state[6 * 64 + 48], 1);
  assert.equal(state[768], 1);
  assert.equal(state[769], 1);
  assert.equal(state[770], 1);
  assert.equal(state[771], 1);
  assert.equal(state[772 + 20], 0);
});

test('chess action encoding stays inside the neural output head', () => {
  const game = new Chess();
  for (const move of game.moves({ verbose: true })) {
    const action = chessMoveIndex(move);
    assert.ok(Number.isInteger(action));
    assert.ok(action >= 0 && action < CHESS_ACTION_COUNT);
  }
  assert.ok(chessMoveIndex({ from: 'a7', to: 'a8', promotion: 'q' }) >= 4096);
});

test('all four legal promotion choices map to distinct AI actions', () => {
  const game = new Chess('7k/P7/6K1/8/8/8/8/8 w - - 0 1');
  const promotions = game.moves({ verbose: true }).filter((move) => move.from === 'a7' && move.to === 'a8');
  assert.deepEqual(promotions.map((move) => move.promotion).sort(), ['b', 'n', 'q', 'r']);
  assert.equal(new Set(promotions.map(chessMoveIndex)).size, 4);
  for (const move of promotions) assert.ok(chessMoveIndex(move) < CHESS_ACTION_COUNT);
});

test('chess neural policy selects a legal move and exposes its learned head', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = await ChessAgent.create();
  try {
    const game = new Chess();
    const move = agent.chooseMove(game);
    assert.ok(game.moves().includes(move.san));
    const visualization = agent.getVisualization(game);
    assert.equal(visualization.inputs.length, 96);
    assert.equal(visualization.hiddenOne.length, 512);
    assert.equal(visualization.outputs.length, game.moves().length);
  } finally {
    agent.dispose();
  }
});

test('chess match results update the separate chess policy weights', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = await ChessAgent.create();
  try {
    const game = new Chess();
    const history = [];
    for (const san of ['f3', 'e5', 'g4', 'Qh4#']) {
      const color = game.turn();
      const state = Array.from(encodeChessState(game));
      const move = game.moves({ verbose: true }).find((candidate) => candidate.san === san);
      history.push({ state, action: chessMoveIndex(move), color });
      game.move(move);
    }
    assert.ok(game.isCheckmate());
    const before = await agent.exportModel();
    await agent.learnFromGame(history, 'b');
    const after = await agent.exportModel();
    assert.equal(after.observedGames, 1);
    assert.ok(before.weights.some((weight, layer) => weight.values.some((value, index) => value !== after.weights[layer].values[index])));
  } finally {
    agent.dispose();
  }
});