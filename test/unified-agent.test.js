import test from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import {
  UnifiedAgent,
  TOTAL_INPUT_SIZE,
  TOTAL_OUTPUT_SIZE,
  CF_GAME_INDEX,
  CHESS_GAME_INDEX,
} from '../src/unified-agent.js';
import { createBoard, drop } from '../src/game.js';
import { Chess } from 'chess.js';

test('unified agent encodes both games into one fixed input vector', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = new UnifiedAgent();
  try {
    const board = createBoard();
    drop(board, 3, 1);
    const cfState = agent.encodeConnectFour(board, 1);
    assert.equal(cfState.length, TOTAL_INPUT_SIZE);
    assert.equal(cfState[CF_GAME_INDEX], 1);

    const game = new Chess();
    const chessState = agent.encodeChess(game);
    assert.equal(chessState.length, TOTAL_INPUT_SIZE);
    assert.equal(chessState[CHESS_GAME_INDEX], 1);
  } finally {
    agent.dispose();
  }
});

test('unified predictions are finite for both games', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = new UnifiedAgent();
  try {
    const board = createBoard();
    const cfValues = await agent.predictConnectFour(board, 1);
    assert.equal(cfValues.length, 7);
    assert.ok(cfValues.every(Number.isFinite));

    const game = new Chess();
    const chessValues = await agent.predictChess(game);
    assert.equal(chessValues.length, 4864);
    assert.ok(chessValues.every(Number.isFinite));
  } finally {
    agent.dispose();
  }
});

test('unified model export and reload preserves its tensor shapes', async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  const agent = new UnifiedAgent();
  try {
    const exported = agent.exportModel();
    assert.equal(exported.format, 'unified-v1');
    assert.equal(exported.unified.weights.length, agent.model.getWeights().length);
    await agent.loadModel(exported);
    const reloaded = agent.exportModel();
    assert.deepEqual(reloaded.unified.shapes, exported.unified.shapes);
  } finally {
    agent.dispose();
  }
});