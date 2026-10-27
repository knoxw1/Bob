import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CHESS_MODEL_WEIGHT_SHAPES, createModelStore, MODEL_WEIGHT_SHAPES } from '../src/model-store.js';

function emptyModel() {
  return {
    version: 1,
    trainingGames: 12,
    observedGames: 3,
    weights: MODEL_WEIGHT_SHAPES.map((shape) => ({
      shape,
      values: new Array(shape.reduce((size, dimension) => size * dimension, 1)).fill(0),
    })),
    chess: {
      version: 1,
      trainingGames: 2,
      observedGames: 1,
      weights: CHESS_MODEL_WEIGHT_SHAPES.map((shape) => ({
        shape,
        values: new Array(shape.reduce((size, dimension) => size * dimension, 1)).fill(0),
      })),
    },
  };
}

test('local model endpoint loads and atomically stores network weights on disk', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect-four-model-'));
  const modelPath = join(directory, 'data', 'model.json');
  const middleware = createModelStore(modelPath);
  const server = createServer((req, res) => middleware(req, res, () => {
    res.writeHead(404).end();
  }));

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => {
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
    await rm(directory, { recursive: true, force: true });
  });

  const url = `http://127.0.0.1:${server.address().port}/api/model`;
  assert.deepEqual(await fetch(url).then((response) => response.json()), { modelData: null });

  const modelData = emptyModel();
  const saved = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelData }),
  });
  assert.equal(saved.status, 200);
  assert.deepEqual(await fetch(url).then((response) => response.json()), { modelData });
  assert.deepEqual(JSON.parse(await readFile(modelPath, 'utf8')), modelData);

  const invalid = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ modelData: { ...modelData, weights: [] } }),
  });
  assert.equal(invalid.status, 422);

  const deleted = await fetch(url, { method: 'DELETE' });
  assert.equal(deleted.status, 200);
  assert.deepEqual(await fetch(url).then((response) => response.json()), { modelData: null });
});