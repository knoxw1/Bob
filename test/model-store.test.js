import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CHESS_MODEL_WEIGHT_SHAPES, createModelStore, MODEL_WEIGHT_SHAPES } from '../src/model-store.js';
import { decodeModel, encodeModel } from '../src/model-binary.js';

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
  const modelPath = join(directory, 'data', 'model.bin');
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
  const initial = await fetch(url);
  assert.equal(initial.status, 204);

  const modelData = emptyModel();
  const saved = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: encodeModel(modelData),
  });
  assert.equal(saved.status, 200);

  const loadedResponse = await fetch(url);
  assert.equal(loadedResponse.status, 200);
  const loaded = decodeModel(new Uint8Array(await loadedResponse.arrayBuffer()));
  assert.deepEqual(loaded, modelData);
  assert.deepEqual(decodeModel(await readFile(modelPath)), modelData);

  const invalid = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: new Uint8Array([1, 2, 3, 4]),
  });
  assert.equal(invalid.status, 400);

  const deleted = await fetch(url, { method: 'DELETE' });
  assert.equal(deleted.status, 200);
  assert.equal((await fetch(url)).status, 204);
});