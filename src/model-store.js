import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { gunzipSync } from 'node:zlib';

export const MODEL_WEIGHT_SHAPES = [[84, 512], [512], [512, 512], [512], [512, 7], [7]];
export const CHESS_MODEL_WEIGHT_SHAPES = [[836, 512], [512], [512, 512], [512], [512, 4864], [4864]];
const MAX_BODY_BYTES = 256_000_000;
const MAX_CHUNK_BYTES = 4_000_000;
const uploads = new Map();

setInterval(() => {
  const cutoff = Date.now() - 5 * 60 * 1000;
  for (const [id, state] of uploads) {
    if (state.last < cutoff) uploads.delete(id);
  }
}, 60 * 1000).unref?.();

function validateWeights(weights, shapes, label) {
  if (!Array.isArray(weights) || weights.length !== shapes.length) {
    throw new Error(`${label} neural network weight count does not match this model.`);
  }

  for (let index = 0; index < shapes.length; index += 1) {
    const weight = weights[index];
    const expectedShape = shapes[index];
    const expectedSize = expectedShape.reduce((product, dimension) => product * dimension, 1);
    if (
      !weight || !Array.isArray(weight.shape)
      || weight.shape.length !== expectedShape.length
      || weight.shape.some((dimension, axis) => dimension !== expectedShape[axis])
      || !Array.isArray(weight.values) || weight.values.length !== expectedSize
      || weight.values.some((value) => typeof value !== 'number' || !Number.isFinite(value))
    ) {
      throw new Error(`${label} neural network layer ${index + 1} has invalid weights.`);
    }
  }
}

function validateChessModel(chess) {
  if (!chess || chess.version !== 1) throw new Error('Invalid chess model data.');
  validateWeights(chess.weights, CHESS_MODEL_WEIGHT_SHAPES, 'Chess');
  for (const field of ['trainingGames', 'observedGames']) {
    if (chess[field] !== undefined && (!Number.isSafeInteger(chess[field]) || chess[field] < 0)) {
      throw new Error(`Invalid chess ${field} value.`);
    }
  }
  return {
    version: 1,
    trainingGames: chess.trainingGames || 0,
    observedGames: chess.observedGames || 0,
    weights: chess.weights,
  };
}

export function validateModelData(modelData) {
  if (!modelData || modelData.version !== 1 || !Array.isArray(modelData.weights)) {
    throw new Error('Invalid neural network model data.');
  }
  validateWeights(modelData.weights, MODEL_WEIGHT_SHAPES, 'Connect Four');

  for (const field of ['trainingGames', 'observedGames']) {
    if (modelData[field] !== undefined && (!Number.isSafeInteger(modelData[field]) || modelData[field] < 0)) {
      throw new Error(`Invalid ${field} value.`);
    }
  }

  const validated = {
    version: 1,
    trainingGames: modelData.trainingGames || 0,
    observedGames: modelData.observedGames || 0,
    weights: modelData.weights,
  };
  if (modelData.chess !== undefined) validated.chess = validateChessModel(modelData.chess);
  return validated;
}

export function createModelStore(filePath) {
  async function readModel() {
    try {
      return validateModelData(JSON.parse(await readFile(filePath, 'utf8')));
    } catch (error) {
      if (error.code === 'ENOENT') return null;
      // Treat unreadable or outdated/incompatible model files as "no saved model"
      // so the app can start fresh instead of failing to load.
      return null;
    }
  }

  async function writeModel(modelData) {
    const validModel = validateModelData(modelData);
    const temporaryPath = `${filePath}.${randomUUID()}.tmp`;
    try {
      await mkdir(dirname(filePath), { recursive: true });
      await writeFile(temporaryPath, JSON.stringify(validModel), { mode: 0o600, flag: 'wx' });
      await rename(temporaryPath, filePath);
    } catch (error) {
      await rm(temporaryPath, { force: true }).catch(() => {});
      throw error;
    }
    return validModel;
  }

  return async function modelStoreMiddleware(req, res, next) {
    const requestPath = (() => {
      try { return new URL(req.url || '/', 'http://localhost').pathname; }
      catch { return (req.url || '').split('?')[0]; }
    })();
    if (requestPath !== '/api/model') {
      next();
      return;
    }
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');

    if (req.method === 'GET') {
      try {
        res.statusCode = 200;
        res.end(JSON.stringify({ modelData: await readModel() }));
      } catch {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: 'The saved model file is invalid or unreadable.' }));
      }
      return;
    }

    if (req.method === 'DELETE') {
      try {
        await rm(filePath, { force: true });
        res.statusCode = 200;
        res.end(JSON.stringify({ deleted: true }));
      } catch {
        res.statusCode = 500;
        res.end(JSON.stringify({ error: 'Could not delete the workspace model file.' }));
      }
      return;
    }

    if (req.method === 'OPTIONS') {
      res.setHeader('Allow', 'GET, PUT, DELETE, OPTIONS');
      res.statusCode = 204;
      res.end();
      return;
    }
    if (req.method !== 'PUT') {
      res.setHeader('Allow', 'GET, PUT, DELETE, OPTIONS');
      res.statusCode = 405;
      res.end(JSON.stringify({ error: 'Use GET or PUT for the local model.' }));
      return;
    }

    const urlObj = new URL(req.url || '/', 'http://localhost');
    const uploadId = urlObj.searchParams.get('upload');
    const chunkIndex = Number(urlObj.searchParams.get('index'));
    const chunkTotal = Number(urlObj.searchParams.get('total'));

    const isChunked = Boolean(uploadId)
      && Number.isInteger(chunkIndex) && chunkIndex >= 0
      && Number.isInteger(chunkTotal) && chunkTotal > 0;

    if (isChunked) {
      const chunks = [];
      let tooLarge = false;
      req.on('data', (chunk) => {
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        if (chunks.reduce((size, item) => size + item.length, 0) + buffer.length > MAX_CHUNK_BYTES) {
          tooLarge = true;
          chunks.length = 0;
        } else if (!tooLarge) {
          chunks.push(buffer);
        }
      });
      req.on('end', async () => {
        if (tooLarge) {
          res.statusCode = 413;
          res.end(JSON.stringify({ error: 'Chunk is too large.' }));
          return;
        }
        const body = Buffer.concat(chunks);
        let state = uploads.get(uploadId);
        if (!state) {
          state = {
            total: chunkTotal,
            chunks: new Array(chunkTotal).fill(null),
            received: 0,
            last: Date.now(),
          };
          uploads.set(uploadId, state);
        }
        if (state.total !== chunkTotal || chunkIndex >= chunkTotal) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: 'Invalid chunk metadata.' }));
          return;
        }
        if (state.chunks[chunkIndex] === null) state.received += 1;
        state.chunks[chunkIndex] = body;
        state.last = Date.now();

        if (state.chunks.some((chunk) => chunk === null)) {
          res.statusCode = 202;
          res.end(JSON.stringify({ received: state.received, total: chunkTotal }));
          return;
        }

        uploads.delete(uploadId);
        const useGzip = urlObj.searchParams.get('gzip') === '1';
        let payloadBuffer = Buffer.concat(state.chunks);
        if (useGzip) {
          try { payloadBuffer = gunzipSync(payloadBuffer); }
          catch (error) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: 'Model data must be valid gzip.' }));
            return;
          }
        }
        try {
          const { modelData } = JSON.parse(payloadBuffer.toString('utf8') || '{}');
          await writeModel(modelData);
          res.statusCode = 200;
          res.end(JSON.stringify({ saved: true }));
        } catch (error) {
          res.statusCode = error instanceof SyntaxError ? 400 : 422;
          res.end(JSON.stringify({ error: error instanceof SyntaxError ? 'Model data must be valid JSON.' : error.message }));
        }
      });
      return;
    }

    const bodyChunks = [];
    let tooLarge = false;
    req.on('data', (chunk) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      const currentSize = bodyChunks.reduce((size, item) => size + item.length, 0);
      if (currentSize + buffer.length > MAX_BODY_BYTES) {
        tooLarge = true;
        bodyChunks.length = 0;
      } else if (!tooLarge) {
        bodyChunks.push(buffer);
      }
    });
    req.on('end', async () => {
      if (tooLarge) {
        res.statusCode = 413;
        res.end(JSON.stringify({ error: 'Model data is too large.' }));
        return;
      }
      const useGzip = urlObj.searchParams.get('gzip') === '1';
      let payloadBuffer = Buffer.concat(bodyChunks);
      if (useGzip) {
        try { payloadBuffer = gunzipSync(payloadBuffer); }
        catch (error) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: 'Model data must be valid gzip.' }));
          return;
        }
      }
      try {
        const { modelData } = JSON.parse(payloadBuffer.toString('utf8') || '{}');
        await writeModel(modelData);
        res.statusCode = 200;
        res.end(JSON.stringify({ saved: true }));
      } catch (error) {
        res.statusCode = error instanceof SyntaxError ? 400 : 422;
        res.end(JSON.stringify({ error: error instanceof SyntaxError ? 'Model data must be valid JSON.' : error.message }));
      }
    });
  };
}