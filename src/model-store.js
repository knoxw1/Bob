import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import {
  CHESS_MODEL_WEIGHT_SHAPES,
  MODEL_WEIGHT_SHAPES,
  decodeModel,
  encodeModel,
  validateModelData,
} from './model-binary.js';

export { CHESS_MODEL_WEIGHT_SHAPES, MODEL_WEIGHT_SHAPES, validateModelData };
const uploads = new Map();
const MAX_BODY_BYTES = 256_000_000;
const MAX_CHUNK_BYTES = 4_000_000;

setInterval(() => {
  const cutoff = Date.now() - 5 * 60 * 1000;
  for (const [id, state] of uploads) {
    if (state.last < cutoff) uploads.delete(id);
  }
}, 60 * 1000).unref?.();

export function createModelStore(filePath) {
  async function readModel() {
    try {
      const buffer = await readFile(filePath);
      return decodeModel(buffer);
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
      await writeFile(temporaryPath, encodeModel(validModel), { mode: 0o600, flag: 'wx' });
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
    res.setHeader('Cache-Control', 'no-store');

    if (req.method === 'GET') {
      try {
        const modelData = await readModel();
        if (!modelData) {
          res.statusCode = 204;
          res.end();
          return;
        }
        const bytes = encodeModel(modelData);
        res.setHeader('Content-Type', 'application/octet-stream');
        res.statusCode = 200;
        res.end(bytes);
      } catch {
        res.statusCode = 500;
        res.end();
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
          const modelData = decodeModel(payloadBuffer);
          await writeModel(modelData);
          res.statusCode = 200;
          res.end(JSON.stringify({ saved: true }));
        } catch (error) {
          res.statusCode = 400;
          res.end(JSON.stringify({ error: error.message || 'Invalid model data.' }));
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
        const modelData = decodeModel(payloadBuffer);
        await writeModel(modelData);
        res.statusCode = 200;
        res.end(JSON.stringify({ saved: true }));
      } catch (error) {
        res.statusCode = 400;
        res.end(JSON.stringify({ error: error.message || 'Invalid model data.' }));
      }
    });
  };
}