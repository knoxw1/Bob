import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createModelStore } from './src/model-store.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.join(__dirname, 'dist');
const dataDir = path.join(__dirname, 'data');
const modelPath = process.env.MODEL_PATH || path.join(__dirname, 'data', 'model.json');
const modelStore = createModelStore(modelPath);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webmanifest': 'application/manifest+json',
  '.ico': 'image/x-icon',
  '.map': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
};

async function sendFile(res, filePath) {
  const data = await fs.readFile(filePath);
  const type = MIME_TYPES[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
  res.setHeader('Content-Type', type);
  res.end(data);
}

async function serveStatic(req, res) {
  const url = new URL(req.url || '/', 'http://localhost');
  let pathname = '/';
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    pathname = url.pathname;
  }
  if (pathname === '/') pathname = '/index.html';

  let baseDir = distDir;
  let relative = pathname.replace(/^\/+/, '');
  if (pathname.startsWith('/data/')) {
    baseDir = dataDir;
    relative = relative.replace(/^data\//, '');
  }
  if (relative === '') relative = 'index.html';

  const filePath = path.resolve(baseDir, relative);
  if (filePath !== baseDir && !filePath.startsWith(baseDir + path.sep)) {
    res.statusCode = 403;
    res.end('Forbidden');
    return;
  }

  try {
    const stat = await fs.stat(filePath);
    if (stat.isDirectory()) {
      await sendFile(res, path.join(filePath, 'index.html'));
      return;
    }
    await sendFile(res, filePath);
    return;
  } catch {
    // Fall back to the SPA entry point for client-side routes.
    if (!pathname.startsWith('/data/')) {
      try {
        await sendFile(res, path.join(distDir, 'index.html'));
        return;
      } catch {
        // Fall through to 404.
      }
    }
    res.statusCode = 404;
    res.end('Not found');
  }
}

const server = http.createServer((req, res) => {
  modelStore(req, res, () => {
    if (req.method === 'GET' && req.url && new URL(req.url, 'http://localhost').pathname === '/healthz') {
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('ok');
      return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.statusCode = 405;
      res.setHeader('Allow', 'GET, HEAD, PUT, DELETE, OPTIONS');
      res.end();
      return;
    }
    serveStatic(req, res).catch((error) => {
      res.statusCode = 500;
      res.end(error.message);
    });
  });
});

const port = Number(process.env.PORT || 4173);
server.listen(port, '0.0.0.0', () => {
  console.log(`Neural Game Lab listening on http://0.0.0.0:${port}`);
  console.log(`Model path: ${modelPath}`);
});
