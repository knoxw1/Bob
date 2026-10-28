import { createRequire } from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Chess } from 'chess.js';
import { ChessAgent, chessMoveIndex, encodeChessState } from '../src/chess-agent.js';
import { decodeModel, encodeModel, validateModelData } from '../src/model-binary.js';
import { MODEL_WEIGHT_SHAPES } from '../src/model-binary.js';

const require = createRequire(import.meta.url);
const initEngine = require('stockfish');
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');

const games = Number(process.env.STOCKFISH_GAMES || 100);
const depth = Number(process.env.STOCKFISH_DEPTH || 8);

const engine = initEngine('lite-single');
let pendingResolve = null;
engine.listener = (line) => {
  const match = line.match(/bestmove (\S+)/);
  if (match && pendingResolve) {
    const resolve = pendingResolve;
    pendingResolve = null;
    resolve(match[1]);
  }
};

await new Promise((resolve) => {
  const readyHandler = (line) => {
    if (line === 'readyok') {
      engine.listener = (line) => {
        const match = line.match(/bestmove (\S+)/);
        if (match && pendingResolve) {
          const resolveMove = pendingResolve;
          pendingResolve = null;
          resolveMove(match[1]);
        }
      };
      resolve();
    }
  };
  engine.listener = readyHandler;
  engine.processCommand('uci');
  engine.processCommand('isready');
});

function stockfishMove(fen) {
  return new Promise((resolve) => {
    pendingResolve = resolve;
    engine.processCommand('ucinewgame');
    engine.processCommand(`position fen ${fen}`);
    engine.processCommand(`go depth ${depth}`);
  });
}

const modelPath = path.join(root, 'data', 'model.bin');
let modelData = null;
try {
  modelData = decodeModel(await fs.readFile(modelPath));
} catch (error) {
  console.warn('No existing model found, starting fresh.');
}

const agent = await ChessAgent.create(modelData?.chess || null);

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const chess = new Chess();
  const ourColor = Math.random() < 0.5 ? 'w' : 'b';
  const history = [];
  while (!chess.isGameOver()) {
    const turn = chess.turn();
    if (turn === ourColor) {
      const move = agent.chooseMove(chess, 0.1);
      if (!move) break;
      const state = Array.from(encodeChessState(chess));
      const action = chessMoveIndex(move);
      history.push({ state, action, color: turn });
      chess.move(move);
    } else {
      const best = await stockfishMove(chess.fen());
      const from = best.slice(0, 2);
      const to = best.slice(2, 4);
      const promotion = best.length > 4 ? best[4] : undefined;
      const legal = chess.moves({ verbose: true }).find((candidate) => candidate.from === from && candidate.to === to);
      if (!legal) break;
      chess.move({ from, to, promotion });
    }
  }

  const winner = chess.isCheckmate() ? (chess.turn() === 'w' ? 'b' : 'w') : null;
  if (history.length) {
    await agent.learnFromGame(history, winner);
  }
  console.log(`[${gameIndex + 1}/${games}] our color ${ourColor}, winner ${winner || 'draw'}, history ${history.length}`);
}

const exported = await agent.exportModel();
const nextModel = modelData ? { ...modelData, chess: exported } : {
  version: 1,
  trainingGames: 0,
  observedGames: 0,
  weights: MODEL_WEIGHT_SHAPES.map((shape) => ({
    shape,
    values: new Array(shape.reduce((size, dimension) => size * dimension, 1)).fill(0),
  })),
  chess: exported,
};
const valid = validateModelData(nextModel);
await fs.mkdir(path.dirname(modelPath), { recursive: true });
await fs.writeFile(modelPath, encodeModel(valid));
console.log(`Saved trained chess model to ${modelPath}`);

engine.terminate?.();
agent.dispose?.();
EOF
echo written