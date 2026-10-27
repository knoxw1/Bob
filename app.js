import { ConnectFourAgent } from './src/agent.js';
import { ChessAgent, chessMoveIndex, encodeChessState } from './src/chess-agent.js';
import { COLS, ROWS, createBoard, drop, encodeState, getCell, isWinningMove, legalMoves, playersForOpening } from './src/game.js';
import { Chess } from 'chess.js';
import { decodeModel, encodeModel } from './src/model-binary.js';

const LEGACY_STORAGE_KEY = 'connect-four-neural-network-v1';
const FIRST_TRAINING_GAMES = 100;
const reduceNetworkMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const BROWSER_MODEL_DB = 'neural-game-lab';
const BROWSER_MODEL_STORE = 'models';
const BROWSER_MODEL_KEY = 'combined-model';

function openBrowserModelDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error('IndexedDB is not available in this browser.'));
      return;
    }
    const request = window.indexedDB.open(BROWSER_MODEL_DB, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(BROWSER_MODEL_STORE)) {
        database.createObjectStore(BROWSER_MODEL_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Could not open browser model storage.'));
  });
}

async function saveBrowserModel(modelData) {
  try {
    const database = await openBrowserModelDb();
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(BROWSER_MODEL_STORE, 'readwrite');
      transaction.objectStore(BROWSER_MODEL_STORE).put(modelData, BROWSER_MODEL_KEY);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error || new Error('Could not save the model in this browser.'));
      transaction.onabort = () => reject(transaction.error || new Error('Browser model save was aborted.'));
    });
    database.close();
  } catch (error) {
    console.warn('Could not back up the trained model in this browser:', error);
  }
}

async function loadBrowserModel() {
  try {
    const database = await openBrowserModelDb();
    const modelData = await new Promise((resolve, reject) => {
      const transaction = database.transaction(BROWSER_MODEL_STORE, 'readonly');
      const request = transaction.objectStore(BROWSER_MODEL_STORE).get(BROWSER_MODEL_KEY);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error('Could not read the browser model backup.'));
    });
    database.close();
    return modelData;
  } catch (error) {
    console.warn('Could not read the trained model backup from this browser:', error);
    return null;
  }
}

const boardElement = document.querySelector('#board');
const columnControls = document.querySelector('#columnControls');
const chessBoardElement = document.querySelector('#chessBoard');
const worker = new Worker(new URL('./src/trainer.worker.js', import.meta.url), { type: 'module' });
let agent = null;
let chessAgent = null;
let gameMode = 'chess';
let board = createBoard();
let openingPlayer = 'human';
let { human: humanPlayer, agent: agentPlayer } = playersForOpening(openingPlayer);
let currentPlayer = humanPlayer;
let gameOver = false;
let training = false;
let learningGame = false;
let winnerCells = [];
let gameHistory = [];
let completedGames = 0;
let sessionScore = { human: 0, learner: 0, draws: 0 };
let chessGame = new Chess();
let chessHistory = [];
let chessHumanColor = 'w';
let chessSelectedSquare = null;
let chessLastMove = null;
let pendingChessPromotion = null;
let chessResultCount = { human: 0, learner: 0, draws: 0 };
let saveQueue = Promise.resolve();
let currentVisualization = null;
let previousVisualization = null;
let visualizationStartedAt = 0;
let visualizationFrame = 0;
let networkMode = 'weights';
let networkZoom = 1;
let networkPan = { x: 0, y: 0 };
let networkNodes = [];
let networkEdges = [];
let hoveredNeuron = null;
let selectedNeuron = null;
let hoveredConnection = null;
let selectedConnection = null;
let signalActivityUntil = 0;
let signalDrainPaths = [];
let previousSignalActivity = false;
let pointerGesture = null;

async function gzipBytes(value) {
  if (!('CompressionStream' in window)) return null;
  try {
    const stream = new Blob([value]).stream().pipeThrough(new CompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch (error) {
    console.warn('Could not compress model data before upload:', error);
    return null;
  }
}

function queueModelSave(modelData) {
  // Keep a browser-side backup so training survives server restarts or ephemeral hosting.
  saveBrowserModel(modelData);
  saveQueue = saveQueue.catch(() => {}).then(async () => {
    document.querySelector('#saveStatus').textContent = 'SAVING...';
    const encoded = encodeModel(modelData);
    const compressed = await gzipBytes(encoded);
    const uploadBytes = compressed || encoded;
    const useGzip = Boolean(compressed);
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const chunkSize = 3.5 * 1024 * 1024;
        const total = Math.max(1, Math.ceil(uploadBytes.length / chunkSize));
        const uploadId = (crypto && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`);
        const concurrency = 4;
        let nextIndex = 0;
        let failure = null;
        async function worker() {
          while (true) {
            const index = nextIndex++;
            if (index >= total) return;
            if (failure) return;
            const chunk = uploadBytes.slice(index * chunkSize, (index + 1) * chunkSize);
            let response;
            let requestError = null;
            for (let innerAttempt = 0; innerAttempt < 2; innerAttempt += 1) {
              try {
                response = await fetch(`/api/model?upload=${encodeURIComponent(uploadId)}&index=${index}&total=${total}${useGzip ? '&gzip=1' : ''}`, {
                  method: 'PUT',
                  headers: { 'Content-Type': useGzip ? 'application/octet-stream' : 'text/plain; charset=utf-8' },
                  body: chunk,
                });
                break;
              } catch (error) {
                requestError = error;
                if (innerAttempt === 0) await new Promise((resolve) => setTimeout(resolve, 500));
              }
            }
            if (!response) { failure = requestError || new Error('Network request failed.'); return; }
            const responseText = await response.text();
            let result = {};
            try { result = responseText ? JSON.parse(responseText) : {}; }
            catch { failure = new Error(`Save endpoint returned non-JSON (${response.status}): ${responseText.slice(0, 160)}`); return; }
            if (!response.ok && response.status !== 202) {
              failure = new Error(result.error || 'Could not save weights to the workspace.');
              return;
            }
          }
        }
        await Promise.all(Array.from({ length: concurrency }, () => worker()));
        if (failure) throw failure;
        document.querySelector('#saveStatus').textContent = 'SAVED TO WORKSPACE';
        return;
      } catch (error) {
        lastError = error;
        if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
    throw lastError;
  });
  return saveQueue;
}

async function loadSavedModel() {
  let serverModel = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetch('/api/model');
      if (response.status === 204) {
        serverModel = null;
        break;
      }
      if (!response.ok) throw new Error('Could not read the workspace model file.');
      serverModel = decodeModel(new Uint8Array(await response.arrayBuffer()));
      break;
    } catch (error) {
      if (attempt === 2) {
        console.warn('Could not load the workspace model:', error);
      } else {
        await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
      }
    }
  }

  if (serverModel) {
    saveBrowserModel(serverModel);
    return serverModel;
  }

  // On ephemeral hosts the server file may be gone while this browser still has
  // the last trained weights. Restore from the browser backup and push it back.
  const browserModel = await loadBrowserModel();
  if (browserModel) {
    try { await queueModelSave(browserModel); } catch (error) { console.warn(error); }
    return browserModel;
  }

  try {
    const legacyModel = JSON.parse(localStorage.getItem(LEGACY_STORAGE_KEY));
    if (legacyModel) {
      await queueModelSave(legacyModel);
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      return legacyModel;
    }
  } catch (error) {
    console.warn('Could not migrate the browser-saved model:', error);
  }
  return null;
}

function networkPositions(count, x, height, top, bottom) {
  const span = height - top - bottom;
  const rows = Math.max(1, count);
  const rowSpan = rows === 1 ? 0 : span / (rows - 1);
  return Array.from({ length: count }, (_, index) => ({
    x,
    y: top + index * rowSpan,
  }));
}

function noiseSample(seed, step) {
  let value = (seed | 0) ^ Math.imul(step | 0, 0x9e3779b1);
  value = Math.imul(value ^ (value >>> 16), 0x85ebca6b);
  value = Math.imul(value ^ (value >>> 13), 0xc2b2ae35);
  value ^= value >>> 16;
  return (value >>> 0) / 0xffffffff * 2 - 1;
}

function smoothNoise(seed, time, rate) {
  const position = time * rate;
  const step = Math.floor(position);
  const fraction = position - step;
  const blend = fraction * fraction * (3 - 2 * fraction);
  const start = noiseSample(seed, step);
  const end = noiseSample(seed, step + 1);
  return start + (end - start) * blend;
}

function edgeWeight(edge, progress, oldWeights) {
  const current = edge.weight || 0;
  const previous = oldWeights?.get(`${edge.from}:${edge.to}`) ?? current;
  return previous + (current - previous) * progress;
}

function getNetworkLayers(visualization) {
  const outputNodes = visualization.outputs.map((value, index) => typeof value === 'number'
    ? { value, label: `C${index + 1}` }
    : value);
  return [
    visualization.inputs,
    visualization.hiddenOne.map((value, index) => ({ value, label: `H1:${index + 1}` })),
    visualization.hiddenTwo.map((value, index) => ({ value, label: `H2:${index + 1}` })),
    outputNodes,
  ];
}

function getNodeId(layer, index) {
  return `${layer}:${index}`;
}

function scheduleNetworkFrame() {
  const signalsActive = networkMode === 'signals' && (training || learningGame || performance.now() < signalActivityUntil);
  const signalsDraining = networkMode === 'signals' && signalDrainPaths.length > 0;
  if (!currentVisualization || document.hidden || visualizationFrame || (reduceNetworkMotion && !signalsActive && !signalsDraining)) return;
  visualizationFrame = requestAnimationFrame(() => {
    visualizationFrame = 0;
    drawNetworkFrame();
  });
}

function drawNetworkFrame(forceComplete = false) {
  if (!currentVisualization) return;
  const now = performance.now();
  const signalsActive = networkMode === 'signals' && (training || learningGame || now < signalActivityUntil);
  if (signalsActive) {
    if (!previousSignalActivity) signalDrainPaths = [];
  } else if (previousSignalActivity) {
    signalDrainPaths = networkEdges.filter((edge) => edge.signalEligible).map((edge) => ({
      ...edge,
      signalPhase: ((now / 720) + edge.signalOffset) % 1,
      startedAt: now,
    }));
  }
  previousSignalActivity = signalsActive;
  if (networkMode !== 'signals') signalDrainPaths = [];
  const canvas = document.querySelector('#networkCanvas');
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(bounds.width * ratio);
  const pixelHeight = Math.round(bounds.height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  const progress = forceComplete ? 1 : Math.min(1, (performance.now() - visualizationStartedAt) / 360);
  context.fillStyle = '#060a14';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = '#7199cf12';
  context.lineWidth = 1;
  for (let x = 18; x < width; x += 26) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = 22; y < height; y += 26) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  context.save();
  context.translate(width / 2 + networkPan.x, height / 2 + networkPan.y);
  context.scale(networkZoom, networkZoom);
  context.translate(-width / 2, -height / 2);

  const top = 48;
  const bottom = 36;
  const xs = [0.105, 0.37, 0.635, 0.89].map((fraction) => width * fraction);
  const drift = reduceNetworkMotion ? 0 : 1;
  const layers = getNetworkLayers(currentVisualization);
  const oldLayers = previousVisualization ? getNetworkLayers(previousVisualization) : layers;
  const time = now / 1000;
  const layerDrift = layers.map((layer, layerIndex) => {
    const dense = layer.length > 72;
    const scale = dense ? 0.55 : 1;
    return {
      x: (smoothNoise(9000 + layerIndex * 211, time, 0.055) * 7.5
        + smoothNoise(12000 + layerIndex * 307, time, 0.14) * 2.5) * scale,
      y: (smoothNoise(15000 + layerIndex * 401, time, 0.061) * 6.5
        + smoothNoise(19000 + layerIndex * 503, time, 0.16) * 2.5) * scale,
    };
  });
  const points = layers.map((layer, layerIndex) => networkPositions(layer.length, xs[layerIndex], height, top, bottom)
    .map((point, index) => {
      const seed = layerIndex * 10000 + index + 1;
      const xRate = 0.07 + (noiseSample(seed, -1) + 1) * 0.055;
      const yRate = 0.075 + (noiseSample(seed, -2) + 1) * 0.06;
      const xPhase = noiseSample(seed, -3) * 40;
      const yPhase = noiseSample(seed, -4) * 40;
      const motionScale = layer.length > 72 ? 0.42 : 1;
      return {
        x: point.x + drift * (
          layerDrift[layerIndex].x
          + smoothNoise(seed, time + xPhase, xRate) * 22 * motionScale
          + smoothNoise(seed + 59, time + xPhase * 0.7, xRate * 2.4) * 8 * motionScale
        ),
        y: point.y + drift * (
          layerDrift[layerIndex].y
          + smoothNoise(seed + 83, time + yPhase, yRate) * 18 * motionScale
          + smoothNoise(seed + 127, time + yPhase * 0.6, yRate * 2.6) * 7 * motionScale
        ),
      };
    }));
  const edgeGroups = [
    [currentVisualization.edges.inputHidden, previousVisualization?.edges.inputHidden],
    [currentVisualization.edges.hiddenHidden, previousVisualization?.edges.hiddenHidden],
    [currentVisualization.edges.hiddenOutput, previousVisualization?.edges.hiddenOutput],
  ];
  const nextNodes = [];
  const nextEdges = [];
  for (let layerIndex = 0; layerIndex < edgeGroups.length; layerIndex += 1) {
    const [edges, oldEdges] = edgeGroups[layerIndex];
    const oldWeights = oldEdges && new Map(oldEdges.map((edge) => [`${edge.from}:${edge.to}`, edge.weight]));
    const maxWeight = Math.max(0.0001, ...edges.map((edge) => Math.abs(edge.weight)));
    for (let index = 0; index < edges.length; index += 1) {
      const edge = edges[index];
      const weight = edgeWeight(edge, progress, oldWeights);
      const strength = Math.min(1, Math.abs(weight) / maxWeight);
      const source = points[layerIndex][edge.from];
      const target = points[layerIndex + 1][edge.to];
      const isSelected = selectedNeuron && (
        (selectedNeuron.layer === layerIndex && selectedNeuron.index === edge.from)
        || (selectedNeuron.layer === layerIndex + 1 && selectedNeuron.index === edge.to)
      );
      const isHovered = hoveredNeuron && (
        (hoveredNeuron.layer === layerIndex && hoveredNeuron.index === edge.from)
        || (hoveredNeuron.layer === layerIndex + 1 && hoveredNeuron.index === edge.to)
      );
      const edgeId = `${layerIndex}:${edge.from}:${edge.to}`;
      const isConnectionHovered = hoveredConnection?.id === edgeId;
      const isConnectionSelected = selectedConnection?.id === edgeId;
      const emphasized = isSelected || isHovered || isConnectionHovered || isConnectionSelected;
      const denseEdgeLayer = points[layerIndex].length > 72 || points[layerIndex + 1].length > 72;
      const signalMode = networkMode === 'signals';
      const opacity = isConnectionHovered || isConnectionSelected
        ? 0.98
        : isSelected || isHovered ? 0.82 : signalMode
          ? 0.02 + strength * 0.07
          : denseEdgeLayer
            ? 0.05 + strength * 0.13
            : 0.08 + strength * 0.29;
      const color = weight >= 0 ? '69, 194, 255' : '255, 94, 143';
      const signalEligible = strength > (denseEdgeLayer ? 0.34 : 0.22)
        && (index % 4 === layerIndex % 4 || emphasized);
      const signalOffset = ((index * 37 + layerIndex * 19) % 100) / 100;
      const shouldDraw = true;
      if (shouldDraw) {
        context.beginPath();
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.strokeStyle = `rgba(${color}, ${opacity})`;
        context.lineWidth = isConnectionHovered || isConnectionSelected
          ? 3.8 + strength * 2
          : emphasized ? 1.9 + strength : denseEdgeLayer ? 0.2 + strength * 0.65 : 0.3 + strength * 1.1;
        if (emphasized) {
          context.shadowColor = `rgb(${color})`;
          context.shadowBlur = isConnectionHovered || isConnectionSelected ? 15 : 8;
        }
        context.stroke();
        context.shadowBlur = 0;
      }
      nextEdges.push({
        id: edgeId,
        layer: layerIndex,
        from: edge.from,
        to: edge.to,
        x1: source.x,
        y1: source.y,
        x2: target.x,
        y2: target.y,
        weight,
        strength,
        signalEligible,
        signalOffset,
      });

      if (signalMode && signalsActive && signalEligible && shouldDraw) {
        const phase = ((now / 720) + signalOffset) % 1;
        const signalX = source.x + (target.x - source.x) * phase;
        const signalY = source.y + (target.y - source.y) * phase;
        context.beginPath();
        context.arc(signalX, signalY, 1.3 + strength * 1.8, 0, Math.PI * 2);
        context.fillStyle = `rgba(${color}, ${0.32 + 0.62 * Math.sin(phase * Math.PI)})`;
        context.shadowColor = `rgb(${color})`;
        context.shadowBlur = 5 + strength * 5;
        context.fill();
        context.shadowBlur = 0;
      }
    }
  }

  if (!signalsActive && networkMode === 'signals' && signalDrainPaths.length) {
    const activePaths = [];
    const edgesById = new Map(nextEdges.map((edge) => [edge.id, edge]));
    for (const path of signalDrainPaths) {
      const edge = edgesById.get(path.id);
      if (!edge) continue;
      const duration = (1 - path.signalPhase) * 720;
      const progress = Math.min(1, (now - path.startedAt) / Math.max(1, duration));
      if (progress >= 1) continue;
      const phase = path.signalPhase + (1 - path.signalPhase) * progress;
      const source = points[edge.layer][edge.from];
      const target = points[edge.layer + 1][edge.to];
      const color = edge.weight >= 0 ? '69, 194, 255' : '255, 94, 143';
      context.beginPath();
      context.arc(
        source.x + (target.x - source.x) * phase,
        source.y + (target.y - source.y) * phase,
        1.3 + edge.strength * 1.8,
        0,
        Math.PI * 2,
      );
      context.fillStyle = `rgba(${color}, ${(0.32 + 0.62 * Math.sin(phase * Math.PI)) * (1 - progress * 0.3)})`;
      context.shadowColor = `rgb(${color})`;
      context.shadowBlur = 5 + edge.strength * 5;
      context.fill();
      context.shadowBlur = 0;
      activePaths.push(path);
    }
    signalDrainPaths = activePaths;
  }

  const layerLabels = currentVisualization.layerLabels || ['BOARD / 84', 'HIDDEN / 512', 'HIDDEN / 512', 'MOVES / 7'];
  context.textAlign = 'center';
  context.font = '9px Courier New';
  for (let layerIndex = 0; layerIndex < points.length; layerIndex += 1) {
    context.fillStyle = '#89a7ce';
    context.fillText(layerLabels[layerIndex], xs[layerIndex], 17);
    const values = layers[layerIndex].map(({ value }) => value);
    const previousValues = oldLayers[layerIndex].map(({ value }) => value);
    const maxActivation = Math.max(0.0001, ...values.map((value) => Math.abs(value)));
    for (let index = 0; index < points[layerIndex].length; index += 1) {
      const point = points[layerIndex][index];
      const oldValue = previousValues[index] ?? values[index];
      const value = oldValue + (values[index] - oldValue) * progress;
      const level = Math.min(1, Math.abs(value) / maxActivation);
      const isInput = layerIndex === 0;
      const isOutput = layerIndex === points.length - 1;
      const denseLayer = points[layerIndex].length > 72;
      const radius = isOutput
        ? 5.2 + level * 3.1
        : isInput
          ? (denseLayer ? 1.25 + level * 0.9 : 1.65 + level * 1.15)
          : (denseLayer ? 1.15 + level * 0.95 : 1.8 + level * 1.65);
      const color = isInput
        ? gameMode === 'chess'
          ? (value ? '#42dcff' : '#174b77')
          : index < 42 ? (value ? '#42dcff' : '#174b77') : (value ? '#9a8bff' : '#322b69')
        : isOutput && value < 0 ? '#ff668e' : isOutput ? '#55e0ff' : '#4d95ff';
      const id = getNodeId(layerIndex, index);
      const isSelected = selectedNeuron?.id === id;
      const isHovered = hoveredNeuron?.id === id;
      nextNodes.push({ id, layer: layerIndex, index, x: point.x, y: point.y, radius, value });
      context.beginPath();
      context.arc(point.x, point.y, radius + (isSelected ? 4 : isHovered ? 3 : 0), 0, Math.PI * 2);
      context.fillStyle = color;
      context.globalAlpha = isInput && !value ? 0.68 : 0.42 + level * 0.58;
      context.shadowColor = color;
      context.shadowBlur = level > 0.16 || isSelected || isHovered ? 5 + level * 11 : 0;
      context.fill();
      context.shadowBlur = 0;
      context.globalAlpha = 1;
      if (isSelected || isHovered || isOutput) {
        context.beginPath();
        context.arc(point.x, point.y, radius + (isSelected ? 4 : isHovered ? 3 : 1), 0, Math.PI * 2);
        context.strokeStyle = isSelected ? '#effaff' : isHovered ? '#a9eaff' : '#c6f7ff99';
        context.lineWidth = isSelected ? 1.6 : 0.85;
        context.stroke();
      }

      if (isOutput) {
        context.fillStyle = '#c5d9ef';
        context.textAlign = 'left';
        context.font = '10px Courier New';
        context.fillText(`${layers[layerIndex][index].label || `C${index + 1}`}  ${value.toFixed(2)}`, point.x + 10, point.y + 3);
      }
    }
  }
  context.restore();
  networkNodes = nextNodes;
  networkEdges = nextEdges;

  if (progress >= 1) previousVisualization = currentVisualization;
  scheduleNetworkFrame();
}

function showNetwork(visualization, activity, signal = false) {
  if (!visualization) return;
  previousVisualization = currentVisualization;
  currentVisualization = visualization;
  visualizationStartedAt = performance.now();
  if (signal) signalActivityUntil = visualizationStartedAt + 1200;
  document.querySelector('#networkActivity').textContent = activity.toUpperCase();
  drawNetworkFrame(true);
}

function canvasPoint(event) {
  const bounds = event.currentTarget.getBoundingClientRect();
  return { x: event.clientX - bounds.left, y: event.clientY - bounds.top, width: bounds.width, height: bounds.height };
}

function toNetworkSpace(point) {
  return {
    x: (point.x - point.width / 2 - networkPan.x) / networkZoom + point.width / 2,
    y: (point.y - point.height / 2 - networkPan.y) / networkZoom + point.height / 2,
  };
}

function findNeuron(point) {
  const position = toNetworkSpace(point);
  let nearest = null;
  let nearestDistance = Infinity;
  for (const node of networkNodes) {
    const distance = Math.hypot(node.x - position.x, node.y - position.y);
    if (distance < Math.max(7, node.radius + 4) && distance < nearestDistance) {
      nearest = node;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function distanceToSegment(point, edge) {
  const deltaX = edge.x2 - edge.x1;
  const deltaY = edge.y2 - edge.y1;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  const projection = lengthSquared
    ? Math.max(0, Math.min(1, ((point.x - edge.x1) * deltaX + (point.y - edge.y1) * deltaY) / lengthSquared))
    : 0;
  return Math.hypot(point.x - (edge.x1 + projection * deltaX), point.y - (edge.y1 + projection * deltaY));
}

function findNetworkTarget(point) {
  const node = findNeuron(point);
  if (node) return { type: 'neuron', node };
  const networkPoint = toNetworkSpace(point);
  const threshold = Math.max(3, 7 / networkZoom);
  let closest = null;
  let closestDistance = threshold;
  for (const edge of networkEdges) {
    const distance = distanceToSegment(networkPoint, edge);
    if (distance <= closestDistance) {
      closest = edge;
      closestDistance = distance;
    }
  }
  return closest ? { type: 'connection', edge: closest } : null;
}

function describeNeuron(node) {
  if (!node || !currentVisualization) return;
  const layerNames = ['BOARD INPUT', 'HIDDEN LAYER 1', 'HIDDEN LAYER 2', 'MOVE OUTPUT'];
  const valueLabel = node.layer === 3 ? 'score' : 'activation';
  let inputLabel = '';
  if (node.layer === 0) {
    if (gameMode === 'chess') {
      const feature = currentVisualization.inputs[node.index]?.index ?? node.index;
      inputLabel = ` / chess feature ${feature + 1}`;
    } else {
      const player = node.index < 42 ? 'current player' : 'opponent';
      const cell = node.index % 42;
      inputLabel = ` / ${player}, row ${Math.floor(cell / 7) + 1}, column ${(cell % 7) + 1}`;
    }
  }

  const groups = [
    currentVisualization.edges.inputHidden,
    currentVisualization.edges.hiddenHidden,
    currentVisualization.edges.hiddenOutput,
  ];
  const connected = [];
  for (let layer = 0; layer < groups.length; layer += 1) {
    for (const edge of groups[layer]) {
      const outgoing = node.layer === layer && edge.from === node.index;
      const incoming = node.layer === layer + 1 && edge.to === node.index;
      if (outgoing || incoming) {
        connected.push({ edge, outgoing, targetLayer: outgoing ? layer + 1 : layer });
      }
    }
  }
  connected.sort((left, right) => Math.abs(right.edge.weight) - Math.abs(left.edge.weight));
  const connection = connected[0];
  const connectionText = connection
    ? `Strongest ${connection.outgoing ? 'outgoing' : 'incoming'} weight ${connection.edge.weight >= 0 ? '+' : ''}${connection.edge.weight.toFixed(4)}.`
    : 'No displayed connections.';
  const outputLabel = node.layer === 3 && gameMode === 'chess'
    ? currentVisualization.outputs[node.index]?.label || `Move ${node.index + 1}`
    : node.layer === 3 ? `Column ${node.index + 1}` : `${layerNames[node.layer]} ${node.index + 1}`;
  const title = node.layer === 3 ? outputLabel : `${layerNames[node.layer]} ${node.index + 1}`;
  document.querySelector('#neuronDetail strong').textContent = title;
  document.querySelector('#neuronDetail span').textContent = `${valueLabel} ${node.value.toFixed(4)}${inputLabel}. ${connectionText}`;
}

function describeConnection(edge) {
  if (!edge) return;
  const source = edge.layer === 0 ? `Input ${edge.from + 1}` : `Layer ${edge.layer} / neuron ${edge.from + 1}`;
  const target = edge.layer === 2 ? `Move ${edge.to + 1}` : `Layer ${edge.layer + 1} / neuron ${edge.to + 1}`;
  document.querySelector('#neuronDetail strong').textContent = 'WEIGHTED CONNECTION';
  document.querySelector('#neuronDetail span').textContent = `${source} to ${target}. Weight ${edge.weight >= 0 ? '+' : ''}${edge.weight.toFixed(5)}. ${edge.strength > 0.5 ? 'Strong' : 'Active'} connection.`;
}

function setFullscreenButton(expanded) {
  const button = document.querySelector('#expandNetwork');
  button.textContent = expanded ? 'Exit' : 'Expand';
  button.setAttribute('aria-label', expanded ? 'Exit full screen' : 'Expand neural network');
  button.title = expanded ? 'Exit full screen' : 'Expand neural network';
}

function setNetworkZoom(zoom, center = null) {
  const canvas = document.querySelector('#networkCanvas');
  const bounds = canvas.getBoundingClientRect();
  const point = center || { x: bounds.width / 2, y: bounds.height / 2, width: bounds.width, height: bounds.height };
  const oldPosition = toNetworkSpace(point);
  networkZoom = Math.max(0.55, Math.min(2.8, zoom));
  networkPan.x = point.x - point.width / 2 - (oldPosition.x - point.width / 2) * networkZoom;
  networkPan.y = point.y - point.height / 2 - (oldPosition.y - point.height / 2) * networkZoom;
  document.querySelector('#zoomReadout').textContent = `${Math.round(networkZoom * 100)}%`;
  drawNetworkFrame(true);
}

function initializeNetworkControls() {
  const canvas = document.querySelector('#networkCanvas');
  const tooltip = document.querySelector('#networkTooltip');
  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    pointerGesture = { clientX: event.clientX, clientY: event.clientY, moved: false, pan: { ...networkPan } };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    const point = canvasPoint(event);
    if (pointerGesture) {
      const deltaX = event.clientX - pointerGesture.clientX;
      const deltaY = event.clientY - pointerGesture.clientY;
      if (Math.hypot(deltaX, deltaY) > 3) pointerGesture.moved = true;
      if (pointerGesture.moved) {
        networkPan.x = pointerGesture.pan.x + deltaX;
        networkPan.y = pointerGesture.pan.y + deltaY;
        drawNetworkFrame(true);
      }
      return;
    }

    const target = findNetworkTarget(point);
    hoveredNeuron = target?.type === 'neuron' ? target.node : null;
    hoveredConnection = target?.type === 'connection' ? target.edge : null;
    canvas.style.cursor = target ? 'pointer' : 'grab';
    if (hoveredNeuron) {
      const layerNames = ['Board input', 'Hidden 1', 'Hidden 2', 'Move output'];
      const value = hoveredNeuron.layer === 3 ? 'score' : 'activation';
      tooltip.textContent = `${layerNames[hoveredNeuron.layer]} ${hoveredNeuron.index + 1} / ${value} ${hoveredNeuron.value.toFixed(4)}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.min(point.x + 12, point.width - 195)}px`;
      tooltip.style.top = `${Math.max(6, Math.min(point.y + 12, point.height - 58))}px`;
    } else if (hoveredConnection) {
      tooltip.textContent = `WEIGHT ${hoveredConnection.weight >= 0 ? '+' : ''}${hoveredConnection.weight.toFixed(5)} / ${Math.round(hoveredConnection.strength * 100)}% STRENGTH`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.min(point.x + 12, point.width - 195)}px`;
      tooltip.style.top = `${Math.max(6, Math.min(point.y + 12, point.height - 58))}px`;
    } else {
      tooltip.hidden = true;
    }
    drawNetworkFrame(true);
  });
  canvas.addEventListener('pointerup', (event) => {
    if (!pointerGesture) return;
    const dragged = pointerGesture.moved;
    pointerGesture = null;
    if (!dragged) {
      const target = findNetworkTarget(canvasPoint(event));
      selectedNeuron = target?.type === 'neuron' ? target.node : null;
      selectedConnection = target?.type === 'connection' ? target.edge : null;
      if (selectedNeuron) {
        describeNeuron(selectedNeuron);
      } else if (selectedConnection) {
        describeConnection(selectedConnection);
      } else {
        document.querySelector('#neuronDetail strong').textContent = 'NETWORK OVERVIEW';
        document.querySelector('#neuronDetail span').textContent = 'Choose a neuron or connection to inspect its current value.';
      }
      drawNetworkFrame(true);
    }
  });
  canvas.addEventListener('pointercancel', () => { pointerGesture = null; });
  canvas.addEventListener('pointerleave', () => {
    if (!pointerGesture) {
      hoveredNeuron = null;
      hoveredConnection = null;
      tooltip.hidden = true;
      drawNetworkFrame(true);
    }
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    const point = canvasPoint(event);
    setNetworkZoom(networkZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12), point);
  }, { passive: false });

  document.querySelector('#zoomIn').addEventListener('click', () => setNetworkZoom(networkZoom * 1.2));
  document.querySelector('#zoomOut').addEventListener('click', () => setNetworkZoom(networkZoom / 1.2));
  document.querySelector('#fitNetwork').addEventListener('click', () => {
    networkZoom = 1;
    networkPan = { x: 0, y: 0 };
    document.querySelector('#zoomReadout').textContent = '100%';
    drawNetworkFrame(true);
  });
  const networkPanel = document.querySelector('.network-visual');
  const expandButton = document.querySelector('#expandNetwork');
  expandButton.addEventListener('click', async () => {
    const isFullscreen = document.fullscreenElement === networkPanel;
    const isFallbackExpanded = networkPanel.classList.contains('is-expanded');
    if (isFullscreen || isFallbackExpanded) {
      if (isFullscreen && document.exitFullscreen) await document.exitFullscreen();
      else networkPanel.classList.remove('is-expanded');
      setFullscreenButton(false);
      drawNetworkFrame(true);
      return;
    }
    if (networkPanel.requestFullscreen) {
      try {
        await networkPanel.requestFullscreen();
      } catch {
        networkPanel.classList.add('is-expanded');
        setFullscreenButton(true);
      }
    } else {
      networkPanel.classList.add('is-expanded');
      setFullscreenButton(true);
    }
    drawNetworkFrame(true);
  });
  document.addEventListener('fullscreenchange', () => {
    setFullscreenButton(document.fullscreenElement === networkPanel);
    drawNetworkFrame(true);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && networkPanel.classList.contains('is-expanded')) {
      networkPanel.classList.remove('is-expanded');
      setFullscreenButton(false);
      drawNetworkFrame(true);
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && visualizationFrame) {
      cancelAnimationFrame(visualizationFrame);
      visualizationFrame = 0;
    } else {
      drawNetworkFrame(true);
    }
  });
  document.querySelectorAll('[data-network-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      networkMode = button.dataset.networkMode;
      document.querySelectorAll('[data-network-mode]').forEach((modeButton) => {
        const selected = modeButton === button;
        modeButton.classList.toggle('selected', selected);
        modeButton.setAttribute('aria-pressed', String(selected));
      });
      drawNetworkFrame(true);
    });
  });
}
function setMessage(message) {
  document.querySelector('#gameMessage').textContent = message;
}

function drawBoard() {
  for (const cell of boardElement.children) {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const piece = getCell(board, row, col);
    cell.replaceChildren();
    const humanPiece = piece === humanPlayer;
    cell.setAttribute('aria-label', `Row ${row + 1}, column ${col + 1}, ${piece ? humanPiece ? 'you' : 'learner' : 'empty'}`);
    if (piece) {
      const disc = document.createElement('span');
      const winning = winnerCells.some(([winRow, winCol]) => winRow === row && winCol === col);
      disc.className = `disc ${humanPiece ? 'human' : 'learner'}${winning ? ' winner' : ''}`;
      cell.append(disc);
    }
  }
  const playerCanMove = !training && !learningGame && !gameOver && currentPlayer === humanPlayer && agent;
  for (let col = 0; col < COLS; col += 1) {
    columnControls.children[col].disabled = !playerCanMove || !legalMoves(board).includes(col);
  }
  document.querySelector('#newGameButton').disabled = training || learningGame;
  document.querySelector('#resetButton').disabled = training || learningGame || !agent || !chessAgent;
  document.querySelectorAll('[data-starter]').forEach((button) => {
    button.disabled = training || learningGame;
  });
  document.querySelector('#turnSwatch').className = `piece-swatch ${currentPlayer === humanPlayer ? 'human-swatch' : 'learner-swatch'}`;
}

function buildBoard() {
  for (let col = 0; col < COLS; col += 1) {
    const control = document.createElement('button');
    control.type = 'button';
    control.className = 'column-button';
    control.setAttribute('aria-label', `Drop in column ${col + 1}`);
    control.addEventListener('click', () => playHumanMove(col));
    columnControls.append(control);
  }
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    for (let col = 0; col < COLS; col += 1) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.setAttribute('role', 'gridcell');
      boardElement.append(cell);
    }
  }
  drawBoard();
}

function updateScore() {
  const score = gameMode === 'chess' ? chessResultCount : sessionScore;
  const ids = gameMode === 'chess'
    ? ['#chessHumanWins', '#chessLearnerWins', '#chessDraws']
    : ['#humanWins', '#learnerWins', '#draws'];
  document.querySelector(ids[0]).textContent = String(score.human);
  document.querySelector(ids[1]).textContent = String(score.learner);
  document.querySelector(ids[2]).textContent = String(score.draws);
}

const CHESS_GLYPHS = {
  wk: '♔', wq: '♕', wr: '♖', wb: '♗', wn: '♘', wp: '♙',
  bk: '♚', bq: '♛', br: '♜', bb: '♝', bn: '♞', bp: '♟',
};
const CHESS_FILES = 'abcdefgh';

function buildChessBoard() {
  chessBoardElement.replaceChildren();
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const square = document.createElement('button');
      const rank = chessHumanColor === 'w' ? 8 - row : row + 1;
      const file = CHESS_FILES[chessHumanColor === 'w' ? col : 7 - col];
      square.type = 'button';
      square.className = `chess-square ${(row + col) % 2 ? 'dark' : 'light'}`;
      square.dataset.square = `${file}${rank}`;
      square.setAttribute('role', 'gridcell');
      square.addEventListener('click', () => handleChessSquare(square.dataset.square));
      chessBoardElement.append(square);
    }
  }
  renderChessBoard();
}

function chessKingSquare(color) {
  const rows = chessGame.board();
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const piece = rows[row][col];
      if (piece?.type === 'k' && piece.color === color) return `${CHESS_FILES[col]}${8 - row}`;
    }
  }
  return null;
}

function renderChessBoard() {
  if (!chessBoardElement.children.length) return;
  const legalFromSelection = chessSelectedSquare
    ? chessGame.moves({ square: chessSelectedSquare, verbose: true })
    : [];
  const legalTargets = new Set(legalFromSelection.map((move) => move.to));
  const lastMoveSquares = new Set(chessLastMove ? [chessLastMove.from, chessLastMove.to] : []);
  const checkedKing = chessGame.isCheck() ? chessKingSquare(chessGame.turn()) : null;
  const playerTurn = chessGame.turn() === chessHumanColor;
  for (const squareElement of chessBoardElement.children) {
    const square = squareElement.dataset.square;
    const piece = chessGame.get(square);
    const isTarget = legalTargets.has(square);
    squareElement.className = `chess-square ${(Number(square[1]) + CHESS_FILES.indexOf(square[0])) % 2 ? 'dark' : 'light'}`;
    if (lastMoveSquares.has(square)) squareElement.classList.add('last-move');
    if (chessSelectedSquare === square) squareElement.classList.add('selected-square');
    if (isTarget) squareElement.classList.add(piece ? 'legal-capture' : 'legal-target');
    if (checkedKing === square) squareElement.classList.add('in-check');
    squareElement.disabled = !chessAgent || training || learningGame || chessGame.isGameOver() || !playerTurn;
    squareElement.setAttribute('aria-label', `${piece ? `${piece.color === 'w' ? 'White' : 'Black'} ${piece.type}` : 'Empty'} ${square}${isTarget ? ', legal move' : ''}`);
    squareElement.replaceChildren();
    if (piece) {
      const glyph = document.createElement('span');
      glyph.className = `chess-piece ${piece.color === 'w' ? 'white-piece' : 'black-piece'}`;
      glyph.textContent = CHESS_GLYPHS[`${piece.color}${piece.type}`];
      squareElement.append(glyph);
    }
  }

  const status = document.querySelector('#chessGameMessage');
  if (chessGame.isCheckmate()) status.textContent = chessGame.turn() === chessHumanColor ? 'Checkmate. Learner wins.' : 'Checkmate. You win.';
  else if (chessGame.isGameOver()) status.textContent = chessGame.isDraw() ? 'Draw game' : 'Game over';
  else if (chessGame.isCheck()) status.textContent = playerTurn ? 'You are in check' : 'Learner is in check';
  else status.textContent = playerTurn ? 'Your move' : 'Learner is thinking';

  document.querySelector('#chessTurnSwatch').className = `piece-swatch ${playerTurn ? 'human-swatch' : 'learner-swatch'}`;
  const moveCount = chessGame.history().length;
  document.querySelector('#chessMoveCount').textContent = `${moveCount} ${moveCount === 1 ? 'half-move' : 'half-moves'}`;
  document.querySelector('#chessOpening').textContent = moveCount ? chessGame.history().slice(-6).join(' ') : 'Initial position';
  document.querySelector('#newChessGameButton').disabled = training || learningGame;
  document.querySelectorAll('[data-chess-side]').forEach((button) => { button.disabled = training || learningGame; });
}

function newChessGame() {
  const promotionDialog = document.querySelector('#promotionDialog');
  if (promotionDialog.open) promotionDialog.close();
  pendingChessPromotion = null;
  chessGame = new Chess();
  chessHistory = [];
  chessSelectedSquare = null;
  chessLastMove = null;
  renderChessBoard();
  if (gameMode === 'chess' && chessHumanColor === 'b' && chessAgent) window.setTimeout(playChessAgentMove, 200);
}

function selectChessSide(color) {
  if (training || learningGame || !['w', 'b'].includes(color)) return;
  chessHumanColor = color;
  document.querySelectorAll('[data-chess-side]').forEach((button) => {
    const selected = button.dataset.chessSide === color;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  newChessGame();
}

function handleChessSquare(square) {
  if (gameMode !== 'chess' || training || learningGame || chessGame.isGameOver() || chessGame.turn() !== chessHumanColor) return;
  if (chessSelectedSquare) {
    const chosenMoves = chessGame.moves({ square: chessSelectedSquare, verbose: true }).filter((move) => move.to === square);
    if (chosenMoves.length) {
      if (chosenMoves.some((move) => move.promotion)) openChessPromotionPicker(chosenMoves);
      else playChessMove(chosenMoves[0], true);
      return;
    }
  }
  const piece = chessGame.get(square);
  chessSelectedSquare = piece?.color === chessHumanColor ? square : null;
  renderChessBoard();
}

function openChessPromotionPicker(moves) {
  pendingChessPromotion = moves;
  const color = chessGame.turn();
  const names = { q: 'Queen', r: 'Rook', b: 'Bishop', n: 'Knight' };
  const dialog = document.querySelector('#promotionDialog');
  for (const button of dialog.querySelectorAll('[data-promotion]')) {
    const type = button.dataset.promotion;
    const glyph = document.createElement('span');
    glyph.className = `chess-piece ${color === 'w' ? 'white-piece' : 'black-piece'}`;
    glyph.textContent = CHESS_GLYPHS[`${color}${type}`];
    const label = document.createElement('small');
    label.textContent = names[type];
    button.replaceChildren(glyph, label);
  }
  dialog.showModal();
}

function chooseChessPromotion(piece) {
  if (!pendingChessPromotion) return;
  const move = pendingChessPromotion.find((candidate) => candidate.promotion === piece);
  pendingChessPromotion = null;
  document.querySelector('#promotionDialog').close();
  if (move) playChessMove(move, true);
}

function playChessMove(move, byHuman) {
  const state = encodeChessState(chessGame);
  const action = chessMoveIndex(move);
  const chessMove = { from: move.from, to: move.to };
  if (move.promotion) chessMove.promotion = move.promotion;
  const result = chessGame.move(chessMove);
  if (!result) return;
  chessHistory.push({ state: Array.from(state), action, color: move.color });
  chessLastMove = { from: move.from, to: move.to };
  chessSelectedSquare = null;
  showNetwork(chessAgent.getVisualization(chessGame), byHuman ? `CHESS / AFTER ${result.san}` : `CHESS AI / ${result.san}`, !byHuman);
  renderChessBoard();
  if (chessGame.isGameOver()) {
    const winner = chessGame.isCheckmate() ? (chessGame.turn() === 'w' ? 'b' : 'w') : null;
    finishChessGame(winner);
  } else if (chessGame.turn() !== chessHumanColor) {
    window.setTimeout(playChessAgentMove, 240);
  }
}

function playChessAgentMove() {
  if (gameMode !== 'chess' || training || learningGame || !chessAgent || chessGame.isGameOver() || chessGame.turn() === chessHumanColor) return;
  const move = chessAgent.chooseMove(chessGame);
  if (move) playChessMove(move, false);
}

function finishChessGame(winner) {
  if (winner === chessHumanColor) chessResultCount.human += 1;
  else if (winner) chessResultCount.learner += 1;
  else chessResultCount.draws += 1;
  learningGame = true;
  document.querySelector('#trainButton').disabled = true;
  document.querySelector('#gamesInput').disabled = true;
  updateModelStatus();
  updateScore();
  document.querySelector('#trainingStatus').textContent = 'Learning from your chess game';
  document.querySelector('#trainingNote').textContent = 'This finished chess game is training the chess policy and will be saved alongside Connect Four.';
  renderChessBoard();
  learnFromChessGame(winner);
}

async function learnFromChessGame(winner) {
  if (!chessAgent || !chessHistory.length) return;
  const history = chessHistory;
  chessHistory = [];
  try {
    worker.postMessage({
      type: 'chess-learn-game',
      gameHistory: history,
      winner,
      modelData: await exportCombinedModel(),
    });
  } catch (error) {
    learningGame = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    updateModelStatus();
    document.querySelector('#trainingStatus').textContent = 'Chess update failed';
    document.querySelector('#trainingNote').textContent = error.message;
    renderChessBoard();
  }
}

async function startChessTraining(games) {
  if (training || learningGame || !Number.isInteger(games) || games < 1 || games > 1000 || !chessAgent) return;
  training = true;
  document.querySelector('#trainButton').disabled = true;
  document.querySelector('#gamesInput').disabled = true;
  document.querySelector('#trainingStatus').textContent = 'Chess self-play preparing';
  document.querySelector('#trainingNote').textContent = 'The chess policy is training locally in the game worker.';
  document.querySelector('#progressBar').style.width = '0%';
  document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '0');
  updateModelStatus();
  renderChessBoard();
  try {
    worker.postMessage({ type: 'chess-train', games, modelData: await exportCombinedModel(), seed: Date.now() });
  } catch (error) {
    training = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    document.querySelector('#trainingStatus').textContent = 'Chess training failed';
    document.querySelector('#trainingNote').textContent = error.message;
    renderChessBoard();
  }
}

async function learnFromChessCom() {
  if (training || learningGame || !chessAgent) return;
  training = true;
  document.querySelector('#trainButton').disabled = true;
  document.querySelector('#gamesInput').disabled = true;
  document.querySelector('#trainingStatus').textContent = 'Chess.com games preparing';
  document.querySelector('#trainingNote').textContent = 'Loading top-player games from data/chesscom-top-games.json.';
  document.querySelector('#progressBar').style.width = '0%';
  document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '0');
  updateModelStatus();
  try {
    const response = await fetch('/data/chesscom-top-games.json');
    if (!response.ok) throw new Error('Chess.com dataset is not available. Run npm run scrape:chesscom first.');
    const dataset = await response.json();
    worker.postMessage({ type: 'chess-learn-pgn', pgnGames: dataset.games, modelData: await exportCombinedModel(), seed: Date.now() });
  } catch (error) {
    training = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    document.querySelector('#trainingStatus').textContent = 'Chess.com learning failed';
    document.querySelector('#trainingNote').textContent = error.message;
    renderChessBoard();
  }
}

function updateModelStatus() {
  const selectedAgent = gameMode === 'chess' ? chessAgent : agent;
  if (!selectedAgent) return;
  completedGames = selectedAgent.trainingGames + selectedAgent.observedGames;
  document.querySelector('#gamesLearned').textContent = completedGames.toLocaleString();
  document.querySelector('#networkShape').textContent = gameMode === 'chess' ? '836 - 512 - 512 - 4864' : '84 - 512 - 512 - 7';
  document.querySelector('#replayCount').textContent = selectedAgent.replay.length.toLocaleString();
  document.querySelector('#resetButton').disabled = training || learningGame || !agent || !chessAgent;
  document.querySelectorAll('[data-game-mode]').forEach((button) => {
    button.disabled = training || learningGame;
  });
}

async function exportCombinedModel() {
  const connectFourModel = await agent.exportModel();
  connectFourModel.chess = await chessAgent.exportModel();
  return connectFourModel;
}

function showCurrentNetwork(activity, signal = false) {
  if (gameMode === 'chess' && chessAgent) {
    showNetwork(chessAgent.getVisualization(chessGame), activity, signal);
  } else if (agent) {
    showNetwork(agent.getVisualization(encodeState(board, agentPlayer)), activity, signal);
  }
}

function selectGameMode(mode) {
  if (training || learningGame || !['connect-four', 'chess'].includes(mode)) return;
  gameMode = mode;
  const isChess = mode === 'chess';
  document.querySelector('#connectFourPanel').hidden = isChess;
  document.querySelector('#chessPanel').hidden = !isChess;
  document.querySelectorAll('[data-game-mode]').forEach((button) => {
    const selected = button.dataset.gameMode === mode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-selected', String(selected));
  });
  document.querySelector('#gameKicker').textContent = isChess ? 'CHESS / DEEP Q-LEARNING' : 'CONNECT FOUR / DEEP Q-LEARNING';
  document.querySelector('#gameTitle').textContent = isChess ? 'Play the chess learner.' : 'Play the neural learner.';
  document.querySelector('#gamesInput').min = isChess ? '1' : '100';
  document.querySelector('#gamesInput').max = isChess ? '1000' : '10000';
  document.querySelector('#gamesInput').step = isChess ? '1' : '100';
  document.querySelector('#gamesInput').value = isChess ? '10' : '1000';
  document.querySelector('.games-label').textContent = isChess ? 'Additional chess self-play games' : 'Additional self-play games';
  document.querySelector('#networkDescription').textContent = isChess
    ? 'The chess policy scores legal moves from the current position. Local self-play and completed matches train its own head while preserving the Connect Four policy.'
    : 'The Connect Four policy maps each board to seven move scores. Self-play adjusts its weights after wins and losses; a replay buffer and target network steady the learning.';
  document.querySelector('#trainingStatus').textContent = isChess
    ? (chessAgent?.trainingGames ? 'Chess network ready' : 'Chess policy ready to learn')
    : agent?.trainingGames ? 'Network ready' : 'First self-play session';
  document.querySelector('#trainingNote').textContent = isChess
    ? 'Chess has its own local policy under the same AI. Play a game or run chess self-play to teach it.'
    : 'Self-play weights are saved alongside chess learning in data/model.bin.';
  document.querySelector('#resetButton').textContent = isChess ? 'Reset chess policy' : 'Reset weights';
  updateModelStatus();
  updateScore();
  showCurrentNetwork(isChess ? 'CHESS POLICY / CURRENT POSITION' : 'CONNECT FOUR POLICY / CURRENT BOARD');
  if (isChess && chessAgent && !chessGame.isGameOver() && chessGame.turn() !== chessHumanColor) {
    window.setTimeout(playChessAgentMove, 200);
  }
}

function gameResult(player, row = -1, col = -1) {
  gameOver = true;
  if (player) {
    const line = winningLine(row, col, player);
    winnerCells = line;
    if (player === humanPlayer) {
      sessionScore.human += 1;
      setMessage('You win');
    } else {
      sessionScore.learner += 1;
      setMessage('Learner wins');
    }
  } else {
    sessionScore.draws += 1;
    setMessage('Draw game');
  }
  updateScore();
  drawBoard();
  learnFromPlayedGame(player);
}

function winningLine(row, col, player) {
  const directions = [[1, 0], [0, 1], [1, 1], [1, -1]];
  for (const [rowStep, colStep] of directions) {
    const cells = [[row, col]];
    for (const sign of [-1, 1]) {
      let nextRow = row + rowStep * sign;
      let nextCol = col + colStep * sign;
      while (
        nextRow >= 0 && nextRow < ROWS && nextCol >= 0 && nextCol < COLS
        && getCell(board, nextRow, nextCol) === player
      ) {
        cells.push([nextRow, nextCol]);
        nextRow += rowStep * sign;
        nextCol += colStep * sign;
      }
    }
    if (cells.length >= 4) return cells;
  }
  return [];
}

function finishMove(row, col, player) {
  if (isWinningMove(board, row, col, player)) {
    gameResult(player, row, col);
    return true;
  }
  if (legalMoves(board).length === 0) {
    gameResult(0);
    return true;
  }
  return false;
}

function playHumanMove(col) {
  if (training || learningGame || gameOver || currentPlayer !== humanPlayer || !agent) return;
  gameHistory.push({ state: Array.from(encodeState(board, humanPlayer)), action: col, player: humanPlayer });
  const row = drop(board, col, humanPlayer);
  if (row < 0) return;
  if (finishMove(row, col, humanPlayer)) return;
  currentPlayer = agentPlayer;
  setMessage('Learner is thinking');
  drawBoard();
  showCurrentNetwork('YOUR MOVE / CURRENT BOARD');
  window.setTimeout(playLearnerMove, 180);
}

function playLearnerMove() {
  if (training || learningGame || gameOver || currentPlayer !== agentPlayer || !agent) return;
  const state = encodeState(board, agentPlayer);
  const visualization = agent.getVisualization(state);
  const col = agent.chooseAction(board, agentPlayer);
  gameHistory.push({ state: Array.from(state), action: col, player: agentPlayer });
  showNetwork(visualization, `AI SELECTED COLUMN ${col + 1}`, true);
  const row = drop(board, col, agentPlayer);
  if (row < 0) return;
  if (finishMove(row, col, agentPlayer)) return;
  currentPlayer = humanPlayer;
  setMessage('Your turn');
  drawBoard();
}

function newGame() {
  board = createBoard();
  currentPlayer = openingPlayer === 'human' ? humanPlayer : agentPlayer;
  gameOver = false;
  winnerCells = [];
  gameHistory = [];
  setMessage(training ? 'Training in background' : !agent ? 'Loading neural network' : currentPlayer === agentPlayer ? 'Learner opens' : 'Your turn');
  drawBoard();
  if (!training && !learningGame && agent && currentPlayer === agentPlayer) window.setTimeout(playLearnerMove, 180);
}

function selectStarter(starter) {
  if (training || learningGame || !['human', 'learner'].includes(starter)) return;
  openingPlayer = starter;
  ({ human: humanPlayer, agent: agentPlayer } = playersForOpening(starter));
  document.querySelectorAll('[data-starter]').forEach((button) => {
    const selected = button.dataset.starter === starter;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  newGame();
  showNetwork(agent.getVisualization(encodeState(board, agentPlayer)), `${starter === 'learner' ? 'LEARNER' : 'YOU'} START / CURRENT BOARD`);
}

async function learnFromPlayedGame(winner) {
  if (!agent || !gameHistory.length) return;
  const history = gameHistory;
  gameHistory = [];
  learningGame = true;
  document.querySelector('#trainButton').disabled = true;
  document.querySelector('#gamesInput').disabled = true;
  document.querySelector('#trainingStatus').textContent = 'Learning from your match';
  document.querySelector('#trainingNote').textContent = 'The finished game is being added to the neural network and saved to the workspace.';
  drawBoard();
  try {
    const modelData = await exportCombinedModel();
    worker.postMessage({ type: 'learn-game', gameHistory: history, winner, modelData });
  } catch (error) {
    learningGame = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    document.querySelector('#trainingStatus').textContent = 'Match update failed';
    document.querySelector('#trainingNote').textContent = error.message;
    drawBoard();
  }
}

async function startTraining(games) {
  if (gameMode === 'chess') {
    startChessTraining(Number(document.querySelector('#gamesInput').value));
    return;
  }
  if (training || learningGame || !Number.isInteger(games) || games < 100 || games > 10000 || !agent) return;
  training = true;
  document.querySelector('#trainButton').disabled = true;
  document.querySelector('#gamesInput').disabled = true;
  document.querySelector('#trainingStatus').textContent = 'Preparing neural network';
  document.querySelector('#trainingNote').textContent = 'The game pauses while self-play runs in a local worker.';
  document.querySelector('#progressBar').style.width = '0%';
  document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '0');
  drawBoard();
  try {
    const modelData = await exportCombinedModel();
    worker.postMessage({ type: 'connect-four-train', games, modelData, seed: completedGames + Date.now() });
  } catch (error) {
    training = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    document.querySelector('#trainingStatus').textContent = 'Training failed';
    document.querySelector('#trainingNote').textContent = error.message;
    drawBoard();
  }
}

worker.addEventListener('message', async (event) => {
  const result = event.data;
  if (result.type === 'progress' || result.type === 'game-progress') {
    const percentage = Math.round((result.completed / result.games) * 100);
    document.querySelector('#progressBar').style.width = `${percentage}%`;
    document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', String(percentage));
    const gameName = result.gameMode === 'chess' ? 'chess' : 'Connect Four';
    document.querySelector('#trainingStatus').textContent = result.type === 'progress'
      ? `${percentage}% - ${gameName} self-play`
      : `Updating ${gameName} match ${percentage}%`;
    document.querySelector('#gamesLearned').textContent = (result.trainingGames + result.observedGames).toLocaleString();
    document.querySelector('#replayCount').textContent = result.replaySize.toLocaleString();
    document.querySelector('#lossValue').textContent = Number.isFinite(result.loss) ? result.loss.toFixed(3) : '--';
    if (result.visualization) showNetwork(result.visualization, result.type === 'progress' ? `${gameName} SELF-PLAY ${percentage}%` : `${gameName} MATCH UPDATE ${percentage}%`);
    if (result.modelData) {
      // Keep the UI in sync during training, but persist weights only when a
      // training run or completed game finishes. Saving the full model on every
      // checkpoint makes large models take a very long time over the network.
      // queueModelSave(result.modelData).catch((error) => {
      //   document.querySelector('#saveStatus').textContent = 'SAVE FAILED';
      //   document.querySelector('#trainingNote').textContent = error.message;
      // });
    }
    return;
  }
  if (result.type === 'error') {
    training = false;
    learningGame = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    document.querySelector('#trainingStatus').textContent = 'Training failed';
    document.querySelector('#trainingNote').textContent = result.message;
    drawBoard();
    renderChessBoard();
    return;
  }
  if (result.type === 'complete' || result.type === 'game-learned') {
    try {
      await queueModelSave(result.modelData);
    } catch (error) {
      document.querySelector('#saveStatus').textContent = 'SAVE FAILED';
      document.querySelector('#trainingNote').textContent = `The model learned in memory but could not be saved: ${error.message}`;
    }
    const [updatedAgent, updatedChessAgent] = await Promise.all([
      ConnectFourAgent.create(result.modelData),
      ChessAgent.create(result.modelData.chess),
    ]);
    agent.dispose();
    chessAgent?.dispose();
    agent = updatedAgent;
    chessAgent = updatedChessAgent;
    updateModelStatus();
    showCurrentNetwork(`${result.gameMode === 'chess' ? 'CHESS' : 'CONNECT FOUR'} NETWORK / UPDATED`);
    training = false;
    learningGame = false;
    document.querySelector('#trainButton').disabled = false;
    document.querySelector('#gamesInput').disabled = false;
    document.querySelector('#trainingStatus').textContent = result.type === 'complete'
      ? `${result.gameMode === 'chess' ? 'Chess policy' : 'Connect Four policy'} ready`
      : `${result.gameMode === 'chess' ? 'Chess game' : 'Connect Four game'} learned and saved`;
    document.querySelector('#progressBar').style.width = '100%';
    document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '100');
    if (result.type === 'complete' && document.querySelector('#saveStatus').textContent !== 'SAVE FAILED') {
      document.querySelector('#trainingNote').textContent = 'Both game policies are saved together in data/model.bin in this workspace.';
    } else if (document.querySelector('#saveStatus').textContent !== 'SAVE FAILED') {
      document.querySelector('#trainingNote').textContent = 'Your completed match updated the network saved in data/model.bin.';
    }
    drawBoard();
    renderChessBoard();
  }
});

document.querySelector('#newGameButton').addEventListener('click', newGame);
document.querySelector('#newChessGameButton').addEventListener('click', newChessGame);
document.querySelectorAll('[data-game-mode]').forEach((button) => {
  button.addEventListener('click', () => selectGameMode(button.dataset.gameMode));
});
document.querySelectorAll('[data-chess-side]').forEach((button) => {
  button.addEventListener('click', () => selectChessSide(button.dataset.chessSide));
});
const promotionDialog = document.querySelector('#promotionDialog');
promotionDialog.querySelectorAll('[data-promotion]').forEach((button) => {
  button.addEventListener('click', () => chooseChessPromotion(button.dataset.promotion));
});
promotionDialog.addEventListener('close', () => { pendingChessPromotion = null; });
document.querySelectorAll('[data-starter]').forEach((button) => {
  button.addEventListener('click', () => selectStarter(button.dataset.starter));
});
document.querySelector('#trainButton').addEventListener('click', () => {
  startTraining(Number(document.querySelector('#gamesInput').value));
});
document.querySelector('#learnChessComButton').addEventListener('click', () => {
  learnFromChessCom();
});
document.querySelector('#resetButton').addEventListener('click', async () => {
  const gameName = gameMode === 'chess' ? 'chess policy' : 'Connect Four policy';
  if (training || learningGame || !window.confirm(`Reset only the ${gameName}? The other game policy will be kept.`)) return;
  if (gameMode === 'chess') {
    const previousChessAgent = chessAgent;
    chessAgent = await ChessAgent.create();
    previousChessAgent?.dispose();
  } else {
    const previousAgent = agent;
    agent = await ConnectFourAgent.create();
    previousAgent?.dispose();
  }
  await queueModelSave(await exportCombinedModel());
  updateModelStatus();
  showCurrentNetwork(`NEW ${gameName.toUpperCase()} / UNTRAINED`);
  if (gameMode === 'chess') {
    newChessGame();
    startChessTraining(5);
  } else {
    newGame();
    startTraining(FIRST_TRAINING_GAMES);
  }
});

buildBoard();
buildChessBoard();
initializeNetworkControls();
updateScore();
document.querySelector('#trainButton').disabled = true;
document.querySelector('#trainingStatus').textContent = 'Loading local network';

new ResizeObserver(() => drawNetworkFrame(true)).observe(document.querySelector('#networkCanvas'));

loadSavedModel().then(async (savedModel) => {
  const [createdAgent, createdChessAgent] = await Promise.all([
    ConnectFourAgent.create(savedModel),
    ChessAgent.create(savedModel?.chess),
  ]);
  agent = createdAgent;
  chessAgent = createdChessAgent;
  updateModelStatus();
  document.querySelector('#trainButton').disabled = false;
  showCurrentNetwork(gameMode === 'chess' ? 'CHESS POLICY / CURRENT POSITION' : 'CONNECT FOUR POLICY / CURRENT BOARD');
  if (gameMode === 'chess') {
    document.querySelector('#trainingStatus').textContent = 'Network ready';
    document.querySelector('#trainingNote').textContent = chessAgent.trainingGames > 0
      ? 'Loaded weights from data/model.bin in this workspace.'
      : 'Run self-play to teach the chess policy.';
    document.querySelector('#progressBar').style.width = '100%';
    document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '100');
    setMessage('Your turn');
  } else if (agent.trainingGames === 0) {
    document.querySelector('#trainingStatus').textContent = 'Network ready';
    document.querySelector('#trainingNote').textContent = 'Run self-play to teach the Connect Four policy.';
    document.querySelector('#progressBar').style.width = '0%';
    document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '0');
    setMessage('Your turn');
  } else {
    document.querySelector('#trainingStatus').textContent = 'Network ready';
    document.querySelector('#trainingNote').textContent = 'Loaded weights from data/model.bin in this workspace.';
    document.querySelector('#progressBar').style.width = '100%';
    document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', '100');
    setMessage('Your turn');
  }
  if (openingPlayer === 'learner' && agent.trainingGames > 0) window.setTimeout(playLearnerMove, 180);
  if (gameMode === 'chess' && chessHumanColor === 'b') window.setTimeout(playChessAgentMove, 200);
  drawBoard();
  renderChessBoard();
}).catch((error) => {
  document.querySelector('#trainingStatus').textContent = 'Network failed to load';
  document.querySelector('#trainingNote').textContent = `Could not load the workspace model: ${error.message}`;
  document.querySelector('#saveStatus').textContent = 'WORKSPACE FILE UNAVAILABLE';
});