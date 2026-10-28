export const MODEL_WEIGHT_SHAPES = [[84, 512], [512], [512, 512], [512], [512, 512], [512], [512, 7], [7]];
export const CHESS_MODEL_WEIGHT_SHAPES = [[836, 512], [512], [512, 512], [512], [512, 512], [512], [512, 4864], [4864]];

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
  if (modelData.chess !== undefined) {
    const chess = modelData.chess;
    if (!chess || chess.version !== 1 || !Array.isArray(chess.weights)) throw new Error('Invalid chess model data.');
    validateWeights(chess.weights, CHESS_MODEL_WEIGHT_SHAPES, 'Chess');
    for (const field of ['trainingGames', 'observedGames']) {
      if (chess[field] !== undefined && (!Number.isSafeInteger(chess[field]) || chess[field] < 0)) {
        throw new Error(`Invalid chess ${field} value.`);
      }
    }
    validated.chess = {
      version: 1,
      trainingGames: chess.trainingGames || 0,
      observedGames: chess.observedGames || 0,
      weights: chess.weights,
    };
  }
  return validated;
}

function concatChunks(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

class Writer {
  constructor() { this.chunks = []; }
  push(bytes) { this.chunks.push(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)); }
  u8(value) { const b = new Uint8Array(1); b[0] = value & 0xff; this.push(b); }
  u32(value) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, value >>> 0, true); this.push(b); }
  f32(value) { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, value, true); this.push(b); }
  finish() { return concatChunks(this.chunks); }
}

function writeWeights(writer, weights) {
  writer.u32(weights.length);
  for (const weight of weights) {
    writer.u8(weight.shape.length);
    for (const dimension of weight.shape) writer.u32(dimension);
    writer.u32(weight.values.length);
    for (const value of weight.values) writer.f32(value);
  }
}

export function encodeModel(modelData) {
  const valid = validateModelData(modelData);
  const writer = new Writer();
  writer.push(new TextEncoder().encode('NGL1'));
  writer.u32(valid.version);
  writer.u32(valid.trainingGames);
  writer.u32(valid.observedGames);
  writeWeights(writer, valid.weights);
  if (valid.chess) {
    writer.u8(1);
    writer.u32(valid.chess.trainingGames);
    writer.u32(valid.chess.observedGames);
    writeWeights(writer, valid.chess.weights);
  } else {
    writer.u8(0);
  }
  return writer.finish();
}

class Reader {
  constructor(buffer) {
    this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    this.offset = 0;
  }
  bytes(length) {
    const result = new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, length);
    this.offset += length;
    return result;
  }
  u8() { const value = this.view.getUint8(this.offset); this.offset += 1; return value; }
  u32() { const value = this.view.getUint32(this.offset, true); this.offset += 4; return value; }
  f32() { const value = this.view.getFloat32(this.offset, true); this.offset += 4; return value; }
  remaining() { return this.view.byteLength - this.offset; }
}

function readWeights(reader) {
  const count = reader.u32();
  const weights = [];
  for (let i = 0; i < count; i += 1) {
    const rank = reader.u8();
    const shape = [];
    for (let axis = 0; axis < rank; axis += 1) shape.push(reader.u32());
    const valueCount = reader.u32();
    const values = new Array(valueCount);
    for (let valueIndex = 0; valueIndex < valueCount; valueIndex += 1) {
      values[valueIndex] = reader.f32();
    }
    weights.push({ shape, values });
  }
  return weights;
}

export function decodeModel(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const reader = new Reader(bytes);
  const magic = new TextDecoder().decode(reader.bytes(4));
  if (magic !== 'NGL1') throw new Error('Invalid model file magic.');
  const version = reader.u32();
  const trainingGames = reader.u32();
  const observedGames = reader.u32();
  const weights = readWeights(reader);
  const hasChess = reader.u8();
  let chess;
  if (hasChess) {
    chess = {
      version: 1,
      trainingGames: reader.u32(),
      observedGames: reader.u32(),
      weights: readWeights(reader),
    };
  }
  if (reader.remaining() !== 0) throw new Error('Invalid model file length.');
  const modelData = { version, trainingGames, observedGames, weights };
  if (chess) modelData.chess = chess;
  return validateModelData(modelData);
}
