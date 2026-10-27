import * as tf from '@tensorflow/tfjs';
import { ConnectFourAgent } from './agent.js';
import { ChessAgent } from './chess-agent.js';

self.onmessage = async (event) => {
  const { type, games, gameHistory, winner, modelData, seed, pgnGames } = event.data;
  let agent;
  try {
    await tf.setBackend('cpu');
    await tf.ready();
    const isChess = type.startsWith('chess-');
    agent = isChess
      ? await ChessAgent.create(modelData?.chess)
      : await ConnectFourAgent.create(modelData);

    if (type === 'chess-train') {
      await agent.trainSelfPlay(games, (progress) => {
        const { modelData: chessModelData, ...status } = progress;
        self.postMessage({ type: 'progress', gameMode: 'chess', ...status,
          ...(chessModelData ? { modelData: { ...modelData, chess: chessModelData } } : {}) });
      }, seed);
    } else if (type === 'chess-learn-pgn') {
      await agent.learnFromChessComGames(pgnGames, (progress) => {
        self.postMessage({ type: 'progress', gameMode: 'chess', ...progress });
      });
    } else if (type === 'chess-learn-game') {
      await agent.learnFromGame(gameHistory, winner, (progress) => {
        self.postMessage({ type: 'game-progress', gameMode: 'chess', ...progress });
      });
    } else if (type === 'learn-game') {
      await agent.learnFromGame(gameHistory, winner, (progress) => {
        self.postMessage({ type: 'game-progress', gameMode: 'connect-four', ...progress });
      });
    } else {
      await agent.trainSelfPlay(games, (progress) => {
        const { modelData: connectFourModelData, ...status } = progress;
        self.postMessage({ type: 'progress', gameMode: 'connect-four', ...status,
          ...(connectFourModelData ? { modelData: { ...connectFourModelData, ...(modelData?.chess ? { chess: modelData.chess } : {}) } } : {}) });
      }, seed);
    }

    const result = await agent.exportModel();
    const combinedModel = isChess ? { ...modelData, chess: result } : { ...result, ...(modelData?.chess ? { chess: modelData.chess } : {}) };
    self.postMessage({
      type: type.endsWith('learn-game') ? 'game-learned' : 'complete',
      gameMode: isChess ? 'chess' : 'connect-four',
      modelData: combinedModel,
    });
  } catch (error) {
    self.postMessage({ type: 'error', message: error.message || 'Local training failed.' });
  } finally {
    agent?.dispose();
  }
};