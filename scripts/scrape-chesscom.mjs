import fs from 'node:fs/promises';
import path from 'node:path';

const userAgent = 'NeuralGameLab/1.0 (local chess learning)';
const now = new Date();
const monthCount = Number(process.env.CHESSCOM_MONTHS || 1);
const months = [];
for (let offset = 0; offset < monthCount; offset += 1) {
  const date = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - offset, 1));
  months.push({ year: date.getUTCFullYear(), month: date.getUTCMonth() + 1 });
}

async function fetchJson(url, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { headers: { 'User-Agent': userAgent }, signal: controller.signal });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

const leaderboard = await fetchJson('https://api.chess.com/pub/leaderboards');
const playerSet = new Set();
for (const key of ['live_rapid', 'live_blitz', 'live_bullet']) {
  for (const entry of leaderboard[key] || []) {
    if (entry?.username) playerSet.add(entry.username);
  }
}
const allPlayers = [...playerSet];
const start = Number(process.env.CHESSCOM_START || 0);
const requestedMax = Number(process.env.CHESSCOM_MAX_PLAYERS || allPlayers.length);
const players = allPlayers.slice(start, start + requestedMax);

const games = [];
const seen = new Set();
const maxGamesPerPlayer = Number(process.env.CHESSCOM_GAMES_PER_PLAYER || 10);

for (let playerIndex = 0; playerIndex < players.length; playerIndex += 1) {
  const username = players[playerIndex];
  console.log(`[${start + playerIndex + 1}/${allPlayers.length}] Fetching ${username}...`);
  let playerGames = 0;
  for (const { year, month } of months) {
    if (playerGames >= maxGamesPerPlayer) break;
    const mm = String(month).padStart(2, '0');
    try {
      const data = await fetchJson(`https://api.chess.com/pub/player/${encodeURIComponent(username)}/games/${year}/${mm}`);
      for (const game of data.games || []) {
        if (playerGames >= maxGamesPerPlayer) break;
        if (!game?.pgn || seen.has(game.url)) continue;
        seen.add(game.url);
        const result = game.white?.result && game.black?.result
          ? (game.white.result === 'win' ? '1-0' : game.black.result === 'win' ? '0-1' : '1/2-1/2')
          : (game.pgn.match(/\[Result "([^"]+)"\]/)?.[1] || '*');
        games.push({
          white: game.white?.username || username,
          black: game.black?.username || username,
          result,
          pgn: game.pgn,
        });
        playerGames += 1;
      }
    } catch (error) {
      console.warn(`  Skipping ${username} ${year}-${mm}: ${error.message}`);
    }
    await sleep(100);
  }
}

const output = {
  source: 'chess.com public API',
  players,
  generatedAt: new Date().toISOString(),
  gameCount: games.length,
  games,
};

const outPath = path.join(process.cwd(), 'data', 'chesscom-top-games.json');
await fs.mkdir(path.dirname(outPath), { recursive: true });

let finalOutput = output;
if (process.env.CHESSCOM_APPEND === '1') {
  try {
    const existing = JSON.parse(await fs.readFile(outPath, 'utf8'));
    const existingPgns = new Set(existing.games.map((game) => game.pgn));
    const mergedGames = [...existing.games];
    for (const game of games) {
      if (!existingPgns.has(game.pgn)) {
        mergedGames.push(game);
        existingPgns.add(game.pgn);
      }
    }
    finalOutput = {
      ...existing,
      players: [...new Set([...(existing.players || []), ...players])],
      generatedAt: new Date().toISOString(),
      gameCount: mergedGames.length,
      games: mergedGames,
    };
  } catch {
    // Keep the fresh output if no existing dataset is available.
  }
}

await fs.writeFile(outPath, JSON.stringify(finalOutput, null, 2));
console.log(`Saved ${finalOutput.gameCount} games to ${outPath}`);
