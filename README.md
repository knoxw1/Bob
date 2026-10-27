# Neural Game Lab

A local game-learning app with chess and Connect Four. One local AI interface and trainer manage a separate TensorFlow.js policy for each game's different board encoding and legal action space. Connect Four uses self-play Q-learning; chess uses `chess.js` for rules/legal moves and a local neural Q-policy trained with self-play and completed matches.

Choose a game from the tabs. Chess supports playing White or Black, including the learner opening when you choose Black. Connect Four supports either side starting.

## Run locally

Requires Node.js 20 or newer.

```sh
npm install
npm run dev
```

Open the local URL Vite prints. The app makes no cloud calls and needs no API key. TensorFlow.js runs on the CPU in a browser worker, keeping the game responsive while training.

On ChromeOS or Microsoft Edge, install the site as a standalone app from the browser's install option. It opens in its own app window and remains interactive. It must still be served by the Codespace; standard ChromeOS/Bing wallpapers cannot host an arbitrary interactive webpage.

## Where learning is saved

The Codespace's `data/model.json` file is the source of truth for the network weights. The local Vite API loads it on startup and atomically writes updates. Self-play saves checkpoints every 50 games and at the end of each training run. Each completed player-vs-AI game also updates and saves the weights. The file is excluded from Git so learned weights remain local to this Codespace.

The browser displays a live network diagram using sampled weights and activations from the currently selected policy. Each policy is stored under the shared `data/model.json` file, so training one preserves the other. Expand the diagram to full screen, zoom or drag to pan, and hover or click a neuron or connection to inspect it.

## Notes

These are small learning demos, not perfect game engines. Chess uses legal moves, but the learned policy is not a strong chess engine and improves with local self-play and completed matches. Reset clears only the currently selected game's policy; the other game's weights remain saved.

```sh
npm test
npm run build
```
## Deploy to Render

This repository includes a production server (`server.js`) and a Render blueprint (`render.yaml`), so it can be deployed as a Render Web Service.

1. Push this repository to GitHub (Render deploys from a Git repository).
2. In Render, choose **New +** -> **Blueprint** (or **Web Service**) and select this repository.
3. If Render asks for commands, use:
   - Build command: `npm install && npm run build`
   - Start command: `npm start`
4. Use the Node 20 environment (the blueprint sets `NODE_VERSION=20`).
5. Deploy. Render will run the Vite build, then start `node server.js`.
6. Render provides a URL such as `https://neural-game-lab.onrender.com`. Open that URL to play chess and Connect Four against the AI.

Notes for Render:
- The production server listens on `process.env.PORT` and binds to `0.0.0.0`, as Render requires.
- `/healthz` is available for Render health checks.
- Learned weights are stored in `data/model.json`. Render's filesystem is ephemeral, so weights reset when the service restarts or redeploys. To persist them, add a Render Disk mounted in the project directory and set `MODEL_PATH` to a file on that disk (for example `/var/data/model.json`).
- The Chess.com dataset at `data/chesscom-top-games.json` is included in the build and is available at `/data/chesscom-top-games.json`.
- The free tier may sleep after inactivity; the first visit after sleep can take a moment to wake up.
