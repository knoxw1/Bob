import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { createModelStore } from './src/model-store.js';

function localModelStore() {
  const middleware = createModelStore(resolve(process.cwd(), 'data', 'model.json'));
  return {
    name: 'local-connect-four-model-store',
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}

export default defineConfig({
  plugins: [localModelStore()],
});