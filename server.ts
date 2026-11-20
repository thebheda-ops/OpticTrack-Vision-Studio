import express from 'express';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { createApiApp } from './backend/index.ts';

const PORT = 3000;
const ROOT_DIR = process.cwd();
const MEDIAPIPE_WASM_DIR = path.resolve(
  ROOT_DIR,
  'node_modules/@mediapipe/tasks-vision/wasm'
);

async function startServer() {
  const app = createApiApp();

  // Serve local MediaPipe WASM files in standalone dev/prod Express mode
  if (fs.existsSync(MEDIAPIPE_WASM_DIR)) {
    app.use('/mediapipe/wasm', express.static(MEDIAPIPE_WASM_DIR));
  }

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`OpticTrack server listening on http://localhost:${PORT}`);
  });
}

startServer();
