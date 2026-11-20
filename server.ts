import express from 'express';
import fs from 'fs';
import path from 'path';
import { createServer as createViteServer } from 'vite';

const PORT = 3000;
const ROOT_DIR = process.cwd();
const LOGS_DIR = path.resolve(ROOT_DIR, 'logs');
const LOG_FILE_PATH = path.resolve(LOGS_DIR, 'detections.txt');
const PYTHON_DIR = path.resolve(ROOT_DIR, 'python');
const MEDIAPIPE_WASM_DIR = path.resolve(
  ROOT_DIR,
  'node_modules/@mediapipe/tasks-vision/wasm'
);

const LOG_HEADER = [
  '# OpticTrack Detection Timestamp Log',
  '# Format: [ISO_TIMESTAMP] TYPE=<type> OBJECT=<label> CONF=<score> BBOX=[x,y,w,h] FPS=<fps> [MOTION=<motion>]',
  '',
].join('\n');

function ensureLogFile(): void {
  if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true });
  }
  if (!fs.existsSync(LOG_FILE_PATH)) {
    fs.writeFileSync(LOG_FILE_PATH, LOG_HEADER, 'utf-8');
  }
}

interface DetectionLogPayload {
  timestamp: string;
  entryType?: 'object' | 'hand_gesture' | 'hand_motion';
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  fps: number;
  motion?: string;
}

interface ParsedLogEntry {
  id: string;
  timestamp: string;
  entryType: 'object' | 'hand_gesture' | 'hand_motion';
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  fps: number;
  motion?: string;
  rawLine: string;
}

function parseLogContent(rawText: string): ParsedLogEntry[] {
  const lines = rawText.split(/\r?\n/);
  const entries: ParsedLogEntry[] = [];
  const lineRegex =
    /^\[([^\]]+)\](?:\s+TYPE=([^\s]+))?\s+OBJECT=([^\s]+)\s+CONF=([0-9.]+)\s+BBOX=\[(-?\d+),(-?\d+),(-?\d+),(-?\d+)\]\s+FPS=([0-9.]+)(?:\s+MOTION=([^\s]+))?/;

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      return;
    }
    const match = trimmed.match(lineRegex);
    if (!match) {
      return;
    }
    const rawType = match[2];
    const entryType: ParsedLogEntry['entryType'] =
      rawType === 'hand_gesture' || rawType === 'hand_motion'
        ? rawType
        : 'object';

    entries.push({
      id: `${index}-${match[1]}-${match[3]}`,
      timestamp: match[1],
      entryType,
      label: match[3],
      confidence: parseFloat(match[4]),
      bbox: [
        parseInt(match[5], 10),
        parseInt(match[6], 10),
        parseInt(match[7], 10),
        parseInt(match[8], 10),
      ],
      fps: parseFloat(match[9]),
      motion: match[10] || undefined,
      rawLine: trimmed,
    });
  });

  return entries;
}

async function startServer() {
  ensureLogFile();

  const app = express();
  app.use(express.json({ limit: '2mb' }));

  // Serve local MediaPipe WASM files so hand tracking loads fast without external CDN drift
  if (fs.existsSync(MEDIAPIPE_WASM_DIR)) {
    app.use('/mediapipe/wasm', express.static(MEDIAPIPE_WASM_DIR));
  }

  // Read local detections.txt log file
  app.get('/api/logs', (_req, res) => {
    try {
      ensureLogFile();
      const rawText = fs.readFileSync(LOG_FILE_PATH, 'utf-8');
      const stats = fs.statSync(LOG_FILE_PATH);
      const entries = parseLogContent(rawText);
      res.json({
        filePath: 'logs/detections.txt',
        byteSize: stats.size,
        totalEntries: entries.length,
        entries: entries.slice(-500).reverse(),
        rawText,
      });
    } catch {
      res.status(500).json({ error: 'Failed to read detection log file.' });
    }
  });

  // Append timestamped detections (objects, hand gestures, hand movements) to local detections.txt
  app.post('/api/logs', (req, res) => {
    try {
      ensureLogFile();
      const entries = req.body?.entries;
      if (!Array.isArray(entries) || entries.length === 0) {
        res.status(400).json({ error: 'Missing entries array.' });
        return;
      }

      const formattedLines = entries
        .filter(
          (item: unknown): item is DetectionLogPayload =>
            Boolean(item) &&
            typeof (item as DetectionLogPayload).timestamp === 'string' &&
            typeof (item as DetectionLogPayload).label === 'string' &&
            typeof (item as DetectionLogPayload).confidence === 'number' &&
            Array.isArray((item as DetectionLogPayload).bbox) &&
            (item as DetectionLogPayload).bbox.length === 4 &&
            typeof (item as DetectionLogPayload).fps === 'number'
        )
        .map((item) => {
          const entryType = item.entryType ?? 'object';
          const safeLabel = item.label.replace(/\s+/g, '_');
          const [x, y, w, h] = item.bbox.map((v) => Math.round(v));
          const motionSuffix = item.motion
            ? ` MOTION=${item.motion.replace(/\s+/g, '_')}`
            : '';
          return `[${item.timestamp}] TYPE=${entryType} OBJECT=${safeLabel} CONF=${item.confidence.toFixed(3)} BBOX=[${x},${y},${w},${h}] FPS=${item.fps.toFixed(1)}${motionSuffix}`;
        });

      if (formattedLines.length === 0) {
        res.status(400).json({ error: 'No valid detection entries provided.' });
        return;
      }

      fs.appendFileSync(LOG_FILE_PATH, `${formattedLines.join('\n')}\n`, 'utf-8');
      const stats = fs.statSync(LOG_FILE_PATH);
      res.json({
        appended: formattedLines.length,
        byteSize: stats.size,
      });
    } catch {
      res.status(500).json({ error: 'Failed to append detection log entries.' });
    }
  });

  // Clear local detections.txt
  app.delete('/api/logs', (_req, res) => {
    try {
      fs.writeFileSync(LOG_FILE_PATH, LOG_HEADER, 'utf-8');
      res.json({ cleared: true });
    } catch {
      res.status(500).json({ error: 'Failed to clear detection log file.' });
    }
  });

  // Download local detections.txt
  app.get('/api/logs/download', (_req, res) => {
    try {
      ensureLogFile();
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Disposition', 'attachment; filename="detections.txt"');
      res.sendFile(LOG_FILE_PATH);
    } catch {
      res.status(500).json({ error: 'Failed to download detection log file.' });
    }
  });

  // Serve Python source files for inspection/download
  app.get('/api/python/:filename', (req, res) => {
    const allowedFiles = new Set([
      'object_detector.py',
      'setup_models.py',
      'requirements.txt',
      'README.md',
    ]);
    const filename = req.params.filename;
    if (!allowedFiles.has(filename)) {
      res.status(404).json({ error: 'File not found.' });
      return;
    }
    const targetPath = path.resolve(PYTHON_DIR, filename);
    if (!fs.existsSync(targetPath)) {
      res.status(404).json({ error: 'File does not exist.' });
      return;
    }
    const content = fs.readFileSync(targetPath, 'utf-8');
    res.json({ filename, content });
  });

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
