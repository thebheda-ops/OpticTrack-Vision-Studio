import express from 'express';
import fs from 'fs';
import path from 'path';

const LOG_HEADER = [
  '# OpticTrack Detection Timestamp Log',
  '# Format: [ISO_TIMESTAMP] TYPE=<type> OBJECT=<label> CONF=<score> BBOX=[x,y,w,h] FPS=<fps> [MOTION=<motion>]',
  '',
].join('\n');

const ALLOWED_PYTHON_FILES = new Set([
  'object_detector.py',
  'setup_models.py',
  'requirements-desktop.txt',
  'requirements.txt',
  'README.md',
]);

function resolveLogsDir(): string {
  const candidates = [
    path.resolve(process.cwd(), 'logs'),
    path.resolve(process.cwd(), '..', 'logs'),
    '/tmp/optictrack-logs',
  ];

  for (const dir of candidates) {
    try {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.accessSync(dir, fs.constants.W_OK);
      return dir;
    } catch {
      // Try next writable directory (e.g. /tmp on serverless runtimes)
    }
  }
  return '/tmp/optictrack-logs';
}

function resolveLocalPythonDir(): string | null {
  const candidates = [
    path.resolve(process.cwd(), 'python'),
    path.resolve(process.cwd(), '..', 'python'),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(dir)) {
      return dir;
    }
  }
  return null;
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

export function createApiApp(): express.Express {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  const ensureLogFile = (): string => {
    const logsDir = resolveLogsDir();
    if (!fs.existsSync(logsDir)) {
      fs.mkdirSync(logsDir, { recursive: true });
    }
    const logFilePath = path.resolve(logsDir, 'detections.txt');
    if (!fs.existsSync(logFilePath)) {
      fs.writeFileSync(logFilePath, LOG_HEADER, 'utf-8');
    }
    return logFilePath;
  };

  app.get('/api/health', async (_req, res) => {
    const pythonBaseUrl = process.env.PYTHON_SERVICE_URL;
    let pythonReachable = false;
    if (pythonBaseUrl) {
      try {
        const probe = await fetch(new URL('health', pythonBaseUrl));
        pythonReachable = probe.ok;
      } catch {
        pythonReachable = false;
      }
    }
    res.json({
      status: 'ok',
      pythonServiceBound: Boolean(pythonBaseUrl),
      pythonReachable,
    });
  });

  // Read detection log file
  app.get('/api/logs', (_req, res) => {
    try {
      const logFilePath = ensureLogFile();
      const rawText = fs.readFileSync(logFilePath, 'utf-8');
      const stats = fs.statSync(logFilePath);
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

  // Append timestamped detections (objects, hand gestures, hand movements)
  app.post('/api/logs', (req, res) => {
    try {
      const logFilePath = ensureLogFile();
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

      fs.appendFileSync(logFilePath, `${formattedLines.join('\n')}\n`, 'utf-8');
      const stats = fs.statSync(logFilePath);
      res.json({
        appended: formattedLines.length,
        byteSize: stats.size,
      });
    } catch {
      res.status(500).json({ error: 'Failed to append detection log entries.' });
    }
  });

  // Clear detection log
  app.delete('/api/logs', (_req, res) => {
    try {
      const logFilePath = ensureLogFile();
      fs.writeFileSync(logFilePath, LOG_HEADER, 'utf-8');
      res.json({ cleared: true });
    } catch {
      res.status(500).json({ error: 'Failed to clear detection log file.' });
    }
  });

  // Download detection log
  app.get('/api/logs/download', (_req, res) => {
    try {
      const logFilePath = ensureLogFile();
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Disposition', 'attachment; filename="detections.txt"');
      res.sendFile(logFilePath);
    } catch {
      res.status(500).json({ error: 'Failed to download detection log file.' });
    }
  });

  // Fetch Python source files via Vercel internal service binding (PYTHON_SERVICE_URL)
  // with local filesystem fallback for standalone single-process dev.
  app.get('/api/python/:filename', async (req, res) => {
    const filename = req.params.filename;
    if (!ALLOWED_PYTHON_FILES.has(filename)) {
      res.status(404).json({ error: 'File not found.' });
      return;
    }

    const pythonServiceUrl = process.env.PYTHON_SERVICE_URL;
    if (pythonServiceUrl) {
      try {
        const upstreamUrl = new URL(
          `sources/${encodeURIComponent(filename)}`,
          pythonServiceUrl
        );
        const upstreamRes = await fetch(upstreamUrl);
        if (!upstreamRes.ok) {
          res.status(upstreamRes.status).json({ error: 'Upstream Python file not found.' });
          return;
        }
        const payload = await upstreamRes.json();
        res.json(payload);
        return;
      } catch {
        res.status(502).json({ error: 'Failed to reach internal Python service.' });
        return;
      }
    }

    const localPythonDir = resolveLocalPythonDir();
    if (!localPythonDir) {
      res.status(503).json({
        error: 'PYTHON_SERVICE_URL is not configured and local python/ directory is unavailable.',
      });
      return;
    }

    const targetPath = path.resolve(localPythonDir, filename);
    if (!fs.existsSync(targetPath)) {
      res.status(404).json({ error: 'File does not exist.' });
      return;
    }
    const content = fs.readFileSync(targetPath, 'utf-8');
    res.json({ filename, content });
  });

  return app;
}

const app = createApiApp();
export default app;
