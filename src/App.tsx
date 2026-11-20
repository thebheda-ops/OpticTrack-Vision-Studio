import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';
import {
  Camera,
  CameraOff,
  Download,
  FileText,
  FileVideo,
  Hand,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Terminal,
  X,
} from 'lucide-react';
import { DetectionLogView } from './components/DetectionLogView';
import { FpsTelemetryChart } from './components/FpsTelemetryChart';
import { PythonOpenCvView } from './components/PythonOpenCvView';
import {
  CocoBaseModel,
  DetectedObject,
  DetectionMode,
  DetectorSettings,
  LogEntry,
  LogEntryType,
  PerformanceSample,
  StreamSourceMode,
  TrackedHand,
  WorkspaceTab,
} from './types/detection';
import { getColorForLabel, renderDetectionOverlay } from './utils/canvasOverlay';
import { extractTrackedHands, HandHistorySample } from './utils/handAnalysis';

type ModelLoadState = 'idle' | 'loading' | 'ready' | 'error';

interface BufferedLogItem {
  timestamp: string;
  entryType: LogEntryType;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  fps: number;
  motion?: string;
}

const GESTURE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';

const DEFAULT_SETTINGS: DetectorSettings = {
  detectionMode: 'all',
  modelBase: 'lite_mobilenet_v2',
  confidenceThreshold: 0.5,
  maxDetections: 12,
  maxHands: 2,
  logCooldownMs: 1000,
  loggingEnabled: true,
  showFpsOnCanvas: true,
  showHandSkeleton: true,
  showOrientationDial: true,
  mirrorCamera: true,
  pythonModel: 'yolov4-tiny',
};

export default function App() {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('monitor');
  const [settings, setSettings] = useState<DetectorSettings>(DEFAULT_SETTINGS);
  const [sourceMode, setSourceMode] = useState<StreamSourceMode>('camera');
  const [mobileControlsOpen, setMobileControlsOpen] = useState(false);

  // Model & backend state
  const [objectModelState, setObjectModelState] = useState<ModelLoadState>('idle');
  const [handModelState, setHandModelState] = useState<ModelLoadState>('idle');
  const [modelError, setModelError] = useState<string | null>(null);
  const [tfBackend, setTfBackend] = useState<string>('initializing');

  // Camera & video stream state
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [videoFileName, setVideoFileName] = useState<string | null>(null);
  const [resolution, setResolution] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  // Real-time detection, hand tracking & telemetry state
  const [detections, setDetections] = useState<DetectedObject[]>([]);
  const [trackedHands, setTrackedHands] = useState<TrackedHand[]>([]);
  const [fps, setFps] = useState<number>(0);
  const [inferenceMs, setInferenceMs] = useState<number>(0);
  const [perfSamples, setPerfSamples] = useState<PerformanceSample[]>([]);

  // Local text file log state
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [rawLogText, setRawLogText] = useState<string>('');
  const [logFilePath, setLogFilePath] = useState<string>('logs/detections.txt');
  const [logByteSize, setLogByteSize] = useState<number>(0);
  const [totalLoggedCount, setTotalLoggedCount] = useState<number>(0);
  const [localFileLinkedName, setLocalFileLinkedName] = useState<string | null>(null);

  // Refs for real-time RAF loop without stale closures
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const objectOffscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const handRecognizerRef = useRef<GestureRecognizer | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const videoObjectUrlRef = useRef<string | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const rvfcIdRef = useRef<number | null>(null);
  const isObjectInferringRef = useRef(false);
  const lastObjectDetectTsRef = useRef<number>(0);
  const latestObjectsRef = useRef<DetectedObject[]>([]);
  const lastVideoTimeRef = useRef<number>(-1);
  const lastHandVideoTsRef = useRef<number>(0);
  const handHistoryRef = useRef<Map<string, HandHistorySample[]>>(new Map());
  const settingsRef = useRef(settings);
  const sourceModeRef = useRef(sourceMode);
  const isStreamingRef = useRef(isStreaming);
  const isPausedRef = useRef(isPaused);

  // Rolling FPS & logging buffers
  const frameTimestampsRef = useRef<number[]>([]);
  const lastUiSyncRef = useRef<number>(0);
  const lastSamplePushRef = useRef<number>(0);
  const lastLoggedByKeyRef = useRef<Map<string, number>>(new Map());
  const pendingLogsRef = useRef<BufferedLogItem[]>([]);
  const localFileHandleRef = useRef<FileSystemFileHandle | null>(null);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    sourceModeRef.current = sourceMode;
  }, [sourceMode]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  // Fetch initial detection log state from server
  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch('/api/logs');
      if (!res.ok) return;
      const data = await res.json();
      setLogEntries(data.entries ?? []);
      setRawLogText(data.rawText ?? '');
      setLogFilePath(data.filePath ?? 'logs/detections.txt');
      setLogByteSize(data.byteSize ?? 0);
      setTotalLoggedCount(data.totalEntries ?? 0);
    } catch {
      // Keep current state on transient error
    }
  }, []);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  // Flush buffered detection & gesture entries to local text file
  const flushLogBuffer = useCallback(async () => {
    if (pendingLogsRef.current.length === 0) return;
    const batch = pendingLogsRef.current.splice(0, pendingLogsRef.current.length);

    try {
      const res = await fetch('/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries: batch }),
      });

      if (res.ok) {
        const resData = await res.json();
        setLogByteSize(resData.byteSize ?? 0);
        setTotalLoggedCount((prev) => prev + batch.length);

        const newParsed: LogEntry[] = batch.map((item, idx) => {
          const safeLabel = item.label.replace(/\s+/g, '_');
          const roundedBbox = item.bbox.map((v) => Math.round(v)) as [
            number,
            number,
            number,
            number,
          ];
          const motionPart = item.motion
            ? ` MOTION=${item.motion.replace(/\s+/g, '_')}`
            : '';
          const rawLine = `[${item.timestamp}] TYPE=${item.entryType} OBJECT=${safeLabel} CONF=${item.confidence.toFixed(3)} BBOX=[${roundedBbox.join(',')}] FPS=${item.fps.toFixed(1)}${motionPart}`;
          return {
            id: `${Date.now()}-${idx}-${safeLabel}`,
            timestamp: item.timestamp,
            entryType: item.entryType,
            label: safeLabel,
            confidence: item.confidence,
            bbox: roundedBbox,
            fps: item.fps,
            motion: item.motion,
            rawLine,
          };
        });

        setLogEntries((prev) => [...newParsed.reverse(), ...prev].slice(0, 500));
        setRawLogText((prev) =>
          `${prev}${newParsed.map((e) => e.rawLine).join('\n')}\n`
        );
      }
    } catch {
      // Ignore transient disconnect
    }

    if (localFileHandleRef.current) {
      try {
        const existingFile = await localFileHandleRef.current.getFile();
        const writable = await localFileHandleRef.current.createWritable({
          keepExistingData: true,
        });
        await writable.seek(existingFile.size);
        const textLines = batch
          .map((item) => {
            const safeLabel = item.label.replace(/\s+/g, '_');
            const roundedBbox = item.bbox.map((v) => Math.round(v));
            const motionPart = item.motion
              ? ` MOTION=${item.motion.replace(/\s+/g, '_')}`
              : '';
            return `[${item.timestamp}] TYPE=${item.entryType} OBJECT=${safeLabel} CONF=${item.confidence.toFixed(3)} BBOX=[${roundedBbox.join(',')}] FPS=${item.fps.toFixed(1)}${motionPart}\n`;
          })
          .join('');
        await writable.write(textLines);
        await writable.close();
      } catch {
        // Ignore if local disk write permission was revoked
      }
    }
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      flushLogBuffer();
    }, 550);
    return () => window.clearInterval(timer);
  }, [flushLogBuffer]);

  // Load pre-trained COCO-SSD object model
  const loadObjectModel = useCallback(async (base: CocoBaseModel) => {
    setObjectModelState('loading');
    setModelError(null);
    try {
      await tf.ready();
      if (tf.getBackend() !== 'webgl') {
        try {
          await tf.setBackend('webgl');
        } catch {
          await tf.setBackend('cpu');
        }
      }
      setTfBackend(tf.getBackend() ?? 'cpu');

      if (objectModelRef.current) {
        objectModelRef.current.dispose();
        objectModelRef.current = null;
      }

      const loaded = await cocoSsd.load({ base });
      objectModelRef.current = loaded;
      setObjectModelState('ready');
    } catch (err) {
      setObjectModelState('error');
      setModelError(
        err instanceof Error
          ? err.message
          : 'Failed to load pre-trained COCO-SSD weights.'
      );
    }
  }, []);

  // Load MediaPipe GestureRecognizer (21-point hand landmarks + gesture classification)
  const loadHandRecognizer = useCallback(async (maxHands: number) => {
    setHandModelState('loading');
    try {
      const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
      if (handRecognizerRef.current) {
        handRecognizerRef.current.close();
        handRecognizerRef.current = null;
      }
      const recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: GESTURE_MODEL_URL,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: maxHands,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });
      handRecognizerRef.current = recognizer;
      setHandModelState('ready');
    } catch {
      // Fallback to CPU delegate if WebGL GPU delegate isn't available
      try {
        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        const recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: GESTURE_MODEL_URL,
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: maxHands,
          minHandDetectionConfidence: 0.45,
          minHandPresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        handRecognizerRef.current = recognizer;
        setHandModelState('ready');
      } catch (err) {
        setHandModelState('error');
        setModelError((prev) =>
          prev ??
          (err instanceof Error
            ? err.message
            : 'Failed to initialize MediaPipe hand gesture recognizer.')
        );
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await loadObjectModel(settings.modelBase);
      if (cancelled && objectModelRef.current) {
        objectModelRef.current.dispose();
        objectModelRef.current = null;
      }
    })();
    return () => {
      cancelled = true;
      if (objectModelRef.current) {
        objectModelRef.current.dispose();
        objectModelRef.current = null;
      }
    };
  }, [settings.modelBase, loadObjectModel]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await loadHandRecognizer(settings.maxHands);
      if (cancelled && handRecognizerRef.current) {
        handRecognizerRef.current.close();
        handRecognizerRef.current = null;
      }
    })();
    return () => {
      cancelled = true;
      if (handRecognizerRef.current) {
        handRecognizerRef.current.close();
        handRecognizerRef.current = null;
      }
    };
  }, [settings.maxHands, loadHandRecognizer]);

  const refreshCameraList = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setCameras(videoInputs);
      if (!selectedCameraId && videoInputs.length > 0 && videoInputs[0].deviceId) {
        setSelectedCameraId(videoInputs[0].deviceId);
      }
    } catch {
      // Ignore enumeration failures
    }
  }, [selectedCameraId]);

  useEffect(() => {
    refreshCameraList();
  }, [refreshCameraList]);

  const stopStream = useCallback(() => {
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    if (rvfcIdRef.current !== null && videoRef.current) {
      const cancelVfc = (
        videoRef.current as HTMLVideoElement & {
          cancelVideoFrameCallback?: (id: number) => void;
        }
      ).cancelVideoFrameCallback;
      cancelVfc?.call(videoRef.current, rvfcIdRef.current);
      rvfcIdRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
      videoObjectUrlRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.removeAttribute('src');
      videoRef.current.load();
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    flushLogBuffer();
    frameTimestampsRef.current = [];
    handHistoryRef.current.clear();
    latestObjectsRef.current = [];
    lastVideoTimeRef.current = -1;
    isObjectInferringRef.current = false;
    setIsStreaming(false);
    setIsPaused(false);
    setDetections([]);
    setTrackedHands([]);
    setFps(0);
    setInferenceMs(0);
    setResolution({ w: 0, h: 0 });
  }, [flushLogBuffer]);

  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (videoObjectUrlRef.current) {
        URL.revokeObjectURL(videoObjectUrlRef.current);
      }
    };
  }, []);

  // Non-blocking background object detection on downscaled offscreen canvas (max 416px wide)
  const triggerAsyncObjectDetection = useCallback(
    (
      video: HTMLVideoElement,
      objectModel: cocoSsd.ObjectDetection,
      maxDetections: number,
      confidenceThreshold: number
    ) => {
      if (isObjectInferringRef.current) return;
      isObjectInferringRef.current = true;
      lastObjectDetectTsRef.current = performance.now();

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      const targetW = Math.min(416, vw);
      const targetH = Math.max(1, Math.round((vh / vw) * targetW));

      if (!objectOffscreenCanvasRef.current) {
        objectOffscreenCanvasRef.current = document.createElement('canvas');
      }
      const offscreen = objectOffscreenCanvasRef.current;
      if (offscreen.width !== targetW || offscreen.height !== targetH) {
        offscreen.width = targetW;
        offscreen.height = targetH;
      }

      const offCtx = offscreen.getContext('2d');
      if (!offCtx) {
        isObjectInferringRef.current = false;
        return;
      }
      offCtx.drawImage(video, 0, 0, targetW, targetH);

      const scaleX = vw / targetW;
      const scaleY = vh / targetH;

      objectModel
        .detect(offscreen, maxDetections, confidenceThreshold)
        .then((predictions) => {
          if (!isStreamingRef.current) return;
          latestObjectsRef.current = predictions.map((pred, idx) => ({
            id: `${idx}-${pred.class}`,
            label: pred.class,
            confidence: pred.score,
            bbox: [
              Math.round(pred.bbox[0] * scaleX),
              Math.round(pred.bbox[1] * scaleY),
              Math.round(pred.bbox[2] * scaleX),
              Math.round(pred.bbox[3] * scaleY),
            ],
          }));
        })
        .catch(() => {
          // Ignore transient tensor read errors
        })
        .finally(() => {
          isObjectInferringRef.current = false;
        });
    },
    []
  );

  // Main real-time loop: runs hand tracking synchronously on every fresh camera frame with zero stalls
  const runDetectionFrame = useCallback(() => {
    if (!isStreamingRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const objectModel = objectModelRef.current;
    const handRecognizer = handRecognizerRef.current;
    const currentSettings = settingsRef.current;

    const wantsObjects =
      currentSettings.detectionMode === 'all' ||
      currentSettings.detectionMode === 'objects';
    const wantsHands =
      currentSettings.detectionMode === 'all' ||
      currentSettings.detectionMode === 'hands';

    const canRunAtLeastOne =
      (wantsObjects && Boolean(objectModel)) ||
      (wantsHands && Boolean(handRecognizer));

    const videoWithVfc = video as
      | (HTMLVideoElement & {
          requestVideoFrameCallback?: (cb: () => void) => number;
        })
      | null;
    const hasRvfc = Boolean(videoWithVfc?.requestVideoFrameCallback);

    if (
      video &&
      canvas &&
      canRunAtLeastOne &&
      !isPausedRef.current &&
      video.readyState >= 2 &&
      video.videoWidth > 0 &&
      video.videoHeight > 0 &&
      (hasRvfc || video.currentTime !== lastVideoTimeRef.current)
    ) {
      lastVideoTimeRef.current = video.currentTime;

      try {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          setResolution({ w: video.videoWidth, h: video.videoHeight });
        }

        const mirrorX =
          currentSettings.mirrorCamera && sourceModeRef.current === 'camera';
        const now = performance.now();

        // 1. Kick off non-blocking object detection when needed (every ~140ms in Combined mode, every frame in Objects mode)
        if (wantsObjects && objectModel) {
          const objectIntervalMs = wantsHands ? 140 : 0;
          if (now - lastObjectDetectTsRef.current >= objectIntervalMs) {
            triggerAsyncObjectDetection(
              video,
              objectModel,
              currentSettings.maxDetections,
              currentSettings.confidenceThreshold
            );
          }
        } else {
          latestObjectsRef.current = [];
        }

        const formattedObjects = latestObjectsRef.current;

        // 2. Run MediaPipe 21-point hand tracking & gesture recognition synchronously on current frame
        const t0 = performance.now();
        let formattedHands: TrackedHand[] = [];
        if (wantsHands && handRecognizer) {
          const nowTs = Math.max(
            lastHandVideoTsRef.current + 1,
            Math.floor(t0)
          );
          lastHandVideoTsRef.current = nowTs;
          const handRes = handRecognizer.recognizeForVideo(video, nowTs);
          formattedHands = extractTrackedHands(
            handRes,
            video.videoWidth,
            video.videoHeight,
            handHistoryRef.current,
            t0,
            mirrorX
          );
        } else {
          handHistoryRef.current.clear();
        }

        const t1 = performance.now();
        const currentInferMs = t1 - t0;

        // Rolling FPS calculation across fresh rendered frames
        const stamps = frameTimestampsRef.current;
        stamps.push(t1);
        if (stamps.length > 24) stamps.shift();

        let currentFps = 0;
        if (stamps.length >= 2) {
          const spanMs = stamps[stamps.length - 1] - stamps[0];
          if (spanMs > 0) {
            currentFps = ((stamps.length - 1) * 1000) / spanMs;
          }
        }

        // Render objects + 21-point hand skeleton + movement trails + gesture labels + FPS HUD
        const ctx = canvas.getContext('2d');
        if (ctx) {
          renderDetectionOverlay(
            ctx,
            canvas.width,
            canvas.height,
            formattedObjects,
            formattedHands,
            {
              mirrorX,
              showFpsHud: currentSettings.showFpsOnCanvas,
              showHandSkeleton: currentSettings.showHandSkeleton,
              showOrientationDial: currentSettings.showOrientationDial,
              fps: currentFps,
              inferenceMs: currentInferMs,
              loggingEnabled: currentSettings.loggingEnabled,
            }
          );
        }

        // Log detected objects, hand gestures, and dynamic swipe movements
        if (
          currentSettings.loggingEnabled &&
          (formattedObjects.length > 0 || formattedHands.length > 0)
        ) {
          const epochNow = Date.now();
          const isoTimestamp = new Date(epochNow).toISOString();

          for (const item of formattedObjects) {
            const key = `obj:${item.label}`;
            const prevLogged = lastLoggedByKeyRef.current.get(key) ?? 0;
            if (epochNow - prevLogged >= currentSettings.logCooldownMs) {
              pendingLogsRef.current.push({
                timestamp: isoTimestamp,
                entryType: 'object',
                label: item.label,
                confidence: item.confidence,
                bbox: item.bbox,
                fps: currentFps,
              });
              lastLoggedByKeyRef.current.set(key, epochNow);
            }
          }

          for (const hand of formattedHands) {
            const gestureLabel = `${hand.handedness}:${hand.gesture}`;
            const gKey = `gesture:${gestureLabel}`;
            const motionSummary = `${hand.direction}@${Math.round(hand.speedPxPerSec)}px/s`;
            const prevGestureLog = lastLoggedByKeyRef.current.get(gKey) ?? 0;
            if (epochNow - prevGestureLog >= currentSettings.logCooldownMs) {
              pendingLogsRef.current.push({
                timestamp: isoTimestamp,
                entryType: 'hand_gesture',
                label: gestureLabel,
                confidence: hand.gestureConfidence,
                bbox: hand.bbox,
                fps: currentFps,
                motion: motionSummary,
              });
              lastLoggedByKeyRef.current.set(gKey, epochNow);
            }

            if (hand.swipeGesture) {
              const swipeLabel = `${hand.handedness}:${hand.swipeGesture}`;
              const sKey = `motion:${swipeLabel}`;
              const swipeCooldown = Math.max(450, currentSettings.logCooldownMs);
              const prevSwipeLog = lastLoggedByKeyRef.current.get(sKey) ?? 0;
              if (epochNow - prevSwipeLog >= swipeCooldown) {
                pendingLogsRef.current.push({
                  timestamp: isoTimestamp,
                  entryType: 'hand_motion',
                  label: swipeLabel,
                  confidence: hand.gestureConfidence,
                  bbox: hand.bbox,
                  fps: currentFps,
                  motion: motionSummary,
                });
                lastLoggedByKeyRef.current.set(sKey, epochNow);
              }
            }
          }
        }

        // Throttled React UI state sync (~7 Hz)
        if (t1 - lastUiSyncRef.current >= 140) {
          lastUiSyncRef.current = t1;
          setDetections(formattedObjects);
          setTrackedHands(formattedHands);
          setFps(currentFps);
          setInferenceMs(currentInferMs);
        }

        if (t1 - lastSamplePushRef.current >= 450) {
          lastSamplePushRef.current = t1;
          setPerfSamples((prev) => [
            ...prev.slice(-59),
            {
              timestamp: Date.now(),
              fps: currentFps,
              inferenceMs: currentInferMs,
              objectCount: formattedObjects.length,
              handCount: formattedHands.length,
            },
          ]);
        }
      } catch {
        // Skip single-frame WebGL texture glitch gracefully
      }
    }

    if (isStreamingRef.current) {
      if (
        videoWithVfc?.requestVideoFrameCallback &&
        !isPausedRef.current &&
        videoWithVfc.readyState >= 2
      ) {
        rvfcIdRef.current = videoWithVfc.requestVideoFrameCallback(runDetectionFrame);
      } else {
        rafIdRef.current = requestAnimationFrame(runDetectionFrame);
      }
    }
  }, [triggerAsyncObjectDetection]);

  const startCamera = useCallback(
    async (deviceIdOverride?: string) => {
      setStreamError(null);
      stopStream();
      setSourceMode('camera');

      if (!navigator.mediaDevices?.getUserMedia) {
        setStreamError(
          'Camera access is unavailable in this browser context. You can upload a local video file to test detection and hand tracking.'
        );
        return;
      }

      try {
        const targetId = deviceIdOverride ?? selectedCameraId;
        const videoConstraints: MediaTrackConstraints = targetId
          ? {
              deviceId: { exact: targetId },
              width: { ideal: 960 },
              height: { ideal: 540 },
              frameRate: { ideal: 60, min: 24 },
            }
          : {
              facingMode: 'user',
              width: { ideal: 960 },
              height: { ideal: 540 },
              frameRate: { ideal: 60, min: 24 },
            };

        const stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
          audio: false,
        });
        mediaStreamRef.current = stream;

        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play();

        setResolution({ w: video.videoWidth, h: video.videoHeight });
        setIsStreaming(true);
        isStreamingRef.current = true;
        setIsPaused(false);
        isPausedRef.current = false;

        await refreshCameraList();
        rafIdRef.current = requestAnimationFrame(runDetectionFrame);
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Unable to access laptop camera.';
        setStreamError(
          `Camera access failed (${message}). Check browser camera permissions or load a video file instead.`
        );
      }
    },
    [refreshCameraList, runDetectionFrame, selectedCameraId, stopStream]
  );

  const handleVideoFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      setStreamError(null);
      stopStream();
      setSourceMode('video');
      setVideoFileName(file.name);

      const url = URL.createObjectURL(file);
      videoObjectUrlRef.current = url;

      const video = videoRef.current;
      if (!video) return;

      video.src = url;
      video.loop = true;
      video.muted = true;

      try {
        await video.play();
        setResolution({ w: video.videoWidth, h: video.videoHeight });
        setIsStreaming(true);
        isStreamingRef.current = true;
        setIsPaused(false);
        isPausedRef.current = false;
        rafIdRef.current = requestAnimationFrame(runDetectionFrame);
      } catch {
        setStreamError('Could not play the selected video file.');
      }
    },
    [runDetectionFrame, stopStream]
  );

  const togglePause = useCallback(() => {
    const video = videoRef.current;
    if (!video || !isStreaming) return;
    if (isPaused) {
      video.play().catch(() => undefined);
      setIsPaused(false);
      isPausedRef.current = false;
    } else {
      video.pause();
      setIsPaused(true);
      isPausedRef.current = true;
    }
  }, [isPaused, isStreaming]);

  const handleSaveSnapshot = useCallback(() => {
    const video = videoRef.current;
    const overlay = canvasRef.current;
    if (!video || !overlay || video.videoWidth === 0) return;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = video.videoWidth;
    exportCanvas.height = video.videoHeight;
    const ctx = exportCanvas.getContext('2d');
    if (!ctx) return;

    if (settings.mirrorCamera && sourceMode === 'camera') {
      ctx.save();
      ctx.translate(exportCanvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height);
      ctx.restore();
    } else {
      ctx.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height);
    }
    ctx.drawImage(overlay, 0, 0);

    const dataUrl = exportCanvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = `optictrack_snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, [settings.mirrorCamera, sourceMode]);

  const handleClearLogs = useCallback(async () => {
    pendingLogsRef.current = [];
    lastLoggedByKeyRef.current.clear();
    try {
      await fetch('/api/logs', { method: 'DELETE' });
      await fetchLogs();
    } catch {
      // Ignore transient error
    }
  }, [fetchLogs]);

  const handleDownloadLogs = useCallback(() => {
    const a = document.createElement('a');
    a.href = '/api/logs/download';
    a.download = 'detections.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, []);

  const handleLinkLocalFile = useCallback(async () => {
    if (!('showSaveFilePicker' in window)) return;
    try {
      const picker = window.showSaveFilePicker as (options?: {
        suggestedName?: string;
        types?: Array<{ description: string; accept: Record<string, string[]> }>;
      }) => Promise<FileSystemFileHandle>;

      const handle = await picker({
        suggestedName: 'detections.txt',
        types: [
          {
            description: 'Text Log File',
            accept: { 'text/plain': ['.txt'] },
          },
        ],
      });
      localFileHandleRef.current = handle;
      setLocalFileLinkedName(handle.name);

      const file = await handle.getFile();
      if (file.size === 0) {
        const writable = await handle.createWritable();
        await writable.write(
          '# OpticTrack Detection Timestamp Log\n# Format: [ISO_TIMESTAMP] TYPE=<type> OBJECT=<label> CONF=<score> BBOX=[x,y,w,h] FPS=<fps> [MOTION=<motion>]\n'
        );
        await writable.close();
      }
    } catch {
      // User canceled file picker
    }
  }, []);

  const groupedDetections = useMemo(() => {
    const counts = new Map<string, { count: number; maxConf: number }>();
    for (const d of detections) {
      const curr = counts.get(d.label);
      if (!curr) {
        counts.set(d.label, { count: 1, maxConf: d.confidence });
      } else {
        counts.set(d.label, {
          count: curr.count + 1,
          maxConf: Math.max(curr.maxConf, d.confidence),
        });
      }
    }
    return Array.from(counts.entries()).sort((a, b) => b[1].count - a[1].count);
  }, [detections]);

  const anyModelLoading =
    objectModelState === 'loading' || handModelState === 'loading';

  const statusText =
    objectModelState === 'ready' && handModelState === 'ready'
      ? `● Models Ready (${tfBackend.toUpperCase()})`
      : anyModelLoading
        ? '○ Loading Models...'
        : objectModelState === 'error' && handModelState === 'error'
          ? '▲ Model Error'
          : '● Ready';

  const statusColor =
    objectModelState === 'error' && handModelState === 'error'
      ? 'text-rose-400'
      : anyModelLoading
        ? 'text-amber-400'
        : 'text-emerald-400';

  return (
    <div className="min-h-screen bg-[#090C10] text-slate-100 flex flex-col">
      {/* Top Bar Contract: Brand Wordmark — Nav Links (Desktop) — 1 Primary Action */}
      <header className="border-b border-slate-800 bg-[#0C1016] px-4 md:px-6 h-14 md:h-16 flex items-center justify-between gap-4 md:gap-8 md:sticky md:top-0 z-30">
        <a
          href="#monitor"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('monitor');
          }}
          className="text-base sm:text-lg font-bold tracking-tight text-slate-100 whitespace-nowrap shrink-0"
        >
          OpticTrack Studio
        </a>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
          <button
            type="button"
            onClick={() => setActiveTab('monitor')}
            className={`py-1 transition-colors border-b-2 whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'monitor'
                ? 'text-slate-100 border-cyan-400'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Live Monitor
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab('logs');
              fetchLogs();
            }}
            className={`py-1 transition-colors border-b-2 whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'logs'
                ? 'text-slate-100 border-cyan-400'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Detection Log
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('python')}
            className={`py-1 transition-colors border-b-2 whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'python'
                ? 'text-slate-100 border-cyan-400'
                : 'text-slate-400 border-transparent hover:text-slate-200'
            }`}
          >
            Python OpenCV
          </button>
        </nav>

        <div className="flex items-center shrink-0">
          {isStreaming ? (
            <button
              type="button"
              onClick={stopStream}
              className="inline-flex items-center justify-center gap-2 min-h-[40px] md:min-h-0 px-4 py-2 text-sm md:text-xs font-semibold text-slate-100 bg-rose-600 hover:bg-rose-500 active:scale-[0.98] rounded-md transition-all whitespace-nowrap shrink-0 cursor-pointer"
            >
              <CameraOff className="w-4 h-4 md:w-3.5 md:h-3.5" />
              Stop Stream
            </button>
          ) : (
            <button
              type="button"
              onClick={() => {
                setActiveTab('monitor');
                startCamera();
              }}
              disabled={anyModelLoading}
              className="inline-flex items-center justify-center gap-2 min-h-[40px] md:min-h-0 px-4 py-2 text-sm md:text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 active:scale-[0.98] rounded-md transition-all whitespace-nowrap shrink-0 disabled:opacity-50 cursor-pointer"
            >
              <Camera className="w-4 h-4 md:w-3.5 md:h-3.5" />
              Start Camera
            </button>
          )}
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 w-full max-w-[1440px] mx-auto px-3 sm:px-6 pt-3 pb-20 sm:py-5 md:pb-6">
        {activeTab === 'monitor' && (
          <div className="flex flex-col lg:flex-row gap-5 items-start">
            {/* Desktop Control & Parameter Dashboard (Hidden on mobile screens) */}
            <aside className="hidden lg:block w-[320px] shrink-0 bg-[#10151C] border border-slate-800 rounded-lg divide-y divide-slate-800">
              <div className="p-4 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
                  <h2 className="text-sm font-semibold text-slate-100">
                    Vision Pipeline
                  </h2>
                </div>
                <span className={`text-xs font-mono ${statusColor}`}>
                  {statusText}
                </span>
              </div>

              {/* Detection Pipeline Mode */}
              <div className="p-4 space-y-2.5">
                <div className="text-xs font-medium text-slate-300">
                  Tracking Mode
                </div>
                <div className="grid grid-cols-3 gap-1 p-1 bg-[#090C10] border border-slate-800 rounded-md">
                  {(
                    [
                      ['all', 'Combined'],
                      ['hands', 'Hands Only'],
                      ['objects', 'Objects'],
                    ] as Array<[DetectionMode, string]>
                  ).map(([mode, label]) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() =>
                        setSettings((prev) => ({ ...prev, detectionMode: mode }))
                      }
                      className={`py-1.5 px-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        settings.detectionMode === mode
                          ? 'bg-slate-800 text-cyan-300'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Video Source */}
              <div className="p-4 space-y-3">
                <div className="text-xs font-medium text-slate-300">
                  Video Input Source
                </div>
                <div className="grid grid-cols-2 gap-1 p-1 bg-[#090C10] border border-slate-800 rounded-md">
                  <button
                    type="button"
                    onClick={() => {
                      setSourceMode('camera');
                      if (!isStreaming) startCamera();
                    }}
                    className={`py-1.5 px-2 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      sourceMode === 'camera'
                        ? 'bg-slate-800 text-slate-100'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Laptop Camera
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className={`py-1.5 px-2 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      sourceMode === 'video'
                        ? 'bg-slate-800 text-slate-100'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Video File
                  </button>
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleVideoFileSelect}
                  className="hidden"
                />

                {sourceMode === 'camera' && cameras.length > 0 && (
                  <div className="space-y-1.5">
                    <label
                      htmlFor="camera-select"
                      className="block text-xs text-slate-400"
                    >
                      Camera Device
                    </label>
                    <select
                      id="camera-select"
                      value={selectedCameraId}
                      onChange={(e) => {
                        const nextId = e.target.value;
                        setSelectedCameraId(nextId);
                        if (isStreaming) {
                          startCamera(nextId);
                        }
                      }}
                      className="w-full px-2.5 py-2 text-xs bg-[#090C10] border border-slate-800 rounded-md text-slate-200 focus:outline-none focus:border-cyan-500"
                    >
                      {cameras.map((cam, idx) => (
                        <option key={cam.deviceId || idx} value={cam.deviceId}>
                          {cam.label || `Camera ${idx + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {sourceMode === 'video' && (
                  <div className="flex items-center justify-between gap-2 text-xs text-slate-400">
                    <span className="truncate">
                      {videoFileName ?? 'No video selected'}
                    </span>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="text-cyan-400 hover:text-cyan-300 font-medium whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      Choose File
                    </button>
                  </div>
                )}
              </div>

              {/* Pre-Trained Model Architecture */}
              <div className="p-4 space-y-3">
                <div className="space-y-1.5">
                  <label
                    htmlFor="web-model-select"
                    className="block text-xs font-medium text-slate-300"
                  >
                    Object Model Architecture
                  </label>
                  <select
                    id="web-model-select"
                    value={settings.modelBase}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        modelBase: e.target.value as CocoBaseModel,
                      }))
                    }
                    className="w-full px-2.5 py-2 text-xs bg-[#090C10] border border-slate-800 rounded-md text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="lite_mobilenet_v2">
                      COCO-SSD Lite MobileNet v2 (High FPS)
                    </option>
                    <option value="mobilenet_v2">
                      COCO-SSD MobileNet v2 (Higher Accuracy)
                    </option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label
                    htmlFor="python-model-select"
                    className="block text-xs font-medium text-slate-300"
                  >
                    Synced Python OpenCV Preset
                  </label>
                  <select
                    id="python-model-select"
                    value={settings.pythonModel}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        pythonModel: e.target.value as DetectorSettings['pythonModel'],
                      }))
                    }
                    className="w-full px-2.5 py-2 text-xs bg-[#090C10] border border-slate-800 rounded-md text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="yolov4-tiny">YOLOv4-tiny + MediaPipe Hands</option>
                    <option value="mobilenet-ssd">MobileNet-SSD + MediaPipe Hands</option>
                  </select>
                </div>
              </div>

              {/* Thresholds & Overlay Toggles */}
              <div className="p-4 space-y-4">
                <div>
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="text-slate-300">Confidence Threshold</span>
                    <span className="font-mono tabular-nums text-cyan-400">
                      {(settings.confidenceThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.2}
                    max={0.9}
                    step={0.05}
                    value={settings.confidenceThreshold}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        confidenceThreshold: parseFloat(e.target.value),
                      }))
                    }
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="text-slate-300">Tracked Hands Limit</span>
                    <span className="font-mono tabular-nums text-cyan-400">
                      {settings.maxHands}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={2}
                    step={1}
                    value={settings.maxHands}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        maxHands: parseInt(e.target.value, 10),
                      }))
                    }
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>

                <div className="space-y-2.5 pt-1">
                  <label className="flex items-center justify-between text-xs text-slate-300 cursor-pointer">
                    <span>21-Point Hand Skeleton</span>
                    <input
                      type="checkbox"
                      checked={settings.showHandSkeleton}
                      onChange={(e) =>
                        setSettings((prev) => ({
                          ...prev,
                          showHandSkeleton: e.target.checked,
                        }))
                      }
                      className="accent-cyan-400 w-4 h-4 rounded cursor-pointer"
                    />
                  </label>
                  <label className="flex items-center justify-between text-xs text-slate-300 cursor-pointer">
                    <span>360° Orientation Dial</span>
                    <input
                      type="checkbox"
                      checked={settings.showOrientationDial}
                      onChange={(e) =>
                        setSettings((prev) => ({
                          ...prev,
                          showOrientationDial: e.target.checked,
                        }))
                      }
                      className="accent-cyan-400 w-4 h-4 rounded cursor-pointer"
                    />
                  </label>
                  <label className="flex items-center justify-between text-xs text-slate-300 cursor-pointer">
                    <span>Show FPS HUD on Video</span>
                    <input
                      type="checkbox"
                      checked={settings.showFpsOnCanvas}
                      onChange={(e) =>
                        setSettings((prev) => ({
                          ...prev,
                          showFpsOnCanvas: e.target.checked,
                        }))
                      }
                      className="accent-cyan-400 w-4 h-4 rounded cursor-pointer"
                    />
                  </label>
                  <label className="flex items-center justify-between text-xs text-slate-300 cursor-pointer">
                    <span>Mirror Laptop Webcam</span>
                    <input
                      type="checkbox"
                      checked={settings.mirrorCamera}
                      onChange={(e) =>
                        setSettings((prev) => ({
                          ...prev,
                          mirrorCamera: e.target.checked,
                        }))
                      }
                      className="accent-cyan-400 w-4 h-4 rounded cursor-pointer"
                    />
                  </label>
                </div>
              </div>

              {/* Timestamp Logging Controls */}
              <div className="p-4 space-y-3">
                <label className="flex items-center justify-between text-xs font-medium text-slate-200 cursor-pointer">
                  <span>Log Timestamps to .txt</span>
                  <input
                    type="checkbox"
                    checked={settings.loggingEnabled}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        loggingEnabled: e.target.checked,
                      }))
                    }
                    className="accent-cyan-400 w-4 h-4 rounded cursor-pointer"
                  />
                </label>

                <div className="space-y-1.5">
                  <label
                    htmlFor="cooldown-select"
                    className="block text-xs text-slate-400"
                  >
                    Per-Class / Gesture Cooldown
                  </label>
                  <select
                    id="cooldown-select"
                    value={settings.logCooldownMs}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        logCooldownMs: parseInt(e.target.value, 10),
                      }))
                    }
                    className="w-full px-2.5 py-2 text-xs bg-[#090C10] border border-slate-800 rounded-md text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value={0}>Every Frame (0.0 s)</option>
                    <option value={500}>0.5 seconds</option>
                    <option value={1000}>1.0 second</option>
                    <option value={2000}>2.0 seconds</option>
                    <option value={5000}>5.0 seconds</option>
                  </select>
                </div>

                <div className="pt-1 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleDownloadLogs}
                    className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Export .txt
                  </button>
                  <button
                    type="button"
                    onClick={handleSaveSnapshot}
                    disabled={!isStreaming}
                    className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 disabled:opacity-40 cursor-pointer"
                  >
                    Save Frame
                  </button>
                </div>
              </div>
            </aside>

            {/* Right Main Viewport & Desktop Performance Monitoring Stage */}
            <section className="flex-1 min-w-0 w-full space-y-3 sm:space-y-4">
              {/* Desktop 5-Metric KPI Dashboard Strip (Hidden on mobile screens) */}
              <div className="hidden lg:grid grid-cols-5 bg-[#10151C] border border-slate-800 rounded-lg divide-x divide-slate-800">
                <div className="p-4">
                  <div className="text-xs text-slate-400">Frame Rate</div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span
                      className={`text-2xl font-mono font-semibold tabular-nums ${
                        !isStreaming
                          ? 'text-slate-500'
                          : fps >= 20
                            ? 'text-emerald-400'
                            : 'text-amber-400'
                      }`}
                    >
                      {isStreaming ? fps.toFixed(1) : '0.0'}
                    </span>
                    <span className="text-xs font-mono text-slate-400">FPS</span>
                  </div>
                </div>

                <div className="p-4">
                  <div className="text-xs text-slate-400">Inference Latency</div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span className="text-2xl font-mono font-semibold tabular-nums text-slate-100">
                      {isStreaming ? inferenceMs.toFixed(1) : '0.0'}
                    </span>
                    <span className="text-xs font-mono text-slate-400">ms</span>
                  </div>
                </div>

                <div className="p-4">
                  <div className="text-xs text-slate-400">Detected Objects</div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span className="text-2xl font-mono font-semibold tabular-nums text-cyan-400">
                      {detections.length}
                    </span>
                    <span className="text-xs font-mono text-slate-400">items</span>
                  </div>
                </div>

                <div className="p-4">
                  <div className="text-xs text-slate-400">Tracked Hands</div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span className="text-2xl font-mono font-semibold tabular-nums text-emerald-400">
                      {trackedHands.length}
                    </span>
                    <span className="text-xs font-mono text-slate-400">active</span>
                  </div>
                </div>

                <div className="p-4">
                  <div className="text-xs text-slate-400">Logged Events</div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span className="text-2xl font-mono font-semibold tabular-nums text-slate-100">
                      {totalLoggedCount}
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      entries
                    </span>
                  </div>
                </div>
              </div>

              {/* Video Feed + Real-Time Bounding Box & Hand Skeleton Canvas */}
              <div className="bg-[#10151C] border border-slate-800 rounded-xl lg:rounded-lg overflow-hidden">
                <div className="px-3.5 sm:px-4 py-2.5 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2.5">
                  <div className="flex flex-wrap items-center gap-2 text-[13px] sm:text-xs text-slate-300">
                    <span
                      className={`font-mono ${
                        isStreaming
                          ? isPaused
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                          : 'text-slate-500'
                      }`}
                    >
                      {isStreaming
                        ? isPaused
                          ? '● Paused'
                          : '● Live Stream'
                        : '○ Standby'}
                    </span>
                    <span className="text-slate-600" aria-hidden="true">·</span>
                    <span className="font-mono tabular-nums text-slate-400">
                      {resolution.w > 0
                        ? `${resolution.w} × ${resolution.h}`
                        : 'Auto Resolution'}
                    </span>
                    {isStreaming && (
                      <>
                        <span className="text-slate-600 lg:hidden" aria-hidden="true">·</span>
                        <span className="font-mono tabular-nums text-emerald-400 lg:hidden">
                          {fps.toFixed(1)} FPS
                        </span>
                      </>
                    )}
                  </div>

                  {isStreaming && (
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        type="button"
                        onClick={togglePause}
                        className="inline-flex items-center justify-center gap-1.5 min-h-[40px] lg:min-h-0 px-3.5 py-2 lg:px-3 lg:py-1 text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-md lg:rounded transition-all whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        {isPaused ? (
                          <>
                            <Play className="w-3.5 h-3.5 lg:w-3 lg:h-3 text-emerald-400" />
                            Resume
                          </>
                        ) : (
                          <>
                            <Pause className="w-3.5 h-3.5 lg:w-3 lg:h-3 text-amber-400" />
                            Pause
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          sourceMode === 'camera'
                            ? startCamera()
                            : fileInputRef.current?.click()
                        }
                        className="inline-flex items-center justify-center gap-1.5 min-h-[40px] lg:min-h-0 px-3.5 py-2 lg:px-3 lg:py-1 text-xs font-medium text-slate-300 hover:text-slate-100 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-md lg:rounded transition-all whitespace-nowrap shrink-0 cursor-pointer"
                      >
                        <RotateCcw className="w-3.5 h-3.5 lg:w-3 lg:h-3" />
                        Restart
                      </button>
                    </div>
                  )}
                </div>

                {/* Responsive Screening Viewport: portrait-friendly on mobile, 16:9 on desktop */}
                <div className="relative w-full aspect-[3/4] sm:aspect-video min-h-[340px] max-h-[66dvh] lg:max-h-none bg-[#05070A] flex items-center justify-center overflow-hidden">
                  <video
                    ref={videoRef}
                    playsInline
                    muted
                    className={`w-full h-full object-contain ${
                      settings.mirrorCamera && sourceMode === 'camera'
                        ? '-scale-x-100'
                        : ''
                    } ${isStreaming ? 'block' : 'hidden'}`}
                  />
                  <canvas
                    ref={canvasRef}
                    className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${
                      isStreaming ? 'block' : 'hidden'
                    }`}
                  />

                  {!isStreaming && (
                    <div className="px-5 sm:px-6 py-8 sm:py-10 text-center max-w-md sm:max-w-lg">
                      <Camera className="w-11 h-11 sm:w-10 sm:h-10 text-slate-600 mx-auto mb-3.5" />
                      <h3 className="text-lg sm:text-base font-semibold text-slate-100 text-balance leading-snug">
                        Connect Camera or Video Stream
                      </h3>
                      <p className="text-sm sm:text-xs text-slate-400 mt-2 leading-relaxed">
                        {modelError
                          ? modelError
                          : streamError
                            ? streamError
                            : 'Start your camera for real-time object recognition, 360° hand & finger tracking, gesture classification, and timestamp logging.'}
                      </p>

                      <div className="mt-6 flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3">
                        <button
                          type="button"
                          onClick={() => startCamera()}
                          disabled={anyModelLoading}
                          className="inline-flex items-center justify-center gap-2 min-h-[44px] px-5 py-2.5 text-sm sm:text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 active:scale-[0.98] rounded-lg sm:rounded-md transition-all whitespace-nowrap shrink-0 disabled:opacity-50 cursor-pointer"
                        >
                          <Camera className="w-4 h-4" />
                          {anyModelLoading
                            ? 'Loading Vision Models...'
                            : 'Start Camera'}
                        </button>
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          disabled={anyModelLoading}
                          className="inline-flex items-center justify-center gap-2 min-h-[44px] px-5 py-2.5 text-sm sm:text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-lg sm:rounded-md transition-all whitespace-nowrap shrink-0 disabled:opacity-50 cursor-pointer"
                        >
                          <FileVideo className="w-4 h-4 text-cyan-400" />
                          Upload Video File
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Mobile Quick Screening Controls (Touch-friendly buttons without full dashboard clutter) */}
                <div className="lg:hidden px-3.5 py-3 border-t border-slate-800 bg-[#0E131A] space-y-2.5">
                  <div className="grid grid-cols-3 gap-1.5 p-1 bg-[#090C10] border border-slate-800 rounded-lg">
                    {(
                      [
                        ['all', 'Combined'],
                        ['hands', 'Hands Only'],
                        ['objects', 'Objects'],
                      ] as Array<[DetectionMode, string]>
                    ).map(([mode, label]) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() =>
                          setSettings((prev) => ({ ...prev, detectionMode: mode }))
                        }
                        className={`min-h-[44px] px-2 text-[13px] font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          settings.detectionMode === mode
                            ? 'bg-slate-800 text-cyan-300'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center justify-center gap-1.5 min-h-[44px] px-3 text-[13px] font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-lg transition-all whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <FileVideo className="w-4 h-4 text-cyan-400 shrink-0" />
                      Video
                    </button>
                    <button
                      type="button"
                      onClick={handleSaveSnapshot}
                      disabled={!isStreaming}
                      className="inline-flex items-center justify-center gap-1.5 min-h-[44px] px-3 text-[13px] font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-lg transition-all whitespace-nowrap shrink-0 disabled:opacity-40 cursor-pointer"
                    >
                      Save Frame
                    </button>
                    <button
                      type="button"
                      onClick={() => setMobileControlsOpen(true)}
                      className="inline-flex items-center justify-center gap-1.5 min-h-[44px] px-3 text-[13px] font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-lg transition-all whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      <SlidersHorizontal className="w-4 h-4 text-cyan-400 shrink-0" />
                      Controls
                    </button>
                  </div>
                </div>

                {/* Clean Unboxed Active Labels & 360° Hand/Finger Telemetry Bar */}
                <div className="px-3.5 sm:px-4 py-3 border-t border-slate-800 bg-[#0C1016] space-y-2.5 text-[13px] sm:text-xs leading-relaxed">
                  <div className="flex flex-wrap items-center justify-between gap-2.5">
                    <div className="flex flex-wrap items-center gap-2 text-slate-300">
                      <span className="text-slate-500">Active Objects:</span>
                      {groupedDetections.length === 0 ? (
                        <span className="text-slate-500">
                          {isStreaming
                            ? settings.detectionMode === 'hands'
                              ? 'Disabled in Hands Only mode'
                              : 'None above threshold'
                            : 'Waiting for stream'}
                        </span>
                      ) : (
                        groupedDetections.map(([label, stats], idx) => (
                          <React.Fragment key={label}>
                            {idx > 0 && (
                              <span className="text-slate-600" aria-hidden="true">
                                ·
                              </span>
                            )}
                            <span className="inline-flex items-center gap-1.5 font-mono tabular-nums">
                              <span
                                className="w-2 h-2 rounded-xs inline-block"
                                style={{ backgroundColor: getColorForLabel(label) }}
                              />
                              <span className="font-sans font-medium text-slate-100">
                                {label}
                              </span>
                              <span className="text-slate-400">
                                ×{stats.count} ({(stats.maxConf * 100).toFixed(0)}%)
                              </span>
                            </span>
                          </React.Fragment>
                        ))
                      )}
                    </div>
                    <div className="hidden sm:block text-slate-400 font-mono text-xs">
                      {settings.loggingEnabled
                        ? `Logging -> ${logFilePath}`
                        : 'Logging Paused'}
                    </div>
                  </div>

                  {/* Real-Time 360° Hand & Per-Finger Articulation Telemetry */}
                  <div className="pt-2 border-t border-slate-800/70 space-y-2 text-slate-300">
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                      <span className="inline-flex items-center gap-1.5 text-slate-500">
                        <Hand className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                        360° Hand & Finger Tracking:
                      </span>
                      {trackedHands.length === 0 ? (
                        <span className="text-slate-500">
                          {isStreaming
                            ? settings.detectionMode === 'objects'
                              ? 'Disabled in Objects Only mode'
                              : 'Show hand at any angle (360° rotation, palm front/back, individual fingers)'
                            : 'Waiting for stream'}
                        </span>
                      ) : (
                        trackedHands.map((hand, idx) => (
                          <React.Fragment key={hand.id}>
                            {idx > 0 && (
                              <span className="text-slate-600" aria-hidden="true">
                                ·
                              </span>
                            )}
                            <span className="inline-flex flex-wrap items-center gap-2 font-mono tabular-nums">
                              <span className="font-sans font-medium text-emerald-400">
                                {hand.handedness}
                              </span>
                              <span className="text-slate-100 font-sans font-semibold">
                                {hand.swipeGesture ?? hand.gesture}
                              </span>
                              <span className="text-slate-400">
                                ({hand.orientation.rollDeg}° · {hand.orientation.facing}
                                {hand.orientation.rotationDirection !== 'Steady'
                                  ? ` · ${hand.orientation.rotationDirection}`
                                  : ''}{' '}
                                · {hand.direction} {Math.round(hand.speedPxPerSec)} px/s)
                              </span>
                            </span>
                          </React.Fragment>
                        ))
                      )}
                    </div>

                    {trackedHands.length > 0 && (
                      <div className="space-y-1.5">
                        {trackedHands.map((hand) => (
                          <div
                            key={`${hand.id}-fingers`}
                            className="flex flex-wrap items-center gap-x-3.5 gap-y-1.5 text-[13px] sm:text-xs font-mono tabular-nums text-slate-400"
                          >
                            <span className="font-sans text-slate-500">
                              {hand.handedness} Fingers:
                            </span>
                            {hand.fingers.map((finger, fIdx) => (
                              <React.Fragment key={finger.name}>
                                {fIdx > 0 && (
                                  <span className="text-slate-700" aria-hidden="true">
                                    ·
                                  </span>
                                )}
                                <span className="inline-flex items-center gap-1.5">
                                  <span
                                    className={
                                      finger.isMoving
                                        ? 'text-amber-400 font-semibold'
                                        : finger.state === 'Extended'
                                          ? 'text-emerald-400'
                                          : 'text-slate-400'
                                    }
                                  >
                                    {finger.name}
                                  </span>
                                  <span className="text-slate-300">
                                    {finger.extensionPct}%
                                  </span>
                                  <span className="text-slate-500">
                                    ({finger.bendAngleDeg}°
                                    {finger.isMoving
                                      ? ` · ${Math.round(finger.speedPxPerSec)} px/s`
                                      : ''}
                                    )
                                  </span>
                                </span>
                              </React.Fragment>
                            ))}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Desktop Bottom Telemetry Dashboard Row (Hidden on mobile screens) */}
              <div className="hidden lg:grid grid-cols-2 gap-4">
                <FpsTelemetryChart samples={perfSamples} />

                <div className="bg-[#10151C] border border-slate-800 rounded-lg p-4 flex flex-col">
                  <div className="flex items-center justify-between gap-3 mb-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100">
                        Recent Timestamp Entries
                      </h3>
                      <p className="text-xs text-slate-400">
                        Objects, gestures, and swipes written to {logFilePath}
                      </p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveTab('logs');
                        fetchLogs();
                      }}
                      className="text-xs font-medium text-cyan-400 hover:text-cyan-300 whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      Open Full Log
                    </button>
                  </div>

                  {logEntries.length === 0 ? (
                    <div className="flex-1 min-h-[148px] flex items-center justify-center border border-dashed border-slate-800 rounded text-xs text-slate-500">
                      Detected objects, hand gestures, and motion timestamps will appear here.
                    </div>
                  ) : (
                    <div className="divide-y divide-slate-800/80 font-mono text-xs tabular-nums overflow-hidden">
                      {logEntries.slice(0, 5).map((entry) => (
                        <div
                          key={entry.id}
                          className="py-2 flex items-center justify-between gap-3"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-slate-500 shrink-0">
                              {entry.timestamp.slice(11, 23)}
                            </span>
                            <span
                              className={`font-sans font-medium truncate ${
                                entry.entryType === 'hand_motion'
                                  ? 'text-amber-300'
                                  : entry.entryType === 'hand_gesture'
                                    ? 'text-emerald-300'
                                    : 'text-cyan-300'
                              }`}
                            >
                              {entry.label}
                            </span>
                          </div>
                          <div className="flex items-center gap-3 text-slate-400 shrink-0">
                            <span>{(entry.confidence * 100).toFixed(1)}%</span>
                            <span className="text-emerald-400">
                              {entry.fps.toFixed(1)} FPS
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'logs' && (
          <DetectionLogView
            entries={logEntries}
            rawText={rawLogText}
            filePath={logFilePath}
            byteSize={logByteSize}
            localFileLinkedName={localFileLinkedName}
            onLinkLocalFile={handleLinkLocalFile}
            onClearLogs={handleClearLogs}
            onDownloadLogs={handleDownloadLogs}
          />
        )}

        {activeTab === 'python' && <PythonOpenCvView settings={settings} />}
      </main>

      {/* Mobile Bottom Sheet for On-Demand Controls (keeps dashboard hidden by default) */}
      {mobileControlsOpen && (
        <div
          className="lg:hidden fixed inset-0 z-50 bg-black/65 backdrop-blur-xs flex items-end"
          onClick={() => setMobileControlsOpen(false)}
        >
          <div
            className="w-full max-h-[82dvh] bg-[#10151C] border-t border-slate-700 rounded-t-3xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="pt-3 pb-2 px-5 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
                <h3 className="text-base font-semibold text-slate-100">
                  Screening Controls
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setMobileControlsOpen(false)}
                aria-label="Close controls"
                className="min-h-[44px] min-w-[44px] inline-flex items-center justify-center text-slate-400 hover:text-slate-100 rounded-lg cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="overflow-y-auto p-5 space-y-5 text-sm">
              {cameras.length > 0 && (
                <div className="space-y-1.5">
                  <label
                    htmlFor="mobile-camera-select"
                    className="block text-sm font-medium text-slate-200"
                  >
                    Camera Device
                  </label>
                  <select
                    id="mobile-camera-select"
                    value={selectedCameraId}
                    onChange={(e) => {
                      const nextId = e.target.value;
                      setSelectedCameraId(nextId);
                      if (isStreaming) startCamera(nextId);
                    }}
                    className="w-full min-h-[44px] px-3 py-2.5 text-sm bg-[#090C10] border border-slate-800 rounded-lg text-slate-100 focus:outline-none focus:border-cyan-500"
                  >
                    {cameras.map((cam, idx) => (
                      <option key={cam.deviceId || idx} value={cam.deviceId}>
                        {cam.label || `Camera ${idx + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="font-medium text-slate-200">
                    Confidence Threshold
                  </span>
                  <span className="font-mono tabular-nums text-cyan-400">
                    {(settings.confidenceThreshold * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min={0.2}
                  max={0.9}
                  step={0.05}
                  value={settings.confidenceThreshold}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      confidenceThreshold: parseFloat(e.target.value),
                    }))
                  }
                  className="w-full h-10 accent-cyan-400 cursor-pointer"
                />
              </div>

              <div className="space-y-1 divide-y divide-slate-800/70">
                <label className="flex items-center justify-between min-h-[48px] text-sm text-slate-200 cursor-pointer">
                  <span>21-Point Hand Skeleton</span>
                  <input
                    type="checkbox"
                    checked={settings.showHandSkeleton}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        showHandSkeleton: e.target.checked,
                      }))
                    }
                    className="accent-cyan-400 w-5 h-5 rounded cursor-pointer"
                  />
                </label>
                <label className="flex items-center justify-between min-h-[48px] text-sm text-slate-200 cursor-pointer">
                  <span>360° Orientation Dial</span>
                  <input
                    type="checkbox"
                    checked={settings.showOrientationDial}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        showOrientationDial: e.target.checked,
                      }))
                    }
                    className="accent-cyan-400 w-5 h-5 rounded cursor-pointer"
                  />
                </label>
                <label className="flex items-center justify-between min-h-[48px] text-sm text-slate-200 cursor-pointer">
                  <span>Show FPS HUD on Video</span>
                  <input
                    type="checkbox"
                    checked={settings.showFpsOnCanvas}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        showFpsOnCanvas: e.target.checked,
                      }))
                    }
                    className="accent-cyan-400 w-5 h-5 rounded cursor-pointer"
                  />
                </label>
                <label className="flex items-center justify-between min-h-[48px] text-sm text-slate-200 cursor-pointer">
                  <span>Mirror Front Camera</span>
                  <input
                    type="checkbox"
                    checked={settings.mirrorCamera}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        mirrorCamera: e.target.checked,
                      }))
                    }
                    className="accent-cyan-400 w-5 h-5 rounded cursor-pointer"
                  />
                </label>
                <label className="flex items-center justify-between min-h-[48px] text-sm text-slate-200 cursor-pointer">
                  <span>Log Timestamps to .txt</span>
                  <input
                    type="checkbox"
                    checked={settings.loggingEnabled}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        loggingEnabled: e.target.checked,
                      }))
                    }
                    className="accent-cyan-400 w-5 h-5 rounded cursor-pointer"
                  />
                </label>
              </div>

              <div className="pt-2 grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={handleDownloadLogs}
                  className="inline-flex items-center justify-center gap-2 min-h-[44px] px-4 py-2.5 text-sm font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 border border-slate-700 rounded-lg cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  Export .txt
                </button>
                <button
                  type="button"
                  onClick={() => setMobileControlsOpen(false)}
                  className="inline-flex items-center justify-center min-h-[44px] px-4 py-2.5 text-sm font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg cursor-pointer"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mobile Bottom Thumb-Zone Navigation Bar (<= 15% sticky height cap) */}
      <nav
        aria-label="Mobile navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 z-40 h-14 bg-[#0C1016]/95 backdrop-blur-md border-t border-slate-800 grid grid-cols-3 items-center"
      >
        <button
          type="button"
          onClick={() => setActiveTab('monitor')}
          className={`flex flex-col items-center justify-center min-h-[44px] gap-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
            activeTab === 'monitor' ? 'text-cyan-400' : 'text-slate-400'
          }`}
        >
          <Camera className="w-5 h-5" />
          <span>Monitor</span>
        </button>
        <button
          type="button"
          onClick={() => {
            setActiveTab('logs');
            fetchLogs();
          }}
          className={`flex flex-col items-center justify-center min-h-[44px] gap-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
            activeTab === 'logs' ? 'text-cyan-400' : 'text-slate-400'
          }`}
        >
          <FileText className="w-5 h-5" />
          <span>Logs</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('python')}
          className={`flex flex-col items-center justify-center min-h-[44px] gap-0.5 text-[11px] font-medium transition-colors cursor-pointer ${
            activeTab === 'python' ? 'text-cyan-400' : 'text-slate-400'
          }`}
        >
          <Terminal className="w-5 h-5" />
          <span>OpenCV</span>
        </button>
      </nav>
    </div>
  );
}
