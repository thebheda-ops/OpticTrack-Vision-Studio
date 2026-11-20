export interface DetectedObject {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, width, height] in video pixels
}

export interface HandLandmarkPoint {
  x: number; // normalized [0..1]
  y: number; // normalized [0..1]
  z: number;
}

export type FingerName = 'Thumb' | 'Index' | 'Middle' | 'Ring' | 'Pinky';

export interface FingerTrackingState {
  name: FingerName;
  tipIndex: number;
  extensionPct: number; // 0 (curled) to 100 (fully extended)
  bendAngleDeg: number; // 0 (straight) to ~170 (tightly curled)
  state: 'Extended' | 'Half-Bent' | 'Curled';
  speedPxPerSec: number;
  isMoving: boolean;
}

export interface HandOrientation {
  rollDeg: number; // 0..360 deg in camera plane (0 = pointing up, 90 = right, 180 = down, 270 = left)
  facing: 'Palm Front' | 'Back of Hand' | 'Side Profile';
  rotationDirection: 'Clockwise' | 'Counter-Clockwise' | 'Steady';
  angularVelocityDegPerSec: number;
}

export interface TrackedHand {
  id: string;
  handedness: 'Left' | 'Right';
  handednessScore: number;
  gesture: string;
  gestureConfidence: number;
  fingerCount: number;
  fingers: FingerTrackingState[];
  orientation: HandOrientation;
  direction: string;
  speedPxPerSec: number;
  swipeGesture: string | null;
  bbox: [number, number, number, number];
  landmarks: HandLandmarkPoint[];
  palmCenter: { x: number; y: number };
}

export type LogEntryType = 'object' | 'hand_gesture' | 'hand_motion';

export interface LogEntry {
  id: string;
  timestamp: string;
  entryType: LogEntryType;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  fps: number;
  motion?: string;
  rawLine: string;
}

export interface PerformanceSample {
  timestamp: number;
  fps: number;
  inferenceMs: number;
  objectCount: number;
  handCount: number;
}

export type DetectionMode = 'all' | 'objects' | 'hands';
export type CocoBaseModel = 'lite_mobilenet_v2' | 'mobilenet_v2';
export type PythonModelPreset = 'yolov4-tiny' | 'mobilenet-ssd';
export type StreamSourceMode = 'camera' | 'video';
export type WorkspaceTab = 'monitor' | 'logs' | 'python';

export interface DetectorSettings {
  detectionMode: DetectionMode;
  modelBase: CocoBaseModel;
  confidenceThreshold: number;
  maxDetections: number;
  maxHands: number;
  logCooldownMs: number;
  loggingEnabled: boolean;
  showFpsOnCanvas: boolean;
  showHandSkeleton: boolean;
  showOrientationDial: boolean;
  mirrorCamera: boolean;
  pythonModel: PythonModelPreset;
}
