import type { GestureRecognizerResult } from '@mediapipe/tasks-vision';
import {
  FingerName,
  FingerTrackingState,
  HandLandmarkPoint,
  HandOrientation,
  TrackedHand,
} from '../types/detection';

const GESTURE_DISPLAY_MAP: Record<string, string> = {
  Open_Palm: 'Open Palm',
  Closed_Fist: 'Closed Fist',
  Pointing_Up: 'Pointing',
  Thumb_Up: 'Thumbs Up',
  Thumb_Down: 'Thumbs Down',
  Victory: 'Victory',
  ILoveYou: 'I Love You',
};

export interface HandHistorySample {
  palmX: number;
  palmY: number;
  rollDeg: number;
  tips: Array<{ x: number; y: number }>; // 5 fingertips [4, 8, 12, 16, 20] in px
  t: number;
}

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

function subVec(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

function dotVec(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

function lenVec(a: Vec3): number {
  return Math.hypot(a.x, a.y, a.z);
}

function crossVec(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

function angleBetweenDeg(v1: Vec3, v2: Vec3): number {
  const m1 = lenVec(v1);
  const m2 = lenVec(v2);
  if (m1 < 1e-6 || m2 < 1e-6) return 0;
  const cosVal = Math.max(-1, Math.min(1, dotVec(v1, v2) / (m1 * m2)));
  return (Math.acos(cosVal) * 180) / Math.PI;
}

function dist3D(a: Vec3, b: Vec3): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

const FINGER_SPECS: Array<{
  name: FingerName;
  joints: [number, number, number, number]; // [base, mid1, mid2, tip]
}> = [
  { name: 'Thumb', joints: [1, 2, 3, 4] },
  { name: 'Index', joints: [5, 6, 7, 8] },
  { name: 'Middle', joints: [9, 10, 11, 12] },
  { name: 'Ring', joints: [13, 14, 15, 16] },
  { name: 'Pinky', joints: [17, 18, 19, 20] },
];

/**
 * 360-degree orientation-invariant finger extension & bend calculation using 3D bone vectors.
 * Works regardless of whether the hand is upright, sideways, upside-down, or tilted.
 */
function computeAllFingersState(
  metricLms: Vec3[],
  imgLms: HandLandmarkPoint[],
  videoWidth: number,
  videoHeight: number,
  prevSample: HandHistorySample | undefined,
  dtSec: number
): FingerTrackingState[] {
  const wrist = metricLms[0];
  const palmSize = Math.max(1e-4, dist3D(metricLms[0], metricLms[9]));

  return FINGER_SPECS.map((spec, fingerIdx) => {
    const [j0, j1, j2, j3] = spec.joints;
    const p0 = metricLms[j0];
    const p1 = metricLms[j1];
    const p2 = metricLms[j2];
    const p3 = metricLms[j3];

    // Segment angles (0 deg = straight, higher = bent)
    const vBase = subVec(p1, p0);
    const vMid = subVec(p2, p1);
    const vTip = subVec(p3, p2);

    const bend1 = angleBetweenDeg(vBase, vMid);
    const bend2 = angleBetweenDeg(vMid, vTip);

    let bendAngleDeg: number;
    let extensionPct: number;

    if (spec.name === 'Thumb') {
      // Thumb flexion is measured by MCP/IP angles + separation from pinky/index base
      bendAngleDeg = (bend1 + bend2) * 0.5;
      const thumbReach = dist3D(p3, metricLms[17]) / palmSize;
      const thumbFromIndex = dist3D(p3, metricLms[5]) / palmSize;
      const angleScore = Math.max(0, Math.min(1, 1 - (bendAngleDeg - 10) / 75));
      const reachScore = Math.max(
        0,
        Math.min(1, (Math.max(thumbReach, thumbFromIndex * 1.1) - 0.55) / 0.75)
      );
      extensionPct = Math.round((angleScore * 0.55 + reachScore * 0.45) * 100);
    } else {
      // For fingers 2..5, combine MCP->PIP->DIP->TIP curl angle + tip-vs-PIP radial distance from wrist
      bendAngleDeg = bend1 * 0.65 + bend2 * 0.35;
      const angleScore = Math.max(0, Math.min(1, 1 - (bendAngleDeg - 14) / 92));
      const tipDist = dist3D(p3, wrist);
      const pipDist = Math.max(1e-4, dist3D(p1, wrist));
      const radialRatio = tipDist / pipDist;
      const radialScore = Math.max(0, Math.min(1, (radialRatio - 0.95) / 0.8));
      extensionPct = Math.round((angleScore * 0.6 + radialScore * 0.4) * 100);
    }

    const state: FingerTrackingState['state'] =
      extensionPct >= 62 ? 'Extended' : extensionPct >= 34 ? 'Half-Bent' : 'Curled';

    // Per-fingertip live speed in px/sec
    const tipImg = imgLms[j3];
    const tipPxX = tipImg.x * videoWidth;
    const tipPxY = tipImg.y * videoHeight;
    let speedPxPerSec = 0;
    if (prevSample && dtSec > 0) {
      const prevTip = prevSample.tips[fingerIdx];
      if (prevTip) {
        speedPxPerSec = Math.hypot(tipPxX - prevTip.x, tipPxY - prevTip.y) / dtSec;
      }
    }

    return {
      name: spec.name,
      tipIndex: j3,
      extensionPct,
      bendAngleDeg: Math.round(bendAngleDeg),
      state,
      speedPxPerSec,
      isMoving: speedPxPerSec >= 115,
    };
  });
}

function classify360Gesture(
  fingers: FingerTrackingState[],
  metricLms: Vec3[]
): { gesture: string; fingerCount: number } {
  const isExt = (idx: number) => fingers[idx].extensionPct >= 58;
  const thumbExt = isExt(0);
  const indexExt = isExt(1);
  const middleExt = isExt(2);
  const ringExt = isExt(3);
  const pinkyExt = isExt(4);

  const fingerCount = [thumbExt, indexExt, middleExt, ringExt, pinkyExt].filter(
    Boolean
  ).length;

  const palmWidth = Math.max(1e-4, dist3D(metricLms[5], metricLms[17]));
  const pinchRatio = dist3D(metricLms[4], metricLms[8]) / palmWidth;

  if (pinchRatio < 0.36 && (middleExt || ringExt)) {
    return { gesture: 'Pinch', fingerCount };
  }
  if (thumbExt && indexExt && middleExt && ringExt && pinkyExt) {
    return { gesture: 'Open Palm', fingerCount };
  }
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    if (thumbExt) {
      const dy = metricLms[4].y - metricLms[2].y;
      const dx = metricLms[4].x - metricLms[2].x;
      if (Math.abs(dy) >= Math.abs(dx)) {
        return { gesture: dy < 0 ? 'Thumbs Up' : 'Thumbs Down', fingerCount };
      }
      return { gesture: 'Thumb Out', fingerCount };
    }
    return { gesture: 'Closed Fist', fingerCount };
  }
  if (indexExt && middleExt && !ringExt && !pinkyExt) {
    return { gesture: 'Victory', fingerCount };
  }
  if (indexExt && !middleExt && !ringExt && !pinkyExt && !thumbExt) {
    return { gesture: 'Pointing', fingerCount };
  }
  if (thumbExt && indexExt && !middleExt && !ringExt && pinkyExt) {
    return { gesture: 'I Love You', fingerCount };
  }
  if (!thumbExt && indexExt && !middleExt && !ringExt && pinkyExt) {
    return { gesture: 'Rock / Horns', fingerCount };
  }
  if (thumbExt && !indexExt && !middleExt && !ringExt && pinkyExt) {
    return { gesture: 'Call Me', fingerCount };
  }
  if (fingerCount === 3) {
    return { gesture: 'Three Fingers', fingerCount };
  }
  if (fingerCount === 4) {
    return { gesture: 'Four Fingers', fingerCount };
  }
  return { gesture: 'Articulated', fingerCount };
}

function computeOrientation(
  imgLms: HandLandmarkPoint[],
  metricLms: Vec3[],
  handedness: 'Left' | 'Right',
  mirrorX: boolean,
  history: HandHistorySample[],
  nowMs: number
): HandOrientation {
  // 2D 360-deg roll angle from wrist (#0) to middle MCP (#9)
  // 0 deg = pointing straight up, 90 = right, 180 = down, 270 = left
  const rawDx = imgLms[9].x - imgLms[0].x;
  const dx = mirrorX ? -rawDx : rawDx;
  const dy = imgLms[9].y - imgLms[0].y;
  let rollDeg = (Math.atan2(dx, -dy) * 180) / Math.PI;
  if (rollDeg < 0) rollDeg += 360;

  // 3D palm normal vector to distinguish Palm Front vs Back of Hand vs Side Profile
  const vIndex = subVec(metricLms[5], metricLms[0]);
  const vPinky = subVec(metricLms[17], metricLms[0]);
  const normal = crossVec(vIndex, vPinky);
  const normalLen = Math.max(1e-6, lenVec(normal));
  const nz = normal.z / normalLen;

  // Sign flips between Left and Right hand anatomy
  const signedNz = handedness === 'Right' ? nz : -nz;
  let facing: HandOrientation['facing'] = 'Side Profile';
  if (signedNz > 0.32) {
    facing = 'Back of Hand';
  } else if (signedNz < -0.32) {
    facing = 'Palm Front';
  }

  // Angular velocity across recent ~140ms window
  let angularVelocityDegPerSec = 0;
  let rotationDirection: HandOrientation['rotationDirection'] = 'Steady';
  if (history.length >= 2) {
    let anchor = history[history.length - 1];
    for (let i = history.length - 1; i >= 0; i -= 1) {
      anchor = history[i];
      if (nowMs - anchor.t >= 90) break;
    }
    const dtSec = Math.max(0.016, (nowMs - anchor.t) / 1000);
    let deltaAngle = rollDeg - anchor.rollDeg;
    if (deltaAngle > 180) deltaAngle -= 360;
    if (deltaAngle < -180) deltaAngle += 360;
    angularVelocityDegPerSec = deltaAngle / dtSec;
    if (angularVelocityDegPerSec >= 65) {
      rotationDirection = 'Clockwise';
    } else if (angularVelocityDegPerSec <= -65) {
      rotationDirection = 'Counter-Clockwise';
    }
  }

  return {
    rollDeg: Math.round(rollDeg),
    facing,
    rotationDirection,
    angularVelocityDegPerSec: Math.round(angularVelocityDegPerSec),
  };
}

function analyzeHandMotion(
  history: HandHistorySample[],
  nowMs: number,
  mirrorX: boolean
): { direction: string; speedPxPerSec: number; swipeGesture: string | null } {
  if (history.length < 2) {
    return { direction: 'Stationary', speedPxPerSec: 0, swipeGesture: null };
  }

  const curr = history[history.length - 1];
  let anchor = history[history.length - 2];
  for (let i = history.length - 2; i >= 0; i -= 1) {
    anchor = history[i];
    if (curr.t - anchor.t >= 60) break;
  }

  const dtSec = Math.max(0.016, (curr.t - anchor.t) / 1000);
  const rawDx = curr.palmX - anchor.palmX;
  const dx = mirrorX ? -rawDx : rawDx;
  const dy = curr.palmY - anchor.palmY;
  const speedPxPerSec = Math.hypot(dx, dy) / dtSec;

  let direction = 'Stationary';
  if (speedPxPerSec >= 85) {
    if (Math.abs(dx) >= Math.abs(dy)) {
      direction = dx > 0 ? 'Moving Right' : 'Moving Left';
    } else {
      direction = dy > 0 ? 'Moving Down' : 'Moving Up';
    }
  }

  let swipeGesture: string | null = null;
  const recentWindow = history.filter((pt) => nowMs - pt.t <= 220);
  if (recentWindow.length >= 3) {
    const first = recentWindow[0];
    const last = recentWindow[recentWindow.length - 1];
    const wDt = Math.max(0.02, (last.t - first.t) / 1000);
    const rawWdx = last.palmX - first.palmX;
    const wdx = mirrorX ? -rawWdx : rawWdx;
    const wdy = last.palmY - first.palmY;
    const dist = Math.hypot(wdx, wdy);
    const avgSpeed = dist / wDt;

    if (dist >= 85 && avgSpeed >= 460) {
      if (Math.abs(wdx) >= Math.abs(wdy)) {
        swipeGesture = wdx > 0 ? 'Swipe Right' : 'Swipe Left';
      } else {
        swipeGesture = wdy > 0 ? 'Swipe Down' : 'Swipe Up';
      }
    }
  }

  return { direction, speedPxPerSec, swipeGesture };
}

export function extractTrackedHands(
  result: GestureRecognizerResult,
  videoWidth: number,
  videoHeight: number,
  historyMap: Map<string, HandHistorySample[]>,
  nowMs: number,
  mirrorX: boolean
): TrackedHand[] {
  const landmarksList = result.landmarks ?? [];
  if (landmarksList.length === 0) {
    for (const [key, pts] of historyMap.entries()) {
      if (pts.length === 0 || nowMs - pts[pts.length - 1].t > 450) {
        historyMap.delete(key);
      }
    }
    return [];
  }

  const activeKeys = new Set<string>();
  const hands: TrackedHand[] = [];

  landmarksList.forEach((rawLms, idx) => {
    const imgLms: HandLandmarkPoint[] = rawLms.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z ?? 0,
    }));

    const worldRaw = result.worldLandmarks?.[idx];
    const metricLms: Vec3[] =
      worldRaw && worldRaw.length === 21
        ? worldRaw.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 }))
        : imgLms;

    const handednessCat = result.handedness?.[idx]?.[0];
    const rawHandLabel = handednessCat?.categoryName === 'Left' ? 'Left' : 'Right';
    const handednessScore = handednessCat?.score ?? 0.9;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of imgLms) {
      const px = p.x * videoWidth;
      const py = p.y * videoHeight;
      if (px < minX) minX = px;
      if (py < minY) minY = py;
      if (px > maxX) maxX = px;
      if (py > maxY) maxY = py;
    }

    const pad = 16;
    const bx = Math.max(0, Math.round(minX - pad));
    const by = Math.max(0, Math.round(minY - pad));
    const bw = Math.min(videoWidth - bx, Math.max(1, Math.round(maxX - minX + pad * 2)));
    const bh = Math.min(videoHeight - by, Math.max(1, Math.round(maxY - minY + pad * 2)));

    const palmIndices = [0, 5, 9, 13, 17];
    const palmX =
      (palmIndices.reduce((acc, i) => acc + imgLms[i].x, 0) / palmIndices.length) *
      videoWidth;
    const palmY =
      (palmIndices.reduce((acc, i) => acc + imgLms[i].y, 0) / palmIndices.length) *
      videoHeight;

    const handKey = `${rawHandLabel}-${idx}`;
    activeKeys.add(handKey);
    const prevHistory = historyMap.get(handKey) ?? [];
    const prevSample = prevHistory[prevHistory.length - 1];
    const dtSec = prevSample ? Math.max(0.016, (nowMs - prevSample.t) / 1000) : 0;

    const orientation = computeOrientation(
      imgLms,
      metricLms,
      rawHandLabel,
      mirrorX,
      prevHistory,
      nowMs
    );

    const fingers = computeAllFingersState(
      metricLms,
      imgLms,
      videoWidth,
      videoHeight,
      prevSample,
      dtSec
    );

    const currSample: HandHistorySample = {
      palmX,
      palmY,
      rollDeg: orientation.rollDeg,
      tips: [4, 8, 12, 16, 20].map((ti) => ({
        x: imgLms[ti].x * videoWidth,
        y: imgLms[ti].y * videoHeight,
      })),
      t: nowMs,
    };

    const nextHistory = [
      ...prevHistory.filter((pt) => nowMs - pt.t <= 450),
      currSample,
    ];
    historyMap.set(handKey, nextHistory);

    const geo360 = classify360Gesture(fingers, metricLms);
    const topGesture = result.gestures?.[idx]?.[0];

    let gesture = geo360.gesture;
    let gestureConfidence = handednessScore;

    if (
      topGesture &&
      topGesture.categoryName &&
      topGesture.categoryName !== 'None' &&
      topGesture.score >= 0.55 &&
      geo360.gesture !== 'Pinch'
    ) {
      gesture =
        GESTURE_DISPLAY_MAP[topGesture.categoryName] ?? topGesture.categoryName;
      gestureConfidence = topGesture.score;
    }

    const motion = analyzeHandMotion(nextHistory, nowMs, mirrorX);

    hands.push({
      id: handKey,
      handedness: rawHandLabel,
      handednessScore,
      gesture,
      gestureConfidence,
      fingerCount: geo360.fingerCount,
      fingers,
      orientation,
      direction: motion.direction,
      speedPxPerSec: motion.speedPxPerSec,
      swipeGesture: motion.swipeGesture,
      bbox: [bx, by, bw, bh],
      landmarks: imgLms,
      palmCenter: { x: palmX, y: palmY },
    });
  });

  for (const key of historyMap.keys()) {
    if (!activeKeys.has(key)) {
      const pts = historyMap.get(key);
      if (!pts || pts.length === 0 || nowMs - pts[pts.length - 1].t > 450) {
        historyMap.delete(key);
      }
    }
  }

  return hands;
}
