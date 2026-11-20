import type { GestureRecognizerResult } from '@mediapipe/tasks-vision';
import { HandLandmarkPoint, HandTrailPoint, TrackedHand } from '../types/detection';

const GESTURE_DISPLAY_MAP: Record<string, string> = {
  Open_Palm: 'Open Palm',
  Closed_Fist: 'Closed Fist',
  Pointing_Up: 'Pointing Up',
  Thumb_Up: 'Thumbs Up',
  Thumb_Down: 'Thumbs Down',
  Victory: 'Victory',
  ILoveYou: 'I Love You',
};

const TRAIL_WINDOW_MS = 450;

function dist2D(a: HandLandmarkPoint, b: HandLandmarkPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function classifyGeometricFallback(
  lms: HandLandmarkPoint[],
  handedness: 'Left' | 'Right'
): { gesture: string; fingerCount: number } {
  const palmWidth = Math.max(1e-4, dist2D(lms[5], lms[17]));
  const pinchRatio = dist2D(lms[4], lms[8]) / palmWidth;

  const thumbOpen =
    handedness === 'Right'
      ? lms[4].x < lms[3].x - 0.02
      : lms[4].x > lms[3].x + 0.02;
  const indexOpen = lms[8].y < lms[6].y - 0.02;
  const middleOpen = lms[12].y < lms[10].y - 0.02;
  const ringOpen = lms[16].y < lms[14].y - 0.02;
  const pinkyOpen = lms[20].y < lms[18].y - 0.02;

  const fingerCount = [
    thumbOpen,
    indexOpen,
    middleOpen,
    ringOpen,
    pinkyOpen,
  ].filter(Boolean).length;

  if (pinchRatio < 0.34 && (middleOpen || ringOpen)) {
    return { gesture: 'Pinch', fingerCount };
  }
  if (indexOpen && middleOpen && ringOpen && pinkyOpen && thumbOpen) {
    return { gesture: 'Open Palm', fingerCount };
  }
  if (!indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    if (dist2D(lms[4], lms[2]) > palmWidth * 0.45) {
      if (lms[4].y < lms[2].y - 0.04) {
        return { gesture: 'Thumbs Up', fingerCount };
      }
      if (lms[4].y > lms[2].y + 0.04) {
        return { gesture: 'Thumbs Down', fingerCount };
      }
    }
    return { gesture: 'Closed Fist', fingerCount };
  }
  if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
    return { gesture: 'Victory', fingerCount };
  }
  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    return { gesture: 'Pointing Up', fingerCount };
  }
  if (fingerCount === 3) {
    return { gesture: 'Three Fingers', fingerCount };
  }
  if (fingerCount === 4) {
    return { gesture: 'Four Fingers', fingerCount };
  }
  return { gesture: 'Tracked Hand', fingerCount };
}

function analyzeTrailMovement(
  trail: HandTrailPoint[],
  nowMs: number,
  mirrorX: boolean
): { direction: string; speedPxPerSec: number; swipeGesture: string | null } {
  if (trail.length < 2) {
    return { direction: 'Stationary', speedPxPerSec: 0, swipeGesture: null };
  }

  // Use a short ~90ms baseline so velocity doesn't spike from 1-frame landmark jitter
  const curr = trail[trail.length - 1];
  let anchor = trail[trail.length - 2];
  for (let i = trail.length - 2; i >= 0; i -= 1) {
    anchor = trail[i];
    if (curr.t - anchor.t >= 60) {
      break;
    }
  }

  const dtSec = Math.max(0.016, (curr.t - anchor.t) / 1000);
  const rawDx = curr.x - anchor.x;
  const dx = mirrorX ? -rawDx : rawDx;
  const dy = curr.y - anchor.y;
  const speedPxPerSec = Math.hypot(dx, dy) / dtSec;

  let direction = 'Stationary';
  if (speedPxPerSec >= 85) {
    if (Math.abs(dx) >= absVal(dy)) {
      direction = dx > 0 ? 'Moving Right' : 'Moving Left';
    } else {
      direction = dy > 0 ? 'Moving Down' : 'Moving Up';
    }
  }

  // Evaluate sustained ~220ms window for rapid directional swipe recognition
  let swipeGesture: string | null = null;
  const recentWindow = trail.filter((pt) => nowMs - pt.t <= 220);
  if (recentWindow.length >= 3) {
    const first = recentWindow[0];
    const last = recentWindow[recentWindow.length - 1];
    const wDt = Math.max(0.02, (last.t - first.t) / 1000);
    const rawWdx = last.x - first.x;
    const wdx = mirrorX ? -rawWdx : rawWdx;
    const wdy = last.y - first.y;
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

function absVal(n: number): number {
  return n < 0 ? -n : n;
}

export function extractTrackedHands(
  result: GestureRecognizerResult,
  videoWidth: number,
  videoHeight: number,
  trailMap: Map<string, HandTrailPoint[]>,
  nowMs: number,
  mirrorX: boolean
): TrackedHand[] {
  const landmarksList = result.landmarks ?? [];
  if (landmarksList.length === 0) {
    for (const [key, pts] of trailMap.entries()) {
      if (pts.length === 0 || nowMs - pts[pts.length - 1].t > TRAIL_WINDOW_MS) {
        trailMap.delete(key);
      }
    }
    return [];
  }

  const activeKeys = new Set<string>();
  const hands: TrackedHand[] = [];

  landmarksList.forEach((rawLms, idx) => {
    const lms: HandLandmarkPoint[] = rawLms.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z ?? 0,
    }));

    const handednessCat = result.handedness?.[idx]?.[0];
    const rawHandLabel = handednessCat?.categoryName === 'Left' ? 'Left' : 'Right';
    const handednessScore = handednessCat?.score ?? 0.9;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of lms) {
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
      (palmIndices.reduce((acc, i) => acc + lms[i].x, 0) / palmIndices.length) *
      videoWidth;
    const palmY =
      (palmIndices.reduce((acc, i) => acc + lms[i].y, 0) / palmIndices.length) *
      videoHeight;

    const handKey = `${rawHandLabel}-${idx}`;
    activeKeys.add(handKey);
    const prevTrail = trailMap.get(handKey) ?? [];
    // Keep strictly time-windowed points (450ms) so trail stays responsive at any FPS
    const nextTrail = [
      ...prevTrail.filter((pt) => nowMs - pt.t <= TRAIL_WINDOW_MS),
      { x: palmX, y: palmY, t: nowMs },
    ];
    trailMap.set(handKey, nextTrail);

    const topGesture = result.gestures?.[idx]?.[0];
    const geometric = classifyGeometricFallback(lms, rawHandLabel);

    let gesture = geometric.gesture;
    let gestureConfidence = handednessScore;

    if (
      topGesture &&
      topGesture.categoryName &&
      topGesture.categoryName !== 'None' &&
      topGesture.score >= 0.48
    ) {
      gesture =
        GESTURE_DISPLAY_MAP[topGesture.categoryName] ?? topGesture.categoryName;
      gestureConfidence = topGesture.score;
    } else if (geometric.gesture === 'Pinch') {
      gesture = 'Pinch';
      gestureConfidence = Math.max(0.75, handednessScore);
    }

    const movement = analyzeTrailMovement(nextTrail, nowMs, mirrorX);

    hands.push({
      id: handKey,
      handedness: rawHandLabel,
      handednessScore,
      gesture,
      gestureConfidence,
      fingerCount: geometric.fingerCount,
      direction: movement.direction,
      speedPxPerSec: movement.speedPxPerSec,
      swipeGesture: movement.swipeGesture,
      bbox: [bx, by, bw, bh],
      landmarks: lms,
      palmCenter: { x: palmX, y: palmY },
      trail: nextTrail,
    });
  });

  for (const key of trailMap.keys()) {
    if (!activeKeys.has(key)) {
      const pts = trailMap.get(key);
      if (!pts || pts.length === 0 || nowMs - pts[pts.length - 1].t > TRAIL_WINDOW_MS) {
        trailMap.delete(key);
      }
    }
  }

  return hands;
}
