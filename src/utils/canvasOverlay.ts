import { DetectedObject, TrackedHand } from '../types/detection';

const CLASS_PALETTE = [
  '#06B6D4', // Cyan
  '#10B981', // Emerald
  '#F59E0B', // Amber
  '#3B82F6', // Blue
  '#EC4899', // Pink
  '#8B5CF6', // Violet
];

// Standard 21-keypoint MediaPipe hand bone connections
const HAND_BONE_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],// Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17],            // Palm arch
];

export function getColorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return CLASS_PALETTE[Math.abs(hash) % CLASS_PALETTE.length];
}

export interface OverlayRenderOptions {
  mirrorX: boolean;
  showFpsHud: boolean;
  showHandSkeleton: boolean;
  showHandTrails: boolean;
  fps: number;
  inferenceMs: number;
  loggingEnabled: boolean;
}

export function renderDetectionOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  detections: DetectedObject[],
  hands: TrackedHand[],
  options: OverlayRenderOptions
): void {
  ctx.clearRect(0, 0, width, height);

  const scaleFactor = Math.max(1, Math.min(width / 960, 1.4));
  const lineW = Math.round(2 * scaleFactor);
  const fontSize = Math.round(13 * scaleFactor);
  ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
  ctx.textBaseline = 'top';

  // 1. Render object bounding boxes & labels
  for (const det of detections) {
    const [rawX, y, w, h] = det.bbox;
    const x = options.mirrorX ? width - (rawX + w) : rawX;
    const color = getColorForLabel(det.label);

    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.strokeRect(x, y, w, h);

    const cornerLen = Math.min(16 * scaleFactor, w * 0.25, h * 0.25);
    ctx.lineWidth = lineW + 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y + cornerLen);
    ctx.lineTo(x, y);
    ctx.lineTo(x + cornerLen, y);
    ctx.moveTo(x + w - cornerLen, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + cornerLen);
    ctx.moveTo(x, y + h - cornerLen);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + cornerLen, y + h);
    ctx.moveTo(x + w - cornerLen, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - cornerLen);
    ctx.stroke();

    const labelText = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
    const metrics = ctx.measureText(labelText);
    const padX = 7 * scaleFactor;
    const tagH = fontSize + 8 * scaleFactor;
    const tagW = metrics.width + padX * 2;
    const tagY = y >= tagH + 2 ? y - tagH : y;
    const tagX = Math.max(0, Math.min(x, width - tagW));

    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, tagW, tagH);

    ctx.fillStyle = '#090C10';
    ctx.fillText(labelText, tagX + padX, tagY + 4 * scaleFactor);
  }

  // 2. Render tracked hands, 21-point skeleton, motion trajectory trails, & gesture labels
  for (const hand of hands) {
    const toScreenX = (rawPxX: number) =>
      options.mirrorX ? width - rawPxX : rawPxX;

    // Motion trajectory trail
    if (options.showHandTrails && hand.trail.length >= 2) {
      const total = hand.trail.length;
      for (let i = 1; i < total; i += 1) {
        const p0 = hand.trail[i - 1];
        const p1 = hand.trail[i];
        const progress = i / total;
        ctx.strokeStyle = hand.swipeGesture
          ? `rgba(245, 158, 11, ${(0.2 + progress * 0.75).toFixed(2)})`
          : `rgba(6, 182, 212, ${(0.15 + progress * 0.75).toFixed(2)})`;
        ctx.lineWidth = Math.max(1.5, progress * 5 * scaleFactor);
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(toScreenX(p0.x), p0.y);
        ctx.lineTo(toScreenX(p1.x), p1.y);
        ctx.stroke();
      }
    }

    // 21-landmark skeleton
    if (options.showHandSkeleton && hand.landmarks.length === 21) {
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = Math.max(1.5, 2 * scaleFactor);
      for (const [a, b] of HAND_BONE_CONNECTIONS) {
        const la = hand.landmarks[a];
        const lb = hand.landmarks[b];
        const ax = toScreenX(la.x * width);
        const ay = la.y * height;
        const bx = toScreenX(lb.x * width);
        const by = lb.y * height;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }

      hand.landmarks.forEach((lm, idx) => {
        const lx = toScreenX(lm.x * width);
        const ly = lm.y * height;
        const isFingertip = idx === 4 || idx === 8 || idx === 12 || idx === 16 || idx === 20;
        ctx.fillStyle = isFingertip ? '#F59E0B' : '#06B6D4';
        ctx.beginPath();
        ctx.arc(
          lx,
          ly,
          (isFingertip ? 4.5 : 3) * scaleFactor,
          0,
          Math.PI * 2
        );
        ctx.fill();
      });
    }

    // Hand bounding box + gesture & movement label
    const [rawBx, by, bw, bh] = hand.bbox;
    const bx = options.mirrorX ? width - (rawBx + bw) : rawBx;
    const handColor = hand.swipeGesture ? '#F59E0B' : '#10B981';

    ctx.strokeStyle = handColor;
    ctx.lineWidth = Math.max(1.5, 1.5 * scaleFactor);
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);

    const activeGesture = hand.swipeGesture ?? hand.gesture;
    const speedLabel =
      hand.speedPxPerSec >= 75
        ? ` · ${hand.direction} (${Math.round(hand.speedPxPerSec)} px/s)`
        : '';
    const handLabel = `${hand.handedness} Hand: ${activeGesture} ${(hand.gestureConfidence * 100).toFixed(0)}%${speedLabel}`;

    ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
    const metrics = ctx.measureText(handLabel);
    const padX = 8 * scaleFactor;
    const tagH = fontSize + 8 * scaleFactor;
    const tagW = metrics.width + padX * 2;
    const tagY = by >= tagH + 2 ? by - tagH : by;
    const tagX = Math.max(0, Math.min(bx, width - tagW));

    ctx.fillStyle = handColor;
    ctx.fillRect(tagX, tagY, tagW, tagH);

    ctx.fillStyle = '#090C10';
    ctx.fillText(handLabel, tagX + padX, tagY + 4 * scaleFactor);
  }

  // 3. Optional OpenCV-style on-frame FPS & telemetry HUD
  if (options.showFpsHud) {
    const hudPad = 12 * scaleFactor;
    const hudW = 320 * scaleFactor;
    const hudH = 62 * scaleFactor;

    ctx.fillStyle = 'rgba(9, 12, 16, 0.82)';
    ctx.fillRect(hudPad, hudPad, hudW, hudH);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.9)';
    ctx.lineWidth = 1;
    ctx.strokeRect(hudPad, hudPad, hudW, hudH);

    const fpsFontSize = Math.round(14 * scaleFactor);
    ctx.font = `600 ${fpsFontSize}px "JetBrains Mono", monospace`;
    ctx.fillStyle = options.fps >= 20 ? '#10B981' : '#F59E0B';
    ctx.fillText(
      `FPS: ${options.fps.toFixed(1)}  |  INFER: ${options.inferenceMs.toFixed(1)} ms`,
      hudPad + 12 * scaleFactor,
      hudPad + 11 * scaleFactor
    );

    const subFontSize = Math.round(11 * scaleFactor);
    ctx.font = `500 ${subFontSize}px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#CBD5E1';
    ctx.fillText(
      `OBJ: ${detections.length}  |  HANDS: ${hands.length}  |  LOG: ${options.loggingEnabled ? 'ACTIVE' : 'PAUSED'}`,
      hudPad + 12 * scaleFactor,
      hudPad + 36 * scaleFactor
    );
  }
}
