import { DetectedObject, TrackedHand } from '../types/detection';

const CLASS_PALETTE = [
  '#06B6D4', // Cyan
  '#10B981', // Emerald
  '#F59E0B', // Amber
  '#3B82F6', // Blue
  '#EC4899', // Pink
  '#8B5CF6', // Violet
];

// Grouped by finger so we can color-code each finger's individual movement & flexion state
const FINGER_BONE_CHAINS: Array<{
  fingerIdx: number; // -1 for palm arch, 0..4 for Thumb..Pinky
  segments: Array<[number, number]>;
}> = [
  { fingerIdx: -1, segments: [[5, 9], [9, 13], [13, 17], [0, 5], [0, 17]] },
  { fingerIdx: 0, segments: [[0, 1], [1, 2], [2, 3], [3, 4]] },
  { fingerIdx: 1, segments: [[5, 6], [6, 7], [7, 8]] },
  { fingerIdx: 2, segments: [[9, 10], [10, 11], [11, 12]] },
  { fingerIdx: 3, segments: [[13, 14], [14, 15], [15, 16]] },
  { fingerIdx: 4, segments: [[17, 18], [18, 19], [19, 20]] },
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
  showOrientationDial: boolean;
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

  const minDim = Math.min(width, height);
  const scaleFactor = Math.max(0.72, Math.min(minDim / 600, 1.35));
  const lineW = Math.max(1.5, Math.round(2 * scaleFactor));
  const fontSize = Math.max(11, Math.round(13 * scaleFactor));
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
    const tagW = Math.min(width - 4, metrics.width + padX * 2);
    const tagY = y >= tagH + 2 ? y - tagH : y;
    const tagX = Math.max(0, Math.min(x, width - tagW));

    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, tagW, tagH);

    ctx.fillStyle = '#090C10';
    ctx.fillText(labelText, tagX + padX, tagY + 4 * scaleFactor, tagW - padX * 2);
  }

  // 2. Render 360-degree articulated 21-point hand skeleton & individual finger states
  //    (No blue gesture tracking trail line)
  for (const hand of hands) {
    const toScreenX = (rawPxX: number) =>
      options.mirrorX ? width - rawPxX : rawPxX;

    // Optional subtle 360° orientation dial centered on the wrist/palm
    if (options.showOrientationDial && hand.landmarks.length === 21) {
      const palmScreenX = toScreenX(hand.palmCenter.x);
      const palmScreenY = hand.palmCenter.y;
      const radius = 22 * scaleFactor;
      const angleRad = ((hand.orientation.rollDeg - 90) * Math.PI) / 180;

      ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
      ctx.lineWidth = 1.5 * scaleFactor;
      ctx.beginPath();
      ctx.arc(palmScreenX, palmScreenY, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Orientation pointer tick showing 360° hand axis
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2.2 * scaleFactor;
      ctx.beginPath();
      ctx.moveTo(
        palmScreenX + Math.cos(angleRad) * (radius - 5 * scaleFactor),
        palmScreenY + Math.sin(angleRad) * (radius - 5 * scaleFactor)
      );
      ctx.lineTo(
        palmScreenX + Math.cos(angleRad) * (radius + 7 * scaleFactor),
        palmScreenY + Math.sin(angleRad) * (radius + 7 * scaleFactor)
      );
      ctx.stroke();
    }

    // Articulated per-finger 21-point skeleton
    if (options.showHandSkeleton && hand.landmarks.length === 21) {
      for (const chain of FINGER_BONE_CHAINS) {
        if (chain.fingerIdx === -1) {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.72)';
          ctx.lineWidth = 1.6 * scaleFactor;
        } else {
          const finger = hand.fingers[chain.fingerIdx];
          if (finger?.isMoving) {
            ctx.strokeStyle = '#F59E0B'; // Active individual finger movement
            ctx.lineWidth = 2.6 * scaleFactor;
          } else if (finger?.state === 'Extended') {
            ctx.strokeStyle = '#10B981'; // Extended finger
            ctx.lineWidth = 2.1 * scaleFactor;
          } else {
            ctx.strokeStyle = '#64748B'; // Curled / half-bent finger
            ctx.lineWidth = 1.8 * scaleFactor;
          }
        }

        for (const [a, b] of chain.segments) {
          const la = hand.landmarks[a];
          const lb = hand.landmarks[b];
          ctx.beginPath();
          ctx.moveTo(toScreenX(la.x * width), la.y * height);
          ctx.lineTo(toScreenX(lb.x * width), lb.y * height);
          ctx.stroke();
        }
      }

      // Draw all 21 joints without blue/cyan blobs
      hand.landmarks.forEach((lm, idx) => {
        const lx = toScreenX(lm.x * width);
        const ly = lm.y * height;

        const fingerIdx =
          idx >= 1 && idx <= 4
            ? 0
            : idx >= 5 && idx <= 8
              ? 1
              : idx >= 9 && idx <= 12
                ? 2
                : idx >= 13 && idx <= 16
                  ? 3
                  : idx >= 17 && idx <= 20
                    ? 4
                    : -1;
        const isTip = idx === 4 || idx === 8 || idx === 12 || idx === 16 || idx === 20;
        const fingerState = fingerIdx >= 0 ? hand.fingers[fingerIdx] : undefined;

        if (isTip && fingerState) {
          ctx.fillStyle = fingerState.isMoving
            ? '#F59E0B'
            : fingerState.state === 'Extended'
              ? '#10B981'
              : '#94A3B8';
          ctx.beginPath();
          ctx.arc(lx, ly, (fingerState.isMoving ? 5 : 4) * scaleFactor, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillStyle = '#E2E8F0';
          ctx.beginPath();
          ctx.arc(lx, ly, 2.6 * scaleFactor, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }

    // Hand bounding box + 360-degree orientation & gesture label
    const [rawBx, by, bw, bh] = hand.bbox;
    const bx = options.mirrorX ? width - (rawBx + bw) : rawBx;
    const handColor = hand.swipeGesture ? '#F59E0B' : '#10B981';

    ctx.strokeStyle = handColor;
    ctx.lineWidth = Math.max(1.5, 1.5 * scaleFactor);
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);

    const activeGesture = hand.swipeGesture ?? hand.gesture;
    const handLabel =
      width < 540
        ? `${hand.handedness[0]} · ${activeGesture} · ${hand.orientation.rollDeg}°`
        : `${hand.handedness} · ${activeGesture} · ${hand.orientation.rollDeg}° · ${hand.orientation.facing}`;

    ctx.font = `600 ${fontSize}px "JetBrains Mono", monospace`;
    const metrics = ctx.measureText(handLabel);
    const padX = 7 * scaleFactor;
    const tagH = fontSize + 8 * scaleFactor;
    const tagW = Math.min(width - 4, metrics.width + padX * 2);
    const tagY = by >= tagH + 2 ? by - tagH : by;
    const tagX = Math.max(0, Math.min(bx, width - tagW));

    ctx.fillStyle = handColor;
    ctx.fillRect(tagX, tagY, tagW, tagH);

    ctx.fillStyle = '#090C10';
    ctx.fillText(handLabel, tagX + padX, tagY + 4 * scaleFactor, tagW - padX * 2);
  }

  // 3. Optional OpenCV-style on-frame FPS & telemetry HUD
  if (options.showFpsHud) {
    const hudPad = Math.max(8, 12 * scaleFactor);
    const hudW = Math.min(width - hudPad * 2, 310 * scaleFactor);
    const hudH = Math.max(46, 58 * scaleFactor);

    ctx.fillStyle = 'rgba(9, 12, 16, 0.82)';
    ctx.fillRect(hudPad, hudPad, hudW, hudH);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.9)';
    ctx.lineWidth = 1;
    ctx.strokeRect(hudPad, hudPad, hudW, hudH);

    const fpsFontSize = Math.max(11, Math.round(13.5 * scaleFactor));
    ctx.font = `600 ${fpsFontSize}px "JetBrains Mono", monospace`;
    ctx.fillStyle = options.fps >= 20 ? '#10B981' : '#F59E0B';
    ctx.fillText(
      `FPS: ${options.fps.toFixed(1)}  |  ${options.inferenceMs.toFixed(1)} ms`,
      hudPad + 10 * scaleFactor,
      hudPad + 9 * scaleFactor,
      hudW - 20 * scaleFactor
    );

    const subFontSize = Math.max(10, Math.round(11 * scaleFactor));
    ctx.font = `500 ${subFontSize}px "JetBrains Mono", monospace`;
    ctx.fillStyle = '#CBD5E1';
    ctx.fillText(
      `OBJ: ${detections.length}  |  HANDS: ${hands.length}  |  LOG: ${options.loggingEnabled ? 'ON' : 'OFF'}`,
      hudPad + 10 * scaleFactor,
      hudPad + 32 * scaleFactor,
      hudW - 20 * scaleFactor
    );
  }
}
