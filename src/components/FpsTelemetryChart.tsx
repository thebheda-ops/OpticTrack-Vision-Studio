import React, { useState } from 'react';
import { PerformanceSample } from '../types/detection';

interface FpsTelemetryChartProps {
  samples: PerformanceSample[];
}

export const FpsTelemetryChart: React.FC<FpsTelemetryChartProps> = ({ samples }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const width = 560;
  const height = 148;
  const padLeft = 36;
  const padRight = 12;
  const padTop = 14;
  const padBottom = 24;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const maxFps = Math.max(
    45,
    ...samples.map((s) => Math.ceil(s.fps / 15) * 15)
  );

  const points = samples.map((sample, idx) => {
    const x =
      samples.length <= 1
        ? padLeft + plotW
        : padLeft + (idx / (samples.length - 1)) * plotW;
    const clampedFps = Math.max(0, Math.min(sample.fps, maxFps));
    const y = padTop + plotH - (clampedFps / maxFps) * plotH;
    return { x, y, sample };
  });

  const linePath =
    points.length > 0
      ? points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
      : '';

  const areaPath =
    points.length > 1
      ? `${linePath} L ${points[points.length - 1].x.toFixed(1)} ${(padTop + plotH).toFixed(1)} L ${points[0].x.toFixed(1)} ${(padTop + plotH).toFixed(1)} Z`
      : '';

  const activePoint =
    hoverIndex !== null && points[hoverIndex]
      ? points[hoverIndex]
      : points[points.length - 1] ?? null;

  const yTicks = [0, Math.round(maxFps / 2), maxFps];

  return (
    <div className="bg-[#10151C] border border-slate-800 rounded-lg p-4">
      <div className="flex items-center justify-between gap-4 mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            Frame Rate Telemetry
          </h3>
          <p className="text-xs text-slate-400">
            Rolling frame rate and inference latency across recent frames
          </p>
        </div>
        {activePoint ? (
          <div className="flex items-center gap-3 text-xs font-mono tabular-nums text-slate-300 shrink-0">
            <span className="text-cyan-400 font-semibold">
              {activePoint.sample.fps.toFixed(1)} FPS
            </span>
            <span className="text-slate-600" aria-hidden="true">·</span>
            <span>{activePoint.sample.inferenceMs.toFixed(1)} ms</span>
            <span className="text-slate-600" aria-hidden="true">·</span>
            <span>{activePoint.sample.objectCount} obj</span>
          </div>
        ) : (
          <span className="text-xs font-mono text-slate-500">Awaiting frames</span>
        )}
      </div>

      {samples.length === 0 ? (
        <div className="h-[148px] flex items-center justify-center border border-dashed border-slate-800 rounded text-xs text-slate-500">
          Start the camera or upload a video stream to monitor live frame rate.
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-[148px] overflow-visible select-none"
          onMouseLeave={() => setHoverIndex(null)}
          onMouseMove={(e) => {
            if (points.length === 0) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = ((e.clientX - rect.left) / rect.width) * width;
            let nearest = 0;
            let bestDist = Infinity;
            points.forEach((p, idx) => {
              const d = Math.abs(p.x - relX);
              if (d < bestDist) {
                bestDist = d;
                nearest = idx;
              }
            });
            setHoverIndex(nearest);
          }}
        >
          <defs>
            <linearGradient id="fpsFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06B6D4" stopOpacity="0.22" />
              <stop offset="100%" stopColor="#06B6D4" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {yTicks.map((tick) => {
            const y = padTop + plotH - (tick / maxFps) * plotH;
            return (
              <g key={tick}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="#1E293B"
                  strokeWidth="1"
                  strokeDasharray={tick === 0 ? undefined : '3 3'}
                />
                <text
                  x={padLeft - 8}
                  y={y + 3}
                  textAnchor="end"
                  className="fill-slate-500 text-[10px] font-mono"
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {areaPath && <path d={areaPath} fill="url(#fpsFill)" />}

          {linePath && (
            <path
              d={linePath}
              fill="none"
              stroke="#06B6D4"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          )}

          {activePoint && (
            <g>
              <line
                x1={activePoint.x}
                y1={padTop}
                x2={activePoint.x}
                y2={padTop + plotH}
                stroke="#334155"
                strokeWidth="1"
              />
              <circle
                cx={activePoint.x}
                cy={activePoint.y}
                r="3.5"
                fill="#06B6D4"
                stroke="#090C10"
                strokeWidth="1.5"
              />
            </g>
          )}
        </svg>
      )}
    </div>
  );
};
