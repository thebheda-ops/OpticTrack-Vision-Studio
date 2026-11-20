import React, { useMemo, useState } from 'react';
import { Download, FileText, FolderOutput, Search, Trash2 } from 'lucide-react';
import { LogEntry, LogEntryType } from '../types/detection';

interface DetectionLogViewProps {
  entries: LogEntry[];
  rawText: string;
  filePath: string;
  byteSize: number;
  localFileLinkedName: string | null;
  onLinkLocalFile: () => void;
  onClearLogs: () => void;
  onDownloadLogs: () => void;
}

const TYPE_LABELS: Record<LogEntryType, string> = {
  object: 'Object',
  hand_gesture: 'Hand Gesture',
  hand_motion: 'Hand Motion',
};

export const DetectionLogView: React.FC<DetectionLogViewProps> = ({
  entries,
  rawText,
  filePath,
  byteSize,
  localFileLinkedName,
  onLinkLocalFile,
  onClearLogs,
  onDownloadLogs,
}) => {
  const [viewMode, setViewMode] = useState<'table' | 'raw'>('table');
  const [typeFilter, setTypeFilter] = useState<'all' | LogEntryType>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredEntries = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return entries.filter((entry) => {
      if (typeFilter !== 'all' && entry.entryType !== typeFilter) {
        return false;
      }
      if (!q) return true;
      return (
        entry.label.toLowerCase().includes(q) ||
        entry.timestamp.toLowerCase().includes(q) ||
        (entry.motion?.toLowerCase().includes(q) ?? false)
      );
    });
  }, [entries, searchQuery, typeFilter]);

  const classCounts = useMemo(() => {
    const map = new Map<string, number>();
    for (const entry of entries) {
      map.set(entry.label, (map.get(entry.label) ?? 0) + 1);
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [entries]);

  const supportsFileSystemPicker =
    typeof window !== 'undefined' && 'showSaveFilePicker' in window;

  return (
    <div className="space-y-4">
      {/* Header & File Actions */}
      <div className="bg-[#10151C] border border-slate-800 rounded-xl sm:rounded-lg p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h2 className="text-base sm:text-lg font-semibold text-slate-100 text-balance">
            Detection & Gesture Timestamp Log
          </h2>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[13px] sm:text-xs font-mono tabular-nums text-slate-400">
            <span>{filePath}</span>
            <span aria-hidden="true">·</span>
            <span>{entries.length} entries</span>
            <span aria-hidden="true">·</span>
            <span>{(byteSize / 1024).toFixed(2)} KB</span>
            {localFileLinkedName && (
              <>
                <span aria-hidden="true">·</span>
                <span className="text-emerald-400">
                  Mirroring to local file: {localFileLinkedName}
                </span>
              </>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          {supportsFileSystemPicker && (
            <button
              type="button"
              onClick={onLinkLocalFile}
              className="inline-flex items-center justify-center gap-2 min-h-[44px] sm:min-h-0 px-4 py-2.5 sm:px-3.5 sm:py-2 text-sm sm:text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 active:scale-[0.98] border border-slate-700 rounded-lg sm:rounded-md transition-all whitespace-nowrap shrink-0 cursor-pointer"
            >
              <FolderOutput className="w-4 h-4 sm:w-3.5 sm:h-3.5 text-cyan-400" />
              {localFileLinkedName ? 'Change Local .txt' : 'Link Local .txt'}
            </button>
          )}
          <button
            type="button"
            onClick={onDownloadLogs}
            className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 min-h-[44px] sm:min-h-0 px-4 py-2.5 sm:px-3.5 sm:py-2 text-sm sm:text-xs font-medium text-slate-950 bg-cyan-400 hover:bg-cyan-300 active:scale-[0.98] rounded-lg sm:rounded-md transition-all whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="w-4 h-4 sm:w-3.5 sm:h-3.5" />
            Download .txt
          </button>
          <button
            type="button"
            onClick={onClearLogs}
            className="inline-flex items-center justify-center gap-2 min-h-[44px] sm:min-h-0 px-4 py-2.5 sm:px-3.5 sm:py-2 text-sm sm:text-xs font-medium text-slate-300 hover:text-rose-300 bg-[#161D27] hover:bg-rose-950/40 active:scale-[0.98] border border-slate-700 hover:border-rose-800/70 rounded-lg sm:rounded-md transition-all whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Trash2 className="w-4 h-4 sm:w-3.5 sm:h-3.5" />
            Clear Log
          </button>
        </div>
      </div>

      {/* Filter & View Switcher Bar */}
      <div className="bg-[#10151C] border border-slate-800 rounded-xl sm:rounded-lg p-3.5 sm:p-4 flex flex-col xl:flex-row xl:items-center justify-between gap-3.5 sm:gap-4">
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 flex-1">
          <div className="relative flex-1 sm:max-w-sm">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter by object, gesture, or timestamp..."
              className="w-full min-h-[44px] sm:min-h-0 pl-9 pr-3 py-2.5 sm:py-2 text-sm sm:text-xs bg-[#090C10] border border-slate-800 rounded-lg sm:rounded-md text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div className="grid grid-cols-4 sm:flex items-center p-1 sm:p-0.5 bg-[#090C10] border border-slate-800 rounded-lg sm:rounded-md shrink-0 overflow-x-auto">
            {(
              [
                ['all', 'All'],
                ['object', 'Objects'],
                ['hand_gesture', 'Gestures'],
                ['hand_motion', 'Swipes'],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setTypeFilter(key)}
                className={`min-h-[40px] sm:min-h-0 px-2.5 py-2 sm:py-1.5 text-[13px] sm:text-xs font-medium rounded-md sm:rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  typeFilter === key
                    ? 'bg-slate-800 text-slate-100'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col sm:flex-row sm:flex-wrap sm:items-center gap-3 sm:gap-4">
          {classCounts.length > 0 && (
            <div className="text-[13px] sm:text-xs text-slate-400 flex flex-wrap items-center gap-2">
              <span className="text-slate-500">Frequent:</span>
              {classCounts.map(([label, count], i) => (
                <React.Fragment key={label}>
                  {i > 0 && <span aria-hidden="true">·</span>}
                  <button
                    type="button"
                    onClick={() =>
                      setSearchQuery((prev) => (prev === label ? '' : label))
                    }
                    className={`hover:text-cyan-400 transition-colors cursor-pointer py-1 sm:py-0 ${
                      searchQuery === label ? 'text-cyan-400 font-medium' : 'text-slate-300'
                    }`}
                  >
                    {label} ({count})
                  </button>
                </React.Fragment>
              ))}
            </div>
          )}

          <div className="grid grid-cols-2 sm:flex items-center p-1 sm:p-0.5 bg-[#090C10] border border-slate-800 rounded-lg sm:rounded-md shrink-0">
            <button
              type="button"
              onClick={() => setViewMode('table')}
              className={`min-h-[40px] sm:min-h-0 px-3 py-2 sm:py-1.5 text-[13px] sm:text-xs font-medium rounded-md sm:rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                viewMode === 'table'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Structured Table
            </button>
            <button
              type="button"
              onClick={() => setViewMode('raw')}
              className={`min-h-[40px] sm:min-h-0 px-3 py-2 sm:py-1.5 text-[13px] sm:text-xs font-medium rounded-md sm:rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                viewMode === 'raw'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Raw Text File
            </button>
          </div>
        </div>
      </div>

      {/* Data Content */}
      <div className="bg-[#10151C] border border-slate-800 rounded-xl sm:rounded-lg overflow-hidden">
        {viewMode === 'raw' ? (
          <pre className="p-4 text-[13px] sm:text-xs font-mono leading-relaxed text-slate-300 overflow-x-auto max-h-[540px] overflow-y-auto whitespace-pre">
            {rawText || '# No detections logged yet.'}
          </pre>
        ) : filteredEntries.length === 0 ? (
          <div className="py-14 px-6 text-center">
            <FileText className="w-8 h-8 text-slate-600 mx-auto mb-3" />
            <h3 className="text-base sm:text-sm font-semibold text-slate-200">
              {entries.length === 0
                ? 'No detection or gesture timestamps logged yet'
                : 'No entries match your filter'}
            </h3>
            <p className="text-sm sm:text-xs text-slate-400 mt-1.5 max-w-md mx-auto leading-relaxed">
              {entries.length === 0
                ? 'Start the camera or video stream in Live Monitor to automatically append detected objects, hand gestures, and motion timestamps to logs/detections.txt.'
                : 'Clear the search or type filter to inspect all recorded timestamps.'}
            </p>
          </div>
        ) : (
          <>
            {/* Mobile Compact List View */}
            <div className="md:hidden divide-y divide-slate-800/80 max-h-[540px] overflow-y-auto">
              {filteredEntries.map((entry) => (
                <div key={entry.id} className="p-4 space-y-1.5">
                  <div className="flex items-center justify-between gap-3">
                    <span
                      className={`text-sm font-medium truncate ${
                        entry.entryType === 'hand_motion'
                          ? 'text-amber-300'
                          : entry.entryType === 'hand_gesture'
                            ? 'text-emerald-300'
                            : 'text-cyan-300'
                      }`}
                    >
                      {entry.label}
                    </span>
                    <span className="text-xs font-mono tabular-nums text-emerald-400 shrink-0">
                      {entry.fps.toFixed(1)} FPS
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs font-mono tabular-nums text-slate-400">
                    <span className="font-sans text-slate-300">
                      {TYPE_LABELS[entry.entryType]}
                    </span>
                    <span aria-hidden="true">·</span>
                    <span>{(entry.confidence * 100).toFixed(1)}%</span>
                    <span aria-hidden="true">·</span>
                    <span>{entry.timestamp.slice(11, 23)} UTC</span>
                    {entry.motion && (
                      <>
                        <span aria-hidden="true">·</span>
                        <span className="text-amber-300">{entry.motion}</span>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Desktop Full Structured Table */}
            <div className="hidden md:block overflow-x-auto max-h-[540px] overflow-y-auto">
              <table className="w-full text-left border-collapse">
                <thead className="bg-[#0C1016] border-b border-slate-800 sticky top-0 z-10">
                  <tr className="text-xs font-medium text-slate-400">
                    <th className="py-2.5 px-4">Timestamp (UTC)</th>
                    <th className="py-2.5 px-4">Category</th>
                    <th className="py-2.5 px-4">Detected Item / Gesture</th>
                    <th className="py-2.5 px-4">Motion</th>
                    <th className="py-2.5 px-4 text-right">Confidence</th>
                    <th className="py-2.5 px-4">Bounding Box [x, y, w, h]</th>
                    <th className="py-2.5 px-4 text-right">Stream FPS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70 text-xs font-mono tabular-nums">
                  {filteredEntries.map((entry) => (
                    <tr
                      key={entry.id}
                      className="hover:bg-slate-800/30 transition-colors"
                    >
                      <td className="py-2.5 px-4 text-slate-300 whitespace-nowrap">
                        {entry.timestamp}
                      </td>
                      <td className="py-2.5 px-4 font-sans text-slate-400 whitespace-nowrap">
                        {TYPE_LABELS[entry.entryType]}
                      </td>
                      <td
                        className={`py-2.5 px-4 font-sans font-medium ${
                          entry.entryType === 'hand_motion'
                            ? 'text-amber-300'
                            : entry.entryType === 'hand_gesture'
                              ? 'text-emerald-300'
                              : 'text-cyan-300'
                        }`}
                      >
                        {entry.label}
                      </td>
                      <td className="py-2.5 px-4 text-slate-400 whitespace-nowrap">
                        {entry.motion ?? '—'}
                      </td>
                      <td className="py-2.5 px-4 text-right text-slate-200">
                        {(entry.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-4 text-slate-400 whitespace-nowrap">
                        [{entry.bbox.join(', ')}]
                      </td>
                      <td className="py-2.5 px-4 text-right text-emerald-400">
                        {entry.fps.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
