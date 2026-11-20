import React, { useEffect, useState } from 'react';
import { Check, Copy, Download, Terminal } from 'lucide-react';
import { DetectorSettings } from '../types/detection';

interface PythonOpenCvViewProps {
  settings: DetectorSettings;
}

const PYTHON_FILES = [
  { key: 'object_detector.py', label: 'object_detector.py' },
  { key: 'setup_models.py', label: 'setup_models.py' },
  { key: 'requirements-desktop.txt', label: 'requirements-desktop.txt' },
  { key: 'README.md', label: 'README.md' },
] as const;

type PythonFileKey = (typeof PYTHON_FILES)[number]['key'];

export const PythonOpenCvView: React.FC<PythonOpenCvViewProps> = ({ settings }) => {
  const [activeFile, setActiveFile] = useState<PythonFileKey>('object_detector.py');
  const [fileContents, setFileContents] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    if (fileContents[activeFile]) return;
    let cancelled = false;
    setLoading(true);
    fetch(`/api/python/${activeFile}`)
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled && typeof data.content === 'string') {
          setFileContents((prev) => ({ ...prev, [activeFile]: data.content }));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFileContents((prev) => ({
            ...prev,
            [activeFile]: '# Unable to load file content from server.',
          }));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeFile, fileContents]);

  const cooldownSeconds = (settings.logCooldownMs / 1000).toFixed(1);
  const cliCommand = `python object_detector.py --mode ${settings.detectionMode} --model ${settings.pythonModel} --confidence ${settings.confidenceThreshold.toFixed(2)} --cooldown ${cooldownSeconds} --log-file detections.txt`;

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId((curr) => (curr === id ? null : curr)), 1600);
    });
  };

  const handleDownload = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const activeText = fileContents[activeFile] ?? '';

  return (
    <div className="space-y-4">
      {/* Synced CLI Launcher Card */}
      <div className="bg-[#10151C] border border-slate-800 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-100">
              Python OpenCV Desktop Pipeline
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Run the native Python + <code className="font-mono text-slate-300">cv2.dnn</code> detector locally with your laptop webcam, synchronized with your current confidence and cooldown parameters.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400 shrink-0">
            <span>Architecture:</span>
            <span className="font-mono text-cyan-400">{settings.pythonModel}</span>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="bg-[#090C10] border border-slate-800 rounded-md p-3">
            <div className="text-xs font-medium text-slate-300 mb-1.5">
              1. Install OpenCV Dependencies
            </div>
            <code className="block text-xs font-mono text-slate-400">
              pip install -r requirements-desktop.txt
            </code>
          </div>
          <div className="bg-[#090C10] border border-slate-800 rounded-md p-3">
            <div className="text-xs font-medium text-slate-300 mb-1.5">
              2. Download Pre-Trained Weights
            </div>
            <code className="block text-xs font-mono text-slate-400">
              python setup_models.py --model {settings.pythonModel}
            </code>
          </div>
          <div className="bg-[#090C10] border border-slate-800 rounded-md p-3 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="text-xs font-medium text-slate-300 mb-1.5 flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                <span>3. Launch Live Detector</span>
              </div>
              <code className="block text-xs font-mono text-cyan-300 truncate">
                {cliCommand}
              </code>
            </div>
            <button
              type="button"
              onClick={() => handleCopy('cli', cliCommand)}
              className="px-2.5 py-1.5 text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 border border-slate-700 rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {copiedId === 'cli' ? 'Copied' : 'Copy'}
            </button>
          </div>
        </div>
      </div>

      {/* Source File Viewer */}
      <div className="bg-[#10151C] border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-1 p-0.5 bg-[#090C10] border border-slate-800 rounded-md">
            {PYTHON_FILES.map((file) => (
              <button
                key={file.key}
                type="button"
                onClick={() => setActiveFile(file.key)}
                className={`px-3 py-1.5 text-xs font-mono rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  activeFile === file.key
                    ? 'bg-slate-800 text-cyan-300'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {file.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => handleCopy(activeFile, activeText)}
              disabled={!activeText}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-[#161D27] hover:bg-slate-800 border border-slate-700 rounded-md transition-colors whitespace-nowrap shrink-0 disabled:opacity-40 cursor-pointer"
            >
              {copiedId === activeFile ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy Source
                </>
              )}
            </button>
            <button
              type="button"
              onClick={() => handleDownload(activeFile, activeText)}
              disabled={!activeText}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-md transition-colors whitespace-nowrap shrink-0 disabled:opacity-40 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download {activeFile}
            </button>
          </div>
        </div>

        <pre className="p-4 text-xs font-mono leading-relaxed text-slate-300 overflow-x-auto max-h-[540px] overflow-y-auto">
          {loading && !activeText ? 'Loading source file...' : activeText}
        </pre>
      </div>
    </div>
  );
};
