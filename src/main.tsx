import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// MediaPipe/TFLite WASM routes C++ GLOG INFO lines via Emscripten stderr (console.error).
// Redirect those informational delegate messages to console.info while preserving real errors.
const originalConsoleError = console.error.bind(console);
console.error = (...args: unknown[]) => {
  const first = args[0];
  const msg = typeof first === 'string' ? first.trimStart() : '';
  if (
    msg.startsWith('INFO:') ||
    /^I\d{4}\s/.test(msg)
  ) {
    console.info(...args);
    return;
  }
  originalConsoleError(...args);
};

createRoot(document.getElementById('root')!).render(<App />);

