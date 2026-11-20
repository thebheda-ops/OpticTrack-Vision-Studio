# OpticTrack Vision Studio

Real-time **object detection**, **21-point hand movement tracking**, and **hand gesture recognition** workbench featuring:

- **Live Browser Studio** (`app` service — React + TypeScript + TensorFlow.js COCO-SSD + MediaPipe GestureRecognizer)
- **Express API Service** (`api` service — `backend/index.ts`, bound to the internal `python` service via `PYTHON_SERVICE_URL`)
- **Internal Python Service & Desktop Pipeline** (`python` service — `python/index.py` on Vercel, plus `python/object_detector.py` for local laptop webcam OpenCV + MediaPipe execution)

## Web Studio & Multi-Service Local Dev

```bash
npm install
# Run single-process local dev (AI Studio / Vite + Express)
npm run dev

# Or run all Vercel services together with bindings injected
vercel dev
```

Open `http://localhost:3000` and click **Start Camera** (or upload a video file).

## Python OpenCV Desktop Setup (Laptop Webcam)

```bash
cd python
pip install -r requirements-desktop.txt
python setup_models.py --model yolov4-tiny
python object_detector.py --mode all --camera 0 --confidence 0.50
```
