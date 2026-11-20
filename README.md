# OpticTrack Vision Studio

Real-time **object detection**, **21-point hand movement tracking**, and **hand gesture recognition** workbench featuring:

- **Live Browser Studio** (React + TypeScript + TensorFlow.js COCO-SSD + MediaPipe GestureRecognizer)
- **Python + OpenCV + MediaPipe Desktop Pipeline** (`python/object_detector.py`)
- **Performance & FPS Telemetry** with real-time sparkline monitoring
- **Timestamped Local Text Logging** (`logs/detections.txt`)

## Web Studio Setup

```bash
npm install
npm run dev
```

Open `http://localhost:3000` and click **Start Camera** (or upload a video file).

## Python OpenCV Desktop Setup

```bash
cd python
pip install -r requirements.txt
python setup_models.py --model yolov4-tiny
python object_detector.py --mode all --camera 0 --confidence 0.50
```
