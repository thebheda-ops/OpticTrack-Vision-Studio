# OpticTrack Python OpenCV + MediaPipe Detector

Real-time laptop webcam **object detection**, **21-point hand movement tracking**, and **hand gesture recognition** using **OpenCV DNN** (`cv2.dnn`) and **MediaPipe Hands**, with live bounding boxes, hand skeleton trails, FPS telemetry, and timestamped local `.txt` logging.

## Quick Start

```bash
cd python
pip install -r requirements.txt
python setup_models.py --model yolov4-tiny
python object_detector.py --mode all --camera 0 --confidence 0.50
```

## CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `--mode` | `all` | `all` (objects + hands), `objects`, or `hands` |
| `--camera` | `0` | Laptop camera device index |
| `--model` | `yolov4-tiny` | `yolov4-tiny` (80 COCO classes) or `mobilenet-ssd` (20 VOC classes) |
| `--confidence` | `0.50` | Minimum confidence threshold (`0.1` to `0.95`) |
| `--nms` | `0.40` | Non-Maximum Suppression threshold (YOLOv4-tiny) |
| `--log-file` | `../logs/detections.txt` | Local text file path for timestamped logs |
| `--cooldown` | `1.0` | Minimum seconds between log writes per object/gesture |

## Supported Hand Gestures & Movement Tracking

- **Static Gestures**: `Open_Palm`, `Closed_Fist`, `Pointing_Up`, `Thumb_Up`, `Thumb_Down`, `Victory`, `ILoveYou`, `Pinch`, `Three_Fingers`, `Four_Fingers`
- **Movement Tracking**: 28-frame palm trajectory trail, instantaneous speed (`px/s`), directional vector (`Moving_Left`, `Moving_Right`, `Moving_Up`, `Moving_Down`), and dynamic swipes (`Swipe_Left`, `Swipe_Right`, `Swipe_Up`, `Swipe_Down`).

## Window Controls

- `Q` or `ESC` — Quit detection stream
- `P` — Pause / resume frame capture
- `L` — Toggle detection file logging
- `S` — Save annotated snapshot image
