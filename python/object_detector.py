#!/usr/bin/env python3
"""
OpticTrack Real-Time Object, Hand Movement & Gesture Detector (Python + OpenCV + MediaPipe)
Decouples heavy object detection into a background worker thread so MediaPipe 21-point
hand tracking, movement trails, gesture recognition, and OpenCV display run at full 30+ FPS
with minimal latency.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import math
import os
import sys
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PALETTE = [
    (6, 182, 212),    # Cyan
    (16, 185, 129),   # Emerald
    (245, 158, 11),   # Amber
    (59, 130, 246),   # Blue
    (236, 72, 153),   # Pink
    (139, 92, 246),   # Violet
]


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    r, g, b = PALETTE[class_id % len(PALETTE)]
    return (b, g, r)


def init_log_file(log_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# OpticTrack Detection Timestamp Log\n")
            f.write(
                "# Format: [ISO_TIMESTAMP] TYPE=<type> OBJECT=<label> CONF=<score> BBOX=[x,y,w,h] FPS=<fps> [MOTION=<motion>]\n"
            )


def append_detection_log(
    log_path: str,
    entry_type: str,
    timestamp: str,
    label: str,
    confidence: float,
    bbox: Tuple[int, int, int, int],
    fps: float,
    motion: Optional[str] = None,
) -> None:
    x, y, w, h = bbox
    safe_label = label.replace(" ", "_")
    motion_part = f" MOTION={motion.replace(' ', '_')}" if motion else ""
    line = (
        f"[{timestamp}] TYPE={entry_type} OBJECT={safe_label} CONF={confidence:.3f} "
        f"BBOX=[{x},{y},{w},{h}] FPS={fps:.1f}{motion_part}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def load_mobilenet_ssd():
    prototxt = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
    weights = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
    if not (os.path.exists(prototxt) and os.path.exists(weights)):
        raise FileNotFoundError(
            "MobileNet-SSD files missing. Run: python setup_models.py --model mobilenet-ssd"
        )
    net = cv2.dnn.readNetFromCaffe(prototxt, weights)
    return net, VOC_CLASSES


def load_yolov4_tiny():
    cfg = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
    weights = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
    names_path = os.path.join(MODELS_DIR, "coco.names")
    if not (os.path.exists(cfg) and os.path.exists(weights) and os.path.exists(names_path)):
        raise FileNotFoundError(
            "YOLOv4-tiny files missing. Run: python setup_models.py --model yolov4-tiny"
        )
    with open(names_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, classes


def detect_mobilenet(
    net,
    frame: np.ndarray,
    classes: List[str],
    conf_threshold: float,
) -> List[Tuple[int, str, float, Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()
    results = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        class_id = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        label = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
        results.append((class_id, label, confidence, (x1, y1, bw, bh)))
    return results


def detect_yolo(
    net,
    frame: np.ndarray,
    classes: List[str],
    conf_threshold: float,
    nms_threshold: float,
) -> List[Tuple[int, str, float, Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()
    ]
    outputs = net.forward(output_layers)

    boxes: List[List[int]] = []
    confidences: List[float] = []
    class_ids: List[int] = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence >= conf_threshold:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = max(0, int(cx - bw / 2))
                y = max(0, int(cy - bh / 2))
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            cid = class_ids[i]
            label = classes[cid] if cid < len(classes) else f"class_{cid}"
            x, y, bw, bh = boxes[i]
            results.append((cid, label, confidences[i], (x, y, bw, bh)))
    return results


class AsyncObjectWorker:
    """Runs OpenCV DNN object inference on a background thread without stalling hand tracking."""

    def __init__(self, net, classes, model_name: str, conf: float, nms: float):
        self.net = net
        self.classes = classes
        self.model_name = model_name
        self.conf = conf
        self.nms = nms
        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_detections: List[Tuple[int, str, float, Tuple[int, int, int, int]]] = []
        self._stop = False
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame.copy()
        self._event.set()

    def get_detections(self) -> List[Tuple[int, str, float, Tuple[int, int, int, int]]]:
        with self._lock:
            return list(self._latest_detections)

    def stop(self) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop:
            self._event.wait(timeout=0.1)
            self._event.clear()
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue

            if self.model_name == "yolov4-tiny":
                res = detect_yolo(self.net, frame, self.classes, self.conf, self.nms)
            else:
                res = detect_mobilenet(self.net, frame, self.classes, self.conf)

            with self._lock:
                self._latest_detections = res


FINGER_SPECS = [
    ("Thumb", (1, 2, 3, 4)),
    ("Index", (5, 6, 7, 8)),
    ("Middle", (9, 10, 11, 12)),
    ("Ring", (13, 14, 15, 16)),
    ("Pinky", (17, 18, 19, 20)),
]


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _len3(a) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _angle_deg(v1, v2) -> float:
    m1, m2 = _len3(v1), _len3(v2)
    if m1 < 1e-6 or m2 < 1e-6:
        return 0.0
    cos_v = max(-1.0, min(1.0, _dot3(v1, v2) / (m1 * m2)))
    return math.degrees(math.acos(cos_v))


def _dist3(a, b) -> float:
    return _len3(_sub3(a, b))


def analyze_fingers_360(landmarks, handedness_label: str) -> Tuple[List[dict], int, str]:
    """Compute 360-degree orientation-invariant extension % and bend angles for all 5 fingers."""
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
    wrist = pts[0]
    palm_size = max(1e-4, _dist3(pts[0], pts[9]))
    palm_width = max(1e-4, _dist3(pts[5], pts[17]))

    fingers_info: List[dict] = []
    for name, (j0, j1, j2, j3) in FINGER_SPECS:
        v1 = _sub3(pts[j1], pts[j0])
        v2 = _sub3(pts[j2], pts[j1])
        v3 = _sub3(pts[j3], pts[j2])
        b1 = _angle_deg(v1, v2)
        b2 = _angle_deg(v2, v3)

        if name == "Thumb":
            bend = (b1 + b2) * 0.5
            reach = max(_dist3(pts[j3], pts[17]), _dist3(pts[j3], pts[5]) * 1.1) / palm_size
            ang_score = max(0.0, min(1.0, 1.0 - (bend - 10.0) / 75.0))
            reach_score = max(0.0, min(1.0, (reach - 0.55) / 0.75))
            ext_pct = int(round((ang_score * 0.55 + reach_score * 0.45) * 100))
        else:
            bend = b1 * 0.65 + b2 * 0.35
            ang_score = max(0.0, min(1.0, 1.0 - (bend - 14.0) / 92.0))
            radial = _dist3(pts[j3], wrist) / max(1e-4, _dist3(pts[j1], wrist))
            rad_score = max(0.0, min(1.0, (radial - 0.95) / 0.8))
            ext_pct = int(round((ang_score * 0.6 + rad_score * 0.4) * 100))

        fingers_info.append(
            {"name": name, "joints": (j0, j1, j2, j3), "ext_pct": ext_pct, "bend_deg": int(round(bend))}
        )

    ext = [f["ext_pct"] >= 58 for f in fingers_info]
    finger_count = sum(ext)
    pinch_ratio = _dist3(pts[4], pts[8]) / palm_width

    if pinch_ratio < 0.36 and (ext[2] or ext[3]):
        gesture = "Pinch"
    elif all(ext):
        gesture = "Open_Palm"
    elif not any(ext[1:]):
        if ext[0]:
            dy = pts[4][1] - pts[2][1]
            dx = pts[4][0] - pts[2][0]
            if abs(dy) >= abs(dx):
                gesture = "Thumb_Up" if dy < 0 else "Thumb_Down"
            else:
                gesture = "Thumb_Out"
        else:
            gesture = "Closed_Fist"
    elif ext[1] and ext[2] and not ext[3] and not ext[4]:
        gesture = "Victory"
    elif ext[1] and not ext[2] and not ext[3] and not ext[4] and not ext[0]:
        gesture = "Pointing"
    elif ext[0] and ext[1] and not ext[2] and not ext[3] and ext[4]:
        gesture = "ILoveYou"
    elif not ext[0] and ext[1] and not ext[2] and not ext[3] and ext[4]:
        gesture = "Rock_Horns"
    elif ext[0] and not ext[1] and not ext[2] and not ext[3] and ext[4]:
        gesture = "Call_Me"
    elif finger_count == 3:
        gesture = "Three_Fingers"
    elif finger_count == 4:
        gesture = "Four_Fingers"
    else:
        gesture = "Articulated"

    return fingers_info, finger_count, gesture


def analyze_hand_movement(
    trail: Deque[Tuple[float, float, float]],
    now_t: float,
) -> Tuple[str, float, Optional[str]]:
    if len(trail) < 2:
        return "Stationary", 0.0, None

    x1, y1, t1 = trail[-1]
    x0, y0, t0 = trail[-2]
    for pt in reversed(list(trail)[:-1]):
        x0, y0, t0 = pt
        if t1 - t0 >= 0.06:
            break

    dt = max(0.016, t1 - t0)
    dx = x1 - x0
    dy = y1 - y0
    speed = math.hypot(dx, dy) / dt

    direction = "Stationary"
    if speed >= 85.0:
        if abs(dx) >= abs(dy):
            direction = "Moving_Right" if dx > 0 else "Moving_Left"
        else:
            direction = "Moving_Down" if dy > 0 else "Moving_Up"

    swipe: Optional[str] = None
    window_pts = [p for p in trail if now_t - p[2] <= 0.22]
    if len(window_pts) >= 3:
        wx0, wy0, wt0 = window_pts[0]
        wx1, wy1, wt1 = window_pts[-1]
        wdt = max(0.02, wt1 - wt0)
        wdx, wdy = wx1 - wx0, wy1 - wy0
        w_dist = math.hypot(wdx, wdy)
        w_speed = w_dist / wdt
        if w_dist >= 85.0 and w_speed >= 460.0:
            if abs(wdx) >= abs(wdy):
                swipe = "Swipe_Right" if wdx > 0 else "Swipe_Left"
            else:
                swipe = "Swipe_Down" if wdy > 0 else "Swipe_Up"

    return direction, speed, swipe


def draw_hud_and_detections(
    frame: np.ndarray,
    detections: List[Tuple[int, str, float, Tuple[int, int, int, int]]],
    hand_results: List[dict],
    fps: float,
    infer_ms: float,
    logging_enabled: bool,
    paused: bool,
) -> np.ndarray:
    annotated = frame.copy()

    for class_id, label, conf, (x, y, bw, bh) in detections:
        color = color_for_class(class_id)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        tag_text = f"{label}: {conf * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tag_y = max(th + 8, y)
        cv2.rectangle(annotated, (x, tag_y - th - 8), (x + tw + 10, tag_y), color, cv2.FILLED)
        cv2.putText(
            annotated,
            tag_text,
            (x + 5, tag_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 14, 20),
            2,
            cv2.LINE_AA,
        )

    for hand in hand_results:
        pts_px = hand["points_px"]
        for c_start, c_end in hand["connections"]:
            cv2.line(annotated, pts_px[c_start], pts_px[c_end], (129, 185, 16), 2, cv2.LINE_AA)

        # Highlight tips based on extension/movement without any blue tracking trail line
        tip_set = {4, 8, 12, 16, 20}
        for idx, (px, py) in enumerate(pts_px):
            if idx in tip_set:
                f_idx = [4, 8, 12, 16, 20].index(idx)
                f_data = hand["fingers"][f_idx]
                dot_col = (11, 158, 245) if f_data.get("moving") else (129, 185, 16)
                cv2.circle(annotated, (px, py), 5, dot_col, -1, cv2.LINE_AA)
            else:
                cv2.circle(annotated, (px, py), 3, (226, 232, 240), -1, cv2.LINE_AA)

        hx, hy, hw, hh = hand["bbox"]
        cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (129, 185, 16), 2)

        gesture_str = hand["swipe"] or hand["gesture"]
        label_str = f"{hand['handedness']} | {gesture_str} | {hand['roll_deg']}deg | {hand['facing']}"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        tag_y = max(th + 10, hy)
        cv2.rectangle(
            annotated,
            (hx, tag_y - th - 10),
            (hx + tw + 12, tag_y),
            (129, 185, 16),
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label_str,
            (hx + 6, tag_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (10, 14, 20),
            2,
            cv2.LINE_AA,
        )

    hud_w, hud_h = 370, 86
    overlay = annotated.copy()
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (12, 16, 23), cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
    cv2.rectangle(annotated, (12, 12), (12 + hud_w, 12 + hud_h), (45, 62, 80), 1)

    fps_bgr = (129, 185, 16) if fps >= 20 else (11, 158, 245)
    cv2.putText(
        annotated,
        f"FPS: {fps:5.1f}  |  HAND: {infer_ms:5.1f} ms",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        fps_bgr,
        2,
        cv2.LINE_AA,
    )

    log_state = "LOG: ON" if logging_enabled else "LOG: PAUSED"
    play_state = "PAUSED" if paused else "LIVE"
    cv2.putText(
        annotated,
        f"OBJ: {len(detections):2d}  |  HANDS: {len(hand_results):1d}  |  {log_state}  |  {play_state}",
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (210, 220, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        "Keys: [Q] Quit  [P] Pause  [L] Toggle Log  [S] Snapshot",
        (24, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 155, 175),
        1,
        cv2.LINE_AA,
    )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time OpenCV object, hand movement, and gesture detector with FPS monitor and local text logging."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--mode",
        choices=["all", "objects", "hands"],
        default="all",
        help="Detection pipeline mode: all (objects + hands), objects, or hands",
    )
    parser.add_argument(
        "--model",
        choices=["yolov4-tiny", "mobilenet-ssd"],
        default="yolov4-tiny",
        help="Pre-trained object detection architecture",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Minimum confidence threshold (0.1 - 0.95)",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.40,
        help="Non-Maximum Suppression threshold for YOLO",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.path.join(BASE_DIR, "..", "logs", "detections.txt"),
        help="Path to local text file for timestamped detection logging",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between log entries for the same object or gesture",
    )
    args = parser.parse_args()

    object_worker: Optional[AsyncObjectWorker] = None
    if args.mode in ("all", "objects"):
        try:
            if args.model == "yolov4-tiny":
                net, classes = load_yolov4_tiny()
            else:
                net, classes = load_mobilenet_ssd()
            object_worker = AsyncObjectWorker(
                net, classes, args.model, args.confidence, args.nms
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    hands_tracker = None
    hand_connections: List[Tuple[int, int]] = []
    if args.mode in ("all", "hands"):
        if mp is None:
            print(
                "Error: mediapipe package is required for hand tracking. Run: pip install -r requirements-desktop.txt",
                file=sys.stderr,
            )
            sys.exit(1)
        mp_hands = mp.solutions.hands
        # model_complexity=0 minimizes per-frame hand tracking latency on laptop CPUs
        hands_tracker = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=args.confidence,
            min_tracking_confidence=0.45,
        )
        hand_connections = list(mp_hands.HAND_CONNECTIONS)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    init_log_file(args.log_file)
    print(f"Started mode={args.mode} on camera {args.camera}")
    print(f"Logging timestamps to: {os.path.abspath(args.log_file)}")

    frame_times: Deque[float] = deque(maxlen=20)
    hand_trails: Dict[str, Deque[Tuple[float, float, float]]] = {}
    last_logged: Dict[str, float] = {}
    logging_enabled = True
    paused = False
    last_frame = None
    last_detections = []
    last_hands: List[dict] = []
    last_infer_ms = 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("Warning: Failed to read frame from camera.")
                    break

                if object_worker is not None:
                    object_worker.submit_frame(frame)
                    detections = object_worker.get_detections()
                else:
                    detections = []

                infer_start = time.perf_counter()
                hand_results: List[dict] = []
                if hands_tracker is not None:
                    fh, fw = frame.shape[:2]
                    # Downscale RGB input to max 640px wide for faster MediaPipe inference
                    if fw > 640:
                        scale_h = int(fh * (640.0 / fw))
                        small_rgb = cv2.cvtColor(
                            cv2.resize(frame, (640, scale_h)), cv2.COLOR_BGR2RGB
                        )
                    else:
                        small_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mp_out = hands_tracker.process(small_rgb)
                    now_t = time.perf_counter()

                    if mp_out.multi_hand_landmarks:
                        for idx, hand_lms in enumerate(mp_out.multi_hand_landmarks):
                            handedness_label = "Right"
                            handedness_score = 0.9
                            if mp_out.multi_handedness and idx < len(mp_out.multi_handedness):
                                cls = mp_out.multi_handedness[idx].classification[0]
                                handedness_label = cls.label
                                handedness_score = float(cls.score)

                            pts_px = [
                                (int(lm.x * fw), int(lm.y * fh)) for lm in hand_lms.landmark
                            ]
                            xs = [p[0] for p in pts_px]
                            ys = [p[1] for p in pts_px]
                            pad = 16
                            hx1 = max(0, min(xs) - pad)
                            hy1 = max(0, min(ys) - pad)
                            hx2 = min(fw, max(xs) + pad)
                            hy2 = min(fh, max(ys) + pad)
                            h_bbox = (hx1, hy1, max(1, hx2 - hx1), max(1, hy2 - hy1))

                            palm_indices = [0, 5, 9, 13, 17]
                            cx = sum(pts_px[i][0] for i in palm_indices) / len(palm_indices)
                            cy = sum(pts_px[i][1] for i in palm_indices) / len(palm_indices)

                            trail_key = f"{handedness_label}_{idx}"
                            if trail_key not in hand_trails:
                                hand_trails[trail_key] = deque(maxlen=24)
                            hand_trails[trail_key].append((cx, cy, now_t))
                            while (
                                hand_trails[trail_key]
                                and now_t - hand_trails[trail_key][0][2] > 0.45
                            ):
                                hand_trails[trail_key].popleft()

                            fingers_info, finger_count, gesture = analyze_fingers_360(
                                hand_lms.landmark, handedness_label
                            )
                            # 360-degree roll angle & 3D palm facing
                            dx_roll = hand_lms.landmark[9].x - hand_lms.landmark[0].x
                            dy_roll = hand_lms.landmark[9].y - hand_lms.landmark[0].y
                            roll_deg = int(round(math.degrees(math.atan2(dx_roll, -dy_roll)) % 360))
                            v_idx = _sub3(
                                (hand_lms.landmark[5].x, hand_lms.landmark[5].y, hand_lms.landmark[5].z),
                                (hand_lms.landmark[0].x, hand_lms.landmark[0].y, hand_lms.landmark[0].z),
                            )
                            v_pnk = _sub3(
                                (hand_lms.landmark[17].x, hand_lms.landmark[17].y, hand_lms.landmark[17].z),
                                (hand_lms.landmark[0].x, hand_lms.landmark[0].y, hand_lms.landmark[0].z),
                            )
                            nrm = _cross3(v_idx, v_pnk)
                            nz = nrm[2] / max(1e-6, _len3(nrm))
                            signed_nz = nz if handedness_label == "Right" else -nz
                            facing = (
                                "Back"
                                if signed_nz > 0.32
                                else "Front"
                                if signed_nz < -0.32
                                else "Side"
                            )

                            direction, speed, swipe = analyze_hand_movement(
                                hand_trails[trail_key], now_t
                            )

                            hand_results.append(
                                {
                                    "handedness": handedness_label,
                                    "score": handedness_score,
                                    "gesture": gesture,
                                    "finger_count": finger_count,
                                    "fingers": fingers_info,
                                    "roll_deg": roll_deg,
                                    "facing": facing,
                                    "direction": direction,
                                    "speed": speed,
                                    "swipe": swipe,
                                    "bbox": h_bbox,
                                    "points_px": pts_px,
                                    "connections": hand_connections,
                                }
                            )

                last_infer_ms = (time.perf_counter() - infer_start) * 1000.0
                last_frame = frame
                last_detections = detections
                last_hands = hand_results

                frame_times.append(time.perf_counter() - loop_start)
                avg_dt = sum(frame_times) / max(1, len(frame_times))
                fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

                if logging_enabled and (detections or hand_results):
                    now_epoch = time.time()
                    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                    for _, label, conf, bbox in detections:
                        key = f"obj:{label}"
                        if now_epoch - last_logged.get(key, 0.0) >= args.cooldown:
                            append_detection_log(
                                args.log_file, "object", iso_ts, label, conf, bbox, fps
                            )
                            last_logged[key] = now_epoch

                    for hand in hand_results:
                        g_label = f"{hand['handedness']}:{hand['gesture']}"
                        g_key = f"gesture:{g_label}"
                        motion_str = f"{hand['direction']}@{hand['speed']:.0f}px/s"
                        if now_epoch - last_logged.get(g_key, 0.0) >= args.cooldown:
                            append_detection_log(
                                args.log_file,
                                "hand_gesture",
                                iso_ts,
                                g_label,
                                hand["score"],
                                hand["bbox"],
                                fps,
                                motion=motion_str,
                            )
                            last_logged[g_key] = now_epoch

                        if hand["swipe"]:
                            s_label = f"{hand['handedness']}:{hand['swipe']}"
                            s_key = f"motion:{s_label}"
                            if now_epoch - last_logged.get(s_key, 0.0) >= max(0.45, args.cooldown):
                                append_detection_log(
                                    args.log_file,
                                    "hand_motion",
                                    iso_ts,
                                    s_label,
                                    hand["score"],
                                    hand["bbox"],
                                    fps,
                                    motion=motion_str,
                                )
                                last_logged[s_key] = now_epoch
            else:
                avg_dt = sum(frame_times) / max(1, len(frame_times)) if frame_times else 0.0
                fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

            if last_frame is not None:
                output = draw_hud_and_detections(
                    last_frame,
                    last_detections,
                    last_hands,
                    fps,
                    last_infer_ms,
                    logging_enabled,
                    paused,
                )
                cv2.imshow("OpticTrack - Object & Hand Gesture Detection", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("l"):
                logging_enabled = not logging_enabled
            elif key == ord("s") and last_frame is not None:
                snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snap_name, output)
                print(f"Saved snapshot: {snap_name}")
    finally:
        if object_worker is not None:
            object_worker.stop()
        if hands_tracker is not None:
            hands_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
