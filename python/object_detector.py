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


def classify_hand_gesture(landmarks, handedness_label: str) -> Tuple[str, int]:
    pts = [(lm.x, lm.y) for lm in landmarks]

    def dist(a: int, b: int) -> float:
        return math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])

    palm_width = max(1e-4, dist(5, 17))
    pinch_ratio = dist(4, 8) / palm_width

    if handedness_label == "Right":
        thumb_open = pts[4][0] < pts[3][0] - 0.02
    else:
        thumb_open = pts[4][0] > pts[3][0] + 0.02

    index_open = pts[8][1] < pts[6][1] - 0.02
    middle_open = pts[12][1] < pts[10][1] - 0.02
    ring_open = pts[16][1] < pts[14][1] - 0.02
    pinky_open = pts[20][1] < pts[18][1] - 0.02

    finger_count = sum([thumb_open, index_open, middle_open, ring_open, pinky_open])

    if pinch_ratio < 0.34 and (middle_open or ring_open):
        return "Pinch", finger_count
    if index_open and middle_open and ring_open and pinky_open and thumb_open:
        return "Open_Palm", finger_count
    if not index_open and not middle_open and not ring_open and not pinky_open:
        if dist(4, 2) > palm_width * 0.45:
            if pts[4][1] < pts[2][1] - 0.04:
                return "Thumb_Up", finger_count
            if pts[4][1] > pts[2][1] + 0.04:
                return "Thumb_Down", finger_count
        return "Closed_Fist", finger_count
    if index_open and middle_open and not ring_open and not pinky_open:
        return "Victory", finger_count
    if index_open and not middle_open and not ring_open and not pinky_open:
        return "Pointing_Up", finger_count
    if thumb_open and index_open and not middle_open and not ring_open and pinky_open:
        return "ILoveYou", finger_count
    if finger_count == 3:
        return "Three_Fingers", finger_count
    if finger_count == 4:
        return "Four_Fingers", finger_count
    return "Tracked_Hand", finger_count


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
        trail: List[Tuple[float, float]] = hand["trail"]
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                p2 = (int(trail[i][0]), int(trail[i][1]))
                thickness = max(1, int(1 + (i / len(trail)) * 4))
                cv2.line(annotated, p1, p2, (212, 182, 6), thickness, cv2.LINE_AA)

        pts_px = hand["points_px"]
        for c_start, c_end in hand["connections"]:
            cv2.line(annotated, pts_px[c_start], pts_px[c_end], (129, 185, 16), 2, cv2.LINE_AA)
        for px, py in pts_px:
            cv2.circle(annotated, (px, py), 4, (212, 182, 6), -1, cv2.LINE_AA)

        hx, hy, hw, hh = hand["bbox"]
        cv2.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (11, 158, 245), 2)

        gesture_str = hand["swipe"] or hand["gesture"]
        label_str = f"{hand['handedness']} | {gesture_str} | {hand['direction']} ({hand['speed']:.0f} px/s)"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        tag_y = max(th + 10, hy)
        cv2.rectangle(
            annotated,
            (hx, tag_y - th - 10),
            (hx + tw + 12, tag_y),
            (11, 158, 245),
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

                            gesture, finger_count = classify_hand_gesture(
                                hand_lms.landmark, handedness_label
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
                                    "direction": direction,
                                    "speed": speed,
                                    "swipe": swipe,
                                    "bbox": h_bbox,
                                    "points_px": pts_px,
                                    "connections": hand_connections,
                                    "trail": [(p[0], p[1]) for p in hand_trails[trail_key]],
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
