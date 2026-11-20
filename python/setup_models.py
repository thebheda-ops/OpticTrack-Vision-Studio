#!/usr/bin/env python3
"""
Model Weight Downloader for OpticTrack OpenCV Object Detector.
Downloads pre-trained MobileNet-SSD (Caffe) or YOLOv4-tiny (Darknet COCO)
weights and configuration files into ./models/.
"""

import argparse
import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_FILES = {
    "mobilenet-ssd": [
        (
            "MobileNetSSD_deploy.prototxt",
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
        ),
        (
            "MobileNetSSD_deploy.caffemodel",
            "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
        ),
    ],
    "yolov4-tiny": [
        (
            "yolov4-tiny.cfg",
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        ),
        (
            "yolov4-tiny.weights",
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        ),
        (
            "coco.names",
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
        ),
    ],
}


def download_file(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[skip] {os.path.basename(dest_path)} already exists")
        return

    print(f"[download] {os.path.basename(dest_path)} <- {url}")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\r  Progress: {pct:5.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print(f"\n  Saved to {dest_path} ({os.path.getsize(dest_path):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pre-trained OpenCV DNN object detection models."
    )
    parser.add_argument(
        "--model",
        choices=["mobilenet-ssd", "yolov4-tiny", "all"],
        default="yolov4-tiny",
        help="Model weights to download (default: yolov4-tiny)",
    )
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    targets = (
        list(MODEL_FILES.keys()) if args.model == "all" else [args.model]
    )

    for model_name in targets:
        print(f"\nPreparing {model_name} assets in {MODELS_DIR}...")
        for filename, url in MODEL_FILES[model_name]:
            download_file(url, os.path.join(MODELS_DIR, filename))

    print("\nAll requested model files are ready.")


if __name__ == "__main__":
    main()
