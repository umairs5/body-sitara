"""
Benchmark YOLO11n-seg inference cost across precisions/runtimes:
  - PyTorch .pt (FP32, current default in blur_yoloseg.py)
  - ONNX FP32 (exported via ultralytics)
  - ONNX INT8 (dynamically quantized via onnxruntime)

Runs get_mask() N times on a fixed real frame from the given clip and
reports mean/median latency. Exploratory only.

Usage: python scripts/bench_yolo_precision.py <video> [--n 50]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from body_sitara.blur_yoloseg import YOLOSegBlur


def bench(name, model_path, frame, n):
    seg = YOLOSegBlur(model_name=model_path, infer_size=320, conf=0.4)
    # warmup
    for _ in range(3):
        seg.get_mask(frame)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        seg.get_mask(frame)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"{name:<20} mean={times.mean():7.1f}ms  median={np.median(times):7.1f}ms  "
          f"min={times.min():7.1f}ms  max={times.max():7.1f}ms")
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--frame-idx", type=int, default=60)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read frame")
        return

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    print(f"Benchmarking on frame {args.frame_idx} of {args.video}, n={args.n} iterations each\n")

    bench("PyTorch FP32 (.pt)", os.path.join(models_dir, "yolo11n-seg.pt"), frame, args.n)
    bench("ONNX FP32", os.path.join(models_dir, "yolo11n-seg.onnx"), frame, args.n)
    bench("ONNX INT8 (dynamic)", os.path.join(models_dir, "yolo11n-seg-int8.onnx"), frame, args.n)


if __name__ == "__main__":
    main()
