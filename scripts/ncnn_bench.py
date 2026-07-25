"""
Clean, correctness-checked benchmark: ONNX INT8 vs NCNN FP32 vs NCNN FP16
segmentation speed, on a real, verified-present clip.

Earlier ad-hoc Pi tests skipped checking cv2.VideoCapture.isOpened() and
cap.read()'s ok flag -- a missing clip path would silently produce empty/
garbage frames rather than erroring, making any timing number from that
run untrustworthy. This script hard-fails loudly instead of silently
measuring nothing.

Usage:
    python scripts/ncnn_bench.py <video> [--max-frames N]
      [--onnx-model path] [--ncnn-fp32-dir path] [--ncnn-fp16-dir path]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def get_mask_ncnn(model, frame, infer_size=320, conf=0.4):
    h, w = frame.shape[:2]
    results = model(frame, imgsz=infer_size, conf=conf, classes=[0], verbose=False)
    combined = np.zeros((h, w), dtype=bool)
    for r in results:
        if r.masks is None:
            continue
        for mt in r.masks.data:
            m = mt.cpu().numpy()
            m_up = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            combined |= m_up
    return combined if combined.any() else None


def load_frames(video_path, max_frames):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open: {video_path}")
    frames = []
    for i in range(max_frames):
        ok, f = cap.read()
        if not ok:
            print(f"  WARNING: cap.read() failed at frame {i} -- stopping early ({len(frames)} frames loaded)")
            break
        frames.append(f)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Zero frames read from {video_path} -- file exists but is unreadable/empty")
    print(f"  Loaded {len(frames)} real frames from {video_path}, shape={frames[0].shape}")
    return frames


def bench(name, fn, frames, warmup=5):
    for f in frames[:warmup]:
        fn(f)
    times = []
    mask_count = 0
    for f in frames:
        t0 = time.perf_counter()
        mask = fn(f)
        times.append((time.perf_counter() - t0) * 1000)
        if mask is not None:
            mask_count += 1
    avg = np.mean(times)
    print(f"{name:20s}: {avg:7.2f} ms/frame  ->  {1000/avg:6.2f} FPS alone  "
          f"(mask found {mask_count}/{len(frames)} frames)")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--onnx-model", default="models/yolo11n-seg-int8.onnx")
    ap.add_argument("--ncnn-fp32-dir", default="models/ncnn_fp32/yolo11n-seg_ncnn_model")
    ap.add_argument("--ncnn-fp16-dir", default="models/ncnn_fp16/yolo11n-seg_ncnn_model")
    args = ap.parse_args()

    print(f"Loading frames from {args.video} ...")
    frames = load_frames(args.video, args.max_frames)

    results = {}

    if os.path.exists(args.onnx_model):
        print(f"\nLoading ONNX ({args.onnx_model}) ...")
        from body_sitara.blur_yoloseg import YOLOSegBlur
        onnx_seg = YOLOSegBlur(model_name=args.onnx_model, infer_size=320, conf=0.4)
        results["ONNX INT8"] = bench("ONNX INT8", lambda f: onnx_seg.get_mask_and_boxes(f)[0], frames)
    else:
        print(f"\nSkipping ONNX -- {args.onnx_model} not found")

    from ultralytics import YOLO

    if os.path.exists(args.ncnn_fp32_dir):
        print(f"\nLoading NCNN FP32 ({args.ncnn_fp32_dir}) ...")
        m32 = YOLO(args.ncnn_fp32_dir)
        results["NCNN FP32"] = bench("NCNN FP32", lambda f: get_mask_ncnn(m32, f), frames)
    else:
        print(f"\nSkipping NCNN FP32 -- {args.ncnn_fp32_dir} not found")

    if os.path.exists(args.ncnn_fp16_dir):
        print(f"\nLoading NCNN FP16 ({args.ncnn_fp16_dir}) ...")
        m16 = YOLO(args.ncnn_fp16_dir)
        results["NCNN FP16"] = bench("NCNN FP16", lambda f: get_mask_ncnn(m16, f), frames)
    else:
        print(f"\nSkipping NCNN FP16 -- {args.ncnn_fp16_dir} not found")

    print("\n=== Summary ===")
    baseline = results.get("ONNX INT8")
    for name, ms in results.items():
        speedup = f"  ({baseline/ms:.2f}x vs ONNX INT8)" if baseline and name != "ONNX INT8" else ""
        print(f"{name:20s}: {ms:7.2f} ms/frame{speedup}")


if __name__ == "__main__":
    main()
