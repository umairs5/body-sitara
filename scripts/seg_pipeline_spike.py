"""
Spike: does overlapping yolo_seg(frame N+1) with pose/draw/write(frame N)
via a one-frame-ahead pipeline beat running everything strictly serially?

Context: yolo_seg now also produces the boxes RTMPose-T needs (see
blur_yoloseg.py's get_mask_and_boxes), so within a single frame, pose
genuinely depends on yolo_seg finishing first -- they can't be parallelized
against each other for the SAME frame. But there's no such dependency
ACROSS frames: yolo_seg for frame N+1 only needs frame N+1's pixels, not
anything pose/draw/write produce for frame N. So this pipelines across
frames instead of within one: submit yolo_seg(N+1) to a background thread
the moment frame N+1 is read, then do pose/draw/write for frame N while
that runs, then join.

This does NOT reduce total CPU work -- it overlaps the two biggest
per-frame costs so wall-clock time approaches max(seg_time, rest_time)
per frame instead of seg_time + rest_time, IF yolo_seg's C++ forward pass
actually releases the GIL enough for real overlap (unverified assumption,
this script's whole point is to measure it, not assume it).

Throwaway/exploratory -- does not touch pipeline.py. Deliberately does
NOT replicate pipeline.py's full feature set (no encryption, no export,
no skip-n/OF, no face canon) -- isolates just the seg+pose timing
question so the comparison is clean and fast to iterate on.

Usage:
    python scripts/seg_pipeline_spike.py <video> [--max-frames N] [--mode serial|pipelined]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from body_sitara.blur_yoloseg import YOLOSegBlur
from body_sitara.detector_patch import apply_detector_patch
from rtmlib import Body

INFER_SIZE = 320


def run_serial(video_path, max_frames, yoloseg, body, kp_scale_x, kp_scale_y):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    t_start = time.perf_counter()
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        mask, boxes = yoloseg.get_mask_and_boxes(frame)

        infer_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
        if boxes is not None and len(boxes) > 0:
            b = boxes.copy().astype(float)
            b[:, 0] /= kp_scale_x; b[:, 2] /= kp_scale_x
            b[:, 1] /= kp_scale_y; b[:, 3] /= kp_scale_y
        else:
            b = np.empty((0, 4), dtype=float)
        keypoints, scores = body.pose_model(infer_frame, bboxes=b)

        frame_idx += 1
    cap.release()
    return time.perf_counter() - t_start, frame_idx


def run_pipelined(video_path, max_frames, yoloseg, body, kp_scale_x, kp_scale_y):
    """
    One-frame-ahead: while pose(N) runs, yolo_seg(N+1) is already running
    in a background thread. Frame 0's seg has no predecessor to overlap
    with, so it's primed serially before the loop starts.
    """
    cap = cv2.VideoCapture(video_path)
    pool = ThreadPoolExecutor(max_workers=1)

    def seg_job(frame):
        return yoloseg.get_mask_and_boxes(frame)

    ok, current_frame = cap.read()
    if not ok:
        cap.release()
        return 0.0, 0

    t_start = time.perf_counter()
    pending = pool.submit(seg_job, current_frame)  # seg for current_frame, in flight
    frame_idx = 0

    while frame_idx < max_frames:
        mask, boxes = pending.result()  # blocks only if pose(N-1) was faster than seg(N)

        # Kick off next frame's seg immediately, before doing this frame's pose --
        # this is the actual overlap: pose(N) runs while seg(N+1) is in flight.
        ok, next_frame = cap.read()
        if ok:
            pending = pool.submit(seg_job, next_frame)

        infer_frame = cv2.resize(current_frame, (INFER_SIZE, INFER_SIZE))
        if boxes is not None and len(boxes) > 0:
            b = boxes.copy().astype(float)
            b[:, 0] /= kp_scale_x; b[:, 2] /= kp_scale_x
            b[:, 1] /= kp_scale_y; b[:, 3] /= kp_scale_y
        else:
            b = np.empty((0, 4), dtype=float)
        keypoints, scores = body.pose_model(infer_frame, bboxes=b)

        frame_idx += 1
        if not ok:
            break
        current_frame = next_frame

    pool.shutdown(wait=True)
    cap.release()
    return time.perf_counter() - t_start, frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--mode", choices=["serial", "pipelined", "both"], default="both")
    args = ap.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    print("Loading RTMPose-T (pose only -- boxes come from yolo_seg)...")
    apply_detector_patch()
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    print("Loading yolo11n-seg-int8...")
    yoloseg = YOLOSegBlur(model_name=os.path.join(models_dir, "yolo11n-seg-int8.onnx"),
                           infer_size=INFER_SIZE, conf=0.4)

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    kp_scale_x, kp_scale_y = w / INFER_SIZE, h / INFER_SIZE

    if args.mode in ("serial", "both"):
        t, n = run_serial(args.video, args.max_frames, yoloseg, body, kp_scale_x, kp_scale_y)
        print(f"\nSERIAL:     {n} frames in {t:.1f}s  ->  {n/t:.2f} FPS")

    if args.mode in ("pipelined", "both"):
        t, n = run_pipelined(args.video, args.max_frames, yoloseg, body, kp_scale_x, kp_scale_y)
        print(f"PIPELINED:  {n} frames in {t:.1f}s  ->  {n/t:.2f} FPS")


if __name__ == "__main__":
    main()
