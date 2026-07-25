"""
Spike: does running yolo_seg in a separate PROCESS (not thread) beat the
cross-frame thread-pipelining approach (seg_pipeline_spike.py)?

Threading relies on yolo_seg's C++ forward pass releasing the GIL for real
overlap -- seg_pipeline_spike.py measured a genuine +19% FPS win from that
on the dev machine, so the GIL isn't the bottleneck there. This script
checks whether a full separate process (true multi-core, no GIL question
at all) does meaningfully better than threading did, which would justify
the extra complexity (model reload per worker, frame serialization across
the process boundary) -- or whether it's not worth it because threading
already captured most of the available overlap.

Uses a persistent worker process (not a fresh process per frame -- that
would reload the ONNX model every frame, dominating the measurement) with
multiprocessing.Pipe for frame-in/result-out, pipelined the same
one-frame-ahead way as seg_pipeline_spike.py's threaded version, so the
two are a fair comparison of "thread pool" vs "process pool" for the same
overlap strategy -- not "no pipelining" vs "process pipelining".

Throwaway/exploratory -- does not touch pipeline.py.

Usage:
    python scripts/seg_process_spike.py <video> [--max-frames N]
"""
import argparse
import os
import sys
import time
import multiprocessing as mp

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

INFER_SIZE = 320


def _seg_worker(model_path, infer_size, conf, in_conn, out_conn):
    """Runs in a separate process: loads its own model once, then services
    frame requests until it receives a sentinel (None) to stop."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from body_sitara.blur_yoloseg import YOLOSegBlur
    yoloseg = YOLOSegBlur(model_name=model_path, infer_size=infer_size, conf=conf)
    while True:
        frame = in_conn.recv()
        if frame is None:
            break
        mask, boxes = yoloseg.get_mask_and_boxes(frame)
        out_conn.send(boxes)  # mask isn't needed by pose -- skip the (larger) IPC payload


def run_process_pipelined(video_path, max_frames, body, kp_scale_x, kp_scale_y, model_path):
    to_worker_parent, to_worker_child = mp.Pipe()
    from_worker_parent, from_worker_child = mp.Pipe()
    worker = mp.Process(
        target=_seg_worker,
        args=(model_path, INFER_SIZE, 0.4, to_worker_child, from_worker_child),
    )
    worker.start()

    cap = cv2.VideoCapture(video_path)
    ok, current_frame = cap.read()
    if not ok:
        cap.release()
        worker.terminate()
        return 0.0, 0

    t_start = time.perf_counter()
    to_worker_parent.send(current_frame)
    frame_idx = 0

    while frame_idx < max_frames:
        boxes = from_worker_parent.recv()

        ok, next_frame = cap.read()
        if ok:
            to_worker_parent.send(next_frame)

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

    to_worker_parent.send(None)
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()
    cap.release()
    return time.perf_counter() - t_start, frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    from body_sitara.detector_patch import apply_detector_patch
    from rtmlib import Body

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

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, "yolo11n-seg-int8.onnx")

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    kp_scale_x, kp_scale_y = w / INFER_SIZE, h / INFER_SIZE

    t, n = run_process_pipelined(args.video, args.max_frames, body, kp_scale_x, kp_scale_y, model_path)
    print(f"\nPROCESS-PIPELINED:  {n} frames in {t:.1f}s  ->  {n/t:.2f} FPS")


if __name__ == "__main__":
    main()
