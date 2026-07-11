"""
Side-by-side comparison of segmentation backends for the Tier 1 anonymizer:
  - selfie_seg1 (MediaPipe SelfieSegmentation, landscape model — active pipeline)
  - YOLOv8n-seg (ultralytics, already in the repo, unused)
  - YOLO11n-seg (ultralytics, newer nano-seg variant)

Runs all three on the same clip, and writes a 2x2 grid video
(ORIGINAL | SELFIE_SEG / YOLOv8n-seg | YOLO11n-seg) plus per-backend timing
and mask-coverage stats. Exploratory only — does not touch pipeline.py or
the real anonymizer selection.

With --skip-n > 1, replicates pipeline.py's actual skip-frame behavior:
RTMPose keypoints are tracked every frame (fresh detection on full frames,
LK optical flow on skip frames); each backend's full-frame segmentation
mask is warped forward on skip frames via an affine transform estimated
from keypoint motion (cv2.estimateAffinePartial2D + cv2.warpAffine), not
simply held static. This matches pipeline.py's last_seg_mask propagation
exactly, so the comparison is fair to how skip-n actually runs in production.

Usage:
    python scripts/compare_segmentation.py <video> [--max-frames N] [--out out.mp4] [--skip-n N]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from body_sitara.blur_seg import SelfieSegBlur
from body_sitara.blur_yoloseg import YOLOSegBlur
from body_sitara.pose import LK_PARAMS
from body_sitara.detector_patch import apply_detector_patch
from rtmlib import Body

HULL_COLOR = (127, 127, 127)
PANEL_LABEL_COLOR = (0, 255, 255)
INFER_SIZE = 320


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, PANEL_LABEL_COLOR, 2, cv2.LINE_AA)
    return out


def warp_mask(mask, old_pts, new_pts, w, h):
    if mask is None or old_pts is None or len(old_pts) < 3:
        return mask
    M, _ = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC)
    if M is None:
        return mask
    return cv2.warpAffine(
        mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST
    ).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="scratch/seg_compare.mp4")
    ap.add_argument("--infer-size", type=int, default=320)
    ap.add_argument("--skip-n", type=int, default=1,
                     help="Full segmentation every Nth frame; skip frames warp "
                          "the last mask via affine-from-keypoint-motion, matching "
                          "pipeline.py's last_seg_mask propagation. 1 = no skipping.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    print("Loading RTMPose (YOLOX-Nano + RTMPose-T) -- drives keypoint motion for skip-frame warp...")
    apply_detector_patch()
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    print("Loading selfie_seg1 (landscape)...")
    selfie = SelfieSegBlur(model_path=os.path.join(models_dir, "selfie_segmenter_landscape.tflite"))

    print("Loading YOLOv8n-seg...")
    yolo8 = YOLOSegBlur(model_name=os.path.join(models_dir, "yolov8n-seg.pt"),
                         infer_size=args.infer_size, conf=0.4)

    print("Loading YOLO11n-seg (auto-downloads to models/ if missing)...")
    yolo11 = YOLOSegBlur(model_name=os.path.join(models_dir, "yolo11n-seg.pt"),
                          infer_size=args.infer_size, conf=0.4)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_w, panel_h = w // 2, h // 2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 2, panel_h * 2))

    stats = {
        "selfie_seg1": {"time": [], "coverage": []},
        "yolov8n-seg": {"time": [], "coverage": []},
        "yolo11n-seg": {"time": [], "coverage": []},
    }

    total_px = h * w
    m_selfie = m_yolo8 = m_yolo11 = None
    prev_gray = None
    last_kpts = None   # (N,2) float32 keypoints used to warp masks forward

    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_full_frame = (frame_idx % args.skip_n == 0)

        # --- keypoint tracking: fresh detection on full frames, LK flow on skip frames ---
        if is_full_frame:
            kpts_all, scores_all = body(frame)
            if kpts_all is not None and len(kpts_all) > 0:
                curr_kpts = kpts_all.reshape(-1, 2).astype(np.float32)
            else:
                curr_kpts = None
        else:
            if last_kpts is not None and prev_gray is not None:
                new_pts, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, last_kpts.reshape(-1, 1, 2), None, **LK_PARAMS)
                curr_kpts = new_pts.reshape(-1, 2)
            else:
                curr_kpts = None

        if is_full_frame:
            t0 = time.perf_counter()
            new_m_selfie = selfie.get_mask(frame, infer_size=args.infer_size)
            t_selfie = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            new_m_yolo8 = yolo8.get_mask(frame)
            t_yolo8 = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            new_m_yolo11 = yolo11.get_mask(frame)
            t_yolo11 = (time.perf_counter() - t0) * 1000

            m_selfie, m_yolo8, m_yolo11 = new_m_selfie, new_m_yolo8, new_m_yolo11

            stats["selfie_seg1"]["time"].append(t_selfie)
            stats["yolov8n-seg"]["time"].append(t_yolo8)
            stats["yolo11n-seg"]["time"].append(t_yolo11)
        else:
            t_selfie = t_yolo8 = t_yolo11 = 0.0
            if last_kpts is not None and curr_kpts is not None and len(last_kpts) == len(curr_kpts):
                m_selfie = warp_mask(m_selfie, last_kpts, curr_kpts, w, h)
                m_yolo8 = warp_mask(m_yolo8, last_kpts, curr_kpts, w, h)
                m_yolo11 = warp_mask(m_yolo11, last_kpts, curr_kpts, w, h)

        if curr_kpts is not None:
            last_kpts = curr_kpts
        prev_gray = curr_gray

        stats["selfie_seg1"]["coverage"].append(m_selfie.sum() / total_px if m_selfie is not None else 0.0)
        stats["yolov8n-seg"]["coverage"].append(m_yolo8.sum() / total_px if m_yolo8 is not None else 0.0)
        stats["yolo11n-seg"]["coverage"].append(m_yolo11.sum() / total_px if m_yolo11 is not None else 0.0)

        orig_p = cv2.resize(frame, (panel_w, panel_h))
        sel_p = cv2.resize(selfie.apply_mask(frame, m_selfie) if m_selfie is not None else frame, (panel_w, panel_h))
        y8_p = cv2.resize(yolo8.apply_mask(frame, m_yolo8) if m_yolo8 is not None else frame, (panel_w, panel_h))
        y11_p = cv2.resize(yolo11.apply_mask(frame, m_yolo11) if m_yolo11 is not None else frame, (panel_w, panel_h))

        tag = "" if is_full_frame else " (warped)"
        top = np.hstack([label(orig_p, "ORIGINAL"), label(sel_p, f"selfie_seg1 {t_selfie:.0f}ms{tag}")])
        bot = np.hstack([label(y8_p, f"YOLOv8n-seg {t_yolo8:.0f}ms{tag}"), label(y11_p, f"YOLO11n-seg {t_yolo11:.0f}ms{tag}")])
        grid = np.vstack([top, bot])
        writer.write(grid)

        if frame_idx % 30 == 0:
            print(f"[F {frame_idx:4d}]{tag} selfie={t_selfie:6.1f}ms  yolov8={t_yolo8:6.1f}ms  yolo11={t_yolo11:6.1f}ms")
        frame_idx += 1

    cap.release()
    writer.release()
    selfie.close()

    print(f"\nWrote comparison video -> {args.out}  ({frame_idx} frames, skip-n={args.skip_n})")
    print("\n=== Summary ===")
    print(f"{'Backend':<14} {'Avg ms/full-frame':<19} {'Effective ms/frame':<20} {'Avg mask coverage':<18}")
    for name, d in stats.items():
        avg_full = np.mean(d["time"]) if d["time"] else 0.0
        effective = avg_full / args.skip_n
        avg_c = np.mean(d["coverage"]) if d["coverage"] else 0.0
        print(f"{name:<14} {avg_full:<19.1f} {effective:<20.1f} {avg_c * 100:<17.2f}%")


if __name__ == "__main__":
    main()
