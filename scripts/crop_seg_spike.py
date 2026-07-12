"""
Spike: does cropping to the detector's person bbox before running YOLO-seg
save latency without hurting mask quality?

Today YOLOSegBlur.get_mask() always runs on the full, uncropped frame
(blur_yoloseg.py) even though pipeline.py already computes a person bbox
via the RTMPose detector shortly after. This script runs yolo11n-seg-int8
both ways -- full frame, and cropped to the (padded) largest detector
bbox -- on the same clip, every frame (no skip-n), and reports timing +
mask agreement (IoU) so the tradeoff can be judged before wiring this into
blur_yoloseg.py/pipeline.py for real.

Throwaway/exploratory only -- does not touch pipeline.py or blur_yoloseg.py.

Usage:
    python scripts/crop_seg_spike.py <video> [--max-frames N] [--out out.mp4] [--pad N]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from body_sitara.blur_yoloseg import YOLOSegBlur
from body_sitara.detector_patch import apply_detector_patch
from rtmlib import Body

PANEL_LABEL_COLOR = (0, 255, 255)
INFER_SIZE = 320


def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, PANEL_LABEL_COLOR, 2, cv2.LINE_AA)
    return out


def largest_box(bboxes):
    """Pick the largest-area box (matches pipeline.py's largest-subject intent)."""
    if bboxes is None or len(bboxes) == 0:
        return None
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return bboxes[int(np.argmax(areas))]


def pad_and_clamp(box, pad, w, h):
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="scratch/crop_seg_spike.mp4")
    ap.add_argument("--pad", type=int, default=30,
                     help="Bbox padding (px) before cropping -- extra context for "
                          "segmentation boundary refinement, a bit more generous than "
                          "pose.py's BODY_CROP_PADDING=20.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    print("Loading RTMPose (YOLOX-Nano + RTMPose-T) -- used only for the person bbox...")
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
    if not cap.isOpened():
        print(f"Error: could not open {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_w, panel_h = w // 2, h // 2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 2, panel_h))

    # --- Pass 1: read all frames, run the detector once per frame, cache the
    # padded crop region. Kept fully separate from Pass 2's get_mask() timing
    # loop -- calling body.det_model() and yoloseg.get_mask() back-to-back
    # causes severe ONNX Runtime CPU thread contention (measured standalone:
    # a lone get_mask() call ~48ms, the same call immediately after a
    # det_model() call ~108ms -- over 2x slower). pipeline.py itself
    # deliberately avoids this ordering ("yolo_seg runs BEFORE det+pose" at
    # pipeline.py:378). This spike's whole point is comparing get_mask()
    # full-frame vs cropped latency, so det-model timing must not leak into
    # either measurement.
    print("Pass 1/2: running detector to cache per-frame crop regions...")
    frames = []
    regions = []  # None or (x1,y1,x2,y2), full-frame pixel space
    kp_scale_x, kp_scale_y = w / INFER_SIZE, h / INFER_SIZE
    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        infer_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
        bboxes = body.det_model(infer_frame)
        box = largest_box(bboxes)
        region = None
        if box is not None:
            sb = box.copy().astype(float)
            sb[0] *= kp_scale_x; sb[2] *= kp_scale_x
            sb[1] *= kp_scale_y; sb[3] *= kp_scale_y
            region = pad_and_clamp(sb, args.pad, w, h)
        regions.append(region)
        frame_idx += 1
    cap.release()
    print(f"Pass 1 done: {len(frames)} frames read.")

    # --- Pass 2: measure get_mask() full-frame vs cropped, with nothing
    # else (no det_model calls) interleaved in this loop.
    print("Pass 2/2: measuring full-frame vs cropped get_mask()...")
    t_full_list, t_crop_list = [], []
    cov_full_list, cov_crop_list = [], []
    iou_list = []
    crop_area_ratio_list = []
    frames_with_box = 0
    total_px = h * w

    for frame_idx, (frame, region) in enumerate(zip(frames, regions)):
        t0 = time.perf_counter()
        mask_full = yoloseg.get_mask(frame)
        t_full = (time.perf_counter() - t0) * 1000
        t_full_list.append(t_full)

        mask_crop = None
        t_crop = 0.0
        if region is not None:
            x1, y1, x2, y2 = region
            crop = frame[y1:y2, x1:x2]
            crop_area_ratio_list.append((crop.shape[0] * crop.shape[1]) / total_px)
            t0 = time.perf_counter()
            local_mask = yoloseg.get_mask(crop)
            t_crop = (time.perf_counter() - t0) * 1000
            if local_mask is not None:
                mask_crop = np.zeros((h, w), dtype=bool)
                mask_crop[y1:y2, x1:x2] = local_mask
            frames_with_box += 1
            t_crop_list.append(t_crop)

        cov_full_list.append(mask_full.sum() / total_px if mask_full is not None else 0.0)
        cov_crop_list.append(mask_crop.sum() / total_px if mask_crop is not None else 0.0)

        if mask_full is not None and mask_crop is not None:
            inter = (mask_full & mask_crop).sum()
            union = (mask_full | mask_crop).sum()
            if union > 0:
                iou_list.append(inter / union)
        elif mask_full is None and mask_crop is None:
            iou_list.append(1.0)
        else:
            iou_list.append(0.0)

        full_p = cv2.resize(
            yoloseg.apply_mask(frame, mask_full) if mask_full is not None else frame,
            (panel_w, panel_h))
        crop_p = cv2.resize(
            yoloseg.apply_mask(frame, mask_crop) if mask_crop is not None else frame,
            (panel_w, panel_h))

        top = np.hstack([
            label(full_p, f"FULL-FRAME {t_full:.0f}ms"),
            label(crop_p, f"CROPPED {t_crop:.0f}ms"),
        ])
        writer.write(top)

        if frame_idx % 30 == 0:
            iou_str = f"{iou_list[-1]:.3f}" if iou_list else "n/a"
            print(f"[F {frame_idx:4d}] full={t_full:6.1f}ms  crop={t_crop:6.1f}ms  iou={iou_str}")

    writer.release()

    print(f"\nWrote comparison video -> {args.out}  ({frame_idx} frames)")
    print("\n=== Summary ===")
    avg_full = np.mean(t_full_list) if t_full_list else 0.0
    avg_crop = np.mean(t_crop_list) if t_crop_list else 0.0
    avg_iou = np.mean(iou_list) if iou_list else 0.0
    avg_cov_full = np.mean(cov_full_list) * 100
    avg_cov_crop = np.mean(cov_crop_list) * 100
    avg_crop_ratio = np.mean(crop_area_ratio_list) * 100 if crop_area_ratio_list else 0.0

    print(f"Frames with a detected bbox: {frames_with_box}/{frame_idx}")
    print(f"Avg crop area vs frame area: {avg_crop_ratio:.1f}%")
    print(f"Avg full-frame inference:    {avg_full:.1f} ms")
    print(f"Avg cropped inference:       {avg_crop:.1f} ms")
    if avg_crop > 0:
        print(f"Speedup (cropped vs full):   {avg_full / avg_crop:.2f}x")
    print(f"Avg mask IoU (full vs crop): {avg_iou:.3f}")
    print(f"Avg coverage -- full:        {avg_cov_full:.2f}%")
    print(f"Avg coverage -- cropped:     {avg_cov_crop:.2f}%")


if __name__ == "__main__":
    main()
