"""
Visual quality check: YOLO11n-seg PyTorch (.pt, FP32) vs ONNX (FP32) export.
bench_yolo_precision.py found ONNX ~2.8x faster than .pt on CPU -- this
confirms mask output is visually equivalent before switching the default
runtime in blur_yoloseg.py.

Writes a 1x3 grid (ORIGINAL | PT | ONNX) plus a per-frame pixel-diff count
between the two masks (0 = identical).

Usage: python scripts/compare_yolo_precision.py <video> [--max-frames N] [--skip-n N] [--out out.mp4]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from body_sitara.blur_yoloseg import YOLOSegBlur
from body_sitara.pose import LK_PARAMS
from body_sitara.detector_patch import apply_detector_patch
from rtmlib import Body

PANEL_LABEL_COLOR = (0, 255, 255)


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
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="scratch/yolo_precision_compare.mp4")
    ap.add_argument("--skip-n", type=int, default=1)
    ap.add_argument("--model-b", default="yolo11n-seg.onnx",
                     help="Second model to compare against PT ground truth, "
                          "e.g. yolo11n-seg.onnx (FP32) or yolo11n-seg-int8.onnx (INT8).")
    ap.add_argument("--label-b", default=None,
                     help="Panel label for model-b; defaults to the filename.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    label_b = args.label_b or args.model_b

    print("Loading RTMPose (drives skip-frame mask warp)...")
    apply_detector_patch()
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    print("Loading YOLO11n-seg PyTorch (.pt)...")
    yolo_pt = YOLOSegBlur(model_name=os.path.join(models_dir, "yolo11n-seg.pt"), infer_size=320, conf=0.4)

    print(f"Loading {label_b}...")
    yolo_onnx = YOLOSegBlur(model_name=os.path.join(models_dir, args.model_b), infer_size=320, conf=0.4)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w, panel_h = w // 2, h // 2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 3, panel_h))

    total_px = h * w
    t_pt_list, t_onnx_list, diff_pct_list, cov_pt_list, cov_onnx_list = [], [], [], [], []
    m_pt = m_onnx = None
    prev_gray = None
    last_kpts = None

    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_full_frame = (frame_idx % args.skip_n == 0)

        if is_full_frame:
            kpts_all, _ = body(frame)
            curr_kpts = kpts_all.reshape(-1, 2).astype(np.float32) if kpts_all is not None and len(kpts_all) > 0 else None
        else:
            if last_kpts is not None and prev_gray is not None:
                new_pts, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, last_kpts.reshape(-1, 1, 2), None, **LK_PARAMS)
                curr_kpts = new_pts.reshape(-1, 2)
            else:
                curr_kpts = None

        if is_full_frame:
            t0 = time.perf_counter()
            m_pt = yolo_pt.get_mask(frame)
            t_pt = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            m_onnx = yolo_onnx.get_mask(frame)
            t_onnx = (time.perf_counter() - t0) * 1000

            t_pt_list.append(t_pt)
            t_onnx_list.append(t_onnx)
        else:
            t_pt = t_onnx = 0.0
            if last_kpts is not None and curr_kpts is not None and len(last_kpts) == len(curr_kpts):
                m_pt = warp_mask(m_pt, last_kpts, curr_kpts, w, h)
                m_onnx = warp_mask(m_onnx, last_kpts, curr_kpts, w, h)

        if curr_kpts is not None:
            last_kpts = curr_kpts
        prev_gray = curr_gray

        a = m_pt if m_pt is not None else np.zeros((h, w), dtype=bool)
        b = m_onnx if m_onnx is not None else np.zeros((h, w), dtype=bool)
        diff_pct = (a != b).sum() / total_px * 100
        diff_pct_list.append(diff_pct)
        cov_pt_list.append(a.sum() / total_px * 100)
        cov_onnx_list.append(b.sum() / total_px * 100)

        orig_p = cv2.resize(frame, (panel_w, panel_h))
        pt_p = cv2.resize(yolo_pt.apply_mask(frame, m_pt) if m_pt is not None else frame, (panel_w, panel_h))
        onnx_p = cv2.resize(yolo_onnx.apply_mask(frame, m_onnx) if m_onnx is not None else frame, (panel_w, panel_h))

        tag = "" if is_full_frame else " (warped)"
        grid = np.hstack([
            label(orig_p, "ORIGINAL"),
            label(pt_p, f"PT {t_pt:.0f}ms{tag}"),
            label(onnx_p, f"{label_b} {t_onnx:.0f}ms{tag}  diff={diff_pct:.2f}%"),
        ])
        writer.write(grid)

        if frame_idx % 30 == 0:
            print(f"[F {frame_idx:4d}]{tag} PT={t_pt:6.1f}ms  {label_b}={t_onnx:6.1f}ms  mask_diff={diff_pct:5.2f}%")
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nWrote comparison video -> {args.out}  ({frame_idx} frames, skip-n={args.skip_n})")
    print("\n=== Summary ===")
    print(f"PT{'':<{len(label_b)-1}} : mean full-frame {np.mean(t_pt_list):.1f}ms   mean coverage {np.mean(cov_pt_list):.2f}%")
    print(f"{label_b} : mean full-frame {np.mean(t_onnx_list):.1f}ms   mean coverage {np.mean(cov_onnx_list):.2f}%")
    print(f"Speedup: {np.mean(t_pt_list) / np.mean(t_onnx_list):.2f}x")
    print(f"Mean mask disagreement (all frames incl. warped): {np.mean(diff_pct_list):.3f}% of pixels")
    print(f"Max mask disagreement: {np.max(diff_pct_list):.3f}% of pixels")


if __name__ == "__main__":
    main()
