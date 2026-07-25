"""
Spike: persistent ROI tracking for YOLO-seg -- does using the PREVIOUS
frame's detected bbox as a stable crop window (re-inferred at a smaller
native size, not resized back up to 320) beat full-frame inference?

Different from crop_seg_spike.py's earlier (negative) result: that test
cropped to the CURRENT frame's own bbox but still ran imgsz=320, so
ultralytics letterboxed the crop back up to 320x320 regardless of source
size -- no FLOPs saved on the actual forward pass, only ~1.12x from
cheaper preprocessing. This spike instead re-exports the model at a
genuinely smaller fixed size (e.g. 192) and feeds it a crop sized to
roughly that native resolution, so less real compute happens per call --
the tradeoff being no full-frame scan every frame, so a fast-moving or
newly-entering person can be missed until the periodic full-frame rescan
catches them.

Tracks and reports:
  - FPS (ROI-tracked vs full-frame baseline)
  - Drift/loss rate: how often the ROI's next-frame crop, built from a
    box padded by --pad px, actually still contains the person (measured
    via IoU between the ROI-path detection and a full-frame "ground
    truth" detection run every frame purely for comparison, not used to
    drive the ROI itself)
  - How often the periodic full-frame rescan was needed to reacquire

Single-person only (multi-person ROI tracking, and the new-person-entry
problem, are real open questions this spike deliberately doesn't
address -- see the summary printed at the end).

Throwaway/exploratory -- does not touch pipeline.py or blur_yoloseg.py.

Usage:
    python scripts/roi_track_spike.py <video> [--max-frames N] [--pad N]
                                       [--roi-infer-size N] [--rescan-every N]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from body_sitara.blur_yoloseg import YOLOSegBlur

FULL_INFER_SIZE = 320


def pad_and_clamp(x1, y1, x2, y2, pad, w, h):
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--pad", type=int, default=60,
                     help="Padding (px, in full-frame space) around the previous "
                          "frame's bbox when building the next frame's ROI crop -- "
                          "needs to absorb one frame's worth of motion.")
    ap.add_argument("--roi-infer-size", type=int, default=192,
                     help="Native network size for the ROI-cropped model (re-exported). "
                          "Smaller than the full-frame 320 -- this is where the real "
                          "FLOPs saving comes from, unlike a same-size crop.")
    ap.add_argument("--rescan-every", type=int, default=10,
                     help="Force a full-frame scan every N frames regardless of ROI "
                          "state, to catch drift/loss and new people entering frame.")
    ap.add_argument("--out", default=None,
                     help="If set, write a visualization video: ROI window box (green), "
                          "resulting mask, RESCAN/LOST labels, so you can see what the "
                          "ROI tracker is actually doing frame to frame.")
    args = ap.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    print(f"Loading yolo11n-seg-int8 @ full frame ({FULL_INFER_SIZE}px, ground truth + rescan)...")
    yoloseg_full = YOLOSegBlur(model_name=os.path.join(models_dir, "yolo11n-seg-int8.onnx"),
                                infer_size=FULL_INFER_SIZE, conf=0.4)

    print(f"Loading yolo11n-seg-int8 @ ROI size ({args.roi_infer_size}px, re-exports if not cached)...")
    yoloseg_roi = YOLOSegBlur(model_name=os.path.join(models_dir, "yolo11n-seg-int8.onnx"),
                               infer_size=args.roi_infer_size, conf=0.4)

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    t_full_list, t_roi_list = [], []
    iou_list = []          # ROI-path bbox vs full-frame "ground truth" bbox, same frame
    rescans = 0
    lost_frames = 0        # frames where the ROI crop produced no mask at all
    roi = None              # current tracking window (x1,y1,x2,y2) in full-frame space

    frame_idx = 0
    while frame_idx < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # Ground truth: full-frame detection every frame, timed separately,
        # used only to measure ROI-path accuracy -- not fed back into the
        # ROI loop itself (that would defeat the point of the spike).
        t0 = time.perf_counter()
        mask_full, _ = yoloseg_full.get_mask_and_boxes(frame)
        t_full_list.append((time.perf_counter() - t0) * 1000)
        gt_bbox = mask_to_bbox(mask_full) if mask_full is not None else None

        need_rescan = (roi is None) or (frame_idx % args.rescan_every == 0)
        roi_used_for_crop = roi  # the window drawn -- None on a rescan frame (no crop was used)
        mask_for_viz = None      # full-frame-sized bool mask, for drawing only

        if need_rescan:
            rescans += 1
            t0 = time.perf_counter()
            mask, boxes = yoloseg_full.get_mask_and_boxes(frame)  # reacquire at full res
            t_roi_list.append((time.perf_counter() - t0) * 1000)  # counts as ROI-path cost
            bbox = mask_to_bbox(mask) if mask is not None else None
            mask_for_viz = mask
            if bbox is not None:
                roi = pad_and_clamp(*bbox, args.pad, w, h)
            else:
                roi = None
                lost_frames += 1
            roi_bbox_full_space = bbox
        else:
            x1, y1, x2, y2 = roi
            crop = frame[y1:y2, x1:x2]
            t0 = time.perf_counter()
            mask_crop, _ = yoloseg_roi.get_mask_and_boxes(crop)
            t_roi_list.append((time.perf_counter() - t0) * 1000)
            if mask_crop is not None:
                local_bbox = mask_to_bbox(mask_crop)
                roi_bbox_full_space = (local_bbox[0] + x1, local_bbox[1] + y1,
                                        local_bbox[2] + x1, local_bbox[3] + y1)
                # Slide the ROI window to follow the person into the next frame.
                roi = pad_and_clamp(*roi_bbox_full_space, args.pad, w, h)
                if writer is not None:
                    mask_for_viz = np.zeros((h, w), dtype=bool)
                    mask_for_viz[y1:y2, x1:x2] = mask_crop
            else:
                roi_bbox_full_space = None
                lost_frames += 1
                roi = None  # force a rescan next frame

        iou_list.append(iou_xyxy(roi_bbox_full_space, gt_bbox))

        if writer is not None:
            vis = frame.copy()
            if mask_for_viz is not None:
                overlay = vis.copy()
                overlay[mask_for_viz] = (127, 127, 127)
                vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
            if roi_used_for_crop is not None:
                x1, y1, x2, y2 = roi_used_for_crop
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green = ROI window used
            status = "RESCAN" if need_rescan else ("TRACKING" if roi_bbox_full_space else "")
            if roi_bbox_full_space is None and not need_rescan:
                status = "LOST"
            color = (0, 0, 255) if status == "LOST" else ((0, 165, 255) if status == "RESCAN" else (0, 255, 0))
            cv2.putText(vis, f"F{frame_idx} {status}  IoU={iou_list[-1]:.2f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, f"F{frame_idx} {status}  IoU={iou_list[-1]:.2f}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            writer.write(vis)

        if frame_idx % 30 == 0:
            print(f"[F {frame_idx:4d}] full={t_full_list[-1]:6.1f}ms  "
                  f"roi={t_roi_list[-1]:6.1f}ms  iou={iou_list[-1]:.3f}  "
                  f"{'RESCAN' if need_rescan else ''}")
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nWrote visualization -> {args.out}")

    avg_full = np.mean(t_full_list)
    avg_roi = np.mean(t_roi_list)
    avg_iou = np.mean(iou_list)

    print("\n=== Summary ===")
    print(f"Frames processed:            {frame_idx}")
    print(f"Full-frame baseline:         {avg_full:.1f} ms/frame  ({1000/avg_full:.2f} FPS if this ran alone)")
    print(f"ROI-tracked (mixed):         {avg_roi:.1f} ms/frame  ({1000/avg_roi:.2f} FPS if this ran alone)")
    print(f"Speedup:                     {avg_full / avg_roi:.2f}x")
    print(f"Rescans triggered:           {rescans}/{frame_idx} ({100*rescans/frame_idx:.1f}%)")
    print(f"Lost-tracking frames:        {lost_frames}/{frame_idx} ({100*lost_frames/frame_idx:.1f}%)")
    print(f"Avg IoU (ROI-path vs full-frame ground truth): {avg_iou:.3f}")
    print("\nNOTE: multi-person tracking and new-person-entry detection are NOT")
    print("modeled here (single tracked ROI only) -- both are real, unresolved")
    print("problems for a real implementation, not solved by this spike.")


if __name__ == "__main__":
    main()
