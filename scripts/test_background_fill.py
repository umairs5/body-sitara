"""
Visual + timing test for the cv2.inpaint background filler
(src/body_sitara/background_fill.py) against a real dense-export bundle.

Usage:
    python scripts/test_background_fill.py <export_dir> [--frame N] [--out out.png]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from body_sitara.background_fill import BackgroundFiller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--frame", type=int, default=90)
    ap.add_argument("--out", default="scratch/bgfill_result.png")
    ap.add_argument("--method", choices=["telea", "ns"], default="telea")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--dilate", type=int, default=0,
                     help="Grow the mask by this many pixels before inpainting (see lama_spike.py's "
                          "--dilate for the rationale: excludes a contaminated boundary ring).")
    args = ap.parse_args()

    method = cv2.INPAINT_TELEA if args.method == "telea" else cv2.INPAINT_NS
    filler = BackgroundFiller(method=method, radius=args.radius)

    video_path = os.path.join(args.export_dir, "output_rtm.mp4")
    mask_path = os.path.join(args.export_dir, "mask.mp4")

    cap_v = cv2.VideoCapture(video_path)
    cap_m = cv2.VideoCapture(mask_path)
    cap_v.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    cap_m.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok_v, frame = cap_v.read()
    ok_m, mask_frame = cap_m.read()
    cap_v.release()
    cap_m.release()
    if not (ok_v and ok_m):
        print("Could not read frame/mask")
        return

    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    mask_bool = mask_gray > 127
    if args.dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate * 2 + 1,) * 2)
        mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
    print(f"Frame {args.frame}: {frame.shape}, mask covers {mask_bool.mean()*100:.1f}% of pixels"
          f"{' (dilated +' + str(args.dilate) + 'px)' if args.dilate > 0 else ''}")

    t0 = time.perf_counter()
    result = filler.fill_frame(frame, mask_bool)
    elapsed = time.perf_counter() - t0
    print(f"Inpaint time: {elapsed*1000:.2f}ms")

    mask_overlay = frame.copy()
    mask_overlay[mask_bool] = (0, 0, 255)
    grid = np.hstack([frame, mask_overlay, result])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Wrote comparison -> {args.out}")
    print(f"\nExtrapolated for a 900-frame clip: {elapsed*900:.2f}s if run serially")


if __name__ == "__main__":
    main()
