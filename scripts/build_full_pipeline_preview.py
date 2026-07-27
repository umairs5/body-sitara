"""
Local Python reproduction of the full 6-step Tier2 pipeline (matching the
iOS BenchmarkView.swift implementation) -- lets pipeline output videos be
inspected directly, without a CI build + Sideloadly install round-trip.

Pipeline: grey silhouette + mask -> Background Reconstruction (align +
trimmed-mean + LaMa, restricted to frame-0's mask region only) -> per-frame
reconstructed-background video (real frames outside the fill region, plate
inside it) -> Lightmap -> silhouette-on-lightmap (outbound-to-server
signal) -> [server placeholder] -> final composite (placeholder avatar on
per-frame reconstructed background).

Usage:
  python build_full_pipeline_preview.py --masked-video <path> --mask-video <path> --out-dir <dir>
"""
import argparse
import os

import cv2
import numpy as np

from reveal_and_fill_static import (
    load_frames, align_pyramid, trimmed_mean_vectorized, run_lama_fill,
)


def build_lightmap(bg_bgr, downscale=48, radius=2, sigma=3.0):
    h, w = bg_bgr.shape[:2]
    small = cv2.resize(bg_bgr, (downscale, downscale), interpolation=cv2.INTER_AREA)
    ksize = 2 * radius + 1
    blurred = cv2.GaussianBlur(small, (ksize, ksize), sigma)
    return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)


def placeholder_character(w, h):
    char = np.zeros((h, w, 3), dtype=np.uint8)
    char[:, :] = (120, 140, 160)  # BGR
    alpha = np.zeros((h, w), dtype=np.float32)
    cy, cx = h / 2, w / 2
    ry, rx = h * 0.38, w * 0.18
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    feather = 6.0 / rx
    alpha = np.clip(1.0 - (dist - 1.0) / feather, 0.0, 1.0)
    alpha[dist <= 1.0] = 1.0
    return char, alpha


def composite(bg_bgr, fg_bgr, alpha):
    a = alpha[:, :, None]
    return (fg_bgr.astype(np.float32) * a + bg_bgr.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)


def write_video(frames, path, fps):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked-video", required=True)
    ap.add_argument("--mask-video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading frames...")
    color_frames = load_frames(args.masked_video)
    mask_frames = load_frames(args.mask_video)
    n = min(len(color_frames), len(mask_frames))
    color_frames, mask_frames = color_frames[:n], mask_frames[:n]
    h, w = color_frames[0].shape[:2]
    bool_masks = [cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > 127 for m in mask_frames]
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in color_frames]
    print(f"  {n} frames, {w}x{h}")

    print("Step 2a: aligning frames...")
    ref_gray = gray_frames[0]
    aligned_color = [color_frames[0]]
    aligned_masks = [bool_masks[0]]
    for i in range(1, n):
        dx, dy = align_pyramid(ref_gray, gray_frames[i])
        M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        warped_color = cv2.warpAffine(color_frames[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(bool_masks[i].astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=1) > 0
        aligned_color.append(warped_color)
        aligned_masks.append(warped_mask)

    print("Step 2b: temporal trimmed-mean...")
    color_stack = np.stack(aligned_color, axis=0).astype(np.float32)
    mask_stack = np.stack(aligned_masks, axis=0)
    valid_stack = ~mask_stack
    plate = np.zeros((h, w, 3), dtype=np.uint8)
    never_revealed = None
    for c in range(3):
        plate_channel, never_revealed = trimmed_mean_vectorized(color_stack[:, :, :, c], valid_stack)
        plate[:, :, c] = np.clip(np.round(plate_channel), 0, 255).astype(np.uint8)

    needs_reconstruction = bool_masks[0]
    plate = np.where(needs_reconstruction[:, :, None], plate, color_frames[0])
    never_revealed = never_revealed & needs_reconstruction

    print("Step 2c: LaMa core-fill (once per clip)...")
    background_final = run_lama_fill(plate, never_revealed)

    print("Building per-frame reconstructed-background video (real frames outside fill region)...")
    reconstructed_bg_frames = []
    for i in range(n):
        frame_out = color_frames[i].copy()
        frame_out[needs_reconstruction] = background_final[needs_reconstruction]
        reconstructed_bg_frames.append(frame_out)
    write_video(reconstructed_bg_frames, os.path.join(args.out_dir, "reconstructed_background.mp4"), args.fps)
    print(f"  wrote reconstructed_background.mp4 ({len(reconstructed_bg_frames)} frames)")

    print("Step 3: lightmap...")
    lightmap = build_lightmap(background_final)
    cv2.imwrite(os.path.join(args.out_dir, "lightmap.png"), lightmap)

    print("Step 4: silhouette-on-lightmap (outbound-to-server signal)...")
    sil_frames = []
    for i in range(n):
        alpha = bool_masks[i].astype(np.float32)
        sil_frames.append(composite(lightmap, color_frames[i], alpha))
    write_video(sil_frames, os.path.join(args.out_dir, "silhouette_on_lightmap.mp4"), args.fps)
    print(f"  wrote silhouette_on_lightmap.mp4 ({len(sil_frames)} frames)")

    print("Steps 5-6: placeholder avatar composited on per-frame reconstructed background...")
    char, alpha = placeholder_character(w, h)
    final_frames = [composite(bg, char, alpha) for bg in reconstructed_bg_frames]
    write_video(final_frames, os.path.join(args.out_dir, "final_output.mp4"), args.fps)
    print(f"  wrote final_output.mp4 ({len(final_frames)} frames)")

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
