"""
Reimplements the STATIC/JITTER path of Danial's Tier-2 Android "Reveal-and-Fill"
background reconstruction (BackgroundInpaint.kt), as documented in
danial/Tier_2_Mobile_Comprehensive_Guide.md and PHONE_APP_STATUS.md. Increment 1
(per user's explicit "step by step" instruction, 2026-07-26): STATIC/JITTER path
only -- no motion-detection branch, no DYNAMIC/windowed cross-window borrowing yet.

Algorithm (matching the documented Kotlin implementation):
  1. Sub-pixel alignment pyramid: coarse->fine->native SAD-based translation
     search, 3 levels ((100px,+-16px), (300px,+-4px), (native,+-3px) per the
     doc), with parabolic sub-pixel refinement:
       dx = bx + 0.5*(SAD(bx-1)-SAD(bx+1)) / (SAD(bx-1)-2*SAD(bx)+SAD(bx+1))
  2. Temporal trimmed-mean aggregation ("robustCenter"): for each pixel,
     collect its value across every frame where it's NOT masked (person
     hole), sort, take the mean of the middle 60% (drop lowest/highest 20%
     each) -- sharper than a plain median per the doc's own reasoning.
  3. big-LaMa fills ONLY the never-revealed core (pixels masked as
     person-hole in EVERY frame) -- ONCE per clip, matching Danial's
     confirmed once-per-clip design (not per-frame).

Input: masked_video.mp4 (person already grey-filled) + mask.mp4 (white=person),
both produced by this project's own yoloseg11ncnn export
(scripts/run.py --export-dir ...).

Usage:
  python reveal_and_fill_static.py --masked-video <path> --mask-video <path> --out-dir <dir>
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch


def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def sad(a, b):
    return np.sum(np.abs(a.astype(np.int32) - b.astype(np.int32)))


def parabolic_subpixel(sad_m1, sad_0, sad_p1):
    denom = sad_m1 - 2 * sad_0 + sad_p1
    if abs(denom) < 1e-6:
        return 0.0
    return 0.5 * (sad_m1 - sad_p1) / denom


def align_translation(ref_gray, tgt_gray, search_radius, scale_hint=None):
    """SAD-based integer-pixel translation search within +-search_radius,
    then parabolic sub-pixel refinement per-axis. Returns (dx, dy)."""
    h, w = ref_gray.shape
    best_sad = None
    best_dx, best_dy = 0, 0
    cy0, cx0 = h // 2, w // 2
    half = min(h, w) // 4  # central patch for SAD comparison, avoids edge effects
    ref_patch = ref_gray[cy0 - half:cy0 + half, cx0 - half:cx0 + half]

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            y0, x0 = cy0 - half + dy, cx0 - half + dx
            if y0 < 0 or x0 < 0 or y0 + 2 * half > h or x0 + 2 * half > w:
                continue
            tgt_patch = tgt_gray[y0:y0 + 2 * half, x0:x0 + 2 * half]
            s = sad(ref_patch, tgt_patch)
            if best_sad is None or s < best_sad:
                best_sad = s
                best_dx, best_dy = dx, dy

    def sad_at(ddx, ddy):
        y0, x0 = cy0 - half + ddy, cx0 - half + ddx
        if y0 < 0 or x0 < 0 or y0 + 2 * half > h or x0 + 2 * half > w:
            return best_sad
        return sad(ref_patch, tgt_gray[y0:y0 + 2 * half, x0:x0 + 2 * half])

    sx = parabolic_subpixel(sad_at(best_dx - 1, best_dy), best_sad, sad_at(best_dx + 1, best_dy))
    sy = parabolic_subpixel(sad_at(best_dx, best_dy - 1), best_sad, sad_at(best_dx, best_dy + 1))
    return best_dx + sx, best_dy + sy


def align_pyramid(ref_gray, tgt_gray):
    """3-level coarse->fine->native alignment pyramid, per the documented
    levels: (100px, +-16px), (300px, +-4px), (native, +-3px)."""
    h, w = ref_gray.shape
    total_dx, total_dy = 0.0, 0.0
    cur_tgt = tgt_gray

    levels = [
        (100, 16),
        (300, 4),
        (max(h, w), 3),
    ]
    for target_dim, radius in levels:
        scale = min(1.0, target_dim / max(h, w))
        if scale < 1.0:
            rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
            ref_s = cv2.resize(ref_gray, (rw, rh), interpolation=cv2.INTER_AREA)
            tgt_s = cv2.resize(cur_tgt, (rw, rh), interpolation=cv2.INTER_AREA)
        else:
            ref_s, tgt_s = ref_gray, cur_tgt

        dx, dy = align_translation(ref_s, tgt_s, radius)
        if scale < 1.0:
            dx, dy = dx / scale, dy / scale
        total_dx += dx
        total_dy += dy

        M = np.array([[1, 0, -total_dx], [0, 1, -total_dy]], dtype=np.float32)
        cur_tgt = cv2.warpAffine(tgt_gray, M, (w, h), flags=cv2.INTER_LINEAR)

    return total_dx, total_dy


def trimmed_mean_vectorized(channel_stack, valid_stack, trim_frac=0.20):
    """Vectorized per-pixel temporal trimmed-mean across the whole frame at
    once (replaces a per-pixel Python loop, which is infeasible at
    1264x1264x302 frames -- millions of pixels x hundreds of samples each
    in pure Python would take hours). Masked-out (person-hole) samples are
    set to +inf before sorting so they always land at the high end and get
    trimmed/ignored, EXCEPT this would bias a pixel with few valid samples
    -- so trimming fractions are computed per-pixel from n_valid, not a
    fixed count (sort, drop the lowest/highest trim_frac of the VALID
    samples each, mean of the remaining middle -- matches the documented
    Kotlin robustCenter()'s semantics per-pixel, not a global N).

    channel_stack: (N, H, W) float32, single color channel
    valid_stack:   (N, H, W) bool, True = real (non-person) pixel
    Returns: (H, W) float32 plate for this channel, and (H,W) bool of
    pixels with zero valid samples (never-revealed -- same for all
    channels, caller only needs to compute this once).
    """
    n, h, w = channel_stack.shape
    n_valid = valid_stack.sum(axis=0)  # (H, W)

    # Push invalid samples to +inf so sorting moves them to the end;
    # per-pixel valid counts (not the fixed N) drive the trim boundaries.
    masked = np.where(valid_stack, channel_stack, np.inf).reshape(n, h * w)
    sorted_vals = np.sort(masked, axis=0)  # (N, H*W), invalid samples now trail

    n_valid_flat = n_valid.reshape(h * w)
    lo = (n_valid_flat * trim_frac).astype(np.int64)
    hi = n_valid_flat - lo

    # Build a per-pixel boolean keep-mask along axis 0 via broadcasting
    idx = np.arange(n).reshape(n, 1)
    keep = (idx >= lo.reshape(1, h * w)) & (idx < hi.reshape(1, h * w))
    # Pixels where hi<=lo (too few samples to trim) keep everything valid instead
    degenerate = hi <= lo
    keep[:, degenerate] = idx[:, 0:1] < n_valid_flat[degenerate].reshape(1, -1)

    sorted_vals_safe = np.where(np.isfinite(sorted_vals), sorted_vals, 0.0)
    sum_kept = (sorted_vals_safe * keep).sum(axis=0)
    count_kept = keep.sum(axis=0).astype(np.float32)
    count_kept[count_kept == 0] = 1  # avoid div-by-zero for never-revealed pixels (result unused there)

    plate_channel = (sum_kept / count_kept).reshape(h, w)
    never_revealed = (n_valid == 0)
    return plate_channel, never_revealed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked-video", required=True)
    ap.add_argument("--mask-video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mask-thresh", type=int, default=127)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading frames...")
    t0 = time.time()
    color_frames = load_frames(args.masked_video)
    mask_frames = load_frames(args.mask_video)
    n = min(len(color_frames), len(mask_frames))
    color_frames, mask_frames = color_frames[:n], mask_frames[:n]
    h, w = color_frames[0].shape[:2]
    print(f"  {n} frames, {w}x{h}, loaded in {time.time()-t0:.1f}s")

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in color_frames]
    bool_masks = [cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) > args.mask_thresh for m in mask_frames]

    print("Aligning frames (sub-pixel pyramid, reference = frame 0)...")
    t0 = time.time()
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
        if i % 50 == 0:
            print(f"  aligned {i}/{n} frames (last shift dx={dx:.2f} dy={dy:.2f})...")
    print(f"  alignment done in {time.time()-t0:.1f}s")

    print("Temporal trimmed-mean aggregation (vectorized)...")
    t0 = time.time()
    color_stack = np.stack(aligned_color, axis=0).astype(np.float32)  # (N, H, W, 3)
    mask_stack = np.stack(aligned_masks, axis=0)   # (N, H, W) True = person (exclude)
    valid_stack = ~mask_stack  # True = usable real background pixel

    plate = np.zeros((h, w, 3), dtype=np.uint8)
    never_revealed = None
    for c in range(3):
        plate_channel, never_revealed = trimmed_mean_vectorized(color_stack[:, :, :, c], valid_stack)
        plate[:, :, c] = np.clip(np.round(plate_channel), 0, 255).astype(np.uint8)
        print(f"  channel {c} done")
    print(f"  trimmed-mean done in {time.time()-t0:.1f}s")

    # Only pixels the REFERENCE frame's own mask actually covered need any
    # reconstruction at all -- everywhere else is already real, untouched
    # background in frame 0, and copying the aggregated/aligned plate there
    # instead only adds alignment-error risk (blur/smear from imperfect
    # cross-frame registration) for zero benefit. Restrict the plate to
    # frame 0's mask region; use frame 0's real pixels everywhere else.
    needs_reconstruction = bool_masks[0]
    plate = np.where(needs_reconstruction[:, :, None], plate, color_frames[0])
    revealed_in_region = (~never_revealed) & needs_reconstruction
    never_revealed = never_revealed & needs_reconstruction

    n_region = max(1, int(needs_reconstruction.sum()))
    real_pct = 100.0 * int(revealed_in_region.sum()) / n_region
    print(f"  reconstruction region: {n_region} px ({100.0*n_region/needs_reconstruction.size:.1f}% of frame)")
    print(f"  {real_pct:.1f}% of the RECONSTRUCTION region has >=1 real pixel sample (upper bound on real-pixel coverage)")

    cv2.imwrite(os.path.join(args.out_dir, "plate_before_lama.png"), plate)
    cv2.imwrite(os.path.join(args.out_dir, "never_revealed_mask.png"), (never_revealed * 255).astype(np.uint8))

    print("\nRunning big-LaMa on the never-revealed core (once, per Danial's confirmed once-per-clip design)...")
    t0 = time.time()
    final_plate = run_lama_fill(plate, never_revealed)
    lama_ms = (time.time() - t0) * 1000
    print(f"  LaMa fill took {lama_ms:.1f}ms")

    cv2.imwrite(os.path.join(args.out_dir, "plate_final.png"), final_plate)

    print(f"\nDone. Outputs in {args.out_dir}:")
    print("  plate_before_lama.png -- trimmed-mean plate, never-revealed core still visible as-is")
    print("  never_revealed_mask.png -- white = pixels needing LaMa")
    print("  plate_final.png -- after LaMa core-fill")
    print(f"\nSummary: real-pixel coverage {real_pct:.1f}%, never-revealed core {100-real_pct:.1f}%, LaMa call {lama_ms:.1f}ms")


def run_lama_fill(plate_bgr, never_revealed_mask, size=512):
    """Runs big-lama.pt on the plate, masking only never_revealed_mask.
    Resizes to a square working resolution (matches the pattern used in
    scripts/lama_onnx_export.py's _resize_square), pastes back at original
    resolution afterward."""
    h, w = plate_bgr.shape[:2]
    model = torch.jit.load(os.path.join(os.path.dirname(__file__), "..", "models", "big-lama.pt"), map_location="cpu")
    model.eval()

    image_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(never_revealed_mask.astype(np.uint8) * 255, (size, size), interpolation=cv2.INTER_NEAREST)

    image_t = torch.from_numpy(image_resized).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)
    mask_t = torch.from_numpy(mask_resized).float().div(255.0).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        result = model(image_t, mask_t)

    result_np = (result[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    result_full = cv2.resize(result_np, (w, h), interpolation=cv2.INTER_LINEAR)
    result_bgr = cv2.cvtColor(result_full, cv2.COLOR_RGB2BGR)

    out = plate_bgr.copy()
    out[never_revealed_mask] = result_bgr[never_revealed_mask]
    return out


if __name__ == "__main__":
    main()
