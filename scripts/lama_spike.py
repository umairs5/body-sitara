"""
LaMa-on-device feasibility spike (Phase 3, background fill / Tier 2B-1).

Loads the TorchScript big-lama.pt checkpoint (MIT-licensed re-package from
enesmsahin/simple-lama-inpainting's GitHub release, user-approved source)
and runs real inference on a real frame + mask from an existing
dense-export bundle.

Preprocessing/postprocessing here is a faithful port of
Acly/comfyui-inpaint-nodes' InpaintWithModel node (the actual node backing
the ComfyUI INPAINT_LoadInpaintModel/INPAINT_InpaintWithModel nodes used
in tier2-workflow/*.json) -- NOT the looser resize+pad-to-modulo-8
approach used in earlier versions of this script, which produced visible
artifacts (a contaminated mask edge letting the model hallucinate
person-like structure, especially bad on some masks). Key differences
that actually matter, confirmed against the real upstream source:

  1. resize_square: pad the shorter side (reflect padding) to make the
     working image square, THEN resize to exactly 256x256 (LaMa's native
     working resolution) via nearest-exact interpolation -- not our old
     "resize to whatever --resize N, pad to modulo-8" approach.
  2. mask_floor(mask, threshold=0.99): a STRICT threshold, not >127/255
     (~0.5). Our mask.mp4 is lossy H.264-compressed, so "should be binary"
     0/255 values have compression noise near edges; a 0.99 threshold
     drops nearly all of that ambiguous boundary from the mask entirely
     -- similar in effect to our earlier "--dilate" workaround, but by
     shrinking the mask at uncertain edges instead of growing it past
     them, which is a more principled fix for the same root problem.
  3. Blend-back: after the model runs and the result is un-padded/resized
     back to source resolution, the FINAL image is composited as
     `original + (lama_output - original) * mask_floor(mask)` -- i.e. the
     unmasked region is guaranteed pixel-identical to the source, and even
     the masked region only takes LaMa's contribution where the (strict)
     mask says so. Our old script used LaMa's raw output directly, which
     let the model's own reconstruction of areas it didn't need to touch
     leak into the final result.

Usage:
    python scripts/lama_spike.py <export_dir_with_mask_and_output_rtm> [--frame N]
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pad_reflect_once(x: torch.Tensor, padding: tuple) -> torch.Tensor:
    # padding = (left, right, top, bottom), matching F.pad's last-dim-first order
    _, _, h, w = x.shape
    padding_arr = np.array(padding)
    size = np.array([w, w, h, h])
    initial_padding = np.minimum(padding_arr, size - 1)
    additional_padding = padding_arr - initial_padding
    x = F.pad(x, tuple(int(v) for v in initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(int(v) for v in additional_padding), mode="constant")
    return x


def resize_square(image: torch.Tensor, mask: torch.Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w

    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w

    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: torch.Tensor, original_size: tuple, upsample: str = "bilinear") -> torch.Tensor:
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        if upsample == "lanczos":
            # cv2's INTER_LANCZOS4 isn't available via F.interpolate -- drop to
            # numpy for this one step. Not part of the real ComfyUI node's
            # default behavior (which always uses bilinear); an explicit
            # opt-in to test whether sharper resampling closes the gap
            # between the clean 256px LaMa output and the blockier final
            # upsampled-to-source-resolution result.
            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            img_np = cv2.resize(img_np, (prev_size, prev_size), interpolation=cv2.INTER_LANCZOS4)
            image = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(image.device)
        else:
            image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0:prev_size - pad_h, 0:prev_size - pad_w]


def mask_floor(mask: torch.Tensor, threshold: float = 0.99) -> torch.Tensor:
    return (mask >= threshold).to(mask.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--frame", type=int, default=90)
    ap.add_argument("--model", default="models/big-lama.pt")
    ap.add_argument("--out", default="scratch/lama_spike_result.png")
    ap.add_argument("--required-size", type=int, default=256,
                     help="LaMa's native working resolution per the real ComfyUI node "
                          "(Acly/comfyui-inpaint-nodes hardcodes 256 for the LaMa architecture).")
    ap.add_argument("--grow-mask", type=int, default=0,
                     help="Dilate the raw mask by this many pixels BEFORE resize_square, "
                          "matching a GrowMask node placed upstream of INPAINT_InpaintWithModel "
                          "in a ComfyUI graph. Not part of the node's own default behavior -- "
                          "an explicit opt-in to test whether a larger source-resolution mask "
                          "changes the 256px-working-resolution result.")
    ap.add_argument("--upsample", choices=["bilinear", "lanczos"], default="bilinear",
                     help="Method for undo_resize_square's upscale back to source resolution. "
                          "bilinear matches the real node's default; lanczos is an opt-in test.")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"Loading TorchScript model from {args.model} ...")
    t0 = time.perf_counter()
    model = torch.jit.load(args.model, map_location=device)
    model.eval()
    print(f"  Loaded in {time.perf_counter() - t0:.2f}s")

    video_path = os.path.join(args.export_dir, "output_rtm.mp4")
    mask_path = os.path.join(args.export_dir, "mask.mp4")

    cap_v = cv2.VideoCapture(video_path)
    cap_m = cv2.VideoCapture(mask_path)
    cap_v.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    cap_m.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok_v, frame_bgr = cap_v.read()
    ok_m, mask_frame = cap_m.read()
    cap_v.release()
    cap_m.release()
    if not (ok_v and ok_m):
        print("Could not read frame/mask")
        return

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    if args.grow_mask > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.grow_mask * 2 + 1,) * 2)
        mask_bin = (mask_gray > 127).astype(np.uint8) * 255
        mask_gray = cv2.dilate(mask_bin, kernel)

    # BHWC uint8 -> BCHW float32 [0,1], matching to_torch()'s permute + our own
    # float normalization (the real node relies on ComfyUI's IMAGE type already
    # being float32 [0,1] BHWC -- we start from raw video frames, so normalize here).
    image_t = torch.from_numpy(frame_rgb).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)
    mask_t = torch.from_numpy(mask_gray).float().div(255.0).unsqueeze(0).unsqueeze(0)

    print(f"Frame {args.frame}: {frame_bgr.shape}, raw mask covers {(mask_gray > 127).mean()*100:.1f}% of pixels"
          f"{' (grown +' + str(args.grow_mask) + 'px)' if args.grow_mask > 0 else ''}")

    work_image, work_mask, original_size = resize_square(image_t, mask_t, args.required_size)
    work_mask_floored = mask_floor(work_mask)
    print(f"  After resize_square -> {tuple(work_image.shape)}, "
          f"mask_floor(0.99) covers {work_mask_floored.mean().item()*100:.1f}% of the {args.required_size}px working frame")

    print("Running inference...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        torch.manual_seed(0)
        result = model(work_image.to(device), work_mask_floored.to(device))
    elapsed = time.perf_counter() - t0
    print(f"  Inference time: {elapsed*1000:.1f}ms")

    # Save the raw 256x256 working-resolution output before any upsampling,
    # to isolate whether artifacts come from the model itself or from the
    # upsample-back/blend step.
    raw_256 = result[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    raw_256_bgr = cv2.cvtColor(raw_256, cv2.COLOR_RGB2BGR)
    raw_out = args.out.replace(".png", "_raw256.png")
    cv2.imwrite(raw_out, raw_256_bgr)
    print(f"Wrote raw 256px model output -> {raw_out}")

    result = undo_resize_square(result, original_size, upsample=args.upsample)
    # Blend back: unmasked region stays exactly the original; masked region
    # takes LaMa's contribution, gated by the SAME strict mask used for inference.
    orig_mask_floored = mask_floor(mask_t)
    final = image_t + (result - image_t) * orig_mask_floored

    final_np = final[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)

    mask_overlay = frame_bgr.copy()
    mask_overlay[mask_gray > 127] = (0, 0, 255)
    grid = np.hstack([frame_bgr, mask_overlay, final_bgr])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Wrote comparison -> {args.out}")
    print(f"\n=== Summary ===")
    print(f"Inference latency (CPU, this machine): {elapsed*1000:.1f}ms for one {w}x{h} frame "
          f"(model runs at {args.required_size}x{args.required_size} regardless of source size)")
    print(f"Extrapolated for a 900-frame clip: {elapsed*900:.1f}s if run serially")


if __name__ == "__main__":
    main()
