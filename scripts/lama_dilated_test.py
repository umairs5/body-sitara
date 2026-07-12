"""
Test qualcomm/LaMa-Dilated (ONNX, dilated-convolution variant of LaMa --
no FFT/FFC layers, so it exports cleanly to ONNX/TFLite/QNN, unlike
big-lama which is blocked on both ONNX and ExecuTorch -- see
lama_onnx_export.py and lama_executorch_export.py) on a real frame, same
preprocessing (mask grown +10px) validated for big-lama this session.

Model: fixed 512x512 I/O (not 256x256 like big-lama's native resolution)
per models/lama_dilated/lama_dilated-onnx-float/metadata.json.

Usage:
    python scripts/lama_dilated_test.py [--frame N]
"""
import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="tier2_export_root/yoloint8_test")
    ap.add_argument("--frame", type=int, default=90)
    ap.add_argument("--model", default="models/lama_dilated/lama_dilated-onnx-float/lama_dilated.onnx")
    ap.add_argument("--grow-mask", type=int, default=10)
    ap.add_argument("--out", default="scratch/lama_dilated_result.png")
    args = ap.parse_args()

    print(f"Loading ONNX model from {args.model} ...")
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        print(f"  input: {inp.name} {inp.shape} {inp.type}")
    for out in sess.get_outputs():
        print(f"  output: {out.name} {out.shape} {out.type}")
    required_size = sess.get_inputs()[0].shape[-1]

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
        mask_gray = cv2.dilate((mask_gray > 127).astype(np.uint8) * 255, kernel)

    print(f"Frame {args.frame}: {frame_bgr.shape}, mask covers {(mask_gray > 127).mean()*100:.1f}% of pixels")

    # Resize (not pad -- this model has no documented pad/undo-pad step;
    # simple square resize, matching its fixed-size I/O contract) to the
    # model's required square resolution.
    frame_resized = cv2.resize(frame_rgb, (required_size, required_size), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask_gray, (required_size, required_size), interpolation=cv2.INTER_NEAREST)

    image_np = frame_resized.astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))[np.newaxis, ...]
    mask_np = (mask_resized > 127).astype(np.float32)[np.newaxis, np.newaxis, ...]

    print("Running inference...")
    t0 = time.perf_counter()
    result = sess.run(None, {"image": image_np, "mask": mask_np})[0]
    elapsed = time.perf_counter() - t0
    print(f"  Inference time: {elapsed*1000:.1f}ms")

    result_np = result[0].transpose(1, 2, 0)
    result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
    result_full = cv2.resize(result_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
    result_bgr = cv2.cvtColor(result_full, cv2.COLOR_RGB2BGR)

    # Blend back against the original at full res, same principle as
    # big-lama's validated pipeline: unmasked region stays exactly original.
    orig_mask_bool = mask_gray > 127
    final = frame_bgr.copy()
    final[orig_mask_bool] = result_bgr[orig_mask_bool]

    mask_overlay = frame_bgr.copy()
    mask_overlay[orig_mask_bool] = (0, 0, 255)
    grid = np.hstack([frame_bgr, mask_overlay, final])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"Wrote comparison -> {args.out}")
    print(f"\nExtrapolated for a 900-frame clip: {elapsed*900:.1f}s if run serially")


if __name__ == "__main__":
    main()
