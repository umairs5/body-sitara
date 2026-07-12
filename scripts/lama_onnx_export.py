"""
Export big-lama.pt (TorchScript) to ONNX, and verify the export produces
the same output as the TorchScript model on a real frame -- the first
real step toward Android feasibility (Tier 2B-1 background fill).
LaMa's FFC (Fast Fourier Convolution) layers are a known risk for ONNX
export / mobile NN-delegate support (see the plan's honest-gaps list);
this script is how we find out whether that risk is real for THIS
checkpoint, rather than assuming either way.

Usage:
    python scripts/lama_onnx_export.py [--model models/big-lama.pt] [--out models/big-lama.onnx]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from body_sitara.background_fill import _resize_square, _mask_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/big-lama.pt")
    ap.add_argument("--out", default="models/big-lama.onnx")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print(f"Loading TorchScript model from {args.model} ...")
    model = torch.jit.load(args.model, map_location="cpu")
    model.eval()

    dummy_image = torch.rand(1, 3, args.size, args.size)
    dummy_mask = torch.randint(0, 2, (1, 1, args.size, args.size)).float()

    print(f"Attempting ONNX export (opset {args.opset}) ...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        torch.onnx.export(
            model,
            (dummy_image, dummy_mask),
            args.out,
            input_names=["image", "mask"],
            output_names=["output"],
            opset_version=args.opset,
            dynamo=False,
        )
        print(f"Export succeeded -> {args.out}")
    except Exception as e:
        print(f"\n=== EXPORT FAILED ===")
        print(f"{type(e).__name__}: {e}")
        print("\nThis would confirm the FFC-layer mobile-export risk flagged in the plan.")
        return

    # Verify: same frame/mask, same preprocessing, compare TorchScript vs ONNX output
    print("\nVerifying ONNX output matches TorchScript output...")
    import onnxruntime as ort

    export_dir = "tier2_export_root/yoloint8_test"
    cap_v = cv2.VideoCapture(os.path.join(export_dir, "output_rtm.mp4"))
    cap_m = cv2.VideoCapture(os.path.join(export_dir, "mask.mp4"))
    cap_v.set(cv2.CAP_PROP_POS_FRAMES, 90)
    cap_m.set(cv2.CAP_PROP_POS_FRAMES, 90)
    ok_v, frame_bgr = cap_v.read()
    ok_m, mask_frame = cap_m.read()
    cap_v.release()
    cap_m.release()
    if not (ok_v and ok_m):
        print("Could not read test frame; skipping numerical verification.")
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_u8 = cv2.dilate((mask_gray > 127).astype(np.uint8) * 255, kernel)

    image_t = torch.from_numpy(frame_rgb).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)
    mask_t = torch.from_numpy(mask_u8).float().div(255.0).unsqueeze(0).unsqueeze(0)
    work_image, work_mask, _ = _resize_square(image_t, mask_t, args.size)
    work_mask_floored = _mask_floor(work_mask)

    with torch.inference_mode():
        torch.manual_seed(0)
        t0 = time.perf_counter()
        ts_result = model(work_image, work_mask_floored)
        ts_time = time.perf_counter() - t0

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    onnx_result = sess.run(
        None,
        {"image": work_image.numpy(), "mask": work_mask_floored.numpy()},
    )[0]
    onnx_time = time.perf_counter() - t0

    ts_np = ts_result.numpy()
    diff = np.abs(ts_np - onnx_result)
    print(f"\n=== Verification ===")
    print(f"TorchScript time: {ts_time*1000:.1f}ms")
    print(f"ONNX Runtime time: {onnx_time*1000:.1f}ms")
    print(f"Max abs difference: {diff.max():.6f}")
    print(f"Mean abs difference: {diff.mean():.6f}")
    if diff.max() < 0.01:
        print("MATCH: ONNX output is numerically equivalent to TorchScript.")
    else:
        print("WARNING: outputs diverge meaningfully -- investigate before trusting the export.")


if __name__ == "__main__":
    main()
