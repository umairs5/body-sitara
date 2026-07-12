"""
Attempt to export big-lama.pt to ExecuTorch's .pte format, as an
alternative to ONNX (which failed: aten::fft_rfftn has no ONNX opset
support at any version available in this toolchain -- see
lama_onnx_export.py). ExecuTorch runs actual PyTorch ops on-device rather
than converting to a different graph representation, so it's a real,
different question whether FFT is supported here.

Usage:
    python scripts/lama_executorch_export.py [--model models/big-lama.pt]
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
    ap.add_argument("--out", default="models/big-lama.pte")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    print(f"Loading TorchScript model from {args.model} ...")
    ts_model = torch.jit.load(args.model, map_location="cpu")
    ts_model.eval()

    dummy_image = torch.rand(1, 3, args.size, args.size)
    dummy_mask = torch.randint(0, 2, (1, 1, args.size, args.size)).float()

    print("Attempting TS2EPConverter (TorchScript -> ExportedProgram)...")
    try:
        # torch.export.export() rejects ScriptModules directly ("Exporting a
        # ScriptModule is not supported" -- confirmed by direct test) but
        # points at this converter as the real path for a pre-scripted graph
        # like ours (torch.jit.load'd, not a plain nn.Module we authored).
        from torch._export.converter import TS2EPConverter
        exported = TS2EPConverter(ts_model, (dummy_image, dummy_mask)).convert()
        print("TS2EPConverter succeeded.")
    except Exception as e:
        print(f"\n=== TS2EPConverter FAILED ===")
        print(f"{type(e).__name__}: {e}")
        return

    print("\nRunning to_edge()...")
    try:
        from executorch.exir import to_edge, EdgeCompileConfig
        # aten.view_as_real (complex-tensor reinterpretation, part of the FFT
        # machinery) isn't in ExecuTorch's Core ATen opset -- its own error
        # message names this exact exception-list workaround. Whether the
        # eventual Android backend (XNNPACK etc.) actually HAS a kernel for
        # it is a separate, later question this alone doesn't answer.
        # Growing exception list -- each retry so far has surfaced exactly
        # one more non-Core-ATen op from the FFT decomposition chain
        # (view_as_real, then complex, ...). Building the full list
        # iteratively rather than guessing it upfront.
        exception_ops = [
            torch.ops.aten.view_as_real.default,
            torch.ops.aten.complex.default,
        ]
        edge_program = to_edge(
            exported,
            compile_config=EdgeCompileConfig(_core_aten_ops_exception_list=exception_ops),
        )
        print("to_edge() succeeded (with view_as_real exception).")
    except Exception as e:
        print(f"\n=== to_edge() FAILED ===")
        print(f"{type(e).__name__}: {e}")
        return

    print("\nRunning to_executorch()...")
    try:
        executorch_program = edge_program.to_executorch()
        print("to_executorch() succeeded.")
    except Exception as e:
        print(f"\n=== to_executorch() FAILED ===")
        print(f"{type(e).__name__}: {e}")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"\nExport succeeded -> {args.out}")

    # Verify against TorchScript on a real frame
    print("\nVerifying .pte output matches TorchScript output on a real frame...")
    from executorch.runtime import Runtime

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
        ts_result = ts_model(work_image, work_mask_floored)
        ts_time = time.perf_counter() - t0

    runtime = Runtime.get()
    program = runtime.load_program(args.out)
    method = program.load_method("forward")
    t0 = time.perf_counter()
    et_result = method.execute([work_image, work_mask_floored])[0]
    et_time = time.perf_counter() - t0

    diff = (ts_result - et_result).abs()
    print(f"\n=== Verification ===")
    print(f"TorchScript time: {ts_time*1000:.1f}ms")
    print(f"ExecuTorch time: {et_time*1000:.1f}ms")
    print(f"Max abs difference: {diff.max().item():.6f}")
    print(f"Mean abs difference: {diff.mean().item():.6f}")


if __name__ == "__main__":
    main()
