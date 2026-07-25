"""
Converts RIFE (IFNet) and big-LaMa to CoreML .mlpackage format, for the
iPhone 15 Pro Max benchmark app.

MUST run on macOS (coremltools' .mlpackage writer -- libcoremlpython /
libmilstoragepython -- is a macOS-only compiled extension; confirmed
2026-07-26 that conversion fails outright on Windows with "No module named
coremltools.libcoremlpython", not just a warning). This script is invoked
by the GitHub Actions macOS runner workflow (.github/workflows/
ios-benchmark.yml), never locally on the Windows dev machine.

CONVERSION PATH (corrected 2026-07-26): coremltools' unified ct.convert()
API dropped direct ONNX support some versions back -- it only accepts
TensorFlow or PyTorch models directly now (confirmed via a real CI failure:
"Unable to determine the type of the model... Please provide source from
[tensorflow, pytorch, milinternal]" -- ONNX isn't even in that list). The
old coremltools.converters.onnx submodule that used to handle this is
unmaintained/deprecated. Fixed by routing ONNX through onnx2torch (ONNX ->
real torch.nn.Module) then torch.jit.trace, landing on coremltools'
actively-supported PyTorch conversion path instead of the abandoned ONNX
one. big-LaMa doesn't need this extra hop -- models/big-lama.pt is ALREADY
a TorchScript module (torch.jit.load works directly), so it goes straight
into ct.convert() without an ONNX round-trip at all.

RESOLUTION: both models benchmarked at 1280x1280, matching RIFE's fixed
pipeline resolution (RifeInterpolator.kt's REQUIRED_SIZE). big-LaMa has no
fixed-resolution constraint of its own (confirmed: scripts/
lama_onnx_export.py's --size is a plain CLI arg, no architectural
constraint) -- traced directly at 1280x1280 here rather than reusing the
existing 512x512 lama_dilated.onnx export, so both models are compared at
the same real, pipeline-relevant resolution.

compute_units=ALL lets CoreML pick ANE/GPU/CPU per-op at runtime -- matches
what the benchmark app needs to inspect via MLComputePlan to confirm actual
ANE usage (see ComputePlanInspector.swift's docstring: same discipline as
the Android RIFE investigation's verbose-logcat check that caught QNN's
silent CPU fallback).

Usage (on the macOS runner):
  python3 convert_models.py
"""
import os

import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "Models")

RIFE_ONNX = os.path.join(REPO_ROOT, "android", "tier1link", "app", "src", "main", "assets", "rife", "rife_ifnet.onnx")
BIG_LAMA_PT = os.path.join(REPO_ROOT, "models", "big-lama.pt")

BENCH_SIZE = 1280  # shared resolution for both models -- matches RifeInterpolator.kt's REQUIRED_SIZE


def convert_rife():
    print(f"Converting RIFE ({RIFE_ONNX}) at {BENCH_SIZE}x{BENCH_SIZE}...")
    print("  Step 1/3: onnx2torch (ONNX -> torch.nn.Module)...")
    torch_model = onnx2torch_convert(RIFE_ONNX)
    torch_model.eval()

    dummy_img0 = torch.rand(1, 3, BENCH_SIZE, BENCH_SIZE)
    dummy_img1 = torch.rand(1, 3, BENCH_SIZE, BENCH_SIZE)

    print("  Step 2/3: torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, (dummy_img0, dummy_img1))

    print("  Step 3/3: coremltools convert...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="img0", shape=(1, 3, BENCH_SIZE, BENCH_SIZE)),
            ct.TensorType(name="img1", shape=(1, 3, BENCH_SIZE, BENCH_SIZE)),
        ],
        outputs=[ct.TensorType(name="interpolated")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )
    out_path = os.path.join(OUT_DIR, "RifeIFNet.mlpackage")
    mlmodel.save(out_path)
    print(f"  saved to {out_path}")


def convert_lama():
    print(f"Converting big-LaMa ({BIG_LAMA_PT}) at {BENCH_SIZE}x{BENCH_SIZE}...")
    print("  Step 1/2: torch.jit.load (already TorchScript, no ONNX hop needed)...")
    model = torch.jit.load(BIG_LAMA_PT, map_location="cpu")
    model.eval()

    dummy_image = torch.rand(1, 3, BENCH_SIZE, BENCH_SIZE)
    dummy_mask = torch.randint(0, 2, (1, 1, BENCH_SIZE, BENCH_SIZE)).float()

    print("  Step 2/2: coremltools convert...")
    mlmodel = ct.convert(
        model,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, BENCH_SIZE, BENCH_SIZE)),
            ct.TensorType(name="mask", shape=(1, 1, BENCH_SIZE, BENCH_SIZE)),
        ],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )
    out_path = os.path.join(OUT_DIR, "BigLama.mlpackage")
    mlmodel.save(out_path)
    print(f"  saved to {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    convert_rife()
    convert_lama()
    print(f"\nBoth models converted at {BENCH_SIZE}x{BENCH_SIZE}.")


if __name__ == "__main__":
    main()
