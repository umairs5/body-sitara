"""
Converts RIFE (IFNet) and big-LaMa to CoreML .mlpackage format, for the
iPhone 15 Pro Max benchmark app.

MUST run on macOS (coremltools' .mlpackage writer -- libcoremlpython /
libmilstoragepython -- is a macOS-only compiled extension; confirmed
2026-07-26 that conversion fails outright on Windows with "No module named
coremltools.libcoremlpython", not just a warning). This script is invoked
by the GitHub Actions macOS runner workflow (.github/workflows/
ios-benchmark.yml), never locally on the Windows dev machine.

RESOLUTION (corrected 2026-07-26): both models benchmarked at 1280x1280,
matching RIFE's fixed pipeline resolution (RifeInterpolator.kt's
REQUIRED_SIZE). The existing models/lama_dilated/lama_dilated-onnx-float/
lama_dilated.onnx export is fixed at 512x512 (baked in at export time via
scripts/lama_onnx_export.py's --size flag, default 256, that particular
artifact exported at 512) -- big-LaMa itself has no fixed-resolution
constraint (confirmed: lama_onnx_export.py's --size is a plain CLI arg), so
this script re-exports directly from models/big-lama.pt at 1280x1280 rather
than converting the existing smaller export, so both models are compared
at the same real, pipeline-relevant resolution.

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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "Models")

RIFE_ONNX = os.path.join(REPO_ROOT, "android", "tier1link", "app", "src", "main", "assets", "rife", "rife_ifnet.onnx")
BIG_LAMA_PT = os.path.join(REPO_ROOT, "models", "big-lama.pt")
LAMA_ONNX_REEXPORTED = os.path.join(OUT_DIR, "big_lama_1280.onnx")

BENCH_SIZE = 1280  # shared resolution for both models -- matches RifeInterpolator.kt's REQUIRED_SIZE


def convert_rife():
    print(f"Converting RIFE ({RIFE_ONNX}) at {BENCH_SIZE}x{BENCH_SIZE}...")
    mlmodel = ct.convert(
        RIFE_ONNX,
        inputs=[
            ct.TensorType(name="img0", shape=(1, 3, BENCH_SIZE, BENCH_SIZE)),
            ct.TensorType(name="img1", shape=(1, 3, BENCH_SIZE, BENCH_SIZE)),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS16,
    )
    out_path = os.path.join(OUT_DIR, "RifeIFNet.mlpackage")
    mlmodel.save(out_path)
    print(f"  saved to {out_path}")


def reexport_lama_onnx():
    """Re-exports big-lama.pt to ONNX at BENCH_SIZE (1280), same procedure
    as scripts/lama_onnx_export.py but with --size matching RIFE's
    resolution instead of that script's default 256/the existing 512
    artifact -- see module docstring for why."""
    print(f"Re-exporting big-lama.pt to ONNX at {BENCH_SIZE}x{BENCH_SIZE}...")
    model = torch.jit.load(BIG_LAMA_PT, map_location="cpu")
    model.eval()

    dummy_image = torch.rand(1, 3, BENCH_SIZE, BENCH_SIZE)
    dummy_mask = torch.randint(0, 2, (1, 1, BENCH_SIZE, BENCH_SIZE)).float()

    torch.onnx.export(
        model,
        (dummy_image, dummy_mask),
        LAMA_ONNX_REEXPORTED,
        input_names=["image", "mask"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print(f"  re-exported to {LAMA_ONNX_REEXPORTED}")


def convert_lama():
    reexport_lama_onnx()
    print(f"Converting big-LaMa ({LAMA_ONNX_REEXPORTED})...")
    mlmodel = ct.convert(
        LAMA_ONNX_REEXPORTED,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, BENCH_SIZE, BENCH_SIZE)),
            ct.TensorType(name="mask", shape=(1, 1, BENCH_SIZE, BENCH_SIZE)),
        ],
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
