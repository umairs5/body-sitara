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
import shutil

import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert

# onnx2torch's Resize converter (as of 1.5.15) is registered for opset
# versions 10/11/13 only -- RIFE's ONNX export uses opset 18 (confirmed via
# onnx.load(...).opset_import), which isn't registered, causing
# NotImplementedError: Converter is not implemented (Resize, version=18)
# during a real CI run (2026-07-26). The opset-13 converter's logic is
# purely attribute-driven (mode/coordinate_transformation_mode/etc, all
# read directly off the ONNX node) with no version-13-specific behavior --
# opset 18's Resize update only added the 'antialias' attribute, which
# that converter doesn't reference at all, so reusing it for opset 18 is
# a safe, minimal patch rather than a real behavior change. Reuses the
# EXACT SAME registered function object (not a reimplementation) by
# looking it up in onnx2torch's own converter registry dict, so there is
# no risk of the patch drifting out of sync with the real opset-13 logic.
import onnx2torch.node_converters.resize  # noqa: F401 (populates the registry)
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
from onnx import defs as _onnx_defs

_resize_v13_key = OperationDescription(domain=_onnx_defs.ONNX_DOMAIN, operation_type="Resize", version=13)
_resize_v18_key = OperationDescription(domain=_onnx_defs.ONNX_DOMAIN, operation_type="Resize", version=18)
if _resize_v18_key not in _CONVERTER_REGISTRY:
    _CONVERTER_REGISTRY[_resize_v18_key] = _CONVERTER_REGISTRY[_resize_v13_key]

# onnx2torch 1.5.15 has NO GridSample converter at all (RIFE's IFNet uses
# it for backward-warping frames by the estimated optical flow -- a
# standard component of virtually every flow-based interpolation network,
# not unusual). ONNX's GridSample op (opset 16, the version RIFE's export
# uses) was explicitly designed to mirror torch.nn.functional.grid_sample
# -- same attribute names (mode/padding_mode/align_corners), same
# semantics (confirmed via the ONNX operator spec) -- so this is a direct,
# low-risk 1:1 mapping, not a reimplementation of unclear behavior.
from onnx2torch.node_converters.registry import add_converter as _add_converter
from onnx2torch.utils.common import OnnxToTorchModule as _OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult as _OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node as _onnx_mapping_from_node
from torch import nn as _nn


class _OnnxGridSample(_nn.Module, _OnnxToTorchModule):
    def __init__(self, mode: str, padding_mode: str, align_corners: bool):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input_tensor, grid):
        return torch.nn.functional.grid_sample(
            input_tensor,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


@_add_converter(operation_type="GridSample", version=16)
def _grid_sample_converter(node, graph):  # noqa: ARG001 (graph unused, matches onnx2torch's own converter signature)
    attrs = node.attributes
    mode = attrs.get("mode", "bilinear")
    padding_mode = attrs.get("padding_mode", "zeros")
    align_corners = bool(attrs.get("align_corners", 0))
    return _OperationConverterResult(
        torch_module=_OnnxGridSample(mode=mode, padding_mode=padding_mode, align_corners=align_corners),
        onnx_mapping=_onnx_mapping_from_node(node),
    )


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(os.path.dirname(__file__), "Models")
SCRATCH_DIR = os.path.join(os.path.dirname(__file__), "_scratch")

RIFE_ONNX = os.path.join(REPO_ROOT, "android", "tier1link", "app", "src", "main", "assets", "rife", "rife_ifnet.onnx")
RIFE_ONNX_DATA = RIFE_ONNX + ".data"
BIG_LAMA_PT = os.path.join(REPO_ROOT, "models", "big-lama.pt")

BENCH_SIZE = 1280  # shared resolution for both models -- matches RifeInterpolator.kt's REQUIRED_SIZE


def convert_rife():
    print(f"Converting RIFE ({RIFE_ONNX}) at {BENCH_SIZE}x{BENCH_SIZE}...")
    print("  Step 1/3: onnx2torch (ONNX -> torch.nn.Module)...")
    # onnx2torch's safe_shape_inference writes a temp file in the SAME
    # directory as the source .onnx (tempfile.NamedTemporaryFile(dir=...))
    # rather than a real temp dir -- fails here because android/.../assets/
    # is a source-tree directory the runner doesn't (and shouldn't) treat
    # as writable scratch space. Fixed by copying the ONNX file (+ its
    # external-data sidecar) into ios_bench/_scratch/ first, which we do
    # have write access to, and converting from there instead.
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    scratch_onnx = os.path.join(SCRATCH_DIR, "rife_ifnet.onnx")
    shutil.copy(RIFE_ONNX, scratch_onnx)
    if os.path.exists(RIFE_ONNX_DATA):
        shutil.copy(RIFE_ONNX_DATA, os.path.join(SCRATCH_DIR, "rife_ifnet.onnx.data"))

    torch_model = onnx2torch_convert(scratch_onnx)
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
