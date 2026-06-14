# bodySITARA — Claude Code Context

## What This Project Is
Extension of SITARA (IEEE PerCom 2026, LUMS) from **face-only** to **full-body** anonymization.
Target: IEEE PerCom 2027 submission. Developer: Muhammad Umair, MS AI Year 1, LUMS.

## Three-Tier Architecture

```
Tier 1 — On-device (Glasses/trusted)
  Detect persons → body pose → face mesh → solid grey fill → encrypt face/body crops → output blurred video

Tier 2 — Companion device + Cloud
  Part A (Mobile): Extract blurred body blob; package 3 signals for cloud
  Part B-1 (Mobile): Fill background (temporal median + LaMa inpaint) — no cloud
  Part B-2 (Cloud, untrusted): WanVideo generates synthetic character from 3 signals only
  Part C (Mobile): Composite character over filled background → FINAL

Tier 3 — TTP (Cloud consent server)
  Cosine similarity match on encrypted embeddings → release AES keys on bystander consent
```

**Privacy invariant:** Cloud (Tier 2B-2) receives ONLY 3 de-identified signals:
1. `C_blurredpart` — grey-filled person silhouette (no background)
2. `facemesh` — identity-free canonical face mesh (expression only)
3. `pose_images` — skeleton stick figures

## Codebase: `src/body_sitara/`

| Module | What it does | Key gap vs workflow |
|---|---|---|
| `pipeline.py` | Main loop: detect → pose → face mesh → blur → encrypt | No SAM2, no background fill, no cloud interface |
| `blur.py` | **Convex hull** of 17 body kpts + 36 face oval pts → solid grey fill | Workflow uses SAM2 pixel mask (more accurate) |
| `pose.py` | Keypoint helpers, LK optical flow params, face/body crop extraction | — |
| `tracking.py` | `PersonState`: per-person AES key, best-crop selection, stream management | — |
| `encryption.py` | AES-128-GCM + RSA-2048 per person stream | Paper used RSA-4096 — needs reconciling |
| `embedding.py` | EdgeFace-s-gamma-05 ONNX, 512-dim L2-normalized face embedding | — |

**Models loaded at runtime:**
- YOLOX-Nano (det) + RTMPose-T (pose) via `rtmlib.Body`, CPU, INFER_SIZE=320
- MediaPipe FaceLandmarker (468 pts) for face mesh
- EdgeFace-s-gamma-05 ONNX for face embedding (optional)

**Skip-frame strategy:** LK optical flow on body kpts + face mesh on skip frames.
`SKIP_N_DEFAULTS = {"slow": 7, "medium": 4, "fast": 1}`, movement-adaptive optional.

## Tier 2 Workflow: ComfyUI (`Blur Trail - V4.4_workflow-*.json`, Downloads folder)

Runs on RunPod GPU. Key models:
- `RTMPoseTinyPoseAndFace` (custom node) — YOLOX-Nano + RTMPose-T on CUDA
- `SITARAFaceCanonicalizer` (custom node) — identity-free canonical face mesh
- `SITARABackgroundFill` (custom node) — temporal median + LaMa (`big-lama.pt`)
- SAM2.1-hiera-base-plus (video mode, fp16) — person segmentation
- WanVideo Wan2.2-Animate-14B fp8 — synthetic character generation
- LoRAs: WanAnimate_relight + lightx2v I2V consistency
- ViTPose-L + YOLOv10m — re-detect generated character in Stage C

Output resolution: input 640×640, WanVideo generates at 832×480, composites back.

## Original SITARA Paper (PerCom 2026) — Key Numbers to Beat/Match

| Metric | Original (face-only) | Our target |
|---|---|---|
| Tier 1 AP | 0.9421 | ≥ 0.94 for body |
| Tier 1 AR | 0.9531 | ≥ 0.95 for body |
| FID (Tier 2) | 63.70 | Lower (WanVideo should improve) |
| SSIM | 0.61 | Higher |
| PSNR | 12.85 dB | Higher |
| Energy overhead | 2.8× baseline | To be measured |
| Storage overhead | 23% | To be measured |

Original Tier 2: MobileFaceSwap on phone (Delaunay triangulation + Poisson blend).
Our Tier 2: WanVideo 14B on cloud — much higher capacity, expect better FID/SSIM.

## Dataset
- Currently downloaded: LUMS face-only dataset (16,500 frames, Ray-Ban Meta Stories)
  - Located: `data/dataset/` (gitignored)
  - Annotations: face bboxes, 4 categories × 3 sub-categories
- Needed: New full-body dataset — to be recorded. Must annotate body bboxes + keypoints.

## File Structure
```
src/body_sitara/    — Python Tier 1 package
scripts/run.py      — CLI entrypoint
data/dataset/       — gitignored, downloaded separately
data/output/        — gitignored, encrypted streams output
models/             — gitignored, ONNX models
requirements.txt    — opencv-python, numpy, mediapipe, rtmlib, onnxruntime, cryptography
```
