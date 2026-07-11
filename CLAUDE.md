# bodySITARA — Claude Code Context

## What This Project Is
Extension of SITARA (IEEE PerCom 2026, LUMS) from **face-only** to **full-body** anonymization.
Target: IEEE PerCom 2027 submission. Developer: Muhammad Umair, MS AI Year 1, LUMS.

## Three-Tier Architecture

```
Tier 1 — On-device (Glasses/trusted)
  Detect persons → body pose → selfie-seg grey fill → canonical expression face → encrypt crops → output blurred video

Tier 2 — Companion device + Cloud
  Part A (Mobile): Extract blurred body blob; package 3 signals for cloud
  Part B-1 (Mobile): Fill background (temporal median + LaMa inpaint) — no cloud
  Part B-2 (Cloud, untrusted): WanVideo generates synthetic character from 3 signals only
  Part C (Mobile): Composite character over filled background → FINAL

Tier 3 — TTP (Cloud consent server)
  Cosine similarity match on encrypted embeddings → release AES keys on bystander consent
```

**Privacy invariant:** Cloud (Tier 2B-2) receives ONLY 3 de-identified signals:
1. `C_blurredpart` — selfie-seg grey-filled person silhouette (no background)
2. `canonical_expression_face` — identity-free cartoon face showing expression only (no texture/colour)
3. `pose_images` — skeleton stick figures

## Codebase: `src/body_sitara/`

| Module | What it does | Key gap vs workflow |
|---|---|---|
| `pipeline.py` | Main loop: detect → pose → selfie-seg → canonical face → blur → encrypt | No SAM2, no background fill, no cloud interface |
| `blur.py` | **Convex hull** of 17 body kpts + 36 face oval pts → solid grey fill | Baseline only — selfie_seg is the active anonymizer |
| `blur_seg.py` | **SelfieSegBlur** — MediaPipe ImageSegmenter (TFLite, selfie_seg0/1) → pixel mask | Active anonymizer in current pipeline |
| `blur_yoloseg.py` | **YOLOSegBlur** — YOLOv8-seg-nano instance segmentation (PT model) | Alternative to selfie_seg; GIL contention issue deferred |
| `face_canonical.py` | **FaceCanonicalizer** — FaceLandmarker → filled cartoon face (512×512) | Tier 2 expression signal; runs full frames only |
| `pose.py` | Keypoint helpers, LK optical flow params, face/body crop extraction | — |
| `tracking.py` | `PersonState`: per-person AES key, best-crop selection, stream management | — |
| `encryption.py` | AES-128-GCM + RSA-2048 per person stream | Paper used RSA-4096 — needs reconciling |
| `embedding.py` | EdgeFace-s-gamma-05 ONNX, 512-dim L2-normalized face embedding | — |

**Models loaded at runtime:**
- YOLOX-Nano (det) + RTMPose-T (pose) via `rtmlib.Body`, CPU, INFER_SIZE=320
- MediaPipe FaceLandmarker (468 pts) — loaded always (used for canonical in selfie_seg mode, for convex hull in baseline mode)
- MediaPipe SelfieSegmenter TFLite — `selfie_segmenter.tflite` (selfie_seg0) or `selfie_segmenter_landscape.tflite` (selfie_seg1)
- EdgeFace-s-gamma-05 ONNX for face embedding (optional)

**Skip-frame strategy:**
- Full frames: det → pose → selfie-seg (parallel) → canonical face → blur
- Skip frames: body LK optical flow only; canonical face reused from last full frame
- `SKIP_N_DEFAULTS = {"slow": 7, "medium": 4, "fast": 1}`, movement-adaptive optional
- Canonical runs full-frames-only because it's hidden in selfie-seg's parallel wait at no extra cost

**FaceCanonicalizer design:**
- Internally runs FaceLandmarker (468 pts) on face crop derived from RTMPose keypoints
- Renders filled cartoon face: skin-tone oval fill, white sclera + dark iris/pupil, dark brows (thick polyline), rose lips (filled outer+inner), subtle nose
- Output: 512×512 BGR on warm-gray background — diffusion-model friendly (no identity, no texture)
- Result frozen on skip frames — expression update rate matches full-frame rate (~6 FPS at skip=5)

**3-panel output video (when canonicalizer active):**
- 1920×640: ORIGINAL | BLURRED | EXPRESSION (each 640×640)
- Run: `python scripts/run.py <video> --anonymizer selfie_seg0 --skip-n 5 --headless`

## Benchmark Numbers (Intel Core Ultra 5 125U, skip=5, no-write, 6_single_face.mp4 1264×1264)

| Pipeline | FPS | Full-frame bottleneck |
|---|---|---|
| Convex hull + facemesh (baseline) | 21.3 FPS | det+pose: ~94ms |
| selfie_seg0 + canonical (active) | 13.1 FPS | selfie-seg: ~130ms (parallel) |

Writing overhead costs ~1.4 FPS additional. Paper benchmarks should use `--no-save` (matches SITARA paper methodology of excluding write overhead).

**RPi 5 8GB (bare, no Hailo, measured 2026-07-11):** selfie_seg1 + canonical, skip=1 (no-skip), no-save, `6_single_face.mp4` (Debian 13 trixie aarch64, Python 3.11.9 venv) — **5.28 FPS** (900/900 full-inference frames, 170.6s total). Full-frame bottleneck: SelfieSeg ~156ms (parallel w/ det+pose); Det ~78ms (no NPU/Hailo — pure ARM CPU). Beat the earlier hand-estimate below, which predated any real hardware test.

Superseded pre-hardware estimate (kept for reference only): ~4–6 FPS convex hull, ~2–3 FPS selfie_seg.
Note: laptop benchmarks inflated by 85% RAM usage (Chrome + VS Code). Close both before benchmarking.

## Tier 2 Workflow: ComfyUI (`Blur Trail - V4.4_workflow-*.json`, Downloads folder)

Runs on RunPod GPU. Key models:
- `RTMPoseTinyPoseAndFace` (custom node) — YOLOX-Nano + RTMPose-T on CUDA
- `SITARAFaceCanonicalizer` (custom node) — identity-free canonical face mesh (node 192, widget_values=[512, 0.7, 0.45])
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

## Reference Paper
Original SITARA paper: `paper/PerCom 2026.pdf` (in this repo)
Full notes extracted from it are in `docs/project-context.md` Section 2.

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

## Pending Cleanup
- `face_mesh` standalone FaceLandmarker instance (pipeline.py [2/4]) still loaded but never called in selfie_seg mode — make it conditional on `anonymizer == "convexhull"`
- yoloseg GIL contention: PyTorch CPU holds GIL causing ORT slowdown when threaded; fix deferred
