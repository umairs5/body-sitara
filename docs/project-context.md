# bodySITARA — Project Context & Research Record

**Last updated:** 2026-06-15  
**Author:** Muhammad Umair, MS AI Year 1, LUMS (25280013@lums.edu.pk)  
**Advisors:** Muhammad Hamad Alizai, Naveed Anwar Bhatti (LUMS CS)  
**Target venue:** IEEE PerCom 2027

---

## 1. Research Goal

Take the original SITARA system (IEEE PerCom 2026) — which anonymizes **faces only** — and extend it to **full-body anonymization**. The core claim to establish: a person's body posture, gait, clothing, and silhouette are as identifying as their face, therefore Tier 1 must protect the whole body, and Tier 2 must replace the whole body with a synthetic character.

The PerCom 2027 paper will need to:
1. Justify why full-body anonymization is necessary (identity leakage beyond the face)
2. Show the upgraded Tier 1 pipeline achieves comparable AP/AR on body detection
3. Show the upgraded Tier 2 (WanVideo) produces synthetic bodies of higher visual quality than the original MobileFaceSwap produced for faces
4. Demonstrate the same privacy guarantees hold under the expanded threat model

---

## 2. Original SITARA (PerCom 2026) — What It Did

**Paper:** "Now You See Me, Now You Don't: Consent-Driven Privacy for Smart Glasses"  
**Hardware:** Raspberry Pi 4B (stand-in for Snapdragon AR1 Gen 1 / Meta Ray-Ban)  
**GitHub:** https://github.com/SYSNET-LUMS/SmartGlassesPrivacy

### Tier 1 (On-device, Glasses)
- **Detector:** YuNet (0.075M params, 0.595G FLOPs) — fastest option on Table II
- **Face mesh:** MediaPipe FaceMesh (468 landmarks)
- **Anonymization:** Convex hull of 468 face landmarks → solid colour fill + AES-128-GCM encryption
- **Skip strategy:** Detector runs every 7 frames; optical flow tracks bboxes in between. Frame-difference MSE triggers early re-detection on scene change.
- **Embedding:** EdgeFace-s-gamma-05 (1.77M params, 0.197G FLOPs, LFW acc 0.9973) — best accuracy/efficiency tradeoff on Table IV
- **Key encryption:** RSA-4096 wraps AES-128 key (note: codebase currently uses RSA-2048 — needs reconciling)
- **Transfer:** Bluetooth to companion phone. Wearer sees blurred video; cannot decrypt.

### Tier 2 (Companion Phone)
- Landmark-driven face swap: 3 critical landmarks (nose tip, mouth corners) → affine via Least-Median → Delaunay triangulation → Poisson blending → MobileFaceSwap refinement
- Synthetic faces from curated pool (not user-uploadable — prevents deepfake abuse)
- Skin tone matching for candidate selection

### Tier 3 (TTP Cloud)
- Cosine similarity threshold: 0.65 (tested on 15-person database)
- TTP is key-server only — never sees video
- Decryption/restoration happens locally on phone

### Key Metrics
| Metric | Value |
|---|---|
| Tier 1 AP (custom dataset) | 0.9421 |
| Tier 1 AR (custom dataset) | 0.9531 |
| vs EgoBlur AP / AR | 0.9354 / 0.9736 (SITARA wins precision, loses recall) |
| FID (Tier 2 vs ground truth) | 63.70 ± 27.78 |
| SSIM | 0.61 ± 0.07 |
| PSNR | 12.85 ± 1.95 dB |
| LPIPS | 0.27 ± 0.07 |
| Landmark distance | 15.94 ± 7.13 |
| Baseline (MobileFaceSwap, unblurred) FID | 31.00 ± 13.83 |
| Energy: full protocol | 112.05 J vs 40.0 J baseline (2.8×) |
| Energy: core (no landmark model) | 67.04 J (1.68×) |
| Processing: 10s clip | 22.88s full / 13.69s core |
| Storage overhead | 23% (14.06 MB per 30-sec clip) |
| Battery impact (full) | ~17.9 min continuous vs 50 min baseline |

### Dataset (Original)
- 16,500 annotated frames, Ray-Ban Meta Stories glasses
- Variations: bystander count (1–5), face size (far/medium/close), motion (static/bystander/wearer)
- 4 human annotators using LabelImg, IoU threshold 0.5 (MS-COCO protocol)
- Also benchmarked on CCV2 dataset

---

## 3. bodySITARA — What We're Building

### What Changes from Original SITARA

| Component | Original | bodySITARA |
|---|---|---|
| Scope | Face only | Full body |
| Detector | YuNet (face) | YOLOX-Nano + RTMPose-T (person + 17 keypoints) |
| Anonymization region | Convex hull of 468 face landmarks | Convex hull of 17 body kpts + 36 face oval pts (codebase) / SAM2 pixel mask (workflow) |
| Face mesh purpose | Define blur boundary | Identity-free canonical expression mesh for WanVideo conditioning |
| Tier 2 method | MobileFaceSwap on phone | WanVideo Wan2.2-Animate-14B on cloud GPU |
| Tier 2 input | Blurred face + face landmarks | 3 signals: blurred body + canonical face mesh + pose skeleton |
| Cloud trust | Tier 2 on trusted companion phone | Tier 2B-2 on untrusted cloud — 3-signal constraint enforces privacy |
| Synthetic output | Synthetic face overlaid on body | Full synthetic character composited over filled background |
| Background | Preserved as-is | Reconstructed (temporal median + LaMa inpainting) |

### What Stays the Same
- Three-tier privacy architecture
- AES-128-GCM + RSA per-person stream encryption
- EdgeFace-s-gamma-05 for identity embedding
- Optical flow skip-frame tracking
- TTP consent protocol (Tier 3 unchanged)

---

## 4. Current Codebase State

### Python Package: `src/body_sitara/`

**`pipeline.py`** — Main orchestrator (`process_video()`)
- YOLOX-Nano + RTMPose-T loaded via `rtmlib.Body` (CPU, ONNX)
- MediaPipe FaceLandmarker for 468-point face mesh (used in canonical mode; used for convex hull in baseline mode)
- Selfie-seg (TFLite) runs in thread pool parallel to det+pose on full frames
- FaceCanonicalizer runs on full frames only, hidden in selfie-seg parallel wait
- Skip frames: body LK only; canonical face reused from last full frame
- 3-panel VideoWriter output: ORIGINAL | BLURRED | EXPRESSION (1920×640) when canonicalizer active
- Benchmark mode (`--no-save`) excludes write overhead (matches paper methodology)
- CSV metrics output for paper tables
- `--anonymizer` flag: `convexhull` | `selfie_seg0` | `selfie_seg1` | `mobilesam` | `yoloseg`

**`blur.py`** — Convex hull anonymization (baseline)
- Builds convex hull from: 17 COCO body keypoints + 36 face oval landmark indices
- Adaptive padding: `shoulder_width × 0.40` (clamped 20–100 px)
- Fallback: bounding rectangle if hull area < 0.5% of frame
- **Known limitation:** hull over-covers gaps (between legs, arm-torso gaps)

**`blur_seg.py`** — SelfieSegBlur (active anonymizer)
- MediaPipe ImageSegmenter (Tasks API), TFLite models
- `selfie_segmenter.tflite` (seg0, general) or `selfie_segmenter_landscape.tflite` (seg1)
- ~130ms on laptop CPU; releases GIL → runs parallel to det+pose in thread pool
- Skip frames: stored mask warped by affine from keypoint motion

**`blur_yoloseg.py`** — YOLOv8-seg-nano instance segmentation
- PT model (47ms) faster than ONNX export (117ms) on CPU
- GIL contention with ORT threads causes slowdown when threaded — fix deferred
- Available via `--anonymizer yoloseg`

**`face_canonical.py`** — FaceCanonicalizer (Tier 2 expression signal)
- Runs FaceLandmarker internally on face crop from RTMPose keypoints
- Renders filled cartoon face onto 512×512 warm-gray canvas:
  - Skin-tone filled face oval
  - White sclera + dark iris + pupil + specular highlight (eye openness visible)
  - Dark eyebrows (7px thick polyline)
  - Rose-pink filled lips (outer + darker inner cavity — mouth open/close visible)
  - Subtle nose bridge + nostril circles
  - GaussianBlur(3,3) for smoothness
- Diffusion-friendly: no identity, no texture, no background — only expression geometry
- ~28ms per full frame, completely hidden in selfie-seg parallel wait

**`pose.py`** — Keypoint utilities
- COCO keypoint index constants (0=nose, 1=left_eye, ... 16=right_ankle)
- `LK_PARAMS` for Lucas-Kanade optical flow
- `derive_face_crop()`, `derive_body_crop()` — extract crops from keypoints
- `get_face_size_tier()`, `get_movement_tier()` — adaptive skip logic

**`tracking.py`** — Per-person state machine
- `PersonState`: holds AES key, best face/body crop (by RTMPose confidence)
- `flush_to_disk()`: encrypts and saves `.packet` files

**`encryption.py`** — Crypto layer
- AES-128-GCM per frame crop
- RSA-2048 wraps AES key (NOTE: paper uses RSA-4096 — needs reconciling)

**`embedding.py`** — Face embedding
- EdgeFace-s-gamma-05 via ONNX Runtime (CPU)
- 112×112 input, (img - 0.5) / 0.5 normalization, L2-normalize output

### Scripts
- `scripts/run.py` — CLI entrypoint, adds `src/` to path, calls `process_video()`

### Configuration Constants (pipeline.py)
```python
BASE_RESOLUTION = 1280.0   # scale factor reference
INFER_SIZE = 320           # resize to this for RTMPose inference
SKIP_N_DEFAULTS = {"slow": 7, "medium": 4, "fast": 1}
FACE_MESH_MIN_CONF = 0.3
```

### Benchmark Numbers (Intel Core Ultra 5 125U, skip=5, --no-save, 6_single_face.mp4 1264×1264)

| Pipeline | FPS | Full-frame wall time |
|---|---|---|
| `convexhull` — det+pose+facemesh+LK | **21.3 FPS** | ~57ms (det+pose bottleneck) |
| `selfie_seg0` + canonical | **13.1 FPS** | ~130ms (selfie-seg bottleneck) |

Note: laptop had 85% RAM usage during tests (Chrome 2.9GB + VS Code 1.2GB). Close both for clean benchmarks.

---

## 5. Tier 2 ComfyUI Workflow

**File:** `Blur Trail - V4.4_workflow-solidfill&fillremoved&rtmpose.json` (Downloads folder)  
**Runs on:** RunPod cloud GPU  
**ComfyUI version:** 1.41.21

### Stage Map

| Stage | Where | What | Custom Node |
|---|---|---|---|
| T1: Detect + Pose | Glasses sim | RTMPoseTinyPoseAndFace (YOLOX-Nano + RTMPose-T, CUDA) | Yes |
| T1: Face canonicalize | Glasses sim | SITARAFaceCanonicalizer → identity-free mesh | Yes |
| T1: Segment | Glasses sim | SAM2.1-hiera-base-plus (video, fp16) | kijai/SAM2 |
| T1: Anonymize | Glasses sim | EmptyImage (grey 0x808080) composited over SAM2 mask | — |
| T1: Pose sticks | Glasses sim | DrawViTPose → pose_images (for WanVideo conditioning) | kijai/WanAnimatePreprocess |
| T2A: Extract blurred part | Mobile | DrawMaskOnImage (black bg) + InvertMask composite | — |
| T2B-1: Fill background | Mobile | SITARABackgroundFill (temporal median 16fr) + LaMa inpaint | Yes |
| T2B-2: Generate character | Cloud | WanVideoAnimateEmbeds + WanVideoSampler (6-step DPM++_SDE) | kijai/WanVideoWrapper |
| T2C: Composite | Mobile | Re-segment character with SAM2 + ImageCompositeMasked | — |

### WanVideo Generation Details
- **Model:** `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors`
- **Resolution:** 832×480, frame_count frames
- **Conditioning:** CLIP-H ref image + pose sticks + canonical face mesh + background + mask
- **Reference character:** `input image (3).webp` (pre-chosen synthetic avatar)
- **LoRA 1:** `WanAnimate_relight_lora_fp16.safetensors` (weight 1.0) — lighting
- **LoRA 2:** `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` (weight 1.2) — consistency
- **Sampler:** DPM++_SDE, 6 steps, CFG=1, shift=5 (distillation enables 6-step quality)
- **Text prompt:** Empty positive, standard WanVideo Chinese negative

### What the Cloud Never Sees
- Original frames
- Real face (only canonical expression mesh)
- Background (cloud derives its own bg from the blurred part via black hole)
- Any personal biometrics beyond skeleton pose

---

## 6. Dataset Plan

### Currently Available
- LUMS face-only dataset: `data/dataset/` (gdown folder download, gitignored)
  - 16,500 frames, face bboxes annotated
  - Useful for testing Tier 1 detection on familiar footage

### Needed for PerCom 2027
- New full-body dataset captured with smart glasses (or equivalent)
- Annotation schema to define: **body bounding box** (not just face), keypoint visibility, occlusion level
- Same variation axes as original: bystander count, body size (far/medium/close), motion state
- Target: ≥10,000 annotated frames minimum for credible AP/AR tables
- Consider: partially occluded bodies (original paper noted this as a gap)

---

## 7. Open Research Questions

These are unresolved decisions that affect both the implementation and the paper framing.

### Privacy & Threat Model
1. **Pose leakage:** Can an adversary re-identify a person from gait/pose skeleton alone? The cloud receives pose sticks — is this sufficient for re-identification? The paper needs to address this or add noise to the skeleton.
2. **Silhouette leakage:** Does the grey-filled body silhouette (sent as `C_blurredpart`) leak enough shape information for re-identification? Body shape + height can be identifying.
3. **RSA key size:** Original paper used RSA-4096, codebase uses RSA-2048. Which do we target for the new paper? 4096 is slower on edge hardware.
4. **Multi-person streams:** Original SITARA's 0.65 cosine similarity threshold was tuned on 15 persons. With full-body, we have body shape as additional signal — does the Tier 3 matching benefit from combining face + body embeddings?

### Technical
5. **SAM2 vs convex hull:** How much does the anonymization coverage improve? Need quantitative comparison on body bboxes (hull can under-cover legs in wide stances, over-cover gaps between limbs). This comparison is a paper contribution.
6. **WanVideo temporal consistency:** The original Tier 2 maintained per-face landmark history for temporal coherence. WanVideo processes the full clip at once — does this eliminate flickering, and how do we measure it? (SSIM across consecutive frames? RAFT optical flow consistency?)
7. **Metrics for body replacement:** SSIM/PSNR/FID on face crops are well-established. Body crops are much larger and more varied — do the same metrics apply, or do we need pose-based metrics (PCK — Percentage of Correct Keypoints on synthetic body vs reference body)?
8. **Runtime on glasses hardware:** SAM2 video mode needs GPU — Raspberry Pi 4B can't run it. What's our target hardware for Tier 1? A phone with NPU? An Android-based glasses platform? This changes the feasibility story.
9. **Frame rate:** Original ran at ~1fps equivalent (7-frame skip on 7fps input). What's our target for bodySITARA, and can WanVideo be batched to avoid per-clip delay?

### Dataset & Evaluation
10. **Annotation schema for body:** Do we annotate full-body bboxes, or just use RTMPose's person bboxes as pseudo-labels? If pseudo-labels, does that bias our AP/AR numbers?
11. **Baseline for body anonymization:** EgoBlur blurs faces only. What is the right comparison system for full-body? There may not be a direct equivalent — this is a research gap worth naming.
12. **Evaluation of Tier 2 identity removal:** How do we formally verify the synthetic body does NOT leak the original person's identity? Face: use face recognizer (ArcFace cosine similarity near 0). Body: use ReID model (Market-1501 trained model)?

### Paper Positioning
13. **Contribution delta from PerCom 2026:** What is the headline novelty? Options: (a) full-body as harder problem with same guarantees; (b) generative upgrade (WanVideo vs MobileFaceSwap) as main contribution; (c) cloud-untrusted architecture (original Tier 2 ran on trusted phone; ours runs on untrusted cloud with 3-signal constraint).
14. **Should we cite bodySITARA as "extension of SITARA" or as an independent system?** Impacts framing of related work section.

---

## 8. Immediate Next Steps (as of 2026-06-15)

**Done:**
- [x] Tier 1 selfie-seg pipeline functional with 3-panel output video
- [x] FaceCanonicalizer implemented (diffusion-friendly filled cartoon face)
- [x] Benchmarked convex hull (21.3 FPS) vs selfie_seg+canonical (13.1 FPS) at skip=5

**Next session (2026-06-16):**
1. Run convex hull baseline on Raspberry Pi 5: `python scripts/run.py <video> --anonymizer convexhull --skip-n 5 --no-save --headless`
2. Run selfie_seg pipeline on RPi 5: same with `--anonymizer selfie_seg0`
3. Compare RPi 5 vs laptop numbers — hardware gap characterization for paper
4. Clean up: make standalone `face_mesh` load conditional on `--anonymizer convexhull`

**Medium term:**
5. Record first batch of full-body test footage (indoor, 1–3 persons, various distances)
6. Run the ComfyUI Tier 2 workflow end-to-end on a short clip to verify outputs
7. Decide SAM2 integration plan for Python codebase (replace `blur_seg.py` or run in parallel for comparison)
8. Reconcile RSA key size (2048 vs 4096) — pick one and be consistent

---

## 9. Key References from Original Paper

- **EgoBlur** [12] — main comparison baseline for Tier 1 blurring
- **YuNet** [20] — original face detector (we replaced with YOLOX-Nano)
- **MediaPipe FaceMesh** [25] — 468 landmarks (we keep this for face mesh)
- **EdgeFace-s-gamma-05** [31] — embedding model (we keep this)
- **MobileFaceSwap** [38] — original Tier 2 (we replace with WanVideo)
- **CCV2 dataset** [39] — secondary evaluation dataset
- **MS-COCO protocol** [24] — AP/AR evaluation at IoU 0.5

For full body: likely new references needed for body ReID evaluation, SAM2, WanVideo, RTMPose.
