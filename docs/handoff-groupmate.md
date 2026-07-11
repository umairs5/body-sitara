# bodySITARA — Project Handoff (as of 2026-07-12)

This document exists so you (and your Claude Code instance) can get up to speed on this
project without re-deriving everything from scratch. It explains what the project is,
what's been built and verified so far, what's still open, and specifically what Phase 2
(the piece you're likely picking up, since you're also running a RunPod pod) needs.

Read this top to bottom once, then treat it as a reference. If anything here turns out to
be stale or wrong once you actually look at the code, trust the code — this doc is a
snapshot, not a live source of truth.

---

## 1. What this project actually is

bodySITARA extends **SITARA** (a prior paper, PerCom 2026, LUMS) from face-only anonymization
to full-body anonymization, targeting a **PerCom 2027** submission. SITARA's original design:
smart glasses detect a face, blur it locally, and reconstruct a synthetic
identity-preserving-but-anonymized face on a paired phone, with a third-party "TTP" server
mediating bystander consent for un-blurring. bodySITARA does the same thing but for whole
bodies, and moves the reconstruction step to a cloud GPU (RunPod) instead of doing it on-phone,
because full-body synthesis needs much more compute than the original face-only approach.

**The original SITARA paper is in the repo**: `paper/PerCom 2026.pdf`. Read it — the whole
system design here is grounded in its threat model and Tier 3 protocol, not invented from
scratch.

### The three tiers

```
Tier 1 — On-device (Raspberry Pi 5, plays the role of "the glasses")
  Detect people → body pose → segmentation mask → canonical (identity-free) expression face
  → encrypt raw crops per person → export a bundle of de-identified signals + one encrypted
  "consent-restoration" bundle

Tier 2 — Companion phone (Samsung Galaxy S25 Ultra) + Cloud (RunPod GPU)
  2A  (phone):  pull the Tier 1 bundle, extract the de-identified signals
  2B-1 (phone): fill in the background where the (blurred) person was, using LaMa inpainting
  2B-2 (cloud, UNTRUSTED): generate a synthetic person from ONLY 3-4 de-identified signals
                using WanVideo (a 14B-param diffusion video model) — this is what your
                RunPod pod is for
  2C  (phone):  composite the cloud-generated synthetic person back onto the phone's own
                locally-filled background → final anonymized video

Tier 3 — TTP (a small cloud key-server, not yet built)
  Matches an encrypted face embedding against a registered bystander database, and if the
  bystander consents, releases the AES key needed to decrypt that one person's original
  (real) crop — for accountability/moderation purposes, not for everyday use.
```

### The critical privacy invariant

**RunPod (Tier 2B-2) must never receive raw pixels, real faces, real fine-grained background,
or audio.** It only ever receives:
1. A skeleton/pose video (rendered from keypoints, not raw video)
2. A canonical, identity-free cartoon face video (expression only, no real facial texture/color)
3. A blockified (heavily pixelated) body silhouette mask
4. A **heavily degraded** ("32×32 downsample → blur → upsample") background color/lighting
   plate — this one is a deliberate, small, explicitly-acknowledged exception to "zero real
   pixel content," see §7 below
5. An `avatar_id` selecting a pre-staged, non-real stand-in identity from a fixed library —
   the cloud never sees the real person's appearance at all, it generates a *different*,
   pre-chosen-looking person doing the same motions/expression

This is because bodySITARA introduces a **new adversary** the original SITARA paper never had
to deal with: an **untrusted cloud operator** (RunPod). The original SITARA's Tier 2 ran
entirely on the trusted phone. Every design decision about what crosses the phone→cloud
boundary exists to defend against a "honest-but-curious" or fully compromised cloud provider.

---

## 2. Where the full design is written down

**`C:\Users\Muhammad Umair\.claude\plans\peppy-frolicking-hinton.md`** (also published as a
Claude Artifact — ask Umair for the link if you want the rendered version) is the full system
architecture design document: every protocol, every data format at every tier boundary, a
full security analysis against 4 threat-model adversaries, tech stack per tier, and a phased
roadmap. **Read section 2.2 and section 3.4 carefully** — those are the ones most relevant to
what you'll be doing (the RunPod/cloud-generation boundary and its privacy analysis).

This doc (the one you're reading now) summarizes what's actually been *built and tested*
against that plan. The plan doc is the design; this doc is the "as-built" status report.

---

## 3. Codebase layout

```
src/body_sitara/       — Tier 1 CV pipeline (Python). Runs today, CPU-only, tested on both
                          a Windows dev machine and a real Raspberry Pi 5.
src/tier1_link/         — NEW this phase: the Pi↔Phone networking layer (FastAPI server,
                          TOFU-pinned HTTPS, mDNS advertisement). Separate from body_sitara
                          because it's networking, not CV.
android/tier1link/      — NEW this phase: a minimal Android app (Kotlin) that pulls a clip
                          bundle from the Tier 1 link server. This is NOT the real Tier 2 app
                          yet — it's a proof-of-concept for the discovery/pull/verify protocol.
scripts/                — CLI entrypoints and one-off test/comparison scripts.
tier2-workflow/          — ComfyUI workflow JSON files (see §5, this is your area).
data/dataset/            — gitignored, LUMS face-only benchmark dataset (16,500 frames).
models/                  — gitignored, downloaded/exported ONNX and .pt model weights.
paper/PerCom 2026.pdf    — the original SITARA paper. Read it.
docs/project-context.md  — older project notes, written before this session's work.
CLAUDE.md                — project instructions/context for Claude Code sessions. Also worth
                          reading — has benchmark numbers, module descriptions, file structure.
```

Two active git branches you'll see: `master` (stable, PR-merged) and `experiment/selfie-seg`
(the working branch everything gets developed on, then PR'd into master). As of this doc,
they're in sync.

---

## 4. What's built and verified — Tier 1 (Raspberry Pi)

**Status: working, tested on real Pi 5 hardware.**

- Full CV pipeline: person detection (YOLOX-Nano) + pose (RTMPose-T) via `rtmlib`, running on
  CPU via ONNX Runtime.
- Segmentation: MediaPipe SelfieSegmentation (the shipped default, `selfie_seg1`) — real Pi 5
  FPS measured at **5.28 FPS**, no-skip, on the standard benchmark clip.
- Alternative segmentation backends were built and benchmarked this session (YOLOv8n-seg,
  YOLO11n-seg, both FP32 ONNX and INT8-quantized) — YOLO gives cleaner per-person masks in some
  multi-person cases but is meaningfully slower even in its fastest (INT8) form (3.68 FPS vs.
  selfie_seg's 5.28 FPS on the same Pi). **selfie_seg1 remains the shipped default.** See
  `scripts/compare_segmentation.py` and `scripts/compare_yolo_precision.py` if you want the
  full comparison data — it's a real, measured tradeoff, not a guess.
- **Canonical (identity-free) face signal**: `src/body_sitara/face_canonical_v2.py` — extracts
  12 normalized scalars (mouth open/width, smile, eye open, brow, gaze, head pose) from a real
  face crop, then renders a cartoon face from those scalars. No real facial texture/color ever
  appears in the output. This is the actual expression signal that goes to the cloud (as a
  *video render* of the cartoon face, not the raw scalars — see §5).
- **Dense per-person export mode** (`process_video(..., export_dir=...)` in `pipeline.py`):
  processes a clip and writes out a directory bundle per clip:
  - `manifest.json` — clip_id, fps, resolution, per-slot stream_id + smile baseline
  - `keypoints_p{i}.npy` — `[N,17,3]` float32 COCO-17 keypoints per frame
  - `bboxes_p{i}.json` — per-frame bounding boxes
  - `face_params_p{i}.npy` — `[N,12]` float32, the canonical face's parametric scalars
  - `output_rtm.mp4` — the full anonymized (grey-filled) video
  - `mask.mp4` — the final segmentation mask video
  
  This bundle uses a **fixed number of person "slots"** (default 3), not one-slot-per-detected-
  person — because the downstream ComfyUI graph needs a stable, known number of person-channels
  to build its compute graph against. A slot tracker (`export_tracking.py`) keeps left-to-right
  screen-position slot assignment stable across frames even through brief occlusion. If a clip
  has fewer real people than `--export-people`, the extra slots are all-zero/empty — this is
  expected, not a bug.
- **A real, confirmed-fixed privacy bug**: earlier in this project, `face_crops_p{i}.mp4` (a
  raw, UNBLURRED face crop) was accidentally part of the export bundle — verified by direct
  code read, fixed by wiring the canonical face signal into the export path instead and
  demoting the raw crop video to a diagnostics-only file (`--export-diagnostics`, off by
  default) that must never be treated as safe to transmit.
- Encryption: AES-128-GCM per person-stream, RSA-4096 to wrap the AES key (matches the
  original paper; was RSA-2048 before this session, fixed). The "TTP" keypair generation in
  `pipeline.py` today is a **local simulation for benchmarking only** — in the real system,
  the TTP (Tier 3, not yet built) owns its own keypair and the Pi only ever fetches its public
  key.

---

## 5. What's built and verified — Tier 1 ↔ Tier 2 link (Pi/phone networking)

**Status: fully working, tested end-to-end on real hardware, currently running on a Windows
machine standing in for the Pi (which was powered off partway through this work).**

- **`src/tier1_link/server.py`** — a FastAPI server implementing the Pi→phone pull protocol:
  - `GET /clips` — list available clip bundles
  - `GET /clips/{id}/manifest` — manifest + per-file SHA-256
  - `GET /clips/{id}/files/{name}` — download a file, Range-request aware (chunked/resumable)
  - `POST /clips/{id}/ack` — phone confirms receipt
  
  Directory layout: `<export_root>/<clip_id>/manifest.json` + the files it lists — this is
  literally what `pipeline.py --export-dir` already produces, just one level up so multiple
  clips can coexist.

- **`src/tier1_link/cert.py`** — self-signed ECDSA P-256 cert generation for TOFU
  (Trust-On-First-Use) HTTPS. The server prints its cert's SHA-256 fingerprint to console on
  startup. There's no real CA for a benchtop Pi, so trust is established by the phone
  remembering ("pinning") this fingerprint the first time it connects, and refusing to talk to
  a server presenting a *different* fingerprint later. This is a deliberate simplification of
  what a real product would do with a manufacturer-signed root key.

- **`src/tier1_link/discovery.py`** — mDNS advertisement via the `zeroconf` library, service
  type `_bodysitara._tcp.local.`. Confirmed working standalone (a separate Python browser
  process finds it correctly).

- **`android/tier1link/`** — a minimal Android app (Kotlin, OkHttp, coroutines) implementing
  the phone side of all of the above: `Tier1LinkClient.kt` (the protocol client, with 3 TLS
  modes — plain HTTP, TOFU-pinned, and first-pairing "capture" mode), `TofuTrustManager.kt`
  (the custom X509TrustManager that does fingerprint-only validation instead of normal CA-chain
  validation), `Tier1Discovery.kt` (NsdManager-based mDNS client), `MainActivity.kt` (a bare
  test UI — list clips, discover, pull, view a log).

  **This has been run on a real Galaxy A31 phone over real WiFi against the Windows-hosted
  server**, and the following is genuinely verified, not assumed:
  - Full discover→list→manifest→download-all-files→sha256-verify→ack cycle, including an
    11-file, ~85MB bundle (video included)
  - TOFU pairing: the fingerprint captured by the phone on first connect was confirmed to
    exactly match the fingerprint independently printed by the server's own console
  - A real bug was found and fixed during this testing: the Android ack call used a deprecated
    OkHttp API that silently threw at runtime — files downloaded fine but the ack never
    reached the server. Fixed; verified server-side (`GET /clips` correctly showed
    `"status": "acknowledged"` afterward) not just by trusting the app's own log output.
  - mDNS: confirmed **not working end-to-end** on this specific network (LUMS WiFi) — verified
    via Logcat diagnostics that the Android-side code is implemented correctly (multicast lock
    acquired, discovery starts, just zero `onServiceFound` callbacks ever arrive), and that
    Windows Firewall already allows mDNS traffic both directions. The most likely cause is
    router/AP-level client isolation, a common and usually non-optional policy on institutional
    WiFi. This should work fine on an unrestricted network or on the real Pi-hosted AP (which
    we'll fully control). **mDNS is convenience-only — no security property depends on it —
    so this is an acceptable known gap, not a blocker.**

**What's NOT done yet in this part**: the real Pi hasn't run any of this yet (it was powered
off mid-session) — everything above was verified using a Windows machine serving the exact
same code/protocol as a stand-in. Swapping in the real Pi should be a drop-in change (same
server code, same directory layout), but hasn't been physically tested. QR-code-based pairing
(vs. the current manual/capture-mode fingerprint entry) also hasn't been built — not needed for
the paper's threat-model claims, just a demo-polish nicety.

---

## 6. What's NOT built yet

- **Tier 2A/2B-1/2C real implementation** — the Android app that exists (`android/tier1link/`)
  only proves the *transport* protocol works. It does not yet render pose/face signal videos,
  do mask blockification, compute the lighting plate, run LaMa background inpainting, upload
  to RunPod, or composite the final result. This is real, substantial work still ahead.
- **Tier 3 (TTP server)** — doesn't exist at all yet. Not urgent; fully decoupled from Tier 2
  work per the roadmap.
- **Pi-as-WiFi-AP setup** (hostapd config) — not done; currently everything's tested over
  shared home/existing WiFi, not the Pi hosting its own access point as the final design calls
  for.
- **Camera capture on the Pi** — the pipeline today is file-in/file-out only; no live camera
  integration exists.
- **Audio path** — doesn't exist anywhere in the Tier 1 stack yet.

---

## 7. Two design corrections made this session (read carefully — differs from the plan doc's original wording in places, though the doc has since been updated to match)

1. **No temporal-median background fill.** The original plan had Tier 2B-1 do temporal-median
   (across all frames of a clip, take the per-pixel median to reconstruct background where the
   person occluded it in some frames) as a first pass, with LaMa inpainting only patching the
   small leftover gaps median couldn't fill. **This has been simplified**: LaMa now inpaints
   the *entire* person-shaped hole directly, per frame, with no median pre-pass. Simpler
   pipeline, but a real tradeoff to be aware of: LaMa is now doing meaningfully more per-frame
   work than originally planned, and because there's no temporal-median baseline anymore, there
   is currently *no* cross-frame consistency mechanism for the filled background — independent
   per-frame inpainting can plausibly produce visible flicker in the background across the
   clip. **This hasn't been evaluated yet** — worth an early visual check once 2B-1 is built,
   not just assumed to be quality-neutral.

2. **Background fill runs concurrently with cloud generation, not before it.** 2B-1 (phone-
   side LaMa background fill) and 2B-2 (cloud generation on RunPod) are independent of each
   other — the cloud only needs the cheap, heavily-degraded "lighting plate" (a 32×32
   downsample of the background), not the fully-filled result. So: as soon as the phone has
   packaged the upload signals, kick off the RunPod upload/generation request **and** start the
   local LaMa fill **at the same time**, not sequentially. 2C (final compositing) waits on
   both to finish. This directly reduces end-to-end latency since the two don't actually depend
   on each other.

---

## 8. What Phase 2 (your likely area, since you're running RunPod) actually needs

This is the next unstarted phase. Read plan doc §1.3, §2.2, and §6 (honest gaps) in full before
starting — they cover this in more detail than this summary.

### The existing ComfyUI workflow

`tier2-workflow/V5-workflow-master-multiperson.json` is the current, authoritative multi-person
ComfyUI graph — 236 nodes, built by hand in ComfyUI's visual editor. It already does real
generation work (WanVideo, 3 person-slots, `BlockifyMask` active, a real lighting-plate
`bg_images` input) when run inside ComfyUI's own UI. There's also an older, single-person,
no-blockify file (`Blur_Trail_-_V4_4_workflow-b1_active_no_blockify.json`) — **V5 is the one to
use**, V4.4 is superseded/kept for reference only. A PDF explanation of the workflow also
exists in that folder.

**Important**: this JSON is a UI-editor export (`nodes`/`links`/`groups` structure, no
`class_type` fields), NOT the "API format" ComfyUI needs to accept programmatic requests via
its `/prompt` endpoint. It also contains a large number of `GetNode`/`SetNode` pairs (60/29) —
ComfyUI's variable-aliasing mechanism for avoiding long visual connection lines in a big graph
— which makes naive JSON-level editing risky in general.

### Do NOT prune the graph's simulation branches

An earlier draft of this doc suggested deleting the nodes that simulate Tier 1's output
(`RTMPoseTinyPoseAndFace`, `Sam2Segmentation`, `INPAINT_LoadInpaintModel`, etc.) from the graph.
**That's wrong — don't do that.** Keep V5's structure fully intact, exactly as it is. The
reason: your first job isn't to build the final pruned production graph, it's to **replicate
this project's actual Tier 1 pipeline in your own ComfyUI setup and confirm your RunPod
environment produces correct results** — i.e. verify your pod, your model weights, your node
versions all actually work end-to-end, using the graph as it already exists and is already
proven to work in Umair's ComfyUI. Deleting nodes before you've even confirmed your own setup
reproduces the known-good result would make it much harder to tell "my pod is misconfigured"
apart from "I broke the graph by editing it."

### What to actually do instead: replicate real Tier 1, not ComfyUI's simulated version

The graph's simulation nodes (`RTMPoseTinyPoseAndFace`, `Sam2Segmentation`, etc.) exist to
approximate Tier 1's output *inside ComfyUI* when you don't have real Pi/pipeline data handy.
But this repo's actual Tier 1 pipeline is real, working code (`src/body_sitara/`, see §4 above)
— use **that**, exactly as it exists in this repo, not ComfyUI's approximation of it, as your
input source. Concretely:

1. Run this repo's real pipeline on a test clip to produce a real dense-export bundle (see the
   command in §8's data-source note below). **Use `--anonymizer yoloseg11int8`** specifically
   (YOLO11n-seg, INT8-quantized ONNX — one of the alternative segmentation backends built and
   benchmarked this session, see §4) rather than the shipped default `selfie_seg1`. This
   backend auto-builds itself on first use (auto-exports ONNX from the auto-downloaded `.pt`
   checkpoint, then auto-quantizes to INT8) — see `src/body_sitara/blur_yoloseg.py` if you want
   the details, you don't need to pre-stage any model files yourself.
2. Feed the real files this produces (`keypoints_p{i}.npy`, `face_params_p{i}.npy`, `mask.mp4`,
   `output_rtm.mp4`) into your ComfyUI setup as the actual source data, in place of whatever
   ComfyUI's own simulation nodes would have derived from a raw test video.
3. Confirm your RunPod pod, when driven by this *real* Tier 1 data, produces a correct,
   sensible generated result — that's the actual validation goal for this step, not graph
   pruning.

Graph pruning (removing the now-genuinely-unnecessary simulation nodes, building the real
pruned API-format production graph) is real, valuable future work, but it comes **after** this
replication step confirms everything upstream actually works — don't jump ahead to it.

### A known bug, for later (not your first priority)

WanVideo currently runs full generation for **all 3 person-slots unconditionally**, even when
a slot has no real detected person in that clip — `SITARAPersonPresent` only gates
*compositing*, not generation itself. This wastes real GPU cycles. Worth fixing once you get to
building the real pruned API-format graph/gateway (later, not part of the replication step
above) — the gateway should construct the per-job prompt dynamically, omitting the WanVideo
generation subgraph entirely for slots with no detections, not just gate the output after the
fact.

### The gateway you'll eventually build (after replication is confirmed working)

A thin FastAPI process on the RunPod pod, sitting in front of ComfyUI's own native API
(`/prompt`, `/history/{id}`, `/view`, and its `/ws` progress socket) — the phone should never
speak ComfyUI's own queue protocol directly, just this small, auditable wrapper. Per the plan
doc §2.2:

| Endpoint | Purpose |
|---|---|
| `POST /v1/jobs` | Multipart upload of the signal bundle + params → `{job_id, job_token}` |
| `GET /v1/jobs/{id}/status` | Poll `{stage, percent}` |
| `GET /v1/jobs/{id}/result/{slot}` | Streams the composited result once ready |

Auth: a bearer API key + short-lived signed per-job token (exact scheme still open — this
wasn't built yet, use your judgement or ask Umair).

### Suggested build order

1. **Replicate real Tier 1 → your ComfyUI setup, using the graph exactly as it exists (no
   pruning)** — generate a real dense-export bundle with this repo's actual pipeline (command
   below, note the `yoloseg11int8` anonymizer), feed those real files into your ComfyUI setup
   in place of what the graph's simulation nodes would otherwise derive, and confirm your
   RunPod pod produces a correct, sensible generated result end-to-end. This is your actual
   first goal — validating your own environment against known-good real data, not graph editing.
2. Once you've confirmed your pod reproduces correct results from real Tier 1 data, *then* look
   at pruning the now-redundant simulation nodes and exporting a real API-format graph (ComfyUI
   "Save (API format)", from inside the UI, not by hand-editing the JSON — see the `GetNode`/
   `SetNode` aliasing caveat above).
3. Build the FastAPI gateway wrapping the pruned graph.
4. Only after that, build the real phone-side signal-rendering code (pose video render, face
   video render from `face_params_p{i}.npy`, mask blockify, lighting-plate reduction) and wire
   it to actually call your gateway.

### Generating a real Tier 1 bundle for step 1

```bash
python scripts/run.py <any_test_clip.mp4> --anonymizer yoloseg11int8 --skip-n 1 \
    --headless --export-dir tier2_export_root/<some_clip_id>
```

Use **`--anonymizer yoloseg11int8`** specifically (YOLO11n-seg, INT8-quantized ONNX) — not the
shipped default `selfie_seg1` — per Umair's instruction for this replication step. This
anonymizer auto-builds itself on first use (exports ONNX from the auto-downloaded `.pt`
checkpoint, then quantizes to INT8) — no manual model staging needed, just let the first run
take a bit longer while it builds these once.

(`--export-dir` also forces `dense_export=True` and no-skip automatically — see
`pipeline.py`'s export-forcing logic if you want the details. As of this doc, the export-dir
gate was widened to allow any `yoloseg*` anonymizer through, not just `selfie_seg*` — if you
see a `NOTE: export mode forces anonymizer='selfie_seg1'` message when using `yoloseg11int8`,
your checkout predates that fix; `git pull` should resolve it.)

This repo doesn't currently ship a pre-generated bundle in git (`tier2_export_root/` is
gitignored, generated output) — you'll need to run the command above yourself once you have a
test clip.

---

## 9. Honest open risks (worth knowing before you build on top of this)

Per the plan doc §3.4 and §6 — these are real, currently-unmitigated privacy gaps, not solved
problems being glossed over:

- **Gait-from-skeleton re-identification**: the pose signal sent to the cloud is precise,
  real-valued joint positions (needed for WanVideo to animate convincingly) — no noise
  injection or gait-normalization exists. Skeleton-based gait re-identification from 2D
  keypoints is a real, demonstrated attack in the literature. Unmitigated.
- **Body-shape-from-silhouette re-identification**: the blockify mask reduction genuinely
  destroys fine contour/texture detail, but coarse skeletal proportions (height, shoulder
  width) likely still survive even at 16px block granularity. Unmitigated.
- **The lighting plate** is a real, if minor, exception to "zero real pixel content crosses the
  boundary" — 32×32-effective-resolution background color/lighting, almost certainly harmless
  as an identification signal but should be named honestly in the paper, not implied away.
- **Background flicker risk** from dropping the temporal-median pre-pass (§7 above) — not yet
  evaluated.
- **LaMa-on-mobile-NPU feasibility** is unverified — this is Phase 3's problem (on-device
  compositor), not yours directly, but worth knowing it's an open question if it affects your
  timeline expectations for when a full end-to-end demo is realistic.

These are exactly the kind of thing a PerCom reviewer will ask about — better to know about
them now than be surprised later.

---

## 10. Quick orientation commands

```bash
# See the real Tier 1 pipeline run on a test clip
python scripts/run.py <clip.mp4> --anonymizer selfie_seg1 --skip-n 1 --headless --no-save

# Generate a real dense-export bundle (the Tier 1 -> Tier 2 handoff format)
python scripts/run.py <clip.mp4> --anonymizer selfie_seg1 --skip-n 1 --headless \
    --export-dir tier2_export_root/test_clip_1

# Start the Tier1 link server (Pi/phone protocol), standing in on your own machine
python -m src.tier1_link.server --root tier2_export_root --port 8443
# (add --http for plain HTTP without TOFU/HTTPS, faster for quick local testing)

# See the segmentation-backend comparison tooling/data from this session
python scripts/compare_segmentation.py <clip.mp4> --skip-n 5 --out scratch/compare.mp4
```

---

If anything in this doc doesn't match what you find in the actual code, the code wins — ping
Umair to reconcile. Good luck with Phase 2.
