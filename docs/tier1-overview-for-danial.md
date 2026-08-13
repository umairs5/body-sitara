# Tier 1 — how it works (short version)

Tier 1 is the on-glasses part. It takes a video, finds every person in each
frame, blurs them out (grey-fills their body), and encrypts a copy of the
original pixels so they can be restored later if the bystander consents. It
never sends raw video anywhere — only the blurred video plus some small
signals leave the device.

## The pipeline, in order (per frame)

1. **Detect people** — find every person's bounding box.
2. **Pose** — find each person's body keypoints (shoulders, elbows, etc.) and
   face keypoints (eyes, nose).
3. **Anonymize** — grey-fill each person, either as a box or a rough mask
   (see "anonymizer backends" below — this is the part with the most
   options).
4. **Face expression** — crop the face, run it through a face-landmark model,
   and turn it into a small "expression only" signal (mouth open, smiling,
   head tilt, etc.) — no identity, just behavior.
5. **Gender guess** — a coarse male/female label per person, used later to
   help the cloud pick a plausible-looking synthetic body (not identity).
6. **Encrypt** — save an encrypted crop of each person's real pixels + a face
   embedding, locked with a key only the consent server (Tier 3) can open.
7. **Track** — keep each person's identity consistent across frames (so
   "person 2" in frame 100 is still "person 2" in frame 150), and pick the
   best-quality frame per person to use for the face embedding/gender guess.

To go faster, steps 1–2 don't run on every single frame — every Nth frame is
a "full" frame (real detection), and in between, boxes are just tracked
forward using optical flow (cheap, no re-detection).

## Files — what does what

| File | What it's for |
|---|---|
| `pipeline.py` | The orchestrator. Runs the loop above, frame by frame. Almost everything else is called from here. |
| `pose.py` | Small helpers: keypoint math, crop extraction, movement/size classification. |
| `tracking.py` | Keeps track of "who is person 1, 2, 3..." across frames; picks best frame per person. |
| `encryption.py` | AES + RSA encryption for the saved crops/embeddings. |
| `embedding.py` | Turns a face crop into a 512-number "fingerprint" (used by Tier 3 to match consent). |
| `gender.py` | The male/female guess, from a face crop. |
| `face_canonical_v2.py` | Turns a face into the 12-number "expression only" signal + a cartoon face image. |

### Anonymizer backends (`blur_*.py`) — pick ONE via `--anonymizer`

These are different ways to do step 3. Only one runs at a time.

| `--anonymizer` value | File | What it actually does |
|---|---|---|
| `convexhull` (default) | `blur.py` | Draws a solid shape around body+face keypoints. No neural net for the shape itself. |
| `selfie_seg0` / `selfie_seg1` | `blur_seg.py` | MediaPipe's person-segmentation model — real pixel mask. |
| `mobilesam` | `blur_mobilesam.py` | MobileSAM, prompted by the keypoints — real pixel mask. |
| `yoloseg` / `yoloseg11` / `yoloseg11int8` / `yoloseg11ncnn` | `blur_yoloseg.py` | YOLO-seg models (different sizes/export formats) — real pixel mask. |
| **`yolo11n_boxfill`** | `blur_yolo11n.py` | **Plain YOLO11n detector, NO segmentation — just grey-fills the rectangular box.** This is what the paper's Table 6/7 numbers (AP/AR) are actually measuring. Was just wired into the CLI today — before that it only existed as a standalone class used by eval scripts. |

Why a box instead of a mask, for the one that matters for the paper: a
mask-shaped cutout still leaks the person's outline (their silhouette is a
real re-identification signal — gait, body shape). A rectangle destroys the
outline completely. Slightly worse for scene realism, better for privacy.

## How to run it

```bash
python scripts/run.py <video_path> --anonymizer yolo11n_boxfill --headless
```

Useful flags:
- `--no-save` — skip writing the output video (just measure speed).
- `--benchmark` — skip crypto + drawing too, pure pipeline speed.
- `--skip-n 5` — how many frames between full detection passes (default 5).
- `--export-dir <path>` — dump per-frame keypoints/boxes/crops to disk (used
  for building datasets / evaluation, not normal runs).

## Where the real numbers come from

The detection accuracy numbers in the paper (Table 6/7) were NOT measured by
running the live pipeline — they were measured by a separate evaluation
pipeline against 16,507 hand-verified frames. That's a different, dedicated
set of scripts (`scripts/eval_*.py` and `scripts/colab_eval_*.ipynb`), not
part of the live camera/video pipeline described above. See
`results/tier1_detection_eval/README.md` for the full writeup, the ground
truth files, and the exact numbers.
