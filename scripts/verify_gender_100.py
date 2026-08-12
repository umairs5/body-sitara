"""
Runs the REAL pipeline gender path (RTMPose detect+pose -> derive_face_crop ->
GenderClassifier.predict, same call sequence as pipeline.py's export block)
on 100 sampled frames from data/output/annotation_frames, and saves a
side-by-side composite per frame: the ACTUAL crop pixels that were fed to
the classifier (left, upscaled for visibility) next to the full original
frame for context (right), with the predicted label + confidence burned
onto the crop side. This is what derive_face_crop() produced and
GenderClassifier.predict() actually saw -- not a label overlaid on the
original frame, which would show nothing about crop framing/tightness/
wrong-person errors. No ground truth is assumed or checked here -- purely
produces labeled images for a human to look at.

Usage:
  python scripts/verify_gender_100.py [--n 100] [--out scratch/gender_verify]
"""
import os
import re
import sys
import glob
import argparse
import random

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from rtmlib import Body
from body_sitara.detector_patch import apply_detector_patch
from body_sitara.pose import derive_face_crop, COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE
from body_sitara.gender import GenderClassifier, GENDER_ONNX_PATH, _ARCFACE_DST_3PT

FRAMES_ROOT = os.path.join(REPO_ROOT, "data", "output", "annotation_frames")

# Single-face categories preferred (one clear, unambiguous subject per frame,
# easiest to hand-verify) -- sampled round-robin so the 100 aren't dominated
# by whichever category happens to have the most files.
CATEGORIES = [
    "NumFaces1", "MovementRest", "MovementHead", "MovementBystander",
    "SizeClose", "SizeMedium", "SizeFar",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "scratch", "gender_verify"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading RTMPose (YOLOX-Nano + RTMPose-T)...")
    apply_detector_patch()
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    print("Loading InsightFace gender classifier...")
    gc = GenderClassifier(GENDER_ONNX_PATH)

    # Group files by (category, subject) -- e.g. "NumFaces1__7_single_face" --
    # not just category, so sampling spreads across the ~10 distinct people
    # recorded per category instead of repeatedly hitting the same subject's
    # consecutive frames (an earlier version of this script did exactly that:
    # naive round-robin-by-category with files sorted lexicographically put
    # every "10_*" frame first, so 100 "samples" were actually ~14 frames
    # each of only 7 people).
    subject_files = {}  # (cat, subject_key) -> [file paths]
    for cat in CATEGORIES:
        files = sorted(glob.glob(os.path.join(FRAMES_ROOT, cat, "*.jpg")))
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            m = re.match(r"(.+?)__f\d+$", stem)
            key = (cat, m.group(1) if m else stem)
            subject_files.setdefault(key, []).append(fp)

    # One representative frame per (category, subject) per pass, subjects
    # shuffled so which frame within a subject's clip gets picked varies
    # instead of always the first, then repeat passes (skipping
    # already-used files) until args.n is reached.
    rng = random.Random(0)
    subject_keys = list(subject_files.keys())
    rng.shuffle(subject_keys)
    used = {k: 0 for k in subject_keys}

    picked = 0
    results = []
    round_num = 0
    while picked < args.n and round_num < 50:
        any_left = False
        for key in subject_keys:
            if picked >= args.n:
                break
            files = subject_files[key]
            i = used[key]
            if i >= len(files):
                continue
            any_left = True
            fp = files[i]
            used[key] += 1
            cat = key[0]
            img = cv2.imread(fp)
            if img is None:
                continue

            boxes = body.det_model(img)
            if boxes is None or len(boxes) == 0:
                continue
            keypoints, scores = body.pose_model(img, bboxes=boxes)
            if keypoints is None or len(keypoints) == 0:
                continue
            # largest-box person in frame (same pick rule as diag_gender_model_compare.py)
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            pick = int(np.argmax(areas))
            kpts_i, scrs_i = keypoints[pick], scores[pick]

            crop, _, _, _, _ = derive_face_crop(img, kpts_i, scrs_i)
            if crop is None or crop.size == 0:
                continue

            # NEW: proper 3-point (eyes+nose) similarity-transform alignment
            # directly on the original frame -- see gender.py's
            # predict_from_keypoints() docstring for why this is preferred
            # over the crude bbox-center-scale fallback below.
            result_new = gc.predict_from_keypoints(
                img, kpts_i[COCO_LEFT_EYE], kpts_i[COCO_RIGHT_EYE], kpts_i[COCO_NOSE]
            )
            # OLD: crude bbox-center-and-scale fallback on the already-cropped
            # derive_face_crop() output -- kept side by side for comparison.
            result_old = gc.predict(crop)
            if result_new is None and result_old is None:
                continue
            label_new, conf_new = result_new if result_new is not None else ("N/A", 0.0)
            label_old, conf_old = result_old if result_old is not None else ("N/A", 0.0)
            label, conf = label_new, conf_new  # what filenames/results.json report

            def _panel(image, txt, color, size=400):
                interp = cv2.INTER_CUBIC if image.shape[0] < size else cv2.INTER_AREA
                disp = cv2.resize(image, (size, size), interpolation=interp)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                cv2.rectangle(disp, (0, 0), (20 + tw, 20 + th), (0, 0, 0), -1)
                cv2.putText(disp, txt, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, color, 3, cv2.LINE_AA)
                cv2.rectangle(disp, (0, 0), (size - 1, size - 1), color, 4)
                return disp

            DISPLAY = 400
            color_new = (0, 140, 255) if label_new == "Male" else (200, 0, 200)
            color_old = (0, 140, 255) if label_old == "Male" else (200, 0, 200)

            # aligned_new comes out already at the model's declared input
            # size (96x96) via warpAffine inside predict_from_keypoints -- we
            # don't have direct access to it here, so recompute the same
            # alignment just for display (identical math, display-only cost).
            attr = gc._attr
            target = attr.input_size[0]
            dst = _ARCFACE_DST_3PT * (target / 112.0)
            src = np.array([kpts_i[COCO_RIGHT_EYE], kpts_i[COCO_LEFT_EYE], kpts_i[COCO_NOSE]], dtype=np.float32)  # swapped -- see gender.py predict_from_keypoints docstring
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            aligned_new = (cv2.warpAffine(img, M, (target, target), borderValue=0.0)
                           if M is not None else np.zeros((target, target, 3), dtype=np.uint8))

            panel_new = _panel(aligned_new, f"NEW {label_new} {conf_new:.2f}", color_new)
            panel_old = _panel(crop, f"OLD {label_old} {conf_old:.2f}", color_old)

            oh, ow = img.shape[:2]
            ctx_w = int(ow * DISPLAY / oh)
            ctx_display = cv2.resize(img, (ctx_w, DISPLAY))

            out_img = np.hstack([panel_new, panel_old, ctx_display])

            out_name = f"{picked:03d}_{cat}_{label}_{conf:.2f}_{os.path.basename(fp)}"
            cv2.imwrite(os.path.join(args.out, out_name), out_img)

            results.append({
                "file": out_name, "source": fp, "category": cat,
                "label_new": label_new, "confidence_new": round(conf_new, 4),
                "label_old": label_old, "confidence_old": round(conf_old, 4),
                "agree": label_new == label_old,
            })
            picked += 1
            flag = "" if label_new == label_old else "  DISAGREE"
            print(f"[{picked}/{args.n}] {cat:20s} {os.path.basename(fp):45s} "
                  f"NEW={label_new}:{conf_new:.2f} OLD={label_old}:{conf_old:.2f}{flag}")
        if not any_left:
            break
        round_num += 1

    import json
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {picked} labeled images -> {args.out}")


if __name__ == "__main__":
    main()
