"""
Quick side-by-side visual test: v1 (position-based) vs v2 (parametric,
identity-baseline-stripped) face canonicalizer, on a single test clip.

v2's smile-baseline correction needs whole-clip statistics, so this runs
two passes: pass 1 extracts a face crop + v2 param vector per frame and
caches both (no detection re-run needed in pass 2); pass 2 renders v1 and
v2 side by side using the cached data and the clip-wide smile baseline.

Usage:
    python scripts/test_face_canon_v2.py <video_path> [--output OUT.mp4]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from body_sitara.detector_patch import apply_detector_patch
from rtmlib import Body
from body_sitara.pose import derive_face_crop, COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE
from body_sitara.face_canonical import FaceCanonicalizer
from body_sitara.face_canonical_v2 import FaceCanonicalizerV2, P_SMILE

INFER_SIZE = 320
PANEL = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--output', default=None)
    ap.add_argument('--face-model', default='face_landmarker.task')
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.video)[0] + '_v1_vs_v2.mp4'

    apply_detector_patch()
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    canon_v2 = FaceCanonicalizerV2(model_path=args.face_model)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[1/2] Extracting face crops + v2 params ({n_frames_total} frames)...")
    cached_crops = []
    cached_params = []
    t0 = time.time()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        infer_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))
        sx, sy = w / INFER_SIZE, h / INFER_SIZE
        bboxes = body.det_model(infer_frame)
        keypoints, scores = body.pose_model(infer_frame, bboxes=bboxes)

        crop, params = None, None
        if keypoints is not None and len(keypoints) > 0:
            kpts = keypoints[0].copy()
            kpts[:, 0] *= sx
            kpts[:, 1] *= sy
            scrs = scores[0]
            if (scrs[COCO_NOSE] > 0.3 and scrs[COCO_LEFT_EYE] > 0.3
                    and scrs[COCO_RIGHT_EYE] > 0.3):
                crop, _, _, _, _ = derive_face_crop(frame, kpts, scrs)
                if crop is not None:
                    params = canon_v2.extract_params(crop)

        cached_crops.append(crop)
        cached_params.append(params)
        idx += 1
        if idx % 100 == 0:
            print(f"  frame {idx}/{n_frames_total}")
    cap.release()
    print(f"  done in {time.time()-t0:.1f}s")

    valid_smiles = [p[P_SMILE] for p in cached_params if p is not None]
    smile_baseline = float(np.median(valid_smiles)) if valid_smiles else 0.0
    print(f"  clip smile baseline: {smile_baseline:.4f}  ({len(valid_smiles)}/{idx} frames had a face)")
    canon_v2.set_smile_baseline(smile_baseline)

    print("[2/2] Rendering v1 vs v2 comparison video...")
    canon_v1 = FaceCanonicalizer(model_path=args.face_model)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (PANEL * 3, PANEL))

    label_font = cv2.FONT_HERSHEY_SIMPLEX
    last_v1, last_v2 = None, None
    for i, (crop, params) in enumerate(zip(cached_crops, cached_params)):
        if crop is not None:
            face_panel = cv2.resize(crop, (PANEL, PANEL))
        else:
            face_panel = np.full((PANEL, PANEL, 3), (40, 40, 40), dtype=np.uint8)

        if crop is not None:
            v1_face = canon_v1.get_canonical_face(crop)
            if v1_face is not None:
                last_v1 = v1_face
        v1_panel = cv2.resize(last_v1, (PANEL, PANEL)) if last_v1 is not None \
            else np.full((PANEL, PANEL, 3), (228, 225, 222), dtype=np.uint8)

        if params is not None:
            last_v2 = canon_v2.render(params)
        v2_panel = cv2.resize(last_v2, (PANEL, PANEL)) if last_v2 is not None \
            else np.full((PANEL, PANEL, 3), (228, 225, 222), dtype=np.uint8)

        cv2.putText(face_panel, "FACE CROP", (8, 24), label_font, 0.7, (255, 255, 255), 2)
        cv2.putText(v1_panel,   "V1 (position-based)", (8, 24), label_font, 0.65, (40, 40, 40), 2)
        cv2.putText(v2_panel,   "V2 (parametric)",     (8, 24), label_font, 0.65, (40, 40, 40), 2)

        out.write(np.hstack([face_panel, v1_panel, v2_panel]))
        if (i + 1) % 100 == 0:
            print(f"  frame {i+1}/{len(cached_crops)}")

    out.release()
    canon_v1.close()
    canon_v2.close()
    print(f"\nOutput: {out_path}")


if __name__ == '__main__':
    main()
