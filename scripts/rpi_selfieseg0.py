"""
bodySITARA — selfie_seg0 standalone script for Raspberry Pi 5
=============================================================
Single-file, no package install from this repo required.

Dependencies (pip install):
    pip install opencv-python-headless numpy mediapipe rtmlib onnxruntime

Models (auto-downloaded on first run):
    face_landmarker.task          (~30 MB)
    selfie_segmenter.tflite       (~0.3 MB)
    YOLOX-Nano + RTMPose-T        (~20 MB, via rtmlib)

Usage:
    python rpi_selfieseg0.py <video_path> [options]

    --skip-n N          frames between full inference (default 5)
    --output PATH       output mp4 path (default /tmp/out_sitara.mp4)
    --no-save           skip writing output video (benchmark FPS only)
    --headless          no cv2.imshow window
    --no-canonical      skip the canonical expression face panel
    --infer-size N      resize for YOLO/RTMPose inference (default 320)
    --cam               use webcam (index 0) instead of a file

Example:
    python rpi_selfieseg0.py myvideo.mp4 --skip-n 5 --headless
"""

import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from rtmlib import Body, draw_skeleton

# ── Constants ─────────────────────────────────────────────────────────────────

INFER_SIZE    = 320
SKIP_N        = 5
DISPLAY_H     = 640
HULL_COLOR    = (127, 127, 127)   # grey fill for anonymized person
CANONICAL_SIZE = 512

BASE_RESOLUTION       = 1280.0
BASE_FAR_THRESHOLD    = 30
BASE_MEDIUM_THRESHOLD = 80
BASE_SLOW_THRESHOLD   = 5
BASE_FAST_THRESHOLD   = 15
FACE_MESH_MIN_CONF    = 0.3
TIMING_INTERVAL       = 30

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# ── COCO keypoint indices ─────────────────────────────────────────────────────
COCO_NOSE      = 0
COCO_LEFT_EYE  = 1
COCO_RIGHT_EYE = 2
BODY_KPT_INDICES  = list(range(0, 17))
BODY_CROP_PADDING = 20

# ── MediaPipe landmark groups for canonical face ──────────────────────────────
_FACE_OVAL  = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
               397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
               172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
_LEFT_EYE   = [33, 246, 161, 160, 159, 158, 157, 173,
               133, 155, 154, 153, 145, 144, 163, 7]
_RIGHT_EYE  = [362, 398, 384, 385, 386, 387, 388, 466,
               263, 249, 390, 373, 374, 380, 381, 382]
_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
_LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
               308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
_NOSE_BRIDGE   = [168, 6, 197, 195, 5, 4]
_LEFT_NOSTRIL  = 48
_RIGHT_NOSTRIL = 278
_NOSE_TIP      = 4

# Diffusion-friendly color palette (BGR)
_BG        = (228, 225, 222)
_SKIN      = (185, 190, 198)
_OUTLINE   = (70,  65,  62)
_BROW      = (38,  44,  58)
_SCLERA    = (250, 250, 248)
_IRIS      = (48,  58,  78)
_PUPIL     = (8,   8,   12)
_HIGHLIGHT = (255, 255, 255)
_LIP       = (90,  82, 195)
_LIP_IN    = (48,  36, 145)
_NOSE_SHD  = (162, 168, 178)

# ── Model download helpers ────────────────────────────────────────────────────

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
SELFIE_SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


def _download_if_missing(path, url):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  Saved -> {path}")


# ── Pose helpers ──────────────────────────────────────────────────────────────

def euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_face_size_tier(inter_eye_px, far_thr, med_thr):
    if inter_eye_px < far_thr:   return "far"
    if inter_eye_px < med_thr:   return "medium"
    return "close"


def get_movement_tier(disp, slow_thr, fast_thr):
    if disp < slow_thr:  return "slow"
    if disp < fast_thr:  return "medium"
    return "fast"


def derive_face_crop(frame, kpts, scores, kpt_thr=0.3):
    h, w = frame.shape[:2]
    if (scores[COCO_NOSE]      < kpt_thr or
            scores[COCO_LEFT_EYE]  < kpt_thr or
            scores[COCO_RIGHT_EYE] < kpt_thr):
        return None, None, None, None
    inter_eye_px = euclidean(kpts[COCO_LEFT_EYE], kpts[COCO_RIGHT_EYE])
    face_radius  = max(int(inter_eye_px * 2.2), 20)
    cx, cy = int(kpts[COCO_NOSE][0]), int(kpts[COCO_NOSE][1])
    x1 = max(cx - face_radius, 0);  y1 = max(cy - face_radius, 0)
    x2 = min(cx + face_radius, w);  y2 = min(cy + face_radius, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None, None
    return crop, x1, y1, (x2 - x1, y2 - y1)


# ── Canonical face renderer ───────────────────────────────────────────────────

def _px(landmarks, size=CANONICAL_SIZE):
    return np.array([[int(lm.x * size), int(lm.y * size)]
                     for lm in landmarks], dtype=np.int32)


def _eye_center_and_height(px_all, indices):
    pts = px_all[indices]
    cx  = int(pts[:, 0].mean());  cy = int(pts[:, 1].mean())
    eh  = int(pts[:, 1].max() - pts[:, 1].min())
    return pts, cx, cy, eh


def _render_canonical(px_all):
    canvas = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), _BG, dtype=np.uint8)
    cv2.fillPoly(canvas, [px_all[_FACE_OVAL]], _SKIN)
    cv2.fillPoly(canvas, [px_all[_LIPS_OUTER]], _LIP)
    cv2.fillPoly(canvas, [px_all[_LIPS_INNER]], _LIP_IN)
    cv2.polylines(canvas, [px_all[_LIPS_OUTER]], True, _OUTLINE, 1, cv2.LINE_AA)
    for eye_idx in (_LEFT_EYE, _RIGHT_EYE):
        pts, cx, cy, eh = _eye_center_and_height(px_all, eye_idx)
        cv2.fillPoly(canvas, [pts], _SCLERA)
        if eh > 5:
            iris_r = max(int(eh * 0.44), 3)
            cv2.circle(canvas, (cx, cy), iris_r, _IRIS, -1)
            cv2.circle(canvas, (cx, cy), max(iris_r // 2, 2), _PUPIL, -1)
            hx = cx - max(iris_r // 4, 1);  hy = cy - max(iris_r // 4, 1)
            cv2.circle(canvas, (hx, hy), max(iris_r // 5, 1), _HIGHLIGHT, -1)
        cv2.polylines(canvas, [pts], True, _OUTLINE, 2, cv2.LINE_AA)
    for brow in (_LEFT_BROW, _RIGHT_BROW):
        cv2.polylines(canvas, [px_all[brow]], False, _BROW, 7, cv2.LINE_AA)
    cv2.polylines(canvas, [px_all[_NOSE_BRIDGE]], False, _NOSE_SHD, 2, cv2.LINE_AA)
    nt = tuple(px_all[_NOSE_TIP])
    for n_idx in (_LEFT_NOSTRIL, _RIGHT_NOSTRIL):
        cv2.circle(canvas, tuple(px_all[n_idx]), 7, _NOSE_SHD, 2, cv2.LINE_AA)
    cv2.circle(canvas, nt, 4, _NOSE_SHD, -1, cv2.LINE_AA)
    cv2.polylines(canvas, [px_all[_FACE_OVAL]], True, _OUTLINE, 2, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


class FaceCanonicalizer:
    def __init__(self, model_path, infer_size=512,
                 min_det_conf=0.7, min_presence_conf=0.45):
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_det_conf,
            min_face_presence_confidence=min_presence_conf,
        )
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)
        self._infer_size = infer_size

    def get_canonical_face(self, face_crop):
        small  = cv2.resize(face_crop, (self._infer_size, self._infer_size))
        result = self._lm.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        )
        if not result.face_landmarks:
            return None
        return _render_canonical(_px(result.face_landmarks[0]))

    def close(self):
        self._lm.close()


# ── SelfieSegBlur ─────────────────────────────────────────────────────────────

class SelfieSegBlur:
    def __init__(self, model_path, threshold=0.5):
        opts = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            output_category_mask=True,
        )
        self._seg = mp_vision.ImageSegmenter.create_from_options(opts)
        self._thr = threshold

    def get_mask(self, frame, infer_size=None):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (infer_size, infer_size)) if infer_size else frame
        result = self._seg.segment(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        )
        if not result.confidence_masks:
            return None
        conf = result.confidence_masks[0].numpy_view().squeeze()
        small_mask = conf > self._thr
        if infer_size:
            return cv2.resize(small_mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        return small_mask

    def apply_mask(self, frame, mask):
        out = frame.copy()
        out[mask] = HULL_COLOR
        return out

    def close(self):
        self._seg.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(
    input_path,
    output_path    = "/tmp/out_sitara.mp4",
    save_video     = True,
    headless       = False,
    skip_n         = 5,
    use_canonical  = True,
    infer_size     = 320,
    seg_model_path = "selfie_segmenter.tflite",
    lm_model_path  = "face_landmarker.task",
):
    print("=" * 60)
    print("  bodySITARA — selfie_seg0 pipeline (RPi 5 standalone)")
    print(f"  Infer size : {infer_size}x{infer_size}  |  Skip-N : {skip_n}")
    print(f"  Canonical  : {use_canonical}  |  Headless : {headless}")
    print(f"  Save video : {save_video} -> {output_path}")
    print("=" * 60)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[1/3] Loading RTMPose (YOLOX-Nano + RTMPose-T) — downloads on first run...")
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    print("\n[2/3] Loading MediaPipe SelfieSegmenter...")
    _download_if_missing(seg_model_path, SELFIE_SEG_URL)
    selfie_seg = SelfieSegBlur(model_path=seg_model_path)

    face_canonicalizer = None
    if use_canonical:
        print("\n[3/3] Loading FaceCanonicalizer (face_landmarker.task)...")
        _download_if_missing(lm_model_path, FACE_LANDMARKER_URL)
        face_canonicalizer = FaceCanonicalizer(model_path=lm_model_path)
    else:
        print("\n[3/3] Canonical face disabled — skipping FaceLandmarker load.")

    # ── Open video/camera ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"\nERROR: Cannot open '{input_path}'")
        sys.exit(1)

    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scale            = min(width, height) / BASE_RESOLUTION
    FAR_THR          = max(int(BASE_FAR_THRESHOLD    * scale), 5)
    MED_THR          = max(int(BASE_MEDIUM_THRESHOLD * scale), 15)
    SLOW_THR         = max(int(BASE_SLOW_THRESHOLD   * scale), 1)
    FAST_THR         = max(int(BASE_FAST_THRESHOLD   * scale), 3)
    kp_scale_x       = width  / infer_size
    kp_scale_y       = height / infer_size

    print(f"\n  Input  : {width}x{height} @ {fps_input:.1f} fps")
    print(f"  Scale  : {scale:.3f}  FAR={FAR_THR}px  MED={MED_THR}px")

    # Three-panel output (original | blurred | expression) when canonical on
    out_w = (DISPLAY_H * 3) if use_canonical else width
    out_h = DISPLAY_H       if use_canonical else height

    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps_input, (out_w, out_h))
        if not out.isOpened():
            print("  WARNING: VideoWriter failed — video will not be saved.")
            out = None

    # ── Per-person state (lightweight — no encryption) ────────────────────────
    prev_noses = {}   # person_idx -> last nose (x, y)

    # ── Thread pools ──────────────────────────────────────────────────────────
    _seg_pool   = ThreadPoolExecutor(max_workers=1)
    _lk_pool    = ThreadPoolExecutor(max_workers=2)
    _write_pool = ThreadPoolExecutor(max_workers=1)

    # ── Timing accumulators ───────────────────────────────────────────────────
    t_det = t_pose = t_seg = t_lk = t_blur = t_canonical = 0.0
    full_frames = skip_frames = frame_idx = 0

    last_keypoints  = None
    last_scores     = None
    last_seg_mask   = None
    seg_mask_kpts   = None
    last_canon      = None
    prev_gray       = None
    movement_tier   = "medium"
    fps_history     = []
    prev_time       = time.time()
    loop_start      = time.time()

    print("\nRunning — press 'q' to quit.\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        annotated     = frame.copy()
        curr_gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_full       = (frame_idx % skip_n == 0)

        if is_full or prev_gray is None:
            full_frames += 1
            infer_frame  = cv2.resize(frame, (infer_size, infer_size))

            # selfie-seg parallel with det+pose (TFLite releases GIL)
            _future_seg = _seg_pool.submit(selfie_seg.get_mask, frame, 256)
            _t_seg0     = time.time()

            t0     = time.time()
            bboxes = body.det_model(infer_frame)
            t_det += time.time() - t0

            t0             = time.time()
            keypoints, scores = body.pose_model(infer_frame, bboxes=bboxes)
            t_pose        += time.time() - t0

            if keypoints is not None and len(keypoints) > 0:
                keypoints[:, :, 0] *= kp_scale_x
                keypoints[:, :, 1] *= kp_scale_y

            last_keypoints = keypoints
            last_scores    = scores

            # canonical face for person 0 (on full frames only)
            if use_canonical and keypoints is not None and len(keypoints) > 0:
                crop, *_ = derive_face_crop(frame, keypoints[0], scores[0])
                if crop is not None:
                    tc0 = time.time()
                    cf  = face_canonicalizer.get_canonical_face(crop)
                    t_canonical += time.time() - tc0
                    if cf is not None:
                        last_canon = cf

            # collect seg result (block only if det+pose finished first)
            last_seg_mask = _future_seg.result()
            t_seg        += time.time() - _t_seg0
            seg_mask_kpts = (keypoints.copy()
                             if keypoints is not None and len(keypoints) > 0
                             else None)

            prev_gray = curr_gray.copy()

            if full_frames % TIMING_INTERVAL == 0:
                n = max(full_frames, 1)
                s = max(skip_frames, 1)
                f = max(frame_idx, 1)
                print(
                    f"[F{frame_idx:4d}] "
                    f"Det: {t_det/n*1000:5.1f}ms | "
                    f"Pose: {t_pose/n*1000:5.1f}ms | "
                    f"Seg: {t_seg/n*1000:5.1f}ms (parallel) | "
                    f"Canon: {t_canonical/n*1000:4.1f}ms | "
                    f"LK: {t_lk/max(s,1)*1000:4.1f}ms | "
                    f"People: {len(keypoints) if keypoints is not None else 0}"
                )

        else:
            skip_frames += 1
            if last_keypoints is not None and len(last_keypoints) > 0:
                n_p = len(last_keypoints)

                def _body_lk(i):
                    old = last_keypoints[i][:, :2].astype(np.float32).reshape(-1, 1, 2)
                    new, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old, None, **LK_PARAMS)
                    return i, new

                tlk0         = time.time()
                futs         = [_lk_pool.submit(_body_lk, i) for i in range(n_p)]
                tracked      = [None] * n_p
                for fut in futs:
                    i, new = fut.result()
                    tk     = last_keypoints[i].copy()
                    tk[:, :2] = new.reshape(-1, 2)
                    tracked[i] = tk
                t_lk += time.time() - tlk0

                keypoints      = np.array(tracked)
                scores         = last_scores
                last_keypoints = keypoints

                # warp stored mask by affine from keypoint motion
                if last_seg_mask is not None and seg_mask_kpts is not None:
                    old_pts = seg_mask_kpts[:, :, :2].reshape(-1, 2).astype(np.float32)
                    new_pts = keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    M, _    = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC)
                    if M is not None:
                        last_seg_mask = cv2.warpAffine(
                            last_seg_mask.astype(np.uint8), M, (width, height),
                            flags=cv2.INTER_NEAREST
                        ).astype(bool)
                    seg_mask_kpts = keypoints.copy()

            prev_gray = curr_gray.copy()

        # ── Apply mask + draw ─────────────────────────────────────────────────
        tb0 = time.time()
        if last_seg_mask is not None:
            annotated = selfie_seg.apply_mask(annotated, last_seg_mask)
        t_blur += time.time() - tb0

        if keypoints is not None and len(keypoints) > 0:
            annotated = draw_skeleton(annotated, keypoints, scores, kpt_thr=0.3)

        # HUD
        now     = time.time()
        elapsed = max(now - prev_time, 1e-6)
        prev_time = now
        fps_history.append(1.0 / elapsed)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_display = sum(fps_history) / len(fps_history)

        for j, line in enumerate([
            f"FPS: {fps_display:.1f}",
            f"{'FULL' if is_full else 'SKIP'}",
            f"People: {len(keypoints) if keypoints is not None else 0}",
        ]):
            cv2.putText(annotated, line, (10, 30 + j * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # ── Write frame ───────────────────────────────────────────────────────
        if out is not None:
            if use_canonical:
                orig_p  = cv2.resize(frame,     (DISPLAY_H, DISPLAY_H))
                blur_p  = cv2.resize(annotated,  (DISPLAY_H, DISPLAY_H))
                expr_p  = np.full((DISPLAY_H, DISPLAY_H, 3), _BG, dtype=np.uint8)
                if last_canon is not None:
                    expr_p[:] = cv2.resize(last_canon, (DISPLAY_H, DISPLAY_H))
                _lbl = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(orig_p,  "ORIGINAL",   (8, 24), _lbl, 0.6, (255, 255, 255), 2)
                cv2.putText(blur_p,  "BLURRED",    (8, 24), _lbl, 0.6, (255, 255, 255), 2)
                cv2.putText(expr_p,  "EXPRESSION", (8, 24), _lbl, 0.6, (40,  40,  40),  2)
                combined = np.hstack([orig_p, blur_p, expr_p])
                _write_pool.submit(out.write, combined)
            else:
                _write_pool.submit(out.write, annotated.copy())

        if not headless:
            cv2.imshow("bodySITARA", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser quit.")
                break

        frame_idx += 1

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _write_pool.shutdown(wait=True)
    _seg_pool.shutdown(wait=False)
    _lk_pool.shutdown(wait=False)
    cap.release()
    if out is not None:
        out.release()
    if not headless:
        cv2.destroyAllWindows()
    selfie_seg.close()
    if face_canonicalizer is not None:
        face_canonicalizer.close()

    total_time = time.time() - loop_start
    avg_fps    = frame_idx / max(total_time, 1e-6)
    f = max(frame_idx, 1)
    n = max(full_frames, 1)
    s = max(skip_frames, 1)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Total frames     : {frame_idx}")
    print(f"  Full inf frames  : {full_frames}  ({full_frames/f*100:.1f}%)")
    print(f"  Skip (LK) frames : {skip_frames}  ({skip_frames/f*100:.1f}%)")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Average FPS      : {avg_fps:.2f}")
    print()
    print(f"  Avg Det/full     : {t_det      / n * 1000:.1f}ms")
    print(f"  Avg Pose/full    : {t_pose     / n * 1000:.1f}ms")
    print(f"  Avg Seg/full     : {t_seg      / n * 1000:.1f}ms  (parallel wall time)")
    print(f"  Avg Canon/full   : {t_canonical / n * 1000:.1f}ms")
    print(f"  Avg LK/skip      : {t_lk       / s * 1000:.1f}ms")
    print(f"  Avg Blur/frame   : {t_blur      / f * 1000:.1f}ms")
    if save_video and out is not None:
        print(f"\n  Output video     : {output_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="bodySITARA selfie_seg0 pipeline — RPi 5 standalone")
    ap.add_argument("input", nargs="?", default=None,
                    help="Path to input video file")
    ap.add_argument("--cam",          action="store_true",
                    help="Use webcam (index 0) instead of a file")
    ap.add_argument("--skip-n",       type=int, default=5,
                    help="Frames between full inference runs (default 5)")
    ap.add_argument("--output",       default="/tmp/out_sitara.mp4",
                    help="Output video path (default /tmp/out_sitara.mp4)")
    ap.add_argument("--no-save",      action="store_true",
                    help="Do not write output video (benchmark FPS only)")
    ap.add_argument("--headless",     action="store_true",
                    help="No display window (for SSH / headless RPi)")
    ap.add_argument("--no-canonical", action="store_true",
                    help="Skip canonical expression face panel")
    ap.add_argument("--infer-size",   type=int, default=320,
                    help="Resize resolution for YOLO/RTMPose (default 320)")
    ap.add_argument("--seg-model",    default="selfie_segmenter.tflite",
                    help="Path to selfie_segmenter.tflite (auto-downloaded if missing)")
    ap.add_argument("--lm-model",     default="face_landmarker.task",
                    help="Path to face_landmarker.task (auto-downloaded if missing)")
    args = ap.parse_args()

    if args.cam:
        src = 0
    elif args.input:
        src = args.input
    else:
        ap.print_help()
        sys.exit(1)

    process_video(
        input_path    = src,
        output_path   = args.output,
        save_video    = not args.no_save,
        headless      = args.headless,
        skip_n        = args.skip_n,
        use_canonical = not args.no_canonical,
        infer_size    = args.infer_size,
        seg_model_path = args.seg_model,
        lm_model_path  = args.lm_model,
    )
