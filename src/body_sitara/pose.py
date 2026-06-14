import numpy as np
import cv2
import mediapipe as mp

COCO_NOSE      = 0
COCO_LEFT_EYE  = 1
COCO_RIGHT_EYE = 2

BODY_KPT_INDICES = list(range(0, 17))
BODY_CROP_PADDING = 20

HEAD_SCALE = 2.5

LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def euclidean(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def get_face_size_tier(inter_eye_px: float, far_thr: int, med_thr: int) -> str:
    if inter_eye_px < far_thr:
        return "far"
    elif inter_eye_px < med_thr:
        return "medium"
    return "close"


def get_movement_tier(disp: float, slow_thr: int, fast_thr: int) -> str:
    if disp < slow_thr:
        return "slow"
    elif disp < fast_thr:
        return "medium"
    return "fast"


def derive_face_crop(frame, kpts, scores, kpt_thr=0.3):
    h, w = frame.shape[:2]
    if (scores[COCO_NOSE] < kpt_thr or
            scores[COCO_LEFT_EYE] < kpt_thr or
            scores[COCO_RIGHT_EYE] < kpt_thr):
        return None, None, None, None, 0.0
    inter_eye_px = euclidean(kpts[COCO_LEFT_EYE], kpts[COCO_RIGHT_EYE])
    face_radius  = max(int(inter_eye_px * 2.2), 20)
    cx, cy = int(kpts[COCO_NOSE][0]), int(kpts[COCO_NOSE][1])
    x1 = max(cx - face_radius, 0)
    y1 = max(cy - face_radius, 0)
    x2 = min(cx + face_radius, w)
    y2 = min(cy + face_radius, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None, None, 0.0
    return crop, x1, y1, (x2 - x1, y2 - y1), inter_eye_px


def derive_body_crop(frame, kpts, scores, kpt_thr=0.3):
    h, w = frame.shape[:2]
    visible_pts = []
    for idx in BODY_KPT_INDICES:
        if scores[idx] > kpt_thr:
            visible_pts.append([kpts[idx][0], kpts[idx][1]])
    if len(visible_pts) < 2:
        return None, 0, 0, 0, 0
    pts = np.array(visible_pts)
    x1 = max(int(pts[:, 0].min()) - BODY_CROP_PADDING, 0)
    y1 = max(int(pts[:, 1].min()) - BODY_CROP_PADDING, 0)
    x2 = min(int(pts[:, 0].max()) + BODY_CROP_PADDING, w)
    y2 = min(int(pts[:, 1].max()) + BODY_CROP_PADDING, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0, 0, 0, 0
    return crop, x1, y1, x2, y2


def compute_frame_confidence(scores, kpt_thr=0.3) -> float:
    visible = scores[scores > kpt_thr]
    return float(visible.mean()) if len(visible) > 0 else 0.0


def project_landmarks(landmarks_list, x_off, y_off, crop_w, crop_h):
    return [
        (int(x_off + lm.x * crop_w), int(y_off + lm.y * crop_h))
        for lm in landmarks_list
    ]


def draw_face_mesh_pts(frame, pts, color=(0, 255, 180), radius=1):
    for (x, y) in pts:
        cv2.circle(frame, (x, y), radius, color, -1)
