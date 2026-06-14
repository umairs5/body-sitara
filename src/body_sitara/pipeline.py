import cv2
import time
import os
import csv
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rtmlib import Body, draw_skeleton
from cryptography.hazmat.primitives import serialization

from .pose import (
    euclidean, get_face_size_tier, get_movement_tier,
    derive_face_crop, derive_body_crop, compute_frame_confidence,
    project_landmarks, draw_face_mesh_pts,
    COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE, LK_PARAMS,
)
from .blur import blur_all_persons
from .tracking import PersonState, propagate_bboxes
from .encryption import generate_ttp_keypair
from .embedding import EmbeddingExtractor, EDGEFACE_ONNX_PATH

BASE_RESOLUTION       = 1280.0
BASE_FAR_THRESHOLD    = 30
BASE_MEDIUM_THRESHOLD = 80
BASE_SLOW_THRESHOLD   = 5
BASE_FAST_THRESHOLD   = 15

SKIP_N_DEFAULTS = {
    "slow":   7,
    "medium": 4,
    "fast":   1,
}

FACE_MESH_MIN_CONF = 0.3
INFER_SIZE         = 320
TIMING_INTERVAL    = 30
DEBUG_DRAW         = True


def process_video(
    input_path,
    output_path       = "/tmp/output_rtm.mp4",
    blur_bodies       = True,
    enc_output_dir    = "data/output/encrypted",
    headless          = False,
    save_video        = True,
    benchmark         = False,
    skip_n            = 5,
    movement_adaptive = False,
    csv_out           = None,
):
    if benchmark:
        save_video = False
        headless   = True

    SKIP_N = SKIP_N_DEFAULTS.copy()
    if not movement_adaptive:
        SKIP_N["slow"]   = skip_n
        SKIP_N["medium"] = skip_n
        SKIP_N["fast"]   = skip_n

    draw_enabled = DEBUG_DRAW and not benchmark

    print("=" * 60)
    print("  RTMPose Pipeline (Optical Flow + EdgeFace + Encryption)")
    print(f"  Infer size        : {INFER_SIZE}x{INFER_SIZE}")
    print(f"  Blur bodies       : {blur_bodies}")
    print(f"  Benchmark mode    : {benchmark}")
    print(f"  Movement adaptive : {movement_adaptive}")
    print(f"  Skip-N            : {SKIP_N}")
    print(f"  Output            : {output_path}")
    print(f"  Enc output        : {enc_output_dir}")
    print(f"  Headless          : {headless}  |  Save video: {save_video}")
    print("=" * 60)

    os.makedirs(enc_output_dir, exist_ok=True)

    # [0] TTP RSA-2048 keypair
    if not benchmark:
        print("\n[0/4] Generating TTP RSA-2048 keypair (simulated)...")
        ttp_private_key, ttp_public_key = generate_ttp_keypair()
        priv_path = os.path.join(enc_output_dir, "ttp_private.pem")
        with open(priv_path, 'wb') as f:
            f.write(ttp_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        print(f"    TTP private key saved to: {priv_path}")
    else:
        print("\n[0/4] Benchmark mode — skipping RSA keypair generation")
        ttp_private_key, ttp_public_key = None, None

    # [1] RTMPose
    print("\n[1/4] Loading RTMPose (YOLOX-Nano + RTMPose-T)...")
    body = Body(
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
        det_input_size=(416, 416),
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        pose_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    # [2] MediaPipe FaceLandmarker
    print("\n[2/4] Loading MediaPipe FaceLandmarker...")
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print("  Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
            model_path,
        )
    face_mesh = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=FACE_MESH_MIN_CONF,
            min_face_presence_confidence=FACE_MESH_MIN_CONF,
        )
    )

    # [3] EdgeFace embedding model
    if not benchmark:
        print("\n[3/4] Loading EdgeFace-s-gamma-05 embedding model...")
        try:
            embedder = EmbeddingExtractor(EDGEFACE_ONNX_PATH)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Running WITHOUT face embedding.")
            embedder = None
    else:
        print("\n[3/4] Benchmark mode — skipping EdgeFace embedding model")
        embedder = None

    # [4] Video IO
    print("\n[4/4] Opening video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"\nError: Could not open '{input_path}'")
        return

    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scale            = min(width, height) / BASE_RESOLUTION
    FAR_THRESHOLD    = max(int(BASE_FAR_THRESHOLD    * scale), 5)
    MEDIUM_THRESHOLD = max(int(BASE_MEDIUM_THRESHOLD * scale), 15)
    SLOW_THRESHOLD   = max(int(BASE_SLOW_THRESHOLD   * scale), 1)
    FAST_THRESHOLD   = max(int(BASE_FAST_THRESHOLD   * scale), 3)
    kp_scale_x       = width  / INFER_SIZE
    kp_scale_y       = height / INFER_SIZE

    print(f"\nInput  : {width}x{height} @ {fps_input:.1f} fps  (scale={scale:.3f})")
    print(f"Infer  : {INFER_SIZE}x{INFER_SIZE}  kp_scale=({kp_scale_x:.3f}, {kp_scale_y:.3f})")
    print(f"FAR={FAR_THRESHOLD}px  MED={MEDIUM_THRESHOLD}px  "
          f"SLOW={SLOW_THRESHOLD}px  FAST={FAST_THRESHOLD}px\n")

    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))
        if not out.isOpened():
            print("  Warning: Could not open VideoWriter — video will not be saved.")
            out = None

    frame_idx        = 0
    full_frame_count = 0
    skip_frame_count = 0
    person_states    = {}
    active_indices   = set()
    current_N        = SKIP_N["medium"]
    movement_tier    = "medium"
    last_keypoints   = None
    last_scores      = None
    last_bboxes      = None
    last_mesh_cache  = {}
    prev_gray        = None

    prev_time   = time.time()
    fps_history = []
    FPS_WINDOW  = 30

    t_det_total      = 0.0
    t_pose_total     = 0.0
    t_facemesh_total = 0.0
    t_of_body_total  = 0.0
    t_of_face_total  = 0.0
    t_blur_total     = 0.0
    t_draw_total     = 0.0
    t_write_total    = 0.0
    t_encrypt_total  = 0.0
    t_embed_total    = 0.0

    streams_flushed = 0
    loop_start      = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated     = frame.copy()
        curr_gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_full_frame = (frame_idx % current_N == 0)

        if is_full_frame or prev_gray is None:
            full_frame_count += 1
            infer_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))

            t0     = time.time()
            bboxes = body.det_model(infer_frame)
            t_det_total += time.time() - t0
            last_bboxes = bboxes

            t2 = time.time()
            keypoints, scores = body.pose_model(infer_frame, bboxes=bboxes)
            t_pose_total += time.time() - t2

            if keypoints is not None and len(keypoints) > 0:
                keypoints[:, :, 0] *= kp_scale_x
                keypoints[:, :, 1] *= kp_scale_y

            last_keypoints  = keypoints
            last_scores     = scores
            last_mesh_cache = {}

            current_indices = (
                set(range(len(keypoints)))
                if keypoints is not None and len(keypoints) > 0
                else set()
            )
            departed = active_indices - current_indices
            for dep_idx in departed:
                if dep_idx in person_states:
                    enc_t, emb_t = person_states[dep_idx].flush_to_disk()
                    t_encrypt_total += enc_t
                    t_embed_total   += emb_t
                    streams_flushed += 1
                    sid = person_states[dep_idx].stream_id[:8]
                    print(f"  [STREAM] Person {dep_idx} departed → "
                          f"stream {sid}... flushed "
                          f"(enc={enc_t*1000:.1f}ms emb={emb_t*1000:.1f}ms)")
                    del person_states[dep_idx]
            active_indices = current_indices

            if keypoints is not None and len(keypoints) > 0:
                for i in range(len(keypoints)):
                    if i not in person_states:
                        person_states[i] = PersonState(
                            ttp_public_key, enc_output_dir,
                            embedder, benchmark=benchmark
                        )
                        print(f"  [STREAM] Person {i} appeared → "
                              f"stream {person_states[i].stream_id[:8]}... created")

                per_person_movement = []

                for i in range(len(keypoints)):
                    kpts  = keypoints[i]
                    scrs  = scores[i]
                    state = person_states[i]

                    nose_xy = kpts[COCO_NOSE]
                    disp    = euclidean(nose_xy, state.prev_nose) if state.prev_nose else 0.0
                    state.prev_nose     = tuple(nose_xy)
                    state.movement_tier = get_movement_tier(disp, SLOW_THRESHOLD, FAST_THRESHOLD)
                    per_person_movement.append(disp)

                    if scrs[COCO_LEFT_EYE] > 0.3 and scrs[COCO_RIGHT_EYE] > 0.3:
                        state.inter_eye_px   = euclidean(kpts[COCO_LEFT_EYE], kpts[COCO_RIGHT_EYE])
                        state.face_size_tier = get_face_size_tier(
                            state.inter_eye_px, FAR_THRESHOLD, MEDIUM_THRESHOLD
                        )
                    else:
                        state.face_size_tier = "far"

                    t_fm0               = time.time()
                    face_crop_for_state = None
                    face_bbox_for_state = None

                    if state.face_size_tier != "far":
                        crop, x_off, y_off, crop_dims, _ = derive_face_crop(frame, kpts, scrs)
                        if crop is not None:
                            face_crop_for_state = crop
                            face_bbox_for_state = (
                                x_off, y_off,
                                x_off + crop_dims[0], y_off + crop_dims[1],
                            )
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                            result   = face_mesh.detect(mp_image)
                            if result.face_landmarks:
                                pts = project_landmarks(
                                    result.face_landmarks[0],
                                    x_off, y_off, crop_dims[0], crop_dims[1],
                                )
                                state.face_mesh_pts = pts
                                last_mesh_cache[i]  = pts
                    else:
                        state.face_mesh_pts = None
                    t_facemesh_total += time.time() - t_fm0

                    body_crop, bx1, by1, bx2, by2 = derive_body_crop(frame, kpts, scrs)
                    body_bbox  = (bx1, by1, bx2, by2) if body_crop is not None else None
                    confidence = compute_frame_confidence(scrs)

                    state.update_best(
                        frame_idx  = frame_idx,
                        confidence = confidence,
                        face_crop  = face_crop_for_state,
                        face_bbox  = face_bbox_for_state,
                        body_crop  = body_crop,
                        body_bbox  = body_bbox,
                    )

                if per_person_movement:
                    movement_tier = get_movement_tier(
                        max(per_person_movement), SLOW_THRESHOLD, FAST_THRESHOLD
                    )
                    if movement_adaptive:
                        current_N = SKIP_N[movement_tier]

            prev_gray = curr_gray.copy()

            if full_frame_count % TIMING_INTERVAL == 0:
                n = max(full_frame_count, 1)
                s = max(skip_frame_count, 1)
                f = max(frame_idx, 1)
                print(
                    f"[F{frame_idx:4d}] "
                    f"Det: {t_det_total/n*1000:5.1f}ms | "
                    f"Pose: {t_pose_total/n*1000:5.1f}ms | "
                    f"FaceMesh: {t_facemesh_total/n*1000:5.1f}ms | "
                    f"OF-body: {t_of_body_total/s*1000:4.1f}ms | "
                    f"OF-face: {t_of_face_total/s*1000:4.1f}ms | "
                    f"Blur: {t_blur_total/f*1000:4.1f}ms | "
                    f"N={current_N} People={len(keypoints) if keypoints is not None else 0}"
                )

        else:
            skip_frame_count += 1

            if last_keypoints is not None and len(last_keypoints) > 0:
                tracked_keypoints = []

                for i in range(len(last_keypoints)):
                    old_pts = last_keypoints[i][:, :2].astype(np.float32).reshape(-1, 1, 2)
                    tob0 = time.time()
                    new_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, old_pts, None, **LK_PARAMS
                    )
                    t_of_body_total += time.time() - tob0

                    tracked_kpts = last_keypoints[i].copy()
                    tracked_kpts[:, :2] = new_pts.reshape(-1, 2)
                    tracked_keypoints.append(tracked_kpts)

                    if i in person_states and person_states[i].face_mesh_pts is not None:
                        old_face_pts = np.array(
                            person_states[i].face_mesh_pts, dtype=np.float32
                        ).reshape(-1, 1, 2)
                        tof0 = time.time()
                        new_face_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray, curr_gray, old_face_pts, None, **LK_PARAMS
                        )
                        t_of_face_total += time.time() - tof0
                        person_states[i].face_mesh_pts = [
                            (int(pt[0][0]), int(pt[0][1])) for pt in new_face_pts
                        ]

                keypoints      = np.array(tracked_keypoints)
                scores         = last_scores
                last_keypoints = keypoints

            prev_gray = curr_gray.copy()

        tb0 = time.time()
        if blur_bodies and keypoints is not None and len(keypoints) > 0:
            all_face_mesh_pts = [
                person_states[i].face_mesh_pts if i in person_states else None
                for i in range(len(keypoints))
            ]
            annotated = blur_all_persons(annotated, keypoints, scores, all_face_mesh_pts)
        t_blur_total += time.time() - tb0

        td0 = time.time()
        if draw_enabled and keypoints is not None and len(keypoints) > 0:
            annotated = draw_skeleton(annotated, keypoints, scores, kpt_thr=0.3)
            for i in range(len(keypoints)):
                if i in person_states and person_states[i].face_mesh_pts:
                    draw_face_mesh_pts(annotated, person_states[i].face_mesh_pts)
        t_draw_total += time.time() - td0

        if not benchmark:
            now     = time.time()
            elapsed = max(now - prev_time, 1e-6)
            prev_time = now
            fps_history.append(1.0 / elapsed)
            if len(fps_history) > FPS_WINDOW:
                fps_history.pop(0)
            fps_display = sum(fps_history) / len(fps_history)

            n_detected = len(keypoints) if keypoints is not None else 0
            frame_type = "SKIP(LK)" if not is_full_frame else "FULL"
            for j, line in enumerate([
                f"FPS: {fps_display:.1f}",
                f"Frame: {frame_type}",
                f"People: {n_detected}",
                f"Movement: {movement_tier}",
                f"Skip-N: {current_N}",
            ]):
                cv2.putText(annotated, line, (10, 35 + j * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

        tw0 = time.time()
        if out is not None:
            out.write(annotated)
        t_write_total += time.time() - tw0

        if not headless:
            cv2.imshow("bodySITARA", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser quit.")
                break

        frame_idx += 1

    print(f"\n  Flushing {len(person_states)} remaining active stream(s)...")
    for idx, state in person_states.items():
        enc_t, emb_t = state.flush_to_disk()
        t_encrypt_total += enc_t
        t_embed_total   += emb_t
        streams_flushed += 1
        print(f"  [STREAM] Person {idx} (EOV) → "
              f"stream {state.stream_id[:8]}... flushed "
              f"(enc={enc_t*1000:.1f}ms emb={emb_t*1000:.1f}ms)")
    person_states.clear()

    cap.release()
    if out is not None:
        out.release()
    if not headless:
        cv2.destroyAllWindows()
    face_mesh.close()

    total_time = time.time() - loop_start
    n  = max(full_frame_count, 1)
    s  = max(skip_frame_count, 1)
    f  = max(frame_idx, 1)
    sf = max(streams_flushed, 1)
    avg_fps = frame_idx / total_time

    print("\n" + "=" * 60)
    print("  FINAL TIMING SUMMARY")
    print("=" * 60)
    print(f"Benchmark mode      : {benchmark}")
    print(f"Movement adaptive   : {movement_adaptive}")
    print(f"Skip-N config       : {SKIP_N}")
    print(f"Total frames        : {frame_idx}")
    print(f"Full inf frames     : {full_frame_count}  ({full_frame_count/f*100:.1f}% of total)")
    print(f"Skip frames (LK)    : {skip_frame_count}  ({skip_frame_count/f*100:.1f}% of total)")
    print(f"Streams flushed     : {streams_flushed}")
    print(f"Total time          : {total_time:.1f}s")
    print(f"Average FPS         : {avg_fps:.2f}")
    print()
    print(f"── Per-component (benchmark-clean) ──────────────────")
    print(f"Avg Det/full frame  : {t_det_total      / n * 1000:.1f}ms")
    print(f"Avg Pose/full frame : {t_pose_total     / n * 1000:.1f}ms")
    print(f"Avg FaceMesh/full   : {t_facemesh_total / n * 1000:.1f}ms")
    print(f"Avg OF-body/skip    : {t_of_body_total  / s * 1000:.1f}ms")
    print(f"Avg OF-face/skip    : {t_of_face_total  / s * 1000:.1f}ms")
    print(f"Avg Blur/frame      : {t_blur_total     / f * 1000:.1f}ms")
    if not benchmark:
        print(f"Avg Draw/frame      : {t_draw_total     / f * 1000:.1f}ms")
        print(f"Avg Write/frame     : {t_write_total    / f * 1000:.1f}ms")
        print(f"Avg Enc/stream      : {t_encrypt_total  / sf * 1000:.1f}ms")
        print(f"Avg Embed/stream    : {t_embed_total    / sf * 1000:.1f}ms")

    if csv_out is not None:
        file_exists = os.path.isfile(csv_out)
        with open(csv_out, 'a', newline='') as csvfile:
            fieldnames = [
                'input_file', 'benchmark', 'movement_adaptive', 'skip_n',
                'total_frames', 'full_frames', 'skip_frames',
                'full_pct', 'skip_pct', 'avg_fps',
                'det_ms', 'pose_ms', 'facemesh_ms',
                'of_body_ms', 'of_face_ms', 'blur_ms',
                'draw_ms', 'write_ms',
                'streams_flushed', 'total_time_s',
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'input_file'       : os.path.basename(input_path),
                'benchmark'        : benchmark,
                'movement_adaptive': movement_adaptive,
                'skip_n'           : skip_n if not movement_adaptive else 'adaptive',
                'total_frames'     : frame_idx,
                'full_frames'      : full_frame_count,
                'skip_frames'      : skip_frame_count,
                'full_pct'         : round(full_frame_count / f * 100, 1),
                'skip_pct'         : round(skip_frame_count / f * 100, 1),
                'avg_fps'          : round(avg_fps, 2),
                'det_ms'           : round(t_det_total      / n * 1000, 1),
                'pose_ms'          : round(t_pose_total     / n * 1000, 1),
                'facemesh_ms'      : round(t_facemesh_total / n * 1000, 1),
                'of_body_ms'       : round(t_of_body_total  / s * 1000, 1),
                'of_face_ms'       : round(t_of_face_total  / s * 1000, 1),
                'blur_ms'          : round(t_blur_total      / f * 1000, 1),
                'draw_ms'          : round(t_draw_total      / f * 1000, 1),
                'write_ms'         : round(t_write_total     / f * 1000, 1),
                'streams_flushed'  : streams_flushed,
                'total_time_s'     : round(total_time, 2),
            })
        print(f"\nMetrics appended → {csv_out}")

    if save_video and out is not None:
        print(f"\nOutput video: {output_path}")
