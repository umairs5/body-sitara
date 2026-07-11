import cv2
import time
import os
import csv
import json
import uuid
import urllib.request
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
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
from .blur_seg import SelfieSegBlur, bbox_region_mask
from .blur_mobilesam import MobileSAMBlur, bboxes_from_keypoints
from .blur_yoloseg import YOLOSegBlur
from .face_canonical import FaceCanonicalizer, CANONICAL_SIZE
from .face_canonical_v2 import FaceCanonicalizerV2, P_SMILE
from .tracking import PersonState, propagate_bboxes
from .encryption import generate_ttp_keypair
from .embedding import EmbeddingExtractor, EDGEFACE_ONNX_PATH
from .detector_patch import apply_detector_patch

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

# Consecutive full frames a stale selfie-seg gate region may be reused for
# when det+pose both fail to produce any usable bbox on a given frame (e.g.
# transient motion-blur pose dropout). Small enough that a person who truly
# leaves the frame stops being blurred within a fraction of a second.
GATE_REGION_TTL = 5

# Minimum ratio (this box's area / largest box's area in the same frame) for
# a YOLOX detection to be treated as a trackable person, when more than one
# box is detected. Filters out distant background bystanders relative to
# whoever is dominant/closest to camera in that frame, without relying on a
# fragile absolute pixel-size cutoff (subject box size varies a lot with
# distance from camera across a clip).
MIN_BOX_AREA_RATIO = 0.20


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
    anonymizer        = "convexhull",   # "convexhull" | "selfie_seg0" | "selfie_seg1" | "mobilesam" | "yoloseg"
    export_dir         = None,   # dense per-person export mode (opt-in, additive -- see export_tracking.py)
    dense_export        = False,
    export_people        = 3,
    export_diagnostics  = False,
):
    if benchmark:
        save_video = False
        headless   = True

    # Dense export needs literal every-frame accuracy (no skip-frame
    # optical-flow propagation) and a consistent segmentation backend to
    # populate mask.mp4/raw_seg_mask.mp4/gate_region.mp4. Both are forced
    # only when export_dir is actually set, so default (export_dir=None)
    # calls are entirely unaffected.
    export_enabled = export_dir is not None
    if export_enabled:
        dense_export = True
        os.makedirs(export_dir, exist_ok=True)
        if not anonymizer.startswith("selfie_seg"):
            print(f"  NOTE: export mode forces anonymizer='selfie_seg1' (was '{anonymizer}')")
            anonymizer = "selfie_seg1"

    SKIP_N = SKIP_N_DEFAULTS.copy()
    if not movement_adaptive:
        SKIP_N["slow"]   = skip_n
        SKIP_N["medium"] = skip_n
        SKIP_N["fast"]   = skip_n
    if dense_export:
        SKIP_N = {"slow": 1, "medium": 1, "fast": 1}

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
        print("\n[0/4] Benchmark mode -- skipping RSA keypair generation")
        ttp_private_key, ttp_public_key = None, None

    # [1] RTMPose
    print("\n[1/4] Loading RTMPose (YOLOX-Nano + RTMPose-T)...")
    apply_detector_patch()
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
        print("\n[3/4] Benchmark mode -- skipping EdgeFace embedding model")
        embedder = None

    # [3b] Anonymizer backend
    selfie_seg       = None
    mobile_sam       = None
    yolo_seg         = None
    face_canonicalizer = None
    if anonymizer.startswith("selfie_seg"):
        _models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        _model_file = ("selfie_segmenter_landscape.tflite"
                       if anonymizer == "selfie_seg1"
                       else "selfie_segmenter.tflite")
        _model_path = os.path.join(_models_dir, _model_file)
        print(f"\n[3b] Loading MediaPipe SelfieSegmentation ({_model_file})...")
        selfie_seg = SelfieSegBlur(model_path=_model_path)
        print(f"     Anonymizer: {anonymizer}")
        print(f"\n[3c] Loading FaceCanonicalizer (expression signal)...")
        face_canonicalizer = FaceCanonicalizer(model_path='face_landmarker.task')
    elif anonymizer == "mobilesam":
        _ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mobile_sam.pt")
        print(f"\n[3b] Loading MobileSAM (ViT-Tiny)...")
        mobile_sam = MobileSAMBlur(checkpoint_path=_ckpt, device="cpu")
        print(f"     Anonymizer: mobilesam")
    elif anonymizer == "yoloseg":
        print(f"\n[3b] Loading YOLOv8-seg-nano (instance segmentation)...")
        yolo_seg = YOLOSegBlur(model_name="yolov8n-seg.pt", infer_size=320, conf=0.4)
        print(f"     Anonymizer: yoloseg")
    else:
        print(f"\n[3b] Anonymizer: convexhull")

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

    # Three-panel output when canonicalizer is active: original | blurred | canonical
    DISPLAY_H  = 640
    _out_w     = (DISPLAY_H * 2 + DISPLAY_H) if face_canonicalizer is not None else width
    _out_h     = DISPLAY_H                   if face_canonicalizer is not None else height

    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps_input, (_out_w, _out_h))
        if not out.isOpened():
            print("  Warning: Could not open VideoWriter -- video will not be saved.")
            out = None

    # --- Dense per-person export setup (opt-in; no-op when export_dir is None) ---
    EXPORT_FACE_SIZE = 512
    slot_tracker              = None
    export_kp_rows            = None
    export_bbox_rows          = None
    export_face_param_rows    = None   # parametric, identity-free signal -- the safe-to-transmit face export
    export_valid_smiles       = None   # smile scalar from genuinely-detected (non-held-over) frames only, for the clip baseline
    export_last_face_crop     = None   # transient only: feeds extract_params() + optional diagnostic write, never itself "the" export
    export_last_face_params   = None   # last-good parametric scalars, held across brief absences (mirrors the old face-crop hold-over)
    export_slot_stream_id     = None
    export_face_canon         = None
    export_face_writers       = None   # diagnostics-only now (see export_diagnostics gating below)
    export_rtm_writer         = None
    export_mask_writer        = None
    export_raw_mask_writer    = None
    export_gate_writer        = None
    export_bbox_overlay_writer = None
    export_clip_id            = None

    if export_enabled:
        from .export_tracking import ExportSlotTracker
        slot_tracker            = ExportSlotTracker(export_people, width, height)
        export_kp_rows          = [[] for _ in range(export_people)]
        export_bbox_rows        = [[] for _ in range(export_people)]
        export_face_param_rows  = [[] for _ in range(export_people)]
        export_valid_smiles     = [[] for _ in range(export_people)]
        export_last_face_crop   = [None] * export_people
        export_last_face_params = [None] * export_people
        export_slot_stream_id   = [str(uuid.uuid4()) for _ in range(export_people)]
        export_face_canon       = FaceCanonicalizerV2(model_path='face_landmarker.task')
        export_clip_id          = str(uuid.uuid4())

        _exp_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        export_rtm_writer  = cv2.VideoWriter(os.path.join(export_dir, "output_rtm.mp4"),
                                              _exp_fourcc, fps_input, (width, height))
        export_mask_writer = cv2.VideoWriter(os.path.join(export_dir, "mask.mp4"),
                                              _exp_fourcc, fps_input, (width, height), isColor=False)
        if export_diagnostics:
            # face_crops_p{i}.mp4 is a RAW, unblurred face crop -- useful for local
            # debugging (visually checking what face_params_p{i}.npy was derived
            # from) but it must never be treated as part of the safe-to-transmit
            # bundle. Gated behind export_diagnostics for exactly that reason,
            # same as the other raw/debug-only outputs below.
            export_face_writers = [
                cv2.VideoWriter(os.path.join(export_dir, f"face_crops_p{i}.mp4"),
                                 _exp_fourcc, fps_input, (EXPORT_FACE_SIZE, EXPORT_FACE_SIZE))
                for i in range(export_people)
            ]
            export_raw_mask_writer = cv2.VideoWriter(
                os.path.join(export_dir, "raw_seg_mask.mp4"),
                _exp_fourcc, fps_input, (width, height), isColor=False)
            export_gate_writer = cv2.VideoWriter(
                os.path.join(export_dir, "gate_region.mp4"),
                _exp_fourcc, fps_input, (width, height), isColor=False)
            export_bbox_overlay_writer = cv2.VideoWriter(
                os.path.join(export_dir, "bbox_overlay.mp4"),
                _exp_fourcc, fps_input, (width, height))
        print(f"\n  Dense export enabled -> {export_dir}  "
              f"(slots={export_people}, diagnostics={export_diagnostics})")

    frame_idx        = 0
    full_frame_count = 0
    skip_frame_count = 0
    person_states    = {}
    active_indices   = set()
    current_N        = SKIP_N["medium"]
    movement_tier    = "medium"
    last_keypoints        = None
    last_scores           = None
    last_bboxes           = None
    last_scaled_bboxes    = None   # YOLOX bboxes scaled to frame resolution for MobileSAM
    last_seg_mask         = None   # last selfie-seg mask (bool H×W), propagated on skip frames
    seg_mask_keypoints    = None   # keypoints at the time last_seg_mask was computed/warped
    last_gate_region      = None   # last non-empty bbox_region_mask, held over brief det+pose dropout
    gate_region_stale_for = 0      # consecutive full frames since last_gate_region was refreshed
    last_mesh_cache       = {}
    last_canonical_face   = None   # last canonical expression image (CANONICAL_SIZE×CANONICAL_SIZE)
    prev_gray        = None

    prev_time   = time.time()
    fps_history = []
    FPS_WINDOW  = 30

    t_det_total      = 0.0
    t_pose_total     = 0.0
    t_facemesh_total = 0.0
    t_of_body_total  = 0.0
    t_of_face_total  = 0.0
    t_seg_total      = 0.0   # selfie-seg inference (runs parallel to det+pose)
    t_canonical_total = 0.0  # face canonicalizer (every frame)
    t_blur_total     = 0.0   # mask apply + warp only
    t_draw_total     = 0.0
    t_write_total    = 0.0
    t_encrypt_total  = 0.0
    t_embed_total    = 0.0

    # Thread pools for parallelism (all C++ backends release the GIL)
    _seg_pool   = ThreadPoolExecutor(max_workers=1)  # selfie seg ∥ det+pose
    _lk_pool    = ThreadPoolExecutor(max_workers=2)  # body LK ∥ face LK
    _write_pool = ThreadPoolExecutor(max_workers=1)  # async VideoWriter

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

            # yolo_seg runs BEFORE det+pose so ONNX Runtime CPU threads are idle.
            # PyTorch and OnnxRuntime both use all CPU threads; running concurrently
            # (or even back-to-back while ORT threads linger) causes severe slowdown.
            if yolo_seg is not None and blur_bodies:
                _t_yolo0 = time.time()
                last_seg_mask = yolo_seg.get_mask(frame)
                t_seg_total += time.time() - _t_yolo0

            # selfie_seg: thread pool (TFLite releases GIL → true parallel with det+pose)
            if selfie_seg is not None and blur_bodies:
                _future_seg = _seg_pool.submit(selfie_seg.get_mask, frame, 256)
            else:
                _future_seg = None
            _t_seg0 = time.time()

            t0     = time.time()
            bboxes = body.det_model(infer_frame)
            t_det_total += time.time() - t0

            # Reject boxes too small relative to the frame's largest detection
            # to plausibly be the video's subject -- e.g. distant background
            # bystanders in a hallway shot that flicker in/out of detection as
            # they walk and otherwise get tracked/encrypted as short-lived
            # phantom person streams. An absolute pixel threshold doesn't work
            # here: the subject's own box size varies hugely with distance
            # from camera, so a bystander can be taller in pixels than the
            # subject is in another frame. Relative-to-largest adapts to that.
            if bboxes is not None and len(bboxes) > 1:
                box_area  = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                max_area  = box_area.max()
                bboxes    = bboxes[box_area >= MIN_BOX_AREA_RATIO * max_area]

            last_bboxes = bboxes
            # scale YOLOX bboxes from infer_frame space to original frame space
            if bboxes is not None and len(bboxes) > 0:
                sb = bboxes.copy().astype(float)
                sb[:, 0] *= kp_scale_x; sb[:, 2] *= kp_scale_x
                sb[:, 1] *= kp_scale_y; sb[:, 3] *= kp_scale_y
                last_scaled_bboxes = sb[:, :4].tolist()
            else:
                last_scaled_bboxes = []

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
                    print(f"  [STREAM] Person {dep_idx} departed -> "
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
                        print(f"  [STREAM] Person {i} appeared -> "
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
                            if face_canonicalizer is not None:
                                # selfie_seg mode: canonicalizer handles face detection
                                # (only run on person 0 — primary subject)
                                if i == 0:
                                    tc0 = time.time()
                                    cf = face_canonicalizer.get_canonical_face(crop)
                                    t_canonical_total += time.time() - tc0
                                    if cf is not None:
                                        last_canonical_face = cf
                            else:
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

            if export_enabled:
                # Stable left-to-right slots are a separate concept from
                # person_states above: person_states is keyed by raw
                # detection index (reused/reshuffled on identity churn),
                # while export needs a fixed small set of identities that
                # keep referring to the same physical person for the whole
                # clip. Detections beyond export_people, or too low-
                # confidence to derive any bbox, simply aren't exported --
                # they're still blurred normally via the existing mask path.
                detections = []
                det_bboxes = {}
                if keypoints is not None and len(keypoints) > 0:
                    for i in range(len(keypoints)):
                        db = bboxes_from_keypoints(
                            [keypoints[i]], [scores[i]], height, width, padding=40
                        )
                        if db:
                            x1, y1, x2, y2 = db[0]
                            detections.append((i, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
                            det_bboxes[i] = [float(x1), float(y1), float(x2), float(y2)]

                slot_matches = slot_tracker.assign(detections)

                for s in range(export_people):
                    crop_s = None
                    if s in slot_matches:
                        di = slot_matches[s]
                        kpts_s, scrs_s = keypoints[di], scores[di]
                        export_kp_rows[s].append(
                            np.concatenate(
                                [kpts_s[:, :2], scrs_s[:, None]], axis=1
                            ).astype(np.float32)
                        )
                        export_bbox_rows[s].append([det_bboxes[di]])

                        crop_s, _, _, _, _ = derive_face_crop(frame, kpts_s, scrs_s)
                        if crop_s is not None:
                            export_last_face_crop[s] = crop_s
                            params_s = export_face_canon.extract_params(crop_s)
                            if params_s is not None:
                                export_last_face_params[s] = params_s
                                export_valid_smiles[s].append(float(params_s[P_SMILE]))
                    else:
                        export_kp_rows[s].append(np.zeros((17, 3), dtype=np.float32))
                        export_bbox_rows[s].append([])

                    # face_params_p{i}.npy is the safe-to-transmit identity-free
                    # signal -- last-good scalars held across brief absences, same
                    # hold-over behaviour the old raw-crop export used, just now
                    # applied to 12 numbers instead of a face image.
                    if export_last_face_params[s] is not None:
                        export_face_param_rows[s].append(export_last_face_params[s].copy())
                    else:
                        export_face_param_rows[s].append(np.zeros(12, dtype=np.float32))

                    if export_diagnostics:
                        if export_last_face_crop[s] is not None:
                            face_frame = cv2.resize(
                                export_last_face_crop[s], (EXPORT_FACE_SIZE, EXPORT_FACE_SIZE)
                            )
                        else:
                            face_frame = np.zeros(
                                (EXPORT_FACE_SIZE, EXPORT_FACE_SIZE, 3), dtype=np.uint8
                            )
                        export_face_writers[s].write(face_frame)

                if export_diagnostics:
                    overlay = frame.copy()
                    if last_scaled_bboxes:
                        for bbox in last_scaled_bboxes:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    export_bbox_overlay_writer.write(overlay)

            # Update seg_mask_keypoints now that keypoints are known (yolo_seg ran before det)
            if yolo_seg is not None and blur_bodies:
                seg_mask_keypoints = keypoints.copy() if keypoints is not None and len(keypoints) > 0 else None

            # Collect selfie-seg result (may already be done; blocks only if det+pose faster)
            if _future_seg is not None:
                raw_seg_mask       = _future_seg.result()
                seg_mask_keypoints = keypoints.copy() if keypoints is not None and len(keypoints) > 0 else None
                t_seg_total       += time.time() - _t_seg0  # wall-time waited, not thread time

                # Selfie-seg segments the whole frame with no notion of "where
                # RTMPose actually detected a person" -- it will happily paint
                # over background clutter (tree branches, textured walls) that
                # merely looks person-shaped. Gate the mask to the union of
                # detected person regions so nothing outside those regions can
                # ever be blurred; if no one was detected, apply no mask at all.
                #
                # last_scaled_bboxes comes from YOLOX and can be [] even when
                # keypoints is non-empty: rtmlib's RTMPose silently falls back
                # to a whole-frame box when given zero detector boxes, so it
                # still produces a pose for a real, visible person the
                # detector merely missed on this frame. Gating on the empty
                # detector boxes alone would wipe the mask for a real person
                # still being tracked -- fall back to a keypoint-derived
                # region in that case instead of gating with nothing.
                #
                # But the fallback pose itself can also be low-confidence
                # (e.g. mid-stride motion blur): every keypoint scores below
                # kpt_thr, so bboxes_from_keypoints returns [] too. Both
                # signals failing on the same frame doesn't mean the person
                # vanished -- it means det+pose had a bad frame. Hold the
                # last known-good gate region for a short TTL instead of
                # collapsing to an all-False mask.
                if raw_seg_mask is not None:
                    if last_scaled_bboxes:
                        gate_bboxes = last_scaled_bboxes
                    elif keypoints is not None and len(keypoints) > 0:
                        gate_bboxes = bboxes_from_keypoints(
                            keypoints, scores, height, width, padding=40
                        )
                    else:
                        gate_bboxes = []

                    fresh_gate = bool(gate_bboxes)
                    if gate_bboxes:
                        region = bbox_region_mask(gate_bboxes, height, width, padding=40)
                        last_gate_region      = region
                        gate_region_stale_for = 0
                    elif last_gate_region is not None and gate_region_stale_for < GATE_REGION_TTL:
                        region = last_gate_region
                        gate_region_stale_for += 1
                    else:
                        region = bbox_region_mask([], height, width, padding=40)
                        last_gate_region       = None
                        gate_region_stale_for  = 0

                    last_seg_mask = raw_seg_mask & region

                    # Selfie-seg can also fail outright on a confidently-detected
                    # person: small/angled figures against a similarly-toned
                    # background sometimes make the confidence field come back
                    # near all-zero even inside a correct, fresh detector bbox.
                    # Gating can't help there since raw_seg_mask has nothing to
                    # gate. If that happens on a fresh (non-stale) detection,
                    # fall back to solid-filling the tight bbox itself -- a
                    # coarser silhouette beats leaving a real person unmasked.
                    if fresh_gate and last_seg_mask.sum() < 0.02 * region.sum():
                        last_seg_mask = bbox_region_mask(gate_bboxes, height, width, padding=0)

                    if export_enabled and export_diagnostics:
                        export_raw_mask_writer.write((raw_seg_mask.astype(np.uint8)) * 255)
                        export_gate_writer.write((region.astype(np.uint8)) * 255)
                else:
                    last_seg_mask = None
                    if export_enabled and export_diagnostics:
                        export_raw_mask_writer.write(np.zeros((height, width), dtype=np.uint8))
                        export_gate_writer.write(np.zeros((height, width), dtype=np.uint8))

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
                n_persons = len(last_keypoints)

                # --- body LK tasks (one per person) ---
                def _body_lk(i):
                    old = last_keypoints[i][:, :2].astype(np.float32).reshape(-1, 1, 2)
                    new, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old, None, **LK_PARAMS)
                    return i, new

                # Face LK only needed for convexhull blur (face_mesh_pts → hull)
                # When canonicalizer active, skip face LK and run canonicalizer instead
                face_mesh_inputs = {}
                if face_canonicalizer is None:
                    for i in range(n_persons):
                        if i in person_states and person_states[i].face_mesh_pts is not None:
                            face_mesh_inputs[i] = np.array(
                                person_states[i].face_mesh_pts, dtype=np.float32
                            ).reshape(-1, 1, 2)

                def _face_lk(i, old_face):
                    new, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old_face, None, **LK_PARAMS)
                    return i, new

                # Submit body LK and face LK (if needed) in parallel
                t_lk0 = time.time()
                body_futures = [_lk_pool.submit(_body_lk, i) for i in range(n_persons)]
                face_futures = {i: _lk_pool.submit(_face_lk, i, old)
                                for i, old in face_mesh_inputs.items()}

                # Collect body results
                tracked_keypoints = [None] * n_persons
                for fut in body_futures:
                    i, new_pts = fut.result()
                    tk = last_keypoints[i].copy()
                    tk[:, :2] = new_pts.reshape(-1, 2)
                    tracked_keypoints[i] = tk
                t_of_body_total += time.time() - t_lk0

                # Collect face LK results (convexhull mode only)
                t_face0 = time.time()
                for i, fut in face_futures.items():
                    _, new_face = fut.result()
                    person_states[i].face_mesh_pts = [
                        (int(pt[0][0]), int(pt[0][1])) for pt in new_face
                    ]
                t_of_face_total += time.time() - t_face0

                # Canonical runs only on full frames (hidden in selfie-seg parallel wait).
                # Skip frames reuse last_canonical_face — expression changes ~10fps is enough.

                keypoints      = np.array(tracked_keypoints)
                scores         = last_scores
                last_keypoints = keypoints

            prev_gray = curr_gray.copy()

        tb0 = time.time()
        if blur_bodies:
            if mobile_sam is not None and keypoints is not None and len(keypoints) > 0:
                # Full frames: use YOLOX bboxes (more accurate, includes head).
                # Skip frames: fall back to keypoint-derived bboxes.
                if is_full_frame and last_scaled_bboxes:
                    sam_bboxes = last_scaled_bboxes
                else:
                    sam_bboxes = bboxes_from_keypoints(
                        keypoints, scores, height, width, padding=80
                    )
                annotated = mobile_sam.blur_frame(annotated, sam_bboxes)
            elif selfie_seg is not None or yolo_seg is not None:
                # Full-frame mask already fetched in parallel above.
                # Skip frames: warp stored mask by affine from keypoint motion.
                if not is_full_frame and last_seg_mask is not None \
                        and seg_mask_keypoints is not None \
                        and keypoints is not None and len(keypoints) > 0:
                    old_pts = seg_mask_keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    new_pts = keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    M, _ = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC)
                    if M is not None:
                        last_seg_mask = cv2.warpAffine(
                            last_seg_mask.astype(np.uint8), M, (width, height),
                            flags=cv2.INTER_NEAREST
                        ).astype(bool)
                    seg_mask_keypoints = keypoints.copy()
                if last_seg_mask is not None:
                    applier = selfie_seg if selfie_seg is not None else yolo_seg
                    annotated = applier.apply_mask(annotated, last_seg_mask)
            elif keypoints is not None and len(keypoints) > 0:
                all_face_mesh_pts = [
                    person_states[i].face_mesh_pts if i in person_states else None
                    for i in range(len(keypoints))
                ]
                annotated = blur_all_persons(annotated, keypoints, scores, all_face_mesh_pts)
        t_blur_total += time.time() - tb0

        if export_enabled:
            # Clean (debug-overlay-free) anonymized frame + final mask, for
            # a downstream machine consumer rather than a human viewer --
            # written here, before the skeleton/bbox debug drawing below.
            export_rtm_writer.write(annotated)
            if last_seg_mask is not None:
                export_mask_writer.write((last_seg_mask.astype(np.uint8)) * 255)
            else:
                export_mask_writer.write(np.zeros((height, width), dtype=np.uint8))

        td0 = time.time()
        if draw_enabled and keypoints is not None and len(keypoints) > 0:
            annotated = draw_skeleton(annotated, keypoints, scores, kpt_thr=0.3)
            for i in range(len(keypoints)):
                if i in person_states and person_states[i].face_mesh_pts:
                    draw_face_mesh_pts(annotated, person_states[i].face_mesh_pts)
        if draw_enabled and last_scaled_bboxes:
            for bbox in last_scaled_bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
            if face_canonicalizer is not None:
                # Three panels: original | blurred | canonical
                orig_panel    = cv2.resize(frame,    (DISPLAY_H, DISPLAY_H))
                blurred_panel = cv2.resize(annotated, (DISPLAY_H, DISPLAY_H))
                canon_panel   = np.full((DISPLAY_H, DISPLAY_H, 3), (228, 225, 222), dtype=np.uint8)
                if last_canonical_face is not None:
                    cf_resized = cv2.resize(last_canonical_face, (DISPLAY_H, DISPLAY_H))
                    canon_panel[:] = cf_resized
                _label = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(orig_panel,    "ORIGINAL",   (8, 24), _label, 0.6, (255, 255, 255), 2)
                cv2.putText(blurred_panel, "BLURRED",    (8, 24), _label, 0.6, (255, 255, 255), 2)
                cv2.putText(canon_panel,   "EXPRESSION", (8, 24), _label, 0.6, (40,  40,  40),  2)
                combined = np.hstack([orig_panel, blurred_panel, canon_panel])
                _write_pool.submit(out.write, combined)
            else:
                _write_pool.submit(out.write, annotated.copy())
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
        print(f"  [STREAM] Person {idx} (EOV) -> "
              f"stream {state.stream_id[:8]}... flushed "
              f"(enc={enc_t*1000:.1f}ms emb={emb_t*1000:.1f}ms)")
    person_states.clear()

    # Flush async write queue before releasing VideoWriter
    _write_pool.shutdown(wait=True)
    _seg_pool.shutdown(wait=False)
    _lk_pool.shutdown(wait=False)

    cap.release()
    if out is not None:
        out.release()

    if export_enabled:
        manifest_slots = []
        for i in range(export_people):
            kp_arr = (np.stack(export_kp_rows[i], axis=0) if export_kp_rows[i]
                      else np.zeros((0, 17, 3), dtype=np.float32))
            np.save(os.path.join(export_dir, f"keypoints_p{i}.npy"), kp_arr)
            with open(os.path.join(export_dir, f"bboxes_p{i}.json"), "w") as f:
                json.dump(export_bbox_rows[i], f)

            fp_arr = (np.stack(export_face_param_rows[i], axis=0) if export_face_param_rows[i]
                      else np.zeros((0, 12), dtype=np.float32))
            np.save(os.path.join(export_dir, f"face_params_p{i}.npy"), fp_arr)

            # Smile baseline uses only genuinely-detected frames (export_valid_smiles),
            # never the held-over values in export_face_param_rows -- matches the
            # two-pass approach in scripts/test_face_canon_v2.py. Downstream
            # rendering applies this correction itself (FaceCanonicalizerV2.
            # set_smile_baseline() + render()) -- exported params are raw/uncorrected.
            smile_baseline = (float(np.median(export_valid_smiles[i]))
                               if export_valid_smiles[i] else 0.0)
            manifest_slots.append({
                "slot": i,
                "stream_id": export_slot_stream_id[i],
                "face_smile_baseline": smile_baseline,
                "frames_with_face": len(export_valid_smiles[i]),
            })

            if export_diagnostics:
                export_face_writers[i].release()

        with open(os.path.join(export_dir, "manifest.json"), "w") as f:
            json.dump({
                "clip_id": export_clip_id,
                "fps": fps_input,
                "width": width,
                "height": height,
                "num_slots": export_people,
                "total_frames": kp_arr.shape[0],
                "slots": manifest_slots,
            }, f, indent=2)

        export_rtm_writer.release()
        export_mask_writer.release()
        export_face_canon.close()
        if export_diagnostics:
            export_raw_mask_writer.release()
            export_gate_writer.release()
            export_bbox_overlay_writer.release()
        print(f"\n  Dense export written -> {export_dir}  "
              f"({kp_arr.shape[0]} frames x {export_people} slots)")

    if not headless:
        cv2.destroyAllWindows()
    face_mesh.close()
    if selfie_seg is not None:
        selfie_seg.close()
    if face_canonicalizer is not None:
        face_canonicalizer.close()
    # MobileSAM has no explicit close; PyTorch model is GC'd

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
    print(f"-- Per-component (benchmark-clean) ------------------")
    print(f"Avg Det/full frame  : {t_det_total      / n * 1000:.1f}ms")
    print(f"Avg Pose/full frame : {t_pose_total     / n * 1000:.1f}ms")
    print(f"Avg FaceMesh/full   : {t_facemesh_total / n * 1000:.1f}ms")
    print(f"Avg SelfieSeg/full  : {t_seg_total        / n * 1000:.1f}ms  (parallel w/ det+pose)")
    print(f"Avg Canonical/full  : {t_canonical_total / n * 1000:.1f}ms  (full frames only, reused on skip)")
    print(f"Avg OF-body/skip    : {t_of_body_total   / s * 1000:.1f}ms  (parallel w/ OF-face)")
    print(f"Avg OF-face/skip    : {t_of_face_total   / s * 1000:.1f}ms  (parallel w/ OF-body)")
    print(f"Avg Blur/frame      : {t_blur_total     / f * 1000:.1f}ms  (mask apply + warp only)")
    if not benchmark:
        print(f"Avg Draw/frame      : {t_draw_total     / f * 1000:.1f}ms")
        print(f"Avg Write/frame     : {t_write_total    / f * 1000:.1f}ms")
        print(f"Avg Enc/stream      : {t_encrypt_total  / sf * 1000:.1f}ms")
        print(f"Avg Embed/stream    : {t_embed_total    / sf * 1000:.1f}ms")

    if csv_out is not None:
        file_exists = os.path.isfile(csv_out)
        with open(csv_out, 'a', newline='') as csvfile:
            fieldnames = [
                'input_file', 'anonymizer', 'benchmark', 'movement_adaptive', 'skip_n',
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
                'anonymizer'       : anonymizer,
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
        print(f"\nMetrics appended -> {csv_out}")

    if save_video and out is not None:
        print(f"\nOutput video: {output_path}")
