import os
import json
import time
import uuid
import numpy as np
import cv2

from .encryption import rsa_encrypt_key, aes_gcm_encrypt, encode_crop
from .pose import LK_PARAMS


class PersonState:
    """Tracks one person stream across frames."""

    def __init__(self, rsa_public_key, enc_output_dir: str,
                 embedder, benchmark: bool = False):

        self.prev_nose       = None
        self.movement_tier   = "medium"
        self.face_size_tier  = "medium"
        self.face_mesh_pts   = None
        self.inter_eye_px    = 0.0
        self.stream_id       = str(uuid.uuid4())
        self.aes_key         = os.urandom(16)

        self.best_confidence = 0.0
        self.best_face_crop  = None
        self.best_body_crop  = None
        self.best_frame_idx  = -1
        self.best_bbox_face  = None
        self.best_bbox_body  = None

        self._embedder       = embedder
        self._enc_output_dir = enc_output_dir
        self._benchmark      = benchmark

        if not benchmark:
            wrapped_key = rsa_encrypt_key(self.aes_key, rsa_public_key)
            key_path    = os.path.join(enc_output_dir, f"stream_{self.stream_id}.key")
            with open(key_path, 'wb') as f:
                f.write(wrapped_key)

    def update_best(self, frame_idx, confidence,
                    face_crop, face_bbox, body_crop, body_bbox):
        if confidence > self.best_confidence:
            self.best_confidence = confidence
            self.best_face_crop  = face_crop.copy() if face_crop is not None else None
            self.best_body_crop  = body_crop.copy() if body_crop is not None else None
            self.best_frame_idx  = frame_idx
            self.best_bbox_face  = face_bbox
            self.best_bbox_body  = body_bbox

    def flush_to_disk(self) -> tuple[float, float]:
        if self._benchmark:
            return 0.0, 0.0

        if self.best_body_crop is None and self.best_face_crop is None:
            return 0.0, 0.0

        t_emb0    = time.time()
        embedding = None
        if self._embedder is not None and self.best_face_crop is not None:
            embedding = self._embedder.extract(self.best_face_crop)
        embed_time = time.time() - t_emb0

        t_enc0 = time.time()

        face_bytes = encode_crop(self.best_face_crop) if self.best_face_crop is not None else b""
        body_bytes = encode_crop(self.best_body_crop) if self.best_body_crop is not None else b""
        meta_bytes = json.dumps({
            "stream_id"    : self.stream_id,
            "frame_idx"    : self.best_frame_idx,
            "confidence"   : round(self.best_confidence, 4),
            "bbox_body"    : self.best_bbox_body,
            "bbox_face"    : self.best_bbox_face,
            "has_embedding": embedding is not None,
            "embedding_dim": 512 if embedding is not None else 0,
        }).encode("utf-8")
        emb_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else b""

        def _enc(data: bytes):
            if not data:
                return b"", b""
            return aes_gcm_encrypt(self.aes_key, data)

        face_nonce, face_ct = _enc(face_bytes)
        body_nonce, body_ct = _enc(body_bytes)
        meta_nonce, meta_ct = _enc(meta_bytes)
        emb_nonce,  emb_ct  = _enc(emb_bytes)

        packet_path = os.path.join(
            self._enc_output_dir, f"stream_{self.stream_id}.packet"
        )

        def write_blob(f, nonce, ct):
            f.write(len(nonce).to_bytes(4, "little"))
            f.write(nonce)
            f.write(len(ct).to_bytes(4, "little"))
            f.write(ct)

        with open(packet_path, "wb") as f:
            write_blob(f, face_nonce, face_ct)
            write_blob(f, body_nonce, body_ct)
            write_blob(f, meta_nonce, meta_ct)
            write_blob(f, emb_nonce,  emb_ct)

        enc_time = time.time() - t_enc0
        return enc_time, embed_time


def propagate_bboxes(last_bboxes, prev_gray, curr_gray):
    if last_bboxes is None or len(last_bboxes) == 0:
        return last_bboxes
    centers = []
    for bbox in last_bboxes:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        centers.append([[cx, cy]])
    old_pts  = np.array(centers, dtype=np.float32)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, old_pts, None, **LK_PARAMS
    )
    updated = last_bboxes.copy().astype(np.float32)
    for i, (old_c, new_c, st) in enumerate(zip(old_pts, new_pts, status)):
        if st[0] == 1:
            dx = new_c[0][0] - old_c[0][0]
            dy = new_c[0][1] - old_c[0][1]
            updated[i][0] += dx
            updated[i][1] += dy
            updated[i][2] += dx
            updated[i][3] += dy
    return updated
