"""
Stable left-to-right person-slot tracking for the dense per-frame export
mode (see process_video's export_dir/dense_export kwargs).

This is deliberately separate from tracking.py's PersonState: PersonState
tracks raw detection indices for the encryption/consent-recovery path,
which can and does change on identity churn (a departed person's index
gets reused by whoever appears next). Export slots instead need a small,
FIXED number of stable identities (e.g. slot 0/1/2) that keep referring to
the same physical person across the whole clip, tolerant of brief misses,
so a downstream per-slot signal stream (keypoints_p0.npy, etc.) doesn't
silently swap identities mid-file.
"""

import numpy as np


class ExportSlotTracker:
    def __init__(self, n_slots: int, frame_w: int, frame_h: int,
                 grace_frames: int = 30, gate_frac: float = 0.25):
        self.n_slots      = n_slots
        self.gate         = gate_frac * float(np.hypot(frame_w, frame_h))
        self.grace_frames = grace_frames
        self.center       = [None] * n_slots   # (cx, cy) or None
        self.missed       = [0] * n_slots
        self.active       = [False] * n_slots

    def assign(self, detections):
        """
        detections: list of (det_idx, (cx, cy)) for this frame's people.
        det_idx indexes that frame's keypoints/scores arrays.

        Returns {slot_idx: det_idx} for slots matched this frame. Slots
        not in the returned dict are either reserved-but-missed (still
        within grace) or genuinely free.
        """
        pairs = []
        for s in range(self.n_slots):
            if not self.active[s] or self.center[s] is None:
                continue
            for det_idx, c in detections:
                d = float(np.hypot(c[0] - self.center[s][0], c[1] - self.center[s][1]))
                if d <= self.gate:
                    pairs.append((d, s, det_idx, c))
        pairs.sort(key=lambda p: p[0])

        matched_slots = {}
        used_slots, used_det = set(), set()
        for d, s, det_idx, c in pairs:
            if s in used_slots or det_idx in used_det:
                continue
            matched_slots[s] = det_idx
            used_slots.add(s)
            used_det.add(det_idx)
            self.center[s] = c
            self.missed[s] = 0

        for s in range(self.n_slots):
            if self.active[s] and s not in used_slots:
                self.missed[s] += 1
                if self.missed[s] > self.grace_frames:
                    self.active[s] = False
                    self.center[s] = None
                    self.missed[s] = 0

        leftover = sorted(
            ((di, c) for di, c in detections if di not in used_det),
            key=lambda x: x[1][0],
        )
        free_slots = [s for s in range(self.n_slots) if not self.active[s]]
        for (det_idx, c), s in zip(leftover, free_slots):
            self.active[s] = True
            self.center[s] = c
            self.missed[s] = 0
            matched_slots[s] = det_idx

        return matched_slots
