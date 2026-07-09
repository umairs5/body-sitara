import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HULL_COLOR = (127, 127, 127)


class SelfieSegBlur:
    """
    MediaPipe ImageSegmenter (Tasks API, mediapipe>=0.10) drop-in replacement
    for convex hull blur. Uses selfie segmentation tflite models.

    model_path: path to .tflite model file
      - selfie_segmenter.tflite          (general, faster)
      - selfie_segmenter_landscape.tflite (landscape/full-body, more accurate)
    """

    def __init__(self, model_path: str, threshold: float = 0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._threshold = threshold

    def blur_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        # category_mask: uint8 array, 1 = person, 0 = background
        if result.category_mask is None:
            return frame
        mask = result.category_mask.numpy_view() == 1
        out = frame.copy()
        out[mask] = HULL_COLOR
        return out

    def close(self):
        self._segmenter.close()
