import cv2
import numpy as np
import os

HULL_COLOR = (127, 127, 127)

_DEFAULT_PT   = "yolov8n-seg.pt"
_DEFAULT_ONNX = "yolov8n-seg.onnx"


class YOLOSegBlur:
    """
    YOLOv8-seg-nano instance segmentation.
    Prefers ONNX model (OnnxRuntime, ~3-5x faster on CPU) when available;
    falls back to PyTorch .pt otherwise.
    Same get_mask / apply_mask interface as SelfieSegBlur.
    """

    def __init__(self, model_name: str = None, infer_size: int = 320, conf: float = 0.4):
        from ultralytics import YOLO

        if model_name is None:
            # PT is faster than ONNX on CPU for this model (~47ms vs ~117ms)
            model_name = _DEFAULT_PT
            print(f"  [YOLOSeg] Using PyTorch model: {_DEFAULT_PT}")

        self._model      = YOLO(model_name)
        self._infer_size = infer_size
        self._conf       = conf

    def get_mask(self, frame: np.ndarray, infer_size: int = None):
        """
        Run YOLO-seg on frame, return combined bool mask (H×W).
        infer_size arg accepted for API compatibility but ignored.
        Returns None if no persons detected.
        """
        h, w = frame.shape[:2]
        results = self._model(
            frame,
            imgsz=self._infer_size,
            conf=self._conf,
            classes=[0],
            verbose=False,
        )
        combined = np.zeros((h, w), dtype=bool)
        for r in results:
            if r.masks is None:
                continue
            for mask_tensor in r.masks.data:
                m = mask_tensor.cpu().numpy()
                m_up = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                combined |= m_up
        return combined if combined.any() else None

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[mask] = HULL_COLOR
        return out

    def blur_frame(self, frame: np.ndarray) -> np.ndarray:
        mask = self.get_mask(frame)
        return self.apply_mask(frame, mask) if mask is not None else frame
