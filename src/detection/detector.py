"""
YOLOv8 + TensorRT enemy detector.

Primary path: loads a pre-exported TensorRT .engine file and runs FP16
inference entirely on-device.  Falls back to the native PyTorch .pt file
when no engine is available (useful during dev / first run before export).

Sub-10 ms inference is achievable on a high-end NVIDIA GPU (RTX 3090/4090)
with the engine path.  The PyTorch path will be ~20–40 ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


def _letterbox(img: np.ndarray, imgsz: int):
    """Resize to square with grey padding. Returns (padded, ratio, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    ratio = imgsz / max(h, w)
    nw, nh = int(w * ratio), int(h * ratio)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
    padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return padded, ratio, (pad_x, pad_y)


@dataclass
class Detection:
    """Single bounding-box detection from the model."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    # Populated by EnemyClassifier after colour analysis
    is_enemy: bool = False
    track_id: Optional[int] = None

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def aim_point(self, y_ratio: float = 0.30) -> tuple[float, float]:
        """Return the aiming point (chest/neck) within the bounding box."""
        ax = self.cx
        ay = self.y1 + self.height * y_ratio
        return ax, ay


@dataclass
class DetectorConfig:
    model_path: str = "models/enemy_detector.engine"
    fallback_model: str = "models/enemy_detector.pt"
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.40
    device: str = "cuda:0"
    input_size: list = field(default_factory=lambda: [640, 640])
    half_precision: bool = True
    # "finetuned" = load model_path/fallback_model as before
    # "coco_person" = load yolov8n.pt base COCO weights, detect class 0 (person) only
    detector_mode: str = "finetuned"


class EnemyDetector:
    """
    Wraps Ultralytics YOLOv8 (or TensorRT engine via Ultralytics) to detect
    potential enemies (class: person) in a single BGR frame.

    The output is a list of Detection objects for all confident bounding boxes.
    EnemyClassifier is responsible for deciding which detections are actual
    enemies vs teammates / static objects.
    """

    def __init__(self, config: dict):
        self._cfg = DetectorConfig(**{
            k: v for k, v in config.items()
            if k in DetectorConfig.__dataclass_fields__
        })
        self._model = None
        self._onnx_session = None
        self._onnx_input_name: Optional[str] = None
        self._warmup_done = False
        self._device = torch.device(self._cfg.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model. Supports ONNX (onnxruntime), TensorRT engine, and PyTorch weights."""
        from ultralytics import YOLO

        if self._cfg.detector_mode == "coco_person":
            # Prefer ONNX if it exists alongside yolov8n.pt — faster on CPU
            onnx_path = Path("models/yolov8n.onnx")
            if onnx_path.exists():
                self._load_onnx(str(onnx_path))
            else:
                log.info("detector_mode=coco_person: loading yolov8n.pt base COCO weights")
                self._model = YOLO("yolov8n.pt")
                log.info("COCO yolov8n loaded.")
            return

        # ONNX path: if model_path explicitly points to a .onnx file
        if self._cfg.model_path.endswith(".onnx"):
            onnx_file = Path(self._cfg.model_path)
            if not onnx_file.exists():
                raise FileNotFoundError(
                    f"ONNX model not found: {onnx_file}. "
                    "Run: python tools/export_onnx.py"
                )
            self._load_onnx(str(onnx_file))
            return

        engine_path = Path(self._cfg.model_path)
        fallback_path = Path(self._cfg.fallback_model)

        if engine_path.exists():
            log.info("Loading TensorRT engine: %s", engine_path)
            self._model = YOLO(str(engine_path), task="detect")
            log.info("TensorRT engine loaded.")
        elif fallback_path.exists():
            log.warning(
                "Engine not found at %s — loading PyTorch weights: %s",
                engine_path, fallback_path,
            )
            self._model = YOLO(str(fallback_path))
        else:
            raise FileNotFoundError(
                f"No model found. Expected engine at '{engine_path}' "
                f"or PyTorch weights at '{fallback_path}'."
            )

    def _load_onnx(self, path: str) -> None:
        """Load ONNX model via onnxruntime (CPU only)."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            )
        providers = ["CPUExecutionProvider"]
        self._onnx_session = ort.InferenceSession(path, providers=providers)
        self._onnx_input_name = self._onnx_session.get_inputs()[0].name
        log.info("ONNX model loaded via onnxruntime (CPU): %s", path)

    def warmup(self, n_iters: int = 10) -> None:
        """Run dummy inference to warm up CUDA kernels and TensorRT calibration."""
        if self._model is None:
            raise RuntimeError("Call load() before warmup().")
        h, w = self._cfg.input_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        log.info("Warming up detector (%d iterations)…", n_iters)
        for _ in range(n_iters):
            self._run_inference(dummy)
        self._warmup_done = True
        log.info("Warmup complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a BGR frame.

        Returns a list of Detection objects that passed the confidence
        threshold and NMS.  Class filtering (enemy vs not) happens later
        in EnemyClassifier.
        """
        if self._model is None:
            raise RuntimeError("Call load() before detect().")
        return self._run_inference(frame)

    def _run_inference(self, frame: np.ndarray) -> List[Detection]:
        if self._onnx_session is not None:
            return self._run_onnx_inference(frame)

        extra = {}
        if self._cfg.detector_mode == "coco_person":
            extra["classes"] = [0]   # COCO class 0 = person

        results = self._model.predict(
            source=frame,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.nms_threshold,
            device=self._device,
            verbose=False,
            half=self._cfg.half_precision and self._device.type == "cuda",
            imgsz=self._cfg.input_size[0],
            **extra,
        )

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy_arr, conf_arr, cls_arr):
            detections.append(Detection(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                confidence=float(conf),
                class_id=int(cls),
            ))

        return detections

    def _run_onnx_inference(self, frame: np.ndarray) -> List[Detection]:
        """Run inference using onnxruntime session (CPU-optimised path)."""
        orig_h, orig_w = frame.shape[:2]
        imgsz = self._cfg.input_size[0]

        # Letterbox resize + normalise
        img, ratio, (pad_x, pad_y) = _letterbox(frame, imgsz)
        img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        img = np.transpose(img, (2, 0, 1))[np.newaxis]      # HWC→BCHW

        output = self._onnx_session.run(None, {self._onnx_input_name: img})[0]
        preds = output[0].T   # (N, 84)  — cx,cy,w,h + 80 class scores

        boxes  = preds[:, :4]
        scores = preds[:, 4:]

        if self._cfg.detector_mode == "coco_person":
            max_scores = scores[:, 0]          # class 0 = person
            class_ids  = np.zeros(len(preds), dtype=int)
        else:
            max_scores = scores.max(axis=1)
            class_ids  = scores.argmax(axis=1)

        mask = max_scores >= self._cfg.confidence_threshold
        boxes      = boxes[mask]
        max_scores = max_scores[mask]
        class_ids  = class_ids[mask]

        if len(boxes) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2 in imgsz space
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        # Undo letterbox → original frame coords
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        # NMS
        xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            xywh, max_scores.tolist(),
            self._cfg.confidence_threshold, self._cfg.nms_threshold,
        )
        if len(indices) == 0:
            return []

        indices = np.array(indices).flatten()
        return [
            Detection(
                x1=float(np.clip(x1[i], 0, orig_w)),
                y1=float(np.clip(y1[i], 0, orig_h)),
                x2=float(np.clip(x2[i], 0, orig_w)),
                y2=float(np.clip(y2[i], 0, orig_h)),
                confidence=float(max_scores[i]),
                class_id=int(class_ids[i]),
            )
            for i in indices
        ]
