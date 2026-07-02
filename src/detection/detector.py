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

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger(__name__)


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
        self._warmup_done = False
        self._device = torch.device(self._cfg.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model. Try TensorRT engine first, then PyTorch weights."""
        from ultralytics import YOLO

        if self._cfg.detector_mode == "coco_person":
            log.info("detector_mode=coco_person: loading yolov8n.pt base COCO weights")
            self._model = YOLO("yolov8n.pt")
            log.info("COCO yolov8n loaded.")
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
