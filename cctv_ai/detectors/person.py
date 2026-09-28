from __future__ import annotations

import math

import cv2
import numpy as np

from .base import BaseDetector, Detection, non_max_suppression


class PersonDetector(BaseDetector):
    """OpenCV HOG + linear SVM people detector.

    Needs no model download and runs on any CPU. The defaults favour speed:
    win_stride (8, 8) is roughly four times faster than (4, 4) on a weak CPU for a
    small accuracy cost. Feed it frames 480 to 640 px wide.
    """

    category = "person"

    def __init__(
        self,
        win_stride=(8, 8),
        padding=(8, 8),
        scale: float = 1.05,
        hit_threshold: float = 0.0,
        min_confidence: float = 0.2,
        nms_threshold: float = 0.4,
    ):
        super().__init__("person")
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = tuple(int(v) for v in win_stride)
        self.padding = tuple(int(v) for v in padding)
        self.scale = float(scale)
        self.hit_threshold = float(hit_threshold)
        self.min_confidence = float(min_confidence)
        self.nms_threshold = float(nms_threshold)

    @staticmethod
    def _confidence(weight: float) -> float:
        # SVM margins are roughly 0..3; squash them into 0..1 so they compare with other detectors.
        return float(1.0 - math.exp(-max(float(weight), 0.0)))

    def detect(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        boxes, weights = self.hog.detectMultiScale(
            gray,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )
        if len(boxes) == 0:
            return []

        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != len(boxes):
            weights = np.full(len(boxes), 1.0)

        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            confidence = self._confidence(weight)
            if confidence < self.min_confidence:
                continue
            detections.append(
                Detection(
                    category="person",
                    class_name="person",
                    confidence=confidence,
                    bbox=[int(x), int(y), int(x + w), int(y + h)],
                    metadata={"source": "hog", "svm_weight": round(float(weight), 3)},
                )
            )

        return non_max_suppression(detections, overlap_threshold=self.nms_threshold)
