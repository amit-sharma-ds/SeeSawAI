from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np


@dataclass
class Detection:
    category: str
    class_name: str
    confidence: float
    bbox: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [int(v) for v in self.bbox],
            "metadata": self.metadata,
        }


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def non_max_suppression(detections, overlap_threshold=0.38):
    if not detections:
        return []

    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    result = []

    while sorted_detections:
        current = sorted_detections.pop(0)
        result.append(current)
        sorted_detections = [
            item for item in sorted_detections if box_iou(current.bbox, item.bbox) < overlap_threshold
        ]

    return result


class BaseDetector:
    category = "base"

    def __init__(self, name: str | None = None):
        self.name = name or self.category

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return []

    def draw(self, frame: np.ndarray, detections: List[Detection]):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(out, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 2)
        return out
