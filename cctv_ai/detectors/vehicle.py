from __future__ import annotations

import cv2
import numpy as np

from .base import BaseDetector, Detection


class VehicleDetector(BaseDetector):
    category = "vehicle"

    def detect(self, frame: np.ndarray):
        return []
