from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class ObjectLeftTakenDetector(BaseDetector):
    category = "object_left_taken"

    def detect(self, frame: np.ndarray):
        return []
