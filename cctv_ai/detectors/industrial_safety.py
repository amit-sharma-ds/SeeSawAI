from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class IndustrialSafetyDetector(BaseDetector):
    category = "industrial_safety"

    def detect(self, frame: np.ndarray):
        return []
