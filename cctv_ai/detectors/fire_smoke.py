from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class FireSmokeDetector(BaseDetector):
    category = "fire_smoke"

    def detect(self, frame: np.ndarray):
        return []
