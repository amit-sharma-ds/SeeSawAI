from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class LoiteringDetector(BaseDetector):
    category = "loitering"

    def detect(self, frame: np.ndarray):
        return []
