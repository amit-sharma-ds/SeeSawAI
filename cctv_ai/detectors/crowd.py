from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class CrowdDetector(BaseDetector):
    category = "crowd"

    def detect(self, frame: np.ndarray):
        return []
