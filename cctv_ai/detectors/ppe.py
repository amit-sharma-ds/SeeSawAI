from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class PPEDetector(BaseDetector):
    category = "ppe"

    def detect(self, frame: np.ndarray):
        return []
