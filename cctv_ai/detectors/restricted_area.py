from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class RestrictedAreaDetector(BaseDetector):
    category = "restricted_area"

    def detect(self, frame: np.ndarray):
        return []
