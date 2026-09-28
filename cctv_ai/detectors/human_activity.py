from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class HumanActivityDetector(BaseDetector):
    category = "human_activity"

    def detect(self, frame: np.ndarray):
        return []
