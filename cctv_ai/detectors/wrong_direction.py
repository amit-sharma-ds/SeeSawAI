from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class WrongDirectionDetector(BaseDetector):
    category = "wrong_direction"

    def detect(self, frame: np.ndarray):
        return []
