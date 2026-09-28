from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class FightDetector(BaseDetector):
    category = "fight"

    def detect(self, frame: np.ndarray):
        return []
