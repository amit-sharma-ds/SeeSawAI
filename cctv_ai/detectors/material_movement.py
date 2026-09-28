from __future__ import annotations

import numpy as np

from .base import BaseDetector, Detection


class MaterialMovementDetector(BaseDetector):
    category = "material_movement"

    def detect(self, frame: np.ndarray):
        return []
