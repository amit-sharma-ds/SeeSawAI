from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from .base import BaseDetector, Detection
from cctv_ai.config import FACE_MODEL, FACE_PROTO


class FaceDetector(BaseDetector):
    category = "face"

    def __init__(self, conf: float = 0.35):
        super().__init__("face")
        self.conf = conf
        self.net = None
        self._load_model()

    def _load_model(self):
        try:
            if not (FACE_PROTO.exists() and FACE_MODEL.exists()):
                self._download_model()
            self.net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
        except Exception:
            self.net = None

    def _download_model(self):
        FACE_PROTO.parent.mkdir(parents=True, exist_ok=True)
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urlretrieve(proto_url, str(FACE_PROTO))
        urlretrieve(model_url, str(FACE_MODEL))

    def detect(self, frame: np.ndarray):
        if frame is None or frame.size == 0 or self.net is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        results = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                results.append(
                    Detection(
                        category="face",
                        class_name="face",
                        confidence=confidence,
                        bbox=[x1, y1, x2, y2],
                        metadata={"source": "ssd"},
                    )
                )

        return results
