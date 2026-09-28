"""Helpers for deterministic end-to-end tests: a synthetic video and a detector that finds its bright blob.

The clip is 12 s at 10 fps, 320x240. A white "person" box walks right for the first
4 s and then stands still on the right side, inside the GATE zone (x >= 70 % of width).
"""
from __future__ import annotations

import cv2
import numpy as np

from cctv_ai.detectors.base import BaseDetector, Detection

W, H, FPS = 320, 240, 10
BOX_W, BOX_H, BOX_Y = 40, 120, 60
WALK_FRAMES = 40        # frames 0..39 walk from x=10 to x=195 (still left of the zone)
STAND_X = 250           # frames 40..119 stand at x=250 (inside the zone)
TOTAL_FRAMES = 120

GATE = {"name": "gate", "kind": "restricted", "polygon": [[0.7, 0.0], [1.0, 0.0], [1.0, 1.0], [0.7, 1.0]]}


def box_at(index: int):
    if index < WALK_FRAMES:
        x = int(round(10 + index * (195 - 10) / (WALK_FRAMES - 1)))
    else:
        x = STAND_X
    return x, BOX_Y, x + BOX_W, BOX_Y + BOX_H


def write_walk_and_stand_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("cannot create the synthetic test video")
    for index in range(TOTAL_FRAMES):
        frame = np.zeros((H, W, 3), np.uint8)
        x1, y1, x2, y2 = box_at(index)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class BlobDetector(BaseDetector):
    """Reports every bright blob as a person. Only for tests."""

    category = "person"

    def __init__(self):
        super().__init__("person")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 200:
                continue
            detections.append(Detection("person", "person", 1.0, [x, y, x + w, y + h], {"source": "blob"}))
        return detections
