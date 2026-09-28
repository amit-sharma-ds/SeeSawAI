from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlretrieve

import cv2
import numpy as np

from .config import DEFAULT_CONF, FACE_MODEL, FACE_PROTO


def find_camera_index(max_index=10):
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return idx
    return 0


def box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def non_max_suppression(boxes, overlap_threshold=0.35):
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    result = []

    while sorted_boxes:
        current = sorted_boxes.pop(0)
        result.append(current)
        sorted_boxes = [
            box for box in sorted_boxes if box_iou(current[:4], box[:4]) < overlap_threshold
        ]
    return result


class CCTVDetector:
    def __init__(self, conf: float = DEFAULT_CONF):
        self.conf = conf
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.face_net = None
        self.face_model_ready = False
        self._load_face_detector()

        self.frontal_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )

    def _download_face_model(self):
        if FACE_PROTO.exists() and FACE_MODEL.exists():
            return

        FACE_PROTO.parent.mkdir(parents=True, exist_ok=True)
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

        try:
            urlretrieve(proto_url, str(FACE_PROTO))
            urlretrieve(model_url, str(FACE_MODEL))
        except Exception:
            raise FileNotFoundError("Face detection model not available. Download failed.")

    def _load_face_detector(self):
        try:
            self._download_face_model()
            self.face_net = cv2.dnn.readNetFromCaffe(str(FACE_PROTO), str(FACE_MODEL))
            self.face_model_ready = True
        except Exception:
            self.face_net = None
            self.face_model_ready = False

    def _dnn_face_boxes(self, frame: np.ndarray):
        if self.face_net is None or not self.face_model_ready:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
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
                results.append((x1, y1, x2, y2, confidence))
        return results

    def _haar_face_boxes(self, gray: np.ndarray):
        results = []
        frontal = self.frontal_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(30, 30),
        )
        for x, y, w, h in frontal:
            results.append((x, y, x + w, y + h, 1.0))

        profile = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30),
        )
        for x, y, w, h in profile:
            results.append((x, y, x + w, y + h, 1.0))

        mirrored = cv2.flip(gray, 1)
        mirrored_boxes = self.profile_cascade.detectMultiScale(
            mirrored,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30),
        )
        for x, y, w, h in mirrored_boxes:
            mirrored_x = gray.shape[1] - x - w
            results.append((mirrored_x, y, mirrored_x + w, y + h, 1.0))
        return results

    def _person_boxes(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        results = []
        for x, y, w, h in boxes:
            results.append((int(x), int(y), int(x + w), int(y + h), 0.8))
        return results

    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            return {"detections": [], "count": 0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        all_boxes = []
        all_boxes.extend(self._dnn_face_boxes(frame))
        all_boxes.extend(self._haar_face_boxes(gray))
        all_boxes.extend(self._person_boxes(frame))

        deduped = non_max_suppression(all_boxes, overlap_threshold=0.38)

        detections = []
        for x1, y1, x2, y2, confidence in deduped:
            label = "person" if (x2 - x1) > 40 and (y2 - y1) > 40 else "face"
            detections.append({
                "class_name": label,
                "confidence": float(confidence),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

        return {"detections": detections, "count": len(detections)}

    def detect_people(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        result = self.predict_frame(frame)
        return [d for d in result["detections"] if d["class_name"] == "person"]

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        result = self.predict_frame(frame)
        return [d for d in result["detections"] if d["class_name"] == "face"]

    def alert_status(self, frame: np.ndarray, crowd_limit: int = 5):
        result = self.predict_frame(frame)
        detections = result["detections"]
        people = [d for d in detections if d["class_name"] == "person"]

        alerts = []
        if len(people) >= crowd_limit:
            alerts.append({
                "type": "crowd",
                "severity": "high",
                "message": f"Crowd detected: {len(people)} people",
                "count": len(people),
            })

        if len(detections) == 0:
            alerts.append({
                "type": "idle",
                "severity": "low",
                "message": "No persons detected",
                "count": 0,
            })

        return {
            "detections": detections,
            "count": len(detections),
            "people_count": len(people),
            "alerts": alerts,
        }

    def save_debug_image(self, frame: np.ndarray, output_path: str | Path, detections: List[Dict[str, Any]]) -> None:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(out, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), out)
