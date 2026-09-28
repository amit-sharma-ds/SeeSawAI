from .base import Detection, box_iou, non_max_suppression
from .face import FaceDetector
from .person import PersonDetector
from .yolo_onnx import COCO_CLASSES, LIVE_CLASSES, YoloOnnxDetector, custom_model_available, discover_custom_models

DETECTION_CATEGORIES = [
    "person",
    "face",
    "vehicle",
    "bag",
    "object",
    "weapon",
    "fire_smoke",
    "ppe",
]


DETECTOR_CHOICES = ("auto", "yolo", "hog")


def build_detectors(faces: bool = False, live: bool = True, prefer: str = "auto"):
    """The detectors to run, plus every custom model found in models/ (weapon, fire_smoke, ppe).

    prefer="auto": the 80-class YOLO model when its file exists, otherwise the HOG person detector.
    prefer="hog": always the fast HOG person detector (weak CPUs, many cameras).
    prefer="yolo": the YOLO model, error if it is missing.
    `live` limits the COCO model to people, vehicles and bags."""
    if prefer not in DETECTOR_CHOICES:
        raise ValueError(f"detector must be one of {DETECTOR_CHOICES}, got '{prefer}'")
    if prefer == "hog":
        detectors = [PersonDetector()]
    elif prefer == "yolo" or YoloOnnxDetector.available():
        detectors = [YoloOnnxDetector(classes=LIVE_CLASSES if live else None)]
    else:
        detectors = [PersonDetector()]
    for name, (path, names) in discover_custom_models().items():
        detectors.append(YoloOnnxDetector(model_path=path, names=names, category=name))
    if faces:
        detectors.append(FaceDetector())
    return detectors


__all__ = [
    "Detection",
    "box_iou",
    "non_max_suppression",
    "PersonDetector",
    "FaceDetector",
    "YoloOnnxDetector",
    "COCO_CLASSES",
    "LIVE_CLASSES",
    "DETECTION_CATEGORIES",
    "DETECTOR_CHOICES",
    "build_detectors",
    "custom_model_available",
    "discover_custom_models",
]
