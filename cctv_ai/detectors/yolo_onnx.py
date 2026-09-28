"""80-class COCO object detector from an ONNX file, run with OpenCV DNN. No PyTorch needed.

Two model families are understood, picked by file name:

  * YOLOv8 / YOLO11 exported by ultralytics (output 1 x 84 x N):
        yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640
  * YOLOX from OpenCV's model zoo (output 1 x N x 85), fetched by download_models.py

Put the file in the models/ folder. The API, live.py and the web page pick it up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cctv_ai.config import MODEL_DIR

from .base import BaseDetector, Detection

COCO_CLASSES: Sequence[str] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "boat"}
BAG_CLASSES = {"backpack", "handbag", "suitcase"}
# what a CCTV camera should track: people, vehicles, bags, knives, and the hand-held objects
# people carry (useful for demos and for object-left rules). Furniture and food stay out.
LIVE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "backpack", "handbag", "suitcase", "umbrella",
    "knife", "bottle", "cup", "cell phone", "laptop", "book", "scissors", "remote",
}
MODEL_CANDIDATES = [
    MODEL_DIR / "yolov8n.onnx",
    MODEL_DIR / "yolo11n.onnx",
    MODEL_DIR / "object_detection_yolox_2022nov.onnx",
    MODEL_DIR / "yolox.onnx",
]
DEFAULT_MODEL = MODEL_CANDIDATES[0]


def find_model(model_path: Optional[str | Path] = None) -> Optional[Path]:
    if model_path:
        path = Path(model_path)
        return path if path.exists() else None
    return next((p for p in MODEL_CANDIDATES if p.exists()), None)


def discover_custom_models(model_dir: Path = MODEL_DIR) -> Dict[str, Tuple[Path, List[str]]]:
    """Models you trained yourself: models/<name>.onnx next to models/<name>.names (one class per
    line). Returns {name: (onnx path, class names)}. The name doubles as the detection category
    and the feature key, e.g. weapon, fire_smoke, ppe."""
    found: Dict[str, Tuple[Path, List[str]]] = {}
    if not Path(model_dir).is_dir():
        return found
    for names_file in sorted(Path(model_dir).glob("*.names")):
        onnx = names_file.with_suffix(".onnx")
        if not onnx.exists():
            continue
        names = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
        if names:
            found[names_file.stem] = (onnx, names)
    return found


def custom_model_available(name: str, model_dir: Path = MODEL_DIR) -> bool:
    return (Path(model_dir) / f"{name}.onnx").exists() and (Path(model_dir) / f"{name}.names").exists()


def category_for(class_name: str) -> str:
    if class_name == "person":
        return "person"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    if class_name in BAG_CLASSES:
        return "bag"
    return "object"


def decode_predictions(
    output,
    scale: float,
    width: int,
    height: int,
    conf: float = 0.35,
    nms: float = 0.45,
    classes: Optional[Iterable[str]] = None,
    names: Sequence[str] = COCO_CLASSES,
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> List[Detection]:
    """Turn YOLOv8-style output (1, 4+classes, N) or (1, N, 4+classes) into Detections in
    original-image pixels. Boxes are centre x, centre y, width, height in input pixels.

    `scale` is the letterbox resize factor and pad_x/pad_y the letterbox offsets that were
    applied before inference. Boxes are clipped to the image and de-duplicated with NMS.
    """
    pred = np.asarray(output, dtype=np.float32)
    while pred.ndim > 2:
        pred = pred[0]
    if pred.shape[0] == 4 + len(names) and pred.shape[1] != 4 + len(names):
        pred = pred.T  # -> (N, 4 + classes)
    boxes = pred[:, :4]
    scores = pred[:, 4:4 + len(names)]
    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]

    keep = confs >= conf
    allowed = set(classes) if classes else None
    if allowed is not None:
        allowed_mask = np.array([name in allowed for name in names])
        keep &= allowed_mask[class_ids]
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]
    if len(boxes) == 0:
        return []

    cx, cy, bw, bh = boxes.T
    x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, width)
    y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, height)
    x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, width)
    y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, height)

    rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    score_list = [float(s) for s in confs]
    if hasattr(cv2.dnn, "NMSBoxesBatched"):
        kept = cv2.dnn.NMSBoxesBatched(rects, score_list, [int(c) for c in class_ids], conf, nms)
    else:
        kept = cv2.dnn.NMSBoxes(rects, score_list, conf, nms)
    kept = np.asarray(kept).reshape(-1)

    results: List[Detection] = []
    for i in kept:
        name = names[int(class_ids[i])]
        results.append(
            Detection(
                category=category_for(name),
                class_name=name,
                confidence=float(confs[i]),
                bbox=[int(round(x1[i])), int(round(y1[i])), int(round(x2[i])), int(round(y2[i]))],
                metadata={"source": "yolo"},
            )
        )
    results.sort(key=lambda d: d.confidence, reverse=True)
    return results


def yolox_grids(input_size: int, strides: Sequence[int] = (8, 16, 32)):
    """Anchor-point grid and stride per output row, in the order YOLOX emits them."""
    grids, stride_col = [], []
    for stride in strides:
        cells = input_size // stride
        yv, xv = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32)
        grids.append(grid)
        stride_col.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))
    return np.concatenate(grids), np.concatenate(stride_col)


def decode_yolox(
    output,
    scale: float,
    width: int,
    height: int,
    input_size: int = 640,
    conf: float = 0.35,
    nms: float = 0.45,
    classes: Optional[Iterable[str]] = None,
    names: Sequence[str] = COCO_CLASSES,
) -> List[Detection]:
    """YOLOX raw output (1, N, 5 + classes): grid offsets, log sizes, objectness, class scores."""
    pred = np.asarray(output, dtype=np.float32)
    while pred.ndim > 2:
        pred = pred[0]
    grids, strides = yolox_grids(input_size)
    if pred.shape[0] != grids.shape[0]:
        raise ValueError(f"unexpected YOLOX output rows {pred.shape[0]} for input size {input_size}")
    xy = (pred[:, :2] + grids) * strides
    wh = np.exp(pred[:, 2:4]) * strides
    scores = pred[:, 4:5] * pred[:, 5:5 + len(names)]
    merged = np.concatenate([xy, wh, scores], axis=1)
    return decode_predictions(merged[np.newaxis], scale, width, height, conf, nms, classes, names)


class YoloOnnxDetector(BaseDetector):
    category = "object"

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf: float = 0.35,
        nms: float = 0.45,
        input_size: int = 640,
        classes: Optional[Iterable[str]] = None,
        names: Optional[Sequence[str]] = None,
        category: Optional[str] = None,
    ):
        """`names` are the model's class labels (COCO by default); `category` forces every
        detection into one category, used for custom models such as weapon or fire_smoke."""
        super().__init__(category or "yolo")
        found = find_model(model_path)
        if found is None:
            wanted = Path(model_path) if model_path else DEFAULT_MODEL
            raise FileNotFoundError(
                f"YOLO ONNX model not found at {wanted}. Run 'py download_models.py' or export one with "
                "'yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640' and copy it to models/."
            )
        self.model_path = found
        self.family = "yolox" if "yolox" in found.name.lower() else "yolov8"
        self.net = cv2.dnn.readNetFromONNX(str(found))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf = float(conf)
        self.nms = float(nms)
        self.input_size = int(input_size)
        self.classes = set(classes) if classes else None
        self.names: Sequence[str] = tuple(names) if names else COCO_CLASSES
        self.fixed_category = category

    @staticmethod
    def available(model_path: Optional[str | Path] = None) -> bool:
        return find_model(model_path) is not None

    def detect(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        if self.family == "yolox":
            blob = cv2.dnn.blobFromImage(canvas, 1.0, (size, size), swapRB=False, crop=False)  # raw BGR 0..255
        else:
            blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        if self.family == "yolox":
            results = decode_yolox(output, scale, w, h, size, self.conf, self.nms, self.classes, self.names)
        else:
            results = decode_predictions(output, scale, w, h, self.conf, self.nms, self.classes, self.names)
        if self.fixed_category:
            for det in results:
                det.category = self.fixed_category
                det.metadata["model"] = self.model_path.stem
        return results
