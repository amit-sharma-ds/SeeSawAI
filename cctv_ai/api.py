from __future__ import annotations

import importlib.util
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import auth
from .detector import CCTVDetector, find_camera_index
from .detectors import (COCO_CLASSES, DETECTOR_CHOICES, FaceDetector, YoloOnnxDetector, build_detectors,
                        non_max_suppression)
from .features import feature_catalog, filter_detections, parse_feature_list, validate_features
from .rules import AlertManager, CameraRules, load_cameras_config
from .worker import CameraWorker

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".3gp", ".ts"}
UPLOAD_DIR = Path(os.environ.get("CCTV_UPLOAD_DIR", PROJECT_DIR / "uploads"))
CAMERAS_CONFIG = PROJECT_DIR / "cameras.json"

workers: Dict[str, CameraWorker] = {}
alert_manager = AlertManager()
_legacy_detector: Optional[CCTVDetector] = None
_image_detectors: Dict[tuple, list] = {}


def image_detectors(faces: bool, detector: str = "auto") -> list:
    """Detectors for uploaded images: all 80 classes when the YOLO model exists."""
    key = (faces, detector)
    if key not in _image_detectors:
        _image_detectors[key] = build_detectors(faces=faces, live=False, prefer=detector)
    return _image_detectors[key]


def _check_detector(detector: str) -> str:
    if detector not in DETECTOR_CHOICES:
        raise HTTPException(status_code=422, detail=f"detector must be one of {DETECTOR_CHOICES}")
    if detector == "yolo" and not YoloOnnxDetector.available():
        raise HTTPException(status_code=422, detail="the object model is not installed; run 'py download_models.py'")
    return detector


def object_detector_name() -> str:
    return "yolo_onnx (80 classes)" if YoloOnnxDetector.available() else "hog (person only)"


# categories that always appear on the sample strips, in this order, even while empty,
# so a visitor can see where gun, smoke or PPE test material belongs (samples/<category>/)
PRODUCT_CATEGORIES = ["person", "vehicle", "crowd", "gun", "knife", "smoke", "fire", "ppe"]

# COCO class -> category shown on the sample strip
SAMPLE_CATEGORIES: Dict[str, str] = {"person": "person", "knife": "knife"}
for _group, _classes in {
    "vehicle": ("bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"),
    "street": ("traffic light", "fire hydrant", "stop sign", "parking meter", "bench"),
    "animal": ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"),
    "bag & accessory": ("backpack", "umbrella", "handbag", "tie", "suitcase"),
    "sports": ("frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"),
    "kitchen": ("bottle", "wine glass", "cup", "fork", "spoon", "bowl"),
    "food": ("banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"),
    "furniture": ("chair", "couch", "potted plant", "bed", "dining table", "toilet"),
    "electronics": ("tv", "laptop", "mouse", "remote", "keyboard", "cell phone"),
    "appliance": ("microwave", "oven", "toaster", "sink", "refrigerator"),
    "indoor": ("book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"),
}.items():
    for _cls in _classes:
        SAMPLE_CATEGORIES[_cls] = _group


def ordered_categories(found) -> List[str]:
    """Product categories first, then whatever else the samples contain."""
    extra = sorted(c for c in set(found) if c not in PRODUCT_CATEGORIES)
    return list(PRODUCT_CATEGORIES) + extra


def _label_info(image_path: Path) -> tuple:
    """YOLO datasets keep images/<split>/x.jpg next to labels/<split>/x.txt with class ids.
    Returns (categories, {class name: count}) from that label file, or ([], {})."""
    parts = list(image_path.parts)
    if "images" not in parts:
        return [], {}
    i = len(parts) - 1 - parts[::-1].index("images")
    label = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
    if not label.exists():
        return [], {}
    categories = set()
    counts: Dict[str, int] = {}
    for line in label.read_text(encoding="utf-8").splitlines():
        bits = line.split()
        if bits and bits[0].isdigit() and int(bits[0]) < len(COCO_CLASSES):
            name = COCO_CLASSES[int(bits[0])]
            categories.add(SAMPLE_CATEGORIES.get(name, "other"))
            counts[name] = counts.get(name, 0) + 1
    return sorted(categories), dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _label_categories(image_path: Path) -> List[str]:
    return _label_info(image_path)[0]


def list_samples() -> Dict[str, Dict[str, Any]]:
    """Demo images with categories: samples/<category>/..., every YOLO dataset under datasets/
    (categories read from its labels), and the two ultralytics photos. Keys are URL paths."""
    found: Dict[str, Dict[str, Any]] = {}
    samples_dir = PROJECT_DIR / "samples"
    if samples_dir.is_dir():
        for path in sorted(samples_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                rel = path.relative_to(samples_dir)
                categories = [rel.parts[0]] if len(rel.parts) > 1 else ["my samples"]
                found[rel.as_posix()] = {"path": path, "categories": categories}
    datasets_dir = PROJECT_DIR / "datasets"
    if datasets_dir.is_dir():
        for images_dir in sorted(datasets_dir.glob("*/images")):
            for path in sorted(images_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    key = f"{images_dir.parent.name}/{path.name}"
                    categories, counts = _label_info(path)
                    found.setdefault(key, {"path": path, "categories": categories or ["unlabelled"], "labels": counts})
    spec = importlib.util.find_spec("ultralytics")
    if spec is not None and spec.submodule_search_locations:
        assets = Path(list(spec.submodule_search_locations)[0]) / "assets"
        known = {"bus.jpg": ["person", "vehicle"], "zidane.jpg": ["person"]}
        if assets.is_dir():
            for path in sorted(assets.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    found.setdefault(path.name, {"path": path, "categories": known.get(path.name, ["other"])})
    return found


def get_legacy_detector() -> CCTVDetector:
    """Single-image detector, created on first use so the API starts instantly and
    never downloads the face model just to run live cameras."""
    global _legacy_detector
    if _legacy_detector is None:
        _legacy_detector = CCTVDetector()
    return _legacy_detector


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    for worker in list(workers.values()):
        worker.stop(join=False)


app = FastAPI(
    title="SeeSaw AI API",
    version="1.2.0",
    description="Live CCTV monitoring: cameras, tracking, zone alerts and an MJPEG stream. "
                "Protected by API keys once any key exists (see make_api_key.py).",
    lifespan=lifespan,
    dependencies=[Depends(auth.require_api_key)],
)

# Web front-ends on other hosts need CORS. Restrict with CCTV_CORS_ORIGINS="http://a.com,http://b.com".
_cors_origins = [o.strip() for o in os.environ.get("CCTV_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraStart(BaseModel):
    camera_id: str = Field(..., min_length=1)
    source: str = Field(..., description="camera index, video file, DroidCam/IP Webcam URL or rtsp:// URL")
    every_n: int = Field(2, ge=1, description="run detection on every Nth frame")
    width: int = Field(640, ge=0, description="resize frames to this width before detection (0 = keep)")
    faces: bool = Field(False, description="also run the face detector")
    rules: Optional[Dict[str, Any]] = Field(None, description="zones, loiter_seconds, crowd_limit, direction, cooldown_seconds")
    features: Optional[List[str]] = Field(None, description="feature keys to switch on (see GET /features); default: every built feature")
    detector: str = Field("auto", description="auto (object model if installed), hog (fast, people only) or yolo (object model)")


class FeaturesUpdate(BaseModel):
    features: List[str] = Field(..., description="feature keys to keep on; everything else is switched off")


def _decode_upload(contents: bytes):
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _get_worker(camera_id: str) -> CameraWorker:
    worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail=f"camera '{camera_id}' not found")
    return worker


def start_camera_worker(camera_id: str, source: str, every_n: int = 2, width: int = 640,
                        faces: bool = False, rules: Optional[dict] = None,
                        features: Optional[List[str]] = None, detector: str = "auto") -> CameraWorker:
    existing = workers.get(camera_id)
    if existing is not None and existing.is_alive():
        raise HTTPException(status_code=409, detail=f"camera '{camera_id}' is already running")
    detector = _check_detector(detector)
    try:
        camera_rules = CameraRules.from_dict(rules, camera_id)
        if features is not None:
            camera_rules.features = validate_features(features)
        elif camera_rules.features is not None:
            camera_rules.features = validate_features(camera_rules.features)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid rules: {exc}")
    # the face detector only runs when asked for, because it downloads its model once
    faces = faces or (camera_rules.features is not None and "face_access" in camera_rules.features)
    detectors = build_detectors(faces=faces, live=True, prefer=detector)
    worker = CameraWorker(
        camera_id, source, detectors,
        rules=camera_rules, alert_manager=alert_manager, every_n=every_n, width=width,
    )
    workers[camera_id] = worker
    worker.start()
    return worker


# -- service ----------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "seesaw-ai",
        "features": [
            "person_detection", "face_detection", "tracking",
            "restricted_area", "industrial_safety", "loitering", "crowd", "wrong_direction",
            "live_cameras", "mjpeg_stream", "alert_log",
        ],
        "cameras_running": sum(1 for w in workers.values() if w.is_alive()),
        "auth": "api_key" if auth.auth_enabled() else "open",
        "object_detector": object_detector_name(),
    }


# -- web page, sample images and single-image detection -----------------------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    # no-cache: browsers must revalidate, so a page update shows on the next reload
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/samples")
def samples() -> Dict[str, Any]:
    """Demo images a visitor can try without uploading anything, grouped by category."""
    items = list_samples()
    return {
        "samples": [
            {"name": key, "url": f"/samples/{key}", "categories": item["categories"], "labels": item.get("labels") or {}}
            for key, item in items.items()
        ],
        "categories": ordered_categories(c for item in items.values() for c in item["categories"]),
        "folder": "samples/<category>/",
    }


@app.get("/samples/{name:path}")
def sample_file(name: str) -> FileResponse:
    item = list_samples().get(name)
    if item is None:
        raise HTTPException(status_code=404, detail=f"sample '{name}' not found")
    return FileResponse(item["path"])


@app.get("/features")
def features() -> Dict[str, Any]:
    """The 14 product features with their build status. Keys are used for the on/off switches."""
    return {"features": feature_catalog()}


@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    faces: bool = False,
    max_width: int = Query(1280, ge=160, le=4096, description="large images are shrunk to this width for detection"),
    features: Optional[str] = Query(None, description="comma separated feature keys to keep, e.g. person,vehicle; default all"),
    detector: str = Query("auto", description="auto, hog (fast, people only) or yolo (object model)"),
) -> Dict[str, Any]:
    """Run the object detector on one uploaded image. Boxes are in original image pixels."""
    detector = _check_detector(detector)
    try:
        enabled = parse_feature_list(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    faces = faces or (enabled is not None and "face_access" in enabled)
    frame = _decode_upload(await file.read())
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded")
    h, w = frame.shape[:2]
    # YOLO letterboxes internally, one pass is enough. The HOG people model only sees people
    # of roughly 100 to 250 px, so run it at three sizes and merge the boxes.
    using_yolo = detector != "hog" and YoloOnnxDetector.available()
    widths = (max_width,) if using_yolo else (640, 480, 360)

    t0 = time.perf_counter()
    detections = []
    for target in widths:
        scale = min(1.0, target / w)
        work = frame if scale == 1.0 else cv2.resize(frame, (int(round(w * scale)), max(1, int(round(h * scale)))))
        for det_model in image_detectors(faces, detector):
            for det in det_model.detect(work):
                if scale != 1.0:
                    det.bbox = [int(round(v / scale)) for v in det.bbox]
                detections.append(det)
    if len(widths) > 1:
        detections = non_max_suppression(detections, overlap_threshold=0.45)
    detections = filter_detections(detections, enabled)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    items = [det.as_dict() for det in detections]
    items.sort(key=lambda d: d["confidence"], reverse=True)
    return {
        "width": w,
        "height": h,
        "time_ms": round(elapsed_ms, 1),
        "detector": object_detector_name() if using_yolo else "hog (person only)",
        "classes": len(COCO_CLASSES) if using_yolo else 1,
        "features": sorted(enabled) if enabled is not None else None,
        "detections": items,
    }


@app.get("/camera-index")
def camera_index() -> Dict[str, Any]:
    idx = find_camera_index()
    return {"camera_index": idx, "message": "Auto-detected camera index"}


@app.get("/live-status")
def live_status() -> Dict[str, Any]:
    return {
        "camera_index": find_camera_index(),
        "status": "ready",
        "message": "POST /cameras with a source (index, file, DroidCam URL or rtsp://) to start live monitoring",
    }


# -- single-image endpoints (legacy) ------------------------------------------

@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    frame = _decode_upload(await file.read())
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image uploaded"})
    result = get_legacy_detector().predict_frame(frame)
    return JSONResponse(content={"detections": result["detections"], "count": result["count"]})


@app.post("/detect-people")
async def detect_people(file: UploadFile = File(...)) -> JSONResponse:
    frame = _decode_upload(await file.read())
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image uploaded"})
    return JSONResponse(content={"detections": get_legacy_detector().detect_people(frame)})


@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)) -> JSONResponse:
    frame = _decode_upload(await file.read())
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image uploaded"})
    return JSONResponse(content={"detections": get_legacy_detector().detect_faces(frame)})


@app.post("/alerts")
async def alerts_for_image(file: UploadFile = File(...), crowd_limit: int = 5) -> JSONResponse:
    frame = _decode_upload(await file.read())
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image uploaded"})
    return JSONResponse(content=get_legacy_detector().alert_status(frame, crowd_limit=crowd_limit))


# -- live cameras -------------------------------------------------------------

@app.get("/cameras-config")
def cameras_config() -> Dict[str, Any]:
    """Camera entries from cameras.json, so the page can offer their zones for video analysis."""
    try:
        cameras = load_cameras_config(CAMERAS_CONFIG)
    except (FileNotFoundError, ValueError):
        return {"cameras": []}
    return {"cameras": [{"camera_id": str(c["camera_id"]), "source": str(c.get("source", "")),
                         "zones": [z.get("name") for z in (c.get("rules") or {}).get("zones", [])]}
                        for c in cameras]}


# -- uploaded and sample videos ---------------------------------------------------

SAMPLE_VIDEO_DIR = PROJECT_DIR / "samples" / "videos"
_thumbs: Dict[str, bytes] = {}


def _video_options(features: Optional[str], rules_from: Optional[str], detector: str):
    detector = _check_detector(detector)
    try:
        enabled = parse_feature_list(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    rules = None
    if rules_from:
        try:
            entry = next(c for c in load_cameras_config(CAMERAS_CONFIG) if str(c["camera_id"]) == rules_from)
        except (FileNotFoundError, ValueError, StopIteration):
            raise HTTPException(status_code=404, detail=f"camera '{rules_from}' not found in cameras.json")
        rules = entry.get("rules")
    return detector, enabled, rules


def _start_video_job(job_id: str, src_path: Path, original_name: str, every_n: int, width: int,
                     enabled, rules, pace: bool, record: bool, detector: str) -> Dict[str, Any]:
    camera_rules = CameraRules.from_dict(rules, job_id)
    if enabled is not None:
        camera_rules.features = enabled
    faces = enabled is not None and "face_access" in enabled
    worker = CameraWorker(
        job_id, str(src_path), build_detectors(faces=faces, live=True, prefer=detector),
        rules=camera_rules, alert_manager=alert_manager, every_n=every_n, width=width,
        realtime=False, reconnect=False, pace=pace,
        record_path=(UPLOAD_DIR / f"{job_id}_annotated.avi") if record else None,
    )
    workers[job_id] = worker
    worker.start()
    return {"job_id": job_id, "original_name": original_name, **worker.status()}


def list_sample_videos() -> Dict[str, Dict[str, Any]]:
    """Clips in samples/videos/ (a subfolder names the category) that visitors can analyse
    without uploading. Keys are paths relative to samples/videos, used in the URLs."""
    if not SAMPLE_VIDEO_DIR.is_dir():
        return {}
    found: Dict[str, Dict[str, Any]] = {}
    for path in sorted(SAMPLE_VIDEO_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            rel = path.relative_to(SAMPLE_VIDEO_DIR)
            category = rel.parts[0] if len(rel.parts) > 1 else ("street" if "vtest" in path.name else "demo")
            found[rel.as_posix()] = {"path": path, "categories": [category]}
    return found


def _video_meta(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "seconds": round(frames / fps, 1) if fps > 0 else None,
        "size_mb": round(path.stat().st_size / 1e6, 1),
    }
    cap.release()
    return meta


@app.get("/sample-videos")
def sample_videos() -> Dict[str, Any]:
    items = list_sample_videos()
    return {
        "videos": [{"name": name, "thumb": f"/sample-videos/{name}/thumb", "categories": item["categories"],
                    **_video_meta(item["path"])} for name, item in items.items()],
        "categories": ordered_categories(c for item in items.values() for c in item["categories"]),
        "folder": "samples/videos/<category>/",
    }


@app.get("/sample-videos/{name:path}/thumb")
def sample_video_thumb(name: str) -> Response:
    item = list_sample_videos().get(name)
    if item is None:
        raise HTTPException(status_code=404, detail=f"sample video '{name}' not found")
    path = item["path"]
    key = f"{name}:{path.stat().st_mtime_ns}"
    if key not in _thumbs:
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(status_code=500, detail="cannot read the first frame of the clip")
        scale = 320 / frame.shape[1]
        small = cv2.resize(frame, (320, max(1, int(round(frame.shape[0] * scale)))))
        _thumbs[key] = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tobytes()
    return Response(content=_thumbs[key], media_type="image/jpeg")


@app.post("/sample-videos/{name:path}/analyze")
def analyze_sample_video(
    name: str,
    every_n: int = Query(2, ge=1),
    width: int = Query(640, ge=0),
    features: Optional[str] = Query(None),
    rules_from: Optional[str] = Query(None),
    pace: bool = Query(True),
    record: bool = Query(True),
    detector: str = Query("auto"),
) -> Dict[str, Any]:
    """Analyse one of the sample clips exactly like an uploaded video."""
    item = list_sample_videos().get(name)
    if item is None:
        raise HTTPException(status_code=404, detail=f"sample video '{name}' not found")
    detector, enabled, rules = _video_options(features, rules_from, detector)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # for the annotated copy
    return _start_video_job("video-" + uuid.uuid4().hex[:8], item["path"], name, every_n, width, enabled, rules,
                            pace, record, detector)


@app.post("/videos")
async def analyze_video(
    file: UploadFile = File(...),
    every_n: int = Query(2, ge=1, description="run detection on every Nth frame"),
    width: int = Query(640, ge=0, description="resize frames to this width (0 = keep)"),
    features: Optional[str] = Query(None, description="comma separated feature keys; default all built features"),
    rules_from: Optional[str] = Query(None, description="camera id in cameras.json whose zones and limits to apply"),
    pace: bool = Query(True, description="play at the video's own speed so the stream looks natural; false = as fast as possible"),
    record: bool = Query(True, description="also write an annotated .avi for download"),
    detector: str = Query("auto", description="auto, hog (fast, people only) or yolo (object model)"),
) -> Dict[str, Any]:
    """Upload a video and analyse it like a camera: watch /cameras/{job_id}/stream, read
    /cameras/{job_id}/alerts and /cameras/{job_id}/summary, download the annotated file."""
    suffix = Path(file.filename or "video").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"unsupported video type '{suffix}'")
    detector, enabled, rules = _video_options(features, rules_from, detector)
    job_id = "video-" + uuid.uuid4().hex[:8]
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    src_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with src_path.open("wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return _start_video_job(job_id, src_path, file.filename or src_path.name, every_n, width, enabled, rules,
                            pace, record, detector)


@app.get("/videos")
def list_videos() -> Dict[str, Any]:
    return {"videos": [w.status() for cid, w in workers.items() if cid.startswith("video-")]}


@app.get("/videos/{job_id}/download")
def download_video(job_id: str) -> FileResponse:
    worker = _get_worker(job_id)
    if worker.record_path is None:
        raise HTTPException(status_code=404, detail="this job was started without recording")
    if worker.is_alive():
        raise HTTPException(status_code=409, detail="still processing, try again when the state is finished")
    if not worker.record_path.exists():
        raise HTTPException(status_code=404, detail="annotated file not found")
    return FileResponse(worker.record_path, media_type="video/x-msvideo", filename=f"{job_id}_annotated.avi")


@app.get("/cameras/{camera_id}/summary")
def camera_summary(camera_id: str) -> Dict[str, Any]:
    return _get_worker(camera_id).summary()


@app.get("/alerts")
def recent_alerts(limit: int = Query(50, ge=1, le=1000), camera_id: Optional[str] = None) -> Dict[str, Any]:
    items = alert_manager.recent(limit=limit, camera_id=camera_id)
    return {"alerts": items, "count": len(items)}


@app.post("/cameras")
def start_camera(body: CameraStart) -> Dict[str, Any]:
    worker = start_camera_worker(body.camera_id, body.source, body.every_n, body.width, body.faces, body.rules,
                                 body.features, body.detector)
    return worker.status()


@app.patch("/cameras/{camera_id}/features")
def set_camera_features(camera_id: str, body: FeaturesUpdate) -> Dict[str, Any]:
    """Switch features on or off for a running camera. Takes effect on the next frame."""
    worker = _get_worker(camera_id)
    try:
        enabled = validate_features(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    worker.rules.features = enabled
    has_face = any(isinstance(d, FaceDetector) for d in worker.detectors)
    if "face_access" in enabled and not has_face:
        worker.detectors.append(FaceDetector())
    elif "face_access" not in enabled and has_face:
        worker.detectors = [d for d in worker.detectors if not isinstance(d, FaceDetector)]
    return worker.status()


@app.post("/cameras/load")
def load_cameras(path: str = "cameras.json") -> Dict[str, Any]:
    try:
        cameras = load_cameras_config(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"config file '{path}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    started, skipped = [], []
    for cam in cameras:
        camera_id = str(cam["camera_id"])
        existing = workers.get(camera_id)
        if existing is not None and existing.is_alive():
            skipped.append(camera_id)
            continue
        start_camera_worker(
            camera_id, str(cam["source"]),
            every_n=int(cam.get("every_n", 2)), width=int(cam.get("width", 640)),
            faces=bool(cam.get("faces", False)), rules=cam.get("rules"),
            features=cam.get("features"), detector=str(cam.get("detector", "auto")),
        )
        started.append(camera_id)
    return {"started": started, "already_running": skipped}


@app.get("/cameras")
def list_cameras() -> Dict[str, Any]:
    return {"cameras": [w.status() for w in workers.values()]}


@app.get("/cameras/{camera_id}")
def camera_status(camera_id: str) -> Dict[str, Any]:
    return _get_worker(camera_id).status()


@app.delete("/cameras/{camera_id}")
def stop_camera(camera_id: str) -> Dict[str, Any]:
    worker = _get_worker(camera_id)
    worker.stop(join=True, timeout=5.0)
    workers.pop(camera_id, None)
    if camera_id.startswith("video-"):  # uploaded clip and its annotated copy are temporary
        for path in [Path(str(worker.source)), worker.record_path]:
            try:
                if path is not None and path.exists() and UPLOAD_DIR in path.resolve().parents:
                    path.unlink()
            except OSError:
                pass
    return {"stopped": camera_id, "state": worker.state, "frames": worker.frame_no, "alerts": worker.alert_count}


@app.get("/cameras/{camera_id}/alerts")
def camera_alerts(camera_id: str, limit: int = Query(50, ge=1, le=1000)) -> Dict[str, Any]:
    _get_worker(camera_id)
    items = alert_manager.recent(limit=limit, camera_id=camera_id)
    return {"camera_id": camera_id, "alerts": items, "count": len(items)}


@app.get("/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str) -> Response:
    jpeg = _get_worker(camera_id).snapshot_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="no frame yet")
    return Response(content=jpeg, media_type="image/jpeg")


def _mjpeg(worker: CameraWorker, fps: float):
    delay = 1.0 / max(1.0, fps)
    last = None
    while True:
        jpeg = worker.snapshot_jpeg()
        if jpeg is not None and jpeg is not last:
            last = jpeg
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )
        if not worker.is_alive():
            break  # a finished job still shows its last annotated frame
        time.sleep(delay)


@app.get("/cameras/{camera_id}/stream")
def camera_stream(camera_id: str, fps: float = Query(12.0, gt=0, le=60)) -> StreamingResponse:
    """MJPEG stream of the annotated frames. Open it directly in a browser <img> tag."""
    worker = _get_worker(camera_id)
    return StreamingResponse(_mjpeg(worker, fps), media_type="multipart/x-mixed-replace; boundary=frame")
