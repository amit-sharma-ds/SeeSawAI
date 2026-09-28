from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

FACE_PROTO = MODEL_DIR / "deploy.prototxt"
FACE_MODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.35

PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

CAMERA_URLS = [
    "http://localhost:8080/video",
]

ALERT_RULES = {
    "restricted_area": {"enabled": True, "min_conf": 0.35},
    "ppe": {"enabled": True, "min_conf": 0.35},
    "fire": {"enabled": True, "min_conf": 0.35},
    "crowd": {"enabled": True, "limit": 10},
}
