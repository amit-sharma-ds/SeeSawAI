"""The 14 product features: what each one is, its build status, and what it needs.

Feature keys are used in cameras.json ("features": [...]), in POST /cameras, in
PATCH /cameras/{id}/features, in /detect-image?features=... and in the web page toggles.
Only enabled features produce detections and alerts. A camera with no explicit list runs
every feature that is built.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .detectors.base import Detection
from .detectors.yolo_onnx import YoloOnnxDetector, custom_model_available

# "model" names the file a feature needs: "coco" is the general 80-class model
# (models/yolov8n.onnx or the OpenCV-zoo YOLOX), anything else is models/<model>.onnx plus
# models/<model>.names, a model you train yourself (see README, "Train your own detector").
FEATURES: List[Dict] = [
    dict(key="person", number=1, name="Person Detection",
         description="A person appears in the camera view",
         needs=[], scope=["image", "live"], status="available"),
    dict(key="vehicle", number=2, name="Vehicle Detection",
         description="Car, truck, bus, motorcycle, bicycle",
         needs=["object model"], model="coco", scope=["image", "live"], status="needs_model"),
    dict(key="restricted_area", number=3, name="Restricted Area Entry",
         description="Someone walks into a prohibited zone",
         needs=["person", "zone:restricted"], scope=["live"], status="available"),
    dict(key="loitering", number=4, name="Loitering Detection",
         description="A person stays unusually long in one place",
         needs=["person"], scope=["live"], status="available"),
    dict(key="fight", number=5, name="Fight / Aggression Detection",
         description="Unusual physical activity between people",
         needs=["action recognition model"], scope=["live"], status="planned"),
    dict(key="fire_smoke", number=6, name="Fire / Smoke Detection",
         description="Signs of smoke or fire; alerts as soon as either is seen",
         needs=["models/fire_smoke.onnx + fire_smoke.names"], model="fire_smoke",
         scope=["image", "live"], status="needs_model"),
    dict(key="ppe", number=7, name="PPE Detection",
         description="Helmet, safety jacket, gloves; alerts on classes named no-helmet, no-vest and so on",
         needs=["models/ppe.onnx + ppe.names"], model="ppe",
         scope=["image", "live"], status="needs_model"),
    dict(key="wrong_direction", number=8, name="Wrong Direction",
         description="Person or vehicle moving against the allowed direction",
         needs=["person", "direction"], scope=["live"], status="available"),
    dict(key="object_left_taken", number=9, name="Object Detection (left / taken)",
         description="Bags, packages and objects; the left-behind timing rule comes next",
         needs=["object model"], model="coco", scope=["image", "live"], status="needs_model"),
    dict(key="crowd", number=10, name="Crowd Detection",
         description="Too many people in one area",
         needs=["person"], scope=["live"], status="available"),
    dict(key="industrial_safety", number=11, name="Industrial Safety",
         description="Person inside a machine danger zone",
         needs=["person", "zone:danger"], scope=["live"], status="available"),
    dict(key="face_access", number=12, name="Face / Identity Access",
         description="Face detection works now; matching against authorised people comes next",
         needs=["face model (auto-download)"], scope=["image", "live"], status="partial"),
    dict(key="human_activity", number=13, name="Human Activity Detection",
         description="What people are doing",
         needs=["action recognition model"], scope=["live"], status="planned"),
    dict(key="material_movement", number=14, name="Material Movement Detection",
         description="Goods crossing a line or leaving a zone",
         needs=["object model", "line-crossing rule"], scope=["live"], status="planned"),
    dict(key="weapon", number=15, name="Gun / Knife Detection",
         description="Knives come from the object model; guns and other weapons need a trained weapon model. Alerts immediately",
         needs=["object model (knife)", "models/weapon.onnx + weapon.names (guns)"], model="weapon",
         scope=["image", "live"], status="needs_model"),
]
# class labels that count as a weapon, from the COCO model or from a custom weapon model
WEAPON_CLASSES = {"knife", "gun", "pistol", "handgun", "rifle", "shotgun", "revolver", "firearm",
                  "weapon", "machete", "sword"}


def is_weapon_class(class_name: str) -> bool:
    """'Knife', 'hand_gun', 'Hand-Gun' and 'handgun' all count."""
    compact = "".join(ch for ch in class_name.lower() if ch.isalnum())
    return compact in WEAPON_CLASSES
FEATURE_KEYS: List[str] = [f["key"] for f in FEATURES]
_BY_KEY: Dict[str, Dict] = {f["key"]: f for f in FEATURES}

# which feature switch controls detections of each Detection.category
CATEGORY_FEATURE = {
    "person": "person",
    "vehicle": "vehicle",
    "bag": "object_left_taken",
    "object": "object_left_taken",
    "face": "face_access",
    "weapon": "weapon",
    "fire_smoke": "fire_smoke",
    "ppe": "ppe",
}


def model_available(model: str) -> bool:
    if model == "coco":
        return YoloOnnxDetector.available()
    return custom_model_available(model)


def feature_status(feature: Dict) -> str:
    if feature["key"] == "weapon":
        if custom_model_available("weapon"):
            return "available"
        return "partial" if YoloOnnxDetector.available() else "needs_model"  # knives only
    model = feature.get("model")
    if model:
        return "available" if model_available(model) else "needs_model"
    return feature["status"]


def feature_note(feature: Dict, status: str) -> str:
    """Short badge text for the web page."""
    if feature["key"] == "weapon" and status == "partial":
        return "knife only, gun model needed"
    return {"partial": "detection only", "needs_model": "needs model file", "planned": "coming soon"}.get(status, "")


def _default_on(feature: Dict) -> bool:
    # face_access stays off until asked for: it downloads its model and only detects faces so far
    return feature["status"] != "planned" and feature["key"] != "face_access"


def feature_catalog() -> List[Dict]:
    """All 14 features with their live status, for the API and the web page."""
    catalog = []
    for feature in FEATURES:
        status = feature_status(feature)
        catalog.append({**feature, "status": status, "note": feature_note(feature, status),
                        "default_on": _default_on(feature)})
    return catalog


def default_features() -> Set[str]:
    return {f["key"] for f in FEATURES if _default_on(f)}


def validate_features(keys: Iterable[str]) -> Set[str]:
    """Normalise a list of feature keys. Raises ValueError for unknown or not-yet-built ones."""
    result: Set[str] = set()
    for raw in keys:
        key = str(raw).strip()
        if not key:
            continue
        feature = _BY_KEY.get(key)
        if feature is None:
            raise ValueError(f"unknown feature '{key}'. Valid keys: {', '.join(FEATURE_KEYS)}")
        if feature["status"] == "planned":
            raise ValueError(f"feature '{key}' ({feature['name']}) is not built yet")
        result.add(key)
    return result


def parse_feature_list(text: Optional[str]) -> Optional[Set[str]]:
    """'person,vehicle' -> {'person', 'vehicle'}; None or '' -> None (meaning all)."""
    if text is None or not text.strip():
        return None
    return validate_features(text.split(","))


def feature_for_category(category: str) -> str:
    return CATEGORY_FEATURE.get(category, "object_left_taken")


def filter_detections(detections: List[Detection], features: Optional[Set[str]]) -> List[Detection]:
    """Keep only detections whose feature switch is on. None means everything is on.
    A knife from the object model passes when either Object Detection or Gun / Knife is on."""
    if features is None:
        return list(detections)
    kept = []
    for det in detections:
        if feature_for_category(det.category) in features:
            kept.append(det)
        elif "weapon" in features and is_weapon_class(det.class_name):
            kept.append(det)
    return kept
