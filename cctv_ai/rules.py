"""Zone, time and direction rules on top of tracked detections, plus alert bookkeeping.

Features covered here need only person/vehicle boxes with track ids:

  restricted_area    a person's feet are inside a zone of kind "restricted"
  industrial_safety  a person's feet are inside a zone of kind "danger" (machine areas)
  loitering          a person stays in a "loiter" zone for loiter_seconds, or, when no
                     loiter zone exists, stays within loiter_radius of one spot that long
  crowd              at least crowd_limit people inside the "crowd" zones (or anywhere)
                     for crowd_min_hits detection rounds in a row
  wrong_direction    a track's movement over the last window points against the
                     allowed direction (image coordinates: x right, y down)

Zone polygons use normalised 0..1 coordinates so they survive a change of resolution.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .features import is_weapon_class
from .tracking import Track

ZONE_KINDS = ("restricted", "danger", "loiter", "crowd")
# BGR colours used when drawing zones
ZONE_COLORS = {
    "restricted": (0, 0, 255),
    "danger": (0, 128, 255),
    "loiter": (0, 220, 220),
    "crowd": (255, 200, 0),
}


@dataclass
class Zone:
    name: str
    kind: str
    polygon: List[List[float]]  # normalised [[x, y], ...], 0..1

    def __post_init__(self):
        if self.kind not in ZONE_KINDS:
            raise ValueError(f"zone '{self.name}': kind must be one of {ZONE_KINDS}, got '{self.kind}'")
        if len(self.polygon) < 3:
            raise ValueError(f"zone '{self.name}': a polygon needs at least 3 points")
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"zone '{self.name}': polygon points must be normalised 0..1, got ({x}, {y})")

    def pixels(self, frame_w: int, frame_h: int) -> np.ndarray:
        pts = [[int(round(x * frame_w)), int(round(y * frame_h))] for x, y in self.polygon]
        return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    def contains(self, x: float, y: float, frame_w: int, frame_h: int) -> bool:
        pts = np.array([[px * frame_w, py * frame_h] for px, py in self.polygon], dtype=np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    @classmethod
    def from_dict(cls, d: dict) -> "Zone":
        return cls(
            name=str(d["name"]),
            kind=str(d.get("kind", "restricted")),
            polygon=[[float(x), float(y)] for x, y in d["polygon"]],
        )

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "polygon": self.polygon}


@dataclass
class CameraRules:
    camera_id: str = "cam"
    zones: List[Zone] = field(default_factory=list)
    loiter_seconds: float = 30.0
    loiter_radius: float = 0.15  # fraction of frame width, only used without a loiter zone
    crowd_limit: int = 5
    crowd_min_hits: int = 3
    direction: Optional[Dict[str, Any]] = None  # {"allowed": [dx, dy], "min_travel": 0.15, "window_seconds": 3}
    cooldown_seconds: float = 30.0
    features: Optional[Set[str]] = None  # enabled feature keys, None = every built feature

    def enabled(self, feature: str) -> bool:
        return self.features is None or feature in self.features

    @classmethod
    def from_dict(cls, d: Optional[dict], camera_id: Optional[str] = None) -> "CameraRules":
        d = dict(d or {})
        features = d.get("features")
        return cls(
            camera_id=camera_id or str(d.get("camera_id", "cam")),
            zones=[Zone.from_dict(z) for z in d.get("zones", [])],
            loiter_seconds=float(d.get("loiter_seconds", 30.0)),
            loiter_radius=float(d.get("loiter_radius", 0.15)),
            crowd_limit=int(d.get("crowd_limit", 5)),
            crowd_min_hits=int(d.get("crowd_min_hits", 3)),
            direction=d.get("direction") or None,
            cooldown_seconds=float(d.get("cooldown_seconds", 30.0)),
            features=set(str(f) for f in features) if features is not None else None,
        )

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "zones": [z.as_dict() for z in self.zones],
            "loiter_seconds": self.loiter_seconds,
            "loiter_radius": self.loiter_radius,
            "crowd_limit": self.crowd_limit,
            "crowd_min_hits": self.crowd_min_hits,
            "direction": self.direction,
            "cooldown_seconds": self.cooldown_seconds,
            "features": sorted(self.features) if self.features is not None else None,
        }


@dataclass
class Alert:
    type: str
    severity: str
    camera_id: str
    message: str
    timestamp: float  # pipeline clock: seconds into a file, or wall time for live streams
    track_id: Optional[int] = None
    zone: Optional[str] = None
    bbox: Optional[List[int]] = None
    wall_time: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.wall_time))
        return d


class AlertManager:
    """Stores alerts, enforces a per-key cooldown, optionally logs JSON lines and notifies subscribers."""

    def __init__(self, cooldown_seconds: float = 30.0, max_alerts: int = 1000, log_path: Optional[str | Path] = None):
        self.cooldown_seconds = float(cooldown_seconds)
        self.alerts: Deque[Alert] = deque(maxlen=max_alerts)
        self.log_path = Path(log_path) if log_path else None
        self._last: Dict[tuple, float] = {}
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        self._callbacks.append(callback)

    def raise_alert(self, alert: Alert, cooldown: Optional[float] = None, key: Optional[tuple] = None) -> Optional[Alert]:
        """Record the alert unless the same key fired within the cooldown. Returns it when recorded."""
        key = key or (alert.camera_id, alert.type, alert.zone, alert.track_id)
        wait = self.cooldown_seconds if cooldown is None else float(cooldown)
        with self._lock:
            last = self._last.get(key)
            if last is not None and alert.timestamp - last < wait:
                return None
            self._last[key] = alert.timestamp
            self.alerts.append(alert)
        if self.log_path is not None:
            try:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(alert.as_dict()) + "\n")
            except OSError:
                pass
        for callback in list(self._callbacks):
            try:
                callback(alert)
            except Exception:
                pass
        return alert

    def recent(self, limit: int = 50, camera_id: Optional[str] = None) -> List[dict]:
        with self._lock:
            items = list(self.alerts)
        if camera_id:
            items = [a for a in items if a.camera_id == camera_id]
        return [a.as_dict() for a in reversed(items[-limit:])]

    def clear(self) -> None:
        with self._lock:
            self.alerts.clear()
            self._last.clear()

    def __len__(self) -> int:
        return len(self.alerts)

    def __bool__(self) -> bool:
        # never falsy, even when empty, so `manager or AlertManager()` cannot drop it
        return True


class RuleEngine:
    """Evaluates one camera's rules against the current tracks. Keeps the per-track timers."""

    def __init__(self, rules: CameraRules, alerts: Optional[AlertManager] = None):
        self.rules = rules
        self.alerts = alerts if alerts is not None else AlertManager(cooldown_seconds=rules.cooldown_seconds)
        self._inside_since: Dict[Tuple[int, str], float] = {}
        self._crowd_hits = 0

    def evaluate(self, tracks: List[Track], frame_shape, now: float) -> List[Alert]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        people = [t for t in tracks if t.class_name == "person"]
        alive = {t.track_id for t in tracks}

        raised: List[Alert] = []
        raised += self._zone_rules(people, w, h, now)
        raised += self._loiter_anywhere(people, w, now)
        raised += self._crowd_rule(people, w, h, now)
        raised += self._direction_rule(tracks, w, now)
        raised += self._model_rules(tracks, now)

        self._inside_since = {k: v for k, v in self._inside_since.items() if k[0] in alive}
        return raised

    # -- individual rules -------------------------------------------------

    def _zone_rules(self, people: List[Track], w: int, h: int, now: float) -> List[Alert]:
        raised: List[Alert] = []
        for track in people:
            fx, fy = track.foot
            for zone in self.rules.zones:
                key = (track.track_id, zone.name)
                if not zone.contains(fx, fy, w, h):
                    self._inside_since.pop(key, None)
                    continue
                since = self._inside_since.setdefault(key, now)
                if zone.kind == "restricted" and self.rules.enabled("restricted_area"):
                    raised += self._emit(
                        "restricted_area", "high",
                        f"Person #{track.track_id} entered restricted zone '{zone.name}'",
                        now, track, zone.name,
                    )
                elif zone.kind == "danger" and self.rules.enabled("industrial_safety"):
                    raised += self._emit(
                        "industrial_safety", "critical",
                        f"Person #{track.track_id} inside danger zone '{zone.name}'",
                        now, track, zone.name,
                    )
                elif (zone.kind == "loiter" and self.rules.enabled("loitering")
                      and now - since >= self.rules.loiter_seconds > 0):
                    raised += self._emit(
                        "loitering", "medium",
                        f"Person #{track.track_id} loitering in '{zone.name}' for {now - since:.0f}s",
                        now, track, zone.name,
                    )
        return raised

    def _loiter_anywhere(self, people: List[Track], w: int, now: float) -> List[Alert]:
        if not self.rules.enabled("loitering"):
            return []
        if self.rules.loiter_seconds <= 0 or any(z.kind == "loiter" for z in self.rules.zones):
            return []
        raised: List[Alert] = []
        radius_px = self.rules.loiter_radius * w
        for track in people:
            covered, radius = track.travel_span(self.rules.loiter_seconds)
            if covered >= self.rules.loiter_seconds and radius <= radius_px:
                raised += self._emit(
                    "loitering", "medium",
                    f"Person #{track.track_id} standing in the same spot for {covered:.0f}s",
                    now, track, None,
                )
        return raised

    def _crowd_rule(self, people: List[Track], w: int, h: int, now: float) -> List[Alert]:
        if self.rules.crowd_limit <= 0 or not self.rules.enabled("crowd"):
            self._crowd_hits = 0
            return []
        crowd_zones = [z for z in self.rules.zones if z.kind == "crowd"]
        if crowd_zones:
            count = sum(1 for t in people if any(z.contains(*t.foot, w, h) for z in crowd_zones))
            zone_name = ",".join(z.name for z in crowd_zones)
        else:
            count = len(people)
            zone_name = None
        if count >= self.rules.crowd_limit:
            self._crowd_hits += 1
        else:
            self._crowd_hits = 0
            return []
        if self._crowd_hits < self.rules.crowd_min_hits:
            return []
        return self._emit(
            "crowd", "high",
            f"Crowd of {count} people (limit {self.rules.crowd_limit})",
            now, None, zone_name,
        )

    def _direction_rule(self, tracks: List[Track], w: int, now: float) -> List[Alert]:
        cfg = self.rules.direction
        if not cfg or not cfg.get("allowed") or not self.rules.enabled("wrong_direction"):
            return []
        ax, ay = float(cfg["allowed"][0]), float(cfg["allowed"][1])
        norm = math.hypot(ax, ay)
        if norm == 0:
            return []
        ax, ay = ax / norm, ay / norm
        min_travel = float(cfg.get("min_travel", 0.15)) * w
        window = float(cfg.get("window_seconds", 3.0))
        classes = set(cfg.get("classes") or [])
        raised: List[Alert] = []
        for track in tracks:
            if classes and track.class_name not in classes:
                continue
            dx, dy, dt = track.displacement(window)
            dist = math.hypot(dx, dy)
            if dt <= 0 or dist < min_travel:
                continue
            cos = (dx * ax + dy * ay) / dist
            if cos < -0.5:  # more than 120 degrees away from the allowed direction
                raised += self._emit(
                    "wrong_direction", "medium",
                    f"{track.class_name.title()} #{track.track_id} moving against the allowed direction",
                    now, track, None,
                )
        return raised

    def _model_rules(self, tracks: List[Track], now: float) -> List[Alert]:
        """Alerts that need nothing but a detection from a trained model: weapon, fire or
        smoke, and missing PPE (classes named no-helmet, without_vest and so on)."""
        raised: List[Alert] = []
        for track in tracks:
            label = track.class_name
            if (track.category == "weapon" or is_weapon_class(label)) and self.rules.enabled("weapon"):
                raised += self._emit("weapon", "critical", f"Weapon detected: {label} (#{track.track_id})",
                                     now, track, None)
            elif track.category == "fire_smoke" and self.rules.enabled("fire_smoke"):
                raised += self._emit("fire_smoke", "critical", f"{label.title()} detected (#{track.track_id})",
                                     now, track, None)
            elif track.category == "ppe" and self.rules.enabled("ppe") and ppe_missing(label):
                raised += self._emit("ppe", "high", f"PPE missing: {label} (#{track.track_id})",
                                     now, track, None)
        return raised

    def _emit(self, type_: str, severity: str, message: str, now: float,
              track: Optional[Track], zone: Optional[str]) -> List[Alert]:
        alert = Alert(
            type=type_,
            severity=severity,
            camera_id=self.rules.camera_id,
            message=message,
            timestamp=now,
            track_id=track.track_id if track else None,
            zone=zone,
            bbox=[int(v) for v in track.bbox] if track else None,
        )
        recorded = self.alerts.raise_alert(alert, cooldown=self.rules.cooldown_seconds)
        return [recorded] if recorded else []


def ppe_missing(class_name: str) -> bool:
    """True for PPE model classes that mean equipment is absent: NO-Hardhat, no_vest, without-mask."""
    label = re.sub(r"[\s_]+", "-", class_name.strip().lower())
    return label.startswith(("no-", "without", "missing", "not-")) or label in {"nohelmet", "novest", "nomask", "noglove", "nogloves"}


# -- config file helpers --------------------------------------------------

def load_cameras_config(path: str | Path) -> List[dict]:
    """Read cameras.json and validate every entry. Returns the list of camera dicts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = data if isinstance(data, list) else data.get("cameras", [])
    for cam in cameras:
        if "camera_id" not in cam or "source" not in cam:
            raise ValueError("each camera entry needs 'camera_id' and 'source'")
        CameraRules.from_dict(cam.get("rules"), str(cam["camera_id"]))
    return cameras


def save_cameras_config(path: str | Path, cameras: List[dict]) -> None:
    Path(path).write_text(json.dumps({"cameras": cameras}, indent=2), encoding="utf-8")
