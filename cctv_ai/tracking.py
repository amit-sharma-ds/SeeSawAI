"""Lightweight multi-object tracker: greedy IoU matching with a centroid-distance fallback.

Good enough for people and vehicles on CCTV at a few detections per second, and it
needs nothing beyond numpy. Every track keeps a position history, which the rule
engine uses for loitering and wrong-direction checks.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from .detectors.base import Detection, box_iou


def centroid_of(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    track_id: int
    bbox: List[int]
    class_name: str
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    category: str = "object"  # person, vehicle, bag, face, weapon, fire_smoke, ppe, object
    # (time, cx, cy) samples, oldest first
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=3000))

    @property
    def centroid(self) -> Tuple[float, float]:
        return centroid_of(self.bbox)

    @property
    def foot(self) -> Tuple[float, float]:
        """Bottom-centre of the box, i.e. where a person stands. Zone checks use this."""
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, float(y2)

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def diagonal(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return math.hypot(x2 - x1, y2 - y1)

    def samples_since(self, seconds: float) -> List[Tuple[float, float, float]]:
        """History covering the last `seconds`, plus the one sample just before the window."""
        cutoff = self.last_seen - seconds
        samples = list(self.history)
        start = max(0, len(samples) - 1)
        for i, sample in enumerate(samples):
            if sample[0] >= cutoff:
                start = max(0, i - 1)
                break
        return samples[start:]

    def displacement(self, window_seconds: float) -> Tuple[float, float, float]:
        """(dx, dy, dt) from the oldest sample in the window to the newest."""
        samples = self.samples_since(window_seconds)
        if len(samples) < 2:
            return 0.0, 0.0, 0.0
        t0, x0, y0 = samples[0]
        t1, x1, y1 = samples[-1]
        return x1 - x0, y1 - y0, t1 - t0

    def travel_span(self, window_seconds: float) -> Tuple[float, float]:
        """(seconds covered by history, farthest distance from the current position) in the window."""
        samples = self.samples_since(window_seconds)
        if not samples:
            return 0.0, 0.0
        cx, cy = self.centroid
        radius = max(math.hypot(x - cx, y - cy) for _, x, y in samples)
        return self.last_seen - samples[0][0], radius

    def as_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "category": self.category,
            "confidence": round(float(self.confidence), 3),
            "bbox": [int(v) for v in self.bbox],
            "age_seconds": round(self.age, 2),
            "hits": self.hits,
        }


class IouTracker:
    """Assigns stable ids to detections across frames.

    Matching is greedy on IoU first; leftovers are matched by centroid distance within
    one box diagonal, which catches fast movers when detection runs at a low rate.
    A track is reported once it has `min_hits` matches, and dropped after `max_misses`
    consecutive detection rounds without a match.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 10,
        min_hits: int = 2,
        distance_ratio: float = 1.0,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)
        self.distance_ratio = float(distance_ratio)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.confirmed_by_category: Dict[str, int] = {}  # distinct objects that reached min_hits

    def reset(self) -> None:
        self.tracks = {}
        self._next_id = 1
        self.confirmed_by_category = {}

    def _confirm(self, track: Track) -> None:
        if track.hits == self.min_hits:
            self.confirmed_by_category[track.category] = self.confirmed_by_category.get(track.category, 0) + 1

    @property
    def total_created(self) -> int:
        """How many distinct tracks have existed so far."""
        return self._next_id - 1

    def update(self, detections: List[Detection], now: float) -> List[Track]:
        tracks = list(self.tracks.values())
        matched_tracks: set = set()
        matched_dets: set = set()

        if tracks and detections:
            pairs = []
            for i, track in enumerate(tracks):
                for j, det in enumerate(detections):
                    if det.class_name != track.class_name:
                        continue
                    score = box_iou(track.bbox, det.bbox)
                    if score >= self.iou_threshold:
                        pairs.append((score, i, j))
            pairs.sort(reverse=True)
            for _, i, j in pairs:
                if i in matched_tracks or j in matched_dets:
                    continue
                matched_tracks.add(i)
                matched_dets.add(j)
                self._absorb(tracks[i], detections[j], now)

            for i, track in enumerate(tracks):
                if i in matched_tracks:
                    continue
                tcx, tcy = track.centroid
                limit = self.distance_ratio * track.diagonal
                best_j, best_dist = None, None
                for j, det in enumerate(detections):
                    if j in matched_dets or det.class_name != track.class_name:
                        continue
                    dcx, dcy = centroid_of(det.bbox)
                    dist = math.hypot(dcx - tcx, dcy - tcy)
                    if dist <= limit and (best_dist is None or dist < best_dist):
                        best_j, best_dist = j, dist
                if best_j is not None:
                    matched_tracks.add(i)
                    matched_dets.add(best_j)
                    self._absorb(track, detections[best_j], now)

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create(det, now)

        for i, track in enumerate(tracks):
            if i not in matched_tracks:
                track.misses += 1

        self.tracks = {tid: t for tid, t in self.tracks.items() if t.misses <= self.max_misses}
        return self.active_tracks()

    def active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits and t.misses == 0]

    def _absorb(self, track: Track, det: Detection, now: float) -> None:
        track.bbox = [int(v) for v in det.bbox]
        track.confidence = float(det.confidence)
        track.hits += 1
        track.misses = 0
        track.last_seen = now
        cx, cy = track.centroid
        track.history.append((now, cx, cy))
        self._confirm(track)

    def _create(self, det: Detection, now: float) -> Track:
        track = Track(
            track_id=self._next_id,
            bbox=[int(v) for v in det.bbox],
            class_name=det.class_name,
            confidence=float(det.confidence),
            first_seen=now,
            last_seen=now,
            category=str(det.category or "object"),
        )
        cx, cy = track.centroid
        track.history.append((now, cx, cy))
        self.tracks[track.track_id] = track
        self._next_id += 1
        self._confirm(track)
        return track
