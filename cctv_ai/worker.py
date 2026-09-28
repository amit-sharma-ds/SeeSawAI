"""Real-time pipeline per camera: frame source -> detectors -> tracker -> rules -> alerts.

The same worker handles a video file, a webcam index, a DroidCam / IP Webcam URL or an
RTSP stream. Files are read frame by frame with media timestamps, so tests are
deterministic. Live streams are read by a background thread that keeps only the newest
frame, so slow detection never builds up lag behind the camera.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from .detectors.base import BaseDetector, Detection
from .features import default_features, filter_detections
from .rules import ZONE_COLORS, Alert, AlertManager, CameraRules, RuleEngine
from .tracking import IouTracker, Track

FONT = cv2.FONT_HERSHEY_SIMPLEX
_CRED_RE = re.compile(r"://[^/@]+@")


def mask_credentials(source) -> str:
    """rtsp://user:pass@host -> rtsp://***@host for logs and API output."""
    return _CRED_RE.sub("://***@", str(source))


def is_live_source(source) -> bool:
    if isinstance(source, int):
        return True
    text = str(source).strip().lower()
    return text.isdigit() or text.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://"))


class FrameSource:
    """Opens any source OpenCV understands and hands out frames.

    realtime=True (default for live sources) reads in a background thread and returns
    only the newest frame. realtime=False (default for files) returns every frame in order.
    """

    def __init__(self, source, realtime: Optional[bool] = None, reconnect_delay: float = 3.0, read_timeout: float = 5.0):
        text = str(source).strip()
        self.source = int(text) if text.isdigit() else source
        self.is_live = is_live_source(self.source)
        self.realtime = self.is_live if realtime is None else bool(realtime)
        self.reconnect_delay = float(reconnect_delay)
        self.read_timeout = float(read_timeout)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.frame_count = 0  # files only
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._latest: Optional[np.ndarray] = None
        self._failed = False

    @property
    def is_open(self) -> bool:
        # Only "released or never opened" counts as closed. End of stream is reported by
        # read() returning None, so the caller decides whether to reconnect or stop.
        return self.cap is not None

    def open(self) -> bool:
        self.release()
        self._generation += 1
        self._failed = False
        self._latest = None
        self._stop = threading.Event()
        self._new = threading.Event()

        cap = None
        if isinstance(self.source, int):
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            if cap is None:
                cap = cv2.VideoCapture(self.source)
        else:
            if str(self.source).lower().startswith("rtsp://"):
                # TCP transport avoids the grey/corrupted frames UDP gives on Wi-Fi
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            cap = cv2.VideoCapture(str(self.source))

        if not cap.isOpened():
            cap.release()
            return False
        if self.is_live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = 0 if self.is_live else max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

        if self.realtime:
            self._thread = threading.Thread(
                target=self._reader,
                args=(cap, self._stop, self._new, self._generation),
                name="frame-reader",
                daemon=True,
            )
            self._thread.start()
        return True

    def _reader(self, cap, stop: threading.Event, new: threading.Event, generation: int) -> None:
        try:
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if generation == self._generation:
                        self._failed = True
                    break
                if generation != self._generation:
                    break
                with self._lock:
                    self._latest = frame
                new.set()
        finally:
            cap.release()
            new.set()

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        if not self.realtime:
            ok, frame = self.cap.read()
            if not ok:
                self._failed = True
                return None
            return frame

        deadline = time.monotonic() + self.read_timeout
        while True:
            with self._lock:
                frame = self._latest
                self._latest = None
                self._new.clear()
            if frame is not None:
                return frame
            if self._failed:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._new.wait(remaining)

    def frame_time(self, frame_no: int) -> float:
        """Pipeline clock: media time for files, wall time for live streams."""
        if self.is_live or self.fps <= 0:
            return time.time()
        return frame_no / self.fps

    def release(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            if thread is not threading.current_thread():
                thread.join(timeout=3.0)
            self.cap = None  # the reader thread releases the capture itself
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = None


class CameraWorker:
    """Runs the full pipeline for one camera, in the calling thread (run) or its own (start)."""

    def __init__(
        self,
        camera_id: str,
        source,
        detectors: List[BaseDetector],
        rules: Optional[CameraRules] = None,
        alert_manager: Optional[AlertManager] = None,
        every_n: int = 2,
        width: int = 640,
        realtime: Optional[bool] = None,
        tracker: Optional[IouTracker] = None,
        on_frame: Optional[Callable[[np.ndarray, "CameraWorker"], None]] = None,
        max_frames: Optional[int] = None,
        reconnect: bool = True,
        reconnect_delay: float = 3.0,
        jpeg_quality: int = 80,
        pace: bool = False,
        record_path: Optional[str | Path] = None,
    ):
        self.camera_id = str(camera_id)
        self.source = source
        self.detectors = list(detectors)
        self.rules = rules if rules is not None else CameraRules(camera_id=self.camera_id)
        # explicit None checks: an empty AlertManager has len() == 0 and must still be reused
        self.alerts = alert_manager if alert_manager is not None else AlertManager(cooldown_seconds=self.rules.cooldown_seconds)
        self.engine = RuleEngine(self.rules, self.alerts)
        self.tracker = tracker if tracker is not None else IouTracker()
        self.every_n = max(1, int(every_n))
        self.width = int(width) if width else 0
        self.realtime = realtime
        self.on_frame = on_frame
        self.max_frames = int(max_frames) if max_frames else None
        self.reconnect = bool(reconnect)
        self.reconnect_delay = float(reconnect_delay)
        self.jpeg_quality = int(jpeg_quality)
        self.pace = bool(pace)  # play files no faster than their own frame rate
        self.record_path = Path(record_path) if record_path else None
        self._writer: Optional[cv2.VideoWriter] = None
        self._pace_start: Optional[float] = None

        self.state = "created"
        self.frame_no = 0
        self.total_frames = 0
        self.detections_run = 0
        self.fps = 0.0
        self.detect_ms = 0.0
        self.last_error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.source_fps = 0.0
        self.latest_tracks: List[Track] = []
        self.latest_detections: List[Detection] = []
        self.alert_count = 0
        self.people_max = 0  # most people seen at the same time

        self._flagged: Dict[int, float] = {}
        self._last_alert: Optional[Alert] = None
        self._last_alert_at = 0.0
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._jpeg: Optional[bytes] = None
        self._jpeg_id = -1
        self._tick_times: Deque[float] = deque(maxlen=40)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "CameraWorker":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name=f"camera-{self.camera_id}", daemon=True)
        self._thread.start()
        return self

    def stop(self, join: bool = True, timeout: float = 5.0) -> None:
        self._stop.set()
        if join and self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self) -> None:
        self.state = "starting"
        self.started_at = time.time()
        source = FrameSource(self.source, realtime=self.realtime, reconnect_delay=self.reconnect_delay)
        try:
            while not self._stop.is_set():
                if not source.is_open:
                    if not source.open():
                        self.last_error = f"cannot open source {mask_credentials(self.source)}"
                        if not (source.is_live and self.reconnect):
                            self.state = "error"
                            break
                        self.state = "reconnecting"
                        self._stop.wait(source.reconnect_delay)
                        continue
                    self.source_fps = source.fps
                    self.total_frames = source.frame_count
                    self.state = "running"
                    self.last_error = None

                frame = source.read()
                if frame is None:
                    if source.is_live and self.reconnect:
                        self.last_error = "stream dropped, reconnecting"
                        self.state = "reconnecting"
                        source.release()
                        self._stop.wait(source.reconnect_delay)
                        continue
                    break

                self.frame_no += 1
                if self.pace and not source.is_live and source.fps > 0:
                    if self._pace_start is None:
                        self._pace_start = time.perf_counter()
                    due = self._pace_start + (self.frame_no - 1) / source.fps
                    wait = due - time.perf_counter()
                    if wait > 0:
                        self._stop.wait(wait)
                now = source.frame_time(self.frame_no)
                frame = self._resize(frame)
                if (self.frame_no - 1) % self.every_n == 0:
                    self._detect(frame, now)
                annotated = self._annotate(frame, now)
                self._record(annotated, source.fps)
                with self._lock:
                    self._frame = annotated
                    self._frame_id += 1
                self._tick()
                if self.on_frame is not None:
                    self.on_frame(annotated, self)
                if self.max_frames and self.frame_no >= self.max_frames:
                    break
        except Exception as exc:  # keep the thread alive long enough to report
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
            traceback.print_exc()
        finally:
            source.release()
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self.state != "error":
                self.state = "stopped" if self._stop.is_set() else "finished"

    def _record(self, frame: np.ndarray, source_fps: float) -> None:
        if self.record_path is None:
            return
        if self._writer is None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            fps = source_fps if 1.0 <= source_fps <= 60.0 else 10.0
            self._writer = cv2.VideoWriter(str(self.record_path), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                                           (frame.shape[1], frame.shape[0]))
            if not self._writer.isOpened():
                self.last_error = f"cannot write {self.record_path}"
                self._writer = None
                self.record_path = None
                return
        self._writer.write(frame)

    # -- per-frame steps ----------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.width <= 0 or frame.shape[1] == self.width:
            return frame
        scale = self.width / frame.shape[1]
        return cv2.resize(frame, (self.width, max(1, int(round(frame.shape[0] * scale)))))

    def _detect(self, frame: np.ndarray, now: float) -> None:
        t0 = time.perf_counter()
        detections: List[Detection] = []
        for detector in self.detectors:
            detections.extend(detector.detect(frame))
        detections = filter_detections(detections, self.rules.features)  # only switched-on features
        self.detect_ms = (time.perf_counter() - t0) * 1000.0
        self.detections_run += 1

        tracks = self.tracker.update(detections, now)
        new_alerts = self.engine.evaluate(tracks, frame.shape, now)
        self.latest_detections = detections
        self.latest_tracks = tracks
        self.people_max = max(self.people_max, self.counts().get("person", 0))
        for alert in new_alerts:
            self.alert_count += 1
            self._last_alert = alert
            self._last_alert_at = now
            if alert.track_id is not None:
                self._flagged[alert.track_id] = now
        self._flagged = {tid: t for tid, t in self._flagged.items() if now - t <= 4.0}

    def _tick(self) -> None:
        # frames per second over the last few seconds, so bursts do not distort it
        self._tick_times.append(time.perf_counter())
        if len(self._tick_times) >= 2:
            span = self._tick_times[-1] - self._tick_times[0]
            self.fps = (len(self._tick_times) - 1) / span if span > 0 else 0.0

    @staticmethod
    def _text(img, text, org, scale=0.5, color=(255, 255, 255), thickness=1):
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

    def counts(self) -> Dict[str, int]:
        """How many tracked objects of each category are in view right now."""
        counts: Dict[str, int] = {}
        for track in self.latest_tracks:
            counts[track.category] = counts.get(track.category, 0) + 1
        return counts

    def object_counts(self) -> Dict[str, int]:
        """Everything in view that is not a person or a vehicle, by class name: bottle, cell phone, knife..."""
        counts: Dict[str, int] = {}
        for track in self.latest_tracks:
            if track.category not in ("person", "vehicle"):
                counts[track.class_name] = counts.get(track.class_name, 0) + 1
        return counts

    def _draw_counter(self, out: np.ndarray) -> None:
        """Top-right counter: people and vehicles in view now, and the running totals."""
        counts = self.counts()
        seen = self.tracker.confirmed_by_category
        people = counts.get("person", 0)
        label = f"PEOPLE {people}"
        if self.rules.enabled("vehicle") or counts.get("vehicle") or seen.get("vehicle"):
            label += f"   VEHICLES {counts.get('vehicle', 0)}"
        h, w = out.shape[:2]
        scale = max(0.7, w / 800)
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 2)
        x = w - tw - 16
        cv2.rectangle(out, (x - 8, 6), (w - 6, th + 22), (0, 0, 0), -1)
        active = people or counts.get("vehicle", 0)
        cv2.putText(out, label, (x, th + 14), FONT, scale, (255, 255, 255) if active else (200, 200, 200), 2, cv2.LINE_AA)
        totals = [f"{k}s {v}" for k, v in sorted(seen.items()) if k in ("person", "vehicle")]
        line2 = ("total passed: " + ", ".join(totals)) if totals else ""
        others = [f"{k} {v}" for k, v in sorted(self.object_counts().items())]
        if others:
            line2 = (line2 + "   " if line2 else "") + "objects: " + ", ".join(others)
        if line2:
            (lw, _), _ = cv2.getTextSize(line2, FONT, 0.5, 1)
            self._text(out, line2, (max(8, w - lw - 14), th + 42), 0.5, (255, 255, 255), 1)

    def _annotate(self, frame: np.ndarray, now: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for zone in self.rules.zones:
            color = ZONE_COLORS.get(zone.kind, (255, 255, 255))
            pts = zone.pixels(w, h)
            cv2.polylines(out, [pts], True, color, 2)
            x, y = int(pts[0][0][0]), int(pts[0][0][1])
            self._text(out, f"{zone.kind}: {zone.name}", (x + 4, min(h - 4, y + 16)), 0.45, color)
        for track in self.latest_tracks:
            x1, y1, x2, y2 = track.bbox
            color = (0, 0, 255) if track.track_id in self._flagged else (200, 200, 200)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            self._text(out, f"{track.class_name} #{track.track_id}", (x1, max(12, y1 - 6)), 0.5, color)
        header = f"{self.camera_id}  {self.fps:.1f} fps  detect {self.detect_ms:.0f} ms  tracks {len(self.latest_tracks)}"
        self._text(out, header, (8, 20), 0.55, (255, 255, 255), 1)
        self._draw_counter(out)
        if self._last_alert is not None and now - self._last_alert_at <= 4.0:
            self._text(out, f"ALERT {self._last_alert.type}: {self._last_alert.message}", (8, h - 12), 0.55, (0, 0, 255), 1)
        return out

    # -- read side ----------------------------------------------------------

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._lock:
            frame, fid = self._frame, self._frame_id
            if frame is None:
                return None
            if self._jpeg_id == fid:
                return self._jpeg
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        data = buf.tobytes()
        with self._lock:
            self._jpeg, self._jpeg_id = data, fid
        return data

    def summary(self) -> dict:
        """Totals for a finished (or running) job: alerts by type, people seen, duration."""
        by_type: Dict[str, int] = {}
        for alert in self.alerts.recent(limit=100000, camera_id=self.camera_id):
            by_type[alert["type"]] = by_type.get(alert["type"], 0) + 1
        return {
            "camera_id": self.camera_id,
            "state": self.state,
            "frames": self.frame_no,
            "total_frames": self.total_frames,
            "duration_seconds": round(self.frame_no / self.source_fps, 1) if self.source_fps > 0 else None,
            "objects_tracked": self.tracker.total_created,
            "seen_by_category": dict(self.tracker.confirmed_by_category),
            "people_max": self.people_max,
            "alerts": self.alert_count,
            "alerts_by_type": by_type,
            "record_path": str(self.record_path) if self.record_path else None,
        }

    def status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source": mask_credentials(self.source),
            "kind": "camera" if is_live_source(self.source) else "video",
            "state": self.state,
            "alive": self.is_alive(),
            "frames": self.frame_no,
            "total_frames": self.total_frames,
            "progress": round(min(1.0, self.frame_no / self.total_frames), 3) if self.total_frames else None,
            "detections_run": self.detections_run,
            "fps": round(self.fps, 1),
            "detect_ms": round(self.detect_ms, 1),
            "every_n": self.every_n,
            "width": self.width,
            "features": sorted(self.rules.features if self.rules.features is not None else default_features()),
            "counts": self.counts(),
            "objects": self.object_counts(),
            "seen": dict(self.tracker.confirmed_by_category),
            "people": self.counts().get("person", 0),
            "people_max": self.people_max,
            "tracks": [t.as_dict() for t in self.latest_tracks],
            "alerts": self.alert_count,
            "last_alert": self._last_alert.as_dict() if self._last_alert else None,
            "last_error": self.last_error,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
        }
