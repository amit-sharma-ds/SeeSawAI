"""Fake camera for demos and tests: an MJPEG stream on http://127.0.0.1:8765/video.

It pans a photo with people back and forth so the person detector, tracker and zone
rules have something to work on when no phone or CCTV camera is available. The photo is
the sample image that ships inside the ultralytics package; without it, white "person"
blocks are used instead.

  py demo_camera.py               # serve on port 8765
  py live.py http://127.0.0.1:8765/video --camera cam1
"""
from __future__ import annotations

import argparse
import importlib.util
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480


def find_sample_photo(name: str = "bus.jpg"):
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.submodule_search_locations:
        return None
    photo = Path(list(spec.submodule_search_locations)[0]) / "assets" / name
    return photo if photo.exists() else None


def load_scene(photo=None):
    """Returns the moving sprite (BGR image) placed on a dark canvas."""
    photo = Path(photo) if photo else find_sample_photo()
    if photo is not None and photo.exists():
        img = cv2.imread(str(photo))
        scale = FRAME_H / img.shape[0]
        sprite = cv2.resize(img, (int(img.shape[1] * scale), FRAME_H))
        if sprite.shape[1] > FRAME_W:  # wide photo: crop the middle so it can still move
            start = (sprite.shape[1] - int(FRAME_W * 0.7)) // 2
            sprite = sprite[:, start:start + int(FRAME_W * 0.7)]
        return sprite, "photo"
    sprite = np.zeros((FRAME_H, 200, 3), np.uint8)
    cv2.rectangle(sprite, (60, 120), (140, 400), (255, 255, 255), -1)
    return sprite, "synthetic"


class DemoCamera:
    def __init__(self, fps: float = 10.0, sweep_seconds: float = 8.0, photo=None, label: str = "DEMO CAM"):
        self.sprite, self.kind = load_scene(photo)
        self.fps = fps
        self.label = label
        self.sweep_frames = max(1, int(sweep_seconds * fps))
        self.travel = max(0, FRAME_W - self.sprite.shape[1])
        self.started = time.time()

    def render(self, index: int) -> np.ndarray:
        """One BGR frame of the moving scene."""
        canvas = np.full((FRAME_H, FRAME_W, 3), 40, np.uint8)
        phase = (index % (2 * self.sweep_frames)) / self.sweep_frames  # 0..2
        progress = phase if phase <= 1 else 2 - phase                   # ping-pong 0..1..0
        x = int(round(progress * self.travel))
        canvas[:, x:x + self.sprite.shape[1]] = self.sprite
        stamp = time.strftime("%H:%M:%S")
        cv2.putText(canvas, f"{self.label} {stamp}", (8, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def frame(self, index: int) -> bytes:
        ok, jpg = cv2.imencode(".jpg", self.render(index), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpg.tobytes()


def make_handler(camera: DemoCamera):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/video") or self.path.startswith("/mjpegfeed"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=dcmjpeg")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                index = 0
                period = 1.0 / camera.fps
                try:
                    while True:
                        jpg = camera.frame(index)
                        index += 1
                        self.wfile.write(b"--dcmjpeg\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                        self.wfile.flush()
                        time.sleep(period)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            elif self.path == "/":
                body = (b"<html><body style='background:#222;color:#ddd;font-family:sans-serif'>"
                        b"<h3>SeeSaw AI demo camera</h3><p>Stream: <code>/video</code></p>"
                        b"<img src='/video' width='640'></body></html>")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return Handler


class Server(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)

    camera = DemoCamera(fps=args.fps)
    server = Server((args.host, args.port), make_handler(camera))
    print(f"demo camera ({camera.kind}) at http://{args.host}:{args.port}/video  Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
