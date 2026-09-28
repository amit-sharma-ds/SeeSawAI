"""Draw a zone, or the allowed walking direction, on a camera view and save it to cameras.json.

  py draw_zones.py <source> --camera cam1 --name gate --kind restricted
  py draw_zones.py <source> --camera cam1 --name machine --kind danger
  py draw_zones.py <source> --camera cam1 --kind direction

Left click adds a point, right click removes the last one, Enter saves, Esc cancels.
For --kind direction click two points: where people come from, then where they go.
Zones are stored with normalised 0..1 coordinates, so they work at any resolution.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from cctv_ai.rules import ZONE_COLORS, ZONE_KINDS, save_cameras_config
from cctv_ai.worker import FrameSource

KINDS = list(ZONE_KINDS) + ["direction"]


def grab_frame(source, width: int):
    src = FrameSource(source, realtime=False)
    if not src.open():
        raise SystemExit(f"cannot open source {source}")
    frame = None
    for _ in range(8):  # skip the first frames so exposure settles on live cameras
        nxt = src.read()
        if nxt is None:
            break
        frame = nxt
    src.release()
    if frame is None:
        raise SystemExit("no frame received from the source")
    if width and frame.shape[1] != width:
        scale = width / frame.shape[1]
        frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))))
    return frame


def collect_points(frame, kind: str, name: str):
    points = []
    window = f"draw {kind} '{name}': left click add, right click undo, Enter save, Esc cancel"
    color = ZONE_COLORS.get(kind, (255, 255, 255))

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if kind == "direction" and len(points) >= 2:
                points[:] = [points[0], (x, y)]
            else:
                points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = frame.copy()
        for p in points:
            cv2.circle(canvas, p, 4, color, -1)
        if kind == "direction" and len(points) == 2:
            cv2.arrowedLine(canvas, points[0], points[1], color, 2, tipLength=0.2)
        elif kind != "direction" and len(points) >= 2:
            outline = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [outline], len(points) >= 3, color, 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:  # Esc or window closed
            points = None
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return points


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="camera index, video file, IP Webcam URL or rtsp:// URL")
    parser.add_argument("--camera", required=True, help="camera id in the config file")
    parser.add_argument("--name", default=None, help="zone name (required for zones)")
    parser.add_argument("--kind", default="restricted", choices=KINDS)
    parser.add_argument("--config", default="cameras.json")
    parser.add_argument("--width", type=int, default=640, help="preview width (default 640)")
    args = parser.parse_args(argv)

    if args.kind != "direction" and not args.name:
        parser.error("--name is required for a zone")

    frame = grab_frame(args.source, args.width)
    h, w = frame.shape[:2]
    points = collect_points(frame, args.kind, args.name or "direction")
    if points is None:
        print("cancelled")
        return 1

    config_path = Path(args.config)
    cameras = []
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        cameras = data if isinstance(data, list) else data.get("cameras", [])
    entry = next((c for c in cameras if str(c.get("camera_id")) == args.camera), None)
    if entry is None:
        entry = {"camera_id": args.camera, "source": args.source, "every_n": 2, "width": 640, "rules": {}}
        cameras.append(entry)
    rules = entry.setdefault("rules", {})

    if args.kind == "direction":
        if len(points) != 2:
            print("direction needs exactly two clicks")
            return 1
        (x0, y0), (x1, y1) = points
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) or 1.0
        rules["direction"] = {
            "allowed": [round(dx / norm, 3), round(dy / norm, 3)],
            "min_travel": 0.15,
            "window_seconds": 3.0,
        }
        print("allowed direction saved:", rules["direction"]["allowed"])
    else:
        if len(points) < 3:
            print("a zone needs at least three points")
            return 1
        polygon = [[round(x / w, 4), round(y / h, 4)] for x, y in points]
        zones = rules.setdefault("zones", [])
        zones[:] = [z for z in zones if z.get("name") != args.name]
        zones.append({"name": args.name, "kind": args.kind, "polygon": polygon})
        print(f"zone '{args.name}' ({args.kind}) saved with {len(polygon)} points")

    save_cameras_config(config_path, cameras)
    print("written to", config_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
