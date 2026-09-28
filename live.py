"""Run the real-time pipeline on one source and watch it in a window.

Examples (run from the project folder):
  py live.py clips/walk_01.mp4
  py live.py 0
    py live.py 0 --camera gate  # DroidCam virtual webcam index
  py live.py "rtsp://admin:pass@192.168.1.10:554/cam/realmonitor?channel=1&subtype=1" --camera cam1

Zones and limits come from cameras.json (draw them with draw_zones.py); --camera picks
the entry by id. Press q in the window to quit. Alerts print to the console and are
appended to alerts.jsonl.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from cctv_ai.detectors import build_detectors
from cctv_ai.rules import AlertManager, CameraRules, load_cameras_config
from cctv_ai.worker import CameraWorker, mask_credentials


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="camera index, video file, IP Webcam URL or rtsp:// URL (default: from config)")
    parser.add_argument("--camera", default="cam1", help="camera id in the config file (default: cam1)")
    parser.add_argument("--config", default="cameras.json", help="cameras config file (default: cameras.json)")
    parser.add_argument("--every", type=int, default=None, help="run detection on every Nth frame (default: config or 2)")
    parser.add_argument("--width", type=int, default=None, help="resize frames to this width (default: config or 640)")
    parser.add_argument("--faces", action="store_true", help="also run the face detector (downloads its model once)")
    parser.add_argument("--detector", default=None, choices=["auto", "hog", "yolo"],
                        help="auto = object model if installed; hog = fast, people only; yolo = object model")
    parser.add_argument("--record", help="save the annotated video to this .avi file")
    parser.add_argument("--alerts-log", default="alerts.jsonl", help="append alerts as JSON lines to this file")
    parser.add_argument("--no-alerts-log", action="store_true", help="do not write the alerts log file")
    parser.add_argument("--headless", action="store_true", help="no window, console output only")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = until the end or q)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    entry = {}
    try:
        cameras = load_cameras_config(args.config)
        entry = next((c for c in cameras if str(c["camera_id"]) == args.camera), {})
    except FileNotFoundError:
        pass

    source = args.source if args.source is not None else entry.get("source")
    if source is None:
        print(f"No source given and no camera '{args.camera}' in {args.config}")
        return 2

    rules = CameraRules.from_dict(entry.get("rules"), args.camera)
    every_n = args.every or int(entry.get("every_n", 2))
    width = args.width if args.width is not None else int(entry.get("width", 640))
    prefer = args.detector or str(entry.get("detector", "auto"))
    detectors = build_detectors(faces=args.faces, live=True, prefer=prefer)
    print("detector:", ", ".join(type(d).__name__ for d in detectors), flush=True)

    log_path = None if args.no_alerts_log else args.alerts_log
    alerts = AlertManager(cooldown_seconds=rules.cooldown_seconds, log_path=log_path)
    alerts.subscribe(lambda a: print(f"ALERT {a.as_dict()['time']} [{a.severity}] {a.type}: {a.message}", flush=True))

    window = f"SeeSaw AI - {args.camera}"
    writer = None

    def on_frame(frame, worker):
        nonlocal writer
        if args.record:
            if writer is None:
                fps = worker.source_fps if 1 <= worker.source_fps <= 60 else 10.0
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"MJPG"), fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if args.headless:
            if worker.frame_no % 50 == 0:
                print(f"frame {worker.frame_no}  {worker.fps:.1f} fps  detect {worker.detect_ms:.0f} ms  "
                      f"tracks {len(worker.latest_tracks)}", flush=True)
            return
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            worker.stop(join=False)

    worker = CameraWorker(
        args.camera, source, detectors,
        rules=rules, alert_manager=alerts, every_n=every_n, width=width,
        on_frame=on_frame, max_frames=args.max_frames or None,
    )
    print(f"source: {mask_credentials(source)}  detection every {every_n} frame(s) at {width or 'native'} px wide. "
          f"{len(rules.zones)} zone(s). Press q to quit.", flush=True)
    try:
        worker.run()
    except KeyboardInterrupt:
        worker.stop(join=False)
    finally:
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    print(f"done: {worker.frame_no} frames, {worker.detections_run} detection runs, "
          f"{worker.fps:.1f} fps at the end, {worker.alert_count} alerts, state {worker.state}")
    if worker.last_error:
        print("last error:", worker.last_error)
    return 0 if worker.state != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
