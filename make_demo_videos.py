"""Create short demo clips in samples/videos/ from the photos already on this PC.

  py make_demo_videos.py

The clips pan a photo with people across the frame, so the Video tab has something to
analyse (tracking, zones, alerts) even before you record real footage. Put your own clips in
samples/videos/ too; anything OpenCV can read (mp4, avi, mov, mkv) shows up on the page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

from demo_camera import FRAME_H, FRAME_W, DemoCamera, find_sample_photo

OUT_DIR = Path(__file__).resolve().parent / "samples" / "videos"
CLIPS = [
    ("demo_street_pan.avi", "bus.jpg", 20.0, "STREET"),
    ("demo_two_people.avi", "zidane.jpg", 15.0, "OFFICE"),
]


def write_clip(path: Path, photo, seconds: float, label: str, fps: float = 10.0) -> int:
    camera = DemoCamera(fps=fps, sweep_seconds=6.0, photo=photo, label=label)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {path}")
    frames = int(seconds * fps)
    for index in range(frames):
        writer.write(camera.render(index))
    writer.release()
    return frames


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = 0
    for name, photo_name, seconds, label in CLIPS:
        photo = find_sample_photo(photo_name)
        if photo is None:
            print(f"skip {name}: {photo_name} not found")
            continue
        frames = write_clip(OUT_DIR / name, photo, seconds, label)
        print(f"wrote {OUT_DIR / name} ({frames} frames, {seconds:.0f} s)")
        made += 1
    print(f"{made} clip(s) in {OUT_DIR}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
