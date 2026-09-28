"""Download the ready-made 80-class object detector from OpenCV's official model zoo.

  py download_models.py

Fetches YOLOX (about 36 MB) from github.com/opencv/opencv_zoo into models/. The download
resumes by itself if the connection drops. After it, the API, live.py and the web page
recognise 80 COCO object types instead of people only. If you prefer YOLOv8n, export it on
Colab instead (see README) and skip this script.
"""
from __future__ import annotations

import http.client
import sys
import time
import urllib.request
from pathlib import Path

from cctv_ai.config import MODEL_DIR

URL = "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
DEST = MODEL_DIR / "object_detection_yolox_2022nov.onnx"
MIN_SIZE = 1_000_000


def download(url: str, dest: Path, attempts: int = 30) -> None:
    """Fetch url into dest, resuming a partial .part file with HTTP range requests."""
    tmp = dest.with_suffix(".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    total = None
    for attempt in range(1, attempts + 1):
        headers = {"User-Agent": "seesaw-ai/1.0"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if done and response.status != 206:  # server ignored the range: start over
                    done = 0
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if "/" in content_range and content_range.rsplit("/", 1)[-1].isdigit():
                        total = int(content_range.rsplit("/", 1)[-1])
                elif response.headers.get("Content-Length", "").isdigit():
                    total = int(response.headers["Content-Length"])
                print(f"  connected (attempt {attempt}), resuming at {done / 1e6:.1f} MB", flush=True)
                with tmp.open("ab" if done else "wb") as out:
                    last_report = time.monotonic()
                    while True:
                        chunk = response.read(64 * 1024)  # small chunks: visible progress on slow lines
                        if not chunk:
                            break
                        out.write(chunk)
                        out.flush()
                        done += len(chunk)
                        if total and time.monotonic() - last_report >= 5:
                            last_report = time.monotonic()
                            print(f"  {done / 1e6:.1f} / {total / 1e6:.1f} MB", flush=True)
        except (OSError, http.client.HTTPException) as exc:
            print(f"\n  connection problem ({exc}); resuming from {done / 1e6:.1f} MB (attempt {attempt}/{attempts})")
            time.sleep(3)
            continue
        if total is None or done >= total:
            break
        print(f"\n  stream ended early at {done / 1e6:.1f} MB; resuming (attempt {attempt}/{attempts})")
        time.sleep(2)
    print()
    if total is not None and done < total:
        raise RuntimeError(f"incomplete download: {done} of {total} bytes; run the script again to resume")
    head = tmp.read_bytes()[:64]
    if tmp.stat().st_size < MIN_SIZE or head.startswith(b"version https://git-lfs"):
        tmp.unlink()
        raise RuntimeError("the server returned a placeholder instead of the model; download it in a browser "
                           f"from {url} and save it as {dest}")
    tmp.replace(dest)


def loads(path: Path) -> bool:
    import cv2
    try:
        cv2.dnn.readNetFromONNX(str(path))
        return True
    except cv2.error:
        return False


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        if loads(DEST):
            print(f"already present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
            return 0
        print(f"{DEST.name} is damaged or incomplete, keeping its bytes to resume")
        DEST.replace(DEST.with_suffix(".part"))
    print(f"downloading {URL}\n  -> {DEST}")
    download(URL, DEST)
    if not loads(DEST):
        DEST.unlink()
        raise RuntimeError("the downloaded file could not be parsed by OpenCV; run the script again")
    print(f"ok: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB). Restart the server to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
