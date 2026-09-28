"""Download test data for the web page.

  py download_samples.py            # 128 labelled COCO photos (coco128, about 7 MB) into datasets/coco128
  py download_samples.py --videos   # OpenCV's pedestrian test video vtest.avi (about 8 MB) into samples/videos

The web page reads the photo labels and files every photo under its categories (person,
vehicle, animal, bag & accessory, food, electronics, ...), so every category has test images.
COCO photos come from Flickr under Creative Commons licences: fine for testing, check the
licence before using one in marketing material. vtest.avi is the street scene shipped with
OpenCV's own samples: people walking in several directions, ideal for tracking, loitering,
crowd and wrong-direction tests.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASETS = PROJECT_DIR / "datasets"
DEST = DATASETS / "coco128"
URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
    "https://ultralytics.com/assets/coco128.zip",
]
VIDEO_DIR = PROJECT_DIR / "samples" / "videos"
VIDEO_URLS = {
    "vtest_pedestrians.avi": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
}


def fetch(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "seesaw-ai/1.0"})
    with urllib.request.urlopen(request) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def download_videos() -> int:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    got = 0
    for name, url in VIDEO_URLS.items():
        target = VIDEO_DIR / name
        if target.exists():
            print(f"already present: {target}")
            got += 1
            continue
        print(f"downloading {url}\n  -> {target}")
        tmp = target.with_suffix(".part")
        try:
            fetch(url, tmp)
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        tmp.replace(target)
        print(f"  ok ({target.stat().st_size / 1e6:.1f} MB)")
        got += 1
    return 0 if got else 1


def main() -> int:
    if "--videos" in sys.argv[1:]:
        return download_videos()
    if (DEST / "images").is_dir():
        count = sum(1 for _ in (DEST / "images").rglob("*.jpg"))
        print(f"already present: {DEST} ({count} images)")
        return 0
    DATASETS.mkdir(exist_ok=True)
    archive = DATASETS / "coco128.zip"
    last_error = None
    for url in URLS:
        try:
            print(f"downloading {url}")
            fetch(url, archive)
            break
        except Exception as exc:  # try the next mirror
            last_error = exc
            print(f"  failed: {exc}")
    else:
        print("could not download coco128:", last_error)
        return 1
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATASETS)
    archive.unlink(missing_ok=True)
    count = sum(1 for _ in (DEST / "images").rglob("*.jpg")) if (DEST / "images").is_dir() else 0
    print(f"ok: {count} images in {DEST}. Reload the web page to see them.")
    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
