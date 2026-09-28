# SeeSaw AI

Real-time CCTV monitoring on top of OpenCV. One pipeline handles a video file, a phone camera
(DroidCam / IP Webcam) or an RTSP CCTV camera:

```
frame source -> detectors -> tracker (stable ids) -> rules (zones, timers, direction) -> alerts
```

No PyTorch is needed for the parts that work today, so it runs on a plain CPU.

## What works now

| Feature | How | Status |
|---|---|---|
| Person detection | OpenCV HOG people detector (`cctv_ai/detectors/person.py`) | working |
| Face detection | OpenCV DNN ResNet-SSD (`cctv_ai/detectors/face.py`), downloads its model once | working, optional (`--faces`) |
| Tracking | IoU + distance tracker with per-track history (`cctv_ai/tracking.py`) | working |
| Restricted area | feet inside a `restricted` zone | working |
| Industrial safety | feet inside a `danger` zone (machines) | working |
| Loitering | dwell time in a `loiter` zone, or standing within a small radius | working |
| Crowd | people count in `crowd` zones (or whole frame) over a limit | working |
| Wrong direction | movement against the allowed direction vector | working |
| Vehicle, object left/taken, material movement | need a COCO object detector (YOLO ONNX) | next |
| PPE, fire/smoke | need custom-trained models (Colab) | later |
| Fight, human activity, face-based access | need specialised models | later |

## Setup

```bash
pip install -r requirements.txt
```

Run the tests (synthetic video, no camera needed):

```bash
py -m unittest -v
```

## Run live

```bash
py live.py clips/walk_01.mp4
py live.py 0 --camera cam1
py live.py "rtsp://admin:pass@192.168.1.10:554/cam/realmonitor?channel=1&subtype=1" --camera cam1
```

Press `q` to quit. Alerts print to the console and append to `alerts.jsonl`.
Useful flags: `--every 3` (detect on every 3rd frame), `--width 480` (smaller = faster),
`--record out.avi`, `--headless`, `--faces`.

Camera sources:

- DroidCam Webcam: connect the phone with the Windows DroidCam client, then use the
  virtual camera index (often `0` or `1`); `4747/video` is not a video URL for DroidCam Remote.
- IP Webcam app: `http://<phone-ip>:8080/video`
- Hikvision: `rtsp://user:pass@<ip>:554/Streaming/Channels/102` (102 = sub stream, lighter)
- CP Plus / Dahua: `rtsp://user:pass@<ip>:554/cam/realmonitor?channel=1&subtype=1`
- Generic ONVIF NVR: check the camera's web page for its RTSP path

Use the sub stream (lower resolution) of a CCTV camera. RTSP is opened over TCP and
reconnects automatically if the stream drops.

## Define zones and direction

Zones live in `cameras.json` as normalised 0..1 polygons, so they survive a resolution
change. Draw them on the real camera view instead of typing numbers:

```bash
py draw_zones.py 0 --camera cam1 --name gate --kind restricted
py draw_zones.py 0 --camera cam1 --name press --kind danger
py draw_zones.py 0 --camera cam1 --kind direction
```

Left click adds points, right click undoes, Enter saves. Zone kinds: `restricted`,
`danger`, `loiter`, `crowd`. Rule limits per camera: `loiter_seconds`, `loiter_radius`,
`crowd_limit`, `crowd_min_hits`, `cooldown_seconds`, `direction`.

## API

```bash
py run_api.py
```

- `POST /cameras` with `{"camera_id": "cam1", "source": "rtsp://...", "rules": {...}}` starts a camera
- `POST /cameras/load?path=cameras.json` starts every camera in the config
- `GET /cameras`, `GET /cameras/{id}` status, tracks, fps
- `GET /cameras/{id}/stream` MJPEG stream of the annotated video (put it in an `<img>` tag)
- `GET /cameras/{id}/snapshot` one JPEG
- `GET /alerts`, `GET /cameras/{id}/alerts` recent alerts, newest first
- `DELETE /cameras/{id}` stops a camera
- `POST /detect`, `/detect-people`, `/detect-faces`, `/alerts` single uploaded image (older endpoints)

## Web page

`http://localhost:8000/` is a small page served by the API:

- **Image test**: drop an image or click a sample, boxes are drawn on it, hover a box or a
  list entry to see the object and its confidence, slide the confidence filter.
- **Live cameras**: pick a running camera, watch the annotated stream and the alert feed,
  or start a camera by typing its source.

Images you put in `samples/` appear on the sample strip, next to the coco8 images and the
two sample photos from the ultralytics package.

## Feature switches

All 14 product features are listed by `GET /features` with their build status. Switch on
only the ones you need; everything else stays silent:

- on the web page: tick boxes in the Features panel (applies to image tests and to the
  selected live camera)
- in `cameras.json`: `"features": ["person", "restricted_area", "crowd"]`
- when starting a camera: `POST /cameras` with `"features": [...]`
- while a camera runs: `PATCH /cameras/{id}/features` with `{"features": [...]}`

Keys: `person`, `vehicle`, `restricted_area`, `loitering`, `fight`, `fire_smoke`, `ppe`,
`wrong_direction`, `object_left_taken`, `crowd`, `industrial_safety`, `face_access`,
`human_activity`, `material_movement`, `weapon`. Features marked "planned" cannot be
switched on yet; features marked "needs model" work as soon as their model file exists.

## Analyse an uploaded video

The **Video** tab on the web page (or `POST /videos`) treats a clip like a camera: it plays
through the pipeline with the chosen features and optional zones from `cameras.json`, shows
the annotated stream while it runs, lists the alerts, gives a summary (people seen, alerts
by type) and an annotated `.avi` to download. Uploads live in `uploads/` and are deleted
when the job is removed.

## 80-class object detection (YOLO ONNX)

Out of the box the detector finds people only. To recognise 80 COCO object types (car,
truck, bus, motorcycle, backpack, handbag, laptop, cell phone, ...) you need one model
file in `models/`. Either download the ready-made one from OpenCV's model zoo:

```bash
py download_models.py
```

or export YOLOv8n on Google Colab (or any machine where ultralytics works) and copy
`yolov8n.onnx` into `models/`:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640
```

The API, `live.py` and the web page switch to it automatically. Live cameras track only
people, vehicles and bags from it; the image page shows every class. YOLOX is the more
accurate of the two but slower on a small CPU (about a second per frame). Choose per use
with the `detector` option (`auto`, `hog` = fast people-only, `yolo` = object model): the
select box in the page header, `--detector` on `live.py`, `"detector"` in `cameras.json`,
or the `detector` field/parameter on `POST /cameras`, `POST /videos` and `POST /detect-image`.
On a weak PC run live cameras with `hog` and `every_n` 2, and use `yolo` for images and
uploaded videos.

## Train your own detector (guns, fire and smoke, PPE)

Knives are already covered: "knife" is one of the 80 classes of the object model, and the
Gun / Knife feature raises a critical alert for every knife seen (on live cameras the object
model tracks knives alongside people, vehicles and bags). Guns, fire and PPE are not among
the 80 classes, so each needs a model trained on its own dataset. The notebook
`colab/train_custom_detector.ipynb` does the whole job on Google Colab's free GPU: pick a
dataset on Roboflow Universe, run the cells, download two files.
The plumbing is ready: any `models/<name>.onnx` with a `models/<name>.names` file (one
class per line) is loaded automatically, its detections get the category `<name>`, and the
matching feature switches on. Recognised names and what they alert on:

| name | alert | dataset to look for |
|---|---|---|
| `weapon` | every detection, critical | "gun detection", "weapon detection" on Roboflow Universe |
| `fire_smoke` | every detection, critical | "fire and smoke detection" |
| `ppe` | classes named `no-helmet`, `NO-Hardhat`, `without_vest` and similar | "hard hat", "PPE detection" |

Recipe on Google Colab (free GPU), once per model:

```bash
pip install ultralytics roboflow
# download the dataset in "YOLOv8" format from Roboflow Universe, it comes with data.yaml
yolo train model=yolov8n.pt data=/content/<dataset>/data.yaml epochs=50 imgsz=640
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12 imgsz=640
```

Copy `best.onnx` to `models/weapon.onnx` (or `fire_smoke.onnx`, `ppe.onnx`) and write the
class names from `data.yaml`, in order, into `models/weapon.names`. Restart the server.

## Sample images by category

The sample strip groups images by what they contain. Sources, in order:

- `samples/<category>/*.jpg`: your own test images, the folder name is the category
- any YOLO dataset under `datasets/` (`images/` next to `labels/`): categories come from the labels
- the two photos shipped with the ultralytics package

`py download_samples.py` adds coco128, 128 labelled COCO photos covering people, vehicles,
animals, bags, food, electronics and more (about 7 MB).

## Sample videos

The Video tab also has a strip of clips that are analysed in place, no upload needed. Sources:

- `samples/videos/*.mp4|avi|mov|mkv`: your own clips
- `py make_demo_videos.py`: two demo clips built from the photos on this PC (people panning
  across a street scene and an office scene)
- `py download_samples.py --videos`: OpenCV's classic pedestrian test video (about 8 MB),
  people walking in several directions, ideal for tracking, loitering, crowd and
  wrong-direction tests

## Share the API with developers

Create one key per developer or app. Keys live in `api_keys.txt` (ignored by git) and the
running server picks them up at once; delete a line to revoke a key.

```bash
py make_api_key.py --label frontend-team
```

While no key exists the API is open. Once any key exists, every endpoint except `/health`
and `/docs` needs the key in the `X-API-Key` header, or as `?api_key=...` on the stream and
snapshot URLs used inside `<img>` tags.

Start the server so other machines on the network can reach it, then share
`http://<your-pc-ip>:8000/docs`, which has an Authorize button and live examples:

```bash
py -m uvicorn cctv_ai.api:app --host 0.0.0.0 --port 8000
```

Example calls:

```bash
curl -H "X-API-Key: THE_KEY" http://192.168.1.20:8000/alerts
```

```html
<img src="http://192.168.1.20:8000/cameras/cam1/stream?api_key=THE_KEY">
```

Notes: Windows Firewall must allow inbound TCP 8000 (it asks on the first start). For
developers outside your network use a tunnel such as ngrok or cloudflared, or deploy the
same command on a Linux server. Web apps on other origins work because CORS is enabled;
restrict it with `CCTV_CORS_ORIGINS="https://app.example.com"`. Keys can also be given by
environment variable: `CCTV_API_KEYS="key1,key2"`.

## How to test before the real camera

1. Record short phone clips, one per scenario: normal walking, standing still in a corner,
   walking into a taped-off area, four people together, walking the wrong way.
2. Run `py live.py clips/<name>.mp4` and check boxes and ids visually.
3. Draw the zones for that view, run again, and check that the expected alert fires and
   the normal clip stays silent. Tune limits in `cameras.json`.
4. Then point `live.py` at the DroidCam virtual webcam index, then at the RTSP camera.

## Performance on a small CPU

The HOG detector takes roughly 60 ms at 480 px and 110 ms at 640 px per run on a
2-core desktop CPU. With `every_n = 2` the loop keeps up with a 10 to 15 fps stream.
Live sources drop old frames automatically, so detection lag never grows.

## Project layout

- `cctv_ai/detectors/` detector classes (`BaseDetector` + `Detection`)
- `cctv_ai/tracking.py` tracker
- `cctv_ai/rules.py` zones, rules, alerts, config loading
- `cctv_ai/worker.py` frame source and per-camera worker
- `cctv_ai/api.py` FastAPI app
- `live.py`, `draw_zones.py` command-line tools
- `tests/` unit and end-to-end tests on a synthetic clip
- `cctv_ai/detector.py` older single-image detector kept for the upload endpoints

## Next steps

1. Export `yolov8n.onnx` on Colab (see above). That unlocks vehicles, bags and the
   object-left and material-movement rules.
2. Train PPE and fire/smoke models on Colab with `train_yolo.py`, export to ONNX the same way.
3. Fight, activity and face-based access after the first ten features are stable.
