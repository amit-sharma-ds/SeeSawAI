# Sample images and videos

Everything in this folder shows up on the web page so visitors can test without uploading.

## Images: `samples/<category>/*.jpg|png`

The folder name is the category shown on the strip. These folders exist and are waiting
for your test photos:

| folder | put here |
|---|---|
| `samples/gun/` | photos with a pistol or rifle visible (toy guns work for demos) |
| `samples/knife/` | photos with a knife visible |
| `samples/smoke/` | smoke, incense, a smoking pan |
| `samples/fire/` | flames, a candle, a lighter |
| `samples/ppe/` | workers with and without helmets and vests |

Photos straight in `samples/` land under "my samples". The page also lists every YOLO
dataset under `datasets/` (categories read from the labels) and the two photos shipped
with the ultralytics package. `py download_samples.py` adds 128 labelled COCO photos.

## Videos: `samples/videos/<category>/*.mp4|avi|mov|mkv`

Same idea for clips. `samples/videos/gun/`, `knife/`, `smoke/`, `fire/` and `ppe/` exist and
are empty. `py make_demo_videos.py` creates two demo clips, `py download_samples.py --videos`
fetches OpenCV's pedestrian street scene. Clips are analysed in place; deleting a finished
analysis never deletes the clip.

Note: gun, smoke, fire and PPE are only *detected* once their models exist in `models/`
(see the README section "Train your own detector"). Knives are detected already.
