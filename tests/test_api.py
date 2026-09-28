import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from cctv_ai import api, auth
from cctv_ai.detectors import YoloOnnxDetector
from tests.synthetic import GATE, write_walk_and_stand_video


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = str(write_walk_and_stand_video(Path(cls.tmp.name) / "clip.avi"))
        # run these tests with auth off and with the fast person detector, whatever keys and
        # models exist on this machine, so they stay quick and deterministic
        cls.patches = [
            mock.patch.object(auth, "KEY_FILE", Path(cls.tmp.name) / "no_keys.txt"),
            mock.patch.dict(os.environ, {"CCTV_API_KEYS": ""}),
            mock.patch.object(YoloOnnxDetector, "available", return_value=False),
        ]
        for p in cls.patches:
            p.start()
        auth._file_cache.update(token=None, keys={})
        api._image_detectors.clear()
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        for worker in list(api.workers.values()):
            worker.stop(join=True)
        api.workers.clear()
        for p in cls.patches:
            p.stop()
        auth._file_cache.update(token=None, keys={})
        api._image_detectors.clear()
        cls.tmp.cleanup()

    def wait_until_done(self, camera_id, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.client.get(f"/cameras/{camera_id}").json()
            if status["state"] in ("finished", "error", "stopped"):
                return status
            time.sleep(0.2)
        self.fail(f"camera {camera_id} did not finish in time")

    def test_health_lists_live_features(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("live_cameras", body["features"])

    def test_camera_lifecycle_on_a_video_file(self):
        response = self.client.post("/cameras", json={
            "camera_id": "t1", "source": self.video, "every_n": 2, "width": 0,
            "rules": {"zones": [GATE], "loiter_seconds": 0, "crowd_limit": 0},
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["camera_id"], "t1")
        self.assertEqual(response.json()["source"], self.video)

        listed = self.client.get("/cameras").json()["cameras"]
        self.assertEqual([c["camera_id"] for c in listed], ["t1"])

        status = self.wait_until_done("t1")
        self.assertEqual(status["state"], "finished", status["last_error"])
        self.assertGreater(status["frames"], 0)

        snapshot = self.client.get("/cameras/t1/snapshot")
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.headers["content-type"], "image/jpeg")
        self.assertTrue(snapshot.content.startswith(b"\xff\xd8"))

        self.assertEqual(self.client.get("/cameras/t1/alerts").json()["camera_id"], "t1")
        self.assertIn("alerts", self.client.get("/alerts").json())

        removed = self.client.delete("/cameras/t1")
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(self.client.get("/cameras/t1").status_code, 404)
        self.assertEqual(self.client.get("/cameras/t1/snapshot").status_code, 404)

    def test_web_page_and_samples(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("SeeSaw AI", page.text)

        body = self.client.get("/samples").json()
        listing = body["samples"]
        self.assertIsInstance(listing, list)
        self.assertIsInstance(body["categories"], list)
        for always in ("person", "gun", "knife", "smoke", "fire", "ppe"):
            self.assertIn(always, body["categories"])   # product categories are listed even while empty
        self.assertEqual(body["categories"][:3], ["person", "vehicle", "crowd"])
        for item in listing:
            self.assertTrue(item["categories"])
        if listing:
            first = self.client.get(listing[0]["url"])
            self.assertEqual(first.status_code, 200)
            self.assertTrue(first.headers["content-type"].startswith("image/"))
        coco8 = [s for s in listing if s["name"].startswith("coco8/")]
        if coco8:  # categories come from the YOLO labels next to the images
            self.assertIn("animal", body["categories"])
            zebra = next((s for s in coco8 if s["name"].endswith("000000000034.jpg")), None)
            if zebra:
                self.assertEqual(zebra["categories"], ["animal"])
                self.assertEqual(list(zebra["labels"]), ["zebra"])   # counts read from the label file
                self.assertGreaterEqual(zebra["labels"]["zebra"], 1)
                self.assertEqual(self.client.get(zebra["url"]).status_code, 200)
        self.assertEqual(self.client.get("/samples/does-not-exist.jpg").status_code, 404)
        self.assertEqual(self.client.get("/samples/coco8/nope.jpg").status_code, 404)

    def test_detect_image_endpoint(self):
        samples = {s["name"]: s["url"] for s in self.client.get("/samples").json()["samples"]}
        if "bus.jpg" in samples:
            image = self.client.get(samples["bus.jpg"]).content
        else:
            import cv2
            import numpy as np
            blank = np.zeros((240, 320, 3), np.uint8)
            image = cv2.imencode(".jpg", blank)[1].tobytes()
        response = self.client.post("/detect-image", files={"file": ("photo.jpg", image, "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body) >= {"width", "height", "time_ms", "detector", "detections"}, True)
        for det in body["detections"]:
            self.assertEqual(len(det["bbox"]), 4)
            self.assertLessEqual(det["bbox"][2], body["width"])
            self.assertLessEqual(det["bbox"][3], body["height"])
        if "bus.jpg" in samples:
            self.assertGreaterEqual(len(body["detections"]), 1)
            self.assertIn("person", [d["class_name"] for d in body["detections"]])

        bad = self.client.post("/detect-image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
        self.assertEqual(bad.status_code, 400)
        wrong = self.client.post("/detect-image", params={"detector": "magic"},
                                 files={"file": ("photo.jpg", image, "image/jpeg")})
        self.assertEqual(wrong.status_code, 422)
        fast = self.client.post("/detect-image", params={"detector": "hog"},
                                files={"file": ("photo.jpg", image, "image/jpeg")})
        self.assertEqual(fast.status_code, 200)
        self.assertEqual(fast.json()["detector"], "hog (person only)")

    def test_feature_catalog_and_switches(self):
        catalog = self.client.get("/features").json()["features"]
        self.assertEqual(len(catalog), 15)
        self.assertEqual(catalog[2]["key"], "restricted_area")
        self.assertEqual(catalog[14]["key"], "weapon")
        self.assertIn("default_on", catalog[0])

        response = self.client.post("/cameras", json={
            "camera_id": "sw", "source": self.video, "width": 0,
            "rules": {"zones": [GATE], "loiter_seconds": 0, "crowd_limit": 0},
            "features": ["person", "restricted_area"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["features"], ["person", "restricted_area"])

        changed = self.client.patch("/cameras/sw/features", json={"features": ["person", "crowd"]})
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(changed.json()["features"], ["crowd", "person"])
        self.assertEqual(self.client.patch("/cameras/sw/features", json={"features": ["fight"]}).status_code, 422)
        self.assertEqual(self.client.patch("/cameras/sw/features", json={"features": ["nope"]}).status_code, 422)
        self.wait_until_done("sw")
        self.client.delete("/cameras/sw")

        bad = self.client.post("/cameras", json={"camera_id": "bad_feature", "source": self.video, "features": ["nope"]})
        self.assertEqual(bad.status_code, 422)
        self.assertNotIn("bad_feature", api.workers)

    def test_detect_image_respects_feature_switches(self):
        samples = {s["name"]: s["url"] for s in self.client.get("/samples").json()["samples"]}
        if "bus.jpg" not in samples:
            self.skipTest("sample photo not available")
        image = self.client.get(samples["bus.jpg"]).content
        only_vehicles = self.client.post("/detect-image", params={"features": "vehicle"},
                                         files={"file": ("bus.jpg", image, "image/jpeg")})
        self.assertEqual(only_vehicles.status_code, 200, only_vehicles.text)
        self.assertTrue(all(d["category"] == "vehicle" for d in only_vehicles.json()["detections"]))
        self.assertEqual(only_vehicles.json()["features"], ["vehicle"])
        with_people = self.client.post("/detect-image", params={"features": "person,vehicle"},
                                       files={"file": ("bus.jpg", image, "image/jpeg")})
        self.assertGreaterEqual(len(with_people.json()["detections"]), 1)
        invalid = self.client.post("/detect-image", params={"features": "nope"},
                                   files={"file": ("bus.jpg", image, "image/jpeg")})
        self.assertEqual(invalid.status_code, 422)

    def test_video_upload_job_lifecycle(self):
        with mock.patch.object(api, "UPLOAD_DIR", Path(self.tmp.name) / "uploads"):
            with open(self.video, "rb") as fh:
                response = self.client.post(
                    "/videos", params={"pace": "false", "every_n": 2, "width": 0, "features": "person"},
                    files={"file": ("clip.avi", fh, "video/x-msvideo")},
                )
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()
            job_id = job["job_id"]
            self.assertTrue(job_id.startswith("video-"))
            self.assertEqual(job["kind"], "video")
            self.assertIn(job_id, [v["camera_id"] for v in self.client.get("/videos").json()["videos"]])

            status = self.wait_until_done(job_id)
            self.assertEqual(status["state"], "finished", status["last_error"])
            self.assertEqual(status["total_frames"], 120)
            self.assertEqual(status["progress"], 1.0)

            summary = self.client.get(f"/cameras/{job_id}/summary").json()
            self.assertEqual(summary["frames"], 120)
            self.assertEqual(summary["duration_seconds"], 12.0)

            download = self.client.get(f"/videos/{job_id}/download")
            self.assertEqual(download.status_code, 200)
            self.assertEqual(download.headers["content-type"], "video/x-msvideo")
            self.assertGreater(len(download.content), 10000)

            uploads = Path(self.tmp.name) / "uploads"
            self.assertEqual(len(list(uploads.glob(f"{job_id}*"))), 2)
            self.assertEqual(self.client.delete(f"/cameras/{job_id}").status_code, 200)
            self.assertEqual(list(uploads.glob(f"{job_id}*")), [])

            bad = self.client.post("/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
            self.assertEqual(bad.status_code, 415)
            missing = self.client.post("/videos", params={"rules_from": "no-such-camera"},
                                       files={"file": ("clip.avi", b"x", "video/x-msvideo")})
            self.assertEqual(missing.status_code, 404)

    def test_sample_videos_can_be_analysed_in_place(self):
        body = self.client.get("/sample-videos").json()
        listing = body["videos"]
        self.assertIsInstance(listing, list)
        self.assertIn("gun", body["categories"])
        self.assertIn("smoke", body["categories"])
        if not listing:
            self.skipTest("no sample clips in samples/videos (run make_demo_videos.py)")
        for item in listing:
            self.assertTrue(item["categories"])
        clip = min(listing, key=lambda v: v.get("seconds") or 1e9)
        thumb = self.client.get(clip["thumb"])
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.headers["content-type"], "image/jpeg")
        self.assertEqual(self.client.get("/sample-videos/nope.avi/thumb").status_code, 404)

        source = api.SAMPLE_VIDEO_DIR / clip["name"]
        with mock.patch.object(api, "UPLOAD_DIR", Path(self.tmp.name) / "uploads"):
            started = self.client.post(f"/sample-videos/{clip['name']}/analyze",
                                       params={"pace": "false", "every_n": 10, "width": 320, "detector": "hog",
                                               "features": "person"})
            self.assertEqual(started.status_code, 200, started.text)
            job_id = started.json()["job_id"]
            self.assertEqual(started.json()["original_name"], clip["name"])
            status = self.wait_until_done(job_id, timeout=120)
            self.assertEqual(status["state"], "finished", status["last_error"])
            self.assertGreater(status["frames"], 0)
            self.assertEqual(self.client.delete(f"/cameras/{job_id}").status_code, 200)
        self.assertTrue(source.exists())  # deleting the job never deletes the sample clip

    def test_cameras_config_listing(self):
        body = self.client.get("/cameras-config").json()
        by_id = {c["camera_id"]: c for c in body["cameras"]}
        self.assertIn("cam1", by_id)
        self.assertIn("gate", by_id["cam1"]["zones"])
        self.assertIn("taped_off_area", by_id["vtest_demo"]["zones"])

    def test_invalid_rules_are_rejected(self):
        response = self.client.post("/cameras", json={
            "camera_id": "bad_rules", "source": self.video,
            "rules": {"zones": [{"name": "z", "kind": "nope", "polygon": [[0, 0], [1, 0], [1, 1]]}]},
        })
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("bad_rules", api.workers)

    def test_missing_source_ends_in_error_state(self):
        response = self.client.post("/cameras", json={"camera_id": "missing", "source": str(Path(self.tmp.name) / "nope.mp4")})
        self.assertEqual(response.status_code, 200)
        status = self.wait_until_done("missing")
        self.assertEqual(status["state"], "error")
        self.client.delete("/cameras/missing")

    def test_load_config_file_not_found(self):
        response = self.client.post("/cameras/load", params={"path": str(Path(self.tmp.name) / "nope.json")})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
