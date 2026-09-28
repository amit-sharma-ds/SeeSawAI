import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cctv_ai.rules import AlertManager, CameraRules
from cctv_ai.tracking import Track
from cctv_ai.worker import CameraWorker, FrameSource, is_live_source, mask_credentials
from tests.synthetic import GATE, TOTAL_FRAMES, BlobDetector, write_walk_and_stand_video


class PipelineOnVideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = str(write_walk_and_stand_video(Path(cls.tmp.name) / "walk_and_stand.avi"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_worker(self, rules, **kwargs):
        alerts = AlertManager(cooldown_seconds=rules.cooldown_seconds)
        worker = CameraWorker("test", self.video, [BlobDetector()], rules=rules, alert_manager=alerts,
                              every_n=2, width=0, **kwargs)
        worker.run()
        return worker, alerts

    def test_restricted_and_loitering_alerts_fire_at_the_right_time(self):
        rules = CameraRules.from_dict({
            "zones": [GATE, {**GATE, "name": "gate_wait", "kind": "loiter"}],
            "loiter_seconds": 5, "crowd_limit": 5, "cooldown_seconds": 30,
        }, "test")
        worker, alerts = self.run_worker(rules)

        self.assertEqual(worker.state, "finished", worker.last_error)
        self.assertEqual(worker.frame_no, TOTAL_FRAMES)
        self.assertEqual(worker.detections_run, TOTAL_FRAMES // 2)

        by_type = {}
        for alert in alerts.alerts:
            by_type.setdefault(alert.type, []).append(alert)
        self.assertEqual(sorted(by_type), ["loitering", "restricted_area"])
        self.assertEqual(len(by_type["restricted_area"]), 1)
        self.assertEqual(len(by_type["loitering"]), 1)

        entered = by_type["restricted_area"][0]
        self.assertAlmostEqual(entered.timestamp, 4.1, delta=0.35)  # walks into the zone at 4 s
        self.assertEqual(entered.zone, "gate")
        self.assertEqual(entered.track_id, 1)
        loiter = by_type["loitering"][0]
        self.assertAlmostEqual(loiter.timestamp, entered.timestamp + 5.0, delta=0.35)
        self.assertEqual(loiter.zone, "gate_wait")
        self.assertEqual(worker.alert_count, 2)
        self.assertEqual(worker.status()["last_alert"]["type"], "loitering")

    def test_default_tracked_boxes_are_not_green(self):
        worker = CameraWorker(
            "test",
            self.video,
            [BlobDetector()],
            rules=CameraRules(camera_id="test", loiter_seconds=0, crowd_limit=0),
            every_n=1,
            width=0,
        )
        worker.latest_tracks = [Track(1, [10, 20, 80, 100], "person", 0.9, 0.0, 0.0, category="person")]
        worker._flagged.clear()

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        annotated = worker._annotate(frame, 0.0)

        self.assertTupleEqual(tuple(annotated[21, 11].tolist()), (200, 200, 200))
        self.assertNotEqual(tuple(annotated[21, 11].tolist()), (0, 255, 0))

    def test_feature_switches_control_detection_and_alerts(self):
        base = {"zones": [GATE], "loiter_seconds": 0, "crowd_limit": 0}
        # person detection switched off: nothing is tracked, so the zone rule cannot fire
        worker, alerts = self.run_worker(CameraRules.from_dict({**base, "features": ["restricted_area"]}, "test"))
        self.assertEqual(list(alerts.alerts), [])
        self.assertEqual(worker.latest_tracks, [])
        # person on but the zone rule off: tracked, no alert
        worker, alerts = self.run_worker(CameraRules.from_dict({**base, "features": ["person"]}, "test"))
        self.assertEqual(list(alerts.alerts), [])
        self.assertEqual(len(worker.latest_tracks), 1)
        # both on: alert
        worker, alerts = self.run_worker(CameraRules.from_dict({**base, "features": ["person", "restricted_area"]}, "test"))
        self.assertEqual([a.type for a in alerts.alerts], ["restricted_area"])
        self.assertEqual(worker.status()["features"], ["person", "restricted_area"])

    def test_pacing_recording_progress_and_summary(self):
        import cv2
        out = Path(self.tmp.name) / "annotated.avi"
        rules = CameraRules.from_dict({"zones": [GATE], "loiter_seconds": 0, "crowd_limit": 0}, "test")
        alerts = AlertManager()
        worker = CameraWorker("test", self.video, [BlobDetector()], rules=rules, alert_manager=alerts,
                              every_n=2, width=0, pace=True, record_path=out, max_frames=15)
        t0 = time.perf_counter()
        worker.run()
        elapsed = time.perf_counter() - t0
        self.assertGreaterEqual(elapsed, 1.2)   # 15 frames at 10 fps take about 1.4 s when paced
        self.assertLess(elapsed, 6.0)
        self.assertEqual(worker.frame_no, 15)
        self.assertEqual(worker.total_frames, TOTAL_FRAMES)
        self.assertEqual(worker.status()["progress"], 0.125)
        self.assertEqual(worker.status()["kind"], "video")

        cap = cv2.VideoCapture(str(out))
        written = 0
        while cap.read()[0]:
            written += 1
        cap.release()
        self.assertEqual(written, 15)

        summary = worker.summary()
        self.assertEqual(summary["frames"], 15)
        self.assertEqual(summary["objects_tracked"], 1)
        self.assertEqual(summary["people_max"], 1)
        self.assertEqual(summary["seen_by_category"], {"person": 1})
        self.assertEqual(worker.status()["counts"], {"person": 1})
        self.assertEqual(worker.status()["objects"], {})
        self.assertEqual(worker.status()["seen"], {"person": 1})
        self.assertEqual(worker.status()["people"], 1)
        self.assertEqual(summary["duration_seconds"], 1.5)
        self.assertEqual(summary["alerts_by_type"], {})

    def test_single_walker_keeps_one_track_id(self):
        worker, _ = self.run_worker(CameraRules(camera_id="test", loiter_seconds=0, crowd_limit=0))
        self.assertEqual(sorted(worker.tracker.tracks), [1])
        self.assertEqual([t.track_id for t in worker.latest_tracks], [1])
        self.assertGreaterEqual(worker.latest_tracks[0].hits, 55)

    def test_wrong_direction_fires_only_against_allowed_direction(self):
        against = CameraRules.from_dict({"direction": {"allowed": [-1, 0], "min_travel": 0.1, "window_seconds": 2},
                                         "loiter_seconds": 0, "crowd_limit": 0}, "test")
        _, alerts = self.run_worker(against)
        self.assertEqual([a.type for a in alerts.alerts], ["wrong_direction"])
        self.assertLess(alerts.alerts[0].timestamp, 4.5)  # during the walk, before standing still

        along = CameraRules.from_dict({"direction": {"allowed": [1, 0], "min_travel": 0.1, "window_seconds": 2},
                                       "loiter_seconds": 0, "crowd_limit": 0}, "test")
        _, alerts = self.run_worker(along)
        self.assertEqual(list(alerts.alerts), [])

    def test_resize_and_zone_scale_together(self):
        rules = CameraRules.from_dict({"zones": [GATE], "loiter_seconds": 0, "crowd_limit": 0}, "test")
        alerts = AlertManager()
        worker = CameraWorker("test", self.video, [BlobDetector()], rules=rules, alert_manager=alerts,
                              every_n=1, width=160)
        worker.run()
        self.assertEqual(worker.state, "finished", worker.last_error)
        self.assertEqual(worker.latest_frame().shape[1], 160)
        self.assertEqual([a.type for a in alerts.alerts], ["restricted_area"])
        self.assertAlmostEqual(alerts.alerts[0].timestamp, 4.1, delta=0.35)

    def test_realtime_reader_thread_finishes_on_file_end(self):
        worker, _ = self.run_worker(CameraRules(camera_id="test", loiter_seconds=0, crowd_limit=0), realtime=True)
        self.assertEqual(worker.state, "finished", worker.last_error)
        self.assertGreaterEqual(worker.frame_no, 1)
        # end of file must not re-open the clip (regression: the worker once looped the file)
        self.assertLessEqual(worker.frame_no, TOTAL_FRAMES)
        self.assertTrue(worker.snapshot_jpeg().startswith(b"\xff\xd8"))
        self.assertIs(worker.snapshot_jpeg(), worker.snapshot_jpeg())  # cached per frame
        status = worker.status()
        self.assertEqual(status["camera_id"], "test")
        self.assertFalse(status["alive"])

    def test_background_thread_can_be_stopped(self):
        worker = CameraWorker("bg", self.video, [BlobDetector()], width=0, every_n=2,
                              on_frame=lambda frame, w: time.sleep(0.02))
        worker.start()
        self.assertTrue(worker.is_alive())
        worker.stop(join=True, timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertIn(worker.state, ("stopped", "finished"))

    def test_live_source_reconnects_after_stream_drop(self):
        # Treat the clip as a live stream: reaching its end looks like a dropped camera,
        # so the worker must release, wait, re-open and keep going until max_frames.
        with mock.patch("cctv_ai.worker.is_live_source", return_value=True):
            worker = CameraWorker("live", self.video, [BlobDetector()], width=0, every_n=2,
                                  max_frames=TOTAL_FRAMES + 30, reconnect_delay=0.05)
            worker.run()
        self.assertEqual(worker.state, "finished", worker.last_error)
        self.assertEqual(worker.frame_no, TOTAL_FRAMES + 30)
        self.assertIsNone(worker.last_error)  # cleared once the re-open succeeded

    def test_live_source_that_never_opens_keeps_retrying_until_stopped(self):
        with mock.patch("cctv_ai.worker.is_live_source", return_value=True):
            worker = CameraWorker("dead", str(Path(self.tmp.name) / "nope.mp4"), [BlobDetector()],
                                  width=0, reconnect_delay=0.05)
            worker.start()
            time.sleep(0.3)
            self.assertTrue(worker.is_alive())
            self.assertEqual(worker.state, "reconnecting")
            self.assertIn("cannot open source", worker.last_error)
            worker.stop(join=True, timeout=5.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.state, "stopped")

    def test_missing_file_reports_error(self):
        worker = CameraWorker("bad", str(Path(self.tmp.name) / "no_such_file.mp4"), [BlobDetector()], width=0)
        worker.run()
        self.assertEqual(worker.state, "error")
        self.assertIn("cannot open source", worker.last_error)

    def test_frame_source_reads_every_frame_with_media_time(self):
        source = FrameSource(self.video)
        self.assertFalse(source.is_live)
        self.assertFalse(source.realtime)
        self.assertTrue(source.open())
        count = 0
        while source.read() is not None:
            count += 1
        source.release()
        self.assertEqual(count, TOTAL_FRAMES)
        self.assertAlmostEqual(source.frame_time(10), 1.0)


class HelpersTest(unittest.TestCase):
    def test_live_source_detection(self):
        self.assertTrue(is_live_source(0))
        self.assertTrue(is_live_source("1"))
        self.assertTrue(is_live_source("rtsp://admin:pass@10.0.0.5:554/stream"))
        self.assertTrue(is_live_source("http://192.168.1.5:4747/video"))
        self.assertFalse(is_live_source("clips/walk.mp4"))

    def test_credentials_are_masked(self):
        self.assertEqual(mask_credentials("rtsp://admin:secret@10.0.0.5:554/x"), "rtsp://***@10.0.0.5:554/x")
        self.assertEqual(mask_credentials("http://192.168.1.5:4747/video"), "http://192.168.1.5:4747/video")
        self.assertEqual(mask_credentials(0), "0")


if __name__ == "__main__":
    unittest.main()
