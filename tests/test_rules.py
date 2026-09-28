import json
import tempfile
import unittest
from pathlib import Path

from cctv_ai.rules import Alert, AlertManager, CameraRules, RuleEngine, Zone, load_cameras_config
from cctv_ai.tracking import Track

W, H = 640, 480
RIGHT_HALF = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
INSIDE = (400, 100, 440, 300)    # feet at (420, 300): right half
OUTSIDE = (100, 100, 140, 300)   # feet at (120, 300): left half


def make_track(track_id, bbox, now, first_seen=None, path=None, cls="person"):
    track = Track(
        track_id=track_id, bbox=list(bbox), class_name=cls, confidence=1.0,
        first_seen=now if first_seen is None else first_seen, last_seen=now, hits=5,
    )
    if path:
        for sample in path:
            track.history.append(tuple(sample))
    else:
        cx, cy = track.centroid
        track.history.append((now, cx, cy))
    return track


def engine(**kwargs):
    rules = CameraRules(camera_id="c", **kwargs)
    return RuleEngine(rules, AlertManager(cooldown_seconds=rules.cooldown_seconds))


def types(alerts):
    return [a.type for a in alerts]


class ZoneRulesTest(unittest.TestCase):
    def test_restricted_zone_alerts_once_per_cooldown(self):
        eng = engine(zones=[Zone("right", "restricted", RIGHT_HALF)], cooldown_seconds=10, loiter_seconds=0, crowd_limit=0)
        inside = make_track(1, INSIDE, 0.0)
        first = eng.evaluate([inside], (H, W), 0.0)
        self.assertEqual(types(first), ["restricted_area"])
        self.assertEqual(first[0].zone, "right")
        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(first[0].severity, "high")
        self.assertEqual(eng.evaluate([inside], (H, W), 5.0), [])
        self.assertEqual(types(eng.evaluate([inside], (H, W), 10.0)), ["restricted_area"])
        self.assertEqual(eng.evaluate([make_track(2, OUTSIDE, 20.0)], (H, W), 20.0), [])

    def test_danger_zone_is_industrial_safety_critical(self):
        eng = engine(zones=[Zone("press", "danger", RIGHT_HALF)], loiter_seconds=0, crowd_limit=0)
        alerts = eng.evaluate([make_track(1, INSIDE, 0.0)], (H, W), 0.0)
        self.assertEqual(types(alerts), ["industrial_safety"])
        self.assertEqual(alerts[0].severity, "critical")

    def test_loiter_zone_needs_dwell_time_and_resets_on_exit(self):
        zone = Zone("wait", "loiter", RIGHT_HALF)
        eng = engine(zones=[zone], loiter_seconds=30, crowd_limit=0)
        self.assertEqual(eng.evaluate([make_track(1, INSIDE, 0.0)], (H, W), 0.0), [])
        self.assertEqual(eng.evaluate([make_track(1, INSIDE, 29.0)], (H, W), 29.0), [])
        self.assertEqual(types(eng.evaluate([make_track(1, INSIDE, 30.0)], (H, W), 30.0)), ["loitering"])

        eng2 = engine(zones=[zone], loiter_seconds=30, crowd_limit=0)
        eng2.evaluate([make_track(1, INSIDE, 0.0)], (H, W), 0.0)
        eng2.evaluate([make_track(1, OUTSIDE, 20.0)], (H, W), 20.0)
        eng2.evaluate([make_track(1, INSIDE, 40.0)], (H, W), 40.0)
        self.assertEqual(eng2.evaluate([make_track(1, INSIDE, 69.0)], (H, W), 69.0), [])
        self.assertEqual(types(eng2.evaluate([make_track(1, INSIDE, 70.0)], (H, W), 70.0)), ["loitering"])

    def test_loitering_anywhere_uses_small_movement_radius(self):
        eng = engine(loiter_seconds=30, loiter_radius=0.1, crowd_limit=0)
        still_path = [(float(t), 300.0, 200.0) for t in range(0, 36)]
        still = make_track(1, (280, 100, 320, 300), 35.0, first_seen=0.0, path=still_path)
        self.assertEqual(types(eng.evaluate([still], (H, W), 35.0)), ["loitering"])

        walk_path = [(float(t), 10.0 + 15.0 * t, 200.0) for t in range(0, 36)]
        walker = make_track(2, (515, 100, 555, 300), 35.0, first_seen=0.0, path=walk_path)
        self.assertEqual(eng.evaluate([walker], (H, W), 35.0), [])

        young_path = [(float(t), 300.0, 200.0) for t in range(0, 11)]
        young = make_track(3, (280, 100, 320, 300), 10.0, first_seen=0.0, path=young_path)
        self.assertEqual(eng.evaluate([young], (H, W), 10.0), [])

    def test_crowd_needs_limit_for_consecutive_rounds(self):
        eng = engine(crowd_limit=3, crowd_min_hits=2, loiter_seconds=0)
        three = [make_track(i, (50 * i, 100, 50 * i + 40, 300), 0.0) for i in range(1, 4)]
        self.assertEqual(eng.evaluate(three, (H, W), 0.0), [])
        alerts = eng.evaluate(three, (H, W), 1.0)
        self.assertEqual(types(alerts), ["crowd"])
        self.assertIn("3 people", alerts[0].message)
        self.assertEqual(eng.evaluate(three[:2], (H, W), 2.0), [])   # under the limit resets the counter
        self.assertEqual(eng.evaluate(three, (H, W), 3.0), [])       # first round over the limit again

    def test_crowd_zone_counts_only_people_inside(self):
        eng = engine(zones=[Zone("hall", "crowd", RIGHT_HALF)], crowd_limit=2, crowd_min_hits=1, loiter_seconds=0)
        left_pair = [make_track(1, (10, 100, 50, 300), 0.0), make_track(2, (100, 100, 140, 300), 0.0)]
        self.assertEqual(eng.evaluate(left_pair, (H, W), 0.0), [])
        right_pair = [make_track(3, (400, 100, 440, 300), 1.0), make_track(4, (500, 100, 540, 300), 1.0)]
        alerts = eng.evaluate(right_pair, (H, W), 1.0)
        self.assertEqual(types(alerts), ["crowd"])
        self.assertEqual(alerts[0].zone, "hall")

    def test_wrong_direction_only_against_allowed_vector(self):
        eng = engine(direction={"allowed": [1, 0], "min_travel": 0.1, "window_seconds": 3}, loiter_seconds=0, crowd_limit=0)
        left_path = [(0.0, 400.0, 200.0), (1.0, 350.0, 200.0), (2.0, 300.0, 200.0)]  # 100 px left, limit is 64 px
        mover = make_track(1, (280, 100, 320, 300), 2.0, first_seen=0.0, path=left_path)
        self.assertEqual(types(eng.evaluate([mover], (H, W), 2.0)), ["wrong_direction"])

        right_path = [(0.0, 300.0, 200.0), (1.0, 350.0, 200.0), (2.0, 400.0, 200.0)]
        ok_mover = make_track(2, (380, 100, 420, 300), 2.0, first_seen=0.0, path=right_path)
        self.assertEqual(eng.evaluate([ok_mover], (H, W), 2.0), [])

        short_path = [(0.0, 320.0, 200.0), (2.0, 300.0, 200.0)]  # only 20 px
        shuffler = make_track(3, (280, 100, 320, 300), 2.0, first_seen=0.0, path=short_path)
        self.assertEqual(eng.evaluate([shuffler], (H, W), 2.0), [])

    def test_feature_switches_silence_rules(self):
        zones = [Zone("right", "restricted", RIGHT_HALF), Zone("press", "danger", RIGHT_HALF)]
        inside = make_track(1, INSIDE, 0.0)
        everything = engine(zones=zones, loiter_seconds=0, crowd_limit=1, crowd_min_hits=1)
        self.assertEqual(sorted(types(everything.evaluate([inside], (H, W), 0.0))),
                         ["crowd", "industrial_safety", "restricted_area"])

        only_crowd = engine(zones=zones, loiter_seconds=0, crowd_limit=1, crowd_min_hits=1, features={"crowd", "person"})
        self.assertEqual(types(only_crowd.evaluate([inside], (H, W), 0.0)), ["crowd"])

        only_zones = engine(zones=zones, loiter_seconds=0, crowd_limit=1, crowd_min_hits=1,
                            features={"restricted_area", "industrial_safety"})
        self.assertEqual(sorted(types(only_zones.evaluate([inside], (H, W), 0.0))),
                         ["industrial_safety", "restricted_area"])

        rules = CameraRules.from_dict({"features": ["person", "loitering"]}, "c")
        self.assertEqual(rules.features, {"person", "loitering"})
        self.assertTrue(rules.enabled("loitering"))
        self.assertFalse(rules.enabled("crowd"))
        self.assertEqual(rules.as_dict()["features"], ["loitering", "person"])

    def test_model_based_alerts_for_weapon_fire_and_missing_ppe(self):
        from cctv_ai.rules import ppe_missing
        eng = engine(loiter_seconds=0, crowd_limit=0)
        gun = make_track(1, (100, 100, 140, 140), 0.0, cls="pistol")
        gun.category = "weapon"
        smoke = make_track(2, (300, 50, 400, 150), 0.0, cls="smoke")
        smoke.category = "fire_smoke"
        no_helmet = make_track(3, (500, 100, 540, 300), 0.0, cls="NO-Hardhat")
        no_helmet.category = "ppe"
        helmet = make_track(4, (10, 100, 50, 300), 0.0, cls="Hardhat")
        helmet.category = "ppe"
        knife = make_track(5, (200, 200, 240, 260), 0.0, cls="knife")   # COCO knife, category object
        alerts = eng.evaluate([gun, smoke, no_helmet, helmet, knife], (H, W), 0.0)
        self.assertEqual(sorted(types(alerts)), ["fire_smoke", "ppe", "weapon", "weapon"])
        messages = {}
        for a in alerts:
            messages.setdefault(a.type, []).append(a.message)
            if a.type == "weapon":
                self.assertEqual(a.severity, "critical")
        self.assertTrue(any("pistol" in m for m in messages["weapon"]))
        self.assertTrue(any("knife" in m for m in messages["weapon"]))
        self.assertIn("NO-Hardhat", messages["ppe"][0])
        self.assertEqual(eng.evaluate([gun], (H, W), 1.0), [])  # cooldown per track

        quiet = engine(loiter_seconds=0, crowd_limit=0, features={"person"})
        self.assertEqual(quiet.evaluate([gun, smoke, no_helmet], (H, W), 0.0), [])

        self.assertTrue(ppe_missing("NO-Hardhat"))
        self.assertTrue(ppe_missing("no_safety_vest"))
        self.assertTrue(ppe_missing("without helmet"))
        self.assertFalse(ppe_missing("Hardhat"))
        self.assertFalse(ppe_missing("Safety Vest"))

    def test_vehicles_ignore_person_zones_but_follow_direction(self):
        eng = engine(zones=[Zone("right", "restricted", RIGHT_HALF)],
                     direction={"allowed": [1, 0], "min_travel": 0.1, "window_seconds": 3},
                     loiter_seconds=0, crowd_limit=0)
        path = [(0.0, 500.0, 200.0), (2.0, 300.0, 200.0)]
        car = make_track(1, (280, 100, 320, 300), 2.0, first_seen=0.0, path=path, cls="vehicle")
        self.assertEqual(types(eng.evaluate([car], (H, W), 2.0)), ["wrong_direction"])


class AlertManagerTest(unittest.TestCase):
    def test_cooldown_log_and_subscribers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "alerts.jsonl"
            manager = AlertManager(cooldown_seconds=10, log_path=log)
            seen = []
            manager.subscribe(seen.append)

            first = Alert("crowd", "high", "c", "Crowd of 5", timestamp=0.0)
            self.assertIs(manager.raise_alert(first), first)
            self.assertIsNone(manager.raise_alert(Alert("crowd", "high", "c", "again", timestamp=5.0)))
            self.assertIsNotNone(manager.raise_alert(Alert("crowd", "high", "c", "later", timestamp=10.0)))
            self.assertIsNotNone(manager.raise_alert(Alert("crowd", "high", "other_cam", "x", timestamp=1.0)))

            self.assertEqual(len(manager), 3)
            self.assertEqual(len(seen), 3)
            recent = manager.recent(limit=2)
            self.assertEqual([a["message"] for a in recent], ["x", "later"])  # newest first
            self.assertEqual(manager.recent(camera_id="other_cam")[0]["camera_id"], "other_cam")
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["type"], "crowd")


class ConfigTest(unittest.TestCase):
    def test_zone_validation(self):
        with self.assertRaises(ValueError):
            Zone("a", "unknown", RIGHT_HALF)
        with self.assertRaises(ValueError):
            Zone("a", "restricted", [[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            Zone("a", "restricted", [[0, 0], [640, 0], [640, 480]])

    def test_rules_from_dict_and_back(self):
        rules = CameraRules.from_dict({"zones": [{"name": "z", "kind": "loiter", "polygon": RIGHT_HALF}],
                                       "loiter_seconds": 12, "crowd_limit": 4}, "gate")
        self.assertEqual(rules.camera_id, "gate")
        self.assertEqual(rules.zones[0].kind, "loiter")
        self.assertEqual(rules.loiter_seconds, 12.0)
        self.assertEqual(rules.as_dict()["crowd_limit"], 4)
        self.assertIsNone(rules.direction)

    def test_project_example_config_loads(self):
        cameras = load_cameras_config(Path(__file__).resolve().parent.parent / "cameras.json")
        by_id = {c["camera_id"]: c for c in cameras}
        rules = CameraRules.from_dict(by_id["cam1"]["rules"], "cam1")
        self.assertEqual([z.kind for z in rules.zones], ["restricted", "danger"])
        demo = CameraRules.from_dict(by_id["vtest_demo"]["rules"], "vtest_demo")
        self.assertEqual([z.name for z in demo.zones], ["taped_off_area"])
        self.assertEqual(by_id["vtest_demo"]["detector"], "yolo")


if __name__ == "__main__":
    unittest.main()
