import unittest
from unittest import mock

from cctv_ai.detectors import YoloOnnxDetector
from cctv_ai.detectors.base import Detection
from cctv_ai.features import (
    FEATURE_KEYS, FEATURES, default_features, feature_catalog, filter_detections, parse_feature_list,
    validate_features,
)


class CatalogTest(unittest.TestCase):
    def test_fifteen_numbered_unique_features(self):
        self.assertEqual(len(FEATURES), 15)
        self.assertEqual([f["number"] for f in FEATURES], list(range(1, 16)))
        self.assertEqual(len(set(FEATURE_KEYS)), 15)
        self.assertEqual(FEATURES[-1]["key"], "weapon")
        for f in feature_catalog():
            self.assertIn(f["status"], ("available", "needs_model", "partial", "planned"))
            self.assertTrue(f["name"] and f["description"])

    def test_status_follows_the_yolo_model(self):
        with mock.patch.object(YoloOnnxDetector, "available", return_value=False):
            vehicle = next(f for f in feature_catalog() if f["key"] == "vehicle")
            self.assertEqual(vehicle["status"], "needs_model")
        with mock.patch.object(YoloOnnxDetector, "available", return_value=True):
            vehicle = next(f for f in feature_catalog() if f["key"] == "vehicle")
            self.assertEqual(vehicle["status"], "available")

    def test_custom_model_features_follow_their_files(self):
        import cctv_ai.features as features_module
        with mock.patch.object(features_module, "custom_model_available", return_value=False), \
             mock.patch.object(YoloOnnxDetector, "available", return_value=False):
            weapon = next(f for f in feature_catalog() if f["key"] == "weapon")
            self.assertEqual(weapon["status"], "needs_model")
            self.assertTrue(any("weapon.names" in need for need in weapon["needs"]))
        with mock.patch.object(features_module, "custom_model_available", return_value=True):
            statuses = {f["key"]: f["status"] for f in feature_catalog()}
            self.assertEqual(statuses["weapon"], "available")
            self.assertEqual(statuses["fire_smoke"], "available")
            self.assertEqual(statuses["ppe"], "available")
            self.assertEqual(statuses["fight"], "planned")

    def test_weapon_is_partial_with_only_the_object_model(self):
        import cctv_ai.features as features_module
        with mock.patch.object(features_module, "custom_model_available", return_value=False), \
             mock.patch.object(YoloOnnxDetector, "available", return_value=True):
            weapon = next(f for f in feature_catalog() if f["key"] == "weapon")
            self.assertEqual(weapon["status"], "partial")
            self.assertIn("knife only", weapon["note"])
        with mock.patch.object(features_module, "custom_model_available", return_value=False), \
             mock.patch.object(YoloOnnxDetector, "available", return_value=False):
            weapon = next(f for f in feature_catalog() if f["key"] == "weapon")
            self.assertEqual(weapon["status"], "needs_model")
            self.assertEqual(weapon["note"], "needs model file")

    def test_weapon_classes(self):
        from cctv_ai.features import is_weapon_class
        for name in ("knife", "Knife", "gun", "hand_gun", "Rifle", "machete"):
            self.assertTrue(is_weapon_class(name), name)
        for name in ("fork", "person", "cell phone"):
            self.assertFalse(is_weapon_class(name), name)

    def test_defaults_skip_planned_and_face_access(self):
        defaults = default_features()
        self.assertIn("person", defaults)
        self.assertIn("restricted_area", defaults)
        self.assertIn("weapon", defaults)
        self.assertNotIn("fight", defaults)
        self.assertNotIn("face_access", defaults)
        self.assertEqual(len(defaults), 11)


class ValidationTest(unittest.TestCase):
    def test_validate_and_parse(self):
        self.assertEqual(validate_features(["person", " crowd "]), {"person", "crowd"})
        self.assertEqual(validate_features(["weapon", "fire_smoke", "ppe"]), {"weapon", "fire_smoke", "ppe"})
        self.assertEqual(parse_feature_list("person,vehicle"), {"person", "vehicle"})
        self.assertIsNone(parse_feature_list(None))
        self.assertIsNone(parse_feature_list("  "))
        with self.assertRaises(ValueError) as unknown:
            validate_features(["person", "teleport"])
        self.assertIn("unknown feature 'teleport'", str(unknown.exception))
        with self.assertRaises(ValueError) as planned:
            validate_features(["fight"])
        self.assertIn("not built yet", str(planned.exception))

    def test_filter_detections_by_switch(self):
        dets = [
            Detection("person", "person", 0.9, [0, 0, 10, 10]),
            Detection("vehicle", "car", 0.8, [0, 0, 10, 10]),
            Detection("bag", "backpack", 0.7, [0, 0, 10, 10]),
            Detection("face", "face", 0.6, [0, 0, 10, 10]),
            Detection("weapon", "pistol", 0.5, [0, 0, 10, 10]),
            Detection("object", "knife", 0.4, [0, 0, 10, 10]),   # knife from the COCO model
        ]
        self.assertEqual(len(filter_detections(dets, None)), 6)
        self.assertEqual([d.class_name for d in filter_detections(dets, {"weapon"})], ["pistol", "knife"])
        self.assertEqual([d.class_name for d in filter_detections(dets, {"object_left_taken"})], ["backpack", "knife"])
        self.assertEqual([d.class_name for d in filter_detections(dets, {"person"})], ["person"])
        self.assertEqual([d.class_name for d in filter_detections(dets, {"person"})], ["person"])
        self.assertEqual([d.class_name for d in filter_detections(dets, {"vehicle", "object_left_taken"})],
                         ["car", "backpack", "knife"])
        self.assertEqual(filter_detections(dets, set()), [])


if __name__ == "__main__":
    unittest.main()
