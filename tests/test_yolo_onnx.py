import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cctv_ai.detectors import PersonDetector, YoloOnnxDetector, build_detectors
from cctv_ai.detectors.yolo_onnx import COCO_CLASSES, category_for, decode_predictions

N_CLASSES = len(COCO_CLASSES)


def raw_output(rows, layout="channels_first"):
    """Build a fake YOLOv8 output tensor from (cx, cy, w, h, class_index, score) rows."""
    pred = np.zeros((len(rows), 4 + N_CLASSES), dtype=np.float32)
    for r, (cx, cy, w, h, cls, score) in enumerate(rows):
        pred[r, :4] = (cx, cy, w, h)
        pred[r, 4 + cls] = score
    if layout == "channels_first":
        return pred.T[np.newaxis]  # (1, 84, N) as exported by ultralytics
    return pred[np.newaxis]        # (1, N, 84)


class DecodeTest(unittest.TestCase):
    ROWS = [
        (100, 100, 50, 80, 0, 0.90),    # person
        (102, 101, 50, 80, 0, 0.80),    # same person again, must be removed by NMS
        (300, 200, 40, 20, 67, 0.60),   # cell phone
        (500, 300, 30, 30, 2, 0.10),    # car, below the confidence threshold
    ]

    def test_scales_back_filters_and_suppresses_duplicates(self):
        # the image was 1280x960 and was shrunk by 0.5 before inference
        dets = decode_predictions(raw_output(self.ROWS), scale=0.5, width=1280, height=960, conf=0.35, nms=0.45)
        self.assertEqual([d.class_name for d in dets], ["person", "cell phone"])
        self.assertEqual(dets[0].bbox, [150, 120, 250, 280])
        self.assertAlmostEqual(dets[0].confidence, 0.9, places=5)
        self.assertEqual(dets[0].category, "person")
        self.assertEqual(dets[1].bbox, [560, 380, 640, 420])
        self.assertEqual(dets[1].category, "object")
        self.assertEqual(dets[1].metadata["source"], "yolo")

    def test_class_filter_and_threshold(self):
        only_people = decode_predictions(raw_output(self.ROWS), 0.5, 1280, 960, classes={"person"})
        self.assertEqual([d.class_name for d in only_people], ["person"])
        nothing = decode_predictions(raw_output(self.ROWS), 0.5, 1280, 960, conf=0.95)
        self.assertEqual(nothing, [])

    def test_transposed_layout_is_accepted(self):
        dets = decode_predictions(raw_output(self.ROWS, layout="channels_last"), 1.0, 640, 480)
        self.assertEqual([d.class_name for d in dets], ["person", "cell phone"])

    def test_boxes_are_clipped_to_the_image(self):
        dets = decode_predictions(raw_output([(5, 5, 40, 40, 0, 0.9)]), 1.0, 100, 100)
        self.assertEqual(dets[0].bbox, [0, 0, 25, 25])

    def test_live_classes_include_knife(self):
        from cctv_ai.detectors.yolo_onnx import LIVE_CLASSES
        self.assertIn("knife", LIVE_CLASSES)
        self.assertIn("person", LIVE_CLASSES)
        self.assertIn("bottle", LIVE_CLASSES)
        self.assertIn("cell phone", LIVE_CLASSES)
        self.assertNotIn("chair", LIVE_CLASSES)
        self.assertNotIn("pizza", LIVE_CLASSES)

    def test_categories(self):
        self.assertEqual(category_for("person"), "person")
        self.assertEqual(category_for("truck"), "vehicle")
        self.assertEqual(category_for("handbag"), "bag")
        self.assertEqual(category_for("laptop"), "object")
        self.assertEqual(len(COCO_CLASSES), 80)


class YoloxDecodeTest(unittest.TestCase):
    def test_grid_decoding(self):
        from cctv_ai.detectors.yolo_onnx import decode_yolox, yolox_grids
        grids, strides = yolox_grids(640)
        self.assertEqual(grids.shape, (8400, 2))
        self.assertEqual(int(strides[0, 0]), 8)
        self.assertEqual(int(strides[6400, 0]), 16)
        self.assertEqual(int(strides[8000, 0]), 32)

        pred = np.zeros((1, 8400, 5 + N_CLASSES), dtype=np.float32)
        row = 20 * 80 + 10  # stride-8 cell at grid x=10, y=20
        pred[0, row, :4] = (0.5, 0.5, np.log(50 / 8), np.log(80 / 8))
        pred[0, row, 4] = 1.0          # objectness
        pred[0, row, 5 + 0] = 0.9      # person
        row32 = 6400 + 1600 + 5 * 20 + 3  # stride-32 cell at grid x=3, y=5
        pred[0, row32, :4] = (0.0, 0.0, np.log(64 / 32), np.log(32 / 32))
        pred[0, row32, 4] = 0.8
        pred[0, row32, 5 + 2] = 0.75   # car -> score 0.6

        dets = decode_yolox(pred, scale=1.0, width=640, height=640)
        self.assertEqual([d.class_name for d in dets], ["person", "car"])
        self.assertEqual(dets[0].bbox, [59, 124, 109, 204])   # centre (84, 164), 50 x 80
        self.assertEqual(dets[1].bbox, [64, 144, 128, 176])   # centre (96, 160), 64 x 32
        self.assertAlmostEqual(dets[1].confidence, 0.6, places=5)
        with self.assertRaises(ValueError):
            decode_yolox(pred[:, :100], 1.0, 640, 640)


class ModelFileTest(unittest.TestCase):
    def test_missing_model_is_reported_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "yolov8n.onnx"
            self.assertFalse(YoloOnnxDetector.available(missing))
            with self.assertRaises(FileNotFoundError) as ctx:
                YoloOnnxDetector(model_path=missing)
            self.assertIn("download_models.py", str(ctx.exception))

    def test_build_detectors_falls_back_to_hog(self):
        from cctv_ai import detectors as detectors_module
        with mock.patch.object(YoloOnnxDetector, "available", return_value=False), \
             mock.patch.object(detectors_module, "discover_custom_models", return_value={}):
            detectors = build_detectors(faces=False, live=True)
        self.assertEqual([type(d) for d in detectors], [PersonDetector])

    def test_detector_preference(self):
        from cctv_ai import detectors as detectors_module
        with mock.patch.object(detectors_module, "discover_custom_models", return_value={}):
            with mock.patch.object(YoloOnnxDetector, "available", return_value=True):
                self.assertEqual([type(d) for d in build_detectors(prefer="hog")], [PersonDetector])
            with mock.patch.object(YoloOnnxDetector, "available", return_value=False), \
                 mock.patch("cctv_ai.detectors.yolo_onnx.find_model", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    build_detectors(prefer="yolo")
                self.assertEqual([type(d) for d in build_detectors(prefer="auto")], [PersonDetector])
            with self.assertRaises(ValueError):
                build_detectors(prefer="magic")

    def test_custom_models_are_discovered_from_names_files(self):
        from cctv_ai.detectors.yolo_onnx import custom_model_available, discover_custom_models
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(discover_custom_models(folder), {})
            (folder / "weapon.names").write_text("# classes\npistol\nknife\n\n", encoding="utf-8")
            self.assertEqual(discover_custom_models(folder), {})  # names without the onnx file
            self.assertFalse(custom_model_available("weapon", folder))
            (folder / "weapon.onnx").write_bytes(b"fake")
            found = discover_custom_models(folder)
            self.assertEqual(list(found), ["weapon"])
            self.assertEqual(found["weapon"], (folder / "weapon.onnx", ["pistol", "knife"]))
            self.assertTrue(custom_model_available("weapon", folder))
            (folder / "orphan.onnx").write_bytes(b"fake")  # onnx without names is ignored
            self.assertEqual(list(discover_custom_models(folder)), ["weapon"])


if __name__ == "__main__":
    unittest.main()
