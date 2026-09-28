import unittest

from cctv_ai.detectors.base import Detection
from cctv_ai.tracking import IouTracker


def det(x, y=50, w=40, h=100, cls="person"):
    return Detection(cls, cls, 0.9, [x, y, x + w, y + h])


class IouTrackerTest(unittest.TestCase):
    def test_moving_object_keeps_its_id(self):
        tracker = IouTracker(min_hits=2)
        ids = []
        for i in range(10):
            tracks = tracker.update([det(10 + 5 * i)], now=i * 0.2)
            ids += [t.track_id for t in tracks]
        self.assertEqual(set(ids), {1})
        self.assertEqual(len(ids), 9)  # confirmed from the second match onwards

    def test_two_objects_get_two_stable_ids(self):
        tracker = IouTracker(min_hits=1)
        seen = set()
        for i in range(8):
            tracks = tracker.update([det(10 + 5 * i), det(300 - 5 * i)], now=i * 0.2)
            self.assertEqual(len(tracks), 2)
            seen |= {t.track_id for t in tracks}
        self.assertEqual(seen, {1, 2})

    def test_lost_track_expires_after_max_misses(self):
        tracker = IouTracker(min_hits=1, max_misses=3)
        tracker.update([det(10)], now=0.0)
        for i in range(1, 4):
            tracker.update([], now=float(i))
            self.assertEqual(len(tracker.tracks), 1)
            self.assertEqual(tracker.active_tracks(), [])
        tracker.update([], now=4.0)
        self.assertEqual(len(tracker.tracks), 0)

    def test_jump_beyond_iou_uses_distance_fallback(self):
        tracker = IouTracker(min_hits=1)
        tracker.update([det(10)], now=0.0)
        tracks = tracker.update([det(60)], now=0.2)  # shift 50 px > box width, IoU is 0
        self.assertEqual([t.track_id for t in tracks], [1])
        self.assertEqual(len(tracker.tracks), 1)

    def test_far_jump_creates_a_new_track(self):
        tracker = IouTracker(min_hits=1)
        tracker.update([det(10)], now=0.0)
        tracker.update([det(400)], now=0.2)  # 390 px away, more than one box diagonal
        self.assertEqual(sorted(tracker.tracks), [1, 2])

    def test_confirmed_totals_count_each_object_once(self):
        tracker = IouTracker(min_hits=2, max_misses=1)
        tracker.update([det(10), det(300, cls="vehicle")], now=0.0)
        self.assertEqual(tracker.confirmed_by_category, {})          # nothing confirmed after one hit
        tracker.update([det(15), det(310, cls="vehicle")], now=0.2)
        self.assertEqual(tracker.confirmed_by_category, {"person": 1, "vehicle": 1})
        for i in range(3, 8):                                          # staying in view adds nothing
            tracker.update([det(10 + 5 * i), det(300 + 10 * i, cls="vehicle")], now=i * 0.2)
        self.assertEqual(tracker.confirmed_by_category, {"person": 1, "vehicle": 1})
        tracker.update([], now=2.0)
        tracker.update([], now=2.2)                                    # both tracks expire
        tracker.update([det(500, cls="vehicle")], now=2.4)
        tracker.update([det(505, cls="vehicle")], now=2.6)             # a new vehicle
        self.assertEqual(tracker.confirmed_by_category, {"person": 1, "vehicle": 2})

    def test_category_is_carried_from_the_detection(self):
        tracker = IouTracker(min_hits=1)
        gun = Detection("weapon", "pistol", 0.9, [10, 10, 50, 50])
        track = tracker.update([gun], now=0.0)[0]
        self.assertEqual(track.category, "weapon")
        self.assertEqual(track.as_dict()["category"], "weapon")

    def test_class_mismatch_creates_new_track(self):
        tracker = IouTracker(min_hits=1)
        tracker.update([det(10, cls="person")], now=0.0)
        tracker.update([det(10, cls="vehicle")], now=0.2)
        self.assertEqual(sorted(tracker.tracks), [1, 2])

    def test_history_displacement_and_travel_span(self):
        tracker = IouTracker(min_hits=1)
        track = None
        for i in range(11):
            track = tracker.update([det(10 + 5 * i)], now=i * 0.2)[0]
        dx, dy, dt = track.displacement(1.0)
        self.assertAlmostEqual(dt, 1.2, places=6)   # window plus the sample just before it
        self.assertAlmostEqual(dx, 30.0, places=6)
        self.assertAlmostEqual(dy, 0.0, places=6)
        covered, radius = track.travel_span(1.0)
        self.assertGreaterEqual(covered, 1.0)
        self.assertAlmostEqual(radius, 30.0, places=6)
        self.assertEqual(track.foot, (80.0, 150.0))
        self.assertEqual(track.as_dict()["hits"], 11)


if __name__ == "__main__":
    unittest.main()
