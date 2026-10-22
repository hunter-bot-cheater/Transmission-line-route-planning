import unittest

from pyproj import Geod

from transmission_path_planner.geometry import path_length_km
from transmission_path_planner.metrics import compute_deviation, eval_error, eval_planned_route_error


class MetricPrecisionTests(unittest.TestCase):
    def test_parallel_routes_use_local_projected_meters(self):
        planned = [(120.0, 24.0), (120.02, 24.0)]
        reference = [(120.0, 24.001), (120.02, 24.001)]

        metrics = compute_deviation(planned, reference)

        self.assertAlmostEqual(metrics["dev_hausdorff_km"], 0.111, delta=0.002)
        self.assertAlmostEqual(metrics["dev_mean_km"], 0.111, delta=0.002)
        self.assertEqual(metrics["deviation_sample_step_m"], 5)
        self.assertIn("+proj=aeqd", metrics["deviation_crs"])

    def test_symmetric_hausdorff_counts_reference_segments_between_vertices(self):
        planned = [(120.0, 24.0), (120.01, 24.0)]
        reference = [(120.0, 24.0), (120.005, 24.001), (120.01, 24.0)]

        metrics = compute_deviation(planned, reference)

        self.assertGreater(metrics["dev_haus_sym_km"], 0.05)
        self.assertEqual(metrics["dev_hausdorff_km"], metrics["dev_haus_sym_km"])

    def test_zero_length_reference_component_remains_in_distance_queries(self):
        planned = [(0.0, 0.0), (0.00009, 0.0)]
        reference = [
            [(0.0, 0.001), (0.00009, 0.001)],
            [(0.000045, 0.0), (0.000045, 0.0)],
        ]

        metrics = compute_deviation(planned, reference)

        self.assertLess(metrics["dev_mean_km"], 0.09)
        self.assertGreater(metrics["dev_hausdorff_km"], 0.09)

    def test_identical_route_crossing_dateline_keeps_local_scale(self):
        route = [(179.99, 10.0), (-179.99, 10.0)]

        metrics = compute_deviation(route, route)
        error = eval_error(metrics["real_length_km"], metrics)

        self.assertLess(metrics["dev_hausdorff_km"], 0.001)
        self.assertGreater(metrics["real_length_km"], 2.0)
        self.assertLess(metrics["real_length_km"], 2.5)
        self.assertTrue(error["pass8"])

    def test_long_identical_route_uses_batched_queries(self):
        route = [(0.0, 0.0), (2.7, 0.0)]

        metrics = compute_deviation(route, route)

        self.assertGreater(metrics["real_length_km"], 250.0)
        self.assertEqual(metrics["dev_hausdorff_km"], 0.0)

    def test_pass8_uses_unrounded_planned_route_length(self):
        from pyproj import Transformer

        forward = Transformer.from_crs(
            "+proj=aeqd +lat_0=24 +lon_0=121 +datum=WGS84 +units=m +no_defs",
            "EPSG:4326", always_xy=True)
        route = [forward.transform(x, 4.11 if index % 2 == 0 else -4.11)
                 for index, x in enumerate(range(0, 1001, 20))]
        reference = [forward.transform(0, 0), forward.transform(1000, 0)]
        dev = compute_deviation(route, reference)
        raw_length = path_length_km(route)

        self.assertGreater(raw_length, 1.08)
        self.assertLessEqual(round(raw_length, 3), 1.08)
        self.assertTrue(eval_error(round(raw_length, 3), dev)["pass8"])
        self.assertFalse(eval_planned_route_error(route, dev)["pass8"])

    def test_reported_error_precision_matches_threshold_decision(self):
        error = eval_error(1.080004, {
            "real_length_km": 1.0,
            "dev_hausdorff_km": 0.0,
            "dev_haus_fwd_km": 0.0,
        })
        self.assertFalse(error["pass8"])
        self.assertGreater(error["worst_err_pct"], 8.0)
        self.assertEqual(error["worst_err_pct"], error["len_err_pct"])

    def test_pass8_uses_unrounded_deviation_and_reference_length(self):
        reference = [(121.0, 24.0), (121.00983, 24.0)]
        planned = [(121.0, 24.000724), (121.00983, 24.000724)]
        plan_length_km = Geod(ellps="WGS84").line_length(
            [point[0] for point in planned], [point[1] for point in planned]
        ) / 1000.0

        metrics = compute_deviation(planned, reference)
        error = eval_error(plan_length_km, metrics)

        self.assertGreater(error["haus_pct"], 8.0)
        self.assertFalse(error["pass8"])


if __name__ == "__main__":
    unittest.main()
