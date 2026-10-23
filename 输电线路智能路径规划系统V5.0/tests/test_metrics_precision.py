import unittest

from pyproj import Transformer

from transmission_path_planner.geometry import path_length_km
from transmission_path_planner.metrics import (
    compute_deviation,
    eval_error,
    eval_planned_route_error,
)


def similarity(metrics):
    return metrics["route_similarity"]


class MetricPrecisionTests(unittest.TestCase):
    def test_parallel_routes_report_bidirectional_distance_and_visual_coverage(self):
        planned = [(120.0, 24.0), (120.02, 24.0)]
        reference = [(120.0, 24.001), (120.02, 24.001)]

        metrics = compute_deviation(planned, reference)
        score = similarity(metrics)

        self.assertAlmostEqual(score["planned_to_reference"]["mean_m"], 111, delta=2)
        self.assertAlmostEqual(score["reference_to_planned"]["mean_m"], 111, delta=2)
        self.assertAlmostEqual(score["symmetric"]["p95_m"], score["planned_to_reference"]["p95_m"])
        self.assertAlmostEqual(score["symmetric"]["sampled_hausdorff_m"], score["planned_to_reference"]["sampled_max_m"])
        self.assertEqual(score["sampling_interval_m"], 20.0)
        self.assertLess(score["symmetric"]["coverage_f1_within_m"]["100"], 0.1)
        self.assertEqual(metrics["route_similarity_version"], 2)

    def test_reference_line_coverage_in_planning_window_is_reported(self):
        route = [(120.0, 24.0), (120.01, 24.0)]

        score = similarity(compute_deviation(
            route, route, planning_bounds_wgs84=(120.0, 23.999, 120.005, 24.001)))

        coverage = score["planning_window_coverage"]
        self.assertAlmostEqual(coverage["reference_length_coverage_ratio"], 0.5, delta=0.002)
        self.assertGreater(coverage["reference_length_outside_window_m"], 500.0)

    def test_short_planned_subroute_does_not_hide_uncovered_reference_ends(self):
        planned = [(120.002, 24.0), (120.008, 24.0)]
        reference = [(120.0, 24.0), (120.01, 24.0)]

        score = similarity(compute_deviation(planned, reference))

        self.assertLess(score["planned_to_reference"]["p95_m"], 1)
        self.assertGreater(score["reference_to_planned"]["p95_m"], 100)
        self.assertGreater(score["symmetric"]["coverage_f1_within_m"]["100"], 0.8)
        self.assertLess(score["symmetric"]["coverage_f1_within_m"]["100"], 0.9)
        self.assertGreater(score["symmetric"]["sampled_hausdorff_m"], 180)

    def test_symmetric_p95_uses_the_worse_direction(self):
        planned = [(120.0, 24.0), (120.02, 24.0)]
        reference = [(120.005, 24.0), (120.015, 24.0)]

        score = similarity(compute_deviation(planned, reference))
        directional_p95 = max(
            score["planned_to_reference"]["p95_m"],
            score["reference_to_planned"]["p95_m"],
        )

        self.assertGreater(score["planned_to_reference"]["p95_m"],
                           score["reference_to_planned"]["p95_m"])
        self.assertEqual(score["symmetric"]["p95_m"], directional_p95)

    def test_reference_segments_between_vertices_are_counted(self):
        planned = [(120.0, 24.0), (120.01, 24.0)]
        reference = [(120.0, 24.0), (120.005, 24.001), (120.01, 24.0)]

        score = similarity(compute_deviation(planned, reference))

        self.assertGreater(score["reference_to_planned"]["sampled_max_m"], 50)
        self.assertGreater(score["symmetric"]["p95_m"], 50)

    def test_zero_length_reference_component_is_ignored_for_sampling(self):
        planned = [(0.0, 0.0), (0.00009, 0.0)]
        reference = [
            [(0.0, 0.001), (0.00009, 0.001)],
            [(0.000045, 0.0), (0.000045, 0.0)],
        ]

        score = similarity(compute_deviation(planned, reference))

        self.assertLess(score["planned_to_reference"]["mean_m"], 90)
        self.assertGreater(score["reference_to_planned"]["sampled_max_m"], 100)

    def test_reversed_reference_keeps_endpoint_and_frechet_similarity(self):
        route = [(121.0, 24.0), (121.005, 24.002), (121.01, 24.0)]

        score = similarity(compute_deviation(route, list(reversed(route))))

        self.assertLess(score["endpoint_offsets"]["planned_start_to_reference_endpoint_m"], 0.01)
        self.assertLess(score["endpoint_offsets"]["planned_end_to_reference_endpoint_m"], 0.01)
        self.assertEqual(score["endpoint_offsets"]["orientation"], "reversed")
        self.assertLess(score["discrete_frechet_approx"]["distance_m"], 0.01)
        self.assertEqual(score["discrete_frechet_approx"]["sample_points_per_route"], 256)

    def test_vertex_density_does_not_change_arc_length_sampled_metrics(self):
        coarse = [(121.0, 24.0), (121.02, 24.0)]
        dense = [(121.0 + i * 0.0001, 24.0) for i in range(201)]
        reference = [(121.0, 24.001), (121.02, 24.001)]

        coarse_score = similarity(compute_deviation(coarse, reference))
        dense_score = similarity(compute_deviation(dense, reference))

        self.assertAlmostEqual(coarse_score["symmetric"]["p95_m"], dense_score["symmetric"]["p95_m"], delta=0.1)
        self.assertEqual(coarse_score["symmetric"]["coverage_f1_within_m"],
                         dense_score["symmetric"]["coverage_f1_within_m"])

    def test_discrete_frechet_distinguishes_ordered_shape_and_buffer_iou_is_exported(self):
        reference = [(121.0, 24.0), (121.005, 24.004), (121.01, 24.0)]
        planned = [(121.0, 24.0), (121.005, 24.006), (121.01, 24.0)]

        score = similarity(compute_deviation(planned, reference))

        self.assertGreater(score["discrete_frechet_approx"]["distance_m"], 100)
        self.assertIn("100", score["buffer_iou"])
        self.assertIn("1000", score["buffer_iou"])
        self.assertLess(score["buffer_iou"]["100"], score["buffer_iou"]["1000"])

    def test_multipart_reference_skips_order_and_endpoint_metrics(self):
        planned = [(121.0, 24.0), (121.01, 24.0)]
        reference = [
            [(121.0, 24.0), (121.005, 24.0)],
            [(121.005, 24.0), (121.01, 24.0)],
        ]

        score = similarity(compute_deviation(planned, reference))

        self.assertIsNone(score["endpoint_offsets"])
        self.assertIsNone(score["discrete_frechet_approx"])

    def test_dateline_projection_and_batched_long_route(self):
        route = [(179.99, 10.0), (-179.99, 10.0)]

        metrics = compute_deviation(route, route)
        score = similarity(metrics)

        self.assertEqual(score["symmetric"]["sampled_hausdorff_m"], 0.0)
        self.assertGreater(score["length"]["reference_km"], 2.0)
        self.assertLess(score["length"]["reference_km"], 2.5)

    def test_long_route_keeps_exact_zero_distance(self):
        route = [(0.0, 0.0), (2.7, 0.0)]

        score = similarity(compute_deviation(route, route))

        self.assertGreater(score["length"]["reference_km"], 250.0)
        self.assertEqual(score["symmetric"]["sampled_hausdorff_m"], 0.0)

    def test_length_difference_is_separate_from_spatial_similarity(self):
        route = [(121.0, 24.0), (121.01, 24.0)]
        dev = compute_deviation(route, route)
        error = eval_error(dev["route_similarity"]["length"]["reference_km"] * 1.2, dev)

        self.assertEqual(error["route_similarity_version"], 2)
        self.assertAlmostEqual(error["route_similarity"]["length"]["absolute_difference_pct"], 20.0)
        self.assertEqual(error["route_similarity"]["symmetric"]["sampled_hausdorff_m"], 0.0)
        self.assertNotIn("pass8", error)
        self.assertNotIn("worst_err_pct", error)

    def test_unrounded_planned_route_length_is_reported(self):
        forward = Transformer.from_crs(
            "+proj=aeqd +lat_0=24 +lon_0=121 +datum=WGS84 +units=m +no_defs",
            "EPSG:4326", always_xy=True)
        route = [forward.transform(x, 4.11 if index % 2 == 0 else -4.11)
                 for index, x in enumerate(range(0, 1001, 20))]
        reference = [forward.transform(0, 0), forward.transform(1000, 0)]
        dev = compute_deviation(route, reference)
        raw_length = path_length_km(route)

        rounded = eval_error(round(raw_length, 3), dev)
        exact = eval_planned_route_error(route, dev)

        self.assertGreater(raw_length, 1.08)
        self.assertNotEqual(rounded["route_similarity"]["length"]["planned_km"],
                            exact["route_similarity"]["length"]["planned_km"])
        self.assertAlmostEqual(exact["route_similarity"]["length"]["planned_km"], raw_length)


if __name__ == "__main__":
    unittest.main()
