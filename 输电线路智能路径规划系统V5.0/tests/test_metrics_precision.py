import unittest
from unittest.mock import patch

import numpy as np
from pyproj import Transformer

from transmission_path_planner.geometry import geodesic_path_length_km
from transmission_path_planner.metrics import (
    _distance_summary,
    _line_samples,
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
        self.assertEqual(metrics["route_similarity_version"], 7)

    def test_custom_sample_interval_is_reported_and_validated(self):
        planned = [(120.0, 24.0), (120.02, 24.0)]
        reference = [(120.0, 24.001), (120.02, 24.001)]

        default = similarity(compute_deviation(planned, reference))
        fine = similarity(compute_deviation(
            planned, reference, sample_interval_m=5))

        self.assertEqual(default["sampling_interval_m"], 20.0)
        self.assertEqual(fine["sampling_interval_m"], 5.0)
        for direction in ("planned_to_reference", "reference_to_planned"):
            self.assertAlmostEqual(
                fine[direction]["p95_m"], default[direction]["p95_m"], delta=0.1)
            self.assertEqual(
                fine[direction]["coverage_within_m"],
                default[direction]["coverage_within_m"],
            )

        with patch("transmission_path_planner.metrics._line_samples",
                   wraps=_line_samples) as sample_calls:
            compute_deviation(planned, reference, sample_interval_m=5.0)
        self.assertEqual([call.args[1] for call in sample_calls.call_args_list],
                         [5.0, 5.0])

        for invalid in (True, 0, 0.5, -1, 1001, 10 ** 1000,
                        float("nan"), float("inf"), "invalid"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "sample_interval_m"):
                compute_deviation(planned, reference, sample_interval_m=invalid)

    def test_routes_share_projection_when_scored_against_same_planning_window(self):
        reference = [(120.0, 24.0), (120.02, 24.0)]
        planning_bounds = (119.9, 23.9, 120.1, 24.1)
        first = [(120.0, 24.001), (120.02, 24.001)]
        second = [(119.99, 24.0), (120.01, 24.001), (120.03, 24.0)]

        first_score = similarity(compute_deviation(
            first, reference, planning_bounds_wgs84=planning_bounds))
        second_score = similarity(compute_deviation(
            second, reference, planning_bounds_wgs84=planning_bounds))

        self.assertEqual(first_score["comparison_crs"], second_score["comparison_crs"])

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

    def test_optional_chain_profile_localizes_corridor_departure_and_aligns_reversed_reference(self):
        planned = [(120.0, 24.0), (120.009, 24.0)]
        reference = [
            (120.0, 24.0), (120.0045, 24.0),
            (120.0045, 24.002), (120.009, 24.002),
        ]
        bounds = (119.99, 23.99, 120.02, 24.02)

        baseline = similarity(compute_deviation(planned, reference, bounds))
        profiled = similarity(compute_deviation(
            planned, reference, bounds, chain_profile_bins=10,
        ))
        reversed_profiled = similarity(compute_deviation(
            planned, list(reversed(reference)), bounds, chain_profile_bins=10,
        ))

        self.assertNotIn("chain_error_profile", baseline)
        profile = profiled["chain_error_profile"]
        self.assertEqual(profile["version"], 1)
        self.assertEqual(profile["bin_count"], 10)
        self.assertEqual(len(profile["directions"]["planned_to_reference"]), 10)
        planned_bins = profile["directions"]["planned_to_reference"]
        self.assertLess(planned_bins[0]["p95_m"], 1.0)
        self.assertGreater(planned_bins[-1]["p95_m"], 100.0)
        self.assertAlmostEqual(
            sum(item["sampled_length_m"] for item in planned_bins),
            profiled["length"]["planned_km"] * 1000,
            delta=1.0,
        )
        self.assertAlmostEqual(
            profiled["symmetric"]["p95_m"], baseline["symmetric"]["p95_m"],
            delta=1e-9,
        )
        forward_bins = profile["directions"]["reference_to_planned"]
        reverse_bins = reversed_profiled["chain_error_profile"]["directions"][
            "reference_to_planned"
        ]
        for forward, reverse in zip(forward_bins, reverse_bins):
            self.assertAlmostEqual(forward["p95_m"], reverse["p95_m"], delta=1e-6)

    def test_chain_profile_is_not_inferred_for_disconnected_reference_lines(self):
        planned = [(121.0, 24.0), (121.01, 24.0)]
        reference = [
            [(121.0, 24.0), (121.004, 24.0)],
            [(121.006, 24.0), (121.01, 24.0)],
        ]

        score = similarity(compute_deviation(planned, reference, chain_profile_bins=10))

        self.assertIsNone(score["chain_error_profile"])

    def test_chain_profile_bin_count_must_be_positive_and_bounded(self):
        route = [(121.0, 24.0), (121.01, 24.0)]
        for invalid in (True, 0, -1, 101, 10.0):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "chain_profile_bins"):
                compute_deviation(route, route, chain_profile_bins=invalid)

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

        self.assertGreater(score["planned_to_reference"]["mean_m"], 100)
        self.assertGreater(score["reference_to_planned"]["sampled_max_m"], 100)

    def test_reversed_reference_keeps_endpoint_and_frechet_similarity(self):
        route = [(121.0, 24.0), (121.005, 24.002), (121.01, 24.0)]

        score = similarity(compute_deviation(route, list(reversed(route))))

        self.assertLess(score["endpoint_offsets"]["planned_start_to_reference_endpoint_m"], 0.01)
        self.assertLess(score["endpoint_offsets"]["planned_end_to_reference_endpoint_m"], 0.01)
        self.assertEqual(score["endpoint_offsets"]["orientation"], "reversed")
        self.assertLess(score["discrete_frechet_approx"]["distance_m"], 0.01)
        frechet = score["discrete_frechet_approx"]
        self.assertEqual(frechet["target_max_sample_interval_m"], 100.0)
        self.assertLessEqual(frechet["planned_max_sample_interval_m"], 100.0)
        self.assertLessEqual(frechet["reference_max_sample_interval_m"], 100.0)
        self.assertGreater(frechet["planned_sample_points"], 2)

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

    def test_connected_reference_fragments_merge_for_order_and_endpoint_metrics(self):
        planned = [(121.0, 24.0), (121.005, 24.004), (121.01, 24.0)]
        reference = [
            planned[1:],
            planned[:2],
        ]

        score = similarity(compute_deviation(planned, reference))
        single_line_score = similarity(compute_deviation(planned, planned))

        self.assertIsNotNone(score["endpoint_offsets"])
        self.assertLess(score["discrete_frechet_approx"]["distance_m"], 0.01)
        self.assertAlmostEqual(score["length"]["reference_km"],
                               single_line_score["length"]["reference_km"], delta=0.001)
        for direction in ("planned_to_reference", "reference_to_planned"):
            for field in ("sampled_length_m", "mean_m", "p95_m", "sampled_max_m"):
                self.assertAlmostEqual(score[direction][field],
                                       single_line_score[direction][field], delta=0.001)
        self.assertEqual(score["symmetric"]["coverage_f1_within_m"],
                         single_line_score["symmetric"]["coverage_f1_within_m"])
        self.assertEqual(score["endpoint_offsets"]["orientation"],
                         single_line_score["endpoint_offsets"]["orientation"])
        self.assertAlmostEqual(score["discrete_frechet_approx"]["distance_m"],
                               single_line_score["discrete_frechet_approx"]["distance_m"],
                               delta=0.001)

    def test_disconnected_reference_components_skip_order_and_endpoint_metrics(self):
        planned = [(121.0, 24.0), (121.01, 24.0)]
        reference = [
            [(121.0, 24.0), (121.004, 24.0)],
            [(121.006, 24.0), (121.01, 24.0)],
        ]

        score = similarity(compute_deviation(planned, reference))

        self.assertIsNone(score["endpoint_offsets"])
        self.assertIsNone(score["discrete_frechet_approx"])

    def test_closed_reference_line_skips_start_dependent_order_metrics(self):
        ring = [(121.0, 24.0), (121.01, 24.0), (121.01, 24.01),
                (121.0, 24.0)]

        score = similarity(compute_deviation(ring, ring))

        self.assertIsNone(score["endpoint_offsets"])
        self.assertIsNone(score["discrete_frechet_approx"])

    def test_multipart_metrics_weight_components_by_arc_length_not_fragment_count(self):
        from shapely.geometry import LineString

        bad_samples, bad_weights, _ = _line_samples(
            [np.asarray([[0.0, 500.0], [10_000.0, 500.0]])], 20.0)
        fragmented_good = [np.asarray(LineString([(x, 0), (x + 10, 0)]).coords)
                           for x in range(10_000, 20_000, 10)]
        good_samples, good_weights, _ = _line_samples(fragmented_good, 20.0)
        self.assertEqual(len(bad_samples), 500)
        self.assertEqual(len(good_samples), 1_000)
        self.assertAlmostEqual(bad_weights.sum(), good_weights.sum(), delta=1e-6)

        reference_distances = np.concatenate((np.full(len(bad_samples), 500.0),
                                              np.zeros(len(good_samples))))
        reference_weights = np.concatenate((bad_weights, good_weights))
        reference = _distance_summary(reference_distances, reference_weights)
        planned = _distance_summary(np.zeros(1_000), np.full(1_000, 20.0))
        f1 = 2 * planned["coverage_within_m"]["100"] * reference["coverage_within_m"]["100"] \
            / (planned["coverage_within_m"]["100"] + reference["coverage_within_m"]["100"])

        self.assertAlmostEqual(reference["coverage_within_m"]["100"], 0.5, delta=0.002)
        self.assertAlmostEqual(f1, 2 / 3, delta=0.003)
        self.assertGreater(reference["p95_m"], 450)

    def test_public_metric_weights_fragmented_reference_by_arc_length(self):
        from pyproj import Transformer
        from shapely.geometry import LineString, MultiLineString
        from shapely.ops import transform

        local = "+proj=aeqd +lat_0=24 +lon_0=121 +datum=WGS84 +units=m +no_defs"
        to_wgs = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
        planned_line = transform(
            to_wgs.transform, LineString([(2000, 0), (4000, 0)]))
        reference_line = transform(to_wgs.transform, MultiLineString(
            [[(0, 500), (2000, 500)]]
            + [[(2000 + 10 * i, 0), (2010 + 10 * i, 0)] for i in range(200)]))
        planned = list(planned_line.coords)
        reference = [list(part.coords) for part in reference_line.geoms]

        score = similarity(compute_deviation(planned, reference))

        self.assertAlmostEqual(
            score["planned_to_reference"]["coverage_within_m"]["100"], 1.0, delta=0.002)
        self.assertAlmostEqual(
            score["reference_to_planned"]["coverage_within_m"]["100"], 0.5, delta=0.002)
        self.assertAlmostEqual(score["symmetric"]["coverage_f1_within_m"]["100"],
                               2 / 3, delta=0.003)
        self.assertGreater(score["reference_to_planned"]["p95_m"], 100)
        self.assertAlmostEqual(score["reference_to_planned"]["sampled_length_m"],
                               4000, delta=0.1)

    def test_symmetric_mean_averages_directional_means_for_unequal_lengths(self):
        from pyproj import Transformer
        from shapely.geometry import LineString
        from shapely.ops import transform

        local = "+proj=aeqd +lat_0=24 +lon_0=121 +datum=WGS84 +units=m +no_defs"
        to_wgs = Transformer.from_crs(local, "EPSG:4326", always_xy=True)
        planned = list(transform(
            to_wgs.transform, LineString([(0, 0), (10_000, 0)])).coords)
        reference = list(transform(
            to_wgs.transform, LineString([(0, 0), (20_000, 0)])).coords)

        score = similarity(compute_deviation(planned, reference))

        self.assertAlmostEqual(score["planned_to_reference"]["mean_m"], 0, delta=0.01)
        self.assertAlmostEqual(score["reference_to_planned"]["mean_m"], 2500, delta=0.01)
        self.assertAlmostEqual(score["symmetric"]["mean_m"], 1250, delta=0.01)

    def test_duplicate_reference_components_do_not_reweight_spatial_similarity(self):
        route = [(121.0, 24.0), (121.01, 24.0)]
        duplicate = [route, route]

        single = similarity(compute_deviation(route, route))
        repeated = similarity(compute_deviation(route, duplicate))

        self.assertAlmostEqual(repeated["length"]["reference_km"],
                               single["length"]["reference_km"], delta=0.001)
        self.assertEqual(repeated["symmetric"]["coverage_f1_within_m"],
                         single["symmetric"]["coverage_f1_within_m"])
        self.assertAlmostEqual(repeated["symmetric"]["mean_m"],
                               single["symmetric"]["mean_m"], delta=0.001)
        self.assertIsNotNone(repeated["endpoint_offsets"])
        self.assertAlmostEqual(
            repeated["discrete_frechet_approx"]["distance_m"],
            single["discrete_frechet_approx"]["distance_m"], delta=0.001)

    def test_partial_overlap_is_dissolved_before_projection_and_length_weighting(self):
        first = [(121.0, 24.0), (121.1, 24.0)]
        second = [(121.05, 24.0), (121.15, 24.0)]
        unique_support = [(121.0, 24.0), (121.15, 24.0)]

        score = similarity(compute_deviation(unique_support, [first, second]))

        self.assertAlmostEqual(score["length"]["reference_km"],
                               score["length"]["planned_km"], delta=0.01)
        self.assertAlmostEqual(score["length"]["absolute_difference_pct"], 0, delta=0.1)
        self.assertGreater(score["symmetric"]["coverage_f1_within_m"]["100"], 0.999)
        self.assertLess(score["symmetric"]["mean_m"], 5)

    def test_partial_overlap_across_dateline_is_dissolved_on_one_longitude_branch(self):
        planned = [(179.9, 10.0), (-179.9, 10.0)]
        reference = [
            [(179.9, 10.0), (-179.95, 10.0)],
            [(180.0, 10.0), (-179.9, 10.0)],
        ]

        score = similarity(compute_deviation(planned, reference))

        self.assertAlmostEqual(score["length"]["reference_km"],
                               score["length"]["planned_km"], delta=0.01)
        self.assertAlmostEqual(score["symmetric"]["sampled_hausdorff_m"], 0.0, delta=0.01)

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

        self.assertEqual(error["route_similarity_version"], 7)
        self.assertAlmostEqual(error["route_similarity"]["length"]["absolute_difference_pct"], 20.0)
        self.assertEqual(error["route_similarity"]["symmetric"]["sampled_hausdorff_m"], 0.0)
        self.assertNotIn("pass8", error)
        self.assertNotIn("worst_err_pct", error)

    def test_route_similarity_lengths_use_wgs84_geodesic(self):
        route = [(121.0, 24.0), (121.1, 24.1)]
        expected = geodesic_path_length_km(route)

        metrics = compute_deviation(route, route)
        length = similarity(metrics)["length"]

        self.assertEqual(metrics["route_similarity_version"], 7)
        self.assertAlmostEqual(length["planned_km"], expected, places=9)
        self.assertAlmostEqual(length["reference_km"], expected, places=9)
        self.assertEqual(
            length["measurement_method"],
            "WGS84 ellipsoidal geodesic (pyproj.Geod)",
        )

    def test_frechet_failure_keeps_other_route_metrics(self):
        route = [(121.0, 24.0), (121.01, 24.0)]

        with patch(
                "transmission_path_planner.metrics._discrete_frechet",
                side_effect=MemoryError("test allocation failure")):
            metrics = compute_deviation(route, route)

        score = similarity(metrics)
        self.assertEqual(score["symmetric"]["sampled_hausdorff_m"], 0.0)
        self.assertIsNone(score["discrete_frechet_approx"])

    def test_unrounded_planned_route_length_is_reported(self):
        forward = Transformer.from_crs(
            "+proj=aeqd +lat_0=24 +lon_0=121 +datum=WGS84 +units=m +no_defs",
            "EPSG:4326", always_xy=True)
        route = [forward.transform(x, 4.11 if index % 2 == 0 else -4.11)
                 for index, x in enumerate(range(0, 1001, 20))]
        reference = [forward.transform(0, 0), forward.transform(1000, 0)]
        dev = compute_deviation(route, reference)
        raw_length = geodesic_path_length_km(route)

        rounded = eval_error(round(raw_length, 3), dev)
        exact = eval_planned_route_error(route, dev)

        self.assertGreater(raw_length, 1.08)
        self.assertNotEqual(rounded["route_similarity"]["length"]["planned_km"],
                            exact["route_similarity"]["length"]["planned_km"])
        self.assertAlmostEqual(exact["route_similarity"]["length"]["planned_km"], raw_length)


if __name__ == "__main__":
    unittest.main()
