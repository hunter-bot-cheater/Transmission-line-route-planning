import unittest
from unittest.mock import patch

import numpy as np
from affine import Affine

from transmission_path_planner.geometry import pixel_spacing_meters, rowcol_to_geo
from transmission_path_planner.metrics import path_cost_integral
from transmission_path_planner.smoothing import (
    ground_grade_limit_report,
    path_hard_violations,
    smooth_path_with_constraints,
)


class DirectShortcutTests(unittest.TestCase):
    def setUp(self):
        self.transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        self.shape = (9, 9)
        self.lat_center = self.transform.f + self.transform.e * (self.shape[0] / 2)
        self.spacing = pixel_spacing_meters(
            self.transform, self.lat_center, "EPSG:4326")
        self.hard = np.ones(self.shape, dtype=np.uint8)

    def _route(self, cells):
        return [rowcol_to_geo(row, col, self.transform) for row, col in cells]

    def _objective(self, route, cost, dem, grade_weight=0.0, turn_weight=0.0):
        return path_cost_integral(
            route, dem, self.transform, cost, grade_weight, turn_weight, self.spacing)

    def test_flat_open_grid_shortcut_removes_direction_zigzag_without_cost_increase(self):
        route = self._route([
            (5, 1), (4, 2), (5, 3), (4, 4), (5, 5), (4, 6), (5, 7),
        ])
        direct = [route[0], route[-1]]
        cost = np.ones(self.shape, dtype=np.float32)
        dem = np.zeros(self.shape, dtype=np.float32)

        result = smooth_path_with_constraints(
            route, self.transform, self.hard, cost, dem,
            grade_penalty_weight=1.0, turn_penalty_weight=2.0,
            pixel_spacing=self.spacing)

        self.assertEqual(result, direct)
        raw_objective = self._objective(route, cost, dem, 1.0, 2.0)
        direct_objective = self._objective(direct, cost, dem, 1.0, 2.0)
        self.assertEqual(raw_objective["status"], "measured")
        self.assertEqual(direct_objective["status"], "measured")
        self.assertLessEqual(
            direct_objective["total_cost_meter"], raw_objective["total_cost_meter"])

    def test_guard_uses_restored_endpoints_but_returns_grid_endpoints_for_cli_offsets(self):
        route = self._route([
            (5, 1), (4, 2), (5, 3), (4, 4), (5, 5), (4, 6), (5, 7),
        ])
        actual_start = self.transform * (1.6, 5.4)
        actual_end = self.transform * (7.4, 5.6)
        actual_route = list(route)
        actual_route[0], actual_route[-1] = actual_start, actual_end
        cost = np.ones(self.shape, dtype=np.float32)
        dem = np.zeros(self.shape, dtype=np.float32)

        result = smooth_path_with_constraints(
            route, self.transform, self.hard, cost, dem,
            pixel_spacing=self.spacing,
            shortcut_endpoints=(actual_start, actual_end))

        self.assertEqual(result, [route[0], route[-1]])
        self.assertEqual(result[0], route[0])
        self.assertEqual(result[-1], route[-1])
        self.assertEqual(
            path_hard_violations([actual_start, actual_end], self.transform, self.hard), 0)
        raw_objective = self._objective(actual_route, cost, dem)
        direct_objective = self._objective(
            [actual_start, actual_end], cost, dem)
        self.assertLessEqual(
            direct_objective["total_cost_meter"], raw_objective["total_cost_meter"])

    def test_shortcut_crossing_hard_barrier_falls_back_to_existing_path(self):
        route = self._route([
            (5, 1), (4, 2), (3, 3), (3, 4), (3, 5), (4, 6), (5, 7),
        ])
        self.hard[5, 4] = 0
        cost = np.ones(self.shape, dtype=np.float32)
        dem = np.zeros(self.shape, dtype=np.float32)

        with patch("transmission_path_planner.smoothing.smooth_path",
                   return_value=route):
            result = smooth_path_with_constraints(
                route, self.transform, self.hard, cost, dem,
                pixel_spacing=self.spacing)

        self.assertEqual(result, route)
        self.assertEqual(path_hard_violations(result, self.transform, self.hard), 0)

    def test_shortcut_across_soft_cost_band_is_rejected(self):
        route = self._route([
            (5, 1), (4, 2), (3, 3), (3, 4), (3, 5), (4, 6), (5, 7),
        ])
        direct = [route[0], route[-1]]
        cost = np.ones(self.shape, dtype=np.float32)
        cost[5, 2:7] = 30.0
        dem = np.zeros(self.shape, dtype=np.float32)

        raw_objective = self._objective(route, cost, dem)
        direct_objective = self._objective(direct, cost, dem)
        self.assertGreater(
            direct_objective["total_cost_meter"], raw_objective["total_cost_meter"])

        with patch("transmission_path_planner.smoothing.smooth_path",
                   return_value=route):
            result = smooth_path_with_constraints(
                route, self.transform, self.hard, cost, dem,
                pixel_spacing=self.spacing)

        self.assertEqual(result, route)

    def test_grade_limit_rejects_shortcut_and_keeps_existing_grade_valid_route(self):
        route = self._route([
            (5, 1), (4, 2), (3, 3), (3, 4), (3, 5), (4, 6), (5, 7),
        ])
        direct = [route[0], route[-1]]
        cost = np.ones(self.shape, dtype=np.float32)
        dem = np.zeros(self.shape, dtype=np.float32)
        dem[5, 4] = 100.0
        threshold = 0.3

        self.assertEqual(
            ground_grade_limit_report(
                direct, dem, self.transform, threshold, "EPSG:4326"
            )["grade_limit_status"],
            "fail",
        )
        self.assertEqual(
            ground_grade_limit_report(
                route, dem, self.transform, threshold, "EPSG:4326"
            )["grade_limit_status"],
            "pass",
        )

        with patch("transmission_path_planner.smoothing.smooth_path",
                   return_value=route):
            result = smooth_path_with_constraints(
                route, self.transform, self.hard, cost, dem,
                pixel_spacing=self.spacing, max_ground_grade=threshold,
                crs="EPSG:4326")

        self.assertEqual(result, route)

    def test_invalid_smoothed_candidate_falls_back_to_grade_valid_raw_route(self):
        transform = Affine(0.0001, 0, 120.0, 0, -0.0001, 25.0)
        dem = np.zeros((12, 12), dtype=np.float64)
        dem[5, 5] = 100.0
        hard = np.ones(dem.shape, dtype=np.uint8)
        cost = np.ones(dem.shape, dtype=np.float32)
        cost[4, :] = 20.0
        cells = [(5, 0), (4, 3), (4, 7), (5, 10)]
        raw = [rowcol_to_geo(row, col, transform) for row, col in cells]
        spacing = pixel_spacing_meters(
            transform, transform.f + transform.e * (dem.shape[0] / 2),
            "EPSG:4326")

        self.assertEqual(
            ground_grade_limit_report(
                raw, dem, transform, 0.3, "EPSG:4326")["grade_limit_status"],
            "pass")
        result = smooth_path_with_constraints(
            raw, transform, hard, cost, dem, pixel_spacing=spacing,
            max_ground_grade=0.3, crs="EPSG:4326")

        self.assertEqual(result, raw)
        self.assertEqual(
            ground_grade_limit_report(
                result, dem, transform, 0.3, "EPSG:4326")["grade_limit_status"],
            "pass")

    def test_rejects_smoothed_and_raw_routes_when_both_fail_grade_limit(self):
        transform = Affine(0.0001, 0, 120.0, 0, -0.0001, 25.0)
        dem = np.zeros((12, 12), dtype=np.float64)
        dem[5, 5] = 100.0
        hard = np.ones(dem.shape, dtype=np.uint8)
        cost = np.ones(dem.shape, dtype=np.float32)
        raw = [rowcol_to_geo(5, 0, transform), rowcol_to_geo(5, 10, transform)]

        result = smooth_path_with_constraints(
            raw, transform, hard, cost, dem, max_ground_grade=0.3,
            crs="EPSG:4326")

        self.assertIsNone(result)

    def test_shortcut_must_pass_grade_check_before_and_after_endpoint_restoration(self):
        route = self._route([
            (5, 1), (4, 2), (5, 3), (4, 4), (5, 5), (4, 6), (5, 7),
        ])
        actual_start = self.transform * (1.6, 5.4)
        actual_end = self.transform * (7.4, 5.6)
        actual_direct = [actual_start, actual_end]
        cost = np.ones(self.shape, dtype=np.float32)
        dem = np.zeros(self.shape, dtype=np.float32)

        def grade_report(candidate, *_args, **_kwargs):
            status = "pass" if candidate in (actual_direct, route) else "fail"
            return {"grade_limit_status": status}

        with patch("transmission_path_planner.smoothing.ground_grade_limit_report",
                   side_effect=grade_report), patch(
                       "transmission_path_planner.smoothing.smooth_path",
                       return_value=route):
            result = smooth_path_with_constraints(
                route, self.transform, self.hard, cost, dem,
                pixel_spacing=self.spacing, max_ground_grade=0.3,
                crs="EPSG:4326", shortcut_endpoints=(actual_start, actual_end))

        # The restored line passes, but the temporary center-to-center path
        # would fail the CLI's pre-restoration grade check; retain the old route.
        self.assertEqual(result, route)


if __name__ == "__main__":
    unittest.main()
