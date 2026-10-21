import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.cli import (
    DEFAULT_GUIDE_MODE,
    _reference_guidance_requested,
    _select_final_route,
    geojson_coordinates,
)
from transmission_path_planner.geometry import geo_to_rowcol, rowcol_to_geo
from transmission_path_planner.methods import astar_search
from transmission_path_planner.smoothing import path_hard_violations, smooth_path_with_constraints


class GuidanceModeTests(unittest.TestCase):
    def test_default_mode_is_reference_independent(self):
        self.assertEqual(DEFAULT_GUIDE_MODE, "none")
        self.assertFalse(_reference_guidance_requested(DEFAULT_GUIDE_MODE, use_real=True))

    def test_none_mode_never_requests_reference_guidance(self):
        self.assertFalse(_reference_guidance_requested("none", use_real=True))

    def test_auto_mode_keeps_legacy_behavior(self):
        self.assertTrue(_reference_guidance_requested("auto", use_real=True))
        self.assertFalse(_reference_guidance_requested("auto", use_real=False))

    def test_reference_mode_is_explicit(self):
        self.assertTrue(_reference_guidance_requested("reference", use_real=False))

    def test_unguided_selection_ignores_reference_errors(self):
        methods = {"VIN-Grad": [(1, 1)], "A*(baseline)": [(1, 1)]}
        metrics = {
            "VIN-Grad": {"worst_err_pct": 1.0, "hard_violations": 0},
            "A*(baseline)": {"worst_err_pct": 9.0, "hard_violations": 0},
        }
        self.assertEqual(
            _select_final_route(methods, metrics, guide_enabled=False),
            ("A*(baseline)", None, "astar_baseline"),
        )

    def test_reference_mode_selects_lowest_reference_error(self):
        methods = {"VIN-Grad": [(1, 1)], "A*(baseline)": [(1, 1)]}
        metrics = {
            "VIN-Grad": {"worst_err_pct": 1.0, "hard_violations": 0},
            "A*(baseline)": {"worst_err_pct": 9.0, "hard_violations": 0},
        }
        self.assertEqual(
            _select_final_route(methods, metrics, guide_enabled=True),
            ("VIN-Grad", 1.0, "reference_error_min"),
        )

    def test_reference_mode_falls_back_when_no_score_exists(self):
        methods = {"VIN-Grad": [(1, 1)], "A*(baseline)": [(1, 1)]}
        metrics = {
            "VIN-Grad": {"hard_violations": 0},
            "A*(baseline)": {"worst_err_pct": None, "hard_violations": 0},
        }
        self.assertEqual(
            _select_final_route(methods, metrics, guide_enabled=True),
            ("A*(baseline)", None, "astar_fallback_no_reference_score"),
        )

    def test_geojson_coordinates_keep_longitude_before_latitude(self):
        self.assertEqual(geojson_coordinates([(120.75, 24.32)]), [[120.75, 24.32]])

    def test_smoothed_shortcut_falls_back_around_hard_obstacle(self):
        transform = Affine(0.0001, 0, 0, 0, -0.0001, 0.001)
        hard_mask = np.ones((12, 12), dtype=np.uint8)
        hard_mask[5, 5] = 0
        raw_route = [rowcol_to_geo(r, c, transform) for r, c in [(5, 3), (4, 4), (4, 6), (5, 7)]]

        result = smooth_path_with_constraints(raw_route, transform, hard_mask)

        self.assertEqual(result, raw_route)
        self.assertEqual(path_hard_violations(result, transform, hard_mask), 0)

    def test_geo_coordinate_round_trips_to_its_pixel(self):
        transform = Affine(0.0001, 0, 120.0, 0, -0.0001, 25.0)
        for row in range(6):
            for col in range(6):
                lon, lat = rowcol_to_geo(row, col, transform)
                self.assertEqual(geo_to_rowcol(lat, lon, transform), (row, col))

    def test_astar_cost_matches_dijkstra_on_variable_cost_grid(self):
        cost = np.full((18, 18), 8.0, dtype=np.float64)
        cost[2:16, 2:16] = 1.0
        cost[8, 2:16] = 0.01
        cost[8, 9] = 50.0
        cost[3:7, 6] = np.inf
        start, goal = (1, 1), (16, 16)
        path = astar_search(cost, start, goal)
        reference = astar_search(cost, start, goal, use_dijkstra=True)

        def path_cost(route):
            return sum(cost[r, c] * (np.sqrt(2) if r != pr and c != pc else 1.0)
                       for (pr, pc), (r, c) in zip(route, route[1:]))

        self.assertTrue(path)
        self.assertAlmostEqual(path_cost(path), path_cost(reference), places=9)


if __name__ == "__main__":
    unittest.main()
