import sys
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ai_path_planning_v5 import (
    DEFAULT_GUIDE_MODE,
    _path_hard_violations,
    _reference_guidance_requested,
    _resolve_guidance_weights,
    _select_final_route,
    astar_search,
    geo_to_rowcol,
    rowcol_to_geo,
    smooth_path_with_constraints,
)


class RouteGuidanceTests(unittest.TestCase):
    def test_default_mode_is_reference_independent(self):
        self.assertEqual(DEFAULT_GUIDE_MODE, "none")
        self.assertFalse(_reference_guidance_requested(DEFAULT_GUIDE_MODE, True))

    def test_none_mode_disables_reference_guidance(self):
        self.assertFalse(_reference_guidance_requested("none", True))

    def test_auto_mode_preserves_legacy_guidance_when_reference_exists(self):
        self.assertTrue(_reference_guidance_requested("auto", True))
        self.assertFalse(_reference_guidance_requested("auto", False))

    def test_explicit_zero_weights_are_respected(self):
        self.assertEqual(_resolve_guidance_weights(True, 0.0, 0.0), (0.0, 0.0))

    def test_unguided_selection_ignores_posthoc_reference_errors(self):
        methods = {"VIN-Grad": [1], "A*(baseline)": [2]}
        metrics = {
            "VIN-Grad": {"worst_err_pct": 1.0, "hard_violations": 0},
            "A*(baseline)": {"worst_err_pct": 5.0, "hard_violations": 0},
        }

        method, rule = _select_final_route(methods, metrics, False)

        self.assertEqual((method, rule), ("A*(baseline)", "astar_baseline"))

    def test_guided_selection_uses_reference_error(self):
        methods = {"VIN-Grad": [1], "A*(baseline)": [2]}
        metrics = {
            "VIN-Grad": {"worst_err_pct": 1.0, "hard_violations": 0},
            "A*(baseline)": {"worst_err_pct": 5.0, "hard_violations": 0},
        }

        method, rule = _select_final_route(methods, metrics, True)

        self.assertEqual((method, rule), ("VIN-Grad", "reference_error_min"))

    def test_smoothed_shortcut_falls_back_when_it_crosses_hard_obstacle(self):
        transform = Affine(0.0001, 0, 0, 0, -0.0001, 0.001)
        hard_mask = np.ones((12, 12), dtype=np.uint8)
        hard_mask[5, 5] = 0
        raw_route = [rowcol_to_geo(r, c, transform) for r, c in [(5, 3), (4, 4), (4, 6), (5, 7)]]

        result = smooth_path_with_constraints(raw_route, transform, hard_mask)

        self.assertEqual(result, raw_route)
        self.assertEqual(_path_hard_violations(result, transform, hard_mask), 0)

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
