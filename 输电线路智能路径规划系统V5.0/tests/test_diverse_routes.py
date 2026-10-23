import unittest
from unittest.mock import patch

import numpy as np

from transmission_path_planner.diverse_routes import (
    _interior_line,
    _distance_to_mask,
    _separation_metrics,
    diverse_astar_routes,
)
from transmission_path_planner.methods import astar_search
from transmission_path_planner.metrics import astar_path_objective


class DiverseRouteTests(unittest.TestCase):
    def test_segment_distance_detects_diagonal_crossing(self):
        result = _separation_metrics(
            [(1, 1), (2, 2)], [[(1, 2), (2, 1)]],
            (35.0, 35.0), 0.0, 5.0)

        self.assertEqual(result["min_separation_m"], 0.0)
        self.assertEqual(result["symmetric_separated_fraction"], 1.0)

    def test_separation_respects_anisotropic_pixel_spacing(self):
        result = _separation_metrics(
            [(1, 0), (1, 1)], [[(3, 0), (3, 1)]],
            (35.0, 70.0), 0.0, 60.0)

        self.assertEqual(result["min_separation_m"], 70.0)
        self.assertEqual(result["symmetric_separated_fraction"], 1.0)

    def test_long_shared_segment_fails_separated_fraction(self):
        selected = [(1, col) for col in range(11)]
        candidate = ([(1, col) for col in range(8)]
                     + [(2, 8), (2, 9), (1, 10)])

        result = _separation_metrics(
            candidate, [selected], (1.0, 1.0), 0.0, 1.5)

        self.assertLess(result["candidate_separated_fraction"], 0.7)
        self.assertLess(result["selected_separated_fraction"], 0.7)

    def test_endpoint_buffer_can_leave_no_comparable_interior(self):
        self.assertIsNone(_interior_line([(0, 0), (0, 1)], (35.0, 35.0), 35.0))

    def test_same_cell_endpoints_return_one_route_without_geometry_error(self):
        routes = diverse_astar_routes(
            np.ones((3, 3)), np.ones((3, 3), dtype=bool),
            (1, 1), (1, 1), pixel_spacing=(35.0, 35.0))

        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["path"], [(1, 1)])

    def test_rejects_noninteger_grid_indices(self):
        with self.assertRaisesRegex(ValueError, r"integer \(row, col\)"):
            diverse_astar_routes(
                np.ones((5, 5)), np.ones((5, 5), dtype=bool),
                (1.5, 0), (4, 4), pixel_spacing=(35.0, 35.0))

    def test_diagonal_move_cannot_squeeze_between_two_hard_cells(self):
        hard = np.ones((2, 2), dtype=bool)
        hard[0, 1] = hard[1, 0] = False
        result = diverse_astar_routes(
            np.ones((2, 2)), hard, (0, 0), (1, 1),
            pixel_spacing=(35.0, 35.0))

        self.assertEqual(result, [])

    def test_finds_two_separated_corridors_and_reports_original_objective(self):
        cost = np.full((21, 31), 30.0)
        hard = np.ones(cost.shape, dtype=bool)
        for row in (4, 16):
            cost[row, :] = 0.2
            cost[min(row, 10):max(row, 10) + 1, 0] = 0.2
            cost[min(row, 10):max(row, 10) + 1, -1] = 0.2
        cost[10, 0] = cost[10, -1] = 0.2
        elevation = np.fromfunction(
            lambda row, col: 0.1 * row + 0.05 * col, cost.shape, dtype=float)

        routes = diverse_astar_routes(
            cost, hard, (10, 0), (10, 30), max_paths=2,
            pixel_spacing=(1, 1), penalty_radius_m=4,
            distance_penalty_weight=40, minimum_separation_m=5,
            endpoint_buffer_m=2, elevation=elevation,
            grade_penalty_weight=0.6, turn_penalty_weight=0.4,
        )

        self.assertEqual(len(routes), 2)
        self.assertTrue(all(route["hard_constraint_legal"] for route in routes))
        self.assertLess(routes[0]["baseline_objective"], 100)
        self.assertGreater(routes[1]["mean_separation_from_selected_m"], 5)
        self.assertGreater(routes[1]["search_distance_penalty_weight"], 0)
        for route in routes:
            expected_baseline = astar_path_objective(
                route["path"], cost, elevation, (1.0, 1.0),
                grade_penalty_weight=0.6, turn_penalty_weight=0.4)["total_cost"]
            self.assertEqual(route["baseline_objective_model"],
                             "production_astar_unperturbed_cost_grade_turn")
            self.assertAlmostEqual(
                route["baseline_objective"], expected_baseline, places=6)
        self.assertAlmostEqual(routes[0]["search_objective"],
                               routes[0]["baseline_objective"], places=6)
        self.assertEqual(routes[1]["search_objective_unit"], "cost_meter")
        self.assertGreater(routes[1]["search_objective"],
                           routes[1]["baseline_objective"])
        second = routes[1]
        selected_mask = np.zeros(cost.shape, dtype=bool)
        for row, col in routes[0]["path"]:
            selected_mask[row, col] = True
        distance = _distance_to_mask(selected_mask, (1.0, 1.0))
        proximity = np.clip(1.0 - distance / 4.0, 0.0, 1.0)
        perturbed_cost = cost + second["search_distance_penalty_weight"] * proximity
        expected_search = astar_path_objective(
            second["path"], perturbed_cost, elevation, (1.0, 1.0),
            grade_penalty_weight=0.6, turn_penalty_weight=0.4)["total_cost"]
        self.assertEqual(second["search_objective_model"],
                         "production_astar_distance_perturbed_cost_grade_turn")
        self.assertAlmostEqual(second["search_objective"], expected_search, places=6)

    def test_candidates_do_not_cross_hard_barrier(self):
        cost = np.ones((11, 17), dtype=float)
        hard = np.ones(cost.shape, dtype=bool)
        hard[:, 8] = False
        hard[5, 8] = True
        routes = diverse_astar_routes(
            cost, hard, (5, 1), (5, 15), max_paths=3,
            pixel_spacing=(1, 1), penalty_radius_m=3,
            distance_penalty_weight=20, minimum_separation_m=2,
            endpoint_buffer_m=1,
        )

        self.assertTrue(routes)
        for route in routes:
            self.assertTrue(route["hard_constraint_legal"])
            self.assertEqual(route["hard_violation_count"], 0)
            self.assertTrue(all(hard[point] for point in route["path"]))
            self.assertIn((5, 8), route["path"])

    def test_returns_fewer_than_k_when_only_one_corridor_exists(self):
        cost = np.full((9, 15), np.inf)
        hard = np.zeros(cost.shape, dtype=bool)
        cost[4, :] = 1.0
        hard[4, :] = True

        routes = diverse_astar_routes(
            cost, hard, (4, 0), (4, 14), max_paths=4,
            pixel_spacing=(1, 1), penalty_radius_m=3,
            distance_penalty_weight=10, minimum_separation_m=2,
        )

        self.assertEqual(len(routes), 1)
        self.assertTrue(routes[0]["hard_constraint_legal"])

    def test_first_route_matches_production_grade_and_turn_aware_astar(self):
        cost = np.full((13, 19), 3.0)
        cost[6, :] = 0.1
        hard = np.ones(cost.shape, dtype=bool)
        hard[6, 6] = False
        elevation = np.fromfunction(
            lambda row, col: row * 0.1 + col * 0.1, cost.shape, dtype=float)
        elevation[6, 9] += 10.0
        kwargs = {
            "elevation": elevation,
            "pixel_m": 45.0,
            "grade_penalty_weight": 0.0,
            "max_ground_grade": 0.05,
            "pixel_spacing": (30.0, 60.0),
            "heuristic_weight": 1.0,
            "turn_penalty_weight": 0.35,
        }

        effective_cost = np.where(hard, cost, np.inf)
        expected = astar_search(effective_cost, (6, 1), (6, 17), **kwargs)
        unconstrained = astar_search(
            cost, (6, 1), (6, 17), elevation=elevation,
            grade_penalty_weight=0.0, pixel_spacing=(30.0, 60.0),
            heuristic_weight=1.0, turn_penalty_weight=0.35)
        with patch("transmission_path_planner.diverse_routes.astar_search",
                   wraps=astar_search) as search:
            routes = diverse_astar_routes(
                cost, hard, (6, 1), (6, 17), max_paths=1, **kwargs,
            )

        search.assert_called_once()
        call_args, call_kwargs = search.call_args
        np.testing.assert_array_equal(call_args[0], effective_cost)
        self.assertEqual(call_args[1:], ((6, 1), (6, 17)))
        for name, value in kwargs.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(call_kwargs[name], value)
            else:
                self.assertEqual(call_kwargs[name], value)
        self.assertEqual(routes[0]["path"], expected)
        self.assertIn((6, 6), unconstrained)
        self.assertNotIn((6, 6), expected)
        self.assertIn((6, 9), unconstrained)
        self.assertNotIn((6, 9), expected)
        self.assertEqual(routes[0]["baseline_objective_model"],
                         "production_astar_unperturbed_cost_grade_turn")
        self.assertEqual(routes[0]["search_objective_model"],
                         "production_astar_unperturbed_cost_grade_turn")
        objective = astar_path_objective(
            expected, cost, elevation, kwargs["pixel_spacing"],
            kwargs["grade_penalty_weight"], kwargs["turn_penalty_weight"])
        self.assertAlmostEqual(routes[0]["baseline_objective"],
                               objective["total_cost"], places=6)
        self.assertAlmostEqual(routes[0]["search_objective"],
                               objective["total_cost"], places=6)
        self.assertEqual(routes[0]["search_objective_unit"], "cost_meter")
        for (r0, c0), (r1, c1) in zip(routes[0]["path"], routes[0]["path"][1:]):
            distance = np.hypot((r1 - r0) * 30.0, (c1 - c0) * 60.0)
            grade = abs(elevation[r1, c1] - elevation[r0, c0]) / distance
            self.assertLessEqual(grade, kwargs["max_ground_grade"])


if __name__ == "__main__":
    unittest.main()
