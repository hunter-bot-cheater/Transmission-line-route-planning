import math
import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.experimental_theta_star import (
    integrate_polyline_cost,
    theta_star_cost_path,
)
from transmission_path_planner.metrics import path_cost_integral
from transmission_path_planner.methods import astar_search
from transmission_path_planner.smoothing import path_hard_violations


def _route_geometry(path):
    return [(col + 0.5, row + 0.5) for row, col in path]


class ExperimentalThetaStarTests(unittest.TestCase):
    def test_open_uniform_grid_returns_direct_any_angle_and_beats_grid_astar_length(self):
        cost = np.ones((30, 30), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        spacing = (2.0, 3.0)
        start, goal = (3, 2), (23, 15)

        route = theta_star_cost_path(
            cost, traversable, start, goal, pixel_spacing=spacing)
        astar_route = astar_search(
            cost, start, goal, pixel_spacing=spacing, heuristic_weight=1.0)

        self.assertEqual(route, [start, goal])
        theta_cost = integrate_polyline_cost(
            route, cost, traversable, pixel_spacing=spacing)
        astar_cost = integrate_polyline_cost(
            astar_route, cost, traversable, pixel_spacing=spacing)
        expected = math.hypot(
            (goal[0] - start[0]) * spacing[0],
            (goal[1] - start[1]) * spacing[1])
        self.assertAlmostEqual(theta_cost, expected, places=9)
        self.assertLess(theta_cost, astar_cost)

    def test_segment_integral_matches_production_piecewise_constant_integral(self):
        cost = np.array([
            [1.0, 2.0, 5.0, 4.0],
            [3.0, 7.0, 2.0, 6.0],
            [2.0, 4.0, 8.0, 3.0],
            [5.0, 1.0, 2.0, 9.0],
        ])
        traversable = np.ones(cost.shape, dtype=bool)
        route = [(0, 0), (2, 3), (3, 3)]
        spacing = (2.5, 4.0)
        exact = integrate_polyline_cost(
            route, cost, traversable, pixel_spacing=spacing)
        production = path_cost_integral(
            _route_geometry(route), np.zeros(cost.shape), Affine.identity(),
            cost, grade_penalty_weight=0.0, turn_penalty_weight=0.0,
            pixel_spacing=spacing)

        self.assertEqual(production["status"], "measured")
        self.assertAlmostEqual(exact, production["base_cost_meter"], places=6)

    def test_hard_obstacle_is_not_crossed_and_route_exists_around_it(self):
        cost = np.ones((15, 21), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        traversable[3:12, 8:13] = False
        start, goal = (7, 1), (7, 19)

        route = theta_star_cost_path(cost, traversable, start, goal)

        self.assertIsNotNone(route)
        self.assertGreater(len(route), 2)
        self.assertEqual(
            path_hard_violations(
                _route_geometry(route), Affine.identity(), traversable.astype(np.uint8)),
            0,
        )
        self.assertTrue(math.isfinite(integrate_polyline_cost(route, cost, traversable)))

    def test_soft_high_cost_region_deflects_route_and_beats_direct_chord(self):
        cost = np.ones((25, 31), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        cost[8:17, 10:21] = 80.0
        start, goal = (12, 1), (12, 29)
        direct = [start, goal]

        route = theta_star_cost_path(cost, traversable, start, goal)
        astar_route = astar_search(
            cost, start, goal, pixel_spacing=(1.0, 1.0), heuristic_weight=1.0)
        route_cost = integrate_polyline_cost(route, cost, traversable)
        direct_cost = integrate_polyline_cost(direct, cost, traversable)
        astar_cost = integrate_polyline_cost(astar_route, cost, traversable)

        self.assertIsNotNone(route)
        self.assertGreater(len(route), 2)
        self.assertLess(route_cost, direct_cost)
        self.assertLess(route_cost, astar_cost)
        self.assertTrue(
            all(cost[row, col] == 1.0 for row, col in route),
            "route vertices should skirt the high-cost region",
        )

    def test_unreachable_when_hard_barrier_spans_entire_raster(self):
        cost = np.ones((11, 21), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        traversable[:, 10] = False

        self.assertIsNone(
            theta_star_cost_path(cost, traversable, (5, 2), (5, 18)))

    def test_baseline_astar_and_theta_star_have_same_feasible_cost_in_open_grid(self):
        cost = np.ones((17, 23), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        start, goal = (2, 2), (14, 19)

        theta_route = theta_star_cost_path(cost, traversable, start, goal)
        astar_route = astar_search(
            cost, start, goal, pixel_spacing=(1.0, 1.0), heuristic_weight=1.0)

        self.assertAlmostEqual(integrate_polyline_cost(
            theta_route, cost, traversable), math.hypot(12, 17), places=12)
        self.assertGreaterEqual(
            integrate_polyline_cost(astar_route, cost, traversable),
            integrate_polyline_cost(theta_route, cost, traversable),
        )


if __name__ == "__main__":
    unittest.main()
