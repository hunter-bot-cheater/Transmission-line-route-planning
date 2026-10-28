import heapq
import math
import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.experimental_astar_16 import (
    NEIGHBORS_8,
    NEIGHBORS_16,
    astar_exact_integral_path,
    astar_16_neighbor_path,
    integrate_path_cost,
)
from transmission_path_planner.metrics import path_cost_integral
from transmission_path_planner.methods import astar_search
from transmission_path_planner.smoothing import path_hard_violations


def _route_geometry(path):
    return [(col + 0.5, row + 0.5) for row, col in path]


def _dijkstra_oracle(cost, traversable, start, goal, spacing,
                     neighbors=NEIGHBORS_16):
    queue = [(0.0, start)]
    best = {start: 0.0}
    while queue:
        score, cell = heapq.heappop(queue)
        if score > best[cell] + 1e-12:
            continue
        if cell == goal:
            return score
        for dr, dc in neighbors:
            neighbor = cell[0] + dr, cell[1] + dc
            if not (0 <= neighbor[0] < cost.shape[0]
                    and 0 <= neighbor[1] < cost.shape[1]):
                continue
            edge = integrate_path_cost(
                [cell, neighbor], cost, traversable, spacing, neighbors)
            if not math.isfinite(edge):
                continue
            candidate = score + edge
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return math.inf


def _lexicographic_dijkstra_oracle(cost, traversable, start, goal, spacing,
                                   neighbors=NEIGHBORS_16, elevation=None,
                                   grade_penalty_weight=0.0):
    queue = [(0.0, 0.0, start)]
    best = {start: (0.0, 0.0)}
    while queue:
        score, length, cell = heapq.heappop(queue)
        if (score, length) != best[cell]:
            continue
        if cell == goal:
            return score, length
        for dr, dc in neighbors:
            neighbor = cell[0] + dr, cell[1] + dc
            if not (0 <= neighbor[0] < cost.shape[0]
                    and 0 <= neighbor[1] < cost.shape[1]):
                continue
            edge_cost = integrate_path_cost(
                [cell, neighbor], cost, traversable, spacing, neighbors,
                elevation=elevation,
                grade_penalty_weight=grade_penalty_weight)
            if not math.isfinite(edge_cost):
                continue
            candidate = (
                score + edge_cost,
                length + math.hypot(dr * spacing[0], dc * spacing[1]),
            )
            if candidate < best.get(neighbor, (math.inf, math.inf)):
                best[neighbor] = candidate
                heapq.heappush(queue, (*candidate, neighbor))
    return math.inf, math.inf


class ExperimentalAstar16Tests(unittest.TestCase):
    def test_positive_grade_penalty_requires_elevation_for_integrals(self):
        cost = np.ones((2, 2), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)

        with self.assertRaisesRegex(ValueError, "elevation is required"):
            integrate_path_cost(
                [(0, 0), (0, 1)], cost, traversable,
                grade_penalty_weight=1.0)

        with self.assertRaisesRegex(ValueError, "elevation is required"):
            astar_exact_integral_path(
                cost, traversable, (0, 0), (1, 1), NEIGHBORS_8,
                grade_penalty_weight=1.0)

    def test_positive_grade_penalty_rejects_nonfinite_endpoint_elevation(self):
        cost = np.ones((2, 2), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        elevation = np.zeros(cost.shape, dtype=np.float64)
        elevation[0, 0] = np.nan

        self.assertTrue(math.isinf(integrate_path_cost(
            [(0, 0), (1, 1)], cost, traversable, elevation=elevation,
            grade_penalty_weight=1.0)))
        self.assertIsNone(astar_exact_integral_path(
            cost, traversable, (0, 0), (1, 1), NEIGHBORS_8,
            elevation=elevation, grade_penalty_weight=1.0))

    def test_positive_grade_penalty_blocks_nodata_crossed_by_search_edges(self):
        cost = np.ones((3, 3), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        elevation = np.zeros(cost.shape, dtype=np.float64)
        elevation[1, 1] = np.nan
        diagonal = [(0, 0), (2, 2)]

        self.assertTrue(math.isinf(integrate_path_cost(
            diagonal, cost, traversable, elevation=elevation,
            grade_penalty_weight=1.0)))
        route = astar_exact_integral_path(
            cost, traversable, (0, 0), (2, 2), NEIGHBORS_16,
            elevation=elevation, grade_penalty_weight=1.0)

        self.assertIsNotNone(route)
        self.assertTrue(all(math.isfinite(integrate_path_cost(
            [first, second], cost, traversable, elevation=elevation,
            grade_penalty_weight=1.0))
            for first, second in zip(route, route[1:])))

    def test_matches_dijkstra_oracle_and_is_direction_invariant(self):
        rng = np.random.default_rng(20261024)
        cost = rng.uniform(0.5, 8.0, size=(8, 10))
        traversable = rng.random(cost.shape) > 0.18
        traversable[0, :] = traversable[-1, :] = True
        traversable[:, 0] = traversable[:, -1] = True
        start, goal = (0, 0), (7, 9)
        spacing = (230.0, 90.0)

        route = astar_16_neighbor_path(
            cost, traversable, start, goal, pixel_spacing=spacing)
        reverse = astar_16_neighbor_path(
            cost, traversable, goal, start, pixel_spacing=spacing)
        route_cost = integrate_path_cost(route, cost, traversable, spacing)
        oracle_cost, oracle_length = _lexicographic_dijkstra_oracle(
            cost, traversable, start, goal, spacing)
        route_length = sum(
            math.hypot((b[0] - a[0]) * spacing[0],
                       (b[1] - a[1]) * spacing[1])
            for a, b in zip(route, route[1:]))

        self.assertEqual(route[0], start)
        self.assertEqual(route[-1], goal)
        self.assertEqual(route, reverse[::-1])
        self.assertAlmostEqual(route_cost, oracle_cost, places=7)
        self.assertAlmostEqual(route_length, oracle_length, places=7)

    def test_zero_cost_plateau_uses_shortest_route_as_secondary_objective(self):
        rng = np.random.default_rng(1)
        cost = np.zeros((10, 10), dtype=np.float64)
        traversable = rng.random(cost.shape) > 0.18
        start, goal = (0, 0), (9, 9)
        traversable[start] = traversable[goal] = True

        route = astar_16_neighbor_path(cost, traversable, start, goal)
        route_cost = integrate_path_cost(route, cost, traversable)
        route_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(route, route[1:]))
        oracle_cost, oracle_length = _lexicographic_dijkstra_oracle(
            cost, traversable, start, goal, (1.0, 1.0))

        self.assertEqual(route_cost, oracle_cost)
        self.assertAlmostEqual(route_length, oracle_length, places=12)

    def test_eight_neighbor_exact_integral_matches_dijkstra(self):
        rng = np.random.default_rng(20261025)
        cost = rng.uniform(0.5, 8.0, size=(8, 10))
        traversable = np.ones(cost.shape, dtype=bool)
        start, goal = (0, 0), (7, 9)
        spacing = (230.0, 90.0)

        route = astar_exact_integral_path(
            cost, traversable, start, goal, NEIGHBORS_8, pixel_spacing=spacing)
        oracle_cost = _dijkstra_oracle(
            cost, traversable, start, goal, spacing, NEIGHBORS_8)

        self.assertAlmostEqual(
            integrate_path_cost(route, cost, traversable, spacing, NEIGHBORS_8),
            oracle_cost,
            places=7,
        )

    def test_shared_eight_neighbor_cost_matches_production_astar(self):
        rng = np.random.default_rng(20261027)
        cost = rng.uniform(0.5, 8.0, size=(8, 10))
        elevation = rng.uniform(0.0, 1200.0, size=cost.shape)
        traversable = rng.random(cost.shape) > 0.12
        traversable[0, :] = traversable[-1, :] = True
        traversable[:, 0] = traversable[:, -1] = True
        start, goal = (0, 0), (7, 9)
        spacing = (230.0, 90.0)
        grade_weight = 0.8

        production_path = astar_search(
            np.where(traversable, cost, np.nan), start, goal,
            elevation=elevation, pixel_spacing=spacing,
            grade_penalty_weight=grade_weight)
        shared_path = astar_exact_integral_path(
            cost, traversable, start, goal, NEIGHBORS_8,
            pixel_spacing=spacing, elevation=elevation,
            grade_penalty_weight=grade_weight)
        production_cost = integrate_path_cost(
            production_path, cost, traversable, spacing, NEIGHBORS_8,
            elevation=elevation, grade_penalty_weight=grade_weight)
        shared_cost = integrate_path_cost(
            shared_path, cost, traversable, spacing, NEIGHBORS_8,
            elevation=elevation, grade_penalty_weight=grade_weight)
        route_length = lambda path: sum(
            math.hypot((b[0] - a[0]) * spacing[0],
                       (b[1] - a[1]) * spacing[1])
            for a, b in zip(path, path[1:]))

        self.assertIsNotNone(production_path)
        self.assertIsNotNone(shared_path)
        self.assertAlmostEqual(production_cost, shared_cost, places=7)
        self.assertAlmostEqual(
            route_length(production_path), route_length(shared_path), places=7)

    def test_shared_grade_objective_matches_dijkstra_for_both_neighbor_sets(self):
        rng = np.random.default_rng(20261026)
        cost = rng.uniform(0.5, 8.0, size=(8, 10))
        elevation = rng.uniform(0.0, 1500.0, size=cost.shape)
        traversable = rng.random(cost.shape) > 0.12
        traversable[0, :] = traversable[-1, :] = True
        traversable[:, 0] = traversable[:, -1] = True
        start, goal = (0, 0), (7, 9)
        spacing = (230.0, 90.0)
        grade_weight = 1.25

        for neighbors in (NEIGHBORS_8, NEIGHBORS_16):
            route = astar_exact_integral_path(
                cost, traversable, start, goal, neighbors,
                pixel_spacing=spacing, elevation=elevation,
                grade_penalty_weight=grade_weight,
            )
            self.assertIsNotNone(route)
            route_cost = integrate_path_cost(
                route, cost, traversable, spacing, neighbors,
                elevation=elevation, grade_penalty_weight=grade_weight)
            route_length = sum(
                math.hypot((b[0] - a[0]) * spacing[0],
                           (b[1] - a[1]) * spacing[1])
                for a, b in zip(route, route[1:]))
            oracle_cost, oracle_length = _lexicographic_dijkstra_oracle(
                cost, traversable, start, goal, spacing, neighbors,
                elevation=elevation, grade_penalty_weight=grade_weight)

            self.assertAlmostEqual(route_cost, oracle_cost, places=7)
            self.assertAlmostEqual(route_length, oracle_length, places=7)

    def test_shared_grade_penalty_is_included_on_knight_edge(self):
        cost = np.ones((3, 2), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        elevation = np.zeros(cost.shape, dtype=np.float64)
        elevation[2, 1] = 10.0
        edge = [(0, 0), (2, 1)]

        base = integrate_path_cost(edge, cost, traversable,
                                   neighbor_offsets=NEIGHBORS_16)
        shared = integrate_path_cost(
            edge, cost, traversable, neighbor_offsets=NEIGHBORS_16,
            elevation=elevation, grade_penalty_weight=2.0)

        self.assertAlmostEqual(base, math.sqrt(5.0), places=12)
        self.assertAlmostEqual(shared, 3.0 * math.sqrt(5.0), places=12)

    def test_knight_edge_integrates_soft_cost_in_crossed_cells(self):
        cost = np.ones((3, 2), dtype=np.float64)
        cost[1, 1] = 100.0
        traversable = np.ones(cost.shape, dtype=bool)

        edge_cost = integrate_path_cost(
            [(0, 0), (2, 1)], cost, traversable,
            neighbor_offsets=NEIGHBORS_16)

        self.assertGreater(edge_cost, math.hypot(2.0, 1.0) * 5.0)

    def test_segment_integral_matches_production_path_integral(self):
        cost = np.array([
            [1.0, 2.0, 5.0, 4.0, 3.0],
            [3.0, 7.0, 2.0, 6.0, 4.0],
            [2.0, 4.0, 8.0, 3.0, 1.0],
            [5.0, 1.0, 2.0, 9.0, 2.0],
            [2.0, 3.0, 4.0, 1.0, 6.0],
        ])
        traversable = np.ones(cost.shape, dtype=bool)
        path = [(0, 0), (2, 1), (3, 3)]
        spacing = (12.0, 7.0)

        actual = integrate_path_cost(path, cost, traversable, spacing)
        expected = path_cost_integral(
            _route_geometry(path), np.zeros(cost.shape), Affine.identity(),
            cost, grade_penalty_weight=0.0, turn_penalty_weight=0.0,
            pixel_spacing=spacing,
        )

        self.assertEqual(expected["status"], "measured")
        self.assertAlmostEqual(actual, expected["base_cost_meter"], places=3)

    def test_knight_edges_never_cross_a_supercover_obstacle(self):
        cost = np.ones((11, 13), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        traversable[1:10, 6] = False
        start, goal = (5, 2), (5, 10)

        route = astar_16_neighbor_path(cost, traversable, start, goal)

        self.assertIsNotNone(route)
        self.assertGreater(len(route), 2)
        self.assertEqual(
            path_hard_violations(
                _route_geometry(route), Affine.identity(), traversable.astype(np.uint8)),
            0,
        )

    def test_grade_cap_rejects_long_edge_when_intermediate_dem_is_too_steep(self):
        cost = np.ones((4, 5), dtype=np.float64)
        traversable = np.zeros(cost.shape, dtype=bool)
        traversable[1, 0] = True
        traversable[1, 1] = True
        traversable[2, 0:3] = True
        elevation = np.zeros(cost.shape, dtype=np.float64)
        elevation[1, 1] = 100.0
        elevation[2, 0:2] = 100.0
        start, goal = (1, 0), (2, 2)

        unconstrained = astar_16_neighbor_path(
            cost, traversable, start, goal, pixel_spacing=(1.0, 1.0))
        constrained = astar_16_neighbor_path(
            cost, traversable, start, goal, pixel_spacing=(1.0, 1.0),
            elevation=elevation, max_ground_grade=0.1,
        )

        self.assertIsNotNone(unconstrained)
        self.assertIsNone(constrained)

    def test_blocked_endpoints_are_not_silently_moved(self):
        cost = np.ones((5, 5), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)
        traversable[1, 1] = False
        traversable[3, 3] = False

        self.assertIsNone(astar_16_neighbor_path(
            cost, traversable, (1, 1), (4, 4)))
        self.assertIsNone(astar_16_neighbor_path(
            cost, traversable, (0, 0), (3, 3)))

    def test_fractional_grid_indices_are_rejected(self):
        cost = np.ones((4, 4), dtype=np.float64)
        traversable = np.ones(cost.shape, dtype=bool)

        with self.assertRaises(ValueError):
            astar_exact_integral_path(
                cost, traversable, (0.8, 0), (3, 3), NEIGHBORS_8)
        with self.assertRaises(ValueError):
            integrate_path_cost(
                [(0, 0), (1.8, 1)], cost, traversable,
                neighbor_offsets=NEIGHBORS_8)


if __name__ == "__main__":
    unittest.main()
