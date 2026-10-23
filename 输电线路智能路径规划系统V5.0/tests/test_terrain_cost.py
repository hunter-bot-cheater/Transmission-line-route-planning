import unittest

import numpy as np

from transmission_path_planner.cost import build_route_cost
from transmission_path_planner.methods import astar_search


class TerrainCostTests(unittest.TestCase):
    def test_positive_tpi_adds_bounded_ridge_penalty(self):
        terrain = {
            "slope": np.zeros((1, 3), dtype=np.float32),
            "roughness_27": np.zeros((1, 3), dtype=np.float32),
            "tpi_300": np.array([[0.0, 100.0, 400.0]], dtype=np.float32),
        }

        cost = build_route_cost(
            terrain, {}, settlement_weight=0.0, ridge_penalty_weight=0.10)

        np.testing.assert_allclose(cost, [[0.0, 0.05, 0.10]])

    def test_negative_tpi_and_missing_tpi_do_not_add_penalty(self):
        terrain = {
            "slope": np.zeros((1, 2), dtype=np.float32),
            "roughness_27": np.zeros((1, 2), dtype=np.float32),
            "tpi_300": np.array([[-200.0, np.nan]], dtype=np.float32),
        }
        expected = build_route_cost(terrain, {}, settlement_weight=0.0)
        without_tpi = build_route_cost(
            {key: value for key, value in terrain.items() if key != "tpi_300"},
            {}, settlement_weight=0.0)

        np.testing.assert_allclose(expected, without_tpi)
        np.testing.assert_allclose(expected, [[0.0, 0.0]])

    def test_forest_and_wetland_add_soft_cost_only_when_mapped(self):
        terrain = {
            "slope": np.zeros((1, 4), dtype=np.float32),
            "roughness_27": np.zeros((1, 4), dtype=np.float32),
        }
        osm = {"landuse_code": np.array([[1.0, 7.0, 2.0, 0.0]], dtype=np.float32)}

        cost = build_route_cost(
            terrain, osm, settlement_weight=0.0, landcover_penalty_weight=0.10)
        missing = build_route_cost(terrain, {}, settlement_weight=0.0)

        np.testing.assert_allclose(cost, [[0.10, 0.10, 0.0, 0.0]])
        np.testing.assert_allclose(missing, np.zeros((1, 4), dtype=np.float32))

    def test_astar_prefers_a_lower_corridor_around_a_high_ridge(self):
        shape = (5, 7)
        terrain = {
            "slope": np.zeros(shape, dtype=np.float32),
            "roughness_27": np.zeros(shape, dtype=np.float32),
            "tpi_300": np.zeros(shape, dtype=np.float32),
        }
        terrain["tpi_300"][2, 1:6] = 400.0
        route_cost = build_route_cost(terrain, {}, settlement_weight=0.0)
        path = astar_search(1.0 + 4.0 * route_cost, (2, 0), (2, 6))

        self.assertIsNotNone(path)
        self.assertTrue(any(row != 2 for row, _ in path))


if __name__ == "__main__":
    unittest.main()
