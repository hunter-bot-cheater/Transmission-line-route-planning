import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.cli import route_hazard_exposure
from transmission_path_planner.cost import build_route_cost


class OfficialHazardCostTests(unittest.TestCase):
    def test_official_hazards_add_one_union_soft_cost_without_double_counting(self):
        shape = (1, 4)
        terrain = {
            "slope": np.zeros(shape, dtype=np.float32),
            "roughness_27": np.zeros(shape, dtype=np.float32),
        }
        masks = {
            "debris_flow_impact": np.array([[1, 1, 0, 0]], dtype=np.uint8),
            "landslide_potential": np.array([[0, 1, 1, 0]], dtype=np.uint8),
            "landslide_impact": np.array([[0, 0, 1, 0]], dtype=np.uint8),
        }

        cost = build_route_cost(
            terrain, {}, settlement_weight=0.0, official_hazard_masks=masks,
            official_hazard_weight=1.5)

        np.testing.assert_allclose(cost, [[1.5, 1.5, 1.5, 0.0]])

    def test_missing_hazard_masks_are_neutral_and_zero_weight_disables_them(self):
        shape = (1, 3)
        terrain = {
            "slope": np.zeros(shape, dtype=np.float32),
            "roughness_27": np.zeros(shape, dtype=np.float32),
        }
        masks = {"landslide_potential": np.array([[0, 1, 0]], dtype=np.uint8)}

        missing = build_route_cost(terrain, {}, settlement_weight=0.0)
        disabled = build_route_cost(
            terrain, {}, settlement_weight=0.0, official_hazard_masks=masks,
            official_hazard_weight=0.0)
        enabled = build_route_cost(
            terrain, {}, settlement_weight=0.0, official_hazard_masks=masks,
            official_hazard_weight=1.0)

        np.testing.assert_array_equal(missing, disabled)
        np.testing.assert_allclose(enabled, [[0.0, 1.0, 0.0]])

    def test_route_exposure_reports_each_hazard_layer(self):
        geo = [(0.1, 1.5), (2.9, 1.5)]
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)
        masks = {
            "debris_flow_impact": np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),
            "landslide_potential": np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),
        }

        result = route_hazard_exposure(geo, masks, transform, (3, 3))

        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["route_cell_count"], 3)
        self.assertEqual(result["layers"]["debris_flow_impact"]["route_cell_count"], 2)
        self.assertAlmostEqual(result["layers"]["landslide_potential"]["route_cell_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
