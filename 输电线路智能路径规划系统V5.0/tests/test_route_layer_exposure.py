import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.metrics import route_layer_exposure


class RouteLayerExposureTests(unittest.TestCase):
    def test_exposure_vector_is_independent_of_cost_weights(self):
        shape = (3, 3)
        terrain = {
            "slope": np.zeros(shape, dtype=np.float32),
            "roughness_27": np.zeros(shape, dtype=np.float32),
        }
        terrain["slope"][1] = [10.0, 20.0, 30.0]
        terrain["roughness_27"][1] = [1.0, 2.0, 3.0]
        terrain["terrain_support_ratio"] = np.ones(shape, dtype=np.float32)
        terrain["terrain_support_ratio"][1] = [1.0, 0.5, 0.0]
        osm = {
            "building_density": np.zeros(shape, dtype=np.float32),
            "residential_landuse_mask": np.zeros(shape, dtype=np.float32),
            "landuse_code": np.zeros(shape, dtype=np.uint8),
        }
        osm["building_density"][1, 0] = 0.2
        osm["residential_landuse_mask"][1, 1] = 0.5
        osm["landuse_code"][1] = [1, 7, 4]
        hazard = np.zeros(shape, dtype=np.uint8)
        hazard[1, 1] = 1
        metadata = {name: {"status": "available"} for name in ("buildings", "landuse")}

        result = route_layer_exposure(
            [(0.5, 1.5), (2.5, 1.5)], Affine.identity(), shape,
            terrain, osm, {"landslide": hazard}, metadata,
            hazard_layer_metadata={"landslide": {"coverage_status": "complete"}})

        self.assertEqual(result["route_cell_count"], 3)
        self.assertAlmostEqual(result["surface_slope_deg"]["mean"], 20.0)
        self.assertAlmostEqual(result["roughness_27_m"]["p95"], 2.9)
        self.assertAlmostEqual(result["terrain_factor_support"]["mean"], 0.5)
        self.assertAlmostEqual(result["terrain_factor_support"]["p05"], 0.05)
        self.assertAlmostEqual(
            result["terrain_factor_support"]["full_support_cell_fraction"], 1 / 3)
        self.assertAlmostEqual(result["mapped_settlement_proxy_mean"], (0.2 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(result["mapped_forest_wetland_cell_fraction"], 2 / 3)
        self.assertAlmostEqual(result["hazard_cell_fractions"]["landslide"], 1 / 3)
        self.assertEqual(result["hazard_observed_positive_route_cell_counts"]["landslide"], 1)
        self.assertEqual(result["hazard_coverage_status"], "unknown")
        self.assertEqual(result["mapped_osm_layer_status"]["buildings"], "available")
        self.assertEqual(
            result["mapped_osm_metric_status"]["building_density"], "mapped_features_only")
        self.assertIn("coverage completeness is unverified",
                      result["mapped_osm_exposure_scope"])
        self.assertIn("not length weighted", result["fraction_denominator"])

    def test_missing_osm_layer_status_is_reported_as_unavailable(self):
        shape = (3, 3)
        result = route_layer_exposure(
            [(0.5, 1.5), (2.5, 1.5)], Affine.identity(), shape,
            {"slope": np.ones(shape), "roughness_27": np.ones(shape)},
            {}, {}, {"buildings": {"status": "missing_file"}},
            hazard_coverage_status="partial")

        self.assertIsNone(result["mapped_settlement_proxy_mean"])
        self.assertIsNone(result["mapped_forest_wetland_cell_fraction"])
        self.assertIsNone(result["terrain_factor_support"])
        self.assertEqual(result["hazard_coverage_status"], "partial")

    def test_incomplete_hazard_coverage_never_reports_zero_risk_fraction(self):
        shape = (3, 3)
        partial_mask = np.zeros(shape, dtype=np.uint8)
        partial_mask[1, 1] = 1
        result = route_layer_exposure(
            [(0.5, 1.5), (2.5, 1.5)], Affine.identity(), shape,
            {}, {}, {"unknown": np.zeros(shape), "partial": partial_mask},
            hazard_coverage_status="partial",
            hazard_layer_metadata={
                "unknown": {"coverage_status": "unknown"},
                "partial": {"coverage_status": "partial"},
            })

        self.assertIsNone(result["hazard_cell_fractions"]["unknown"])
        self.assertIsNone(result["hazard_cell_fractions"]["partial"])
        self.assertEqual(result["hazard_observed_positive_route_cell_counts"]["unknown"], 0)
        self.assertEqual(result["hazard_observed_positive_route_cell_counts"]["partial"], 1)
        self.assertIn("loaded masks only", result["hazard_positive_route_cell_counts_scope"])

    def test_available_osm_components_are_reported_independently(self):
        shape = (3, 3)
        density = np.zeros(shape, dtype=np.float32)
        density[1, 1] = 0.5
        result = route_layer_exposure(
            [(0.5, 1.5), (2.5, 1.5)], Affine.identity(), shape,
            {}, {"building_density": density}, {},
            {"buildings": {"status": "available", "coverage_status": "partial"},
             "landuse": {"status": "missing_file", "coverage_status": "unknown"}})

        self.assertAlmostEqual(result["mapped_building_density_mean"], 0.5 / 3)
        self.assertIsNone(result["mapped_forest_wetland_cell_fraction"])
        self.assertIsNone(result["mapped_settlement_proxy_mean"])
        self.assertEqual(
            result["mapped_osm_metric_status"]["building_density"], "mapped_features_only")
        self.assertEqual(result["mapped_osm_coverage_status"]["buildings"], "partial")

    def test_missing_osm_arrays_are_not_replaced_with_zero_exposure(self):
        shape = (3, 3)
        result = route_layer_exposure(
            [(0.5, 1.5), (2.5, 1.5)], Affine.identity(), shape,
            {}, {}, {},
            {"buildings": {"status": "available"},
             "landuse": {"status": "available"}})

        self.assertIsNone(result["mapped_building_density_mean"])
        self.assertIsNone(result["mapped_settlement_proxy_mean"])
        self.assertEqual(result["mapped_osm_metric_status"]["building_density"], "array_missing")

    def test_exposure_rejects_rasters_on_a_different_grid(self):
        cases = (
            ({"slope": np.ones((2, 2))}, {}, {}),
            ({}, {"building_density": np.ones((2, 2))}, {}),
            ({}, {}, {"landslide": np.ones((2, 2))}),
        )
        for terrain, osm, hazards in cases:
            with self.subTest(terrain=bool(terrain), osm=bool(osm), hazards=bool(hazards)):
                with self.assertRaisesRegex(ValueError, "does not match route grid"):
                    route_layer_exposure(
                        [(0.5, 1.5), (2.5, 1.5)], Affine.identity(), (3, 3),
                        terrain, osm, hazards)


if __name__ == "__main__":
    unittest.main()
