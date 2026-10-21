import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.data import compute_terrain_factors
from transmission_path_planner.geometry import pixel_spacing_meters


class GeospatialSpacingTests(unittest.TestCase):
    def test_geographic_row_and_column_spacing_are_distinct(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 60.0)

        row_m, col_m = pixel_spacing_meters(transform, 60.0, "EPSG:4326")

        self.assertAlmostEqual(row_m / col_m, 2.0, delta=0.005)

    def test_projected_spacing_uses_crs_linear_units(self):
        transform = Affine(10.0, 0, 0, 0, -20.0, 0)

        row_m, col_m = pixel_spacing_meters(transform, 0.0, "EPSG:32651")

        self.assertAlmostEqual(row_m, 20.0, places=6)
        self.assertAlmostEqual(col_m, 10.0, places=6)

    def test_dem_slope_uses_north_south_ground_spacing(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 60.0)
        dem = np.tile(np.arange(8, dtype=np.float32)[:, None], (1, 8))
        row_m, _ = pixel_spacing_meters(transform, 60.0, "EPSG:4326")

        terrain = compute_terrain_factors(dem, row_m, transform, "EPSG:4326")

        expected = np.degrees(np.arctan(1.0 / row_m))
        self.assertAlmostEqual(float(terrain["slope"][3, 3]), expected, delta=0.02)


if __name__ == "__main__":
    unittest.main()
