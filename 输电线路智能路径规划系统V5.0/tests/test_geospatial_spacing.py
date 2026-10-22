import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon

from transmission_path_planner.data import compute_terrain_factors, load_dem_window
from transmission_path_planner.geometry import pixel_meters, pixel_spacing_meters


PREPARE_BUILDINGS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_official_buildings.py"
PREPARE_BUILDINGS_SPEC = importlib.util.spec_from_file_location("prepare_official_buildings", PREPARE_BUILDINGS_PATH)
prepare_buildings = importlib.util.module_from_spec(PREPARE_BUILDINGS_SPEC)
PREPARE_BUILDINGS_SPEC.loader.exec_module(prepare_buildings)


class GeospatialSpacingTests(unittest.TestCase):
    def test_dem_window_keeps_native_grid_when_max_side_allows_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dem.tif"
            data = np.arange(80 * 60, dtype=np.float32).reshape(60, 80)
            transform = Affine(0.0002, 0, 120.0, 0, -0.0002, 24.0)
            with rasterio.open(
                path, "w", driver="GTiff", width=80, height=60, count=1,
                dtype="float32", crs="EPSG:4326", transform=transform,
            ) as dst:
                dst.write(data, 1)

            dem, actual_transform, crs, actual_pixel_m = load_dem_window(
                path, 120.0, 23.988, 120.016, 24.0, max_side=4096
            )

        self.assertEqual(dem.shape, (60, 80))
        self.assertEqual(actual_transform, transform)
        self.assertEqual(crs, "EPSG:4326")
        center_lat = transform.f + transform.e * (dem.shape[0] / 2.0)
        self.assertAlmostEqual(
            actual_pixel_m, pixel_meters(transform, center_lat, crs), places=6
        )

    def test_official_building_import_reprojects_and_records_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "buildings.shp"
            output = Path(folder) / "buildings.geojson"
            frame = gpd.GeoDataFrame(
                {"UNUSED": ["discarded"]},
                geometry=[Polygon([(300000, 2740000), (300020, 2740000),
                                   (300020, 2740020), (300000, 2740020)])],
                crs="EPSG:3826",
            )
            frame.to_file(source)

            count, source_crs = prepare_buildings.convert_buildings(
                source, output, source_date="2026-08", license_note="approved academic use"
            )
            document = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(source_crs, "EPSG:3826")
        self.assertEqual(document["metadata"]["constraint_mode"], "hard")
        self.assertEqual(document["features"][0]["properties"]["mode"], "hard")
        self.assertNotIn("UNUSED", document["features"][0]["properties"])
        self.assertEqual(document["data_sources"][0]["license"], "approved academic use")

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

    def test_terrain_roughness_matches_population_std_at_interior_cell(self):
        transform = Affine(0.0002, 0, 120.0, 0, -0.0002, 24.0)
        dem = np.arange(31 * 31, dtype=np.float32).reshape(31, 31) % 17
        row_m, col_m = pixel_spacing_meters(transform, 24.0, "EPSG:4326")

        terrain = compute_terrain_factors(dem, row_m, transform, "EPSG:4326")

        spacing = (row_m * col_m) ** 0.5
        radius = max(1, round(100 / spacing))
        window = dem[15 - radius:16 + radius, 15 - radius:16 + radius]
        self.assertAlmostEqual(
            float(terrain["roughness_3"][15, 15]), float(window.std()), delta=1e-4
        )


if __name__ == "__main__":
    unittest.main()
