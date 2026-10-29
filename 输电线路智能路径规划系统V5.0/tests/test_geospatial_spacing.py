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

from transmission_path_planner.config import HAS_SCIPY, ndi
from transmission_path_planner.data import (
    _hillshade, _masked_local_stats, compute_terrain_factors, load_dem_window,
)
from transmission_path_planner.geometry import pixel_meters, pixel_spacing_meters
from transmission_path_planner.visualization import plot_comparison


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

    def test_terrain_factors_support_single_row_or_column_dem(self):
        for shape in ((1, 1), (1, 3), (3, 1)):
            with self.subTest(shape=shape):
                dem = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

                terrain = compute_terrain_factors(dem, 20.0)

                self.assertEqual(terrain["slope"].shape, shape)
                expected_slope = np.degrees(np.arctan(1 / 20)) if max(shape) > 1 else 0.0
                np.testing.assert_allclose(terrain["slope"], expected_slope, atol=1e-6)

    def test_hillshade_supports_single_row_or_column_dem(self):
        for shape in ((1, 1), (1, 3), (3, 1)):
            with self.subTest(shape=shape):
                hillshade = _hillshade(np.ones(shape, dtype=np.float32))

                self.assertEqual(hillshade.shape, shape)
                np.testing.assert_array_equal(hillshade, np.full(shape, 0.5))

    def test_comparison_plot_supports_degenerate_dem(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        for shape in ((1, 1), (1, 3), (3, 1)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as folder:
                first = (120.0005, 24.9995)
                last = (120.0 + (shape[1] - 0.5) * 0.001,
                        25.0 - (shape[0] - 0.5) * 0.001)
                output = Path(folder) / "comparison.png"

                plot_comparison(
                    np.ones(shape, dtype=np.float32), transform,
                    {"A*(baseline)": [first, last]},
                    (first[1], first[0]), (last[1], last[0]), output,
                )

                self.assertGreater(output.stat().st_size, 0)

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

    def test_terrain_factors_at_nodata_edge_ignore_distant_elevations(self):
        dem_a = np.zeros((41, 41), dtype=np.float32)
        dem_a[20, 21] = np.nan
        dem_b = dem_a.copy()
        rows, cols = np.indices(dem_b.shape)
        distant = np.maximum(np.abs(rows - 20), np.abs(cols - 20)) > 10
        dem_b[distant] = 1000.0

        factors_a = compute_terrain_factors(dem_a, 100.0)
        factors_b = compute_terrain_factors(dem_b, 100.0)

        for name in (
            "slope", "aspect_cos", "aspect_sin", "tri",
            "roughness_3", "roughness_9", "roughness_27",
            "tpi_100", "tpi_300", "tpi_900",
            "profile_curvature", "plan_curvature", "terrain_support_ratio",
        ):
            with self.subTest(factor=name):
                np.testing.assert_allclose(
                    factors_a[name][20, 20], factors_b[name][20, 20],
                    rtol=0.0, atol=1e-6, equal_nan=True,
                )
        self.assertGreater(float(factors_a["terrain_support_ratio"][20, 20]), 0.0)
        self.assertLess(float(factors_a["terrain_support_ratio"][20, 20]), 1.0)

    @unittest.skipUnless(HAS_SCIPY, "masked local stats require scipy")
    def test_chunked_masked_local_stats_match_full_grid_reference(self):
        rows, cols = np.indices((11, 9))
        raw = (2500.0 + 0.25 * rows + 0.125 * cols
               + 0.03125 * ((rows * cols) % 7)).astype(np.float32)
        nodata = raw.copy()
        nodata[0, 0] = np.nan
        nodata[3, 4] = np.nan
        nodata[8, 1] = np.nan
        cases = ((np.ones(raw.shape, dtype=bool), raw), (np.isfinite(nodata), nodata))

        for valid, source in cases:
            for size in (3, 7, 15):
                with self.subTest(size=size, has_nodata=not valid.all()):
                    expected_weights = valid.astype(np.float64)
                    expected_values = np.where(valid, source, 0.0).astype(np.float64)
                    expected_support = ndi.uniform_filter(
                        expected_weights, size=size, mode="reflect")
                    expected_mean = ndi.uniform_filter(
                        expected_values, size=size, mode="reflect")
                    expected_mean = np.divide(
                        expected_mean, expected_support, out=expected_mean,
                        where=expected_support > 0)
                    expected_mean_sq = ndi.uniform_filter(
                        expected_values * expected_values, size=size, mode="reflect")
                    expected_mean_sq = np.divide(
                        expected_mean_sq, expected_support, out=expected_mean_sq,
                        where=expected_support > 0)
                    np.subtract(expected_mean_sq, expected_mean * expected_mean,
                                out=expected_mean_sq)
                    np.maximum(expected_mean_sq, 0.0, out=expected_mean_sq)
                    np.sqrt(expected_mean_sq, out=expected_mean_sq)
                    expected_mean[~valid] = 0.0
                    expected_mean_sq[~valid] = 0.0
                    expected_support[~valid] = 0.0

                    actual = _masked_local_stats(source, valid, size, chunk_rows=3)

                    for got, expected in zip(actual, (
                            expected_mean, expected_mean_sq, expected_support)):
                        np.testing.assert_array_equal(got, expected.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
