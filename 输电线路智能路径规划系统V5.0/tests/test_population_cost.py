import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine
from pyproj import Geod
from rasterio.transform import from_origin

from transmission_path_planner.cli import main, route_population_coverage
from transmission_path_planner.cost import build_route_cost
from transmission_path_planner.data import (
    POPULATION_DENSITY_NORMALIZATION_PEOPLE_PER_KM2,
    load_population_density_window,
    population_cost_for_routing,
    population_density_cost,
)


class PopulationCostTests(unittest.TestCase):
    @staticmethod
    def _write_population(path, values, transform, nodata=-99999, tags=None):
        with rasterio.open(
                path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                nodata=nodata) as dst:
            dst.write(values.astype(np.float32), 1)
            if tags:
                dst.update_tags(**tags)

    @staticmethod
    def _grid_population_total(density, transform):
        geod = Geod(ellps="WGS84")
        total = 0.0
        for row in range(density.shape[0]):
            y0 = transform.f + row * transform.e
            y1 = y0 + transform.e
            for col in range(density.shape[1]):
                x0 = transform.c + col * transform.a
                x1 = x0 + transform.a
                area = abs(geod.polygon_area_perimeter(
                    [x0, x1, x1, x0], [y0, y0, y1, y1])[0])
                total += float(density[row, col]) * area / 1_000_000.0
        return total

    def test_disabled_population_weight_is_bitwise_legacy_cost(self):
        terrain = {
            "slope": np.array([[0.0, 25.0], [45.0, 10.0]], dtype=np.float32),
            "roughness_27": np.array([[0.0, 5.0], [20.0, 2.0]], dtype=np.float32),
        }
        proxy = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
        baseline = build_route_cost(terrain, {})
        disabled = build_route_cost(
            terrain, {}, population_proxy=proxy, population_weight=0.0)
        np.testing.assert_array_equal(disabled, baseline)

    def test_enabled_population_proxy_adds_soft_cost_and_validates_input(self):
        terrain = {"slope": np.zeros((1, 3)), "roughness_27": np.zeros((1, 3))}
        proxy = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        cost = build_route_cost(
            terrain, {}, population_proxy=proxy, population_weight=2.0)
        np.testing.assert_allclose(cost, [[0.0, 1.0, 2.0]])
        with self.assertRaisesRegex(ValueError, "required"):
            build_route_cost(terrain, {}, population_weight=1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            build_route_cost(terrain, {}, population_proxy=np.zeros((2, 2)),
                             population_weight=1.0)
        with self.assertRaisesRegex(ValueError, "finite values"):
            build_route_cost(terrain, {}, population_proxy=np.array([[0.0, np.nan, 0.0]]),
                             population_weight=1.0)
        for weight in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                build_route_cost(terrain, {}, population_proxy=proxy,
                                 population_weight=weight)

    def test_density_normalization_uses_fixed_scale(self):
        scale = POPULATION_DENSITY_NORMALIZATION_PEOPLE_PER_KM2
        transformed = population_density_cost(np.array([[0.0, scale, 2 * scale]]))
        np.testing.assert_allclose(transformed, [[0.0, 1.0, 1.0]])
        one_value = population_density_cost(np.array([[100.0]]))
        same_value = population_density_cost(np.array([[100.0, 100.0, 100.0]]))
        np.testing.assert_array_equal(one_value[0, 0], same_value[0, 0])

    def test_unknown_density_keeps_mask_and_gets_conservative_routing_penalty(self):
        density = np.array([[0.0, np.nan, 100.0]], dtype=np.float32)
        proxy, unknown = population_cost_for_routing(density)
        self.assertEqual(unknown.tolist(), [[False, True, False]])
        np.testing.assert_array_equal(proxy[0, 1], 1.0)
        self.assertEqual(proxy[0, 0], 0.0)

    def test_population_nodata_median_and_zero_are_explicit_sensitivity_policies(self):
        density = np.array([[0.0, np.nan, 10000.0]], dtype=np.float32)
        median_proxy, median_unknown = population_cost_for_routing(
            density, nodata_policy="median")
        zero_proxy, zero_unknown = population_cost_for_routing(
            density, nodata_policy="zero")

        np.testing.assert_array_equal(median_unknown, [[False, True, False]])
        np.testing.assert_allclose(median_proxy, [[0.0, 0.5, 1.0]])
        np.testing.assert_array_equal(zero_unknown, median_unknown)
        np.testing.assert_array_equal(zero_proxy, [[0.0, 0.0, 1.0]])

    def test_population_nodata_median_requires_valid_cells_and_policy_is_validated(self):
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            population_cost_for_routing(
                np.full((2, 2), np.nan), nodata_policy="median")
        with self.assertRaisesRegex(ValueError, "must be maximum, median, or zero"):
            population_cost_for_routing(
                np.array([[1.0, np.nan]]), nodata_policy="nearest")

    def test_route_coverage_flags_unknown_cells(self):
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
        valid = np.array([[True, False, True], [True, True, True]])
        route = [(0.1, 1.5), (2.9, 1.5)]
        measured = route_population_coverage(route, valid, transform, valid.shape)
        self.assertEqual(measured["status"], "incomplete_unknown_population_cells")
        self.assertEqual(measured["unknown_route_cell_count"], 1)
        self.assertAlmostEqual(measured["coverage_ratio"], 2 / 3)

    def test_density_alignment_preserves_population_total_on_aligned_finer_grid(self):
        src_transform = from_origin(121.0, 25.0, 0.002, 0.002)
        dst_transform = from_origin(121.0, 25.0, 0.001, 0.001)
        geod = Geod(ellps="WGS84")
        source_density = np.array([[1200.0, 50.0], [300.0, 800.0]])
        counts = np.empty(source_density.shape, dtype=np.float32)
        for row in range(2):
            y0 = src_transform.f + row * src_transform.e
            y1 = y0 + src_transform.e
            x0 = src_transform.c
            x1 = x0 + src_transform.a
            area = abs(geod.polygon_area_perimeter(
                [x0, x1, x1, x0], [y0, y0, y1, y1])[0])
            counts[row] = source_density[row] * area / 1_000_000.0

        with TemporaryDirectory() as temp_dir:
            source_path = f"{temp_dir}/population.tif"
            self._write_population(source_path, counts, src_transform)
            aligned, metadata = load_population_density_window(
                source_path, (4, 4), dst_transform, "EPSG:4326",
                required_mask=np.ones((4, 4), dtype=bool),
            )
            coarse_transform = from_origin(121.0, 25.0, 0.004, 0.004)
            coarse, coarse_metadata = load_population_density_window(
                source_path, (1, 1), coarse_transform, "EPSG:4326",
                required_mask=np.ones((1, 1), dtype=bool),
            )

        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["aligned_value_unit"], "people_per_km2")
        self.assertEqual(metadata["resampling"], "area_weighted_average_on_density")
        self.assertEqual(metadata["coverage_ratio"], 1.0)
        np.testing.assert_allclose(aligned[::2, ::2], source_density, rtol=2e-4)
        self.assertAlmostEqual(
            self._grid_population_total(aligned, dst_transform),
            float(counts.sum()), delta=float(counts.sum()) * 2e-4)

        self.assertEqual(coarse_metadata["status"], "complete")
        self.assertAlmostEqual(
            self._grid_population_total(coarse, coarse_transform),
            float(counts.sum()), delta=float(counts.sum()) * 2e-4)

    def test_nodata_stays_missing_and_reduces_reported_coverage(self):
        transform = from_origin(121.0, 25.0, 0.001, 0.001)
        counts = np.array([[10.0, -99999.0], [0.0, 3.0]], dtype=np.float32)
        with TemporaryDirectory() as temp_dir:
            source_path = f"{temp_dir}/population.tif"
            self._write_population(source_path, counts, transform)
            required = np.ones((2, 2), dtype=bool)
            aligned, metadata = load_population_density_window(
                source_path, (2, 2), transform, "EPSG:4326", required)

        self.assertTrue(np.isnan(aligned[0, 1]))
        self.assertEqual(metadata["status"], "partial")
        self.assertEqual(metadata["coverage_ratio"], 0.75)
        self.assertEqual(metadata["missing_required_cell_count"], 1)
        self.assertEqual(metadata["nodata_policy"], "preserve_as_nan_for_explicit_unknown_policy")
        self.assertTrue(np.isfinite(aligned[0, 0]))

    def test_source_license_and_attribution_are_read_from_geotiff_tags(self):
        transform = from_origin(121.0, 25.0, 0.001, 0.001)
        tags = {
            "TIFFTAG_IMAGEDESCRIPTION": "TWN population 2025 [WorldPop R2025A v1]",
            "TIFFTAG_DOCUMENTNAME": "twn_pop_2025_CN_100m_R2025A_v1",
            "TIFFTAG_COPYRIGHT": "CC-BY-4.0",
        }
        with TemporaryDirectory() as temp_dir:
            source_path = f"{temp_dir}/renamed_population.tif"
            self._write_population(source_path, np.ones((1, 1)), transform, tags=tags)
            _, metadata = load_population_density_window(
                source_path, (1, 1), transform, "EPSG:4326")

        source = metadata["source_metadata"]
        self.assertEqual(source["provider"], "WorldPop")
        self.assertEqual(source["dataset_title"], tags["TIFFTAG_IMAGEDESCRIPTION"])
        self.assertEqual(source["license_id"], "CC-BY-4.0")
        self.assertEqual(source["license_url"], "https://creativecommons.org/licenses/by/4.0/")
        self.assertIn("WorldPop", source["attribution"])
        self.assertIsNone(source["doi"])

    def test_matching_stac_sidecar_adds_population_doi_and_release(self):
        transform = from_origin(121.0, 25.0, 0.001, 0.001)
        item = {
            "assets": {"data": {"href": "https://example.test/renamed_population.tif"}},
            "license": "CC-BY-4.0",
            "properties": {
                "title": "WorldPop Taiwan population 2025",
                "Digital Object Identifier (DOI)": "10.5258/SOTON/WP00839",
                "Release": "R2025A",
                "Version": "v1",
                "year": 2025,
            },
        }
        with TemporaryDirectory() as temp_dir:
            source_path = f"{temp_dir}/renamed_population.tif"
            self._write_population(source_path, np.ones((1, 1)), transform)
            (Path(temp_dir) / "stac_item_2025.json").write_text(
                json.dumps(item), encoding="utf-8")
            _, metadata = load_population_density_window(
                source_path, (1, 1), transform, "EPSG:4326")

        source = metadata["source_metadata"]
        self.assertEqual(source["provider"], "WorldPop")
        self.assertEqual(source["doi"], "10.5258/SOTON/WP00839")
        self.assertEqual(source["release"], "R2025A")
        self.assertEqual(source["version"], "v1")
        self.assertEqual(source["year"], 2025)
        self.assertEqual(source["license_id"], "CC-BY-4.0")
        self.assertEqual(source["metadata_basis"],
                         "embedded GeoTIFF tags and matching adjacent STAC item")

    def test_cli_rejects_nonfinite_or_negative_population_weight_before_io(self):
        for value in ("nan", "inf", "-1"):
            with patch.object(sys, "argv", ["planner", "--population-weight", value]):
                with self.assertRaises(SystemExit) as exc:
                    main()
            self.assertEqual(exc.exception.code, 2)

    def test_cli_requires_raster_when_population_weight_is_positive(self):
        with patch.object(sys, "argv", ["planner", "--population-weight", "0.5"]):
            with self.assertRaises(SystemExit) as exc:
                main()
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
