import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from affine import Affine
from shapely.geometry import box

from transmission_path_planner import osm
from transmission_path_planner.cost import build_route_cost


def _way(coords, tags=None):
    return {
        "type": "way",
        "tags": tags or {},
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in coords],
    }


class OSMSettlementMetadataTests(unittest.TestCase):
    def setUp(self):
        osm._PKG_CACHE.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _save_cache(self, name, elements):
        path = Path(self.temp_dir.name) / name
        with path.open("wb") as stream:
            pickle.dump({"elements": elements}, stream)
        return path

    def test_invalid_building_polygon_is_repaired_before_rasterization(self):
        cache = self._save_cache("bow_tie_building.pkl", [
            _way([(0.1, 1.9), (1.9, 0.1), (0.1, 0.1),
                  (1.9, 1.9), (0.1, 1.9)], {"building": "yes"})
        ])

        buildings, metadata = osm.load_osm_geoms_with_status(
            cache, "polygon", (0, 0, 2, 2))
        mask = osm.rasterize_on(buildings, Affine(1, 0, 0, 0, -1, 2), (2, 2))

        self.assertEqual(metadata["geometry_repaired_count"], 1)
        self.assertTrue(buildings[0][0].is_valid)
        self.assertTrue(mask.any())

    def test_nonfinite_osm_coordinates_fail_closed(self):
        cache = self._save_cache("nonfinite_building.pkl", [
            _way([(0.1, 0.1), (1.9, 0.1), (1.9, float("nan")),
                  (0.1, 0.1)], {"building": "yes"})
        ])

        with self.assertRaisesRegex(ValueError, "含非有限坐标"):
            osm.load_osm_geoms_with_status(cache, "polygon", (0, 0, 2, 2))

    def test_nonfinite_geometry_outside_request_bbox_does_not_fail_case(self):
        cache = self._save_cache("remote_nonfinite_building.pkl", [
            _way([(120.1, 24.1), (120.2, 24.1), (120.2, 24.2),
                  (120.1, 24.2), (120.1, 24.1)], {"building": "yes"}),
            _way([(110.1, 10.1), (110.2, 10.1), (float("nan"), 10.2),
                  (110.1, 10.1)], {"building": "yes"}),
        ])

        buildings, metadata = osm.load_osm_geoms_with_status(
            cache, "polygon", (120, 24, 121, 25))

        self.assertEqual(len(buildings), 1)
        self.assertEqual(metadata["nonfinite_geometry_outside_bbox_count"], 1)
        self.assertEqual(metadata["status"], "available")
        self.assertEqual(metadata["coverage_status"], "partial")
        self.assertEqual(
            metadata["coverage_reason"],
            "nonfinite_geometry_skipped_outside_bbox")

    def test_only_skipped_nonfinite_geometry_cannot_claim_observed_coverage(self):
        cache = self._save_cache("only_remote_nonfinite_building.pkl", [
            _way([(110.1, 10.1), (110.2, 10.1), (float("nan"), 10.2),
                  (110.1, 10.1)], {"building": "yes"}),
        ])

        buildings, metadata = osm.load_osm_geoms_with_status(
            cache, "polygon", (120, 24, 121, 25))

        self.assertEqual(buildings, [])
        self.assertEqual(metadata["status"], "no_usable_geometry")
        self.assertEqual(metadata["coverage_status"], "partial")
        self.assertEqual(metadata["nonfinite_geometry_outside_bbox_count"], 1)

    def test_nonfinite_geometry_with_no_finite_location_still_fails_closed(self):
        cache = self._save_cache("unlocated_nonfinite_building.pkl", [
            _way([(float("nan"), float("inf")), (float("nan"), float("inf")),
                  (float("nan"), float("inf"))], {"building": "yes"})
        ])

        with self.assertRaisesRegex(ValueError, "无法定位的非有限坐标"):
            osm.load_osm_geoms_with_status(cache, "polygon", (120, 24, 121, 25))

    def test_rasterize_on_propagates_rasterization_errors(self):
        with patch.object(osm, "rio_rasterize", side_effect=RuntimeError("rasterization failed")):
            with self.assertRaisesRegex(RuntimeError, "rasterization failed"):
                osm.rasterize_on([(box(0, 0, 1, 1), {})], Affine.identity(), (2, 2))

    def test_distance_band_uses_axis_spacing_in_geographic_and_projected_crs(self):
        cases = (
            ("EPSG:4326", Affine(0.001, 0, 120, 0, -0.002, 25)),
            ("EPSG:3857", Affine(250, 0, 0, 0, -100, 1000)),
        )
        shape = (5, 5)
        for crs, transform in cases:
            with self.subTest(crs=crs):
                lat_center = transform.f + transform.b * (shape[1] / 2) + transform.e * (shape[0] / 2)
                row_m, col_m = osm.pixel_spacing_meters(transform, lat_center, crs)
                pad_rows = int(np.ceil(1000 / row_m)) + 1
                pad_cols = int(np.ceil(1000 / col_m)) + 1
                padded_on = np.zeros((shape[0] + 2 * pad_rows,
                                      shape[1] + 2 * pad_cols), dtype=np.uint8)
                padded_on[pad_rows + 2, pad_cols + 2] = 1

                def rasterize_padded(_geoms, padded_transform, padded_shape):
                    self.assertEqual(padded_shape, padded_on.shape)
                    self.assertEqual(
                        padded_transform,
                        transform * Affine.translation(-pad_cols, -pad_rows))
                    return padded_on

                with patch.object(osm, "rasterize_on", side_effect=rasterize_padded):
                    distance = osm.compute_distance_band(
                        [(box(0, 0, 1, 1), {})], transform, shape, max_dist_m=1000, crs=crs)

                self.assertAlmostEqual(float(distance[1, 2]), row_m, places=3)
                self.assertAlmostEqual(float(distance[2, 3]), col_m, places=3)
                self.assertNotAlmostEqual(row_m, col_m, places=1)

    @unittest.skipUnless(osm.HAS_SCIPY and osm.HAS_RASTERIO,
                         "distance bands require scipy and rasterio")
    def test_distance_band_includes_buildings_outside_all_four_window_edges(self):
        transform = Affine(100, 0, 0, 0, -100, 500)
        shape = (5, 5)
        cases = (
            ("left", (2, 0), box(-99, 201, -1, 299),
             box(-4899, 201, -4801, 299)),
            ("right", (2, 4), box(501, 201, 599, 299),
             box(5301, 201, 5399, 299)),
            ("top", (0, 2), box(201, 501, 299, 599),
             box(201, 5301, 299, 5399)),
            ("bottom", (4, 2), box(201, -99, 299, -1),
             box(201, -4899, 299, -4801)),
        )

        for side, edge_cell, near_building, far_building in cases:
            for distance, building in ((100.0, near_building), (4900.0, far_building)):
                with self.subTest(side=side, expected_distance_m=distance):
                    band = osm.compute_distance_band(
                        [(building, {})], transform, shape, max_dist_m=5000,
                        crs="EPSG:3857")
                    row, col = edge_cell
                    self.assertAlmostEqual(float(band[row, col]), distance, places=3)

    @unittest.skipUnless(osm.HAS_SCIPY, "building density requires scipy")
    def test_building_density_uses_axis_specific_kernel_and_returns_cell_fraction(self):
        cases = (
            ("EPSG:4326", Affine(0.001, 0, 120, 0, -0.002, 25)),
            ("EPSG:3857", Affine(250, 0, 0, 0, -100, 1000)),
        )
        shape = (9, 9)
        for crs, transform in cases:
            with self.subTest(crs=crs):
                lat_center = transform.f + transform.b * (shape[1] / 2) + transform.e * (shape[0] / 2)
                row_m, col_m = osm.pixel_spacing_meters(transform, lat_center, crs)
                sigma = (500.0 / row_m, 500.0 / col_m)
                pad_rows, pad_cols = int(np.ceil(4 * sigma[0])), int(np.ceil(4 * sigma[1]))
                padded_shape = (shape[0] + 2 * pad_rows, shape[1] + 2 * pad_cols)
                padded_on = np.zeros(padded_shape, dtype=np.uint8)
                padded_on[pad_rows + 4, pad_cols + 4] = 1
                expected_smooth = osm.gaussian_filter(
                    padded_on.astype(np.float32), sigma=sigma, mode="constant")
                expected = expected_smooth[
                    pad_rows:pad_rows + shape[0], pad_cols:pad_cols + shape[1]]

                def rasterize_padded(_geoms, _transform, requested_shape):
                    self.assertEqual(requested_shape, padded_shape)
                    return padded_on

                with patch.object(osm, "rasterize_on", side_effect=rasterize_padded):
                    density = osm.compute_building_density(
                        [(box(0, 0, 1, 1), {})], transform, shape, crs=crs)

                np.testing.assert_allclose(density, expected, rtol=1e-6)
                self.assertTrue(np.all((density >= 0.0) & (density <= 1.0)))
                self.assertNotEqual(sigma[0], sigma[1])

    @unittest.skipUnless(osm.HAS_SCIPY, "building density requires scipy")
    def test_building_density_is_stable_when_same_cell_moves_to_window_edge(self):
        building_center = (2050.0, 2050.0)
        windows = (
            (Affine(100, 0, 0, 0, -100, 4100), (41, 41), (20, 20)),
            (Affine(100, 0, 0, 0, -100, 3100), (21, 21), (10, 20)),
        )

        def rasterize_same_world_building(_geoms, transform, shape):
            mask = np.zeros(shape, dtype=np.uint8)
            col, row = ~transform * building_center
            mask[int(np.floor(row)), int(np.floor(col))] = 1
            return mask

        sampled = []
        with patch.object(osm, "rasterize_on", side_effect=rasterize_same_world_building):
            for transform, shape, cell in windows:
                density = osm.compute_building_density(
                    [(box(2050.0, 2050.0, 2051.0, 2051.0), {})],
                    transform, shape, crs="EPSG:3857")
                sampled.append(float(density[cell]))

        self.assertAlmostEqual(sampled[0], sampled[1], places=6)

    def test_missing_and_empty_caches_have_distinct_statuses(self):
        missing, missing_meta = osm.load_osm_geoms_with_status(
            Path(self.temp_dir.name) / "missing.pkl", "line", (0, 0, 1, 1))
        empty, empty_meta = osm.load_osm_geoms_with_status(
            self._save_cache("empty.pkl", []), "line", (0, 0, 1, 1))

        self.assertEqual(missing, [])
        self.assertEqual(missing_meta["status"], "missing_file")
        self.assertEqual(empty, [])
        self.assertEqual(empty_meta["status"], "empty_source")

    def test_outside_observed_extent_is_distinct_from_no_local_features(self):
        cache = self._save_cache("lines.pkl", [
            _way([(0, 0), (1, 1)]),
            _way([(10, 10), (11, 11)]),
        ])
        _, uncovered = osm.load_osm_geoms_with_status(cache, "line", (20, 20, 21, 21))
        _, empty_window = osm.load_osm_geoms_with_status(cache, "line", (4, 4, 5, 5))

        self.assertEqual(uncovered["status"], "uncovered_bbox")
        self.assertEqual(empty_window["status"], "empty_in_bbox")
        self.assertEqual(empty_window["coverage_status"], "unverified_within_feature_extent")

    def test_outside_features_skip_shapely_geometry_construction(self):
        cache = self._save_cache("far_polygons.pkl", [
            _way([(10, 10), (11, 10), (11, 11), (10, 11)], {"landuse": "residential"}),
        ])
        with patch.object(osm, "Polygon", side_effect=AssertionError("unexpected geometry build")):
            geometries, metadata = osm.load_osm_geoms_with_status(
                cache, "polygon", (0, 0, 1, 1))

        self.assertEqual(geometries, [])
        self.assertEqual(metadata["status"], "uncovered_bbox")

    def test_compute_osm_bands_returns_sidecar_layer_metadata(self):
        requested = {}

        def empty_layer(path, _geom_type, query_bbox):
            requested[Path(path).name] = query_bbox
            return [], {"status": "missing_file", "geometry_count_in_bbox": 0}

        dem = np.zeros((3, 3), dtype=np.float32)
        transform = Affine(0.01, 0, 120, 0, -0.01, 25)
        with patch.object(osm, "load_osm_geoms_with_status", side_effect=empty_layer):
            result = osm.compute_osm_bands(dem, transform, (120, 24.97, 120.03, 25))

        self.assertEqual(result.metadata["layers"]["buildings"]["status"], "missing_file")
        self.assertEqual(result.metadata["layers"]["protected_areas"]["planning_role"], "informational_only")
        self.assertTrue(all(isinstance(value, np.ndarray) for value in result.values()))
        self.assertLess(requested["taiwan_buildings.pkl"][1], 24.93)
        self.assertGreater(requested["taiwan_buildings.pkl"][3], 25.04)
        self.assertEqual(requested["taiwan_landuse.pkl"], (120, 24.97, 120.03, 25))

    def test_a_star_mode_loads_only_cost_and_hard_mask_layers(self):
        paths = []
        requested = {}

        def empty_layer(path, _geom_type, bbox):
            paths.append(Path(path).name)
            requested[Path(path).name] = bbox
            return [], {"status": "empty_in_bbox", "geometry_count_in_bbox": 0}

        dem = np.zeros((3, 3), dtype=np.float32)
        transform = Affine(0.01, 0, 120, 0, -0.01, 25)
        with patch.object(osm, "load_osm_geoms_with_status", side_effect=empty_layer):
            result = osm.compute_osm_bands(
                dem, transform, (120, 24.97, 120.03, 25),
                include_model_features=False, include_building_distance=False)

        self.assertCountEqual(paths, ["taiwan_landuse.pkl", "taiwan_buildings.pkl"])
        self.assertIn("landuse_code", result)
        self.assertIn("residential_landuse_mask", result)
        self.assertIn("building_density", result)
        self.assertIn("building_mask", result)
        self.assertNotIn("dist_road", result)
        self.assertNotIn("building_distance_m", result)
        self.assertEqual(result.metadata["layers"]["roads"]["status"], "not_loaded")
        self.assertAlmostEqual(requested["taiwan_buildings.pkl"][1], 24.952, places=3)

    def test_a_star_core_bands_match_full_feature_mode(self):
        landuse = [
            (box(120.002, 24.992, 120.012, 25.0), {"landuse": "commercial"}),
            (box(120.012, 24.992, 120.022, 25.0), {"landuse": "residential"}),
        ]
        buildings = [(box(120.002, 24.992, 120.007, 24.997), {})]

        def fixture(path, _geom_type, _bbox):
            name = Path(path).name
            features = landuse if name == "taiwan_landuse.pkl" else buildings if name == "taiwan_buildings.pkl" else []
            return features, {"status": "available" if features else "empty_in_bbox",
                              "geometry_count_in_bbox": len(features)}

        dem = np.arange(9, dtype=np.float32).reshape(3, 3)
        transform = Affine(0.01, 0, 120, 0, -0.01, 25)
        terrain = {"slope": np.full((3, 3), 5.0),
                   "roughness_27": np.full((3, 3), 10.0)}
        with patch.object(osm, "load_osm_geoms_with_status", side_effect=fixture):
            full = osm.compute_osm_bands(dem, transform, (120, 24.97, 120.03, 25))
            astar = osm.compute_osm_bands(
                dem, transform, (120, 24.97, 120.03, 25),
                include_model_features=False, include_building_distance=False)

        for name in ("landuse_code", "residential_landuse_mask", "building_density", "building_mask"):
            np.testing.assert_array_equal(astar[name], full[name])
        np.testing.assert_array_equal(
            build_route_cost(terrain, astar), build_route_cost(terrain, full))

    def test_compute_osm_bands_uses_explicit_cache_directory(self):
        paths = []

        def empty_layer(path, _geom_type, _bbox):
            paths.append(Path(path))
            return [], {"status": "missing_file", "geometry_count_in_bbox": 0}

        cache_dir = Path(self.temp_dir.name) / "selected-cache"
        dem = np.zeros((2, 2), dtype=np.float32)
        transform = Affine(0.01, 0, 120, 0, -0.01, 25)
        with patch.object(osm, "load_osm_geoms_with_status", side_effect=empty_layer):
            osm.compute_osm_bands(dem, transform, (120, 24.98, 120.02, 25), cache_dir=cache_dir)

        self.assertEqual(len(paths), 8)
        self.assertTrue(all(path.parent == cache_dir for path in paths))
        self.assertIn(cache_dir / "taiwan_buildings.pkl", paths)
        self.assertIn(cache_dir / "taiwan_landuse.pkl", paths)

    def test_residential_landuse_contributes_soft_cost_without_becoming_a_hard_mask(self):
        residential = box(0.1, 1.1, 1.9, 2.9)
        industrial = box(2.1, 0.1, 2.9, 0.9)
        transform = Affine(1, 0, 0, 0, -1, 3)
        mask = osm.compute_residential_landuse_mask([
            (residential, {"landuse": "residential"}),
            (industrial, {"landuse": "industrial"}),
        ], transform, (3, 3))
        terrain = {"slope": np.zeros((3, 3)), "roughness_27": np.zeros((3, 3))}
        cost = build_route_cost(terrain, {
            "building_density": np.zeros((3, 3)),
            "residential_landuse_mask": mask,
        })

        self.assertEqual(mask[0, 0], 1)
        self.assertEqual(mask[2, 2], 0)
        self.assertGreater(cost[0, 0], cost[2, 2])

    def test_built_up_landuse_code_contributes_soft_cost_without_double_counting(self):
        terrain = {"slope": np.zeros((2, 2)), "roughness_27": np.zeros((2, 2))}
        landuse_code = osm.compute_landuse_code([
            (box(0.1, 1.1, 0.9, 1.9), {"landuse": "commercial"}),
        ], Affine(1, 0, 0, 0, -1, 2), (2, 2))
        self.assertEqual(int(landuse_code[0, 0]), 4)

        landuse_only = build_route_cost(terrain, {"landuse_code": landuse_code})
        self.assertEqual(float(landuse_only[0, 0]), 2.0)
        self.assertEqual(float(landuse_only[1, 1]), 0.0)
        self.assertEqual(float(build_route_cost(terrain, {})[0, 0]), 0.0)
        nan_landuse = build_route_cost(terrain, {"landuse_code": np.full((2, 2), np.nan)})
        self.assertEqual(float(nan_landuse[0, 0]), 0.0)

        overlapping = build_route_cost(terrain, {
            "landuse_code": landuse_code,
            "residential_landuse_mask": landuse_code == 4,
            "building_density": np.array([[5.0, 0.0], [0.0, 0.0]]),
        })
        self.assertEqual(float(overlapping[0, 0]), float(landuse_only[0, 0]))

    def test_mapped_forest_and_wetland_codes_feed_the_soft_cost(self):
        geoms = [
            (box(0.1, 1.1, 0.9, 1.9), {"natural": "wood"}),
            (box(1.1, 1.1, 1.9, 1.9), {"natural": "wetland"}),
        ]
        landuse_code = osm.compute_landuse_code(
            geoms, Affine(1, 0, 0, 0, -1, 2), (2, 2))
        terrain = {"slope": np.zeros((2, 2)), "roughness_27": np.zeros((2, 2))}

        self.assertEqual(int(landuse_code[0, 0]), 1)
        self.assertEqual(int(landuse_code[0, 1]), 7)
        cost = build_route_cost(
            terrain, {"landuse_code": landuse_code}, landcover_penalty_weight=0.10)
        self.assertAlmostEqual(float(cost[0, 0]), 0.10)
        self.assertAlmostEqual(float(cost[0, 1]), 0.10)

    def test_legacy_wrapped_tag_strings_are_parsed_and_failures_are_counted(self):
        cache = self._save_cache("legacy_landuse.pkl", [
            _way([(0.1, 1.1), (0.9, 1.1), (0.9, 1.9), (0.1, 1.9)],
                 {"tags": "{'landuse': 'commercial'}"}),
            _way([(1.1, 1.1), (1.9, 1.1), (1.9, 1.9), (1.1, 1.9)],
                 {"tags": "{'landuse': 'residential'}"}),
            _way([(2.1, 1.1), (2.9, 1.1), (2.9, 1.9), (2.1, 1.9)],
                 {"tags": "not a dictionary"}),
        ])
        geoms, metadata = osm.load_osm_geoms_with_status(
            cache, "polygon", (0, 0, 3, 2))

        self.assertEqual(metadata["tag_wrapper_count"], 3)
        self.assertEqual(metadata["tag_parse_error_count"], 1)
        commercial = osm.compute_landuse_code(geoms[:1], Affine(1, 0, 0, 0, -1, 2), (2, 3))
        residential = osm.compute_residential_landuse_mask(
            geoms, Affine(1, 0, 0, 0, -1, 2), (2, 3))
        terrain = {"slope": np.zeros((2, 3)), "roughness_27": np.zeros((2, 3))}
        cost = build_route_cost(terrain, {"landuse_code": commercial})

        self.assertEqual(int(commercial[0, 0]), 4)
        self.assertEqual(int(residential[0, 1]), 1)
        self.assertEqual(int(residential[0, 2]), 0)
        self.assertGreater(float(cost[0, 0]), 0.0)

    def test_user_geojson_constraints_rasterize_hard_and_soft_polygons(self):
        features = [
            {"type": "Feature", "properties": {"mode": "hard", "source": "official zoning",
                                                   "category": "no_build"},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]}},
            {"type": "Feature", "properties": {"mode": "soft", "source": "survey", "weight": 0.75,
                                                   "category": "settlement"},
             "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]]]}},
            {"type": "Feature", "properties": {"mode": "soft", "source": "planning brief", "weight": 0.25},
             "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]]] }},
            {"type": "Feature", "properties": {"mode": "soft", "source": "zero weight", "weight": 0},
             "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]] }},
            {"type": "Feature", "properties": {"mode": "hard", "source": "outside extent"},
             "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [10, 12], [12, 12], [12, 10], [10, 10]]]}}]
        path = Path(self.temp_dir.name) / "constraints.geojson"
        dataset_metadata = {"generated_at": "2026-10-22T00:00:00+00:00", "feature_count": len(features)}
        data_sources = [{"dataset_id": "demo", "license": "open"}]
        path.write_text(json.dumps({"type": "FeatureCollection", "metadata": dataset_metadata,
                                    "data_sources": data_sources, "features": features}), encoding="utf-8")

        result = osm.load_user_constraint_overlays(path, Affine(1, 0, 0, 0, -1, 4), (4, 4))

        self.assertGreater(result["hard_mask"].sum(), 0)
        self.assertAlmostEqual(float(result["soft_cost"].max()), 1.0)
        self.assertEqual(result["metadata"]["hard_feature_count"], 2)
        self.assertEqual(result["metadata"]["soft_feature_count"], 3)
        self.assertEqual(result["metadata"]["sources"],
                         ["official zoning", "outside extent", "planning brief", "survey", "zero weight"])
        source_layers = {
            (layer["source"], layer["mode"]): layer
            for layer in result["metadata"]["source_layers"]
        }
        self.assertGreater(source_layers[("official zoning", "hard")]["affected_cell_count"], 0)
        self.assertGreater(source_layers[("survey", "soft")]["covered_cell_count"], 0)
        self.assertGreater(source_layers[("zero weight", "soft")]["covered_cell_count"], 0)
        self.assertEqual(source_layers[("zero weight", "soft")]["affected_cell_count"], 0)
        self.assertEqual(source_layers[("outside extent", "hard")]["covered_cell_count"], 0)
        self.assertEqual(source_layers[("outside extent", "hard")]["affected_cell_count"], 0)
        self.assertEqual(result["metadata"]["dataset_metadata"], dataset_metadata)
        self.assertEqual(result["metadata"]["data_sources"], data_sources)

    def test_user_geojson_soft_feature_requires_source_and_weight(self):
        feature = {"type": "Feature", "properties": {"mode": "soft"},
                   "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}}
        path = Path(self.temp_dir.name) / "invalid.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "source"):
            osm.load_user_constraint_overlays(path, Affine.identity(), (2, 2))

    def test_tower_selection_requires_declared_hard_constraint_evidence(self):
        geometry = {"type": "Polygon", "coordinates":
                    [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
        verified = {"type": "Feature", "properties": {
            "mode": "hard", "source": "surveyed permit boundary",
            "verification_status": "verified",
            "verification_reference": "survey-register-2026-01",
        }, "geometry": geometry}
        path = Path(self.temp_dir.name) / "verified-constraints.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [verified]}),
                        encoding="utf-8")

        result = osm.load_user_constraint_overlays(
            path, Affine.identity(), (2, 2),
            require_verified_hard_constraints=True)
        self.assertTrue(result["metadata"]["hard_constraint_declarations_complete"])
        self.assertEqual(result["metadata"]["hard_constraint_evidence_validation_scope"],
                         "caller_declarations_only")
        self.assertEqual(result["metadata"]["hard_constraint_evidence"][0][
            "verification_reference"], "survey-register-2026-01")

        unverified = {**verified, "properties": {
            **verified["properties"], "verification_status": "unknown"}}
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [unverified]}),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification_status=verified"):
            osm.load_user_constraint_overlays(
                path, Affine.identity(), (2, 2),
                require_verified_hard_constraints=True)

        missing_reference = {**verified, "properties": {
            **verified["properties"], "verification_reference": ""}}
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [missing_reference]}),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification_status=verified"):
            osm.load_user_constraint_overlays(
                path, Affine.identity(), (2, 2),
                require_verified_hard_constraints=True)

        soft_only = {"type": "Feature", "properties": {
            "mode": "soft", "source": "survey", "weight": 1.0},
            "geometry": geometry}
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [soft_only]}),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "至少需要一个"):
            osm.load_user_constraint_overlays(
                path, Affine.identity(), (2, 2),
                require_verified_hard_constraints=True)

        outside = {**verified, "geometry": {"type": "Polygon", "coordinates":
                   [[[10, 10], [10, 11], [11, 11], [11, 10], [10, 10]]]}}
        path.write_text(json.dumps({"type": "FeatureCollection", "features": [outside]}),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "没有覆盖当前规划网格"):
            osm.load_user_constraint_overlays(
                path, Affine.identity(), (2, 2),
                require_verified_hard_constraints=True)


if __name__ == "__main__":
    unittest.main()
