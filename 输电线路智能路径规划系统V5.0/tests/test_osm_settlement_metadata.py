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
        def empty_layer(_path, _geom_type, _bbox):
            return [], {"status": "missing_file", "geometry_count_in_bbox": 0}

        dem = np.zeros((3, 3), dtype=np.float32)
        transform = Affine(0.01, 0, 120, 0, -0.01, 25)
        with patch.object(osm, "load_osm_geoms_with_status", side_effect=empty_layer):
            result = osm.compute_osm_bands(dem, transform, (120, 24.97, 120.03, 25))

        self.assertEqual(result.metadata["layers"]["buildings"]["status"], "missing_file")
        self.assertEqual(result.metadata["layers"]["protected_areas"]["planning_role"], "informational_only")
        self.assertTrue(all(isinstance(value, np.ndarray) for value in result.values()))

    def test_a_star_mode_loads_only_cost_and_hard_mask_layers(self):
        paths = []

        def empty_layer(path, _geom_type, _bbox):
            paths.append(Path(path).name)
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


if __name__ == "__main__":
    unittest.main()
