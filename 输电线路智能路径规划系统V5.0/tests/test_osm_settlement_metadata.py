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


if __name__ == "__main__":
    unittest.main()
