import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from transmission_path_planner.cli import build_tower_masks, validate_endpoint_cells
from transmission_path_planner.config import HAS_RASTERIO, HAS_SHAPELY
from transmission_path_planner.osm import load_land_mask_geojson


@unittest.skipUnless(HAS_RASTERIO and HAS_SHAPELY, "requires rasterio and shapely")
class LandMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "land.geojson"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_geojson(self, feature, crs=None):
        document = {"type": "FeatureCollection", "features": [feature]}
        if crs:
            document["crs"] = crs
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_land_polygons_become_allowed_cells_and_block_outside_endpoints(self):
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [1, 1], [3, 1], [3, 3], [1, 3], [1, 1],
            ]]},
        }
        self.write_geojson(feature)

        result = load_land_mask_geojson(
            self.path, Affine(1, 0, 0, 0, -1, 4), (4, 4))
        hard_mask = (result["land_mask"] == 0).astype(np.uint8)
        span_mask, _ = build_tower_masks(
            np.zeros((4, 4)), np.zeros((4, 4)), custom_hard_mask=hard_mask)

        self.assertTrue(span_mask[1, 1])
        self.assertFalse(span_mask[0, 1])
        self.assertFalse(span_mask[0, 3])
        with self.assertRaisesRegex(ValueError, "陆域范围外"):
            validate_endpoint_cells((0, 3), (1, 1), (4, 4), span_mask,
                                    {"陆域范围外": result["land_mask"] == 0})
        self.assertEqual(result["metadata"]["status"], "loaded")

    def test_land_mask_rejects_projected_crs_and_nonoverlapping_data(self):
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [10, 10], [11, 10], [11, 11], [10, 11], [10, 10],
            ]]},
        }
        self.write_geojson(feature, {"type": "name", "properties": {"name": "EPSG:3857"}})
        with self.assertRaisesRegex(ValueError, "WGS84"):
            load_land_mask_geojson(
                self.path, Affine(1, 0, 0, 0, -1, 4), (4, 4))

        self.write_geojson(feature)
        with self.assertRaisesRegex(ValueError, "没有覆盖"):
            load_land_mask_geojson(
                self.path, Affine(1, 0, 0, 0, -1, 4), (4, 4))

    def test_nested_multipolygon_components_are_dissolved_for_land_coverage(self):
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "MultiPolygon", "coordinates": [
                [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
                [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]],
            ]},
        }
        self.write_geojson(feature)

        result = load_land_mask_geojson(
            self.path, Affine(1, 0, 0, 0, -1, 4), (4, 4))

        self.assertEqual(result["metadata"]["repaired_feature_count"], 1)
        self.assertEqual(result["metadata"]["repaired_feature_indices"], [1])
        self.assertTrue(np.all(result["land_mask"]))


if __name__ == "__main__":
    unittest.main()
