import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from transmission_path_planner.rebuild_terrarium_dem import rebuild_dem
from transmission_path_planner.terrarium import (
    WEB_MERCATOR_RADIUS_M,
    terrarium_rgb_to_elevation,
    xyz_pixel_to_lonlat,
    xyz_tile_transform,
)


class TerrariumGeoreferencingTests(unittest.TestCase):
    def test_terrarium_rgb_decodes_fractional_elevation(self):
        rgb = np.array([[[128, 123, 128]]], dtype=np.uint8)

        self.assertAlmostEqual(float(terrarium_rgb_to_elevation(rgb)[0, 0]), 123.5)

    def test_xyz_tile_transform_uses_web_mercator_pixel_rows(self):
        zoom, tile_x, tile_y, pixel_x, pixel_y = 12, 3429, 1773, 42, 50
        transform = xyz_tile_transform(zoom, tile_x, tile_y)
        x, y = transform * (pixel_x + 0.5, pixel_y + 0.5)
        n = (2 ** zoom) * 256
        expected_lon = ((tile_x * 256 + pixel_x + 0.5) / n) * 360.0 - 180.0
        mercator_y = math.pi * (1.0 - 2.0 * (tile_y * 256 + pixel_y + 0.5) / n)
        expected_lat = math.degrees(math.atan(math.sinh(mercator_y)))

        self.assertAlmostEqual(x / WEB_MERCATOR_RADIUS_M,
                               math.radians(expected_lon), places=12)
        self.assertAlmostEqual(y / WEB_MERCATOR_RADIUS_M,
                               mercator_y, places=12)
        self.assertAlmostEqual(xyz_pixel_to_lonlat(
            zoom, tile_x, tile_y, pixel_x, pixel_y)[1], expected_lat, places=10)

    def test_terrarium_rgb_rejects_non_image_shapes(self):
        with self.assertRaises(ValueError):
            terrarium_rgb_to_elevation(np.zeros((2, 3), dtype=np.uint8))

    def test_rebuild_dem_reprojects_xyz_rows_before_geographic_sampling(self):
        zoom, tile_x, tile_y = 4, 8, 8
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = root / "tiles"
            cache.mkdir()
            rows = np.arange(256, dtype=np.uint8)[:, None]
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb[..., 0] = 128
            rgb[..., 1] = rows
            Image.fromarray(rgb, mode="RGB").save(
                cache / f"t_{zoom}_{tile_x}_{tile_y}.png")

            output = root / "reprojected.tif"
            rebuild_dem(cache, output, (10, -13, 13, -9), zoom=zoom)
            sample_lon, sample_lat = xyz_pixel_to_lonlat(
                zoom, tile_x, tile_y, 128, 128)

            with rasterio.open(output) as dem:
                sampled_elevation = next(dem.sample([(sample_lon, sample_lat)]))[0]
                self.assertEqual(dem.crs.to_epsg(), 4326)
                self.assertEqual(dem.tags()["source_crs"], "EPSG:3857")
                self.assertAlmostEqual(float(sampled_elevation), 128.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
