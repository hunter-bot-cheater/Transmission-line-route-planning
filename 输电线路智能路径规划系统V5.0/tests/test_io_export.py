import json
import tempfile
import unittest
from pathlib import Path

from rasterio.crs import CRS

from transmission_path_planner import io_export


class RouteExportTests(unittest.TestCase):
    def test_exports_geojson_even_when_shapefile_succeeds(self):
        with tempfile.TemporaryDirectory() as folder:
            shp_path = Path(folder) / "route.shp"
            geojson_path = Path(folder) / "route.geojson"

            result = io_export.export_route(
                [(121.0, 24.0), (121.1, 24.1)], "EPSG:4326", shp_path, geojson_path
            )

            self.assertTrue(geojson_path.is_file())
            document = json.loads(geojson_path.read_text(encoding="utf-8"))
            self.assertEqual(
                document["features"][0]["geometry"]["coordinates"],
                [[121.0, 24.0], [121.1, 24.1]],
            )
            if io_export.HAS_GPD:
                self.assertTrue(shp_path.is_file())
                self.assertEqual(result, shp_path)
            else:
                self.assertEqual(result, geojson_path)

    def test_multiline_exports_use_the_same_valid_segments(self):
        segments = [[(121.0, 24.0)], [(122.0, 25.0), (122.1, 25.1)]]
        with tempfile.TemporaryDirectory() as folder:
            shp_path = Path(folder) / "routes.shp"
            geojson_path = Path(folder) / "routes.geojson"

            io_export.export_route(segments, "EPSG:4326", shp_path, geojson_path)

            document = json.loads(geojson_path.read_text(encoding="utf-8"))
            self.assertEqual(len(document["features"]), 1)
            self.assertEqual(document["features"][0]["properties"]["name"], "seg0")
            self.assertEqual(
                document["features"][0]["geometry"]["coordinates"],
                [[122.0, 25.0], [122.1, 25.1]],
            )
            if io_export.HAS_GPD:
                frame = io_export.gpd.read_file(shp_path)
                self.assertEqual(len(frame), 1)
                self.assertEqual(list(frame.geometry.iloc[0].coords), [(122.0, 25.0), (122.1, 25.1)])

    def test_empty_multiline_has_no_export(self):
        with tempfile.TemporaryDirectory() as folder:
            shp_path = Path(folder) / "routes.shp"
            geojson_path = Path(folder) / "routes.geojson"

            result = io_export.export_route([[]], "EPSG:4326", shp_path, geojson_path)

            self.assertIsNone(result)
            self.assertFalse(shp_path.exists())
            self.assertFalse(geojson_path.exists())

    def test_crs_object_is_serialized_in_geojson(self):
        with tempfile.TemporaryDirectory() as folder:
            shp_path = Path(folder) / "route.shp"
            geojson_path = Path(folder) / "route.geojson"

            io_export.export_route(
                [(121.0, 24.0), (121.1, 24.1)], CRS.from_epsg(4326), shp_path, geojson_path
            )

            document = json.loads(geojson_path.read_text(encoding="utf-8"))
            self.assertEqual(document["crs"]["properties"]["name"], "EPSG:4326")

    def test_non_finite_coordinates_are_not_exported(self):
        for invalid_coordinate in (float("nan"), float("inf"), float("-inf"), "not-a-number", None):
            with self.subTest(invalid_coordinate=invalid_coordinate):
                with tempfile.TemporaryDirectory() as folder:
                    shp_path = Path(folder) / "route.shp"
                    geojson_path = Path(folder) / "route.geojson"

                    result = io_export.export_route(
                        [(invalid_coordinate, 24.0), (121.1, 24.1)],
                        "EPSG:4326",
                        shp_path,
                        geojson_path,
                    )

                    self.assertIsNone(result)
                    self.assertFalse(geojson_path.exists())
                    self.assertFalse(shp_path.exists())
                    self.assertEqual(list(Path(folder).iterdir()), [])

    def test_malformed_coordinate_points_are_not_exported(self):
        routes = (
            ["12", "34"],
            [{"lon": 121.0, "lat": 24.0}, (121.1, 24.1)],
            [(121.0, 24.0, 5.0), (121.1, 24.1, 5.0)],
            [("121.0", "24.0"), ("121.1", "24.1")],
            [[(121.0, 24.0), (121.1, 24.1)], [(float("nan"), 25.0), (122.1, 25.1)]],
        )
        for route in routes:
            with self.subTest(route=route):
                with tempfile.TemporaryDirectory() as folder:
                    shp_path = Path(folder) / "route.shp"
                    geojson_path = Path(folder) / "route.geojson"

                    result = io_export.export_route(route, "EPSG:4326", shp_path, geojson_path)

                    self.assertIsNone(result)
                    self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
