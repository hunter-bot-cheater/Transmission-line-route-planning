import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_taiwan_constraints.py"
SPEC = importlib.util.spec_from_file_location("fetch_taiwan_constraints", SCRIPT_PATH)
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


class TaiwanConstraintFetcherTests(unittest.TestCase):
    def test_selects_latest_moa_shapefile_and_ignores_kml(self):
        records = [
            {"FileName": "nature_2024_SHP", "FileUrl": "https://example.test/old.zip", "CreateDate": "2024-01-01"},
            {"FileName": "nature_2026.kml", "FileUrl": "https://example.test/new.kml", "CreateDate": "2026-06-24"},
            {"FileName": "nature_2025_SHP", "FileUrl": "https://example.test/new.zip", "CreateDate": "2025-06-24"},
        ]

        selected = fetcher.latest_moa_shapefile(records)

        self.assertEqual(selected["FileUrl"], "https://example.test/new.zip")

    def test_selects_latest_moenv_zip(self):
        records = [
            {"filename": "old.zip", "url": "https://example.test/old", "upload_date": "20240603T170512"},
            {"filename": "new.zip", "url": "https://example.test/new", "upload_date": "20260720T153652"},
        ]

        self.assertEqual(fetcher.latest_moenv_zip(records)["url"], "https://example.test/new")

    def test_generated_features_are_soft_and_keep_provenance(self):
        feature = fetcher.make_feature(
            {"type": "Polygon", "coordinates": []}, {"name": "reserve"},
            source="Ministry of Agriculture", category="nature_reserve", weight=1.25,
            dataset_id="24914", updated_at="20260624", layer="reserve",
        )

        self.assertEqual(feature["properties"]["mode"], "soft")
        self.assertEqual(feature["properties"]["weight"], 1.25)
        self.assertEqual(feature["properties"]["dataset_id"], "24914")
        self.assertEqual(feature["properties"]["name"], "reserve")

    def test_rejects_non_finite_or_negative_weight(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                fetcher.validate_weight(weight)

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.shp", "not a shapefile")

            with self.assertRaises(ValueError):
                fetcher._extract_zip(archive_path, root / "extract")

    def test_reprojects_source_polygon_to_wgs84_geojson(self):
        import geopandas as gpd
        from shapely.geometry import Polygon, shape

        with tempfile.TemporaryDirectory() as temporary:
            shapefile = Path(temporary) / "reserve.shp"
            frame = gpd.GeoDataFrame(
                {"NAME": ["sample reserve"]},
                geometry=[Polygon([(300000, 2740000), (300100, 2740000),
                                   (300100, 2740100), (300000, 2740100)])],
                crs="EPSG:3826",
            )
            frame.to_file(shapefile)

            features, layers = fetcher._read_layers(
                [shapefile], agency="Ministry of Agriculture",
                category="nature_reserve", dataset_id="24914",
                updated_at="20260624", weight=1.0,
            )

        geometry = shape(features[0]["geometry"])
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        self.assertTrue(120 <= min_lon < max_lon <= 122)
        self.assertTrue(24 <= min_lat < max_lat <= 26)
        self.assertEqual(features[0]["properties"]["mode"], "soft")
        self.assertEqual(features[0]["properties"]["NAME"], "sample reserve")
        self.assertEqual(layers[0]["source_crs"], "EPSG:3826")

    def test_corrects_oca_geojson_that_declares_crs84_but_contains_twd97(self):
        from shapely.geometry import Polygon, mapping, shape

        polygon = Polygon([(184000, 2660000), (184100, 2660000),
                           (184100, 2660100), (184000, 2660100)])
        document = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": [{"type": "Feature", "properties": {"name": "habitat"},
                          "geometry": mapping(polygon)}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marine.geojson"
            path.write_text(json.dumps(document), encoding="utf-8")
            features, layers = fetcher._read_marine_habitat(path, weight=1.0)

        geometry = shape(features[0]["geometry"])
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        self.assertTrue(120 <= min_lon < max_lon <= 121)
        self.assertTrue(24 <= min_lat < max_lat <= 25)
        self.assertEqual(layers[0]["declared_crs"], "urn:ogc:def:crs:OGC:1.3:CRS84")
        self.assertEqual(layers[0]["source_crs_used"], "EPSG:3826")
        self.assertEqual(features[0]["properties"]["mode"], "soft")


if __name__ == "__main__":
    unittest.main()
