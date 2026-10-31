import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "transmission_path_planner" / "official_hazards.py")
_SPEC = importlib.util.spec_from_file_location("official_hazards_under_test", _MODULE_PATH)
hazards = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hazards)


class OfficialHazardMaskTests(unittest.TestCase):
    bbox = (0.0, 0.0, 1.0, 1.0)
    zoom = 4

    def _fixture(self, root, *, failed_layer=None, expected_but_missing=False,
                 nonempty_but_missing=False):
        tiles = hazards._tiles_for_bbox(self.bbox, self.zoom)
        self.assertEqual(len(tiles), 1)
        x, y = tiles[0]
        layer_docs = {}
        checksum_lines = []
        for key, layer_name in hazards.LAYER_SPECS.items():
            layer_dir = root / layer_name / str(self.zoom) / str(y)
            layer_dir.mkdir(parents=True, exist_ok=True)
            if key == failed_layer:
                failure = f"{layer_name}/{self.zoom}/{y}/{x}.pbf"
                layer_docs[layer_name] = {
                    "tiles_requested": 1, "nonempty_tiles": 0,
                    "empty_tiles": 0, "failed_tiles": [failure],
                    "description": layer_name,
                }
            elif key == "landslide_potential":
                layer_docs[layer_name] = {
                    "tiles_requested": 1,
                    "nonempty_tiles": 1 if nonempty_but_missing else 0,
                    "empty_tiles": 0 if nonempty_but_missing else 1,
                    "failed_tiles": [],
                    "description": layer_name,
                }
                if expected_but_missing:
                    checksum_lines.append(
                        f"{'0' * 64}  {layer_name}/{self.zoom}/{y}/{x}.pbf")
            else:
                path = layer_dir / f"{x}.pbf"
                path.write_bytes(f"mock-{key}".encode("ascii"))
                checksum_lines.append(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                    f"{layer_name}/{self.zoom}/{y}/{x}.pbf")
                layer_docs[layer_name] = {
                    "tiles_requested": 1, "nonempty_tiles": 1,
                    "empty_tiles": 0, "failed_tiles": [],
                    "description": layer_name,
                }

        (root / "tiles.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        manifest = {
            "source": "official test source",
            "downloaded_at_utc": "2026-10-23T00:00:00+00:00",
            "zoom": self.zoom,
            "bbox_wgs84": {"west": 0, "south": 0, "east": 1, "north": 1},
            "tile_x_range_inclusive": [x, x],
            "tile_y_range_inclusive": [y, y],
            "layers": layer_docs,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return x, y

    def test_requested_tiles_union_masks_and_preserve_empty_vs_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, failed_layer="landslide_impact")
            written = []

            def rasterize(geometries, _transform, shape):
                written.append(list(geometries))
                mask = np.zeros(shape, dtype=np.uint8)
                if geometries:
                    mask[0, 0] = 1
                return mask

            with patch.object(hazards, "_read_tile_geometries", return_value=["fragment", "fragment"]), \
                    patch.object(hazards, "_rasterize_union", side_effect=rasterize):
                masks, metadata = hazards.load_official_hazard_masks(
                    root, self.bbox, transform="test-transform", shape=(3, 4))

        self.assertEqual(set(masks), set(hazards.LAYER_SPECS))
        self.assertEqual(masks["debris_flow_impact"].dtype, np.uint8)
        self.assertEqual(int(masks["debris_flow_impact"].sum()), 1)
        self.assertEqual(len(written[0]), 2)  # duplicate tile fragments merge in raster union
        self.assertEqual(int(masks["landslide_potential"].sum()), 0)
        self.assertEqual(int(masks["landslide_impact"].sum()), 0)
        self.assertEqual(metadata["layers"]["landslide_potential"]["status"], "official_empty")
        self.assertEqual(metadata["layers"]["landslide_potential"]["coverage_status"], "complete")
        self.assertEqual(metadata["layers"]["landslide_impact"]["tiles"]["failed"], 1)
        self.assertEqual(metadata["coverage_status"], "partial")
        self.assertIn("soft_only", metadata["risk_mode"])

    def _vector_fixture(self, root, *, include_address=False,
                        geometry_type="Polygon", feature_count=1):
        sanitized_files = []
        frames = {}
        layer_info = {}
        read_calls = []

        class CRS:
            @staticmethod
            def to_epsg():
                return 3826

        class GeometrySeries(list):
            @property
            def is_valid(self):
                return np.asarray([geometry.is_valid for geometry in self])

        class Frame:
            crs = CRS()

            def __init__(self, geometry, columns):
                self.geometry = GeometrySeries([geometry])
                self.columns = columns

            def __len__(self):
                return len(self.geometry)

            def to_crs(self, _crs):
                return self

        from shapely.geometry import box

        for index, (layer, (filename, _dataset_id)) in enumerate(
                hazards.VECTOR_LAYER_SPECS.items()):
            archive = root / filename
            archive.write_bytes(f"test-source-{layer}".encode("ascii"))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            sanitized_files.append({
                "name": filename,
                "sha256": digest,
                "feature_count": feature_count,
                "address_field_removed": True,
            })
            fields = ["Risk"]
            if include_address and index == 0:
                fields.insert(0, "Address")
            frames[filename] = Frame(box(0.1, 0.1, 0.8, 0.8), ["geometry"])
            layer_info[filename] = {
                "fields": fields,
                "features": feature_count,
                "crs": "EPSG:3826",
                "geometry_type": geometry_type,
            }

        (root / "manifest.json").write_text(
            json.dumps({
                "sanitized_files": sanitized_files,
                "address_values_read_or_emitted": False,
            }), encoding="utf-8")

        def read_file(path, *, columns, engine):
            read_calls.append((columns, engine))
            return frames[Path(path.split("/")[-1]).name]

        gpd_stub = SimpleNamespace(read_file=read_file)
        pyogrio_stub = SimpleNamespace(
            read_info=lambda path: layer_info[Path(path.split("/")[-1]).name])
        return gpd_stub, pyogrio_stub, read_calls

    def test_sanitized_vector_layers_are_verified_and_rasterized_as_soft_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpd_stub, pyogrio_stub, read_calls = self._vector_fixture(root)

            def rasterize(geometries, _transform, shape):
                mask = np.zeros(shape, dtype=np.uint8)
                if list(geometries):
                    mask[0, 0] = 1
                return mask

            with patch.dict("sys.modules", {
                    "geopandas": gpd_stub, "pyogrio": pyogrio_stub}), \
                    patch.object(hazards, "_rasterize_union", side_effect=rasterize):
                masks, metadata = hazards.load_official_hazard_vector_masks(
                    root, self.bbox, transform="test-transform", shape=(3, 4))

        self.assertEqual(set(masks), set(hazards.VECTOR_LAYER_SPECS))
        self.assertTrue(all(mask[0, 0] == 1 for mask in masks.values()))
        self.assertEqual(metadata["source_type"], "sanitized_shapefile")
        self.assertEqual(metadata["coverage_status"], "unknown")
        self.assertEqual(metadata["source_integrity_status"],
                         "validated_against_local_manifest")
        self.assertEqual(metadata["spatial_coverage_status"],
                         "not_independently_verified")
        self.assertFalse(metadata["address_values_read_or_emitted"])
        self.assertEqual(read_calls, [([], "pyogrio")] * len(hazards.VECTOR_LAYER_SPECS))
        self.assertTrue(all(item["hard_exclusion_allowed"] is False
                            for item in metadata["layers"].values()))
        self.assertTrue(all(item["coverage_status"] == "unknown"
                            for item in metadata["layers"].values()))

    def test_vector_loader_rejects_an_address_field_even_if_manifest_claims_sanitized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpd_stub, pyogrio_stub, read_calls = self._vector_fixture(
                root, include_address=True)
            with patch.dict("sys.modules", {
                    "geopandas": gpd_stub, "pyogrio": pyogrio_stub}):
                with self.assertRaisesRegex(ValueError, "地址字段"):
                    hazards.load_official_hazard_vector_masks(
                        root, self.bbox, transform="test-transform", shape=(2, 2))
            self.assertEqual(read_calls, [])

    def test_vector_loader_rejects_nonpolygon_archive_before_geometry_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpd_stub, pyogrio_stub, read_calls = self._vector_fixture(
                root, geometry_type="Point")
            with patch.dict("sys.modules", {
                    "geopandas": gpd_stub, "pyogrio": pyogrio_stub}):
                with self.assertRaisesRegex(ValueError, "仅包含面几何"):
                    hazards.load_official_hazard_vector_masks(
                        root, self.bbox, transform="test-transform", shape=(2, 2))
            self.assertEqual(read_calls, [])

    def test_vector_loader_rejects_symlink_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_name = next(iter(hazards.VECTOR_LAYER_SPECS.values()))[0]
            gpd_stub, pyogrio_stub, _ = self._vector_fixture(root)
            original_is_symlink = Path.is_symlink

            def is_target_symlink(path):
                return path.name == target_name or original_is_symlink(path)

            with patch.dict("sys.modules", {
                    "geopandas": gpd_stub, "pyogrio": pyogrio_stub}), \
                    patch.object(hazards.Path, "is_symlink", is_target_symlink):
                with self.assertRaisesRegex(ValueError, "不能是符号链接"):
                    hazards.load_official_hazard_vector_masks(
                        root, self.bbox, transform="test-transform", shape=(2, 2))

    def test_missing_source_returns_zero_masks_and_unknown_coverage(self):
        missing = Path(tempfile.gettempdir()) / "definitely-missing-official-hazard-source"
        masks, metadata = hazards.load_official_hazard_masks(
            missing, self.bbox, transform=None, shape=(2, 3))
        self.assertTrue(all(mask.shape == (2, 3) and not mask.any() for mask in masks.values()))
        self.assertEqual(metadata["status"], "missing_source")
        self.assertEqual(metadata["coverage_status"], "unknown")

    def test_manifestless_absent_tile_is_unknown_not_official_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            masks, metadata = hazards.load_official_hazard_masks(
                temporary, self.bbox, transform=None, shape=(2, 2))
        self.assertTrue(all(not mask.any() for mask in masks.values()))
        self.assertEqual(metadata["status"], "no_manifest")
        self.assertEqual(metadata["coverage_status"], "unknown")

    def test_tile_expected_by_checksum_but_missing_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, expected_but_missing=True)
            with patch.object(hazards, "_read_tile_geometries", return_value=[]):
                _, metadata = hazards.load_official_hazard_masks(
                    root, self.bbox, transform=None, shape=(2, 2))
        layer = metadata["layers"]["landslide_potential"]
        self.assertEqual(layer["tiles"]["unknown"], 1)
        self.assertEqual(layer["tiles"]["official_empty"], 0)
        self.assertEqual(layer["coverage_status"], "partial")

    def test_manifest_nonempty_tile_missing_from_file_and_checksums_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root, nonempty_but_missing=True)
            with patch.object(hazards, "_read_tile_geometries", return_value=[]):
                _, metadata = hazards.load_official_hazard_masks(
                    root, self.bbox, transform=None, shape=(2, 2))

        layer = metadata["layers"]["landslide_potential"]
        self.assertEqual(layer["tiles"]["checksummed_nonempty"], 0)
        self.assertFalse(layer["manifest_checksum_count_match"])
        self.assertEqual(layer["tiles"]["unknown"], 1)
        self.assertEqual(layer["tiles"]["official_empty"], 0)
        self.assertEqual(layer["coverage_status"], "partial")
        self.assertEqual(metadata["coverage_status"], "partial")

    def test_actual_bundled_tile_is_read_by_gdal_mvt_driver(self):
        tile = (Path(__file__).resolve().parents[1] / "data" / "official_sources"
                / "ardswc_115_mvt_z10_20261023" / "DEBRIS1753_20260126"
                / "10" / "437" / "857.pbf")
        if not tile.is_file():
            self.skipTest("official sample MVT tile is not present in this checkout")
        try:
            import pyogrio
        except ImportError:
            self.skipTest("pyogrio/GDAL is not installed")

        info = pyogrio.read_info(tile, layer=0)
        meta, _, geometries, _ = pyogrio.raw.read(tile, layer=0, columns=[])
        self.assertEqual(info["driver"], "MVT")
        self.assertEqual(meta["crs"], "EPSG:3857")
        self.assertTrue(len(geometries) > 0)

    def test_actual_z_y_x_tile_is_geolocated_and_rasterized_in_taiwan(self):
        source = (Path(__file__).resolve().parents[1] / "data" / "official_sources"
                  / "ardswc_115_mvt_z10_20261023")
        tile_x, tile_y, zoom = 857, 437, 10
        n = 1 << zoom

        def tile_latitude(y):
            return np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))

        bbox = (tile_x / n * 360 - 180,
                tile_latitude(tile_y + 1),
                (tile_x + 1) / n * 360 - 180,
                tile_latitude(tile_y))
        tile = (source / "DEBRIS1753_20260126" / str(zoom)
                / str(tile_y) / f"{tile_x}.pbf")
        if not tile.is_file():
            self.skipTest("official sample MVT tile is not present in this checkout")
        try:
            from rasterio.transform import from_bounds
        except ImportError:
            self.skipTest("rasterio is not installed")

        geometries = hazards._read_tile_geometries(tile, bbox, tile_x, tile_y, zoom)
        self.assertGreater(len(geometries), 0)
        self.assertTrue(all(geometry.bounds[0] >= bbox[0] - 1e-8
                            and geometry.bounds[2] <= bbox[2] + 1e-8
                            and geometry.bounds[1] >= bbox[1] - 1e-8
                            and geometry.bounds[3] <= bbox[3] + 1e-8
                            for geometry in geometries))

        masks, metadata = hazards.load_official_hazard_masks(
            source, bbox, from_bounds(*bbox, 256, 256), (256, 256))
        self.assertTrue(masks["debris_flow_impact"].any())
        layer = metadata["layers"]["debris_flow_impact"]
        self.assertEqual(layer["coverage_status"], "complete")
        self.assertEqual(layer["tile_records"][0]["status"], "read")
        self.assertEqual(metadata["coverage_status"], "complete")


if __name__ == "__main__":
    unittest.main()
