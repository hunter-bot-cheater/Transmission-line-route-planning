import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import MultiPolygon, box, mapping

from transmission_path_planner.cli import (
    annotate_route_validation_context,
    building_validation_metadata,
    open_safe_building_endpoint_pixels,
    validate_route_against_buildings,
)
from transmission_path_planner.osm import (
    OSMBands,
    load_official_buildings_geojson,
    merge_official_buildings_into_bands,
)


class OfficialBuildingLayerTests(unittest.TestCase):
    bbox = (0.0, 0.0, 2.0, 2.0)
    transform = Affine(1, 0, 0, 0, -1, 2)
    grid_shape = (2, 2)

    def _feature(self, geometry, source="NLSC Taiwan e-Map BUILD; source date 2026-08"):
        return {
            "type": "Feature",
            "geometry": mapping(geometry),
            "properties": {
                "mode": "hard",
                "source": source,
                "category": "building_footprint",
                "dataset_id": "NLSC_TAIWAN_EMAP_BUILD",
            },
        }

    def _write_geojson(self, path, *, features=None, crs=None):
        features = features or [self._feature(box(0.2, 0.2, 0.8, 0.8))]
        document = {
            "type": "FeatureCollection",
            "features": features,
            "data_sources": [{
                "agency": "National Land Surveying and Mapping Center",
                "dataset_id": "Taiwan e-Map BUILD",
                "license": "authorized academic use",
            }],
            "metadata": {
                "coordinate_system": "OGC:CRS84 (WGS84 longitude, latitude)",
                "constraint_mode": "hard",
                "feature_count": len(features),
            },
        }
        if crs is not None:
            document["crs"] = crs
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_loader_clips_polygon_and_multipolygon_and_reports_unknown_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "buildings.geojson"
            self._write_geojson(path, features=[
                self._feature(box(0.2, 0.2, 1.2, 1.2)),
                self._feature(MultiPolygon([
                    box(1.2, 1.2, 1.8, 1.8), box(3.0, 3.0, 4.0, 4.0),
                ])),
                self._feature(box(3.0, 3.0, 4.0, 4.0)),
            ])

            layer = load_official_buildings_geojson(
                path, self.bbox, self.transform, self.grid_shape)

        self.assertEqual(layer["metadata"]["source_feature_count"], 3)
        self.assertEqual(layer["metadata"]["source_features_intersecting_window"], 2)
        self.assertEqual(layer["metadata"]["coverage_status"],
                         "not_independently_verified")
        self.assertEqual(layer["metadata"]["rasterization"], "all_touched")
        self.assertEqual(layer["metadata"]["data_sources"][0]["dataset_id"],
                         "Taiwan e-Map BUILD")
        self.assertGreater(int(layer["mask"].sum()), 0)
        self.assertTrue(all(geometry.geom_type in {"Polygon", "MultiPolygon"}
                            for geometry, _ in layer["geometries"]))

    def test_loader_rejects_non_wgs84_crs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "buildings.geojson"
            self._write_geojson(path, crs={
                "type": "name", "properties": {"name": "EPSG:3826"},
            })

            with self.assertRaisesRegex(ValueError, "必须使用 WGS84"):
                load_official_buildings_geojson(
                    path, self.bbox, self.transform, self.grid_shape)

    def test_loader_rejects_nonpolygon_and_invalid_geometry(self):
        from shapely.geometry import LineString, Polygon

        bad_geometries = [
            LineString([(0.1, 0.1), (0.9, 0.9)]),
            Polygon([(0.1, 0.1), (0.9, 0.9), (0.1, 0.9),
                     (0.9, 0.1), (0.1, 0.1)]),
        ]
        for geometry in bad_geometries:
            with self.subTest(geometry=geometry.geom_type), \
                    tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "buildings.geojson"
                self._write_geojson(path, features=[self._feature(geometry)])
                with self.assertRaisesRegex(ValueError, "有效 Polygon 或 MultiPolygon"):
                    load_official_buildings_geojson(
                        path, self.bbox, self.transform, self.grid_shape)

    def test_clearance_buffer_loads_outside_window_building_into_distance_band(self):
        bbox = (120.0, 24.0, 120.002, 24.002)
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 24.002)
        shape = (2, 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "buildings.geojson"
            self._write_geojson(path, features=[
                self._feature(box(119.997, 24.0008, 119.998, 24.0012)),
            ])
            layer = load_official_buildings_geojson(
                path, bbox, transform, shape, buffer_m=400.0)

        self.assertEqual(layer["metadata"]["load_buffer_m"], 400.0)
        self.assertEqual(layer["metadata"]["planning_window_bbox_wgs84"], list(bbox))
        self.assertEqual(layer["metadata"]["source_features_intersecting_window"], 0)
        self.assertEqual(layer["metadata"]["source_features_intersecting_query_bbox"], 1)
        self.assertEqual(layer["metadata"]["status"], "loaded_no_features_in_window")
        self.assertEqual(int(layer["mask"].sum()), 0)
        bands = OSMBands(
            {"building_mask": np.zeros(shape, dtype=np.uint8)},
            geometries={"buildings": []},
        )
        merge_official_buildings_into_bands(
            bands, layer, transform, shape, include_building_distance=True)
        self.assertLessEqual(float(bands["building_distance_m"][0, 0]), 400.0)

    def test_union_mask_and_exact_route_intersection_include_official_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "buildings.geojson"
            official_building = box(0.2, 0.2, 0.8, 0.8)
            self._write_geojson(path, features=[self._feature(official_building)])
            layer = load_official_buildings_geojson(
                path, self.bbox, self.transform, self.grid_shape)

        osm_building = box(1.2, 1.2, 1.8, 1.8)
        osm_mask = np.zeros(self.grid_shape, dtype=np.uint8)
        osm_mask[0, 1] = 1
        bands = OSMBands(
            {"building_mask": osm_mask},
            geometries={"buildings": [(osm_building, {})]},
        )
        merged = merge_official_buildings_into_bands(
            bands, layer, self.transform, self.grid_shape)

        self.assertEqual(len(merged), 2)
        self.assertEqual(int(bands["building_mask"].sum()),
                         int(osm_mask.sum() + layer["mask"].sum()))
        accepted, report = validate_route_against_buildings(
            [(0.1, 0.5), (0.9, 0.5)], merged)
        self.assertIsNone(accepted)
        self.assertEqual(report["building_geometry_status"], "fail")
        self.assertEqual(report["building_geometry_intersections"], 1)
        self.assertFalse(report["building_geometry_absence_claim"])

    def test_official_geometry_is_used_for_exact_endpoint_safety_check(self):
        transform = Affine.identity()
        bbox = (0.0, 0.0, 3.0, 3.0)
        shape = (3, 3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "buildings.geojson"
            self._write_geojson(path, features=[
                self._feature(box(1.75, 1.75, 1.9, 1.9)),
            ])
            layer = load_official_buildings_geojson(path, bbox, transform, shape)

        hard_mask = np.ones(shape, dtype=np.uint8)
        hard_mask[1, 1] = 0
        blockers = {"OSM+官方建筑足迹": layer["mask"] > 0}
        exceptions = open_safe_building_endpoint_pixels(
            (("起点", (1, 1), (1.1, 1.5)),), hard_mask, blockers,
            layer["route_geometries"], transform,
            building_blocker_name="OSM+官方建筑足迹",
        )

        self.assertEqual(len(exceptions), 1)
        self.assertEqual(exceptions[0]["point_xy"], (1.1, 1.5))

    def test_pass_metadata_names_sources_and_limits_claim_to_loaded_features(self):
        osm_metadata = {
            "status": "available", "coverage_status": "features_observed",
            "geometry_count_in_bbox": 4,
        }
        official_metadata = {
            "source": "NLSC Taiwan e-Map BUILD", "status": "loaded",
            "coverage_status": "not_independently_verified",
            "source_feature_count": 10, "source_features_intersecting_window": 6,
            "feature_sources": ["NLSC Taiwan e-Map BUILD; source date 2026-08"],
            "data_sources": [{"dataset_id": "Taiwan e-Map BUILD"}],
            "coverage_reason": "source coverage is unverified",
        }
        metadata = building_validation_metadata(osm_metadata, official_metadata)
        report = annotate_route_validation_context(
            {"building_geometry_status": "pass"}, [], metadata)

        self.assertEqual(metadata["check_scope"],
                         "loaded_official_and_osm_features_only")
        self.assertEqual([item["source"] for item in report["building_geometry_sources"]],
                         ["OpenStreetMap", "NLSC Taiwan e-Map BUILD"])
        self.assertEqual(report["building_geometry_coverage_status"],
                         "not_independently_verified")
        self.assertEqual(report["building_geometry_status_semantics"],
                         "pass_means_no_intersection_with_loaded_features_only")
        self.assertFalse(report["building_geometry_absence_claim"])

    def test_empty_official_window_and_unavailable_osm_do_not_produce_pass(self):
        official_metadata = {
            "source": "NLSC Taiwan e-Map BUILD",
            "status": "loaded_no_features_in_window",
            "coverage_status": "not_independently_verified",
            "source_feature_count": 5,
            "source_features_intersecting_window": 0,
            "data_sources": [{"dataset_id": "Taiwan e-Map BUILD"}],
            "coverage_reason": "source coverage is unverified",
        }
        metadata = building_validation_metadata(
            {"status": "missing_file", "coverage_status": "unknown"},
            official_metadata)
        route = [(0.1, 0.1), (1.9, 1.9)]
        accepted, report = validate_route_against_buildings(route, [])
        annotate_route_validation_context(report, [], metadata)

        self.assertEqual(accepted, route)
        self.assertEqual(metadata["status"], "no_loaded_features")
        self.assertEqual(report["building_geometry_status"],
                         "not_checked_no_loaded_features")
        self.assertEqual(report["building_geometry_status_semantics"],
                         "no_loaded_features_no_intersection_check_performed")
        self.assertFalse(report["building_geometry_absence_claim"])


if __name__ == "__main__":
    unittest.main()
