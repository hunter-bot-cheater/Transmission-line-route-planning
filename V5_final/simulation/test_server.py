import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from threading import Event, Thread
from urllib.request import Request, urlopen

import benchmark
import server
import geographic_route_api
from benchmark import build_cost as benchmark_cost, path_cost, run_policy, valid_route
from server import (_get_session, build_cost_grid, configure_session_mission, create_world_session,
                    make_world, plan_path, plan_session, select_scan_target, select_session_scan,
                    sense_flight_segment, sense_session, Handler)


class GeographicRoutePointValidationTests(unittest.TestCase):
    def test_public_status_explains_engineering_checks_not_evaluated(self):
        status = geographic_route_api.GeographicRouteJobs().public_status()

        self.assertEqual(status["engineering_feasibility"], "not_evaluated")
        self.assertEqual(status["engineering_gate"]["status"], "not_evaluated")
        blocker_codes = {item["code"] for item in status["engineering_gate"]["blockers"]}
        self.assertIn("station_terminals_unverified", blocker_codes)
        self.assertIn("tower_and_span_graph_not_run", blocker_codes)
        self.assertIn("sag_and_clearance_not_evaluated", blocker_codes)

    def test_valid_point_is_checked_against_configured_dem(self):
        jobs = server.GEOGRAPHIC_ROUTE_JOBS
        dem_path = Path("/data1/B24041626/pp/taiwan-dem.tif")
        land_mask_path = Path("/data1/B24041626/pp/taiwan-land-mask.geojson")
        with patch.object(jobs, "_config", return_value={
                "dem": dem_path, "land_mask": land_mask_path}), \
                patch.object(geographic_route_api, "_require_supported_planner_dem_crs"), \
                patch.object(jobs, "_validate_dem_point") as validate_dem_point, \
                patch.object(geographic_route_api, "_validate_taiwan_land_point") as validate_land:
            result = jobs.validate_point({"lat": 23.5, "lon": 120.9})

        self.assertEqual(result, {"valid": True, "lat": 23.5, "lon": 120.9})
        validate_dem_point.assert_called_once_with(
            dem_path, {"lat": "23.5", "lon": "120.9"}, "地图所选位置",
        )
        validate_land.assert_called_once_with(land_mask_path, 120.9, 23.5, "地图所选位置")

    def test_out_of_range_point_is_rejected_before_dem_sampling(self):
        jobs = server.GEOGRAPHIC_ROUTE_JOBS
        with patch.object(jobs, "_config", return_value={
                "dem": Path("dem.tif"), "land_mask": Path("land-mask.geojson")}), \
                patch.object(jobs, "_validate_dem_point") as validate_dem_point:
            with self.assertRaisesRegex(ValueError, "纬度超出允许范围"):
                jobs.validate_point({"lat": 91, "lon": 120.9})

        validate_dem_point.assert_not_called()

    def test_point_validation_requires_taiwan_land_mask(self):
        jobs = server.GEOGRAPHIC_ROUTE_JOBS
        with patch.object(jobs, "_config", return_value={"dem": Path("dem.tif")}):
            with self.assertRaisesRegex(RuntimeError, "未配置台湾陆域边界"):
                jobs.validate_point({"lat": 23.5, "lon": 120.9})

    def test_submit_requires_taiwan_land_mask(self):
        jobs = server.GEOGRAPHIC_ROUTE_JOBS
        with patch.object(jobs, "_config", return_value={"dem": Path("dem.tif")}):
            with self.assertRaisesRegex(RuntimeError, "台湾范围规划已禁用"):
                jobs.submit({})

    def test_submit_rejects_endpoint_outside_land_mask_before_creating_job(self):
        jobs = geographic_route_api.GeographicRouteJobs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            land_mask = GeographicRouteOutputValidationTests._write_land_mask(
                root / "land-mask.geojson",
            )
            job_root = root / "jobs"
            config = {
                "dem": root / "dem.tif", "land_mask": land_mask,
                "job_root": job_root, "access_token": "x" * 32,
            }
            payload = {
                "endpoint_mode": "exploratory",
                "start": {"station_id": "A", "name": "海上", "lat": 23.5, "lon": 122.5},
                "end": {"station_id": "B", "name": "台湾", "lat": 23.5, "lon": 120.9},
            }
            with patch.object(jobs, "_config", return_value=config), \
                    patch.object(geographic_route_api, "_require_supported_planner_dem_crs"), \
                    patch.object(jobs, "_validate_dem_point"), \
                    patch.object(geographic_route_api, "_TAIWAN_LAND_MASK_SHA256",
                                 hashlib.sha256(land_mask.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "起点坐标位于台湾县市陆域边界外"):
                    jobs.submit(payload)

            self.assertFalse(job_root.exists())

    def test_public_status_is_not_ready_without_taiwan_land_mask(self):
        jobs = geographic_route_api.GeographicRouteJobs()
        config = {"dem": Path("dem.tif"), "max_side": 4096}
        with patch.object(jobs, "_config", return_value=config), \
                patch.object(geographic_route_api, "_require_supported_planner_dem_crs"), \
                patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                    "west": 120.0, "south": 22.0, "east": 122.0, "north": 25.0,
                }):
            status = jobs.public_status()

        self.assertFalse(status["ready"])
        self.assertFalse(status["data_layers"]["land_mask"])
        self.assertIn("台湾陆域边界", status["message"])

    def test_public_status_requires_both_dem_and_valid_taiwan_land_mask(self):
        jobs = geographic_route_api.GeographicRouteJobs()
        config = {
            "dem": Path("dem.tif"), "land_mask": Path("land-mask.geojson"),
            "max_side": 4096,
        }
        from shapely.geometry import box

        with patch.object(jobs, "_config", return_value=config), \
                patch.object(geographic_route_api, "_require_supported_planner_dem_crs"), \
                patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                    "west": 120.0, "south": 22.0, "east": 122.0, "north": 25.0,
                }), \
                patch.object(geographic_route_api, "_taiwan_land_geometry",
                             return_value=box(120.0, 22.0, 122.0, 25.0)), \
                patch.object(geographic_route_api, "_dem_has_valid_taiwan_land_cell",
                             return_value=True):
            status = jobs.public_status()

        self.assertTrue(status["ready"])
        self.assertTrue(status["data_layers"]["land_mask"])

        with patch.object(jobs, "_config", return_value=config), \
                patch.object(geographic_route_api, "_require_supported_planner_dem_crs"), \
                patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                    "west": 120.0, "south": 22.0, "east": 122.0, "north": 25.0,
                }), \
                patch.object(geographic_route_api, "_taiwan_land_geometry",
                             return_value=box(130.0, 30.0, 131.0, 31.0)):
            disjoint_status = jobs.public_status()

        self.assertFalse(disjoint_status["ready"])
        self.assertFalse(disjoint_status["data_layers"]["land_mask"])

        with patch.object(jobs, "_config", return_value=config), \
                patch.object(geographic_route_api, "_require_supported_planner_dem_crs"), \
                patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                    "west": 120.8, "south": 22.0, "east": 121.2, "north": 25.0,
                }), \
                patch.object(geographic_route_api, "_taiwan_land_geometry",
                             return_value=box(120.0, 22.0, 120.8, 25.0)):
            touching_status = jobs.public_status()

        self.assertFalse(touching_status["ready"])
        self.assertFalse(touching_status["data_layers"]["land_mask"])

    def test_public_status_rejects_projected_dem_even_when_land_checks_pass(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box

        jobs = geographic_route_api.GeographicRouteJobs()
        with tempfile.TemporaryDirectory() as temp_dir:
            dem_path = Path(temp_dir) / "projected-dem.tif"
            with rasterio.open(
                    dem_path, "w", driver="GTiff", width=2, height=2, count=1,
                    dtype="float32", crs="EPSG:3826",
                    transform=from_origin(250000, 2800000, 20, 20)) as dataset:
                dataset.write(np.ones((2, 2), dtype=np.float32), 1)
            config = {
                "dem": dem_path, "land_mask": Path("land-mask.geojson"),
                "max_side": 4096,
            }
            with patch.object(jobs, "_config", return_value=config), \
                    patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                        "west": 120.0, "south": 22.0, "east": 122.0, "north": 25.0,
                    }), \
                    patch.object(geographic_route_api, "_taiwan_land_geometry",
                                 return_value=box(120.0, 22.0, 122.0, 25.0)), \
                    patch.object(geographic_route_api, "_dem_has_valid_taiwan_land_cell",
                                 return_value=True) as check_valid_cell:
                status = jobs.public_status()

        self.assertFalse(status["ready"])
        self.assertFalse(status["dem_crs_supported"])
        self.assertFalse(status["data_layers"]["dem"])
        self.assertIn("EPSG:4326", status["message"])
        check_valid_cell.assert_not_called()

    def test_projected_dem_is_rejected_before_point_validation_or_job_creation(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        jobs = geographic_route_api.GeographicRouteJobs()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dem_path = root / "projected-dem.tif"
            with rasterio.open(
                    dem_path, "w", driver="GTiff", width=2, height=2, count=1,
                    dtype="float32", crs="EPSG:3826",
                    transform=from_origin(250000, 2800000, 20, 20)) as dataset:
                dataset.write(np.ones((2, 2), dtype=np.float32), 1)
            config = {
                "dem": dem_path, "land_mask": root / "land-mask.geojson",
                "job_root": root / "jobs", "access_token": "x" * 32,
                "max_side": 4096,
            }
            with patch.object(jobs, "_config", return_value=config):
                with self.assertRaisesRegex(RuntimeError, "仅支持 EPSG:4326"):
                    jobs.validate_point({"lat": 23.5, "lon": 120.9})
                with self.assertRaisesRegex(RuntimeError, "仅支持 EPSG:4326"):
                    jobs.submit({})

            self.assertFalse(config["job_root"].exists())

    def test_public_status_requires_a_valid_dem_pixel_inside_taiwan_land(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import box

        jobs = geographic_route_api.GeographicRouteJobs()
        with tempfile.TemporaryDirectory() as temp_dir:
            dem_path = Path(temp_dir) / "dem.tif"
            nodata = -9999.0
            with rasterio.open(
                    dem_path, "w", driver="GTiff", width=2, height=2, count=1,
                    dtype="float32", crs="EPSG:4326", transform=from_origin(120, 25, 1, 1),
                    nodata=nodata) as dataset:
                dataset.write(np.array([[nodata, 1.0], [2.0, 3.0]], dtype=np.float32), 1)
            config = {
                "dem": dem_path, "land_mask": Path("land-mask.geojson"),
                "max_side": 4096,
            }
            land_geometry = box(120.1, 24.1, 120.9, 24.9)
            with patch.object(jobs, "_config", return_value=config), \
                    patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                        "west": 120.0, "south": 23.0, "east": 122.0, "north": 25.0,
                    }), \
                    patch.object(geographic_route_api, "_taiwan_land_geometry",
                                 return_value=land_geometry):
                nodata_status = jobs.public_status()

            self.assertFalse(nodata_status["ready"])
            self.assertTrue(nodata_status["data_layers"]["land_mask"])
            self.assertIn("没有有效 DEM 像元", nodata_status["message"])

            with rasterio.open(dem_path, "r+") as dataset:
                dataset.write(np.array([[123.0, nodata], [nodata, nodata]], dtype=np.float32), 1)
            with patch.object(jobs, "_config", return_value=config), \
                    patch.object(geographic_route_api, "_dem_bounds_wgs84", return_value={
                        "west": 120.0, "south": 23.0, "east": 122.0, "north": 25.0,
                    }), \
                    patch.object(geographic_route_api, "_taiwan_land_geometry",
                                 return_value=land_geometry):
                valid_status = jobs.public_status()

            self.assertTrue(valid_status["ready"])

    def test_point_validation_requires_planner_configuration(self):
        jobs = server.GEOGRAPHIC_ROUTE_JOBS
        with patch.object(jobs, "_config", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "真实 DEM 规划未配置"):
                jobs.validate_point({"lat": 23.5, "lon": 120.9})

    def test_http_point_validation_rejects_missing_planner_token(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        jobs = server.GEOGRAPHIC_ROUTE_JOBS
        try:
            request = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/routes/validate-point",
                data=json.dumps({"lat": 23.5, "lon": 120.9}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with patch.object(jobs, "authorize", return_value=False), \
                    patch.object(jobs, "validate_point") as validate_point:
                with self.assertRaises(HTTPError) as rejected:
                    urlopen(request, timeout=5)

            self.assertEqual(rejected.exception.code, 403)
            validate_point.assert_not_called()
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)


class GeographicRouteOutputValidationTests(unittest.TestCase):
    class FakeDataset:
        def __init__(self, invalid_cells=()):
            from affine import Affine

            self.crs = "EPSG:4326"
            self.transform = Affine(0.125, 0.0, 120.0, 0.0, -0.125, 25.0)
            self.width = 10
            self.height = 10
            self.invalid_cells = set(invalid_cells)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def sample(self, coordinates, indexes=1, masked=False):
            import numpy as np

            self.assert_sample_options(indexes, masked)
            inverse = ~self.transform
            for x, y in coordinates:
                col, row = inverse * (x, y)
                cell = (math.floor(row), math.floor(col))
                yield np.ma.array(
                    [1.0], mask=[cell in self.invalid_cells],
                )

        @staticmethod
        def assert_sample_options(indexes, masked):
            if indexes != 1 or masked is not True:
                raise AssertionError("DEM coverage validation must request masked band-1 samples")

    def _validate(self, coordinates, invalid_cells=()):
        import rasterio

        dataset = self.FakeDataset(invalid_cells)
        with patch.object(rasterio, "open", return_value=dataset):
            geographic_route_api._validate_route_dem_coverage(
                Path("/data1/B24041626/pp/taiwan-dem.tif"), coordinates,
            )

    @staticmethod
    def _write_land_mask(path, *, hole=None):
        exterior = [[120, 22], [122, 22], [122, 25], [120, 25], [120, 22]]
        rings = [exterior]
        if hole:
            rings.append([[hole[0], hole[1]], [hole[2], hole[1]],
                          [hole[2], hole[3]], [hole[0], hole[3]],
                          [hole[0], hole[1]]])
        document = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": {
                "type": "Polygon", "coordinates": rings,
            }}],
        }
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_taiwan_land_mask_accepts_inside_point_and_rejects_outside_point(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            land_mask = self._write_land_mask(Path(temp_dir) / "land-mask.geojson")
            with patch.object(geographic_route_api, "_TAIWAN_LAND_MASK_SHA256",
                              hashlib.sha256(land_mask.read_bytes()).hexdigest()):
                geographic_route_api._validate_taiwan_land_point(
                    land_mask, 120.9, 23.5, "起点",
                )
                with self.assertRaisesRegex(ValueError, "位于台湾县市陆域边界外"):
                    geographic_route_api._validate_taiwan_land_point(
                        land_mask, 122.5, 23.5, "起点",
                    )

    def test_taiwan_land_mask_rejects_unpinned_hash_and_invalid_feature_or_crs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            land_mask = self._write_land_mask(Path(temp_dir) / "land-mask.geojson")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                geographic_route_api._validate_taiwan_land_point(
                    land_mask, 120.9, 23.5, "起点",
                )

            document = json.loads(land_mask.read_text(encoding="utf-8"))
            document["features"][0]["type"] = "Polygon"
            land_mask.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(geographic_route_api, "_TAIWAN_LAND_MASK_SHA256",
                              hashlib.sha256(land_mask.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(RuntimeError, "要素"):
                    geographic_route_api._validate_taiwan_land_point(
                        land_mask, 120.9, 23.5, "起点",
                    )

            land_mask = self._write_land_mask(Path(temp_dir) / "land-mask.geojson")
            document = json.loads(land_mask.read_text(encoding="utf-8"))
            document["crs"] = {}
            land_mask.write_text(json.dumps(document), encoding="utf-8")
            with patch.object(geographic_route_api, "_TAIWAN_LAND_MASK_SHA256",
                              hashlib.sha256(land_mask.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(RuntimeError, "CRS"):
                    geographic_route_api._validate_taiwan_land_point(
                        land_mask, 120.9, 23.5, "起点",
                    )

    def test_taiwan_land_mask_hash_is_rechecked_after_same_size_file_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            land_mask = self._write_land_mask(Path(temp_dir) / "land-mask.geojson")
            original = land_mask.read_bytes()
            original_stat = land_mask.stat()
            changed = original.replace(b"120", b"130", 1)
            self.assertEqual(len(changed), len(original))

            with patch.object(geographic_route_api, "_TAIWAN_LAND_MASK_SHA256",
                              hashlib.sha256(original).hexdigest()):
                geographic_route_api._load_taiwan_land_geometry_cached.cache_clear()
                geographic_route_api._validate_taiwan_land_point(
                    land_mask, 120.9, 23.5, "起点",
                )
                land_mask.write_bytes(changed)
                os.utime(land_mask, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                self.assertEqual(land_mask.stat().st_size, original_stat.st_size)
                self.assertEqual(land_mask.stat().st_mtime_ns, original_stat.st_mtime_ns)
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    geographic_route_api._validate_taiwan_land_point(
                        land_mask, 120.9, 23.5, "起点",
                    )
                geographic_route_api._load_taiwan_land_geometry_cached.cache_clear()

    def test_final_route_must_be_covered_by_taiwan_land_polygon(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            land_mask = self._write_land_mask(
                Path(temp_dir) / "land-mask.geojson", hole=(120.8, 23.4, 121.0, 23.6),
            )
            with patch.object(geographic_route_api, "_TAIWAN_LAND_MASK_SHA256",
                              hashlib.sha256(land_mask.read_bytes()).hexdigest()):
                geographic_route_api._validate_route_taiwan_land_coverage(
                    land_mask, [[120.2, 23.0], [120.6, 23.0]],
                )
                with self.assertRaisesRegex(RuntimeError, "路线超出台湾县市陆域边界"):
                    geographic_route_api._validate_route_taiwan_land_coverage(
                        land_mask, [[120.5, 23.5], [121.5, 23.5]],
                    )

    def test_route_inside_valid_dem_cells_is_accepted(self):
        self._validate([[120.1875, 24.8125], [121.0625, 24.8125]])

    def test_diagonal_route_inside_valid_dem_cells_is_accepted(self):
        self._validate([[120.1875, 24.8125], [121.0625, 23.9375]])

    def test_route_crossing_dem_nodata_between_valid_vertices_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "穿过无效数据区"):
            self._validate(
                [[120.1875, 24.8125], [121.0625, 24.8125]], invalid_cells={(1, 5)},
            )

    def test_route_touching_nodata_cell_only_at_diagonal_corner_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "像元边界接触"):
            self._validate(
                [[120.1875, 24.8125], [120.3125, 24.6875]], invalid_cells={(1, 2)},
            )

    def test_route_following_nodata_cell_edge_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "像元边界接触"):
            self._validate(
                [[120.1875, 24.75], [120.3125, 24.75]], invalid_cells={(1, 1)},
            )

    def test_route_vertex_outside_dem_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "超出数据范围"):
            self._validate([[120.1875, 24.8125], [121.3125, 24.8125]])

    def test_successful_route_load_requires_dem_coverage_validation(self):
        route_path = Path("results/final_route.geojson")
        start = {"station_id": "A", "lon": "120.1875", "lat": "24.8125"}
        end = {"station_id": "B", "lon": "120.8125", "lat": "24.8125"}
        route_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[120.1875, 24.8125], [120.8125, 24.8125]],
                },
                "properties": {
                    "planning_status": "screening_candidate",
                    "hard_constraint_status": "pass",
                    "hard_violations": 0,
                    "length_km": 1.0,
                    "dem_sample_count": 2,
                    "dem_valid_sample_count": 2,
                    "dem_sample_coverage_ratio": 1.0,
                    "engineering_feasibility": "not_evaluated",
                    "from_station_id": "A",
                    "to_station_id": "B",
                },
            }],
        }
        dem_path = Path("/data1/B24041626/pp/taiwan-dem.tif")
        land_mask_path = Path("/data1/B24041626/pp/taiwan-land-mask.geojson")
        with patch.object(Path, "rglob", return_value=[route_path]), \
                patch.object(Path, "read_text", return_value=json.dumps(route_data)), \
                patch.object(geographic_route_api, "_validate_route_dem_coverage") as validate_dem, \
                patch.object(geographic_route_api, "_validate_route_taiwan_land_coverage") as validate_land:
            result = geographic_route_api._load_final_route_properties(
                Path("results"), start, end, dem_path, land_mask_path,
            )

        self.assertEqual(result, (route_path, route_data["features"][0]["properties"]))
        coordinates = route_data["features"][0]["geometry"]["coordinates"]
        validate_dem.assert_called_once_with(dem_path, coordinates)
        validate_land.assert_called_once_with(land_mask_path, coordinates)


class PlanPathTests(unittest.TestCase):
    def test_diagonal_cannot_cut_across_unobserved_corner(self):
        result = plan_path(
            observed=[[True, False], [False, True]],
            cost=[[1.0, None], [None, 1.0]],
            start={"x": 0, "y": 0},
            goal={"x": 1, "y": 1},
        )

        self.assertEqual(result["path"], [])

    def test_diagonal_cannot_cut_through_hard_blocked_corner(self):
        result = plan_path(
            observed=[[True, True], [True, True]],
            cost=[[1.0, 100.0], [100.0, 1.0]],
            start={"x": 0, "y": 0},
            goal={"x": 1, "y": 1},
        )

        self.assertEqual(result["path"], [])

    def test_diagonal_route_is_available_when_surrounding_cells_are_passable(self):
        result = plan_path(
            observed=[[True, True], [True, True]],
            cost=[[1.0, 1.0], [1.0, 1.0]],
            start={"x": 0, "y": 0},
            goal={"x": 1, "y": 1},
        )

        self.assertEqual(result["path"], [{"x": 0, "y": 0}, {"x": 1, "y": 1}])
        self.assertEqual(result["planner"], "V5 A*")

    def test_unknown_endpoint_waits_for_scan(self):
        result = plan_path(
            observed=[[True, False]],
            cost=[[1.0, 0.01]],
            start={"x": 0, "y": 0},
            goal={"x": 1, "y": 0},
        )

        self.assertTrue(result["waiting_for_scan"])
        self.assertEqual(result["path"], [])

    def test_unknown_low_cost_is_not_used_as_shortcut(self):
        result = plan_path(
            observed=[[True, False, True], [True, True, True]],
            cost=[[1.0, 0.01, 1.0], [1.0, 1.0, 1.0]],
            start={"x": 0, "y": 0},
            goal={"x": 2, "y": 0},
        )

        self.assertNotIn({"x": 1, "y": 0}, result["path"])
        self.assertEqual(result["path"][0], {"x": 0, "y": 0})
        self.assertEqual(result["path"][-1], {"x": 2, "y": 0})


class ScanTargetTests(unittest.TestCase):
    def test_mission_progress_grid_matches_exact_segment_reference(self):
        width, height = 32, 24
        paths = [
            [(0, 0)],
            [(0, 0), (31, 23)],
            [(1, 22), (30, 1), (2, 1), (29, 22)],
            [(5, 5), (5, 5), (15, 5), (15, 18), (5, 18), (5, 5)],
            [(5, 5), (15, 5)],
        ]
        snake = [(x, y) for y in range(height)
                 for x in (range(width) if y % 2 == 0 else range(width - 1, -1, -1))]
        paths.append(snake[:200])

        def reference(path):
            points = [(int(point["x"]), int(point["y"])) for point in path]
            segments = [(x0, y0, x1 - x0, y1 - y0, (x1 - x0) ** 2 + (y1 - y0) ** 2)
                        for (x0, y0), (x1, y1) in zip(points, points[1:])]
            values = []
            for y in range(height):
                row = []
                for x in range(width):
                    nearest, progress = math.inf, 0.0
                    if not segments:
                        nearest = math.hypot(x - points[0][0], y - points[0][1])
                        progress = 1.0
                    for index, (x0, y0, dx, dy, length_squared) in enumerate(segments):
                        along = ((x - x0) * dx + (y - y0) * dy) / length_squared if length_squared else 0.0
                        along = server.clamp(along)
                        distance = math.hypot(x - x0 - along * dx, y - y0 - along * dy)
                        if distance < nearest:
                            nearest, progress = distance, (index + along) / len(segments)
                    row.append(max(0.0, 1.0 - nearest / server.GOAL_CORRIDOR_RADIUS)
                               * (0.25 + 0.75 * progress))
                values.append(row)
            return values

        for path in paths:
            mission_path = [{"x": x, "y": y} for x, y in path]
            actual = server._mission_progress_grid(width, height, mission_path)
            expected = reference(mission_path)
            for actual_row, expected_row in zip(actual, expected):
                for actual_value, expected_value in zip(actual_row, expected_row):
                    self.assertAlmostEqual(actual_value, expected_value, places=12)

        boundary = server._mission_progress_grid(width, height, [{"x": 5, "y": 5}, {"x": 15, "y": 5}])
        self.assertEqual(boundary[15][5], 0.0)
        crossing = server._mission_progress_grid(
            width, height, [{"x": 2, "y": 2}, {"x": 8, "y": 8}, {"x": 2, "y": 8}, {"x": 8, "y": 2}]
        )
        self.assertAlmostEqual(crossing[5][5], 0.375, places=12)

    def test_seeded_scan_rasters_change_direction_without_using_hidden_terrain(self):
        width, height = 24, 18
        start, goal = {"x": 1, "y": 1}, {"x": 22, "y": 16}

        def trace(pattern_seed):
            candidates = server._make_scan_candidates(width, height, start, pattern_seed)
            observed = [[False] * width for _ in range(height)]
            cost = [[None] * width for _ in range(height)]
            drone, previous, route, distance = start, None, [], 0.0
            for _ in range(width * height):
                choice = select_scan_target(
                    observed, cost, candidates, drone, radius=2, guidance_weight=0.75,
                    priority_points=[start, goal], mission_path=[start, goal],
                    previous=previous,
                )
                target = choice["target"]
                if target is None:
                    break
                route.append((target["x"], target["y"]))
                distance += math.hypot(target["x"] - drone["x"], target["y"] - drone["y"])
                for y in range(max(0, target["y"] - 2), min(height, target["y"] + 3)):
                    for x in range(max(0, target["x"] - 2), min(width, target["x"] + 3)):
                        if (x - target["x"]) ** 2 + (y - target["y"]) ** 2 <= 5:
                            observed[y][x] = True
                            cost[y][x] = 0.2
                previous, drone = drone, target
            unknown = sum(not cell for row in observed for cell in row)
            return candidates, route, distance, unknown

        patterns = [trace(seed) for seed in range(4)]
        candidate_sets = [tuple((point["x"], point["y"]) for point in candidates)
                          for candidates, _, _, _ in patterns]
        routes = [tuple(route) for _, route, _, _ in patterns]
        distances = [distance for _, _, distance, _ in patterns]

        self.assertEqual(len(set(candidate_sets)), 4, "session variants should include horizontal/vertical sweeps and lane offsets")
        self.assertEqual(len(set(routes)), 4, "unknown-terrain exploration should not reuse one fixed route")
        self.assertLessEqual(max(distances), min(distances) * 1.15,
                             "sweep diversity must retain comparable total travel on the same terrain")
        for candidates, _, _, unknown in patterns:
            self.assertTrue(all(0 <= point["x"] < width and 0 <= point["y"] < height for point in candidates))
            self.assertEqual(unknown, 0, "every sweep variant must finish full sensor coverage")

    def test_nearby_scan_frontier_prevents_long_hops_for_extra_gain(self):
        observed = [[True] * 6 for _ in range(6)]
        observed[0][0] = False
        for y in range(2, 5):
            for x in range(2, 5):
                observed[y][x] = False
        cost = [[0.2 if seen else None for seen in row] for row in observed]

        result = select_scan_target(observed, cost, [{"x": 0, "y": 0}, {"x": 3, "y": 3}],
                                   {"x": 0, "y": 0}, radius=1, local_first=True)

        self.assertEqual(result["target"], {"x": 0, "y": 0})

    def test_shorter_nearby_flight_wins_over_larger_scan_gain(self):
        observed = [[True] * 6 for _ in range(6)]
        observed[1][2] = False
        observed[1][3] = observed[2][3] = observed[3][3] = False
        cost = [[0.2 if seen else None for seen in row] for row in observed]

        result = select_scan_target(observed, cost,
                                    [{"x": 2, "y": 1}, {"x": 3, "y": 2}],
                                    {"x": 1, "y": 1}, radius=1, local_first=True)

        self.assertEqual(result["target"], {"x": 2, "y": 1})

    def test_scan_selection_penalizes_sharp_heading_changes(self):
        observed = [[True] * 4 for _ in range(3)]
        observed[0][2] = observed[1][3] = False
        cost = [[0.2 if seen else None for seen in row] for row in observed]

        result = select_scan_target(observed, cost,
                                    [{"x": 2, "y": 0}, {"x": 3, "y": 1}],
                                    {"x": 2, "y": 1}, radius=0, local_first=False,
                                    previous={"x": 1, "y": 1})

        self.assertEqual(result["target"], {"x": 3, "y": 1})

    def test_frontier_fallback_and_completed_coverage(self):
        observed = [[True, False, True]]
        cost = [[0.2, None, 0.2]]
        candidates = [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]

        result = select_scan_target(observed, cost, candidates, {"x": 0, "y": 0}, radius=0)
        self.assertEqual(result["target"], {"x": 1, "y": 0})
        self.assertEqual(result["reason"], "frontier")

        complete = select_scan_target([[True, True, True]], [[0.2] * 3], candidates,
                                      {"x": 0, "y": 0}, radius=0)
        self.assertIsNone(complete["target"])
        self.assertEqual(complete["reason"], "coverage_complete")

        exhausted = select_scan_target([[True, False, True]], [[0.2, None, 0.2]],
                                       [{"x": 0, "y": 0}, {"x": 2, "y": 0}],
                                       {"x": 0, "y": 0}, radius=0)
        self.assertEqual(exhausted["target"], {"x": 1, "y": 0})
        self.assertEqual(exhausted["unknown_cells"], 1)
        self.assertTrue(exhausted["residual_fallback"])

    def test_residual_scan_targets_only_uncovered_cells_after_raster_exhaustion(self):
        observed = [[True, False, True]]
        cost = [[0.2, None, 0.2]]

        result = select_scan_target(observed, cost,
                                    [{"x": 0, "y": 0}, {"x": 2, "y": 0}],
                                    {"x": 0, "y": 0}, radius=0)

        self.assertEqual(result["target"], {"x": 1, "y": 0})
        self.assertEqual(result["new_cells"], 1)
        self.assertTrue(result["residual_fallback"])

    def test_residual_scan_chooses_nearest_sensor_center_not_unknown_cell(self):
        observed = [[True, True, True, True, True, False]]
        cost = [[0.2, 0.2, 0.2, 0.2, 0.2, None]]

        result = select_scan_target(observed, cost, [{"x": 0, "y": 0}],
                                    {"x": 3, "y": 0}, radius=1, local_first=True)

        self.assertEqual(result["target"], {"x": 4, "y": 0})
        self.assertEqual(result["travel_cells"], 1.0)
        self.assertTrue(result["residual_fallback"])

    def test_unobserved_endpoint_is_prioritized_for_scanning(self):
        observed = [[True] * 4 for _ in range(3)]
        observed[0][0] = observed[0][2] = False
        cost = [[0.2 if seen else None for seen in row] for row in observed]

        result = select_scan_target(observed, cost, [{"x": 0, "y": 0}, {"x": 2, "y": 0}],
                                   {"x": 0, "y": 0}, radius=0,
                                   priority_points=[{"x": 0, "y": 0}])

        self.assertEqual(result["target"], {"x": 0, "y": 0})
        self.assertEqual(result["priority_cells"], 1)

    def test_goal_corridor_can_beat_a_nearer_off_corridor_scan(self):
        observed = [[True] * 7 for _ in range(7)]
        observed[0][0] = observed[3][2] = False
        cost = [[0.2 if seen else None for seen in row] for row in observed]
        candidates = [{"x": 0, "y": 0}, {"x": 2, "y": 3}]

        without_corridor = select_scan_target(observed, cost, candidates, {"x": 0, "y": 2}, radius=0)
        with_corridor = select_scan_target(
            observed, cost, candidates, {"x": 0, "y": 2}, radius=0,
            mission_path=[{"x": 0, "y": 3}, {"x": 6, "y": 3}],
        )

        self.assertEqual(without_corridor["target"], {"x": 0, "y": 0})
        self.assertEqual(with_corridor["target"], {"x": 2, "y": 3})

    def test_existing_route_replaces_the_straight_corridor(self):
        observed = [[True] * 7 for _ in range(7)]
        observed[0][3] = observed[3][3] = False
        cost = [[0.2 if seen else None for seen in row] for row in observed]
        candidates = [{"x": 3, "y": 0}, {"x": 3, "y": 3}]

        result = select_scan_target(
            observed, cost, candidates, {"x": 0, "y": 3}, radius=0,
            mission_path=[{"x": 0, "y": 3}, {"x": 0, "y": 0},
                          {"x": 6, "y": 0}, {"x": 6, "y": 3}],
        )

        self.assertEqual(result["target"], {"x": 3, "y": 0})


class ServerAuthoritativeMapTests(unittest.TestCase):
    def test_default_resolution_keeps_physical_extent_and_uses_finer_scan_stops(self):
        world = make_world(515, 144, 108)
        self.assertEqual(world["cell_size_m"], 50)
        self.assertEqual((world["width"] * world["cell_size_m"],
                          world["height"] * world["cell_size_m"]), (7200, 5400))
        self.assertEqual(len(world["layers"]["elevation"]), 108)
        self.assertEqual(len(world["layers"]["elevation"][0]), 144)

        footprint = server._scan_footprint_offsets(4)
        for pattern_seed in range(4):
            candidates = server._make_scan_candidates(144, 108, world["start"], pattern_seed)
            self.assertTrue(all(0 <= point["x"] < 144 and 0 <= point["y"] < 108
                                for point in candidates))
            covered = [[False] * world["width"] for _ in range(world["height"])]
            for point in candidates:
                for dx, dy in footprint:
                    x, y = point["x"] + dx, point["y"] + dy
                    if 0 <= x < world["width"] and 0 <= y < world["height"]:
                        covered[y][x] = True
            self.assertTrue(all(map(all, covered)),
                            f"200m footprint for scan pattern {pattern_seed} should cover the full map")
            if pattern_seed == 0:
                # Along-sweep waypoints are now 200m apart on the 50m grid.
                first_lane = [point["x"] for point in candidates if point["y"] == 0]
                self.assertTrue(any(b - a == 4 for a, b in zip(first_lane, first_lane[1:])))
                lanes = sorted({point["y"] for point in candidates})
                self.assertLessEqual(max(b - a for a, b in zip(lanes, lanes[1:])), 6)
        reference_radius_cells = sum(dx * dx + dy * dy <= 4 for dy in range(-2, 3) for dx in range(-2, 3))
        sensor_radius_cells = sum(dx * dx + dy * dy <= 16 for dy in range(-4, 5) for dx in range(-4, 5))
        self.assertLess(abs(sensor_radius_cells * 50**2 / (reference_radius_cells * 100**2) - 1), 0.1)
        self.assertLess(abs(sensor_radius_cells * 50**2 / (math.pi * 200**2) - 1), 0.03)
        self.assertEqual(server.GOAL_CORRIDOR_RADIUS * world["cell_size_m"], 500)
        self.assertEqual(server.LOCAL_SCAN_RADIUS * world["cell_size_m"], 225)

    def test_scan_selection_prediction_matches_cells_sensed_at_selected_center(self):
        world = create_world_session(516, 24, 18)
        center = {"x": 12, "y": 9}
        observed = [[False] * world["width"] for _ in range(world["height"])]
        cost = [[0.2] * world["width"] for _ in range(world["height"])]

        choice = select_scan_target(observed, cost, [center], center, radius=4)
        measurement = sense_session(world["session_id"], center, radius=4)
        predicted_cells = {
            (center["x"] + dx, center["y"] + dy)
            for dx, dy in server._scan_footprint_offsets(4)
        }
        sensed_cells = {(cell["x"], cell["y"]) for cell in measurement["cells"]}

        self.assertEqual(choice["target"], center)
        self.assertEqual(choice["new_cells"], len(predicted_cells))
        self.assertEqual(sensed_cells, predicted_cells)

    def test_sensor_noise_is_reproducible_and_remains_near_standard_normal(self):
        def seed(sequence, y, x, layer):
            return (515 * 73856093 + sequence * 19349663 + y * 83492791
                    + x * 2654435761 + layer * 97531) & 0xFFFFFFFF

        sequence_values = [server._sensor_noise(seed(index, 41, 47, 2))
                           for index in range(1, 4097)]
        spatial_values, other_layer_values = [], []
        for index in range(4096):
            y, x = divmod(index, 128)
            spatial_values.append(server._sensor_noise(seed(9, y, x, 2)))
            other_layer_values.append(server._sensor_noise(seed(9, y, x, 3)))

        self.assertEqual(sequence_values,
                         [server._sensor_noise(seed(index, 41, 47, 2)) for index in range(1, 4097)])
        self.assertLess(abs(statistics.mean(sequence_values)), 0.05)
        self.assertLess(abs(statistics.pstdev(sequence_values) - 1.0), 0.05)
        within_two_sigma = sum(abs(value) <= 2 for value in sequence_values) / len(sequence_values)
        beyond_three_sigma = sum(abs(value) > 3 for value in sequence_values) / len(sequence_values)
        self.assertLess(abs(within_two_sigma - 0.9545), 0.01)
        self.assertLess(abs(beyond_three_sigma - 0.0027), 0.0015)
        self.assertLess(abs(statistics.correlation(sequence_values[:-1], sequence_values[1:])), 0.05)
        self.assertLess(abs(statistics.correlation(spatial_values, other_layer_values)), 0.05)
        self.assertLess(abs(statistics.correlation(spatial_values[:-1], spatial_values[1:])), 0.05)

    def test_identical_world_seeds_produce_identical_sensor_measurements(self):
        first = create_world_session(517, 24, 18)
        second = create_world_session(517, 24, 18)
        center = {"x": 12, "y": 9}

        first_measurement = sense_session(first["session_id"], center, radius=1)
        second_measurement = sense_session(second["session_id"], center, radius=1)

        self.assertEqual(first_measurement["cells"], second_measurement["cells"])

    def setUp(self):
        with server._SESSIONS_LOCK:
            server._SESSIONS.clear()

    def test_scan_helpers_reject_non_integer_radii_clearly(self):
        for radius in (2.5, True, "2"):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "扫描半径"):
                    sense_session("missing", {"x": 0, "y": 0}, radius=radius)
                with self.assertRaisesRegex(ValueError, "扫描半径"):
                    sense_flight_segment("missing", {"x": 0, "y": 0}, {"x": 1, "y": 0}, radius=radius)

    def test_benchmark_releases_each_policy_session_before_the_next_run(self):
        with patch("benchmark._run_policy", return_value={"complete": True}):
            for seed in range(server.MAX_SESSIONS + 1):
                with self.subTest(seed=seed):
                    self.assertEqual(run_policy(seed, 24, 18, active=False), {"complete": True})
                    self.assertEqual(server._SESSIONS, {})

    def test_benchmark_releases_policy_session_after_an_error(self):
        with patch("benchmark._run_policy", side_effect=RuntimeError("benchmark failed")):
            with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                run_policy(123, 24, 18, active=False)

        self.assertEqual(server._SESSIONS, {})

    def test_benchmark_oracle_scores_the_session_truth(self):
        result = run_policy(202, 24, 18, active=False)

        self.assertTrue(result["oracle_reachable"])
        self.assertTrue(result["route_found"])
        self.assertFalse(result["unknown_or_obstacle_violation"])
        self.assertLess(result["cost_regret_ratio"], 1.5)
        self.assertEqual(server._SESSIONS, {})

    def test_benchmark_seed_recreates_the_same_world_across_sessions(self):
        script = """
import hashlib, json, benchmark, server
def capture(world, *_args):
    layers = server._get_session(world['session_id'])['truth']['layers']
    print(hashlib.sha256(json.dumps(layers, sort_keys=True).encode()).hexdigest())
    return {}
benchmark._run_policy = capture
benchmark.run_policy(202, 24, 18, active=False)
"""
        process_hashes = []
        for _ in range(2):
            output = subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parent,
                text=True,
            )
            process_hashes.append(output.strip().splitlines()[-1])

        self.assertEqual(process_hashes[0], process_hashes[1])

    def test_default_mission_endpoints_move_off_hard_blocked_cells(self):
        page_id = "benchmark-safe-endpoints"
        world = server.create_world_session(
            151, 36, 28, page_id=page_id,
            _world_seed_key=benchmark._BENCHMARK_WORLD_SEED_KEY,
        )
        try:
            truth = _get_session(world["session_id"])["truth"]
            cost = benchmark_cost(truth)
            oracle = plan_path([[True] * truth["width"] for _ in range(truth["height"])],
                               cost, world["start"], world["goal"])

            self.assertLess(cost[world["start"]["y"]][world["start"]["x"]], 50)
            self.assertLess(cost[world["goal"]["y"]][world["goal"]["x"]], 50)
            self.assertTrue(oracle["path"])
        finally:
            server.release_world_session(world["session_id"], world["release_token"], page_id)

    def test_benchmark_reports_missing_route_separately_from_path_violations(self):
        with patch("benchmark.plan_session", return_value={"path": [], "cost": None}):
            result = run_policy(202, 24, 18, active=False)

        self.assertTrue(result["oracle_reachable"])
        self.assertFalse(result["route_found"])
        self.assertIsNone(result["unknown_or_obstacle_violation"])

    def test_benchmark_route_legality_requires_requested_endpoints(self):
        observed = [[True, True, True]]
        cost = [[0.1, 0.1, 0.1]]
        start, goal = {"x": 0, "y": 0}, {"x": 2, "y": 0}

        self.assertTrue(valid_route([start, {"x": 1, "y": 0}, goal], observed, cost, start, goal))
        self.assertFalse(valid_route([start, {"x": 1, "y": 0}], observed, cost, start, goal))

    def test_benchmark_path_cost_keeps_precision_until_reporting(self):
        route = [{"x": 0, "y": 0}, {"x": 1, "y": 0}]
        self.assertEqual(path_cost(route, [[0.0, 0.123456]]), 0.123456)

    def test_infrastructure_distance_proxies_follow_their_feature_layers(self):
        world = make_world(123, 72, 54)
        for feature_name in ("railway", "fault", "existing_line"):
            feature = world["layers"][feature_name]
            proxy = world["layers"][f"dist_{feature_name}"]
            feature_cells = [(x, y) for y, row in enumerate(feature)
                             for x, value in enumerate(row) if value > 0.5]
            self.assertTrue(feature_cells)
            for y in range(world["height"]):
                for x in range(world["width"]):
                    nearest = min((x - fx) ** 2 + (y - fy) ** 2 for fx, fy in feature_cells)
                    expected = min(1.0, math.sqrt(nearest) / (800.0 / world["cell_size_m"]))
                    self.assertAlmostEqual(proxy[y][x], expected, delta=0.08,
                                           msg=f"{feature_name} at ({x}, {y})")

    def test_random_maps_show_water_and_keep_the_hard_channel_inside_the_map(self):
        for width, height in ((24, 18), (72, 54), (100, 72)):
            for seed in range(5):
                with self.subTest(width=width, height=height, seed=seed):
                    layers = make_world(seed, width, height)["layers"]
                    water = layers["water"]
                    visible_cells = sum(value > 0.35 for row in water for value in row)
                    river_by_x = {}
                    for y, row in enumerate(water):
                        for x, value in enumerate(row):
                            if value > 0.90:
                                river_by_x.setdefault(x, []).append(y)
                    hard_water_cells = [(x, y) for x, ys in river_by_x.items() for y in ys]
                    self.assertGreater(visible_cells, width * height * 0.05)
                    self.assertLess(visible_cells, width * height * 0.20)
                    self.assertGreaterEqual(len(hard_water_cells), int(width * 0.4))
                    self.assertTrue(all(width * 0.15 < x < width * 0.85
                                        for x, _ in hard_water_cells))
                    river_columns = sorted(river_by_x)
                    self.assertEqual(river_columns, list(range(river_columns[0], river_columns[-1] + 1)))
                    self.assertTrue(all(min(abs(y - neighbor_y)
                                            for y in river_by_x[x]
                                            for neighbor_y in river_by_x[x + 1]) <= 1
                                        for x in river_columns[:-1]))
                    self.assertTrue(all(layers["hard_blocked"][y][x] == 1.0
                                        for x, y in hard_water_cells))

    def test_flight_scan_request_id_replays_the_committed_result(self):
        world = create_world_session(914, 24, 18)
        session = _get_session(world["session_id"])
        args = (world["session_id"], {"x": 2, "y": 2}, {"x": 8, "y": 2})

        first = sense_flight_segment(*args, radius=1, request_id="scan-retry-1")
        sequence = session["scan_sequence"]
        samples = sum(sum(row) for row in session["samples"])
        replay = sense_flight_segment(*args, radius=1, request_id="scan-retry-1")

        self.assertEqual(replay, first)
        self.assertEqual(session["scan_sequence"], sequence)
        self.assertEqual(sum(sum(row) for row in session["samples"]), samples)
        with self.assertRaisesRegex(ValueError, "其他航段"):
            sense_flight_segment(world["session_id"], {"x": 2, "y": 2}, {"x": 9, "y": 2},
                                 radius=1, request_id="scan-retry-1")

    def test_single_scan_rolls_back_partial_cell_fusion_on_error(self):
        world = create_world_session(918, 24, 18)
        session = _get_session(world["session_id"])
        before = copy.deepcopy({
            "layers": session["layers"], "cost": session["cost"], "samples": session["samples"],
            "scan_sequence": session["scan_sequence"], "snapshot_revision": session["snapshot_revision"],
        })
        original_cost = server._cell_cost
        calls = 0

        def fail_on_third_cell(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("cost update failed")
            return original_cost(*args, **kwargs)

        with patch.object(server, "_cell_cost", side_effect=fail_on_third_cell):
            with self.assertRaisesRegex(RuntimeError, "cost update failed"):
                sense_session(world["session_id"], {"x": 10, "y": 9}, radius=1)

        self.assertEqual(session["layers"], before["layers"])
        self.assertEqual(session["cost"], before["cost"])
        self.assertEqual(session["samples"], before["samples"])
        self.assertEqual(session["scan_sequence"], before["scan_sequence"])
        self.assertEqual(session["snapshot_revision"], before["snapshot_revision"])

    def test_flight_scan_rolls_back_partial_samples_before_retry(self):
        world = create_world_session(919, 24, 18)
        session = _get_session(world["session_id"])
        sense_session(world["session_id"], {"x": 5, "y": 15}, radius=1)
        start, end = {"x": 2, "y": 15}, {"x": 8, "y": 15}
        request_id = "atomic-scan-retry"
        session["drone_position"] = dict(start)
        session["active_scan_leg"] = {
            "leg_id": "atomic-scan-leg", "start": dict(start), "target": dict(end), "radius": 2,
            "choice": {"target": dict(end), "leg_id": "atomic-scan-leg"},
        }
        before = copy.deepcopy({
            "layers": session["layers"], "cost": session["cost"], "samples": session["samples"],
            "scan_sequence": session["scan_sequence"], "snapshot_revision": session["snapshot_revision"],
            "drone_position": session["drone_position"], "previous_drone_position": session["previous_drone_position"],
            "active_scan_leg": session["active_scan_leg"], "flight_distance": session.get("flight_distance", 0),
            "scan_requests": session["scan_requests"], "scan_request_receipts": session["scan_request_receipts"],
        })
        original_sense = server.sense_session
        original_cost = server._cell_cost
        centers = 0
        cost_calls = 0

        def fail_on_second_cell_cost(*args, **kwargs):
            nonlocal cost_calls
            cost_calls += 1
            if cost_calls == 2:
                raise RuntimeError("cost update failed")
            return original_cost(*args, **kwargs)

        def fail_inside_third_center(*args, **kwargs):
            nonlocal centers
            centers += 1
            if centers == 3:
                with patch.object(server, "_cell_cost", side_effect=fail_on_second_cell_cost):
                    return original_sense(*args, **kwargs)
            return original_sense(*args, **kwargs)

        with patch.object(server, "sense_session", side_effect=fail_inside_third_center):
            with self.assertRaisesRegex(RuntimeError, "cost update failed"):
                sense_flight_segment(world["session_id"], start, end, request_id=request_id,
                                     leg_id="atomic-scan-leg", require_authorization=True)

        self.assertEqual(centers, 3)
        self.assertEqual(cost_calls, 2)

        self.assertEqual(session["layers"], before["layers"])
        self.assertEqual(session["cost"], before["cost"])
        self.assertEqual(session["samples"], before["samples"])
        self.assertEqual(session["scan_sequence"], before["scan_sequence"])
        self.assertEqual(session["snapshot_revision"], before["snapshot_revision"])
        self.assertEqual(session["drone_position"], before["drone_position"])
        self.assertEqual(session["previous_drone_position"], before["previous_drone_position"])
        self.assertEqual(session["active_scan_leg"], before["active_scan_leg"])
        self.assertEqual(session.get("flight_distance", 0), before["flight_distance"])
        self.assertEqual(session["scan_requests"], before["scan_requests"])
        self.assertEqual(session["scan_request_receipts"], before["scan_request_receipts"])

        result = sense_flight_segment(world["session_id"], start, end, request_id=request_id,
                                      leg_id="atomic-scan-leg", require_authorization=True)
        self.assertEqual(result["sample_centers"], 4)
        self.assertEqual(session["scan_sequence"], before["scan_sequence"] + 4)
        self.assertEqual(session["snapshot_revision"], before["snapshot_revision"] + 4)
        sample_increment = sum(sum(row) for row in session["samples"]) - sum(sum(row) for row in before["samples"])
        reported_increment = sum(cell["samples"] - before["samples"][cell["y"]][cell["x"]]
                                 for cell in result["cells"])
        self.assertEqual(sample_increment, reported_increment)
        self.assertEqual(session["drone_position"], end)
        self.assertEqual(session["flight_distance"], 6)
        self.assertIn(request_id, session["scan_requests"])
        self.assertIn(request_id, session["scan_request_receipts"])

    def test_replayed_flight_scan_returns_current_fused_values_after_an_overlapping_scan(self):
        world = create_world_session(913, 24, 18)
        session = _get_session(world["session_id"])
        args = (world["session_id"], {"x": 2, "y": 2}, {"x": 8, "y": 2})

        first = sense_flight_segment(*args, radius=1, request_id="overlap-retry-a")
        cell = first["cells"][0]
        first_samples = cell["samples"]
        sense_flight_segment(*args, radius=1, request_id="overlap-retry-b")
        sequence = session["scan_sequence"]
        samples = sum(sum(row) for row in session["samples"])

        replay = sense_flight_segment(*args, radius=1, request_id="overlap-retry-a")
        replay_cell = next(item for item in replay["cells"] if (item["x"], item["y"]) == (cell["x"], cell["y"]))
        current_samples = session["samples"][cell["y"]][cell["x"]]
        current_layers = {name: session["layers"][name][cell["y"]][cell["x"]] for name in server.LAYERS}

        self.assertGreater(replay_cell["samples"], first_samples)
        self.assertEqual(replay_cell["samples"], current_samples)
        self.assertEqual(replay_cell["confidence"], round(current_samples / (current_samples + 2.0), 3))
        self.assertEqual(replay_cell["layers"], current_layers)
        self.assertEqual(session["scan_sequence"], sequence, "replay must not collect another sensor sample")
        self.assertEqual(sum(sum(row) for row in session["samples"]), samples)

    def test_authorized_flight_segments_allow_slow_response_batches_and_ordered_replay(self):
        world = create_world_session(912, 24, 18)
        session = _get_session(world["session_id"])
        session["drone_position"] = {"x": 2, "y": 2}
        session["active_scan_leg"] = {
            "leg_id": "long-leg", "start": {"x": 2, "y": 2},
            "target": {"x": 20, "y": 2}, "radius": 2,
            "choice": {"target": {"x": 20, "y": 2}, "leg_id": "long-leg"},
        }

        first = sense_flight_segment(world["session_id"], {"x": 2, "y": 2}, {"x": 10, "y": 2},
                                     request_id="long-first", leg_id="long-leg", require_authorization=True)
        sequence = session["scan_sequence"]
        replay = sense_flight_segment(world["session_id"], {"x": 2, "y": 2}, {"x": 10, "y": 2},
                                      request_id="long-first", leg_id="long-leg", require_authorization=True)
        self.assertEqual(replay, first)
        self.assertEqual(session["scan_sequence"], sequence)
        self.assertEqual(session["drone_position"], {"x": 10, "y": 2})

        second = sense_flight_segment(world["session_id"], {"x": 10, "y": 2}, {"x": 20, "y": 2},
                                      request_id="long-second", leg_id="long-leg", require_authorization=True)
        self.assertEqual(second["sample_centers"], 7)
        self.assertEqual(session["drone_position"], {"x": 20, "y": 2})
        self.assertIsNone(session["active_scan_leg"])

        overlap = next(cell for cell in first["cells"]
                       if session["samples"][cell["y"]][cell["x"]] > cell["samples"])
        sequence = session["scan_sequence"]
        samples = sum(sum(row) for row in session["samples"])
        refreshed = sense_flight_segment(world["session_id"], {"x": 2, "y": 2}, {"x": 10, "y": 2},
                                         request_id="long-first", leg_id="long-leg", require_authorization=True)
        refreshed_overlap = next(cell for cell in refreshed["cells"]
                                 if (cell["x"], cell["y"]) == (overlap["x"], overlap["y"]))
        current_samples = session["samples"][overlap["y"]][overlap["x"]]
        self.assertEqual(refreshed_overlap["samples"], current_samples)
        self.assertEqual(refreshed_overlap["layers"], {
            name: session["layers"][name][overlap["y"]][overlap["x"]] for name in server.LAYERS
        })
        self.assertEqual(refreshed["flight_distance"], second["flight_distance"],
                         "an older request replay must report the latest committed session distance")
        self.assertEqual(refreshed["scan_sequence"], sequence,
                         "an older request replay must report the latest fused scan sequence")
        self.assertEqual(session["scan_sequence"], sequence, "authorized replay must not collect another sensor sample")
        self.assertEqual(sum(sum(row) for row in session["samples"]), samples)

    def test_authorized_flight_rejects_jumps_without_mutating_scan_state(self):
        world = create_world_session(911, 24, 18)
        session = _get_session(world["session_id"])
        session["drone_position"] = {"x": 2, "y": 9}
        session["active_scan_leg"] = {
            "leg_id": "straight-leg", "start": {"x": 2, "y": 9},
            "target": {"x": 12, "y": 9}, "radius": 2,
            "choice": {"target": {"x": 12, "y": 9}, "leg_id": "straight-leg"},
        }

        with self.assertRaisesRegex(ValueError, "授权目标方向"):
            sense_flight_segment(world["session_id"], {"x": 2, "y": 9}, {"x": 12, "y": 12},
                                 request_id="off-path", leg_id="straight-leg", require_authorization=True)

        self.assertEqual(session["scan_sequence"], 0)
        self.assertEqual(session["drone_position"], {"x": 2, "y": 9})
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_authorized_flight_cannot_accumulate_reverse_progress(self):
        world = create_world_session(915, 64, 32)
        session = _get_session(world["session_id"])
        origin, target = {"x": 20, "y": 20}, {"x": 60, "y": 20}
        session["drone_position"] = dict(origin)
        session["active_scan_leg"] = {
            "leg_id": "reverse-leg", "start": origin, "target": target, "radius": 2,
            "choice": {"target": target, "leg_id": "reverse-leg"},
        }

        for index in range(20):
            start = dict(session["drone_position"])
            with self.subTest(attempt=index), self.assertRaises(ValueError):
                sense_flight_segment(world["session_id"], start, {"x": start["x"] - 1, "y": 20},
                                     request_id=f"reverse-{index}",
                                     leg_id="reverse-leg", require_authorization=True)
            self.assertEqual(session["drone_position"], origin)
            self.assertEqual(session["scan_sequence"], 0)
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_authorized_flight_rejects_parallel_drift_outside_rounding_tolerance(self):
        world = create_world_session(916, 64, 32)
        session = _get_session(world["session_id"])
        origin, target = {"x": 20, "y": 20}, {"x": 60, "y": 20}
        session["drone_position"] = dict(origin)
        session["active_scan_leg"] = {
            "leg_id": "drift-leg", "start": origin, "target": target, "radius": 2,
            "choice": {"target": target, "leg_id": "drift-leg"},
        }

        with self.assertRaisesRegex(ValueError, "授权目标方向"):
            sense_flight_segment(world["session_id"], origin, {"x": 21, "y": 21},
                                 request_id="parallel-drift", leg_id="drift-leg",
                                 require_authorization=True)

        self.assertEqual(session["drone_position"], origin)
        self.assertEqual(session["scan_sequence"], 0)
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_authorized_diagonal_flight_accepts_rounded_points(self):
        world = create_world_session(917, 64, 32)
        session = _get_session(world["session_id"])
        origin, target = {"x": 20, "y": 10}, {"x": 50, "y": 30}
        session["drone_position"] = dict(origin)
        session["active_scan_leg"] = {
            "leg_id": "rounded-diagonal", "start": origin, "target": target, "radius": 2,
            "choice": {"target": target, "leg_id": "rounded-diagonal"},
        }

        first = sense_flight_segment(world["session_id"], origin, {"x": 21, "y": 11},
                                     request_id="rounded-first", leg_id="rounded-diagonal",
                                     require_authorization=True)
        self.assertTrue(first["cells"])
        self.assertEqual(session["drone_position"], {"x": 21, "y": 11})

    def test_authorized_shallow_diagonal_distance_tracks_progress_not_rounded_chords(self):
        world = create_world_session(9171, 64, 32)
        session = _get_session(world["session_id"])
        origin, target = {"x": 0, "y": 0}, {"x": 40, "y": 2}
        session["drone_position"] = dict(origin)
        session["active_scan_leg"] = {
            "leg_id": "shallow-diagonal", "start": origin, "target": target, "radius": 2,
            "choice": {"target": target, "leg_id": "shallow-diagonal"},
        }
        stops = [
            {"x": 5, "y": 0}, {"x": 10, "y": 1}, {"x": 15, "y": 1},
            {"x": 20, "y": 1}, {"x": 25, "y": 1}, {"x": 30, "y": 2},
            {"x": 35, "y": 2}, target,
        ]

        distances = []
        for index, stop in enumerate(stops):
            result = sense_flight_segment(
                world["session_id"], session["drone_position"], stop,
                request_id=f"shallow-diagonal-{index}", leg_id="shallow-diagonal",
                require_authorization=True,
            )
            distances.append(result["flight_distance"])

        expected = math.hypot(target["x"] - origin["x"], target["y"] - origin["y"])
        self.assertEqual(distances, sorted(distances), "confirmed distance must never move backward")
        self.assertAlmostEqual(distances[-1], expected, places=6)

    def test_server_owned_scan_pattern_finishes_residual_cells_and_cost_fusion(self):
        world = create_world_session(910, 24, 18)
        session = _get_session(world["session_id"])

        for index in range(400):
            choice = select_session_scan(world["session_id"], radius=2, guidance_weight=1.0)
            self.assertEqual(choice["drone_position"], session["drone_position"])
            if choice["target"] is None:
                break
            sense_flight_segment(world["session_id"], session["drone_position"], choice["target"],
                                 radius=2, request_id=f"coverage-{index}",
                                 leg_id=choice["leg_id"], require_authorization=True)

        self.assertIsNone(choice["target"])
        self.assertEqual(sum(value == 0 for row in session["samples"] for value in row), 0)
        self.assertTrue(all(value is not None for row in session["cost"] for value in row))
        self.assertIsNone(session["active_scan_leg"])

    def test_scan_selection_discards_route_guidance_for_old_endpoints(self):
        world = create_world_session(909, 24, 18)
        session = _get_session(world["session_id"])
        session["route"] = [world["start"], {"x": 10, "y": 10}, world["goal"]]
        changed_goal = {"x": world["goal"]["x"], "y": world["goal"]["y"] - 1}

        configure_session_mission(world["session_id"], world["start"], changed_goal)
        choice = select_session_scan(world["session_id"])

        self.assertEqual(session["route"], [])
        self.assertEqual(choice["reason"], "goal_corridor")
        self.assertNotIn((world["goal"]["x"], world["goal"]["y"]), session["guidance_key"])

    def test_mission_change_revokes_unstarted_scan_leg_and_reanchors_drone(self):
        world = create_world_session(908, 24, 18)
        session = _get_session(world["session_id"])
        selected_start = {"x": 5, "y": 5}
        configure_session_mission(world["session_id"], selected_start, world["goal"])
        choice = select_session_scan(world["session_id"])
        self.assertEqual(session["drone_position"], selected_start)

        changed_start = {"x": 8, "y": 8}
        configured = configure_session_mission(world["session_id"], changed_start, world["goal"])
        self.assertEqual(configured["drone_position"], changed_start,
                         "a selection that has not begun moving must re-anchor to the edited start")
        self.assertIsNone(session["active_scan_leg"])
        repeated = select_session_scan(world["session_id"])
        self.assertNotEqual(repeated["leg_id"], choice["leg_id"])
        self.assertEqual(session["drone_position"], changed_start)

        sense_flight_segment(world["session_id"], changed_start, repeated["target"], radius=2,
                             request_id="chosen-start", leg_id=repeated["leg_id"],
                             require_authorization=True)
        moved_to = dict(session["drone_position"])
        configured = configure_session_mission(world["session_id"], changed_start, world["goal"])
        self.assertEqual(configured["drone_position"], moved_to)
        select_session_scan(world["session_id"])
        self.assertEqual(session["drone_position"], moved_to)

    def test_flight_scan_finishes_if_expired_session_is_removed_mid_request(self):
        world = create_world_session(913, 24, 18)
        session = _get_session(world["session_id"])
        original = server.sense_session
        expired_time = session["lease_until"] + 1

        def evict_then_measure(*args, **kwargs):
            with patch("server.time.monotonic", return_value=expired_time):
                create_world_session(920, 24, 18)
            return original(*args, **kwargs)

        with patch("server.sense_session", side_effect=evict_then_measure):
            result = sense_flight_segment(
                world["session_id"], {"x": 2, "y": 2}, {"x": 3, "y": 2}, radius=1
            )

        self.assertEqual(result["sample_centers"], 1)
        self.assertTrue(result["cells"])
        self.assertEqual(session["scan_sequence"], 1)

    def test_replacing_a_session_waits_for_its_authorized_scan_to_commit(self):
        world = create_world_session(915, 24, 18)
        session = _get_session(world["session_id"])
        choice = select_session_scan(world["session_id"])
        scan_entered = Event()
        allow_scan = Event()
        replacement_ready = Event()
        replacement_done = Event()
        scan_result = []
        replacement_result = []
        thread_errors = []
        original_sense = server.sense_session
        original_make_world = server.make_world

        def hold_scan(*args, **kwargs):
            scan_entered.set()
            if not allow_scan.wait(2):
                raise RuntimeError("scan test timed out")
            return original_sense(*args, **kwargs)

        def make_replacement(*args, **kwargs):
            result = original_make_world(*args, **kwargs)
            replacement_ready.set()
            return result

        def scan():
            try:
                scan_result.append(sense_flight_segment(
                    world["session_id"], choice["drone_position"], choice["target"], radius=2,
                    request_id="replace-race-scan", leg_id=choice["leg_id"], require_authorization=True,
                ))
            except Exception as error:
                thread_errors.append(error)

        def replace():
            try:
                replacement_result.append(create_world_session(
                    916, 24, 18, replace_session_id=world["session_id"],
                ))
            except Exception as error:
                thread_errors.append(error)
            finally:
                replacement_done.set()

        with patch.object(server, "sense_session", side_effect=hold_scan), \
                patch.object(server, "make_world", side_effect=make_replacement):
            scan_thread = Thread(target=scan)
            scan_thread.start()
            replacement_thread = None
            try:
                self.assertTrue(scan_entered.wait(2), "the authorized scan should enter its critical section")
                replacement_thread = Thread(target=replace)
                replacement_thread.start()
                self.assertTrue(replacement_ready.wait(2), "replacement terrain should be generated")
                self.assertFalse(replacement_done.wait(.05), "replacement must wait for the active scan lock")
                self.assertIs(server._SESSIONS.get(world["session_id"]), session)
            finally:
                allow_scan.set()
                scan_thread.join(2)
                if replacement_thread:
                    replacement_thread.join(2)

        self.assertFalse(scan_thread.is_alive())
        self.assertIsNotNone(replacement_thread)
        self.assertFalse(replacement_thread.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertEqual(len(scan_result), 1)
        self.assertEqual(len(replacement_result), 1)
        self.assertTrue(scan_result[0]["cells"])
        self.assertGreater(session["scan_sequence"], 0, "the acknowledged scan must commit before retirement")
        self.assertNotIn(world["session_id"], server._SESSIONS)
        self.assertIn(replacement_result[0]["session_id"], server._SESSIONS)

    def test_releasing_a_session_waits_for_its_authorized_scan_to_commit(self):
        world = create_world_session(917, 24, 18, page_id="scan-page")
        session = _get_session(world["session_id"])
        choice = select_session_scan(world["session_id"])
        scan_entered = Event()
        allow_scan = Event()
        release_started = Event()
        release_done = Event()
        scan_result = []
        release_result = []
        thread_errors = []
        original_sense = server.sense_session

        def hold_scan(*args, **kwargs):
            scan_entered.set()
            if not allow_scan.wait(2):
                raise RuntimeError("scan test timed out")
            return original_sense(*args, **kwargs)

        def scan():
            try:
                scan_result.append(sense_flight_segment(
                    world["session_id"], choice["drone_position"], choice["target"], radius=2,
                    request_id="release-race-scan", leg_id=choice["leg_id"], require_authorization=True,
                ))
            except Exception as error:
                thread_errors.append(error)

        def release():
            release_started.set()
            try:
                release_result.append(server.release_world_session(
                    world["session_id"], world["release_token"], "scan-page",
                ))
            except Exception as error:
                thread_errors.append(error)
            finally:
                release_done.set()

        with patch.object(server, "sense_session", side_effect=hold_scan):
            scan_thread = Thread(target=scan)
            scan_thread.start()
            release_thread = None
            try:
                self.assertTrue(scan_entered.wait(2), "the authorized scan should enter its critical section")
                release_thread = Thread(target=release)
                release_thread.start()
                self.assertTrue(release_started.wait(2))
                self.assertFalse(release_done.wait(.05), "release must wait for the active scan lock")
                self.assertIs(server._SESSIONS.get(world["session_id"]), session)
            finally:
                allow_scan.set()
                scan_thread.join(2)
                if release_thread:
                    release_thread.join(2)

        self.assertFalse(scan_thread.is_alive())
        self.assertIsNotNone(release_thread)
        self.assertFalse(release_thread.is_alive())
        self.assertEqual(thread_errors, [])
        self.assertEqual(len(scan_result), 1)
        self.assertTrue(scan_result[0]["cells"])
        self.assertGreater(session["scan_sequence"], 0, "the acknowledged scan must commit before release")
        self.assertEqual(release_result, [True])
        self.assertNotIn(world["session_id"], server._SESSIONS)

    def test_replayed_world_requests_register_separate_page_owners(self):
        first = create_world_session(918, 24, 18, request_id="shared-world-request", page_id="page-one")
        replay = create_world_session(918, 24, 18, request_id="shared-world-request", page_id="page-two")
        session = server._SESSIONS[first["session_id"]]

        self.assertEqual(replay["session_id"], first["session_id"])
        self.assertTrue(replay["request_replayed"])
        self.assertNotEqual(replay["release_token"], first["release_token"])
        self.assertEqual(set(session["owners"]), {"page-one", "page-two"})
        self.assertTrue(server.release_world_session(first["session_id"], first["release_token"], "page-one"))
        self.assertIn(first["session_id"], server._SESSIONS,
                      "releasing one replay owner must retain the other page's session")
        self.assertTrue(server.release_world_session(replay["session_id"], replay["release_token"], "page-two"))
        self.assertNotIn(first["session_id"], server._SESSIONS)

    def test_session_snapshot_waits_for_a_complete_scan_commit(self):
        world = create_world_session(9182, 24, 18)
        session = _get_session(world["session_id"])
        started, completed = Event(), Event()
        snapshot_result = []

        def read_snapshot():
            started.set()
            snapshot_result.append(server.session_scan_snapshot(world["session_id"]))
            completed.set()

        session["lock"].acquire()
        reader = Thread(target=read_snapshot)
        try:
            reader.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(completed.wait(.05), "snapshot must wait for the in-flight session update")
            sense_session(world["session_id"], {"x": 7, "y": 8}, radius=1, _session=session)
        finally:
            session["lock"].release()
            reader.join(2)

        self.assertFalse(reader.is_alive())
        self.assertEqual(snapshot_result[0]["scan_sequence"], session["scan_sequence"])
        self.assertEqual(len(snapshot_result[0]["cells"]), sum(
            count > 0 for row in session["samples"] for count in row
        ))

    def test_session_snapshot_racing_last_owner_release_is_atomic(self):
        world = create_world_session(9184, 24, 18, page_id="snapshot-race-page")
        sense_session(world["session_id"], {"x": 7, "y": 8}, radius=1)
        session = _get_session(world["session_id"])
        snapshot_started, release_started = Event(), Event()
        outcomes = []

        def read_snapshot():
            snapshot_started.set()
            try:
                outcomes.append(("snapshot", server.session_scan_snapshot(world["session_id"])))
            except ValueError as error:
                outcomes.append(("expired", str(error)))

        def release_owner():
            release_started.set()
            outcomes.append(("released", server.release_world_session(
                world["session_id"], world["release_token"], "snapshot-race-page",
            )))

        session["lock"].acquire()
        reader, releaser = Thread(target=read_snapshot), Thread(target=release_owner)
        try:
            reader.start()
            releaser.start()
            self.assertTrue(snapshot_started.wait(2))
            self.assertTrue(release_started.wait(2))
            self.assertFalse(outcomes, "both operations should wait for the held session lock")
        finally:
            session["lock"].release()
            reader.join(2)
            releaser.join(2)

        self.assertFalse(reader.is_alive())
        self.assertFalse(releaser.is_alive())
        self.assertEqual(next(value for kind, value in outcomes if kind == "released"), True)
        self.assertNotIn(world["session_id"], server._SESSIONS)
        result = next((value for kind, value in outcomes if kind == "snapshot"), None)
        error = next((value for kind, value in outcomes if kind == "expired"), None)
        self.assertTrue(result is not None or error is not None)
        if result is not None:
            self.assertTrue(result["cells"])
            self.assertEqual(len(result["cells"]), sum(
                count > 0 for row in session["samples"] for count in row
            ))
        else:
            self.assertRegex(error, "不存在|过期|替换")

    def test_missed_pagehide_release_is_bounded_by_session_lease(self):
        with patch("server.time.monotonic", return_value=100.0):
            first = create_world_session(9180, 24, 18, request_id="same-tab-world-request", page_id="old-document")
            replay = create_world_session(9180, 24, 18, request_id="same-tab-world-request", page_id="reloaded-document")
            self.assertTrue(server.release_world_session(
                replay["session_id"], replay["release_token"], "reloaded-document",
            ))

        self.assertEqual(server._SESSIONS[first["session_id"]]["owners"], {"old-document"})
        with patch("server.time.monotonic", return_value=100.0 + server.SESSION_LEASE_SECONDS + 1):
            create_world_session(9183, 24, 18)
        self.assertNotIn(first["session_id"], server._SESSIONS)

    def test_stale_page_owner_does_not_block_active_page_replacement(self):
        with patch("server.time.monotonic", return_value=100.0):
            shared = create_world_session(9185, 24, 18, request_id="stale-owner-world", page_id="dead-page")
            other_worlds = [create_world_session(seed, 24, 18, page_id=f"other-page-{index}")
                            for index, seed in enumerate(range(9186, 9189), 1)]
            replay = create_world_session(9185, 24, 18, request_id="stale-owner-world", page_id="live-page")

        with patch("server.time.monotonic", return_value=150.0):
            self.assertTrue(server.touch_session(shared["session_id"], "live-page"))
            for index, world in enumerate(other_worlds, 1):
                self.assertTrue(server.touch_session(world["session_id"], f"other-page-{index}"))

        with patch("server.time.monotonic", return_value=191.0):
            replacement = create_world_session(
                9189, 24, 18, replace_session_id=replay["session_id"], page_id="live-page",
            )

        self.assertNotIn(shared["session_id"], server._SESSIONS)
        self.assertIn(replacement["session_id"], server._SESSIONS)
        self.assertEqual(len(server._SESSIONS), server.MAX_SESSIONS)
        self.assertTrue(all(world["session_id"] in server._SESSIONS for world in other_worlds))

    def test_replacing_one_shared_owner_keeps_the_other_page_session_active(self):
        first = create_world_session(9181, 24, 18, request_id="replace-shared-world", page_id="page-one")
        replay = create_world_session(9181, 24, 18, request_id="replace-shared-world", page_id="page-two")
        replacement = create_world_session(9182, 24, 18, replace_session_id=first["session_id"],
                                           page_id="page-one")

        self.assertNotIn("page-one", server._SESSIONS[first["session_id"]]["owners"])
        self.assertEqual(server._SESSIONS[first["session_id"]]["owners"], {"page-two"})
        self.assertIn(first["session_id"], server._SESSIONS)
        self.assertTrue(server.touch_session(first["session_id"], "page-two"))
        self.assertIn("target", server.select_session_scan(first["session_id"]))
        self.assertIn(replacement["session_id"], server._SESSIONS)
        self.assertTrue(server.release_world_session(replay["session_id"], replay["release_token"], "page-two"))
        self.assertNotIn(first["session_id"], server._SESSIONS)

    def test_replacement_from_a_different_page_keeps_the_active_owner_session(self):
        active = create_world_session(919, 24, 18, page_id="active-page")
        replacement = create_world_session(920, 24, 18, replace_session_id=active["session_id"],
                                           page_id="copied-page")

        self.assertIn(active["session_id"], server._SESSIONS)
        self.assertIn(replacement["session_id"], server._SESSIONS)

    def test_page_owner_can_replace_its_own_session(self):
        active = create_world_session(921, 24, 18, page_id="same-page")
        replacement = create_world_session(922, 24, 18, replace_session_id=active["session_id"],
                                           page_id="same-page")

        self.assertNotIn(active["session_id"], server._SESSIONS)
        self.assertIn(replacement["session_id"], server._SESSIONS)

    def test_page_owner_can_replace_its_session_when_capacity_is_full(self):
        worlds = [create_world_session(seed, 24, 18, page_id=f"page-{seed}") for seed in range(960, 964)]

        replacement = create_world_session(964, 24, 18, replace_session_id=worlds[0]["session_id"],
                                            page_id="page-960")

        self.assertNotIn(worlds[0]["session_id"], server._SESSIONS)
        self.assertIn(replacement["session_id"], server._SESSIONS)
        self.assertEqual(len(server._SESSIONS), server.MAX_SESSIONS)

    def test_failed_full_capacity_replacement_keeps_shared_owner_registered(self):
        shared = create_world_session(970, 24, 18, request_id="shared-full-capacity", page_id="page-one")
        create_world_session(970, 24, 18, request_id="shared-full-capacity", page_id="page-two")
        [create_world_session(seed, 24, 18, page_id=f"page-{seed}") for seed in range(971, 974)]

        with self.assertRaisesRegex(RuntimeError, "容量已满"):
            create_world_session(974, 24, 18, replace_session_id=shared["session_id"], page_id="page-one")

        self.assertEqual(server._SESSIONS[shared["session_id"]]["owners"], {"page-one", "page-two"})
        self.assertTrue(server.touch_session(shared["session_id"], "page-one"))
        self.assertIn(shared["session_id"], server._SESSIONS)

    def test_health_touch_during_world_generation_prevents_replacing_a_shared_session(self):
        worlds = [create_world_session(seed, 24, 18, page_id=f"page-{seed}") for seed in range(980, 984)]
        old, *others = worlds
        original_make_world = server.make_world
        generation_started, allow_generation = Event(), Event()
        results, errors = [], []

        def hold_generation(*args):
            generation_started.set()
            if not allow_generation.wait(2):
                raise RuntimeError("replacement test timed out")
            return original_make_world(*args)

        def replace():
            try:
                results.append(create_world_session(984, 24, 18, old["session_id"], page_id="page-980"))
            except Exception as error:
                errors.append(error)

        with patch("server.make_world", side_effect=hold_generation):
            thread = Thread(target=replace)
            thread.start()
            try:
                self.assertTrue(generation_started.wait(2), "replacement should pass preflight and start generation")
                self.assertTrue(server.touch_session(old["session_id"], "sibling-page"))
            finally:
                allow_generation.set()
                thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(server._SESSIONS[old["session_id"]]["owners"], {"page-980", "sibling-page"})
        self.assertTrue(all(world["session_id"] in server._SESSIONS for world in [old, *others]))
        self.assertEqual(len(server._SESSIONS), server.MAX_SESSIONS)

    def test_releasing_old_owner_after_preflight_allows_world_replacement(self):
        worlds = [create_world_session(seed, 24, 18, page_id=f"page-{seed}") for seed in range(985, 989)]
        old, *others = worlds
        original_make_world = server.make_world
        generation_started, allow_generation = Event(), Event()
        results, errors = [], []

        def hold_generation(*args):
            generation_started.set()
            if not allow_generation.wait(2):
                raise RuntimeError("replacement test timed out")
            return original_make_world(*args)

        def replace():
            try:
                results.append(create_world_session(989, 24, 18, old["session_id"], page_id="page-985"))
            except Exception as error:
                errors.append(error)

        with patch("server.make_world", side_effect=hold_generation):
            thread = Thread(target=replace)
            thread.start()
            try:
                self.assertTrue(generation_started.wait(2), "replacement should pass preflight and start generation")
                self.assertTrue(server.release_world_session(old["session_id"], old["release_token"], "page-985"))
            finally:
                allow_generation.set()
                thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertNotIn(old["session_id"], server._SESSIONS)
        self.assertTrue(all(world["session_id"] in server._SESSIONS for world in others))
        self.assertIn(results[0]["session_id"], server._SESSIONS)
        self.assertEqual(len(server._SESSIONS), server.MAX_SESSIONS)

    def test_expired_replacement_during_world_generation_is_pruned_before_insert(self):
        worlds = [create_world_session(seed, 24, 18, page_id=f"page-{seed}") for seed in range(990, 994)]
        old, *others = worlds
        original_make_world = server.make_world
        generation_started, allow_generation = Event(), Event()
        results, errors = [], []

        def hold_generation(*args):
            generation_started.set()
            if not allow_generation.wait(2):
                raise RuntimeError("replacement test timed out")
            return original_make_world(*args)

        def replace():
            try:
                results.append(create_world_session(994, 24, 18, old["session_id"], page_id="page-990"))
            except Exception as error:
                errors.append(error)

        with patch("server.make_world", side_effect=hold_generation):
            thread = Thread(target=replace)
            thread.start()
            try:
                self.assertTrue(generation_started.wait(2), "replacement should pass preflight and start generation")
                with server._SESSIONS_LOCK:
                    server._SESSIONS[old["session_id"]]["lease_until"] = server.time.monotonic() - 1
            finally:
                allow_generation.set()
                thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertNotIn(old["session_id"], server._SESSIONS)
        self.assertTrue(all(world["session_id"] in server._SESSIONS for world in others))
        self.assertIn(results[0]["session_id"], server._SESSIONS)
        self.assertEqual(len(server._SESSIONS), server.MAX_SESSIONS)

    def test_active_sessions_are_not_evicted_when_capacity_is_full(self):
        worlds = [create_world_session(seed, 24, 18) for seed in range(930, 934)]

        with patch("server.make_world") as generate_world:
            with self.assertRaisesRegex(RuntimeError, "容量已满"):
                create_world_session(934, 24, 18)
            generate_world.assert_not_called()

        for world in worlds:
            self.assertNotIn("seed", world)
            self.assertIsNotNone(_get_session(world["session_id"]))

    def test_lookup_removes_an_expired_session(self):
        world = create_world_session(935, 24, 18)
        with server._SESSIONS_LOCK:
            server._SESSIONS[world["session_id"]]["lease_until"] = server.time.monotonic() - 1

        with self.assertRaisesRegex(ValueError, "不存在或已过期"):
            _get_session(world["session_id"])
        self.assertNotIn(world["session_id"], server._SESSIONS)

    def test_expired_sessions_are_removed_to_make_room(self):
        with patch("server.time.monotonic", return_value=100.0):
            expired_worlds = [create_world_session(seed, 24, 18) for seed in range(940, 943)]
        with patch("server.time.monotonic", return_value=120.0):
            active_world = create_world_session(943, 24, 18)
        with patch("server.time.monotonic", return_value=100.0 + server.SESSION_LEASE_SECONDS + 1):
            replacement = create_world_session(944, 24, 18)

        for world in expired_worlds:
            self.assertNotIn(world["session_id"], server._SESSIONS)
        self.assertIn(active_world["session_id"], server._SESSIONS)
        self.assertIn(replacement["session_id"], server._SESSIONS)

    def test_same_tab_can_replace_its_session_when_capacity_is_full(self):
        worlds = [create_world_session(seed, 24, 18) for seed in range(950, 954)]

        replacement = create_world_session(954, 24, 18, worlds[0]["session_id"])

        self.assertNotIn(worlds[0]["session_id"], server._SESSIONS)
        self.assertIn(replacement["session_id"], server._SESSIONS)
        for world in worlds[1:]:
            self.assertIn(world["session_id"], server._SESSIONS)

    def test_failed_world_replacement_keeps_the_old_session(self):
        world = create_world_session(960, 24, 18)

        with patch("server.make_world", side_effect=RuntimeError("generation failed")):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                create_world_session(961, 24, 18, world["session_id"])

        self.assertIn(world["session_id"], server._SESSIONS)

    def test_public_world_does_not_expose_a_seed_that_reconstructs_truth(self):
        world = create_world_session(77, 24, 18)

        self.assertNotIn("layers", world)
        self.assertNotIn("seed", world)
        self.assertTrue(world["map_id"])
        self.assertTrue(world["session_id"])
        self.assertTrue(world["instance_id"])
        self.assertEqual(world["api_protocol"], server.API_PROTOCOL)
        self.assertEqual(set(world["layer_names"]), set(make_world(77, 24, 18)["layers"]))
        truth = _get_session(world["session_id"])["truth"]
        self.assertNotEqual(truth["layers"], make_world(77, 24, 18)["layers"])

        reset = create_world_session(77, 24, 18)
        reset_truth = _get_session(reset["session_id"])["truth"]
        self.assertEqual(
            truth["layers"], reset_truth["layers"],
            "resetting with the same request seed should keep the map",
        )

    def test_sensor_returns_only_footprint_and_fuses_repeat_readings(self):
        world = create_world_session(78, 24, 18)
        center = {"x": 10, "y": 9}

        first = sense_session(world["session_id"], center, radius=0)
        second = sense_session(world["session_id"], center, radius=0)

        self.assertEqual(len(first["cells"]), 1)
        self.assertEqual((first["cells"][0]["x"], first["cells"][0]["y"]), (10, 9))
        self.assertEqual(first["cells"][0]["samples"], 1)
        self.assertEqual(second["cells"][0]["samples"], 2)
        self.assertGreater(second["cells"][0]["confidence"], first["cells"][0]["confidence"])
        stored = _get_session(world["session_id"])
        self.assertNotEqual(first["cells"][0]["layers"]["elevation"],
                            stored["truth"]["layers"]["elevation"][9][10])
        self.assertEqual(first["cells"][0]["layers"]["road"], stored["truth"]["layers"]["road"][9][10])
        self.assertIsNone(stored["layers"]["elevation"][8][10])
        self.assertIsNotNone(stored["cost"][9][10])
        self.assertIsNone(stored["cost"][8][10])

    def test_sensor_rejects_non_integer_centers_without_sampling(self):
        world = create_world_session(780, 24, 18)
        session = _get_session(world["session_id"])

        for center in ({"x": 3.9, "y": 4}, {"x": 3, "y": True}):
            with self.subTest(center=center):
                with self.assertRaisesRegex(ValueError, "扫描中心必须为整数坐标"):
                    sense_session(world["session_id"], center, radius=0)

        self.assertEqual(session["scan_sequence"], 0)
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_sensor_fuses_footprints_along_flown_segment(self):
        world = create_world_session(79, 14, 6)

        result = sense_flight_segment(world["session_id"], {"x": 1, "y": 2},
                                      {"x": 11, "y": 2}, radius=1)

        self.assertEqual(result["sample_centers"], 7)
        cells = {(cell["x"], cell["y"]): cell for cell in result["cells"]}
        self.assertIn((6, 2), cells)
        self.assertIn((11, 2), cells)
        self.assertGreater(len(cells), 18)
        self.assertTrue(all(cell["samples"] >= 1 for cell in cells.values()))

    def test_flight_segment_rejects_non_integer_endpoints_without_sampling(self):
        world = create_world_session(790, 24, 18)
        session = _get_session(world["session_id"])

        invalid_segments = (
            ({"x": 3.9, "y": 4}, {"x": 8, "y": 4}),
            ({"x": 3, "y": 4}, {"x": 8, "y": False}),
        )
        for start, end in invalid_segments:
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, "航段.*必须为整数坐标"):
                    sense_flight_segment(world["session_id"], start, end, radius=0)

        self.assertEqual(session["scan_sequence"], 0)
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_scan_selection_rejects_non_integer_mission_start_before_session_changes(self):
        world = create_world_session(791, 24, 18)
        session = _get_session(world["session_id"])

        with self.assertRaisesRegex(ValueError, "任务起点必须为整数坐标"):
            configure_session_mission(world["session_id"], {"x": 3.9, "y": 4}, world["goal"])

        self.assertIsNone(session["guidance_grid"])
        self.assertEqual(session["scan_sequence"], 0)
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_invalid_flight_segment_does_not_partially_scan_map(self):
        world = create_world_session(902, 12, 8)
        session = _get_session(world["session_id"])

        with self.assertRaisesRegex(ValueError, "超出地图范围"):
            sense_flight_segment(world["session_id"], {"x": 1, "y": 2},
                                 {"x": 99, "y": 2}, radius=1)

        self.assertEqual(session["scan_sequence"], 0)
        self.assertTrue(all(value == 0 for row in session["samples"] for value in row))

    def test_planner_waits_for_scanned_endpoints_and_ignores_unknown_cells(self):
        world = create_world_session(79, 24, 18)
        start, goal = world["start"], world["goal"]

        waiting = plan_session(world["session_id"], start, goal)
        self.assertTrue(waiting["waiting_for_scan"])
        self.assertEqual(waiting["path"], [])

        sense_session(world["session_id"], start, radius=0)
        sense_session(world["session_id"], goal, radius=0)
        disconnected = plan_session(world["session_id"], start, goal)
        self.assertFalse(disconnected["waiting_for_scan"])
        self.assertEqual(disconnected["path"], [])

    def test_stale_plan_waiting_on_session_lock_cannot_overwrite_new_mission_route(self):
        world = create_world_session(795, 24, 18)
        current_start, current_goal = {"x": 4, "y": 4}, {"x": 20, "y": 13}
        session = _get_session(world["session_id"])
        current_route = [current_start, {"x": 12, "y": 8}, current_goal]
        old_plan_started, old_plan_done = Event(), Event()
        old_plan_result = []

        def plan_old_endpoints():
            old_plan_started.set()
            try:
                old_plan_result.append(plan_session(world["session_id"], world["start"], world["goal"]))
            except ValueError as error:
                old_plan_result.append(error)
            finally:
                old_plan_done.set()

        session["lock"].acquire()
        old_plan_thread = Thread(target=plan_old_endpoints)
        try:
            old_plan_thread.start()
            self.assertTrue(old_plan_started.wait(2))
            self.assertFalse(old_plan_done.wait(.05), "the old plan should wait for the mission lock")
            configure_session_mission(world["session_id"], current_start, current_goal)
            with patch("server.plan_v5_path", return_value={"path": current_route, "cost": 1}) as planner:
                current_result = plan_session(world["session_id"], current_start, current_goal)
                self.assertEqual(current_result["path"], current_route)
                planner.assert_called_once()
        finally:
            session["lock"].release()
            old_plan_thread.join(2)

        self.assertFalse(old_plan_thread.is_alive())
        self.assertEqual(len(old_plan_result), 1)
        self.assertIsInstance(old_plan_result[0], ValueError)
        self.assertRegex(str(old_plan_result[0]), "规划请求端点与当前任务不一致")
        self.assertEqual(session["route"], current_route)

    def test_server_cost_matches_full_truth_benchmark(self):
        world = make_world(80, 24, 18)
        sensed = build_cost_grid(world["layers"], world["layer_weights"])
        reference = benchmark_cost(world)
        endpoints = {(world["start"]["x"], world["start"]["y"]),
                     (world["goal"]["x"], world["goal"]["y"])}

        for y in range(world["height"]):
            for x in range(world["width"]):
                if (x, y) not in endpoints:
                    self.assertAlmostEqual(sensed[y][x], reference[y][x], places=5)

    def test_http_world_metadata_hides_the_generator_seed(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        request = Request(f"http://127.0.0.1:{httpd.server_port}/api/world",
                          data=json.dumps({"seed": 81, "width": 24, "height": 18}).encode(),
                          headers={"Content-Type": "application/json", "X-Page-ID": "metadata-page"})
        try:
            with urlopen(request, timeout=5) as response:
                world = json.loads(response.read())
            self.assertNotIn("layers", world)
            self.assertNotIn("seed", world)
            self.assertTrue(world["map_id"])
            self.assertTrue(server.release_world_session(
                world["session_id"], world["release_token"], "metadata-page",
            ))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_http_flow_never_accepts_client_owned_map_for_planning(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        control_token = None

        def post(path, payload):
            nonlocal control_token
            headers = {"Content-Type": "application/json", "X-Page-ID": "api-test-page"}
            if control_token:
                headers["X-Control-Token"] = control_token
            request = Request(f"http://127.0.0.1:{httpd.server_port}{path}",
                              data=json.dumps(payload).encode(),
                              headers=headers)
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read())
                control_token = result.get("control_token", control_token)
                return result

        try:
            world = post("/api/world", {"seed": 81, "width": 24, "height": 18})
            self.assertNotIn("layers", world)
            self.assertNotIn("seed", world)
            self.assertTrue(world["map_id"])
            start, goal = world["start"], world["goal"]

            with self.assertRaises(HTTPError) as rejected_center:
                post("/api/scan/measure", {"session_id": world["session_id"],
                                           "center": {"x": start["x"] + .5, "y": start["y"]},
                                           "radius": 0})
            self.assertEqual(rejected_center.exception.code, 410)
            with self.assertRaises(HTTPError) as rejected_segment:
                post("/api/scan/path", {"session_id": world["session_id"],
                                        "start": {"x": start["x"], "y": start["y"]},
                                        "end": {"x": start["x"] + .5, "y": start["y"]},
                                        "radius": 2, "leg_id": "forged", "request_id": "bad"})
            self.assertEqual(rejected_segment.exception.code, 400)
            with self.assertRaises(HTTPError) as rejected:
                post("/api/plan", {"observed": [[True]], "cost": [[0.01]],
                                   "start": start, "goal": goal})
            self.assertEqual(rejected.exception.code, 400)
            self.assertTrue(post("/api/plan", {"session_id": world["session_id"],
                                                "start": start, "goal": goal})["waiting_for_scan"])
            choice = post("/api/scan/next", {"session_id": world["session_id"],
                                              "candidates": [goal], "drone": goal,
                                              "start": start, "goal": goal, "radius": 2})
            self.assertIn(choice["target"], server._SESSIONS[world["session_id"]]["scan_candidates"])
            self.assertEqual(choice["drone_position"], start)
            path_scan = post("/api/scan/path", {"session_id": world["session_id"],
                                                 "start": start, "end": choice["target"],
                                                 "radius": 2, "leg_id": choice["leg_id"],
                                                 "request_id": "authorized-http-scan"})
            self.assertTrue(path_scan["cells"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_http_replayed_world_returns_only_observed_session_state(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        control_token = None

        def post(path, payload):
            nonlocal control_token
            request = Request(f"http://127.0.0.1:{httpd.server_port}{path}",
                              data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json", "X-Page-ID": "snapshot-page-one",
                                       **({"X-Control-Token": control_token} if control_token else {})})
            with urlopen(request, timeout=5) as response:
                result = json.loads(response.read())
                control_token = result.get("control_token", control_token)
                return result

        try:
            create = {"seed": 918, "width": 24, "height": 18,
                      "request_id": "snapshot-world-request", "page_id": "snapshot-page-one"}
            world = post("/api/world", create)
            start, goal = {"x": 3, "y": 14}, {"x": 20, "y": 3}
            configured = post("/api/mission/points", {"session_id": world["session_id"],
                                                       "start": start, "goal": goal})
            self.assertEqual(configured["drone_position"], start)
            choice = post("/api/scan/next", {"session_id": world["session_id"], "radius": 2,
                                               "guidance_weight": .75})
            scan = post("/api/scan/path", {"session_id": world["session_id"],
                                            "start": start, "end": choice["target"],
                                            "radius": 2, "leg_id": choice["leg_id"],
                                            "request_id": "snapshot-scan-request"})
            self.assertTrue(scan["cells"])
            replay = post("/api/world", {**create, "page_id": "snapshot-page-two"})
            snapshot = post("/api/session/scan-state", {"session_id": replay["session_id"]})

            self.assertTrue(replay["request_replayed"])
            self.assertEqual(replay["session_id"], world["session_id"])
            self.assertEqual(snapshot["mission_start"], start)
            self.assertEqual(snapshot["mission_goal"], goal)
            self.assertEqual(snapshot["drone_position"], choice["target"])
            self.assertEqual(snapshot["previous_drone_position"], start)
            session = server._SESSIONS[world["session_id"]]
            expected = {(x, y) for y, row in enumerate(session["samples"])
                        for x, count in enumerate(row) if count > 0}
            actual = {(cell["x"], cell["y"]) for cell in snapshot["cells"]}
            self.assertEqual(actual, expected)
            self.assertTrue(all(cell["samples"] > 0 and set(cell["layers"]) == set(server.LAYERS)
                                for cell in snapshot["cells"]))
            self.assertNotIn("truth", snapshot)
            self.assertNotIn("seed", snapshot)
            self.assertNotIn("layers", snapshot)
            self.assertEqual(session["owners"], {"snapshot-page-one", "snapshot-page-two"})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_http_session_release_is_validated_and_idempotent(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        def post(path, payload):
            request = Request(f"http://127.0.0.1:{httpd.server_port}{path}",
                              data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read())

        try:
            create_payload = {"seed": 82, "width": 24, "height": 18,
                              "request_id": "http-replayed-world", "page_id": "http-page-one"}
            world = post("/api/world", create_payload)
            replay = post("/api/world", {**create_payload, "page_id": "http-page-two"})
            self.assertEqual(replay["session_id"], world["session_id"])
            self.assertNotEqual(replay["release_token"], world["release_token"])
            controller = post("/api/session/control", {
                "session_id": world["session_id"], "action": "acquire", "page_id": "http-page-one",
                "release_token": world["release_token"],
            })
            forged_release = post("/api/session/release", {
                "session_id": world["session_id"], "release_token": replay["release_token"],
                "page_id": "http-page-one",
            })
            self.assertEqual(forged_release, {"released": False})
            session = server._SESSIONS[world["session_id"]]
            self.assertEqual(session["control_page_id"], "http-page-one")
            self.assertEqual(session["owners"], {"http-page-one", "http-page-two"})
            missing_reattach = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/session/control",
                data=json.dumps({"session_id": world["session_id"], "action": "acquire",
                                 "page_id": "http-page-one"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as missing_reattach_blocked:
                urlopen(missing_reattach, timeout=5)
            self.assertEqual(missing_reattach_blocked.exception.code, 409)
            reattached = post("/api/session/control", {
                "session_id": world["session_id"], "action": "acquire", "page_id": "http-page-one",
                "release_token": world["release_token"],
            })
            self.assertEqual(reattached["control_token"], controller["control_token"])
            self.assertFalse(reattached["control_claimed"])
            forged_reattach = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/session/control",
                data=json.dumps({"session_id": world["session_id"], "action": "acquire",
                                 "page_id": "http-page-one", "release_token": replay["release_token"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as reattach_blocked:
                urlopen(forged_reattach, timeout=5)
            self.assertEqual(reattach_blocked.exception.code, 409)
            takeover = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/session/control",
                data=json.dumps({"session_id": world["session_id"], "action": "acquire",
                                 "page_id": "http-page-two"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as blocked:
                urlopen(takeover, timeout=5)
            self.assertEqual(blocked.exception.code, 409)
            release_payload = {"session_id": world["session_id"], "release_token": world["release_token"],
                               "page_id": "http-page-one"}
            self.assertEqual(post("/api/session/release", release_payload), {"released": True})
            self.assertIn(world["session_id"], server._SESSIONS,
                          "releasing one page must retain the replayed session for its sibling")
            self.assertEqual(post("/api/session/release", release_payload), {"released": False})
            released_owner_reacquire = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/session/control",
                data=json.dumps({"session_id": world["session_id"], "action": "acquire",
                                 "page_id": "http-page-one", "release_token": world["release_token"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as released_owner_blocked:
                urlopen(released_owner_reacquire, timeout=5)
            self.assertEqual(released_owner_blocked.exception.code, 409)
            peer_lease = post("/api/session/control", {
                "session_id": world["session_id"], "action": "acquire", "page_id": "http-page-two",
                "release_token": replay["release_token"],
            })
            self.assertEqual(post("/api/session/release", {
                **release_payload, "release_token": replay["release_token"], "page_id": "http-page-two",
                "control_token": peer_lease["control_token"],
            }),
                             {"released": True})
            self.assertNotIn(world["session_id"], server._SESSIONS)
            second_world = post("/api/world", {"seed": 83, "width": 24, "height": 18, "page_id": "http-page-three"})
            cross_origin = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/session/release",
                data=json.dumps({"session_id": second_world["session_id"],
                                 "release_token": second_world["release_token"],
                                 "page_id": "http-page-three"}).encode(),
                headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
            )
            with self.assertRaises(HTTPError) as rejected_origin:
                urlopen(cross_origin, timeout=5)
            self.assertEqual(rejected_origin.exception.code, 403)
            self.assertIn(second_world["session_id"], server._SESSIONS)
            wrong_token = {"session_id": second_world["session_id"], "release_token": "copied-tab-token",
                           "page_id": "http-page-three"}
            self.assertEqual(post("/api/session/release", wrong_token), {"released": False})
            self.assertIn(second_world["session_id"], server._SESSIONS)
            with self.assertRaises(HTTPError) as invalid:
                post("/api/session/release", {"session_id": second_world["session_id"], "release_token": 123,
                                               "page_id": "http-page-three"})
            self.assertEqual(invalid.exception.code, 400)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_session_control_is_exclusive_and_stale_tokens_cannot_release_a_new_lease(self):
        world = create_world_session(941, 24, 18, request_id="lease-owner-sharing", page_id="controller-a")
        world_b = create_world_session(941, 24, 18, request_id="lease-owner-sharing", page_id="controller-b")
        session_id = world["session_id"]
        first = server.acquire_session_control(session_id, "controller-a", owner_token=world["release_token"])
        token_a = first["control_token"]
        self.assertTrue(first["control_claimed"])
        self.assertEqual(first["control_epoch"], 1)
        same_page_renewal = server.acquire_session_control(
            session_id, "controller-a", owner_token=world["release_token"],
        )
        self.assertFalse(same_page_renewal["control_claimed"])
        self.assertEqual(same_page_renewal["control_token"], token_a)

        session = server._get_session(session_id, renew=False)
        original_lease = session["lease_until"]
        with self.assertRaises(server.SessionControlConflict):
            server.configure_session_mission(
                session_id, {"x": 2, "y": 2}, {"x": 20, "y": 14}, page_id="controller-c",
            )
        session = server._get_session(session_id, renew=False)
        self.assertNotIn("controller-c", session["owners"], "rejected writers must not renew themselves as page owners")
        self.assertEqual(session["lease_until"], original_lease, "a rejected writer must not prolong the map session")

        configured = server.configure_session_mission(
            session_id, {"x": 2, "y": 2}, {"x": 20, "y": 14},
            page_id="controller-a", control_token=token_a,
        )
        self.assertEqual(configured["control_token"], token_a)
        server.sense_session(session_id, {"x": 2, "y": 2}, radius=0)
        choice = server.select_session_scan(
            session_id, radius=2, page_id="controller-a", control_token=token_a,
        )
        self.assertIsNotNone(session["active_scan_leg"])

        session["control_lease_until"] = time.monotonic() - 1
        takeover = server.acquire_session_control(
            session_id, "controller-b", owner_token=world_b["release_token"],
        )
        self.assertTrue(takeover["control_taken_over"])
        self.assertEqual(takeover["control_page_id"], "controller-b")
        self.assertEqual(takeover["drone_position"], session["drone_position"])
        self.assertTrue(takeover["cells"], "takeover must return committed sensor observations")
        self.assertIsNone(session["active_scan_leg"], "takeover must revoke the previous active scan leg")
        with self.assertRaises(server.SessionControlConflict):
            server.select_session_scan(
                session_id, page_id="controller-a", control_token=token_a,
            )

        token_b = takeover["control_token"]
        self.assertTrue(server.release_session_control(session_id, "controller-b", token_b))
        second_lease = server.acquire_session_control(
            session_id, "controller-a", owner_token=world["release_token"],
        )
        self.assertNotEqual(second_lease["control_token"], token_a)
        self.assertGreater(second_lease["control_epoch"], takeover["control_epoch"])
        self.assertFalse(server.release_session_control(session_id, "controller-a", token_a))
        status = server.session_control_status(session_id, "controller-a", second_lease["control_token"])
        self.assertTrue(status["scan_controlled_by_you"])

    def test_authorized_scan_helper_requires_page_identity_when_a_control_lease_exists(self):
        page_id = "scan-boundary-page"
        world = create_world_session(9411, 24, 18, page_id=page_id)
        session_id = world["session_id"]
        lease = server.acquire_session_control(session_id, page_id, owner_token=world["release_token"])
        choice = server.select_session_scan(session_id, page_id=page_id, control_token=lease["control_token"])
        session = _get_session(session_id)
        near_expiry = time.monotonic() + 10
        session["lease_until"] = near_expiry
        session["owner_lease_until"][page_id] = near_expiry
        session["control_lease_until"] = near_expiry
        before = (
            session["scan_sequence"], session["snapshot_revision"], dict(session["drone_position"]),
            session["lease_until"], session["owner_lease_until"][page_id], session["control_lease_until"],
        )

        with self.assertRaisesRegex(ValueError, "页面实例编号必填"):
            server.sense_flight_segment(
                session_id, choice["drone_position"], choice["target"], radius=2,
                request_id="missing-page-id", leg_id=choice["leg_id"], require_authorization=True,
            )

        self.assertEqual(
            (session["scan_sequence"], session["snapshot_revision"], session["drone_position"],
             session["lease_until"], session["owner_lease_until"][page_id], session["control_lease_until"]), before,
        )
        result = server.sense_flight_segment(
            session_id, choice["drone_position"], choice["target"], radius=2,
            request_id="valid-page-id", leg_id=choice["leg_id"], require_authorization=True,
            page_id=page_id, control_token=lease["control_token"],
        )
        self.assertTrue(result["cells"])
        self.assertGreater(session["lease_until"], before[3])
        self.assertGreater(session["owner_lease_until"][page_id], before[4])
        self.assertGreater(session["control_lease_until"], before[5])

    def test_releasing_or_replacing_a_page_revokes_its_old_control_lease(self):
        shared = create_world_session(945, 24, 18, request_id="replace-controlled-world", page_id="replace-page-a")
        shared_b = create_world_session(945, 24, 18, request_id="replace-controlled-world", page_id="replace-page-b")
        old_session_id = shared["session_id"]
        first = server.acquire_session_control(old_session_id, "replace-page-a", owner_token=shared["release_token"])
        old_token = first["control_token"]

        replacement = create_world_session(
            946, 24, 18, replace_session_id=old_session_id, page_id="replace-page-a",
        )

        old_session = server._SESSIONS[old_session_id]
        self.assertEqual(old_session["owners"], {"replace-page-b"})
        self.assertIsNone(old_session["control_page_id"])
        with self.assertRaises(server.SessionControlConflict):
            server.configure_session_mission(
                old_session_id, shared["start"], shared["goal"],
                page_id="replace-page-a", control_token=old_token,
            )
        takeover = server.acquire_session_control(
            old_session_id, "replace-page-b", owner_token=shared_b["release_token"],
        )
        self.assertEqual(takeover["control_page_id"], "replace-page-b")
        self.assertNotEqual(replacement["session_id"], old_session_id)

        self.assertTrue(server.release_world_session(
            old_session_id, shared_b["release_token"], "replace-page-b",
        ))
        self.assertIsNone(old_session["control_page_id"])
        self.assertNotIn("replace-page-b", old_session["owners"])

    def test_pagehide_owner_credential_releases_a_lease_before_acquire_response_arrives(self):
        first = create_world_session(9462, 24, 18, request_id="pagehide-race", page_id="pagehide-a")
        peer = create_world_session(9462, 24, 18, request_id="pagehide-race", page_id="pagehide-b")
        session_id = first["session_id"]
        server.acquire_session_control(session_id, "pagehide-a", owner_token=first["release_token"])

        self.assertTrue(server.release_world_session(session_id, first["release_token"], "pagehide-a"))
        session = server._SESSIONS[session_id]
        self.assertIsNone(session["control_page_id"])
        self.assertNotIn("pagehide-a", session["owners"])
        next_lease = server.acquire_session_control(
            session_id, "pagehide-b", owner_token=peer["release_token"],
        )
        self.assertEqual(next_lease["control_page_id"], "pagehide-b")

    def test_replayed_owner_credentials_are_bounded_per_session(self):
        world = create_world_session(9463, 24, 18, request_id="bounded-owners", page_id="owner-0")
        session = server._SESSIONS[world["session_id"]]
        for index in range(1, server.MAX_SESSION_OWNERS):
            server.create_world_session(
                9463, 24, 18, request_id="bounded-owners", page_id=f"owner-{index}",
            )

        self.assertEqual(len(session["owners"]), server.MAX_SESSION_OWNERS)
        self.assertEqual(len(session["owner_release_tokens"]), server.MAX_SESSION_OWNERS)
        with self.assertRaisesRegex(RuntimeError, "页面数量已达上限"):
            server.create_world_session(
                9463, 24, 18, request_id="bounded-owners", page_id="owner-overflow",
            )

        with session["lock"]:
            with server._SESSIONS_LOCK:
                session["owner_lease_until"]["owner-0"] = time.monotonic() - 1
                server._prune_session_owners(session, time.monotonic())
        server.create_world_session(
            9463, 24, 18, request_id="bounded-owners", page_id="owner-overflow",
        )
        self.assertEqual(len(session["owner_release_tokens"]), server.MAX_SESSION_OWNERS)
        self.assertNotIn("owner-0", session["owner_release_tokens"])

    def test_shared_snapshot_revision_tracks_scans_mission_and_route(self):
        world = create_world_session(94635, 24, 18, page_id="snapshot-version-page")
        session_id = world["session_id"]
        start, goal = {"x": 3, "y": 14}, {"x": 20, "y": 3}
        initial = server.session_scan_snapshot(session_id)["snapshot_revision"]

        sense_session(session_id, {"x": 5, "y": 5}, radius=0)
        scan_revision = server.session_scan_snapshot(session_id)["snapshot_revision"]
        configured = configure_session_mission(session_id, start, goal)
        mission_revision = server.session_scan_snapshot(session_id)["snapshot_revision"]
        with patch("server.plan_v5_path", return_value={"path": [start, goal], "cost": 1.0, "planner": "test"}):
            plan_session(session_id, start, goal)
        route_revision = server.session_control_status(session_id, "snapshot-reader", None)["snapshot_revision"]

        self.assertEqual(initial, 0)
        self.assertGreater(scan_revision, initial)
        self.assertGreater(mission_revision, scan_revision)
        self.assertGreater(route_revision, mission_revision)
        self.assertEqual(configured["start"], start)
        self.assertEqual(route_revision, server.session_scan_snapshot(session_id)["snapshot_revision"])

    def test_authorized_scan_distance_is_persisted_and_replay_does_not_double_count(self):
        world = create_world_session(9464, 24, 18, page_id="distance-page")
        session_id = world["session_id"]
        lease = server.acquire_session_control(
            session_id, "distance-page", owner_token=world["release_token"],
        )
        token = lease["control_token"]
        choice = server.select_session_scan(session_id, page_id="distance-page", control_token=token)
        start, target = choice["drone_position"], choice["target"]
        result = server.sense_flight_segment(
            session_id, start, target, request_id="distance-scan", leg_id=choice["leg_id"],
            require_authorization=True, page_id="distance-page", control_token=token,
        )
        expected = math.hypot(target["x"] - start["x"], target["y"] - start["y"])

        snapshot = server.session_scan_snapshot(session_id)
        replay = server.sense_flight_segment(
            session_id, start, target, request_id="distance-scan", leg_id=choice["leg_id"],
            require_authorization=True, page_id="distance-page", control_token=token,
        )

        self.assertAlmostEqual(snapshot["flight_distance"], expected, places=6)
        self.assertEqual(result["flight_distance"], replay["flight_distance"])
        self.assertAlmostEqual(server.session_scan_snapshot(session_id)["flight_distance"], expected, places=6)

    def test_replaying_an_older_authorized_scan_reports_later_committed_distance(self):
        world = create_world_session(9465, 24, 18, page_id="distance-replay-page")
        session_id = world["session_id"]
        lease = server.acquire_session_control(
            session_id, "distance-replay-page", owner_token=world["release_token"],
        )
        token = lease["control_token"]
        choice = server.select_session_scan(session_id, page_id="distance-replay-page", control_token=token)
        start, target = choice["drone_position"], choice["target"]
        if start == target:
            server.sense_flight_segment(
                session_id, start, target, request_id="distance-replay-seed", leg_id=choice["leg_id"],
                require_authorization=True, page_id="distance-replay-page", control_token=token,
            )
            choice = server.select_session_scan(session_id, page_id="distance-replay-page", control_token=token)
            start, target = choice["drone_position"], choice["target"]
        first_stop = {"x": round(start["x"] + (target["x"] - start["x"]) * 0.35),
                      "y": round(start["y"] + (target["y"] - start["y"]) * 0.35)}
        second_stop = {"x": round(start["x"] + (target["x"] - start["x"]) * 0.7),
                       "y": round(start["y"] + (target["y"] - start["y"]) * 0.7)}
        first = server.sense_flight_segment(
            session_id, start, first_stop, request_id="distance-replay-first", leg_id=choice["leg_id"],
            require_authorization=True, page_id="distance-replay-page", control_token=token,
        )
        second = server.sense_flight_segment(
            session_id, first_stop, second_stop, request_id="distance-replay-second", leg_id=choice["leg_id"],
            require_authorization=True, page_id="distance-replay-page", control_token=token,
        )
        replay = server.sense_flight_segment(
            session_id, start, first_stop, request_id="distance-replay-first", leg_id=choice["leg_id"],
            require_authorization=True, page_id="distance-replay-page", control_token=token,
        )

        self.assertLess(first["flight_distance"], second["flight_distance"])
        self.assertEqual(replay["flight_distance"], second["flight_distance"])
        self.assertEqual(replay["drone_position"], second_stop)
        self.assertEqual(replay["scan_sequence"], server.session_scan_snapshot(session_id)["scan_sequence"])

    def test_scan_request_receipts_are_bounded_without_partial_mutation(self):
        world = create_world_session(9466, 24, 18, page_id="bounded-scan-receipts")
        session_id = world["session_id"]
        lease = server.acquire_session_control(
            session_id, "bounded-scan-receipts", owner_token=world["release_token"],
        )
        token = lease["control_token"]
        choice = server.select_session_scan(session_id, page_id="bounded-scan-receipts", control_token=token)
        start = choice["drone_position"]
        if choice["target"] == start:
            server.sense_flight_segment(
                session_id, start, start, request_id="bounded-scan-seed", leg_id=choice["leg_id"],
                require_authorization=True, page_id="bounded-scan-receipts", control_token=token,
            )
            choice = server.select_session_scan(session_id, page_id="bounded-scan-receipts", control_token=token)
            start = choice["drone_position"]

        session = _get_session(session_id)
        with patch.object(server, "MAX_SCAN_REQUEST_RECEIPTS", len(session["scan_request_receipts"]) + 1):
            self.assertFalse(server.session_scan_snapshot(session_id)["scan_request_capacity_reached"])
            server.sense_flight_segment(
                session_id, start, start, request_id="bounded-scan-first", leg_id=choice["leg_id"],
                require_authorization=True, page_id="bounded-scan-receipts", control_token=token,
            )
            before_rejected_request = server.session_scan_snapshot(session_id)
            self.assertTrue(before_rejected_request["scan_request_capacity_reached"])
            receipt_count = len(session["scan_request_receipts"])
            with self.assertRaisesRegex(ValueError, "扫描请求编号已达上限"):
                server.sense_flight_segment(
                    session_id, start, start, request_id="bounded-scan-overflow", leg_id=choice["leg_id"],
                    require_authorization=True, page_id="bounded-scan-receipts", control_token=token,
                )
            after_rejected_request = server.session_scan_snapshot(session_id)

        self.assertEqual(after_rejected_request, before_rejected_request)
        self.assertEqual(len(session["scan_request_receipts"]), receipt_count)

    def test_pruned_page_can_rebind_with_its_preserved_owner_credential(self):
        world = create_world_session(9461, 24, 18, page_id="returning-page")
        session = server._SESSIONS[world["session_id"]]
        owner_token = world["release_token"]
        with session["lock"]:
            with server._SESSIONS_LOCK:
                session["owner_lease_until"]["returning-page"] = time.monotonic() - 1
                server._prune_session_owners(session, time.monotonic())

        self.assertNotIn("returning-page", session["owners"])
        self.assertEqual(session["owner_release_tokens"]["returning-page"], owner_token)
        lease = server.acquire_session_control(
            world["session_id"], "returning-page", owner_token=owner_token,
        )
        self.assertIn("returning-page", session["owners"])
        self.assertEqual(session["owner_release_tokens"]["returning-page"], owner_token)
        self.assertTrue(server.release_world_session(
            world["session_id"], owner_token, "returning-page", lease["control_token"],
        ))

    def test_invalid_authenticated_writes_do_not_claim_or_extend_control(self):
        world = create_world_session(947, 24, 18, page_id="invalid-write-page")
        session = server._SESSIONS[world["session_id"]]
        original_lease = session["lease_until"]

        with self.assertRaisesRegex(ValueError, "超出地图范围"):
            server.configure_session_mission(
                world["session_id"], {"x": 999, "y": 3}, world["goal"],
                page_id="invalid-write-page",
            )

        self.assertIsNone(session["control_page_id"])
        self.assertEqual(session["lease_until"], original_lease)
        self.assertEqual(session["owner_lease_until"]["invalid-write-page"], original_lease)

    def test_scan_replay_from_released_page_cannot_restore_owner_or_control(self):
        shared = create_world_session(948, 24, 18, request_id="replay-shared-world", page_id="replay-page-a")
        create_world_session(948, 24, 18, request_id="replay-shared-world", page_id="replay-page-b")
        session_id = shared["session_id"]
        token = server.acquire_session_control(
            session_id, "replay-page-a", owner_token=shared["release_token"],
        )["control_token"]
        start, goal = {"x": 2, "y": 2}, {"x": 20, "y": 14}
        server.configure_session_mission(session_id, start, goal, page_id="replay-page-a", control_token=token)
        choice = server.select_session_scan(session_id, page_id="replay-page-a", control_token=token)
        payload = (start, choice["target"])
        original = server.sense_flight_segment(
            session_id, *payload, request_id="released-page-replay", leg_id=choice["leg_id"],
            require_authorization=True, page_id="replay-page-a", control_token=token,
        )
        session = server._SESSIONS[session_id]
        lease_before_release = session["lease_until"]
        self.assertTrue(server.release_world_session(
            session_id, shared["release_token"], "replay-page-a", token,
        ))
        lease_after_release = session["lease_until"]
        scan_sequence = session["scan_sequence"]
        samples = [row[:] for row in session["samples"]]

        with self.assertRaises(server.SessionControlConflict):
            server.sense_flight_segment(
                session_id, *payload, request_id="released-page-replay", leg_id=choice["leg_id"],
                require_authorization=True, page_id="replay-page-a", control_token=token,
            )

        self.assertTrue(original["cells"])
        self.assertEqual(lease_after_release, lease_before_release)
        self.assertEqual(session["lease_until"], lease_after_release)
        self.assertEqual(session["samples"], samples)
        self.assertEqual(session["owners"], {"replay-page-b"})
        self.assertIsNone(session["control_page_id"])
        self.assertEqual(session["scan_sequence"], scan_sequence)

    def test_http_control_conflicts_are_409_and_health_does_not_transfer_ownership(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        def post(path, payload, page_id, token=None):
            headers = {"Content-Type": "application/json", "X-Page-ID": page_id}
            if token:
                headers["X-Control-Token"] = token
            request = Request(f"http://127.0.0.1:{httpd.server_port}{path}",
                              data=json.dumps(payload).encode(), headers=headers)
            with urlopen(request, timeout=5) as response:
                return json.loads(response.read())

        try:
            world = post("/api/world", {"seed": 944, "width": 24, "height": 18,
                                        "page_id": "http-controller-a"}, "http-controller-a")
            session_id = world["session_id"]
            lease = post("/api/session/control", {
                "session_id": session_id, "action": "acquire", "release_token": world["release_token"],
            }, "http-controller-a")
            token = lease["control_token"]
            old_start = lease["mission_start"]
            denied = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/mission/points",
                data=json.dumps({"session_id": session_id, "start": {"x": 3, "y": 3},
                                 "goal": {"x": 19, "y": 14}}).encode(),
                headers={"Content-Type": "application/json", "X-Page-ID": "http-controller-b"},
            )
            with self.assertRaises(HTTPError) as rejected:
                urlopen(denied, timeout=5)
            self.assertEqual(rejected.exception.code, 409)
            error = json.loads(rejected.exception.read())
            self.assertEqual(error["code"], "control_conflict")
            self.assertEqual(error["control_page_id"], "http-controller-a")
            session = server._get_session(session_id, renew=False)
            self.assertEqual(session["mission_start"], old_start, "a rejected page must not change mission endpoints")

            health = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/health",
                headers={"X-Session-ID": session_id, "X-Page-ID": "http-controller-b"},
            )
            with urlopen(health, timeout=5) as response:
                status = json.loads(response.read())
            self.assertEqual(status["scan_control_page_id"], "http-controller-a")
            self.assertFalse(status["scan_controlled_by_you"], "a heartbeat cannot acquire or renew another page's control")
            self.assertEqual(status["snapshot_revision"], 0)
            self.assertNotIn("http-controller-b", session["owners"], "a heartbeat cannot restore a released page owner")

            configured = post("/api/mission/points", {"session_id": session_id,
                                                        "start": {"x": 4, "y": 4},
                                                        "goal": {"x": 18, "y": 13}},
                              "http-controller-a", token)
            self.assertEqual(configured["start"], {"x": 4, "y": 4})
            self.assertEqual(configured["control_token"], token)
            with urlopen(health, timeout=5) as response:
                updated_status = json.loads(response.read())
            self.assertGreater(updated_status["snapshot_revision"], status["snapshot_revision"])
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_http_health_treats_non_ascii_control_token_as_unauthorized(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            world = create_world_session(9441, 24, 18, page_id="health-token-page")
            lease = server.acquire_session_control(
                world["session_id"], "health-token-page", owner_token=world["release_token"],
            )
            request = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/health",
                headers={
                    "X-Session-ID": world["session_id"],
                    "X-Page-ID": "health-token-page",
                    "X-Control-Token": "é",
                },
            )

            with urlopen(request, timeout=5) as response:
                health = json.loads(response.read())

            self.assertTrue(health["session_valid"])
            self.assertFalse(health["scan_controlled_by_you"])
            self.assertEqual(health["scan_control_page_id"], "health-token-page")
            self.assertNotEqual(lease["control_token"], "é")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_http_health_does_not_keep_session_valid_after_concurrent_release(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            page_id = "health-release-page"
            world = create_world_session(9442, 24, 18, page_id=page_id)
            request = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/health",
                headers={"X-Session-ID": world["session_id"], "X-Page-ID": page_id},
            )
            touch_session = server.touch_session

            def touch_then_release(*args, **kwargs):
                valid = touch_session(*args, **kwargs)
                server.release_world_session(world["session_id"], world["release_token"], page_id)
                return valid

            with patch.object(server, "touch_session", side_effect=touch_then_release):
                with urlopen(request, timeout=5) as response:
                    health = json.loads(response.read())

            self.assertFalse(health["session_valid"])
            self.assertNotIn("snapshot_revision", health)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_http_scan_request_capacity_returns_actionable_conflict(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            world = create_world_session(9467, 24, 18, page_id="capacity-http-page")
            lease = server.acquire_session_control(
                world["session_id"], "capacity-http-page", owner_token=world["release_token"],
            )
            choice = server.select_session_scan(
                world["session_id"], page_id="capacity-http-page", control_token=lease["control_token"],
            )
            position = choice["drone_position"]
            payload = {
                "session_id": world["session_id"], "start": position, "end": position,
                "radius": 2, "leg_id": choice["leg_id"], "request_id": "capacity-http-request",
                "page_id": "capacity-http-page",
            }
            request = Request(
                f"http://127.0.0.1:{httpd.server_port}/api/scan/path",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json", "X-Control-Token": lease["control_token"]},
            )
            before = server.session_scan_snapshot(world["session_id"])
            with patch.object(server, "MAX_SCAN_REQUEST_RECEIPTS", 0):
                with self.assertRaises(HTTPError) as rejected:
                    urlopen(request, timeout=5)
            self.assertEqual(rejected.exception.code, 409)
            self.assertEqual(json.loads(rejected.exception.read())["code"], "scan_request_capacity")
            self.assertEqual(server.session_scan_snapshot(world["session_id"]), before)
            self.assertEqual(server._get_session(world["session_id"])["scan_request_receipts"], {})
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=1)

    def test_evicted_committed_scan_request_reconciles_from_snapshot_without_resampling(self):
        world = create_world_session(942, 24, 18, page_id="receipt-page")
        session_id = world["session_id"]
        lease = server.acquire_session_control(session_id, "receipt-page", owner_token=world["release_token"])
        token = lease["control_token"]
        start, goal = {"x": 2, "y": 2}, {"x": 20, "y": 14}
        server.configure_session_mission(
            session_id, start, goal, page_id="receipt-page", control_token=token,
        )
        choice = server.select_session_scan(
            session_id, radius=2, page_id="receipt-page", control_token=token,
        )
        args = (session_id, start, choice["target"])
        request = server.sense_flight_segment(
            *args, radius=2, request_id="committed-then-evicted", leg_id=choice["leg_id"],
            require_authorization=True, page_id="receipt-page", control_token=token,
        )
        session = server._get_session(session_id, renew=False)
        sequence = session["scan_sequence"]
        samples = sum(sum(row) for row in session["samples"])
        session["scan_requests"].clear()

        replay = server.sense_flight_segment(
            *args, radius=2, request_id="committed-then-evicted", leg_id=choice["leg_id"],
            require_authorization=True, page_id="receipt-page", control_token=token,
        )

        self.assertTrue(request["cells"])
        self.assertEqual(replay["cells"], [])
        self.assertTrue(replay["reconcile_snapshot"])
        self.assertEqual(replay["drone_position"], session["drone_position"])
        self.assertEqual(session["scan_sequence"], sequence)
        self.assertEqual(sum(sum(row) for row in session["samples"]), samples)

    def test_takeover_waits_for_inflight_scan_and_returns_its_committed_position(self):
        world = create_world_session(943, 24, 18, request_id="race-shared-world", page_id="race-page-a")
        world_b = create_world_session(943, 24, 18, request_id="race-shared-world", page_id="race-page-b")
        session_id = world["session_id"]
        token_a = server.acquire_session_control(
            session_id, "race-page-a", owner_token=world["release_token"],
        )["control_token"]
        start, goal = {"x": 2, "y": 2}, {"x": 20, "y": 14}
        server.configure_session_mission(
            session_id, start, goal, page_id="race-page-a", control_token=token_a,
        )
        choice = server.select_session_scan(
            session_id, radius=2, page_id="race-page-a", control_token=token_a,
        )
        session = server._get_session(session_id, renew=False)
        entered_scan, resume_scan, takeover_done = Event(), Event(), Event()
        original_sense = server.sense_session
        scan_results, takeover_results = [], []

        def slow_sense(*args, **kwargs):
            if not entered_scan.is_set():
                entered_scan.set()
                self.assertTrue(resume_scan.wait(2))
            return original_sense(*args, **kwargs)

        def scan():
            scan_results.append(server.sense_flight_segment(
                session_id, start, choice["target"], radius=2, request_id="race-scan",
                leg_id=choice["leg_id"], require_authorization=True,
                page_id="race-page-a", control_token=token_a,
            ))

        def takeover():
            takeover_results.append(server.acquire_session_control(
                session_id, "race-page-b", owner_token=world_b["release_token"],
            ))
            takeover_done.set()

        with patch("server.sense_session", side_effect=slow_sense):
            scanner = Thread(target=scan)
            scanner.start()
            self.assertTrue(entered_scan.wait(2))
            session["control_lease_until"] = time.monotonic() - 1
            taker = Thread(target=takeover)
            taker.start()
            self.assertFalse(takeover_done.wait(.05), "takeover must wait for the in-flight scan's session lock")
            resume_scan.set()
            scanner.join(2)
            taker.join(2)

        self.assertFalse(scanner.is_alive())
        self.assertFalse(taker.is_alive())
        self.assertEqual(len(scan_results), 1)
        self.assertTrue(takeover_done.is_set())
        self.assertEqual(takeover_results[0]["drone_position"], choice["target"])
        self.assertEqual(takeover_results[0]["scan_sequence"], session["scan_sequence"])
        self.assertTrue(takeover_results[0]["control_taken_over"])
        self.assertIsNone(session["active_scan_leg"])


if __name__ == "__main__":
    unittest.main()
