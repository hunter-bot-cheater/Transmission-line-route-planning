import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from transmission_path_planner.cost import build_route_cost
from transmission_path_planner.cli import (
    _select_final_route,
    attach_station_endpoints,
    screening_data_quality,
    validate_route_hard_constraints,
)
from transmission_path_planner.geometry import rowcol_to_geo
from transmission_path_planner.methods import astar_search
from transmission_path_planner.metrics import compute_metrics, route_profile_metrics, write_elevation_profile_csv
from transmission_path_planner.smoothing import (
    filter_path_by_ground_grade,
    path_hard_violations,
    sample_dem,
    smooth_path,
    smooth_path_with_constraints,
    snap_to_valid,
)
from transmission_path_planner.stations import load_station_tasks, station_elevation_metrics


class StationPlanningTests(unittest.TestCase):
    def _csv(self, folder, name, headers, rows):
        path = Path(folder) / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_loads_station_pair_and_route_voltage(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "name", "lat", "lon", "voltage_kv", "ground_elevation_m",
                                  "ground_elevation_source", "vertical_datum"],
                                 [["S1", "North", 24.1, 120.1, 220, 120.5, "survey A", "datum A"],
                                  ["S2", "South", 24.2, 120.2, 220, 143.0, "survey B", "datum A"]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            result = load_station_tasks(stations, tasks)
        self.assertEqual(result[0]["route_id"], "R1")
        self.assertEqual(result[0]["from_station"]["station_id"], "S1")
        self.assertEqual(result[0]["voltage_kv"], "220")
        self.assertEqual(result[0]["from_station"]["ground_elevation_m"], 120.5)

    def test_rejects_duplicate_station_csv_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "name", "lat", "lat", "lon", "voltage_kv"],
                                 [["S1", "North", 24, 35, 120, 110]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"],
                              [["R1", "S1", "S1"]])

            with self.assertRaisesRegex(ValueError, "重复列名"):
                load_station_tasks(stations, tasks)

    def test_rejects_duplicate_route_task_csv_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon"],
                                 [["S1", 24, 120], ["S2", 25, 121]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id", "to_station_id"],
                              [["R1", "S1", "S2", "S1"]])

            with self.assertRaisesRegex(ValueError, "重复列名"):
                load_station_tasks(stations, tasks)

    def test_reports_surveyed_station_delta_only_for_matching_datum(self):
        task = {
            "from_station": {"ground_elevation_m": 120.5, "ground_elevation_source": "survey A",
                             "vertical_datum": "1985 National Height Datum"},
            "to_station": {"ground_elevation_m": 143.0, "ground_elevation_source": "survey B",
                           "vertical_datum": "1985 National Height Datum"},
        }

        report = station_elevation_metrics(task)

        self.assertEqual(report["station_ground_elevation_status"], "comparable")
        self.assertEqual(report["station_ground_elevation_delta_m"], 22.5)

    def test_does_not_compare_station_heights_with_mismatched_datum_or_missing_source(self):
        task = {
            "from_station": {"ground_elevation_m": 120.5, "ground_elevation_source": "survey A",
                             "vertical_datum": "datum A"},
            "to_station": {"ground_elevation_m": 143.0, "ground_elevation_source": "survey B",
                           "vertical_datum": "datum B"},
        }
        report = station_elevation_metrics(task)
        self.assertEqual(report["station_ground_elevation_status"], "incompatible_vertical_datums")
        self.assertIsNone(report["station_ground_elevation_delta_m"])

        task["to_station"]["vertical_datum"] = "datum A"
        task["to_station"]["ground_elevation_source"] = ""
        report = station_elevation_metrics(task)
        self.assertEqual(report["station_ground_elevation_status"], "elevation_source_missing")
        self.assertIsNone(report["station_ground_elevation_delta_m"])

    def test_rejects_nonfinite_station_ground_elevation(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon", "ground_elevation_m"],
                                 [["S1", 24.1, 120.1, "NaN"], ["S2", 24.2, 120.2, ""]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            with self.assertRaisesRegex(ValueError, "有限数值"):
                load_station_tasks(stations, tasks)

    def test_rejects_unknown_station_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv", ["station_id", "lat", "lon"], [["S1", 24.1, 120.1]])
            tasks = self._csv(folder, "routes.csv", ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            with self.assertRaisesRegex(ValueError, "不存在"):
                load_station_tasks(stations, tasks)

    def test_building_density_increases_route_cost(self):
        terrain = {"slope": np.zeros((3, 3)), "roughness_27": np.zeros((3, 3))}
        osm = {"building_density": np.zeros((3, 3))}
        osm["building_density"][1, 1] = 10
        cost = build_route_cost(terrain, osm, settlement_weight=2.0)
        self.assertEqual(cost[0, 0], 0)
        self.assertGreater(cost[1, 1], 0)

    def test_astar_can_enforce_local_ground_grade(self):
        cost = np.ones((9, 9), dtype=float)
        elevation = np.zeros((9, 9), dtype=float)
        elevation[4, 4] = 100.0
        path = astar_search(cost, (4, 1), (4, 7), elevation=elevation, pixel_m=100,
                            max_ground_grade=0.5)
        self.assertTrue(path)
        self.assertNotIn((4, 4), path)

    def test_smoothed_candidate_fails_grade_cap_and_reports_violations(self):
        transform = Affine(0.0002, 0, 120.0, 0, -0.0002, 25.0)
        dem = np.zeros((12, 12), dtype=float)
        dem[4, 5] = 200.0
        hard_mask = np.ones(dem.shape, dtype=np.uint8)
        raw = [rowcol_to_geo(r, c, transform) for r, c in [(4, 1), (3, 2), (3, 4), (3, 6), (4, 8)]]
        smoothed = smooth_path_with_constraints(raw, transform, hard_mask)

        filtered, report = filter_path_by_ground_grade(smoothed, dem, transform, 0.3)

        self.assertIsNone(filtered)
        self.assertEqual(report["grade_limit_status"], "fail")
        self.assertGreater(report["grade_limit_violation_count"], 0)
        self.assertEqual(report["grade_limit_threshold_ratio"], 0.3)

    def test_smoothing_keeps_features_that_exceed_50_meter_tolerance(self):
        x_scale = 111320.0 * np.cos(np.radians(25.0))
        y_scale = 110574.0
        route = [
            (120.0, 25.0),
            (120.0 + 100.0 / x_scale, 25.0 + 60.0 / y_scale),
            (120.0 + 200.0 / x_scale, 25.0),
        ]

        smoothed = smooth_path(route, spacing_m=90.0)

        self.assertEqual(len(smoothed), 3)
        self.assertAlmostEqual(smoothed[1][0], route[1][0], places=10)
        self.assertAlmostEqual(smoothed[1][1], route[1][1], places=10)

    def test_snap_to_valid_respects_meter_limit_and_fails_closed(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        hard_mask = np.ones((5, 5), dtype=np.uint8)
        hard_mask[2, 2] = 0
        blocked_center = rowcol_to_geo(2, 2, transform)

        snapped = snap_to_valid([blocked_center], transform, hard_mask, max_offset_m=50.0)

        self.assertIsNone(snapped)

    def test_hard_constraint_check_detects_short_graze_through_blocked_cell(self):
        hard_mask = np.ones((10, 10), dtype=np.uint8)
        hard_mask[5, 5] = 0

        violations = path_hard_violations([(4.99, 5.02), (5.02, 4.99)], Affine.identity(), hard_mask)

        self.assertEqual(violations, 1)

    def test_hard_constraint_check_detects_corner_touch_and_grid_edge_travel(self):
        hard_mask = np.ones((10, 10), dtype=np.uint8)
        hard_mask[4, 5] = 0
        corner_violations = path_hard_violations([(4.5, 4.5), (5.0, 5.0)], Affine.identity(), hard_mask)
        self.assertGreater(corner_violations, 0)

        hard_mask = np.ones((10, 10), dtype=np.uint8)
        hard_mask[3, 4] = 0
        edge_violations = path_hard_violations([(5.0, 2.2), (5.0, 4.8)], Affine.identity(), hard_mask)
        self.assertGreater(edge_violations, 0)

    def test_astar_grade_limit_uses_anisotropic_pixel_spacing(self):
        vertical_dem = np.array([[0.0], [50.0]])
        vertical = astar_search(np.ones((2, 1)), (0, 0), (1, 0), elevation=vertical_dem,
                                pixel_m=100.0, max_ground_grade=0.3, pixel_spacing=(200.0, 100.0))
        horizontal_dem = np.array([[0.0, 50.0]])
        horizontal = astar_search(np.ones((1, 2)), (0, 0), (0, 1), elevation=horizontal_dem,
                                  pixel_m=100.0, max_ground_grade=0.3, pixel_spacing=(200.0, 100.0))

        self.assertEqual(vertical, [(0, 0), (1, 0)])
        self.assertIsNone(horizontal)

    def test_grade_limit_uses_projected_crs_distances(self):
        transform = Affine(100.0, 0.0, 500000.0, 0.0, -200.0, 4600000.0)
        dem = np.array([[0.0, 50.0], [50.0, 100.0]])
        vertical = [rowcol_to_geo(r, 0, transform) for r in (0, 1)]
        horizontal = [rowcol_to_geo(0, c, transform) for c in (0, 1)]

        vertical_path, vertical_report = filter_path_by_ground_grade(
            vertical, dem, transform, 0.3, "EPSG:32651")
        horizontal_path, horizontal_report = filter_path_by_ground_grade(
            horizontal, dem, transform, 0.3, "EPSG:32651")

        self.assertIsNotNone(vertical_path)
        self.assertEqual(vertical_report["grade_limit_status"], "pass")
        self.assertIsNone(horizontal_path)
        self.assertEqual(horizontal_report["grade_limit_status"], "fail")

    def test_anisotropic_astar_matches_dijkstra_cost(self):
        cost = np.array([[1.0, 2.0, 3.0, 1.0, 2.0],
                         [2.0, 1.0, 4.0, 2.0, 1.0],
                         [3.0, 2.0, 1.0, 3.0, 2.0],
                         [1.0, 3.0, 2.0, 1.0, 4.0]])
        spacing = (200.0, 100.0)
        start, end = (0, 0), (3, 4)
        astar = astar_search(cost, start, end, pixel_spacing=spacing)
        dijkstra = astar_search(cost, start, end, use_dijkstra=True, pixel_spacing=spacing)

        def physical_cost(path):
            return sum(0.5 * (cost[pr, pc] + cost[r, c])
                       * np.hypot((r - pr) * spacing[0], (c - pc) * spacing[1])
                       for (pr, pc), (r, c) in zip(path, path[1:]))

        self.assertAlmostEqual(physical_cost(astar), physical_cost(dijkstra))

    def test_weighted_astar_respects_bounded_suboptimality(self):
        cost = np.array([[1.0, 4.0, 4.0, 4.0, 1.0],
                         [1.0, 4.0, 1.0, 4.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
        start, end = (1, 0), (1, 4)
        exact = astar_search(cost, start, end)
        weighted = astar_search(cost, start, end, heuristic_weight=1.5)

        def route_cost(path):
            return sum(0.5 * (cost[pr, pc] + cost[r, c]) * np.hypot(r - pr, c - pc)
                       for (pr, pc), (r, c) in zip(path, path[1:]))

        self.assertLessEqual(route_cost(weighted), 1.5 * route_cost(exact) + 1e-9)

    def test_exports_station_height_and_profile_csv(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        dem = np.zeros((10, 10), dtype=float)
        dem[5, 2:6] = [10.0, 11.0, 12.0, 13.0]
        route = [rowcol_to_geo(5, 2, transform), rowcol_to_geo(5, 5, transform)]
        metrics = route_profile_metrics(route, dem, transform)
        self.assertEqual(metrics["endpoint_dem_elevation_delta_m"], 3.0)
        self.assertNotIn("station_elevation_delta_m", metrics)
        self.assertGreater(metrics["dem_sample_count"], len(route))
        self.assertEqual(metrics["dem_sample_coverage_ratio"], 1.0)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "profile.csv"
            write_elevation_profile_csv(route, dem, transform, output)
            self.assertEqual(output.read_text(encoding="utf-8-sig").splitlines()[0],
                             "point_index,longitude,latitude,distance_from_start_km,dem_elevation_m")

    def test_outside_dem_samples_are_nan_and_profile_reports_coverage(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        dem = np.ones((3, 3), dtype=float)
        route = [rowcol_to_geo(1, -1, transform), rowcol_to_geo(1, 1, transform)]

        self.assertTrue(np.isnan(sample_dem(route[:1], dem, transform)[0]))
        metrics = route_profile_metrics(route, dem, transform)
        self.assertLess(metrics["dem_sample_coverage_ratio"], 1.0)
        self.assertLess(metrics["dem_valid_sample_count"], metrics["dem_sample_count"])

    def test_station_coordinates_replace_cell_centers_and_report_offsets(self):
        route = [(120.0015, 24.9985), (120.0045, 24.9985)]
        station_task = {
            "from_station": {"lon": 120.0012, "lat": 24.9987},
            "to_station": {"lon": 120.0048, "lat": 24.9983},
        }

        adjusted, offsets = attach_station_endpoints(route, station_task)

        self.assertEqual(adjusted[0], (120.0012, 24.9987))
        self.assertEqual(adjusted[-1], (120.0048, 24.9983))
        self.assertGreater(offsets["from_station_to_grid_center_offset_m"], 0)
        self.assertGreater(offsets["to_station_to_grid_center_offset_m"], 0)

    def test_single_cell_station_route_keeps_grid_center_offsets(self):
        station_task = {
            "from_station": {"lat": 24.0001, "lon": 121.0001},
            "to_station": {"lat": 24.0002, "lon": 121.0002},
        }

        route, offsets = attach_station_endpoints([(121.0, 24.0)], station_task)

        self.assertEqual(route, [(121.0001, 24.0001), (121.0002, 24.0002)])
        self.assertGreater(offsets["from_station_to_grid_center_offset_m"], 0.0)
        self.assertGreater(offsets["to_station_to_grid_center_offset_m"], 0.0)

    def test_station_endpoint_restore_rejects_new_hard_obstacle_crossing(self):
        transform = Affine.identity()
        hard_mask = np.ones((6, 6), dtype=np.uint8)
        hard_mask[2, 1] = 0
        raster_route = [(2.5, 2.5), (1.5, 0.5)]
        station_task = {
            "from_station": {"lon": 2.038894157, "lat": 2.302094932},
            "to_station": {"lon": 1.5, "lat": 0.5},
        }

        self.assertEqual(path_hard_violations(raster_route, transform, hard_mask), 0)
        restored, _ = attach_station_endpoints(raster_route, station_task)
        self.assertGreater(path_hard_violations(restored, transform, hard_mask), 0)

        accepted, report = validate_route_hard_constraints(restored, transform, hard_mask)

        self.assertIsNone(accepted)
        self.assertEqual(report["hard_constraint_status"], "fail")
        self.assertGreater(report["hard_violations"], 0)

    def test_final_route_selection_skips_hard_constraint_violations(self):
        path = [(0.0, 0.0), (1.0, 1.0)]
        methods = {"A*(baseline)": path, "VIN-Grad": path}
        metrics = {
            "A*(baseline)": {"hard_violations": 1},
            "VIN-Grad": {"hard_violations": 0},
        }

        self.assertEqual(
            _select_final_route(methods, metrics),
            (None, None, "no_hard_constraint_compliant_route"),
        )

    def test_final_route_selection_fails_closed_without_valid_candidate(self):
        path = [(0.0, 0.0), (1.0, 1.0)]

        self.assertEqual(
            _select_final_route(
                {"A*(baseline)": path},
                {"A*(baseline)": {"hard_violations": 2}},
            ),
            (None, None, "no_hard_constraint_compliant_route"),
        )

    def test_missing_osm_proxy_layers_mark_data_quality_incomplete(self):
        status, missing = screening_data_quality({"buildings": {"status": "available"}})

        self.assertEqual(status, "incomplete_osm_proxies")
        self.assertEqual(missing, ["landuse"])

    def test_mean_cost_samples_route_interior_cells(self):
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        dem = np.zeros((5, 5), dtype=float)
        cost = np.zeros((5, 5), dtype=float)
        cost[2, 2] = 10.0
        hard_mask = np.ones((5, 5), dtype=np.uint8)
        route = [rowcol_to_geo(2, col, transform) for col in (0, 4)]

        metrics = compute_metrics(route, dem, transform, cost, hard_mask,
                                  start_ll=(2.5, 0.5), end_ll=(2.5, 4.5))

        self.assertEqual(metrics["mean_cost"], 2.0)


if __name__ == "__main__":
    unittest.main()
