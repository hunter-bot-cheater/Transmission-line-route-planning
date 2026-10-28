import csv
import heapq
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from affine import Affine

from transmission_path_planner.cost import build_route_cost
from transmission_path_planner.cli import (
    TOWER_GRAPH_METHOD,
    _select_final_route,
    attach_station_endpoints,
    build_tower_masks,
    overhead_span_grade_report,
    route_planning_status,
    screening_data_quality,
    tower_candidate_validation_status,
    validate_endpoint_cells,
    validate_route_hard_constraints,
    validate_tower_span_lengths,
)
from transmission_path_planner.geometry import pixel_spacing_meters, rowcol_to_geo
from transmission_path_planner.methods import _astar_parent_index_dtype, astar_search
from transmission_path_planner.metrics import compute_metrics, route_profile_metrics, write_elevation_profile_csv
from transmission_path_planner.smoothing import (
    filter_path_by_ground_grade,
    ground_grade_limit_report,
    path_hard_violations,
    sample_dem,
    smooth_path,
    smooth_path_with_constraints,
    snap_to_valid,
)
from transmission_path_planner.stations import (
    load_station_tasks,
    station_coordinate_crs_properties,
    station_endpoint_properties,
    station_clearance_elevation_overrides,
    station_elevation_metrics,
)
from transmission_path_planner.experimental_tower_graph import tower_graph_route


class StationPlanningTests(unittest.TestCase):
    def _csv(self, folder, name, headers, rows):
        path = Path(folder) / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_tower_masks_allow_steep_spans_but_not_steep_sites(self):
        dem = np.zeros((5, 5), dtype=np.float64)
        slope = np.zeros_like(dem)
        slope[:, 2] = 45.0
        user_hard = np.zeros_like(dem, dtype=np.uint8)
        user_hard[1, 1] = 1
        buildings = np.zeros_like(user_hard)
        buildings[3, 3] = 1
        dem[0, 0] = np.nan
        building_distance = np.full_like(dem, 100.0)
        building_distance[4, 4] = 5.0

        span_mask, site_mask = build_tower_masks(
            dem, slope, user_hard, max_surface_slope_deg=30.0,
            building_mask=buildings, building_distance_m=building_distance,
            building_clearance_m=10.0)

        self.assertTrue(span_mask[2, 2])
        self.assertFalse(site_mask[2, 2])
        self.assertFalse(span_mask[1, 1])
        self.assertFalse(site_mask[1, 1])
        self.assertFalse(span_mask[3, 3])
        self.assertFalse(site_mask[3, 3])
        self.assertFalse(span_mask[0, 0])
        self.assertFalse(span_mask[4, 4])

        span_cost = np.where(span_mask > 0, 1.0, np.inf)
        span_path = astar_search(span_cost, (2, 0), (2, 4))
        self.assertIn((2, 2), span_path)
        with self.assertRaisesRegex(ValueError, "地表坡度阈值"):
            validate_endpoint_cells(
                (2, 2), (2, 0), site_mask.shape, site_mask,
                {"地表坡度阈值": slope > 30.0},
            )

        route = tower_graph_route(
            np.ones((5, 5)), span_mask, site_mask, (2, 0), (2, 4),
            min_span_m=1.0, max_span_m=2.5, candidate_stride=1)
        self.assertIsNotNone(route)
        self.assertTrue(all(col != 2 for _, col in route[1:-1]))
        self.assertTrue(any(min(first[1], second[1]) < 2 < max(first[1], second[1])
                            for first, second in zip(route, route[1:])))

    def test_overhead_span_grade_is_explicitly_unassessed(self):
        report = overhead_span_grade_report()
        self.assertEqual(report["grade_limit_status"], "not_applicable_overhead_spans")
        self.assertFalse(report["grade_limit_assessed"])
        self.assertIsNone(report["grade_limit_violation_count"])
        self.assertIsNone(report["grade_limit_unassessed_count"])

    def test_explicit_tower_graph_selection_requires_all_screening_checks(self):
        route = [(121.0, 25.0), (121.003, 25.0)]
        accepted_metrics = {
            "candidate_validation_status": "survived_configured_screening",
            "hard_constraint_status": "pass",
            "hard_violations": 0,
            "building_geometry_status": "pass",
            "tower_span_status": "pass",
            "tower_span_violation_count": 0,
            "tower_site_slope_status": "interior_candidates_screened",
            "tower_site_max_surface_slope_deg": 30.0,
            "clearance_profile_status": "dem_only_profile_screening",
            "clearance_profile_spans": [{"status": "sampled_clearance_ok"}],
        }
        selected = _select_final_route(
            {TOWER_GRAPH_METHOD: route},
            {TOWER_GRAPH_METHOD: accepted_metrics},
            select_tower_graph=True)
        self.assertEqual(selected,
                         (TOWER_GRAPH_METHOD, None,
                          "explicit_tower_graph_screen_passed"))

        failed_metrics = {**accepted_metrics, "hard_violations": 1}
        fallback = _select_final_route(
            {TOWER_GRAPH_METHOD: route, "A*(baseline)": route},
            {TOWER_GRAPH_METHOD: failed_metrics,
             "A*(baseline)": {"hard_violations": 0}},
            select_tower_graph=True)
        self.assertEqual(fallback,
                         (None, None, "no_hard_constraint_compliant_route"))

        unknown_clearance = {
            **accepted_metrics,
            "clearance_profile_status": "not_requested",
        }
        self.assertIsNone(_select_final_route(
            {TOWER_GRAPH_METHOD: route},
            {TOWER_GRAPH_METHOD: unknown_clearance},
            select_tower_graph=True)[0])

        missing_building_check = {
            **accepted_metrics,
            "building_geometry_status": "not_checked",
        }
        self.assertIsNone(_select_final_route(
            {TOWER_GRAPH_METHOD: route},
            {TOWER_GRAPH_METHOD: missing_building_check},
            select_tower_graph=True)[0])

        unassessed_site_slope = {
            **accepted_metrics,
            "tower_site_slope_status": "not_evaluated",
        }
        self.assertIsNone(_select_final_route(
            {TOWER_GRAPH_METHOD: route},
            {TOWER_GRAPH_METHOD: unassessed_site_slope},
            select_tower_graph=True)[0])

        missing_site_slope = {
            key: value for key, value in accepted_metrics.items()
            if key not in {"tower_site_slope_status", "tower_site_max_surface_slope_deg"}
        }
        self.assertIsNone(_select_final_route(
            {TOWER_GRAPH_METHOD: route},
            {TOWER_GRAPH_METHOD: missing_site_slope},
            select_tower_graph=True)[0])

        for invalid_limit in (0, -1, 91, float("nan"), float("inf"), True, "30"):
            with self.subTest(invalid_limit=invalid_limit):
                invalid_slope = {
                    **accepted_metrics,
                    "tower_site_max_surface_slope_deg": invalid_limit,
                }
                self.assertIsNone(_select_final_route(
                    {TOWER_GRAPH_METHOD: route},
                    {TOWER_GRAPH_METHOD: invalid_slope},
                    select_tower_graph=True)[0])

    def test_tower_candidate_validation_status_tracks_post_search_result(self):
        route = [(121.0, 25.0), (121.003, 25.0)]
        self.assertEqual(
            tower_candidate_validation_status(
                route, route, "not_requested", "pass", "interior_candidates_screened"),
            "survived_configured_screening")
        self.assertEqual(
            tower_candidate_validation_status(route, route, "not_requested", "pass"),
            "site_slope_not_evaluated")
        self.assertEqual(
            tower_candidate_validation_status(
                None, route, "dem_only_profile_screening", "pass"),
            "rejected_after_search")
        self.assertEqual(
            tower_candidate_validation_status(None, None, "not_requested", None),
            "no_path")
        self.assertEqual(
            tower_candidate_validation_status(
                None, None, "unknown_dem_vertical_datum_missing", None),
            "unknown_clearance_reference")
        self.assertEqual(
            tower_candidate_validation_status(route, route, "dem_only_profile_screening", None),
            "incomplete_building_geometry_check")

    def test_restored_tower_endpoints_are_rechecked_against_span_limits(self):
        invalid_route = [(121.0, 25.0), (121.001, 25.0), (121.02, 25.0)]
        rejected, report = validate_tower_span_lengths(
            invalid_route, min_span_m=200.0, max_span_m=500.0)

        self.assertIsNone(rejected)
        self.assertEqual(report["tower_span_status"], "fail")
        self.assertEqual(report["tower_span_violation_count"], 2)
        self.assertEqual(report["tower_span_invalid_indices"], [0, 1])

        valid_route = [(121.0, 25.0), (121.003, 25.0), (121.006, 25.0)]
        accepted, valid_report = validate_tower_span_lengths(
            valid_route, min_span_m=200.0, max_span_m=500.0)
        self.assertEqual(accepted, valid_route)
        self.assertEqual(valid_report["tower_span_status"], "pass")

    def test_loads_station_pair_and_route_voltage(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "name", "lat", "lon", "coordinate_crs", "voltage_kv",
                                  "source", "point_role", "verification_status", "verification_reference",
                                  "ground_elevation_m", "ground_elevation_source", "vertical_datum"],
                                 [["S1", "North", 24.1, 120.1, "EPSG:4326", 220, "survey A",
                                   "line_connection_point", "verified", "record-A", 120.5, "survey A", "datum A"],
                                  ["S2", "South", 24.2, 120.2, "EPSG:4326", 220, "survey B",
                                   "line_connection_point", "verified", "record-B", 143.0, "survey B", "datum A"]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            result = load_station_tasks(stations, tasks)
        self.assertEqual(result[0]["route_id"], "R1")
        self.assertEqual(result[0]["from_station"]["station_id"], "S1")
        self.assertEqual(result[0]["from_station"]["coordinate_crs"], "EPSG:4326")
        self.assertEqual(
            station_coordinate_crs_properties(result[0]),
            {"from_station_coordinate_crs": "EPSG:4326",
             "to_station_coordinate_crs": "EPSG:4326"},
        )
        self.assertEqual(result[0]["voltage_kv"], "220")
        self.assertEqual(result[0]["from_station"]["ground_elevation_m"], 120.5)
        self.assertEqual(result[0]["endpoint_qualification"], "declared_verified_line_connection_points")
        self.assertEqual(station_endpoint_properties(result[0])["from_station_point_role"],
                         "line_connection_point")

    def test_unverified_or_non_connection_endpoints_are_rejected_by_default(self):
        headers = ["station_id", "lat", "lon", "coordinate_crs", "source", "point_role",
                   "verification_status", "verification_reference"]
        cases = [
            (["S1", 24.1, 120.1, "EPSG:4326", "survey", "line_connection_point", "unknown", ""],
             "声明为已核验线路接入点"),
            (["S1", 24.1, 120.1, "EPSG:4326", "survey", "station_centroid", "verified", "record-A"],
             "声明为已核验线路接入点"),
            (["S1", 24.1, 120.1, "EPSG:4326", "", "line_connection_point", "verified", "record-A"],
             "声明为已核验线路接入点"),
            (["S1", 24.1, 120.1, "EPSG:4326", "survey", "line_connection_point", "verified", ""],
             "声明为已核验线路接入点"),
        ]
        for first_station, message in cases:
            with self.subTest(first_station=first_station), tempfile.TemporaryDirectory() as folder:
                stations = self._csv(folder, "stations.csv", headers,
                                     [first_station,
                                      ["S2", 24.2, 120.2, "EPSG:4326", "survey", "line_connection_point",
                                       "verified", "record-B"]])
                tasks = self._csv(folder, "routes.csv", ["route_id", "from_station_id", "to_station_id"],
                                  [["R1", "S1", "S2"]])
                with self.assertRaisesRegex(ValueError, message):
                    load_station_tasks(stations, tasks)

    def test_exploratory_endpoints_are_explicitly_qualified_and_never_screening_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon", "coordinate_crs"],
                                 [["S1", 24.1, 120.1, "EPSG:4326"], ["S2", 24.2, 120.2, "EPSG:4326"]])
            tasks = self._csv(folder, "routes.csv", ["route_id", "from_station_id", "to_station_id"],
                              [["R1", "S1", "S2"]])
            result = load_station_tasks(stations, tasks, allow_unverified=True)[0]

        self.assertEqual(result["endpoint_qualification"], "exploratory_only_unverified_endpoints")
        self.assertEqual(route_planning_status([(1.0, 1.0), (2.0, 2.0)], result, []),
                         "exploratory_only_unverified_endpoints")
        self.assertEqual(route_planning_status(None, result, []), "no_hard_constraint_compliant_route")
        self.assertEqual(route_planning_status([(1.0, 1.0), (2.0, 2.0)], None, ["buildings"]),
                         "screening_candidate_data_incomplete")
        self.assertEqual(
            route_planning_status([(1.0, 1.0), (2.0, 2.0)], None,
                                  ["official_hazards"],
                                  osm_coverage_unverified=True),
            "screening_candidate_data_incomplete",
        )

    def test_declared_verified_endpoint_does_not_open_blocked_dem_or_building_cell(self):
        hard_mask = np.ones((3, 3), dtype=np.uint8)
        hard_mask[0, 0] = 0
        blocks = {"DEM NoData": np.zeros((3, 3), dtype=bool),
                  "OSM建筑足迹": np.zeros((3, 3), dtype=bool)}
        blocks["OSM建筑足迹"][0, 0] = True
        with self.assertRaisesRegex(ValueError, "OSM建筑足迹"):
            validate_endpoint_cells((0, 0), (2, 2), hard_mask.shape, hard_mask, blocks)

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

    def test_station_coordinate_crs_properties_are_empty_without_station_task(self):
        self.assertEqual(station_coordinate_crs_properties(None), {})

    def test_rejects_duplicate_route_task_csv_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon", "coordinate_crs"],
                                 [["S1", 24, 120, "EPSG:4326"], ["S2", 25, 121, "EPSG:4326"]])
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

    def test_station_clearance_elevations_require_matching_declared_dem_datum(self):
        task = {
            "from_station": {"ground_elevation_m": 120.5,
                             "ground_elevation_source": "survey A",
                             "vertical_datum": "1985 National Height Datum"},
            "to_station": {"ground_elevation_m": 143.0,
                           "ground_elevation_source": "survey B",
                           "vertical_datum": "1985 National Height Datum"},
        }

        overrides, status = station_clearance_elevation_overrides(
            task, "1985 national height datum")
        self.assertEqual(status, "station_and_dem_vertical_datums_declared_matching")
        self.assertEqual(overrides, {
            "start_ground_elevation_m": 120.5,
            "goal_ground_elevation_m": 143.0,
        })
        self.assertEqual(
            station_clearance_elevation_overrides(task, None),
            (None, "dem_vertical_datum_missing"))
        self.assertEqual(
            station_clearance_elevation_overrides(task, "ellipsoid"),
            (None, "incompatible_dem_vertical_datum"))

    def test_station_clearance_elevation_statuses_fail_closed_on_incomplete_data(self):
        task = {
            "from_station": {"ground_elevation_m": None,
                             "ground_elevation_source": "",
                             "vertical_datum": ""},
            "to_station": {"ground_elevation_m": None,
                           "ground_elevation_source": "",
                           "vertical_datum": ""},
        }
        self.assertEqual(station_clearance_elevation_overrides(task, "datum"),
                         (None, "dem_only"))

        task["from_station"].update({
            "ground_elevation_m": 10.0,
            "ground_elevation_source": "survey",
            "vertical_datum": "datum",
        })
        self.assertEqual(
            station_clearance_elevation_overrides(task, "datum"),
            (None, "incomplete_station_measurements"))

        task["to_station"].update({
            "ground_elevation_m": 20.0,
            "ground_elevation_source": "",
            "vertical_datum": "datum",
        })
        self.assertEqual(
            station_clearance_elevation_overrides(task, "datum"),
            (None, "elevation_source_missing"))

    def test_rejects_nonfinite_station_ground_elevation(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon", "coordinate_crs", "ground_elevation_m"],
                                 [["S1", 24.1, 120.1, "EPSG:4326", "NaN"],
                                  ["S2", 24.2, 120.2, "EPSG:4326", ""]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            with self.assertRaisesRegex(ValueError, "有限数值"):
                load_station_tasks(stations, tasks)

    def test_rejects_unknown_station_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon", "coordinate_crs"],
                                 [["S1", 24.1, 120.1, "EPSG:4326"]])
            tasks = self._csv(folder, "routes.csv", ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            with self.assertRaisesRegex(ValueError, "不存在"):
                load_station_tasks(stations, tasks)

    def test_rejects_station_coordinates_without_declared_crs(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv", ["station_id", "lat", "lon"],
                                 [["S1", 24.1, 120.1]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"],
                              [["R1", "S1", "S1"]])
            with self.assertRaisesRegex(ValueError, "缺少列: coordinate_crs"):
                load_station_tasks(stations, tasks)

    def test_rejects_unsupported_station_coordinate_crs(self):
        with tempfile.TemporaryDirectory() as folder:
            stations = self._csv(folder, "stations.csv",
                                 ["station_id", "lat", "lon", "coordinate_crs"],
                                 [["S1", 24.1, 120.1, "EPSG:3826"]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"],
                              [["R1", "S1", "S1"]])
            with self.assertRaisesRegex(ValueError, "必须明确为 EPSG:4326"):
                load_station_tasks(stations, tasks)

    def test_building_density_increases_route_cost(self):
        terrain = {"slope": np.zeros((3, 3)), "roughness_27": np.zeros((3, 3))}
        osm = {"building_density": np.zeros((3, 3))}
        osm["building_density"][1, 1] = 0.1
        cost = build_route_cost(terrain, osm, settlement_weight=2.0)
        self.assertEqual(cost[0, 0], 0)
        self.assertAlmostEqual(cost[1, 1], 0.2)

    def test_terrain_component_weights_preserve_defaults_and_can_be_ablated(self):
        terrain = {
            "slope": np.array([[45.0, 0.0]], dtype=np.float32),
            "roughness_27": np.array([[0.0, 100.0]], dtype=np.float32),
        }
        default_cost = build_route_cost(terrain, {})
        explicit_defaults = build_route_cost(
            terrain, {}, terrain_slope_weight=0.75, roughness_weight=0.25)
        ablated = build_route_cost(
            terrain, {}, terrain_slope_weight=0.0, roughness_weight=0.0)

        np.testing.assert_allclose(default_cost, explicit_defaults)
        np.testing.assert_allclose(default_cost, [[0.75, 0.25]])
        np.testing.assert_allclose(ablated, 0.0)

    def test_building_density_cost_is_independent_of_task_window_extremes(self):
        terrain = {"slope": np.zeros((3, 3)), "roughness_27": np.zeros((3, 3))}
        local = {"building_density": np.zeros((3, 3), dtype=np.float32)}
        local["building_density"][1, 1] = 0.1
        expanded = {"building_density": np.full((5, 5), 0.9, dtype=np.float32)}
        expanded["building_density"][2, 2] = 0.1
        local_cost = build_route_cost(terrain, local)[1, 1]
        expanded_cost = build_route_cost(
            {"slope": np.zeros((5, 5)), "roughness_27": np.zeros((5, 5))},
            expanded)[2, 2]
        self.assertAlmostEqual(float(local_cost), float(expanded_cost))

    def test_astar_can_enforce_local_ground_grade(self):
        cost = np.ones((9, 9), dtype=float)
        elevation = np.zeros((9, 9), dtype=float)
        elevation[4, 4] = 100.0
        path = astar_search(cost, (4, 1), (4, 7), elevation=elevation, pixel_m=100,
                            max_ground_grade=0.5)
        self.assertTrue(path)
        self.assertNotIn((4, 4), path)

    def test_astar_parent_indices_use_smallest_safe_integer_type(self):
        int32_max = np.iinfo(np.int32).max
        self.assertIs(_astar_parent_index_dtype(4096 * 4096), np.int32)
        self.assertIs(_astar_parent_index_dtype(int32_max + 1), np.int32)
        self.assertIs(_astar_parent_index_dtype(int32_max + 2), np.int64)

    def test_astar_grade_cap_matches_final_validator_on_diagonal_edges(self):
        transform = Affine(0.0001, 0, 120.0, 0, -0.0001, 25.0)
        elevation = np.zeros((12, 12), dtype=float)
        elevation[5, 5] = 100.0
        cost = np.full(elevation.shape, 20.0)
        cost[5, :] = 1.0
        spacing = pixel_spacing_meters(
            transform, transform.f + transform.e * (elevation.shape[0] / 2),
            "EPSG:4326")

        for start, goal in (((5, 0), (5, 10)), ((5, 10), (5, 0))):
            for turn_weight in (0.0, 0.4):
                with self.subTest(start=start, turn_penalty_weight=turn_weight):
                    path = astar_search(
                        cost, start, goal, elevation=elevation,
                        pixel_spacing=spacing, max_ground_grade=0.3,
                        turn_penalty_weight=turn_weight)
                    self.assertTrue(path)
                    route = [rowcol_to_geo(row, col, transform) for row, col in path]
                    report = ground_grade_limit_report(
                        route, elevation, transform, 0.3, "EPSG:4326")
                    self.assertEqual(report["grade_limit_status"], "pass")

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

    def test_smoothing_rejects_higher_cost_corridor(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        raw = [rowcol_to_geo(3, 0, transform), rowcol_to_geo(2, 0, transform),
               rowcol_to_geo(2, 6, transform), rowcol_to_geo(3, 6, transform)]
        straight = [raw[0], raw[-1]]
        cost = np.ones((7, 7), dtype=np.float32)
        cost[3, 1:6] = 20.0
        dem = np.zeros(cost.shape, dtype=np.float32)
        hard_mask = np.ones(cost.shape, dtype=np.uint8)

        with patch("transmission_path_planner.smoothing.smooth_path", return_value=straight):
            result = smooth_path_with_constraints(
                raw, transform, hard_mask, cost, dem, 0.0,
                pixel_spacing=(100.0, 100.0))

        self.assertEqual(result, raw)

    def test_smoothing_keeps_equal_cost_route(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        raw = [rowcol_to_geo(3, 0, transform), rowcol_to_geo(2, 0, transform),
               rowcol_to_geo(2, 6, transform), rowcol_to_geo(3, 6, transform)]
        straight = [raw[0], raw[-1]]
        cost = np.ones((7, 7), dtype=np.float32)
        dem = np.zeros(cost.shape, dtype=np.float32)
        hard_mask = np.ones(cost.shape, dtype=np.uint8)

        with patch("transmission_path_planner.smoothing.smooth_path", return_value=straight):
            result = smooth_path_with_constraints(
                raw, transform, hard_mask, cost, dem, 0.0,
                pixel_spacing=(100.0, 100.0))

        self.assertEqual(result, straight)

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

    def test_astar_uses_zero_lower_bound_when_raster_has_zero_cost(self):
        cost = np.full((3, 22), np.inf, dtype=np.float64)
        cost[0, :] = 0.0
        cost[1, :] = 1e-7
        cost[1, [0, -1]] = 0.0
        start, end = (1, 0), (1, 21)

        def route_cost(path):
            return sum(0.5 * (cost[r0, c0] + cost[r1, c1])
                       * math.hypot(r1 - r0, c1 - c0)
                       for (r0, c0), (r1, c1) in zip(path, path[1:]))

        astar = astar_search(cost, start, end)
        dijkstra = astar_search(cost, start, end, use_dijkstra=True)

        self.assertAlmostEqual(route_cost(dijkstra), 0.0)
        self.assertAlmostEqual(route_cost(astar), route_cost(dijkstra))

    def test_astar_uses_physical_length_to_break_equal_cost_routes(self):
        spacing = (3.0, 1.0)
        path = astar_search(
            np.zeros((7, 9)), (0, 0), (6, 8), pixel_spacing=spacing)
        length = sum(
            math.hypot((r1 - r0) * spacing[0], (c1 - c0) * spacing[1])
            for (r0, c0), (r1, c1) in zip(path, path[1:]))
        expected = 6 * math.hypot(*spacing) + 2 * spacing[1]

        self.assertAlmostEqual(length, expected)

    def test_astar_keeps_lower_cost_longer_route_over_shorter_costlier_route(self):
        cost = np.full((3, 7), np.inf, dtype=np.float64)
        cost[0, :] = 0.0
        cost[1, :] = 4.0
        cost[1, [0, -1]] = 0.0
        start, goal = (1, 0), (1, 6)
        path = astar_search(cost, start, goal)

        def objective(route):
            total_cost = 0.0
            total_length = 0.0
            for (r0, c0), (r1, c1) in zip(route, route[1:]):
                step_length = math.hypot(r1 - r0, c1 - c0)
                total_length += step_length
                total_cost += 0.5 * (cost[r0, c0] + cost[r1, c1]) * step_length
            return total_cost, total_length

        planned_cost, planned_length = objective(path)
        short_direct_cost, short_direct_length = objective(
            [(1, col) for col in range(7)])

        self.assertAlmostEqual(planned_cost, 0.0)
        self.assertGreater(planned_length, short_direct_length)
        self.assertGreater(short_direct_cost, planned_cost)

    def test_astar_lexicographic_objective_matches_small_dijkstra_oracle(self):
        cost = np.array([
            [0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, np.inf, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 1.0, 0.0],
        ], dtype=np.float64)
        spacing = (1.5, 1.0)
        start, goal = (0, 0), (3, 5)
        directions = ((-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1))
        best = {start: (0.0, 0.0)}
        queue = [(0.0, 0.0, start)]
        while queue:
            total_cost, total_length, (row, col) = heapq.heappop(queue)
            if (total_cost, total_length) != best[(row, col)]:
                continue
            if (row, col) == goal:
                break
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < cost.shape[0] and 0 <= nc < cost.shape[1]):
                    continue
                if not np.isfinite(cost[nr, nc]):
                    continue
                if dr and dc and (not np.isfinite(cost[row + dr, col])
                                  or not np.isfinite(cost[row, col + dc])):
                    continue
                step_length = math.hypot(dr * spacing[0], dc * spacing[1])
                candidate = (
                    total_cost + 0.5 * (cost[row, col] + cost[nr, nc]) * step_length,
                    total_length + step_length,
                )
                if candidate < best.get((nr, nc), (math.inf, math.inf)):
                    best[(nr, nc)] = candidate
                    heapq.heappush(queue, (*candidate, (nr, nc)))

        path = astar_search(cost, start, goal, pixel_spacing=spacing)
        objective = (0.0, 0.0)
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            step_length = math.hypot((r1 - r0) * spacing[0],
                                     (c1 - c0) * spacing[1])
            objective = (
                objective[0] + 0.5 * (cost[r0, c0] + cost[r1, c1]) * step_length,
                objective[1] + step_length,
            )

        self.assertEqual(objective, best[goal])

    def test_zero_cost_astar_uses_goal_distance_only_as_tie_break(self):
        cost = np.zeros((32, 64), dtype=np.float32)
        pop_count = 0
        heap_pop = heapq.heappop

        def counted_pop(heap):
            nonlocal pop_count
            pop_count += 1
            return heap_pop(heap)

        with patch("heapq.heappop", side_effect=counted_pop):
            path = astar_search(cost, (16, 2), (16, 61))

        self.assertEqual(len(path), 60)
        self.assertLess(pop_count, 2 * len(path))

    def test_zero_cost_turn_astar_uses_goal_distance_only_as_tie_break(self):
        cost = np.zeros((32, 64), dtype=np.float32)
        pop_count = 0
        heap_pop = heapq.heappop

        def counted_pop(heap):
            nonlocal pop_count
            pop_count += 1
            return heap_pop(heap)

        with patch("heapq.heappop", side_effect=counted_pop):
            path = astar_search(cost, (16, 2), (16, 61), turn_penalty_weight=1.0)

        self.assertEqual(len(path), 60)
        self.assertLess(pop_count, 2 * len(path))

    def test_weighted_astar_cost_on_nonuniform_fixture(self):
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

    def test_weighted_astar_reopening_keeps_cost_within_weight_bound(self):
        rng = np.random.default_rng(417)
        weight = 3.0

        def route_cost(path, cost):
            return sum(0.5 * (cost[pr, pc] + cost[r, c])
                       * np.hypot(r - pr, c - pc)
                       for (pr, pc), (r, c) in zip(path, path[1:]))

        for trial in range(12):
            cost = rng.choice([1.0, 1.5, 3.0, 8.0], size=(14, 20))
            gap_row = 2 + trial
            cost[1:13, 10] = np.inf
            cost[gap_row, 10] = rng.choice([1.0, 1.5, 3.0, 8.0])
            start, end = (7, 1), (7, 18)
            exact = astar_search(cost, start, end, use_dijkstra=True)
            weighted = astar_search(cost, start, end, heuristic_weight=weight)

            self.assertIsNotNone(exact)
            self.assertIsNotNone(weighted)
            self.assertLessEqual(
                route_cost(weighted, cost),
                weight * route_cost(exact, cost) + 1e-9,
            )

    def test_weighted_astar_actually_reopens_a_better_closed_node(self):
        rng = np.random.default_rng(0)
        cost = rng.choice([1.0, 1.5, 3.0, 8.0, 20.0], size=(14, 20))
        cost[1:13, 10] = np.inf
        cost[2, 10] = rng.choice([1.0, 1.5, 3.0, 8.0, 20.0])
        best_g = {7 * cost.shape[1] + 1: 0.0}
        expanded_g = {}
        events = {"reopens": 0, "stale_pops": 0}
        real_push, real_pop = heapq.heappush, heapq.heappop

        def tracked_push(heap, item):
            node, queued_g = item[3], item[4]
            if node in expanded_g and queued_g < expanded_g[node]:
                events["reopens"] += 1
            best_g[node] = min(best_g.get(node, math.inf), queued_g)
            real_push(heap, item)

        def tracked_pop(heap):
            item = real_pop(heap)
            node, queued_g = item[3], item[4]
            if queued_g != best_g[node]:
                events["stale_pops"] += 1
            else:
                expanded_g[node] = queued_g
            return item

        with patch("heapq.heappush", side_effect=tracked_push), \
                patch("heapq.heappop", side_effect=tracked_pop):
            path = astar_search(cost, (7, 1), (7, 18), heuristic_weight=3.0)

        self.assertIsNotNone(path)
        self.assertGreater(events["reopens"], 0)
        self.assertGreater(events["stale_pops"], 0)

    def test_astar_route_is_invariant_to_endpoint_order(self):
        cost = np.array([[1, 2, 2, 4, 1, 1],
                         [1, 3, 2, 2, 1, 1],
                         [1, 1, 1, 4, 2, 1],
                         [3, 1, 2, 1, 1, 1]], dtype=np.float64)
        start, end = (0, 0), (3, 5)

        for weight in (1.0, 1.5):
            forward = astar_search(cost, start, end, heuristic_weight=weight)
            backward = astar_search(cost, end, start, heuristic_weight=weight)
            self.assertEqual(forward, backward[::-1])

    @staticmethod
    def _turn_route_cost(path, cost, turn_weight, pixel_spacing=None):
        edges = []
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            dr, dc = r1 - r0, c1 - c0
            length = (math.hypot(dr * pixel_spacing[0], dc * pixel_spacing[1])
                      if pixel_spacing is not None else math.hypot(dr, dc))
            edges.append((dr, dc, 0.5 * (cost[r0, c0] + cost[r1, c1]) * length))
        total = sum(edge[2] for edge in edges)
        for first, second in zip(edges, edges[1:]):
            if pixel_spacing is None:
                v0 = first[:2]
                v1 = second[:2]
            else:
                v0 = (first[0] * pixel_spacing[0], first[1] * pixel_spacing[1])
                v1 = (second[0] * pixel_spacing[0], second[1] * pixel_spacing[1])
            cosine = ((v0[0] * v1[0] + v0[1] * v1[1])
                      / (math.hypot(*v0) * math.hypot(*v1)))
            total += turn_weight * 0.5 * (first[2] + second[2]) * (1 - cosine)
        return total

    def _assert_route_legal(self, path, cost):
        self.assertIsNotNone(path)
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            dr, dc = r1 - r0, c1 - c0
            self.assertLessEqual(max(abs(dr), abs(dc)), 1)
            self.assertNotEqual((dr, dc), (0, 0))
            self.assertTrue(np.isfinite(cost[r1, c1]))
            if dr and dc:
                self.assertTrue(np.isfinite(cost[r0 + dr, c0]))
                self.assertTrue(np.isfinite(cost[r0, c0 + dc]))

    @staticmethod
    def _direction_changes(path):
        directions = [(r1 - r0, c1 - c0)
                      for (r0, c0), (r1, c1) in zip(path, path[1:])]
        return sum(first != second for first, second in zip(directions, directions[1:]))

    def test_turn_penalty_reduces_corners_around_uniform_cost_obstacle(self):
        cost = np.ones((15, 25), dtype=np.float64)
        cost[4:11, 8:17] = np.inf
        start, end = (7, 0), (7, 24)

        baseline = astar_search(cost, start, end)
        rounded = astar_search(cost, start, end, turn_penalty_weight=0.5)

        self._assert_route_legal(baseline, cost)
        self._assert_route_legal(rounded, cost)
        self.assertLess(self._direction_changes(rounded), self._direction_changes(baseline))
        self.assertEqual(rounded[0], start)
        self.assertEqual(rounded[-1], end)

    def test_turn_penalty_astar_matches_dijkstra_on_fixed_random_maps(self):
        rng = np.random.default_rng(20261023)
        spacing = (180.0, 100.0)
        turn_weight = 0.7

        for case in range(16):
            with self.subTest(case=case):
                cost = rng.uniform(1.0, 8.0, size=(9, 11))
                cost[rng.random(cost.shape) < 0.18] = np.inf
                cost[0, :] = cost[-1, :] = 1.0
                cost[:, 0] = cost[:, -1] = 1.0
                start, end = (0, 0), (8, 10)

                astar = astar_search(
                    cost, start, end, pixel_spacing=spacing,
                    turn_penalty_weight=turn_weight)
                dijkstra = astar_search(
                    cost, start, end, pixel_spacing=spacing, use_dijkstra=True,
                    turn_penalty_weight=turn_weight)

                self._assert_route_legal(astar, cost)
                self._assert_route_legal(dijkstra, cost)
                self.assertAlmostEqual(
                    self._turn_route_cost(astar, cost, turn_weight, spacing),
                    self._turn_route_cost(dijkstra, cost, turn_weight, spacing),
                    places=8,
                )

    def test_zero_lower_bound_turn_astar_matches_dijkstra(self):
        cost = np.full((9, 15), 2.0, dtype=np.float64)
        cost[0, :] = 0.0
        cost[2:7, 7] = np.inf
        spacing = (240.0, 100.0)
        turn_weight = 0.8
        start, end = (4, 0), (4, 14)

        astar = astar_search(
            cost, start, end, pixel_spacing=spacing,
            turn_penalty_weight=turn_weight)
        dijkstra = astar_search(
            cost, start, end, pixel_spacing=spacing, use_dijkstra=True,
            turn_penalty_weight=turn_weight)

        self._assert_route_legal(astar, cost)
        self._assert_route_legal(dijkstra, cost)
        self.assertGreater(self._turn_route_cost(astar, cost, turn_weight, spacing), 0.0)
        self.assertAlmostEqual(
            self._turn_route_cost(astar, cost, turn_weight, spacing),
            self._turn_route_cost(dijkstra, cost, turn_weight, spacing),
            places=8,
        )

    def test_turn_penalty_route_is_symmetric_with_anisotropic_spacing(self):
        cost = np.ones((13, 19), dtype=np.float64)
        cost[3:10, 6:13] = np.inf
        start, end = (6, 0), (6, 18)
        spacing = (240.0, 100.0)

        forward = astar_search(
            cost, start, end, pixel_spacing=spacing, turn_penalty_weight=0.8)
        backward = astar_search(
            cost, end, start, pixel_spacing=spacing, turn_penalty_weight=0.8)

        self.assertEqual(forward, backward[::-1])
        self._assert_route_legal(forward, cost)

    def test_turn_angle_uses_physical_anisotropic_pixel_vectors(self):
        cost = np.ones((2, 4), dtype=np.float64)
        path = [(1, 1), (0, 2), (0, 3)]
        spacing = (2.0, 1.0)
        turn_weight = 1.0
        diagonal_length = math.sqrt(5.0)
        physical_cosine = 1.0 / diagonal_length
        expected = (diagonal_length + 1.0
                    + 0.5 * (diagonal_length + 1.0) * (1.0 - physical_cosine))

        actual = self._turn_route_cost(path, cost, turn_weight, spacing)

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertNotAlmostEqual(
            actual,
            self._turn_route_cost(path, cost, turn_weight, (1.0, 1.0)),
            places=6,
        )

    def test_turn_penalty_search_does_not_cut_diagonal_obstacle_corners(self):
        cost = np.array([[1.0, np.inf], [1.0, 1.0]])

        path = astar_search(cost, (0, 0), (1, 1), turn_penalty_weight=0.5)

        self.assertEqual(path, [(0, 0), (1, 0), (1, 1)])
        self._assert_route_legal(path, cost)

        diagonal_only = np.array([[1.0, np.inf], [np.inf, 1.0]])
        self.assertIsNone(astar_search(
            diagonal_only, (0, 0), (1, 1), turn_penalty_weight=0.5))

    def test_turn_penalty_weight_must_be_finite_and_nonnegative(self):
        cost = np.ones((3, 3), dtype=np.float64)
        for weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaisesRegex(
                    ValueError, "turn_penalty_weight"):
                astar_search(cost, (0, 0), (2, 2), turn_penalty_weight=weight)

    def test_astar_rejects_negative_costs_for_both_search_modes(self):
        cost = np.ones((3, 3), dtype=np.float64)
        cost[1, 1] = -0.1
        for turn_weight in (0.0, 0.5):
            with self.subTest(turn_weight=turn_weight), self.assertRaisesRegex(
                    ValueError, "cost_raster must be non-negative"):
                astar_search(cost, (0, 0), (2, 2),
                             turn_penalty_weight=turn_weight)

    def test_astar_rejects_invalid_pixel_spacing(self):
        cost = np.ones((3, 3), dtype=np.float64)
        for spacing in ((0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0),
                        (1.0, float("inf")), (1.0,)):
            with self.subTest(spacing=spacing), self.assertRaisesRegex(
                    ValueError, "pixel_spacing"):
                astar_search(cost, (0, 0), (2, 2), pixel_spacing=spacing)

    def test_astar_rejects_invalid_legacy_pixel_size(self):
        cost = np.ones((3, 3), dtype=np.float64)
        for pixel_m in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(pixel_m=pixel_m), self.assertRaisesRegex(
                    ValueError, "pixel_m"):
                astar_search(cost, (0, 0), (2, 2), pixel_m=pixel_m)

    def test_zero_turn_penalty_keeps_legacy_path(self):
        cost = np.array([[1, 2, 2, 4, 1, 1],
                         [1, 3, 2, 2, 1, 1],
                         [1, 1, 1, 4, 2, 1],
                         [3, 1, 2, 1, 1, 1]], dtype=np.float64)

        self.assertEqual(
            astar_search(cost, (0, 0), (3, 5)),
            astar_search(cost, (0, 0), (3, 5), turn_penalty_weight=0.0),
        )

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
        self.assertEqual(report["hard_constraint_validation_scope"], "loaded_raster_masks_only")
        self.assertEqual(report["hard_constraint_completeness_status"], "not_independently_verified")

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

    def test_observed_osm_features_do_not_claim_complete_coverage(self):
        metadata = {
            name: {"status": "available", "coverage_status": "features_observed"}
            for name in ("buildings", "landuse")
        }

        status, missing = screening_data_quality(metadata)

        self.assertEqual(status, "osm_proxy_features_observed_coverage_unverified")
        self.assertEqual(missing, [])
        self.assertEqual(
            route_planning_status([(1.0, 1.0), (2.0, 2.0)], None, missing,
                                  osm_coverage_unverified=True),
            "screening_candidate_coverage_unverified",
        )

    def test_complete_required_osm_coverage_can_be_screening_candidate(self):
        metadata = {
            name: {"status": "available", "coverage_status": "complete"}
            for name in ("buildings", "landuse")
        }

        status, missing = screening_data_quality(metadata)

        self.assertEqual(status, "osm_proxy_coverage_complete")
        self.assertEqual(missing, [])
        self.assertEqual(
            route_planning_status(
                [(1.0, 1.0), (2.0, 2.0)],
                {"endpoint_qualification": "declared_verified_line_connection_points"},
                missing,
                osm_coverage_unverified=False),
            "screening_candidate",
        )

    def test_complete_screening_without_station_task_stays_exploratory(self):
        self.assertEqual(
            route_planning_status(
                [(1.0, 1.0), (2.0, 2.0)], None, [],
                osm_coverage_unverified=False),
            "exploratory_only_unverified_endpoints",
        )

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

    def test_path_objective_integral_matches_astar_and_dijkstra(self):
        from transmission_path_planner.metrics import astar_path_objective, path_cost_integral

        transform = Affine(0.001, 0.0, 120.0, 0.0, -0.001, 25.0)
        cost = np.array([[1.0, 2.0, 1.5, 1.0],
                         [2.0, 1.0, 3.0, 2.0],
                         [1.0, 1.5, 2.0, 1.0],
                         [2.0, 1.0, 1.0, 2.0]])
        dem = np.array([[0.0, 10.0, 20.0, 10.0],
                        [5.0, 15.0, 10.0, 5.0],
                        [0.0, 10.0, 5.0, 0.0],
                        [5.0, 0.0, 0.0, 5.0]])
        start, end = (0, 0), (3, 3)
        spacing = (111.0, 101.0)
        weight = 1.2

        astar_path = astar_search(
            cost, start, end, elevation=dem, grade_penalty_weight=weight,
            pixel_spacing=spacing)
        dijkstra_path = astar_search(
            cost, start, end, use_dijkstra=True, elevation=dem,
            grade_penalty_weight=weight, pixel_spacing=spacing)
        astar = astar_path_objective(astar_path, cost, dem, spacing, weight)
        dijkstra = astar_path_objective(dijkstra_path, cost, dem, spacing, weight)

        self.assertEqual(astar["status"], "measured")
        self.assertAlmostEqual(astar["total_cost"], dijkstra["total_cost"], places=5)
        route = [rowcol_to_geo(row, col, transform) for row, col in astar_path]
        sampled = path_cost_integral(route, dem, transform, cost, weight, 0.0, spacing)
        self.assertAlmostEqual(sampled["base_cost_meter"] + sampled["grade_cost_meter"],
                               astar["total_cost"], places=2)

    def test_continuous_base_cost_integrates_exact_cell_crossing_lengths(self):
        from transmission_path_planner.metrics import path_cost_integral

        transform = Affine(0.001, 0.0, 120.0, 0.0, -0.001, 25.0)
        cost = np.array([[0.0, 100.0], [0.0, 100.0]])
        dem = np.zeros((2, 2), dtype=float)
        route = [transform * (0.1, 0.5), transform * (1.1, 0.5)]

        measured = path_cost_integral(route, dem, transform, cost, 0.0,
                                      pixel_spacing=(100.0, 100.0))

        self.assertEqual(measured["base_cost_meter"], 1000.0)
        self.assertEqual(measured["base_cost_method"], "piecewise_constant_grid_cell_exact")

    def test_continuous_base_cost_splits_simultaneous_row_and_column_crossing(self):
        from transmission_path_planner.metrics import path_cost_integral

        transform = Affine(0.001, 0.0, 120.0, 0.0, -0.001, 25.0)
        cost = np.array([[1.0, 0.0], [0.0, 3.0]])
        dem = np.zeros((2, 2), dtype=float)
        route = [transform * (0.5, 0.5), transform * (1.5, 1.5)]

        forward = path_cost_integral(route, dem, transform, cost, 0.0,
                                     pixel_spacing=(100.0, 100.0))
        reverse = path_cost_integral(route[::-1], dem, transform, cost, 0.0,
                                     pixel_spacing=(100.0, 100.0))

        self.assertAlmostEqual(forward["base_cost_meter"], 2.0 * math.sqrt(2) * 100.0, places=3)
        self.assertEqual(forward["base_cost_meter"], reverse["base_cost_meter"])

    def test_discrete_objective_matches_pixel_m_only_grade_units(self):
        from transmission_path_planner.metrics import astar_path_objective

        measured = astar_path_objective(
            [(0, 0), (0, 1)], np.ones((1, 2)), np.array([[0.0, 10.0]]),
            grade_penalty_weight=1.0, pixel_m=100.0)

        self.assertEqual(measured["unit"], "cost_pixel")
        self.assertAlmostEqual(measured["base_cost"], 1.0)
        self.assertAlmostEqual(measured["grade_cost"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(measured["total_cost"], 4.0 / 3.0, places=6)

    def test_discrete_objective_rejects_diagonal_corner_cutting(self):
        from transmission_path_planner.metrics import astar_path_objective

        cost = np.array([[1.0, np.inf], [np.inf, 1.0]])

        measured = astar_path_objective([(0, 0), (1, 1)], cost)

        self.assertEqual(measured["status"], "corner_cut_path")

if __name__ == "__main__":
    unittest.main()
