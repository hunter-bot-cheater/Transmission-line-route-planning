import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from affine import Affine

from transmission_path_planner.cost import build_route_cost
from transmission_path_planner.cli import attach_station_endpoints
from transmission_path_planner.geometry import rowcol_to_geo
from transmission_path_planner.methods import astar_search
from transmission_path_planner.metrics import route_profile_metrics, write_elevation_profile_csv
from transmission_path_planner.smoothing import filter_path_by_ground_grade, sample_dem, smooth_path_with_constraints
from transmission_path_planner.stations import load_station_tasks


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
                                 ["station_id", "name", "lat", "lon", "voltage_kv"],
                                 [["S1", "North", 24.1, 120.1, 220], ["S2", "South", 24.2, 120.2, 220]])
            tasks = self._csv(folder, "routes.csv",
                              ["route_id", "from_station_id", "to_station_id"], [["R1", "S1", "S2"]])
            result = load_station_tasks(stations, tasks)
        self.assertEqual(result[0]["route_id"], "R1")
        self.assertEqual(result[0]["from_station"]["station_id"], "S1")
        self.assertEqual(result[0]["voltage_kv"], "220")

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
            return sum(cost[r, c] * np.hypot((r - pr) * spacing[0], (c - pc) * spacing[1])
                       for (pr, pc), (r, c) in zip(path, path[1:]))

        self.assertAlmostEqual(physical_cost(astar), physical_cost(dijkstra))

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


if __name__ == "__main__":
    unittest.main()
