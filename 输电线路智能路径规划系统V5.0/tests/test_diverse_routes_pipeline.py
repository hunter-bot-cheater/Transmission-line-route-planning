import unittest

import numpy as np
from affine import Affine

from transmission_path_planner.cli import (
    attach_station_endpoints,
    validate_route_hard_constraints,
)
from transmission_path_planner.diverse_routes import diverse_astar_routes
from transmission_path_planner.geometry import pixel_spacing_meters, rowcol_to_geo
from transmission_path_planner.methods import astar_search
from transmission_path_planner.smoothing import (
    filter_path_by_ground_grade,
    smooth_path_with_constraints,
)


class DiverseRoutePipelineTests(unittest.TestCase):
    def setUp(self):
        self.transform = Affine(0.001, 0, 119.9, 0, -0.001, 25.0)
        self.spacing = pixel_spacing_meters(self.transform, 24.99, "EPSG:4326")
        self.dem = np.zeros((21, 31), dtype=np.float64)
        self.start = (10, 0)
        self.end = (10, 30)
        start_lon, start_lat = rowcol_to_geo(*self.start, self.transform)
        end_lon, end_lat = rowcol_to_geo(*self.end, self.transform)
        self.station_task = {
            "from_station": {"lon": start_lon + 0.00005, "lat": start_lat + 0.00005},
            "to_station": {"lon": end_lon - 0.00005, "lat": end_lat - 0.00005},
        }

    def _finalize_candidate(self, route, cost, hard):
        geo = [rowcol_to_geo(row, col, self.transform)
               for row, col in route["path"]]
        geo = smooth_path_with_constraints(
            geo, self.transform, hard, cost, self.dem,
            grade_penalty_weight=0.6, turn_penalty_weight=0.4,
            pixel_spacing=self.spacing,
        )
        self.assertIsNotNone(geo)
        geo, grade_report = filter_path_by_ground_grade(
            geo, self.dem, self.transform, 0.05, "EPSG:4326")
        self.assertEqual(grade_report["grade_limit_status"], "pass")
        geo, _ = attach_station_endpoints(geo, self.station_task)
        geo, grade_report = filter_path_by_ground_grade(
            geo, self.dem, self.transform, 0.05, "EPSG:4326")
        self.assertEqual(grade_report["grade_limit_status"], "pass")
        accepted, hard_report = validate_route_hard_constraints(
            geo, self.transform, hard)
        self.assertEqual(hard_report["hard_constraint_status"], "pass")
        self.assertEqual(hard_report["hard_violations"], 0)
        self.assertEqual(accepted[0], (self.station_task["from_station"]["lon"],
                                       self.station_task["from_station"]["lat"]))
        self.assertEqual(accepted[-1], (self.station_task["to_station"]["lon"],
                                        self.station_task["to_station"]["lat"]))
        return accepted

    def test_candidates_survive_smoothing_and_station_endpoint_validation(self):
        cost = np.full(self.dem.shape, 30.0)
        hard = np.ones(self.dem.shape, dtype=bool)
        for row in (4, 16):
            cost[row, :] = 0.2
            cost[min(row, 10):max(row, 10) + 1, 0] = 0.2
            cost[min(row, 10):max(row, 10) + 1, -1] = 0.2
        cost[10, 0] = cost[10, -1] = 0.2
        kwargs = {
            "elevation": self.dem,
            "pixel_spacing": self.spacing,
            "grade_penalty_weight": 0.6,
            "max_ground_grade": 0.05,
            "heuristic_weight": 1.0,
            "turn_penalty_weight": 0.4,
        }
        expected_first = astar_search(
            np.where(hard, cost, np.inf), self.start, self.end, **kwargs)

        candidates = diverse_astar_routes(
            cost, hard, self.start, self.end, max_paths=3,
            penalty_radius_m=4 * max(self.spacing),
            distance_penalty_weight=40,
            minimum_separation_m=5 * min(self.spacing),
            endpoint_buffer_m=2 * max(self.spacing),
            **kwargs,
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["path"], expected_first)
        processed = [self._finalize_candidate(candidate, cost, hard)
                     for candidate in candidates]
        self.assertEqual(len(processed), len(candidates))

    def test_single_passable_corridor_returns_one_legal_final_route(self):
        cost = np.full(self.dem.shape, np.inf)
        hard = np.zeros(self.dem.shape, dtype=bool)
        cost[10, :] = 1.0
        hard[10, :] = True
        candidates = diverse_astar_routes(
            cost, hard, self.start, self.end, max_paths=4,
            pixel_spacing=self.spacing,
        )

        self.assertEqual(len(candidates), 1)
        processed = [self._finalize_candidate(candidate, cost, hard)
                     for candidate in candidates]
        self.assertEqual(len(processed), 1)


if __name__ == "__main__":
    unittest.main()
