from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from affine import Affine
from shapely.geometry import LineString, MultiLineString

from transmission_path_planner.cli import (
    _select_final_route,
    expand_single_cell_route,
    geojson_coordinates,
    load_case_reference,
    main,
    resolve_real_path,
    method_visual_similarity_summary,
    run_method_or_skip_on_oom,
    train_vin_or_skip_on_oom,
    validate_endpoint_cells,
    validate_cases_available,
    validate_real_shp_argument,
    validate_unique_case_names,
    configure_console_output,
    summarize_run_completion,
)
from transmission_path_planner.geometry import geo_to_rowcol, rowcol_to_geo
from transmission_path_planner import methods as route_methods
from transmission_path_planner.methods import astar_search, gradient_track_path
from transmission_path_planner.smoothing import path_hard_violations, smooth_path_with_constraints
from transmission_path_planner import metrics


class ReferenceIndependentPlanningTests(unittest.TestCase):
    def test_continuous_tracers_stay_in_degenerate_grid_bounds(self):
        transform = Affine(0.001, 0, 120.0, 0, -0.001, 25.0)
        for shape in ((1, 3), (3, 1)):
            with self.subTest(shape=shape):
                start = (0, 0)
                end = (shape[0] - 1, shape[1] - 1)
                cost = np.zeros(shape, dtype=np.float32)
                hard_mask = np.ones(shape, dtype=np.uint8)
                V = np.zeros(shape, dtype=np.float32)

                vin_route = gradient_track_path(
                    V, cost, hard_mask, start, end, transform, max_steps=8, eps=0.25
                )
                geo_routes = [smooth_path_with_constraints(vin_route, transform, hard_mask)]
                if route_methods.HAS_TORCH:
                    ppo_route = route_methods.ppo_rollout_path(
                        None, None, cost, hard_mask, end, start,
                        max_steps=8, n_tries=1,
                    )
                    ppo_geo = [rowcol_to_geo(row, col, transform) for row, col in ppo_route]
                    geo_routes.append(smooth_path_with_constraints(ppo_geo, transform, hard_mask))

                for route in geo_routes:
                    self.assertEqual(path_hard_violations(route, transform, hard_mask), 0)
                    for lon, lat in route:
                        col_corner, row_corner = ~transform * (lon, lat)
                        row, col = row_corner - 0.5, col_corner - 0.5
                        self.assertGreaterEqual(row, 0.0)
                        self.assertLessEqual(row, shape[0] - 1)
                        self.assertGreaterEqual(col, 0.0)
                        self.assertLessEqual(col, shape[1] - 1)

    @unittest.skipUnless(route_methods.HAS_TORCH, "torch unavailable")
    def test_vin_preserves_single_row_and_column_value_map_shape(self):
        for shape in ((1, 1), (1, 3), (3, 1)):
            with self.subTest(shape=shape):
                goal = np.zeros(shape, dtype=np.float32)
                goal[-1, -1] = 1.0
                cost = np.ones(shape, dtype=np.float32)
                hard_mask = np.ones(shape, dtype=np.uint8)
                extra = np.zeros((*shape, 1), dtype=np.float32)

                value_map = route_methods.train_vin(
                    cost, goal, extra, hard_mask, n_steps=1
                )

                self.assertEqual(value_map.shape, shape)
                self.assertTrue(np.isfinite(value_map).all())

    def test_console_output_replaces_symbols_unavailable_in_gbk(self):
        buffer = BytesIO()
        stdout = TextIOWrapper(buffer, encoding="gbk", errors="strict")
        with patch("sys.stdout", stdout), patch("sys.stderr", StringIO()):
            configure_console_output()
            print("ℹ")
            stdout.flush()
        self.assertIn("?", buffer.getvalue().decode("gbk"))

    def test_final_route_selection_ignores_reference_errors(self):
        methods = {"VIN-Grad": [(1, 1)], "A*(baseline)": [(1, 1)]}
        for vin_error, astar_error in ((1.0, 9.0), (99.0, 0.1)):
            def reference_score(value):
                return {"route_similarity": {
                    "symmetric": {"p95_m": value},
                    "length": {"absolute_difference_pct": value},
                }, "hard_violations": 0}
            metrics = {
                "VIN-Grad": reference_score(vin_error),
                "A*(baseline)": reference_score(astar_error),
            }
            self.assertEqual(
                _select_final_route(methods, metrics),
                ("A*(baseline)", None, "astar_feasible_priority"),
            )

    def test_final_route_falls_back_to_valid_candidate_if_astar_fails(self):
        methods = {"VIN-Grad": [(1, 1)], "A*(baseline)": None}
        metrics = {"VIN-Grad": {"hard_violations": 0}}

        self.assertEqual(
            _select_final_route(methods, metrics),
            ("VIN-Grad", None, "valid_candidate_fallback"),
        )

    def test_final_route_fallback_order_does_not_depend_on_mapping_order(self):
        methods = {"PPO": [(1, 1)], "VIN-Grad": [(1, 1)], "A*(baseline)": None}
        metrics = {
            "PPO": {"hard_violations": 0},
            "VIN-Grad": {"hard_violations": 0},
        }

        self.assertEqual(
            _select_final_route(methods, metrics),
            ("VIN-Grad", None, "valid_candidate_fallback"),
        )

    def test_final_route_uses_ppo_when_vin_is_unavailable(self):
        methods = {"PPO": [(1, 1)], "VIN-Grad": None, "A*(baseline)": None}
        metrics = {"PPO": {"hard_violations": 0}}

        self.assertEqual(
            _select_final_route(methods, metrics),
            ("PPO", None, "valid_candidate_fallback"),
        )

    def test_cli_does_not_expose_reference_guidance_options(self):
        output = StringIO()
        with patch("sys.argv", ["planner", "--help"]), redirect_stdout(output):
            with self.assertRaises(SystemExit):
                main()

        help_text = output.getvalue()
        for removed_option in (
            "--guide-mode", "--corridor-w", "--corridor-band",
            "--real-attract", "--real-attract-cutoff", "--baseline",
            "--shp", "--existing-network",
        ):
            self.assertNotIn(removed_option, help_text)
        self.assertIn("--no-vin", help_text)

    def test_geojson_coordinates_keep_longitude_before_latitude(self):
        self.assertEqual(geojson_coordinates([(120.75, 24.32)]), [[120.75, 24.32]])

    def test_smoothed_shortcut_falls_back_around_hard_obstacle(self):
        transform = Affine(0.0001, 0, 0, 0, -0.0001, 0.001)
        hard_mask = np.ones((12, 12), dtype=np.uint8)
        hard_mask[5, 5] = 0
        raw_route = [rowcol_to_geo(r, c, transform) for r, c in [(5, 3), (4, 4), (4, 6), (5, 7)]]

        result = smooth_path_with_constraints(raw_route, transform, hard_mask)

        self.assertEqual(result, raw_route)
        self.assertEqual(path_hard_violations(result, transform, hard_mask), 0)

    def test_geo_coordinate_round_trips_to_its_pixel(self):
        transform = Affine(0.0001, 0, 120.0, 0, -0.0001, 25.0)
        for row in range(6):
            for col in range(6):
                lon, lat = rowcol_to_geo(row, col, transform)
                self.assertEqual(geo_to_rowcol(lat, lon, transform), (row, col))

    def test_astar_cost_matches_dijkstra_on_variable_cost_grid(self):
        cost = np.full((18, 18), 8.0, dtype=np.float64)
        cost[2:16, 2:16] = 1.0
        cost[8, 2:16] = 0.01
        cost[8, 9] = 50.0
        cost[3:7, 6] = np.inf
        start, goal = (1, 1), (16, 16)
        path = astar_search(cost, start, goal)
        reference = astar_search(cost, start, goal, use_dijkstra=True)

        def path_cost(route):
            return sum(cost[r, c] * (np.sqrt(2) if r != pr and c != pc else 1.0)
                       for (pr, pc), (r, c) in zip(route, route[1:]))

        self.assertTrue(path)
        self.assertAlmostEqual(path_cost(path), path_cost(reference), places=9)

    def test_fixed_endpoints_must_be_inside_complete_hard_mask(self):
        hard_mask = np.ones((5, 5), dtype=np.uint8)
        hard_mask[2, 2] = 0
        hard_mask[4, 4] = 0

        self.assertEqual(
            validate_endpoint_cells((0, 0), (3, 3), hard_mask.shape, hard_mask),
            ((0, 0), (3, 3)),
        )
        before = hard_mask.copy()
        with self.assertRaisesRegex(ValueError, "禁区"):
            validate_endpoint_cells((2, 2), (4, 4), hard_mask.shape, hard_mask)
        with self.assertRaisesRegex(ValueError, "超出DEM范围"):
            validate_endpoint_cells((-1, 2), (4, 4), hard_mask.shape, hard_mask)
        for out_of_bounds in ((5, 0), (0, 5)):
            with self.assertRaisesRegex(ValueError, "超出DEM范围"):
                validate_endpoint_cells((0, 0), out_of_bounds, hard_mask.shape, hard_mask)
        with self.assertRaisesRegex(ValueError, "禁区"):
            validate_endpoint_cells((0, 0), (4, 4), hard_mask.shape, hard_mask)
        np.testing.assert_array_equal(hard_mask, before)

    def test_vin_out_of_memory_skips_only_vin(self):
        class FakeOutOfMemoryError(RuntimeError):
            pass

        clear_cache = Mock()
        fake_torch = SimpleNamespace(
            OutOfMemoryError=FakeOutOfMemoryError,
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=clear_cache),
        )
        with patch("transmission_path_planner.cli.torch", fake_torch), \
             patch("transmission_path_planner.cli.train_vin",
                   side_effect=FakeOutOfMemoryError("memory full")):
            with redirect_stdout(StringIO()):
                self.assertIsNone(train_vin_or_skip_on_oom())
        clear_cache.assert_called_once_with()

    def test_vin_unrelated_runtime_error_is_not_suppressed(self):
        clear_cache = Mock()
        fake_torch = SimpleNamespace(
            OutOfMemoryError=type("FakeOutOfMemoryError", (RuntimeError,), {}),
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=clear_cache),
        )
        with patch("transmission_path_planner.cli.torch", fake_torch), \
             patch("transmission_path_planner.cli.train_vin",
                   side_effect=RuntimeError("unexpected failure")):
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                train_vin_or_skip_on_oom()
        clear_cache.assert_not_called()

    def test_non_oom_method_error_is_not_suppressed(self):
        fake_torch = SimpleNamespace(
            OutOfMemoryError=type("FakeOutOfMemoryError", (RuntimeError,), {}),
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()),
        )
        with patch("transmission_path_planner.cli.torch", fake_torch):
            with self.assertRaisesRegex(ValueError, "regression"):
                run_method_or_skip_on_oom("PPO", lambda: (_ for _ in ()).throw(ValueError("regression")))

    def test_real_shp_requires_one_existing_target_route_file(self):
        case = ("route-a", (24.0, 121.0), (24.1, 121.1))
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            validate_real_shp_argument(str(route_file), [case])
            with self.assertRaisesRegex(ValueError, "单任务"):
                validate_real_shp_argument(str(route_file), [case, case])
            with self.assertRaisesRegex(ValueError, "存在"):
                validate_real_shp_argument(str(route_file / "missing"), [case])

    def test_duplicate_case_names_are_rejected(self):
        cases = [
            ("route/a", (24.0, 121.0), (24.1, 121.1)),
            ("route/a", (24.2, 121.2), (24.3, 121.3)),
        ]
        with self.assertRaisesRegex(ValueError, "必须唯一"):
            validate_unique_case_names(cases)

    def test_sanitized_reference_filename_collisions_are_rejected(self):
        cases = [
            ("R/1", (24.0, 121.0), (24.1, 121.1)),
            ("R?1", (24.2, 121.2), (24.3, 121.3)),
        ]
        validate_unique_case_names(cases)
        with self.assertRaisesRegex(ValueError, "净化后发生重复"):
            validate_unique_case_names(cases, check_reference_names=True)

    def test_reference_filename_collisions_ignore_case(self):
        cases = [
            ("RouteA", (24.0, 121.0), (24.1, 121.1)),
            ("routea", (24.2, 121.2), (24.3, 121.3)),
        ]
        with self.assertRaisesRegex(ValueError, "净化后发生重复"):
            validate_unique_case_names(cases, check_reference_names=True)

    def test_empty_task_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未找到有效规划任务"):
            validate_cases_available([])

    def test_conflicting_task_sources_are_rejected_before_planning(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "cases.csv"
            csv_path.write_text(
                "name,start_lat,start_lon,end_lat,end_lon\nroute,24,121,24.1,121.1\n",
                encoding="utf-8")
            cases = [
                ["--start", "24,121", "--end", "24.1,121.1",
                 "--real-cases-csv", str(csv_path)],
                ["--stations-csv", str(Path(temp_dir) / "stations.csv"),
                 "--route-tasks-csv", str(Path(temp_dir) / "tasks.csv"),
                 "--real-cases-csv", str(csv_path)],
            ]
            for options in cases:
                output = StringIO()
                with self.subTest(options=options), \
                     patch("sys.argv", ["planner", *options]), \
                     redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as raised:
                        main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("任务输入来源只能选一种", output.getvalue())

    def test_invalid_direct_coordinates_and_grid_size_are_rejected(self):
        scenarios = [
            (["--max-side", "0"], "--max-side 必须是正整数"),
            (["--max-ground-grade", "nan"], "必须是有限数值"),
            (["--grade-penalty-weight", "inf"], "必须是有限数值"),
            (["--max-terrain-elevation-m", "nan"], "必须是有限数值"),
            (["--start", "91,121", "--end", "24,122"], "纬度须在[-90,90]"),
            (["--start", "nan,121", "--end", "24,122"], "经纬度无效"),
            (["--start", "24,121,0", "--end", "24,122"], "经纬度无效"),
        ]
        with TemporaryDirectory() as temp_dir:
            for index, (options, expected_message) in enumerate(scenarios):
                output = StringIO()
                with self.subTest(options=options), \
                     patch("sys.argv", ["planner", "--out-dir",
                                         str(Path(temp_dir) / str(index)), *options]), \
                     redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as raised:
                        main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected_message, output.getvalue())

    def test_single_cell_route_keeps_distinct_exact_endpoints(self):
        path = [(121.0, 24.0)]

        result = expand_single_cell_route(path, (24.0001, 121.0001), (24.0002, 121.0002))

        self.assertEqual(result, [(121.0001, 24.0001), (121.0002, 24.0002)])
        with self.assertRaisesRegex(ValueError, "起终点坐标相同"):
            expand_single_cell_route(path, (24.0, 121.0), (24.0, 121.0))

    def test_zero_length_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "起终点坐标相同"):
            validate_cases_available([("same", (24.0, 121.0), (24.0, 121.0))])

    def test_empty_real_cases_csv_is_rejected_before_planning(self):
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "empty.csv"
            csv_path.write_text("name,start_lat,start_lon,end_lat,end_lon\n", encoding="utf-8")
            output = StringIO()
            with patch("sys.argv", ["planner", "--real-cases-csv", str(csv_path)]), \
                 redirect_stdout(output), redirect_stderr(output):
                with self.assertRaises(SystemExit) as raised:
                    main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("未找到有效规划任务", output.getvalue())

    def test_no_route_does_not_count_as_completed_batch(self):
        results = [{"case": "blocked", "metrics": {"最终路线": {
            "planning_status": "no_hard_constraint_compliant_route"}}}]
        summary = summarize_run_completion(1, results, [])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_cases"], 0)
        self.assertEqual(summary["no_route_cases"], ["blocked"])

    def test_no_route_and_failed_cases_are_unmeasured(self):
        records = [
            {"case": "blocked", "metrics": {
                "VIN-Grad": {"route_similarity": {"symmetric": {"p95_m": 1}}},
                "最终路线": {"planning_status": "no_hard_constraint_compliant_route"},
            }},
            {"case": "failed-after-record", "metrics": {
                "VIN-Grad": {"route_similarity": {"symmetric": {"p95_m": 20}}},
                "最终路线": {"planning_status": "screening_candidate"},
            }},
        ]
        self.assertEqual(
            method_visual_similarity_summary(records, "VIN-Grad", expected_cases=2,
                                             failed_case_names={"failed-after-record"}),
            {"measured": 0, "unmeasured": 2, "median_symmetric_p95_m": None,
             "median_sampled_hausdorff_m": None, "median_coverage_f1_100m": None,
             "median_coverage_f1_1km": None, "routes_f1_100m_at_least_95pct": 0,
             "median_length_difference_pct": None},
        )

    def test_mixed_valid_and_invalid_real_cases_csv_is_rejected(self):
        csv_text = (
            "name,start_lat,start_lon,end_lat,end_lon\n"
            "valid,24.0,121.0,24.1,121.1\n"
            "broken,not-a-number,121.0,24.2,121.2\n"
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "第 3 行"):
                metrics.load_real_cases_csv(path)

    def test_case_reference_read_error_does_not_fail_planning_case(self):
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", side_effect=OSError("bad SHP")):
                with redirect_stdout(StringIO()):
                    route, status = load_case_reference(route_file)
        self.assertIsNone(route)
        self.assertEqual(status, "reference_invalid_or_unreadable")

    def test_missing_geopandas_is_reported_as_unreadable_reference(self):
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", False), redirect_stdout(StringIO()):
                route, status = load_case_reference(route_file)
        self.assertIsNone(route)
        self.assertEqual(status, "reference_invalid_or_unreadable")

    def test_reader_value_error_is_reported_as_unreadable_reference(self):
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", side_effect=ValueError("decoder failure")), \
                 redirect_stdout(StringIO()):
                route, status = load_case_reference(route_file)
        self.assertIsNone(route)
        self.assertEqual(status, "reference_invalid_or_unreadable")

    def test_reference_coordinates_outside_wgs84_range_are_invalid(self):
        line = LineString([(200.0, 24.0), (201.0, 24.1)])
        frame = SimpleNamespace(crs="EPSG:4326", geometry=[line])
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                with self.assertRaisesRegex(ValueError, "无效经纬度"):
                    metrics.load_real_path(route_file, require_single_line=True)
                with redirect_stdout(StringIO()):
                    route, status = load_case_reference(route_file, require_single_line=True)
        self.assertIsNone(route)
        self.assertEqual(status, "reference_invalid_or_empty")

    def test_local_case_reference_precedes_global_name_mapping(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapped_dir = root / "mapped"
            local_dir = root / "local"
            mapped_dir.mkdir()
            local_dir.mkdir()
            mapped_path = mapped_dir / "route.shp"
            local_path = local_dir / "RouteA_real_path.shp"
            mapped_path.touch()
            local_path.touch()
            with patch("transmission_path_planner.io_export.REAL_PATH_MAP", {"RouteA": "route.shp"}), \
                 patch("transmission_path_planner.io_export.REAL_PATH_DIR", mapped_dir):
                self.assertEqual(resolve_real_path("RouteA", local_dir), local_path)

    def test_custom_reference_directory_does_not_fall_back_to_global_mapping(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mapped_dir = root / "mapped"
            custom_dir = root / "custom"
            mapped_dir.mkdir()
            custom_dir.mkdir()
            (mapped_dir / "route.shp").touch()
            with patch("transmission_path_planner.io_export.REAL_PATH_MAP", {"RouteA": "route.shp"}), \
                 patch("transmission_path_planner.io_export.REAL_PATH_DIR", mapped_dir):
                self.assertIsNone(resolve_real_path(
                    "RouteA", custom_dir, include_global_mapping=False))

    def test_ambiguous_recursive_reference_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one").mkdir()
            (root / "two").mkdir()
            (root / "one" / "RouteA_real_path.shp").touch()
            (root / "two" / "RouteA_real_path.shp").touch()
            with patch("transmission_path_planner.io_export.REAL_PATH_MAP", {}):
                with self.assertRaisesRegex(ValueError, "匹配到多个文件"):
                    resolve_real_path("RouteA", root)

    def test_reference_without_crs_is_invalid_not_unreadable(self):
        line = LineString([(121.0, 24.0), (121.1, 24.1)])
        frame = SimpleNamespace(crs=None, geometry=[line])
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                with redirect_stdout(StringIO()):
                    route, status = load_case_reference(route_file)
        self.assertIsNone(route)
        self.assertEqual(status, "reference_invalid_or_empty")

    def test_invalid_direct_reference_is_reported_without_raising(self):
        line = LineString([(121.0, 24.0), (121.1, 24.1)])
        frame = SimpleNamespace(crs="EPSG:4326", geometry=[line, line])
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                with redirect_stdout(StringIO()):
                    route, status = load_case_reference(
                        route_file, require_single_line=True)
        self.assertIsNone(route)
        self.assertEqual(status, "reference_invalid_or_empty")

    def test_3d_reference_lines_use_horizontal_coordinates(self):
        line = LineString([(121.0, 24.0, 100.0), (121.1, 24.1, 200.0)])
        frame = SimpleNamespace(crs="EPSG:4326", geometry=[line])
        with TemporaryDirectory() as temp_dir:
            route_file = Path(temp_dir) / "route.shp"
            route_file.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                self.assertEqual(
                    metrics.load_real_path(route_file, require_single_line=True),
                    [[(121.0, 24.0), (121.1, 24.1)]],
                )

    def test_real_cases_csv_rejects_unheaded_extra_fields(self):
        csv_text = (
            "name,start_lat,start_lon,end_lat,end_lon\n"
            "valid,24.0,121.0,24.1,121.1,unexpected\n"
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "字段数多于表头"):
                metrics.load_real_cases_csv(path)

    def test_real_cases_csv_rejects_duplicate_headers(self):
        csv_text = (
            "name,start_lat,start_lon,end_lat,end_lon,end_lon\n"
            "route,24,121,24.1,121.1,0\n"
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.csv"
            path.write_text(csv_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复列名"):
                metrics.load_real_cases_csv(path)

    def test_missing_method_rows_are_reported_as_unmeasured(self):
        route_similarity = {
            "symmetric": {"p95_m": 120, "sampled_hausdorff_m": 400,
                          "coverage_f1_within_m": {"1000": 0.75}},
            "length": {"absolute_difference_pct": 5.0},
        }
        records = [
            {"case": "route-a", "metrics": {"A*(baseline)": {"route_similarity": route_similarity}}},
            {"case": "route-b", "metrics": {}},
        ]
        self.assertEqual(
            method_visual_similarity_summary(records, "VIN-Grad"),
            {"measured": 0, "unmeasured": 2, "median_symmetric_p95_m": None,
             "median_sampled_hausdorff_m": None, "median_coverage_f1_100m": None,
             "median_coverage_f1_1km": None, "routes_f1_100m_at_least_95pct": 0,
             "median_length_difference_pct": None},
        )
        self.assertEqual(
            method_visual_similarity_summary(records, "A*(baseline)", expected_cases=3),
            {"measured": 1, "unmeasured": 2, "median_symmetric_p95_m": 120.0,
             "median_sampled_hausdorff_m": 400.0, "median_coverage_f1_100m": None,
             "median_coverage_f1_1km": 0.75, "routes_f1_100m_at_least_95pct": 0,
             "median_length_difference_pct": 5.0},
        )

    def test_direct_real_shp_rejects_multiple_line_features(self):
        line = LineString([(121.0, 24.0), (121.1, 24.1)])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route.shp"
            path.touch()
            single_frame = SimpleNamespace(crs="EPSG:4326", geometry=[line])
            multi_frame = SimpleNamespace(crs="EPSG:4326", geometry=[line, line])
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=single_frame):
                self.assertEqual(
                    metrics.load_real_path(path, require_single_line=True),
                    [[(121.0, 24.0), (121.1, 24.1)]],
                )
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=multi_frame):
                with self.assertRaisesRegex(ValueError, "恰含一个 LineString"):
                    metrics.load_real_path(path, require_single_line=True)

    def test_real_dir_loader_accepts_segmented_route_but_direct_input_rejects_it(self):
        lines = MultiLineString([
            [(121.0, 24.0), (121.1, 24.1)],
            [(121.1, 24.1), (121.2, 24.2)],
        ])
        frame = SimpleNamespace(crs="EPSG:4326", geometry=[lines])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route.shp"
            path.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                self.assertEqual(len(metrics.load_real_path(path)), 2)
                with self.assertRaisesRegex(ValueError, "恰含一个 LineString"):
                    metrics.load_real_path(path, require_single_line=True)

    def test_reference_without_crs_is_not_treated_as_wgs84(self):
        frame = SimpleNamespace(crs=None, geometry=[LineString([(121.0, 24.0), (121.1, 24.1)])])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route.shp"
            path.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                with self.assertRaisesRegex(ValueError, "缺少 CRS"):
                    metrics.load_real_path(path, require_single_line=True)
                self.assertIsNone(metrics.load_real_path(path))

    def test_zero_length_reference_line_is_rejected(self):
        frame = SimpleNamespace(crs="EPSG:4326", geometry=[LineString([(121.0, 24.0), (121.0, 24.0)])])
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "route.shp"
            path.touch()
            with patch.object(metrics, "HAS_GPD", True), \
                 patch.object(metrics.gpd, "read_file", return_value=frame):
                with self.assertRaisesRegex(ValueError, "长度必须大于零"):
                    metrics.load_real_path(path, require_single_line=True)
                self.assertIsNone(metrics.load_real_path(path))


if __name__ == "__main__":
    unittest.main()
