import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shapely.geometry import LineString


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "compare_constraint_ab_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_constraint_ab_runs", SCRIPT_PATH)
comparator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparator)


class ConstraintReportTests(unittest.TestCase):
    def _write(self, temporary, records, expected_case_count=1):
        root = Path(temporary)
        dem = root / "dem.tif"
        constraints = root / "constraints.geojson"
        dem.write_bytes(b"dem")
        constraints.write_bytes(b"constraints")
        report = root / "report.md"
        comparator._write_report(
            records, report, {"drinking_water_protection": 1},
            {"max_side": 4096, "window_margin_deg": 0.3},
            dem, constraints, root / "baseline", root / "treatment",
            expected_case_count,
            {"baseline_commit": "base", "treatment_commit": "treatment",
             "excluded_source_file_differences": []})
        return report.read_text(encoding="utf-8")

    def test_report_uses_current_grid_and_run_paths_without_stale_experiment_claims(self):
        record = {
            "case": "case-a", "status_without": "route", "status_with": "route",
            "length_without_km": 12.0, "length_with_km": 12.5,
            "length_delta_km": 0.5, "raster_soft_cells_without": 4,
            "route_geometry_identical": False,
            "route_vertex_hausdorff_m": 300.0,
            "raster_soft_cells_with": 2, "vector_overlap_km_without": 1.0,
            "vector_overlap_km_with": 0.5, "symmetric_p95_m_without": 2000.0,
            "symmetric_p95_m_with": 2500.0, "f1_100m_without": 0.1,
            "f1_100m_with": 0.08, "pixel_size_m": 35.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = self._write(temporary, [record])

        self.assertIn("`4096`", report)
        self.assertIn("`35.0 to 35.0 m`", report)
        self.assertIn("Baseline run:", report)
        self.assertIn("Treatment run:", report)
        self.assertIn("Routes with changed geometry: `1/1`", report)
        self.assertIn("Vertex Hausdorff (m)", report)
        for stale in ("1024-cell cap", "10.1 km", "seven successful pairs",
                      "--max-side 1024", "8cases_1024"):
            self.assertNotIn(stale, report)

    def test_only_disabled_tower_graph_source_may_differ(self):
        tower_path = (
            "输电线路智能路径规划系统V5.0/transmission_path_planner/"
            "experimental_tower_graph.py")
        common = {"runtime_source_files": [
            {"path": "transmission_path_planner/methods.py",
             "size_bytes": 4, "sha256": "same"},
            {"path": tower_path, "size_bytes": 4, "sha256": "base"}],
            "commit": "base"}
        treatment = {"runtime_source_files": [
            {"path": "transmission_path_planner/methods.py",
             "size_bytes": 4, "sha256": "same"},
            {"path": tower_path, "size_bytes": 4, "sha256": "new"}],
            "commit": "treatment"}
        config = {"experimental_tower_graph": False,
                  "select_experimental_tower_graph": False}
        comparison = comparator._compare_effective_source(
            {"source": common, "configuration": config},
            {"source": treatment, "configuration": config})
        self.assertEqual(comparison["excluded_source_file_differences"],
                         [tower_path])

        treatment["runtime_source_files"][0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "active runtime source"):
            comparator._compare_effective_source(
                {"source": common, "configuration": config},
                {"source": treatment, "configuration": config})

    def test_route_map_allocates_an_axis_for_each_case(self):
        line_maps = [
            (f"case-{index}", LineString([(index, 0), (index, 1)]),
             LineString([(index, 0.1), (index, 1.1)]), 1.0, 1.1)
            for index in range(12)]

        def inspect_figure(figure, _path, **_kwargs):
            self.assertEqual(len(figure.axes), 12)

        before = set(comparator.plt.get_fignums())
        with patch("matplotlib.figure.Figure.savefig", new=inspect_figure):
            comparator._write_map(line_maps, {}, "unused.png")
        self.assertEqual(set(comparator.plt.get_fignums()), before)

    def test_incomplete_pairs_do_not_receive_pooled_route_statistics(self):
        record = {"case": "case-a", "status_without": "route",
                  "status_with": "failed", "failure_reason": "not reachable"}
        with tempfile.TemporaryDirectory() as temporary:
            report = self._write(temporary, [record], expected_case_count=2)

        self.assertIn("Pooled medians withheld", report)
        self.assertIn("0 paired routes", report)
        self.assertIn("failed", report)


if __name__ == "__main__":
    unittest.main()
