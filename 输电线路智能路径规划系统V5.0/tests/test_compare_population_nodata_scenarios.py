import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_population_nodata_scenarios.py"
SPEC = importlib.util.spec_from_file_location("population_nodata_comparator", MODULE_PATH)
comparator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparator)


class PopulationNoDataTripletTests(unittest.TestCase):
    def _write_run(self, root, policy, latitude_offset=0.0):
        run = root / policy
        run.mkdir()
        config = {
            "population_nodata_policy": policy,
            "population_raster": "population.tif",
            "population_weight": 1.0,
            "out_dir": str(run),
            "max_side": 4096,
        }
        provenance = {
            "configuration": config,
            "planning_inputs": [{
                "role": "population_raster", "path": "population.tif",
                "sha256": "a" * 64, "size_bytes": 10, "status": "hashed",
            }],
            "source": {"runtime_source_files": [{
                "path": "planner.py", "sha256": "source-hash", "size_bytes": 20,
            }]},
        }
        (run / "run_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        (run / "run_summary.json").write_text(json.dumps({
            "complete": True, "expected_cases": 1, "completed_cases": 1,
            "failed_cases": [], "no_route_cases": [],
        }), encoding="utf-8")
        feature = {"type": "Feature", "properties": {}, "geometry": {
            "type": "LineString", "coordinates": [[121.0, 24.0 + latitude_offset], [121.01, 24.0 + latitude_offset]],
        }}
        case_dir = run / "case-1"
        case_dir.mkdir()
        for name in ("planned_route_aastar.geojson", "real_route.geojson"):
            (case_dir / name).write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
        (case_dir / "final_route.geojson").write_text(json.dumps({
            "type": "FeatureCollection", "features": [{
                **feature, "properties": {"case": "case-1"},
            }],
        }), encoding="utf-8")
        metrics = [{"case": "case-1", "metrics": {"A*(baseline)": {
            "length_km": 1.0,
            "population_data_coverage": {"coverage_ratio": 0.8, "unknown_route_cell_count": 2,
                                          "cell_fraction_is_approximate": True},
            "route_similarity": {"symmetric": {"p95_m": 10,
                                                  "coverage_f1_within_m": {"100": 0.5}}},
        }}}]
        (run / "comparison_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return run

    def test_pair_metrics_are_projected_and_corridor_scores_are_defined(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            line_a = comparator.transform(comparator.PROJECT.transform,
                                          comparator.shape({"type": "LineString", "coordinates": [[121, 24], [121.01, 24]]}))
            line_b = comparator.transform(comparator.PROJECT.transform,
                                          comparator.shape({"type": "LineString", "coordinates": [[121, 24.001], [121.01, 24.001]]}))
            result = comparator.pair_geometry_metrics(line_a, line_b)
        self.assertGreater(result["mean_m"], 100)
        self.assertGreater(result["p95_m"], 100)
        self.assertLess(result["corridor_f1_100m"], 0.1)
        self.assertGreater(result["corridor_f1_500m"], 0.9)

    def test_triplet_requires_only_policy_and_output_directory_to_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {policy: self._write_run(root, policy) for policy in comparator.POLICIES}
            audit = comparator.validate_triplet(runs, expected_cases=1)
            self.assertEqual(audit["cases"], ["case-1"])
            self.assertEqual(len(audit["planning_input_hashes"]), 1)

            provenance_path = runs["median"] / "run_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["configuration"]["max_side"] = 1024
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration differences"):
                comparator.validate_triplet(runs, expected_cases=1)

    def test_population_coverage_is_labeled_approximate_and_reference_is_posthoc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {policy: self._write_run(root, policy) for policy in comparator.POLICIES}
            rows, pair_rows = comparator.build_rows(comparator.validate_triplet(runs, expected_cases=1))
        row = rows[0]
        self.assertTrue(row["maximum_population_cell_fraction_is_approximate"])
        self.assertIsNotNone(row["maximum_posthoc_reference_p95_m"])
        self.assertEqual(len(pair_rows), 3)

    def test_triplet_rejects_population_without_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {policy: self._write_run(root, policy) for policy in comparator.POLICIES}
            for run in runs.values():
                provenance_path = run / "run_provenance.json"
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                provenance["planning_inputs"][0]["sha256"] = None
                provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be present and SHA-256 hashed"):
                comparator.validate_triplet(runs, expected_cases=1)

    def test_triplet_rejects_different_population_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {policy: self._write_run(root, policy) for policy in comparator.POLICIES}
            provenance_path = runs["zero"] / "run_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["planning_inputs"][0]["sha256"] = "b" * 64
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "planning input hashes"):
                comparator.validate_triplet(runs, expected_cases=1)

    def test_triplet_rejects_duplicate_case_geometry_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {policy: self._write_run(root, policy) for policy in comparator.POLICIES}
            run = runs["median"]
            duplicate = run / "case-duplicate"
            duplicate.mkdir()
            original = run / "case-1"
            for name in ("planned_route_aastar.geojson", "real_route.geojson", "final_route.geojson"):
                (duplicate / name).write_bytes((original / name).read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate planned-route case"):
                comparator.validate_triplet(runs, expected_cases=1)

    def test_compare_copies_run_manifests_and_route_geometries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {policy: self._write_run(root, policy) for policy in comparator.POLICIES}
            output = root / "evidence"
            report = comparator.compare(runs, output, expected_cases=1)
            self.assertTrue((output / "runs/maximum/run_provenance.json").is_file())
            self.assertTrue((output / "routes/median/case-1/planned_route_aastar.geojson").is_file())
            self.assertEqual(len(report["copied_run_artifacts"]), 18)


if __name__ == "__main__":
    unittest.main()
