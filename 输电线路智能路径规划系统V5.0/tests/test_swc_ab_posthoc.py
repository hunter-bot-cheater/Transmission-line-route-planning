import sys
import unittest
from pathlib import Path


SCORER_DIR = (Path(__file__).resolve().parents[1] / "docs" / "evidence"
              / "swc_zones_soft_sensitivity_20261030_01")
sys.path.insert(0, str(SCORER_DIR))

from score_swc_ab_posthoc import (  # noqa: E402
    compare_planning_input_provenance,
    planning_argument_comparison,
)


def input_record(role, path, digest):
    return {
        "role": role,
        "path": path,
        "status": "hashed",
        "size_bytes": 10,
        "sha256": digest * 64,
    }


class SWCPosthocGateTests(unittest.TestCase):
    def setUp(self):
        self.control = {"planning_inputs": [
            input_record("dem", "/data/dem.tif", "a"),
            input_record("land_mask", "/data/land.geojson", "b"),
            input_record("user_constraints", "/data/base.geojson", "c"),
        ]}
        self.treatment = {"planning_inputs": [
            input_record("dem", "/data/dem.tif", "a"),
            input_record("land_mask", "/data/land.geojson", "b"),
            input_record("user_constraints", "/data/swc.geojson", "d"),
        ]}

    def test_only_constraint_input_may_change(self):
        result = compare_planning_input_provenance(self.control, self.treatment)
        self.assertTrue(result["all_non_constraint_planning_inputs_match"])
        self.assertEqual(result["changed_input_role"], "user_constraints")
        self.assertEqual(result["shared_input_count"], 2)

    def test_rejects_changed_hazard_or_dem_input(self):
        self.treatment["planning_inputs"][0]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "outside the intended constraint layer"):
            compare_planning_input_provenance(self.control, self.treatment)

    def test_rejects_unhashed_input(self):
        self.treatment["planning_inputs"][0]["status"] = "missing"
        with self.assertRaisesRegex(ValueError, "unhashed planning input"):
            compare_planning_input_provenance(self.control, self.treatment)

    def test_rejects_unchanged_constraint(self):
        self.treatment["planning_inputs"][2]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "did not change"):
            compare_planning_input_provenance(self.control, self.treatment)

    def test_hazard_path_is_not_normalized_away(self):
        control = ["--out-dir", "/runs/control", "--constraint-geojson", "/data/base",
                   "--official-hazards-dir", "/data/hazards-a"]
        treatment = ["--out-dir", "/runs/treatment", "--constraint-geojson", "/data/swc",
                     "--official-hazards-dir", "/data/hazards-b"]
        result = planning_argument_comparison(control, treatment)
        self.assertFalse(result["non_source_arguments_match"])


if __name__ == "__main__":
    unittest.main()
