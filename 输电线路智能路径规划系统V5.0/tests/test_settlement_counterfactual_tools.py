import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.run_settlement_counterfactual_12 import endpoint_cases
from tools.settlement_counterfactual_12 import normalized_argv, parsed_cli_args


class SettlementCounterfactualToolTests(unittest.TestCase):
    def write_endpoint_csv(self, path: Path, bad_value=None):
        fields = ["name", "start_lat", "start_lon", "end_lat", "end_lon"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(12):
                writer.writerow({
                    "name": f"case_{index}", "start_lat": 24.0,
                    "start_lon": 120.0, "end_lat": 24.5, "end_lon": 121.0,
                })
        if bad_value is not None:
            rows = path.read_text(encoding="utf-8").splitlines()
            rows[1] = rows[1].replace("24.0", str(bad_value), 1)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_endpoint_preflight_validates_coordinates(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.csv"
            self.write_endpoint_csv(path)
            self.assertEqual(endpoint_cases(path), [f"case_{index}" for index in range(12)])

    def test_endpoint_preflight_rejects_non_finite_coordinates(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.csv"
            self.write_endpoint_csv(path, "nan")
            with self.assertRaises(SystemExit):
                endpoint_cases(path)

    def test_variant_normalization_removes_only_weight_and_output_path(self):
        first = {"argv": [
            "planner.py", "--dem", "dem.tif", "--settlement-weight", "2.0",
            "--out-dir", "weight_2", "--no-vin",
        ]}
        second = {"argv": [
            "planner.py", "--dem", "dem.tif", "--settlement-weight", "0.0",
            "--out-dir", "weight_0", "--no-vin",
        ]}

        self.assertEqual(normalized_argv(first), normalized_argv(second))
        self.assertEqual(normalized_argv(first), ["planner.py", "--dem", "dem.tif", "--no-vin"])

    def test_cli_parser_preserves_effective_values_and_switches(self):
        summary = {"argv": [
            "planner.py", "--dem", "dem.tif", "--settlement-weight", "2.0",
            "--out-dir", "weight_2", "--max-side", "4096", "--no-vin", "--no-ppp",
        ]}

        self.assertEqual(parsed_cli_args(summary), {
            "--dem": "dem.tif", "--settlement-weight": "2.0",
            "--out-dir": "weight_2", "--max-side": "4096",
            "--no-vin": True, "--no-ppp": True,
        })

    def test_cli_parser_rejects_unregistered_inputs(self):
        with self.assertRaises(SystemExit):
            parsed_cli_args({"argv": ["planner.py", "--use-costunet"]})

    def test_cli_parser_rejects_duplicate_parameters(self):
        with self.assertRaises(SystemExit):
            parsed_cli_args({"argv": ["planner.py", "--settlement-weight", "2",
                                       "--settlement-weight", "0"]})


if __name__ == "__main__":
    unittest.main()
