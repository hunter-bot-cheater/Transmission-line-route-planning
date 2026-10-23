import unittest

from transmission_path_planner.visualization import _algo_comparison_data


class AlgorithmComparisonDataTests(unittest.TestCase):
    def test_no_path_is_not_aggregated_as_zero_hard_violations(self):
        results = [
            {"metrics": {
                "A*(baseline)": {
                    "hard_constraint_status": "no_path",
                    "hard_violations": 0,
                    "time_s": 10.0,
                },
            }},
            {"metrics": {
                "A*(baseline)": {
                    "hard_constraint_status": "pass",
                    "hard_violations": 0,
                    "time_s": 5.0,
                },
                "A*(16-neighbor experimental)": {
                    "hard_constraint_status": "no_path",
                    "hard_violations": 0,
                    "time_s": 7.0,
                },
            }},
        ]

        methods, aggregates = _algo_comparison_data(results)

        self.assertEqual(methods, ["A*(baseline)", "A*(16-neighbor experimental)"])
        self.assertEqual(aggregates["A*(baseline)"]["hard_violations"], [0])
        self.assertEqual(aggregates["A*(16-neighbor experimental)"]["hard_violations"], [])
        self.assertEqual(aggregates["A*(16-neighbor experimental)"]["time_s"], [7.0])


if __name__ == "__main__":
    unittest.main()
