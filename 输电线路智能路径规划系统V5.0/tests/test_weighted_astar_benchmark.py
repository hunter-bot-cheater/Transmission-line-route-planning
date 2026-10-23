import json

from scripts.benchmark_weighted_astar_synthetic import WEIGHTS, _summarize_runs


def test_failed_baseline_makes_candidate_case_uncomparable_and_fails_screen():
    runs = [
        {
            "scenario": "case",
            "pair": 0,
            "repeat": 0,
            "weight": weight,
            "time_s": 1.0,
            "path_cost_meter": None if weight == 1.0 else 10.0,
            "hard_violations": 1 if weight == 1.0 else 0,
        }
        for weight in WEIGHTS
    ]

    summary = _summarize_runs(runs)["1.05"]

    assert summary["cases"] == 1
    assert summary["compared_cases"] == 0
    assert summary["baseline_failures"] == 1
    assert summary["candidate_failures"] == 0
    assert summary["failures"] == 1
    assert summary["screen_status"] == "fail"
    json.dumps(_summarize_runs(runs), allow_nan=False)
