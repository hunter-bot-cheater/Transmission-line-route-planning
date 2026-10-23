"""Bounded, reference-free weighted-A* search tradeoff benchmark."""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from transmission_path_planner.methods import astar_search


WEIGHTS = (1.0, 1.05, 1.10, 1.20)
SPACING_M = 35.0
REPEATS = 3
ORDER_SEED = 20261024


def _cost_surface(rows: int, cols: int, scenario: str) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    xn = x / max(cols - 1, 1)
    yn = y / max(rows - 1, 1)
    if scenario == "ridge":
        elevation = (180 * np.sin(6 * np.pi * xn)
                     + 130 * np.cos(4 * np.pi * yn)
                     + 90 * np.sin(4 * np.pi * (xn + yn)))
    elif scenario == "settlement":
        elevation = 110 * np.sin(3 * np.pi * xn) * np.cos(5 * np.pi * yn)
    else:
        elevation = (150 * np.cos(5 * np.pi * xn)
                     + 100 * np.sin(7 * np.pi * yn)
                     + 70 * np.cos(3 * np.pi * (xn - yn)))
    gy, gx = np.gradient(elevation, SPACING_M, SPACING_M)
    cost = 1.0 + 8.0 * np.hypot(gx, gy)

    if scenario == "settlement":
        for cx, cy in ((0.32, 0.35), (0.68, 0.67), (0.55, 0.23)):
            cost += 5.0 * np.exp(-((xn - cx) ** 2 + (yn - cy) ** 2) / 0.004)
    elif scenario == "ridge":
        cost += 2.0 * np.exp(-((xn - 0.52) ** 2) / 0.004)
    else:
        cost += 2.5 * np.exp(-((yn - 0.48) ** 2) / 0.006)

    blocked = np.zeros((rows, cols), dtype=bool)
    if scenario == "ridge":
        blocked[rows // 2 - 3:rows // 2 + 3, 24:cols - 24] = True
        blocked[rows // 2 - 3:rows // 2 + 3, cols // 3 - 10:cols // 3 + 10] = False
    elif scenario == "settlement":
        for row in range(24, rows - 24):
            col = int(row * cols / rows + cols / 3)
            blocked[row, max(0, col - 2):min(cols, col + 3)] = True
        blocked[24:rows // 3, :] = False
        blocked[2 * rows // 3:rows - 24, :] = False
    else:
        for col, gate_row in ((cols // 3, rows // 3), (2 * cols // 3, 2 * rows // 3)):
            blocked[24:rows - 24, col - 3:col + 3] = True
            blocked[gate_row - 12:gate_row + 12, col - 3:col + 3] = False
    cost[blocked] = np.inf
    return cost


def _path_cost(path, cost):
    total = 0.0
    for (row, col), (next_row, next_col) in zip(path, path[1:]):
        dr, dc = next_row - row, next_col - col
        if abs(dr) > 1 or abs(dc) > 1:
            return None, 1
        if dr and dc and (not np.isfinite(cost[row + dr, col])
                          or not np.isfinite(cost[row, col + dc])):
            return None, 1
        if not np.isfinite(cost[row, col]) or not np.isfinite(cost[next_row, next_col]):
            return None, 1
        distance = math.hypot(dr * SPACING_M, dc * SPACING_M)
        total += 0.5 * (cost[row, col] + cost[next_row, next_col]) * distance
    return float(total), 0


def _hard_violations(path, cost):
    if not path:
        return 1
    if any(not np.isfinite(cost[row, col]) for row, col in path):
        return 1
    return 0


def _run_search(cost, start, end, weight):
    started = time.perf_counter()
    try:
        path = astar_search(
            cost, start, end, pixel_spacing=(SPACING_M, SPACING_M),
            heuristic_weight=weight)
    except Exception as exc:
        return None, time.perf_counter() - started, str(exc), 1, None
    elapsed = time.perf_counter() - started
    path_cost, edge_violations = _path_cost(path, cost) if path else (None, 0)
    violations = max(_hard_violations(path, cost), edge_violations)
    return path, elapsed, None, violations, path_cost


def _summarize_runs(raw):
    groups = defaultdict(lambda: defaultdict(dict))
    for result in raw:
        case_key = (result["scenario"], result["pair"])
        groups[case_key][result["repeat"]][result["weight"]] = result

    summary = {}
    for weight in WEIGHTS:
        regrets, speedups = [], []
        failures = violations = baseline_failures = candidate_failures = 0
        for runs_by_repeat in groups.values():
            case_regrets, case_speedups = [], []
            for result_by_weight in runs_by_repeat.values():
                baseline = result_by_weight[1.0]
                current = result_by_weight[weight]
                baseline_failed = baseline["path_cost_meter"] is None
                candidate_failed = current["path_cost_meter"] is None
                baseline_failures += baseline_failed
                candidate_failures += candidate_failed
                failures += candidate_failed or (weight != 1.0 and baseline_failed)
                violations += current["hard_violations"]
                if weight != 1.0:
                    violations += baseline["hard_violations"]
                if not baseline_failed and not candidate_failed:
                    case_regrets.append(
                        100 * (current["path_cost_meter"] / baseline["path_cost_meter"] - 1))
                    case_speedups.append(baseline["time_s"] / current["time_s"])
            if len(case_regrets) == len(runs_by_repeat):
                regrets.append(max(case_regrets))
                speedups.append(statistics.median(case_speedups))
        p95_regret = float(np.percentile(regrets, 95)) if regrets else None
        max_regret = max(regrets) if regrets else None
        median_speedup = statistics.median(speedups) if speedups else 0.0
        summary[str(weight)] = {
            "cases": len(groups),
            "compared_cases": len(regrets),
            "search_runs": sum(len(runs) for runs in groups.values()),
            "failures": failures,
            "baseline_failures": baseline_failures,
            "candidate_failures": candidate_failures,
            "hard_violations": violations,
            "cost_regret_p95_pct": p95_regret,
            "cost_regret_max_pct": max_regret,
            "median_speedup": median_speedup,
            "screen_status": (
                "control" if weight == 1.0 else
                "pass" if failures == 0 and violations == 0
                and p95_regret is not None and max_regret is not None
                and p95_regret <= 1.0 and max_regret <= 3.0
                and median_speedup >= 1.25 else "fail"
            ),
        }
    return summary


def run_benchmark(output_path: Path) -> dict:
    rows, cols = 256, 384
    pairs = (
        ((16, 16), (rows - 17, cols - 17)),
        ((16, cols - 17), (rows - 17, 16)),
        ((rows // 2, 16), (rows // 2, cols - 17)),
    )
    raw = []
    execution_orders = {}
    for scenario_index, scenario in enumerate(("ridge", "settlement", "maze")):
        cost = _cost_surface(rows, cols, scenario)
        for pair_id, (start, end) in enumerate(pairs):
            # Warm every weight once, outside the measured repetitions.
            for weight in WEIGHTS:
                _run_search(cost, start, end, weight)
            for repeat in range(REPEATS):
                order = list(WEIGHTS)
                random.Random(ORDER_SEED + scenario_index * len(pairs) * REPEATS
                              + pair_id * REPEATS + repeat).shuffle(order)
                execution_orders[f"{scenario}:{pair_id}:{repeat}"] = order
                for weight in order:
                    path, elapsed, fail_reason, violations, path_cost = _run_search(
                        cost, start, end, weight)
                    raw.append({
                        "scenario": scenario,
                        "pair": pair_id,
                        "repeat": repeat,
                        "weight": weight,
                        "time_s": elapsed,
                        "path_cost_meter": path_cost,
                        "path_points": len(path) if path else 0,
                        "hard_violations": violations,
                        "fail_reason": fail_reason,
                    })

    result = {
        "experiment": "weighted_astar_synthetic_no_reference",
        "grid_shape": [rows, cols],
        "pixel_spacing_m": SPACING_M,
        "repeats": REPEATS,
        "warmups_per_case_and_weight": 1,
        "execution_order_seed": ORDER_SEED,
        "execution_orders": execution_orders,
        "weights": list(WEIGHTS),
        "criteria": {
            "interpretation": "Provisional engineering screening gates, not a theorem or statistical guarantee.",
            "failures": 0,
            "hard_violations": 0,
            "cost_regret_p95_pct_max": 1.0,
            "cost_regret_max_pct_max": 3.0,
            "median_speedup_min": 1.25,
        },
        "summary": _summarize_runs(raw),
        "runs": raw,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    output = Path("outputs/optimization_loop_20261024/weighted_astar_synthetic/results.json")
    report = run_benchmark(output)
    print(json.dumps(report["summary"], indent=2))
    print(f"saved: {output}")
