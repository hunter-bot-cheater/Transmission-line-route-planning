"""Compare raster search geometries on deterministic synthetic cost maps only."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from transmission_path_planner.experimental_astar_16 import (
    NEIGHBORS_16,
    astar_exact_integral_path,
)
from transmission_path_planner.experimental_theta_star import (
    integrate_polyline_cost,
    theta_star_cost_path,
)
from transmission_path_planner.methods import astar_search

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "search_geometry_synthetic_20261024.json"
SPACING_M = (30.0, 30.0)


def scenarios():
    shape = (61, 81)
    start, goal = (30, 2), (30, 78)

    uniform = np.ones(shape, dtype=np.float64)
    diagonal = np.ones(shape, dtype=np.float64)

    barrier = np.ones(shape, dtype=np.float64)
    barrier[:43, 40] = np.inf

    friction = np.ones(shape, dtype=np.float64)
    friction[15:46, 30:51] = 12.0

    return (
        ("uniform_open", uniform, start, goal),
        ("uniform_open_diagonal", diagonal, (5, 5), (55, 72)),
        ("hard_barrier_gap", barrier, start, goal),
        ("high_cost_band", friction, start, goal),
    )


def route_length_m(path) -> float:
    return sum(math.hypot((r1 - r0) * SPACING_M[0],
                          (c1 - c0) * SPACING_M[1])
               for (r0, c0), (r1, c1) in zip(path, path[1:]))


def turn_summary(path) -> tuple[int, float]:
    vectors = [((r1 - r0) * SPACING_M[0], (c1 - c0) * SPACING_M[1])
               for (r0, c0), (r1, c1) in zip(path, path[1:])]
    angles = [math.atan2(row, col) for row, col in vectors]
    changes = [abs((second - first + math.pi) % (2 * math.pi) - math.pi)
               for first, second in zip(angles, angles[1:])]
    return sum(angle > 1e-9 for angle in changes), math.degrees(sum(changes))


def run_scenario(name, cost, start, goal):
    traversable = np.isfinite(cost)
    methods = (
        ("A*_8", lambda: astar_search(
            cost, start, goal, pixel_spacing=SPACING_M, heuristic_weight=1.0)),
        ("Dijkstra_8", lambda: astar_search(
            cost, start, goal, use_dijkstra=True,
            pixel_spacing=SPACING_M, heuristic_weight=1.0)),
        ("A*_16_exact", lambda: astar_exact_integral_path(
            cost, traversable, start, goal, NEIGHBORS_16,
            pixel_spacing=SPACING_M)),
        ("Theta*_cost_aware", lambda: theta_star_cost_path(
            cost, traversable, start, goal, pixel_spacing=SPACING_M)),
    )
    timing_samples = {method: [] for method, _ in methods}
    paths = {}
    for _, search in methods:
        search()  # Warm each implementation before collecting timings.
    for repetition in range(5):
        order = methods[repetition % len(methods):] + methods[:repetition % len(methods)]
        for method, search in order:
            started = time.perf_counter()
            paths[method] = search()
            timing_samples[method].append(time.perf_counter() - started)

    rows = []
    for method, _ in methods:
        path = paths[method]
        samples = np.asarray(timing_samples[method])
        if path is None:
            rows.append({"method": method, "status": "no_route",
                         "runtime_s_median": round(float(np.median(samples)), 6),
                         "runtime_s_iqr": round(float(np.percentile(samples, 75)
                                                      - np.percentile(samples, 25)), 6)})
            continue
        turn_count, turn_angle_sum_deg = turn_summary(path)
        integral_cost = integrate_polyline_cost(
            path, cost, traversable, pixel_spacing=SPACING_M)
        rows.append({
            "method": method,
            "status": "feasible" if math.isfinite(integral_cost) else "hard_violation",
            "runtime_s_median": round(float(np.median(samples)), 6),
            "runtime_s_iqr": round(float(np.percentile(samples, 75)
                                          - np.percentile(samples, 25)), 6),
            "route_vertices": len(path),
            "length_m": round(route_length_m(path), 3),
            "continuous_cost_integral": round(integral_cost, 6),
            "turn_count": turn_count,
            "sum_turn_angle_deg": round(turn_angle_sum_deg, 3),
        })
    return {"scenario": name, "start_rc": start, "goal_rc": goal,
            "shape": cost.shape, "spacing_m": SPACING_M, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    results = [run_scenario(*scenario) for scenario in scenarios()]
    report = {
        "experiment": "synthetic_search_geometry_comparison",
        "historical_route_data_loaded": False,
        "route_generation_inputs": "deterministic synthetic cost maps only",
        "primary_metric": "same exact piecewise-constant continuous cost integral",
        "timing_protocol": "one warmup, five runs per method, rotated method order; report median and IQR",
        "algorithms": ["A*_8", "Dijkstra_8", "A*_16_exact", "Theta*_cost_aware"],
        "scenarios": results,
        "interpretation": (
            "This checks search geometry and objective behavior only. It does not "
            "select an algorithm using historical-route similarity or establish "
            "transmission engineering feasibility."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
