"""Measure candidate-route stability across all phases on a synthetic surface.

This is a deterministic algorithm diagnostic only; it does not load Taiwan DEM
or reference-line data and must not be read as real-line accuracy evidence.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_tower_graph_sensitivity import (  # noqa: E402
    GOAL_RC,
    MIN_CLEARANCE_M,
    MIN_SPAN_M,
    SPACING_M,
    START_RC,
    synthetic_surface,
)
from transmission_path_planner.experimental_tower_graph import (  # noqa: E402
    tower_graph_path_objective,
    tower_graph_route,
)


STRIDE = 4
MAX_SPAN_M = 700.0
ATTACHMENT_HEIGHT_M = 160.0
MAX_SAG_M = 80.0
SAMPLE_STEP_M = 10.0


def sample_route(path):
    samples = []
    for first, second in zip(path, path[1:]):
        row_delta = (second[0] - first[0]) * SPACING_M[0]
        col_delta = (second[1] - first[1]) * SPACING_M[1]
        length = math.hypot(row_delta, col_delta)
        segments = max(1, math.ceil(length / SAMPLE_STEP_M))
        for index in range(segments):
            fraction = index / segments
            samples.append((
                (first[0] + fraction * (second[0] - first[0])) * SPACING_M[0],
                (first[1] + fraction * (second[1] - first[1])) * SPACING_M[1],
            ))
    samples.append((path[-1][0] * SPACING_M[0], path[-1][1] * SPACING_M[1]))
    return np.asarray(samples, dtype=np.float64)


def run(output_dir):
    dem, slope, cost = synthetic_surface()
    traversable = np.ones(dem.shape, dtype=bool)
    tower_allowed = slope <= 20.0
    routes, phase_rows = {}, []
    for row_phase in range(STRIDE):
        for col_phase in range(STRIDE):
            phase = (row_phase, col_phase)
            path = tower_graph_route(
                cost, traversable, tower_allowed, START_RC, GOAL_RC,
                MIN_SPAN_M, MAX_SPAN_M, pixel_spacing=SPACING_M,
                candidate_stride=STRIDE, candidate_phase=phase,
                dem_raster=dem, attachment_height_m=ATTACHMENT_HEIGHT_M,
                max_midspan_sag_m=MAX_SAG_M,
                minimum_clearance_m=MIN_CLEARANCE_M,
            )
            if path:
                routes[phase] = sample_route(path)
                objective = tower_graph_path_objective(path, cost, SPACING_M)
                length = sum(
                    math.hypot((b[0] - a[0]) * SPACING_M[0],
                               (b[1] - a[1]) * SPACING_M[1])
                    for a, b in zip(path, path[1:]))
                phase_rows.append({
                    "phase_row": row_phase,
                    "phase_col": col_phase,
                    "status": "route_found",
                    "sample_count": int(len(routes[phase])),
                    "route_length_m": round(length, 3),
                    "objective_cost_meter": objective["total_cost_meter"],
                })
            else:
                phase_rows.append({
                    "phase_row": row_phase,
                    "phase_col": col_phase,
                    "status": "no_route",
                    "sample_count": 0,
                    "route_length_m": None,
                    "objective_cost_meter": None,
                })

    pair_rows = []
    for index, first_phase in enumerate(sorted(routes)):
        for second_phase in sorted(routes)[index + 1:]:
            first, second = routes[first_phase], routes[second_phase]
            distances = np.concatenate((
                cKDTree(second).query(first)[0],
                cKDTree(first).query(second)[0],
            ))
            pair_rows.append({
                "phase_a": f"{first_phase[0]},{first_phase[1]}",
                "phase_b": f"{second_phase[0]},{second_phase[1]}",
                "mean_bidirectional_distance_m": round(float(distances.mean()), 3),
                "p95_bidirectional_distance_m": round(float(np.quantile(distances, .95)), 3),
                "max_bidirectional_distance_m": round(float(distances.max()), 3),
                "within_100m_fraction": round(float(np.mean(distances <= 100.0)), 6),
                "within_500m_fraction": round(float(np.mean(distances <= 500.0)), 6),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("phase_routes.csv", phase_rows), ("phase_pairs.csv", pair_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "experiment_type": "synthetic_phase_stability_not_real_line_accuracy",
        "surface_source": "scripts/benchmark_tower_graph_sensitivity.py::synthetic_surface",
        "shape": list(dem.shape),
        "pixel_spacing_m": list(SPACING_M),
        "start_rc": list(START_RC),
        "goal_rc": list(GOAL_RC),
        "tower_site_slope_limit_deg": 20.0,
        "candidate_stride": STRIDE,
        "phase_count": STRIDE ** 2,
        "successful_phases": len(routes),
        "pair_count": len(pair_rows),
        "minimum_span_m": MIN_SPAN_M,
        "maximum_span_m": MAX_SPAN_M,
        "attachment_height_m": ATTACHMENT_HEIGHT_M,
        "maximum_midspan_sag_m": MAX_SAG_M,
        "minimum_clearance_m": MIN_CLEARANCE_M,
        "sampling_step_m": SAMPLE_STEP_M,
        "reference_geometry_loaded": False,
        "outputs": ["phase_routes.csv", "phase_pairs.csv"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(output_dir / "manifest.json"),
        "successful_phases": len(routes),
        "phase_pairs": len(pair_rows),
        "median_pair_p95_m": float(np.median([
            row["p95_bidirectional_distance_m"] for row in pair_rows])),
        "max_pair_p95_m": max(row["p95_bidirectional_distance_m"] for row in pair_rows),
        "median_within_100m_fraction": float(np.median([
            row["within_100m_fraction"] for row in pair_rows])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
