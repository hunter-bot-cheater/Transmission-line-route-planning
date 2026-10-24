"""Run a deterministic synthetic sensitivity scan for the tower-graph screen."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transmission_path_planner.experimental_tower_graph import (
    span_clearance_status,
    tower_graph_route,
)
from transmission_path_planner.methods import astar_search


SPACING_M = (100.0, 100.0)
START_RC = (25, 3)
GOAL_RC = (25, 57)
MIN_SPAN_M = 200.0
MIN_CLEARANCE_M = 15.0
MAX_SPANS_M = (500.0, 700.0, 900.0)
STRIDES = (1, 2, 3, 4)
ATTACHMENT_HEIGHTS_M = (80.0, 160.0)
SAG_LIMITS_M = (20.0, 80.0, 140.0)


def synthetic_surface():
    rows, cols = np.mgrid[0:51, 0:61]
    pass_factor = 1.0 - 0.82 * np.exp(-((rows - 12.0) / 3.5) ** 2)
    ridge = 420.0 * np.exp(-((cols - 30.0) / 4.0) ** 2) * pass_factor
    dem = (250.0 + 0.15 * np.abs(rows - 25.0) * 100.0 + ridge).astype(np.float64)
    row_gradient, col_gradient = np.gradient(dem, *SPACING_M)
    slope_deg = np.degrees(np.arctan(np.hypot(row_gradient, col_gradient)))
    cost = (1.0 + slope_deg / 25.0).astype(np.float32)
    return dem, slope_deg, cost


def route_summary(path, dem, attachment_height_m=None, sag_limit_m=None):
    if not path:
        return {
            "status": "no_route",
            "route_length_m": None,
            "span_count": None,
            "max_span_m": None,
            "max_site_elevation_m": None,
            "minimum_sampled_clearance_m": None,
        }

    span_lengths = [
        math.hypot((second[0] - first[0]) * SPACING_M[0],
                   (second[1] - first[1]) * SPACING_M[1])
        for first, second in zip(path, path[1:])
    ]
    clearances = []
    if attachment_height_m is not None and sag_limit_m is not None:
        clearances = [
            span_clearance_status(
                first, second, dem, SPACING_M, attachment_height_m,
                sag_limit_m, MIN_CLEARANCE_M,
            )["sampled_minimum_clearance_m"]
            for first, second in zip(path, path[1:])
        ]
    return {
        "status": "route_found",
        "route_length_m": round(sum(span_lengths), 2),
        "span_count": len(span_lengths),
        "max_span_m": round(max(span_lengths, default=0.0), 2),
        "max_site_elevation_m": round(max(float(dem[row, col]) for row, col in path), 2),
        "minimum_sampled_clearance_m": round(min(clearances), 2) if clearances else None,
    }


def run_scan():
    dem, slope_deg, cost = synthetic_surface()
    traversable = np.ones(dem.shape, dtype=bool)
    tower_allowed = slope_deg <= 20.0
    rows = []
    routes = {}

    astar = astar_search(cost, START_RC, GOAL_RC, pixel_spacing=SPACING_M)
    rows.append({
        "method": "A* grid baseline",
        "candidate_stride": None,
        "max_span_limit_m": None,
        "attachment_height_m": None,
        "max_midspan_sag_m": None,
        **route_summary(astar, dem, 0.0, 0.0),
    })
    routes["A* grid baseline"] = astar

    for stride in STRIDES:
        for max_span in MAX_SPANS_M:
            for attachment_height in ATTACHMENT_HEIGHTS_M:
                for sag in SAG_LIMITS_M:
                    path = tower_graph_route(
                        cost, traversable, tower_allowed, START_RC, GOAL_RC,
                        MIN_SPAN_M, max_span, pixel_spacing=SPACING_M,
                        candidate_stride=stride, dem_raster=dem,
                        attachment_height_m=attachment_height,
                        max_midspan_sag_m=sag,
                        minimum_clearance_m=MIN_CLEARANCE_M,
                    )
                    rows.append({
                        "method": "tower graph",
                        "candidate_stride": stride,
                        "max_span_limit_m": max_span,
                        "attachment_height_m": attachment_height,
                        "max_midspan_sag_m": sag,
                        **route_summary(path, dem, attachment_height, sag),
                    })
                    key = (stride, max_span, attachment_height, sag)
                    routes[key] = path
    return dem, rows, routes


def write_outputs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dem, rows, routes = run_scan()

    csv_path = output_dir / "tower_graph_sensitivity.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "experiment_type": "synthetic_sensitivity_only_not_real_route_or_engineering_validation",
        "surface": {
            "shape": list(dem.shape),
            "pixel_spacing_m": list(SPACING_M),
            "formula": "250 + 0.15*abs(row-25)*100 + 420*exp(-((col-30)/4)^2)*(1-0.82*exp(-((row-12)/3.5)^2))",
            "start_rc": list(START_RC),
            "goal_rc": list(GOAL_RC),
            "tower_site_slope_limit_deg": 20.0,
        },
        "sweep": {
            "minimum_span_m": MIN_SPAN_M,
            "maximum_span_m": list(MAX_SPANS_M),
            "candidate_stride_cells": list(STRIDES),
            "attachment_height_m": list(ATTACHMENT_HEIGHTS_M),
            "maximum_midspan_sag_m": list(SAG_LIMITS_M),
            "minimum_clearance_m": MIN_CLEARANCE_M,
            "parameter_status": "illustrative_values_not_selected_from_an_engineering_standard",
        },
        "result_rows": len(rows),
        "csv": csv_path.name,
    }
    manifest_path = output_dir / "tower_graph_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax = axes[0]
    image = ax.imshow(dem, origin="upper", cmap="terrain", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Synthetic elevation (m)", shrink=0.8)
    astar = routes["A* grid baseline"]
    if astar:
        ax.plot([col for row, col in astar], [row for row, col in astar],
                color="#111111", linestyle="--", linewidth=2.0, label="A* grid")
    colors = {500.0: "#00a6d6", 700.0: "#d62728", 900.0: "#7b2cbf"}
    for max_span, color in colors.items():
        path = routes[(2, max_span, 160.0, 80.0)]
        if path:
            ax.plot([col for row, col in path], [row for row, col in path],
                    color=color, linewidth=1.8, marker="o", markersize=2.5,
                    label=f"Tower graph max span {int(max_span)} m")
    ax.set_title("Route changes under span limits")
    ax.set_xlabel("Column (100 m cells)")
    ax.set_ylabel("Row (100 m cells)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    matrix = np.full((len(STRIDES), len(MAX_SPANS_M)), np.nan)
    for row_index, stride in enumerate(STRIDES):
        for col_index, max_span in enumerate(MAX_SPANS_M):
            path = routes[(stride, max_span, 160.0, 80.0)]
            summary = route_summary(path, dem, 160.0, 80.0)
            if summary["route_length_m"] is not None:
                matrix[row_index, col_index] = summary["route_length_m"] / 1000.0
    grid = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(grid, ax=ax, label="Route length (km)", shrink=0.8)
    ax.set_xticks(range(len(MAX_SPANS_M)), [str(int(value)) for value in MAX_SPANS_M])
    ax.set_yticks(range(len(STRIDES)), [str(value) for value in STRIDES])
    ax.set_xlabel("Maximum span limit (m)")
    ax.set_ylabel("Candidate stride (cells)")
    ax.set_title("Length sensitivity (height 160 m, sag 80 m)")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            text = "no route" if np.isnan(matrix[row_index, col_index]) else f"{matrix[row_index, col_index]:.2f}"
            ax.text(col_index, row_index, text, ha="center", va="center",
                    color="white" if np.isnan(matrix[row_index, col_index]) or matrix[row_index, col_index] < 6.5 else "black",
                    fontsize=8)

    fig.suptitle("Synthetic tower-graph sensitivity — not a Taiwan engineering result", fontsize=13)
    image_path = output_dir / "tower_graph_sensitivity.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    return csv_path, manifest_path, image_path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    csv_path, manifest_path, image_path, rows = write_outputs(Path(args.output_dir))
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"manifest={manifest_path}")
    print(f"figure={image_path}")
    baseline = next(row for row in rows if row["method"] == "A* grid baseline")
    print(f"astar_length_m={baseline['route_length_m']}")
    print(f"tower_graph_routes={sum(row['status'] == 'route_found' for row in rows if row['method'] == 'tower graph')}/"
          f"{sum(row['method'] == 'tower graph' for row in rows)}")


if __name__ == "__main__":
    main()
