"""Run a no-reference sensitivity experiment on a synthetic terrain corridor.

The terrain and exclusion polygons are analytic physical-space functions. The
same scene is sampled on two resolutions and two half-cell grid origins; active
production cost weights are perturbed one at a time by +/-10 percent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import LineString, Point, mapping, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transmission_path_planner.cost import build_route_cost
from transmission_path_planner.data import compute_terrain_factors
from transmission_path_planner.metrics import astar_path_objective
from transmission_path_planner.methods import astar_search


DOMAIN_X_M = 14_000.0
DOMAIN_Y_M = 10_000.0
START_XY = (1_000.0, 5_000.0)
END_XY = (13_000.0, 5_000.0)
HARD_POLYGON = box(6_250.0, 3_700.0, 7_750.0, 6_300.0)
SETTLEMENT_POLYGON = box(0.0, 0.0, DOMAIN_X_M, 4_500.0)
HAZARD_POLYGON = box(0.0, 5_500.0, DOMAIN_X_M, DOMAIN_Y_M)

BASE_WEIGHTS = {
    "terrain_slope_weight": 0.75,
    "roughness_weight": 0.25,
    "ridge_penalty_weight": 0.10,
    "settlement_weight": 2.0,
    "official_hazard_weight": 1.0,
    "grade_penalty_weight": 1.0,
}


def terrain_surface(x, y):
    """A fixed, smooth hill/ridge field in physical metres."""
    hill = 300.0 * np.exp(-0.5 * (((x - 7_000.0) / 1_350.0) ** 2
                                  + ((y - 5_000.0) / 1_700.0) ** 2))
    north_ridge = 125.0 * np.exp(-0.5 * (((x - 7_000.0) / 1_800.0) ** 2
                                         + ((y - 7_500.0) / 620.0) ** 2))
    south_saddle = 45.0 * np.exp(-0.5 * (((x - 7_000.0) / 2_200.0) ** 2
                                         + ((y - 2_500.0) / 900.0) ** 2))
    texture = 16.0 * np.sin(x / 270.0) * np.cos(y / 360.0)
    texture *= np.exp(-0.5 * (((x - 7_000.0) / 4_000.0) ** 2
                              + ((y - 5_000.0) / 3_800.0) ** 2))
    return (90.0 + 0.004 * x + 0.0015 * y + hill + north_ridge
            - south_saddle + texture).astype(np.float32)


def make_grid(resolution_m, origin_fraction):
    shift = resolution_m * origin_fraction
    left, top = -shift, DOMAIN_Y_M + shift
    width = int(math.ceil((DOMAIN_X_M + 2.0 * shift) / resolution_m))
    height = int(math.ceil((DOMAIN_Y_M + 2.0 * shift) / resolution_m))
    transform = Affine(resolution_m, 0.0, left, 0.0, -resolution_m, top)
    xs = left + (np.arange(width, dtype=np.float64) + 0.5) * resolution_m
    ys = top - (np.arange(height, dtype=np.float64) + 0.5) * resolution_m
    x, y = np.meshgrid(xs, ys)
    dem = terrain_surface(x, y)
    terrain = compute_terrain_factors(dem, pixel_m=resolution_m)

    hard_cells = rasterize(
        [(HARD_POLYGON, 1)], out_shape=(height, width), transform=transform,
        fill=0, all_touched=True, dtype=np.uint8,
    ).astype(bool)
    hard_mask = ~hard_cells
    settlement_area = rasterize(
        [(SETTLEMENT_POLYGON, 1)], out_shape=(height, width), transform=transform,
        fill=0, all_touched=False, dtype=np.uint8,
    ).astype(np.float32)
    hazard_area = rasterize(
        [(HAZARD_POLYGON, 1)], out_shape=(height, width), transform=transform,
        fill=0, all_touched=False, dtype=np.uint8,
    ).astype(np.float32)
    # A fractional, smoothed mapped-building proxy makes settlement weight
    # sensitivity visible without treating the synthetic zone as a hard mask.
    settlement_proxy = 0.50 * settlement_area
    osm = {
        "building_density": settlement_proxy,
        "residential_landuse_mask": np.zeros((height, width), dtype=np.float32),
        "landuse_code": np.zeros((height, width), dtype=np.float32),
    }
    hazards = {"synthetic_hazard": hazard_area}

    def cell_for(xy):
        x0, y0 = xy
        col = int(math.floor((x0 - left) / resolution_m))
        row = int(math.floor((top - y0) / resolution_m))
        return row, col

    start_rc, end_rc = cell_for(START_XY), cell_for(END_XY)
    return {
        "resolution_m": resolution_m,
        "origin_fraction": origin_fraction,
        "transform": transform,
        "dem": dem,
        "terrain": terrain,
        "hard_mask": hard_mask,
        "osm": osm,
        "hazards": hazards,
        "start_rc": start_rc,
        "end_rc": end_rc,
        "shape": (height, width),
    }


def weight_cases():
    cases = [("baseline", dict(BASE_WEIGHTS))]
    for name, value in BASE_WEIGHTS.items():
        for factor, label in ((0.9, "minus10"), (1.1, "plus10")):
            values = dict(BASE_WEIGHTS)
            values[name] = value * factor
            cases.append((f"{name}_{label}", values))
    return cases


def route_line(path, grid):
    resolution = grid["resolution_m"]
    transform = grid["transform"]
    coords = [transform * (col + 0.5, row + 0.5) for row, col in path]
    if coords[0] != START_XY:
        coords.insert(0, START_XY)
    if coords[-1] != END_XY:
        coords.append(END_XY)
    return LineString(coords)


def symmetric_distance(source, target, sample_spacing_m=20.0):
    def directed(a, b):
        count = max(1, math.ceil(a.length / sample_spacing_m))
        distances = [b.distance(a.interpolate((i + 0.5) * a.length / count))
                     for i in range(count)]
        return np.asarray(distances, dtype=np.float64)

    both = np.concatenate((directed(source, target), directed(target, source)))
    return {
        "mean_m": float(np.mean(both)),
        "p95_m": float(np.percentile(both, 95)),
        "sampled_max_m": float(np.max(both)),
        "minimum_segment_distance_m": float(source.distance(target)),
    }


def path_cost_parts(path, grid, weights, cost_grid):
    terrain, osm, hazards = grid["terrain"], grid["osm"], grid["hazards"]
    slope_proxy = np.clip(terrain["slope"] / 45.0, 0.0, 1.0) ** 1.5
    roughness_proxy = np.clip(terrain["roughness_27"] / 100.0, 0.0, 1.0)
    ridge_proxy = np.clip(np.nan_to_num(terrain["tpi_300"], nan=0.0,
                                         posinf=200.0, neginf=0.0) / 200.0,
                          0.0, 1.0)
    settlement_proxy = np.clip(osm["building_density"], 0.0, 1.0)
    hazard_proxy = np.maximum.reduce(list(hazards.values()))
    proxies = {
        "slope": 4.0 * weights["terrain_slope_weight"] * slope_proxy,
        "roughness": 4.0 * weights["roughness_weight"] * roughness_proxy,
        "ridge": 4.0 * weights["ridge_penalty_weight"] * ridge_proxy,
        "settlement": 4.0 * weights["settlement_weight"] * settlement_proxy,
        "hazard": 4.0 * weights["official_hazard_weight"] * hazard_proxy,
    }
    parts = {name: 0.0 for name in ("movement", *proxies, "grade")}
    resolution = grid["resolution_m"]
    elevation = grid["dem"]
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        step = math.hypot((r1 - r0) * resolution, (c1 - c0) * resolution)
        parts["movement"] += step
        for name, surface in proxies.items():
            parts[name] += 0.5 * (float(surface[r0, c0]) + float(surface[r1, c1])) * step
        grade = abs(float(elevation[r1, c1]) - float(elevation[r0, c0])) / step
        parts["grade"] += weights["grade_penalty_weight"] * min(grade / 0.30, 1.0) * step
    parts["total"] = sum(parts.values())
    objective = astar_path_objective(
        path, cost_grid, elevation, (resolution, resolution),
        grade_penalty_weight=weights["grade_penalty_weight"],
    )
    parts["objective_check_delta"] = parts["total"] - objective["total_cost"]
    return parts


def hard_check(path, line, grid):
    mask = grid["hard_mask"]
    violations = 0
    for index, (row, col) in enumerate(path):
        if not mask[row, col]:
            violations += 1
        if index:
            prev_row, prev_col = path[index - 1]
            dr, dc = row - prev_row, col - prev_col
            if max(abs(dr), abs(dc)) != 1:
                violations += 1
            elif dr and dc and (not mask[prev_row + dr, prev_col]
                                or not mask[prev_row, prev_col + dc]):
                violations += 1
    return {
        "raster_hard_violations": violations,
        "physical_hard_polygon_intersects": bool(line.intersects(HARD_POLYGON)),
        "physical_hard_polygon_clearance_m": float(line.distance(HARD_POLYGON)),
    }


def solve(grid, weights):
    terrain, osm = grid["terrain"], grid["osm"]
    cost = build_route_cost(
        terrain, osm,
        settlement_weight=weights["settlement_weight"],
        ridge_penalty_weight=weights["ridge_penalty_weight"],
        official_hazard_masks=grid["hazards"],
        official_hazard_weight=weights["official_hazard_weight"],
        terrain_slope_weight=weights["terrain_slope_weight"],
        roughness_weight=weights["roughness_weight"],
    )
    cost_grid = np.where(grid["hard_mask"], 1.0 + 4.0 * cost, np.inf)
    start, end = grid["start_rc"], grid["end_rc"]
    if not grid["hard_mask"][start] or not grid["hard_mask"][end]:
        raise RuntimeError("synthetic endpoint fell inside the fixed hard polygon")
    start_time = time.perf_counter()
    path = astar_search(
        cost_grid, start, end,
        elevation=grid["dem"],
        pixel_spacing=(grid["resolution_m"], grid["resolution_m"]),
        grade_penalty_weight=weights["grade_penalty_weight"],
        heuristic_weight=1.0,
    )
    elapsed = time.perf_counter() - start_time
    if not path:
        raise RuntimeError("no synthetic route found")
    line = route_line(path, grid)
    first_center = grid["transform"] * (path[0][1] + 0.5, path[0][0] + 0.5)
    last_center = grid["transform"] * (path[-1][1] + 0.5, path[-1][0] + 0.5)
    parts = path_cost_parts(path, grid, weights, cost_grid)
    hard = hard_check(path, line, grid)
    return {
        "path": path,
        "line": line,
        "parts": parts,
        "hard": hard,
        "elapsed_s": elapsed,
        "path_length_m": float(line.length),
        "grid_path_length_m": float(sum(
            math.hypot((r1 - r0) * grid["resolution_m"],
                       (c1 - c0) * grid["resolution_m"])
            for (r0, c0), (r1, c1) in zip(path, path[1:]))),
        "path_bounds_x_min_m": float(line.bounds[0]),
        "path_bounds_y_min_m": float(line.bounds[1]),
        "path_bounds_x_max_m": float(line.bounds[2]),
        "path_bounds_y_max_m": float(line.bounds[3]),
        "start_snap_m": float(Point(START_XY).distance(Point(*first_center))),
        "end_snap_m": float(Point(END_XY).distance(Point(*last_center))),
        "weights": weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/synthetic_route_stability_20261024")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid_specs = [(resolution, shift) for resolution in (35.0, 70.0)
                  for shift in (0.0, 0.5)]
    grids = {(res, shift): make_grid(res, shift) for res, shift in grid_specs}
    results = {}
    rows = []

    # First establish each grid's own baseline and deterministic repeat check.
    for res, shift in grid_specs:
        grid = grids[(res, shift)]
        key = (res, shift, "baseline")
        reps = [solve(grid, dict(BASE_WEIGHTS)) for _ in range(args.repeats)]
        results[key] = reps[0]
        for repeat, result in enumerate(reps):
            row = make_row(result, grid, "baseline", repeat)
            row["repeat_path_identical_to_first"] = (
                result["path"] == reps[0]["path"])
            row["repeat_offset_mean_m"] = symmetric_distance(
                result["line"], reps[0]["line"])["mean_m"]
            rows.append(row)

    cases = weight_cases()
    for case_name, weights in cases:
        if case_name == "baseline":
            continue
        for res, shift in grid_specs:
            grid = grids[(res, shift)]
            result = solve(grid, weights)
            results[(res, shift, case_name)] = result
            row = make_row(result, grid, case_name, 0)
            local_baseline = results[(res, shift, "baseline")]
            row.update({f"offset_from_local_baseline_{name}": value
                        for name, value in symmetric_distance(
                            result["line"], local_baseline["line"]).items()})
            rows.append(row)

    anchor = results[(35.0, 0.0, "baseline")]
    for row in rows:
        result = results[(row["resolution_m"], row["origin_fraction"], row["weight_case"])]
        row.update({f"offset_from_35m_origin0_baseline_{name}": value
                    for name, value in symmetric_distance(
                        result["line"], anchor["line"]).items()})

    fieldnames = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    anchor = results[(35.0, 0.0, "baseline")]
    grid_comparisons = []
    for res, shift in grid_specs:
        baseline = results[(res, shift, "baseline")]
        grid_comparisons.append({
            "resolution_m": res,
            "origin_fraction": shift,
            **{f"from_35m_origin0_{name}": value
               for name, value in symmetric_distance(baseline["line"], anchor["line"]).items()},
        })
    origin_comparisons = []
    for res in (35.0, 70.0):
        low = results[(res, 0.0, "baseline")]
        half = results[(res, 0.5, "baseline")]
        origin_comparisons.append({
            "resolution_m": res,
            **symmetric_distance(low["line"], half["line"]),
        })

    features = []
    for (res, shift, case_name), result in results.items():
        features.append({
            "type": "Feature",
            "geometry": mapping(result["line"]),
            "properties": {
                "resolution_m": res,
                "origin_fraction": shift,
                "weight_case": case_name,
                "weights": result["weights"],
                "path_length_m": result["path_length_m"],
                **result["hard"],
            },
        })
    with (output / "routes.geojson").open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features},
                  handle, ensure_ascii=False)

    summary = {
        "experiment": "synthetic fixed-physics cost and grid stability",
        "uses_historical_routes": False,
        "physical_domain_m": [DOMAIN_X_M, DOMAIN_Y_M],
        "fixed_endpoints_m": {"start": START_XY, "end": END_XY},
        "fixed_hard_polygon_bounds_m": list(HARD_POLYGON.bounds),
        "grid_settings": [
            {"resolution_m": res, "origin_fraction_of_cell": shift,
             "shape": list(grids[(res, shift)]["shape"])}
            for res, shift in grid_specs
        ],
        "base_weights": BASE_WEIGHTS,
        "weight_cases": [name for name, _ in cases],
        "repeats_per_baseline_grid": args.repeats,
        "route_solves": len(rows),
        "summary_csv": "metrics.csv",
        "routes_geojson": "routes.geojson",
        "baseline_grid_comparisons": grid_comparisons,
        "same_resolution_origin_shift_comparisons": origin_comparisons,
        "interpretation_limit": (
            "Synthetic sensitivity only; no empirical route accuracy or engineering feasibility claim."),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_row(result, grid, weight_case, repeat):
    return {
        "weight_case": weight_case,
        "resolution_m": grid["resolution_m"],
        "origin_fraction": grid["origin_fraction"],
        "repeat": repeat,
        "height_cells": grid["shape"][0],
        "width_cells": grid["shape"][1],
        "path_length_m": result["path_length_m"],
        "grid_path_length_m": result["grid_path_length_m"],
        "path_bounds_x_min_m": result["path_bounds_x_min_m"],
        "path_bounds_y_min_m": result["path_bounds_y_min_m"],
        "path_bounds_x_max_m": result["path_bounds_x_max_m"],
        "path_bounds_y_max_m": result["path_bounds_y_max_m"],
        "path_cells": len(result["path"]),
        "start_snap_m": result["start_snap_m"],
        "end_snap_m": result["end_snap_m"],
        "search_time_s": result["elapsed_s"],
        **result["hard"],
        **{f"cost_{name}": value for name, value in result["parts"].items()},
        **{f"weight_{name}": value for name, value in result["weights"].items()},
    }


if __name__ == "__main__":
    main()
