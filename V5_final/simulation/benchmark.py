"""Compare fixed and active scan order through the same sensor and V5 planner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import server as simulation_server
from server import (
    CELL_SIZE_METERS, _get_session, build_cost_grid, clamp, create_world_session, plan_path,
    plan_session, release_world_session, select_session_scan, sense_flight_segment,
)

_BENCHMARK_WORLD_SEED_KEY = hashlib.sha256(b"v5-simulation-benchmark-v1").digest()


def scan_pattern(width, height, start):
    lane_spacing = round(300 / CELL_SIZE_METERS)
    sweep_spacing = round(200 / CELL_SIZE_METERS)
    rows = list(range(0, height, lane_spacing))
    if rows[-1] != height - 1:
        rows.append(height - 1)
    if start["y"] > height / 2:
        rows.reverse()
    reverse = start["x"] > width / 2
    points = []
    for y in rows:
        xs = range(width - 2, 0, -sweep_spacing) if reverse else range(1, width - 1, sweep_spacing)
        points.extend({"x": x, "y": y} for x in xs)
        reverse = not reverse
    return points


def build_cost(world):
    """Independent full-truth reference used only to score the experiment."""
    layers, weights = world["layers"], world["layer_weights"]
    height, width = world["height"], world["width"]
    cost = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            n = lambda name: clamp(layers[name][y][x])
            hard = bool(layers["hard_blocked"][y][x]) or n("water") > .90 or n("slope") > .96 or n("building") > .96
            value = (.08 + n("slope") * weights["slope"] + n("tri") * weights["tri"]
                     + n("roughness_9") * weights["roughness_9"]
                     + n("landuse") * weights["landuse"] + n("building") * weights["building"]
                     + n("vegetation") * weights["vegetation"] + n("protected") * weights["protected"]
                     + n("seismic") * weights["seismic"] + n("landslide") * weights["landslide"]
                     + n("typhoon") * weights["typhoon"] + n("ice") * weights["ice"]
                     + n("lightning") * weights["lightning"]
                     + (1 - n("dist_water")) * weights["dist_water"]
                     + (1 - n("dist_fault")) * weights["dist_fault"]
                     + (1 - n("dist_railway")) * weights["dist_railway"]
                     - (1 - n("dist_road")) * weights["dist_road"])
            cost[y][x] = 100.0 if hard else max(.02, value)
    return cost


def path_cost(path, cost):
    total = 0.0
    for index, point in enumerate(path[1:], start=1):
        previous = path[index - 1]
        step = math.sqrt(2) if point["x"] != previous["x"] and point["y"] != previous["y"] else 1.0
        total += float(cost[point["y"]][point["x"]]) * step
    return total


def valid_route(path, observed, cost, start, goal):
    if not path or path[0] != start or path[-1] != goal:
        return False
    for index, point in enumerate(path):
        x, y = point["x"], point["y"]
        if not observed[y][x] or cost[y][x] is None or cost[y][x] >= 50:
            return False
        if index:
            previous = path[index - 1]
            dx, dy = x - previous["x"], y - previous["y"]
            if abs(dx) > 1 or abs(dy) > 1 or (dx == 0 and dy == 0):
                return False
            if dx and dy and (not observed[y][previous["x"]] or not observed[previous["y"]][x]
                              or cost[y][previous["x"]] is None or cost[previous["y"]][x] is None
                              or cost[y][previous["x"]] >= 50 or cost[previous["y"]][x] >= 50):
                return False
    return True


def run_policy(seed, width, height, active, goal_guided=True):
    page_id = f"benchmark-{seed}-{int(active)}-{int(goal_guided)}"
    world = create_world_session(seed, width, height, page_id=page_id,
                                 _world_seed_key=_BENCHMARK_WORLD_SEED_KEY)
    try:
        _get_session(world["session_id"])["scan_pattern_seed"] = seed
        return _run_policy(world, active, goal_guided)
    finally:
        release_world_session(world["session_id"], world["release_token"], page_id)


def _run_policy(world, active, goal_guided):
    truth = _get_session(world["session_id"])["truth"]
    truth_cost = build_cost(truth)
    oracle = plan_path([[True] * world["width"] for _ in range(world["height"])],
                       truth_cost, world["start"], world["goal"])
    observed = [[False] * world["width"] for _ in range(world["height"])]
    candidates = scan_pattern(world["width"], world["height"], world["start"])
    drone, first_connected = world["start"], None
    flight_distance = 0.0
    max_flight_leg = 0.0
    estimated_seconds = 0.0
    scan_legs = 0
    sensor_samples = 0
    replanned_at = 0
    result = {"path": [], "cost": None, "waiting_for_scan": True}

    while candidates or (active and not all(map(all, observed))):
        coverage = sum(map(sum, observed)) / (world["width"] * world["height"])
        if active:
            radius = round(200 / CELL_SIZE_METERS)
            choice = select_session_scan(world["session_id"], radius=radius,
                                         guidance_weight=clamp(.75 + coverage * 1.25, .75, 2.0)
                                         if goal_guided else 0.0,
                                         goal_guided=goal_guided)
            point = choice["target"]
            if point is None:
                break
            leg_id = choice["leg_id"]
            if point in candidates:
                candidates.remove(point)
        else:
            point = candidates.pop(0)
            leg_id = None

        measurement = sense_flight_segment(world["session_id"], drone, point,
                                           radius=round(200 / CELL_SIZE_METERS),
                                           leg_id=leg_id, require_authorization=active)
        for cell in measurement["cells"]:
            observed[cell["y"]][cell["x"]] = True
        leg_distance = math.hypot(point["x"] - drone["x"], point["y"] - drone["y"])
        flight_distance += leg_distance
        max_flight_leg = max(max_flight_leg, leg_distance)
        estimated_seconds += max(.09, leg_distance * (CELL_SIZE_METERS / 100) * .045) + .05
        drone = point
        scan_legs += 1
        sensor_samples += measurement["sample_centers"]
        coverage = sum(map(sum, observed)) / (world["width"] * world["height"])
        threshold = math.floor(coverage / .20)
        if threshold > replanned_at:
            replanned_at = threshold
            result = plan_session(world["session_id"], world["start"], world["goal"])
            if result["path"] and first_connected is None:
                first_true_route_cost = path_cost(result["path"], truth_cost)
                first_connected = {
                    "scan_legs": scan_legs,
                    "sensor_samples": sensor_samples,
                    "coverage": round(coverage, 4),
                    "flight_distance_cells": round(flight_distance, 2),
                    "estimated_route_cost": result["cost"],
                    "true_route_cost": round(first_true_route_cost, 3),
                    "true_route_regret_ratio": round(first_true_route_cost / oracle["cost"], 4)
                    if oracle["cost"] else None,
                    "route_nodes": len(result["path"]),
                }

    result = plan_session(world["session_id"], world["start"], world["goal"])
    stored = _get_session(world["session_id"])
    observed_cost = build_cost_grid(stored["layers"], truth["layer_weights"])
    route = result["path"]
    true_cost = path_cost(route, truth_cost) if route else None
    oracle_cost = oracle["cost"]
    return {
        "scan_legs": scan_legs,
        "sensor_samples": sensor_samples,
        "flight_distance_cells": round(flight_distance, 2),
        "max_flight_leg_cells": round(max_flight_leg, 2),
        "estimated_animation_seconds_at_1x": round(estimated_seconds, 2),
        "coverage": round(sum(map(sum, observed)) / (world["width"] * world["height"]), 4),
        "first_connected": first_connected,
        "estimated_route_cost": result["cost"],
        "true_route_cost": round(true_cost, 3) if true_cost is not None else None,
        "cost_regret_ratio": round(true_cost / oracle_cost, 4) if true_cost and oracle_cost else None,
        "route_found": bool(route),
        "unknown_or_obstacle_violation": (None if not route else
                                           not valid_route(route, observed, observed_cost,
                                                           world["start"], world["goal"])),
        "oracle_reachable": bool(oracle["path"]),
        "remaining_candidates": len(candidates),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--width", type=int, default=36)
    parser.add_argument("--height", type=int, default=28)
    parser.add_argument("--goal-progress-gain", type=float, default=None,
                        help="override the mission-corridor progress reward for sensitivity checks")
    args = parser.parse_args()
    if args.goal_progress_gain is not None:
        if not math.isfinite(args.goal_progress_gain) or args.goal_progress_gain < 0:
            parser.error("--goal-progress-gain must be a finite non-negative number")
        simulation_server.GOAL_PROGRESS_GAIN = args.goal_progress_gain
    results = []
    for seed in args.seeds:
        results.append({
            "seed": seed,
            "fixed_sweep": run_policy(seed, args.width, args.height, active=False),
            "information_gain_only": run_policy(seed, args.width, args.height,
                                                 active=True, goal_guided=False),
            "mission_corridor_scan": run_policy(seed, args.width, args.height,
                                                active=True, goal_guided=True),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
