"""Experimental generator for spatially diverse raster route candidates.

It can use the production A* terrain-grade and turn objective, but does not
model towers, span feasibility, conductor clearance, or historical routes.
"""

import math
from numbers import Integral

import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring, unary_union

from .methods import astar_search


def _distance_to_mask(mask, spacing):
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - optional dependency fallback
        raise RuntimeError("scipy is required for route separation distances") from exc
    return distance_transform_edt(~mask, sampling=spacing)


def _route_mask(path, shape):
    mask = np.zeros(shape, dtype=bool)
    for row, col in path:
        mask[row, col] = True
    return mask


def _production_objective(path, cost, elevation, spacing, grade_penalty_weight,
                          turn_penalty_weight):
    from .metrics import astar_path_objective

    result = astar_path_objective(
        path, cost, elevation, spacing, grade_penalty_weight, turn_penalty_weight)
    if result["status"] != "measured":
        raise ValueError(f"candidate path has no valid production objective: {result['status']}")
    return float(result["total_cost"])


def _hard_violations(path, hard_mask):
    violations = 0
    height, width = hard_mask.shape
    for index, (row, col) in enumerate(path):
        if not (0 <= row < height and 0 <= col < width) or not hard_mask[row, col]:
            violations += 1
            continue
        if index == 0:
            continue
        pr, pc = path[index - 1]
        dr, dc = row - pr, col - pc
        if max(abs(dr), abs(dc)) != 1:
            violations += 1
        elif dr and dc and (not hard_mask[pr + dr, pc] or not hard_mask[pr, pc + dc]):
            violations += 1
    return violations


def _grid_point(value, name):
    try:
        if len(value) != 2:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer (row, col) pair") from None
    point = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral):
            raise ValueError(f"{name} must be an integer (row, col) pair")
        point.append(int(item))
    return tuple(point)


def _interior_line(path, spacing, endpoint_buffer_m):
    if len(path) < 2:
        return None
    line = LineString([(col * spacing[1], row * spacing[0]) for row, col in path])
    if line.length <= 2.0 * endpoint_buffer_m:
        return None
    return substring(line, endpoint_buffer_m, line.length - endpoint_buffer_m)


def _sample_line_distances(source, target, sample_spacing_m):
    count = max(1, math.ceil(source.length / sample_spacing_m))
    return np.asarray([
        target.distance(source.interpolate((index + 0.5) * source.length / count))
        for index in range(count)
    ], dtype=np.float64)


def _separation_metrics(candidate, selected, spacing, endpoint_buffer_m,
                        minimum_separation_m):
    candidate_line = _interior_line(candidate, spacing, endpoint_buffer_m)
    selected_lines = [line for path in selected
                      if (line := _interior_line(path, spacing, endpoint_buffer_m)) is not None]
    if candidate_line is None or not selected_lines:
        return None
    selected_line = unary_union(selected_lines)
    sample_spacing = min(spacing)
    forward = _sample_line_distances(candidate_line, selected_line, sample_spacing)
    reverse = _sample_line_distances(selected_line, candidate_line, sample_spacing)
    distances = np.concatenate((forward, reverse))
    minimum_distance = min(float(np.min(distances)), float(candidate_line.distance(selected_line)))
    candidate_fraction = float(np.mean(forward >= minimum_separation_m))
    selected_fraction = float(np.mean(reverse >= minimum_separation_m))
    return {
        "mean_separation_m": float(np.mean(distances)),
        "min_separation_m": minimum_distance,
        "candidate_separated_fraction": candidate_fraction,
        "selected_separated_fraction": selected_fraction,
        "symmetric_separated_fraction": min(candidate_fraction, selected_fraction),
    }


def diverse_astar_routes(
    cost_raster,
    hard_mask,
    start_rc,
    end_rc,
    *,
    pixel_spacing,
    max_paths=3,
    penalty_radius_m=None,
    distance_penalty_weight=10.0,
    minimum_separation_m=None,
    minimum_separated_fraction=0.7,
    endpoint_buffer_m=None,
    max_penalty_retries=3,
    penalty_growth=2.0,
    elevation=None,
    pixel_m=None,
    grade_penalty_weight=1.0,
    max_ground_grade=None,
    heuristic_weight=1.0,
    turn_penalty_weight=0.0,
):
    """Return up to ``max_paths`` legal A* routes with spatial separation.

    The next search adds a linear distance-decay cost around selected paths.
    A candidate is accepted only if at least ``minimum_separated_fraction`` of
    both route interiors lie at least ``minimum_separation_m`` away. Distances
    are measured to the route line segments, so diagonal crossings count as
    zero separation. Endpoint buffers are removed before the comparison.

    ``pixel_spacing`` is required to keep all physical distances in meters.
    Optional production A* grade and turn parameters are passed through unchanged.
    A retry adds its distance-decay penalty to the raster before rerunning A*;
    turn costs during that search therefore use perturbed cell costs. Results
    report both the unperturbed baseline and the objective used for each search.
    This remains an experimental corridor generator, not an engineering
    feasibility or historical-route matching guarantee.
    """
    cost = np.asarray(cost_raster, dtype=np.float64)
    raw_mask = np.asarray(hard_mask)
    if cost.ndim != 2 or raw_mask.shape != cost.shape:
        raise ValueError("cost_raster and hard_mask must be same-shape 2D arrays")
    if raw_mask.dtype != np.bool_:
        if (not np.issubdtype(raw_mask.dtype, np.number)
                or not np.all(np.isfinite(raw_mask))
                or not np.all(np.isin(raw_mask, (0, 1)))):
            raise ValueError("hard_mask must contain only boolean or 0/1 values")
    traversable = raw_mask.astype(bool, copy=False)
    if elevation is not None:
        elevation = np.asarray(elevation, dtype=np.float64)
        if elevation.shape != cost.shape:
            raise ValueError("elevation must have the same shape as cost_raster")
    spacing = tuple(map(float, pixel_spacing))
    if len(spacing) != 2 or any(not math.isfinite(x) or x <= 0 for x in spacing):
        raise ValueError("pixel_spacing must contain two finite positive values")
    if (isinstance(max_paths, bool) or not isinstance(max_paths, Integral) or max_paths < 1
            or isinstance(max_penalty_retries, bool)
            or not isinstance(max_penalty_retries, Integral) or max_penalty_retries < 1):
        raise ValueError("max_paths and max_penalty_retries must be positive integers")
    if not math.isfinite(float(minimum_separated_fraction)) or not (
            0 < minimum_separated_fraction <= 1):
        raise ValueError("minimum_separated_fraction must be in (0, 1]")
    weights = (grade_penalty_weight, heuristic_weight, turn_penalty_weight)
    if any(not math.isfinite(float(value)) for value in weights):
        raise ValueError("A* weights must be finite")
    if float(grade_penalty_weight) < 0 or float(heuristic_weight) < 1 \
            or float(turn_penalty_weight) < 0:
        raise ValueError("A* weights must be non-negative and heuristic_weight at least 1")
    if max_ground_grade is not None and (
            not math.isfinite(float(max_ground_grade)) or float(max_ground_grade) <= 0):
        raise ValueError("max_ground_grade must be finite and positive")

    penalty_radius_m = (2.0 * max(spacing) if penalty_radius_m is None
                        else float(penalty_radius_m))
    minimum_separation_m = (1.5 * min(spacing) if minimum_separation_m is None
                            else float(minimum_separation_m))
    endpoint_buffer_m = (2.0 * max(spacing) if endpoint_buffer_m is None
                         else float(endpoint_buffer_m))
    numeric = (penalty_radius_m, distance_penalty_weight,
               minimum_separation_m, endpoint_buffer_m, penalty_growth)
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("penalty and separation parameters must be finite")
    if (penalty_radius_m <= 0 or distance_penalty_weight < 0
            or minimum_separation_m <= 0 or endpoint_buffer_m < 0 or penalty_growth < 1):
        raise ValueError("invalid penalty or separation parameter")

    usable = traversable & np.isfinite(cost)
    if np.any(cost[usable] < 0):
        raise ValueError("cost_raster must be non-negative on traversable cells")
    start = _grid_point(start_rc, "start_rc")
    end = _grid_point(end_rc, "end_rc")
    for point in (start, end):
        if (not (0 <= point[0] < cost.shape[0] and 0 <= point[1] < cost.shape[1])
                or not usable[point]):
            return []

    base_cost = np.where(usable, cost, np.inf)

    def search(surface):
        return astar_search(
            surface, start, end, elevation=elevation, pixel_m=pixel_m,
            grade_penalty_weight=grade_penalty_weight,
            max_ground_grade=max_ground_grade, pixel_spacing=spacing,
            heuristic_weight=heuristic_weight,
            turn_penalty_weight=turn_penalty_weight,
        )

    first = search(base_cost)
    if not first or _hard_violations(first, usable):
        return []
    selected = [first]

    def record(path, separation=None, search_weight=0.0, search_cost_surface=None):
        search_cost_surface = (base_cost if search_cost_surface is None
                               else search_cost_surface)
        item = {
            "path": [tuple(map(int, point)) for point in path],
            "baseline_objective": _production_objective(
                path, base_cost, elevation, spacing,
                grade_penalty_weight, turn_penalty_weight),
            "baseline_objective_unit": "cost_meter",
            "baseline_objective_model": "production_astar_unperturbed_cost_grade_turn",
            "search_objective": _production_objective(
                path, search_cost_surface, elevation, spacing,
                grade_penalty_weight, turn_penalty_weight),
            "search_objective_unit": "cost_meter",
            "search_objective_model": (
                "production_astar_distance_perturbed_cost_grade_turn"
                if search_weight else "production_astar_unperturbed_cost_grade_turn"
            ),
            "hard_constraint_legal": True,
            "hard_violation_count": 0,
            "search_distance_penalty_weight": float(search_weight),
            "mean_separation_from_selected_m": None,
            "min_separation_from_selected_m": None,
            "symmetric_separated_fraction": None,
        }
        if separation is not None:
            item["mean_separation_from_selected_m"] = separation["mean_separation_m"]
            item["min_separation_from_selected_m"] = separation["min_separation_m"]
            item["candidate_separated_fraction"] = separation["candidate_separated_fraction"]
            item["selected_separated_fraction"] = separation["selected_separated_fraction"]
            item["symmetric_separated_fraction"] = separation["symmetric_separated_fraction"]
        return item

    results = [record(first)]
    while len(selected) < max_paths:
        selected_mask = np.zeros(cost.shape, dtype=bool)
        for path in selected:
            selected_mask |= _route_mask(path, cost.shape)
        distance = _distance_to_mask(selected_mask, spacing)
        proximity = np.clip(1.0 - distance / penalty_radius_m, 0.0, 1.0)
        accepted = None
        last_weight = None
        for retry in range(max_penalty_retries):
            weight = float(distance_penalty_weight) * float(penalty_growth) ** retry
            if weight == last_weight:
                break
            last_weight = weight
            search_cost = np.where(usable, base_cost + weight * proximity, np.inf)
            candidate = search(search_cost)
            if not candidate or _hard_violations(candidate, usable):
                break
            separation = _separation_metrics(
                candidate, selected, spacing, endpoint_buffer_m, minimum_separation_m)
            if (separation is not None and separation["symmetric_separated_fraction"]
                    >= minimum_separated_fraction):
                accepted = (candidate, separation, weight, search_cost)
                break
        if accepted is None:
            break
        path, separation, weight, search_cost = accepted
        selected.append(path)
        results.append(record(path, separation, weight, search_cost))
    return results
