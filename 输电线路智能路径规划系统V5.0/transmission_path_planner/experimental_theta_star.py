"""Experimental cost-aware Theta* planner.

This module is intentionally not wired into the production CLI. It searches
cell-center vertices with 8-neighbor expansion while allowing parent-to-neighbor
any-angle links. Segment costs are exact integrals of a piecewise-constant raster
cost (cost per metre) under a fixed local row/column metric. Hard constraints use
the planner's shared conservative supercover raster test.

Scope: non-negative isotropic terrain costs and a traversability mask, with an
optional sampled ground-grade edge limit. There is no turn/curvature term,
tower/span model, endpoint snapping, or global optimality guarantee for
weighted Theta*.
"""

from __future__ import annotations

import heapq
import math
from itertools import count

import numpy as np

from .smoothing import _grid_cells_touched, _ground_grade_grid_edge_passes


_NEIGHBORS = (
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
)


def _validated_inputs(cost_raster, traversable_mask, pixel_spacing):
    cost = np.asarray(cost_raster, dtype=np.float64)
    raw_mask = np.asarray(traversable_mask)
    if cost.ndim != 2 or raw_mask.shape != cost.shape:
        raise ValueError("cost_raster and traversable_mask must be same-shaped 2D arrays")
    if raw_mask.dtype != np.bool_ and not np.all(np.isin(raw_mask, (0, 1))):
        raise ValueError("traversable_mask must contain only booleans or 0/1 values")
    traversable = raw_mask.astype(bool, copy=False) & np.isfinite(cost)
    if np.any(cost[traversable] < 0.0):
        raise ValueError("finite traversable costs must be non-negative")

    spacing = tuple(map(float, pixel_spacing))
    if (len(spacing) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in spacing)):
        raise ValueError("pixel_spacing must contain positive finite row/column metres")
    return cost, traversable, spacing


def _checked_cell(value, name):
    if len(value) != 2:
        raise ValueError(f"{name} must be a (row, column) pair")
    row, col = value
    try:
        row_index, col_index = int(row), int(col)
        if row_index != row or col_index != col:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} indices must be integers") from exc
    return row_index, col_index


def _cell_center(row_col):
    row, col = row_col
    return float(col) + 0.5, float(row) + 0.5


def _segment_is_clear(start_rc, end_rc, traversable):
    """Use the production supercover cell enumeration for a center-to-center chord."""
    start = (float(start_rc[0]) + 0.5, float(start_rc[1]) + 0.5)
    end = (float(end_rc[0]) + 0.5, float(end_rc[1]) + 0.5)
    return _segment_is_clear_positions(start, end, traversable)


def _segment_is_clear_positions(start_rc, end_rc, traversable):
    """Check a chord between continuous raster-corner row/column positions."""
    height, width = traversable.shape
    r0, c0 = map(float, start_rc)
    r1, c1 = map(float, end_rc)
    cells = _grid_cells_touched(c0, r0, c1, r1)
    return all(0 <= row < height and 0 <= col < width and traversable[row, col]
               for row, col in cells)


def _segment_cost(start_rc, end_rc, cost, pixel_spacing):
    """Integrate cost exactly over cell interiors crossed by a straight segment.

    The raster is piecewise constant per cell. The segment is split at each
    integer row/column boundary; each subsegment is charged the midpoint cell's
    cost times its physical length. Boundary-only contacts have zero measure
    for cost; they are still handled conservatively by _segment_is_clear().
    """
    first = (float(start_rc[0]) + 0.5, float(start_rc[1]) + 0.5)
    second = (float(end_rc[0]) + 0.5, float(end_rc[1]) + 0.5)
    return _segment_cost_positions(first, second, cost, pixel_spacing)


def _segment_cost_positions(start_rc, end_rc, cost, pixel_spacing,
                            segment_length_m=None):
    """Integrate piecewise-constant raster cost between continuous points."""
    r0, c0 = map(float, start_rc)
    r1, c1 = map(float, end_rc)
    dc, dr = c1 - c0, r1 - r0
    row_m, col_m = pixel_spacing

    cuts = [0.0, 1.0]
    for start, end in ((c0, c1), (r0, r1)):
        delta = end - start
        if delta == 0.0:
            continue
        low, high = sorted((start, end))
        for boundary in range(math.floor(low) + 1, math.ceil(high)):
            fraction = (boundary - start) / delta
            if 0.0 < fraction < 1.0:
                cuts.append(fraction)
    cuts = sorted(set(cuts))

    full_length = (math.hypot(dr * row_m, dc * col_m)
                   if segment_length_m is None else float(segment_length_m))
    if not math.isfinite(full_length) or full_length < 0.0:
        return math.inf
    total = 0.0
    height, width = cost.shape
    for first, second in zip(cuts, cuts[1:]):
        middle = 0.5 * (first + second)
        row = math.floor(r0 + middle * dr)
        col = math.floor(c0 + middle * dc)
        if not (0 <= row < height and 0 <= col < width):
            return math.inf
        value = float(cost[row, col])
        if not math.isfinite(value):
            return math.inf
        total += value * full_length * (second - first)
    return total


def integrate_polyline_cost(path, cost_raster, traversable_mask, pixel_spacing=(1.0, 1.0)):
    """Return exact piecewise-constant cost-meter integral for cell-center vertices.

    Returns infinity when any segment's conservative supercover touches a
    blocked/out-of-grid cell. A one-cell route has zero length and zero cost.
    """
    cost, traversable, spacing = _validated_inputs(
        cost_raster, traversable_mask, pixel_spacing)
    points = [_checked_cell(point, "path cell") for point in path]
    if not points:
        return math.inf
    height, width = traversable.shape
    if any(not (0 <= row < height and 0 <= col < width and traversable[row, col])
           for row, col in points):
        return math.inf

    total = 0.0
    for start, end in zip(points, points[1:]):
        if not _segment_is_clear(start, end, traversable):
            return math.inf
        total += _segment_cost(start, end, cost, spacing)
    return total


def theta_star_cost_path(cost_raster, traversable_mask, start_rc, goal_rc,
                         pixel_spacing=(1.0, 1.0), elevation=None,
                         max_ground_grade=None):
    """Find an experimental any-angle route over non-negative raster costs.

    Args:
        cost_raster: 2D cost-per-metre raster. Non-finite cells are blocked.
        traversable_mask: same-shaped mask where True/1 means traversable.
        start_rc, goal_rc: exact in-bounds cell indices; no snapping is performed.
        pixel_spacing: physical metres per row and column pixel.
        elevation: optional DEM used to enforce the ground-grade edge limit.
        max_ground_grade: optional positive grade ratio applied to every segment.

    Returns:
        A list of (row, column) cell-center vertices, or None when unreachable.
        The output is feasible under the supercover mask and uses the exact
        piecewise-constant segment integral, but Basic Theta* is not guaranteed
        globally optimal for weighted terrain.
    """
    cost, traversable, spacing = _validated_inputs(
        cost_raster, traversable_mask, pixel_spacing)
    height, width = traversable.shape
    if max_ground_grade is not None:
        try:
            max_ground_grade = float(max_ground_grade)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_ground_grade must be positive and finite") from exc
        if not math.isfinite(max_ground_grade) or max_ground_grade <= 0.0:
            raise ValueError("max_ground_grade must be positive and finite")
        if elevation is None:
            raise ValueError("elevation is required when max_ground_grade is set")
        elevation = np.asarray(elevation, dtype=np.float64)
        if elevation.shape != cost.shape:
            raise ValueError("elevation must match the 2D cost raster")
    elif elevation is not None:
        elevation = np.asarray(elevation, dtype=np.float64)
        if elevation.shape != cost.shape:
            raise ValueError("elevation must match the 2D cost raster")

    start = _checked_cell(start_rc, "start_rc")
    goal = _checked_cell(goal_rc, "goal_rc")
    for cell in (start, goal):
        row, col = cell
        if not (0 <= row < height and 0 <= col < width):
            return None
        if not traversable[row, col]:
            return None
    if start == goal:
        return [start]

    row_m, col_m = spacing
    finite_costs = cost[traversable]
    minimum_cost = float(finite_costs.min())

    def heuristic(row, col):
        dr = (goal[0] - row) * row_m
        dc = (goal[1] - col) * col_m
        return minimum_cost * math.hypot(dr, dc)

    def grade_passes(first, second):
        return (max_ground_grade is None or _ground_grade_grid_edge_passes(
            first, second, elevation, spacing, max_ground_grade))

    g_score = np.full((height, width), np.inf, dtype=np.float64)
    parent_row = np.full((height, width), -1, dtype=np.int32)
    parent_col = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=bool)
    g_score[start] = 0.0
    parent_row[start], parent_col[start] = start

    sequence = count()
    open_set = [(heuristic(*start), heuristic(*start), next(sequence),
                 start[0], start[1], 0.0)]

    while open_set:
        _, _, _, row, col, popped_g = heapq.heappop(open_set)
        if popped_g > g_score[row, col] + 1e-12 or closed[row, col]:
            continue
        if (row, col) == goal:
            route = []
            current = goal
            while current != start:
                route.append(current)
                current = (int(parent_row[current]), int(parent_col[current]))
                if current[0] < 0 or current[1] < 0:
                    return None
            route.append(start)
            route.reverse()
            return route
        closed[row, col] = True

        parent = (int(parent_row[row, col]), int(parent_col[row, col]))
        for dr, dc in _NEIGHBORS:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if closed[nr, nc] or not traversable[nr, nc]:
                continue

            candidate_parent = None
            candidate_g = math.inf

            # Unlike geometric Theta*, a visible long chord can cross expensive
            # cells. Compare it with the adjacent predecessor edge every time.
            if (_segment_is_clear(parent, neighbor, traversable)
                    and grade_passes(parent, neighbor)):
                direct_g = (g_score[parent]
                            + _segment_cost(parent, neighbor, cost, spacing))
                candidate_parent, candidate_g = parent, direct_g

            current = (row, col)
            if (_segment_is_clear(current, neighbor, traversable)
                    and grade_passes(current, neighbor)):
                via_g = (g_score[row, col]
                         + _segment_cost(current, neighbor, cost, spacing))
                if via_g < candidate_g:
                    candidate_parent, candidate_g = current, via_g

            if candidate_parent is None or candidate_g + 1e-12 >= g_score[nr, nc]:
                continue
            g_score[nr, nc] = candidate_g
            parent_row[nr, nc], parent_col[nr, nc] = candidate_parent
            h = heuristic(nr, nc)
            heapq.heappush(open_set, (candidate_g + h, h, next(sequence),
                                      nr, nc, candidate_g))
    return None
