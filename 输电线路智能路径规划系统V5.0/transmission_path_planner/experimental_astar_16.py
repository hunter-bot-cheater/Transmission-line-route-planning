"""Opt-in 16-neighbor A* for a bounded raster-search experiment.

The search objective is the exact piecewise-constant raster cost integral.
Optional ground-grade limits are hard edge constraints; soft grade and turn
penalties are intentionally excluded. This planner is not a production default.
"""

from __future__ import annotations

import heapq
import math
from itertools import count
from operator import index

import numpy as np

from .smoothing import _grid_cells_touched, _ground_grade_grid_edge_passes


NEIGHBORS_8 = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)
NEIGHBORS_16 = NEIGHBORS_8 + (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
)


def _grid_point(point):
    try:
        if len(point) != 2 or any(isinstance(value, bool) for value in point):
            raise ValueError
        return tuple(index(value) for value in point)
    except (TypeError, ValueError):
        raise ValueError("grid points must contain two integer indices") from None


def _validated_inputs(cost_raster, traversable_mask, pixel_spacing,
                      elevation, max_ground_grade):
    cost = np.asarray(cost_raster, dtype=np.float64)
    raw_mask = np.asarray(traversable_mask)
    if cost.ndim != 2 or raw_mask.shape != cost.shape:
        raise ValueError("cost_raster and traversable_mask must be same-shaped 2D arrays")
    if raw_mask.dtype != np.bool_ and not np.all(np.isin(raw_mask, (0, 1))):
        raise ValueError("traversable_mask must contain only booleans or 0/1 values")
    mask = raw_mask.astype(bool, copy=False) & np.isfinite(cost)
    if np.any(cost[mask] < 0.0):
        raise ValueError("finite traversable costs must be non-negative")

    spacing = tuple(map(float, pixel_spacing))
    if (len(spacing) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in spacing)):
        raise ValueError("pixel_spacing must contain positive finite row/column metres")
    if elevation is not None:
        elevation = np.asarray(elevation, dtype=np.float64)
        if elevation.shape != cost.shape:
            raise ValueError("elevation must have the same shape as cost_raster")
    if max_ground_grade is not None:
        if (not math.isfinite(float(max_ground_grade)) or max_ground_grade <= 0.0
                or elevation is None):
            raise ValueError("max_ground_grade requires a positive finite limit and elevation")
    return cost, mask, spacing, elevation


def _validated_grade_weight(value):
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError("grade_penalty_weight must be finite and non-negative") from None
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("grade_penalty_weight must be finite and non-negative")
    return weight


def _cell_center(row_col):
    row, col = row_col
    return float(col) + 0.5, float(row) + 0.5


def _segment_is_clear(start_rc, end_rc, traversable):
    height, width = traversable.shape
    cells = _grid_cells_touched(*_cell_center(start_rc), *_cell_center(end_rc))
    return all(0 <= row < height and 0 <= col < width and traversable[row, col]
               for row, col in cells)


def _segment_cost(start_rc, end_rc, cost, spacing):
    """Integrate raster cost exactly by splitting at every grid boundary."""
    c0, r0 = _cell_center(start_rc)
    c1, r1 = _cell_center(end_rc)
    dc, dr = c1 - c0, r1 - r0
    row_m, col_m = spacing
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
    cuts.sort()

    height, width = cost.shape
    length = math.hypot(dr * row_m, dc * col_m)
    total = 0.0
    for first, second in zip(cuts, cuts[1:]):
        middle = 0.5 * (first + second)
        row = math.floor(r0 + middle * dr)
        col = math.floor(c0 + middle * dc)
        if not (0 <= row < height and 0 <= col < width):
            return math.inf
        value = float(cost[row, col])
        if not math.isfinite(value):
            return math.inf
        total += value * length * (second - first)
    return total


def _segment_grade_penalty(start_rc, end_rc, elevation, spacing, weight):
    """Apply the production endpoint-grade formula to one search edge."""
    if weight == 0.0 or elevation is None:
        return 0.0
    start_elevation = float(elevation[start_rc])
    end_elevation = float(elevation[end_rc])
    if not math.isfinite(start_elevation) or not math.isfinite(end_elevation):
        return 0.0
    dr = (end_rc[0] - start_rc[0]) * spacing[0]
    dc = (end_rc[1] - start_rc[1]) * spacing[1]
    length = math.hypot(dr, dc)
    if length == 0.0:
        return 0.0
    grade = abs(end_elevation - start_elevation) / length
    return weight * min(grade / 0.30, 1.0) * length


def integrate_path_cost(path, cost_raster, traversable_mask, pixel_spacing=(1.0, 1.0),
                        neighbor_offsets=NEIGHBORS_16, elevation=None,
                        grade_penalty_weight=0.0):
    """Return exact raster integral plus an optional endpoint-grade penalty."""
    grade_penalty_weight = _validated_grade_weight(grade_penalty_weight)
    cost, traversable, spacing, _ = _validated_inputs(
        cost_raster, traversable_mask, pixel_spacing, elevation, None)
    points = [_grid_point(point) for point in path]
    if not points:
        return math.inf
    height, width = traversable.shape
    if any(not (0 <= row < height and 0 <= col < width and traversable[row, col])
           for row, col in points):
        return math.inf
    total = 0.0
    for start, end in zip(points, points[1:]):
        delta = (end[0] - start[0], end[1] - start[1])
        if delta not in neighbor_offsets or not _segment_is_clear(start, end, traversable):
            return math.inf
        total += (_segment_cost(start, end, cost, spacing)
                  + _segment_grade_penalty(
                      start, end, elevation, spacing, grade_penalty_weight))
    return total


def astar_exact_integral_path(cost_raster, traversable_mask, start_rc, goal_rc,
                              neighbor_offsets, pixel_spacing=(1.0, 1.0),
                              elevation=None, max_ground_grade=None,
                              grade_penalty_weight=0.0):
    """Search selected raster offsets using exact segment-integral cost.

    The search minimizes (integrated cost plus optional grade penalty, then
    length) lexicographically.
    The cost lower bound is consistent because each edge costs at least the
    minimum traversable raster cost times its length and the optional grade
    penalty is non-negative. The Euclidean length lower bound is consistent
    because edge length bounds endpoint displacement.
    """
    grade_penalty_weight = _validated_grade_weight(grade_penalty_weight)
    cost, traversable, spacing, elevation = _validated_inputs(
        cost_raster, traversable_mask, pixel_spacing, elevation, max_ground_grade)
    height, width = cost.shape
    start = _grid_point(start_rc)
    goal = _grid_point(goal_rc)
    for row, col in (start, goal):
        if not (0 <= row < height and 0 <= col < width):
            return None
    if not traversable[start] or not traversable[goal]:
        return None

    reverse_result = start[0] * width + start[1] > goal[0] * width + goal[1]
    if reverse_result:
        start, goal = goal, start
    if start == goal:
        return [start]

    minimum_cost = float(cost[traversable].min())

    def cost_heuristic(row, col):
        dr = (goal[0] - row) * spacing[0]
        dc = (goal[1] - col) * spacing[1]
        return minimum_cost * math.hypot(dr, dc)

    def length_heuristic(row, col):
        dr = (goal[0] - row) * spacing[0]
        dc = (goal[1] - col) * spacing[1]
        return math.hypot(dr, dc)

    def edge_cost(first, second):
        if not _segment_is_clear(first, second, traversable):
            return math.inf
        if (max_ground_grade is not None
                and not _ground_grade_grid_edge_passes(
                    first, second, elevation, spacing, max_ground_grade)):
            return math.inf
        return (_segment_cost(first, second, cost, spacing)
                + _segment_grade_penalty(
                    first, second, elevation, spacing, grade_penalty_weight))

    start_index = start[0] * width + start[1]
    goal_index = goal[0] * width + goal[1]
    scores = np.full((height, width), np.inf, dtype=np.float64)
    lengths = np.full((height, width), np.inf, dtype=np.float64)
    parents = np.full((height, width), -1, dtype=np.int64)
    closed = np.zeros((height, width), dtype=bool)
    scores[start] = 0.0
    lengths[start] = 0.0
    sequence = count()
    open_set = [(cost_heuristic(*start), length_heuristic(*start),
                 next(sequence), start_index, 0.0, 0.0)]

    while open_set:
        _, _, _, index, popped_cost, popped_length = heapq.heappop(open_set)
        row, col = divmod(index, width)
        if (popped_cost != scores[row, col]
                or popped_length != lengths[row, col]
                or closed[row, col]):
            continue
        if index == goal_index:
            path = []
            while index >= 0:
                path.append(divmod(index, width))
                if index == start_index:
                    break
                index = int(parents[row, col])
                if index < 0:
                    return None
                row, col = divmod(index, width)
            path.reverse()
            return path[::-1] if reverse_result else path

        closed[row, col] = True
        for dr, dc in neighbor_offsets:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width) or closed[nr, nc]:
                continue
            segment_cost = edge_cost((row, col), (nr, nc))
            if not math.isfinite(segment_cost):
                continue
            candidate = scores[row, col] + segment_cost
            step_length = math.hypot(dr * spacing[0], dc * spacing[1])
            candidate_length = lengths[row, col] + step_length
            if (candidate > scores[nr, nc]
                    or (candidate == scores[nr, nc]
                        and candidate_length >= lengths[nr, nc])):
                continue
            scores[nr, nc] = candidate
            lengths[nr, nc] = candidate_length
            parents[nr, nc] = row * width + col
            h_cost = cost_heuristic(nr, nc)
            h_length = length_heuristic(nr, nc)
            heapq.heappush(
                open_set,
                (candidate + h_cost, candidate_length + h_length,
                 next(sequence), nr * width + nc, candidate, candidate_length),
            )
    return None


def astar_16_neighbor_path(cost_raster, traversable_mask, start_rc, goal_rc,
                           pixel_spacing=(1.0, 1.0), elevation=None,
                           max_ground_grade=None):
    """Search with 8 adjacent and 8 knight offsets."""
    return astar_exact_integral_path(
        cost_raster, traversable_mask, start_rc, goal_rc, NEIGHBORS_16,
        pixel_spacing, elevation, max_ground_grade,
    )
