"""Experimental station-to-station routing over candidate tower sites.

This is a screening model, not an overhead-line design tool. It separates
per-metre span resistance from per-tower site cost and enforces horizontal span
length plus raster traversability. It does not model conductors, sag, clearance,
load cases, permits, or engineering tower-spacing standards.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from itertools import count

import numpy as np

from .experimental_theta_star import (
    _checked_cell,
    _segment_cost,
    _segment_is_clear,
    _validated_inputs,
)


def tower_graph_route(cost_raster, traversable_mask, tower_allowed_mask,
                      start_rc, goal_rc, min_span_m, max_span_m,
                      pixel_spacing=(1.0, 1.0), candidate_stride=2,
                      tower_cost_raster=None, turn_penalty_weight=0.0):
    """Find a route on a sparse graph of candidate tower cells.

    ``cost_raster`` is a non-negative per-metre span resistance. Optional
    ``tower_cost_raster`` contains per-site penalties in compatible objective
    units. Start and goal are fixed terminals and do not need to pass the tower
    siting mask. ``turn_penalty_weight`` is a dimensionless multiplier. Invalid
    terminal cells are rejected; no snapping is performed.
    """
    cost, traversable, spacing = _validated_inputs(
        cost_raster, traversable_mask, pixel_spacing)
    tower_allowed = np.asarray(tower_allowed_mask)
    if tower_allowed.shape != cost.shape or tower_allowed.ndim != 2:
        raise ValueError("tower_allowed_mask must match the 2D cost raster")
    if tower_allowed.dtype != np.bool_ and not np.all(np.isin(tower_allowed, (0, 1))):
        raise ValueError("tower_allowed_mask must contain only booleans or 0/1 values")
    tower_allowed = tower_allowed.astype(bool, copy=False)

    try:
        stride = int(candidate_stride)
        if stride != candidate_stride or stride < 1:
            raise ValueError
        min_span = float(min_span_m)
        max_span = float(max_span_m)
        turn_weight = float(turn_penalty_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid span, stride, or turn-penalty parameter") from exc
    if (not math.isfinite(min_span) or not math.isfinite(max_span)
            or min_span < 0.0 or max_span <= min_span):
        raise ValueError("span limits must be finite with 0 <= min < max")
    if not math.isfinite(turn_weight) or turn_weight < 0.0:
        raise ValueError("turn_penalty_weight must be finite and non-negative")

    height, width = cost.shape
    start = _checked_cell(start_rc, "start_rc")
    goal = _checked_cell(goal_rc, "goal_rc")
    if tower_cost_raster is None:
        tower_cost = np.zeros(cost.shape, dtype=np.float64)
    else:
        tower_cost = np.asarray(tower_cost_raster, dtype=np.float64)
        if tower_cost.shape != cost.shape or tower_cost.ndim != 2:
            raise ValueError("tower_cost_raster must match the 2D cost raster")
        site_mask = tower_allowed & traversable
        for row, col in (start, goal):
            if 0 <= row < height and 0 <= col < width:
                site_mask[row, col] = False
        site_values = tower_cost[site_mask]
        if np.any(~np.isfinite(site_values)) or np.any(site_values < 0.0):
            raise ValueError("allowed tower sites must have finite non-negative costs")

    for terminal in (start, goal):
        row, col = terminal
        if not (0 <= row < height and 0 <= col < width and traversable[row, col]):
            return None
    if start == goal:
        return [start]

    reversed_result = start[0] * width + start[1] > goal[0] * width + goal[1]
    if reversed_result:
        start, goal = goal, start

    cells = {start, goal}
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            if tower_allowed[row, col] and traversable[row, col]:
                cells.add((row, col))
    nodes = sorted(cells)
    node_id = {cell: index for index, cell in enumerate(nodes)}
    start_id, goal_id = node_id[start], node_id[goal]

    rows_to_cols = {}
    for index, (row, col) in enumerate(nodes):
        rows_to_cols.setdefault(row, []).append((col, index))
    candidate_rows = sorted(rows_to_cols)
    row_m, col_m = spacing
    row_radius = max_span / row_m
    col_radius = max_span / col_m
    adjacency = [[] for _ in nodes]
    edge_cost = {} if turn_weight > 0.0 else None

    for first_id, (row, col) in enumerate(nodes):
        first_row = bisect_left(candidate_rows, row - row_radius)
        last_row = bisect_right(candidate_rows, row + row_radius)
        for next_row in candidate_rows[first_row:last_row]:
            if abs(next_row - row) * row_m > max_span:
                continue
            row_candidates = rows_to_cols[next_row]
            left = bisect_left(row_candidates, (col - col_radius, -1))
            right = bisect_right(row_candidates, (col + col_radius, len(nodes)))
            for next_col, second_id in row_candidates[left:right]:
                if second_id <= first_id:
                    continue
                distance = math.hypot((next_row - row) * row_m,
                                      (next_col - col) * col_m)
                if distance < min_span or distance > max_span:
                    continue
                second = (next_row, next_col)
                if not _segment_is_clear((row, col), second, traversable):
                    continue
                span_cost = _segment_cost((row, col), second, cost, spacing)
                if not math.isfinite(span_cost):
                    continue
                adjacency[first_id].append((second_id, span_cost))
                adjacency[second_id].append((first_id, span_cost))
                if edge_cost is not None:
                    edge_cost[first_id, second_id] = span_cost
                    edge_cost[second_id, first_id] = span_cost

    if turn_weight == 0.0:
        best_cost = {start_id: 0.0}
        parent = {}
        sequence = count()
        open_set = [(0.0, next(sequence), start_id)]
        while open_set:
            current_cost, _, current_id = heapq.heappop(open_set)
            if current_cost != best_cost[current_id]:
                continue
            if current_id == goal_id:
                route_ids = []
                while current_id != start_id:
                    route_ids.append(current_id)
                    current_id = parent[current_id]
                route_ids.append(start_id)
                route_ids.reverse()
                route = [nodes[index] for index in route_ids]
                return route[::-1] if reversed_result else route

            for next_id, span_cost in adjacency[current_id]:
                added_cost = span_cost
                if next_id != goal_id and next_id != start_id:
                    next_row, next_col = nodes[next_id]
                    added_cost += float(tower_cost[next_row, next_col])
                candidate_cost = current_cost + added_cost
                if candidate_cost >= best_cost.get(next_id, math.inf):
                    continue
                best_cost[next_id] = candidate_cost
                parent[next_id] = current_id
                heapq.heappush(open_set,
                               (candidate_cost, next(sequence), next_id))
        return None

    start_state = (-1, start_id)
    best = {start_state: 0.0}
    parent = {}
    sequence = count()
    open_set = [(0.0, next(sequence), start_state)]
    while open_set:
        current_cost, _, state = heapq.heappop(open_set)
        if current_cost != best[state]:
            continue
        previous_id, current_id = state
        if current_id == goal_id:
            route_ids = []
            while state != start_state:
                route_ids.append(state[1])
                state = parent[state]
            route_ids.append(start_id)
            route_ids.reverse()
            route = [nodes[index] for index in route_ids]
            return route[::-1] if reversed_result else route

        for next_id, span_cost in adjacency[current_id]:
            # A tower span cannot be used as an immediate out-and-back detour
            # to manipulate the direction penalty.
            if next_id == previous_id:
                continue
            added_cost = span_cost
            if next_id != goal_id and next_id != start_id:
                next_row, next_col = nodes[next_id]
                added_cost += float(tower_cost[next_row, next_col])
            if previous_id >= 0:
                previous, current, following = (
                    nodes[previous_id], nodes[current_id], nodes[next_id])
                first_vector = ((current[0] - previous[0]) * row_m,
                                (current[1] - previous[1]) * col_m)
                second_vector = ((following[0] - current[0]) * row_m,
                                 (following[1] - current[1]) * col_m)
                first_length = math.hypot(*first_vector)
                second_length = math.hypot(*second_vector)
                cosine = ((first_vector[0] * second_vector[0]
                           + first_vector[1] * second_vector[1])
                          / (first_length * second_length))
                turn_cost = (turn_weight
                             * 0.5 * (edge_cost[previous_id, current_id] + span_cost)
                             * (1.0 - min(max(cosine, -1.0), 1.0)))
                added_cost += turn_cost

            next_state = (current_id, next_id)
            candidate_cost = current_cost + added_cost
            if candidate_cost >= best.get(next_state, math.inf):
                continue
            best[next_state] = candidate_cost
            parent[next_state] = state
            heapq.heappush(open_set,
                           (candidate_cost, next(sequence), next_state))
    return None
