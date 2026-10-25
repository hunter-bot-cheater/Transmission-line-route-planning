"""Experimental station-to-station routing over candidate tower sites.

This is a screening model, not an overhead-line design tool. It separates
per-metre span resistance from per-tower site cost and enforces horizontal span
length plus raster traversability. An optional DEM profile screen uses caller-
supplied attachment heights, a parabolic sag bound, and minimum clearance; it
does not model conductor-specific catenaries, load cases, permits, or validated
engineering tower-spacing standards.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left
from itertools import count

import numpy as np

from .experimental_theta_star import (
    _checked_cell,
    _segment_cost_positions,
    _segment_is_clear_positions,
    _validated_inputs,
)


def _checked_position(value, name):
    if len(value) != 2:
        raise ValueError(f"{name} must be a (row, column) pair")
    try:
        position = tuple(map(float, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite coordinates") from exc
    if not all(math.isfinite(coordinate) for coordinate in position):
        raise ValueError(f"{name} must contain finite coordinates")
    return position


def _dem_data_and_validity(dem_raster, dem_valid_mask=None,
                           dem_nodata_value=None):
    masked_dem = np.ma.asarray(dem_raster, dtype=np.float64)
    dem = np.asarray(masked_dem.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(masked_dem) & np.isfinite(dem)
    if dem_valid_mask is not None:
        masked_valid = np.ma.asarray(dem_valid_mask)
        raw_valid = np.asarray(masked_valid.data)
        mask_values = np.ma.getmaskarray(masked_valid)
        if (raw_valid.shape != dem.shape
                or (raw_valid.dtype != np.bool_
                    and not np.all(np.isin(raw_valid[~mask_values], (0, 1))))):
            raise ValueError("dem_valid_mask must match the DEM and contain booleans or 0/1")
        valid &= ~mask_values & raw_valid.astype(bool, copy=False)
    if dem_nodata_value is not None:
        try:
            nodata_value = float(dem_nodata_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dem_nodata_value must be a finite number") from exc
        if not math.isfinite(nodata_value):
            raise ValueError("dem_nodata_value must be a finite number")
        valid &= dem != nodata_value
    return dem, valid


def tower_graph_route(cost_raster, traversable_mask, tower_allowed_mask,
                      start_rc, goal_rc, min_span_m, max_span_m,
                      pixel_spacing=(1.0, 1.0), candidate_stride=2,
                      tower_cost_raster=None, turn_penalty_weight=0.0,
                      dem_raster=None, attachment_height_m=None,
                      max_midspan_sag_m=None, minimum_clearance_m=None,
                      dem_valid_mask=None, dem_nodata_value=None,
                      start_ground_elevation_m=None,
                      goal_ground_elevation_m=None,
                      start_position_rc=None, goal_position_rc=None):
    """Find a route on a sparse graph of candidate tower cells.

    ``cost_raster`` is a non-negative per-metre span resistance. Optional
    ``tower_cost_raster`` contains per-site penalties in compatible objective
    units. Start and goal are fixed terminals and do not need to pass the tower
    siting mask. ``turn_penalty_weight`` is a dimensionless multiplier. Invalid
    terminal cells are rejected; no snapping is performed. Optional continuous
    terminal positions use cell-center row/column coordinates and must remain in
    their declared terminal cells. If all four profile
    inputs are supplied, each span also needs a complete DEM profile and must
    pass the parabolic-sag clearance screen. Missing profile values reject that
    span; no profile inputs means the result remains a 2D screening route.
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

    profile_values = (dem_raster, attachment_height_m,
                      max_midspan_sag_m, minimum_clearance_m)
    profile_enabled = any(value is not None for value in profile_values)
    if not profile_enabled and (dem_valid_mask is not None or dem_nodata_value is not None):
        raise ValueError("DEM validity inputs require profile screening")
    if profile_enabled and any(value is None for value in profile_values):
        raise ValueError("DEM, attachment height, sag, and clearance must be supplied together")
    if profile_enabled:
        dem, dem_valid = _dem_data_and_validity(
            dem_raster, dem_valid_mask, dem_nodata_value)
        if dem.ndim != 2 or dem.shape != cost.shape:
            raise ValueError("dem_raster must match the 2D cost raster")
        try:
            attachment_height = float(attachment_height_m)
            sag_limit = float(max_midspan_sag_m)
            clearance_limit = float(minimum_clearance_m)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid span profile parameter") from exc
        if (not math.isfinite(attachment_height) or attachment_height < 0.0
                or not math.isfinite(sag_limit) or sag_limit < 0.0
                or not math.isfinite(clearance_limit) or clearance_limit < 0.0):
            raise ValueError("profile heights, sag, and clearance must be finite and non-negative")
        terminal_ground_elevations = (start_ground_elevation_m,
                                      goal_ground_elevation_m)
        for value in terminal_ground_elevations:
            if value is not None:
                try:
                    valid_elevation = math.isfinite(float(value))
                except (TypeError, ValueError, OverflowError):
                    valid_elevation = False
                if not valid_elevation:
                    raise ValueError("terminal ground elevations must be finite numbers")
    else:
        dem = None
        dem_valid = None
        attachment_height = None
        sag_limit = clearance_limit = None
        if start_ground_elevation_m is not None or goal_ground_elevation_m is not None:
            raise ValueError("terminal ground elevations require profile screening")

    height, width = cost.shape
    start = _checked_cell(start_rc, "start_rc")
    goal = _checked_cell(goal_rc, "goal_rc")
    start_node = (_checked_position(start_position_rc, "start_position_rc")
                  if start_position_rc is not None else start)
    goal_node = (_checked_position(goal_position_rc, "goal_position_rc")
                 if goal_position_rc is not None else goal)
    for cell, position in ((start, start_node), (goal, goal_node)):
        if (math.floor(position[0] + 0.5), math.floor(position[1] + 0.5)) != cell:
            raise ValueError("terminal position must lie within its declared terminal cell")
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
    if start_node == goal_node:
        return [start_node]

    reversed_result = start_node[0] * width + start_node[1] > goal_node[0] * width + goal_node[1]
    if reversed_result:
        start, goal = goal, start
        start_node, goal_node = goal_node, start_node
        start_ground_elevation_m, goal_ground_elevation_m = (
            goal_ground_elevation_m, start_ground_elevation_m)

    candidate_rows, candidate_cols = np.nonzero(
        tower_allowed[::stride, ::stride] & traversable[::stride, ::stride])
    candidate_nodes = np.column_stack((candidate_rows * stride,
                                       candidate_cols * stride))
    nodes = candidate_nodes.astype(np.float64)
    for terminal in sorted((start_node, goal_node)):
        first = int(np.searchsorted(nodes[:, 0], terminal[0], side="left"))
        last = int(np.searchsorted(nodes[:, 0], terminal[0], side="right"))
        offset = int(np.searchsorted(nodes[first:last, 1], terminal[1], side="left"))
        node_index = first + offset
        if node_index >= last or not np.array_equal(nodes[node_index], terminal):
            nodes = np.insert(nodes, node_index, terminal, axis=0)

    def find_node_id(position):
        first = int(np.searchsorted(nodes[:, 0], position[0], side="left"))
        last = int(np.searchsorted(nodes[:, 0], position[0], side="right"))
        offset = int(np.searchsorted(nodes[first:last, 1], position[1], side="left"))
        node_index = first + offset
        if node_index >= last or not np.array_equal(nodes[node_index], position):
            raise RuntimeError("terminal position is missing from tower graph")
        return node_index

    start_id, goal_id = find_node_id(start_node), find_node_id(goal_node)
    row_starts = np.concatenate((
        np.array([0]), np.flatnonzero(nodes[1:, 0] != nodes[:-1, 0]) + 1))
    candidate_rows = nodes[row_starts, 0]
    row_offsets = np.append(row_starts, len(nodes))

    def route_coordinate(node_index):
        if node_index == start_id:
            return start_node
        if node_index == goal_id:
            return goal_node
        row, col = nodes[node_index]
        return int(row), int(col)
    row_m, col_m = spacing
    row_radius = max_span / row_m
    col_radius = max_span / col_m
    adjacency = {}

    def get_neighbors(first_id):
        if first_id in adjacency:
            return adjacency[first_id]

        neighbors = []
        row, col = nodes[first_id]
        first_row = np.searchsorted(candidate_rows, row - row_radius, side="left")
        last_row = np.searchsorted(candidate_rows, row + row_radius, side="right")
        for row_index in range(first_row, last_row):
            next_row = candidate_rows[row_index]
            if abs(next_row - row) * row_m > max_span:
                continue
            row_start, row_end = row_offsets[row_index:row_index + 2]
            row_candidates = nodes[row_start:row_end, 1]
            left = np.searchsorted(row_candidates, col - col_radius, side="left")
            right = np.searchsorted(row_candidates, col + col_radius, side="right")
            for second_id in range(row_start + left, row_start + right):
                if second_id == first_id:
                    continue
                edge_first_id = min(first_id, second_id)
                edge_second_id = max(first_id, second_id)
                first, second = nodes[edge_first_id], nodes[edge_second_id]
                distance = math.hypot((second[0] - first[0]) * row_m,
                                      (second[1] - first[1]) * col_m)
                if distance < min_span or distance > max_span:
                    continue
                first_position = (first[0] + 0.5, first[1] + 0.5)
                second_position = (second[0] + 0.5, second[1] + 0.5)
                if not _segment_is_clear_positions(first_position, second_position,
                                                   traversable):
                    continue
                if profile_enabled:
                    first_ground = (start_ground_elevation_m if edge_first_id == start_id
                                    else goal_ground_elevation_m if edge_first_id == goal_id
                                    else None)
                    second_ground = (start_ground_elevation_m if edge_second_id == start_id
                                     else goal_ground_elevation_m if edge_second_id == goal_id
                                     else None)
                    profile = span_clearance_status(
                        first, second, dem, spacing, attachment_height,
                        sag_limit, clearance_limit, dem_valid_mask=dem_valid,
                        start_ground_elevation_m=first_ground,
                        end_ground_elevation_m=second_ground)
                    if profile["status"] != "sampled_clearance_ok":
                        continue
                span_cost = _segment_cost_positions(
                    first_position, second_position, cost, spacing)
                if not math.isfinite(span_cost):
                    continue
                neighbors.append((second_id, span_cost))

        adjacency[first_id] = neighbors
        return neighbors

    minimum_cost = float(cost[traversable].min())

    def heuristic(node_id):
        row, col = nodes[node_id]
        goal_row, goal_col = nodes[goal_id]
        distance = math.hypot((goal_row - row) * row_m,
                              (goal_col - col) * col_m)
        return minimum_cost * distance

    if turn_weight == 0.0:
        best_cost = {start_id: 0.0}
        parent = {}
        sequence = count()
        open_set = [(heuristic(start_id), 0.0, next(sequence), start_id)]
        while open_set:
            _, current_cost, _, current_id = heapq.heappop(open_set)
            if current_cost != best_cost[current_id]:
                continue
            if current_id == goal_id:
                route_ids = []
                while current_id != start_id:
                    route_ids.append(current_id)
                    current_id = parent[current_id]
                route_ids.append(start_id)
                route_ids.reverse()
                route = [route_coordinate(index) for index in route_ids]
                return route[::-1] if reversed_result else route

            for next_id, span_cost in get_neighbors(current_id):
                added_cost = span_cost
                if next_id != goal_id and next_id != start_id:
                    next_row, next_col = nodes[next_id]
                    added_cost += float(tower_cost[int(next_row), int(next_col)])
                candidate_cost = current_cost + added_cost
                if candidate_cost >= best_cost.get(next_id, math.inf):
                    continue
                best_cost[next_id] = candidate_cost
                parent[next_id] = current_id
                estimated_total = candidate_cost + heuristic(next_id)
                heapq.heappush(open_set, (estimated_total, candidate_cost,
                                          next(sequence), next_id))
        return None

    start_state = (-1, start_id)
    best = {start_state: 0.0}
    parent = {}
    sequence = count()
    open_set = [(heuristic(start_id), 0.0, next(sequence), start_state)]
    while open_set:
        _, current_cost, _, state = heapq.heappop(open_set)
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
            route = [route_coordinate(index) for index in route_ids]
            return route[::-1] if reversed_result else route

        neighbors = get_neighbors(current_id)
        previous_span_cost = None
        if previous_id >= 0:
            previous_edge = bisect_left(neighbors, (previous_id, -math.inf))
            if (previous_edge == len(neighbors)
                    or neighbors[previous_edge][0] != previous_id):
                raise RuntimeError("previous tower span is missing from cached neighbors")
            previous_span_cost = neighbors[previous_edge][1]
        for next_id, span_cost in neighbors:
            # A tower span cannot be used as an immediate out-and-back detour
            # to manipulate the direction penalty.
            if next_id == previous_id:
                continue
            added_cost = span_cost
            if next_id != goal_id and next_id != start_id:
                next_row, next_col = nodes[next_id]
                added_cost += float(tower_cost[int(next_row), int(next_col)])
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
                             * 0.5 * (previous_span_cost + span_cost)
                             * (1.0 - min(max(cosine, -1.0), 1.0)))
                added_cost += turn_cost

            next_state = (current_id, next_id)
            candidate_cost = current_cost + added_cost
            if candidate_cost >= best.get(next_state, math.inf):
                continue
            best[next_state] = candidate_cost
            parent[next_state] = state
            estimated_total = candidate_cost + heuristic(next_id)
            heapq.heappush(open_set, (estimated_total, candidate_cost,
                                      next(sequence), next_state))
    return None


def span_clearance_status(start_rc, end_rc, dem_raster, pixel_spacing,
                          attachment_height_m, max_midspan_sag_m,
                          minimum_clearance_m, dem_valid_mask=None,
                          dem_nodata_value=None,
                          start_ground_elevation_m=None,
                          end_ground_elevation_m=None):
    """Screen one cell-center span against a sampled DEM profile.

    Terrain is bilinearly interpolated between DEM cells. On each crossed
    cell interval, the minimum between that surface and the parabolic conductor
    approximation is solved analytically. ``unknown`` means required terrain
    samples are masked, non-finite, or equal to the declared NoData value.
    """
    dem, valid = _dem_data_and_validity(
        dem_raster, dem_valid_mask, dem_nodata_value)
    if dem.ndim != 2:
        raise ValueError("dem_raster must be a 2D array")
    spacing = tuple(map(float, pixel_spacing))
    if (len(spacing) != 2
            or any(not math.isfinite(value) or value <= 0.0 for value in spacing)):
        raise ValueError("pixel_spacing must contain positive finite row/column metres")
    try:
        start = _checked_position(start_rc, "start_rc")
        end = _checked_position(end_rc, "end_rc")
        attachment_height = float(attachment_height_m)
        sag = float(max_midspan_sag_m)
        required_clearance = float(minimum_clearance_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid span profile input") from exc
    if (not math.isfinite(attachment_height) or attachment_height < 0.0
            or not math.isfinite(sag) or sag < 0.0
            or not math.isfinite(required_clearance) or required_clearance < 0.0):
        raise ValueError("profile heights, sag, and clearance must be finite and non-negative")

    reversed_input = end < start
    if reversed_input:
        start, end = end, start
        start_ground_elevation_m, end_ground_elevation_m = (
            end_ground_elevation_m, start_ground_elevation_m)

    for value in (start_ground_elevation_m, end_ground_elevation_m):
        if value is not None:
            try:
                valid_elevation = math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                valid_elevation = False
            if not valid_elevation:
                raise ValueError("terminal ground elevations must be finite numbers")

    height, width = dem.shape
    if any(not (0 <= row <= height - 1 and 0 <= col <= width - 1)
           for row, col in (start, end)):
        return {"status": "unknown", "reason": "endpoint_outside_dem"}

    row_m, col_m = spacing
    dr, dc = end[0] - start[0], end[1] - start[1]
    span_length = math.hypot(dr * row_m, dc * col_m)
    if not math.isfinite(span_length):
        return {"status": "unknown", "reason": "numerical_overflow"}
    minimum_clearance = math.inf
    limiting_distance = 0.0

    def dem_bilinear_elevation(row, col):
        row0, col0 = math.floor(row), math.floor(col)
        row1, col1 = min(row0 + 1, height - 1), min(col0 + 1, width - 1)
        row_fraction, col_fraction = row - row0, col - col0
        samples = (
            (row0, col0, (1.0 - row_fraction) * (1.0 - col_fraction)),
            (row0, col1, (1.0 - row_fraction) * col_fraction),
            (row1, col0, row_fraction * (1.0 - col_fraction)),
            (row1, col1, row_fraction * col_fraction),
        )
        elevation = 0.0
        for sample_row, sample_col, weight in samples:
            if weight == 0.0:
                continue
            if not valid[sample_row, sample_col]:
                return None
            elevation += weight * float(dem[sample_row, sample_col])
        return elevation

    def bilinear_elevation(row, col):
        if (row, col) == start and start_ground_elevation_m is not None:
            return float(start_ground_elevation_m)
        if (row, col) == end and end_ground_elevation_m is not None:
            return float(end_ground_elevation_m)
        return dem_bilinear_elevation(row, col)

    cuts = [0.0, 1.0]
    for first, last in ((start[0], end[0]), (start[1], end[1])):
        delta = last - first
        if delta == 0:
            continue
        low, high = sorted((first, last))
        for boundary in range(math.floor(low) + 1, math.ceil(high)):
            fraction = (boundary - first) / delta
            if 0.0 < fraction < 1.0:
                cuts.append(fraction)
    cuts = sorted(set(cuts))

    start_dem_ground = dem_bilinear_elevation(*start)
    end_dem_ground = dem_bilinear_elevation(*end)
    if start_dem_ground is None or end_dem_ground is None:
        return {"status": "unknown", "reason": "dem_nodata"}
    start_ground = (float(start_ground_elevation_m)
                    if start_ground_elevation_m is not None else start_dem_ground)
    end_ground = (float(end_ground_elevation_m)
                  if end_ground_elevation_m is not None else end_dem_ground)
    start_attach = start_ground + attachment_height
    end_attach = end_ground + attachment_height
    if not math.isfinite(start_attach) or not math.isfinite(end_attach):
        return {"status": "unknown", "reason": "numerical_overflow"}

    def conductor_at(fraction):
        chord = (1.0 - fraction) * start_attach + fraction * end_attach
        return chord - 4.0 * sag * fraction * (1.0 - fraction)

    for first, last in zip(cuts, cuts[1:]):
        middle = 0.5 * (first + last)
        terrain_values = []
        conductor_values = []
        for fraction in (first, middle, last):
            row = start[0] + fraction * dr
            col = start[1] + fraction * dc
            ground = bilinear_elevation(row, col)
            if ground is None:
                return {"status": "unknown", "reason": "dem_nodata"}
            terrain_values.append(ground)
            conductor = conductor_at(fraction)
            if not math.isfinite(conductor):
                return {"status": "unknown", "reason": "numerical_overflow"}
            conductor_values.append(conductor)

        # Both terrain and conductor are quadratic within this cell interval.
        terrain_a = 2.0 * (terrain_values[2] + terrain_values[0]
                           - 2.0 * terrain_values[1])
        terrain_b = terrain_values[2] - terrain_values[0] - terrain_a
        conductor_a = 2.0 * (conductor_values[2] + conductor_values[0]
                             - 2.0 * conductor_values[1])
        conductor_b = conductor_values[2] - conductor_values[0] - conductor_a
        clearance_a = conductor_a - terrain_a
        clearance_b = conductor_b - terrain_b
        clearance_c = conductor_values[0] - terrain_values[0]
        if not all(math.isfinite(value) for value in (
                terrain_a, terrain_b, conductor_a, conductor_b,
                clearance_a, clearance_b, clearance_c)):
            return {"status": "unknown", "reason": "numerical_overflow"}
        candidates = [0.0, 1.0]
        if clearance_a > 0.0:
            vertex_ratio = clearance_b / clearance_a
            if not math.isfinite(vertex_ratio):
                return {"status": "unknown", "reason": "numerical_overflow"}
            vertex = -0.5 * vertex_ratio
            if 0.0 < vertex < 1.0:
                candidates.append(vertex)
        for fraction_in_interval in candidates:
            clearance = (clearance_a * fraction_in_interval ** 2
                         + clearance_b * fraction_in_interval + clearance_c)
            if not math.isfinite(clearance):
                return {"status": "unknown", "reason": "numerical_overflow"}
            if clearance < minimum_clearance:
                minimum_clearance = clearance
                fraction = first + fraction_in_interval * (last - first)
                limiting_distance = fraction * span_length

    status = ("sampled_clearance_ok" if minimum_clearance >= required_clearance
              else "fail")
    if reversed_input:
        limiting_distance = span_length - limiting_distance
    if not math.isfinite(limiting_distance):
        return {"status": "unknown", "reason": "numerical_overflow"}
    return {"status": status,
            "sampled_minimum_clearance_m": minimum_clearance,
            "required_clearance_m": required_clearance,
            "limiting_distance_m": limiting_distance,
            "profile_interval_count": len(cuts) - 1}
