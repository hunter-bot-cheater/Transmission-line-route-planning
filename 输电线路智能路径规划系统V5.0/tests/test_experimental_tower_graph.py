import math

import numpy as np

from transmission_path_planner.experimental_tower_graph import (
    span_clearance_status,
    tower_graph_route,
)


def _small_graph_dijkstra_cost(nodes, start, goal, min_span, max_span,
                               site_cost, turn_weight, spacing=(1.0, 1.0)):
    """Independent full-graph Dijkstra oracle for an all-open unit-cost grid."""
    adjacency = {node: [] for node in nodes}
    span_cost = {}
    for index, first in enumerate(nodes):
        for second in nodes[index + 1:]:
            distance = math.hypot((second[0] - first[0]) * spacing[0],
                                  (second[1] - first[1]) * spacing[1])
            if min_span <= distance <= max_span:
                adjacency[first].append(second)
                adjacency[second].append(first)
                span_cost[first, second] = distance
                span_cost[second, first] = distance

    if turn_weight == 0.0:
        start_state = start
    else:
        start_state = (None, start)
    best = {start_state: 0.0}
    unsettled = {start_state}

    while unsettled:
        state = min(unsettled, key=best.__getitem__)
        unsettled.remove(state)
        current = state if turn_weight == 0.0 else state[1]
        previous = None if turn_weight == 0.0 else state[0]
        current_cost = best[state]
        if current == goal:
            return current_cost

        for following in adjacency[current]:
            if following == previous:
                continue
            added = span_cost[current, following]
            if following not in (start, goal):
                added += site_cost[following]
            if previous is not None:
                first_vector = ((current[0] - previous[0]) * spacing[0],
                                (current[1] - previous[1]) * spacing[1])
                second_vector = ((following[0] - current[0]) * spacing[0],
                                 (following[1] - current[1]) * spacing[1])
                cosine = (sum(a * b for a, b in zip(first_vector, second_vector))
                          / (math.hypot(*first_vector)
                             * math.hypot(*second_vector)))
                added += (turn_weight
                          * 0.5 * (span_cost[previous, current]
                                   + span_cost[current, following])
                          * (1.0 - cosine))

            next_state = (current, following) if turn_weight > 0.0 else following
            candidate = current_cost + added
            if candidate < best.get(next_state, math.inf):
                best[next_state] = candidate
                unsettled.add(next_state)
    return math.inf


def _small_route_objective(route, site_cost, turn_weight, spacing=(1.0, 1.0)):
    spans = [math.hypot((b[0] - a[0]) * spacing[0],
                        (b[1] - a[1]) * spacing[1])
             for a, b in zip(route, route[1:])]
    total = sum(spans) + sum(site_cost[cell] for cell in route[1:-1])
    for index in range(1, len(route) - 1):
        previous, current, following = route[index - 1:index + 2]
        first_vector = ((current[0] - previous[0]) * spacing[0],
                        (current[1] - previous[1]) * spacing[1])
        second_vector = ((following[0] - current[0]) * spacing[0],
                         (following[1] - current[1]) * spacing[1])
        cosine = (sum(a * b for a, b in zip(first_vector, second_vector))
                  / (math.hypot(*first_vector) * math.hypot(*second_vector)))
        total += (turn_weight * 0.5 * (spans[index - 1] + spans[index])
                  * (1.0 - cosine))
    return total


def test_tower_graph_enforces_span_limits_and_is_direction_invariant():
    cost = np.ones((11, 41), dtype=np.float64)
    allowed = np.ones(cost.shape, dtype=bool)
    start, goal = (5, 1), (5, 37)
    spacing = (35.0, 35.0)

    route = tower_graph_route(
        cost, allowed, allowed, start, goal, 175.0, 350.0,
        pixel_spacing=spacing, candidate_stride=2, turn_penalty_weight=0.5)
    reverse = tower_graph_route(
        cost, allowed, allowed, goal, start, 175.0, 350.0,
        pixel_spacing=spacing, candidate_stride=2, turn_penalty_weight=0.5)

    assert route[0] == start
    assert route[-1] == goal
    assert len(route) >= 5
    spans = [np.hypot((b[0] - a[0]) * spacing[0],
                      (b[1] - a[1]) * spacing[1])
             for a, b in zip(route, route[1:])]
    assert all(175.0 <= span <= 350.0 for span in spans)
    assert route == reverse[::-1]


def test_tower_site_cost_and_exclusion_apply_to_nodes_not_span_cells():
    cost = np.ones((11, 41), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    tower_allowed = traversable.copy()
    tower_allowed[:, 20] = False
    tower_cost = np.zeros(cost.shape, dtype=np.float64)
    tower_cost[4, ::2] = 1000.0

    route = tower_graph_route(
        cost, traversable, tower_allowed, (5, 1), (5, 37), 175.0, 350.0,
        pixel_spacing=(35.0, 35.0), candidate_stride=2,
        tower_cost_raster=tower_cost)

    assert route is not None
    assert all(tower_allowed[row, col] for row, col in route[1:-1])
    assert all(row != 4 for row, _ in route[1:-1])
    assert any(a[1] < 20 < b[1] or b[1] < 20 < a[1]
               for a, b in zip(route, route[1:]))


def test_tower_graph_returns_no_path_across_a_hard_wall():
    cost = np.ones((11, 41), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    traversable[:, 20] = False

    route = tower_graph_route(
        cost, traversable, traversable, (5, 1), (5, 37), 175.0, 350.0,
        pixel_spacing=(35.0, 35.0), candidate_stride=2)

    assert route is None


def test_tower_graph_rejects_span_touching_blocked_diagonal_corner():
    cost = np.ones((2, 2), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    traversable[0, 1] = False

    route = tower_graph_route(
        cost, traversable, np.ones(cost.shape, dtype=bool), (0, 0), (1, 1),
        1.3, 1.5, pixel_spacing=(1.0, 1.0), candidate_stride=10)

    assert route is None


def test_tower_graph_does_not_immediately_retrace_a_span_for_turn_cost():
    cost = np.ones((5, 5), dtype=np.float64)
    allowed = np.ones(cost.shape, dtype=bool)

    route = tower_graph_route(
        cost, allowed, allowed, (0, 0), (0, 1), 1.4, 3.1,
        candidate_stride=1, turn_penalty_weight=100.0)

    assert route is not None
    assert all(route[index - 1] != route[index + 1]
               for index in range(1, len(route) - 1))


def test_candidate_stride_can_change_reachability():
    cost = np.ones((5, 5), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    traversable[2, 2] = False
    tower_allowed = np.zeros(cost.shape, dtype=bool)
    tower_allowed[1, 1] = True
    tower_allowed[1, 3] = True

    dense = tower_graph_route(
        cost, traversable, tower_allowed, (2, 0), (2, 4), 1.0, 3.0,
        candidate_stride=1)
    sparse = tower_graph_route(
        cost, traversable, tower_allowed, (2, 0), (2, 4), 1.0, 3.0,
        candidate_stride=2)

    assert dense is not None
    assert sparse is None


def test_zero_turn_search_matches_independent_two_path_cost_oracle():
    cost = np.ones((3, 5), dtype=np.float64)
    tower_allowed = np.zeros(cost.shape, dtype=bool)
    tower_allowed[0, 2] = True
    tower_allowed[2, 2] = True
    tower_allowed[1, 0] = True
    tower_allowed[1, 4] = True
    tower_cost = np.zeros(cost.shape, dtype=np.float64)
    tower_cost[0, 2] = 10.0
    tower_cost[2, 2] = 3.0
    tower_cost[1, 0] = np.nan
    tower_cost[1, 4] = np.nan

    start, goal = (1, 0), (1, 4)
    route = tower_graph_route(
        cost, np.ones(cost.shape, dtype=bool), tower_allowed, start, goal,
        2.1, 2.3, candidate_stride=1, tower_cost_raster=tower_cost)
    reverse = tower_graph_route(
        cost, np.ones(cost.shape, dtype=bool), tower_allowed, goal, start,
        2.1, 2.3, candidate_stride=1, tower_cost_raster=tower_cost)

    # Only the two indicated two-span paths are feasible in this candidate
    # graph. The south site is cheaper; terminal site costs are excluded.
    expected_cost = 2.0 * np.sqrt(5.0) + 3.0
    route_cost = sum(
        np.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(route, route[1:])
    ) + sum(tower_cost[cell] for cell in route[1:-1])

    assert route == [start, (2, 2), goal]
    assert route == reverse[::-1]
    assert np.isclose(route_cost, expected_cost)


def test_lazy_a_star_matches_full_dijkstra_oracle_and_reverse_endpoints():
    cost = np.ones((5, 7), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    tower_allowed = np.zeros(cost.shape, dtype=bool)
    tower_allowed[0, 3] = True
    tower_allowed[1, 3] = True
    tower_cost = np.zeros(cost.shape, dtype=np.float64)
    tower_cost[0, 3] = 0.0
    tower_cost[1, 3] = 10.0
    tower_cost[2, 0] = 1000.0
    tower_cost[2, 6] = 2000.0
    start, goal = (2, 0), (2, 6)
    candidate_nodes = sorted([start, goal, (0, 3), (1, 3)])

    for turn_weight, expected_site in ((0.0, (0, 3)), (6.0, (1, 3))):
        route = tower_graph_route(
            cost, traversable, tower_allowed, start, goal, 3.0, 3.7,
            candidate_stride=1, tower_cost_raster=tower_cost,
            turn_penalty_weight=turn_weight)
        reverse = tower_graph_route(
            cost, traversable, tower_allowed, goal, start, 3.0, 3.7,
            candidate_stride=1, tower_cost_raster=tower_cost,
            turn_penalty_weight=turn_weight)
        oracle = _small_graph_dijkstra_cost(
            candidate_nodes, start, goal, 3.0, 3.7,
            {(0, 3): 0.0, (1, 3): 10.0}, turn_weight)

        assert route == [start, expected_site, goal]
        assert route == reverse[::-1]
        assert np.isclose(_small_route_objective(
            route, tower_cost, turn_weight), oracle)


def test_anisotropic_turn_cost_matches_independent_dijkstra_oracle():
    cost = np.ones((7, 9), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    tower_allowed = np.zeros(cost.shape, dtype=bool)
    tower_allowed[2, 4] = True
    tower_allowed[5, 4] = True
    tower_cost = np.zeros(cost.shape, dtype=np.float64)
    tower_cost[2, 4] = 20.0
    spacing = (2.0, 1.0)
    start, goal = (3, 0), (3, 8)
    nodes = [start, goal, (2, 4), (5, 4)]

    route = tower_graph_route(
        cost, traversable, tower_allowed, start, goal, 2.5, 5.8,
        pixel_spacing=spacing, candidate_stride=1,
        tower_cost_raster=tower_cost, turn_penalty_weight=10.0)
    reverse = tower_graph_route(
        cost, traversable, tower_allowed, goal, start, 2.5, 5.8,
        pixel_spacing=spacing, candidate_stride=1,
        tower_cost_raster=tower_cost, turn_penalty_weight=10.0)
    oracle = _small_graph_dijkstra_cost(
        nodes, start, goal, 2.5, 5.8,
        {(2, 4): 20.0, (5, 4): 0.0}, 10.0, spacing)

    assert route == reverse[::-1]
    assert np.isclose(_small_route_objective(
        route, tower_cost, 10.0, spacing), oracle)


def test_tower_graph_finds_valid_route_around_a_finite_hard_wall():
    cost = np.ones((7, 9), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    traversable[1:6, 4] = False
    tower_allowed = np.zeros(cost.shape, dtype=bool)
    tower_allowed[0, 3] = True
    tower_allowed[0, 6] = True
    start, goal = (3, 0), (3, 8)

    route = tower_graph_route(
        cost, traversable, tower_allowed, start, goal, 2.5, 5.5,
        candidate_stride=1)

    assert route is not None
    assert route[0] == start and route[-1] == goal
    assert (0, 3) in route and (0, 6) in route
    assert all(np.hypot(b[0] - a[0], b[1] - a[1]) <= 5.5
               for a, b in zip(route, route[1:]))


def test_lazy_a_star_prefers_lower_cost_route_with_anisotropic_pixels():
    cost = np.ones((5, 7), dtype=np.float64)
    cost[3, :] = 3.0
    traversable = np.ones(cost.shape, dtype=bool)
    tower_allowed = np.zeros(cost.shape, dtype=bool)
    tower_allowed[1, 3] = True
    tower_allowed[3, 3] = True
    start, goal = (2, 0), (2, 6)

    route = tower_graph_route(
        cost, traversable, tower_allowed, start, goal, 3.5, 3.7,
        pixel_spacing=(2.0, 1.0), candidate_stride=1)
    reverse = tower_graph_route(
        cost, traversable, tower_allowed, goal, start, 3.5, 3.7,
        pixel_spacing=(2.0, 1.0), candidate_stride=1)

    # Only the two middle sites connect both terminals. The upper route has
    # uniform unit cost; the lower route spends half of each span at cost 3.
    upper_cost = 2.0 * math.sqrt(13.0)
    lower_cost = 4.0 * math.sqrt(13.0)
    assert upper_cost < lower_cost
    assert route == [start, (1, 3), goal]
    assert route == reverse[::-1]


def test_lazy_a_star_handles_zero_cost_and_inclusive_span_boundaries():
    cost = np.zeros((1, 3), dtype=np.float64)
    traversable = np.ones(cost.shape, dtype=bool)
    tower_allowed = np.zeros(cost.shape, dtype=bool)

    for min_span, max_span in ((2.0, 2.1), (1.9, 2.0)):
        route = tower_graph_route(
            cost, traversable, tower_allowed, (0, 0), (0, 2),
            min_span, max_span, candidate_stride=1)
        assert route == [(0, 0), (0, 2)]

    assert tower_graph_route(
        cost, traversable, tower_allowed, (0, 0), (0, 2),
        2.0001, 2.1, candidate_stride=1) is None
    assert tower_graph_route(
        cost, traversable, tower_allowed, (-1, 0), (0, 2),
        1.9, 2.1, candidate_stride=1) is None
    blocked = traversable.copy()
    blocked[0, 0] = False
    assert tower_graph_route(
        cost, blocked, tower_allowed, (0, 0), (0, 2),
        1.9, 2.1, candidate_stride=1) is None
    assert tower_graph_route(
        cost, traversable, tower_allowed, (0, 1), (0, 1),
        1.9, 2.1, candidate_stride=1) == [(0, 1)]


def test_profile_clearance_rejects_ridge_and_routes_around_it():
    cost = np.ones((7, 11), dtype=np.float64)
    dem = np.zeros(cost.shape, dtype=np.float64)
    dem[3, 5] = 20.0
    dem[3, 0] = 2.0
    dem[3, 10] = 10.0
    allowed = np.zeros(cost.shape, dtype=bool)
    allowed[0, 5] = True
    allowed[6, 5] = True
    traversable = np.ones(cost.shape, dtype=bool)
    start, goal = (3, 0), (3, 10)

    direct = span_clearance_status(
        start, goal, dem, (1.0, 1.0), 20.0, 5.0, 3.0)
    reverse_direct = span_clearance_status(
        goal, start, dem, (1.0, 1.0), 20.0, 5.0, 3.0)
    route = tower_graph_route(
        cost, traversable, allowed, start, goal, 4.0, 10.5,
        pixel_spacing=(1.0, 1.0), candidate_stride=1,
        dem_raster=dem, attachment_height_m=20.0,
        max_midspan_sag_m=5.0, minimum_clearance_m=3.0)
    reverse_route = tower_graph_route(
        cost, traversable, allowed, goal, start, 4.0, 10.5,
        pixel_spacing=(1.0, 1.0), candidate_stride=1,
        dem_raster=dem, attachment_height_m=20.0,
        max_midspan_sag_m=5.0, minimum_clearance_m=3.0)

    assert direct["status"] == "fail"
    assert reverse_direct["status"] == direct["status"]
    assert np.isclose(reverse_direct["sampled_minimum_clearance_m"],
                      direct["sampled_minimum_clearance_m"])
    assert route is not None
    assert route == reverse_route[::-1]
    assert route[1] in {(0, 5), (6, 5)}
    assert all(span_clearance_status(
        first, second, dem, (1.0, 1.0), 20.0, 5.0, 3.0)["status"] == "sampled_clearance_ok"
        for first, second in zip(route, route[1:]))


def test_profile_nodata_is_unknown_and_rejects_span():
    cost = np.ones((1, 11), dtype=np.float64)
    dem = np.zeros(cost.shape, dtype=np.float64)
    dem[0, 5] = np.nan
    allowed = np.zeros(cost.shape, dtype=bool)
    traversable = np.ones(cost.shape, dtype=bool)
    start, goal = (0, 0), (0, 10)

    status = span_clearance_status(
        start, goal, dem, (1.0, 1.0), 20.0, 5.0, 3.0)
    route = tower_graph_route(
        cost, traversable, allowed, start, goal, 1.0, 11.0,
        candidate_stride=1, dem_raster=dem, attachment_height_m=20.0,
        max_midspan_sag_m=5.0, minimum_clearance_m=3.0)

    assert status["status"] == "unknown"
    assert status["reason"] == "dem_nodata"
    assert route is None


def test_profile_mask_and_finite_nodata_are_unknown():
    dem = np.ma.array(np.zeros((1, 5)), mask=False)
    dem.mask[0, 2] = True
    masked = span_clearance_status(
        (0, 0), (0, 4), dem, (1.0, 1.0), 20.0, 5.0, 3.0)

    sentinel_dem = np.zeros((1, 5), dtype=np.float64)
    sentinel_dem[0, 2] = -9999.0
    sentinel = span_clearance_status(
        (0, 0), (0, 4), sentinel_dem, (1.0, 1.0), 20.0, 5.0, 3.0,
        dem_nodata_value=-9999.0)
    masked_by_validity = span_clearance_status(
        (0, 0), (0, 4), sentinel_dem, (1.0, 1.0), 20.0, 5.0, 3.0,
        dem_valid_mask=np.array([[True, True, False, True, True]]))
    masked_validity = np.ma.array(np.ones((1, 5), dtype=bool), mask=False)
    masked_validity.mask[0, 2] = True
    validity_masked = span_clearance_status(
        (0, 0), (0, 4), np.zeros((1, 5)), (1.0, 1.0), 20.0, 5.0, 3.0,
        dem_valid_mask=masked_validity)

    assert masked["status"] == "unknown"
    assert sentinel["status"] == "unknown"
    assert masked_by_validity["status"] == "unknown"
    assert validity_masked["status"] == "unknown"


def test_profile_numerical_overflow_is_unknown():
    dem = np.full((1, 3), 1e308, dtype=np.float64)
    status = span_clearance_status(
        (0, 0), (0, 2), dem, (1.0, 1.0), 1e308, 5.0, 3.0)

    assert status["status"] == "unknown"
    assert status["reason"] == "numerical_overflow"


def test_profile_span_length_overflow_is_unknown():
    status = span_clearance_status(
        (0, 0), (0, 2), np.zeros((1, 3)), (1.0, 1e308), 20.0, 5.0, 3.0)

    assert status["status"] == "unknown"
    assert status["reason"] == "numerical_overflow"


def test_profile_checks_interior_minimum_when_vertex_denominator_would_overflow():
    dem = np.array([[0.0, 5e307], [5e307, 0.0]])

    status = span_clearance_status(
        (0, 0), (1, 1), dem, (1.0, 1.0), 0.0, 0.0, 0.0)

    assert status["status"] == "fail"
    assert np.isclose(status["sampled_minimum_clearance_m"], -2.5e307)


def test_profile_clearance_is_direction_invariant_at_threshold():
    dem = np.array([
        [-30.0, 26.0, -17.0, 69.0],
        [-50.0, 11.0, -75.0, -58.0],
        [-66.0, 49.0, -60.0, 34.0],
        [-31.0, 13.0, 75.0, 73.0],
    ])
    start, end = (0, 3), (3, 2)

    forward = span_clearance_status(
        start, end, dem, (7.0, 5.0), 0.0, 10.0, 0.0)
    reverse = span_clearance_status(
        end, start, dem, (7.0, 5.0), 0.0, 10.0, 0.0)

    assert forward["status"] == reverse["status"]
    assert (forward["sampled_minimum_clearance_m"]
            == reverse["sampled_minimum_clearance_m"])
    assert np.isclose(
        forward["limiting_distance_m"] + reverse["limiting_distance_m"],
        np.hypot(3.0 * 7.0, 5.0))


def test_profile_solves_interior_clearance_minimum_between_raster_cells():
    dem = np.array([[0.0, 20.0], [20.0, 0.0]])
    status = span_clearance_status(
        (0, 0), (1, 1), dem, (1.0, 1.0), 20.0, 5.0, 6.0)

    assert status["status"] == "fail"
    assert np.isclose(status["sampled_minimum_clearance_m"], 5.0)
    assert np.isclose(status["limiting_distance_m"], np.sqrt(2.0) / 2.0)


def test_profile_parameters_must_be_supplied_together():
    cost = np.ones((1, 3), dtype=np.float64)
    allowed = np.ones(cost.shape, dtype=bool)

    try:
        tower_graph_route(
            cost, allowed, allowed, (0, 0), (0, 2), 1.0, 3.0,
            dem_raster=cost, attachment_height_m=20.0)
    except ValueError as error:
        assert "supplied together" in str(error)
    else:
        raise AssertionError("partial profile inputs must be rejected")
