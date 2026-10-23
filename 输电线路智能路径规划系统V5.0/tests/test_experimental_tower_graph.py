import numpy as np

from transmission_path_planner.experimental_tower_graph import tower_graph_route


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
