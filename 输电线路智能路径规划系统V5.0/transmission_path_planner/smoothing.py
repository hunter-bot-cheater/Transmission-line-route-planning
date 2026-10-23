# transmission_path_planner/smoothing.py
"""路径平滑、硬性约束吸附、高程采样。"""

from .config import *
from .geometry import *

def _rdp(points, epsilon):
    """Ramer-Douglas-Peucker 折线简化。

    改用显式栈的迭代实现, 避免对长且高度弯曲的路径递归过深触发
    'maximum recursion depth exceeded' (Case 3 PPO 路径点数常达数千, 单边反复
    细分时递归深度可超 1000)。迭代版结果与递归版完全一致。
    """
    points = list(points)
    n = len(points)
    if n < 3:
        return points
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True
    stack = [(0, n - 1)]
    while stack:
        s_i, e_i = stack.pop()
        if e_i <= s_i + 1:
            continue
        s = np.array(points[s_i], np.float64)
        e = np.array(points[e_i], np.float64)
        lv = e - s
        ll = np.linalg.norm(lv)
        if ll < 1e-12:
            # 退化段: 端点已标记保留, 丢弃内部点(与原递归版行为一致)
            continue
        unit = lv / ll
        dmax, idx = 0.0, s_i + 1
        for i in range(s_i + 1, e_i):
            v = np.array(points[i], np.float64) - s
            proj = np.clip(np.dot(v, unit), 0, ll)
            closest = s + proj * unit
            d = np.linalg.norm(np.array(points[i], np.float64) - closest)
            if d > dmax:
                dmax, idx = d, i
        if dmax > epsilon:
            keep[idx] = True
            stack.append((s_i, idx))
            stack.append((idx, e_i))
    return [points[i] for i in range(n) if keep[i]]


def smooth_path(geo, spacing_m=90.0):
    if len(geo) < 3:
        return geo
    arr = np.array([(lon, lat) for lon, lat in geo], np.float64)
    from pyproj import CRS, Transformer

    lons = np.radians(arr[:, 0])
    lon0 = math.degrees(math.atan2(float(np.sin(lons).mean()), float(np.cos(lons).mean())))
    lat0 = float(np.mean(arr[:, 1]))
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    project = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    unproject = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True).transform
    metric = np.asarray([project(lon, lat) for lon, lat in arr], dtype=np.float64)
    epsilon_m = min(max(float(spacing_m), 0.0) * 0.5, 50.0)
    simplified = _rdp(metric.tolist(), epsilon_m)
    if len(simplified) < 2:
        return geo
    simplified = np.array(simplified)
    xs, ys = simplified[:, 0], simplified[:, 1]
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return geo
    n = max(int(total / max(float(spacing_m), 1e-6)), len(simplified))
    sd = np.linspace(0, total, n)
    xi = np.interp(sd, cum, xs); yi = np.interp(sd, cum, ys)
    result = [tuple(map(float, unproject(xi[i], yi[i]))) for i in range(n)]
    result[0], result[-1] = tuple(geo[0]), tuple(geo[-1])
    return result


def snap_to_valid(geo, transform, hard_mask, max_offset_m=100.0):
    """把平滑后的 geo 路径中落到硬约束屏蔽格(陡坡/超高/无效DEM)的点, 吸附到最近的合法格中心。
    推理 rollout 只走合法邻居, 原始格路径已是 0 违规; 但 smooth_path 在地理空间等距重采样时会
    越过硬约束格。吸附距离使用米制上限, 超出范围则拒绝候选并由调用方回退到原始路径。"""
    H, W = hard_mask.shape
    center_lat = transform.f + transform.e * (H / 2.0)
    row_m, col_m = pixel_spacing_meters(transform, center_lat)
    max_cells = int(math.ceil(max(float(max_offset_m), 0.0) / min(row_m, col_m)))
    out = []
    for lon, lat in geo:
        r, c = geo_to_rowcol(lat, lon, transform)
        r = int(round(r)); c = int(round(c))
        if 0 <= r < H and 0 <= c < W and hard_mask[r, c] == 1:
            out.append((lon, lat)); continue
        candidates = []
        for dr in range(-max_cells, max_cells + 1):
            for dc in range(-max_cells, max_cells + 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W) or hard_mask[rr, cc] != 1:
                    continue
                distance_m = haversine_km(lon, lat, *rowcol_to_geo(rr, cc, transform)) * 1000.0
                if distance_m <= max_offset_m:
                    candidates.append((distance_m, rr, cc))
        if not candidates:
            return None
        _, rr, cc = min(candidates)
        out.append(rowcol_to_geo(rr, cc, transform))
    return out


def _grid_cells_touched(c0, r0, c1, r1):
    """Return every raster cell touched by a segment in grid-corner coordinates."""
    dx, dy = c1 - c0, r1 - r0
    cells = set()

    def axis_cells(value):
        nearest = round(value)
        if abs(value - nearest) <= 1e-10:
            return (int(nearest) - 1, int(nearest))
        return (int(math.floor(value)),)

    def add_touching(x, y):
        for row in axis_cells(y):
            for col in axis_cells(x):
                cells.add((row, col))

    add_touching(c0, r0)
    add_touching(c1, r1)

    sx = 1 if dx > 0 else -1 if dx < 0 else 0
    sy = 1 if dy > 0 else -1 if dy < 0 else 0
    col, row = int(math.floor(c0)), int(math.floor(r0))
    if sx:
        t_max_x = ((col + 1 - c0) if sx > 0 else (c0 - col)) / abs(dx)
        t_delta_x = 1.0 / abs(dx)
    else:
        t_max_x = t_delta_x = math.inf
    if sy:
        t_max_y = ((row + 1 - r0) if sy > 0 else (r0 - row)) / abs(dy)
        t_delta_y = 1.0 / abs(dy)
    else:
        t_max_y = t_delta_y = math.inf

    while min(t_max_x, t_max_y) <= 1.0 + 1e-12:
        if abs(t_max_x - t_max_y) <= 1e-12:
            cells.add((row, col + sx))
            cells.add((row + sy, col))
            col += sx
            row += sy
            cells.add((row, col))
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            col += sx
            cells.add((row, col))
            t_max_x += t_delta_x
        else:
            row += sy
            cells.add((row, col))
            t_max_y += t_delta_y

    if sx == 0 and abs(c0 - round(c0)) <= 1e-10:
        boundary_col = int(round(c0))
        cells.update((row, boundary_col - 1) for row, col in tuple(cells)
                     if col == boundary_col)
    if sy == 0 and abs(r0 - round(r0)) <= 1e-10:
        boundary_row = int(round(r0))
        cells.update((boundary_row - 1, col) for row, col in tuple(cells)
                     if row == boundary_row)
    return cells


def path_hard_violations(geo, transform, hard_mask):
    """Count blocked/out-of-grid cells crossed by a path, including segment interiors."""
    if not geo:
        return 0
    inv = ~transform
    cells = set()
    grid_points = [inv * point for point in geo]
    for (c0, r0), (c1, r1) in zip(grid_points, grid_points[1:]):
        cells.update(_grid_cells_touched(c0, r0, c1, r1))
    if len(grid_points) == 1:
        c, r = grid_points[0]
        cells.update(_grid_cells_touched(c, r, c, r))
    return sum(1 for r, c in cells
               if not (0 <= r < hard_mask.shape[0] and 0 <= c < hard_mask.shape[1])
               or hard_mask[r, c] == 0)


def _ground_grade_segment_cells(c0, r0, c1, r1, dem):
    """Sample the DEM cells along one segment using the final route validator's rule."""
    H, W = dem.shape
    steps = max(1, int(math.ceil(4 * max(abs(r1 - r0), abs(c1 - c0)))))
    cells = []
    for i in range(steps + 1):
        t = i / steps
        r, c = r0 + (r1 - r0) * t, c0 + (c1 - c0) * t
        rr, cc = int(math.floor(r)), int(math.floor(c))
        cell = ((rr, cc) if 0 <= rr < H and 0 <= cc < W
                and np.isfinite(dem[rr, cc]) else None)
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def _ground_grade_cells_report(cells, dem, row_spacing_m, col_spacing_m,
                               max_ground_grade):
    violations = 0
    unassessed = 0
    max_ratio = 0.0
    for first, second in zip(cells, cells[1:]):
        if first is None or second is None:
            unassessed += 1
            continue
        r0, c0 = first
        r1, c1 = second
        if first == second:
            continue
        distance_m = math.hypot((r1 - r0) * row_spacing_m,
                                (c1 - c0) * col_spacing_m)
        if distance_m <= 0:
            unassessed += 1
            continue
        ratio = abs(float(dem[r1, c1]) - float(dem[r0, c0])) / distance_m
        max_ratio = max(max_ratio, ratio)
        if ratio > max_ground_grade + 1e-9:
            violations += 1
    return violations, unassessed, max_ratio


def _ground_grade_grid_edge_passes(start_rc, end_rc, dem, pixel_spacing,
                                   max_ground_grade):
    """Check a cell-center A* edge with the same DEM sampling as route validation."""
    row0, col0 = start_rc
    row1, col1 = end_rc
    cells = _ground_grade_segment_cells(
        col0 + 0.5, row0 + 0.5, col1 + 0.5, row1 + 0.5, dem)
    violations, unassessed, _ = _ground_grade_cells_report(
        cells, dem, pixel_spacing[0], pixel_spacing[1], max_ground_grade)
    return violations == 0 and unassessed == 0


def ground_grade_limit_report(geo, dem, transform, max_ground_grade, crs=None):
    """Check consecutive DEM cells crossed by a route against a ground-grade ratio cap."""
    if max_ground_grade is None:
        return {
            "grade_limit_status": "not_configured",
            "grade_limit_violation_count": 0,
            "grade_limit_unassessed_count": 0,
            "grade_limit_max_sampled_grade_ratio": None,
            "grade_limit_threshold_ratio": None,
        }
    if not geo or len(geo) < 2:
        return {
            "grade_limit_status": "no_path",
            "grade_limit_violation_count": 0,
            "grade_limit_unassessed_count": 0,
            "grade_limit_max_sampled_grade_ratio": None,
            "grade_limit_threshold_ratio": max_ground_grade,
        }

    inv = ~transform
    lat_center = transform.f + transform.e * (dem.shape[0] / 2.0)
    row_spacing_m, col_spacing_m = pixel_spacing_meters(transform, lat_center, crs)
    cells = []
    last_cell = object()
    grid_points = [inv * point for point in geo]
    if len(grid_points) == 1:
        segments = [(grid_points[0], grid_points[0])]
    else:
        segments = zip(grid_points, grid_points[1:])
    for (c0, r0), (c1, r1) in segments:
        for marker in _ground_grade_segment_cells(c0, r0, c1, r1, dem):
            if marker != last_cell:
                cells.append(marker)
                last_cell = marker

    violations, unassessed, max_ratio = _ground_grade_cells_report(
        cells, dem, row_spacing_m, col_spacing_m, max_ground_grade)

    status = "fail" if violations or unassessed else "pass"
    return {
        "grade_limit_status": status,
        "grade_limit_violation_count": violations,
        "grade_limit_unassessed_count": unassessed,
        "grade_limit_max_sampled_grade_ratio": round(max_ratio, 6),
        "grade_limit_threshold_ratio": max_ground_grade,
    }


def filter_path_by_ground_grade(geo, dem, transform, max_ground_grade, crs=None):
    """Reject a route when a configured grade cap is violated or cannot be assessed."""
    report = ground_grade_limit_report(geo, dem, transform, max_ground_grade, crs)
    if max_ground_grade is not None and report["grade_limit_status"] != "pass":
        return None, report
    return geo, report


def smooth_path_with_constraints(geo, transform, hard_mask, cost_surface=None, dem=None,
                                grade_penalty_weight=1.0, turn_penalty_weight=0.0,
                                pixel_spacing=None, max_ground_grade=None, crs=None,
                                shortcut_endpoints=None):
    """Try a legal, non-increasing direct shortcut, then use the existing smoother."""
    if not geo:
        return None

    def grade_is_valid(path):
        return (max_ground_grade is None or (dem is not None and
                ground_grade_limit_report(
                    path, dem, transform, max_ground_grade, crs
                )["grade_limit_status"] == "pass"))

    if len(geo) >= 3 and cost_surface is not None and dem is not None:
        direct = [geo[0], geo[-1]]
        raw_for_guard = list(geo)
        direct_for_guard = direct
        if shortcut_endpoints is not None:
            start, end = shortcut_endpoints
            start = tuple(map(float, start))
            end = tuple(map(float, end))
            raw_for_guard[0], raw_for_guard[-1] = start, end
            direct_for_guard = [start, end]
        if path_hard_violations(direct_for_guard, transform, hard_mask) == 0:
            grade_routes = (
                [direct] if direct_for_guard == direct else [direct, direct_for_guard]
            )
            grade_ok = max_ground_grade is None or all(
                ground_grade_limit_report(
                    route, dem, transform, max_ground_grade, crs
                )["grade_limit_status"] == "pass"
                for route in grade_routes
            )
            if grade_ok:
                # Import at call time to avoid the metrics -> smoothing import cycle.
                from .metrics import path_cost_integral

                raw_objective = path_cost_integral(
                    raw_for_guard, dem, transform, cost_surface, grade_penalty_weight,
                    turn_penalty_weight, pixel_spacing)
                direct_objective = path_cost_integral(
                    direct_for_guard, dem, transform, cost_surface, grade_penalty_weight,
                    turn_penalty_weight, pixel_spacing)
                if (raw_objective.get("status") == "measured"
                        and direct_objective.get("status") == "measured"):
                    raw_cost = raw_objective["total_cost_meter"]
                    direct_cost = direct_objective["total_cost_meter"]
                    tolerance = max(abs(raw_cost), 1.0) * 1e-9
                    if direct_cost <= raw_cost + tolerance:
                        return direct

    candidate = snap_to_valid(smooth_path(geo), transform, hard_mask)
    if (candidate and path_hard_violations(candidate, transform, hard_mask) == 0
            and grade_is_valid(candidate)):
        if cost_surface is None:
            return candidate
        if dem is not None:
            # Import at call time to avoid the metrics -> smoothing import cycle.
            from .metrics import path_cost_integral

            raw_objective = path_cost_integral(
                geo, dem, transform, cost_surface, grade_penalty_weight,
                turn_penalty_weight, pixel_spacing)
            smooth_objective = path_cost_integral(
                candidate, dem, transform, cost_surface, grade_penalty_weight,
                turn_penalty_weight, pixel_spacing)
            if (raw_objective.get("status") == "measured"
                    and smooth_objective.get("status") == "measured"):
                raw_cost = raw_objective["total_cost_meter"]
                smooth_cost = smooth_objective["total_cost_meter"]
                tolerance = max(abs(raw_cost), 1.0) * 1e-9
                if smooth_cost <= raw_cost + tolerance:
                    return candidate
    if (path_hard_violations(geo, transform, hard_mask) == 0
            and grade_is_valid(geo)):
        return geo
    return None


def sample_dem(coords, dem, transform):
    out = []
    H, W = dem.shape
    for lon, lat in coords:
        r, c = geo_to_rowcol(lat, lon, transform)
        if 0 <= r < H and 0 <= c < W and np.isfinite(dem[r, c]):
            out.append(float(dem[r, c]))
        else:
            out.append(np.nan)
    return out


def densify_path_for_raster(coords, transform):
    """Interpolate each route segment so no step exceeds one pixel on either axis."""
    coords = list(coords or [])
    if len(coords) < 2:
        return coords
    inv = ~transform
    grid = [inv * point for point in coords]
    dense = [coords[0]]
    for i, ((c0, r0), (c1, r1)) in enumerate(zip(grid, grid[1:])):
        steps = max(1, int(math.ceil(max(abs(r1 - r0), abs(c1 - c0)))))
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        dense.extend((x0 + (x1 - x0) * step / steps,
                      y0 + (y1 - y0) * step / steps) for step in range(1, steps + 1))
    return dense


def cumulative_relief(hs):
    hs = np.array([h for h in hs if np.isfinite(h)], np.float64)
    if hs.size < 2:
        return 0.0, 0.0
    return float(np.sum(np.abs(np.diff(hs)))), float(hs[-1] - hs[0])
