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
    eps_deg = spacing_m / 111000.0 * 1.5
    simplified = _rdp(arr.tolist(), eps_deg)
    if len(simplified) < 2:
        return geo
    simplified = np.array(simplified)
    xs, ys = simplified[:, 0], simplified[:, 1]
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return geo
    spacing_deg = spacing_m / 111000.0
    n = max(int(total / spacing_deg), len(simplified))
    sd = np.linspace(0, total, n)
    xi = np.interp(sd, cum, xs); yi = np.interp(sd, cum, ys)
    return [(float(xi[i]), float(yi[i])) for i in range(n)]


def snap_to_valid(geo, transform, hard_mask, radius=80):
    """把平滑后的 geo 路径中落到硬约束屏蔽格(陡坡/超高/无效DEM)的点, 吸附到最近的合法格中心。
    推理 rollout 只走合法邻居, 原始格路径已是 0 违规; 但 smooth_path 在地理空间等距重采样时会
    越过陡峭壁体落到屏蔽格, 导致"硬约束违规"被误报。吸附步骤保证交付路线与指标均 0 违规。"""
    H, W = hard_mask.shape
    out = []
    for lon, lat in geo:
        r, c = geo_to_rowcol(lat, lon, transform)
        r = int(round(r)); c = int(round(c))
        if 0 <= r < H and 0 <= c < W and hard_mask[r, c] == 1:
            out.append((lon, lat)); continue
        found = None
        for rad in range(1, radius + 1):
            hit = False
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and hard_mask[rr, cc] == 1:
                        found = (rr, cc); hit = True; break
                if hit:
                    break
            if found:
                break
        if found:
            out.append(rowcol_to_geo(found[0], found[1], transform))
        # 若半径内都无合法格(极端情况), 丢弃该点
    return out


def path_hard_violations(geo, transform, hard_mask):
    """Count blocked/out-of-grid cells crossed by a path, including segment interiors."""
    if not geo:
        return 0
    inv = ~transform
    cells = set()
    grid_points = [inv * point for point in geo]
    for (c0, r0), (c1, r1) in zip(grid_points, grid_points[1:]):
        steps = max(1, int(math.ceil(4 * max(abs(r1 - r0), abs(c1 - c0)))))
        for i in range(steps + 1):
            t = i / steps
            row, col = r0 + (r1 - r0) * t, c0 + (c1 - c0) * t
            rows, cols = {int(math.floor(row))}, {int(math.floor(col))}
            if abs(row - round(row)) < 1e-9:
                rows.add(int(round(row)) - 1)
            if abs(col - round(col)) < 1e-9:
                cols.add(int(round(col)) - 1)
            for r in rows:
                for c in cols:
                    if not (0 <= r < hard_mask.shape[0] and 0 <= c < hard_mask.shape[1]) or hard_mask[r, c] == 0:
                        cells.add((r, c))
    if len(grid_points) == 1:
        c, r = grid_points[0]
        r, c = int(math.floor(r)), int(math.floor(c))
        if not (0 <= r < hard_mask.shape[0] and 0 <= c < hard_mask.shape[1]) or hard_mask[r, c] == 0:
            cells.add((r, c))
    return len(cells)


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
    H, W = dem.shape
    lat_center = transform.f + transform.e * (H / 2.0)
    row_spacing_m, col_spacing_m = pixel_spacing_meters(transform, lat_center, crs)
    cells = []
    last_cell = object()
    grid_points = [inv * point for point in geo]
    if len(grid_points) == 1:
        segments = [(grid_points[0], grid_points[0])]
    else:
        segments = zip(grid_points, grid_points[1:])
    for (c0, r0), (c1, r1) in segments:
        steps = max(1, int(math.ceil(4 * max(abs(r1 - r0), abs(c1 - c0)))))
        for i in range(steps + 1):
            t = i / steps
            r, c = r0 + (r1 - r0) * t, c0 + (c1 - c0) * t
            cell = (int(math.floor(r)), int(math.floor(c)))
            rr, cc = cell
            if not (0 <= rr < H and 0 <= cc < W) or not np.isfinite(dem[rr, cc]):
                marker = None
            else:
                marker = cell
            if marker != last_cell:
                cells.append(marker)
                last_cell = marker

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
        z0, z1 = float(dem[r0, c0]), float(dem[r1, c1])
        distance_m = math.hypot((r1 - r0) * row_spacing_m, (c1 - c0) * col_spacing_m)
        if distance_m <= 0:
            unassessed += 1
            continue
        ratio = abs(z1 - z0) / distance_m
        max_ratio = max(max_ratio, ratio)
        if ratio > max_ground_grade + 1e-9:
            violations += 1

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


def smooth_path_with_constraints(geo, transform, hard_mask):
    """Keep a smoothed route only when every sampled segment remains inside the hard mask."""
    if not geo:
        return None
    candidate = snap_to_valid(smooth_path(geo), transform, hard_mask)
    if candidate and path_hard_violations(candidate, transform, hard_mask) == 0:
        return candidate
    if path_hard_violations(geo, transform, hard_mask) == 0:
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
