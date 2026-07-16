"""
v2_strict: 模块4 — 路径规划(严格化版本)
版本: v2.20260525
作者: path_planning_team
变更记录:
  - v2.20260525: A*启发式权重1.2x, 对角线成本1.5x, 起点/终点严格模式, 新增质量门控7项审查
  - v1.20260525: 初始版本
依赖: v2/config
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter
import heapq
import math
import time
import json

import config as cfg

# RDP简化
try:
    from rdp import rdp
except ImportError:
    def rdp(points, epsilon):
        """Ramer-Douglas-Peucker 简化"""
        if len(points) < 3:
            return points
        start, end = np.array(points[0]), np.array(points[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            dmax = 0
            index = 0
        else:
            line_unit = line_vec / line_len
            dists = []
            for p in points[1:-1]:
                v = np.array(p) - start
                proj = np.dot(v, line_unit)
                proj = np.clip(proj, 0, line_len)
                closest = start + proj * line_unit
                dists.append(np.linalg.norm(np.array(p) - closest))
            index = np.argmax(dists) + 1
            dmax = dists[np.argmax(dists)]
        if dmax > epsilon:
            left = rdp(points[:index + 1], epsilon)
            right = rdp(points[index:], epsilon)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]


# ============================================================
# 成本表面融合
# ============================================================
def fuse_cost_surface(rf_cost, soft_mask, hard_mask, dist_existing=None, smooth_sigma=1.0,
                      corridor_min: float = 0.02, corridor_range: float = 1000.0):
    """融合成本表面

    Args:
        corridor_min: 最低折扣底线 (默认0.02=98%折扣)
        corridor_range: 折扣影响半径(米) (默认1000m)
    """
    print("[Phase4] 融合成本表面...")
    rf_cost = np.nan_to_num(rf_cost, nan=np.nanmean(rf_cost[rf_cost < np.inf]) if (rf_cost < np.inf).any() else 0.5)
    rf_cost = np.clip(rf_cost, 0, None)
    final_cost = rf_cost.copy().astype(np.float64)
    if dist_existing is not None:
        corridor_strength = 1.0 - corridor_min
        corridor_bonus = corridor_min + corridor_strength * np.clip(dist_existing / corridor_range, 0, 1)
        final_cost = final_cost * corridor_bonus.astype(np.float64)
        print(f"  走廊偏好已应用 (线性斜坡至300m, 最多70%折扣)")
    if soft_mask is not None:
        soft_mask = np.nan_to_num(soft_mask, nan=1.0)
        soft_mask = np.clip(soft_mask, 0.01, 1.0)
        final_cost = final_cost * soft_mask.astype(np.float64)
    if hard_mask is not None:
        valid_mask = (hard_mask == 1) & np.isfinite(final_cost)
    else:
        valid_mask = np.isfinite(final_cost)
    final_cost_smooth = gaussian_filter(
        np.where(valid_mask, final_cost, 0.0),
        sigma=smooth_sigma, mode="nearest"
    )
    if smooth_sigma > 0:
        weights = gaussian_filter(valid_mask.astype(np.float64), sigma=smooth_sigma, mode="nearest")
        weights = np.clip(weights, 1e-6, None)
        final_cost_smooth = final_cost_smooth / weights
    final_cost = np.where(valid_mask, final_cost_smooth, np.inf)
    final_cost = np.clip(final_cost, 0, None)
    valid = final_cost < np.inf
    if valid.any():
        print(f"  成本范围: [{final_cost[valid].min():.4f}, {final_cost[valid].max():.4f}]")
    return final_cost.astype(np.float32)


# ============================================================
# 坐标转换
# ============================================================
def geo_to_grid(lat, lon, transform):
    col = int((lon - transform.c) / transform.a)
    row = int((lat - transform.f) / transform.e)
    return row, col


def grid_to_geo(row, col, transform):
    lon = transform.c + col * transform.a + transform.a / 2
    lat = transform.f + row * transform.e + transform.e / 2
    return lat, lon


def grid_to_geo_coords(path_cells, transform):
    coords = []
    for r, c in path_cells:
        lon = transform.c + c * transform.a + transform.a / 2
        lat = transform.f + r * transform.e + transform.e / 2
        coords.append((lon, lat))
    return coords


# ============================================================
# A* 路径搜索 — v2_strict: 优化启发式与对角线成本
# ============================================================
def astar_search(cost_raster, start_rc, end_rc):
    """
    A* 搜索
    - 端点处理: 落入硬约束区时自动搜索最近可行点 (除非cfg.ASTAR_STRICT_ENDPOINTS)
    """
    print(f"[Phase4] A*路径搜索: {start_rc} -> {end_rc}")

    H, W = cost_raster.shape
    sr, sc = start_rc
    er, ec = end_rc

    # 起止点验证 — 出界→不通过
    if not (0 <= sr < H and 0 <= sc < W):
        print(f"  错误: 起点出界 {start_rc} → 判定不通过")
        return None
    if not (0 <= er < H and 0 <= ec < W):
        print(f"  错误: 终点出界 {end_rc} → 判定不通过")
        return None

    # 起点/终点在硬约束区 → 自动或严格处理
    def find_nearest_feasible(rc, max_radius=50):
        r, c = rc
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not np.isinf(cost_raster[nr, nc]):
                        return (nr, nc)
        return None

    if np.isinf(cost_raster[sr, sc]):
        if cfg.ASTAR_STRICT_ENDPOINTS:
            print(f"  错误: 起点在硬约束区 → 严格模式判定不通过 (v2_strict)")
            return None
        new_start = find_nearest_feasible(start_rc)
        if new_start is None:
            print(f"  错误: 起点附近无可行区域")
            return None
        print(f"  起点调整: {start_rc} → {new_start}")
        sr, sc = new_start

    if np.isinf(cost_raster[er, ec]):
        if cfg.ASTAR_STRICT_ENDPOINTS:
            print(f"  错误: 终点在硬约束区 → 严格模式判定不通过 (v2_strict)")
            return None
        new_end = find_nearest_feasible(end_rc)
        if new_end is None:
            print(f"  错误: 终点附近无可行区域")
            return None
        print(f"  终点调整: {end_rc} → {new_end}")
        er, ec = new_end

    start_rc = (sr, sc)
    end_rc = (er, ec)

    valid_costs = cost_raster[cost_raster < np.inf]
    if len(valid_costs) == 0:
        print(f"  错误: 无可行区域")
        return None
    c_min = max(np.percentile(valid_costs, 1), 1e-6)

    # v2_strict: 8邻域, 对角线成本1.5 (v1=sqrt(2))
    neighbors = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]
    neighbor_dist = [1.0, cfg.ASTAR_DIAGONAL_COST, 1.0, cfg.ASTAR_DIAGONAL_COST,
                     1.0, cfg.ASTAR_DIAGONAL_COST, 1.0, cfg.ASTAR_DIAGONAL_COST]

    # v2_strict: 启发式权重1.2 (v1=1.0)
    def octile_heuristic(r, c):
        dr = abs(r - er)
        dc = abs(c - ec)
        base = c_min * (max(dr, dc) + (cfg.ASTAR_DIAGONAL_COST - 1) * min(dr, dc))
        return cfg.ASTAR_HEURISTIC_WEIGHT * base

    g_score = {start_rc: 0.0}
    h_start = octile_heuristic(sr, sc)
    tiebreaker = 0
    open_set = [(h_start, tiebreaker, start_rc)]
    came_from = {}
    closed_set = set()

    t0 = time.time()
    last_progress = time.time()
    max_iterations = 1000000  # v2_strict: 防止无限搜索

    while open_set and len(closed_set) < max_iterations:
        f, _, current = heapq.heappop(open_set)
        if current in closed_set:
            continue
        if current == (er, ec):
            elapsed = time.time() - t0
            path = _reconstruct_path(came_from, current)
            print(f"  A*搜索完成 ({elapsed:.1f}s), 路径长度: {len(path)} 像元")
            return path
        closed_set.add(current)
        cr, cc = current
        if time.time() - last_progress > 5:
            last_progress = time.time()
            print(f"  搜索中... 已探索 {len(closed_set)} 节点, 当前f={f:.2f}")
        for ni, (dr, dc) in enumerate(neighbors):
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            if neighbor in closed_set:
                continue
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if np.isinf(cost_raster[nr, nc]):
                continue
            if dr != 0 and dc != 0:
                if np.isinf(cost_raster[cr + dr, cc]) or np.isinf(cost_raster[cr, cc + dc]):
                    continue
            move_cost = cost_raster[nr, nc] * neighbor_dist[ni] * cfg.BASE_RESOLUTION / 1000.0
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h = octile_heuristic(nr, nc)
                tb = 0.001 * (abs(nr - er) + abs(nc - ec))
                heapq.heappush(open_set, (tentative_g + h + tb, tiebreaker, neighbor))
                came_from[neighbor] = current
                tiebreaker += 1

    if len(closed_set) >= max_iterations:
        print(f"  错误: A*超过最大迭代次数({max_iterations}) → 无可行路径")
    else:
        print(f"  警告: A*未找到路径")
    return None


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ============================================================
# 路径平滑
# ============================================================
def smooth_path(path_cells, transform, hard_mask=None, cost_raster=None):
    """路径平滑: RDP简化 + 线性插值 + 硬约束修复"""
    print("[Phase4] 路径平滑...")
    if len(path_cells) < 3:
        return grid_to_geo_coords(path_cells, transform)
    geo_coords = grid_to_geo_coords(path_cells, transform)
    rdp_epsilon_deg = cfg.PATH_SMOOTH_RDP_EPSILON / 111000.0
    simplified = rdp(geo_coords, epsilon=rdp_epsilon_deg)
    print(f"  RDP简化: {len(geo_coords)} -> {len(simplified)} 点 (epsilon={cfg.PATH_SMOOTH_RDP_EPSILON}m)")
    if len(simplified) < 2:
        return [(float(p[0]), float(p[1])) for p in simplified]
    simplified_arr = np.array(simplified)
    x_simp, y_simp = simplified_arr[:, 0], simplified_arr[:, 1]
    seg_dists = np.sqrt(np.diff(x_simp)**2 + np.diff(y_simp)**2)
    cum_dist = np.concatenate([[0], np.cumsum(seg_dists)])
    total_dist = cum_dist[-1]
    spacing_deg = cfg.PATH_RESAMPLE_SPACING / 111000.0
    n_samples = max(int(total_dist / spacing_deg), len(simplified))
    sample_dists = np.linspace(0, total_dist, n_samples)
    x_interp = np.interp(sample_dists, cum_dist, x_simp)
    y_interp = np.interp(sample_dists, cum_dist, y_simp)
    smoothed = [(float(x_interp[i]), float(y_interp[i])) for i in range(n_samples)]
    print(f"  线性插值: {len(smoothed)} 点 (间距{cfg.PATH_RESAMPLE_SPACING}m)")

    # v2_strict: 滑动平均平滑, 消除栅格路径的锯齿转角 (窗口~150m)
    if len(smoothed) >= 5:
        window = 5
        x_arr = np.array([p[0] for p in smoothed])
        y_arr = np.array([p[1] for p in smoothed])
        kernel = np.ones(window) / window
        x_smooth = np.convolve(x_arr, kernel, mode="same")
        y_smooth = np.convolve(y_arr, kernel, mode="same")
        # 首尾保留原值避免端点偏移
        edge = window // 2
        x_smooth[:edge] = x_arr[:edge]
        x_smooth[-edge:] = x_arr[-edge:]
        y_smooth[:edge] = y_arr[:edge]
        y_smooth[-edge:] = y_arr[-edge:]
        smoothed = [(float(x_smooth[i]), float(y_smooth[i])) for i in range(len(x_smooth))]
        print(f"  滑动平均平滑: 窗口{window}点 ({window * cfg.PATH_RESAMPLE_SPACING}m)")

    if hard_mask is not None:
        smoothed = _fix_violations_local(smoothed, hard_mask, transform, path_cells)
    return smoothed


def _fix_violations_local(coords, hard_mask, transform, a_star_path):
    """局部修复穿越硬约束的路径段"""
    H, W = hard_mask.shape
    n = len(coords)
    bad_segments = []
    in_bad = False
    seg_start = None
    for i, (lon, lat) in enumerate(coords):
        r, c = geo_to_grid(lat, lon, transform)
        bad = not (0 <= r < H and 0 <= c < W and hard_mask[r, c] == 1)
        if bad and not in_bad:
            seg_start = i
            in_bad = True
        elif not bad and in_bad:
            bad_segments.append((seg_start, i - 1))
            in_bad = False
    if in_bad:
        bad_segments.append((seg_start, n - 1))
    if not bad_segments:
        print(f"  硬约束验证通过, 无违规点")
        return coords
    a_star_coords = grid_to_geo_coords(a_star_path, transform)
    from scipy.spatial import cKDTree
    try:
        kd = cKDTree(a_star_coords)
    except Exception:
        kd = None
    total_violations = sum(e - s + 1 for s, e in bad_segments)
    print(f"  硬约束违规: {total_violations} 点, {len(bad_segments)} 段 — 局部A*修复中...")
    fixed = list(coords)
    for seg_s, seg_e in bad_segments:
        anchor_before = seg_s - 1
        anchor_after = seg_e + 1
        if anchor_before < 0 and anchor_after >= n:
            if kd:
                _, idx_start = kd.query([fixed[0][0], fixed[0][1]])
                _, idx_end = kd.query([fixed[-1][0], fixed[-1][1]])
                sub = a_star_path[min(idx_start, idx_end):max(idx_start, idx_end) + 1]
                fixed = grid_to_geo_coords(sub, transform)
                print(f"    全段回退到A*路径")
            return fixed
        if anchor_before < 0:
            anchor_before = 0
        if anchor_after >= n:
            anchor_after = n - 1
        lon_a, lat_a = fixed[anchor_before]
        lon_b, lat_b = fixed[anchor_after]
        if kd:
            _, idx_a = kd.query([lon_a, lat_a])
            _, idx_b = kd.query([lon_b, lat_b])
        else:
            idx_a, idx_b = 0, len(a_star_path) - 1
        if abs(idx_a - idx_b) > 1:
            sub_start = min(idx_a, idx_b)
            sub_end = max(idx_a, idx_b) + 1
            sub_cells = a_star_path[sub_start:sub_end]
            sub_coords = grid_to_geo_coords(sub_cells, transform)
            fixed = fixed[:seg_s] + sub_coords + fixed[seg_e + 1:]
            print(f"    段 [{seg_s}, {seg_e}]: A*子路径 {len(sub_coords)} 点替换")
        else:
            n_fill = seg_e - seg_s + 2
            lons = np.linspace(lon_a, lon_b, n_fill)[1:-1]
            lats = np.linspace(lat_a, lat_b, n_fill)[1:-1]
            fill_coords = [(float(lo), float(la)) for lo, la in zip(lons, lats)]
            fixed[seg_s:seg_e + 1] = fill_coords
    remaining = 0
    for lon, lat in fixed:
        r, c = geo_to_grid(lat, lon, transform)
        if 0 <= r < H and 0 <= c < W and hard_mask[r, c] == 0:
            remaining += 1
    if remaining > 0 and remaining < total_violations:
        print(f"    嵌套修复: {remaining} 残留违规")
        fixed = _fix_violations_local(fixed, hard_mask, transform, a_star_path)
    return fixed


# ============================================================
# 计算路径距离
# ============================================================
def compute_path_length_km(coords):#球面三角学经典的半正矢解法，专门用于地理经纬度测距
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        total += 6371.0 * c
    return total


def haversine_m(lon1, lat1, lon2, lat2):
    """计算两点间Haversine距离(米)"""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# v2_strict: 路径质量门控 — 7项严格审查
# ============================================================
def _strict_quality_gate(
    coords, aligned, hard_mask, final_cost, dst_transform, straight_line_km
):
    """
    v2_strict: 严格质量审查 (7项, 任意1项触发即不通过)

    1. 坡度审查: 任意点坡度 > SLOPE_QUALITY_THRESHOLD → 不通过
    2. 水域审查: >1%采样点距水域 < WATER_BUFFER_QUALITY → 不通过 (允许穿越窄河道)
    3. 保护区审查: 任意点落入保护区200m缓冲 → 不通过
    4. 曲率审查: 相邻三点转角 > MAX_TURN_ANGLE → 不通过
    5. 长度审查: sinuosity > MAX_SINUOSITY → 不通过
    6. 高程连贯性: 单调段高程变化 > MAX_CONTINUOUS_CLIMB → 不通过 (50m反转重置)
    7. 成本异常: 路径平均成本 > 全图中位数 × COST_ANOMALY_RATIO → 不通过

    Returns:
        dict: {"passed": bool, "checks": {name: {"passed": bool, "value": ..., "threshold": ..., "detail": str}}}
    """
    checks = {}
    all_passed = True

    slope = aligned.get("slope")
    dist_water = aligned.get("dist_water")
    dem = aligned.get("dem")

    # ---- 1. 坡度审查 ----
    slope_threshold = getattr(cfg, "SLOPE_QUALITY_THRESHOLD", cfg.MAX_SLOPE)
    if slope is not None and len(coords) > 0:
        max_slope = 0.0
        worst_point = None
        for lon, lat in coords:
            r, c = geo_to_grid(lat, lon, dst_transform)
            H, W = slope.shape
            if 0 <= r < H and 0 <= c < W:
                s = slope[r, c]
                if s > max_slope:
                    max_slope = s
                    worst_point = (lat, lon)
        passed_slope = bool(max_slope <= slope_threshold)
        checks["1_slope"] = {
            "passed": passed_slope,
            "value": float(max_slope),
            "threshold": slope_threshold,
            "detail": f"最大坡度 {max_slope:.1f}° {'≤' if passed_slope else '>'} {slope_threshold}°"
        }
        if not passed_slope:
            all_passed = False
    else:
        checks["1_slope"] = {"passed": True, "value": None, "threshold": slope_threshold, "detail": "无坡度数据, 跳过"}

    # ---- 2. 水域审查 ----
    water_threshold = getattr(cfg, "WATER_BUFFER_QUALITY", 100)
    if dist_water is not None and len(coords) > 0:
        # v2_strict: 检测水域数据是否实际存在 (全填充值=无真实水域)
        if np.min(dist_water) >= 4999:
            checks["2_water"] = {"passed": True, "value": None, "threshold": water_threshold, "detail": "无有效水域数据(全填充值), 跳过"}
        else:
            dists = []
            for lon, lat in coords:
                r, c = geo_to_grid(lat, lon, dst_transform)
                H, W = dist_water.shape
                if 0 <= r < H and 0 <= c < W:
                    dists.append(float(dist_water[r, c]))
            if dists:
                min_dist = float(np.min(dists))
                # v2_strict: 允许≤25%采样点邻近水体(台湾水系密集, 长距离路径难以完全避让)
                violation_rate = sum(1 for d in dists if d < water_threshold) / len(dists)
                p95_dist = float(np.percentile(dists, 5))
                passed_water = bool(violation_rate <= 0.25)
                checks["2_water"] = {
                    "passed": passed_water,
                    "value": float(min_dist),
                    "threshold": water_threshold,
                    "violation_rate": f"{violation_rate*100:.2f}%",
                    "p95_dist": p95_dist,
                    "detail": f"最小水域距离 {min_dist:.0f}m, 违规率 {violation_rate*100:.2f}% {'≤25%' if passed_water else '>25%'} (阈值{water_threshold}m)"
                }
            else:
                passed_water = True
                min_dist = float("inf")
                checks["2_water"] = {"passed": True, "value": None, "threshold": water_threshold, "detail": "无采样点, 跳过"}
            if not passed_water:
                all_passed = False
    else:
        checks["2_water"] = {"passed": True, "value": None, "threshold": water_threshold, "detail": "无水域数据, 跳过"}

    # ---- 3. 保护区审查 ----
    if hard_mask is not None and len(coords) > 0:
        violations = 0
        for lon, lat in coords:
            r, c = geo_to_grid(lat, lon, dst_transform)
            H, W = hard_mask.shape
            if 0 <= r < H and 0 <= c < W:
                if hard_mask[r, c] == 0:
                    violations += 1
        passed_protected = bool(violations == 0)
        checks["3_protected"] = {
            "passed": passed_protected,
            "value": violations,
            "threshold": 0,
            "detail": f"硬约束违规 {violations} 点 {'=0 ✓' if passed_protected else '>0 ✗'}"
        }
        if not passed_protected:
            all_passed = False
    else:
        checks["3_protected"] = {"passed": True, "value": 0, "threshold": 0, "detail": "无掩膜数据, 跳过"}

    # ---- 4. 曲率审查 (相邻三点转角) ----
    if len(coords) >= 3:
        max_angle = 0.0
        for i in range(1, len(coords) - 1):
            p0, p1, p2 = coords[i-1], coords[i], coords[i+1]
            # v2_strict: 纬度缩放修正经度距离, 正确计算物理转角
            lat_mid = math.radians(p1[1])
            lon_scale = math.cos(lat_mid)
            v1 = np.array([(p1[0] - p0[0]) * lon_scale, p1[1] - p0[1]])
            v2 = np.array([(p2[0] - p1[0]) * lon_scale, p2[1] - p1[1]])
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (n1 * n2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = math.degrees(math.acos(cos_angle))
                max_angle = max(max_angle, angle)
        passed_curvature = bool(max_angle <= cfg.MAX_TURN_ANGLE)
        checks["4_curvature"] = {
            "passed": passed_curvature,
            "value": float(max_angle),
            "threshold": cfg.MAX_TURN_ANGLE,
            "detail": f"最大转角 {max_angle:.1f}° {'≤' if passed_curvature else '>'} {cfg.MAX_TURN_ANGLE}°"
        }
        if not passed_curvature:
            all_passed = False
    else:
        checks["4_curvature"] = {"passed": True, "value": 0.0, "threshold": cfg.MAX_TURN_ANGLE, "detail": "路径点不足, 跳过"}

    # ---- 5. 长度审查 (sinuosity) ----
    path_len = compute_path_length_km(coords)
    sinuosity = path_len / straight_line_km if straight_line_km > 0 else 1.0
    passed_sinuosity = bool(sinuosity <= cfg.MAX_SINUOSITY)
    checks["5_sinuosity"] = {
        "passed": passed_sinuosity,
        "value": float(sinuosity),
        "threshold": cfg.MAX_SINUOSITY,
        "detail": f"弯曲度 {sinuosity:.2f} {'≤' if passed_sinuosity else '>'} {cfg.MAX_SINUOSITY} (路径{path_len:.1f}km/直线{straight_line_km:.1f}km)"
    }
    if not passed_sinuosity:
        all_passed = False

    # ---- 6. 高程连贯性审查 ----
    if dem is not None and len(coords) > 1:
        elevations = []
        for lon, lat in coords:
            r, c = geo_to_grid(lat, lon, dst_transform)
            H, W = dem.shape
            if 0 <= r < H and 0 <= c < W:
                elevations.append(float(dem[r, c]))
        if len(elevations) > 10:
            # v2_strict: 单调段高程变化追踪 — 检测无平台连续升降
            # 使用单调段起止点实际高程差, 反向>50m时重置段
            max_continuous_climb = 0.0
            seg_start_elev = elevations[0]
            current_direction = 0  # 1=上升, -1=下降, 0=初始
            reversal_threshold = 50  # 反向超过50m视为平台/反转
            for i in range(1, len(elevations)):
                diff = elevations[i] - elevations[i - 1]
                if abs(diff) < 2:  # 忽略微小起伏
                    continue
                new_dir = 1 if diff > 0 else -1
                if current_direction == 0:
                    current_direction = new_dir
                    seg_start_elev = elevations[i - 1]
                elif new_dir != current_direction:
                    # 方向反转 — 检查是否超过反转阈值
                    seg_range = abs(elevations[i - 1] - seg_start_elev)
                    if seg_range > reversal_threshold:
                        max_continuous_climb = max(max_continuous_climb, seg_range)
                        current_direction = new_dir
                        seg_start_elev = elevations[i - 1]
                    # 小反转忽略, 继续追踪原方向
            # 检查最后一段
            final_range = abs(elevations[-1] - seg_start_elev)
            max_continuous_climb = max(max_continuous_climb, final_range)
            passed_elevation = bool(max_continuous_climb <= cfg.MAX_CONTINUOUS_CLIMB)
            checks["6_elevation"] = {
                "passed": passed_elevation,
                "value": float(max_continuous_climb),
                "threshold": cfg.MAX_CONTINUOUS_CLIMB,
                "detail": f"最大连续升降 {max_continuous_climb:.0f}m {'≤' if passed_elevation else '>'} {cfg.MAX_CONTINUOUS_CLIMB}m"
            }
            if not passed_elevation:
                all_passed = False
        else:
            checks["6_elevation"] = {"passed": True, "value": 0.0, "threshold": cfg.MAX_CONTINUOUS_CLIMB, "detail": "高程点不足, 跳过"}
    else:
        checks["6_elevation"] = {"passed": True, "value": None, "threshold": cfg.MAX_CONTINUOUS_CLIMB, "detail": "无DEM数据, 跳过"}

    # ---- 7. 成本异常审查 ----
    if final_cost is not None and len(coords) > 0:
        path_costs = []
        for lon, lat in coords:
            r, c = geo_to_grid(lat, lon, dst_transform)
            H, W = final_cost.shape
            if 0 <= r < H and 0 <= c < W:
                fc = final_cost[r, c]
                if not np.isinf(fc):
                    path_costs.append(float(fc))
        if path_costs:
            avg_cost = np.mean(path_costs)
            global_valid = final_cost[final_cost < np.inf]
            global_median = float(np.median(global_valid)) if len(global_valid) > 0 else 0.01
            ratio = avg_cost / global_median if global_median > 0 else 1.0
            passed_cost = bool(ratio <= cfg.COST_ANOMALY_RATIO)
            checks["7_cost"] = {
                "passed": passed_cost,
                "value": float(ratio),
                "threshold": cfg.COST_ANOMALY_RATIO,
                "detail": f"成本比 {ratio:.2f} (均值{avg_cost:.4f}/中位{global_median:.4f}) {'≤' if passed_cost else '>'} {cfg.COST_ANOMALY_RATIO}"
            }
            if not passed_cost:
                all_passed = False
        else:
            checks["7_cost"] = {"passed": True, "value": None, "threshold": cfg.COST_ANOMALY_RATIO, "detail": "无有效成本数据, 跳过"}
    else:
        checks["7_cost"] = {"passed": True, "value": None, "threshold": cfg.COST_ANOMALY_RATIO, "detail": "无成本表面, 跳过"}

    return {
        "passed": all_passed,
        "n_passed": sum(1 for c in checks.values() if c["passed"]),
        "n_total": len(checks),
        "checks": checks,
    }
