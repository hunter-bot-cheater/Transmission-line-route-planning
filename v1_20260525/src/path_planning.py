"""
模块4: 路径规划
- 成本表面融合(硬约束+软约束+RF预测)
- A* 路径搜索(8邻域 Moore图, octile启发式)
- 路径平滑(RDP简化 + B样条拟合)
- 路径验证
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import heapq
import math
import time
from pathlib import Path

import config as cfg

# RDP简化
try:
    from rdp import rdp
except ImportError:
    # 内置简易RDP实现
    def rdp(points, epsilon):
        """Ramer-Douglas-Peucker 简化"""
        if len(points) < 3:
            return points

        # 找距首尾连线最远的点
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
def fuse_cost_surface(rf_cost, soft_mask, hard_mask, dist_existing=None, smooth_sigma=1.0):
    """
    融合成本表面:
    final_cost = rf_cost * soft_mask * corridor_bonus
    硬约束区域 = INF
    走廊偏好: 距现有输电线越近, 成本越低(共享基础设施)
    """
    print("[Phase4] 融合成本表面...")

    rf_cost = np.nan_to_num(rf_cost, nan=np.nanmean(rf_cost[rf_cost < np.inf]) if (rf_cost < np.inf).any() else 0.5)
    rf_cost = np.clip(rf_cost, 0, None)

    final_cost = rf_cost.copy().astype(np.float64)

    # 走廊偏好: 线性斜坡, 1000m内获得成本折扣
    if dist_existing is not None:
        corridor_bonus = 0.02 + 0.98 * np.clip(dist_existing / 1000.0, 0, 1)
        final_cost = final_cost * corridor_bonus.astype(np.float64)
        print(f"  走廊偏好已应用 (线性斜坡至1000m, 最多98%折扣)")

    # 应用软约束
    if soft_mask is not None:
        soft_mask = np.nan_to_num(soft_mask, nan=1.0)
        soft_mask = np.clip(soft_mask, 0.01, 1.0)
        final_cost = final_cost * soft_mask.astype(np.float64)

    # 确定有效区域
    if hard_mask is not None:
        valid_mask = (hard_mask == 1) & np.isfinite(final_cost)
    else:
        valid_mask = np.isfinite(final_cost)

    # 高斯平滑 (仅在有效区域内, 用有效值填充无效区)
    cost_smooth = final_cost.copy()
    cost_smooth[~valid_mask] = np.nan
    from scipy.ndimage import generic_filter
    final_cost_smooth = gaussian_filter(
        np.where(valid_mask, final_cost, 0.0),
        sigma=smooth_sigma, mode="nearest"
    )
    # 归一化: 补偿平滑造成的能量损失
    if smooth_sigma > 0:
        weights = gaussian_filter(valid_mask.astype(np.float64), sigma=smooth_sigma, mode="nearest")
        weights = np.clip(weights, 1e-6, None)
        final_cost_smooth = final_cost_smooth / weights

    # 恢复硬约束区域为inf
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
    """WGS84坐标 -> 栅格行列号"""
    col = int((lon - transform.c) / transform.a)
    row = int((lat - transform.f) / transform.e)
    return row, col


def grid_to_geo(row, col, transform):
    """栅格行列号 -> WGS84坐标"""
    lon = transform.c + col * transform.a + transform.a / 2
    lat = transform.f + row * transform.e + transform.e / 2
    return lat, lon


def grid_to_geo_coords(path_cells, transform):
    """将路径栅格坐标列表转为经纬度列表"""
    coords = []
    for r, c in path_cells:
        lon = transform.c + c * transform.a + transform.a / 2
        lat = transform.f + r * transform.e + transform.e / 2
        coords.append((lon, lat))
    return coords


# ============================================================
# A* 路径搜索
# ============================================================
def astar_search(cost_raster, start_rc, end_rc):
    """
    A* 搜索在成本栅格上寻找最优路径
    8邻域Moore图, octile距离启发式

    参数:
        cost_raster: 2D numpy array (H, W), inf=不可通行
        start_rc: (row, col)
        end_rc: (row, col)

    返回:
        list of (row, col) 路径坐标, 或None
    """
    print(f"[Phase4] A*路径搜索: {start_rc} -> {end_rc}")

    H, W = cost_raster.shape
    sr, sc = start_rc
    er, ec = end_rc

    # 验证起止点
    if not (0 <= sr < H and 0 <= sc < W):
        print(f"  错误: 起点出界 {start_rc}")
        return None
    if not (0 <= er < H and 0 <= ec < W):
        print(f"  错误: 终点出界 {end_rc}")
        return None
    if np.isinf(cost_raster[sr, sc]):
        # 将起点移出硬约束区
        print(f"  警告: 起点在硬约束区, 搜索最近的可行点...")
        sr, sc = _find_nearest_valid(cost_raster, sr, sc)
        if sr is None:
            print(f"  错误: 起点附近无可行区域")
            return None
        print(f"  起点调整为: ({sr}, {sc})")
    if np.isinf(cost_raster[er, ec]):
        print(f"  警告: 终点在硬约束区, 搜索最近的可行点...")
        er, ec = _find_nearest_valid(cost_raster, er, ec)
        if er is None:
            print(f"  错误: 终点附近无可行区域")
            return None
        print(f"  终点调整为: ({er}, {ec})")

    # 最小非零成本(用于启发式)
    valid_costs = cost_raster[cost_raster < np.inf]
    if len(valid_costs) == 0:
        print(f"  错误: 无可行区域")
        return None
    c_min = max(np.percentile(valid_costs, 1), 1e-6)

    # 8邻域: N, NE, E, SE, S, SW, W, NW
    neighbors = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]
    # 对角线 vs 正交的距离比
    neighbor_dist = [1.0, math.sqrt(2), 1.0, math.sqrt(2),
                     1.0, math.sqrt(2), 1.0, math.sqrt(2)]

    def octile_heuristic(r, c):
        """八分位距离启发式"""
        dr = abs(r - er)
        dc = abs(c - ec)
        return c_min * (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))

    # 初始化
    g_score = {start_rc: 0.0}
    h_start = octile_heuristic(sr, sc)
    tiebreaker = 0
    open_set = [(h_start, tiebreaker, start_rc)]
    came_from = {}
    closed_set = set()

    t0 = time.time()
    last_progress = time.time()

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if current == (er, ec):
            elapsed = time.time() - t0
            # 重建路径
            path = _reconstruct_path(came_from, current)
            print(f"  A*搜索完成 ({elapsed:.1f}s), 路径长度: {len(path)} 像元")
            return path

        closed_set.add(current)
        cr, cc = current

        # 进度报告
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

            # 检查对角线穿越: 两侧不能都是inf
            if dr != 0 and dc != 0:
                if np.isinf(cost_raster[cr + dr, cc]) or np.isinf(cost_raster[cr, cc + dc]):
                    continue

            move_cost = cost_raster[nr, nc] * neighbor_dist[ni] * cfg.BASE_RESOLUTION / 1000.0
            tentative_g = g_score[current] + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h = octile_heuristic(nr, nc)
                # 小tiebreaker, 偏向目标方向
                tb = 0.001 * (abs(nr - er) + abs(nc - ec))
                heapq.heappush(open_set, (tentative_g + h + tb, tiebreaker, neighbor))
                came_from[neighbor] = current
                tiebreaker += 1

    print(f"  警告: A*未找到路径")
    return None


def _reconstruct_path(came_from, current):
    """从came_from字典重建路径"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _find_nearest_valid(cost_raster, r, c, search_radius=50):
    """搜索最近的可行点"""
    H, W = cost_raster.shape
    for radius in range(1, search_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not np.isinf(cost_raster[nr, nc]):
                    return nr, nc
    return None, None


# ============================================================
# 路径平滑
# ============================================================
def smooth_path(path_cells, transform, hard_mask=None, cost_raster=None):
    """
    路径平滑:
    1. 栅格坐标 -> 地理坐标
    2. RDP简化 (保留关键拐点)
    3. 线性插值 (分段直线, 更符合真实输电线路)
    4. 等距重采样
    5. 硬约束局部A*修复 (沿原始A*路径微调, 不跳远)
    """
    print("[Phase4] 路径平滑...")

    if len(path_cells) < 3:
        return grid_to_geo_coords(path_cells, transform)

    # 1. 转为地理坐标
    geo_coords = grid_to_geo_coords(path_cells, transform)

    # 2. RDP简化 — 使用更小的epsilon保留更多细节
    rdp_epsilon_deg = cfg.PATH_SMOOTH_RDP_EPSILON / 111000.0
    simplified = rdp(geo_coords, epsilon=rdp_epsilon_deg)
    print(f"  RDP简化: {len(geo_coords)} -> {len(simplified)} 点 (epsilon={cfg.PATH_SMOOTH_RDP_EPSILON}m)")

    if len(simplified) < 2:
        return [(float(p[0]), float(p[1])) for p in simplified]

    # 3. 线性插值 — 分段直线, 等距重采样
    simplified_arr = np.array(simplified)
    x_simp, y_simp = simplified_arr[:, 0], simplified_arr[:, 1]

    # 计算分段累积弧长
    seg_dists = np.sqrt(np.diff(x_simp)**2 + np.diff(y_simp)**2)
    cum_dist = np.concatenate([[0], np.cumsum(seg_dists)])
    total_dist = cum_dist[-1]

    spacing_deg = cfg.PATH_RESAMPLE_SPACING / 111000.0
    n_samples = max(int(total_dist / spacing_deg), len(simplified))
    sample_dists = np.linspace(0, total_dist, n_samples)

    # 线性插值
    x_interp = np.interp(sample_dists, cum_dist, x_simp)
    y_interp = np.interp(sample_dists, cum_dist, y_simp)
    smoothed = [(float(x_interp[i]), float(y_interp[i])) for i in range(n_samples)]
    print(f"  线性插值: {len(smoothed)} 点 (间距{cfg.PATH_RESAMPLE_SPACING}m)")

    # 4. 硬约束检测与局部修复
    if hard_mask is not None:
        smoothed = _fix_violations_local(smoothed, hard_mask, transform, path_cells)

    return smoothed


def _fix_violations_local(coords, hard_mask, transform, a_star_path):
    """
    局部修复穿越硬约束的路径段。
    对违规段, 在A*路径上找该段首尾的最近点, 用A*子路径替换。
    这比"跳到最近有效像元"更忠实于走廊偏好。
    """
    H, W = hard_mask.shape
    n = len(coords)

    # 找出所有违规段
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

    # 构建A*路径的栅格索引以便快速查找最近点
    a_star_grid = {(r, c): geo_idx for geo_idx, (r, c) in enumerate(a_star_path)}
    a_star_coords = [(c, r) for r, c in a_star_path]  # (col, row) for spatial KD-tree

    from scipy.spatial import cKDTree
    try:
        kd = cKDTree(a_star_coords)
    except Exception:
        kd = None

    total_violations = sum(e - s + 1 for s, e in bad_segments)
    print(f"  硬约束违规: {total_violations} 点, {len(bad_segments)} 段 — 局部A*修复中...")

    fixed = list(coords)

    for seg_s, seg_e in bad_segments:
        # 找到段前和段后的有效锚点
        anchor_before = seg_s - 1
        anchor_after = seg_e + 1

        if anchor_before < 0 and anchor_after >= n:
            # 全路径无效 — 回退到A*路径
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
        ra, ca = geo_to_grid(lat_a, lon_a, transform)
        rb, cb = geo_to_grid(lat_b, lon_b, transform)

        # 在A*路径中找锚点对应的最近点
        if kd:
            _, idx_a = kd.query([lon_a, lat_a])
            _, idx_b = kd.query([lon_b, lat_b])
        else:
            idx_a, idx_b = 0, len(a_star_path) - 1

        if abs(idx_a - idx_b) > 1:
            # 用A*子路径替换违规段
            sub_start = min(idx_a, idx_b)
            sub_end = max(idx_a, idx_b) + 1
            sub_cells = a_star_path[sub_start:sub_end]
            sub_coords = grid_to_geo_coords(sub_cells, transform)

            # 替换
            fixed = fixed[:seg_s] + sub_coords + fixed[seg_e + 1:]
            print(f"    段 [{seg_s}, {seg_e}]: A*子路径 {len(sub_coords)} 点替换")
        else:
            # 太短, 简单线性插值并检查
            n_fill = seg_e - seg_s + 2
            lons = np.linspace(lon_a, lon_b, n_fill)[1:-1]
            lats = np.linspace(lat_a, lat_b, n_fill)[1:-1]
            fill_coords = [(float(lo), float(la)) for lo, la in zip(lons, lats)]
            fixed[seg_s:seg_e + 1] = fill_coords

    # 递归检查一次, 防止嵌套违规
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
def compute_path_length_km(coords):
    """计算路径长度(km) - coords为(lon, lat)列表"""
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        # Haversine
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        total += 6371.0 * c
    return total
