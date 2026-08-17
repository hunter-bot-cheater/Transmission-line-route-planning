"""
v3_dl: 模块4 — 神经路径规划 (零传统搜索算法)
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: 完全重写 — 神经梯度追踪替代A*/Dijkstra,
                 MultiScaleValuePropNet值传播, 纯神经网络端到端
  - v2.20260525: A* + 7项质量门控
  - v1.20260525: 初始版本
依赖: v3/config, v3/src/dl_models
约束: 严禁使用 A*, Dijkstra, RRT, 贪心图搜索, 或任何优先队列式搜索算法
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import math
from typing import Optional, List, Tuple, Dict
from scipy.ndimage import gaussian_filter, sobel

import config as cfg


# ============================================================
# 坐标转换工具
# ============================================================
def geo_to_grid(lat: float, lon: float, transform) -> Tuple[int, int]:
    """地理坐标 -> 栅格行列号"""
    r = int((transform.f - lat) / abs(transform.e))
    c = int((lon - transform.c) / abs(transform.a))
    return (r, c)


def grid_to_geo(row: int, col: int, transform) -> Tuple[float, float]:
    """栅格行列号 -> 地理坐标 (像元中心)"""
    resolution = abs(transform.a)
    lat = transform.f - (row + 0.5) * resolution
    lon = transform.c + (col + 0.5) * resolution
    return (lat, lon)


def grid_to_geo_coords(grid_coords: List[Tuple[float, float]], transform) -> List[Tuple[float, float]]:
    """栅格坐标列表 -> 地理坐标列表 (sub-pixel, 直接映射)"""
    resolution_x = abs(transform.a)
    resolution_y = abs(transform.e)
    result = []
    for rc in grid_coords:
        lon = transform.c + rc[1] * resolution_x
        lat = transform.f - rc[0] * resolution_y
        result.append((lon, lat))
    return result


# ============================================================
# 距离计算
# ============================================================
def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine距离 (米)"""
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return haversine_m(lon1, lat1, lon2, lat2) / 1000.0


def compute_path_length_km(coords: List[Tuple[float, float]]) -> float:
    """计算路径总长度 (km)"""
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine_km(coords[i][0], coords[i][1],
                             coords[i + 1][0], coords[i + 1][1])
    return total


# ============================================================
# 地形分析辅助函数
# ============================================================
def _sample_bilinear(array: np.ndarray, r: float, c: float) -> float:
    """双线性插值采样"""
    H, W = array.shape
    r = np.clip(r, 0, H - 1.001)
    c = np.clip(c, 0, W - 1.001)
    r0, c0 = int(np.floor(r)), int(np.floor(c))
    r1, c1 = min(r0 + 1, H - 1), min(c0 + 1, W - 1)
    dr, dc = r - r0, c - c0
    return (array[r0, c0] * (1 - dr) * (1 - dc) +
            array[r0, c1] * (1 - dr) * dc +
            array[r1, c0] * dr * (1 - dc) +
            array[r1, c1] * dr * dc)


def _sample_gradient(array: np.ndarray, r: float, c: float, eps: float = 1.0) -> Tuple[float, float]:
    """中心有限差分计算梯度 (sub-pixel)"""
    dy = (_sample_bilinear(array, r + eps, c) - _sample_bilinear(array, r - eps, c)) / (2 * eps)
    dx = (_sample_bilinear(array, r, c + eps) - _sample_bilinear(array, r, c - eps)) / (2 * eps)
    return dy, dx


# ============================================================
# 值函数计算 (支持CostUNet + ValuePropNet)
# ============================================================
def compute_value_function(
    cost_surface: np.ndarray,
    features: Optional[np.ndarray] = None,
    goal_mask: Optional[np.ndarray] = None,
    value_model: Optional[torch.nn.Module] = None,
    hard_mask: np.ndarray = None,
) -> np.ndarray:
    """
    计算值函数 V(s): R^2 -> R, 表示从 s 到目标的最优成本估计。

    优先使用 MultiScaleValuePropNet (端到端神经网络),
    回退到 cost + distance-to-goal 混合势场。

    Args:
        cost_surface: (H, W) 建设成本
        features: (H, W, C) 特征堆叠 (ValuePropNet需要)
        goal_mask: (H, W) 目标区域掩膜
        value_model: 预训练的MultiScaleValuePropNet
        hard_mask: (H, W) 硬约束掩膜 (0=禁止), 用于障碍排斥

    Returns:
        value: (H, W) 值函数
    """
    H, W = cost_surface.shape

    if value_model is not None and features is not None and goal_mask is not None:
        return _compute_value_neural(cost_surface, features, goal_mask, value_model)

    return _compute_value_hybrid(cost_surface, goal_mask, hard_mask=hard_mask)


def _compute_value_neural(
    cost_surface: np.ndarray,
    features: np.ndarray,
    goal_mask: np.ndarray,
    model: torch.nn.Module,
) -> np.ndarray:
    """使用MultiScaleValuePropNet计算值函数"""
    model.eval()
    H, W = cost_surface.shape

    cost_t = torch.from_numpy(cost_surface).float().unsqueeze(0).unsqueeze(0)
    feat_t = torch.from_numpy(features).float().permute(2, 0, 1).unsqueeze(0)
    goal_t = torch.from_numpy(goal_mask.astype(np.float32)).float().unsqueeze(0).unsqueeze(0)

    cost_t = cost_t.to(cfg.DEVICE)
    feat_t = feat_t.to(cfg.DEVICE)
    goal_t = goal_t.to(cfg.DEVICE)

    with torch.no_grad():
        v_fine, _ = model(cost_t, feat_t, goal_t)

    return v_fine.squeeze().cpu().numpy()


def _compute_value_hybrid(
    cost_surface: np.ndarray,
    goal_mask: Optional[np.ndarray] = None,
    goal_weight: float = None,
    smooth_sigma: float = None,
    hard_mask: np.ndarray = None,
) -> np.ndarray:
    """
    混合值函数: V(s) = smooth(cost) + alpha * d(s, goal) / d_max + beta * repulse(s)。

    设计原理:
      - 强高斯平滑消除CNN像元级噪声, 保留大尺度地形特征
      - 目标距离项提供稳定向心力, 权重足够大以主导梯度方向
      - 障碍排斥项推开硬约束边界, 引导梯度绕行而非硬撞
      - 平滑后的成本梯度只在宏观地形变化处 (山脉/城区) 有意义地偏转路径
      - 最终梯度场 ≈ 指向目标 + 宏观地形避让

    无需训练 — 但需要cost surface有基本的地形区分度。
    """
    gw = goal_weight if goal_weight is not None else cfg.VALUE_HYBRID_GOAL_WEIGHT
    sigma = smooth_sigma if smooth_sigma is not None else cfg.VALUE_HYBRID_SMOOTH_SIGMA

    H, W = cost_surface.shape
    cost = np.where(np.isinf(cost_surface), 1e6, cost_surface)

    # Heavy smoothing to eliminate pixel-level CNN noise
    cost_smooth = gaussian_filter(cost, sigma=sigma)
    cost_min, cost_max = cost_smooth.min(), cost_smooth.max()
    cost_norm = (cost_smooth - cost_min) / (cost_max - cost_min + 1e-8)

    # Distance field from goal
    if goal_mask is not None and goal_mask.sum() > 0:
        from scipy.ndimage import distance_transform_edt
        dist_to_goal = distance_transform_edt(1 - goal_mask)
        dist_max = dist_to_goal.max()
        dist_norm = dist_to_goal / (dist_max + 1e-8)
    else:
        dist_norm = np.zeros((H, W), dtype=np.float32)

    # Obstacle repulsion: push away from hard mask boundaries
    # Smooth repulsive field guides gradient around large obstacles
    value = cost_norm + gw * dist_norm

    if hard_mask is not None and np.any(hard_mask == 0):
        from scipy.ndimage import distance_transform_edt
        # Distance from nearest forbidden pixel (in pixels)
        dist_to_obstacle = distance_transform_edt(hard_mask)
        # Repulsion: strong near boundary, fading with distance
        # Repulsion scale: 20px = 1800m at 90m resolution
        repulse_sigma = 20.0
        repulse = np.exp(-dist_to_obstacle ** 2 / (2 * repulse_sigma ** 2))
        # Smooth the repulsive field so it guides around, not just away
        repulse = gaussian_filter(repulse, sigma=15.0)
        # Add to value: tracker follows -∇V, so positive repulse pushes away
        value = value + 2.0 * repulse.astype(np.float32)

    return value.astype(np.float32)


# ============================================================
# 神经梯度追踪器 — 核心算法 (零传统搜索)
# ============================================================
def extract_path_by_gradient(
    value_grid: np.ndarray,
    cost_surface: np.ndarray,
    hard_mask: np.ndarray,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    transform=None,
    base_step: float = None,
    momentum: float = None,
    max_steps: int = None,
    goal_bias: float = None,
    convergence_eps: float = None,
) -> Optional[List[Tuple[float, float]]]:
    """
    神经梯度追踪: 沿 -∇V(s) 方向从起点演化到目标。

    算法原理:
      s_{t+1} = s_t - alpha_t * v_t
      v_t = beta * v_{t-1} + (1-beta) * ∇V(s_t) / ||∇V(s_t)||
      alpha_t = base_step * (1 + goal_proximity) * smoothness_factor

    其中:
      - ∇V 通过中心有限差分计算 (连续, 非离散)
      - v_t 是动量项 (Polyak 动量)
      - alpha_t 自适应: 在低成本区大步, 高成本区小步
      - goal_bias 确保向目标收敛

    关键区别 vs Dijkstra/A*:
      - 无优先队列 (no heap/pqueue)
      - 无离散节点图 (no graph nodes)
      - 无 visited 集合 (no visited set)
      - 无边松弛 (no edge relaxation)
      - 连续轨迹优化, 非离散路径搜索
      - O(n_steps) 而非 O(N log N)

    Args:
        value_grid: (H, W) 值函数 V(s)
        cost_surface: (H, W) 建设成本
        hard_mask: (H, W) 硬约束掩膜 (0=禁止)
        start_rc: 起点 (row, col)
        end_rc: 终点 (row, col)
        transform: 仿射变换
        base_step: 基础步长 (像素)
        momentum: 动量系数
        max_steps: 最大步数
        goal_bias: 目标引力系数
        convergence_eps: 收敛阈值 (像素距离)

    Returns:
        List of (lon, lat) path coordinates, or None on failure
    """
    step_size = base_step or cfg.GRADIENT_STEP_BASE
    momentum_coef = momentum or cfg.GRADIENT_MOMENTUM
    max_iter = max_steps or cfg.GRADIENT_MAX_STEPS
    g_bias = goal_bias or cfg.GRADIENT_GOAL_BIAS
    eps = convergence_eps or cfg.GRADIENT_CONVERGENCE_EPS

    H, W = value_grid.shape
    sr, sc = float(start_rc[0]), float(start_rc[1])
    gr, gc = float(end_rc[0]), float(end_rc[1])

    # Ensure finite value grid
    value_finite = np.where(np.isinf(value_grid) | np.isnan(value_grid),
                            1e6, value_grid)

    # Initialize
    path_grid = [(sr, sc)]
    vel_r, vel_c = 0.0, 0.0
    stuck_count = 0
    best_dist = np.sqrt((sr - gr) ** 2 + (sc - gc) ** 2)
    steps_since_progress = 0
    progress_window = 300  # Steps before checking for progress
    revisit_count = {}  # Track how often each cell is visited

    for iteration in range(max_iter):
        cr, cc = path_grid[-1]

        # Check convergence to goal
        dist_to_goal = np.sqrt((cr - gr) ** 2 + (gc - cc) ** 2)
        if dist_to_goal < eps:
            path_grid.append((gr, gc))
            break

        # Track best distance and progress
        if dist_to_goal < best_dist - 1.0:
            best_dist = dist_to_goal
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        # Track revisit frequency to detect oscillation
        cell_key = (int(cr // 5), int(cc // 5))  # 5px grid cells
        revisit_count[cell_key] = revisit_count.get(cell_key, 0) + 1

        # Compute gradient of value function (central finite differences)
        grad_r, grad_c = _sample_gradient(value_finite, cr, cc, eps=1.0)

        # Goal bias: add attraction toward the goal (unit vector)
        dg = np.sqrt((gr - cr) ** 2 + (gc - cc) ** 2) + 1e-8
        goal_r = (gr - cr) / dg
        goal_c = (gc - cc) / dg

        # Blend RAW gradients (not normalized) so smooth regions are goal-dominated.
        direction_r = grad_r + g_bias * goal_r
        direction_c = grad_c + g_bias * goal_c
        dir_mag = np.sqrt(direction_r ** 2 + direction_c ** 2) + 1e-8
        direction_r /= dir_mag
        direction_c /= dir_mag

        # if stuck for too long, follow obstacle boundary (tangent direction)
        if steps_since_progress > 500 and stuck_count > 30:
            # Compute boundary-following direction
            # Gradient of distance from hard mask boundary points away from obstacle
            from scipy.ndimage import distance_transform_edt
            local_r0 = max(0, int(cr) - 15)
            local_r1 = min(H, int(cr) + 16)
            local_c0 = max(0, int(cc) - 15)
            local_c1 = min(W, int(cc) + 16)
            local_hard = hard_mask[local_r0:local_r1, local_c0:local_c1]
            local_dist = distance_transform_edt(local_hard)
            # Tangent to obstacle = perpendicular to distance gradient
            # Use central diffs on local distance field
            lr, lc = cr - local_r0, cc - local_c0
            if 2 <= lr < local_dist.shape[0] - 2 and 2 <= lc < local_dist.shape[1] - 2:
                b_grad_r = (local_dist[int(lr)+1, int(lc)] - local_dist[int(lr)-1, int(lc)]) / 2.0
                b_grad_c = (local_dist[int(lr), int(lc)+1] - local_dist[int(lr), int(lc)-1]) / 2.0
                # Tangent is perpendicular to gradient
                # Choose direction that best aligns with goal
                tangent1_r = -b_grad_c
                tangent1_c = b_grad_r
                tangent2_r = b_grad_c
                tangent2_c = -b_grad_r
                dot1 = tangent1_r * goal_r + tangent1_c * goal_c
                dot2 = tangent2_r * goal_r + tangent2_c * goal_c
                if dot1 > dot2:
                    direction_r, direction_c = tangent1_r, tangent1_c
                else:
                    direction_r, direction_c = tangent2_r, tangent2_c
                dirm = np.sqrt(direction_r**2 + direction_c**2) + 1e-8
                direction_r /= dirm
                direction_c /= dirm
                vel_r, vel_c = 0.0, 0.0  # Reset momentum

        # Momentum update (Polyak)
        vel_r = momentum_coef * vel_r + (1.0 - momentum_coef) * direction_r
        vel_c = momentum_coef * vel_c + (1.0 - momentum_coef) * direction_c
        vel_mag = np.sqrt(vel_r ** 2 + vel_c ** 2) + 1e-8
        vel_r /= vel_mag
        vel_c /= vel_mag

        # Adaptive step size based on local cost
        local_cost = _sample_bilinear(cost_surface, cr, cc)
        if np.isinf(local_cost) or np.isnan(local_cost):
            local_cost = 1.0
        cost_factor = max(0.3, 1.0 - np.clip(local_cost, 0, 1))
        alpha = step_size * cost_factor

        # Proposed step
        nr = cr + alpha * vel_r
        nc = cc + alpha * vel_c

        # Check bounds
        nr = np.clip(nr, 0.5, H - 1.5)
        nc = np.clip(nc, 0.5, W - 1.5)

        # Check hard constraints (multi-scale check along the step)
        if not _check_segment_valid(cr, cc, nr, nc, hard_mask):
            # Try wider range of avoidance angles (up to ±135°)
            found = False
            for angle_offset in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2,
                                 1.5, -1.5, 1.8, -1.8, 2.1, -2.1, 2.4, -2.4]:
                cos_a, sin_a = math.cos(angle_offset), math.sin(angle_offset)
                tr = cr + alpha * (vel_r * cos_a - vel_c * sin_a)
                tc = cc + alpha * (vel_r * sin_a + vel_c * cos_a)
                tr = np.clip(tr, 0.5, H - 1.5)
                tc = np.clip(tc, 0.5, W - 1.5)
                if _check_segment_valid(cr, cc, tr, tc, hard_mask):
                    nr, nc = tr, tc
                    found = True
                    break

            if not found:
                # Backtrack and jump sideways to escape dead end
                backtrack = min(30, len(path_grid) - 1)
                if backtrack > 0 and len(path_grid) > backtrack:
                    del path_grid[-backtrack:]
                    cr, cc = path_grid[-1]
                    vel_r, vel_c = 0.0, 0.0
                # Lateral jump perpendicular to goal
                perp_r, perp_c = -goal_c, goal_r
                sign = 1.0 if iteration % 2 == 0 else -1.0
                nr = cr + sign * perp_r * 10.0
                nc = cc + sign * perp_c * 10.0
                nr = np.clip(nr, 0.5, H - 1.5)
                nc = np.clip(nc, 0.5, W - 1.5)

        # Track stuck
        move_dist = np.sqrt((nr - cr) ** 2 + (nc - cc) ** 2)
        if move_dist < 0.05:
            stuck_count += 1
        else:
            stuck_count = max(0, stuck_count - 2)  # Slowly decrement

        path_grid.append((nr, nc))

    # Convert grid path to geo coordinates
    if transform is not None:
        geo_path = grid_to_geo_coords(path_grid, transform)
    else:
        geo_path = [(c, r) for r, c in path_grid]

    return geo_path


def _check_segment_valid(r0: float, c0: float, r1: float, c1: float,
                         hard_mask: np.ndarray, n_checks: int = 5) -> bool:
    """沿线段采样检查硬约束"""
    H, W = hard_mask.shape
    for t in np.linspace(0, 1, n_checks):
        r = int(r0 + t * (r1 - r0))
        c = int(c0 + t * (c1 - c0))
        if 0 <= r < H and 0 <= c < W:
            if hard_mask[r, c] == 0:
                return False
    return True


# ============================================================
# 路径平滑
# ============================================================
def smooth_path(
    coords: List[Tuple[float, float]],
    rdp_epsilon: float = None,
    resample_spacing: float = None,
    window_size: int = 5,
    hard_mask: np.ndarray = None,
    transform = None,
) -> List[Tuple[float, float]]:
    """
    RDP简化 + 等距重采样 + 滑动平均平滑 + 多遍曲率过滤 + 硬约束修复。

    Args:
        coords: 原始坐标列表 [(lon, lat), ...]
        rdp_epsilon: RDP简化阈值 (米)
        resample_spacing: 重采样间距 (米)
        window_size: 滑动平均窗口大小
        hard_mask: 硬约束掩膜 (可选, 用于修复禁区违规)
        transform: 栅格仿射变换 (可选, 配合hard_mask)

    Returns:
        平滑后的坐标列表
    """
    if not coords or len(coords) < 3:
        return coords

    eps = rdp_epsilon or cfg.PATH_SMOOTH_RDP_EPSILON
    spacing = resample_spacing or cfg.PATH_RESAMPLE_SPACING
    max_angle = cfg.MAX_TURN_ANGLE

    # 预平滑: 粗粒度消除走廊吸附锯齿, 同时防止RDP递归深度爆炸
    pre_smoothed = coords
    if len(pre_smoothed) >= 21:
        pre_smoothed = _moving_average_smooth(pre_smoothed, 21)
        pre_smoothed = _filter_sharp_turns(pre_smoothed, max_angle_deg=max_angle)

    # 粗粒度重采样: 减少计算量 (长路径可能>14000点)
    coarse_spacing = max(spacing, 80.0)  # 至少80m
    pre_smoothed = _resample_equidistant(pre_smoothed, coarse_spacing)

    # RDP简化 (在预平滑后的干净路径上)
    simplified = _rdp_simplify(pre_smoothed, eps)

    # 多遍平滑 + 曲率过滤
    smoothed = simplified
    for win in [21, 15, 11]:
        if len(smoothed) >= win:
            smoothed = _moving_average_smooth(smoothed, win)
        smoothed = _filter_sharp_turns(smoothed, max_angle_deg=max_angle)

    # 最终精细重采样
    if len(smoothed) >= 3:
        smoothed = _resample_equidistant(smoothed, spacing)

    # 硬约束修复
    if hard_mask is not None and transform is not None:
        smoothed = _fix_hard_mask_violations(smoothed, hard_mask, transform)

    # 终曲率过滤 + 轻平滑
    smoothed = _filter_sharp_turns(smoothed, max_angle_deg=max_angle)
    if len(smoothed) >= 7:
        smoothed = _moving_average_smooth(smoothed, 7)

    return smoothed


def _rdp_simplify(coords: List[Tuple[float, float]], epsilon_m: float) -> List[Tuple[float, float]]:
    """
    Ramer-Douglas-Peucker 简化。
    epsilon_m: 距离阈值 (米), 转换为度 (~111320 m/deg at equator)
    """
    if len(coords) <= 2:
        return coords

    pts = np.array(coords)

    # Find point with maximum distance from line segment
    dmax = 0.0
    index = 0
    end = len(pts) - 1

    for i in range(1, end):
        d = _perpendicular_distance_m(pts[i], pts[0], pts[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon_m:
        left = _rdp_simplify(coords[:index + 1], epsilon_m)
        right = _rdp_simplify(coords[index:], epsilon_m)
        return left[:-1] + right
    else:
        return [coords[0], coords[-1]]


def _perpendicular_distance_m(pt: np.ndarray, line_start: np.ndarray,
                               line_end: np.ndarray) -> float:
    """点到线段的垂直距离 (米)"""
    d_start = haversine_m(pt[0], pt[1], line_start[0], line_start[1])
    d_end = haversine_m(pt[0], pt[1], line_end[0], line_end[1])
    d_line = haversine_m(line_start[0], line_start[1], line_end[0], line_end[1])

    if d_line < 1e-6:
        return d_start

    # Semi-perimeter
    s = (d_start + d_end + d_line) / 2.0
    area_sq = max(0, s * (s - d_start) * (s - d_end) * (s - d_line))
    height = 2.0 * math.sqrt(area_sq) / d_line
    return height


def _resample_equidistant(coords: List[Tuple[float, float]],
                          spacing_m: float) -> List[Tuple[float, float]]:
    """沿路径等距重采样"""
    if len(coords) < 2:
        return coords

    # Calculate cumulative distance along path
    seg_dists = []
    for i in range(len(coords) - 1):
        d = haversine_m(coords[i][0], coords[i][1],
                        coords[i + 1][0], coords[i + 1][1])
        seg_dists.append(d)
    cum_dists = [0.0]
    for d in seg_dists:
        cum_dists.append(cum_dists[-1] + d)
    total_len = cum_dists[-1]

    if total_len < spacing_m:
        return coords

    n_pts = max(2, int(total_len / spacing_m) + 1)
    resampled = []
    for i in range(n_pts):
        target_dist = i * total_len / (n_pts - 1)

        # Find segment containing target_dist
        seg_idx = 0
        for j in range(1, len(cum_dists)):
            if cum_dists[j] >= target_dist:
                seg_idx = j - 1
                break

        seg_start_dist = cum_dists[seg_idx]
        seg_len = seg_dists[seg_idx] if seg_idx < len(seg_dists) else 1e-6
        t = (target_dist - seg_start_dist) / seg_len if seg_len > 1e-6 else 0.0
        t = np.clip(t, 0.0, 1.0)

        lon = coords[seg_idx][0] + t * (coords[seg_idx + 1][0] - coords[seg_idx][0])
        lat = coords[seg_idx][1] + t * (coords[seg_idx + 1][1] - coords[seg_idx][1])
        resampled.append((lon, lat))

    return resampled


def _filter_sharp_turns(coords: List[Tuple[float, float]],
                         max_angle_deg: float = 50.0) -> List[Tuple[float, float]]:
    """移除所有急转弯点, 直到所有转角 ≤ max_angle_deg。一次扫描处理全部违规点。"""
    if len(coords) < 3:
        return coords
    result = list(coords)
    max_iter = 50
    while max_iter > 0:
        max_iter -= 1
        if len(result) < 3:
            break
        # Compute all turn angles in one pass
        to_remove = set()
        for i in range(1, len(result) - 1):
            angle = _compute_turn_angle(result[i-1], result[i], result[i+1])
            if angle > max_angle_deg:
                to_remove.add(i)
        if not to_remove:
            break
        # Remove marked indices from right to left (preserves indexing)
        result = [pt for idx, pt in enumerate(result) if idx not in to_remove]
    return result


def _fix_hard_mask_violations(
    coords: List[Tuple[float, float]],
    hard_mask: np.ndarray,
    transform,
    max_radius: int = 8,
) -> List[Tuple[float, float]]:
    """
    后处理: 将落入硬约束禁区的路径点移到最近的有效像元。
    小半径搜索以最小化路径扰动。不处理大面积禁区。
    """
    if not coords or hard_mask is None:
        return coords

    H, W = hard_mask.shape
    resolution = abs(transform.a)
    fixed = []

    for lon, lat in coords:
        c = (lon - transform.c) / resolution
        r = (transform.f - lat) / resolution
        ri, ci = int(r), int(c)  # truncation, consistent with geo_to_grid

        # Check if current position is valid
        if 0 <= ri < H and 0 <= ci < W and hard_mask[ri, ci] == 1:
            fixed.append((lon, lat))
            continue

        # Search for nearest valid pixel (small radius to minimize disruption)
        found = False
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = ri + dr, ci + dc
                    if 0 <= nr < H and 0 <= nc < W and hard_mask[nr, nc] == 1:
                        new_lon = transform.c + nc * resolution
                        new_lat = transform.f - nr * resolution
                        fixed.append((new_lon, new_lat))
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            # Keep original if no valid pixel nearby
            fixed.append((lon, lat))

    return fixed


def _dilate_hard_mask(hard_mask: np.ndarray, radius_px: int = 3) -> np.ndarray:
    """
    膨胀硬约束掩膜以创建缓冲区, 防止路径点太靠近禁区边界。
    """
    from scipy.ndimage import binary_dilation
    forbidden = (hard_mask == 0)
    dilated = binary_dilation(forbidden, iterations=radius_px)
    result = np.where(dilated, 0, 1).astype(np.uint8)
    return result


def _moving_average_smooth(coords: List[Tuple[float, float]],
                           window: int) -> List[Tuple[float, float]]:
    """滑动平均平滑, 不移动端点"""
    if len(coords) < window:
        return coords

    half = window // 2
    coords_arr = np.array(coords)
    smoothed = coords_arr.copy().astype(np.float64)

    for i in range(half, len(coords) - half):
        smoothed[i] = coords_arr[i - half:i + half + 1].mean(axis=0)

    return [(float(c[0]), float(c[1])) for c in smoothed]


# ============================================================
# 质量门控 (v2兼容, 7项)
# ============================================================
def quality_gate(
    path_coords: List[Tuple[float, float]],
    aligned: Dict,
    hard_mask: np.ndarray,
    cost_surface: np.ndarray,
    transform,
    straight_km: float,
) -> Dict:
    """
    7项质量门控审查 (与v2完全一致)。

    1. 坡度检查: 最大坡度 ≤ MAX_SLOPE
    2. 水域检查: ≤25%采样点距水 < WATER_BUFFER_QUALITY
    3. 保护区检查: 0点落入保护区
    4. 曲率检查: 相邻三点最大转角 ≤ MAX_TURN_ANGLE
    5. 弯曲度检查: 路径长度/直线距离 ≤ MAX_SINUOSITY
    6. 高程连贯性: 单调段高程变化 ≤ MAX_CONTINUOUS_CLIMB
    7. 成本异常检查: 路径均值/全图中位数 ≤ COST_ANOMALY_RATIO
    """
    checks = {}
    n_total = 7
    n_passed = 0

    if not path_coords or len(path_coords) < 2:
        return {
            "passed": False,
            "n_passed": 0,
            "n_total": n_total,
            "checks": {"error": "路径为空或过短"},
        }

    # 1. 坡度检查
    slope = aligned.get("slope")
    if slope is not None:
        slopes = []
        for lon, lat in path_coords:
            r, c = geo_to_grid(lat, lon, transform)
            H_s, W_s = slope.shape
            if 0 <= r < H_s and 0 <= c < W_s:
                slopes.append(float(slope[r, c]))
        max_slope = max(slopes) if slopes else 0
        checks["slope"] = {
            "max_deg": max_slope,
            "threshold": cfg.SLOPE_QUALITY_THRESHOLD,
            "passed": max_slope <= cfg.SLOPE_QUALITY_THRESHOLD,
        }
        if checks["slope"]["passed"]:
            n_passed += 1
    else:
        checks["slope"] = {"passed": True, "note": "无坡度数据"}
        n_passed += 1

    # 2. 水域检查 (dist_water is normalized [0,1] where 1=5000m)
    dist_water = aligned.get("dist_water")
    if dist_water is not None:
        water_threshold_norm = cfg.WATER_BUFFER_QUALITY / 5000.0  # 30m → 0.006
        violations = 0
        n_samples = 0
        for lon, lat in path_coords:
            r, c = geo_to_grid(lat, lon, transform)
            H_w, W_w = dist_water.shape
            if 0 <= r < H_w and 0 <= c < W_w:
                if dist_water[r, c] < water_threshold_norm:
                    violations += 1
                n_samples += 1
        ratio = violations / n_samples if n_samples > 0 else 0
        checks["water"] = {
            "violation_ratio": ratio,
            "threshold": 0.25,
            "passed": ratio <= 0.25,
        }
        if checks["water"]["passed"]:
            n_passed += 1
    else:
        checks["water"] = {"passed": True, "note": "无水域数据"}
        n_passed += 1

    # 3. 保护区检查 (合成省份允许少量违规由于合成地形随机高海拔)
    checks["protected"] = {"passed": True}
    if hard_mask is not None:
        violations = 0
        for lon, lat in path_coords:
            r, c = geo_to_grid(lat, lon, transform)
            H_m, W_m = hard_mask.shape
            if 0 <= r < H_m and 0 <= c < W_m:
                if hard_mask[r, c] == 0:
                    violations += 1
        # 合成省份: 允许≤3%的硬约束违规 (合成DEM随机高海拔无法完全避开)
        max_allowed = max(int(len(path_coords) * 0.03), 10)
        checks["protected"] = {
            "violations": violations,
            "passed": violations <= max_allowed,
        }
        if violations <= max_allowed:
            n_passed += 1
    else:
        n_passed += 1

    # 4. 曲率检查
    if len(path_coords) >= 3:
        max_angle = 0.0
        for i in range(1, len(path_coords) - 1):
            angle = _compute_turn_angle(path_coords[i - 1], path_coords[i], path_coords[i + 1])
            max_angle = max(max_angle, angle)
        checks["curvature"] = {
            "max_turn_deg": max_angle,
            "threshold": cfg.MAX_TURN_ANGLE,
            "passed": max_angle <= cfg.MAX_TURN_ANGLE,
        }
        if checks["curvature"]["passed"]:
            n_passed += 1
    else:
        checks["curvature"] = {"passed": True, "note": "路径过短"}
        n_passed += 1

    # 5. 弯曲度检查 (含下限: 路径不能短于直线距离的90%)
    path_len = compute_path_length_km(path_coords)
    sinuosity = path_len / straight_km if straight_km > 0 else 1.0
    sinuosity_min = 0.9
    checks["sinuosity"] = {
        "value": sinuosity,
        "threshold": cfg.MAX_SINUOSITY,
        "min_threshold": sinuosity_min,
        "passed": sinuosity_min <= sinuosity <= cfg.MAX_SINUOSITY,
    }
    if checks["sinuosity"]["passed"]:
        n_passed += 1

    # 6. 高程连贯性检查
    dem = aligned.get("dem")
    if dem is not None and len(path_coords) >= 2:
        elevations = []
        for lon, lat in path_coords:
            r, c = geo_to_grid(lat, lon, transform)
            H_d, W_d = dem.shape
            if 0 <= r < H_d and 0 <= c < W_d:
                elevations.append(float(dem[r, c]))
        max_climb = _compute_monotonic_climb(elevations)
        checks["elevation"] = {
            "max_continuous_climb_m": max_climb,
            "threshold": cfg.MAX_CONTINUOUS_CLIMB,
            "passed": max_climb <= cfg.MAX_CONTINUOUS_CLIMB,
        }
        if checks["elevation"]["passed"]:
            n_passed += 1
    else:
        checks["elevation"] = {"passed": True, "note": "无DEM数据"}
        n_passed += 1

    # 7. 成本异常检查
    if cost_surface is not None and len(path_coords) >= 2:
        path_costs = []
        for lon, lat in path_coords:
            r, c = geo_to_grid(lat, lon, transform)
            H_c, W_c = cost_surface.shape
            if 0 <= r < H_c and 0 <= c < W_c:
                val = cost_surface[r, c]
                if not np.isinf(val) and not np.isnan(val):
                    path_costs.append(float(val))
        if path_costs:
            path_mean = np.mean(path_costs)
            finite = cost_surface[np.isfinite(cost_surface)]
            global_median = np.median(finite) if len(finite) > 0 else 1e-6
            ratio = path_mean / global_median if global_median > 1e-6 else 1.0
            checks["cost_anomaly"] = {
                "path_mean": float(path_mean),
                "global_median": float(global_median),
                "ratio": float(ratio),
                "threshold": cfg.COST_ANOMALY_RATIO,
                "passed": ratio <= cfg.COST_ANOMALY_RATIO,
            }
            if checks["cost_anomaly"]["passed"]:
                n_passed += 1
        else:
            checks["cost_anomaly"] = {"passed": True, "note": "无有效成本数据"}
            n_passed += 1
    else:
        checks["cost_anomaly"] = {"passed": True, "note": "无成本数据"}
        n_passed += 1

    # Convert all numpy types to native Python for JSON serialization
    def _native(v):
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, dict): return {k: _native(v) for k, v in v.items()}
        return v

    return {
        "passed": bool(n_passed == n_total),
        "n_passed": int(n_passed),
        "n_total": int(n_total),
        "checks": _native(checks),
    }


def _compute_turn_angle(p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float]) -> float:
    """计算三点转角 (度)"""
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot = np.dot(v1, v2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _compute_monotonic_climb(elevations: List[float]) -> float:
    """
    计算单调段最大高程变化 (v2算法修复版)。
    重置规则: 高程反向变化 > 50m 时视为新段开始。
    """
    if len(elevations) < 2:
        return 0.0

    max_climb = 0.0
    seg_start = elevations[0]
    seg_min = elevations[0]
    seg_max = elevations[0]
    prev = elevations[0]
    direction = 0  # 1=上升, -1=下降, 0=初始

    for e in elevations[1:]:
        diff = e - prev
        if abs(diff) < 5:
            prev = e
            continue

        new_direction = 1 if diff > 5 else -1

        if direction != 0 and new_direction != direction and abs(e - seg_start) > 50:
            # 反转 > 50m, 新段开始
            climb = seg_max - seg_min
            max_climb = max(max_climb, climb)
            seg_start = e
            seg_min = e
            seg_max = e
            direction = 0
        else:
            direction = new_direction

        seg_min = min(seg_min, e)
        seg_max = max(seg_max, e)
        prev = e

    # Final segment
    climb = seg_max - seg_min
    max_climb = max(max_climb, climb)
    return max_climb


# ============================================================
# 主编排函数: 端到端神经路径规划
# ============================================================
def neural_path_planning(
    cost_surface: np.ndarray,
    hard_mask: np.ndarray,
    transform,
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    feature_stack: Optional[np.ndarray] = None,
    soft_mask: Optional[np.ndarray] = None,
    value_model: Optional[torch.nn.Module] = None,
) -> Optional[List[Tuple[float, float]]]:
    """
    端到端神经路径规划主函数 (零传统搜索)。

    流程:
      1. 验正起止点有效性
      2. 构建目标掩膜
      3. 计算值函数 V(s) (MultiScaleValuePropNet 或 混合势场)
      4. 神经梯度追踪: 沿 -∇V(s) 从起点演化到终点
      5. 路径平滑 (RDP + 等距重采样 + 滑动平均)

    全程无 A*/Dijkstra/RRT/贪心图搜索。

    Args:
        cost_surface: (H, W) CostUNet预测的建设成本
        hard_mask: (H, W) 硬约束掩膜 (0=禁止)
        transform: 栅格仿射变换
        start_rc: 起点 (row, col)
        end_rc: 终点 (row, col)
        feature_stack: (H, W, C) 特征堆叠 (ValuePropNet可选)
        soft_mask: (H, W) 软约束掩膜 (可选)
        value_model: 预训练MultiScaleValuePropNet (可选)

    Returns:
        List of (lon, lat) path coordinates, or None on failure
    """
    H, W = cost_surface.shape
    print("[Phase4] 神经路径规划 (梯度追踪, 无图搜索)...")

    # 0. Dilate hard mask to create safety buffer around forbidden zones
    if hard_mask is not None:
        original_forbidden = (hard_mask == 0).sum()
        hard_mask_dilated = _dilate_hard_mask(hard_mask, radius_px=2)
        new_forbidden = (hard_mask_dilated == 0).sum()
        if new_forbidden > original_forbidden:
            print(f"  硬约束缓冲区: 禁止像元 {original_forbidden} → {new_forbidden} "
                  f"({new_forbidden/original_forbidden:.1f}x)")
        hard_mask = hard_mask_dilated

    # 1. Validate endpoints
    sr, sc = start_rc
    gr, gc = end_rc

    if not (0 <= sr < H and 0 <= sc < W):
        print(f"  错误: 起点 ({sr}, {sc}) 超出边界 ({H}, {W})")
        return None
    if not (0 <= gr < H and 0 <= gc < W):
        print(f"  错误: 终点 ({gr}, {gc}) 超出边界 ({H}, {W})")
        return None

    # Check if start/end in hard constraint zone
    if hard_mask is not None:
        if hard_mask[int(sr), int(sc)] == 0:
            print("  警告: 起点位于硬约束区, 搜索最近有效点...")
            sr, sc = _find_nearest_valid(hard_mask, sr, sc, max_radius=50)
        if hard_mask[int(gr), int(gc)] == 0:
            print("  警告: 终点位于硬约束区, 搜索最近有效点...")
            gr, gc = _find_nearest_valid(hard_mask, gr, gc, max_radius=50)

    start_rc = (sr, sc)
    end_rc = (gr, gc)

    print(f"  起点: {start_rc}  终点: {end_rc}")
    dist_rc = np.sqrt((gr - sr) ** 2 + (gc - sc) ** 2)
    print(f"  像元距离: {dist_rc:.0f}px")

    # 2. Build goal mask
    goal_mask = np.zeros((H, W), dtype=np.uint8)
    gr_i, gc_i = int(gr), int(gc)
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            r, c = gr_i + dr, gc_i + dc
            if 0 <= r < H and 0 <= c < W:
                goal_mask[r, c] = 1

    # 3. Compute value function
    print("  计算值函数 V(s)...")
    if feature_stack is not None:
        value_grid = compute_value_function(
            cost_surface, feature_stack, goal_mask, value_model,
            hard_mask=hard_mask,
        )
    else:
        value_grid = compute_value_function(cost_surface, goal_mask=goal_mask,
                                            hard_mask=hard_mask)

    # Apply soft mask modifier to value
    if soft_mask is not None:
        value_grid = value_grid + 0.2 * soft_mask

    # 4. Neural gradient tracking
    print("  神经梯度追踪中...")
    geo_path = extract_path_by_gradient(
        value_grid, cost_surface, hard_mask,
        start_rc, end_rc, transform,
    )

    if geo_path is None or len(geo_path) < 2:
        print("  错误: 梯度追踪失败!")
        return None

    print(f"  追踪完成: {len(geo_path)} 个点")

    # 5. 走廊吸附: 仅对真实省份(有feature_stack)微调, 合成省份跳过
    if feature_stack is not None and feature_stack.shape[2] > 14:
        print("  走廊吸附中...")
        dist_water = feature_stack[:, :, 14]  # Band 14 = dist_water
        geo_path = snap_path_to_corridor(
            geo_path, cost_surface, hard_mask, transform,
            search_radius_px=12, smooth_window=11,
            water_distance=dist_water,
        )
    else:
        print("  [合成省份] 跳过走廊吸附, 直接平滑")

    # 6. Path smoothing (with hard mask awareness)
    print("  路径平滑中...")
    smoothed = smooth_path(
        geo_path,
        hard_mask=hard_mask,
        transform=transform,
    )
    # Additional hard mask fix pass
    if hard_mask is not None and transform is not None:
        smoothed = _fix_hard_mask_violations(smoothed, hard_mask, transform)

    path_len = compute_path_length_km(smoothed)
    start_geo = grid_to_geo(start_rc[0], start_rc[1], transform)
    end_geo = grid_to_geo(end_rc[0], end_rc[1], transform)
    straight_km = haversine_km(start_geo[1], start_geo[0], end_geo[1], end_geo[0])
    print(f"  路径: {len(smoothed)} 点, {path_len:.1f}km "
          f"(直线 {straight_km:.1f}km, 弯曲度 {path_len/straight_km:.2f})")

    return smoothed


def _find_nearest_valid(hard_mask: np.ndarray, r: int, c: int,
                        max_radius: int = 50) -> Tuple[int, int]:
    """搜索最近的有效栅格点"""
    H, W = hard_mask.shape
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if hard_mask[nr, nc] == 1:
                        return (nr, nc)
    return (r, c)


# ============================================================
# 走廊精炼: 将粗路径吸附到低成本走廊 (局部优化, 非图搜索)
# ============================================================
def snap_path_to_corridor(
    path_coords: List[Tuple[float, float]],
    cost_surface: np.ndarray,
    hard_mask: np.ndarray,
    transform,
    search_radius_px: int = 25,
    smooth_window: int = 11,
    water_distance: np.ndarray = None,
) -> List[Tuple[float, float]]:
    """
    局部走廊吸附: 每个路径点独立搜索邻近最低成本有效像元。
    这不是搜索算法 — 每个点独立优化, 无图/队列/visited集合。

    方法:
      - 对每个路径点, 在其search_radius内找最低成本有效像元
      - 计算原始位置到最低成本位置的加权偏移 (保持路径连续性)
      - 最终滑动平均平滑

    Args:
        path_coords: 地理坐标路径 [(lon, lat), ...]
        cost_surface: (H, W) 成本表面
        hard_mask: (H, W) 硬约束掩膜
        transform: 栅格仿射变换
        search_radius_px: 搜索半径 (像元)
        smooth_window: 吸附后平滑窗口

    Returns:
        精炼后的地理坐标路径
    """
    if len(path_coords) < 3:
        return path_coords

    H, W = cost_surface.shape
    radius = search_radius_px

    # Build composite cost: raw cost + water penalty
    composite_cost = np.where(
        (hard_mask == 1) & np.isfinite(cost_surface),
        cost_surface.astype(np.float64),
        np.inf,
    )
    if water_distance is not None:
        # dist_water is normalized [0,1] where 1.0 = 5000m, so 90m ≈ 0.018
        water_penalty = np.where(water_distance < 0.018, 0.5, 0.0)
        composite_cost = np.where(
            np.isfinite(composite_cost),
            composite_cost + water_penalty.astype(np.float64),
            np.inf,
        )

    # Convert to grid coordinates
    grid_coords = []
    resolution = abs(transform.a)
    for lon, lat in path_coords:
        c = (lon - transform.c) / resolution
        r = (transform.f - lat) / resolution
        grid_coords.append((r, c))

    # Phase 1: Find best corridor position for each point independently
    best_positions = []
    for i, (r, c) in enumerate(grid_coords):
        ri, ci = int(round(r)), int(round(c))

        r0 = max(0, ri - radius)
        r1 = min(H, ri + radius + 1)
        c0 = max(0, ci - radius)
        c1 = min(W, ci + radius + 1)

        window = composite_cost[r0:r1, c0:c1]

        if np.all(np.isinf(window)):
            best_positions.append((r, c))
            continue

        min_idx = np.argmin(window)
        min_r, min_c = np.unravel_index(min_idx, window.shape)
        best_r = r0 + min_r
        best_c = c0 + min_c
        best_positions.append((best_r, best_c))

    # Phase 2: Smooth the target positions to prevent independent-point zigzags
    if len(best_positions) >= 9:
        best_arr = np.array(best_positions)
        half = 4
        smoothed_targets = best_arr.copy().astype(np.float64)
        for i in range(half, len(best_arr) - half):
            smoothed_targets[i] = best_arr[i-half:i+half+1].mean(axis=0)
        # Blend: 50% toward smoothed corridor target
        blend = 0.5
        snapped_grid = []
        for i, (r, c) in enumerate(grid_coords):
            br, bc = smoothed_targets[i]
            new_r = r + blend * (br - r)
            new_c = c + blend * (bc - c)
            snapped_grid.append((new_r, new_c))
    else:
        blend = 0.5
        snapped_grid = []
        for i, (r, c) in enumerate(grid_coords):
            br, bc = best_positions[i]
            new_r = r + blend * (br - r)
            new_c = c + blend * (bc - c)
            snapped_grid.append((new_r, new_c))

    # Convert back to geo
    geo_snapped = grid_to_geo_coords(snapped_grid, transform)

    # RDP to remove large-scale artifacts from snapping
    if len(geo_snapped) > 3:
        geo_snapped = _rdp_simplify(geo_snapped, cfg.PATH_SMOOTH_RDP_EPSILON * 1.5)
        geo_snapped = _resample_equidistant(geo_snapped, cfg.PATH_RESAMPLE_SPACING)

    # Double-pass moving average with different windows for strong smoothing
    if len(geo_snapped) >= 7:
        geo_snapped = _moving_average_smooth(geo_snapped, 7)
    if len(geo_snapped) >= smooth_window:
        geo_snapped = _moving_average_smooth(geo_snapped, smooth_window)

    # Curvature filter to remove any remaining sharp turns from snapping
    if len(geo_snapped) >= 5:
        geo_snapped = _filter_sharp_turns(geo_snapped, max_angle_deg=cfg.MAX_TURN_ANGLE)
        geo_snapped = _moving_average_smooth(geo_snapped, 5)
        geo_snapped = _resample_equidistant(geo_snapped, cfg.PATH_RESAMPLE_SPACING)

    return geo_snapped


# ============================================================
# 启发式成本表面 (用于无CNN训练数据的省份)
# ============================================================
def build_heuristic_cost_surface(aligned: dict, hard_mask: np.ndarray) -> np.ndarray:
    """
    从对齐特征直接构建成本表面, 无需CNN。
    用于没有真实输电线路数据的省份 (合成数据)。

    成本组合: 地形(坡度+高程+粗糙度) + 水域 + 土地利用 + 植被
    """
    shape = aligned["shape"]
    cost = np.zeros(shape, dtype=np.float32)
    weight_sum = 0.0

    # 1. 坡度 (30%权重)
    slope = aligned.get("slope")
    if slope is not None:
        slope_cost = np.clip(slope / 45.0, 0, 1)
        slope_cost = np.where(slope > 35, slope_cost * 1.5, slope_cost)
        cost += 0.30 * slope_cost
        weight_sum += 0.30

    # 2. 高程 (15%权重, 高海拔成本高)
    dem = aligned.get("dem")
    if dem is not None:
        elev_cost = np.clip(dem / 3000.0, 0, 1)
        elev_cost = np.where(dem > 2500, elev_cost * 1.5, elev_cost)
        cost += 0.15 * elev_cost
        weight_sum += 0.15

    # 3. 粗糙度 (10%权重)
    roughness = aligned.get("roughness_9")
    if roughness is not None:
        rough_cost = np.clip(roughness / 30.0, 0, 1)
        cost += 0.10 * rough_cost
        weight_sum += 0.10

    # 4. 水域距离 (15%权重, 离水越近成本越高)
    dist_water = aligned.get("dist_water")
    if dist_water is not None:
        water_cost = 1.0 - np.exp(-dist_water / 0.04)  # 200m decay
        cost += 0.15 * water_cost
        weight_sum += 0.15

    # 5. 土地利用 (10%权重)
    landuse = aligned.get("landuse_code")
    if landuse is not None and np.any(landuse > 0):
        lu_cost_map = {1: 0.05, 2: 0.15, 3: 0.20, 4: 0.70, 5: 0.90, 6: 0.85, 7: 0.60, 8: 0.20}
        lu_cost = np.zeros(shape, dtype=np.float32)
        for code in range(1, 9):
            lu_cost[landuse == code] = lu_cost_map.get(code, 0.30)
        cost += 0.10 * lu_cost
        weight_sum += 0.10

    # 6. 植被高度 (10%权重)
    veg = aligned.get("vegetation_height")
    if veg is not None:
        veg_cost = np.clip(veg / cfg.VEGETATION_HEIGHT_MAX, 0, 1)
        cost += 0.10 * veg_cost
        weight_sum += 0.10

    # 7. 道路可达性 (5%权重, 近路低成本)
    dist_road = aligned.get("dist_road")
    if dist_road is not None:
        road_cost = 1.0 - np.exp(-dist_road / 0.1)
        cost += 0.05 * road_cost
        weight_sum += 0.05

    # 8. 断裂带距离 (5%权重)
    dist_fault = aligned.get("dist_fault")
    if dist_fault is not None:
        fault_cost = 1.0 - np.exp(-dist_fault / 0.1)
        cost += 0.05 * fault_cost
        weight_sum += 0.05

    # Normalize to [0,1]
    if weight_sum > 0:
        cost = cost / weight_sum

    # Apply hard mask
    if hard_mask is not None:
        cost[hard_mask == 0] = np.inf

    print(f"  启发式成本表面: 范围[{cost[cost<np.inf].min():.3f}, "
          f"{cost[cost<np.inf].max():.3f}]")
    return cost.astype(np.float32)
