"""
v3: 模块 — 群体智能路径规划基类
版本: v3.20260710
作者: path_planning_team
描述: 提取 IPSO-SA 和 DBO 的共性 — 路径编码、适应度评估、硬约束修复、路标→路径转换
依赖: v3/config, v2_20260525/src/path_planning (工具函数)
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent.parent
_v2_dir = _v3_dir.parent / "v2_20260525"
_shared_dir = _v3_dir.parent / "shared"
for _d in reversed([_v3_dir, _v2_dir, _v2_dir / "src", _shared_dir]):
    sys.path.insert(0, str(_d))

import numpy as np
from scipy.ndimage import map_coordinates
import math
import time

import config as cfg

from path_planning import (
    geo_to_grid, grid_to_geo, grid_to_geo_coords,
    compute_path_length_km, haversine_m,
    _strict_quality_gate, smooth_path,
)


class BaseSwarmPlanner:
    """
    群体智能路径规划基类

    路径编码: M 个中间路标点 (lat, lon), 首尾固定为起止点
    完整路径 = 起点 + M路标点 + 终点 → 等距采样 → 双线性插值查成本
    适应度 = 平均成本 + 硬约束惩罚 + 回退惩罚 + 长度惩罚

    子类需实现:
        _update_positions() — 算法特定的位置更新逻辑
    """

    def __init__(
        self,
        cost_surface: np.ndarray,
        hard_mask: np.ndarray,
        transform,
        start_latlon: tuple,
        end_latlon: tuple,
        num_individuals: int | None = None,
        num_waypoints: int | None = None,
        max_iterations: int | None = None,
        corridor_width_km: float | None = None,
        aligned: dict | None = None,
        a_star_path: list | None = None,
        random_seed: int = 42,
        algorithm_name: str = "Base",
    ):
        self.cost_surface = cost_surface
        self.hard_mask = hard_mask
        self.transform = transform
        self.start_ll = start_latlon    # (lat, lon)
        self.end_ll = end_latlon        # (lat, lon)
        self.aligned = aligned or {}
        self.a_star_path = a_star_path
        self.algorithm_name = algorithm_name

        self.H, self.W = cost_surface.shape
        self.rng = np.random.RandomState(random_seed)

        # 种群规模
        self.num_individuals = num_individuals or 30
        self.max_iterations = max_iterations or 200

        # 路标点数量 — 按直线距离自动计算
        self.straight_km = haversine_m(
            start_latlon[1], start_latlon[0],
            end_latlon[1], end_latlon[0],
        ) / 1000.0

        if num_waypoints is None:
            self.num_waypoints = max(
                3,
                int(self.straight_km * cfg.SWARM_WAYPOINTS_PER_KM),
            )
        else:
            self.num_waypoints = num_waypoints

        # 搜索走廊 (经纬度)
        corridor_km = corridor_width_km or cfg.SWARM_CORRIDOR_WIDTH_KM
        self.corridor_deg = corridor_km / 111.0
        slat, slon = start_latlon
        elat, elon = end_latlon
        self.lat_min = max(min(slat, elat) - self.corridor_deg, cfg.TAIWAN_BBOX[1])
        self.lat_max = min(max(slat, elat) + self.corridor_deg, cfg.TAIWAN_BBOX[3])
        self.lon_min = max(min(slon, elon) - self.corridor_deg, cfg.TAIWAN_BBOX[0])
        self.lon_max = min(max(slon, elon) + self.corridor_deg, cfg.TAIWAN_BBOX[2])

        # 惩罚系数
        self.hard_penalty = cfg.SWARM_HARD_PENALTY
        self.backward_penalty = cfg.SWARM_BACKWARD_PENALTY

        # 种群
        self.population: np.ndarray | None = None   # (N, M, 2) — (lat, lon)
        self.fitness: np.ndarray | None = None        # (N,)
        self.convergence_curve: list = []

        # 全局最优
        self.best_position: np.ndarray | None = None  # (M, 2)
        self.best_fitness: float = np.inf

        # 早停
        self.early_stop_stall = cfg.SWARM_EARLY_STOP_STALL

        print(
            f"[{self.algorithm_name}] 初始化: "
            f"个体={self.num_individuals}, 路标点={self.num_waypoints}, "
            f"走廊={corridor_km}km, 直线={self.straight_km:.1f}km"
        )

    # ============================================================
    # 种群初始化
    # ============================================================
    def _initialize_population(self, use_astar_hotstart: bool = True) -> np.ndarray:
        """初始化种群: 直线 + 可选A*热启动 + 走廊随机

        Args:
            use_astar_hotstart: 是否使用A*路径作为个体1的热启动, False则全部随机
        """
        pop = np.zeros(
            (self.num_individuals, self.num_waypoints, 2),
            dtype=np.float64,
        )
        slat, slon = self.start_ll
        elat, elon = self.end_ll

        base_lats = np.linspace(slat, elat, self.num_waypoints + 2)[1:-1]
        base_lons = np.linspace(slon, elon, self.num_waypoints + 2)[1:-1]

        for i in range(self.num_individuals):
            if i == 0:
                # 个体 0: 直线路径
                pop[i, :, 0] = base_lats
                pop[i, :, 1] = base_lons
            elif use_astar_hotstart and i == 1 and self.a_star_path is not None and len(self.a_star_path) > 2:
                # 个体 1: A* 路径降采样 (热启动)
                a_coords = np.array(self.a_star_path)
                n_a = len(a_coords)
                indices = np.linspace(0, n_a - 1, self.num_waypoints + 2, dtype=int)[1:-1]
                pop[i, :, 1] = a_coords[indices, 0]
                pop[i, :, 0] = a_coords[indices, 1]
            else:
                # 走廊内随机扰动 (扩大范围以增加多样性)
                noise_lat = self.rng.uniform(-self.corridor_deg * 0.8, self.corridor_deg * 0.8, self.num_waypoints)
                noise_lon = self.rng.uniform(-self.corridor_deg * 0.8, self.corridor_deg * 0.8, self.num_waypoints)
                pop[i, :, 0] = np.clip(base_lats + noise_lat, self.lat_min, self.lat_max)
                pop[i, :, 1] = np.clip(base_lons + noise_lon, self.lon_min, self.lon_max)

        return pop

    # ============================================================
    # 适应度评估
    # ============================================================
    def _evaluate_fitness(self, waypoints: np.ndarray) -> float:
        """
        评估路径适应度 (越低越好)

        适应度 = 平均成本 + 硬约束惩罚 + 回退惩罚 + 长度惩罚

        Args:
            waypoints: (M, 2) — [(lat, lon), ...] 中间路标点
        Returns:
            float: 适应度值
        """
        # 拼接完整路径
        full_lats = np.concatenate([
            [self.start_ll[0]], waypoints[:, 0], [self.end_ll[0]],
        ])
        full_lons = np.concatenate([
            [self.start_ll[1]], waypoints[:, 1], [self.end_ll[1]],
        ])

        # 沿路径等距采样
        n_samples = max(50, int(self.straight_km * 1000 / cfg.BASE_RESOLUTION))
        n_samples = min(n_samples, 2000)

        # 每段距离 (近似, 米)
        cos_lat = np.cos(np.radians((full_lats[:-1] + full_lats[1:]) / 2))
        seg_dists = np.sqrt(
            ((full_lons[1:] - full_lons[:-1]) * cfg.METERS_PER_DEG * cos_lat) ** 2 +
            ((full_lats[1:] - full_lats[:-1]) * cfg.METERS_PER_DEG) ** 2
        )
        total_dist = np.sum(seg_dists)
        if total_dist < 1:
            return 1e10

        cum_dist = np.concatenate([[0], np.cumsum(seg_dists)])
        sample_dists = np.linspace(0, total_dist, n_samples)
        sample_lats = np.interp(sample_dists, cum_dist, full_lats)
        sample_lons = np.interp(sample_dists, cum_dist, full_lons)
        sample_lats = np.clip(sample_lats, cfg.TAIWAN_BBOX[1], cfg.TAIWAN_BBOX[3])
        sample_lons = np.clip(sample_lons, cfg.TAIWAN_BBOX[0], cfg.TAIWAN_BBOX[2])

        # 转换为栅格坐标
        sample_rows = (self.transform.f - sample_lats) / self.transform.e
        sample_cols = (sample_lons - self.transform.c) / self.transform.a

        # 双线性插值查询成本表面
        rows_idx = np.clip(sample_rows, 0, self.H - 1.001)
        cols_idx = np.clip(sample_cols, 0, self.W - 1.001)
        costs = map_coordinates(
            self.cost_surface, [rows_idx, cols_idx],
            order=1, mode="constant", cval=np.inf,
        )

        # 成本项
        finite_mask = np.isfinite(costs)
        if finite_mask.sum() < n_samples * 0.5:
            return 1e10

        mean_cost = np.mean(costs[finite_mask])
        inf_ratio = 1.0 - finite_mask.sum() / n_samples

        # 硬约束惩罚
        hard_penalty = inf_ratio * self.hard_penalty * 100

        # 回退惩罚 (路标点不能倒着走)
        backward = 0.0
        for k in range(1, self.num_waypoints):
            prev_p = self._progress_along_line(
                waypoints[k - 1, 0], waypoints[k - 1, 1],
            )
            curr_p = self._progress_along_line(
                waypoints[k, 0], waypoints[k, 1],
            )
            if curr_p < prev_p - 0.05:
                backward += (prev_p - curr_p)
        backward_penalty = backward * self.backward_penalty

        # 长度惩罚 (sinuosity)
        path_len_km = total_dist / 1000.0
        sinuosity = path_len_km / self.straight_km if self.straight_km > 0 else 1.0
        length_penalty = max(0, sinuosity - cfg.MAX_SINUOSITY) * 5.0

        return float(mean_cost + hard_penalty + backward_penalty + length_penalty)

    def _progress_along_line(self, lat: float, lon: float) -> float:
        """计算点在起点→终点方向上的投影进度 [0, 1]"""
        slat, slon = self.start_ll
        elat, elon = self.end_ll
        dlat, dlon = elat - slat, elon - slon
        line_len_sq = dlat ** 2 + dlon ** 2
        if line_len_sq < 1e-15:
            return 0.5
        t = ((lat - slat) * dlat + (lon - slon) * dlon) / line_len_sq
        return float(np.clip(t, 0, 1))

    # ============================================================
    # 硬约束修复
    # ============================================================
    def _repair_to_feasible(self, waypoints: np.ndarray) -> np.ndarray:
        """将落入硬约束区的路标点移动到最近可行点 (搜索半径30像素)"""
        for j in range(self.num_waypoints):
            lat, lon = waypoints[j, 0], waypoints[j, 1]
            r, c = geo_to_grid(lat, lon, self.transform)
            if 0 <= r < self.H and 0 <= c < self.W:
                if np.isinf(self.cost_surface[r, c]):
                    found = False
                    for radius in range(1, 31):
                        for dr in range(-radius, radius + 1):
                            for dc in range(-radius, radius + 1):
                                nr, nc = r + dr, c + dc
                                if (
                                    0 <= nr < self.H
                                    and 0 <= nc < self.W
                                    and not np.isinf(self.cost_surface[nr, nc])
                                ):
                                    new_lat, new_lon = grid_to_geo(
                                        nr, nc, self.transform,
                                    )
                                    waypoints[j, 0] = new_lat
                                    waypoints[j, 1] = new_lon
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
        return waypoints

    # ============================================================
    # 路标点 → 密集路径坐标
    # ============================================================
    def _waypoints_to_path(
        self, waypoints: np.ndarray, sample_spacing_m: int = 30,
    ) -> list:
        """将 M 个路标点插值为密集路径坐标 [(lon, lat), ...]"""
        full_lats = np.concatenate([
            [self.start_ll[0]], waypoints[:, 0], [self.end_ll[0]],
        ])
        full_lons = np.concatenate([
            [self.start_ll[1]], waypoints[:, 1], [self.end_ll[1]],
        ])

        n_pts = len(full_lats)
        coords = []
        for k in range(n_pts - 1):
            d = haversine_m(full_lons[k], full_lats[k],
                            full_lons[k + 1], full_lats[k + 1])
            n_seg = max(2, int(d / sample_spacing_m))
            for j in range(n_seg):
                t = j / n_seg
                lon = full_lons[k] + t * (full_lons[k + 1] - full_lons[k])
                lat = full_lats[k] + t * (full_lats[k + 1] - full_lats[k])
                coords.append((float(lon), float(lat)))
        coords.append((float(full_lons[-1]), float(full_lats[-1])))
        return coords

    # ============================================================
    # 工厂函数
    # ============================================================
    @staticmethod
    def create(algo_type: str, **kwargs):
        """工厂方法: 根据算法类型创建规划器实例

        Args:
            algo_type: "A*" | "IPSO-SA" | "DBO"
            **kwargs: 传递给具体规划器的参数
        """
        from ipso_sa_planner import IPSOSAPlanner
        from dbo_planner import DBOPlanner

        planners = {
            "ipso_sa": IPSOSAPlanner,
            "IPSO-SA": IPSOSAPlanner,
            "dbo": DBOPlanner,
            "DBO": DBOPlanner,
        }
        if algo_type not in planners:
            raise ValueError(f"未知算法类型: {algo_type}。支持: {list(planners.keys())}")
        return planners[algo_type](**kwargs)

    # ============================================================
    # 位置更新 (子类实现)
    # ============================================================
    def _update_positions(self, iteration: int, max_iterations: int):
        """算法特定的位置更新逻辑 — 子类必须重写"""
        raise NotImplementedError("子类必须实现 _update_positions()")

    # ============================================================
    # 主优化循环
    # ============================================================
    def optimize(self, verbose: bool = True, use_astar_hotstart: bool = True) -> tuple:
        """
        运行优化主循环

        Args:
            verbose: 是否输出日志
            use_astar_hotstart: 是否使用A*热启动 (DBO建议False以增加差异化)
        """
        t0 = time.time()

        if verbose:
            print(f"[{self.algorithm_name}] 初始化种群...")
        self.population = self._initialize_population(use_astar_hotstart=use_astar_hotstart)
        self.fitness = np.full(self.num_individuals, np.inf)

        for i in range(self.num_individuals):
            self.population[i] = self._repair_to_feasible(self.population[i])
            self.fitness[i] = self._evaluate_fitness(self.population[i])

        # 找初始全局最优
        best_idx = np.argmin(self.fitness)
        self.best_position = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.convergence_curve = [self.best_fitness]

        if verbose:
            print(
                f"[{self.algorithm_name}] 初始最优={self.best_fitness:.4f}"
            )

        stall_count = 0

        for it in range(self.max_iterations):
            self._update_positions(it, self.max_iterations)

            # 评估新一代
            for i in range(self.num_individuals):
                self.population[i] = self._repair_to_feasible(
                    self.population[i],
                )
                self.fitness[i] = self._evaluate_fitness(self.population[i])

            # 更新全局最优
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_position = self.population[best_idx].copy()
                stall_count = 0
            else:
                stall_count += 1

            self.convergence_curve.append(self.best_fitness)

            if verbose and (it + 1) % 20 == 0:
                print(
                    f"[{self.algorithm_name}] 迭代 {it + 1}/{self.max_iterations}, "
                    f"最优={self.best_fitness:.4f}, 停滞={stall_count}"
                )

            if stall_count >= self.early_stop_stall:
                if verbose:
                    print(
                        f"[{self.algorithm_name}] 早停于迭代 {it + 1} "
                        f"({self.early_stop_stall}代无改进)"
                    )
                break

        elapsed = time.time() - t0
        if verbose:
            print(
                f"[{self.algorithm_name}] 完成 ({elapsed:.1f}s), "
                f"最优={self.best_fitness:.4f}, "
                f"迭代={len(self.convergence_curve)}"
            )

        best_path = self._waypoints_to_path(self.best_position)
        return best_path, self.best_fitness, self.convergence_curve

    # ============================================================
    # 便捷接口: 完整规划 + 平滑 + 质量门控
    # ============================================================
    def plan_path(
        self,
        apply_smoothing: bool = True,
        verbose: bool = True,
    ) -> tuple:
        """
        完整路径规划: 优化 → 路径生成 → 平滑 → 质量门控

        Returns:
            smoothed_path:   平滑后路径 [(lon, lat), ...]
            quality_report:  7项质量门控结果 dict
            info:            附加信息 dict
        """
        use_hs = getattr(self, '_USE_ASTAR_HOTSTART', True)
        raw_path, best_fitness, convergence = self.optimize(verbose=verbose, use_astar_hotstart=use_hs)

        # 路径平滑
        if apply_smoothing and len(raw_path) > 2:
            smoothed = self._smooth_raw_path(raw_path)
        else:
            smoothed = raw_path

        # 质量门控
        straight_km = haversine_m(
            self.start_ll[1], self.start_ll[0],
            self.end_ll[1], self.end_ll[0],
        ) / 1000.0

        quality_report = _strict_quality_gate(
            smoothed,
            self.aligned,
            self.hard_mask,
            self.cost_surface,
            self.transform,
            straight_km,
        )

        info = {
            "raw_path": raw_path,
            "best_fitness": best_fitness,
            "convergence_curve": convergence,
            "num_waypoints": self.num_waypoints,
            "algorithm": self.algorithm_name,
        }

        if verbose:
            passed = quality_report["passed"]
            n_pass = quality_report["n_passed"]
            n_total = quality_report["n_total"]
            print(
                f"[{self.algorithm_name}] 质量门控: "
                f"{'通过' if passed else '不通过'} ({n_pass}/{n_total})"
            )

        return smoothed, quality_report, info

    def _smooth_raw_path(self, raw_path: list) -> list:
        """RDP简化 + 线性插值 + 滑动平均"""
        if len(raw_path) < 3:
            return raw_path

        try:
            from rdp import rdp as _rdp
        except ImportError:
            from path_planning import rdp as _rdp

        rdp_epsilon_deg = cfg.PATH_SMOOTH_RDP_EPSILON / 111000.0
        simplified = _rdp(raw_path, epsilon=rdp_epsilon_deg)

        if len(simplified) < 3:
            return raw_path

        simplified_arr = np.array(simplified)
        xs, ys = simplified_arr[:, 0], simplified_arr[:, 1]
        seg_dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        cum_dist = np.concatenate([[0], np.cumsum(seg_dists)])
        total_dist = cum_dist[-1]

        spacing_deg = cfg.PATH_RESAMPLE_SPACING / 111000.0
        n_samples = max(int(total_dist / spacing_deg), len(simplified))
        sample_dists = np.linspace(0, total_dist, n_samples)
        x_interp = np.interp(sample_dists, cum_dist, xs)
        y_interp = np.interp(sample_dists, cum_dist, ys)

        if n_samples >= 5:
            window = 5
            kernel = np.ones(window) / window
            x_smooth = np.convolve(x_interp, kernel, mode="same")
            y_smooth = np.convolve(y_interp, kernel, mode="same")
            edge = window // 2
            x_smooth[:edge] = x_interp[:edge]
            x_smooth[-edge:] = x_interp[-edge:]
            y_smooth[:edge] = y_interp[:edge]
            y_smooth[-edge:] = y_interp[-edge:]
            return [
                (float(x_smooth[i]), float(y_smooth[i]))
                for i in range(n_samples)
            ]

        return [
            (float(x_interp[i]), float(y_interp[i]))
            for i in range(n_samples)
        ]
