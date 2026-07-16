"""
v3: 模块 — DBO (Dung Beetle Optimizer) 蜣螂优化路径规划器
版本: v3.20260710
作者: path_planning_team
描述: 基于 2022年 Xue & Shen 原论文的五行为模型
       滚球(探索) + 跳舞(避障) + 育雏(局部开发) + 觅食(精细搜索) + 偷窃(多样性)
       每条蜣螂代表一条路径 (M个中间路标点)
参考: Xue & Shen, "Dung Beetle Optimizer", J. Supercomputing, 2023
依赖: v3/config, v3/src/base_planner
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent.parent
_v3_src = _v3_dir / "src"
for _d in reversed([_v3_src, _v3_dir]):
    sys.path.insert(0, str(_d))

import numpy as np
from scipy.ndimage import map_coordinates
import config as cfg
from base_planner import BaseSwarmPlanner


class DBOPlanner(BaseSwarmPlanner):
    """
    DBO 蜣螂优化路径规划器

    种群分四组:
        roll:  滚球蜣螂 (利用天体线索直线移动, 遇障跳舞转向)
        brood: 育雏蜣螂 (在最优个体附近产卵, 局部开发)
        forage: 觅食蜣螂 (在最优区域周围寻找食物)
        steal: 偷窃蜣螂 (从其他个体窃取位置)

    参数:
        k: 滚球偏转系数 (默认 0.1, 控制转向幅度)
        b: 育雏区域半径系数 (默认 0.3)
        S: 觅食区域半径系数 (默认 0.5)

    DBO 不使用 A* 热启动——全种群随机初始化以区别于 IPSO-SA,
    在更宽的 30km 走廊内独立探索。
    """

    # DBO 不走 A* 热启动，全随机初始化 + 独立惩罚权重以区别于 IPSO-SA
    _USE_ASTAR_HOTSTART = False

    def __init__(
        self,
        cost_surface,
        hard_mask,
        transform,
        start_latlon,
        end_latlon,
        num_beetles=None,
        num_waypoints=None,
        max_iterations=None,
        corridor_width_km=None,
        aligned=None,
        a_star_path=None,
        random_seed=42,
    ):
        super().__init__(
            cost_surface=cost_surface,
            hard_mask=hard_mask,
            transform=transform,
            start_latlon=start_latlon,
            end_latlon=end_latlon,
            num_individuals=num_beetles or cfg.DBO_NUM_BEETLES,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations or cfg.DBO_MAX_ITERATIONS,
            corridor_width_km=corridor_width_km,
            aligned=aligned,
            a_star_path=a_star_path,
            random_seed=random_seed,
            algorithm_name="DBO",
        )

        # DBO 独立惩罚权重 — 更低hard_penalty鼓励探索, 更高backward_penalty保持方向
        self.hard_penalty = cfg.DBO_HARD_PENALTY
        self.backward_penalty = cfg.DBO_BACKWARD_PENALTY

        # 行为比例
        p_roll = cfg.DBO_P_ROLL
        p_brood = cfg.DBO_P_BROOD
        p_forage = cfg.DBO_P_FORAGE
        p_steal = cfg.DBO_P_STEAL
        total_p = p_roll + p_brood + p_forage + p_steal
        self.p_roll = p_roll / total_p
        self.p_brood = p_brood / total_p
        self.p_forage = p_forage / total_p

        # 各行为个体数
        self.n_roll = max(1, int(self.num_individuals * self.p_roll))
        self.n_brood = max(1, int(self.num_individuals * self.p_brood))
        self.n_forage = max(1, int(self.num_individuals * self.p_forage))
        self.n_steal = self.num_individuals - self.n_roll - self.n_brood - self.n_forage

        # 行为参数
        self.k = cfg.DBO_K   # 滚球偏转系数
        self.b = cfg.DBO_B   # 育雏区域半径
        self.S = cfg.DBO_S   # 觅食区域半径

        # 历史最优 (用于育雏和觅食的边界)
        self.gbest_history = None

        print(
            f"[{self.algorithm_name}] 行为分配: "
            f"滚球={self.n_roll}, 育雏={self.n_brood}, "
            f"觅食={self.n_forage}, 偷窃={self.n_steal}"
        )

    # ============================================================
    # 行为 1: 滚球 (探索)
    # ============================================================
    def _rolling_behavior(self, i: int):
        """滚球蜣螂: 沿当前方向移动, 遇障碍偏转"""
        # 根据光线(全局最优方向)更新位置
        light_intensity = self.rng.random()

        if light_intensity < 0.8:
            # 有光线 → 向全局最优点偏转
            delta = self.rng.uniform(0, 1) * self.k * (
                self.best_position - self.population[i]
            )
            self.population[i] = self.population[i] + delta
        else:
            # 无光线 → 随机偏转 (跳舞)
            theta = self.rng.uniform(0, np.pi) if self.rng.random() < 0.5 else self.rng.uniform(0, np.pi / 2)
            self.population[i] = self.population[i] + np.tan(theta) * (
                self.population[i] - self.population[i - 1]
                if i > 0
                else np.zeros_like(self.population[i])
            )

        # 裁剪
        self.population[i, :, 0] = np.clip(self.population[i, :, 0], self.lat_min, self.lat_max)
        self.population[i, :, 1] = np.clip(self.population[i, :, 1], self.lon_min, self.lon_max)

    # ============================================================
    # 行为 2: 育雏 (局部开发)
    # ============================================================
    def _brood_behavior(self, i: int):
        """育雏蜣螂: 在最优个体附近的育雏球内产卵"""
        # 育雏区域边界
        R = 1.0 - (self._current_iteration / self.max_iterations)
        Lb_local = self.best_position * (1 - R * self.b)
        Ub_local = self.best_position * (1 + R * self.b)

        # 在边界内随机产卵
        b1 = self.rng.random((self.num_waypoints, 2))
        b2 = self.rng.random((self.num_waypoints, 2))
        self.population[i] = self.best_position + b1 * (self.population[i] - Lb_local) + b2 * (self.population[i] - Ub_local)

        self.population[i, :, 0] = np.clip(self.population[i, :, 0], self.lat_min, self.lat_max)
        self.population[i, :, 1] = np.clip(self.population[i, :, 1], self.lon_min, self.lon_max)

    # ============================================================
    # 行为 3: 觅食 (精细搜索)
    # ============================================================
    def _foraging_behavior(self, i: int):
        """觅食蜣螂: 在最优觅食区域内搜索"""
        # 觅食区域边界
        R = 1.0 - (self._current_iteration / self.max_iterations)
        Lb_forage = self.best_position * (1 - R * self.S)
        Ub_forage = self.best_position * (1 + R * self.S)

        # 在边界内搜索
        C1 = self.rng.random((self.num_waypoints, 2))
        self.population[i] = self.population[i] + C1 * (self.population[i] - Lb_forage) + C1 * (self.population[i] - Ub_forage)

        self.population[i, :, 0] = np.clip(self.population[i, :, 0], self.lat_min, self.lat_max)
        self.population[i, :, 1] = np.clip(self.population[i, :, 1], self.lon_min, self.lon_max)

    # ============================================================
    # 行为 4: 偷窃 (多样性)
    # ============================================================
    def _stealing_behavior(self, i: int):
        """偷窃蜣螂: 从更好个体附近窃取位置"""
        # 随机选一个比当前好的个体
        better_indices = [j for j in range(self.num_individuals)
                          if self.fitness[j] < self.fitness[i] and j != i]
        if not better_indices:
            # 没有更好的 → 随机扰动
            noise = self.rng.uniform(-0.5, 0.5, (self.num_waypoints, 2)) * self.corridor_deg * 0.1
            self.population[i] = self.population[i] + noise
        else:
            target_idx = self.rng.choice(better_indices)
            # 向更优个体移动
            g = self.rng.random((self.num_waypoints, 2)) * 0.5
            self.population[i] = self.population[i] + g * (
                self.population[target_idx] - self.population[i]
            )

        self.population[i, :, 0] = np.clip(self.population[i, :, 0], self.lat_min, self.lat_max)
        self.population[i, :, 1] = np.clip(self.population[i, :, 1], self.lon_min, self.lon_max)

    # ============================================================
    # 位置更新 — 按行为分组
    # ============================================================
    def _update_positions(self, iteration: int, max_iterations: int):
        """按五行为分组更新蜣螂位置"""
        self._current_iteration = iteration

        # 滚球组 (0 ~ n_roll-1)
        for i in range(self.n_roll):
            self._rolling_behavior(i)

        # 育雏组 (n_roll ~ n_roll+n_brood-1)
        for i in range(self.n_roll, self.n_roll + self.n_brood):
            self._brood_behavior(i)

        # 觅食组
        for i in range(self.n_roll + self.n_brood, self.n_roll + self.n_brood + self.n_forage):
            self._foraging_behavior(i)

        # 偷窃组 (剩余)
        for i in range(self.n_roll + self.n_brood + self.n_forage, self.num_individuals):
            self._stealing_behavior(i)


# ============================================================
# 便捷函数: 完整 DBO 路径规划
# ============================================================
def dbo_plan_path(
    cost_surface,
    hard_mask,
    transform,
    start_latlon,
    end_latlon,
    aligned=None,
    a_star_path=None,
    num_beetles=None,
    num_waypoints=None,
    max_iterations=None,
    corridor_width_km=None,
    apply_smoothing=True,
    random_seed=42,
    verbose=True,
):
    """
    使用 DBO 规划路径, 返回平滑后的路径 + 质量门控结果
    """
    planner = DBOPlanner(
        cost_surface=cost_surface,
        hard_mask=hard_mask,
        transform=transform,
        start_latlon=start_latlon,
        end_latlon=end_latlon,
        num_beetles=num_beetles,
        num_waypoints=num_waypoints,
        max_iterations=max_iterations,
        corridor_width_km=corridor_width_km,
        aligned=aligned,
        a_star_path=a_star_path,
        random_seed=random_seed,
    )

    return planner.plan_path(apply_smoothing=apply_smoothing, verbose=verbose)
