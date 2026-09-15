"""
v3: 模块 — IPSO-SA (Improved PSO + Simulated Annealing) 路径规划器
版本: v3.20260710
作者: path_planning_team
描述: 改进粒子群优化 + 模拟退火, 作为申报书基准对照组
       每个粒子代表一条路径 (M个中间路标点), PSO全局搜索 + SA跳出局部最优
参考: 项目申报书 — 基于IPSO-SA与无人机协同的复杂山区高空输电线路路径规划研究
依赖: v3/config, v3/src/base_planner
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent.parent
_v3_src = _v3_dir / "src"
for _d in reversed([_v3_src, _v3_dir]):
    sys.path.insert(0, str(_d))

import numpy as np
import config as cfg
from base_planner import BaseSwarmPlanner


class IPSOSAPlanner(BaseSwarmPlanner):
    """
    IPSO-SA 路径规划器

    PSO: 惯性权重线性递减 (w: 0.9 → 0.4)
         c1=c2=2.0, 速度限幅在走廊宽度的 20%

    SA:  温度指数衰减 (T = alpha * T, alpha=0.95)
         以概率 exp(-Δf/T) 接受劣解, 内循环5次
    """

    def __init__(
        self,
        cost_surface,
        hard_mask,
        transform,
        start_latlon,
        end_latlon,
        num_particles=None,
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
            num_individuals=num_particles or cfg.IPSO_NUM_PARTICLES,
            num_waypoints=num_waypoints,
            max_iterations=max_iterations or cfg.IPSO_MAX_ITERATIONS,
            corridor_width_km=corridor_width_km,
            aligned=aligned,
            a_star_path=a_star_path,
            random_seed=random_seed,
            algorithm_name="IPSO-SA",
        )

        # PSO 参数
        self.w_start = cfg.IPSO_W_START
        self.w_end = cfg.IPSO_W_END
        self.c1 = cfg.IPSO_C1
        self.c2 = cfg.IPSO_C2

        # SA 参数
        self.sa_T0 = cfg.IPSO_SA_T0
        self.sa_alpha = cfg.IPSO_SA_ALPHA
        self.sa_K = cfg.IPSO_SA_K

        # 速度 (纬度, 经度)
        self.velocity = np.zeros_like(self.population) if self.population is not None else None

        # 个体最优
        self.pbest_positions: np.ndarray | None = None    # (N, M, 2)
        self.pbest_fitness: np.ndarray | None = None       # (N,)

    # ============================================================
    # 位置更新 — PSO + SA
    # ============================================================
    def _update_positions(self, iteration: int, max_iterations: int):
        """PSO 速度+位置更新, SA 接受/拒绝判断"""
        # 惯性权重线性递减
        w = self.w_start - (self.w_start - self.w_end) * (iteration / max_iterations)

        # SA 温度
        T = self.sa_T0 * (self.sa_alpha ** iteration)

        # 速度限幅 (走廊宽度的 20%)
        v_max_lat = self.corridor_deg * 0.2
        v_max_lon = self.corridor_deg * 0.2

        for i in range(self.num_individuals):
            # 保存旧位置(用于SA回退)
            old_position = self.population[i].copy()
            old_fitness = self.fitness[i]

            # PSO 速度更新
            r1 = self.rng.random((self.num_waypoints, 2))
            r2 = self.rng.random((self.num_waypoints, 2))

            cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.population[i])
            social = self.c2 * r2 * (self.best_position - self.population[i])

            self.velocity[i] = w * self.velocity[i] + cognitive + social
            self.velocity[i, :, 0] = np.clip(self.velocity[i, :, 0], -v_max_lat, v_max_lat)
            self.velocity[i, :, 1] = np.clip(self.velocity[i, :, 1], -v_max_lon, v_max_lon)

            # 位置更新
            self.population[i] = self.population[i] + self.velocity[i]
            self.population[i, :, 0] = np.clip(
                self.population[i, :, 0], self.lat_min, self.lat_max,
            )
            self.population[i, :, 1] = np.clip(
                self.population[i, :, 1], self.lon_min, self.lon_max,
            )

            # 修复 + 评估新位置
            self.population[i] = self._repair_to_feasible(self.population[i])
            new_fitness = self._evaluate_fitness(self.population[i])
            delta = new_fitness - old_fitness

            # SA接受准则
            if delta < 0:
                # 更好 → 接受
                self.fitness[i] = new_fitness
            else:
                # 更差 → 以SA概率接受
                prob = np.exp(-delta / (T + 1e-10))
                if self.rng.random() < prob:
                    self.fitness[i] = new_fitness
                else:
                    # 拒绝 → 回退
                    self.population[i] = old_position
                    self.fitness[i] = old_fitness

            # 更新个体最优
            if self.fitness[i] < self.pbest_fitness[i]:
                self.pbest_fitness[i] = self.fitness[i]
                self.pbest_positions[i] = self.population[i].copy()

    # ============================================================
    # 重写: 初始化 (额外初始化速度和个体最优)
    # ============================================================
    def optimize(self, verbose: bool = True, use_astar_hotstart: bool = True):
        """运行 IPSO-SA 优化 (初始化速度+个体最优后调用父类主循环)"""
        # 先调父类初始化种群
        import time
        t0 = time.time()

        if verbose:
            print(f"[{self.algorithm_name}] 初始化种群...")
        self.population = self._initialize_population(use_astar_hotstart=use_astar_hotstart)

        # 初始化速度为零
        self.velocity = np.zeros_like(self.population)

        # 初始适应度
        self.fitness = np.full(self.num_individuals, np.inf)
        for i in range(self.num_individuals):
            self.population[i] = self._repair_to_feasible(self.population[i])
            self.fitness[i] = self._evaluate_fitness(self.population[i])

        # 个体最优 = 初始位置
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()

        # 全局最优
        best_idx = np.argmin(self.fitness)
        self.best_position = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.convergence_curve = [self.best_fitness]

        if verbose:
            print(f"[{self.algorithm_name}] 初始最优={self.best_fitness:.4f}")

        stall_count = 0

        for it in range(self.max_iterations):
            self._update_positions(it, self.max_iterations)

            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_position = self.population[best_idx].copy()
                stall_count = 0
            else:
                stall_count += 1

            self.convergence_curve.append(self.best_fitness)

            if verbose and (it + 1) % 20 == 0:
                w = self.w_start - (self.w_start - self.w_end) * (it / self.max_iterations)
                print(
                    f"[{self.algorithm_name}] 迭代 {it + 1}/{self.max_iterations}, "
                    f"w={w:.3f}, 最优={self.best_fitness:.4f}, 停滞={stall_count}"
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
                f"最优={self.best_fitness:.4f}"
            )

        best_path = self._waypoints_to_path(self.best_position)
        return best_path, self.best_fitness, self.convergence_curve


# ============================================================
# 便捷函数: 完整 IPSO-SA 路径规划
# ============================================================
def ipso_sa_plan_path(
    cost_surface,
    hard_mask,
    transform,
    start_latlon,
    end_latlon,
    aligned=None,
    a_star_path=None,
    num_particles=None,
    num_waypoints=None,
    max_iterations=None,
    corridor_width_km=None,
    apply_smoothing=True,
    random_seed=42,
    verbose=True,
):
    """
    使用 IPSO-SA 规划路径, 返回平滑后的路径 + 质量门控结果
    """
    planner = IPSOSAPlanner(
        cost_surface=cost_surface,
        hard_mask=hard_mask,
        transform=transform,
        start_latlon=start_latlon,
        end_latlon=end_latlon,
        num_particles=num_particles,
        num_waypoints=num_waypoints,
        max_iterations=max_iterations,
        corridor_width_km=corridor_width_km,
        aligned=aligned,
        a_star_path=a_star_path,
        random_seed=random_seed,
    )

    return planner.plan_path(apply_smoothing=apply_smoothing, verbose=verbose)
