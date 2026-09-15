"""
v4: CNN增强群体智能路径规划对比 — 全局配置
版本: v4.20260718
作者: path_planning_team
描述: 继承 v2 全部参数 + IPSO-SA/DBO/CNN 超参
依赖: v2_20260525/config.py
"""
import sys
from pathlib import Path

# 导入 v2 config 全部符号
import importlib.util
_v4_dir = Path(__file__).resolve().parent
_v2_cfg_path = _v4_dir.parent / "v2_20260525" / "config.py"
_spec = importlib.util.spec_from_file_location("v2_config", _v2_cfg_path)
_v2_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2_cfg)
for _attr in dir(_v2_cfg):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_v2_cfg, _attr)

# ============================================================
# v4 路径配置
# ============================================================
V4_DIR = _v4_dir
OUTPUT_DIR = V4_DIR / "output"

# ============================================================
# 通用群体智能参数
# ============================================================
# 路标点密度 (每公里直线距离的路标点数)
SWARM_WAYPOINTS_PER_KM = 4.0

# 搜索走廊宽度 (km, 限制种群在起终点连线附近)
SWARM_CORRIDOR_WIDTH_KM = 30.0

# 硬约束惩罚系数 (适应度 = 平均成本 + 硬约束违规比例 * HARD_PENALTY)
SWARM_HARD_PENALTY = 100.0

# 回退惩罚系数 (路标点顺序倒退的惩罚)
SWARM_BACKWARD_PENALTY = 10.0

# DBO 独立惩罚参数 — 更低硬约束惩罚鼓励探索禁区边缘, 更高回退惩罚保持方向
DBO_HARD_PENALTY = 50.0
DBO_BACKWARD_PENALTY = 30.0

# 收敛早停代数 (连续多少代最优解无改进则停止)
SWARM_EARLY_STOP_STALL = 50

# ============================================================
# IPSO-SA 参数
# ============================================================
IPSO_NUM_PARTICLES = 30         # 粒子群规模
IPSO_MAX_ITERATIONS = 200       # 最大迭代次数
IPSO_W_START = 0.9              # 惯性权重起始值
IPSO_W_END = 0.4                # 惯性权重终止值
IPSO_C1 = 2.0                   # 个体学习因子
IPSO_C2 = 2.0                   # 社会学习因子
IPSO_SA_T0 = 100.0              # SA 初始温度
IPSO_SA_ALPHA = 0.95            # SA 降温系数
IPSO_SA_K = 5                   # SA 内循环次数

# ============================================================
# DBO 参数 (Dung Beetle Optimizer, 2022)
# ============================================================
DBO_NUM_BEETLES = 40            # 蜣螂种群规模 (加大以探索CNN复杂表面)
DBO_MAX_ITERATIONS = 300        # 最大迭代次数 (延长以收敛到平滑路径)
DBO_P_ROLL = 0.2                # 滚球行为比例
DBO_P_BROOD = 0.2               # 育雏行为比例
DBO_P_FORAGE = 0.3              # 觅食行为比例
DBO_P_STEAL = 0.3               # 偷窃行为比例
DBO_K = 0.1                     # 滚球偏转系数
DBO_B = 0.3                     # 育雏区域半径系数
DBO_S = 0.5                     # 觅食区域半径系数

# ============================================================
# 对比实验参数
# ============================================================
COMPARISON_N_RUNS = 30          # 每个算法独立运行次数
COMPARISON_RANDOM_SEED = 42     # 基准随机种子

# CNN 模型路径
CNN_COST_SURFACE_PATH = _v3_dir / "cnn_model" / "cnn_cost_surface.npy"

# 自动创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
