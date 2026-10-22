# transmission_path_planner/config.py
"""全局配置：第三方库导入、常量、路径、随机种子确定性初始化。"""

import os
import sys
import time
import math
import json
import argparse
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. 依赖导入（缺失时给出明确提示，但核心逻辑尽量自包含）
# ----------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    sys.exit("[FATAL] 缺少 numpy，请先运行: pip install numpy")

try:
    import scipy.ndimage as ndi
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.windows import from_bounds
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    try:
        import tifffile  # 回退读 DEM
    except ImportError:
        tifffile = None

try:
    from shapely.geometry import box, Point, LineString, Polygon
    from shapely.geometry.base import BaseGeometry
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None


# ============================================================================
# 1. 配置
# ============================================================================
PACKAGE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = PACKAGE_DIR.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DIR = next((p for p in (
    PACKAGE_DIR / "data" / "downloaded",
    BASE_DIR / "V5_final" / "data" / "downloaded",
    DATA_DIR / "downloaded",
) if p.exists()), PACKAGE_DIR / "data" / "downloaded")
MODELS_DIR = DATA_DIR / "models" / "v3_dl"

DEM_CANDIDATES = [
    PACKAGE_DIR / "data" / "taiwan_dem_35m.tif",
    Path(r"D:\地形数据\台湾省_DEM_30m分辨率_SRTM数据.tif"),
    BASE_DIR / "data" / "downloaded" / "taiwan_dem.tif",
]
SHP_CANDIDATES = [
    BASE_DIR / "输电线数据" / "示例数据-中国输电线路矢量.shp",
    Path(r"D:\输电线数据\示例数据-中国输电线路矢量.shp"),
]
MODEL_WEIGHTS = [
    MODELS_DIR / "cost_unet_improved.pt",
    MODELS_DIR / "cost_unet_best.pt",
]

# 26 维特征波段顺序（与 v3 config.FEATURE_BANDS 一致）
FEATURE_BANDS = {
    "elevation": 0, "slope": 1, "aspect_cos": 2, "aspect_sin": 3,
    "tri": 4, "tpi_100": 5, "tpi_300": 6, "tpi_900": 7,
    "profile_curvature": 8, "plan_curvature": 9,
    "roughness_3": 10, "roughness_9": 11, "roughness_27": 12,
    "dist_road": 13, "dist_water": 14, "dist_existing_line": 15,
    "dist_railway": 16, "dist_fault": 17,
    "landuse_code": 18, "building_density": 19, "vegetation_height": 20,
    "typhoon_risk": 21, "seismic_risk": 22, "landslide_risk": 23,
    "ice_cover_risk": 24, "lightning_risk": 25,
}
N_FEATURES = 26

# 硬约束阈值
METERS_PER_DEG = 111320.0

# 工作网格 / 训练超参
WORK_MAX_SIDE = 4096       # 尽量保留 20–35m 地形栅格；可用 --max-side 按内存调低
PPO_GRID = 256             # PPO 训练环境网格
RANDOM_SEED = 42

# 全局固定随机种子: 消除 torch/numpy 随机初始化导致的训练发散与 run-to-run 方差
# (此前 深美~冬山線 VIN-Grad 在 7.79% 与 71.85% 间随机跳变、PPO 时好时坏均源于此)。
import random as _random
_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # CUDA 卷积/原子操作为非确定性, 不受 manual_seed 控制 -> 同种子下 run-to-run 仍方差
    # (深美~冬山線曾单跑 3.91% PASS、全量跑 20% FAIL 即源于此)。开启确定性卷积,
    # 配合每条走廊训练前独立重置种子, 使 VIN/PPO 训练逐走廊可复现、顺序无关。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# 默认台湾山区走廊测试用例 (lat, lon)，均落在台湾陆地 DEM 内
DEFAULT_CASES = [
    ("T01_横贯中央山脉", (23.85, 120.70), (23.95, 121.50)),
    ("T02_脊梁南北向",   (24.20, 121.00), (23.20, 120.90)),
    ("T03_西南-东北",     (23.30, 120.60), (24.00, 121.30)),
    ("T04_纵谷走廊",      (24.10, 120.80), (22.95, 121.20)),
]

# ============================================================================
# 1.1 真实/参考路径（地面真值）配置
# ----------------------------------------------------------------------------
# v3 产出每条走廊一个 real_path.shp (EPSG:4326, (lon,lat) 与本项目 geo 约定一致)。
# 注意: 这些是 v3 的真实走廊, 与上方 T01~T04 不重叠; T01~T04 目前无对应真实线。
# 阶段二: 把 T01~T04 各自真实线路放入 REAL_PATH_DIR，供规划后参考线路评估自动匹配。
REAL_PATH_DIR = BASE_DIR / "v3_20260525" / "output" / "comparisons"
REAL_CASES = [
    # (名称, 起点(lat,lon), 终点(lat,lon), real_path 文件名)
    ("R01_南部屏东",     (22.418, 120.610), (21.960, 120.749), "case_01_real_path.shp"),
    ("R02_中南部纵谷",   (23.065, 120.427), (23.891, 120.770), "case_02_real_path.shp"),
    ("R03_东南部",       (22.783, 121.090), (23.335, 121.305), "case_03_real_path.shp"),
    ("R04_北部台北宜兰", (24.820, 121.195), (25.034, 121.363), "case_04_real_path.shp"),
]
# 名称 -> v3 real_path 文件名 的显式映射(阶段一)
REAL_PATH_MAP = {name: fname for (name, _, _, fname) in REAL_CASES}
