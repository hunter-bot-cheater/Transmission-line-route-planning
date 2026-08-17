"""
v3_dl: 输电线路智能路径规划系统 — 深度学习全面重构
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: CNN成本模型 + 神经值传播 + 梯度路径提取, 无传统搜索, 多省份数据
  - v2.20260525: 收紧硬约束阈值, 7项质量门控
  - v1.20260525: 初始版本
依赖: shared/*, torch
"""
from pathlib import Path
import copy

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(r"D:\大创")
V3_DIR = BASE_DIR / "v3_20260525"
SHARED_DIR = BASE_DIR / "shared"
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DIR = DATA_DIR / "downloaded"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
V3_MODELS_DIR = MODELS_DIR / "v3_dl"
OUTPUT_DIR = V3_DIR / "output"
OUTPUT_MAPS_DIR = OUTPUT_DIR / "maps"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"
OUTPUT_COMPARISON_DIR = OUTPUT_DIR / "comparisons"

DEM_PATH = Path(r"D:\地形数据\台湾省_DEM_30m分辨率_SRTM数据.tif")
SHP_PATH = Path(r"D:\输电线数据\示例数据-中国输电线路矢量.shp")

for _d in [DATA_DIR, DOWNLOADED_DIR, PROCESSED_DIR, MODELS_DIR, V3_MODELS_DIR,
           OUTPUT_DIR, OUTPUT_MAPS_DIR, OUTPUT_DATA_DIR, OUTPUT_REPORTS_DIR,
           OUTPUT_COMPARISON_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 多省份边界定义 (WGS84)
# ============================================================
PROVINCE_BBOX = {
    "taiwan":    (120.00, 21.90, 122.00, 25.40),
    "sichuan":   (97.35,  26.05, 108.52, 34.32),
    "yunnan":    (97.53,  21.14, 106.20, 29.25),
    "xizang":    (78.40,  26.85, 99.12,  36.50),
    "guizhou":   (103.60, 24.62, 109.60, 29.22),
    "gansu":     (92.21,  32.52, 108.77, 42.80),
    "shaanxi":   (105.50, 31.70, 111.25, 39.58),
    "fujian":    (115.83, 23.53, 120.72, 28.32),
    "chongqing": (105.28, 28.16, 110.19, 32.21),
}

ACTIVE_REGION = "taiwan"

# 向后兼容别名 (shared/data_acquisition.py 引用)
TAIWAN_BBOX = PROVINCE_BBOX["taiwan"]

def get_active_bbox():
    return PROVINCE_BBOX.get(ACTIVE_REGION, PROVINCE_BBOX["taiwan"])

# ============================================================
# 坐标参考系
# ============================================================
WGS84 = "EPSG:4326"
PROJECTED_CRS = "EPSG:32651"

# ============================================================
# 栅格参数
# ============================================================
BASE_RESOLUTION = 90
RESAMPLE_METHOD = "bilinear"
METERS_PER_DEG = 111320.0
SRTM_NATIVE_M = 30

def meters_to_deg(meters):
    return meters / METERS_PER_DEG

# ============================================================
# 地形因子参数
# ============================================================
TPI_MULTI_SCALE = [100, 300, 900]
ROUGHNESS_MULTI_SCALE = [3, 9, 27]

# ============================================================
# 硬约束参数
# ============================================================
MAX_SLOPE = 45
SLOPE_QUALITY_THRESHOLD = 45
WATER_BUFFER = 0
WATER_BUFFER_QUALITY = 30
MAX_ELEVATION = 3000
PROTECTED_BUFFER = 200
BUILDING_DENSITY_LIMIT = 500
MAX_TURN_ANGLE = 50
MAX_SINUOSITY = 3.0
MAX_CONTINUOUS_CLIMB = 2500
COST_ANOMALY_RATIO = 3.0

# v3新增硬约束
FAULT_BUFFER = 500
ICE_COVER_THRESHOLD = 0.6
LIGHTNING_THRESHOLD = 0.7
VEGETATION_HEIGHT_MAX = 30

EXISTING_LINE_BUFFER = 30
AIRPORT_BUFFER = 800
URBAN_DENSITY_THRESHOLD = 1500

# ============================================================
# 伪标签生成权重
# ============================================================
LABEL_WEIGHTS = {
    "dist_existing": 0.40,
    "slope": 0.12,
    "landuse": 0.06,
    "road_access": 0.03,
    "railway": 0.03,
    "water": 0.06,
    "protected": 0.05,
    "fault": 0.05,
    "ice_cover": 0.06,
    "lightning": 0.05,
    "vegetation": 0.04,
    "roughness": 0.05,
}

PSEUDO_LABEL_PARAMS = {
    "dist_existing_decay": 250,
    "slope_threshold": 28,
    "slope_extra_penalty": 20,
    "water_decay": 200,
    "roughness_threshold": 30,
    "fault_decay": 500,
    "vegetation_decay": 500,
    "ice_threshold": 0.4,
    "lightning_threshold": 0.5,
}

# ============================================================
# 深度学习超参数
# ============================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32

# CostUNet
DL_BATCH_SIZE = 16
DL_N_EPOCHS = 200
DL_LEARNING_RATE = 1e-3
DL_WEIGHT_DECAY = 1e-5
DL_EARLY_STOP = 25
DL_TRAIN_SPLIT = 0.8

UNET_ENCODER_CHANNELS = [32, 64, 128, 256, 512]
UNET_DECODER_CHANNELS = [256, 128, 64, 32]
UNET_BOTTLENECK = 512
UNET_DROPOUT = 0.1

# MultiScaleValuePropNet
COARSE_SIZE = (256, 256)
FINE_SIZE = (1024, 1024)
VALUE_PROP_K_COARSE = 20
VALUE_PROP_K_FINE = 10
VALUE_PROP_HIDDEN_DIM = 64
VALUE_PROP_GAMMA = 0.99
VALUE_PROP_LR = 5e-4
SOFTMIN_TEMPERATURE = 0.5

# PathRefiner
REFINER_HIDDEN = [128, 64]
REFINER_LR = 1e-4
REFINER_N_ITERATIONS = 5
REFINER_SAMPLE_STEPS = 200

# Neural Gradient Tracker
GRADIENT_STEP_BASE = 1.0
GRADIENT_MOMENTUM = 0.7
GRADIENT_MAX_STEPS = 30000
GRADIENT_CONVERGENCE_EPS = 3.0
GRADIENT_GOAL_BIAS = 0.6

# Hybrid value function parameters
VALUE_HYBRID_GOAL_WEIGHT = 3.0    # Strong goal attraction
VALUE_HYBRID_SMOOTH_SIGMA = 12.0  # ~1km smoothing at 90m resolution

# 路径平滑参数
PATH_SMOOTH_RDP_EPSILON = 90
PATH_RESAMPLE_SPACING = 30

# ============================================================
# 特征波段定义 (v3: 26维)
# ============================================================
FEATURE_BANDS = {
    # 地形 (13维)
    "elevation": 0,
    "slope": 1,
    "aspect_cos": 2,
    "aspect_sin": 3,
    "tri": 4,
    "tpi_100": 5,
    "tpi_300": 6,
    "tpi_900": 7,
    "profile_curvature": 8,
    "plan_curvature": 9,
    "roughness_3": 10,
    "roughness_9": 11,
    "roughness_27": 12,
    # 距离 (5维)
    "dist_road": 13,
    "dist_water": 14,
    "dist_existing_line": 15,
    "dist_railway": 16,
    "dist_fault": 17,
    # 分类/密度 (3维)
    "landuse_code": 18,
    "building_density": 19,
    "vegetation_height": 20,
    # 风险 (5维)
    "typhoon_risk": 21,
    "seismic_risk": 22,
    "landslide_risk": 23,
    "ice_cover_risk": 24,
    "lightning_risk": 25,
}
N_FEATURES = len(FEATURE_BANDS)

# ============================================================
# 土地利用代码 -> 软约束权值
# ============================================================
LANDUSE_SOFT_COST = {
    1: 0.1,
    2: 0.3,
    3: 0.5,
    4: 0.7,
    5: 0.8,
    6: 0.9,
    7: 0.6,
    8: 0.2,
}

# ============================================================
# 可视化参数
# ============================================================
FIGURE_DPI = 300
DETAIL_BUFFER_KM = 2.0

# ============================================================
# 10条标准测试线路 (覆盖≥4省份)
# ============================================================
TEST_CASES = [
    # --- 台湾 (4条) ---
    {
        "case_id": "case_01",
        "province": "taiwan",
        "description": "屏东恒春线(台湾, 345kV, 短距离)",
        "way_id": "way/365662519",
    },
    {
        "case_id": "case_02",
        "province": "taiwan",
        "description": "嘉义台中纵贯线(台湾, 345kV, 长距离)",
        "way_id": "way/203692582",
    },
    {
        "case_id": "case_03",
        "province": "taiwan",
        "description": "台东花东纵谷线(台湾, 161kV, 中距离)",
        "way_id": "way/179686272",
    },
    {
        "case_id": "case_04",
        "province": "taiwan",
        "description": "北部沿海线(台湾, 345kV, 中距离)",
        "way_id": "way/129716882",
    },
    # --- 四川 (2条) ---
    {
        "case_id": "case_05",
        "province": "sichuan",
        "description": "四川盆地输电走廊(四川, 500kV, 长距离)",
        "way_id": None,
        "start": (30.5, 103.0),  # (lat, lon) — 成都附近
        "end": (30.5, 106.5),     # 重庆附近
    },
    {
        "case_id": "case_06",
        "province": "sichuan",
        "description": "川西高原线路(四川, 220kV, 山区)",
        "way_id": None,
        "start": (31.0, 99.0),    # 甘孜
        "end": (31.0, 104.0),     # 绵阳
    },
    # --- 云南 (2条) ---
    {
        "case_id": "case_07",
        "province": "yunnan",
        "description": "云南高原输电线路(云南, 500kV, 中距离)",
        "way_id": None,
        "start": (28.0, 100.5),   # 迪庆
        "end": (23.0, 102.0),     # 普洱
    },
    {
        "case_id": "case_08",
        "province": "yunnan",
        "description": "滇西北山区线路(云南, 220kV, 山区)",
        "way_id": None,
        "start": (28.0, 98.5),    # 怒江
        "end": (25.0, 103.0),     # 曲靖
    },
    # --- 福建/贵州 (2条) ---
    {
        "case_id": "case_09",
        "province": "fujian",
        "description": "福建沿海输电走廊(福建, 500kV, 中距离)",
        "way_id": None,
        "start": (27.0, 118.0),   # 南平
        "end": (24.5, 118.5),     # 泉州
    },
    {
        "case_id": "case_10",
        "province": "guizhou",
        "description": "贵州喀斯特山区线路(贵州, 220kV, 中距离)",
        "way_id": None,
        "start": (27.0, 105.0),   # 毕节
        "end": (26.5, 108.5),     # 黔东南
    },
]

# 默认起止点 (台湾)
START_POINT = (21.95, 120.75)
END_POINT = (25.03, 121.53)

# OSM下载参数
OSM_CACHE_DAYS = 7
RANDOM_SEED = 42
