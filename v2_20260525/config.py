"""
v2_strict: 台湾输电线路智能路径规划系统 — 严格化全局配置
版本: v2.20260525
作者: path_planning_team
变更记录:
  - v2.20260525: 收紧所有硬约束阈值, 新增质量门控, A*优化
  - v1.20260525: 初始版本(宽松阈值)
依赖: shared/*
"""
from pathlib import Path
import copy

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(r"c:\Users\86133\Desktop\大创项目文件夹\Transmission-line-route-planning")
V2_DIR = BASE_DIR / "v2_20260525"
SHARED_DIR = BASE_DIR / "shared"
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DIR = DATA_DIR / "downloaded"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = V2_DIR / "output"
OUTPUT_MAPS_DIR = OUTPUT_DIR / "maps"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

DEM_PATH = DATA_DIR / "dem" / "台湾.tif"
SHP_PATH = DATA_DIR / "shp" / "示例数据-中国输电线路矢量.shp"

for _d in [DATA_DIR, DOWNLOADED_DIR, PROCESSED_DIR, MODELS_DIR,
           OUTPUT_DIR, OUTPUT_MAPS_DIR, OUTPUT_DATA_DIR, OUTPUT_REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 台湾地区边界 (WGS84)
# ============================================================
TAIWAN_BBOX = (120.0, 21.9, 122.0, 25.4)

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
TPI_WINDOW = 300
ROUGHNESS_WINDOW = 9

# ============================================================
# 硬约束参数 — v2_strict: 收紧阈值
# ============================================================
# 硬约束坡度上限 (仅阻断极端陡坡)
MAX_SLOPE = 45
# 质量门控坡度阈值 (独立于硬约束)
SLOPE_QUALITY_THRESHOLD = 45
# 水域硬约束 (0=不纳入硬约束, 仅质量门控检测; >0=阻断水域像元, 小值如10m不产生膨胀)
WATER_BUFFER = 0
# 质量门控水域距离阈值
WATER_BUFFER_QUALITY = 30
# v2_strict: MAX_ELEVATION 从3500m降低至3000m
MAX_ELEVATION = 3000
# v2_strict: 新增保护区缓冲区200m
PROTECTED_BUFFER = 200
# v2_strict: 新增建筑密度硬约束 (>5栋/公顷 = 500栋/km²)
BUILDING_DENSITY_LIMIT = 500
# v2_strict: 新增路径最大曲率半径限制
MAX_CURVATURE_RADIUS = 500
# v2_strict: 新增最大转角 (相邻三点, 度) — 50°适配栅格路径经平滑后
MAX_TURN_ANGLE = 50
# v2_strict: 新增最大sinuosity (路径长度/直线距离)
MAX_SINUOSITY = 3.0
# v2_strict: 新增高程连贯性阈值 (单调段高程变化, 米)
MAX_CONTINUOUS_CLIMB = 2500
# v2_strict: 新增成本异常阈值 (路径均值/全图中位数的倍数)
COST_ANOMALY_RATIO = 3.0

EXISTING_LINE_BUFFER = 30
AIRPORT_BUFFER = 800
URBAN_DENSITY_THRESHOLD = 1500

# ============================================================
# 伪标签生成权重 — v2_strict: 强化距现有线/坡度/粗糙度/水域
# ============================================================
# v2_strict: dist_existing从0.60提升至0.65, 衰减距离从400缩短至250
LABEL_WEIGHTS = {
    "dist_existing": 0.65,
    "slope": 0.15,
    "landuse": 0.04,
    "road_access": 0.02,
    "railway": 0.02,
    "water": 0.08,
    "protected": 0.04,
}

# v2_strict: 伪标签中各因素的衰减/阈值参数
PSEUDO_LABEL_PARAMS = {
    "dist_existing_decay": 250,    # v1=400, 半衰从~280m缩至~173m
    "slope_threshold": 28,          # v1=40(MAX_SLOPE), 使用28作为满分参考
    "slope_extra_penalty": 20,      # v1=25, 陡坡额外惩罚起点从25°降至20°
    "water_decay": 200,             # v1=150, 水域惩罚距离从150m扩至200m
    "roughness_threshold": 30,      # v2_strict: 新增粗糙度阈值(米)
}

# ============================================================
# 随机森林参数
# ============================================================
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 10
RF_TEST_SIZE = 0.2
RF_RANDOM_STATE = 42

# ============================================================
# A* 路径搜索参数 — v2_strict: 优化搜索行为
# ============================================================
ASTAR_NEIGHBORHOOD = 8
# A*启发式权重 (1.0=标准A*)
ASTAR_HEURISTIC_WEIGHT = 1.0
# 对角线移动成本 (sqrt(2)标准值)
ASTAR_DIAGONAL_COST = 1.414
# v2_strict: 起点/终点落入硬约束区时自动寻找最近可行点
ASTAR_STRICT_ENDPOINTS = False

PATH_SMOOTH_RDP_EPSILON = 90
PATH_RESAMPLE_SPACING = 30

# ============================================================
# 默认起止点 (WGS84)
# ============================================================
START_POINT = (21.95, 120.75)
END_POINT = (25.03, 121.53)

# ============================================================
# OSM下载参数
# ============================================================
OSM_CACHE_DAYS = 7

# ============================================================
# 特征波段定义
# ============================================================
FEATURE_BANDS = {
    "elevation": 0,
    "slope": 1,
    "aspect_cos": 2,
    "aspect_sin": 3,
    "tri": 4,
    "tpi": 5,
    "profile_curvature": 6,
    "plan_curvature": 7,
    "roughness": 8,
    "dist_road": 9,
    "dist_water": 10,
    "dist_existing_line": 11,
    "landuse_code": 12,
    "building_density": 13,
    "typhoon_risk": 14,
    "seismic_risk": 15,
    "landslide_risk": 16,
    "dist_railway": 17,
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
# 10条标准测试线路 — v2_strict
# ============================================================
TEST_CASES = [
    {
        "case_id": "case_01",
        "description": "屏东恒春线(南台湾, 345kV, 短距离)",
        "way_id": "way/365662519",
    },
    {
        "case_id": "case_02",
        "description": "嘉义台中纵贯线(中台湾, 345kV, 长距离)",
        "way_id": "way/203692582",
    },
    {
        "case_id": "case_03",
        "description": "新竹桃园线(西北台湾, 161kV, 中距离)",
        "way_id": "way/202162589",
    },
    {
        "case_id": "case_04",
        "description": "台东花东纵谷线(东台湾, 161kV, 中距离)",
        "way_id": "way/179686272",
    },
    {
        "case_id": "case_05",
        "description": "台中山线(中台湾山区, 345kV, 中距离)",
        "way_id": "way/199445383",
    },
    {
        "case_id": "case_06",
        "description": "台中苗栗线(中西部, 161kV, 中距离)",
        "way_id": "way/184391013",
    },
    {
        "case_id": "case_07",
        "description": "南台湾联络线(345kV, 中距离)",
        "way_id": "way/365662507",
    },
    {
        "case_id": "case_08",
        "description": "中部横贯线(345kV, 长距离山区)",
        "way_id": "way/199783876",
    },
    {
        "case_id": "case_09",
        "description": "北部沿海线(345kV, 中距离)",
        "way_id": "way/129716882",
    },
    {
        "case_id": "case_10",
        "description": "东部联络线(161kV, 中距离)",
        "way_id": "way/136343018",
    },
]
