"""
台湾输电线路智能路径规划系统 — 全局配置
"""
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(r"c:\Users\86133\Desktop\大创项目文件夹\Transmission-line-route-planning")
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DIR = DATA_DIR / "downloaded"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_MAPS_DIR = OUTPUT_DIR / "maps"
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

DEM_PATH = DATA_DIR / "dem" / "台湾.tif"
SHP_PATH = DATA_DIR / "shp" / "示例数据-中国输电线路矢量.shp"

# 自动创建所有目录
for _d in [DATA_DIR, DOWNLOADED_DIR, PROCESSED_DIR, MODELS_DIR,
           OUTPUT_DIR, OUTPUT_MAPS_DIR, OUTPUT_DATA_DIR, OUTPUT_REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 台湾地区边界 (WGS84)
# ============================================================
TAIWAN_BBOX = (120.0, 21.9, 122.0, 25.4)  # lon_min, lat_min, lon_max, lat_max

# ============================================================
# 坐标参考系
# ============================================================
WGS84 = "EPSG:4326"
PROJECTED_CRS = "EPSG:32651"  # UTM 51N (适用于台湾, 单位=米)

# ============================================================
# 栅格参数
# ============================================================
BASE_RESOLUTION = 90  # 米 (DEM原始30m, 降采样3倍)
RESAMPLE_METHOD = "bilinear"
METERS_PER_DEG = 111320.0  # 赤道处每度约111.32km
SRTM_NATIVE_M = 30  # SRTM原始分辨率(米)

def meters_to_deg(meters):
    """米转度(近似, 适用于台湾纬度)"""
    return meters / METERS_PER_DEG

# ============================================================
# 地形因子参数
# ============================================================
TPI_WINDOW = 300    # TPI窗口半径(米)
ROUGHNESS_WINDOW = 9  # 粗糙度窗口(像元数)

# ============================================================
# 硬约束参数
# ============================================================
MAX_SLOPE = 40            # 坡度>40°禁止建设
WATER_BUFFER = 50         # 水域缓冲距离(米)
EXISTING_LINE_BUFFER = 30 # 与现有线路安全间距(米)
AIRPORT_BUFFER = 800      # 机场缓冲区(米) — 净空限制(仅跑道/大型机场)

# ============================================================
# 伪标签生成权重
# ============================================================
LABEL_WEIGHTS = {
    "dist_existing": 0.60,
    "slope": 0.12,
    "landuse": 0.08,
    "road_access": 0.04,
    "railway": 0.04,
    "water": 0.06,
    "protected": 0.06,
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
# A* 路径搜索参数
# ============================================================
ASTAR_NEIGHBORHOOD = 8  # 8邻域Moore
PATH_SMOOTH_RDP_EPSILON = 50  # RDP简化阈值(米)
PATH_RESAMPLE_SPACING = 30    # 最终路径采样间距(米)

# ============================================================
# 默认起止点 (WGS84)
# 起点: 核三厂(屏东恒春)  终点: 台北
# ============================================================
START_POINT = (21.95, 120.75)   # (lat, lon)
END_POINT = (25.03, 121.53)     # (lat, lon)

# ============================================================
# OSM下载参数
# ============================================================
OSM_CACHE_DAYS = 7  # OSM数据缓存有效期(天)

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
    1: 0.1,   # 森林/灌木
    2: 0.3,   # 农田/草地
    3: 0.5,   # 裸地
    4: 0.7,   # 居民区
    5: 0.8,   # 工业区
    6: 0.9,   # 水域(软约束, 非硬约束)
    7: 0.6,   # 湿地
    8: 0.2,   # 其他
}

# ============================================================
# 可视化参数
# ============================================================
FIGURE_DPI = 300
DETAIL_BUFFER_KM = 2.0  # 局部放大图缓冲区(公里)
