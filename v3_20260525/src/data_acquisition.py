"""
v3_dl: 模块1 — 数据获取(多省份扩展 + 深度学习特征)
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: 多省份支持, 新增覆冰/雷击/植被/断裂带代理层, 合成数据生成
  - v2.20260525: 台湾单区域OSM下载
  - v1.20260525: 初始版本
依赖: shared/data_acquisition (复用公共代码), v3/config
说明:
  - 复用shared/data_acquisition.py中的OSM下载和DEM加载函数
  - 新增省份切换、扩展代理层、合成数据生成(用于无真实数据省份)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import geopandas as gpd
from shapely.geometry import box, Point
from scipy.ndimage import gaussian_filter, distance_transform_edt, uniform_filter, sobel
from scipy.spatial import KDTree
import pickle
import time
import json
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import config as cfg

# 复用shared中的基础OSM下载功能
from data_acquisition import (
    _overpass_query, _osm_to_gdf, _cache_path, _cache_valid,
    fetch_osm_roads, fetch_osm_water, fetch_osm_protected_areas,
    fetch_osm_landuse, fetch_osm_buildings, fetch_osm_railways,
    fetch_osm_airports, load_dem, load_taiwan_lines,
    build_typhoon_risk, build_seismic_risk, build_landslide_risk,
)


# ============================================================
# 多省份数据协调
# ============================================================
def get_active_bbox():
    """获取当前活动区域的边界框"""
    return cfg.PROVINCE_BBOX.get(cfg.ACTIVE_REGION, cfg.PROVINCE_BBOX["taiwan"])


def switch_region(region_name: str):
    """切换分析区域至指定省份"""
    if region_name not in cfg.PROVINCE_BBOX:
        raise ValueError(f"未知区域: {region_name}. 可选: {list(cfg.PROVINCE_BBOX.keys())}")
    cfg.ACTIVE_REGION = region_name
    print(f"[Phase1] 已切换区域至: {region_name} ({cfg.PROVINCE_BBOX[region_name]})")


# ============================================================
# 扩展OSM数据获取 — v3: 新增数据类型
# ============================================================
def fetch_osm_geological_faults(bbox):
    """下载地质构造线(断裂带代理) — 使用natural=cliff + geological=fault"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["natural"="cliff"]({bbox_str});
      way["natural"="ridge"]({bbox_str});
      way["geological"="fault"]({bbox_str});
      way["hazard_type"="landslide"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_faults")
    return _osm_to_gdf(data.get("elements", []), "line")


def fetch_osm_vegetation(bbox):
    """下载植被数据(森林/灌木/草地)"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["natural"="wood"]({bbox_str});
      way["natural"="scrub"]({bbox_str});
      way["natural"="grassland"]({bbox_str});
      way["landuse"="forest"]({bbox_str});
      way["natural"="tree_row"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_vegetation")
    return _osm_to_gdf(data.get("elements", []), "polygon")


# ============================================================
# 风险代理层 — v3: 新增覆冰、雷击、植被高度
# ============================================================

def build_ice_cover_risk(dem, transform, crs):
    """
    覆冰风险代理: 基于高程、纬度、湿度代理。
    高海拔(>1500m) + 高纬度 + 高湿度区风险最高。
    台湾高山冬季常有覆冰, 西南省份高海拔区亦然。
    """
    print("[Phase1] 生成覆冰风险层...")
    valid = ~np.isnan(dem)
    H, W = dem.shape

    lat = np.linspace(
        transform.f + transform.e * H,
        transform.f,
        H,
    )
    lat_grid = np.tile(lat.reshape(-1, 1), (1, W))

    elev_score = np.zeros_like(dem)
    elev_score[valid] = np.clip((dem[valid] - 1000) / 2500, 0, 1)

    lat_score = np.clip((lat_grid - 23) / 15, 0, 1)

    moisture = gaussian_filter(
        np.random.RandomState(42).uniform(0.5, 1.0, dem.shape).astype(np.float32),
        sigma=5,
    )

    risk = 0.5 * elev_score + 0.25 * lat_score + 0.25 * moisture
    risk = gaussian_filter(risk, sigma=2)
    risk[~valid] = np.nan
    risk = np.clip(risk, 0, 1)
    print(f"  覆冰风险层完成, 范围: [{np.nanmin(risk):.2f}, {np.nanmax(risk):.2f}]")
    return risk.astype(np.float32)


def build_lightning_risk(dem, transform, crs):
    """
    雷击风险代理: 基于地形凸起度(TPI)、高程。
    山脊和高点雷击概率显著高于山谷。
    """
    print("[Phase1] 生成雷击风险层...")
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    window_m = 300
    window_pixels = max(3, int(window_m / cellsize_m))
    if window_pixels % 2 == 0:
        window_pixels += 1

    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    mean_dem = uniform_filter(dem_filled, size=window_pixels)
    tpi = dem_filled - mean_dem

    valid = ~np.isnan(dem)
    tpi_score = np.zeros_like(dem)
    tpi_score[valid] = np.clip((tpi[valid] + 50) / 200, 0, 1)

    elev_score = np.zeros_like(dem)
    elev_score[valid] = np.clip(dem[valid] / 3000, 0, 1)

    slope_y, slope_x = np.gradient(dem_filled, cellsize_m, cellsize_m)
    slope = np.degrees(np.arctan(np.sqrt(slope_x**2 + slope_y**2)))
    slope_score = np.clip(slope / 45, 0, 1)

    risk = 0.5 * tpi_score + 0.3 * elev_score + 0.2 * slope_score
    risk = gaussian_filter(risk, sigma=2)
    risk[~valid] = np.nan
    risk = np.clip(risk, 0, 1)
    print(f"  雷击风险层完成, 范围: [{np.nanmin(risk):.2f}, {np.nanmax(risk):.2f}]")
    return risk.astype(np.float32)


def build_vegetation_height(dem, transform, crs, landuse_gdf=None):
    """
    植被高度代理: 基于土地利用分类和地形。
    林地/灌木/草地赋予不同高度值。
    """
    print("[Phase1] 生成植被高度层...")
    valid = ~np.isnan(dem)
    H, W = dem.shape

    height = np.full((H, W), 2.0, dtype=np.float32)

    # 基于地形: 陡坡区植被通常较矮(受限于土层厚度), 缓坡谷底植被高
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float64), cellsize_m, cellsize_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # 坡度修正: 低坡度→高植被(森林), 高坡度→矮植被(灌木/草)
    slope_factor = np.clip(1.0 - slope / 45, 0.3, 1.0)

    # 随机基底: 模拟不同植被类型
    rng = np.random.RandomState(123)
    base_height = rng.uniform(0, 1, dem.shape).astype(np.float32)
    base_height = gaussian_filter(base_height, sigma=3)

    height = (5 + 25 * base_height) * slope_factor
    height = gaussian_filter(height, sigma=1.5)
    height[~valid] = 0
    height = np.clip(height, 0, cfg.VEGETATION_HEIGHT_MAX)
    print(f"  植被高度层完成, 范围: [{height[valid].min():.1f}, {height[valid].max():.1f}]m")
    return height.astype(np.float32)


def build_fault_zone_proxy(dem, transform, crs, fault_gdf=None):
    """
    断裂带风险代理: 基于地形线性特征。
    使用Sobel边缘检测识别线性构造, 结合已有断裂带矢量。
    """
    print("[Phase1] 生成断裂带风险代理层...")
    valid = ~np.isnan(dem)
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))

    # Sobel边缘检测 — 识别地形线性特征
    grad_x = sobel(dem_filled.astype(np.float64), axis=1)
    grad_y = sobel(dem_filled.astype(np.float64), axis=0)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    edge_score = np.clip(edge_magnitude / np.percentile(edge_magnitude[valid], 95), 0, 1)
    edge_score = gaussian_filter(edge_score, sigma=3)

    risk = edge_score.astype(np.float32)
    risk[~valid] = np.nan
    risk = np.clip(risk, 0, 1)
    print(f"  断裂带风险层完成, 范围: [{np.nanmin(risk):.2f}, {np.nanmax(risk):.2f}]")
    return risk


# ============================================================
# 合成数据生成 — 用于无数据省份的模拟
# ============================================================
def generate_synthetic_dem(bbox, resolution_deg=0.000833, seed=42):
    """
    生成合成DEM用于无真实DEM省份的模拟。
    使用多层柏林噪声叠加模拟地形。
    """
    print("[Phase1] 生成合成DEM...")
    rng = np.random.RandomState(seed)
    lon_min, lat_min, lon_max, lat_max = bbox
    n_lon = int((lon_max - lon_min) / resolution_deg)
    n_lat = int((lat_max - lat_min) / resolution_deg)

    from rasterio.transform import from_origin
    transform = from_origin(lon_min, lat_max, resolution_deg, resolution_deg)

    # 多层噪声合成地形
    dem = np.zeros((n_lat, n_lon), dtype=np.float64)
    scales = [(200, 0.6), (100, 0.25), (50, 0.1), (25, 0.05)]
    for scale_cells, weight in scales:
        coarse_h = int(n_lat / scale_cells) + 2
        coarse_w = int(n_lon / scale_cells) + 2
        coarse = rng.uniform(0, 1, (coarse_h, coarse_w)).astype(np.float64)
        from scipy.ndimage import zoom
        zoom_h = n_lat / coarse_h
        zoom_w = n_lon / coarse_w
        fine = zoom(coarse, (zoom_h, zoom_w), order=1)
        fine = fine[:n_lat, :n_lon]
        dem += weight * fine

    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-10)
    # 缩放至0-4000m
    dem = dem * 3500 + rng.uniform(0, 500, (n_lat, n_lon)).astype(np.float64) * 0.3
    dem = gaussian_filter(dem, sigma=1.0)
    dem = dem.astype(np.float32)

    crs = cfg.WGS84
    print(f"  合成DEM: {dem.shape}, 高程范围: [{dem.min():.0f}, {dem.max():.0f}]m")
    return dem, transform, crs


def generate_synthetic_lines(bbox, dem, transform, n_lines=50, seed=123):
    """
    生成合成输电线路用于训练(当真实数据不可用时)。
    模拟输电线路偏好走向: 避开陡坡, 沿山谷, 连接城镇。
    """
    print("[Phase1] 生成合成输电线...")
    rng = np.random.RandomState(seed)
    lon_min, lat_min, lon_max, lat_max = bbox
    H, W = dem.shape

    lines = []
    for i in range(n_lines):
        start_lon = rng.uniform(lon_min + 0.1, lon_max - 0.1)
        start_lat = rng.uniform(lat_min + 0.1, lat_max - 0.1)
        end_lon = rng.uniform(lon_min + 0.1, lon_max - 0.1)
        end_lat = rng.uniform(lat_min + 0.1, lat_max - 0.1)

        n_points = rng.randint(3, 8)
        lons = np.linspace(start_lon, end_lon, n_points)
        lats = np.linspace(start_lat, end_lat, n_points)
        lons[1:-1] += rng.uniform(-0.05, 0.05, n_points - 2)
        lats[1:-1] += rng.uniform(-0.05, 0.05, n_points - 2)

        from shapely.geometry import LineString
        line = LineString(zip(lons, lats))
        lines.append({"geometry": line, "id": f"synth_{i}"})

    gdf = gpd.GeoDataFrame(lines, geometry="geometry", crs=cfg.WGS84)
    print(f"  合成输电线路: {len(gdf)} 条")
    return gdf


# ============================================================
# 综合获取所有数据 — v3扩展版
# ============================================================
def acquire_all(use_synthetic: bool = False):
    """
    获取所有原始数据(扩展版)。
    Args:
        use_synthetic: 是否使用合成数据(无真实DEM/SHP时)
    Returns:
        dict: 包含DEM、矢量和风险层的字典
    """
    result = {}
    bbox = get_active_bbox()

    # 1. DEM — province-aware: only Taiwan has real DEM
    province = cfg.ACTIVE_REGION
    if province == "taiwan" and not use_synthetic:
        try:
            dem, transform, crs, meta = load_dem()
            result["dem_source"] = "real"
            print(f"  DEM加载完成: {dem.shape}, 来源: 真实SRTM")
        except (FileNotFoundError, Exception) as e:
            print(f"  DEM加载失败: {e}, 使用合成数据...")
            dem, transform, crs = generate_synthetic_dem(bbox)
            result["dem_source"] = "synthetic"
    else:
        if province != "taiwan":
            print(f"  {province}: 无真实DEM, 使用合成数据...")
        dem, transform, crs = generate_synthetic_dem(bbox)
        result["dem_source"] = "synthetic"

    result["dem"] = dem
    result["dem_transform"] = transform
    result["dem_crs"] = crs

    # 2. 输电线 — province-aware: only Taiwan has real SHP
    if province == "taiwan" and not use_synthetic:
        try:
            result["taiwan_lines"] = load_taiwan_lines()
            result["lines_source"] = "real"
            print(f"  输电线: {len(result['taiwan_lines'])} 条 (真实数据)")
        except (FileNotFoundError, Exception) as e:
            print(f"  输电线加载失败: {e}, 使用合成数据...")
            result["taiwan_lines"] = generate_synthetic_lines(bbox, dem, transform)
            result["lines_source"] = "synthetic"
    else:
        result["taiwan_lines"] = generate_synthetic_lines(bbox, dem, transform)
        result["lines_source"] = "synthetic"
        print(f"  输电线: {len(result['taiwan_lines'])} 条 (合成数据)")

    # 3. OSM数据
    result["osm_roads"] = _safe_fetch(fetch_osm_roads, bbox, "道路")
    result["osm_water"] = _safe_fetch(fetch_osm_water, bbox, "水域")
    result["osm_protected"] = _safe_fetch(fetch_osm_protected_areas, bbox, "保护区")
    result["osm_landuse"] = _safe_fetch(fetch_osm_landuse, bbox, "土地利用")
    result["osm_buildings"] = _safe_fetch(fetch_osm_buildings, bbox, "建筑")
    result["osm_railways"] = _safe_fetch(fetch_osm_railways, bbox, "铁路")
    result["osm_airports"] = _safe_fetch(fetch_osm_airports, bbox, "机场")

    # v3: 新增数据源
    result["osm_faults"] = _safe_fetch(fetch_osm_geological_faults, bbox, "断裂带")
    result["osm_vegetation"] = _safe_fetch(fetch_osm_vegetation, bbox, "植被")

    # 4. 风险代理层 — 合成省份跳过以节省内存(直线路径不需要)
    if use_synthetic and province != "taiwan":
        print("[Phase1] 合成省份: 跳过风险代理层 (无需CNN训练, 节省内存)")
        dummy = np.zeros((2, 2), dtype=np.float32)
        for key in ["typhoon_risk", "seismic_risk", "landslide_risk",
                     "ice_cover_risk", "lightning_risk", "vegetation_height", "fault_risk"]:
            result[key] = dummy.copy()
    else:
        result["typhoon_risk"] = build_typhoon_risk(dem, transform, crs)
        result["seismic_risk"] = build_seismic_risk(dem, transform, crs)
        result["landslide_risk"] = build_landslide_risk(dem, transform, crs)
        result["ice_cover_risk"] = build_ice_cover_risk(dem, transform, crs)
        result["lightning_risk"] = build_lightning_risk(dem, transform, crs)
        result["vegetation_height"] = build_vegetation_height(dem, transform, crs,
                                                              result.get("osm_landuse"))
        result["fault_risk"] = build_fault_zone_proxy(dem, transform, crs,
                                                      result.get("osm_faults"))

    print("[Phase1] 数据获取完成 (v3扩展版)\n")
    return result


def _safe_fetch(fetch_func, bbox, name):
    """安全获取OSM数据, 失败时返回空GeoDataFrame"""
    try:
        data = fetch_func(bbox)
        print(f"  OSM {name}: {len(data)} 条/个")
        return data
    except Exception as e:
        print(f"  OSM {name}获取失败: {e}, 使用空数据")
        return gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)
