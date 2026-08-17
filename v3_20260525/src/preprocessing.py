"""
v3_dl: 模块2 — 数据预处理(扩展多尺度特征 + 深度学习就绪)
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: 多尺度TPI(100/300/900m), 多尺度粗糙度(3/9/27像元),
                 扩展约束(断裂带/覆冰/雷击/植被), 深度学习特征归一化
  - v2.20260525: 收紧硬约束阈值
  - v1.20260525: 初始版本
依赖: v3/config, v3/src/data_acquisition
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio import features
from scipy.ndimage import (
    uniform_filter, sobel, gaussian_filter,
    distance_transform_edt, binary_dilation,
)
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import box
import pickle
from typing import Optional, Dict, Any, Tuple

import config as cfg


# ============================================================
# CRS 统一与重采样
# ============================================================
def reproject_raster(src_array, src_transform, src_crs, dst_crs, dst_resolution=None):
    if src_crs == dst_crs and dst_resolution is None:
        return src_array, src_transform
    if dst_resolution is None:
        dst_resolution = abs(src_transform.a)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs,
        src_array.shape[1], src_array.shape[0],
        left=src_transform.c, bottom=src_transform.f + src_transform.e * src_array.shape[0],
        right=src_transform.c + src_transform.a * src_array.shape[1], top=src_transform.f,
        resolution=dst_resolution,
    )
    dst_array = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=src_array, destination=dst_array,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform


def vector_to_raster(gdf, transform, shape, burn_value=1, all_touched=True):
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=np.float32)
    shapes = [(geom, burn_value) for geom in gdf.geometry if geom and not geom.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=np.float32)
    raster = features.rasterize(
        shapes, out_shape=shape, transform=transform,
        fill=0, all_touched=all_touched, dtype=np.float32,
    )
    return raster


def resample_to_target(src_array, src_transform, target_transform, target_shape, method="bilinear"):
    dst_array = np.full(target_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_array, destination=dst_array,
        src_transform=src_transform, src_crs=cfg.WGS84,
        dst_transform=target_transform, dst_crs=cfg.WGS84,
        resampling=Resampling.bilinear if method == "bilinear" else Resampling.nearest,
    )
    return dst_array


# ============================================================
# 地形因子提取 — v3: 多尺度扩展
# ============================================================
def compute_slope(dem, transform):
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float32), cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad)


def compute_aspect(dem, transform):
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float32), cellsize_y, cellsize_x)
    aspect_rad = np.arctan2(-dy, dx)
    return np.cos(aspect_rad), np.sin(aspect_rad)


def compute_tri(dem, transform):
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    mean_neighbors = uniform_filter(dem_filled, size=3)
    mean_neighbors = (9 * mean_neighbors - dem_filled) / 8
    return np.abs(dem_filled - mean_neighbors)


def compute_tpi_multiscale(dem, transform, scales_m=None):
    """多尺度地形位置指数"""
    if scales_m is None:
        scales_m = cfg.TPI_MULTI_SCALE
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    results = {}
    for scale_m in scales_m:
        window_pixels = max(3, int(scale_m / cellsize_m))
        if window_pixels % 2 == 0:
            window_pixels += 1
        mean_dem = uniform_filter(dem_filled, size=window_pixels)
        tpi = dem_filled - mean_dem
        results[f"tpi_{scale_m}"] = tpi
    return results


def compute_curvature(dem, transform):
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float32), cellsize_y, cellsize_x)
    dyy, dyx = np.gradient(dy, cellsize_y, cellsize_x)
    dxy, dxx = np.gradient(dx, cellsize_y, cellsize_x)
    p = dx ** 2 + dy ** 2
    p_safe = np.where(p > 1e-6, p, 1e-6)
    profile_curv = -(dxx * dx ** 2 + 2 * dxy * dx * dy + dyy * dy ** 2) / (p_safe ** 1.5)
    plan_curv = -(dxx * dy ** 2 - 2 * dxy * dx * dy + dyy * dx ** 2) / (p_safe ** 1.5)
    return profile_curv, plan_curv


def compute_roughness_multiscale(dem, transform, windows=None):
    """多尺度地形粗糙度"""
    if windows is None:
        windows = cfg.ROUGHNESS_MULTI_SCALE
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    results = {}
    for w in windows:
        mean = uniform_filter(dem_filled, size=w)
        mean_sq = uniform_filter(dem_filled ** 2, size=w)
        variance = mean_sq - mean ** 2
        variance = np.clip(variance, 0, None)
        results[f"roughness_{w}"] = np.sqrt(variance)
    return results


def derive_terrain_factors(dem, transform):
    """提取所有地形因子 (v3: 多尺度扩展)"""
    print("[Phase2] 提取地形因子 (v3多尺度)...")
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    factors = {}

    factors["slope"] = compute_slope(dem_filled, transform)
    print("  - 坡度完成")

    aspect_cos, aspect_sin = compute_aspect(dem_filled, transform)
    factors["aspect_cos"] = aspect_cos
    factors["aspect_sin"] = aspect_sin
    print("  - 坡向完成")

    factors["tri"] = compute_tri(dem_filled, transform)
    print("  - TRI完成")

    tpi_results = compute_tpi_multiscale(dem_filled, transform)
    factors.update(tpi_results)
    print(f"  - 多尺度TPI完成: {list(tpi_results.keys())}")

    profile_curv, plan_curv = compute_curvature(dem_filled, transform)
    factors["profile_curvature"] = profile_curv
    factors["plan_curvature"] = plan_curv
    print("  - 曲率完成")

    rough_results = compute_roughness_multiscale(dem_filled, transform)
    factors.update(rough_results)
    print(f"  - 多尺度粗糙度完成: {list(rough_results.keys())}")

    return factors


# ============================================================
# 约束掩膜生成 — v3: 扩展硬/软约束
# ============================================================
def generate_hard_mask(data, transform, shape):
    """
    v3: 生成硬约束掩膜 (0=禁止建设, 1=允许)
    禁止区域:
      1. 坡度 > MAX_SLOPE
      2. 水域 (如启用缓冲)
      3. 自然保护区 + 缓冲区
      4. 高程 > MAX_ELEVATION
      5. 密集建筑区
      6. [v3新增] 断裂带 + 缓冲区
      7. [v3新增] 极高覆冰风险区
      8. [v3新增] 极高雷击风险区
    """
    print("[Phase2] 生成硬约束掩膜 (v3扩展)...")
    hard_mask = np.ones(shape, dtype=np.uint8)
    pixel_deg = abs(transform.a)

    # 1. 坡度
    slope = data.get("slope")
    if slope is not None:
        slope_aligned = resample_to_target(slope, transform, transform, shape)
        hard_mask[slope_aligned > cfg.MAX_SLOPE] = 0
        print(f"  坡度>{cfg.MAX_SLOPE}°: {(slope_aligned > cfg.MAX_SLOPE).sum()} 像元")

    # 2. 水域
    water_gdf = data.get("osm_water")
    if water_gdf is not None and len(water_gdf) > 0:
        water_raster = vector_to_raster(water_gdf, transform, shape)
        if cfg.WATER_BUFFER > 0:
            buffer_pixels = max(1, int(cfg.meters_to_deg(cfg.WATER_BUFFER) / pixel_deg))
            water_buffer = binary_dilation(water_raster > 0, iterations=buffer_pixels)
            hard_mask[water_buffer] = 0
            print(f"  水域+{cfg.WATER_BUFFER}m缓冲: {water_buffer.sum()} 像元")
        else:
            print("  水域未纳入硬约束(WATER_BUFFER=0)")

    # 3. 自然保护区
    protected_gdf = data.get("osm_protected")
    if protected_gdf is not None and len(protected_gdf) > 0:
        protected_raster = vector_to_raster(protected_gdf, transform, shape)
        if cfg.PROTECTED_BUFFER > 0:
            buffer_pixels = max(1, int(cfg.meters_to_deg(cfg.PROTECTED_BUFFER) / pixel_deg))
            protected_buffer = binary_dilation(protected_raster > 0, iterations=buffer_pixels)
            hard_mask[protected_buffer] = 0
            print(f"  保护区+{cfg.PROTECTED_BUFFER}m缓冲: {protected_buffer.sum()} 像元")

    # 4. 高海拔
    dem_aligned = data.get("dem")
    if dem_aligned is not None:
        dem_r = resample_to_target(dem_aligned, transform, transform, shape)
        hard_mask[dem_r > cfg.MAX_ELEVATION] = 0
        print(f"  高程>{cfg.MAX_ELEVATION}m: {(dem_r > cfg.MAX_ELEVATION).sum()} 像元")

    # 5. 密集建筑区
    build_density = data.get("building_density")
    if build_density is not None and np.any(build_density > 0):
        bd_aligned = resample_to_target(build_density, transform, transform, shape)
        hard_mask[bd_aligned > cfg.BUILDING_DENSITY_LIMIT] = 0
        print(f"  建筑密度>{cfg.BUILDING_DENSITY_LIMIT}: {(bd_aligned > cfg.BUILDING_DENSITY_LIMIT).sum()} 像元")

    # 6. [v3] 断裂带
    fault_gdf = data.get("osm_faults")
    if fault_gdf is not None and len(fault_gdf) > 0:
        fault_raster = vector_to_raster(fault_gdf, transform, shape)
        buffer_pixels = max(1, int(cfg.meters_to_deg(cfg.FAULT_BUFFER) / pixel_deg))
        fault_buffer = binary_dilation(fault_raster > 0, iterations=buffer_pixels)
        hard_mask[fault_buffer] = 0
        print(f"  断裂带+{cfg.FAULT_BUFFER}m缓冲: {fault_buffer.sum()} 像元")

    # 7. [v3] 极高覆冰风险
    ice_risk = data.get("ice_cover_risk")
    if ice_risk is not None:
        ice_aligned = resample_to_target(ice_risk, transform, transform, shape)
        hard_mask[ice_aligned > cfg.ICE_COVER_THRESHOLD] = 0
        print(f"  覆冰>{cfg.ICE_COVER_THRESHOLD}: {(ice_aligned > cfg.ICE_COVER_THRESHOLD).sum()} 像元")

    # 8. [v3] 极高雷击风险
    lightning = data.get("lightning_risk")
    if lightning is not None:
        light_aligned = resample_to_target(lightning, transform, transform, shape)
        hard_mask[light_aligned > cfg.LIGHTNING_THRESHOLD] = 0
        print(f"  雷击>{cfg.LIGHTNING_THRESHOLD}: {(light_aligned > cfg.LIGHTNING_THRESHOLD).sum()} 像元")

    blocked = (hard_mask == 0).sum()
    print(f"  硬约束禁止: {blocked} 像元 ({blocked / max(hard_mask.size, 1) * 100:.1f}%)")
    return hard_mask


def generate_soft_mask(data, transform, shape):
    """v3: 生成软约束掩膜 (0-1, 值越高成本越高)"""
    print("[Phase2] 生成软约束掩膜...")
    soft_mask = np.ones(shape, dtype=np.float32)

    landuse_code = data.get("landuse_code")
    if landuse_code is not None and np.any(landuse_code > 0):
        for code, factor in cfg.LANDUSE_SOFT_COST.items():
            soft_mask[landuse_code == code] = factor
        print(f"  土地利用软约束已应用")
    else:
        dem = data.get("dem")
        if dem is not None:
            slope = compute_slope(np.nan_to_num(dem, nan=0), transform)
            slope_aligned = resample_to_target(slope, transform, transform, shape)
            soft_mask = 1.0 - 0.7 * np.clip(slope_aligned / cfg.MAX_SLOPE, 0, 1)
            print(f"  基于坡度的软约束代理已应用")

    # [v3] 植被高度软约束
    veg_height = data.get("vegetation_height")
    if veg_height is not None:
        veg_aligned = resample_to_target(veg_height, transform, transform, shape)
        veg_factor = 1.0 + 0.3 * np.clip(veg_aligned / cfg.VEGETATION_HEIGHT_MAX, 0, 1)
        soft_mask = soft_mask * veg_factor
        print(f"  植被高度软约束已应用")

    print(f"  软约束完成")
    return np.clip(soft_mask, 0.01, 2.0)


# ============================================================
# 距离栅格计算 — v3: 扩展
# ============================================================
def compute_distance_raster(gdf, transform, shape, max_dist_m=5000):
    if gdf is None or len(gdf) == 0:
        return np.full(shape, float(max_dist_m), dtype=np.float32)
    raster = vector_to_raster(gdf, transform, shape, burn_value=1)
    if raster.sum() == 0:
        return np.full(shape, float(max_dist_m), dtype=np.float32)
    cellsize_deg = abs(transform.a)
    cellsize_m = cellsize_deg * cfg.METERS_PER_DEG
    dist_pixels = distance_transform_edt(1 - raster)
    dist_m = dist_pixels * cellsize_m
    return np.clip(dist_m, 0, max_dist_m).astype(np.float32)


# ============================================================
# 土地利用分类栅格化
# ============================================================
def _landuse_tag_to_code(tags):
    if not tags:
        return 8
    lu = tags.get("landuse", "").lower()
    nat = tags.get("natural", "").lower()
    mapping = {
        "forest": 1, "wood": 1, "scrub": 1, "heath": 1,
        "farmland": 2, "farmyard": 2, "orchard": 2, "vineyard": 2,
        "meadow": 2, "grass": 2, "grassland": 2, "greenfield": 2,
        "bare_rock": 3, "bare_ground": 3, "scree": 3, "sand": 3, "beach": 3,
        "residential": 4, "retail": 4, "commercial": 4, "urban": 4,
        "industrial": 5, "quarry": 5, "brownfield": 5, "construction": 5,
        "reservoir": 6, "basin": 6, "water": 6, "salt_pond": 6, "aquaculture": 6,
        "wetland": 7, "marsh": 7, "swamp": 7, "mud": 7,
        "cemetery": 8, "recreation_ground": 8, "village_green": 8,
        "military": 8, "allotments": 8, "plant_nursery": 8,
    }
    for key, code in mapping.items():
        if key in lu or key in nat:
            return code
    return 8


def rasterize_landuse(gdf, transform, shape):
    if gdf is None or len(gdf) == 0:
        return np.full(shape, 8, dtype=np.uint8)
    from collections import defaultdict
    groups = defaultdict(list)
    for _, row in gdf.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            tags = row.get("tags", {}) if hasattr(row, "tags") else {}
            code = _landuse_tag_to_code(tags)
            groups[code].append(geom)
        except Exception:
            continue
    result = np.zeros(shape, dtype=np.uint8)
    for code in sorted(groups.keys()):
        shapes_list = [(g, code) for g in groups[code] if g is not None]
        if not shapes_list:
            continue
        mask = features.rasterize(
            shapes_list, out_shape=shape, transform=transform,
            fill=0, all_touched=True, dtype=np.uint8,
        )
        result = np.where(mask == code, code, result)
    result[result == 0] = 8
    return result


def rasterize_building_density(gdf, transform, shape):
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=np.float32)
    building_raster = vector_to_raster(gdf, transform, shape, burn_value=1, all_touched=True)
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    sigma_pixels = max(2, int(500 / cellsize_m))
    density = gaussian_filter(building_raster.astype(np.float32), sigma=sigma_pixels)
    cell_area_km2 = (cellsize_m / 1000) ** 2
    return (density / cell_area_km2).astype(np.float32)


# ============================================================
# 栅格对齐 — v3: 扩展更多层
# ============================================================
def align_all_rasters(data, reference_dem, ref_transform, osm_data=None):
    """将所有栅格对齐至统一的90m分辨率参考网格 (v3扩展)"""
    print("[Phase2] 栅格对齐至统一分辨率 (v3扩展)...")
    orig_res_deg = abs(ref_transform.a)
    dst_res_deg = orig_res_deg * (cfg.BASE_RESOLUTION / 30.0)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        cfg.WGS84, cfg.WGS84,
        reference_dem.shape[1], reference_dem.shape[0],
        left=ref_transform.c,
        bottom=ref_transform.f + ref_transform.e * reference_dem.shape[0],
        right=ref_transform.c + ref_transform.a * reference_dem.shape[1],
        top=ref_transform.f,
        resolution=dst_res_deg,
    )
    print(f"  目标分辨率: {cfg.BASE_RESOLUTION}m, 网格: {dst_width}x{dst_height}")

    aligned = {}
    aligned["transform"] = dst_transform
    aligned["shape"] = (dst_height, dst_width)
    aligned["extent"] = (
        dst_transform.c,
        dst_transform.c + dst_transform.a * dst_width,
        dst_transform.f + dst_transform.e * dst_height,
        dst_transform.f,
    )

    # 地形因子层
    terrain_layers = [
        "dem", "slope", "aspect_cos", "aspect_sin", "tri",
        "tpi_100", "tpi_300", "tpi_900",
        "profile_curvature", "plan_curvature",
        "roughness_3", "roughness_9", "roughness_27",
    ]
    for name in terrain_layers:
        arr = data.get(name)
        if arr is not None:
            aligned[name] = resample_to_target(arr, ref_transform, dst_transform,
                                               (dst_height, dst_width))

    # 风险层
    risk_layers = [
        "typhoon_risk", "seismic_risk", "landslide_risk",
        "ice_cover_risk", "lightning_risk", "fault_risk",
    ]
    for name in risk_layers:
        arr = data.get(name)
        if arr is not None:
            aligned[name] = resample_to_target(arr, ref_transform, dst_transform,
                                               (dst_height, dst_width))
            print(f"  对齐: {name}")

    # 距离栅格
    print("  计算距离栅格...")
    aligned["dist_road"] = compute_distance_raster(
        data.get("osm_roads"), dst_transform, (dst_height, dst_width))
    aligned["dist_water"] = compute_distance_raster(
        data.get("osm_water"), dst_transform, (dst_height, dst_width))
    aligned["dist_existing_line"] = compute_distance_raster(
        data.get("taiwan_lines"), dst_transform, (dst_height, dst_width), max_dist_m=10000)
    aligned["dist_railway"] = compute_distance_raster(
        data.get("osm_railways"), dst_transform, (dst_height, dst_width), max_dist_m=5000)
    # [v3] 断裂带距离
    aligned["dist_fault"] = compute_distance_raster(
        data.get("osm_faults"), dst_transform, (dst_height, dst_width), max_dist_m=10000)

    # 植被高度
    veg = data.get("vegetation_height")
    if veg is not None:
        aligned["vegetation_height"] = resample_to_target(veg, ref_transform, dst_transform,
                                                          (dst_height, dst_width))
        print(f"  对齐: vegetation_height")

    # 分类层
    if osm_data:
        airports_gdf = osm_data.get("osm_airports")
        if airports_gdf is not None and len(airports_gdf) > 0:
            aligned["airport_raster"] = vector_to_raster(
                airports_gdf, dst_transform, (dst_height, dst_width))
            print(f"  机场栅格化: {(aligned['airport_raster'] > 0).sum()} 像元")

        landuse_gdf = osm_data.get("osm_landuse")
        if landuse_gdf is not None and len(landuse_gdf) > 0:
            print("  栅格化土地利用分类...")
            aligned["landuse_code"] = rasterize_landuse(landuse_gdf, dst_transform,
                                                        (dst_height, dst_width))

        buildings_gdf = osm_data.get("osm_buildings")
        if buildings_gdf is not None and len(buildings_gdf) > 0:
            print("  计算建筑密度...")
            aligned["building_density"] = rasterize_building_density(
                buildings_gdf, dst_transform, (dst_height, dst_width))

    print("[Phase2] 栅格对齐完成\n")
    return aligned


# ============================================================
# 深度学习特征归一化
# ============================================================
def normalize_features(aligned):
    """
    v3: 对各特征层进行归一化/标准化, 适配深度学习训练。
    对每个特征层, 计算全局统计量并规范化到合理范围。
    """
    print("[Phase2] 特征归一化 (深度学习就绪)...")
    shape = aligned["shape"]
    norm_params = {}

    for key in cfg.FEATURE_BANDS:
        if key not in aligned:
            continue
        arr = aligned[key].copy()
        valid = np.isfinite(arr)

        if key in ("elevation",):
            norm_params[key] = {"mean": float(np.mean(arr[valid])), "std": float(np.std(arr[valid]))}
            arr = (arr - norm_params[key]["mean"]) / max(norm_params[key]["std"], 1e-6)
        elif key.startswith("slope") or key.startswith("aspect") or key.startswith("roughness"):
            norm_params[key] = {"max": float(np.percentile(arr[valid], 99))}
            arr = np.clip(arr / max(norm_params[key]["max"], 1e-6), -3, 3)
        elif key.startswith("tpi") or key.startswith("profile") or key.startswith("plan"):
            p99 = float(np.percentile(np.abs(arr[valid]), 99))
            norm_params[key] = {"p99": p99}
            arr = np.clip(arr / max(p99, 1e-6), -3, 3)
        elif key.startswith("dist_"):
            norm_params[key] = {"max": 5000.0}
            arr = arr / 5000.0
        elif key in ("landuse_code",):
            norm_params[key] = {"num_classes": 9}
            arr = arr / 8.0
        elif key in ("building_density",):
            p99 = float(np.percentile(arr[valid], 99))
            norm_params[key] = {"p99": max(p99, 1)}
            arr = np.clip(arr / max(p99, 1), 0, 1)
        elif "risk" in key or key.startswith("vegetation"):
            norm_params[key] = {"max": 1.0}
            arr = np.clip(arr, 0, 1)

        aligned[key] = arr.astype(np.float32)

    aligned["_norm_params"] = norm_params
    print(f"  归一化完成: {len(norm_params)} 层")
    return aligned
