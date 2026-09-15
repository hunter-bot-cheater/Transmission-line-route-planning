"""
模块2: 数据预处理
- CRS统一至WGS84
- 提取地形因子(坡度/坡向/TRI/TPI/曲率/粗糙度)
- 生成硬约束和软约束掩膜
- 栅格对齐至统一分辨率
"""
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio import features
from scipy.ndimage import uniform_filter, sobel, gaussian_filter, distance_transform_edt, binary_dilation
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import box
import pickle
from pathlib import Path

import config as cfg


# ============================================================
# CRS 统一
# ============================================================
def reproject_raster(src_array, src_transform, src_crs, dst_crs, dst_resolution=None):
    """将栅格重投影至目标CRS"""
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
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform


def vector_to_raster(gdf, transform, shape, burn_value=1, all_touched=True):
    """矢量转为栅格"""
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=np.float32)

    shapes = [(geom, burn_value) for geom in gdf.geometry if geom and not geom.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=np.float32)

    raster = features.rasterize(
        shapes, out_shape=shape, transform=transform,
        fill=0, all_touched=all_touched, dtype=np.float32
    )
    return raster


def resample_to_target(src_array, src_transform, target_transform, target_shape, method="bilinear"):
    """将栅格重采样至目标网格"""
    dst_array = np.full(target_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=cfg.WGS84,
        dst_transform=target_transform,
        dst_crs=cfg.WGS84,
        resampling=Resampling.bilinear if method == "bilinear" else Resampling.nearest,
    )
    return dst_array


# ============================================================
# 地形因子提取
# ============================================================
def compute_slope(dem, transform):
    """Horn(1981)坡度 (度)"""
    # 度->米转换
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float64), cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad)


def compute_aspect(dem, transform):
    """坡向: 返回 (cos, sin) 两个分量"""
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float64), cellsize_y, cellsize_x)
    aspect_rad = np.arctan2(-dy, dx)
    return np.cos(aspect_rad), np.sin(aspect_rad)


def compute_tri(dem, transform):
    """地形粗糙度指数(TRI): 3x3窗口内中心像元与邻域均值差的绝对值"""
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    mean_neighbors = uniform_filter(dem_filled, size=3)
    mean_neighbors = (9 * mean_neighbors - dem_filled) / 8
    return np.abs(dem_filled - mean_neighbors)


def compute_tpi(dem, transform, window_meters=300):
    """地形位置指数: 中心 - 邻域均值"""
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    window_pixels = max(3, int(window_meters / cellsize_m))
    if window_pixels % 2 == 0:
        window_pixels += 1

    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    mean_dem = uniform_filter(dem_filled, size=window_pixels)
    tpi = dem_filled - mean_dem
    return tpi


def compute_curvature(dem, transform):
    """计算剖面曲率和平面曲率"""
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG

    dy, dx = np.gradient(dem.astype(np.float64), cellsize_y, cellsize_x)
    dyy, dyx = np.gradient(dy, cellsize_y, cellsize_x)
    dxy, dxx = np.gradient(dx, cellsize_y, cellsize_x)

    p = dx ** 2 + dy ** 2
    p_safe = np.where(p > 1e-6, p, 1e-6)

    # 剖面曲率 (沿坡向)
    profile_curv = -(dxx * dx ** 2 + 2 * dxy * dx * dy + dyy * dy ** 2) / (p_safe ** 1.5)
    # 平面曲率 (沿等高线)
    plan_curv = -(dxx * dy ** 2 - 2 * dxy * dx * dy + dyy * dx ** 2) / (p_safe ** 1.5)

    return profile_curv, plan_curv


def compute_roughness(dem, transform, window=9):
    """地形粗糙度: 局部标准差(米)"""
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    mean = uniform_filter(dem_filled, size=window)
    mean_sq = uniform_filter(dem_filled ** 2, size=window)
    variance = mean_sq - mean ** 2
    variance = np.clip(variance, 0, None)
    return np.sqrt(variance)


def derive_terrain_factors(dem, transform):
    """提取所有地形因子"""
    print("[Phase2] 提取地形因子...")

    # 处理NaN
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

    factors["tpi"] = compute_tpi(dem_filled, transform, cfg.TPI_WINDOW)
    print("  - TPI完成")

    profile_curv, plan_curv = compute_curvature(dem_filled, transform)
    factors["profile_curvature"] = profile_curv
    factors["plan_curvature"] = plan_curv
    print("  - 曲率完成")

    factors["roughness"] = compute_roughness(dem_filled, transform, cfg.ROUGHNESS_WINDOW)
    print("  - 粗糙度完成")

    return factors


# ============================================================
# 约束掩膜生成
# ============================================================
def generate_hard_mask(data, transform, shape):
    """
    生成硬约束掩膜 (0=禁止建设, 1=允许)
    禁止区域: 坡度>35°, 水域缓冲区, 自然保护区
    """
    print("[Phase2] 生成硬约束掩膜...")
    hard_mask = np.ones(shape, dtype=np.uint8)
    pixel_deg = abs(transform.a)

    # 1. 坡度约束
    slope = data.get("slope")
    if slope is not None:
        slope_aligned = resample_to_target(slope, transform, transform, shape)
        hard_mask[slope_aligned > cfg.MAX_SLOPE] = 0
        print(f"  坡度>35°禁止: {(slope_aligned > cfg.MAX_SLOPE).sum()} 像元")

    # 2. 水域缓冲区
    dem = data.get("dem")
    water_gdf = data.get("osm_water")
    if water_gdf is not None and len(water_gdf) > 0:
        water_raster = vector_to_raster(water_gdf, transform, shape)
        buffer_deg = cfg.meters_to_deg(cfg.WATER_BUFFER)
        buffer_pixels = max(1, int(buffer_deg / pixel_deg))
        water_buffer = binary_dilation(water_raster > 0, iterations=buffer_pixels)
        hard_mask[water_buffer] = 0
        print(f"  水域缓冲区禁止: {water_buffer.sum()} 像元")
    else:
        print("  水域数据缺失, 跳过水域约束")

    # 3. 自然保护区
    protected_gdf = data.get("osm_protected")
    if protected_gdf is not None and len(protected_gdf) > 0:
        protected_raster = vector_to_raster(protected_gdf, transform, shape)
        hard_mask[protected_raster > 0] = 0
        print(f"  自然保护区禁止: {(protected_raster > 0).sum()} 像元")
    else:
        print("  保护区数据缺失, 跳过")

    # 4. 高海拔(>3500m)视为不可建设
    dem_aligned = data.get("dem")
    if dem_aligned is not None:
        dem_r = resample_to_target(dem_aligned, transform, transform, shape)
        hard_mask[dem_r > 3500] = 0

    print(f"  硬约束禁止区域: {(hard_mask == 0).sum()} 像元 ({(hard_mask == 0).sum() / max(hard_mask.size, 1) * 100:.1f}%)")
    return hard_mask


def generate_soft_mask(data, transform, shape):
    """
    生成软约束掩膜 (0-1, 值越高成本越高)
    """
    print("[Phase2] 生成软约束掩膜...")
    soft_mask = np.ones(shape, dtype=np.float32)

    # 土地利用软约束
    landuse_gdf = data.get("osm_landuse")
    if landuse_gdf is not None and len(landuse_gdf) > 0:
        # 根据landuse标签赋予权值
        landuse_map = {
            "forest": 1, "wood": 1, "scrub": 1, "grassland": 2,
            "farmland": 2, "meadow": 2, "orchard": 2, "vineyard": 2,
            "residential": 4, "urban": 4, "commercial": 5, "industrial": 5,
            "bare_rock": 3, "quarry": 5, "brownfield": 5, "landfill": 5,
            "wetland": 7, "salt_pond": 7, "beach": 3, "sand": 3,
        }

        for _, row in landuse_gdf.iterrows():
            tags = row.get("tags", {}) if hasattr(row, "tags") else {}
            landuse_val = tags.get("landuse", tags.get("natural", "")).lower()
            code = landuse_map.get(landuse_val, 8)
            cost_factor = cfg.LANDUSE_SOFT_COST.get(code, 0.5)

            try:
                geom = row.geometry
                shapes = [(geom, cost_factor)]
                mask_part = features.rasterize(shapes, out_shape=shape, transform=transform,
                                               fill=1.0, all_touched=True, dtype=np.float32)
                soft_mask = np.minimum(soft_mask, mask_part)
            except Exception:
                continue
        print(f"  土地利用软约束已应用")

    # 如果没有土地利用数据, 基于DEM坡度生成代理
    else:
        dem = data.get("dem")
        if dem is not None:
            slope = compute_slope(np.nan_to_num(dem, nan=0), transform)
            slope_aligned = resample_to_target(slope, transform, transform, shape)
            # 坡度越大, 成本越高
            soft_mask = 1.0 - 0.7 * np.clip(slope_aligned / cfg.MAX_SLOPE, 0, 1)
            print(f"  基于坡度的软约束代理已应用")

    print(f"  软约束完成")
    return soft_mask


# ============================================================
# 距离栅格计算
# ============================================================
def compute_distance_raster(gdf, transform, shape, max_dist_m=5000):
    """计算到矢量的欧氏距离栅格(米)"""
    if gdf is None or len(gdf) == 0:
        return np.full(shape, float(max_dist_m), dtype=np.float32)

    raster = vector_to_raster(gdf, transform, shape, burn_value=1)
    if raster.sum() == 0:
        return np.full(shape, float(max_dist_m), dtype=np.float32)

    # EDT返回像元距离, 转为米
    cellsize_deg = abs(transform.a)
    cellsize_m = cellsize_deg * cfg.METERS_PER_DEG
    dist_pixels = distance_transform_edt(1 - raster)
    dist_m = dist_pixels * cellsize_m
    dist_m = np.clip(dist_m, 0, max_dist_m)
    return dist_m.astype(np.float32)


# ============================================================
# 栅格对齐
# ============================================================
def _landuse_tag_to_code(tags):
    """将OSM landuse标签映射为分类编码"""
    if not tags:
        return 8  # other/unknown
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
    """
    将OSM土地利用GeoDataFrame转为分类栅格(1-8)
    各类别: 1=森林, 2=农田, 3=裸地, 4=居民区, 5=工业区, 6=水域, 7=湿地, 8=其他
    使用批量处理: 按类别分组后一次性栅格化, 大幅提速
    """
    if gdf is None or len(gdf) == 0:
        return np.full(shape, 8, dtype=np.uint8)

    from rasterio import features
    from collections import defaultdict

    # 按编码分组几何
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

    # 批量栅格化 (按类别优先级: 低编号先填, 高编号覆盖)
    result = np.zeros(shape, dtype=np.uint8)
    for code in sorted(groups.keys()):
        shapes_list = [(g, code) for g in groups[code] if g is not None]
        if not shapes_list:
            continue
        mask = features.rasterize(
            shapes_list, out_shape=shape, transform=transform,
            fill=0, all_touched=True, dtype=np.uint8
        )
        result = np.where(mask == code, code, result)
        print(f"    类别{code}: {len(shapes_list)}个面")

    # 未分配像元设为默认类别8(其他)
    result[result == 0] = 8
    return result


def rasterize_building_density(gdf, transform, shape):
    """
    计算建筑密度(栋/km²)
    """
    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, dtype=np.float32)

    # 将建筑点转为二值栅格
    building_raster = vector_to_raster(gdf, transform, shape, burn_value=1, all_touched=True)

    # 使用高斯核密度估计
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    sigma_pixels = max(2, int(500 / cellsize_m))  # 500m带宽
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(building_raster.astype(np.float32), sigma=sigma_pixels)
    # 转为栋/km²
    cell_area_km2 = (cellsize_m / 1000) ** 2
    density = density / cell_area_km2

    return density.astype(np.float32)


def align_all_rasters(data, reference_dem, ref_transform, osm_data=None):
    """
    将所有栅格对齐至统一的90m分辨率参考网格
    osm_data: dict with 'osm_landuse', 'osm_buildings' GeoDataFrames
    """
    print("[Phase2] 栅格对齐至统一分辨率...")

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

    layers_to_align = [
        "dem", "slope", "aspect_cos", "aspect_sin", "tri", "tpi",
        "profile_curvature", "plan_curvature", "roughness",
        "typhoon_risk", "seismic_risk", "landslide_risk",
    ]

    for name in layers_to_align:
        arr = data.get(name)
        if arr is not None:
            aligned[name] = resample_to_target(arr, ref_transform, dst_transform, (dst_height, dst_width))
            print(f"  对齐: {name}")

    # 距离栅格
    print("  计算距离栅格...")
    roads = data.get("osm_roads")
    aligned["dist_road"] = compute_distance_raster(roads, dst_transform, (dst_height, dst_width))

    water = data.get("osm_water")
    aligned["dist_water"] = compute_distance_raster(water, dst_transform, (dst_height, dst_width))

    lines = data.get("taiwan_lines")
    aligned["dist_existing_line"] = compute_distance_raster(lines, dst_transform, (dst_height, dst_width), max_dist_m=10000)

    railways = data.get("osm_railways")
    aligned["dist_railway"] = compute_distance_raster(railways, dst_transform, (dst_height, dst_width), max_dist_m=5000)

    # 机场数据传递(在hard_mask中使用)
    if osm_data:
        airports_gdf = osm_data.get("osm_airports")
        if airports_gdf is not None and len(airports_gdf) > 0:
            aligned["airport_raster"] = vector_to_raster(airports_gdf, dst_transform, (dst_height, dst_width))
            print(f"  机场栅格化: {(aligned['airport_raster'] > 0).sum()} 像元")

    # 土地利用分类栅格
    if osm_data:
        landuse_gdf = osm_data.get("osm_landuse")
        if landuse_gdf is not None and len(landuse_gdf) > 0:
            print("  栅格化土地利用分类...")
            aligned["landuse_code"] = rasterize_landuse(landuse_gdf, dst_transform, (dst_height, dst_width))
            print(f"    分类分布: {np.bincount(aligned['landuse_code'].ravel(), minlength=9)[1:]}")

        buildings_gdf = osm_data.get("osm_buildings")
        if buildings_gdf is not None and len(buildings_gdf) > 0:
            print("  计算建筑密度...")
            aligned["building_density"] = rasterize_building_density(buildings_gdf, dst_transform, (dst_height, dst_width))
            vals = aligned["building_density"][aligned["building_density"] > 0]
            if len(vals) > 0:
                print(f"    密度范围: {vals.min():.0f} - {vals.max():.0f} 栋/km^2")

    print("[Phase2] 栅格对齐完成\n")
    return aligned
