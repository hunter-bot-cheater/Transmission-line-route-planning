"""
模块1: 数据获取
- 加载台湾DEM GeoTIFF
- 读取中国输电线SHP并空间过滤至台湾
- Overpass API下载OSM数据(道路/水域/自然保护区/土地利用)
- 生成气象风险代理层(台风/地震/滑坡)
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
import json
import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import requests
from scipy.ndimage import gaussian_filter, distance_transform_edt, sobel
from scipy.spatial import KDTree

import config as cfg


# ============================================================
# DEM 重采样辅助函数
# ============================================================
def _resample_dem_to_standard(dem, src_transform, src_crs):
    """将任意分辨率/CRS的DEM重采样到WGS84 90m标准网格"""
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.transform import from_origin

    target_crs = "EPSG:4326"
    target_res_deg = cfg.BASE_RESOLUTION / cfg.METERS_PER_DEG
    bbox = cfg.TAIWAN_BBOX

    target_width = int((bbox[2] - bbox[0]) / target_res_deg)
    target_height = int((bbox[3] - bbox[1]) / target_res_deg)
    # from_origin y_step应为负值(north-up栅格)
    target_transform = from_origin(bbox[0], bbox[3], target_res_deg, -target_res_deg)
    print(f"  DEM重采样目标: {target_width}x{target_height} (90m WGS84)")

    if "WGS84" not in str(src_crs).upper() and "4326" not in str(src_crs):
        dem_bounds = transform_bounds(
            src_crs, target_crs,
            src_transform.c,
            src_transform.f + src_transform.e * dem.shape[0],
            src_transform.c + src_transform.a * dem.shape[1],
            src_transform.f,
        )
    else:
        dem_bounds = (
            src_transform.c,
            src_transform.f + src_transform.e * dem.shape[0],
            src_transform.c + src_transform.a * dem.shape[1],
            src_transform.f,
        )

    dem_wgs = np.zeros((target_height, target_width), dtype=np.float32)
    reproject(
        dem.astype(np.float32), dem_wgs,
        src_transform=src_transform, dst_transform=target_transform,
        src_crs=src_crs, dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    print(f"  DEM重采样: {dem.shape}({src_crs}) → {dem_wgs.shape}(WGS84 90m)")
    return dem_wgs, target_transform, target_crs


# ============================================================
# DEM 加载
# ============================================================
def load_dem():
    """加载台湾DEM，返回 (array, transform, crs, meta)"""
    print("[Phase1] 加载台湾DEM...")
    with rasterio.open(cfg.DEM_PATH) as src:
        # 计算台湾范围对应的窗口
        bbox = cfg.TAIWAN_BBOX
        # rasterio.warp.transform_bounds: WGS84 -> DEM CRS
        from rasterio.warp import transform_bounds
        if src.crs and src.crs.to_string() != cfg.WGS84:
            dem_bounds = transform_bounds(cfg.WGS84, src.crs, *bbox)
        else:
            dem_bounds = bbox

        window = src.window(*dem_bounds)
        window = window.round_offsets().round_lengths()
        # 确保window不越界
        window = Window(
            max(0, int(window.col_off)), max(0, int(window.row_off)),
            min(src.width - int(window.col_off), int(window.width)),
            min(src.height - int(window.row_off), int(window.height))
        )

        dem = src.read(1, window=window).astype(np.float32)
        transform = src.window_transform(window)
        meta = src.meta.copy()
        meta.update({"transform": transform, "width": window.width, "height": window.height})

        # 处理NoData
        dem[dem <= -9999] = np.nan
        dem[dem > 9000] = np.nan

        print(f"  DEM加载完成: {dem.shape}, 范围: {dem_bounds}")
        return dem, transform, str(src.crs), meta


# ============================================================
# 输电线数据
# ============================================================
def load_taiwan_lines():
    """加载中国输电线SHP，过滤至台湾范围"""
    print("[Phase1] 加载输电线数据...")
    gdf = gpd.read_file(cfg.SHP_PATH)
    print(f"  原始输电线: {len(gdf)} 条")

    # 空间过滤至台湾
    bbox = cfg.TAIWAN_BBOX
    taiwan_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    gdf = gdf[gdf.geometry.intersects(taiwan_poly)].copy()

    # 裁剪几何至台湾bbox
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g.intersection(taiwan_poly) if g else None)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # 确保WGS84
    if gdf.crs and gdf.crs.to_string() != cfg.WGS84:
        gdf = gdf.to_crs(cfg.WGS84)

    print(f"  台湾地区输电线: {len(gdf)} 条")
    return gdf


# ============================================================
# Overpass API 数据下载
# ============================================================
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_cache_dir = cfg.DOWNLOADED_DIR


def _cache_path(name):
    return _cache_dir / f"{name}.pkl"


def _cache_valid(path, days=cfg.OSM_CACHE_DAYS):
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=days)


def _overpass_query(query, cache_name):
    """执行Overpass API查询，带缓存"""
    cache_path = _cache_path(cache_name)
    if _cache_valid(cache_path):
        print(f"  使用缓存: {cache_name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  下载OSM数据: {cache_name} ...")
    try:
        # 速率限制
        time.sleep(2)
        headers = {
            "Accept": "application/json",
            "User-Agent": "TaiwanTransmissionPlanning/1.0",
        }
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        return data
    except Exception as e:
        print(f"  Overpass查询失败({cache_name}): {e}")
        # 尝试使用缓存(即使过期)
        if cache_path.exists():
            print(f"  使用过期缓存: {cache_name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return {"elements": []}


def _osm_to_gdf(elements, geom_type="line"):
    """将OSM elements转为GeoDataFrame"""
    from shapely.geometry import Point, LineString, Polygon, MultiLineString

    records = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        geometry = el.get("geometry")
        if not geometry:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geometry]
        try:
            if geom_type == "point":
                if len(coords) >= 1:
                    records.append({"geometry": Point(coords[0]), "tags": tags})
            elif geom_type == "polygon":
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    records.append({"geometry": Polygon(coords), "tags": tags})
            else:
                if len(coords) >= 2:
                    records.append({"geometry": LineString(coords), "tags": tags})
        except Exception:
            continue

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=cfg.WGS84)
    return gdf


def fetch_osm_roads(bbox):
    """下载台湾道路网"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"="motorway"]({bbox_str});
      way["highway"="trunk"]({bbox_str});
      way["highway"="primary"]({bbox_str});
      way["highway"="secondary"]({bbox_str});
      way["highway"="tertiary"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_roads")
    return _osm_to_gdf(data.get("elements", []), "line")


def fetch_osm_water(bbox):
    """下载台湾水域(河流+湖泊)"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["waterway"="river"]({bbox_str});
      way["waterway"="stream"]({bbox_str});
      way["natural"="water"]({bbox_str});
      way["water"="lake"]({bbox_str});
      way["water"="reservoir"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_water")
    return _osm_to_gdf(data.get("elements", []), "line")


def fetch_osm_protected_areas(bbox):
    """下载自然保护区/国家公园"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["boundary"="national_park"]({bbox_str});
      way["boundary"="protected_area"]({bbox_str});
      way["leisure"="nature_reserve"]({bbox_str});
      relation["boundary"="national_park"]({bbox_str});
      relation["boundary"="protected_area"]({bbox_str});
      relation["leisure"="nature_reserve"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_protected")
    elements = data.get("elements", [])
    ways = [e for e in elements if e.get("type") == "way"]
    return _osm_to_gdf(ways, "polygon")


def fetch_osm_landuse(bbox):
    """下载土地利用数据"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["landuse"]({bbox_str});
      way["natural"="wood"]({bbox_str});
      way["natural"="scrub"]({bbox_str});
      way["natural"="grassland"]({bbox_str});
      way["natural"="bare_rock"]({bbox_str});
      way["natural"="wetland"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_landuse")
    return _osm_to_gdf(data.get("elements", []), "polygon")


def fetch_osm_buildings(bbox):
    """下载建筑数据(用于密度代理)"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["building"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_buildings")
    return _osm_to_gdf(data.get("elements", []), "polygon")


def fetch_osm_railways(bbox):
    """下载铁路网(高铁+台铁+捷运)"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["railway"="rail"]({bbox_str});
      way["railway"="high_speed"]({bbox_str});
      way["railway"="subway"]({bbox_str});
      way["railway"="light_rail"]({bbox_str});
      way["railway"="narrow_gauge"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_railways")
    return _osm_to_gdf(data.get("elements", []), "line")


def fetch_osm_airports(bbox):
    """下载机场/飞行区(仅跑道和大型机场, 不含小型直升机坪)"""
    bbox_str = f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}"
    query = f"""
    [out:json][timeout:180];
    (
      way["aeroway"="aerodrome"]({bbox_str});
      way["aeroway"="runway"]({bbox_str});
      relation["aeroway"="aerodrome"]({bbox_str});
    );
    out geom;
    """
    data = _overpass_query(query, "taiwan_airports")
    return _osm_to_gdf(data.get("elements", []), "polygon")


# ============================================================
# 气象/灾害风险代理层
# ============================================================
def build_typhoon_risk(dem, transform, crs):
    """
    台风风险代理: 基于地形暴露度
    东南坡向(45°-225°)+低高程+陡坡 = 高台风暴露
    """
    print("[Phase1] 生成台风风险层...")
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG

    dy, dx = np.gradient(dem.astype(np.float64), cellsize_y, cellsize_x)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect_rad = np.arctan2(-dy, dx)  # 地理坡向
    aspect_deg = np.degrees(aspect_rad) % 360

    valid = ~np.isnan(dem)
    slope_norm = np.zeros_like(dem)
    aspect_score = np.zeros_like(dem)

    slope_norm[valid] = np.clip(np.degrees(slope_rad[valid]) / 45, 0, 1)
    # 东南坡向(45-225度)得分最高
    mask = valid & (aspect_deg >= 45) & (aspect_deg < 225)
    aspect_score[mask] = 1.0
    mask2 = valid & ~mask
    aspect_score[mask2] = 0.3

    elev_norm = np.zeros_like(dem)
    elev_norm[valid] = 1.0 - np.clip(dem[valid] / 3000, 0, 1)

    risk = 0.4 * slope_norm + 0.35 * aspect_score + 0.25 * elev_norm
    risk = gaussian_filter(risk, sigma=2)
    risk[~valid] = np.nan
    print(f"  台风风险层完成")
    return risk


def build_seismic_risk(dem, transform, crs):
    """
    地震风险代理: 基于坡度和地形破碎度
    台湾全岛地震风险均较高,地形陡峭区风险更大
    """
    print("[Phase1] 生成地震风险层...")
    cellsize_m = abs(transform.a) * cfg.METERS_PER_DEG
    dy, dx = np.gradient(dem.astype(np.float64), cellsize_m, cellsize_m)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    valid = ~np.isnan(dem)
    base_risk = np.full_like(dem, 0.6)

    slope_factor = np.zeros_like(dem)
    slope_factor[valid] = np.clip(slope[valid] / 40, 0, 1)
    risk = base_risk + 0.3 * slope_factor
    risk = np.clip(risk, 0, 1)
    risk[~valid] = np.nan
    print(f"  地震风险层完成")
    return risk


def build_landslide_risk(dem, transform, crs):
    """
    滑坡风险: 基于坡度+曲率+地形粗糙度组合
    """
    print("[Phase1] 生成滑坡风险层...")
    cellsize_y = abs(transform.e) * cfg.METERS_PER_DEG
    cellsize_x = abs(transform.a) * cfg.METERS_PER_DEG

    dem_f32 = dem.astype(np.float32)
    dy, dx = np.gradient(dem_f32, cellsize_y, cellsize_x)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    dyy, dyx = np.gradient(dy, cellsize_y, cellsize_x)
    dxy, dxx = np.gradient(dx, cellsize_y, cellsize_x)
    p = dx ** 2 + dy ** 2
    p_safe = np.where(p > 1e-6, p, 1e-6)
    profile_curv = -(dxx * dx ** 2 + 2 * dxy * dx * dy + dyy * dy ** 2) / (p_safe ** 1.5)

    # TRI - 3x3窗口中心与邻域差
    from scipy.ndimage import uniform_filter
    dem_filled = np.nan_to_num(dem, nan=np.nanmean(dem))
    mean_neighbors = (9 * uniform_filter(dem_filled, size=3) - dem_filled) / 8
    tri = np.abs(dem_filled - mean_neighbors)

    valid = ~np.isnan(dem)
    slope_score = np.zeros_like(dem)
    curv_score = np.zeros_like(dem)
    tri_score = np.zeros_like(dem)

    slope_score[valid] = np.clip((slope[valid] - 15) / 30, 0, 1)
    curv_score[valid] = np.clip((profile_curv[valid] + 0.01) / 0.02, 0, 1)
    tri_score[valid] = np.clip(tri[valid] / 200, 0, 1)

    risk = 0.45 * slope_score + 0.30 * tri_score + 0.25 * curv_score
    risk = gaussian_filter(risk, sigma=2)
    risk[~valid] = np.nan
    print(f"  滑坡风险层完成")
    return risk


# ============================================================
# 综合获取所有数据
# ============================================================
def acquire_all():
    """获取所有原始数据，返回字典"""
    result = {}

    # 1. DEM — 加载后立即重采样到WGS84 90m标准网格
    dem_raw, transform_raw, crs_raw, meta = load_dem()
    dem, transform, crs = _resample_dem_to_standard(dem_raw, transform_raw, crs_raw)
    result["dem"] = dem
    result["dem_transform"] = transform
    result["dem_crs"] = crs
    result["dem_meta"] = meta

    # 2. 输电线
    result["taiwan_lines"] = load_taiwan_lines()

    # 3. OSM数据
    bbox = cfg.TAIWAN_BBOX
    try:
        result["osm_roads"] = fetch_osm_roads(bbox)
        print(f"  OSM道路: {len(result['osm_roads'])} 条")
    except Exception as e:
        print(f"  OSM道路获取失败: {e}")
        result["osm_roads"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    try:
        result["osm_water"] = fetch_osm_water(bbox)
        print(f"  OSM水域: {len(result['osm_water'])} 条")
    except Exception as e:
        print(f"  OSM水域获取失败: {e}")
        result["osm_water"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    try:
        result["osm_protected"] = fetch_osm_protected_areas(bbox)
        print(f"  OSM保护区: {len(result['osm_protected'])} 个")
    except Exception as e:
        print(f"  OSM保护区获取失败: {e}")
        result["osm_protected"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    try:
        result["osm_landuse"] = fetch_osm_landuse(bbox)
        print(f"  OSM土地利用: {len(result['osm_landuse'])} 个")
    except Exception as e:
        print(f"  OSM土地利用获取失败: {e}")
        result["osm_landuse"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    try:
        result["osm_buildings"] = fetch_osm_buildings(bbox)
        print(f"  OSM建筑: {len(result['osm_buildings'])} 个")
    except Exception as e:
        print(f"  OSM建筑获取失败: {e}")
        result["osm_buildings"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    try:
        result["osm_railways"] = fetch_osm_railways(bbox)
        print(f"  OSM铁路: {len(result['osm_railways'])} 条")
    except Exception as e:
        print(f"  OSM铁路获取失败: {e}")
        result["osm_railways"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    try:
        result["osm_airports"] = fetch_osm_airports(bbox)
        print(f"  OSM机场: {len(result['osm_airports'])} 个")
    except Exception as e:
        print(f"  OSM机场获取失败: {e}")
        result["osm_airports"] = gpd.GeoDataFrame(geometry=[], crs=cfg.WGS84)

    # 4. 风险代理层
    result["typhoon_risk"] = build_typhoon_risk(dem, transform, crs)
    result["seismic_risk"] = build_seismic_risk(dem, transform, crs)
    result["landslide_risk"] = build_landslide_risk(dem, transform, crs)

    print("[Phase1] 数据获取完成\n")
    return result
