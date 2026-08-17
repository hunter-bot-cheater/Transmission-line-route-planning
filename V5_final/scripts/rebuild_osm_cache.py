#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成 v4 / v3 共用的台湾 OSM 缓存 pkl。
问题: 现有 data/downloaded/taiwan_landuse.pkl 等文件包含中国大陆坐标,
      运行时 bbox 过滤后为空, 导致土地利用等波段恒为 0。
做法: 通过 shared.data_acquisition 以正确 TAIWAN_BBOX 重新拉取并保存为
      v4 期望的 raw Overpass dict 格式, 覆盖旧缓存。
"""
import sys
import types
import pickle
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "downloaded"

TAIWAN_BBOX = (120.0, 21.9, 122.0, 25.4)  # (min_lon, min_lat, max_lon, max_lat)
METERS_PER_DEG = 111320.0


def _import_shared():
    """注入 config shim 后 import shared.data_acquisition。"""
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    cfg = types.SimpleNamespace(
        TAIWAN_BBOX=TAIWAN_BBOX,
        DOWNLOADED_DIR=DATA_DIR,
        OSM_CACHE_DAYS=9999,  # 强制优先读本地缓存, 但重抓时忽略它
        METERS_PER_DEG=METERS_PER_DEG,
        WGS84="EPSG:4326",
    )
    if "config" in sys.modules:
        old = sys.modules["config"]
    else:
        old = None
    sys.modules["config"] = cfg
    try:
        import shared.data_acquisition as shared
        return shared, old
    finally:
        if old is not None:
            sys.modules["config"] = old


def _gdf_to_overpass_dict(gdf):
    """把 GeoDataFrame 转成 v4 load_osm_geoms 能解析的 Overpass raw dict。

    v4 的 load_osm_geoms 只看 elements 列表中每个 el 的:
      - type (way|relation)
      - geometry (way 点列表) 或 members (relation 成员)
      - tags (属性字典)
    这里统一输出 way 元素, geometry 为经纬度坐标序列。
    """
    elements = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        tags = {k: str(v) for k, v in row.items() if k != "geometry" and v is not None and str(v) not in ("nan", "None", "")}
        if geom is None or geom.is_empty:
            continue
        # 统一用多边形/线坐标序列
        coords = []
        try:
            if geom.geom_type == "Polygon":
                ring = list(geom.exterior.coords)
            elif geom.geom_type in ("MultiPolygon",):
                # 取最大 polygon 的外环, 够用即可
                largest = max(geom.geoms, key=lambda g: g.area)
                ring = list(largest.exterior.coords)
            elif geom.geom_type in ("LineString", "MultiLineString"):
                # 线元素: 取最长分支
                if geom.geom_type == "LineString":
                    ring = list(geom.coords)
                else:
                    longest = max(geom.geoms, key=lambda g: g.length)
                    ring = list(longest.coords)
            else:
                continue
        except Exception:
            continue
        if len(ring) < 2:
            continue
        # Overpass 风格: lat/lon 键
        geometry = [{"lat": float(y), "lon": float(x)} for (x, y) in ring]
        elements.append({"type": "way", "geometry": geometry, "tags": tags})
    return {"elements": elements, "generator": "rebuild_osm_cache (shared)"}


def _bbox_filter(d, bbox):
    """粗略过滤: 若 element 任一点在台湾 bbox 内则保留。"""
    min_lon, min_lat, max_lon, max_lat = bbox
    keep = []
    for el in d.get("elements", []):
        geom = el.get("geometry") or []
        ok = False
        for p in geom:
            lat = p.get("lat"); lon = p.get("lon")
            if lat is None or lon is None:
                continue
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                ok = True
                break
        if ok:
            keep.append(el)
    d["elements"] = keep
    return d


def rebuild():
    shared, _ = _import_shared()
    print("=" * 60)
    print("开始重建台湾 OSM 缓存 (shared.data_acquisition)")
    print("TAIWAN_BBOX =", TAIWAN_BBOX)
    print("输出目录:", DATA_DIR)
    print("=" * 60)

    fetchers = {
        "taiwan_roads.pkl": "fetch_osm_roads",
        "taiwan_water.pkl": "fetch_osm_water",
        "taiwan_landuse.pkl": "fetch_osm_landuse",
        "taiwan_buildings.pkl": "fetch_osm_buildings",
        "taiwan_railways.pkl": "fetch_osm_railways",
        "taiwan_protected.pkl": "fetch_osm_protected",
        "taiwan_airports.pkl": "fetch_osm_airports",
        "taiwan_faults.pkl": "fetch_osm_faults",
        "taiwan_vegetation.pkl": "fetch_osm_vegetation",
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pkl_name, fetcher_name in fetchers.items():
        fetch_fn = getattr(shared, fetcher_name, None)
        if fetch_fn is None:
            print(f"[SKIP] shared.{fetcher_name} 不存在, 跳过 {pkl_name}")
            continue
        out = DATA_DIR / pkl_name
        try:
            print(f"\n[FETCH] {fetcher_name}() -> {pkl_name}")
            gdf = fetch_fn(TAIWAN_BBOX)
            count = len(gdf) if gdf is not None else 0
            print(f"  原始记录数: {count}")
            d = _gdf_to_overpass_dict(gdf)
            print(f"  转换 way 数: {len(d['elements'])}")
            d = _bbox_filter(d, TAIWAN_BBOX)
            print(f"  bbox 过滤后: {len(d['elements'])}")
            with open(out, "wb") as f:
                pickle.dump(d, f)
            print(f"  [SAVED] {out}")
        except Exception as e:
            print(f"  [ERR] {pkl_name} 失败: {e}")

    print("\n" + "=" * 60)
    print("重建完成。下次运行 v4 / v3 时将使用新的台湾 OSM 缓存。")
    print("=" * 60)


if __name__ == "__main__":
    rebuild()
