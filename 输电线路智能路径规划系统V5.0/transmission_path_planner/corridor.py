# -*- coding: utf-8 -*-
"""corridor.py — 输电走廊先验模块(V6)

从全网既有线路(默认 SHP)构造两类先验场, 均支持留一法排除待规划线:
  1) dist_existing: 到最近既有线路的米级距离场 + 方向场(用于 CostUNet 通道15 与 VIN 引导)
  2) regional_density: 高斯平滑的区域走廊密度场(反映"本区域工程师习惯把线放哪")
"""
from __future__ import annotations
import pickle
import threading
from pathlib import Path

import numpy as np


def pickle_to_np(obj):
    """任意 Python 对象 -> np.uint8 数组(可存入 npz)。"""
    return np.frombuffer(pickle.dumps(obj, protocol=4), dtype=np.uint8)


def pickle_from_np(arr):
    """np.uint8 数组 -> 原对象。"""
    return pickle.loads(arr.tobytes())


def cache_load_real_geo(arr):
    """兼容新旧缓存格式读取 real_geo, 统一为 折线列表, 每条 = [(lon,lat),...]。

    旧格式: np.array([real_geo], dtype=object) 可能落成 (1,1,N,2) 数值数组 或 (1,n) 对象数组;
    新格式: pickle 字节。
    """
    if arr.dtype == object:
        if arr.ndim == 1:
            val = arr[0]
        else:
            val = arr[0]  # (1,1,N,2) -> (1,N,2); (1,n) -> (n,)
    else:
        try:
            val = pickle.loads(arr.tobytes())
        except Exception:
            val = arr
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, tuple):
        val = [val]
    out = []
    for poly in val:
        if isinstance(poly, np.ndarray):
            poly = poly.tolist()
        out.append([(float(x), float(y)) for x, y in poly])
    return out

_network_lock = threading.Lock()
_NETWORK_CACHE = {}  # path -> (coords_tuple_list, geom_list)


def load_network_lines(shp_path):
    """加载 SHP 中全部 LineString(将 MultiLineString 拆开), 进程级缓存。"""
    with _network_lock:
        if shp_path in _NETWORK_CACHE:
            return _NETWORK_CACHE[shp_path]
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    coords_list, geom_list = [], []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords_list.append(tuple(geom.coords))
            geom_list.append(geom)
        elif geom.geom_type == "MultiLineString":
            for sub in geom.geoms:
                coords_list.append(tuple(sub.coords))
                geom_list.append(sub)
    with _network_lock:
        _NETWORK_CACHE[shp_path] = (coords_list, geom_list)
    return coords_list, geom_list


def _norm(c):
    return tuple((round(float(x), 6), round(float(y), 6)) for x, y in c)


def match_exclude_index(real_geo_coords, coords_list):
    """按坐标序列精确匹配待规划线在全网中的索引(留一排除); 支持反向。"""
    target = _norm(real_geo_coords)
    rev = tuple(reversed(target))
    for i, c in enumerate(coords_list):
        nc = _norm(c)
        if nc == target or nc == rev:
            return i
    return None


def corridor_dist_band(coords_list, geom_list, exclude_idx, transform, H, W,
                       max_dist_m=12000.0):
    """栅格化除 exclude_idx 外全部线路, 返回(到最近线路米距离场, 方向场(r,c))。"""
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import mapping
    from scipy.ndimage import distance_transform_edt
    from .geometry import pixel_meters

    if exclude_idx is not None:
        geoms = [g for i, g in enumerate(geom_list) if i != exclude_idx]
    else:
        geoms = geom_list
    mask = rio_rasterize([(mapping(g), 1) for g in geoms], out_shape=(H, W),
                         transform=transform, fill=0, dtype="uint8", all_touched=False)
    dist_px, idx = distance_transform_edt(1 - mask, return_indices=True)
    lat0 = (transform.f + transform.e * H) / 2.0
    pm = pixel_meters(transform, lat0)
    dist_m = np.minimum(dist_px * pm, max_dist_m).astype(np.float64)
    ir, ic = idx
    rr = np.arange(H)[:, None]
    cc = np.arange(W)[None, :]
    ar = (ir - rr).astype(np.float64)
    ac = (ic - cc).astype(np.float64)
    mag = np.hypot(ar, ac) + 1e-8
    ar /= mag
    ac /= mag
    return dist_m, (ar, ac), pm


def regional_density_band(geom_list, exclude_idx, transform, H, W, sigma_px=12.0):
    """区域走廊密度: 栅格化既有线路(留一)后高斯平滑, 归一化到 [0,1]。

    sigma_px 建议 8~25 像元(对应约 1.5~5km @150m/px), 表达'区域走廊偏好'。
    """
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import mapping
    from scipy.ndimage import gaussian_filter

    geoms = [g for i, g in enumerate(geom_list) if i != exclude_idx] if exclude_idx is not None else geom_list
    mask = rio_rasterize([(mapping(g), 1) for g in geoms], out_shape=(H, W),
                         transform=transform, fill=0, dtype="float32", all_touched=False)
    dens = gaussian_filter(mask, sigma=sigma_px)
    mx = dens.max()
    if mx <= 0:
        return np.zeros((H, W), np.float32)
    return (dens / mx).astype(np.float32)

def corridor_orientation_band(geom_list, exclude_idx, transform, H, W, win=5):
    """每像元'最近既有线路段'的局部方向(弧度, 模π∈[0,π))。

    思路: 线像元上按窗口PCA取主轴方向, 再用距离变换把方向传播到全图。
    无线路像元(孤立区)返回 NaN。
    """
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import mapping
    from scipy.ndimage import distance_transform_edt
    from numpy.linalg import eig
    import math

    geoms = [g for i, g in enumerate(geom_list) if i != exclude_idx] if exclude_idx is not None else geom_list
    mask = rio_rasterize([(mapping(g), 1) for g in geoms], out_shape=(H, W),
                         transform=transform, fill=0, dtype="uint8", all_touched=False)
    orient = np.full((H, W), np.nan, np.float64)
    ys, xs = np.nonzero(mask)
    w = max(win, 3)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - w), min(H, y + w + 1)
        x0, x1 = max(0, x - w), min(W, x + w + 1)
        sub = mask[y0:y1, x0:x1]
        pts = np.argwhere(sub > 0).astype(np.float64) + np.array([y0, x0])
        if len(pts) < 3:
            orient[y, x] = 0.0
            continue
        mu = pts.mean(axis=0)
        cov = (pts - mu).T @ (pts - mu) / len(pts)
        wv, v = eig(cov)
        ang = math.atan2(v[1, int(np.argmax(wv))], v[0, int(np.argmax(wv))])
        orient[y, x] = ang % math.pi
    _, idx = distance_transform_edt(1 - mask, return_indices=True)
    ir, ic = idx
    return orient[ir, ic]
