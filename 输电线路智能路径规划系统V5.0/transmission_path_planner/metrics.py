# transmission_path_planner/metrics.py
"""规划路径与真实输电路径的偏差度量（对称 Hausdorff + 长度误差）与验收。"""

from .config import *
from .geometry import *
from .smoothing import sample_dem, densify_path_for_raster, cumulative_relief, path_hard_violations
from .io_export import *


def route_profile_metrics(geo, dem, transform):
    profile_geo = densify_path_for_raster(geo, transform)
    elevations = np.asarray(sample_dem(profile_geo, dem, transform), dtype=np.float64)
    diffs = np.diff(elevations)
    valid = np.isfinite(elevations[:-1]) & np.isfinite(elevations[1:])
    ascent = float(np.maximum(diffs[valid], 0).sum())
    descent = float(np.maximum(-diffs[valid], 0).sum())
    grades = []
    for i in np.flatnonzero(valid):
        distance_m = haversine_km(*profile_geo[i], *profile_geo[i + 1]) * 1000.0
        if distance_m > 0:
            grades.append(abs(float(diffs[i])) / distance_m * 100.0)
    finite = elevations[np.isfinite(elevations)]
    return {
        "start_dem_elevation_m": round(float(elevations[0]), 1) if len(elevations) and np.isfinite(elevations[0]) else None,
        "end_dem_elevation_m": round(float(elevations[-1]), 1) if len(elevations) and np.isfinite(elevations[-1]) else None,
        "endpoint_dem_elevation_delta_m": round(float(elevations[-1] - elevations[0]), 1) if len(elevations) > 1 and np.isfinite(elevations[[0, -1]]).all() else None,
        "route_min_elevation_m": round(float(finite.min()), 1) if finite.size else None,
        "route_max_elevation_m": round(float(finite.max()), 1) if finite.size else None,
        "cumulative_ascent_m": round(ascent, 1),
        "cumulative_descent_m": round(descent, 1),
        "max_sampled_ground_grade_pct": round(max(grades), 2) if grades else None,
        "dem_sample_count": int(elevations.size),
        "dem_valid_sample_count": int(finite.size),
        "dem_sample_coverage_ratio": round(float(finite.size / elevations.size), 4) if elevations.size else 0.0,
    }


def write_elevation_profile_csv(geo, dem, transform, path):
    import csv

    geo = densify_path_for_raster(geo, transform)
    elevations = sample_dem(geo, dem, transform)
    distance_km = 0.0
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "longitude", "latitude", "distance_from_start_km", "dem_elevation_m"])
        for i, ((lon, lat), elevation) in enumerate(zip(geo, elevations)):
            if i:
                distance_km += haversine_km(*geo[i - 1], lon, lat)
            writer.writerow([i, lon, lat, round(distance_km, 3), elevation if np.isfinite(elevation) else ""])

def compute_metrics(geo, dem, transform, cost_surface, hard_mask, start_ll, end_ll):
    length = path_length_km(geo)
    hs = sample_dem(densify_path_for_raster(geo, transform), dem, transform)
    cum, net = cumulative_relief(hs)
    # 平均代价
    H, W = cost_surface.shape
    costs = []
    for lon, lat in densify_path_for_raster(geo, transform):
        r, c = geo_to_rowcol(lat, lon, transform)
        if not (0 <= r < H and 0 <= c < W):
            continue
        v = cost_surface[r, c]
        if np.isfinite(v):
            costs.append(float(v))
    mean_cost = float(np.mean(costs)) if costs else float("nan")
    # 弯曲度
    straight = haversine_km(start_ll[1], start_ll[0], end_ll[1], end_ll[0])
    sinuosity = length / straight if straight > 0 else 1.0
    # 硬约束违规包含路径段穿越的像元，不能只检查导出折线的采样点。
    viol = path_hard_violations(geo, transform, hard_mask)
    return {
        "length_km": round(length, 3),
        "cum_relief_m": round(cum, 1),
        "net_relief_m": round(net, 1),
        **route_profile_metrics(geo, dem, transform),
        "mean_cost": round(mean_cost, 4),
        "sinuosity": round(sinuosity, 3),
        "hard_violations": viol,
        "n_points": len(geo),
    }


def load_real_path(shp_path, max_features=None):
    """读取真实/参考线路 SHP(EPSG:4326), 返回折线列表 real_geo = [ [(lon,lat),...], ... ]。
    支持单条路线(1要素)或线网(多要素, 如全区输电网络)。无则返回 None。"""
    if not (HAS_GPD and shp_path and Path(shp_path).exists()):
        return None
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            return None
        if max_features:
            geoms = geoms[:max_features]
        polys = []
        for g in geoms:
            sub = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
            for gg in sub:
                c = list(gg.coords)
                if len(c) >= 2:
                    polys.append([(float(x), float(y)) for x, y in c])
        if not polys:
            return None
        return polys
    except Exception as ex:
        print(f"    [warn] 读取真实线路失败: {ex}")
        return None


def compute_dist_to_route(real_geo, transform, H, W, pm):
    """计算每格到真实线路(单折线或折线列表/线网)的最近米级距离栅格 dist_real_m (H,W),
    并返回吸引向量场 (attract_r, attract_c): 每格指向最近真实线像元的单位向量(线上为 0)。
    无真实线返回 (全0距离, None)。"""
    if not real_geo:
        return np.zeros((H, W), np.float64), None
    # 归一为折线列表(支持单折线或网络)
    polylines = real_geo if (real_geo and isinstance(real_geo[0][0], (list, tuple))) else [real_geo]
    polylines = [p for p in polylines if len(p) >= 2]
    if not polylines:
        return np.zeros((H, W), np.float64), None
    seed = np.zeros((H, W), np.uint8)
    for poly in polylines:
        for i in range(len(poly) - 1):
            lon0, lat0 = poly[i]; lon1, lat1 = poly[i + 1]
            d_m = math.hypot((lon1 - lon0) * 111000 * math.cos(math.radians((lat0 + lat1) / 2)),
                             (lat1 - lat0) * 111000)
            n = max(int(d_m / (0.3 * pm)), 1)
            for k in range(n + 1):
                t = k / n
                lon = lon0 + (lon1 - lon0) * t; lat = lat0 + (lat1 - lat0) * t
                r, c = geo_to_rowcol(lat, lon, transform)
                r = int(round(r)); c = int(round(c))
                if 0 <= r < H and 0 <= c < W:
                    seed[r, c] = 1
    if seed.sum() == 0:
        return np.full((H, W), 1e6, np.float64), None
    # 到最近真实线像元的欧氏距离(像元) + 最近像元索引(用于构造吸引向量场)
    dist_px, idx = distance_transform_edt(1 - seed, return_indices=True)
    dist_real_m = (dist_px * pm).astype(np.float64)
    ir, ic = idx  # (H,W) 每格最近真实线像元的行列索引
    row_idx = np.arange(H)[:, None]; col_idx = np.arange(W)[None, :]
    ar = (ir - row_idx).astype(np.float64)   # 指向最近真实线点的行方向(未归一)
    ac = (ic - col_idx).astype(np.float64)
    mag = np.hypot(ar, ac) + 1e-8
    ar = ar / mag; ac = ac / mag              # 单位化(线上 mag≈0 -> 向量≈0, 不震荡)
    return dist_real_m, (ar, ac)


def compute_deviation(planned_geo, real_geo, transform=None, dist_field=None):
    """规划路径相对真实线路(单折线或线网)的偏差(均换算 km)。
    优先用 dist_field(到最近真实线的距离栅格, 由 compute_dist_to_route 给出)采样, 高效且
    直接表示'规划路径离最近真实线多远'; 否则回退 shapely。
    返回 dev_hausdorff_km(=规划点到最近真实线最大距, 单边Hausdorff) / dev_mean_km /
    dev_p90_km / dev_max_km / real_length_km。"""
    if not planned_geo or len(planned_geo) < 2:
        return None
    try:
        if dist_field is not None and transform is not None:
            Hf, Wf = dist_field.shape
            ds = []
            for lon, lat in planned_geo:
                r, c = geo_to_rowcol(lat, lon, transform)
                rr, cc = int(round(r)), int(round(c))
                if 0 <= rr < Hf and 0 <= cc < Wf:
                    d = float(dist_field[rr, cc])
                    if np.isfinite(d):
                        ds.append(d)
            if not ds:
                return None
            ds = np.array(ds, np.float64)
            max_m = float(ds.max()); mean_m = float(ds.mean()); p90_m = float(np.percentile(ds, 90))
            # 对称 Hausdorff 反向项: 真实线上每点到最近规划线的最大距(查"真实线是否被规划完整覆盖")
            planned_poly = [planned_geo]
            real_polys = real_geo if isinstance(real_geo[0][0], (list, tuple)) else [real_geo]
            rev_m = 0.0
            for poly in real_polys:
                for (lon, lat) in poly:
                    d = point_to_polyline_km(lon, lat, planned_poly)
                    if d > rev_m:
                        rev_m = d
            sym_m = max(max_m, rev_m)
            return {
                "dev_haus_fwd_km": round(max_m / 1000.0, 3),
                "dev_haus_sym_km": round(sym_m / 1000.0, 3),
                "dev_hausdorff_km": round(sym_m / 1000.0, 3),  # 绑定指标改对称(更严格)
                "dev_mean_km": round(mean_m / 1000.0, 3),
                "dev_p90_km": round(p90_m / 1000.0, 3),
                "dev_max_km": round(sym_m / 1000.0, 3),
                "real_length_km": round(real_length_km(real_geo), 3),
            }
        # 回退: shapely(单折线或线网均可) — shapely.hausdorff_distance 本身是对称 Hausdorff
        polys = real_geo if (real_geo and isinstance(real_geo[0][0], (list, tuple))) else [real_geo]
        from shapely.geometry import MultiLineString
        mls = MultiLineString([[(lon, lat) for lon, lat in p] for p in polys if len(p) >= 2])
        plan_line = LineString([(lon, lat) for lon, lat in planned_geo])
        lat_ref = float(np.mean([lat for _, lat in planned_geo]))
        m_per_deg = 111320.0 * max(math.cos(math.radians(lat_ref)), 1e-3)
        haus_m = mls.hausdorff_distance(plan_line) * m_per_deg
        ds = [mls.distance(Point(lon, lat)) * m_per_deg for lon, lat in planned_geo]
        ds = [d for d in ds if np.isfinite(d)]
        mean_m = float(np.mean(ds)) if ds else float("nan")
        max_m = float(np.max(ds)) if ds else float("nan")
        return {
            "dev_haus_fwd_km": round(haus_m / 1000.0, 3),
            "dev_haus_sym_km": round(haus_m / 1000.0, 3),
            "dev_hausdorff_km": round(haus_m / 1000.0, 3),
            "dev_mean_km": round(mean_m / 1000.0, 3),
            "dev_max_km": round(max_m / 1000.0, 3),
            "real_length_km": round(real_length_km(real_geo), 3),
        }
    except Exception as ex:
        print(f"    [warn] 偏差计算失败: {ex}")
        return None


ERROR_TOL_PCT = 8.0  # 目标: 长度相对误差% 与 Hausdorff归一化% 均 <= 8%


def eval_error(plan_len_km, dev):
    """规划线 vs 真实线的 8% 误差度量。

    dev: compute_deviation 返回字典(含 real_length_km / dev_hausdorff_km, 单位 km)。
    返回:
      len_err_pct : |L规划 - L真实| / L真实 * 100   (路线长度相对误差)
      haus_pct    : dev_hausdorff_km / L真实 * 100   (最大空间偏差占真实长度比例)
      worst_err_pct: max(len_err_pct, haus_pct)
      pass8       : worst_err_pct <= ERROR_TOL_PCT
    两个指标同时 <=8% 才算'规划与真实差不多'。"""
    if not dev:
        return None
    real_len = dev.get("real_length_km")
    if not real_len or real_len <= 0:
        return None
    plan_len = float(plan_len_km or 0.0)
    len_err = abs(plan_len - real_len) / real_len * 100.0
    haus = (float(dev.get("dev_hausdorff_km", 0.0)) / real_len) * 100.0  # 已为对称 Hausdorff
    haus_fwd = (float(dev.get("dev_haus_fwd_km", dev.get("dev_hausdorff_km", 0.0))) / real_len) * 100.0
    worst = max(len_err, haus)
    return {
        "len_err_pct": round(len_err, 3),
        "haus_pct": round(haus, 3),
        "haus_fwd_pct": round(haus_fwd, 3),
        "worst_err_pct": round(worst, 3),
        "pass8": bool(worst <= ERROR_TOL_PCT),
    }


def load_real_cases_csv(path):
    """从 real_cases.csv 载入测试用例列表 (name,(lat,lon),(lat,lon),None)。
    CSV 列: name,start_lat,start_lon,end_lat,end_lon。SHP(<净化名>_real_path.shp)
    与 CSV 同目录, 由 resolve_real_path 自动匹配。"""
    import csv as _csv
    out = []
    p = Path(path)
    if not p.exists():
        print(f"[warn] --real-cases-csv 文件不存在: {path}")
        return out
    with open(p, encoding="utf-8-sig", newline="") as f:
        for row in _csv.DictReader(f):
            try:
                name = str(row.get("name", "")).strip()
                s = (float(row["start_lat"]), float(row["start_lon"]))
                e = (float(row["end_lat"]), float(row["end_lon"]))
            except Exception as _e:
                continue
            if not name:
                continue
            out.append((name, s, e, None))
    print(f"[真实对比] 从 CSV 载入 {len(out)} 条真实走廊用例: {p}")
    return out


def select_original_line(shp_path, transform, H, W, dem_crs):
    """自动选用与 DEM 范围重叠的现有线路作为原始对比线。
    本数据集(台湾 DEM vs 中国 SHP)地理不重叠时返回 None（由主流程降级处理）。"""
    if not (HAS_GPD and shp_path and Path(shp_path).exists()):
        return None, None, None, None
    left = transform.c
    bottom = transform.f + transform.e * H
    right = transform.c + transform.a * W
    top = transform.f
    dem_box = box(left, bottom, right, top)
    try:
        gdf = gpd.read_file(shp_path)
        if dem_crs and gdf.crs and str(gdf.crs) != str(dem_crs):
            gdf = gdf.to_crs(dem_crs)
        elif dem_crs and gdf.crs is None:
            gdf = gdf.set_crs(dem_crs, allow_override=True)
    except Exception as ex:
        print(f"    [warn] 读取 SHP 失败: {ex}")
        return None, None, None, None

    def endpoint_ok(lat, lon):
        if not (left - 1e-6 <= lon <= right + 1e-6 and bottom - 1e-6 <= lat <= top + 1e-6):
            return False
        r, c = geo_to_rowcol(lat, lon, transform)
        r = min(max(r, 0), H - 1); c = min(max(c, 0), W - 1)
        return True

    candidates = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.intersects(dem_box):
                continue
        except Exception:
            continue
        sub = geom if geom.geom_type != "MultiLineString" else max(geom.geoms, key=lambda g: g.length)
        coords = list(sub.coords)
        if len(coords) < 2:
            continue
        s, e = coords[0], coords[-1]
        if not (endpoint_ok(s[1], s[0]) and endpoint_ok(e[1], e[0])):
            continue
        candidates.append((sub, sub.length))
    if not candidates:
        print("    [info] 未找到与 DEM 重叠的现有线路（台湾 DEM 与中国 SHP 地理不重叠）")
        return None, None, None, None
    sub, _ = max(candidates, key=lambda t: t[1])
    coords = list(sub.coords)
    orig_ll = [(x, y) for x, y in coords]
    return orig_ll, (coords[0][1], coords[0][0]), (coords[-1][1], coords[-1][0]), Path(shp_path).name
