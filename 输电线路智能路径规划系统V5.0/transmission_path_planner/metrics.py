# transmission_path_planner/metrics.py
"""规划路径与真实输电路径的偏差度量（对称 Hausdorff + 长度误差）及内部阈值对照。"""

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


def load_real_path(shp_path, max_features=None, require_single_line=False, raise_errors=False):
    """读取真实/参考线路 SHP(EPSG:4326), 返回折线列表 real_geo = [ [(lon,lat),...], ... ]。
    require_single_line 用于 --real-shp，拒绝多要素和 MultiLineString 线网。"""
    if not shp_path or not Path(shp_path).exists():
        return None
    if not HAS_GPD:
        if require_single_line or raise_errors:
            raise RuntimeError("读取参考线路需要安装 geopandas")
        return None
    try:
        try:
            gdf = gpd.read_file(shp_path)
        except Exception as ex:
            raise RuntimeError(f"SHP 数据读取失败: {ex}") from ex
        if gdf.crs is None:
            raise ValueError("参考线路缺少 CRS；不能安全地把坐标当作经纬度")
        if str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            return None
        if require_single_line and (len(geoms) != 1 or geoms[0].geom_type != "LineString"):
            raise ValueError("单任务 --real-shp 必须恰含一个 LineString 要素，避免区域线网污染误差")
        if require_single_line and geoms[0].length <= 0:
            raise ValueError("单任务目标线路长度必须大于零")
        if max_features:
            geoms = geoms[:max_features]
        polys = []
        for g in geoms:
            sub = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
            for gg in sub:
                c = list(gg.coords)
                if len(c) >= 2 and gg.length > 0:
                    poly = [(float(coord[0]), float(coord[1])) for coord in c]
                    if any(
                        not math.isfinite(lon) or not math.isfinite(lat)
                        or not -180.0 <= lon <= 180.0
                        or not -90.0 <= lat <= 90.0
                        for lon, lat in poly
                    ):
                        raise ValueError("参考线路含无效经纬度；经度须在[-180,180]、纬度须在[-90,90]")
                    polys.append(poly)
        if not polys:
            return None
        return polys
    except Exception as ex:
        print(f"    [warn] 读取真实线路失败: {ex}")
        if require_single_line or raise_errors:
            raise
        return None


def compute_deviation(planned_geo, real_geo):
    """Project both routes to local meters and estimate symmetric Hausdorff at 5m spacing.
    """
    if not planned_geo or len(planned_geo) < 2 or not real_geo:
        return None
    try:
        from pyproj import CRS, Transformer
        from shapely import linestrings, points
        from shapely.strtree import STRtree

        real_polys = real_geo if isinstance(real_geo[0][0], (list, tuple)) else [real_geo]
        real_polys = [poly for poly in real_polys if len(poly) >= 2]
        if not real_polys:
            return None
        coords = list(planned_geo) + [point for poly in real_polys for point in poly]
        lons = np.radians([point[0] for point in coords])
        lon0 = math.degrees(math.atan2(float(np.sin(lons).mean()), float(np.cos(lons).mean())))
        lat0 = float(np.mean([point[1] for point in coords]))
        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
        )
        project = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform

        def project_coordinates(poly):
            return np.asarray([project(lon, lat) for lon, lat in poly], dtype=np.float64)

        def densify(projected):
            samples = [projected[:1]]
            for start, end in zip(projected, projected[1:]):
                length = math.hypot(end[0] - start[0], end[1] - start[1])
                steps = max(1, int(math.ceil(length / 5.0)))
                fractions = np.arange(1, steps + 1, dtype=np.float64)[:, None] / steps
                samples.append(start + fractions * (end - start))
            return np.concatenate(samples, axis=0)

        planned_projected = project_coordinates(planned_geo)
        real_projected = [project_coordinates(poly) for poly in real_polys]
        planned_samples = densify(planned_projected)
        real_samples = [densify(poly) for poly in real_projected]

        def segment_tree(projected_lines):
            segments = []
            for coordinates in projected_lines:
                if len(coordinates) >= 2:
                    segments.append(np.stack((coordinates[:-1], coordinates[1:]), axis=1))
            if not segments:
                raise ValueError("偏差计算线路长度必须大于零")
            return STRtree(linestrings(np.concatenate(segments, axis=0)))

        def nearest_distances(sample_coordinates, tree):
            result = np.full(len(sample_coordinates), np.inf, dtype=np.float64)
            for start in range(0, len(sample_coordinates), 50000):
                end = min(start + 50000, len(sample_coordinates))
                query_points = points(sample_coordinates[start:end])
                indices, distances = tree.query_nearest(
                    query_points, all_matches=False, return_distance=True)
                result[start + indices[0]] = distances
            if not np.isfinite(result).all():
                raise ValueError("空间索引未返回所有采样点的最近距离")
            return result

        real_samples_all = np.concatenate(real_samples, axis=0)
        planned_distances = nearest_distances(planned_samples, segment_tree(real_projected))
        reverse_distances = nearest_distances(real_samples_all, segment_tree([planned_projected]))
        forward_m = float(planned_distances.max())
        reverse_m = float(reverse_distances.max())
        hausdorff_m = max(forward_m, reverse_m)
        return {
            "dev_haus_fwd_km": forward_m / 1000.0,
            "dev_haus_sym_km": hausdorff_m / 1000.0,
            "dev_hausdorff_km": hausdorff_m / 1000.0,
            "dev_mean_km": float(planned_distances.mean()) / 1000.0,
            "dev_p90_km": float(np.percentile(planned_distances, 90)) / 1000.0,
            "dev_max_km": hausdorff_m / 1000.0,
            "real_length_km": sum(
                float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
                for line in real_projected
            ) / 1000.0,
            "deviation_sample_step_m": 5,
            "deviation_crs": local_crs.to_string(),
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
        "len_err_pct": round(len_err, 6),
        "haus_pct": round(haus, 6),
        "haus_fwd_pct": round(haus_fwd, 6),
        "worst_err_pct": round(worst, 6),
        "pass8": bool(worst <= ERROR_TOL_PCT),
    }


def eval_planned_route_error(planned_geo, dev):
    """Score using full route length; output metrics are rounded for display only."""
    return eval_error(path_length_km(planned_geo), dev)


def load_real_cases_csv(path):
    """从 real_cases.csv 载入测试用例列表 (name,(lat,lon),(lat,lon),None)。
    CSV 列: name,start_lat,start_lon,end_lat,end_lon。SHP(<净化名>_real_path.shp)
    与 CSV 同目录, 由 resolve_real_path 自动匹配。"""
    import csv as _csv
    out = []
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"--real-cases-csv 文件不存在: {path}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if any(not field for field in fieldnames) or len(fieldnames) != len(set(fieldnames)):
            raise ValueError("--real-cases-csv 表头存在空列名或重复列名")
        required = {"name", "start_lat", "start_lon", "end_lat", "end_lon"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"--real-cases-csv 缺少列: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"--real-cases-csv 第 {row_number} 行字段数多于表头")
            try:
                name = str(row.get("name") or "").strip()
                s = (float(row["start_lat"]), float(row["start_lon"]))
                e = (float(row["end_lat"]), float(row["end_lon"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"--real-cases-csv 第 {row_number} 行任务字段无效: {exc}") from exc
            if not name:
                raise ValueError(f"--real-cases-csv 第 {row_number} 行缺少任务名称")
            if not all(math.isfinite(value) for value in (*s, *e)):
                raise ValueError(f"--real-cases-csv 第 {row_number} 行坐标必须为有限数值")
            if not (-90 <= s[0] <= 90 and -90 <= e[0] <= 90
                    and -180 <= s[1] <= 180 and -180 <= e[1] <= 180):
                raise ValueError(f"--real-cases-csv 第 {row_number} 行经纬度超出有效范围")
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
