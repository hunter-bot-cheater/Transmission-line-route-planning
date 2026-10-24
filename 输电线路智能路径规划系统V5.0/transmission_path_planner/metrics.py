# transmission_path_planner/metrics.py
"""规划路径与参考线路的双向空间相似度指标。"""

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


def path_cost_integral(geo, dem, transform, cost_surface, grade_penalty_weight=1.0,
                       turn_penalty_weight=0.0, pixel_spacing=None):
    """Integrate cell cost exactly under an axis-aligned local grid metric.

    Grade and turn terms remain one-pixel sampled estimates for non-grid paths.
    """
    dense = densify_path_for_raster(geo, transform)
    if len(dense) < 2:
        return {"status": "insufficient_path", "base_cost_meter": None,
                "grade_cost_meter": None, "turn_cost_meter": None,
                "total_cost_meter": None,
                "base_cost_method": "piecewise_constant_grid_cell_exact",
                "grade_cost_method": "one_pixel_sampled_segment_estimate"}

    height, width = cost_surface.shape
    center_lat = float(np.mean([lat for _, lat in dense]))
    row_m, col_m = (pixel_spacing if pixel_spacing is not None
                    else pixel_spacing_meters(transform, center_lat))

    # Split each segment at every raster boundary. Midpoint cell lookup then
    # integrates the piecewise-constant surface over the actual traversed length.
    inverse = ~transform
    base_total = 0.0
    for (lon0, lat0), (lon1, lat1) in zip(geo, geo[1:]):
        c0, r0 = inverse * (lon0, lat0)
        c1, r1 = inverse * (lon1, lat1)
        cuts = [0.0, 1.0]
        for start, end in ((r0, r1), (c0, c1)):
            delta = end - start
            if delta == 0.0:
                continue
            low, high = sorted((start, end))
            for boundary in range(math.floor(low) + 1, math.ceil(high)):
                t = (boundary - start) / delta
                if 0.0 < t < 1.0:
                    cuts.append(t)
        cuts.sort()
        for t0, t1 in zip(cuts, cuts[1:]):
            midpoint = (t0 + t1) * 0.5
            row = math.floor(r0 + midpoint * (r1 - r0))
            col = math.floor(c0 + midpoint * (c1 - c0))
            if (not (0 <= row < height and 0 <= col < width)
                    or not np.isfinite(cost_surface[row, col])):
                return {"status": "outside_or_invalid_cost", "base_cost_meter": None,
                        "grade_cost_meter": None, "turn_cost_meter": None,
                        "total_cost_meter": None}
            dr = (r1 - r0) * (t1 - t0)
            dc = (c1 - c0) * (t1 - t0)
            base_total += float(cost_surface[row, col]) * math.hypot(dr * row_m, dc * col_m)

    cells = [geo_to_rowcol(lat, lon, transform) for lon, lat in dense]
    if any(not (0 <= row < height and 0 <= col < width)
           or not np.isfinite(cost_surface[row, col]) for row, col in cells):
        return {"status": "outside_or_invalid_cost", "base_cost_meter": None,
                "grade_cost_meter": None, "turn_cost_meter": None,
                "total_cost_meter": None,
                "base_cost_method": "piecewise_constant_grid_cell_exact",
                "grade_cost_method": "one_pixel_sampled_segment_estimate"}

    elevations = sample_dem(dense, dem, transform)
    edge_base_estimate = []
    edge_grade = []
    edge_vectors = []
    grade_total = 0.0
    for index, ((r0, c0), (r1, c1)) in enumerate(zip(cells, cells[1:])):
        # Use transformed fractional coordinates for metric length and direction;
        # integer cells above are only used to sample raster values.
        c0f, r0f = (~transform) * dense[index]
        c1f, r1f = (~transform) * dense[index + 1]
        drf, dcf = r1f - r0f, c1f - c0f
        length_m = math.hypot(drf * row_m, dcf * col_m)
        if length_m <= 0.0:
            edge_base_estimate.append(0.0)
            edge_grade.append(0.0)
            edge_vectors.append((0.0, 0.0))
            continue
        base_estimate = 0.5 * (float(cost_surface[r0, c0]) + float(cost_surface[r1, c1])) * length_m
        grade_cost = 0.0
        e0, e1 = elevations[index], elevations[index + 1]
        if np.isfinite(e0) and np.isfinite(e1):
            grade = abs(float(e1) - float(e0)) / length_m
            grade_cost = max(float(grade_penalty_weight), 0.0) * min(grade / 0.30, 1.0) * length_m
        edge_base_estimate.append(base_estimate)
        edge_grade.append(grade_cost)
        edge_vectors.append((drf * row_m, dcf * col_m))
        grade_total += grade_cost

    turn_total = 0.0
    turn_weight = max(float(turn_penalty_weight), 0.0)
    if turn_weight:
        edge_costs = [base + grade for base, grade in zip(edge_base_estimate, edge_grade)]
        for index in range(1, len(edge_vectors)):
            previous, following = edge_vectors[index - 1], edge_vectors[index]
            previous_length = math.hypot(*previous)
            following_length = math.hypot(*following)
            if previous_length == 0.0 or following_length == 0.0:
                continue
            cosine = ((previous[0] * following[0] + previous[1] * following[1])
                      / (previous_length * following_length))
            turn_total += (turn_weight * 0.5 * (edge_costs[index - 1] + edge_costs[index])
                           * (1.0 - min(max(cosine, -1.0), 1.0)))

    return {
        "status": "measured",
        "base_cost_meter": round(base_total, 3),
        "grade_cost_meter": round(grade_total, 3),
        "turn_cost_meter": round(turn_total, 3),
        "total_cost_meter": round(base_total + grade_total + turn_total, 3),
        "base_cost_method": "piecewise_constant_grid_cell_exact",
        "grade_cost_method": "one_pixel_sampled_segment_estimate",
        "turn_cost_method": "one_pixel_sampled_segment_estimate" if turn_weight else "disabled",
    }


def astar_path_objective(path_rc, cost_surface, elevation=None, pixel_spacing=None,
                         grade_penalty_weight=1.0, turn_penalty_weight=0.0,
                         pixel_m=None):
    """Recompute the discrete A* objective from a valid cell-center search path."""
    path = list(path_rc or [])
    if not path:
        return {"status": "insufficient_path", "total_cost": None, "unit": None}
    height, width = cost_surface.shape
    if any(not (0 <= row < height and 0 <= col < width)
           or not np.isfinite(cost_surface[row, col]) for row, col in path):
        return {"status": "outside_or_invalid_cost", "total_cost": None, "unit": None}

    spacing = None if pixel_spacing is None else tuple(map(float, pixel_spacing))
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    edges = []
    base_total = grade_total = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = r1 - r0, c1 - c0
        if (dr, dc) not in directions:
            return {"status": "non_neighbor_path", "total_cost": None, "unit": None}
        if (dr and dc and
                (not np.isfinite(cost_surface[r0 + dr, c0])
                 or not np.isfinite(cost_surface[r0, c0 + dc]))):
            return {"status": "corner_cut_path", "total_cost": None, "unit": None}
        step_length = (math.hypot(dr * spacing[0], dc * spacing[1])
                       if spacing is not None else math.hypot(dr, dc))
        base = 0.5 * (float(cost_surface[r0, c0]) + float(cost_surface[r1, c1])) * step_length
        grade_cost = 0.0
        if elevation is not None and (spacing is not None or pixel_m):
            e0, e1 = float(elevation[r0, c0]), float(elevation[r1, c1])
            if np.isfinite(e0) and np.isfinite(e1):
                horizontal_m = (step_length if spacing is not None
                                else float(pixel_m) * step_length)
                grade = abs(e1 - e0) / horizontal_m
                grade_cost = (max(float(grade_penalty_weight), 0.0)
                              * min(grade / 0.30, 1.0) * step_length)
        scale = spacing if spacing is not None else (1.0, 1.0)
        vector = (dr * scale[0], dc * scale[1])
        edges.append((base + grade_cost, vector))
        base_total += base
        grade_total += grade_cost

    turn_total = 0.0
    for index in range(1, len(edges)):
        previous_cost, previous = edges[index - 1]
        following_cost, following = edges[index]
        previous_length = math.hypot(*previous)
        following_length = math.hypot(*following)
        cosine = (previous[0] * following[0] + previous[1] * following[1]) / (previous_length * following_length)
        turn_total += (max(float(turn_penalty_weight), 0.0)
                       * 0.5 * (previous_cost + following_cost)
                       * (1.0 - min(max(cosine, -1.0), 1.0)))

    return {
        "status": "measured",
        "unit": "cost_meter" if spacing is not None else "cost_pixel",
        "base_cost": round(base_total, 6),
        "grade_cost": round(grade_total, 6),
        "turn_cost": round(turn_total, 6),
        "total_cost": round(base_total + grade_total + turn_total, 6),
    }


def route_layer_exposure(geo, transform, shape, terrain, osm, hazard_masks,
                         osm_layer_metadata=None, hazard_coverage_status="unknown",
                         hazard_layer_metadata=None):
    """Summarize route exposures independently of the weights used to plan it."""
    if not geo or len(geo) < 2:
        return {"status": "route_too_short", "route_cell_count": 0}
    route_mask = rio_rasterize(
        [(LineString(geo), 1)], out_shape=shape, transform=transform,
        fill=0, all_touched=True, dtype=np.uint8,
    ) > 0
    count = int(np.count_nonzero(route_mask))
    if not count:
        return {"status": "route_not_rasterized", "route_cell_count": 0}

    result = {
        "status": "measured",
        "route_cell_count": count,
        "fraction_denominator": "all_touched_route_cells; approximate, not length weighted",
    }

    def route_band(mapping, key, dtype=np.float64):
        band = mapping.get(key)
        if band is None:
            return None
        values = np.asarray(band, dtype=dtype)
        if values.shape != tuple(shape):
            raise ValueError(f"{key} shape {values.shape} does not match route grid {tuple(shape)}")
        return values[route_mask]

    for name, key in (("surface_slope_deg", "slope"),
                      ("roughness_27_m", "roughness_27")):
        values = route_band(terrain, key)
        if values is None:
            result[name] = None
            continue
        values = values[np.isfinite(values)]
        result[name] = ({
            "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "valid_cell_fraction": float(len(values) / count),
        } if values.size else None)

    layer_status = osm_layer_metadata or {}
    result["mapped_osm_layer_status"] = {
        name: layer_status.get(name, {}).get("status", "unknown")
        for name in ("buildings", "landuse")
    }
    result["mapped_osm_coverage_status"] = {
        name: layer_status.get(name, {}).get("coverage_status", "unknown")
        for name in ("buildings", "landuse")
    }
    result["mapped_osm_exposure_scope"] = (
        "mapped-feature inventory only; coverage completeness is unverified "
        "unless each layer reports complete")
    result["hazard_coverage_status"] = hazard_coverage_status
    building_available = layer_status.get("buildings", {}).get("status") == "available"
    landuse_available = layer_status.get("landuse", {}).get("status") == "available"
    density = route_band(osm, "building_density", dtype=np.float32)
    residential = route_band(osm, "residential_landuse_mask", dtype=np.float32)
    landuse = route_band(osm, "landuse_code", dtype=np.float32)
    result["mapped_building_density_mean"] = (
        float(np.mean(density)) if building_available and density is not None else None)
    result["mapped_residential_landuse_cell_fraction"] = (
        float(np.mean(residential > 0))
        if landuse_available and residential is not None else None)
    result["mapped_forest_wetland_cell_fraction"] = (
        float(np.mean(np.isin(landuse, (1.0, 7.0))))
        if landuse_available and landuse is not None else None)
    if (building_available and landuse_available and density is not None
            and residential is not None and landuse is not None):
        settlement = np.maximum(np.maximum(density, residential), landuse == 4.0)
        result["mapped_settlement_proxy_mean"] = float(np.mean(settlement))
    else:
        result["mapped_settlement_proxy_mean"] = None
    result["mapped_osm_metric_status"] = {
        "building_density": "mapped_features_only" if building_available and density is not None
        else "array_missing" if building_available else "layer_unavailable",
        "landuse": "mapped_features_only" if landuse_available and landuse is not None
        else "array_missing" if landuse_available else "layer_unavailable",
        "residential_landuse": "mapped_features_only" if landuse_available and residential is not None
        else "array_missing" if landuse_available else "layer_unavailable",
    }

    hazard_layer_metadata = hazard_layer_metadata or {}
    hazard_fractions = {}
    hazard_observed_positive_counts = {}
    hazard_status = {}
    for name, mask in (hazard_masks or {}).items():
        values = np.asarray(mask)
        if values.shape != tuple(shape):
            raise ValueError(f"{name} shape {values.shape} does not match route grid {tuple(shape)}")
        positive_count = int(np.count_nonzero(values[route_mask] > 0))
        coverage = hazard_layer_metadata.get(name, {}).get(
            "coverage_status", hazard_coverage_status)
        hazard_status[name] = coverage
        hazard_observed_positive_counts[name] = positive_count
        hazard_fractions[name] = (
            positive_count / count if coverage == "complete" else None)
    result["hazard_layer_coverage_status"] = hazard_status
    result["hazard_positive_route_cell_counts_scope"] = (
        "observed positive cells in loaded masks only; incomplete coverage may omit cells")
    result["hazard_observed_positive_route_cell_counts"] = hazard_observed_positive_counts
    result["hazard_cell_fractions"] = hazard_fractions
    return result


def compute_metrics(geo, dem, transform, cost_surface, hard_mask, start_ll, end_ll,
                    grade_penalty_weight=1.0, turn_penalty_weight=0.0,
                    pixel_spacing=None):
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
    path_cost = path_cost_integral(
        geo, dem, transform, cost_surface, grade_penalty_weight, turn_penalty_weight,
        pixel_spacing)
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
        "path_cost_integral": path_cost,
        "sinuosity": round(sinuosity, 3),
        "hard_violations": viol,
        "n_points": len(geo),
    }


def load_real_path(shp_path, max_features=None, require_single_line=False, raise_errors=False):
    """读取真实/参考线路 SHP(EPSG:4326), 返回折线列表 real_geo = [ [(lon,lat),...], ... ]。
    require_single_line 用于单任务评测，拒绝多要素和 MultiLineString 线网。"""
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
            raise ValueError("单任务参考线路必须恰含一个 LineString 要素，避免多资产几何污染误差")
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


VISUAL_TOLERANCES_M = (100, 500, 1000, 5000)
ROUTE_SAMPLE_STEP_M = 20.0
FRECHET_SAMPLE_INTERVAL_M = 100.0


def _line_samples(projected, step_m):
    """Return arc-length midpoint samples, sample weights, and component lengths."""
    from shapely.geometry import LineString

    samples = []
    weights = []
    lengths = []
    for coordinates in projected:
        line = LineString(coordinates)
        if line.length <= 0:
            continue
        count = max(1, int(math.ceil(line.length / step_m)))
        distances = (np.arange(count, dtype=np.float64) + 0.5) * line.length / count
        samples.append(np.asarray([line.interpolate(float(d)).coords[0] for d in distances]))
        weights.append(np.full(count, line.length / count, dtype=np.float64))
        lengths.append(float(line.length))
    if not samples:
        raise ValueError("偏差计算线路长度必须大于零")
    return np.concatenate(samples, axis=0), np.concatenate(weights), lengths


def _weighted_percentile(values, weights, percentile):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    if np.allclose(sorted_weights, sorted_weights[0], rtol=1e-12, atol=0.0):
        return float(np.percentile(sorted_values, percentile))
    mid_cdf = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return float(np.interp(percentile / 100.0, mid_cdf, sorted_values,
                           left=sorted_values[0], right=sorted_values[-1]))


def _distance_summary(distances, sample_weights):
    weights = np.asarray(sample_weights, dtype=np.float64)
    return {
        "mean_m": float(np.average(distances, weights=weights)),
        "median_m": _weighted_percentile(distances, weights, 50),
        "p90_m": _weighted_percentile(distances, weights, 90),
        "p95_m": _weighted_percentile(distances, weights, 95),
        "p99_m": _weighted_percentile(distances, weights, 99),
        "sampled_max_m": float(distances.max()),
        "sampled_length_m": float(weights.sum()),
        "coverage_within_m": {
            str(tolerance): float(np.average(distances <= tolerance, weights=weights))
            for tolerance in VISUAL_TOLERANCES_M
        },
    }


def _discrete_frechet(first, second):
    """Compute discrete Frechet distance with GEOS over sampled vertices."""
    from shapely import frechet_distance
    from shapely.geometry import LineString

    return float(frechet_distance(LineString(first), LineString(second)))


def _resample_line(coordinates, max_interval_m):
    line = LineString(coordinates)
    count = max(2, int(math.ceil(line.length / max_interval_m)) + 1)
    return np.asarray([
        line.interpolate(float(distance)).coords[0]
        for distance in np.linspace(0.0, line.length, count)
    ], dtype=np.float64)


def compute_deviation(planned_geo, real_geo, planning_bounds_wgs84=None):
    """Compare route geometries in both directions using arc-length samples in meters.

    Distance quantiles and coverage are sampled approximations at
    ``ROUTE_SAMPLE_STEP_M``. The maximum is diagnostic and is not a pass/fail score.
    """
    if not planned_geo or len(planned_geo) < 2 or not real_geo:
        return None
    try:
        from pyproj import CRS, Transformer
        from shapely import linestrings, points
        from shapely.geometry import LineString, Polygon
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

        planned_projected = project_coordinates(planned_geo)

        def source_coordinates(poly):
            # Put date-line coordinates on the same longitude branch before
            # GEOS unions them; e.g. -179.9 becomes 180.1 when lon0 is 180.
            return [
                (lon0 + (float(lon) - lon0 + 180.0) % 360.0 - 180.0, float(lat))
                for lon, lat in poly
            ]

        # Dissolve every reference input in source coordinates. This removes
        # repeated and partially overlapping support before projection can
        # introduce tiny non-coincident gaps or double-counted length.
        from shapely.ops import linemerge, unary_union
        reference_geometry = unary_union([
            LineString(source_coordinates(poly)) for poly in real_polys
        ])
        if reference_geometry.geom_type == "MultiLineString":
            reference_geometry = linemerge(reference_geometry)

        def line_parts(geometry):
            if geometry.geom_type == "LineString":
                if geometry.length == 0:
                    return []
                # GEOS inserts nodes at overlap boundaries. Remove only exact
                # collinear vertices so projection does not make equivalent
                # unsplit and split reference lines measure differently.
                return [geometry.simplify(0.0, preserve_topology=True)]
            if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
                return [part for child in geometry.geoms for part in line_parts(child)]
            return []

        reference_lines = line_parts(reference_geometry)
        if not reference_lines:
            return None
        real_projected = [
            project_coordinates(list(line.coords)) for line in reference_lines
        ]
        planned_samples, planned_sample_weights, planned_component_lengths = _line_samples(
            [planned_projected], ROUTE_SAMPLE_STEP_M)
        real_samples, real_sample_weights, real_component_lengths = _line_samples(
            real_projected, ROUTE_SAMPLE_STEP_M)

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
            result[result < 1e-9] = 0.0
            return result

        planned_distances = nearest_distances(planned_samples, segment_tree(real_projected))
        reference_distances = nearest_distances(real_samples, segment_tree([planned_projected]))
        planned_to_reference = _distance_summary(planned_distances, planned_sample_weights)
        reference_to_planned = _distance_summary(reference_distances, real_sample_weights)
        combined = np.concatenate((planned_distances, reference_distances))
        tolerance_f1 = {}
        for tolerance in VISUAL_TOLERANCES_M:
            precision = planned_to_reference["coverage_within_m"][str(tolerance)]
            recall = reference_to_planned["coverage_within_m"][str(tolerance)]
            tolerance_f1[str(tolerance)] = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

        reference_length_m = float(sum(real_component_lengths))
        planned_length_m = float(sum(planned_component_lengths))
        planning_window_coverage = None
        if planning_bounds_wgs84 is not None:
            left, bottom, right, top = map(float, planning_bounds_wgs84)
            window_geo = [
                (left + (right - left) * step / 20.0, bottom)
                for step in range(21)
            ] + [
                (right, bottom + (top - bottom) * step / 20.0)
                for step in range(1, 21)
            ] + [
                (right - (right - left) * step / 20.0, top)
                for step in range(1, 21)
            ] + [
                (left, top - (top - bottom) * step / 20.0)
                for step in range(1, 20)
            ]
            window_polygon = Polygon(project_coordinates(window_geo))
            reference_lines = [LineString(line) for line in real_projected if len(line) >= 2]
            inside_m = float(sum(line.intersection(window_polygon).length
                                 for line in reference_lines))
            planning_window_coverage = {
                "reference_length_within_window_m": inside_m,
                "reference_length_outside_window_m": max(reference_length_m - inside_m, 0.0),
                "reference_length_coverage_ratio": (
                    min(max(inside_m / reference_length_m, 0.0), 1.0)
                    if reference_length_m > 0.0 else None
                ),
                "bounds_wgs84": [left, bottom, right, top],
            }
        route_similarity = {
            "comparison_crs": local_crs.to_string(),
            "sampling_interval_m": ROUTE_SAMPLE_STEP_M,
            "planned_to_reference": planned_to_reference,
            "reference_to_planned": reference_to_planned,
            "symmetric": {
                "mean_m": 0.5 * (planned_to_reference["mean_m"]
                                + reference_to_planned["mean_m"]),
                "p95_m": max(planned_to_reference["p95_m"], reference_to_planned["p95_m"]),
                "sampled_hausdorff_m": float(combined.max()),
                "coverage_f1_within_m": tolerance_f1,
            },
            "length": {
                "planned_km": planned_length_m / 1000.0,
                "reference_km": reference_length_m / 1000.0,
                "signed_difference_km": (planned_length_m - reference_length_m) / 1000.0,
                "absolute_difference_km": abs(planned_length_m - reference_length_m) / 1000.0,
                "absolute_difference_pct": abs(planned_length_m - reference_length_m) / reference_length_m * 100.0,
            },
            "planning_window_coverage": planning_window_coverage,
            "endpoint_offsets": None,
            "discrete_frechet_approx": None,
            "buffer_iou": {},
        }

        reference_is_open_line = (
            len(real_projected) == 1
            and not np.allclose(real_projected[0][0], real_projected[0][-1],
                                rtol=0.0, atol=1e-6)
        )
        if reference_is_open_line:
            endpoint_a = np.asarray(planned_projected[0])
            endpoint_b = np.asarray(planned_projected[-1])
            reference_a = np.asarray(real_projected[0][0])
            reference_b = np.asarray(real_projected[0][-1])
            forward_sum = np.linalg.norm(endpoint_a - reference_a) + np.linalg.norm(endpoint_b - reference_b)
            reverse_sum = np.linalg.norm(endpoint_a - reference_b) + np.linalg.norm(endpoint_b - reference_a)
            if forward_sum <= reverse_sum:
                endpoint_offsets = (float(np.linalg.norm(endpoint_a - reference_a)),
                                    float(np.linalg.norm(endpoint_b - reference_b)))
                orientation = "forward"
            else:
                endpoint_offsets = (float(np.linalg.norm(endpoint_a - reference_b)),
                                    float(np.linalg.norm(endpoint_b - reference_a)))
                orientation = "reversed"
            route_similarity["endpoint_offsets"] = {
                "orientation": orientation,
                "planned_start_to_reference_endpoint_m": endpoint_offsets[0],
                "planned_end_to_reference_endpoint_m": endpoint_offsets[1],
            }

            planned_shape = _resample_line(planned_projected, FRECHET_SAMPLE_INTERVAL_M)
            reference_shape = _resample_line(real_projected[0], FRECHET_SAMPLE_INTERVAL_M)
            frechet = min(
                _discrete_frechet(planned_shape, reference_shape),
                _discrete_frechet(planned_shape, reference_shape[::-1]),
            )
            route_similarity["discrete_frechet_approx"] = {
                "distance_m": frechet,
                "target_max_sample_interval_m": FRECHET_SAMPLE_INTERVAL_M,
                "planned_sample_points": len(planned_shape),
                "reference_sample_points": len(reference_shape),
                "planned_max_sample_interval_m": float(
                    planned_length_m / (len(planned_shape) - 1)),
                "reference_max_sample_interval_m": float(
                    reference_length_m / (len(reference_shape) - 1)),
            }

        try:
            from shapely.ops import unary_union
            planned_line = LineString(planned_projected)
            reference_geometry = unary_union([LineString(line) for line in real_projected])
            for radius in (100, 1000):
                planned_buffer = planned_line.buffer(radius, quad_segs=8)
                reference_buffer = reference_geometry.buffer(radius, quad_segs=8)
                union_area = planned_buffer.union(reference_buffer).area
                route_similarity["buffer_iou"][str(radius)] = (
                    float(planned_buffer.intersection(reference_buffer).area / union_area)
                    if union_area else 1.0
                )
        except Exception:
            route_similarity["buffer_iou"] = {}

        return {
            "route_similarity_version": 5,
            "route_similarity": route_similarity,
        }
    except Exception as ex:
        print(f"    [warn] 偏差计算失败: {ex}")
        return None


def eval_error(plan_len_km, dev):
    """Return v4 arc-length-weighted geometry and length metrics."""
    if not dev:
        return None
    similarity = dev.get("route_similarity")
    if not similarity:
        return None
    plan_len = float(plan_len_km or 0.0)
    reference_len = float(similarity["length"]["reference_km"])
    if reference_len <= 0:
        return None
    length = dict(similarity["length"])
    length.update({
        "planned_km": plan_len,
        "signed_difference_km": plan_len - reference_len,
        "absolute_difference_km": abs(plan_len - reference_len),
        "absolute_difference_pct": abs(plan_len - reference_len) / reference_len * 100.0,
    })
    similarity = dict(similarity)
    similarity["length"] = length
    return {
        "route_similarity_version": 5,
        "route_similarity": similarity,
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
