# transmission_path_planner/cli.py
"""命令行入口与多走廊批量编排。"""

from .config import *
from .geometry import *
from .data import *
from .osm import *
from .cost import *
from .methods import *
from .smoothing import *
from .metrics import *
from .visualization import *
from .io_export import *
from .stations import *
from .official_hazards import load_official_hazard_masks
from .experimental_theta_star import theta_star_cost_path
from shapely import make_valid


FINAL_ROUTE_LABEL = "最终路线"


def _select_final_route(methods, case_metrics):
    for method, rule in (
        ("A*(turn-aware)", "turn_penalized_screening_candidate"),
        ("A*(baseline)", "astar_min_cost_priority"),
        ("A*(weighted)", "weighted_astar_only_candidate"),
    ):
        metrics = case_metrics.get(method, {})
        if methods.get(method) and metrics.get("hard_violations") == 0:
            return method, None, rule
    return None, None, "no_hard_constraint_compliant_route"


def attach_station_endpoints(geo, station_task):
    """Replace snapped raster-center endpoints with the supplied station coordinates."""
    if not station_task or not geo:
        return geo, {}
    start, end = station_task["from_station"], station_task["to_station"]
    start_point = (float(start["lon"]), float(start["lat"]))
    end_point = (float(end["lon"]), float(end["lat"]))
    offsets = {
        "from_station_to_grid_center_offset_m": round(haversine_km(*start_point, *geo[0]) * 1000.0, 2),
        "to_station_to_grid_center_offset_m": round(haversine_km(*end_point, *geo[-1]) * 1000.0, 2),
    }
    if len(geo) == 1:
        return [start_point, end_point], offsets
    adjusted = list(geo)
    adjusted[0], adjusted[-1] = start_point, end_point
    return adjusted, offsets


def expand_single_cell_route(geo, start_ll, end_ll):
    """Turn a one-cell route into the short segment between distinct input endpoints."""
    if not geo or len(geo) != 1:
        return geo
    if tuple(start_ll) == tuple(end_ll):
        raise ValueError("起终点坐标相同，不能生成零长度线路")
    return [(start_ll[1], start_ll[0]), (end_ll[1], end_ll[0])]


def attach_input_endpoints(geo, start_ll, end_ll):
    """Restore both exact input endpoints after routing on snapped raster cells."""
    if not geo:
        return geo
    start_xy = (float(start_ll[1]), float(start_ll[0]))
    end_xy = (float(end_ll[1]), float(end_ll[0]))
    if len(geo) == 1:
        if start_xy == end_xy:
            raise ValueError("起终点坐标相同，不能生成零长度线路")
        return [start_xy, end_xy]
    adjusted = list(geo)
    adjusted[0], adjusted[-1] = start_xy, end_xy
    return adjusted


def validate_route_hard_constraints(geo, transform, hard_mask):
    """Reject a candidate whose exported geometry crosses a blocked grid cell."""
    if not geo:
        return None, {"hard_constraint_status": "no_path", "hard_violations": 0}
    violations = path_hard_violations(geo, transform, hard_mask)
    report = {
        "hard_constraint_status": "pass" if violations == 0 else "fail",
        "hard_violations": int(violations),
    }
    return (geo if violations == 0 else None), report


def screening_data_quality(osm_layer_metadata):
    """Describe missing settlement proxy layers without treating them as empty land."""
    required_layers = ("buildings", "landuse")
    missing = [
        name for name in required_layers
        if osm_layer_metadata.get(name, {}).get("status") != "available"
    ]
    return ("incomplete_osm_proxies" if missing
            else "osm_proxy_features_observed_coverage_unverified"), missing


def route_hazard_exposure(geo, hazard_masks, transform, shape):
    """Report raster-cell contact with hazard masks for the selected route."""
    if not geo or len(geo) < 2:
        return {"status": "route_too_short", "route_cell_count": 0, "layers": {}}
    route_mask = rasterize_on([(LineString(geo), {})], transform, shape) > 0
    route_cells = int(np.count_nonzero(route_mask))
    layers = {}
    for name, mask in hazard_masks.items():
        hazard = np.asarray(mask) > 0
        cells = int(np.count_nonzero(route_mask & hazard))
        layers[name] = {
            "route_cell_count": cells,
            "route_cell_fraction": cells / route_cells if route_cells else None,
        }
    return {
        "status": "measured" if route_cells else "route_not_rasterized",
        "route_cell_count": route_cells,
        "cell_fraction_is_approximate": True,
        "layers": layers,
    }


def validate_endpoint_cells(start, end, shape, hard_mask, blocking_layers=None):
    """Require fixed endpoints to lie inside the DEM and the complete passable mask."""
    height, width = shape
    for label, (row, col) in (("起点", start), ("终点", end)):
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label}超出DEM范围")
        if not hard_mask[row, col]:
            reasons = [name for name, mask in (blocking_layers or {}).items()
                       if mask is not None and bool(mask[row, col])]
            if reasons:
                reason_text = "、".join(reasons)
                raise ValueError(f"{label}位于不可通行像元（{reason_text}）；请核对接入点坐标或约束")
            raise ValueError(f"{label}位于NoData/建筑/配置禁区内；请核对接入点坐标或约束")
    return start, end


def _geometry_intersects_features(geometry, features):
    """Check exact intersections against OSM geometry rows or bare geometries."""
    return _geometry_intersection_count(geometry, features) > 0


def _geometry_intersection_summary(geometry, features):
    """Count intersections and repaired nearby OSM geometries."""
    minx, miny, maxx, maxy = geometry.bounds
    count = 0
    repaired = 0
    for feature in features:
        obstacle = feature[0] if isinstance(feature, (tuple, list)) else feature
        left, bottom, right, top = obstacle.bounds
        if left <= maxx and right >= minx and bottom <= maxy and top >= miny:
            if not obstacle.is_valid:
                obstacle = make_valid(obstacle)
                if obstacle.is_empty or not obstacle.is_valid:
                    raise ValueError("相交建筑几何修复失败，拒绝通过精确建筑复核")
                repaired += 1
            if geometry.intersects(obstacle):
                count += 1
    return count, repaired


def _geometry_intersection_count(geometry, features):
    """Count mapped features intersecting a geometry after validity repair."""
    return _geometry_intersection_summary(geometry, features)[0]


def open_safe_building_endpoint_pixels(endpoints, hard_mask, blocking_layers,
                                       building_geometries, transform,
                                       building_clearance_m=0.0):
    """Open only endpoint cells blocked by rasterization when exact geometry is clear."""
    if not building_geometries or building_clearance_m > 0:
        return []

    exceptions = []
    for label, (row, col), point_xy in endpoints:
        if not (0 <= row < hard_mask.shape[0] and 0 <= col < hard_mask.shape[1]):
            continue
        if hard_mask[row, col]:
            continue
        reasons = [name for name, mask in blocking_layers.items()
                   if mask is not None and bool(mask[row, col])]
        if reasons != ["OSM建筑足迹"]:
            continue

        point = Point(*point_xy)
        if _geometry_intersects_features(point, building_geometries):
            continue
        center_xy = transform * (col + 0.5, row + 0.5)
        connector = LineString([point_xy, center_xy])
        if _geometry_intersects_features(connector, building_geometries):
            continue

        probe_mask = hard_mask.copy()
        probe_mask[row, col] = 1
        if path_hard_violations([point_xy, center_xy], transform, probe_mask):
            continue
        exceptions.append({"label": label, "row": row, "col": col,
                           "point_xy": tuple(point_xy)})

    for exception in exceptions:
        hard_mask[exception["row"], exception["col"]] = 1
    return exceptions


def validate_route_against_buildings(geo, building_geometries):
    """Fail closed when an exported route intersects an available building footprint."""
    if not geo:
        return None, {"building_geometry_status": "no_path",
                      "building_geometry_intersections": 0}
    route = LineString(geo) if len(geo) > 1 else Point(*geo[0])
    intersections, repaired = _geometry_intersection_summary(route, building_geometries)
    report = {
        "building_geometry_status": "fail" if intersections else "pass",
        "building_geometry_intersections": intersections,
        "building_geometry_repaired_features": repaired,
    }
    return (None if intersections else geo), report


def annotate_route_validation_context(report, endpoint_exceptions, building_metadata):
    """Record endpoint exceptions and the scope of the available building check."""
    report["endpoint_rasterization_exceptions"] = [
        {"label": item["label"], "row": int(item["row"]),
         "col": int(item["col"]), "point_xy": list(item["point_xy"])}
        for item in endpoint_exceptions
    ]
    report["building_geometry_coverage_status"] = building_metadata.get(
        "coverage_status", "unknown")
    report["building_geometry_coverage_reason"] = building_metadata.get(
        "coverage_reason")
    report["building_geometry_check_scope"] = (
        "loaded_osm_features_only"
        if building_metadata.get("status") == "available" else "unavailable")
    report["building_geometry_source_repair_count"] = int(
        building_metadata.get("geometry_repaired_count", 0))
    report["building_geometry_skipped_nonfinite_count"] = int(
        building_metadata.get("nonfinite_geometry_outside_bbox_count", 0))
    return report


def run_method_or_skip_on_oom(method_name, run_method):
    """Skip a neural candidate only for PyTorch OOM; let other errors surface."""
    try:
        return run_method(), None
    except torch.OutOfMemoryError as exc:
        message = str(exc)
    print(f"  [{method_name}] 显存不足，跳过该方法并继续其余规划: {message}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None, message


def train_vin_or_skip_on_oom(*args, **kwargs):
    """Skip VIN on PyTorch OOM without suppressing unrelated failures."""
    result, _ = run_method_or_skip_on_oom("VIN-Grad", lambda: train_vin(*args, **kwargs))
    return result


def validate_unique_case_names(cases, check_reference_names=False):
    names = [case[0] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("用例名称必须唯一，重复名称会覆盖输出目录")
    if check_reference_names:
        # Reference files must resolve uniquely on Windows as well as POSIX.
        reference_names = [safe_reference_name(name).casefold() for name in names]
        if len(set(reference_names)) != len(reference_names):
            raise ValueError("用例名净化后发生重复，无法安全匹配逐任务参考线路文件")


def validate_cases_available(cases):
    if not cases:
        raise ValueError("未找到有效规划任务；请检查输入 CSV 或站点任务")
    if any(tuple(start) == tuple(end) for _, start, end in cases):
        raise ValueError("起终点坐标相同，不能生成零长度线路")


def load_case_reference(path, require_single_line=False):
    if not path or not Path(path).is_file():
        return None, "reference_missing"
    try:
        real_geo = load_real_path(
            path, require_single_line=require_single_line, raise_errors=True)
    except ValueError as exc:
        print(f"  [warn] 目标线路参考文件无效: {exc}")
        return None, "reference_invalid_or_empty"
    except Exception as exc:
        print(f"  [warn] 目标线路参考文件无法读取: {exc}")
        return None, "reference_invalid_or_unreadable"
    return (real_geo, "reference_loaded" if real_geo else "reference_invalid_or_empty")


def summarize_run_completion(expected_cases, results, failed_cases):
    failed_names = {item["case"] for item in failed_cases}
    no_route_cases = [
        record["case"] for record in results
        if record["metrics"].get(FINAL_ROUTE_LABEL, {}).get("planning_status")
        == "no_hard_constraint_compliant_route"
    ]
    no_route_names = set(no_route_cases)
    completed = {
        record["case"] for record in results
        if record["case"] not in failed_names and record["case"] not in no_route_names
    }
    return {
        "expected_cases": expected_cases,
        "completed_cases": len(completed),
        "recorded_cases": len(results),
        "failed_cases": failed_cases,
        "no_route_cases": no_route_cases,
        "complete": expected_cases > 0 and len(completed) == expected_cases
        and not failed_cases and not no_route_cases,
    }


def method_visual_similarity_summary(results, method, expected_cases=None, failed_case_names=None):
    excluded = set(failed_case_names or ())
    eligible = [record for record in results
                if record["case"] not in excluded
                and record["metrics"].get(FINAL_ROUTE_LABEL, {}).get("planning_status")
                != "no_hard_constraint_compliant_route"]
    comparisons = [record["metrics"].get(method, {}).get("route_similarity")
                   for record in eligible]
    comparisons = [value for value in comparisons if value]
    total_cases = max(len(results), expected_cases or 0)
    p95 = [value["symmetric"]["p95_m"] for value in comparisons]
    hausdorff = [value["symmetric"]["sampled_hausdorff_m"] for value in comparisons]
    coverage = [value["symmetric"]["coverage_f1_within_m"]["1000"]
                for value in comparisons]
    coverage_100m = [value["symmetric"]["coverage_f1_within_m"].get("100")
                     for value in comparisons]
    coverage_100m = [value for value in coverage_100m if value is not None]
    length_error = [value["length"]["absolute_difference_pct"] for value in comparisons]
    return {
        "measured": len(comparisons),
        "unmeasured": max(0, total_cases - len(comparisons)),
        "median_symmetric_p95_m": float(np.median(p95)) if p95 else None,
        "median_sampled_hausdorff_m": float(np.median(hausdorff)) if hausdorff else None,
        "median_coverage_f1_100m": float(np.median(coverage_100m)) if coverage_100m else None,
        "median_coverage_f1_1km": float(np.median(coverage)) if coverage else None,
        "median_length_difference_pct": float(np.median(length_error)) if length_error else None,
    }


def validate_real_shp_argument(real_shp, cases):
    if real_shp is None:
        return
    if not Path(real_shp).is_file():
        raise ValueError("--real-shp 必须指向一个存在的目标线路文件")
    if len(cases) != 1:
        raise ValueError("--real-shp 只支持单任务目标线路；多任务请使用 --real-dir 提供逐任务线路")


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="replace")


def parse_lat_lon(value, label):
    try:
        parts = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label}必须是纬度,经度数值") from exc
    if (len(parts) != 2 or not all(math.isfinite(part) for part in parts)
            or not -90 <= parts[0] <= 90 or not -180 <= parts[1] <= 180):
        raise ValueError(f"{label}经纬度无效；纬度须在[-90,90]、经度须在[-180,180]")
    return tuple(parts)


def main():
    configure_console_output()
    t0 = time.time()
    ap = argparse.ArgumentParser(description="复杂山区输电线路启发式路径规划 (V5.0)")
    ap.add_argument("--dem", type=str, default=None)
    ap.add_argument("--start", type=str, default=None, help="起点 lat,lon")
    ap.add_argument("--end", type=str, default=None, help="终点 lat,lon")
    ap.add_argument("--stations-csv", type=str, default=None,
                    help="站点表: station_id,name,lat,lon,voltage_kv; 坐标应为线路接入点")
    ap.add_argument("--route-tasks-csv", type=str, default=None,
                    help="站间任务表: route_id,from_station_id,to_station_id")
    ap.add_argument("--constraint-geojson", type=str, default=None,
                    help="可选 WGS84 多边形图层; properties 需含 mode=hard/soft、source；soft 还需 weight")
    default_hazard_dir = PACKAGE_DIR / "data" / "official_sources" / "ardswc_115_mvt_z10_20261023"
    ap.add_argument("--official-hazards-dir", type=str, default=str(default_hazard_dir),
                    help="农业部 115 年土石流/大规模崩塌 MVT 数据目录；默认读取软件包内离线图层")
    ap.add_argument("--official-hazard-weight", type=float, default=1.0,
                    help="官方土石流/崩塌图层联合软代价权重；重叠类别不重复累计，0 表示关闭")
    ap.add_argument("--turn-penalty-weight", type=float, default=0.0,
                    help="A* 转弯平滑代价系数；0 关闭。启用后使用8方向状态、内存占用显著增加")
    ap.add_argument("--settlement-weight", type=float, default=2.0,
                    help="OSM建筑密度/住宅与商业用地代理的软避让权重(方案筛选参数, 默认2.0)")
    ap.add_argument("--ridge-penalty-weight", type=float, default=0.10,
                    help="基于300米TPI的高脊软惩罚权重；仅用地形，不读取线路几何")
    ap.add_argument("--landcover-penalty-weight", type=float, default=0.0,
                    help="OSM林地/湿地软代价权重；默认0关闭，不完整环境图层仅作可选筛查")
    ap.add_argument("--building-clearance-m", type=float, default=0.0,
                    help="建筑物栅格硬避让距离(米, 必须小于5000); 0仅屏蔽已映射建筑足迹，不代表规范净距")
    ap.add_argument("--grade-penalty-weight", type=float, default=1.0,
                    help="沿线相邻DEM高差软惩罚权重(以30%%地面坡度作为一个参考级)")
    ap.add_argument("--terrain-slope-weight", type=float, default=0.75,
                    help="DEM坡度面软代价权重；默认0.75，0可用于消融测试")
    ap.add_argument("--roughness-weight", type=float, default=0.25,
                    help="900米地形粗糙度软代价权重；默认0.25，0可用于消融测试")
    ap.add_argument("--astar-heuristic-weight", type=float, default=1.0,
                    help="A*启发函数权重；1.0对当前离散代价面求最优，大于1时为近似搜索且不保证误差界")
    ap.add_argument("--experimental-theta-star", action="store_true",
                    help="额外生成 Theta* 实验候选；复用 A* 栅格代价面，不计软坡度/转弯代价，"
                         "经常规约束复核但不参与最终选路")
    ap.add_argument("--max-ground-grade", type=float, default=None,
                    help="可选地面相邻栅格最大坡度比，如0.30; 属于筛选阈值，不是线路设计限值")
    ap.add_argument("--max-surface-slope-deg", type=float, default=None,
                    help="可选地表坡度硬限制(度); 默认不设工程阈值")
    ap.add_argument("--max-terrain-elevation-m", type=float, default=None,
                    help="可选地形海拔硬限制(米); 默认不设工程阈值")
    ap.add_argument("--use-costunet", action="store_true",
                    help="把CostUNet作为50%%软先验混入规则代价; 默认关闭，避免未知训练来源影响基线")
    ap.add_argument("--max-side", type=int, default=WORK_MAX_SIDE,
                    help=f"规划网格最长边；默认{WORK_MAX_SIDE}，尽量保留 DEM 原始分辨率，内存不足时可调低")
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--include-ai-diagnostics", action="store_true",
                    help="可选运行 VIN/PPO 作为诊断对照；AI 候选不会被选为最终路线")
    ap.add_argument("--no-ppp", action="store_true", help="在 --include-ai-diagnostics 下跳过 PPO")
    ap.add_argument("--no-vin", action="store_true", help="在 --include-ai-diagnostics 下跳过 VIN-Grad")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--with-real", action="store_true",
                    help="加载参考线路，仅在规划选路完成后计算 v2 双向空间相似度")
    ap.add_argument("--real-cases", action="store_true",
                    help="改用 v3 的 4 条真实走廊(R01~R04)并启用真实线对照")
    ap.add_argument("--real-cases-csv", type=str, default=None,
                    help="从真实走廊 CSV(列: name,start_lat,start_lon,end_lat,end_lon)载入测试用例；"
                         "默认在 CSV 同目录查找参考线，可用 --real-dir 指定其他目录。真实线只用于规划后的对照。")
    ap.add_argument("--real-dir", type=str, default=None,
                    help="真实线路 SHP 所在目录(按 '<净化名>_real_path.shp' 匹配)")
    ap.add_argument("--real-shp", type=str, default=None,
                    help="单任务指定只含一个 LineString 且 CRS 明确的目标线路 SHP 作为事后对照; "
                         "多任务请用 --real-dir 按任务分别提供线路。规划阶段不会读取该线。")
    ap.add_argument("--osm-cache-dir", type=str, default=None,
                    help="OSM 环境图层缓存目录，内含 taiwan_buildings.pkl、taiwan_landuse.pkl 等文件")
    ap.add_argument("--rebuild-osm-cache", action="store_true",
                    help="调用 shared.data_acquisition 重新生成台湾 OSM 缓存 pkl(覆盖错误区域数据), "
                         "然后继续正常流程。首次使用或土地利用波段恒为0时请启用。")
    # PPO 强度(显存/耗时敏感, 默认中等; RTX 4060 8GB 可调低防 OOM)
    ap.add_argument("--ppo-updates", type=int, default=60, help="PPO 训练更新轮数(默认60)")
    ap.add_argument("--ppo-episodes", type=int, default=8, help="PPO 每轮采样的轨迹数(默认8)")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="PPO 每轮策略优化 epoch(默认4)")
    ap.add_argument("--ppo-bc", type=int, default=60, help="PPO 行为克隆预热步数(默认60)")
    args = ap.parse_args()
    if bool(args.stations_csv) != bool(args.route_tasks_csv):
        ap.error("--stations-csv 和 --route-tasks-csv 必须同时提供")
    if bool(args.start) != bool(args.end):
        ap.error("--start 和 --end 必须同时提供")
    task_sources = sum((
        bool(args.stations_csv), bool(args.start or args.end),
        bool(args.real_cases_csv), bool(args.real_cases),
    ))
    if task_sources > 1:
        ap.error("任务输入来源只能选一种：站点 CSV、--start/--end、--real-cases-csv 或 --real-cases")
    if args.real_shp and (args.real_cases_csv or args.real_cases):
        ap.error("--real-shp 不能与 --real-cases-csv/--real-cases 同时使用")
    if args.real_shp and args.real_dir:
        ap.error("--real-shp 不能与 --real-dir 同时使用")
    if args.max_side < 1:
        ap.error("--max-side 必须是正整数")
    float_parameters = {
        "--building-clearance-m": args.building_clearance_m,
        "--ridge-penalty-weight": args.ridge_penalty_weight,
        "--landcover-penalty-weight": args.landcover_penalty_weight,
        "--settlement-weight": args.settlement_weight,
        "--grade-penalty-weight": args.grade_penalty_weight,
        "--terrain-slope-weight": args.terrain_slope_weight,
        "--roughness-weight": args.roughness_weight,
        "--official-hazard-weight": args.official_hazard_weight,
        "--turn-penalty-weight": args.turn_penalty_weight,
        "--astar-heuristic-weight": args.astar_heuristic_weight,
        "--max-ground-grade": args.max_ground_grade,
        "--max-surface-slope-deg": args.max_surface_slope_deg,
        "--max-terrain-elevation-m": args.max_terrain_elevation_m,
    }
    for option, value in float_parameters.items():
        if value is not None and not math.isfinite(value):
            ap.error(f"{option} 必须是有限数值")
    if (args.building_clearance_m < 0 or args.ridge_penalty_weight < 0
            or args.landcover_penalty_weight < 0
            or args.settlement_weight < 0
            or args.grade_penalty_weight < 0
            or args.terrain_slope_weight < 0
            or args.roughness_weight < 0
            or args.official_hazard_weight < 0
            or args.turn_penalty_weight < 0):
        ap.error("避让距离与代价权重不能为负数")
    if args.astar_heuristic_weight < 1:
        ap.error("--astar-heuristic-weight 必须大于或等于 1.0")
    if args.building_clearance_m >= 5000:
        ap.error("--building-clearance-m 必须小于5000米（当前距离栅格上限）")
    if args.max_ground_grade is not None and args.max_ground_grade <= 0:
        ap.error("--max-ground-grade 必须大于0")

    if args.rebuild_osm_cache:
        print("[OSM] 请求重建台湾 OSM 缓存 ...")
        try:
            from . import rebuild_osm_cache as _roc
            _roc.rebuild()
        except Exception as _e:
            print(f"[warn] OSM 缓存重建失败: {_e}")
        print("[OSM] 缓存重建完成, 继续后续流程。")

    OUT = Path(args.out_dir) if args.out_dir else OUTPUTS_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("  复杂山区输电线路启发式路径规划 V5.0")
    print("=" * 70)

    dem_path = Path(args.dem) if args.dem else find_dem()
    if dem_path is None:
        sys.exit("[FATAL] 未找到 DEM，请用 --dem 指定。")
    print(f"[DEM] {dem_path}")

    station_task_by_route = {}
    if args.stations_csv:
        try:
            station_tasks = load_station_tasks(args.stations_csv, args.route_tasks_csv)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        cases = []
        for task in station_tasks:
            cases.append((task["route_id"], task["start"], task["end"]))
            station_task_by_route[task["route_id"]] = task
    elif args.start and args.end:
        try:
            start = parse_lat_lon(args.start, "--start")
            end = parse_lat_lon(args.end, "--end")
        except ValueError as exc:
            ap.error(str(exc))
        cases = [("Custom", start, end)]
    else:
        if args.real_cases_csv:
            if not Path(args.real_cases_csv).is_file():
                ap.error(f"--real-cases-csv 文件不存在: {args.real_cases_csv}")
            try:
                csv_cases = load_real_cases_csv(args.real_cases_csv)
            except (OSError, ValueError) as exc:
                ap.error(str(exc))
            # 默认从 CSV 同目录找参考线；显式 --real-dir 时使用指定目录。
            args.real_dir = args.real_dir or str(Path(args.real_cases_csv).parent)
            cases = [(n, s, e) for (n, s, e, _) in csv_cases]
        elif args.real_cases:
            cases = [(n, s, e) for (n, s, e, _) in REAL_CASES]
        else:
            cases = DEFAULT_CASES[:max(1, args.cases)]

    allow_global_reference_mapping = not any((
        args.real_dir, args.stations_csv, args.start, args.real_cases_csv))
    args.real_dir = args.real_dir or str(REAL_PATH_DIR)
    try:
        validate_cases_available(cases)
        validate_unique_case_names(
            cases,
            check_reference_names=not args.real_shp and bool(
                args.real_cases_csv or args.real_cases or args.with_real),
        )
        validate_real_shp_argument(args.real_shp, cases)
    except ValueError as exc:
        ap.error(str(exc))

    global_real_exists = args.real_shp is not None
    use_real = bool(args.with_real or args.real_cases or args.real_cases_csv or global_real_exists)
    if use_real:
        mode = "单任务目标线路" if global_real_exists else args.real_dir
        print(f"[真实对比] 已启用: 模式={mode}；真实线仅用于规划后的事后评估")

    # CostUNet（全局加载一次）
    costunet = load_costunet() if args.use_costunet and HAS_TORCH else None
    if args.use_costunet and costunet is None:
        print("[warn] CostUNet不可用，继续使用规则代价面")

    all_results = []
    main_done = False

    failed_cases = []
    for ci, (cname, (slat, slon), (elat, elon)) in enumerate(cases):
        print(f"\n--- 用例 {ci + 1}/{len(cases)}: {cname} ---")
        try:
            margin = 0.15
            minlon, maxlon = min(slon, elon) - margin, max(slon, elon) + margin
            minlat, maxlat = min(slat, elat) - margin, max(slat, elat) + margin
            dem, transform, crs, pm = load_dem_window(dem_path, minlon, minlat, maxlon, maxlat, args.max_side)
            H, W = dem.shape
            if args.turn_penalty_weight > 0:
                state_gib = H * W * (8 * 8 + 8) / (1024 ** 3)
                if state_gib >= 0.5:
                    print(f"  [warn] 转弯状态数组约需 {state_gib:.2f} GiB；还未包含其他栅格和搜索队列")

            sr, sc = geo_to_rowcol(slat, slon, transform)
            er, ec = geo_to_rowcol(elat, elon, transform)
            station_task = station_task_by_route.get(cname)
            shortcut_endpoints = (
                (float(station_task["from_station"]["lon"]),
                 float(station_task["from_station"]["lat"])),
                (float(station_task["to_station"]["lon"]),
                 float(station_task["to_station"]["lat"])),
            ) if station_task else ((float(slon), float(slat)), (float(elon), float(elat)))

            terrain = compute_terrain_factors(dem, pm, transform, crs)
            left = transform.c; top = transform.f
            right = left + transform.a * W; bottom = top + transform.e * H
            bbox = (left, bottom, right, top)
            official_hazard_masks, official_hazard_metadata = load_official_hazard_masks(
                args.official_hazards_dir, bbox, transform, dem.shape)
            official_hazard_metadata["soft_weight_applied_in_run"] = args.official_hazard_weight
            if official_hazard_metadata.get("coverage_status") != "complete":
                print("  [warn] 官方灾害图层覆盖不完整；未知瓦片不按无风险处理。"
                      f" status={official_hazard_metadata.get('status')}, "
                      f"coverage={official_hazard_metadata.get('coverage_status')}")
            needs_feature_stack = costunet is not None or (
                args.include_ai_diagnostics and HAS_TORCH and not args.no_vin)
            osm = compute_osm_bands(
                dem, transform, bbox, cache_dir=args.osm_cache_dir,
                include_model_features=needs_feature_stack,
                include_building_distance=args.building_clearance_m > 0,
            )
            osm_layer_metadata = getattr(osm, "metadata", {}).get("layers", {})
            custom_constraints = (
                load_user_constraint_overlays(args.constraint_geojson, transform, dem.shape)
                if args.constraint_geojson else None
            )
            custom_constraint_metadata = custom_constraints["metadata"] if custom_constraints else None
            for layer_name in ("buildings", "landuse"):
                layer = osm_layer_metadata.get(layer_name, {})
                if layer.get("status") != "available":
                    print(f"  [warn] OSM {layer_name} 图层状态={layer.get('status', 'unknown')}；"
                          "不能据此认定该范围内没有居民点/建筑")
            stack = build_feature_stack(dem, terrain, osm) if needs_feature_stack else None

            model_cost = costunet.predict(stack) if costunet is not None else None
            cost = build_route_cost(
                terrain, osm, args.settlement_weight, model_cost,
                ridge_penalty_weight=args.ridge_penalty_weight,
                landcover_penalty_weight=args.landcover_penalty_weight,
                official_hazard_masks=official_hazard_masks,
                official_hazard_weight=args.official_hazard_weight,
                terrain_slope_weight=args.terrain_slope_weight,
                roughness_weight=args.roughness_weight,
            )

            real_geo = None
            if custom_constraints:
                soft_cost = custom_constraints["soft_cost"]
                cost = (cost + soft_cost).astype(np.float32)
            print(f"  代价面范围 [{np.nanmin(cost):.3f}, {np.nanmax(cost):.3f}]")

            slope = terrain["slope"]; elev = terrain["elevation"]
            hard_mask = np.ones((H, W), np.uint8)
            hard_mask[~np.isfinite(dem)] = 0
            if custom_constraints:
                hard_mask[custom_constraints["hard_mask"] > 0] = 0
            if args.max_surface_slope_deg is not None:
                if not 0 < args.max_surface_slope_deg <= 90:
                    raise ValueError("--max-surface-slope-deg 必须在(0,90]内")
                hard_mask[slope > args.max_surface_slope_deg] = 0
            if args.max_terrain_elevation_m is not None:
                hard_mask[elev > args.max_terrain_elevation_m] = 0
            building_mask = osm.get("building_mask")
            if building_mask is not None:
                hard_mask[np.asarray(building_mask) > 0] = 0
            if args.building_clearance_m > 0 and "building_distance_m" in osm:
                hard_mask[osm["building_distance_m"] <= args.building_clearance_m] = 0
            endpoint_blockers = {"DEM NoData": ~np.isfinite(dem)}
            if custom_constraints:
                endpoint_blockers["用户硬约束"] = custom_constraints["hard_mask"] > 0
            if args.max_surface_slope_deg is not None:
                endpoint_blockers["地表坡度阈值"] = slope > args.max_surface_slope_deg
            if args.max_terrain_elevation_m is not None:
                endpoint_blockers["地形海拔阈值"] = elev > args.max_terrain_elevation_m
            if building_mask is not None:
                endpoint_blockers["OSM建筑足迹"] = np.asarray(building_mask) > 0
            if args.building_clearance_m > 0 and "building_distance_m" in osm:
                endpoint_blockers["建筑净距阈值"] = (
                    osm["building_distance_m"] <= args.building_clearance_m)
            building_geometries = getattr(osm, "geometries", {}).get("buildings", [])
            endpoint_building_exceptions = open_safe_building_endpoint_pixels(
                (("起点", (sr, sc), (slon, slat)),
                 ("终点", (er, ec), (elon, elat))),
                hard_mask, endpoint_blockers, building_geometries, transform,
                args.building_clearance_m,
            )
            validate_endpoint_cells((sr, sc), (er, ec), dem.shape, hard_mask,
                                    endpoint_blockers)
            same_grid_cell = (sr, sc) == (er, ec)
            cost_grid = (1.0 + 4.0 * cost).astype(np.float64)
            cost_grid = np.where(hard_mask == 0, np.inf, cost_grid)
            if args.include_ai_diagnostics:
                cost_finite = np.where(np.isfinite(cost), cost, 1.0).astype(np.float32)
                cost_vin = np.where(hard_mask > 0, cost_grid, 0.0).astype(np.float32)
            lat_center = transform.f + transform.e * (H / 2.0)
            route_pixel_spacing = pixel_spacing_meters(transform, lat_center, crs)

            methods = {}
            grade_reports = {}
            V_main = None
            t_vin = 0.0
            t_ppo = 0.0
            t_theta = 0.0
            # AI methods are opt-in diagnostics; route selection below excludes them.
            if args.include_ai_diagnostics and HAS_TORCH and not args.no_vin and not same_grid_cell:
                t_v = time.time()
                extra = build_vin_extra(stack)
                goal_mask = np.zeros((H, W), np.float32); goal_mask[er, ec] = 1.0
                torch.manual_seed(RANDOM_SEED)
                V = train_vin_or_skip_on_oom(cost_vin, goal_mask, extra, hard_mask,
                                             route_pixel_spacing, n_steps=80)
                if V is None:
                    print("  [VIN-Grad] 本用例未生成 AI 候选")
                else:
                    V_main = np.nan_to_num(V, nan=1e6, posinf=1e6, neginf=1e6)
                    geo_vin = smooth_path_with_constraints(gradient_track_path(
                        V, cost_finite, hard_mask, (sr, sc), (er, ec), transform),
                        transform, hard_mask, cost_grid, dem,
                        0.0, pixel_spacing=route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints)
                    geo_vin, grade_reports["VIN-Grad"] = filter_path_by_ground_grade(
                        geo_vin, dem, transform, args.max_ground_grade, crs)
                    if geo_vin:
                        grade_reports["VIN-Grad"] = ground_grade_limit_report(
                            geo_vin, dem, transform, args.max_ground_grade, crs)
                    else:
                        grade_reports["VIN-Grad"] = ground_grade_limit_report(
                            None, dem, transform, args.max_ground_grade, crs)
                    # 保存真实 V/cost/hard/起终点, 便于 PPO 推理调试(无需重跑训练即可秒级重测)
                    try:
                        np.savez(str(OUT / "_last_V.npz"), V=V_main, cost=cost_finite,
                                 hard=hard_mask, start=np.array([sr, sc]), goal=np.array([er, ec]))
                    except Exception:
                        pass
                    methods["VIN-Grad"] = geo_vin
                    t_vin = time.time() - t_v
                    print(f"  [VIN-Grad] 路径 {len(geo_vin) if geo_vin else 0} 点, 耗时 {t_vin:.1f}s")
            else:
                reason = ("未启用 AI 诊断" if not args.include_ai_diagnostics else
                          "起终点位于同一栅格" if same_grid_cell else
                          "已禁用或 torch 不可用")
                print(f"  [skip] VIN-Grad {reason}")

            # --- 方法2: RL/PPO 策略梯度路径生成器 (每用例独立训练: 代价图与目标不同) ---
            if args.include_ai_diagnostics and HAS_TORCH and not args.no_ppp and not same_grid_cell:
                def build_ppo_candidate():
                    torch.manual_seed(RANDOM_SEED + 100)
                    print("  [PPO] 训练策略网络 ...")
                    t_p = time.time()
                    ppo_policy, _ = train_ppo(
                        cost_finite, hard_mask, (er, ec), (sr, sc), value_map=V_main,
                        n_updates=args.ppo_updates, episodes=args.ppo_episodes,
                        ppo_epochs=args.ppo_epochs, bc_steps=args.ppo_bc)
                    print(f"  [PPO] 训练完成 {time.time() - t_p:.1f}s")
                    ctx = build_ppo_context(np.clip(cost_finite, 0, 1), (er, ec), H, W, value_map=V_main)
                    ppo_path = ppo_rollout_path(ppo_policy, ctx, cost_finite, hard_mask, (er, ec), (sr, sc),
                                               n_tries=6, guide_v=V_main, guide_w=6.0, v_tol=0.35)
                    geo_ppo = smooth_path_with_constraints(
                        [rowcol_to_geo(r, c, transform) for r, c in ppo_path],
                        transform, hard_mask, cost_grid, dem,
                        0.0, pixel_spacing=route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints)
                    geo_ppo, grade_reports["PPO"] = filter_path_by_ground_grade(
                        geo_ppo, dem, transform, args.max_ground_grade, crs)
                    if geo_ppo:
                        grade_reports["PPO"] = ground_grade_limit_report(
                            geo_ppo, dem, transform, args.max_ground_grade, crs)
                    else:
                        grade_reports["PPO"] = ground_grade_limit_report(
                            None, dem, transform, args.max_ground_grade, crs)
                    nonlocal t_ppo
                    t_ppo = time.time() - t_p
                    return geo_ppo

                geo_ppo, ppo_oom = run_method_or_skip_on_oom("PPO", build_ppo_candidate)
                if ppo_oom is None:
                    methods["PPO"] = geo_ppo
                    print(f"  [PPO] 路径 {len(geo_ppo) if geo_ppo else 0} 点")
            else:
                reason = ("未启用 AI 诊断" if not args.include_ai_diagnostics else
                          "起终点位于同一栅格" if same_grid_cell else
                          "已禁用或 torch 不可用")
                print(f"  [skip] PPO {reason}")

            # Weight 1 is optimal A*. Larger weights are approximate; because
            # closed states are not reopened, no general cost bound is claimed.
            astar_method = (
                "A*(turn-aware)" if args.turn_penalty_weight > 0 else
                "A*(baseline)" if args.astar_heuristic_weight == 1.0 else
                "A*(weighted)"
            )
            t_astar = time.time()
            a_path = astar_search(
                cost_grid, (sr, sc), (er, ec), elevation=dem, pixel_m=pm,
                grade_penalty_weight=args.grade_penalty_weight,
                max_ground_grade=args.max_ground_grade,
                pixel_spacing=route_pixel_spacing,
                heuristic_weight=args.astar_heuristic_weight,
                turn_penalty_weight=args.turn_penalty_weight,
            )
            raw_astar_geo = ([rowcol_to_geo(r, c, transform) for r, c in a_path]
                             if a_path else None)
            geo_astar = [rowcol_to_geo(r, c, transform) for r, c in a_path] if a_path else None
            if geo_astar:
                geo_astar = smooth_path_with_constraints(
                    geo_astar, transform, hard_mask, cost_grid, dem,
                    args.grade_penalty_weight, args.turn_penalty_weight,
                    route_pixel_spacing, max_ground_grade=args.max_ground_grade,
                    crs=crs, shortcut_endpoints=shortcut_endpoints)
            geo_astar, grade_reports[astar_method] = filter_path_by_ground_grade(
                geo_astar, dem, transform, args.max_ground_grade, crs)
            methods[astar_method] = geo_astar
            t_astar = time.time() - t_astar
            print(f"  [{astar_method}] 路径 {len(geo_astar) if geo_astar else 0} 点, 耗时 {t_astar:.1f}s")

            if args.experimental_theta_star:
                t_theta = time.time()
                theta_path = theta_star_cost_path(
                    cost_grid, hard_mask, (sr, sc), (er, ec),
                    pixel_spacing=route_pixel_spacing)
                geo_theta = ([rowcol_to_geo(r, c, transform) for r, c in theta_path]
                             if theta_path else None)
                if geo_theta:
                    geo_theta = smooth_path_with_constraints(
                        geo_theta, transform, hard_mask, cost_grid, dem,
                        0.0, 0.0, route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints)
                geo_theta, grade_reports["Theta*(experimental)"] = filter_path_by_ground_grade(
                    geo_theta, dem, transform, args.max_ground_grade, crs)
                methods["Theta*(experimental)"] = geo_theta
                t_theta = time.time() - t_theta
                print(f"  [Theta*(experimental)] 路径 {len(geo_theta) if geo_theta else 0} 点, "
                      f"耗时 {t_theta:.1f}s；仅作对照候选")

            station_offsets_by_method = {}
            station_measurements = {}
            hard_violation_reports = {}
            if station_task:
                for method, geo in list(methods.items()):
                    if not geo:
                        continue
                    geo, offsets = attach_station_endpoints(geo, station_task)
                    geo, grade_report = filter_path_by_ground_grade(
                        geo, dem, transform, args.max_ground_grade, crs)
                    geo, hard_report = validate_route_hard_constraints(geo, transform, hard_mask)
                    if geo and osm_layer_metadata.get("buildings", {}).get("status") == "available":
                        geo, building_report = validate_route_against_buildings(
                            geo, building_geometries)
                        hard_report.update(building_report)
                        if building_report["building_geometry_status"] == "fail":
                            hard_report["hard_constraint_status"] = "fail"
                            hard_report["hard_violations"] += 1
                    annotate_route_validation_context(
                        hard_report, endpoint_building_exceptions,
                        osm_layer_metadata.get("buildings", {}))
                    methods[method] = geo
                    grade_reports[method] = grade_report
                    hard_violation_reports[method] = hard_report
                    station_offsets_by_method[method] = offsets
                    if geo:
                        print(f"  [{method}] 站点端点恢复后的硬约束复核通过")
                    elif hard_report["hard_constraint_status"] == "fail":
                        print(f"  [{method}] 站点端点恢复后穿越硬禁区，拒绝该候选")
            else:
                for method, geo in list(methods.items()):
                    geo = expand_single_cell_route(geo, (slat, slon), (elat, elon))
                    geo = attach_input_endpoints(geo, (slat, slon), (elat, elon))
                    geo, grade_reports[method] = filter_path_by_ground_grade(
                        geo, dem, transform, args.max_ground_grade, crs)
                    geo, hard_report = validate_route_hard_constraints(
                        geo, transform, hard_mask)
                    if geo and osm_layer_metadata.get("buildings", {}).get("status") == "available":
                        geo, building_report = validate_route_against_buildings(
                            geo, building_geometries)
                        hard_report.update(building_report)
                        if building_report["building_geometry_status"] == "fail":
                            hard_report["hard_constraint_status"] = "fail"
                            hard_report["hard_violations"] += 1
                    annotate_route_validation_context(
                        hard_report, endpoint_building_exceptions,
                        osm_layer_metadata.get("buildings", {}))
                    hard_violation_reports[method] = hard_report
                    methods[method] = geo

            # 指标
            start_ll = (slat, slon); end_ll = (elat, elon)
            case_metrics = {}
            for m, geo in methods.items():
                if geo:
                    method_turn_weight = (args.turn_penalty_weight
                                          if m == "A*(turn-aware)" else 0.0)
                    met = compute_metrics(
                        geo, dem, transform, cost_grid, hard_mask, start_ll, end_ll,
                        grade_penalty_weight=args.grade_penalty_weight,
                        turn_penalty_weight=method_turn_weight,
                        pixel_spacing=route_pixel_spacing)
                    met["weight_independent_exposure"] = route_layer_exposure(
                        geo, transform, dem.shape, terrain, osm,
                        official_hazard_masks, osm_layer_metadata,
                        official_hazard_metadata.get("coverage_status", "unknown"),
                        official_hazard_metadata.get("layers", {}))
                    if m == astar_method and raw_astar_geo:
                        met["astar_search_path_objective"] = astar_path_objective(
                            a_path, cost_grid, dem, route_pixel_spacing,
                            args.grade_penalty_weight, args.turn_penalty_weight)
                    if m in ("A*(baseline)", "A*(weighted)"):
                        met["astar_heuristic_weight"] = args.astar_heuristic_weight
                        met["ridge_penalty_weight"] = args.ridge_penalty_weight
                        met["landcover_penalty_weight"] = args.landcover_penalty_weight
                        met["official_hazard_weight"] = args.official_hazard_weight
                    if m == "A*(turn-aware)":
                        met["astar_heuristic_weight"] = args.astar_heuristic_weight
                        met["turn_penalty_weight"] = args.turn_penalty_weight
                        met["ridge_penalty_weight"] = args.ridge_penalty_weight
                        met["landcover_penalty_weight"] = args.landcover_penalty_weight
                        met["official_hazard_weight"] = args.official_hazard_weight
                    met["time_s"] = round({
                        "VIN-Grad": t_vin,
                        "PPO": t_ppo,
                        "Theta*(experimental)": t_theta,
                        "A*(baseline)": t_astar,
                        "A*(weighted)": t_astar,
                        "A*(turn-aware)": t_astar,
                    }.get(m, 0.0), 2)
                    if not station_task:
                        start_center = rowcol_to_geo(sr, sc, transform)
                        end_center = rowcol_to_geo(er, ec, transform)
                        met["start_input_to_grid_center_offset_m"] = round(
                            haversine_km(slon, slat, *start_center) * 1000.0, 2)
                        met["end_input_to_grid_center_offset_m"] = round(
                            haversine_km(elon, elat, *end_center) * 1000.0, 2)
                elif m in grade_reports:
                    met = {}
                else:
                    continue
                if m in grade_reports:
                    met.update(grade_reports[m])
                met.update(hard_violation_reports.get(m, {}))
                if geo and "hard_constraint_status" not in met:
                    met["hard_constraint_status"] = (
                        "pass" if met.get("hard_violations", 0) == 0 else "fail")
                met.update(station_offsets_by_method.get(m, {}))
                case_metrics[m] = met

            # 按固定优先规则筛选硬约束可行路线，再单独读取真实线作事后评分。
            best_method, _, selection_rule = _select_final_route(methods, case_metrics)
            final_geo = methods.get(best_method) if best_method else None
            final_is_ai = False if best_method else None
            osm_data_quality, missing_osm_layers = screening_data_quality(osm_layer_metadata)
            hazard_coverage_complete = (
                official_hazard_metadata.get("coverage_status") == "complete")
            missing_hazard_layers = (
                [] if args.official_hazard_weight == 0 or hazard_coverage_complete
                else ["official_hazards"])
            missing_screening_layers = missing_osm_layers + missing_hazard_layers
            screening_quality = (
                "incomplete_screening_layers" if missing_screening_layers
                else "configured_layers_present_osm_coverage_unverified")
            hazard_route_exposure = route_hazard_exposure(
                final_geo, official_hazard_masks, transform, dem.shape)
            selected_metrics = case_metrics.get(best_method, {}) if best_method else {}

            real_evaluation_status = "not_requested"
            real_evaluation_methods = []
            if use_real:
                real_evaluation_status = "reference_missing"
                if global_real_exists:
                    real_geo, real_evaluation_status = load_case_reference(
                        args.real_shp, require_single_line=True)
                else:
                    try:
                        rp = resolve_real_path(
                            cname, args.real_dir,
                            include_global_mapping=allow_global_reference_mapping)
                    except AmbiguousReferenceError as exc:
                        print(f"  [warn] 目标线路参考文件匹配不唯一: {exc}")
                        real_geo, real_evaluation_status = None, "reference_ambiguous"
                    else:
                        real_geo, real_evaluation_status = load_case_reference(rp)
                if real_geo:
                    print(f"  [POST-EVAL] 加载真实线路: {len(real_geo)}条折线")
                    grid_corners = [transform * point for point in (
                        (0, 0), (W, 0), (W, H), (0, H))]
                    planning_bounds = (
                        min(point[0] for point in grid_corners),
                        min(point[1] for point in grid_corners),
                        max(point[0] for point in grid_corners),
                        max(point[1] for point in grid_corners),
                    )
                    for method, geo in methods.items():
                        if not geo or method not in case_metrics:
                            continue
                        dev = compute_deviation(
                            geo, real_geo, planning_bounds_wgs84=planning_bounds)
                        if not dev:
                            continue
                        metrics = case_metrics[method]
                        err = eval_planned_route_error(geo, dev)
                        if err:
                            metrics.update(err)
                        real_evaluation_methods.append(method)
                        if err:
                            similarity = err["route_similarity"]
                            symmetric = similarity["symmetric"]
                            coverage_100m = symmetric["coverage_f1_within_m"]["100"]
                            coverage_1km = symmetric["coverage_f1_within_m"]["1000"]
                            print(f"  [{method}] 较差方向P95={symmetric['p95_m']:.0f}m "
                                  f"采样最大偏移={symmetric['sampled_hausdorff_m']:.0f}m "
                                  f"100m覆盖F1={coverage_100m:.1%} "
                                  f"1km覆盖F1={coverage_1km:.1%} "
                                  f"长度差={similarity['length']['absolute_difference_km']:.2f}km")
                    real_evaluation_status = (
                        "comparison_completed" if real_evaluation_methods
                        else "reference_loaded_no_comparable_route")

            selected_metrics = case_metrics.get(best_method, {}) if best_method else {}
            planning_status = (
                "no_hard_constraint_compliant_route" if not final_geo else
                "screening_candidate_data_incomplete" if missing_screening_layers else
                "screening_candidate")
            case_metrics[FINAL_ROUTE_LABEL] = {
                "method": best_method,
                "selection_rule": selection_rule,
                "route_similarity_version": selected_metrics.get("route_similarity_version"),
                "route_similarity": selected_metrics.get("route_similarity"),
                "is_ai_method": final_is_ai,
                "real_evaluation_status": real_evaluation_status,
                "real_evaluation_methods": real_evaluation_methods,
                "planning_status": planning_status,
                "hard_constraint_status": selected_metrics.get("hard_constraint_status", "no_path"),
                "hard_violations": selected_metrics.get("hard_violations"),
                "osm_data_quality": osm_data_quality,
                "screening_data_quality": screening_quality,
                "missing_osm_screening_layers": missing_osm_layers,
                "missing_screening_layers": missing_screening_layers,
                "official_hazard_metadata": official_hazard_metadata,
                "official_hazard_route_exposure": hazard_route_exposure,
                "constraint_layer_metadata": custom_constraint_metadata,
                "length_km": round(path_length_km(final_geo), 3) if final_geo else None}
            case_metrics[FINAL_ROUTE_LABEL].update(selected_metrics)
            if final_geo:
                case_metrics[FINAL_ROUTE_LABEL].update(route_profile_metrics(final_geo, dem, transform))
            if best_method in grade_reports:
                case_metrics[FINAL_ROUTE_LABEL].update(grade_reports[best_method])
            case_metrics[FINAL_ROUTE_LABEL].update(station_offsets_by_method.get(best_method, {}))
            case_metrics[FINAL_ROUTE_LABEL].update({
                "settlement_weight": args.settlement_weight,
                "official_hazard_weight": args.official_hazard_weight,
                "turn_penalty_weight": args.turn_penalty_weight,
                "ridge_penalty_weight": args.ridge_penalty_weight,
                "landcover_penalty_weight": args.landcover_penalty_weight,
                "building_clearance_m": args.building_clearance_m,
                "grade_penalty_weight": args.grade_penalty_weight,
                "terrain_slope_weight": args.terrain_slope_weight,
                "roughness_weight": args.roughness_weight,
                "max_ground_grade_ratio": args.max_ground_grade,
                "max_surface_slope_deg": args.max_surface_slope_deg,
                "max_terrain_elevation_m": args.max_terrain_elevation_m,
                "osm_layer_metadata": osm_layer_metadata,
                "voltage_rules_applied": False,
                "route_constraints_limitations": (
                    "No population/sensitive-target inventory, official protected-area boundaries, "
                    "conductor clearance, tower siting or span feasibility checked"),
            })
            if station_task:
                start_station = station_task["from_station"]
                end_station = station_task["to_station"]
                station_measurements = station_elevation_metrics(station_task)
                for metrics in case_metrics.values():
                    if isinstance(metrics, dict):
                        metrics.update({
                            "route_id": cname,
                            "from_station_id": start_station["station_id"],
                            "from_station_name": start_station["name"],
                            "from_station_type": start_station["station_type"] or None,
                            "from_station_source": start_station["source"] or None,
                            "from_station_verified": start_station["verified"] or None,
                            "to_station_id": end_station["station_id"],
                            "to_station_name": end_station["name"],
                            "to_station_type": end_station["station_type"] or None,
                            "to_station_source": end_station["source"] or None,
                            "to_station_verified": end_station["verified"] or None,
                            "voltage_kv": station_task["voltage_kv"] or None,
                            **station_measurements,
                            "settlement_weight": args.settlement_weight,
                            "ridge_penalty_weight": args.ridge_penalty_weight,
                            "landcover_penalty_weight": args.landcover_penalty_weight,
                            "building_clearance_m": args.building_clearance_m,
                            "grade_penalty_weight": args.grade_penalty_weight,
                            "terrain_slope_weight": args.terrain_slope_weight,
                            "roughness_weight": args.roughness_weight,
                            "max_ground_grade_ratio": args.max_ground_grade,
                            "max_surface_slope_deg": args.max_surface_slope_deg,
                            "max_terrain_elevation_m": args.max_terrain_elevation_m,
                            "route_constraints_limitations": "No population layer, official protected-area layer, conductor clearance, tower siting or span feasibility checked",
                        })
            print(f"  [集成] 最终路线采用 {best_method} ({selection_rule})；线路相似度仅作事后指标")
            # 导出最终路线矢量 (GeoJSON)
            safe_now = route_output_name(cname)
            final_dir = OUT / safe_now
            final_dir.mkdir(parents=True, exist_ok=True)
            try:
                if final_geo:
                    _fg = [{"type": "Feature",
                            "geometry": {"type": "LineString",
                                         "coordinates": geojson_coordinates(final_geo)},
                            "properties": {"case": cname, "method": best_method,
                                            "route_similarity_version": selected_metrics.get("route_similarity_version"),
                                            "route_similarity": selected_metrics.get("route_similarity"),
                                           "selection_rule": selection_rule,
                                           "planning_status": planning_status,
                                           "hard_constraint_status": selected_metrics.get("hard_constraint_status", "no_path"),
                                           "hard_violations": selected_metrics.get("hard_violations"),
                                           "osm_data_quality": osm_data_quality,
                                           "screening_data_quality": screening_quality,
                                           "missing_osm_screening_layers": missing_osm_layers,
                                           "missing_screening_layers": missing_screening_layers,
                                           "official_hazard_metadata": official_hazard_metadata,
                                           "official_hazard_route_exposure": hazard_route_exposure,
                                           "constraint_layer_metadata": custom_constraint_metadata,
                                           "length_km": selected_metrics.get("length_km"),
                                           "start_input_to_grid_center_offset_m": selected_metrics.get("start_input_to_grid_center_offset_m"),
                                           "end_input_to_grid_center_offset_m": selected_metrics.get("end_input_to_grid_center_offset_m"),
                                           "mean_cost": selected_metrics.get("mean_cost"),
                                           "path_cost_integral": selected_metrics.get("path_cost_integral"),
                                           "astar_search_path_objective": selected_metrics.get("astar_search_path_objective"),
                                           "sinuosity": selected_metrics.get("sinuosity"),
                                           "cum_relief_m": selected_metrics.get("cum_relief_m"),
                                           "net_relief_m": selected_metrics.get("net_relief_m"),
                                           "from_station_id": station_task["from_station"]["station_id"] if station_task else None,
                                           "to_station_id": station_task["to_station"]["station_id"] if station_task else None,
                                           "from_station_name": station_task["from_station"]["name"] if station_task else None,
                                           "to_station_name": station_task["to_station"]["name"] if station_task else None,
                                           "from_station_source": station_task["from_station"]["source"] or None if station_task else None,
                                           "from_station_verified": station_task["from_station"]["verified"] or None if station_task else None,
                                           "to_station_source": station_task["to_station"]["source"] or None if station_task else None,
                                           "to_station_verified": station_task["to_station"]["verified"] or None if station_task else None,
                                           "voltage_kv": station_task["voltage_kv"] or None if station_task else None,
                                           **station_measurements,
                                           "voltage_rules_applied": False,
                                           "settlement_weight": args.settlement_weight,
                                           "official_hazard_weight": args.official_hazard_weight,
                                           "turn_penalty_weight": args.turn_penalty_weight,
                                           "ridge_penalty_weight": args.ridge_penalty_weight,
                                           "landcover_penalty_weight": args.landcover_penalty_weight,
                                           "building_clearance_m": args.building_clearance_m,
                                           "grade_penalty_weight": args.grade_penalty_weight,
                                           "terrain_slope_weight": args.terrain_slope_weight,
                                           "roughness_weight": args.roughness_weight,
                                           "max_ground_grade_ratio": args.max_ground_grade,
                                           "max_surface_slope_deg": args.max_surface_slope_deg,
                                           "max_terrain_elevation_m": args.max_terrain_elevation_m,
                                           "osm_layer_metadata": osm_layer_metadata,
                                           "route_constraints_limitations": (
                                               "No population/sensitive-target inventory, official protected-area boundaries, "
                                               "conductor clearance, tower siting or span feasibility checked"),
                                           **grade_reports.get(best_method, {}),
                                           **station_offsets_by_method.get(best_method, {}),
                                           **(route_profile_metrics(final_geo, dem, transform) if final_geo else {})}}]
                    with open(final_dir / "final_route.geojson", "w", encoding="utf-8") as _ff:
                        json.dump({"type": "FeatureCollection", "features": _fg}, _ff, ensure_ascii=False)
                    if final_geo:
                        write_elevation_profile_csv(final_geo, dem, transform, final_dir / "elevation_profile.csv")
            except Exception as exc:
                raise RuntimeError(f"最终路线必需文件写出失败: {exc}") from exc
            all_results.append({"case": cname, "metrics": case_metrics})
            # 增量写出真实对比 JSON: 任一走廊完成后即落盘, 即使后续走廊异常/SIGKILL 终止也能保留已完成结果
            try:
                _real_out = []
                for _rec in all_results:
                    _entry = {"case": _rec["case"]}
                    for _m, _met in _rec["metrics"].items():
                        if "route_similarity" in _met:
                            _entry[_m] = {
                                "method": _met.get("method", _m),
                                "length_km": _met.get("length_km"),
                                "route_similarity_version": _met.get("route_similarity_version"),
                                "route_similarity": _met["route_similarity"],
                            }
                    if _entry:
                        _real_out.append(_entry)
                with open(OUT / "real_comparison_metrics.json", "w", encoding="utf-8") as _f:
                    json.dump(_real_out, _f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            # 每走廊结束释放 GPU 显存, 避免 RTX 4060(8GB) 在多走廊连续训练时累积 OOM
            if HAS_TORCH:
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

            # 逐走廊绘制详细路径对比图 + 导出矢量（每个用例一张对比图, 满足"多条路径对比"）
            safe = route_output_name(cname)
            cdir = OUT / safe
            cdir.mkdir(parents=True, exist_ok=True)
            plot_comparison(dem, transform, methods, start_ll, end_ll,
                            cdir / f"path_comparison_{safe}.png",
                            title=f"多算法路径对比 - {cname}", real_ll=real_geo,
                            hazard_masks=official_hazard_masks)
            # 逐用例导出矢量与坐标(便于复用/二次绘图)
            for m, geo in methods.items():
                tag = m.split("(")[0].strip().lower().replace("*", "astar").replace("-", "")
                export_route(geo, crs, cdir / f"planned_route_{tag}.shp",
                             cdir / f"planned_route_{tag}.geojson")
            # 导出真实/参考线路(便于下游复用与二次对比)
            if real_geo:
                export_route(real_geo, crs, cdir / "real_route.shp", cdir / "real_route.geojson")
            paths_json = {m: [[round(lon, 6), round(lat, 6)] for lon, lat in geo]
                          for m, geo in methods.items() if geo}
            with open(cdir / "paths.json", "w", encoding="utf-8") as f:
                json.dump(paths_json, f, ensure_ascii=False)

            # 仅首个用例额外交互兼容产物(单图 + 值图 + 主方法指标)
            OUT.mkdir(parents=True, exist_ok=True)
            if not main_done:
                # 立即置位, 防止任一子步骤异常被外层 per-corridor except 捕获后 main_done 未设置,
                # 导致后续走廊重新进入本块并错误覆盖根目录产物。
                main_done = True
                try:
                    plot_comparison(dem, transform, methods, start_ll, end_ll,
                                    OUT / "path_comparison.png", title=f"多算法路径对比 - {cname}",
                                    real_ll=real_geo, hazard_masks=official_hazard_masks)
                except Exception as _pe:
                    print(f"[WARN] 主对比图绘制失败(跳过, 不影响本走廊其余产出): {_pe}")
                if "VIN-Grad" in methods and HAS_TORCH and V_main is not None:
                    # 复用主流程已计算的 V（避免重复 80 步 Bellman 训练）
                    try:
                        plot_value_map(V_main, transform, methods["VIN-Grad"], start_ll, end_ll, OUT / "value_map.png")
                    except Exception as _ve:
                        print(f"[WARN] 值图绘制失败(跳过): {_ve}")
                if "VIN-Grad" in case_metrics:
                    try:
                        with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
                            json.dump(case_metrics["VIN-Grad"], f, ensure_ascii=False, indent=2)
                    except Exception as _me:
                        print(f"[WARN] metrics.json 写出失败(跳过): {_me}")
        except Exception as _exc:
            import traceback as _tb
            _tb.print_exc()
            print(f"[WARN] 用例 {cname} 处理失败, 跳过此用例: {_exc}")
            failed_cases.append({"case": cname, "error": str(_exc)})

    run_summary = {
        "scope": "per_case_planning_and_required_route_artifacts",
        **summarize_run_completion(len(cases), all_results, failed_cases),
        "real_evaluation_by_case": [
            {"case": record["case"],
             "status": record["metrics"].get(FINAL_ROUTE_LABEL, {}).get("real_evaluation_status", "not_recorded"),
             "methods": record["metrics"].get(FINAL_ROUTE_LABEL, {}).get("real_evaluation_methods", [])}
            for record in all_results
        ],
        "dem": str(dem_path),
        "max_side": args.max_side,
        "argv": sys.argv,
    }
    try:
        with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
    except Exception as _exc:
        print(f"[WARN] 批次运行摘要写出失败: {_exc}")
    print(f"[批次状态] 应测 {len(cases)} 条，完成 {run_summary['completed_cases']} 条，"
          f"无可行路线 {len(run_summary['no_route_cases'])} 条，失败 {len(failed_cases)} 条")

    # 聚合对比图 + 指标表
    try:
        plot_algo_comparison(all_results, OUT / "algo_comparison.png")
        with open(OUT / "comparison_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    except Exception as _exc2:
        import traceback as _tb2
        _tb2.print_exc()
        print(f"[WARN] 聚合对比图/指标写出失败(不影响各走廊已写出结果): {_exc2}")
    # CSV
    rows = []
    for rec in all_results:
        for m, met in rec["metrics"].items():
            row = {"case": rec["case"], "method": m, **met}
            if "route_similarity" in row:
                row["route_similarity_json"] = json.dumps(
                    row.pop("route_similarity"), ensure_ascii=False)
            rows.append(row)
    if rows:
        import csv
        # 用所有行的键并集作为表头, 避免不同方法(A*/VIN/PPO)或有无真实对比导致字段不一致
        all_keys = []
        for r in rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(OUT / "comparison_metrics.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader(); w.writerows(rows)

    # 参考线路空间相似度 JSON + 控制台摘要
    if use_real:
        real_out = []
        for rec in all_results:
            entry = {"case": rec["case"]}
            for m, met in rec["metrics"].items():
                if "route_similarity" in met:
                    entry[m] = {
                        "method": met.get("method", m),
                        "length_km": met.get("length_km"),
                        "route_similarity_version": met.get("route_similarity_version"),
                        "route_similarity": met["route_similarity"],
                    }
            if entry:
                real_out.append(entry)
        try:
            with open(OUT / "real_comparison_metrics.json", "w", encoding="utf-8") as f:
                json.dump(real_out, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print("\n" + "=" * 70)
        print("  参考线路空间相似度 (双向米制偏移 + 线路覆盖；仅事后评估)")
        print("=" * 70)
        print(f"  {'走廊':16s} {'方法':16s} {'较差方向P95(m)':>14s} {'采样最大(m)':>12s} "
              f"{'F1@100m':>9s} {'F1@1km':>9s} {'长度差(km)':>11s}")
        candidate_methods = ["A*(baseline)", "A*(weighted)", "A*(turn-aware)",
                             "Theta*(experimental)", "VIN-Grad", "PPO"]
        failed_case_names = {item["case"] for item in failed_cases}
        for rec in all_results:
            if rec["case"] in failed_case_names or rec["metrics"].get(FINAL_ROUTE_LABEL, {}).get("planning_status") == "no_hard_constraint_compliant_route":
                continue
            for m in candidate_methods:
                met = rec["metrics"].get(m, {})
                similarity = met.get("route_similarity")
                if similarity:
                    symmetric = similarity["symmetric"]
                    print(f"  {rec['case']:16s} {m:16s} "
                          f"{symmetric['p95_m']:11.0f} "
                          f"{symmetric['sampled_hausdorff_m']:12.0f} "
                          f"{symmetric['coverage_f1_within_m']['100']:8.1%} "
                          f"{symmetric['coverage_f1_within_m']['1000']:8.1%} "
                          f"{similarity['length']['absolute_difference_km']:11.2f}")
        print("  说明：误差与覆盖率只作事后诊断，不设贴合率达标线，也不参与选路。")
        print("-" * 70)
        for m in candidate_methods:
            summary = method_visual_similarity_summary(
                all_results, m, expected_cases=len(cases),
                failed_case_names=failed_case_names,
            )
            p95 = summary["median_symmetric_p95_m"]
            max_offset = summary["median_sampled_hausdorff_m"]
            f1_100m = summary["median_coverage_f1_100m"]
            f1 = summary["median_coverage_f1_1km"]
            length_difference = summary["median_length_difference_pct"]
            p95_text = f"{p95:.1f}m" if p95 is not None else "NA"
            max_text = f"{max_offset:.1f}m" if max_offset is not None else "NA"
            f1_100m_text = f"{f1_100m:.1%}" if f1_100m is not None else "NA"
            f1_text = f"{f1:.1%}" if f1 is not None else "NA"
            length_text = f"{length_difference:.1f}%" if length_difference is not None else "NA"
            print(f"  {m:16s} 已测={summary['measured']:2d} 未测={summary['unmeasured']:2d} "
                  f"中位较差方向P95={p95_text} 中位最大偏移={max_text} "
                  f"中位F1@100m={f1_100m_text} 中位F1@1km={f1_text} "
                  f"中位长度差={length_text}")
        heuristic_selected = [r["case"] for r in all_results
                          if r["case"] not in failed_case_names
                          and r["metrics"].get(FINAL_ROUTE_LABEL, {}).get("planning_status")
                          != "no_hard_constraint_compliant_route"
                          if r["metrics"].get(FINAL_ROUTE_LABEL, {}).get("method")
                          in ("A*(baseline)", "A*(weighted)", "A*(turn-aware)")]
        if heuristic_selected:
            print(f"  启发式搜索生成最终路线的走廊: {heuristic_selected}")
        print("=" * 70)

    # 控制台摘要
    print("\n" + "=" * 70)
    print("  结果摘要（各走廊均值）")
    print("=" * 70)
    for m in ["A*(baseline)", "A*(weighted)", "A*(turn-aware)",
              "Theta*(experimental)", "VIN-Grad", "PPO"]:
        lens = [r["metrics"][m]["length_km"] for r in all_results if m in r["metrics"]]
        mc = [r["metrics"][m]["mean_cost"] for r in all_results if m in r["metrics"]]
        if lens:
            print(f"  {m:14s} 平均长度 {np.mean(lens):7.2f}km  平均代价 {np.mean(mc):.3f}")
    print(f"  总耗时: {time.time() - t0:.1f}s")
    print(f"  输出目录: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
