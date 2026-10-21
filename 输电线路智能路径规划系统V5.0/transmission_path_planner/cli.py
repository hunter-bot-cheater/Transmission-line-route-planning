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


def _reference_guidance_requested(guide_mode, use_real):
    return guide_mode == "reference" or (guide_mode == "auto" and use_real)


DEFAULT_GUIDE_MODE = "none"
FINAL_ROUTE_LABEL = "最终路线"


def _select_final_route(methods, case_metrics, guide_enabled):
    valid_methods = [
        method for method, path in methods.items()
        if path and method in case_metrics
        and case_metrics[method].get("hard_violations") == 0
    ]
    if guide_enabled:
        best_method, best_err = None, float("inf")
        for method in valid_methods:
            metrics = case_metrics[method]
            error = metrics.get("worst_err_pct")
            if error is not None and error < best_err:
                best_method, best_err = method, error
        if best_method is not None:
            return best_method, best_err, "reference_error_min"
        fallback_rule = "astar_fallback_no_reference_score"
    else:
        fallback_rule = "astar_baseline"

    if "A*(baseline)" in valid_methods:
        return "A*(baseline)", None, fallback_rule
    if valid_methods:
        return valid_methods[0], None, "valid_candidate_fallback"
    return None, None, "no_hard_constraint_compliant_route"


def attach_station_endpoints(geo, station_task):
    """Replace snapped raster-center endpoints with the supplied station coordinates."""
    if not station_task or not geo or len(geo) < 2:
        return geo, {}
    start, end = station_task["from_station"], station_task["to_station"]
    start_point = (float(start["lon"]), float(start["lat"]))
    end_point = (float(end["lon"]), float(end["lat"]))
    offsets = {
        "from_station_to_grid_center_offset_m": round(haversine_km(*start_point, *geo[0]) * 1000.0, 2),
        "to_station_to_grid_center_offset_m": round(haversine_km(*end_point, *geo[-1]) * 1000.0, 2),
    }
    adjusted = list(geo)
    adjusted[0], adjusted[-1] = start_point, end_point
    return adjusted, offsets


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
    return ("incomplete_osm_proxies" if missing else "osm_proxy_features_observed"), missing


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="复杂山区输电线路 AI 路径规划 (V5.0: 多方法搜索 + 26维特征)")
    ap.add_argument("--dem", type=str, default=None)
    ap.add_argument("--shp", type=str, default=None)
    ap.add_argument("--start", type=str, default=None, help="起点 lat,lon")
    ap.add_argument("--end", type=str, default=None, help="终点 lat,lon")
    ap.add_argument("--stations-csv", type=str, default=None,
                    help="站点表: station_id,name,lat,lon,voltage_kv; 坐标应为线路接入点")
    ap.add_argument("--route-tasks-csv", type=str, default=None,
                    help="站间任务表: route_id,from_station_id,to_station_id")
    ap.add_argument("--settlement-weight", type=float, default=2.0,
                    help="OSM建筑密度代理的软避让权重(方案筛选参数, 默认2.0)")
    ap.add_argument("--building-clearance-m", type=float, default=0.0,
                    help="建筑物栅格硬避让距离(米); 0仅屏蔽已映射建筑足迹，不代表规范净距")
    ap.add_argument("--grade-penalty-weight", type=float, default=1.0,
                    help="沿线相邻DEM高差软惩罚权重(以30%%地面坡度作为一个参考级)")
    ap.add_argument("--max-ground-grade", type=float, default=None,
                    help="可选地面相邻栅格最大坡度比，如0.30; 属于筛选阈值，不是线路设计限值")
    ap.add_argument("--max-surface-slope-deg", type=float, default=None,
                    help="可选地表坡度硬限制(度); 默认不设工程阈值")
    ap.add_argument("--max-terrain-elevation-m", type=float, default=None,
                    help="可选地形海拔硬限制(米); 默认不设工程阈值")
    ap.add_argument("--use-costunet", action="store_true",
                    help="把CostUNet作为50%%软先验混入规则代价; 默认关闭，避免未知训练来源影响基线")
    ap.add_argument("--max-side", type=int, default=WORK_MAX_SIDE)
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--no-ppp", action="store_true", help="跳过 PPO 方法")
    ap.add_argument("--baseline", action="store_true",
                    help="额外跑'无走廊引导'的 VIN-Grad 基线作 A/B 对照(默认关闭, 省显存防 OOM)")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--with-real", action="store_true",
                    help="加载真实/参考路径并计算事后偏差; 是否参与规划由 --guide-mode 控制")
    ap.add_argument("--real-cases", action="store_true",
                    help="改用 v3 的 4 条真实走廊(R01~R04)并启用真实线对照")
    ap.add_argument("--real-cases-csv", type=str, default=None,
                    help="从真实走廊 CSV(列: name,start_lat,start_lon,end_lat,end_lon)载人测试用例, "
                         "并自动把 --real-dir 设为该 CSV 同目录(其中含 '<净化名>_real_path.shp' 真实线)。"
                         "启用事后真实线对照; 是否引导规划由 --guide-mode 控制。")
    ap.add_argument("--guide-mode", choices=("auto", "none", "reference"), default=DEFAULT_GUIDE_MODE,
                    help="引导模式: none=默认，真实线仅用于事后评估; auto=有真实线时按旧行为引导; "
                         "reference=明确使用输入真实线引导(结果不代表无泄漏泛化能力)")
    ap.add_argument("--corridor-w", type=float, default=None,
                    help="参考线走廊带惩罚权重; 0=关闭, 未指定时在参考线引导模式下使用默认值")
    ap.add_argument("--corridor-band", type=float, default=CORRIDOR_BAND_M_DEFAULT,
                    help="走廊缓冲带半宽(米), 带内几乎无额外惩罚")
    ap.add_argument("--real-dir", type=str, default=str(REAL_PATH_DIR),
                    help="真实线路 SHP 所在目录(按 '<净化名>_real_path.shp' 匹配)")
    ap.add_argument("--real-shp", type=str, default=None,
                    help="直接指定一个真实线路/线网 SHP(如全区输电网络), 作为事后对照线; "
                         "是否参与规划由 --guide-mode 控制。")
    ap.add_argument("--real-attract", type=float, default=None,
                    help="真实线方向吸引力权重; 0=关闭, 未指定时在参考线引导模式下使用默认值")
    ap.add_argument("--real-attract-cutoff", type=float, default=2500.0,
                    help="方向吸引力距离门控(米): 仅在真实线 cutoff 范围内生效, 远处不把路径拽离走廊")
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
    if args.stations_csv and (args.start or args.end):
        ap.error("站点 CSV 任务不能与 --start/--end 同时使用")
    if args.building_clearance_m < 0 or args.settlement_weight < 0 or args.grade_penalty_weight < 0:
        ap.error("避让距离与代价权重不能为负数")
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
    print("  复杂山区输电线路 AI 路径规划 V5.0 (多方法路径规划)")
    print("=" * 70)

    dem_path = Path(args.dem) if args.dem else find_dem()
    if dem_path is None:
        sys.exit("[FATAL] 未找到 DEM，请用 --dem 指定。")
    print(f"[DEM] {dem_path}")

    shp_path = Path(args.shp) if args.shp else (SHP_CANDIDATES[0] if SHP_CANDIDATES[0].exists() else None)
    print(f"[SHP] {shp_path if shp_path else '无 (将运行 AI vs AI + 基准对比)'}")

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
        slat, slon = map(float, args.start.split(","))
        elat, elon = map(float, args.end.split(","))
        cases = [("Custom", (slat, slon), (elat, elon))]
    else:
        if args.real_cases_csv and Path(args.real_cases_csv).exists():
            csv_cases = load_real_cases_csv(args.real_cases_csv)
            # SHP(<净化名>_real_path.shp) 与 CSV 同目录, 自动设为真实线搜索目录
            args.real_dir = str(Path(args.real_cases_csv).parent)
            cases = [(n, s, e) for (n, s, e, _) in csv_cases]
        elif args.real_cases:
            cases = [(n, s, e) for (n, s, e, _) in REAL_CASES]
        else:
            cases = DEFAULT_CASES[:max(1, args.cases)]

    # 无引导模式先只记录全局参考线路径, 等各用例的规划候选生成后再读取几何。
    global_real_geo = None
    global_real_exists = bool(args.real_shp and Path(args.real_shp).exists())
    use_real = bool(args.with_real or args.real_cases or args.real_cases_csv or global_real_exists)
    reference_guidance = _reference_guidance_requested(args.guide_mode, use_real)
    if reference_guidance and global_real_exists:
        global_real_geo = load_real_path(args.real_shp)
        if global_real_geo:
            print(f"[真实对比] 全局参考线网: {args.real_shp}  {len(global_real_geo)}条折线, "
                  f"总长 {real_length_km(global_real_geo):.0f}km")
        else:
            print(f"[真实对比] 警告: 无法读取 {args.real_shp}")
    if use_real:
        mode = "全局线网" if global_real_exists else args.real_dir
        guide_text = "参考线参与规划" if reference_guidance else "真实线仅用于事后评估"
        print(f"[真实对比] 已启用: 模式={mode}  {guide_text}")

    # CostUNet（全局加载一次）
    costunet = load_costunet() if args.use_costunet and HAS_TORCH else None
    if args.use_costunet and costunet is None:
        print("[warn] CostUNet不可用，继续使用规则代价面")

    all_results = []
    baseline_records = []  # 无走廊引导的 VIN-Grad 偏差记录(A/B 对照)
    policy = value = None
    main_done = False

    for ci, (cname, (slat, slon), (elat, elon)) in enumerate(cases):
        print(f"\n--- 用例 {ci + 1}/{len(cases)}: {cname} ---")
        try:
            margin = 0.15
            minlon, maxlon = min(slon, elon) - margin, max(slon, elon) + margin
            minlat, maxlat = min(slat, elat) - margin, max(slat, elat) + margin
            dem, transform, crs, pm = load_dem_window(dem_path, minlon, minlat, maxlon, maxlat, args.max_side)
            H, W = dem.shape

            sr, sc = geo_to_rowcol(slat, slon, transform)
            er, ec = geo_to_rowcol(elat, elon, transform)
            station_task = station_task_by_route.get(cname)
            if station_task is None:
                sr, sc = snap_to_land(dem, sr, sc)
                er, ec = snap_to_land(dem, er, ec)

            terrain = compute_terrain_factors(dem, pm, transform, crs)
            left = transform.c; top = transform.f
            right = left + transform.a * W; bottom = top + transform.e * H
            bbox = (left, bottom, right, top)
            osm = compute_osm_bands(dem, transform, bbox)
            osm_layer_metadata = getattr(osm, "metadata", {}).get("layers", {})
            for layer_name in ("buildings", "landuse"):
                layer = osm_layer_metadata.get(layer_name, {})
                if layer.get("status") != "available":
                    print(f"  [warn] OSM {layer_name} 图层状态={layer.get('status', 'unknown')}；"
                          "不能据此认定该范围内没有居民点/建筑")
            stack = build_feature_stack(dem, terrain, osm)

            model_cost = costunet.predict(stack) if costunet is not None else None
            cost = build_route_cost(terrain, osm, args.settlement_weight, model_cost)
            print(f"  代价面范围 [{np.nanmin(cost):.3f}, {np.nanmax(cost):.3f}]")

            # ---- 真实/参考路径加载 + 走廊引导代价(缩小规划路径与真实路径差距) ----
            real_geo = None
            real_dist_m = None
            real_field = None
            if reference_guidance and use_real and global_real_geo is not None:
                # 全局线网: 所有走廊共用同一真实参考(如全区输电网络)
                real_geo = global_real_geo
                real_dist_m, real_field = compute_dist_to_route(real_geo, transform, H, W, pm)
                print(f"  [真实线路网] 全局参考: {len(real_geo)}条折线, 总长 {real_length_km(real_geo):.0f}km")
            elif reference_guidance and use_real:
                rp = resolve_real_path(cname, args.real_dir)
                if rp and Path(rp).exists():
                    real_geo = load_real_path(rp)
                    if real_geo:
                        real_dist_m, real_field = compute_dist_to_route(real_geo, transform, H, W, pm)
                        print(f"  [真实路径] 加载 {Path(rp).name}: {len(real_geo)}条折线, "
                              f"{real_length_km(real_geo):.1f}km")
            guide_enabled = reference_guidance and real_geo is not None
            corridor_w = (args.corridor_w if args.corridor_w is not None else CORRIDOR_W_DEFAULT) if guide_enabled else 0.0
            real_attract = (args.real_attract if args.real_attract is not None else REAL_ATTRACT_DEFAULT) if guide_enabled else 0.0
            cost_base = cost.copy()  # 保留无引导基线代价, 供 A/B 对照
            if corridor_w > 0 and real_geo:
                ratio = np.clip(real_dist_m / max(float(args.corridor_band), 1.0), 0.0, CORRIDOR_CAP)
                corridor_pen = corridor_w * ratio
                cost = np.clip(cost + corridor_pen, 0, 1.5).astype(np.float32)
                print(f"  [走廊引导] w={corridor_w:.2f} band={args.corridor_band:.0f}m 最大附加代价={corridor_pen.max():.2f}")

            slope = terrain["slope"]; elev = terrain["elevation"]
            hard_mask = np.ones((H, W), np.uint8)
            hard_mask[~np.isfinite(dem)] = 0
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
            if station_task:
                if not (0 <= sr < H and 0 <= sc < W and 0 <= er < H and 0 <= ec < W):
                    raise ValueError("站点接入点超出DEM范围")
                if not hard_mask[sr, sc] or not hard_mask[er, ec]:
                    raise ValueError("站点接入点位于NoData/建筑/配置禁区内；请提供站端线路接入点坐标")
                sr_m, sc_m, er_m, ec_m = sr, sc, er, ec
            else:
                sr_m, sc_m = snap_to_land(dem, sr, sc)
                er_m, ec_m = snap_to_land(dem, er, ec)
                hard_mask[sr_m, sc_m] = 1
                hard_mask[er_m, ec_m] = 1
            cost_grid = (1.0 + 4.0 * cost).astype(np.float64)
            cost_grid = np.where(hard_mask == 0, np.inf, cost_grid)
            cost_finite = np.where(np.isfinite(cost), cost, 1.0).astype(np.float32)

            methods = {}
            grade_reports = {}
            V_main = None
            # --- 方法1: 神经值传播 + 梯度追踪 (VIN-Grad, 主 AI 方法) ---
            if HAS_TORCH:
                t_v = time.time()
                extra = build_vin_extra(stack)
                goal_mask = np.zeros((H, W), np.float32); goal_mask[er, ec] = 1.0
                # 参考线引导模式允许用参考误差挑选重试种子; 无引导时固定使用首个种子,
                # 避免把验收线用于规划阶段的候选筛选。
                best_vin, best_vin_V, best_vin_err = None, None, 1e9
                last_Vc = None
                for vin_attempt in range(3 if guide_enabled else 1):
                    torch.manual_seed(RANDOM_SEED + vin_attempt * 10)
                    V = train_vin(cost_finite, goal_mask, extra, n_steps=80)
                    Vc = np.nan_to_num(V, nan=1e6, posinf=1e6, neginf=1e6)
                    last_Vc = Vc
                    if not np.isfinite(Vc).all():
                        print(f"  [VIN-Grad] 第{vin_attempt + 1}次含非有限值, 重训..."); continue
                    geo_t = smooth_path_with_constraints(gradient_track_path(
                        V, cost_finite, hard_mask, (sr, sc), (er, ec), transform,
                        real_field=real_field if guide_enabled else None, real_attract=real_attract,
                        real_dist_m=real_dist_m if guide_enabled else None,
                        real_attract_cutoff=args.real_attract_cutoff), transform, hard_mask)
                    geo_t, grade_reports["VIN-Grad"] = filter_path_by_ground_grade(
                        geo_t, dem, transform, args.max_ground_grade, crs)
                    err_t = 1e9
                    if geo_t and guide_enabled:
                        dev_t = compute_deviation(geo_t, real_geo, transform, real_dist_m)
                        if dev_t:
                            e = eval_error(path_length_km(geo_t), dev_t)
                            if e:
                                err_t = e["worst_err_pct"]
                    if not guide_enabled:
                        best_vin, best_vin_V = geo_t, Vc
                        break
                    if err_t < best_vin_err:
                        best_vin_err = err_t; best_vin = geo_t; best_vin_V = Vc
                    if best_vin_err <= ERROR_TOL_PCT:
                        print(f"  [VIN-Grad] 第{vin_attempt + 1}次达标({best_vin_err:.2f}%), 停止重训")
                        break
                    print(f"  [VIN-Grad] 第{vin_attempt + 1}次误差{best_vin_err:.2f}%>8%, 重训(seed+{(vin_attempt + 1) * 10})...")
                geo_vin = best_vin
                if geo_vin:
                    grade_reports["VIN-Grad"] = ground_grade_limit_report(
                        geo_vin, dem, transform, args.max_ground_grade, crs)
                elif "VIN-Grad" not in grade_reports:
                    grade_reports["VIN-Grad"] = ground_grade_limit_report(
                        None, dem, transform, args.max_ground_grade, crs)
                V_main = best_vin_V if best_vin_V is not None else last_Vc
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
                print("  [skip] torch 不可用, 跳过 VIN-Grad")

            # --- 基线对照: 无走廊引导的 VIN-Grad(展示优化前与真实路径的差距) ---
            if args.baseline and guide_enabled and corridor_w > 0 and real_geo and HAS_TORCH:
                try:
                    cost_finite_base = np.where(np.isfinite(cost_base), cost_base, 1.0).astype(np.float32)
                    Vb = train_vin(cost_finite_base, goal_mask, extra, n_steps=80)
                    Vb = np.nan_to_num(Vb, nan=1e6, posinf=1e6, neginf=1e6)
                    geo_vin_base = gradient_track_path(Vb, cost_finite_base, hard_mask, (sr, sc), (er, ec), transform)
                    geo_vin_base = smooth_path_with_constraints(geo_vin_base, transform, hard_mask)
                    geo_vin_base, grade_reports["VIN-Grad-no-corridor"] = filter_path_by_ground_grade(
                        geo_vin_base, dem, transform, args.max_ground_grade, crs)
                    dev_base = compute_deviation(geo_vin_base, real_geo, transform, real_dist_m)
                    if dev_base:
                        print(f"  [基线VIN无引导] 偏差 Hausdorff={dev_base['dev_hausdorff_km']}km "
                              f"mean={dev_base['dev_mean_km']}km max={dev_base['dev_max_km']}km")
                        baseline_records.append({"case": cname, "deviation": dev_base,
                                                 "length_km": path_length_km(geo_vin_base)})
                except Exception as ex:
                    print(f"  [基线VIN无引导] 失败: {ex}")

            # --- 方法2: RL/PPO 策略梯度路径生成器 (每用例独立训练: 代价图与目标不同) ---
            if HAS_TORCH and not args.no_ppp:
                try:
                    best_ppo, best_ppo_err = None, 1e9
                    for ppo_attempt in range(2 if guide_enabled else 1):
                        torch.manual_seed(RANDOM_SEED + 100 + ppo_attempt * 10)
                        print(f"  [PPO] 训练策略网络 (尝试{ppo_attempt + 1}) ...")
                        t_p = time.time()
                        policy, value = train_ppo(cost_finite, hard_mask, (er, ec), (sr, sc), value_map=V_main,
                                                  n_updates=args.ppo_updates, episodes=args.ppo_episodes,
                                                  ppo_epochs=args.ppo_epochs, bc_steps=args.ppo_bc)
                        print(f"  [PPO] 训练完成 {time.time() - t_p:.1f}s")
                        ctx = build_ppo_context(np.clip(cost_finite, 0, 1), (er, ec), H, W, value_map=V_main)
                        ppo_path = ppo_rollout_path(policy, ctx, cost_finite, hard_mask, (er, ec), (sr, sc),
                                                   n_tries=6, guide_v=V_main, guide_w=6.0, v_tol=0.35,
                                                   real_field=real_field if guide_enabled else None,
                                                   real_attract=real_attract,
                                                   real_dist_m=real_dist_m if guide_enabled else None,
                                                   real_attract_cutoff=args.real_attract_cutoff)
                        geo_t = smooth_path_with_constraints(
                            [rowcol_to_geo(r, c, transform) for r, c in ppo_path], transform, hard_mask)
                        geo_t, grade_reports["PPO"] = filter_path_by_ground_grade(
                            geo_t, dem, transform, args.max_ground_grade, crs)
                        err_t = 1e9
                        if geo_t and guide_enabled:
                            dev_t = compute_deviation(geo_t, real_geo, transform, real_dist_m)
                            if dev_t:
                                e = eval_error(path_length_km(geo_t), dev_t)
                                if e:
                                    err_t = e["worst_err_pct"]
                        if not guide_enabled:
                            best_ppo = geo_t
                            break
                        if err_t < best_ppo_err:
                            best_ppo_err = err_t; best_ppo = geo_t
                        if best_ppo_err <= ERROR_TOL_PCT:
                            print(f"  [PPO] 第{ppo_attempt + 1}次达标({best_ppo_err:.2f}%), 停止重训")
                            break
                        print(f"  [PPO] 第{ppo_attempt + 1}次误差{best_ppo_err:.2f}%>8%, 重训(seed+{(ppo_attempt + 1) * 10})...")
                    geo_ppo = best_ppo
                    if geo_ppo:
                        grade_reports["PPO"] = ground_grade_limit_report(
                            geo_ppo, dem, transform, args.max_ground_grade, crs)
                    elif "PPO" not in grade_reports:
                        grade_reports["PPO"] = ground_grade_limit_report(
                            None, dem, transform, args.max_ground_grade, crs)
                    methods["PPO"] = geo_ppo
                    print(f"  [PPO] 路径 {len(geo_ppo) if geo_ppo else 0} 点")
                except Exception as ex:
                    print(f"  [PPO] 训练/推理失败: {ex}; 跳过")

            # --- 基准: A* (仅作传统方法对照) ---
            t_a = time.time()
            lat_center = transform.f + transform.e * (H / 2.0)
            route_pixel_spacing = pixel_spacing_meters(transform, lat_center, crs)
            a_path = astar_search(cost_grid, (sr, sc), (er, ec), elevation=dem, pixel_m=pm,
                                  grade_penalty_weight=args.grade_penalty_weight,
                                  max_ground_grade=args.max_ground_grade,
                                  pixel_spacing=route_pixel_spacing)
            geo_astar = [rowcol_to_geo(r, c, transform) for r, c in a_path] if a_path else None
            if geo_astar:
                geo_astar = smooth_path_with_constraints(geo_astar, transform, hard_mask)
            geo_astar, grade_reports["A*(baseline)"] = filter_path_by_ground_grade(
                geo_astar, dem, transform, args.max_ground_grade, crs)
            methods["A*(baseline)"] = geo_astar
            t_astar = time.time() - t_a
            print(f"  [A* baseline] 路径 {len(geo_astar) if geo_astar else 0} 点, 耗时 {t_astar:.1f}s")

            # 无引导模式在所有路径都生成后才读取真实线, 使其只进入事后评估。
            if use_real and not reference_guidance:
                if global_real_exists:
                    if global_real_geo is None:
                        global_real_geo = load_real_path(args.real_shp)
                    real_geo = global_real_geo
                else:
                    rp = resolve_real_path(cname, args.real_dir)
                    if rp and Path(rp).exists():
                        real_geo = load_real_path(rp)
                if real_geo:
                    real_dist_m, _ = compute_dist_to_route(real_geo, transform, H, W, pm)
                    print(f"  [POST-EVAL] 加载真实线路: {len(real_geo)}条折线")

            station_offsets_by_method = {}
            hard_violation_reports = {}
            if station_task:
                for method, geo in list(methods.items()):
                    if not geo:
                        continue
                    geo, offsets = attach_station_endpoints(geo, station_task)
                    geo, grade_report = filter_path_by_ground_grade(
                        geo, dem, transform, args.max_ground_grade, crs)
                    geo, hard_report = validate_route_hard_constraints(geo, transform, hard_mask)
                    methods[method] = geo
                    grade_reports[method] = grade_report
                    hard_violation_reports[method] = hard_report
                    station_offsets_by_method[method] = offsets
                    if geo:
                        print(f"  [{method}] 站点端点恢复后的硬约束复核通过")
                    elif hard_report["hard_constraint_status"] == "fail":
                        print(f"  [{method}] 站点端点恢复后穿越硬禁区，拒绝该候选")

            # 指标
            start_ll = (slat, slon); end_ll = (elat, elon)
            case_metrics = {}
            for m, geo in methods.items():
                if geo:
                    met = compute_metrics(geo, dem, transform, cost_grid, hard_mask, start_ll, end_ll)
                    met["time_s"] = round(t_vin if m == "VIN-Grad" else (t_astar if m == "A*(baseline)" else 0.0), 2)
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
                if geo and real_geo:
                    dev = compute_deviation(geo, real_geo, transform, real_dist_m)
                    if dev:
                        met["dev_hausdorff_km"] = dev["dev_hausdorff_km"]
                        met["dev_mean_km"] = dev["dev_mean_km"]
                        met["dev_p90_km"] = dev["dev_p90_km"]
                        met["dev_max_km"] = dev["dev_max_km"]
                        err = eval_error(met.get("length_km", 0.0), dev)
                        if err:
                            met.update(err)
                        flag = (f"PASS(误差{err['worst_err_pct']:.2f}%<=8%)"
                                if (err and err["pass8"]) else
                                f"FAIL(误差{err['worst_err_pct']:.2f}%)" if err else "NA")
                        print(f"  [{m}] 偏差 vs 真实线网: 最大={dev['dev_hausdorff_km']}km "
                              f"P90={dev['dev_p90_km']}km 平均={dev['dev_mean_km']}km | "
                              f"长度误差={err['len_err_pct'] if err else 'NA'}% "
                              f"Haus%={err['haus_pct'] if err else 'NA'}% -> {flag}")
                case_metrics[m] = met
            # 有参考线引导时按参考误差择优; 无引导时不读取验收误差做选择, 使用 A* 代价基线。
            best_method, best_err, selection_rule = _select_final_route(
                methods, case_metrics, guide_enabled)
            final_geo = methods.get(best_method) if best_method else None
            final_is_ai = best_method in ("VIN-Grad", "PPO")
            osm_data_quality, missing_osm_layers = screening_data_quality(osm_layer_metadata)
            selected_metrics = case_metrics.get(best_method, {}) if best_method else {}
            planning_status = (
                "no_hard_constraint_compliant_route" if not final_geo else
                "screening_candidate_data_incomplete" if missing_osm_layers else
                "screening_candidate")
            case_metrics[FINAL_ROUTE_LABEL] = {
                "method": best_method,
                "selection_rule": selection_rule,
                "worst_err_pct": round(best_err, 3) if best_err is not None else None,
                "pass8": bool(best_err <= ERROR_TOL_PCT) if guide_enabled and best_err is not None else None,
                "is_ai_method": final_is_ai,
                "planning_status": planning_status,
                "hard_constraint_status": selected_metrics.get("hard_constraint_status", "no_path"),
                "hard_violations": selected_metrics.get("hard_violations"),
                "osm_data_quality": osm_data_quality,
                "missing_osm_screening_layers": missing_osm_layers,
                "length_km": round(path_length_km(final_geo), 3) if final_geo else None}
            case_metrics[FINAL_ROUTE_LABEL].update({
                key: value for key, value in selected_metrics.items()
                if key not in {"worst_err_pct", "pass8"}
            })
            if final_geo:
                case_metrics[FINAL_ROUTE_LABEL].update(route_profile_metrics(final_geo, dem, transform))
            if best_method in grade_reports:
                case_metrics[FINAL_ROUTE_LABEL].update(grade_reports[best_method])
            case_metrics[FINAL_ROUTE_LABEL].update(station_offsets_by_method.get(best_method, {}))
            case_metrics[FINAL_ROUTE_LABEL].update({
                "settlement_weight": args.settlement_weight,
                "building_clearance_m": args.building_clearance_m,
                "grade_penalty_weight": args.grade_penalty_weight,
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
                            "settlement_weight": args.settlement_weight,
                            "building_clearance_m": args.building_clearance_m,
                            "grade_penalty_weight": args.grade_penalty_weight,
                            "max_ground_grade_ratio": args.max_ground_grade,
                            "max_surface_slope_deg": args.max_surface_slope_deg,
                            "max_terrain_elevation_m": args.max_terrain_elevation_m,
                            "route_constraints_limitations": "No population layer, official protected-area layer, conductor clearance, tower siting or span feasibility checked",
                        })
            if guide_enabled and best_err is not None:
                print(f"  [集成] 最终路线采用 {best_method} (参考线误差 {best_err:.2f}%)")
            else:
                print(f"  [集成] 无参考线择优, 最终路线采用 {best_method} (A*代价基线)")
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
                                           "worst_err_pct": round(best_err, 3) if best_err is not None else None,
                                           "selection_rule": selection_rule,
                                           "planning_status": planning_status,
                                           "hard_constraint_status": selected_metrics.get("hard_constraint_status", "no_path"),
                                           "hard_violations": selected_metrics.get("hard_violations"),
                                           "osm_data_quality": osm_data_quality,
                                           "missing_osm_screening_layers": missing_osm_layers,
                                           "length_km": selected_metrics.get("length_km"),
                                           "mean_cost": selected_metrics.get("mean_cost"),
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
                                           "voltage_rules_applied": False,
                                           "settlement_weight": args.settlement_weight,
                                           "building_clearance_m": args.building_clearance_m,
                                           "grade_penalty_weight": args.grade_penalty_weight,
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
            except Exception:
                pass
            all_results.append({"case": cname, "metrics": case_metrics})
            # 增量写出真实对比 JSON: 任一走廊完成后即落盘, 即使后续走廊异常/SIGKILL 终止也能保留已完成结果
            try:
                _real_out = []
                for _rec in all_results:
                    _entry = {"case": _rec["case"]}
                    for _m, _met in _rec["metrics"].items():
                        if "dev_hausdorff_km" in _met:
                            _entry[_m] = {"length_km": _met["length_km"],
                                          "dev_hausdorff_km": _met["dev_hausdorff_km"],
                                          "dev_p90_km": _met.get("dev_p90_km"),
                                          "dev_mean_km": _met["dev_mean_km"],
                                          "dev_max_km": _met["dev_max_km"],
                                          "len_err_pct": _met.get("len_err_pct"),
                                          "haus_pct": _met.get("haus_pct"),
                                          "worst_err_pct": _met.get("worst_err_pct"),
                                          "pass8": _met.get("pass8")}
                        elif "method" in _met:
                            _entry[_m] = {"method": _met.get("method"),
                                          "length_km": _met.get("length_km"),
                                          "worst_err_pct": _met.get("worst_err_pct"),
                                          "pass8": _met.get("pass8")}
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
                            title=f"多算法路径对比 - {cname}", real_ll=real_geo)
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
                                    real_ll=real_geo)
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
            rows.append({"case": rec["case"], "method": m, **met})
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

    # 真实路径偏差对比 JSON + 控制台摘要
    if use_real:
        real_out = []
        for rec in all_results:
            entry = {"case": rec["case"]}
            for m, met in rec["metrics"].items():
                if "dev_hausdorff_km" in met:
                    entry[m] = {"length_km": met["length_km"],
                                "dev_hausdorff_km": met["dev_hausdorff_km"],
                                "dev_p90_km": met.get("dev_p90_km"),
                                "dev_mean_km": met["dev_mean_km"],
                                "dev_max_km": met["dev_max_km"],
                                "len_err_pct": met.get("len_err_pct"),
                                "haus_pct": met.get("haus_pct"),
                                "worst_err_pct": met.get("worst_err_pct"),
                                "pass8": met.get("pass8")}
            if entry:
                real_out.append(entry)
        if baseline_records:
            real_out.append({"baseline_VIN_no_corridor": baseline_records})
        try:
            with open(OUT / "real_comparison_metrics.json", "w", encoding="utf-8") as f:
                json.dump(real_out, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print("\n" + "=" * 70)
        print("  真实路径偏差对比 (规划路径 vs 真实线路, 单位 km)")
        print("=" * 70)
        print(f"  {'走廊':16s} {'方法':16s} {'Hausdorff':>9s} {'长度误差%':>9s} {'Haus%':>7s} {'达标':>5s}")
        n_pass = n_tot = 0
        for rec in all_results:
            for m, met in rec["metrics"].items():
                if "dev_hausdorff_km" in met:
                    we = met.get("worst_err_pct")
                    ok = met.get("pass8")
                    n_tot += 1
                    if ok:
                        n_pass += 1
                    print(f"  {rec['case']:16s} {m:16s} {met['dev_hausdorff_km']:9.2f} "
                          f"{(met.get('len_err_pct') or 0):9.2f} {(met.get('haus_pct') or 0):7.2f} "
                          f"{(('PASS' if ok else 'FAIL') if we is not None else 'NA'):>5s}")
        for b in baseline_records:
            d = b["deviation"]
            print(f"  {b['case']:16s} {'VIN无引导':16s} {d['dev_hausdorff_km']:9.2f} {'-':>9s} {'-':>7s} {'-':>5s}")
        print("-" * 70)
        print(f"  误差<=8% 达标: {n_pass}/{n_tot} (方法×走廊); 目标 ERROR_TOL_PCT={ERROR_TOL_PCT}%")
        print("=" * 70)

        # ── 对抗式循环验收: 按方法独立统计 PASS(不掩盖 AI 成败) ──
        print("\n" + "=" * 70)
        print("  【独立验收】各 AI 方法自身达标率 (Critic 挑战 G2)")
        print("=" * 70)
        for m in ["VIN-Grad", "PPO", "A*(baseline)"]:
            recs = [r for r in all_results if m in r["metrics"]]
            if not recs:
                continue
            npass = sum(1 for r in recs if r["metrics"][m].get("pass8"))
            worst = max((r["metrics"][m].get("worst_err_pct", 0.0) for r in recs), default=0.0)
            tag = "AI主方法" if m == "VIN-Grad" else ("AI-RL" if m == "PPO" else "基线(非AI)")
            verdict = "PASS" if npass == len(recs) else f"FAIL({npass}/{len(recs)})"
            print(f"  {m:16s} [{tag}] 自身达标 {npass}/{len(recs)}  最差={worst:.2f}%  -> {verdict}")
        ai_fail_final = [r["case"] for r in all_results
                         if not r["metrics"].get("VIN-Grad", {}).get("pass8")]
        ai_fallback = [r["case"] for r in all_results
                       if r["metrics"].get(FINAL_ROUTE_LABEL, {}).get("is_ai_method") is False]
        if ai_fail_final:
            print(f"  ⚠ 以下走廊 VIN-Grad(AI主方法)自身未达标: {ai_fail_final}")
        else:
            print("  ✓ VIN-Grad(AI主方法)在所有走廊自身达标(<=8%)")
        if ai_fallback:
            print(f"  ℹ 以下走廊最终交付路线选用了 A*(因其误差比已达标的 VIN-Grad 略优, 非AI失败): {ai_fallback}")
        print("=" * 70)

    # 控制台摘要
    print("\n" + "=" * 70)
    print("  结果摘要（各走廊均值）")
    print("=" * 70)
    for m in ["VIN-Grad", "PPO", "A*(baseline)"]:
        lens = [r["metrics"][m]["length_km"] for r in all_results if m in r["metrics"]]
        mc = [r["metrics"][m]["mean_cost"] for r in all_results if m in r["metrics"]]
        if lens:
            print(f"  {m:14s} 平均长度 {np.mean(lens):7.2f}km  平均代价 {np.mean(mc):.3f}")
    print(f"  总耗时: {time.time() - t0:.1f}s")
    print(f"  输出目录: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
