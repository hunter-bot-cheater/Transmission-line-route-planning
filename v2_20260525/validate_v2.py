"""
v2_strict: 多输电路径规划模型严格验证脚本
版本: v2.20260525
作者: path_planning_team
变更记录:
  - v2.20260525: 10条标准线路, 集成质量门控, 每条线路独立输出
  - v1.20260525: 6条线路, 宽松验证
依赖: v2/config, v2/src/*, shared/data_acquisition
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import json
import math
import time

import config as cfg
from data_acquisition import load_dem, load_taiwan_lines, acquire_all
from src.preprocessing import (
    derive_terrain_factors, generate_hard_mask, generate_soft_mask,
    align_all_rasters,
)
from src.cost_model import (
    build_feature_stack, generate_pseudo_labels,
    train_random_forest, predict_cost_surface, save_cost_geotiff,
)
from src.path_planning import (
    fuse_cost_surface, astar_search, smooth_path,
    geo_to_grid, grid_to_geo_coords, compute_path_length_km,
    haversine_m, _strict_quality_gate,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _dense_sample_line(coords, spacing_m=100):
    if len(coords) < 2:
        return coords
    sampled = [coords[0]]
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        d = haversine_m(p1[0], p1[1], p2[0], p2[1])
        n_segs = max(1, int(d / spacing_m))
        for j in range(1, n_segs + 1):
            t = j / n_segs
            lon = p1[0] + t * (p2[0] - p1[0])
            lat = p1[1] + t * (p2[1] - p1[1])
            sampled.append(np.array([lon, lat]))
    return np.array(sampled)


def compute_hausdorff(coords_a, coords_b, sample_step=2):#豪斯多夫距离
    a = _dense_sample_line(np.array(coords_a), spacing_m=100)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]
    max_min_a2b = 0#a to b
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        max_min_a2b = max(max_min_a2b, min_d)
    max_min_b2a = 0
    for pb in b:
        min_d = min(haversine_m(pb[0], pb[1], pa[0], pa[1]) for pa in a)
        max_min_b2a = max(max_min_b2a, min_d)
    return max(max_min_a2b, max_min_b2a)


def compute_mean_distance(coords_a, coords_b, sample_step=2):
    a = _dense_sample_line(np.array(coords_a), spacing_m=100)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]
    total = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        total += min_d
    return total / len(a)


def compute_overlap_ratio(coords_a, coords_b, threshold_m=500):
    a = _dense_sample_line(np.array(coords_a), spacing_m=100)
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)
    within = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        if min_d < threshold_m:
            within += 1
    return within / len(a)


def validate_single_line(case, aligned, rf, scaler, dst_transform, crs, hard_mask, soft_mask):
    """对单条线路运行完整v2严格验证"""
    taiwan_lines = load_taiwan_lines()
    way_id = case["way_id"]
    rows = taiwan_lines[taiwan_lines["id"] == way_id]
    if len(rows) == 0:
        print(f"  警告: 未找到线路 {way_id}")
        return None
    row = rows.iloc[0]
    real_coords = list(row.geometry.coords)
    real_start = real_coords[0]
    real_end = real_coords[-1]
    real_length = compute_path_length_km([(c[0], c[1]) for c in real_coords])

    start_lat, start_lon = real_start[1], real_start[0]
    end_lat, end_lon = real_end[1], real_end[0]
    voltage = row.get("voltage", "N/A")

    # 直线距离 (用于sinuosity计算)
    straight_km = haversine_m(start_lon, start_lat, end_lon, end_lat) / 1000.0

    case_id = case["case_id"]
    print(f"\n  {'─'*60}")
    print(f"  案例: {case_id} — {case['description']}")
    print(f"  线路: {way_id} | 电压: {voltage}V | 长度: {real_length:.1f}km")
    print(f"  起点: ({start_lat:.4f}, {start_lon:.4f})  终点: ({end_lat:.4f}, {end_lon:.4f})")
    print(f"  {'─'*60}")

    t0 = time.time()

    # Phase 4: 路径搜索
    feature_stack = build_feature_stack(aligned)
    predicted_cost = predict_cost_surface(rf, scaler, feature_stack, hard_mask)
    dist_existing = aligned.get("dist_existing_line")
    final_cost = fuse_cost_surface(predicted_cost, soft_mask, hard_mask, dist_existing=dist_existing)

    # 保存成本表面到案例目录
    case_output = cfg.OUTPUT_DIR / case_id
    case_output.mkdir(parents=True, exist_ok=True)
    save_cost_geotiff(final_cost, dst_transform, crs, case_output / f"{case_id}_final_cost_surface.tif")

    start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
    end_rc = geo_to_grid(end_lat, end_lon, dst_transform)

    # 起止点验证
    H, W = final_cost.shape
    endpoint_fail = False
    if not (0 <= start_rc[0] < H and 0 <= start_rc[1] < W):
        print(f"  错误: 起点出界 → 不通过")
        endpoint_fail = True
    if not (0 <= end_rc[0] < H and 0 <= end_rc[1] < W):
        print(f"  错误: 终点出界 → 不通过")
        endpoint_fail = True
    # 非严格模式下, 端点硬约束由astar_search内部自动调整
    if endpoint_fail:
        quality = {
            "passed": False,
            "n_passed": 0,
            "n_total": 7,
            "checks": {"endpoint": {"passed": False, "detail": "起点或终点出界"}},
        }
        return {
            "case_id": case_id,
            "way_id": way_id,
            "description": case["description"],
            "voltage": voltage,
            "real_length_km": real_length,
            "predicted_length_km": None,
            "length_error_pct": None,
            "hausdorff_m": None,
            "mean_distance_m": None,
            "overlap_500m_pct": None,
            "overlap_1km_pct": None,
            "overlap_2km_pct": None,
            "quality_passed": False,
            "quality_n_passed": 0,
            "quality_n_total": 7,
            "quality_checks": quality["checks"],
            "acceptable": False,
            "fail_reason": "ENDPOINT_INVALID",
            "n_real_vertices": len(real_coords),
            "n_predicted_vertices": 0,
            "real_coords": real_coords,
            "predicted_coords": [],
            "start": (float(start_lat), float(start_lon)),
            "end": (float(end_lat), float(end_lon)),
        }

    path_cells = astar_search(final_cost, start_rc, end_rc)
    if path_cells is None:
        print(f"  错误: A*未找到路径 → 不通过")
        quality = {
            "passed": False,
            "n_passed": 0,
            "n_total": 7,
            "checks": {"astar": {"passed": False, "detail": "A*搜索失败"}},
        }
        return {
            "case_id": case_id,
            "way_id": way_id,
            "description": case["description"],
            "voltage": voltage,
            "real_length_km": real_length,
            "predicted_length_km": None,
            "length_error_pct": None,
            "hausdorff_m": None,
            "mean_distance_m": None,
            "overlap_500m_pct": None,
            "overlap_1km_pct": None,
            "overlap_2km_pct": None,
            "quality_passed": False,
            "quality_n_passed": 0,
            "quality_n_total": 7,
            "quality_checks": quality["checks"],
            "acceptable": False,
            "fail_reason": "ASTAR_FAILED",
            "n_real_vertices": len(real_coords),
            "n_predicted_vertices": 0,
            "real_coords": real_coords,
            "predicted_coords": [],
            "start": (float(start_lat), float(start_lon)),
            "end": (float(end_lat), float(end_lon)),
        }

    predicted_coords = smooth_path(path_cells, dst_transform, hard_mask)
    pred_length = compute_path_length_km(predicted_coords)
    elapsed = time.time() - t0

    # v2_strict: 质量门控
    quality = _strict_quality_gate(
        predicted_coords, aligned, hard_mask, final_cost, dst_transform, straight_km
    )

    # v1兼容指标
    hausdorff = compute_hausdorff(predicted_coords, real_coords)
    mean_dist = compute_mean_distance(predicted_coords, real_coords)
    overlap_500m = compute_overlap_ratio(predicted_coords, real_coords, 500)
    overlap_1km = compute_overlap_ratio(predicted_coords, real_coords, 1000)
    overlap_2km = compute_overlap_ratio(predicted_coords, real_coords, 2000)
    len_error_pct = abs(pred_length - real_length) / real_length * 100

    # 综合判定: v1指标 + v2质量门控
    acceptable = quality["passed"]
    if not acceptable:
        fail_reasons = [k for k, v in quality["checks"].items() if not v["passed"]]
    else:
        fail_reasons = []

    print(f"  Hausdorff={hausdorff:.0f}m | 平均距离={mean_dist:.0f}m | "
          f"500m重叠={overlap_500m*100:.0f}% | 长度误差={len_error_pct:.1f}%")
    print(f"  质量门控: {'通过' if quality['passed'] else '不通过'} "
          f"({quality['n_passed']}/{quality['n_total']}项)")
    if fail_reasons:
        print(f"  失败项: {', '.join(fail_reasons)}")
    print(f"  耗时: {elapsed:.0f}s")

    return {
        "case_id": case_id,
        "way_id": way_id,
        "description": case["description"],
        "voltage": voltage,
        "real_length_km": real_length,
        "predicted_length_km": pred_length,
        "length_error_pct": len_error_pct,
        "hausdorff_m": hausdorff,
        "mean_distance_m": mean_dist,
        "overlap_500m_pct": overlap_500m * 100,
        "overlap_1km_pct": overlap_1km * 100,
        "overlap_2km_pct": overlap_2km * 100,
        "quality_passed": quality["passed"],
        "quality_n_passed": quality["n_passed"],
        "quality_n_total": quality["n_total"],
        "quality_checks": quality["checks"],
        "acceptable": acceptable,
        "fail_reason": ", ".join(fail_reasons) if fail_reasons else "NONE",
        "n_real_vertices": len(real_coords),
        "n_predicted_vertices": len(predicted_coords),
        "real_coords": real_coords,
        "predicted_coords": predicted_coords,
        "start": (float(start_lat), float(start_lon)),
        "end": (float(end_lat), float(end_lon)),
    }


def export_case_outputs(result, aligned, final_cost, dst_transform, hard_mask):
    """导出单条线路的完整输出"""
    case_id = result["case_id"]
    case_dir = cfg.OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    coords = result["predicted_coords"]
    if not coords:
        return

    # 1. SHP / GeoJSON
    geom = LineString([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame({
        "case_id": [case_id],
        "way_id": [result["way_id"]],
        "length_km": [result["predicted_length_km"]],
        "geometry": [geom],
    }, crs=cfg.WGS84)
    gdf.to_file(case_dir / f"{case_id}_optimal_path.shp")
    gdf.to_file(case_dir / f"{case_id}_optimal_path.geojson", driver="GeoJSON")

    # 2. statistics.json
    stats = {
        "case_id": case_id,
        "way_id": result["way_id"],
        "description": result["description"],
        "start": result["start"],
        "end": result["end"],
        "real_length_km": result["real_length_km"],
        "predicted_length_km": result["predicted_length_km"],
        "length_error_pct": result["length_error_pct"],
        "hausdorff_m": result["hausdorff_m"],
        "mean_distance_m": result["mean_distance_m"],
        "overlap_500m_pct": result["overlap_500m_pct"],
        "quality_passed": result["quality_passed"],
        "quality_n_passed": result["quality_n_passed"],
        "fail_reason": result["fail_reason"],
    }
    with open(case_dir / f"{case_id}_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 3. quality_report.json
    with open(case_dir / f"{case_id}_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(result["quality_checks"], f, ensure_ascii=False, indent=2)

    # 4. 可视化
    _create_case_map(result, case_dir, aligned, dst_transform)
    _create_elevation_profile(result, case_dir, aligned, dst_transform)


def _create_case_map(result, case_dir, aligned, dst_transform):
    """创建单条线路概览图"""
    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
    case_id = result["case_id"]

    dem = aligned.get("dem")
    if dem is not None:
        sample = 6
        dem_sub = dem[::sample, ::sample]
        dem_sub = np.nan_to_num(dem_sub, nan=0)
        dy, dx = np.gradient(dem_sub.astype(np.float64))
        az, alt = np.radians(315), np.radians(45)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(dy, dx)
        hs = np.cos(alt)*np.cos(slope) + np.sin(alt)*np.sin(slope)*np.cos(az-aspect)
        hs = np.clip(hs*255, 0, 255)
        extent = (dst_transform.c, dst_transform.c + dst_transform.a*dem.shape[1],
                  dst_transform.f + dst_transform.e*dem.shape[0], dst_transform.f)
        ax.imshow(hs, extent=extent, cmap="gray", alpha=0.4, origin="upper")

    # 真实线路
    rc = result["real_coords"]
    ax.plot([c[0] for c in rc], [c[1] for c in rc], color="#3498db", lw=2.0, label="真实线路", alpha=0.8)

    # 预测路径
    pc = result["predicted_coords"]
    if pc:
        ax.plot([c[0] for c in pc], [c[1] for c in pc], color="#e74c3c", lw=1.5, linestyle="--", label="预测路径", alpha=0.8)

    status_str = "通过" if result["quality_passed"] else "不通过"
    ax.set_title(f"{case_id}: {result['description']}\n质量门控: {status_str} ({result['quality_n_passed']}/{result['quality_n_total']}项)", fontsize=12, fontweight="bold")
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(case_dir / f"{case_id}_map_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _create_elevation_profile(result, case_dir, aligned, dst_transform):
    """创建高程剖面图"""
    coords = result["predicted_coords"]
    if not coords:
        return
    dem = aligned.get("dem")
    slope = aligned.get("slope")
    if dem is None:
        return

    distances = [0.0]
    elevations = []
    slopes_vals = []
    for i, (lon, lat) in enumerate(coords):
        r, c = geo_to_grid(lat, lon, dst_transform)
        H, W = dem.shape
        if 0 <= r < H and 0 <= c < W:
            elevations.append(float(dem[r, c]))
            if slope is not None:
                slopes_vals.append(float(slope[r, c]))
        if i > 0:
            d = haversine_m(coords[i-1][0], coords[i-1][1], lon, lat)
            distances.append(distances[-1] + d / 1000.0)

    # 对齐距离和采样
    n_pts = min(len(distances), len(elevations))
    dists = distances[:n_pts]
    elevs = elevations[:n_pts]

    fig, ax1 = plt.subplots(figsize=(16, 6), dpi=150)
    ax1.fill_between(dists, elevs, min(elevs), alpha=0.3, color="#3498db")
    ax1.plot(dists, elevs, color="#2c3e50", lw=1.5)
    ax1.set_xlabel("距离 (km)")
    ax1.set_ylabel("高程 (m)", color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")

    if slopes_vals and len(slopes_vals) >= n_pts:
        ax2 = ax1.twinx()
        slope_vals = slopes_vals[:n_pts]
        ax2.plot(dists, slope_vals, color="#e74c3c", lw=0.8, alpha=0.6)
        ax2.axhline(y=cfg.MAX_SLOPE, color="red", linestyle="--", lw=1, label=f"坡度上限{cfg.MAX_SLOPE}°")
        ax2.set_ylabel("坡度 (°)", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.legend(loc="upper right")

    ax1.set_title(f"{result['case_id']}: 高程剖面图 (质量门控: {'通过' if result['quality_passed'] else '不通过'})")
    fig.savefig(case_dir / f"{result['case_id']}_elevation_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("  v2严格化验证 — 10条标准测试线路")
    print("  台湾输电线路智能路径规划系统")
    print("=" * 60)

    # ============================================================
    # 1. 选择验证线路
    # ============================================================
    print("\n[Step1] 选择10条标准测试线路...")
    gdf = load_taiwan_lines()

    test_cases = cfg.TEST_CASES
    selected = []
    for case in test_cases:
        rows = gdf[gdf["id"] == case["way_id"]]
        if len(rows) > 0:
            selected.append(case)
        else:
            print(f"  警告: 未找到 {case['case_id']} — {case['way_id']}")

    print(f"  选中 {len(selected)} 条线路:")
    for case in selected:
        print(f"    {case['case_id']}: {case['description']}")

    # ============================================================
    # 2. 公共预处理
    # ============================================================
    print("\n" + "=" * 60)
    print("  [Step2] 公共预处理 (Phase 1-3)")
    print("=" * 60)
    t0 = time.time()

    data = acquire_all()
    dem = data["dem"]
    transform = data["dem_transform"]
    crs = data["dem_crs"]

    terrain_factors = derive_terrain_factors(dem, transform)
    data.update(terrain_factors)

    # 排除10条测试线路，防止数据泄露
    # dist_existing_line不应包含目标线路本身，否则伪标签和成本表面会不公平地偏向正确路径
    test_way_ids = {case["way_id"] for case in cfg.TEST_CASES}
    taiwan_lines_all = data.get("taiwan_lines")
    if taiwan_lines_all is not None:
        data["taiwan_lines"] = taiwan_lines_all[~taiwan_lines_all["id"].isin(test_way_ids)].copy()
        n_excluded = len(taiwan_lines_all) - len(data["taiwan_lines"])
        print(f"  已排除 {n_excluded} 条测试线路, 剩余 {len(data['taiwan_lines'])} 条用于距离场计算")

    osm_data = {
        "osm_landuse": data.get("osm_landuse"),
        "osm_buildings": data.get("osm_buildings"),
        "osm_railways": data.get("osm_railways"),
        "osm_airports": data.get("osm_airports"),
    }
    aligned = align_all_rasters(data, dem, transform, osm_data=osm_data)
    dst_transform = aligned["transform"]
    dst_shape = aligned["shape"]
    # 合并对齐的栅格层与原始矢量层, 供硬/软约束掩膜使用
    aligned_for_mask = dict(aligned)
    for key in ["osm_water", "osm_protected"]:
        if key in data and data[key] is not None:
            aligned_for_mask[key] = data[key]
    hard_mask = generate_hard_mask(aligned_for_mask, dst_transform, dst_shape)
    soft_mask = generate_soft_mask(aligned_for_mask, dst_transform, dst_shape)

    feature_stack = build_feature_stack(aligned)
    labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
    rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)

    elapsed_common = time.time() - t0
    print(f"  公共预处理完成, 耗时: {elapsed_common:.0f}s")

    # ============================================================
    # 3. 逐条验证
    # ============================================================
    print("\n" + "=" * 60)
    print(f"  [Step3] 逐条验证 ({len(selected)}条)")
    print("=" * 60)

    results = []
    for i, case in enumerate(selected):
        print(f"\n{'='*50}")
        print(f"  案例 {i+1}/{len(selected)}")
        print(f"{'='*50}")
        result = validate_single_line(case, aligned, rf, scaler, dst_transform, crs, hard_mask, soft_mask)
        if result is not None:
            results.append(result)
            export_case_outputs(result, aligned, None, dst_transform, hard_mask)

    # ============================================================
    # 4. 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("  [Step4] 汇总报告")
    print("=" * 60)

    n_quality_pass = sum(1 for r in results if r["quality_passed"])
    n_valid = sum(1 for r in results if r["predicted_coords"])

    print(f"\n  总测试线路: {len(results)}")
    print(f"  成功生成路径: {n_valid}/{len(results)}")
    print(f"  质量门控通过: {n_quality_pass}/{len(results)}")

    print(f"\n  {'─'*100}")
    print(f"  {'案例':<12} {'线路ID':<22} {'长度误差':>8} {'Hausdorff':>10} {'500m重叠':>8} {'质量门控':>10} {'状态':>8}")
    print(f"  {'─'*100}")

    for r in results:
        le = f"{r['length_error_pct']:.1f}%" if r['length_error_pct'] is not None else "N/A"
        hd = f"{r['hausdorff_m']:.0f}m" if r['hausdorff_m'] is not None else "N/A"
        ov = f"{r['overlap_500m_pct']:.1f}%" if r['overlap_500m_pct'] is not None else "N/A"
        qs = f"{r['quality_n_passed']}/{r['quality_n_total']}" if r['quality_n_passed'] is not None else "N/A"
        status = "通过" if r["acceptable"] else "不通过"
        print(f"  {r['case_id']:<12} {r['way_id']:<22} {le:>8} {hd:>10} {ov:>8} {qs:>10} {status:>8}")

    print(f"  {'─'*100}")

    # 统计
    valid_results = [r for r in results if r["hausdorff_m"] is not None]
    if valid_results:
        hausdorffs = [r["hausdorff_m"] for r in valid_results]
        mean_dists = [r["mean_distance_m"] for r in valid_results]
        overlaps = [r["overlap_500m_pct"] for r in valid_results]
        len_errors = [r["length_error_pct"] for r in valid_results]

        print(f"\n  统计汇总 (有效结果 {len(valid_results)}/{len(results)}):")
        print(f"  {'指标':<20} {'均值':>10} {'中位数':>10} {'最小':>10} {'最大':>10}")
        print(f"  {'─'*60}")
        print(f"  {'Hausdorff距离(m)':<20} {np.mean(hausdorffs):>10.0f} {np.median(hausdorffs):>10.0f} {np.min(hausdorffs):>10.0f} {np.max(hausdorffs):>10.0f}")
        print(f"  {'平均距离(m)':<20} {np.mean(mean_dists):>10.0f} {np.median(mean_dists):>10.0f} {np.min(mean_dists):>10.0f} {np.max(mean_dists):>10.0f}")
        print(f"  {'500m重叠率(%)':<20} {np.mean(overlaps):>10.1f} {np.median(overlaps):>10.1f} {np.min(overlaps):>10.1f} {np.max(overlaps):>10.1f}")
        print(f"  {'长度误差(%)':<20} {np.mean(len_errors):>10.1f} {np.median(len_errors):>10.1f} {np.min(len_errors):>10.1f} {np.max(len_errors):>10.1f}")

    print(f"\n  >>> v2严格验证完成: 质量门控通过 {n_quality_pass}/{len(results)} <<<")

    # ============================================================
    # 5. 导出汇总报告
    # ============================================================
    report_data = []
    for r in results:
        d = {k: v for k, v in r.items() if k not in ("real_coords", "predicted_coords", "quality_checks")}
        d["quality_checks"] = r["quality_checks"]
        report_data.append(d)

    report = {
        "version": "v2.20260525",
        "n_lines_tested": len(results),
        "n_paths_generated": n_valid,
        "n_quality_passed": n_quality_pass,
        "pass_rate": f"{n_quality_pass}/{len(results)}",
        "test_cases": report_data,
    }

    report_path = cfg.OUTPUT_DIR / "validation_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    main()
