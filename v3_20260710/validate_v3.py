"""
v3: 三算法对比验证 — A* vs IPSO-SA vs DBO
版本: v3.20260710
作者: path_planning_team
描述: 复用 v2 预处理+成本建模, 运行三算法在 10 条线路上对比
依赖: v3/config, v2_20260525/src/*, shared/data_acquisition
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent
_v3_src = _v3_dir / "src"
_v2_dir = _v3_dir.parent / "v2_20260525"
_v2_src = _v2_dir / "src"
_shared_dir = _v3_dir.parent / "shared"
# 绝对路径 + reversed + insert(0) → v3/src在最前,v2在最后
for _d in reversed([_v3_src, _v3_dir, _v2_dir, _v2_src, _shared_dir]):
    sys.path.insert(0, str(_d))  # 不检查重复,确保v3在v2前面

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

import config as cfg
from data_acquisition import load_taiwan_lines, acquire_all
from preprocessing import (
    derive_terrain_factors, generate_hard_mask, generate_soft_mask,
    align_all_rasters,
)
from cost_model import (
    build_feature_stack, generate_pseudo_labels,
    train_random_forest, predict_cost_surface,
)
from path_planning import (
    fuse_cost_surface, astar_search, smooth_path,
    geo_to_grid, grid_to_geo_coords, compute_path_length_km,
    haversine_m, _strict_quality_gate,
)
from ipso_sa_planner import ipso_sa_plan_path
from dbo_planner import dbo_plan_path
from output_v3 import (
    export_path, export_statistics, export_quality_report,
    export_convergence, export_comparison_chart,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 验证工具函数 (复用 v2 的验证逻辑)
# ============================================================
def _dense_sample_line(coords, spacing_m=100):
    """路径加密采样"""
    if len(coords) < 2:
        return np.array(coords)
    sampled = [np.array(coords[0])]
    for i in range(len(coords) - 1):
        p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
        d = haversine_m(p1[0], p1[1], p2[0], p2[1])
        n_segs = max(1, int(d / spacing_m))
        for j in range(1, n_segs + 1):
            t = j / n_segs
            lon = p1[0] + t * (p2[0] - p1[0])
            lat = p1[1] + t * (p2[1] - p1[1])
            sampled.append(np.array([lon, lat]))
    return np.array(sampled)


def compute_hausdorff(coords_a, coords_b, sample_step=2):
    a = _dense_sample_line(np.array(coords_a), spacing_m=100)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]
    max_min_a2b = max(
        min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        for pa in a
    )
    max_min_b2a = max(
        min(haversine_m(pb[0], pb[1], pa[0], pa[1]) for pa in a)
        for pb in b
    )
    return max(max_min_a2b, max_min_b2a)


def compute_overlap_ratio(coords_a, coords_b, threshold_m=500):
    a = _dense_sample_line(np.array(coords_a), spacing_m=100)
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)
    within = sum(
        1 for pa in a
        if min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b) < threshold_m
    )
    return within / len(a)


def compute_mean_distance(coords_a, coords_b, sample_step=2):
    a = _dense_sample_line(np.array(coords_a), spacing_m=100)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]
    total = sum(
        min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        for pa in a
    )
    return total / len(a)


# ============================================================
# 单线路三算法验证
# ============================================================
def validate_single_line_v3(
    case: dict,
    aligned: dict,
    rf, scaler,
    dst_transform, crs,
    hard_mask, soft_mask,
):
    """对单条线路运行 A* + IPSO-SA + DBO"""
    way_id = case["way_id"]
    real_coords = case["real_coords"]
    real_length = case["real_length"]
    voltage = case["voltage"]
    real_start = real_coords[0]
    real_end = real_coords[-1]
    start_lat, start_lon = real_start[1], real_start[0]
    end_lat, end_lon = real_end[1], real_end[0]
    straight_km = haversine_m(start_lon, start_lat, end_lon, end_lat) / 1000.0

    case_id = case["case_id"]
    print(f"\n  {'─' * 60}")
    print(f"  案例: {case_id} — {case['description']}")
    print(f"  线路: {way_id} | 电压: {voltage}V | 长度: {real_length:.1f}km")
    print(f"  直线距离: {straight_km:.1f}km")
    print(f"  {'─' * 60}")

    case_output = cfg.OUTPUT_DIR / case_id
    case_output.mkdir(parents=True, exist_ok=True)

    # 成本表面 (复用 v2)
    feature_stack = build_feature_stack(aligned)
    predicted_cost = predict_cost_surface(rf, scaler, feature_stack, hard_mask)
    dist_existing = aligned.get("dist_existing_line")
    final_cost = fuse_cost_surface(
        predicted_cost, soft_mask, hard_mask, dist_existing=dist_existing,
    )

    start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
    end_rc = geo_to_grid(end_lat, end_lon, dst_transform)
    start_latlon = (start_lat, start_lon)
    end_latlon = (end_lat, end_lon)

    results = {"case_id": case_id, "way_id": way_id, "algorithms": {}}

    # ── 算法 1: A* (v2 现成) ──
    print("\n  [A*] 运行中...")
    t0 = time.time()
    a_star_raw = astar_search(final_cost, start_rc, end_rc)
    if a_star_raw is not None:
        a_star_smoothed = smooth_path(a_star_raw, dst_transform)
        a_star_quality = _strict_quality_gate(
            a_star_smoothed, aligned, hard_mask, final_cost, dst_transform, straight_km,
        )
    else:
        a_star_smoothed = None
        a_star_quality = {"passed": False, "checks": {"A*失败": {"passed": False, "detail": "A*未找到路径"}}}
    t_astar = time.time() - t0

    if a_star_smoothed:
        results["algorithms"]["A*"] = {
            "time_s": t_astar,
            "path_length_km": compute_path_length_km(a_star_smoothed),
            "hausdorff_m": compute_hausdorff(a_star_smoothed, real_coords),
            "overlap_500m_pct": compute_overlap_ratio(a_star_smoothed, real_coords) * 100,
            "mean_distance_m": compute_mean_distance(a_star_smoothed, real_coords),
            "quality_passed": a_star_quality["passed"],
            "quality_n": f"{a_star_quality.get('n_passed', 0)}/{a_star_quality.get('n_total', 7)}",
            "convergence": None,
        }
        # 导出 A* 结果
        export_path(a_star_smoothed, case_output, case_id, "A*")
        export_statistics(a_star_smoothed, aligned, hard_mask, dst_transform, case_output, case_id, "A*")
        export_quality_report(a_star_quality, case_output, case_id, "A*")
        _create_case_map(case_id, case["description"], real_coords, a_star_smoothed,
                        a_star_quality["passed"], f"{a_star_quality.get('n_passed', 0)}/{a_star_quality.get('n_total', 7)}",
                        aligned, dst_transform, case_output)
        _create_elevation_profile(case_id, a_star_smoothed, a_star_quality["passed"],
                                 aligned, dst_transform, case_output)
        print(f"  [A*] 完成 ({t_astar:.1f}s), 质量: {'通过' if a_star_quality['passed'] else '不通过'}")
    else:
        results["algorithms"]["A*"] = {"time_s": t_astar, "status": "failed"}
        print(f"  [A*] 失败")

    # ── 算法 2: IPSO-SA ──
    print("\n  [IPSO-SA] 运行中...")
    t0 = time.time()
    try:
        ipso_path, ipso_quality, ipso_info = ipso_sa_plan_path(
            cost_surface=final_cost,
            hard_mask=hard_mask,
            transform=dst_transform,
            start_latlon=start_latlon,
            end_latlon=end_latlon,
            aligned=aligned,
            a_star_path=a_star_smoothed,
            random_seed=cfg.COMPARISON_RANDOM_SEED,
            verbose=False,
        )
        t_ipso = time.time() - t0
        results["algorithms"]["IPSO-SA"] = {
            "time_s": t_ipso,
            "path_length_km": compute_path_length_km(ipso_path),
            "hausdorff_m": compute_hausdorff(ipso_path, real_coords),
            "overlap_500m_pct": compute_overlap_ratio(ipso_path, real_coords) * 100,
            "mean_distance_m": compute_mean_distance(ipso_path, real_coords),
            "quality_passed": ipso_quality["passed"],
            "quality_n": f"{ipso_quality.get('n_passed', 0)}/{ipso_quality.get('n_total', 7)}",
            "convergence": ipso_info["convergence_curve"],
            "best_fitness": ipso_info["best_fitness"],
        }
        export_path(ipso_path, case_output, case_id, "IPSO-SA")
        export_statistics(ipso_path, aligned, hard_mask, dst_transform, case_output, case_id, "IPSO-SA")
        export_quality_report(ipso_quality, case_output, case_id, "IPSO-SA")
        export_convergence(ipso_info["convergence_curve"], ipso_info["best_fitness"],
                          case_output, case_id, "IPSO-SA")
        _create_case_map(case_id, case["description"], real_coords, ipso_path,
                        ipso_quality["passed"], f"{ipso_quality.get('n_passed', 0)}/{ipso_quality.get('n_total', 7)}",
                        aligned, dst_transform, case_output)
        _create_elevation_profile(case_id, ipso_path, ipso_quality["passed"],
                                 aligned, dst_transform, case_output)
        print(f"  [IPSO-SA] 完成 ({t_ipso:.1f}s), 适应度={ipso_info['best_fitness']:.4f}, "
              f"质量: {'通过' if ipso_quality['passed'] else '不通过'}")
    except Exception as e:
        t_ipso = time.time() - t0
        results["algorithms"]["IPSO-SA"] = {"time_s": t_ipso, "status": "failed", "error": str(e)}
        print(f"  [IPSO-SA] 异常: {e}")

    # ── 算法 3: DBO ──
    print("\n  [DBO] 运行中...")
    t0 = time.time()
    try:
        dbo_path, dbo_quality, dbo_info = dbo_plan_path(
            cost_surface=final_cost,
            hard_mask=hard_mask,
            transform=dst_transform,
            start_latlon=start_latlon,
            end_latlon=end_latlon,
            aligned=aligned,
            a_star_path=a_star_smoothed,
            random_seed=cfg.COMPARISON_RANDOM_SEED,
            verbose=False,
        )
        t_dbo = time.time() - t0
        results["algorithms"]["DBO"] = {
            "time_s": t_dbo,
            "path_length_km": compute_path_length_km(dbo_path),
            "hausdorff_m": compute_hausdorff(dbo_path, real_coords),
            "overlap_500m_pct": compute_overlap_ratio(dbo_path, real_coords) * 100,
            "mean_distance_m": compute_mean_distance(dbo_path, real_coords),
            "quality_passed": dbo_quality["passed"],
            "quality_n": f"{dbo_quality.get('n_passed', 0)}/{dbo_quality.get('n_total', 7)}",
            "convergence": dbo_info["convergence_curve"],
            "best_fitness": dbo_info["best_fitness"],
        }
        export_path(dbo_path, case_output, case_id, "DBO")
        export_statistics(dbo_path, aligned, hard_mask, dst_transform, case_output, case_id, "DBO")
        export_quality_report(dbo_quality, case_output, case_id, "DBO")
        export_convergence(dbo_info["convergence_curve"], dbo_info["best_fitness"],
                          case_output, case_id, "DBO")
        _create_case_map(case_id, case["description"], real_coords, dbo_path,
                        dbo_quality["passed"], f"{dbo_quality.get('n_passed', 0)}/{dbo_quality.get('n_total', 7)}",
                        aligned, dst_transform, case_output)
        _create_elevation_profile(case_id, dbo_path, dbo_quality["passed"],
                                 aligned, dst_transform, case_output)
        print(f"  [DBO] 完成 ({t_dbo:.1f}s), 适应度={dbo_info['best_fitness']:.4f}, "
              f"质量: {'通过' if dbo_quality['passed'] else '不通过'}")
    except Exception as e:
        t_dbo = time.time() - t0
        results["algorithms"]["DBO"] = {"time_s": t_dbo, "status": "failed", "error": str(e)}
        print(f"  [DBO] 异常: {e}")

    # 导出对比报告
    report_path = case_output / f"{case_id}_comparison_v3.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ============================================================
# 可视化辅助函数
# ============================================================
def _create_case_map(case_id, description, real_coords, predicted_coords,
                     quality_passed, quality_n, aligned, dst_transform, output_dir):
    """创建单条线路概览图 (自适应dem背景 + 真实线路 + 加密预测路径)"""
    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
    dem = aligned.get("dem")
    if dem is not None:
        # 自适应降采样: 短边至少800像素, 防止马赛克
        target_pixels = 800
        sample = max(1, min(dem.shape) // target_pixels)
        dem_sub = dem[::sample, ::sample]
        dem_sub = np.nan_to_num(dem_sub, nan=0)
        dy, dx = np.gradient(dem_sub.astype(np.float64))
        az, alt = np.radians(315), np.radians(45)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect_rad = np.arctan2(dy, dx)
        hill = (np.cos(alt) * np.cos(slope_rad) +
                np.sin(alt) * np.sin(slope_rad) * np.cos(az - aspect_rad))
        extent = (dst_transform.c, dst_transform.c + dst_transform.a * dem.shape[1],
                  dst_transform.f + dst_transform.e * dem.shape[0], dst_transform.f)
        ax.imshow(np.clip(hill * 255, 0, 255), extent=extent, cmap="gray", alpha=0.4, origin="upper")
    if real_coords:
        ax.plot([c[0] for c in real_coords], [c[1] for c in real_coords],
                color="#3498db", lw=2.5, label="真实线路", alpha=0.9)
    if predicted_coords:
        # 加密路径点: RDP简化后的稀疏拐点 → 100m间距密集采样, 曲线不再变形为直线
        dense_pred = _dense_sample_line(np.array(predicted_coords), spacing_m=100)
        # 发光底层: 宽半透明线让路径从hillshade背景中浮出来
        ax.plot([p[0] for p in dense_pred], [p[1] for p in dense_pred],
                color="#e74c3c", lw=5.0, alpha=0.25, solid_capstyle="round")
        # 主线: 红色虚线
        ax.plot([p[0] for p in dense_pred], [p[1] for p in dense_pred],
                color="#e74c3c", lw=2.0, linestyle="--", label="预测路径", alpha=0.9)
    status_str = "通过" if quality_passed else "不通过"
    ax.set_title(f"{case_id}: {description}\n质量门控: {status_str} ({quality_n})", fontsize=12, fontweight="bold")
    ax.set_xlabel("经度 (°E)"); ax.set_ylabel("纬度 (°N)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"{case_id}_map_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _create_elevation_profile(case_id, coords, quality_passed, aligned, dst_transform, output_dir):
    """创建高程剖面图"""
    if not coords:
        return
    dem = aligned.get("dem")
    slope = aligned.get("slope")
    if dem is None:
        return
    distances = [0.0]; elevations = []; slopes_vals = []
    for i, (lon, lat) in enumerate(coords):
        r = int((lat - dst_transform.f) / dst_transform.e)
        c = int((lon - dst_transform.c) / dst_transform.a)
        H, W = dem.shape
        if 0 <= r < H and 0 <= c < W:
            elevations.append(float(dem[r, c]))
            if slope is not None:
                slopes_vals.append(float(slope[r, c]))
        if i > 0:
            d = haversine_m(coords[i-1][0], coords[i-1][1], lon, lat)
            distances.append(distances[-1] + d / 1000.0)
    n_pts = min(len(distances), len(elevations))
    dists = distances[:n_pts]; elevs = elevations[:n_pts]
    fig, ax1 = plt.subplots(figsize=(16, 6), dpi=150)
    ax1.fill_between(dists, elevs, min(elevs), alpha=0.3, color="#3498db")
    ax1.plot(dists, elevs, color="#2c3e50", lw=1.5)
    ax1.set_xlabel("距离 (km)"); ax1.set_ylabel("高程 (m)", color="#2c3e50")
    if slopes_vals and len(slopes_vals) >= n_pts:
        ax2 = ax1.twinx()
        ax2.plot(dists, slopes_vals[:n_pts], color="#e74c3c", lw=0.8, alpha=0.6)
        ax2.axhline(y=cfg.MAX_SLOPE, color="red", linestyle="--", lw=1, label=f"坡度上限{cfg.MAX_SLOPE}°")
        ax2.set_ylabel("坡度 (°)", color="#e74c3c"); ax2.legend(loc="upper right")
    ax1.set_title(f"{case_id}: 高程剖面图 (质量: {'通过' if quality_passed else '不通过'})")
    fig.savefig(output_dir / f"{case_id}_elevation_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  V3 三算法对比验证: A* vs IPSO-SA vs DBO")
    print("=" * 60)

    # Step 1: 选择验证线路
    print("\n[Step1] 选择10条标准测试线路...")
    gdf = load_taiwan_lines()
    test_cases = cfg.TEST_CASES
    selected = []
    for case in test_cases:
        rows = gdf[gdf["id"] == case["way_id"]]
        if len(rows) > 0:
            row = rows.iloc[0]
            case["real_coords"] = list(row.geometry.coords)
            case["voltage"] = row.get("voltage", "N/A")
            case["real_length"] = compute_path_length_km(
                [(c[0], c[1]) for c in case["real_coords"]]
            )
            selected.append(case)
        else:
            print(f"  警告: 未找到 {case['case_id']} — {case['way_id']}")
    print(f"  选中 {len(selected)} 条线路")

    # Step 2: 公共预处理 (复用 v2)
    print("\n" + "=" * 60)
    print("  [Step2] 公共预处理 (复用 v2 Phase 1-3)")
    print("=" * 60)
    t0 = time.time()

    data = acquire_all()
    dem = data["dem"]
    transform = data["dem_transform"]
    crs = data["dem_crs"]

    terrain_factors = derive_terrain_factors(dem, transform)
    data.update(terrain_factors)

    # 排除测试线路
    test_way_ids = {case["way_id"] for case in cfg.TEST_CASES}
    taiwan_lines_all = data.get("taiwan_lines")
    if taiwan_lines_all is not None:
        data["taiwan_lines"] = taiwan_lines_all[
            ~taiwan_lines_all["id"].isin(test_way_ids)
        ].copy()
        n_excluded = len(taiwan_lines_all) - len(data["taiwan_lines"])
        print(f"  已排除 {n_excluded} 条测试线路")

    osm_data = {
        "osm_landuse": data.get("osm_landuse"),
        "osm_buildings": data.get("osm_buildings"),
        "osm_railways": data.get("osm_railways"),
        "osm_airports": data.get("osm_airports"),
    }
    aligned = align_all_rasters(data, dem, transform, osm_data=osm_data)
    dst_transform = aligned["transform"]
    dst_shape = aligned["shape"]

    aligned_for_mask = dict(aligned)
    for key in ["osm_water", "osm_protected"]:
        if key in data and data[key] is not None:
            aligned_for_mask[key] = data[key]

    hard_mask = generate_hard_mask(aligned_for_mask, dst_transform, dst_shape)
    soft_mask = generate_soft_mask(aligned_for_mask, dst_transform, dst_shape)

    feature_stack = build_feature_stack(aligned)
    labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
    rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)

    print(f"  公共预处理完成, 耗时: {time.time() - t0:.0f}s")

    # Step 3: 逐条三算法验证
    print("\n" + "=" * 60)
    print(f"  [Step3] 逐条三算法验证 ({len(selected)}条)")
    print("=" * 60)

    all_results = []
    for i, case in enumerate(selected):
        print(f"\n{'=' * 50}")
        print(f"  案例 {i + 1}/{len(selected)}")
        print(f"{'=' * 50}")
        result = validate_single_line_v3(
            case, aligned, rf, scaler, dst_transform, crs, hard_mask, soft_mask,
        )
        if result is not None:
            all_results.append(result)

    # Step 4: 汇总报告
    print("\n" + "=" * 60)
    print("  [Step4] 汇总对比")
    print("=" * 60)

    print(f"\n  {'案例':<12} {'算法':<10} {'耗时(s)':<10} {'长度(km)':<10} "
          f"{'Hausdorff(m)':<14} {'重叠率':<10} {'质量':<10}")
    print("  " + "-" * 76)
    for r in all_results:
        for algo in ["A*", "IPSO-SA", "DBO"]:
            algo_data = r["algorithms"].get(algo, {})
            if "status" in algo_data and algo_data["status"] == "failed":
                print(f"  {r['case_id']:<12} {algo:<10} {'FAILED':<10}")
            elif "path_length_km" in algo_data:
                print(
                    f"  {r['case_id']:<12} {algo:<10} "
                    f"{algo_data['time_s']:<10.1f} "
                    f"{algo_data['path_length_km']:<10.1f} "
                    f"{algo_data.get('hausdorff_m', 0):<14.0f} "
                    f"{algo_data.get('overlap_500m_pct', 0):<10.1f} "
                    f"{algo_data.get('quality_n', '?'):<10}"
                )

    # 汇总统计
    print("\n  --- 汇总统计 ---")
    for algo in ["A*", "IPSO-SA", "DBO"]:
        hausdorffs = []
        overlaps = []
        passed_count = 0
        total_valid = 0
        for r in all_results:
            ad = r["algorithms"].get(algo, {})
            if "hausdorff_m" in ad:
                hausdorffs.append(ad["hausdorff_m"])
                overlaps.append(ad["overlap_500m_pct"])
                total_valid += 1
                if ad.get("quality_passed"):
                    passed_count += 1

        if hausdorffs:
            print(
                f"  {algo:<10}: "
                f"Hausdorff 均值={np.mean(hausdorffs):.0f}m, "
                f"重叠率 均值={np.mean(overlaps):.1f}%, "
                f"质量通过={passed_count}/{total_valid}"
            )

    # 保存汇总报告
    summary_path = cfg.OUTPUT_DIR / "validation_v3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  汇总报告: {summary_path}")

    # 导出对比图表
    chart_path = export_comparison_chart(all_results, cfg.OUTPUT_DIR)
    print(f"  对比图表: {chart_path}")

    print("\n" + "=" * 60)
    print("  V3 对比验证完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
