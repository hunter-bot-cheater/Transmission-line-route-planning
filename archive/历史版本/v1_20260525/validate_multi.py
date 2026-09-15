"""
多输电线路径规划模型验证脚本
- 选择多条已有输电线, 提取真实起止点
- 用模型预测最优路径
- 对比预测路径与真实路径的重合度
- 生成综合对比报告和可视化
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import json
import math
import time

import config as cfg
from src.data_acquisition import load_dem, load_taiwan_lines, acquire_all
from src.preprocessing import (
    derive_terrain_factors, generate_hard_mask, generate_soft_mask,
    align_all_rasters,
)
from src.cost_model import (
    build_feature_stack, generate_pseudo_labels,
    train_random_forest, predict_cost_surface,
)
from src.path_planning import (
    fuse_cost_surface, astar_search, smooth_path,
    geo_to_grid, compute_path_length_km,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def haversine_m(lon1, lat1, lon2, lat2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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


def compute_hausdorff(coords_a, coords_b, sample_step=2):
    a = np.array(coords_a)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]

    max_min_a2b = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        max_min_a2b = max(max_min_a2b, min_d)

    max_min_b2a = 0
    for pb in b:
        min_d = min(haversine_m(pb[0], pb[1], pa[0], pa[1]) for pa in a)
        max_min_b2a = max(max_min_b2a, min_d)

    return max(max_min_a2b, max_min_b2a)


def compute_mean_distance(coords_a, coords_b, sample_step=2):
    a = np.array(coords_a)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]

    total = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        total += min_d
    return total / len(a)


def compute_overlap_ratio(coords_a, coords_b, threshold_m=500):
    a = np.array(coords_a)
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)

    within = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        if min_d < threshold_m:
            within += 1
    return within / len(a)


def validate_single_line(row, aligned, rf, scaler, dst_transform, hard_mask, soft_mask):
    """对单条输电线运行完整验证, 返回指标字典"""
    real_coords = list(row.geometry.coords)
    real_start = real_coords[0]
    real_end = real_coords[-1]
    real_length = compute_path_length_km([(c[0], c[1]) for c in real_coords])

    start_lat, start_lon = real_start[1], real_start[0]
    end_lat, end_lon = real_end[1], real_end[0]

    line_id = row["id"]
    voltage = row.get("voltage", "N/A")

    print(f"\n  {'─'*50}")
    print(f"  线路: {line_id}")
    print(f"  电压: {voltage} | 长度: {real_length:.1f}km | 节点: {len(real_coords)}")
    print(f"  起点: ({real_start[0]:.4f}, {real_start[1]:.4f})")
    print(f"  终点: ({real_end[0]:.4f}, {real_end[1]:.4f})")
    print(f"  {'─'*50}")

    t0 = time.time()

    # Phase 4: 路径搜索
    dist_existing = aligned.get("dist_existing_line")
    final_cost = fuse_cost_surface(
        predict_cost_surface(rf, scaler,
            build_feature_stack(aligned), hard_mask),
        soft_mask, hard_mask, dist_existing=dist_existing)

    start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
    end_rc = geo_to_grid(end_lat, end_lon, dst_transform)

    if np.isinf(final_cost[start_rc[0], start_rc[1]]):
        print(f"  警告: 起点在硬约束区, 搜索最近可行点...")
        from src.path_planning import _find_nearest_valid
        sr2, sc2 = _find_nearest_valid(final_cost, start_rc[0], start_rc[1])
        if sr2 is None:
            print(f"  错误: 起点附近无可行区域, 跳过")
            return None
        start_rc = (sr2, sc2)
        print(f"  起点调整为: {start_rc}")

    if np.isinf(final_cost[end_rc[0], end_rc[1]]):
        print(f"  警告: 终点在硬约束区, 搜索最近可行点...")
        from src.path_planning import _find_nearest_valid
        er2, ec2 = _find_nearest_valid(final_cost, end_rc[0], end_rc[1])
        if er2 is None:
            print(f"  错误: 终点附近无可行区域, 跳过")
            return None
        end_rc = (er2, ec2)
        print(f"  终点调整为: {end_rc}")

    path_cells = astar_search(final_cost, start_rc, end_rc)
    if path_cells is None:
        print(f"  错误: A*未找到路径, 跳过")
        return None

    predicted_coords = smooth_path(path_cells, dst_transform, hard_mask)
    pred_length = compute_path_length_km(predicted_coords)
    elapsed = time.time() - t0

    # 对比指标
    hausdorff = compute_hausdorff(predicted_coords, real_coords)
    mean_dist = compute_mean_distance(predicted_coords, real_coords)
    overlap_500m = compute_overlap_ratio(predicted_coords, real_coords, 500)
    overlap_1km = compute_overlap_ratio(predicted_coords, real_coords, 1000)
    overlap_2km = compute_overlap_ratio(predicted_coords, real_coords, 2000)
    len_error_pct = abs(pred_length - real_length) / real_length * 100

    # 可接受性
    acceptable = True
    reasons = []
    if hausdorff < 5000:
        reasons.append(f"Hausdorff {hausdorff:.0f}m < 5km [通过]")
    else:
        reasons.append(f"Hausdorff {hausdorff:.0f}m > 5km [偏高]")
        acceptable = False
    if mean_dist < 2000:
        reasons.append(f"平均距离 {mean_dist:.0f}m < 2km [通过]")
    else:
        reasons.append(f"平均距离 {mean_dist:.0f}m > 2km [偏高]")
        acceptable = False
    if overlap_500m > 0.3:
        reasons.append(f"500m重叠 {overlap_500m*100:.0f}% > 30% [通过]")
    else:
        reasons.append(f"500m重叠 {overlap_500m*100:.0f}% < 30% [偏低]")
        acceptable = False
    if len_error_pct < 30:
        reasons.append(f"长度误差 {len_error_pct:.1f}% < 30% [通过]")
    else:
        reasons.append(f"长度误差 {len_error_pct:.1f}% > 30% [偏高]")
        acceptable = False

    status = "通过" if acceptable else "部分指标偏高"

    print(f"  Hausdorff={hausdorff:.0f}m | 平均距离={mean_dist:.0f}m | "
          f"500m重叠={overlap_500m*100:.0f}% | 1km重叠={overlap_1km*100:.0f}% | "
          f"2km重叠={overlap_2km*100:.0f}%")
    print(f"  长度: 真实{real_length:.1f}km vs 预测{pred_length:.1f}km (误差{len_error_pct:.1f}%)")
    print(f"  耗时: {elapsed:.0f}s | 评估: {status}")

    return {
        "line_id": line_id,
        "voltage": voltage,
        "real_length_km": real_length,
        "predicted_length_km": pred_length,
        "length_error_pct": len_error_pct,
        "hausdorff_m": hausdorff,
        "mean_distance_m": mean_dist,
        "overlap_500m_pct": overlap_500m * 100,
        "overlap_1km_pct": overlap_1km * 100,
        "overlap_2km_pct": overlap_2km * 100,
        "acceptable": acceptable,
        "reasons": reasons,
        "n_real_vertices": len(real_coords),
        "n_predicted_vertices": len(predicted_coords),
        "real_coords": real_coords,
        "predicted_coords": predicted_coords,
        "start": (float(real_start[1]), float(real_start[0])),
        "end": (float(real_end[1]), float(real_end[0])),
    }


def create_summary_chart(results, output_dir):
    """创建多线路对比汇总图"""
    n = len(results)
    if n == 0:
        return

    line_labels = [r["line_id"].replace("way/", "") for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=200)

    # 1. Hausdorff距离
    ax = axes[0, 0]
    vals = [r["hausdorff_m"] for r in results]
    colors = ["#2ecc71" if r["acceptable"] else "#e74c3c" for r in results]
    bars = ax.bar(range(n), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=5000, color="orange", linestyle="--", linewidth=1, label="阈值5km")
    ax.set_xticks(range(n))
    ax.set_xticklabels(line_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hausdorff距离 (m)")
    ax.set_title("Hausdorff距离对比")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{v:.0f}",
                ha="center", fontsize=7)

    # 2. 平均距离
    ax = axes[0, 1]
    vals = [r["mean_distance_m"] for r in results]
    bars = ax.bar(range(n), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=2000, color="orange", linestyle="--", linewidth=1, label="阈值2km")
    ax.set_xticks(range(n))
    ax.set_xticklabels(line_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("平均最近距离 (m)")
    ax.set_title("平均距离对比")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f"{v:.0f}",
                ha="center", fontsize=7)

    # 3. 500m重叠率
    ax = axes[0, 2]
    vals = [r["overlap_500m_pct"] for r in results]
    bars = ax.bar(range(n), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=30, color="orange", linestyle="--", linewidth=1, label="阈值30%")
    ax.set_xticks(range(n))
    ax.set_xticklabels(line_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("500m重叠率 (%)")
    ax.set_title("500m内重叠率对比")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{v:.1f}%",
                ha="center", fontsize=7)

    # 4. 长度误差
    ax = axes[1, 0]
    vals = [r["length_error_pct"] for r in results]
    bars = ax.bar(range(n), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=30, color="orange", linestyle="--", linewidth=1, label="阈值30%")
    ax.set_xticks(range(n))
    ax.set_xticklabels(line_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("长度误差 (%)")
    ax.set_title("路径长度误差对比")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{v:.1f}%",
                ha="center", fontsize=7)

    # 5. 多阈值重叠率汇总
    ax = axes[1, 1]
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, [r["overlap_500m_pct"] for r in results], w, label="500m内", color="#3498db")
    ax.bar(x, [r["overlap_1km_pct"] for r in results], w, label="1km内", color="#f39c12")
    ax.bar(x + w, [r["overlap_2km_pct"] for r in results], w, label="2km内", color="#2ecc71")
    ax.set_xticks(range(n))
    ax.set_xticklabels(line_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("重叠率 (%)")
    ax.set_title("多阈值重叠率对比")
    ax.legend(fontsize=8)

    # 6. 通过/未通过汇总
    ax = axes[1, 2]
    n_pass = sum(1 for r in results if r["acceptable"])
    n_fail = n - n_pass
    ax.pie([n_pass, n_fail], labels=[f"通过({n_pass})", f"未通过({n_fail})"],
           colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 12})
    ax.set_title("综合评估汇总")

    fig.suptitle(f"多线路验证对比 ({n}条输电线)", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = output_dir / "maps" / "validation_multi_summary.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n汇总图已保存: {save_path}")
    return save_path


def create_overlay_map(results, aligned, dst_transform, output_dir):
    """创建所有线路的路径对比叠加图"""
    fig, ax = plt.subplots(figsize=(18, 16), dpi=200)

    # 山影底图
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
        ax.imshow(hs, extent=extent, cmap="gray", alpha=0.5, origin="upper")

    colors_real = plt.cm.Blues(np.linspace(0.3, 0.9, len(results)))
    colors_pred = plt.cm.Reds(np.linspace(0.3, 0.9, len(results)))

    for i, r in enumerate(results):
        # 真实线路
        rc = r["real_coords"]
        lons_r = [c[0] for c in rc]
        lats_r = [c[1] for c in rc]
        ax.plot(lons_r, lats_r, color=colors_real[i], lw=2.0, alpha=0.7,
                label=f"真实 {r['line_id'].replace('way/','')}" if i < 8 else "")
        # 预测路径
        pc = r["predicted_coords"]
        lons_p = [c[0] for c in pc]
        lats_p = [c[1] for c in pc]
        ax.plot(lons_p, lats_p, color=colors_pred[i], lw=1.5, alpha=0.7, linestyle="--",
                label=f"预测 {r['line_id'].replace('way/','')}" if i < 8 else "")

    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_title(f"多线路路径对比 ({len(results)}条输电线) — 实线=真实, 虚线=预测", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    save_path = output_dir / "maps" / "validation_multi_overlay.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"叠加图已保存: {save_path}")
    return save_path


def main():
    print("=" * 60)
    print("  多输电线路径规划模型验证")
    print("  对比: 模型预测 vs 真实线路 (多条)")
    print("=" * 60)

    # ============================================================
    # 1. 选择验证线路
    # ============================================================
    print("\n[Step1] 选择验证线路...")
    gdf = load_taiwan_lines()
    gdf["est_len_km"] = gdf.geometry.length * 111.32

    # 选6条代表性线路: 不同长度、电压、区域
    target_ids = [
        "way/365662519",  # 58.3km, 345kV, 屏东→恒春
        "way/203692582",  # 112.8km, 345kV, 最长
        "way/202162589",  # 75.6km, 161kV, 中长
        "way/179686272",  # 72.1km, 161kV, 中长
        "way/199445383",  # 55.4km, 345kV, 中等
        "way/184391013",  # 57.2km, 161kV, 中等
    ]

    selected_lines = []
    for tid in target_ids:
        rows = gdf[gdf["id"] == tid]
        if len(rows) > 0:
            selected_lines.append(rows.iloc[0])
        else:
            print(f"  警告: 未找到线路 {tid}")

    print(f"  选中 {len(selected_lines)} 条线路:")
    for row in selected_lines:
        print(f"    {row['id']}: {row['est_len_km']:.1f}km, {row.get('voltage','N/A')}V")

    # ============================================================
    # 2. 公共预处理 (Phase 1-3, 只运行一次)
    # ============================================================
    print("\n" + "=" * 60)
    print("  [Step2] 公共预处理 (Phase 1-3, 一次运行)")
    print("=" * 60)

    t0 = time.time()

    # Phase 1
    data = acquire_all()
    dem = data["dem"]
    transform = data["dem_transform"]
    crs = data["dem_crs"]

    # Phase 2
    terrain_factors = derive_terrain_factors(dem, transform)
    data.update(terrain_factors)
    osm_data = {
        "osm_landuse": data.get("osm_landuse"),
        "osm_buildings": data.get("osm_buildings"),
        "osm_railways": data.get("osm_railways"),
        "osm_airports": data.get("osm_airports"),
    }
    aligned = align_all_rasters(data, dem, transform, osm_data=osm_data)
    dst_transform = aligned["transform"]
    dst_shape = aligned["shape"]
    hard_mask = generate_hard_mask(aligned, dst_transform, dst_shape)
    soft_mask = generate_soft_mask(aligned, dst_transform, dst_shape)

    # Phase 3
    feature_stack = build_feature_stack(aligned)
    labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
    rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)

    elapsed_common = time.time() - t0
    print(f"  公共预处理完成, 耗时: {elapsed_common:.0f}s")

    # ============================================================
    # 3. 逐条验证
    # ============================================================
    print("\n" + "=" * 60)
    print(f"  [Step3] 逐条验证 ({len(selected_lines)}条)")
    print("=" * 60)

    results = []
    for i, row in enumerate(selected_lines):
        print(f"\n{'='*50}")
        print(f"  线路 {i+1}/{len(selected_lines)}")
        print(f"{'='*50}")
        result = validate_single_line(row, aligned, rf, scaler, dst_transform, hard_mask, soft_mask)
        if result is not None:
            results.append(result)

    # ============================================================
    # 4. 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("  [Step4] 汇总报告")
    print("=" * 60)

    output_dir = cfg.OUTPUT_DIR

    n_acceptable = sum(1 for r in results if r["acceptable"])
    print(f"\n  验证线路数: {len(results)} (成功{len(results)}条)")
    print(f"  通过数: {n_acceptable}/{len(results)}")
    print(f"\n  {'─'*80}")
    print(f"  {'线路ID':<25} {'长度误差':>8} {'Hausdorff':>10} {'平均距离':>10} {'500m重叠':>10} {'评估':>8}")
    print(f"  {'─'*80}")

    for r in results:
        status = "通过" if r["acceptable"] else "偏高"
        print(f"  {r['line_id']:<25} {r['length_error_pct']:>7.1f}% {r['hausdorff_m']:>9.0f}m "
              f"{r['mean_distance_m']:>9.0f}m {r['overlap_500m_pct']:>9.1f}% {status:>8}")

    print(f"  {'─'*80}")

    # 统计
    hausdorffs = [r["hausdorff_m"] for r in results]
    mean_dists = [r["mean_distance_m"] for r in results]
    overlaps_500 = [r["overlap_500m_pct"] for r in results]
    len_errors = [r["length_error_pct"] for r in results]

    print(f"\n  统计汇总:")
    print(f"  {'指标':<20} {'均值':>10} {'中位数':>10} {'最小':>10} {'最大':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Hausdorff距离(m)':<20} {np.mean(hausdorffs):>10.0f} {np.median(hausdorffs):>10.0f} {np.min(hausdorffs):>10.0f} {np.max(hausdorffs):>10.0f}")
    print(f"  {'平均距离(m)':<20} {np.mean(mean_dists):>10.0f} {np.median(mean_dists):>10.0f} {np.min(mean_dists):>10.0f} {np.max(mean_dists):>10.0f}")
    print(f"  {'500m重叠率(%)':<20} {np.mean(overlaps_500):>10.1f} {np.median(overlaps_500):>10.1f} {np.min(overlaps_500):>10.1f} {np.max(overlaps_500):>10.1f}")
    print(f"  {'长度误差(%)':<20} {np.mean(len_errors):>10.1f} {np.median(len_errors):>10.1f} {np.min(len_errors):>10.1f} {np.max(len_errors):>10.1f}")

    # 综合可接受性
    if n_acceptable >= len(results) * 0.67:
        overall = "整体可接受"
    else:
        overall = "需进一步优化"

    print(f"\n  >>> 综合评估: {overall} ({n_acceptable}/{len(results)}通过) <<<")

    # ============================================================
    # 5. 可视化
    # ============================================================
    print(f"\n[Step5] 生成可视化...")
    create_summary_chart(results, output_dir)
    create_overlay_map(results, aligned, dst_transform, output_dir)

    # ============================================================
    # 6. 导出完整报告
    # ============================================================
    # 清理结果中的坐标数据(太大)
    report_data = []
    for r in results:
        d = {k: v for k, v in r.items() if k not in ("real_coords", "predicted_coords", "reasons")}
        d["reasons"] = r["reasons"]
        report_data.append(d)

    report = {
        "n_lines_tested": len(results),
        "n_acceptable": n_acceptable,
        "overall_assessment": overall,
        "aggregate_stats": {
            "hausdorff_mean": float(np.mean(hausdorffs)),
            "hausdorff_median": float(np.median(hausdorffs)),
            "mean_distance_mean": float(np.mean(mean_dists)),
            "mean_distance_median": float(np.median(mean_dists)),
            "overlap_500m_mean": float(np.mean(overlaps_500)),
            "overlap_500m_median": float(np.median(overlaps_500)),
            "length_error_mean": float(np.mean(len_errors)),
            "length_error_median": float(np.median(len_errors)),
        },
        "lines": report_data,
    }

    report_path = output_dir / "reports" / "validation_multi_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {report_path}")

    # 导出对比GeoJSON
    geoms = []
    names = []
    types = []
    for r in results:
        geoms.append(LineString([(c[0], c[1]) for c in r["real_coords"]]))
        names.append(f"{r['line_id']}_真实")
        types.append("actual")
        geoms.append(LineString(r["predicted_coords"]))
        names.append(f"{r['line_id']}_预测")
        types.append("predicted")

    comp_gdf = gpd.GeoDataFrame({
        "name": names, "type": types, "geometry": geoms
    }, crs=cfg.WGS84)
    comp_gdf.to_file(output_dir / "data" / "validation_multi_comparison.geojson", driver="GeoJSON")
    print(f"对比GeoJSON已保存: {output_dir / 'validation_multi_comparison.geojson'}")

    print("\n" + "=" * 60)
    print("  多线路验证完成!")
    print("=" * 60)
    return report


if __name__ == "__main__":
    main()
