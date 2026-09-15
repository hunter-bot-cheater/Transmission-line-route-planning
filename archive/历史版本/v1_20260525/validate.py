"""
输电线路径规划模型验证脚本
- 选择一条已有输电线, 提取其真实起止点
- 用模型预测最优路径
- 对比预测路径与真实路径的重合度
- 输出对比可视化和误差分析
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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
from src.data_acquisition import load_dem, load_taiwan_lines
from src.preprocessing import (
    derive_terrain_factors, generate_hard_mask, generate_soft_mask,
    align_all_rasters, compute_slope,
)
from src.cost_model import (
    build_feature_stack, generate_pseudo_labels,
    train_random_forest, predict_cost_surface,
)
from src.path_planning import (
    fuse_cost_surface, astar_search, smooth_path,
    geo_to_grid, grid_to_geo_coords, compute_path_length_km,
)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def haversine_m(lon1, lat1, lon2, lat2):
    """两点的Haversine距离(米)"""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_hausdorff(coords_a, coords_b, sample_step=2):
    """采样Hausdorff距离(米) — 两个路径之间的最大最小距离"""
    a = np.array(coords_a)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]

    # a -> b
    max_min_a2b = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        max_min_a2b = max(max_min_a2b, min_d)

    # b -> a
    max_min_b2a = 0
    for pb in b:
        min_d = min(haversine_m(pb[0], pb[1], pa[0], pa[1]) for pa in a)
        max_min_b2a = max(max_min_b2a, min_d)

    return max(max_min_a2b, max_min_b2a)


def compute_mean_distance(coords_a, coords_b, sample_step=2):
    """平均最近距离(米)"""
    a = np.array(coords_a)[::sample_step]
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)[::sample_step]

    total = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        total += min_d
    return total / len(a)


def compute_overlap_ratio(coords_a, coords_b, threshold_m=500):
    """阈值内重叠比例: coords_a中距离coords_b < threshold的点占比"""
    a = np.array(coords_a)
    # 沿真实线路密采样以准确测量距离
    b = _dense_sample_line(np.array(coords_b), spacing_m=100)

    within = 0
    for pa in a:
        min_d = min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b)
        if min_d < threshold_m:
            within += 1
    return within / len(a)


def _dense_sample_line(coords, spacing_m=100):
    """沿线路密采样，间距约spacing_m米"""
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


def main():
    print("=" * 60)
    print("  输电线路径规划模型验证")
    print("  对比: 模型预测 vs 真实线路")
    print("=" * 60)

    # ============================================================
    # 1. 选择验证线路
    # ============================================================
    print("\n[Step1] 选择验证线路...")
    gdf = load_taiwan_lines()

    # 选 way/365662519 (58.3km, 345kV, 屏东→恒春)
    target_id = "way/365662519"
    target_row = gdf[gdf["id"] == target_id]
    if len(target_row) == 0:
        # fallback: 选最长的
        gdf["len_est"] = gdf.geometry.length
        target_row = gdf.nlargest(1, "len_est")

    row = target_row.iloc[0]
    real_coords = list(row.geometry.coords)
    real_start = real_coords[0]
    real_end = real_coords[-1]
    real_length = compute_path_length_km([(c[0], c[1]) for c in real_coords])

    print(f"  线路ID: {row['id']}")
    print(f"  电压: {row.get('voltage', 'N/A')}")
    print(f"  真实长度: {real_length:.2f} km")
    print(f"  真实起点: ({real_start[0]:.4f}, {real_start[1]:.4f})")
    print(f"  真实终点: ({real_end[0]:.4f}, {real_end[1]:.4f})")
    print(f"  真实节点数: {len(real_coords)}")

    start_lat, start_lon = real_start[1], real_start[0]
    end_lat, end_lon = real_end[1], real_end[0]

    # ============================================================
    # 2. 运行模型预测
    # ============================================================
    print("\n[Step2] 运行模型预测路径...")
    t0 = time.time()

    # Phase 1: 数据
    from src.data_acquisition import acquire_all
    data = acquire_all()
    dem = data["dem"]
    transform = data["dem_transform"]
    crs = data["dem_crs"]

    # Phase 2: 预处理
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

    # Phase 3: 成本建模
    feature_stack = build_feature_stack(aligned)
    labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
    rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)
    predicted_cost = predict_cost_surface(rf, scaler, feature_stack, hard_mask)

    # Phase 4: 路径搜索
    dist_existing = aligned.get("dist_existing_line")
    final_cost = fuse_cost_surface(predicted_cost, soft_mask, hard_mask, dist_existing=dist_existing)
    start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
    end_rc = geo_to_grid(end_lat, end_lon, dst_transform)

    print(f"  起点栅格: {start_rc}")
    print(f"  终点栅格: {end_rc}")

    path_cells = astar_search(final_cost, start_rc, end_rc)
    if path_cells is None:
        print("  [错误] A*未找到路径!")
        sys.exit(1)

    predicted_coords = smooth_path(path_cells, dst_transform, hard_mask)
    pred_length = compute_path_length_km(predicted_coords)
    elapsed = time.time() - t0
    print(f"  预测路径长度: {pred_length:.2f} km")
    print(f"  预测耗时: {elapsed:.1f}s")

    # ============================================================
    # 3. 对比分析
    # ============================================================
    print("\n[Step3] 对比分析...")

    hausdorff = compute_hausdorff(predicted_coords, real_coords)
    mean_dist = compute_mean_distance(predicted_coords, real_coords)
    overlap_500m = compute_overlap_ratio(predicted_coords, real_coords, 500)
    overlap_1km = compute_overlap_ratio(predicted_coords, real_coords, 1000)
    overlap_2km = compute_overlap_ratio(predicted_coords, real_coords, 2000)

    # 路径长度误差
    len_error_pct = abs(pred_length - real_length) / real_length * 100

    print(f"\n  对比指标:")
    print(f"  {'─'*40}")
    print(f"  真实路径长度:    {real_length:.2f} km")
    print(f"  预测路径长度:    {pred_length:.2f} km")
    print(f"  长度误差:        {len_error_pct:.1f}%")
    print(f"  Hausdorff距离:   {hausdorff:.0f} m")
    print(f"  平均最近距离:    {mean_dist:.0f} m")
    print(f"  500m内重叠率:    {overlap_500m*100:.1f}%")
    print(f"  1km内重叠率:     {overlap_1km*100:.1f}%")
    print(f"  2km内重叠率:     {overlap_2km*100:.1f}%")

    # ============================================================
    # 4. 误差可接受性评估
    # ============================================================
    print(f"\n  误差可接受性评估:")
    print(f"  {'─'*40}")

    acceptable = True
    reasons = []

    if hausdorff < 5000:
        reasons.append(f"Hausdorff距离 {hausdorff:.0f}m < 5km [通过]")
    else:
        reasons.append(f"Hausdorff距离 {hausdorff:.0f}m > 5km [偏高]")
        acceptable = False

    if mean_dist < 2000:
        reasons.append(f"平均距离 {mean_dist:.0f}m < 2km [通过]")
    else:
        reasons.append(f"平均距离 {mean_dist:.0f}m > 2km [偏高]")
        acceptable = False

    if overlap_500m > 0.3:
        reasons.append(f"500m重叠率 {overlap_500m*100:.1f}% > 30% [通过]")
    else:
        reasons.append(f"500m重叠率 {overlap_500m*100:.1f}% < 30% [偏低]")
        acceptable = False

    if len_error_pct < 30:
        reasons.append(f"长度误差 {len_error_pct:.1f}% < 30% [通过]")
    else:
        reasons.append(f"长度误差 {len_error_pct:.1f}% > 30% [偏高]")
        acceptable = False

    for r in reasons:
        print(f"  {r}")

    if acceptable:
        print(f"\n  >>> 整体评估: 误差可接受 <<<")
    else:
        print(f"\n  >>> 整体评估: 部分指标需关注 <<<")

    # ============================================================
    # 5. 对比可视化 (双面板)
    # ============================================================
    print("\n[Step4] 生成对比可视化...")
    output_dir = cfg.OUTPUT_DIR
    dem_aligned = aligned.get("dem")

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=300)

    # --- 左图: 路径对比 ---
    ax = axes[0]
    _draw_hillshade_bg(ax, dem_aligned, dst_transform)
    _draw_line(ax, real_coords, color="#3388FF", lw=2.5, label="真实输电线")
    _draw_line(ax, predicted_coords, color="#FF2D2D", lw=2.5, label="模型预测路径")
    ax.scatter(real_start[0], real_start[1], c="green", s=150, zorder=10,
               edgecolors="white", linewidth=1.5, marker="^", label="起点")
    ax.scatter(real_end[0], real_end[1], c="red", s=150, zorder=10,
               edgecolors="white", linewidth=1.5, marker="s", label="终点")
    ax.set_title("路径对比: 模型预测 vs 真实输电线", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")

    # --- 右图: 距离偏差分布 ---
    ax2 = axes[1]
    # 沿预测路径采样, 计算到真实路径的距离
    pred_sample = np.array(predicted_coords)[::5]
    real_sample = _dense_sample_line(np.array(real_coords), spacing_m=100)
    distances = []
    for pp in pred_sample:
        d = min(haversine_m(pp[0], pp[1], rp[0], rp[1]) for rp in real_sample)
        distances.append(d)
    cumul_km = np.arange(len(distances)) * cfg.PATH_RESAMPLE_SPACING * 10 / 1000

    ax2.fill_between(cumul_km, distances, alpha=0.3, color="#FF8844")
    ax2.plot(cumul_km, distances, color="#CC3300", lw=1.2)
    ax2.axhline(y=500, color="orange", linestyle="--", alpha=0.6, label="500m阈值")
    ax2.axhline(y=1000, color="red", linestyle="--", alpha=0.6, label="1km阈值")
    ax2.axhline(y=mean_dist, color="blue", linestyle="-", alpha=0.8, label=f"平均={mean_dist:.0f}m")
    ax2.set_xlabel("沿预测路径距离 (km)", fontsize=11)
    ax2.set_ylabel("到真实线路最近距离 (m)", fontsize=11)
    ax2.set_title("预测路径→真实线路距离偏差分布", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # 标题信息
    info_text = (f"Hausdorff={hausdorff:.0f}m | 平均距离={mean_dist:.0f}m | "
                 f"500m重叠={overlap_500m*100:.0f}% | 长度误差={len_error_pct:.1f}%")
    fig.suptitle(f"输电线路路径规划模型验证\n{info_text}",
                 fontsize=13, fontweight="bold", y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    val_path = output_dir / "maps" / "validation_comparison.png"
    fig.savefig(val_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  对比图已保存: {val_path}")

    # ============================================================
    # 6. 高程剖面对比
    # ============================================================
    fig2, ax_elev = plt.subplots(figsize=(18, 7), dpi=300)

    # 预测路径高程
    pred_elevs, pred_dists = _extract_elev_profile(predicted_coords, dem_aligned, dst_transform)
    # 真实路径高程
    real_elevs, real_dists = _extract_elev_profile(
        [(c[0], c[1]) for c in real_coords], dem_aligned, dst_transform)

    ax_elev.plot(real_dists, real_elevs, color="#3388FF", lw=2.0, alpha=0.7, label="真实线路高程")
    ax_elev.plot(pred_dists, pred_elevs, color="#FF2D2D", lw=2.0, alpha=0.7, label="预测路径高程")
    ax_elev.fill_between(real_dists, real_elevs, alpha=0.1, color="#3388FF")
    ax_elev.fill_between(pred_dists, pred_elevs, alpha=0.1, color="#FF2D2D")
    ax_elev.set_xlabel("距离 (km)", fontsize=12)
    ax_elev.set_ylabel("高程 (m)", fontsize=12)
    ax_elev.set_title("高程剖面对比: 真实线路 vs 预测路径", fontsize=14, fontweight="bold")
    ax_elev.legend(fontsize=11)
    ax_elev.grid(True, alpha=0.3)

    elev_path = output_dir / "maps" / "validation_elevation.png"
    fig2.savefig(elev_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  高程对比图已保存: {elev_path}")

    # ============================================================
    # 7. 导出对比GeoJSON (用于QGIS)
    # ============================================================
    pred_line = LineString(predicted_coords)
    real_line = LineString([(c[0], c[1]) for c in real_coords])

    comp_gdf = gpd.GeoDataFrame({
        "name": ["真实输电线", "模型预测路径"],
        "length_km": [real_length, pred_length],
        "type": ["actual", "predicted"],
        "geometry": [real_line, pred_line],
    }, crs=cfg.WGS84)
    comp_gdf.to_file(output_dir / "data" / "validation_comparison.geojson", driver="GeoJSON")
    print(f"  对比GeoJSON已保存: {output_dir / 'data' / 'validation_comparison.geojson'}")

    # ============================================================
    # 8. 保存分析报告
    # ============================================================
    report = {
        "validation_line_id": target_id,
        "start": (float(real_start[1]), float(real_start[0])),
        "end": (float(real_end[1]), float(real_end[0])),
        "real_length_km": real_length,
        "predicted_length_km": pred_length,
        "length_error_pct": len_error_pct,
        "hausdorff_distance_m": hausdorff,
        "mean_nearest_distance_m": mean_dist,
        "overlap_within_500m_pct": overlap_500m * 100,
        "overlap_within_1km_pct": overlap_1km * 100,
        "overlap_within_2km_pct": overlap_2km * 100,
        "acceptable": acceptable,
        "reasons": reasons,
    }
    report_path = output_dir / "reports" / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  验证报告已保存: {report_path}")

    print("\n" + "=" * 60)
    print("  验证完成!")
    print("=" * 60)
    return report


def _draw_hillshade_bg(ax, dem, transform):
    if dem is None:
        return
    sample = 4
    dem_sub = dem[::sample, ::sample]
    dem_sub = np.nan_to_num(dem_sub, nan=0)
    dy, dx = np.gradient(dem_sub.astype(np.float64))
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(dy, dx)
    hs = np.cos(alt)*np.cos(slope) + np.sin(alt)*np.sin(slope)*np.cos(az-aspect)
    hs = np.clip(hs*255, 0, 255)
    extent = (transform.c, transform.c + transform.a*dem.shape[1],
              transform.f + transform.e*dem.shape[0], transform.f)
    ax.imshow(hs, extent=extent, cmap="gray", alpha=0.7, origin="upper")


def _draw_line(ax, coords, color, lw, label):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    ax.plot(lons, lats, color=color, lw=lw, label=label, alpha=0.85)


def _extract_elev_profile(coords, dem, transform):
    dists = [0.0]
    elevs = []
    for i, (lon, lat) in enumerate(coords):
        r, c = geo_to_grid(lat, lon, transform)
        if i > 0:
            d = haversine_m(coords[i-1][0], coords[i-1][1], lon, lat) / 1000
            dists.append(dists[-1] + d)
        if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
            elevs.append(float(dem[r, c]))
        else:
            elevs.append(np.nan)
    return elevs, dists[:len(elevs)]


if __name__ == "__main__":
    main()
