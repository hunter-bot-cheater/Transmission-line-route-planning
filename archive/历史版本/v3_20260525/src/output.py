"""
v3_dl: 模块5 — 输出与可视化(深度学习对比版)
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: 新增真实vs规划对比图、高程剖面对比、局部细节图、综合汇总
  - v2.20260525: SHP/GeoJSON导出+基本统计+质量报告
  - v1.20260525: 基本可视化
依赖: v3/config
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely.geometry import LineString, Point, box as sbox
from scipy.ndimage import gaussian_filter
import json
import math
import rasterio
from rasterio.features import rasterize
from typing import Optional, List, Tuple, Dict

import config as cfg

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 矢量导出
# ============================================================
def export_path(coords, output_dir, case_id="optimal"):
    if not coords:
        return None
    geom = LineString([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame({"id": [case_id], "geometry": [geom]}, crs=cfg.WGS84)
    shp_path = output_dir / f"{case_id}_path_v3.shp"
    geojson_path = output_dir / f"{case_id}_path_v3.geojson"
    gdf.to_file(shp_path)
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"  导出: {shp_path}")
    return shp_path, geojson_path


def export_real_path(coords, output_dir, case_id="case"):
    """导出真实线路路径"""
    if not coords:
        return None
    geom = LineString([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame({"id": [f"{case_id}_real"], "geometry": [geom]}, crs=cfg.WGS84)
    shp_path = output_dir / f"{case_id}_real_path.shp"
    gdf.to_file(shp_path)
    return shp_path


# ============================================================
# 统计指标计算
# ============================================================
def compute_statistics(coords, final_cost, dem, slope, hard_mask, transform,
                       output_dir, case_id="case", version="v3.20260525"):
    stats = {"case_id": case_id, "version": version}

    path_len_km = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
        path_len_km += 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    stats["length_km"] = path_len_km

    if dem is not None and len(coords) > 0:
        elevs = []
        for lon, lat in coords:
            r = int((lat - transform.f) / transform.e)
            c = int((lon - transform.c) / transform.a)
            if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
                elevs.append(float(dem[r, c]))
        if elevs:
            stats["elevation_min_m"] = min(elevs)
            stats["elevation_max_m"] = max(elevs)
            stats["elevation_mean_m"] = float(np.mean(elevs))

    if slope is not None and len(coords) > 0:
        slopes = []
        for lon, lat in coords:
            r = int((lat - transform.f) / transform.e)
            c = int((lon - transform.c) / transform.a)
            if 0 <= r < slope.shape[0] and 0 <= c < slope.shape[1]:
                slopes.append(float(slope[r, c]))
        if slopes:
            stats["slope_max_deg"] = max(slopes)
            stats["slope_mean_deg"] = float(np.mean(slopes))

    if hard_mask is not None:
        violations = 0
        for lon, lat in coords:
            r = int((lat - transform.f) / transform.e)
            c = int((lon - transform.c) / transform.a)
            if 0 <= r < hard_mask.shape[0] and 0 <= c < hard_mask.shape[1]:
                if hard_mask[r, c] == 0:
                    violations += 1
        stats["hard_constraint_violations"] = violations

    if len(coords) >= 2:
        start = coords[0]
        end = coords[-1]
        dlat = math.radians(end[1] - start[1])
        dlon = math.radians(end[0] - start[0])
        a = (math.sin(dlat/2)**2 + math.cos(math.radians(start[1])) *
             math.cos(math.radians(end[1])) * math.sin(dlon/2)**2)
        straight_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        stats["straight_line_km"] = straight_km
        stats["sinuosity"] = path_len_km / straight_km if straight_km > 0 else 1.0

    stats_path = output_dir / f"{case_id}_statistics_v3.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def compute_comparison_metrics(planned_coords, real_coords):
    """
    v3: 计算规划路径与真实路径的对比指标。
    """
    if not planned_coords or not real_coords:
        return {}

    planned_arr = np.array(planned_coords)
    real_arr = np.array(real_coords)

    # Hausdorff距离
    from scipy.spatial import cKDTree
    tree_p = cKDTree(planned_arr)
    tree_r = cKDTree(real_arr)

    dist_p_to_r, _ = tree_r.query(planned_arr)
    dist_r_to_p, _ = tree_p.query(real_arr)
    hausdorff = max(np.max(dist_p_to_r), np.max(dist_r_to_p)) * cfg.METERS_PER_DEG

    # 平均最近距离
    mean_dist = float(np.mean(dist_p_to_r) * cfg.METERS_PER_DEG)

    # 500m缓冲区重叠率
    buffer_m = 500
    buffer_deg = buffer_m / cfg.METERS_PER_DEG
    overlap_rate = _compute_buffer_overlap(planned_coords, real_coords, buffer_deg)

    # 路径长度
    planned_len = 0.0
    for i in range(len(planned_coords) - 1):
        planned_len += _haversine_km(planned_coords[i], planned_coords[i+1])
    real_len = 0.0
    for i in range(len(real_coords) - 1):
        real_len += _haversine_km(real_coords[i], real_coords[i+1])

    return {
        "hausdorff_m": float(hausdorff),
        "mean_distance_m": float(mean_dist),
        "overlap_500m": float(overlap_rate),
        "planned_length_km": float(planned_len),
        "real_length_km": float(real_len),
        "length_error_pct": float(abs(planned_len - real_len) / real_len * 100),
    }


def _haversine_km(p1, p2):
    lon1, lat1 = p1
    lon2, lat2 = p2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _compute_buffer_overlap(coords_a, coords_b, buffer_deg):
    from shapely.geometry import LineString
    line_a = LineString([(c[0], c[1]) for c in coords_a])
    line_b = LineString([(c[0], c[1]) for c in coords_b])
    buf_a = line_a.buffer(buffer_deg)
    buf_b = line_b.buffer(buffer_deg)
    intersection = buf_a.intersection(buf_b)
    union = buf_a.union(buf_b)
    if union.area == 0:
        return 0.0
    return intersection.area / union.area


# ============================================================
# 对比可视化 — v3核心
# ============================================================
def plot_comparison_overview(
    planned_coords, real_coords, dem, transform, case_id,
    output_dir, hard_mask=None, extent=None,
):
    """
    v3: 路径对比总览图 — 真实vs规划。
    """
    print(f"  [{case_id}] 绘制对比总览图...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=cfg.FIGURE_DPI)

    # 公共渲染参数
    dem_vmin = np.nanpercentile(dem, 2)
    dem_vmax = np.nanpercentile(dem, 98)

    for ax_idx, (coords, title, color) in enumerate([
        (planned_coords, f"{case_id} — DL规划路径 (v3)", "#E31A1C"),
        (real_coords, f"{case_id} — 真实输电线路", "#2166AC"),
    ]):
        ax = axes[ax_idx]
        ax.imshow(dem, cmap="terrain", vmin=dem_vmin, vmax=dem_vmax,
                  extent=extent, aspect="auto", alpha=0.8)

        if hard_mask is not None:
            mask_overlay = np.ma.masked_where(hard_mask == 1, hard_mask)
            ax.imshow(mask_overlay, cmap="Reds", alpha=0.15, extent=extent, aspect="auto")

        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            ax.plot(lons, lats, color=color, linewidth=1.8, label=title)
            ax.scatter(lons[0], lats[0], c="green", s=80, marker="o",
                      edgecolors="white", linewidths=1, zorder=5, label="起点")
            ax.scatter(lons[-1], lats[-1], c="red", s=80, marker="s",
                      edgecolors="white", linewidths=1, zorder=5, label="终点")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("经度 (°E)")
        ax.set_ylabel("纬度 (°N)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"v3深度学习路径规划 vs 真实线路对比 — {case_id}",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    save_path = output_dir / f"{case_id}_comparison_overview_v3.png"
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return save_path


def plot_elevation_profile_comparison(
    planned_coords, real_coords, dem, transform, case_id, output_dir,
):
    """
    v3: 高程剖面对比图 — 规划vs真实。
    """
    print(f"  [{case_id}] 绘制高程剖面对比图...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=cfg.FIGURE_DPI)
    fig.subplots_adjust(hspace=0.35)

    for ax, (coords, label, color) in [
        (ax1, planned_coords, "DL规划路径 (v3)", "#E31A1C"),
        (ax2, real_coords, "真实输电线路", "#2166AC"),
    ]:
        if not coords:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
            continue

        elevations = []
        distances = [0]
        for i, (lon, lat) in enumerate(coords):
            r = int((lat - transform.f) / transform.e)
            c = int((lon - transform.c) / transform.a)
            if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
                elevations.append(float(dem[r, c]))
            if i > 0:
                d = _haversine_km(coords[i-1], coords[i])
                distances.append(distances[-1] + d)

        dists_km = np.array(distances[:len(elevations)])
        elevs = np.array(elevations)

        ax.fill_between(dists_km, elevs - 50, elevs, alpha=0.3, color=color)
        ax.plot(dists_km, elevs, color=color, linewidth=1.5)
        ax.set_xlabel("沿线距离 (km)")
        ax.set_ylabel("高程 (m)")
        ax.set_title(f"{label} — 高程剖面", fontweight="bold")
        ax.grid(True, alpha=0.3)

        stats_text = (f"长度: {dists_km[-1]:.1f}km | "
                      f"高程: {elevs.min():.0f}-{elevs.max():.0f}m | "
                      f"均值: {elevs.mean():.0f}m")
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle(f"高程剖面对比 — {case_id}", fontsize=14, fontweight="bold")
    save_path = output_dir / f"{case_id}_elevation_profile_v3.png"
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return save_path


def plot_single_path_overview(
    coords, dem, slope, hard_mask, transform, case_id, output_dir, extent=None,
):
    """
    单路径总览图 — DEM背景 + 路径 + 硬约束叠加。
    """
    print(f"  [{case_id}] 绘制路径总览图...")
    fig, ax = plt.subplots(figsize=(12, 10), dpi=cfg.FIGURE_DPI)

    dem_vmin = np.nanpercentile(dem, 2)
    dem_vmax = np.nanpercentile(dem, 98)

    dem_show = ax.imshow(dem, cmap="terrain", vmin=dem_vmin, vmax=dem_vmax,
                         extent=extent, aspect="auto", alpha=0.85)

    if hard_mask is not None:
        mask_overlay = np.ma.masked_where(hard_mask == 1, hard_mask)
        ax.imshow(mask_overlay, cmap="Reds", alpha=0.2, extent=extent, aspect="auto")

    if slope is not None:
        slope_overlay = np.ma.masked_where(slope < 25, slope)
        ax.imshow(slope_overlay, cmap="Oranges", alpha=0.1, extent=extent, aspect="auto")

    if coords:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        ax.plot(lons, lats, color="#E31A1C", linewidth=2.0, label="v3规划路径")
        ax.scatter(lons[0], lats[0], c="green", s=100, marker="o",
                  edgecolors="white", linewidths=1.5, zorder=5, label="起点")
        ax.scatter(lons[-1], lats[-1], c="red", s=100, marker="s",
                  edgecolors="white", linewidths=1.5, zorder=5, label="终点")

    cbar = plt.colorbar(dem_show, ax=ax, shrink=0.75)
    cbar.set_label("高程 (m)")

    ax.set_title(f"v3 DL路径规划 — {case_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    save_path = output_dir / f"{case_id}_overview_v3.png"
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return save_path


# ============================================================
# 汇总报告
# ============================================================
def generate_summary_report(all_results, output_dir):
    """
    v3: 生成10条线路的综合对比汇总报告。
    """
    print("\n[Phase5] 生成汇总对比报告...")
    report_path = output_dir / "v3_comparison_summary.json"

    # 汇总指标
    summary = {
        "version": "v3.20260525",
        "total_cases": len(all_results),
        "passed_cases": sum(1 for r in all_results if r.get("quality_passed")),
        "comparison_metrics": {},
        "aggregate_stats": {},
    }

    hausdorffs = []
    overlaps = []
    length_errors = []
    for r in all_results:
        case_id = r.get("case_id", "unknown")
        metrics = r.get("comparison_metrics", {})
        summary["comparison_metrics"][case_id] = metrics
        if metrics.get("hausdorff_m"):
            hausdorffs.append(metrics["hausdorff_m"])
        if metrics.get("overlap_500m"):
            overlaps.append(metrics["overlap_500m"])
        if metrics.get("length_error_pct"):
            length_errors.append(metrics["length_error_pct"])

    if hausdorffs:
        summary["aggregate_stats"]["hausdorff_mean_m"] = float(np.mean(hausdorffs))
        summary["aggregate_stats"]["hausdorff_median_m"] = float(np.median(hausdorffs))
    if overlaps:
        summary["aggregate_stats"]["overlap_500m_mean"] = float(np.mean(overlaps))
    if length_errors:
        summary["aggregate_stats"]["length_error_mean_pct"] = float(np.mean(length_errors))

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  汇总报告: {report_path}")

    # 绘制汇总图表
    _plot_summary_charts(all_results, output_dir)
    return summary


def _plot_summary_charts(all_results, output_dir):
    """绘制10条线路的综合对比图表"""
    print("  生成综合对比图表...")
    case_ids = [r.get("case_id", "") for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=cfg.FIGURE_DPI)

    # 1. Hausdorff距离柱状图
    ax = axes[0, 0]
    hausdorffs = [r.get("comparison_metrics", {}).get("hausdorff_m", 0) for r in all_results]
    bars = ax.bar(range(len(case_ids)), hausdorffs, color="#E31A1C", alpha=0.8)
    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hausdorff距离 (m)")
    ax.set_title("Hausdorff距离对比 (越小越好)")
    ax.axhline(y=np.mean(hausdorffs), color="blue", linestyle="--",
               label=f"均值: {np.mean(hausdorffs):.0f}m")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 500m重叠率柱状图
    ax = axes[0, 1]
    overlaps = [r.get("comparison_metrics", {}).get("overlap_500m", 0) for r in all_results]
    ax.bar(range(len(case_ids)), overlaps, color="#2166AC", alpha=0.8)
    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("500m缓冲重叠率")
    ax.set_title("空间重叠率 (越大越好)")
    ax.axhline(y=np.mean(overlaps), color="red", linestyle="--",
               label=f"均值: {np.mean(overlaps):.2%}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 长度误差柱状图
    ax = axes[1, 0]
    length_errs = [r.get("comparison_metrics", {}).get("length_error_pct", 0) for r in all_results]
    ax.bar(range(len(case_ids)), length_errs, color="#4DAF4A", alpha=0.8)
    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(case_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("长度误差 (%)")
    ax.set_title("路径长度误差 (越小越好)")
    ax.axhline(y=np.mean(length_errs), color="red", linestyle="--",
               label=f"均值: {np.mean(length_errs):.1f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 质量门控通过率
    ax = axes[1, 1]
    quality_results = [
        r.get("quality_passed", False) for r in all_results
    ]
    n_passed = sum(quality_results)
    ax.pie([n_passed, len(quality_results) - n_passed],
           labels=[f"通过 ({n_passed})", f"未通过 ({len(quality_results) - n_passed})"],
           colors=["#4DAF4A", "#E31A1C"], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"质量门控通过率 (v3)")

    plt.suptitle(f"v3深度学习路径规划 — 10条线路综合对比",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()

    save_path = output_dir / "v3_summary_comparison_charts.png"
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  综合图表: {save_path}")

    # 训练曲线图
    _plot_training_curves(all_results, output_dir)


def _plot_training_curves(all_results, output_dir):
    """绘制CNN训练曲线"""
    train_curves = [r.get("train_losses", []) for r in all_results
                    if r.get("train_losses")]
    if not train_curves:
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=cfg.FIGURE_DPI)

    # 使用第一个结果的曲线
    curves = train_curves[0]
    epochs = range(1, len(curves) + 1)

    ax.plot(epochs, curves, color="#E31A1C", linewidth=1.5, label="训练损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CostUNet 训练曲线 (v3)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = output_dir / "v3_training_curve.png"
    plt.savefig(save_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()


def export_quality_report(quality_result, output_dir, case_id="case"):
    """导出质量审查报告"""
    report_path = output_dir / f"{case_id}_quality_report_v3.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_result, f, ensure_ascii=False, indent=2)
    return report_path
