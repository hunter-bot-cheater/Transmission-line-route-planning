"""
v2_strict: 模块5 — 输出与可视化(严格化版本)
版本: v2.20260525
作者: path_planning_team
变更记录:
  - v2.20260525: 新增quality_report.json导出, 输出路径含版本号
  - v1.20260525: 初始版本
依赖: v2/config
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import json
import rasterio
from rasterio.features import rasterize
import math
from src.path_planning import compute_path_length_km

import config as cfg

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def export_path(coords, output_dir, case_id="optimal"):
    """导出路径为SHP和GeoJSON"""
    if not coords:
        return None
    geom = LineString([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame({"id": [case_id], "geometry": [geom]}, crs=cfg.WGS84)
    shp_path = output_dir / f"{case_id}_path_v2.shp"
    geojson_path = output_dir / f"{case_id}_path_v2.geojson"
    gdf.to_file(shp_path)
    gdf.to_file(geojson_path, driver="GeoJSON")
    return shp_path, geojson_path


def compute_statistics(coords, final_cost, dem, slope, hard_mask, transform, output_dir, case_id="case"):
    """计算并保存统计指标"""
    stats = {"case_id": case_id, "version": "v2.20260525"}
    stats["length_km"] = compute_path_length_km(coords)

    if dem is not None and len(coords) > 0:
        elevs = []
        for lon, lat in coords:
            r = int((lat - transform.f) / transform.e)
            c = int((lon - transform.c) / transform.a)
            H, W = dem.shape
            if 0 <= r < H and 0 <= c < W:
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
            H, W = slope.shape
            if 0 <= r < H and 0 <= c < W:
                slopes.append(float(slope[r, c]))
        if slopes:
            stats["slope_max_deg"] = max(slopes)
            stats["slope_mean_deg"] = float(np.mean(slopes))

    if hard_mask is not None:
        violations = 0
        for lon, lat in coords:
            r = int((lat - transform.f) / transform.e)
            c = int((lon - transform.c) / transform.a)
            H, W = hard_mask.shape
            if 0 <= r < H and 0 <= c < W and hard_mask[r, c] == 0:
                violations += 1
        stats["hard_constraint_violations"] = violations

    stats_path = output_dir / f"{case_id}_statistics_v2.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def export_quality_report(quality_result, output_dir, case_id="case"):
    """v2_strict: 导出7项质量审查报告"""
    report_path = output_dir / f"{case_id}_quality_report_v2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_result, f, ensure_ascii=False, indent=2)
    return report_path
