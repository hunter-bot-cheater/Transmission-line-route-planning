"""
v3: 输出模块 — 遵循 OUTPUT_SPEC.md 规范
版本: v3.20260710
作者: path_planning_team
描述: 每条线路每种算法独立输出 SHP/GeoJSON/统计/质量报告/收敛曲线
依赖: v3/config, v2/src/path_planning (工具函数)
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent.parent
_v2_dir = _v3_dir.parent / "v2_20260525"
for _d in reversed([_v3_dir, _v2_dir, _v2_dir / "src"]):
    sys.path.insert(0, str(_d))

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import json
import config as cfg
from path_planning import compute_path_length_km

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

VERSION = "v3"


def _algo_slug(algorithm: str) -> str:
    """算法名 → 文件安全标识"""
    return {"A*": "astar", "IPSO-SA": "ipso-sa", "DBO": "dbo"}.get(algorithm, algorithm.lower().replace(" ", "-"))


def export_path(coords: list, output_dir: Path, case_id: str, algorithm: str) -> tuple | None:
    """导出路径为 SHP + GeoJSON"""
    if not coords:
        return None
    slug = _algo_slug(algorithm)
    geom = LineString([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame({"id": [case_id], "algorithm": [algorithm]}, geometry=[geom], crs=cfg.WGS84)
    shp_path = output_dir / f"{case_id}_{slug}_path_v3.shp"
    geojson_path = output_dir / f"{case_id}_{slug}_path_v3.geojson"
    gdf.to_file(shp_path)
    gdf.to_file(geojson_path, driver="GeoJSON")
    return shp_path, geojson_path


def export_statistics(coords: list, aligned: dict, hard_mask: np.ndarray,
                       transform, output_dir: Path, case_id: str,
                       algorithm: str) -> dict:
    """计算并导出路径统计"""
    slug = _algo_slug(algorithm)
    stats = {"case_id": case_id, "algorithm": algorithm, "version": f"v3.{cfg.VERSION_DATE}" if hasattr(cfg, "VERSION_DATE") else "v3"}
    stats["length_km"] = compute_path_length_km(coords)

    dem = aligned.get("dem")
    slope = aligned.get("slope")

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
        violations = sum(1 for lon, lat in coords
                         if 0 <= int((lat - transform.f) / transform.e) < hard_mask.shape[0]
                         and 0 <= int((lon - transform.c) / transform.a) < hard_mask.shape[1]
                         and hard_mask[int((lat - transform.f) / transform.e), int((lon - transform.c) / transform.a)] == 0)
        stats["hard_constraint_violations"] = violations

    stats_path = output_dir / f"{case_id}_{slug}_statistics_v3.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def export_quality_report(quality_result: dict, output_dir: Path, case_id: str,
                           algorithm: str) -> Path:
    """导出质量门控报告"""
    slug = _algo_slug(algorithm)
    report = {"case_id": case_id, "algorithm": algorithm, **quality_result}
    report_path = output_dir / f"{case_id}_{slug}_quality_report_v3.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def export_convergence(convergence_curve: list | None, best_fitness: float,
                        output_dir: Path, case_id: str, algorithm: str) -> Path | None:
    """导出收敛曲线数据 + PNG 图"""
    if convergence_curve is None:
        return None
    slug = _algo_slug(algorithm)

    # JSON 数据
    conv_data = {
        "case_id": case_id, "algorithm": algorithm,
        "iterations": len(convergence_curve),
        "best_fitness": best_fitness,
        "curve": convergence_curve,
    }
    json_path = output_dir / f"{case_id}_{slug}_convergence_v3.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(conv_data, f, ensure_ascii=False, indent=2)

    # PNG 图
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(convergence_curve)), convergence_curve, linewidth=1)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("最优适应度")
    ax.set_title(f"{case_id} — {algorithm} 收敛曲线")
    ax.grid(True, alpha=0.3)
    png_path = output_dir / f"{case_id}_{slug}_convergence_v3.png"
    fig.savefig(png_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path


def export_comparison_chart(all_results: list, output_dir: Path) -> Path:
    """导出三算法对比汇总图 (Hausdorff / 重叠率 / 耗时)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    algos = ["A*", "IPSO-SA", "DBO"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    cases = [r["case_id"] for r in all_results]

    # Hausdorff
    ax = axes[0]
    x = np.arange(len(cases))
    for algo, color in zip(algos, colors):
        vals = [r["algorithms"].get(algo, {}).get("hausdorff_m", 0) for r in all_results]
        ax.bar(x + algos.index(algo) * 0.25, vals, 0.25, label=algo, color=color)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hausdorff (m)")
    ax.set_title("Hausdorff 距离对比")
    ax.legend()

    # Overlap
    ax = axes[1]
    for algo, color in zip(algos, colors):
        vals = [r["algorithms"].get(algo, {}).get("overlap_500m_pct", 0) for r in all_results]
        ax.bar(x + algos.index(algo) * 0.25, vals, 0.25, label=algo, color=color)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("重叠率 (%)")
    ax.set_title("500m 重叠率对比")
    ax.legend()

    # Time
    ax = axes[2]
    for algo, color in zip(algos, colors):
        vals = [r["algorithms"].get(algo, {}).get("time_s", 0) for r in all_results]
        ax.bar(x + algos.index(algo) * 0.25, vals, 0.25, label=algo, color=color)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("耗时 (s)")
    ax.set_title("计算耗时对比")
    ax.legend()

    fig.tight_layout()
    png_path = output_dir / "validation_v3_summary.png"
    fig.savefig(png_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return png_path
