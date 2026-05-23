"""
模块5: 输出与可视化
- 导出最优路径SHP/GeoJSON
- 计算并导出统计指标(JSON/XLSX)
- 生成4幅高质量可视化图
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
import json
import math
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

import config as cfg

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# SHP / GeoJSON 导出
# ============================================================
def export_path(coords, output_dir):
    """
    导出最优路径为SHP和GeoJSON
    coords: [(lon, lat), ...]
    """
    print("[Phase5] 导出路径文件...")

    line = LineString(coords)
    gdf = gpd.GeoDataFrame({
        "path_id": ["optimal_path"],
        "length_km": [compute_length(coords)],
        "vertices": [len(coords)],
        "geometry": [line],
    }, crs=cfg.WGS84)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # SHP
    shp_path = data_dir / "optimal_path.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"  SHP已保存: {shp_path}")

    # GeoJSON
    geojson_path = data_dir / "optimal_path.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON", encoding="utf-8")
    print(f"  GeoJSON已保存: {geojson_path}")

    # 起止点
    start_pt = Point(coords[0])
    end_pt = Point(coords[-1])
    pts_gdf = gpd.GeoDataFrame({
        "name": ["起点", "终点"],
        "geometry": [start_pt, end_pt],
    }, crs=cfg.WGS84)
    pts_gdf.to_file(data_dir / "start_end_points.shp", driver="ESRI Shapefile", encoding="utf-8")


def compute_length(coords):
    """计算Haversine距离(km)"""
    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        total += 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return total


# ============================================================
# 统计计算
# ============================================================
def compute_statistics(coords, cost_raster, dem, slope, hard_mask, transform, output_dir):
    """计算所有统计指标并导出JSON/XLSX"""
    print("[Phase5] 计算统计指标...")

    stats = {}
    stats["path_length_km"] = compute_length(coords)

    # 直线距离
    lon1, lat1 = coords[0]
    lon2, lat2 = coords[-1]
    stats["straight_line_km"] = haversine(lon1, lat1, lon2, lat2)
    stats["sinuosity"] = stats["path_length_km"] / max(stats["straight_line_km"], 0.01)

    # 提取路径沿线值
    path_elevs = []
    path_slopes = []
    path_costs = []

    for lon, lat in coords:
        r, c = geo_to_grid(lat, lon, transform)
        if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
            if not np.isnan(dem[r, c]):
                path_elevs.append(float(dem[r, c]))
            if slope is not None and not np.isnan(slope[r, c]):
                path_slopes.append(float(slope[r, c]))
            if cost_raster is not None and not np.isinf(cost_raster[r, c]):
                path_costs.append(float(cost_raster[r, c]))

    if path_elevs:
        stats["elevation_min_m"] = min(path_elevs)
        stats["elevation_max_m"] = max(path_elevs)
        stats["elevation_mean_m"] = np.mean(path_elevs)
        stats["total_ascent_m"] = sum(
            max(0, path_elevs[i + 1] - path_elevs[i])
            for i in range(len(path_elevs) - 1)
        )
        stats["total_descent_m"] = sum(
            max(0, path_elevs[i] - path_elevs[i + 1])
            for i in range(len(path_elevs) - 1)
        )

    if path_slopes:
        stats["slope_mean_deg"] = np.mean(path_slopes)
        stats["slope_max_deg"] = np.max(path_slopes)
        stats["slope_pct_gt15"] = sum(1 for s in path_slopes if s > 15) / len(path_slopes) * 100
        stats["slope_pct_gt25"] = sum(1 for s in path_slopes if s > 25) / len(path_slopes) * 100
        stats["slope_pct_gt35"] = sum(1 for s in path_slopes if s > 35) / len(path_slopes) * 100

    if path_costs:
        stats["cost_total"] = sum(path_costs)
        stats["cost_mean"] = np.mean(path_costs)
        stats["cost_max"] = np.max(path_costs)

    # 硬约束穿越检查
    if hard_mask is not None:
        crossings = 0
        for lon, lat in coords:
            r, c = geo_to_grid(lat, lon, transform)
            if 0 <= r < hard_mask.shape[0] and 0 <= c < hard_mask.shape[1]:
                if hard_mask[r, c] == 0:
                    crossings += 1
        stats["hard_constraint_crossings"] = crossings

    stats["vertices"] = len(coords)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # 保存JSON
    json_path = reports_dir / "statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  统计JSON已保存: {json_path}")

    # 保存XLSX
    try:
        import openpyxl
        import pandas as pd
        df = pd.DataFrame([stats])
        xlsx_path = reports_dir / "statistics.xlsx"
        df.T.to_excel(xlsx_path, sheet_name="统计指标", header=False)
        print(f"  统计XLSX已保存: {xlsx_path}")
    except ImportError:
        print("  openpyxl未安装, 跳过XLSX导出")

    # 打印摘要
    print("\n" + "=" * 50)
    print("  路径规划统计摘要")
    print("=" * 50)
    print(f"  路径长度:   {stats['path_length_km']:.2f} km")
    print(f"  直线距离:   {stats['straight_line_km']:.2f} km")
    print(f"  弯曲度:     {stats['sinuosity']:.2f}")
    print(f"  最低高程:   {stats.get('elevation_min_m', 'N/A')} m")
    print(f"  最高高程:   {stats.get('elevation_max_m', 'N/A')} m")
    print(f"  总爬升:     {stats.get('total_ascent_m', 'N/A')} m")
    print(f"  平均坡度:   {stats.get('slope_mean_deg', 'N/A'):.1f}°")
    print(f"  硬约束穿越: {stats.get('hard_constraint_crossings', 'N/A')}")
    print("=" * 50)

    return stats


def haversine(lon1, lat1, lon2, lat2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geo_to_grid(lat, lon, transform):
    c = int((lon - transform.c) / transform.a)
    r = int((lat - transform.f) / transform.e)
    return r, c


# ============================================================
# 可视化 1: 概览图
# ============================================================
def create_overview_map(coords, cost_raster, dem, taiwan_lines, hard_mask, transform, output_dir):
    """台湾全岛概览图 (300dpi)"""
    print("[Phase5] 生成概览图...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 14), dpi=cfg.FIGURE_DPI)

    # --- 左图: 成本表面 + 路径 ---
    ax = axes[0]
    _draw_cost_background(ax, cost_raster, transform)
    _draw_path(ax, coords, color="#FF2D2D", lw=2.0, label="最优路径")
    _draw_start_end(ax, coords)
    _draw_lines(ax, taiwan_lines)

    # 硬约束叠加
    if hard_mask is not None:
        _draw_hard_mask_overlay(ax, hard_mask, transform)

    ax.set_xlim(cfg.TAIWAN_BBOX[0], cfg.TAIWAN_BBOX[2])
    ax.set_ylim(cfg.TAIWAN_BBOX[1], cfg.TAIWAN_BBOX[3])
    ax.set_xlabel("经度 (°E)", fontsize=11)
    ax.set_ylabel("纬度 (°N)", fontsize=11)
    ax.set_title("台湾输电线路最优路径规划 — 成本表面", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect(1.0 / math.cos(math.radians(23.5)))

    # --- 右图: 山影地形 + 路径 ---
    ax2 = axes[1]
    _draw_hillshade_background(ax2, dem, transform)
    _draw_path(ax2, coords, color="#FFD700", lw=2.5, label="最优路径")
    _draw_start_end(ax2, coords)
    _draw_lines(ax2, taiwan_lines)

    ax2.set_xlim(cfg.TAIWAN_BBOX[0], cfg.TAIWAN_BBOX[2])
    ax2.set_ylim(cfg.TAIWAN_BBOX[1], cfg.TAIWAN_BBOX[3])
    ax2.set_xlabel("经度 (°E)", fontsize=11)
    ax2.set_ylabel("纬度 (°N)", fontsize=11)
    ax2.set_title("台湾输电线路最优路径规划 — 地形山影", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_aspect(1.0 / math.cos(math.radians(23.5)))

    fig.suptitle("台湾输电线路智能路径规划", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / "maps" / "map_overview.png"
    fig.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  概览图已保存: {path}")


# ============================================================
# 可视化 2: 局部放大图
# ============================================================
def create_detail_map(coords, cost_raster, dem, hard_mask, transform, output_dir):
    """路径走廊局部放大图 (600dpi)"""
    print("[Phase5] 生成局部放大图...")

    # 计算路径包围盒 + 缓冲区
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    buffer_deg = cfg.DETAIL_BUFFER_KM / 111.0
    xlim = (min(lons) - buffer_deg, max(lons) + buffer_deg)
    ylim = (max(lats) + buffer_deg, min(lats) - buffer_deg)  # 翻转(纬度大的在上)

    fig, ax = plt.subplots(figsize=(16, 12), dpi=600)

    _draw_hillshade_background(ax, dem, transform)
    _draw_path(ax, coords, color="#FF2D2D", lw=3.0, label="最优路径")
    _draw_start_end(ax, coords)

    if hard_mask is not None:
        _draw_hard_mask_overlay(ax, hard_mask, transform)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("经度 (°E)", fontsize=12)
    ax.set_ylabel("纬度 (°N)", fontsize=12)
    ax.set_title(f"路径走廊详情 ({cfg.DETAIL_BUFFER_KM}km 缓冲区)", fontsize=15, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect(1.0 / math.cos(math.radians(np.mean(lats))))

    path = output_dir / "maps" / "map_detail.png"
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  局部放大图已保存: {path}")


# ============================================================
# 可视化 3: 高程剖面图
# ============================================================
def create_elevation_profile(coords, dem, slope, transform, output_dir):
    """路径高程剖面图"""
    print("[Phase5] 生成高程剖面图...")

    # 沿路径累积距离和高程
    distances = [0.0]
    elevations = []
    slopes_deg = []

    for i, (lon, lat) in enumerate(coords):
        r, c = geo_to_grid(lat, lon, transform)
        if i > 0:
            d = haversine(coords[i - 1][0], coords[i - 1][1], lon, lat)
            distances.append(distances[-1] + d)

        if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]:
            elevations.append(float(dem[r, c]))
            if slope is not None:
                slopes_deg.append(float(slope[r, c]))
        else:
            elevations.append(np.nan)
            slopes_deg.append(np.nan)

    distances = distances[:len(elevations)]

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), dpi=cfg.FIGURE_DPI,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # 上: 高程剖面
    # 按坡度着色
    elev_arr = np.array(elevations)
    slope_arr = np.array(slopes_deg)
    dist_arr = np.array(distances)

    scatter = ax1.scatter(dist_arr, elev_arr, c=slope_arr, cmap="RdYlGn_r",
                          s=2, vmin=0, vmax=45, alpha=0.8)
    ax1.plot(dist_arr, elev_arr, color="#333333", lw=0.5, alpha=0.5)
    ax1.fill_between(dist_arr, elev_arr, min(elev_arr) - 50, alpha=0.15, color="#3388FF")
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label("坡度 (°)", fontsize=11)

    # 标注起止点
    ax1.annotate(f"起点\n{elev_arr[0]:.0f}m", (dist_arr[0], elev_arr[0]),
                 xytext=(5, 20), textcoords="offset points", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                 arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax1.annotate(f"终点\n{elev_arr[-1]:.0f}m", (dist_arr[-1], elev_arr[-1]),
                 xytext=(-60, -30), textcoords="offset points", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax1.set_ylabel("高程 (m)", fontsize=12)
    ax1.set_title("路径高程剖面", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 下: 坡度分布
    ax2.fill_between(dist_arr, slope_arr, 0, alpha=0.5, color="#FF8844")
    ax2.plot(dist_arr, slope_arr, color="#CC3300", lw=0.8)
    ax2.axhline(y=15, color="orange", linestyle="--", alpha=0.5, label="15°警戒线")
    ax2.axhline(y=35, color="red", linestyle="--", alpha=0.5, label="35°禁止线")
    ax2.set_xlabel("距离 (km)", fontsize=12)
    ax2.set_ylabel("坡度 (°)", fontsize=12)
    ax2.set_title("路径坡度分布", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "maps" / "elevation_profile.png"
    fig.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  高程剖面图已保存: {path}")


# ============================================================
# 可视化 4: 交互式HTML地图
# ============================================================
def create_interactive_map(coords, cost_raster, dem, taiwan_lines, hard_mask, transform, output_dir):
    """生成独立Leaflet交互式HTML地图"""
    print("[Phase5] 生成交互式地图...")

    # 将路径和输电线数据内嵌为JSON
    path_json = json.dumps([[lat, lon] for lon, lat in coords], ensure_ascii=False)

    # 输电线(最多500条以控制文件大小)
    lines_json = []
    lines_gdf = taiwan_lines if taiwan_lines is not None else None
    if lines_gdf is not None and len(lines_gdf) > 0:
        for _, row in lines_gdf.head(500).iterrows():
            try:
                if row.geometry and hasattr(row.geometry, "coords"):
                    coords_list = [[lat, lon] for lon, lat in row.geometry.coords[:100]]
                    if coords_list:
                        lines_json.append(coords_list)
            except Exception:
                continue
    lines_geojson = json.dumps(lines_json, ensure_ascii=False)

    # 起止点
    start_json = json.dumps([coords[0][1], coords[0][0]], ensure_ascii=False)
    end_json = json.dumps([coords[-1][1], coords[-1][0]], ensure_ascii=False)

    # 路径中心
    center_lat = sum(c[1] for c in coords) / len(coords)
    center_lon = sum(c[0] for c in coords) / len(coords)

    html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>台湾输电线路最优路径 — 交互式地图</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; padding:0; font-family: "Microsoft YaHei", sans-serif; }}
  #map {{ width:100vw; height:100vh; }}
  .info-panel {{
    position:absolute; top:10px; right:10px; z-index:1000;
    background:rgba(255,255,255,0.95); padding:15px 20px;
    border-radius:8px; box-shadow:0 2px 12px rgba(0,0,0,0.3);
    max-width:320px; font-size:14px;
  }}
  .info-panel h3 {{ margin:0 0 10px 0; color:#002255; font-size:18px; }}
  .info-panel table {{ width:100%; border-collapse:collapse; }}
  .info-panel td {{ padding:4px 6px; border-bottom:1px solid #eee; }}
  .info-panel .label {{ color:#666; font-size:12px; }}
  .info-panel .value {{ font-weight:bold; color:#333; text-align:right; }}
  .legend {{ position:absolute; bottom:30px; left:10px; z-index:1000;
    background:rgba(255,255,255,0.9); padding:10px 14px;
    border-radius:6px; box-shadow:0 1px 8px rgba(0,0,0,0.2); font-size:13px; }}
  .legend span {{ display:inline-block; width:20px; height:12px; margin-right:6px; border-radius:2px; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel">
  <h3>台湾输电线路最优路径</h3>
  <table>
    <tr><td class="label">路径长度</td><td class="value" id="info-length">--</td></tr>
    <tr><td class="label">起点</td><td class="value">{coords[0][1]:.4f}, {coords[0][0]:.4f}</td></tr>
    <tr><td class="label">终点</td><td class="value">{coords[-1][1]:.4f}, {coords[-1][0]:.4f}</td></tr>
    <tr><td class="label">顶点数</td><td class="value">{len(coords)}</td></tr>
  </table>
</div>
<div class="legend">
  <div><span style="background:#FF2D2D;"></span> 最优路径</div>
  <div><span style="background:#3388FF;"></span> 现有输电线</div>
  <div><span style="background:green;"></span> 起点</div>
  <div><span style="background:red;"></span> 终点</div>
  <div><span style="background:rgba(255,0,0,0.3);"></span> 硬约束区</div>
</div>
<script>
const pathCoords = {path_json};
const existingLines = {lines_geojson};
const startPt = {start_json};
const endPt = {end_json};

const map = L.map("map").setView([{center_lat}, {center_lon}], 8);

// 底图
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
  attribution: '&copy; OpenStreetMap contributors',
  maxZoom: 18,
}}).addTo(map);

// 最优路径
L.polyline(pathCoords, {{
  color: "#FF2D2D", weight: 4, opacity: 0.9,
  dashArray: null,
}}).addTo(map).bindPopup("<b>最优输电线路路径</b>");

// 现有输电线
existingLines.forEach(function(coords) {{
  L.polyline(coords, {{
    color: "#3388FF", weight: 1.5, opacity: 0.5,
  }}).addTo(map);
}});

// 起止点
L.circleMarker(startPt, {{
  radius: 10, color: "green", fillColor: "#00FF00", fillOpacity: 0.8
}}).addTo(map).bindPopup("<b>起点: 核三厂(屏东)</b>");

L.circleMarker(endPt, {{
  radius: 10, color: "red", fillColor: "#FF0000", fillOpacity: 0.8
}}).addTo(map).bindPopup("<b>终点: 台北</b>");

// 计算路径长度
let totalDist = 0;
for (let i = 1; i < pathCoords.length; i++) {{
  const lat1 = pathCoords[i-1][0] * Math.PI / 180;
  const lon1 = pathCoords[i-1][1] * Math.PI / 180;
  const lat2 = pathCoords[i][0] * Math.PI / 180;
  const lon2 = pathCoords[i][1] * Math.PI / 180;
  const dlat = lat2 - lat1;
  const dlon = lon2 - lon1;
  const a = Math.sin(dlat/2)**2 + Math.cos(lat1)*Math.cos(lat2)*Math.sin(dlon/2)**2;
  totalDist += 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}
document.getElementById("info-length").textContent = totalDist.toFixed(2) + " km";

// 添加比例尺
L.control.scale({{imperial: false, metric: true}}).addTo(map);
</script>
</body>
</html>'''

    html_path = output_dir / "maps" / "interactive_map.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  交互式地图已保存: {html_path}")


# ============================================================
# 辅助绘图函数
# ============================================================
def _draw_cost_background(ax, cost_raster, transform):
    """绘制成本表面背景"""
    if cost_raster is None:
        return

    # 降采样渲染
    sample = 4
    cost_sub = cost_raster[::sample, ::sample]
    cost_sub = np.where(np.isinf(cost_sub), np.nan, cost_sub)

    height, width = cost_sub.shape
    extent = (
        transform.c,
        transform.c + transform.a * cost_raster.shape[1],
        transform.f + transform.e * cost_raster.shape[0],
        transform.f,
    )

    im = ax.imshow(cost_sub, extent=extent, cmap="RdYlGn_r", alpha=0.7,
                   vmin=np.nanpercentile(cost_sub, 2), vmax=np.nanpercentile(cost_sub, 98),
                   origin="upper", aspect="auto", interpolation="bilinear")
    plt.colorbar(im, ax=ax, shrink=0.6, label="建设成本指数")


def _draw_hillshade_background(ax, dem, transform):
    """绘制山影背景"""
    if dem is None:
        return

    sample = 4
    dem_sub = dem[::sample, ::sample]
    dem_sub = np.nan_to_num(dem_sub, nan=0)

    # 山影计算
    dy, dx = np.gradient(dem_sub.astype(np.float64))
    azimuth = 315.0
    altitude = 45.0
    az_rad = math.radians(360 - azimuth + 90)
    alt_rad = math.radians(altitude)

    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect_rad = np.arctan2(dy, dx)

    hillshade = (np.cos(alt_rad) * np.cos(slope_rad) +
                 np.sin(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad))
    hillshade = np.clip(hillshade * 255, 0, 255)

    extent = (
        transform.c,
        transform.c + transform.a * dem.shape[1],
        transform.f + transform.e * dem.shape[0],
        transform.f,
    )

    ax.imshow(hillshade, extent=extent, cmap="gray", alpha=0.8,
              origin="upper", aspect="auto")


def _draw_path(ax, coords, color, lw, label):
    """绘制路径线"""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    ax.plot(lons, lats, color=color, lw=lw, label=label, zorder=5, alpha=0.9)


def _draw_start_end(ax, coords):
    """标注起止点"""
    ax.scatter(coords[0][0], coords[0][1], c="green", s=120, zorder=10,
               edgecolors="white", linewidth=1.5, marker="^", label="起点")
    ax.scatter(coords[-1][0], coords[-1][1], c="red", s=120, zorder=10,
               edgecolors="white", linewidth=1.5, marker="s", label="终点")


def _draw_lines(ax, gdf):
    """绘制现有输电线"""
    if gdf is None or len(gdf) == 0:
        return
    for _, row in gdf.head(300).iterrows():
        try:
            if row.geometry and hasattr(row.geometry, "coords"):
                coords = list(row.geometry.coords)
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                ax.plot(lons, lats, color="#3388FF", lw=0.5, alpha=0.4)
        except Exception:
            continue


def _draw_hard_mask_overlay(ax, hard_mask, transform):
    """绘制硬约束区叠加"""
    if hard_mask is None:
        return

    # 降采样
    sample = 8
    mask_sub = hard_mask[::sample, ::sample]
    height, width = mask_sub.shape
    extent = (
        transform.c,
        transform.c + transform.a * hard_mask.shape[1],
        transform.f + transform.e * hard_mask.shape[0],
        transform.f,
    )

    # 红色半透明叠加
    mask_rgba = np.zeros((*mask_sub.shape, 4), dtype=np.float32)
    mask_rgba[mask_sub == 0, 0] = 1.0  # Red
    mask_rgba[mask_sub == 0, 3] = 0.25  # Alpha
    ax.imshow(mask_rgba, extent=extent, origin="upper", aspect="auto", interpolation="nearest")
