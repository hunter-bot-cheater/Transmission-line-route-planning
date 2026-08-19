#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ai_path_planning.py
===================================================================
基于 AI 的复杂山区输电线路路径规划与可视化 (独立脚本)
===================================================================

功能概述
--------
1) 自动检测现有 DEM (*.tif) 与输电线路矢量 (*.shp)，不硬编码任何可能
   不存在的路径；若传入外部目录也一并搜索 (D:\\地形数据, D:\\输电线数据)。
2) 基于现有 Attention U-Net (CostUNet) 权重生成"建设代价图"；
   若 torch 或权重缺失，则自动回退到由坡度/曲率/高程派生出的启发式代价图。
3) 在代价栅格上用 A* (8 邻域 Moore 图, octile 启发式) 或 Dijkstra 寻路，
   找到起点→终点的优化线路，并做 RDP 简化 + 等距重采样。
4) 输出:
     - D:\\大创\\outputs\\planned_route.shp   (优化矢量路径, geopandas;
        不可用时回退写 planned_route.geojson)
     - D:\\大创\\outputs\\path_comparison.png (DEM 山体阴影底图 + 原始线路红色虚线
        + 规划线路蓝色实线 + 起终点标记 + 清晰图例, matplotlib 绘制)
5) 控制台打印: 路径长度(km)、累计高差(m)、长度变化率(%)、高差改善率(%)。

依赖库与安装命令 (Python >= 3.9)
-------------------------------------------------------------------
    pip install numpy scipy matplotlib shapely rasterio geopandas torch

说明:
    - numpy/scipy/matplotlib/rasterio 为核心 (读 DEM / 计算地形 / 绘图)。
    - geopandas 用于写出 .shp (缺失时自动回退为 .geojson，核心逻辑不受影响)。
    - torch 用于加载现有 CostUNet 权重 (缺失时自动使用启发式代价图)。
    - 所有重计算 (A*/Dijkstra、RDP、Haversine、坐标转换) 均为纯 Python/NumPy 实现。

用法示例
-------------------------------------------------------------------
    python ai_path_planning.py
    python ai_path_planning.py --method auto          # 默认: 优先 U-Net, 回退启发式
    python ai_path_planning.py --method unet           # 强制使用 U-Net 权重
    python ai_path_planning.py --method heuristic      # 强制启发式代价图
    python ai_path_planning.py --use-dijkstra          # 用 Dijkstra 替代 A*
    python ai_path_planning.py --dem "D:/地形数据/台湾省_DEM_30m分辨率_SRTM数据.tif" \
                               --line "D:/输电线数据/示例数据-中国输电线路矢量.shp"
    python ai_path_planning.py --start 22.0,120.5 --end 25.0,121.5

作者: path_planning_team (大创项目)
"""

import os
import sys
import time
import math
import json
import pickle
import argparse
import heapq
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. 基础依赖导入 (核心库缺失时给出明确提示)
# ----------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    sys.exit("[FATAL] 缺少 numpy，请先运行: pip install numpy")

# 模块级导入滑动窗口视图 (供 _box_mean 复用, 避免每次调用重复 import)
from numpy.lib.stride_tricks import sliding_window_view

# scipy 用于形态学/距离变换/滤波 (代价图的 TPI/粗糙度/距离变换)
try:
    import scipy.ndimage as ndi
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import matplotlib
matplotlib.use("Agg")  # 无界面后端, 适合脚本/服务器
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
# 优先使用常见中文字体, 避免图例/标题出现方块
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# rasterio 用于读取 GeoTIFF DEM
try:
    import rasterio
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# shapely 用于几何判断
try:
    from shapely.geometry import LineString, box, Point, Polygon
    from shapely.ops import transform as shapely_transform
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

# geopandas 用于写出 .shp (可选)
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

# torch (可选) —— 用于加载现有 Attention U-Net (CostUNet) 权重
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


# ============================================================================
# 1. 现有 Attention U-Net (CostUNet) 架构 —— 与 v3/src/dl_models.py 完全一致
#    (内嵌以保证脚本自包含, 可直接 load_state_dict 现有权重)
#    仅在 torch 可用时定义, 否则置为 None (不影响启发式回退路径)。
# ============================================================================
if HAS_TORCH:
    class DoubleConv(nn.Module):
        """双卷积 + BatchNorm + ReLU"""
        def __init__(self, in_ch, out_ch, dropout=0.0):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            return self.dropout(self.conv(x))

    class ResidualBlock(nn.Module):
        def __init__(self, channels, dropout=0.0):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual)

    class CostUNet(nn.Module):
        """
        与 v3 完全一致的 U-Net 成本预测模型。
        输入: (B, N_FEATURES=26, H, W) 多波段特征
        输出: (B, 1, H, W) 逐像元建设成本 (Sigmoid -> [0,1])
        """
        def __init__(self, n_features=26, encoder_channels=None,
                     decoder_channels=None, bottleneck=None, dropout=0.1):
            super().__init__()
            enc_ch = encoder_channels or [32, 64, 128, 256, 512]
            dec_ch = decoder_channels or [256, 128, 64, 32]
            bn = bottleneck or 512

            self.encoder_blocks = nn.ModuleList()
            in_ch = n_features
            for out_ch in enc_ch:
                self.encoder_blocks.append(DoubleConv(in_ch, out_ch, dropout))
                in_ch = out_ch

            self.bottleneck = nn.Sequential(
                DoubleConv(enc_ch[-1], bn, dropout),
                ResidualBlock(bn, dropout), ResidualBlock(bn, dropout),
            )

            self.decoder_blocks = nn.ModuleList()
            self.up_convs = nn.ModuleList()
            for i, out_ch in enumerate(dec_ch):
                skip_ch = enc_ch[-(i + 1)] if i < len(enc_ch) else enc_ch[-1]
                self.up_convs.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
                self.decoder_blocks.append(DoubleConv(out_ch + skip_ch, out_ch, dropout))
                in_ch = out_ch

            self.final_conv = nn.Sequential(
                nn.Conv2d(dec_ch[-1], 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1), nn.Sigmoid(),
            )

        def forward(self, x):
            input_shape = x.shape[2:]
            skips = []
            for enc in self.encoder_blocks:
                x = enc(x)
                skips.append(x)
                x = F.max_pool2d(x, 2)
            x = self.bottleneck(x)
            for i, (up, dec) in enumerate(zip(self.up_convs, self.decoder_blocks)):
                x = up(x)
                skip = skips[-(i + 1)]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
                x = torch.cat([x, skip], dim=1)
                x = dec(x)
            x = self.final_conv(x)
            if x.shape[2:] != input_shape:
                x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)
            return x

        def predict_cost_surface(self, feature_stack, batch_size=4):
            """对 (H, W, C) 特征堆叠分块预测成本面, 返回 (H, W) 的 [0,1] 数组。"""
            self.eval()
            H, W, C = feature_stack.shape
            x = torch.from_numpy(np.ascontiguousarray(feature_stack)).permute(2, 0, 1).unsqueeze(0).float()
            if H <= 512 and W <= 512:
                with torch.no_grad():
                    pred = self.forward(x)
                return pred.squeeze().cpu().numpy().astype(np.float32)
            patch, overlap = 256, 32
            result = np.zeros((H, W), dtype=np.float32)
            weight = np.zeros((H, W), dtype=np.float32)
            for r in range(0, H, patch - overlap):
                for c in range(0, W, patch - overlap):
                    r_end = min(r + patch, H); c_end = min(c + patch, W)
                    r0 = max(0, r_end - patch); c0 = max(0, c_end - patch)
                    patch_t = x[:, :, r0:r_end, c0:c_end]
                    with torch.no_grad():
                        pred = self.forward(patch_t)
                    pred_np = pred.squeeze().cpu().numpy().astype(np.float32)
                    result[r0:r_end, c0:c_end] += pred_np
                    weight[r0:r_end, c0:c_end] += 1.0
            result = result / np.clip(weight, 1e-6, None)
            return result.astype(np.float32)

    def load_costunet(path):
        """加载现有 CostUNet 权重 (cost_unet_improved.pt)。"""
        # 项目本地可信 checkpoint, 仅含 state_dict 张量: 优先用最小权限 weights_only=True
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # 极少数旧格式含非张量元数据时退回; 文件来源可信(项目本地)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        model = CostUNet(n_features=26)
        model.load_state_dict(sd)
        model.eval()
        return model
else:
    CostUNet = None
    load_costunet = None


# ============================================================================
# 2. 路径配置 (全部运行时探测, 不硬编码不存在的路径)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"
DOWNLOADED_DIR = DATA_DIR / "downloaded"   # OSM 原始缓存 (.pkl), 由 shared/data_acquisition 读取
SHARED_DIR = BASE_DIR / "shared"
# 已知外部数据目录 (项目历史约定)
KNOWN_DEM_DIRS = [Path(r"D:\地形数据"), BASE_DIR / "data"]
KNOWN_LINE_DIRS = [Path(r"D:\输电线数据"), BASE_DIR]
MODEL_WEIGHTS_CANDIDATES = [
    BASE_DIR / "data" / "models" / "v3_dl" / "cost_unet_improved.pt",
    BASE_DIR / "data" / "models" / "v3_dl" / "cost_unet_best.pt",
]
# 任务要求的输出文件名 (输出目录默认 OUTPUTS_DIR, 可用 --out-dir 覆盖)
ROUTE_SHP_NAME = "planned_route.shp"
ROUTE_GEOJSON_NAME = "planned_route.geojson"
COMPARISON_PNG_NAME = "path_comparison.png"


# ============================================================================
# 3. 文件自动探测
# ============================================================================
def find_dem_files():
    """递归搜索所有候选目录中的 *.tif (排除明显的输出代价面 *_cost* / overviews)。"""
    candidates = []
    search_roots = KNOWN_DEM_DIRS + [BASE_DIR]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.tif"):
            name = p.name.lower()
            if "cost" in name or name.endswith(".ovr") or "constraint" in name:
                continue
            # 排除明显非 DEM 的 tif (如 hillshade/产物)
            if p in seen:
                continue
            seen.add(p)
            candidates.append(p)
    return candidates


def find_line_files():
    """递归搜索所有候选目录中的 *.shp (输电线路矢量)。"""
    candidates = []
    search_roots = KNOWN_LINE_DIRS + [BASE_DIR / "v1_20260525", BASE_DIR / "v2_20260525",
                                       BASE_DIR / "v3_20260525"]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.shp"):
            # 跳过临时/索引类文件
            if p.name.lower().startswith("."):
                continue
            if p in seen:
                continue
            seen.add(p)
            candidates.append(p)
    return candidates


def find_model_weights():
    for c in MODEL_WEIGHTS_CANDIDATES:
        if c.exists():
            return c
    # 兜底: 任意 v3_dl 下的 .pt
    d = BASE_DIR / "data" / "models" / "v3_dl"
    if d.exists():
        pts = sorted(d.glob("*.pt"))
        if pts:
            return pts[0]
    return None


# ============================================================================
# 4. DEM 读取与下采样
# ============================================================================
def load_dem(path, max_side=2200):
    """
    读取 GeoTIFF DEM, 必要时下采样到最长边 <= max_side (控制 A* 网格规模)。
    返回: dem(np.float32, 已填 NaN), transform(affine.Affine), crs(str),
          pixel_m(单像元边长, 米, 近似), dem_nodata
    """
    if not HAS_RASTERIO:
        raise RuntimeError("需要 rasterio 读取 DEM, 请运行: pip install rasterio")
    path = Path(path)
    with rasterio.open(path) as src:
        crs = src.crs
        nodata = src.nodata
        h, w = src.height, src.width
        # 计算下采样目标尺寸
        scale = max(h, w) / max_side
        if scale <= 1.0:
            th, tw = h, w
        else:
            th, tw = int(round(h / scale)), int(round(w / scale))
        # 用 GDAL 重采样读取
        dem = src.read(1, out_shape=(th, tw)).astype(np.float32)
        # 新变换 (保持左上角与朝向)
        a = src.transform.a * (w / tw)
        e = src.transform.e * (h / th)
        new_transform = Affine(a, 0.0, src.transform.c, 0.0, e, src.transform.f)
    # 处理 nodata / 异常值
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem = np.where(np.isfinite(dem), dem, np.nan)
    # 粗略剔除明显异常高程 (如 <= -100 或 > 9000)
    dem = np.where((dem < -100) | (dem > 9000), np.nan, dem)
    # 单像元边长(米) 近似 (地理坐标下随纬度变化, 取中心纬度)
    lat0 = (new_transform.f + e * th) / 2.0  # 约中心纬度(度)
    pixel_m = _deg_pixel_m(a, lat0)
    crs_str = str(crs) if crs is not None else "EPSG:4326"
    print(f"  DEM: {path.name}  网格 {tw}x{th}  像元≈{pixel_m:.0f}m  CRS={crs_str}")
    return dem, new_transform, crs_str, pixel_m, nodata


# ============================================================================
# 5. 地形因子与特征栈 (26 维, 与 v3 config.FEATURE_BANDS 对应)
# ============================================================================
def _fill_nan(arr):
    """用有限值的中值填充 NaN, 便于梯度/滤波计算。"""
    if not np.isfinite(arr).any():
        return np.zeros_like(arr)
    m = np.nanmedian(arr)
    return np.where(np.isfinite(arr), arr, m)


def crop_dem_to_bbox(dem, transform, lon0, lat0, lon1, lat1, margin_deg=0.2):
    """裁剪 DEM 到起终点包围盒 + 边距, 返回 (sub_dem, new_transform)。
    用于把分析聚焦到走廊/山区, 降低海洋(void)占比、提升有效分辨率。
    注: margin_deg=0.2 的余量假设基于经纬度 DEM (如 SRTM)。当前 SRTM 为地理
    坐标系(度), 度->米的近似在走廊尺度合理; 若改用投影(米)DEM 需相应改余量单位。"""
    H, W = dem.shape
    left = transform.c
    top = transform.f
    right = left + transform.a * W
    bottom = top + transform.e * H
    minlon = max(min(lon0, lon1) - margin_deg, left)
    maxlon = min(max(lon0, lon1) + margin_deg, right)
    minlat = max(min(lat0, lat1) - margin_deg, bottom)
    maxlat = min(max(lat0, lat1) + margin_deg, top)
    c0 = int(math.floor((minlon - left) / transform.a))
    c1 = int(math.ceil((maxlon - left) / transform.a))
    r_top = int(math.floor((maxlat - top) / transform.e))   # e<0: 高纬 -> 小行号
    r_bot = int(math.ceil((minlat - top) / transform.e))
    c0 = min(max(c0, 0), W - 1); c1 = min(max(c1, 0), W - 1)
    r_top = min(max(r_top, 0), H - 1); r_bot = min(max(r_bot, 0), H - 1)
    if r_top > r_bot:
        r_top, r_bot = r_bot, r_top
    if c0 > c1:
        c0, c1 = c1, c0
    # 退化保护: 起终点极近时裁剪框可能塌缩为 1 像素, 保证至少 2x2 以便后续寻路
    if c1 - c0 < 1:
        c1 = min(c0 + 1, W - 1)
    if r_bot - r_top < 1:
        r_bot = min(r_top + 1, H - 1)
    sub = dem[r_top:r_bot + 1, c0:c1 + 1].copy()
    new_transform = Affine(transform.a, 0.0, left + transform.a * c0,
                           0.0, transform.e, top + transform.e * r_top)
    return sub, new_transform


def compute_terrain_factors(dem, pixel_m):
    """由 DEM 计算 13 维地形因子 (与 v3 preprocessing 对应)。"""
    z = _fill_nan(dem)
    dy, dx = np.gradient(z, pixel_m, pixel_m)  # 单位: m/m

    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))        # 弧度
    slope_deg = np.degrees(slope)
    aspect = np.arctan2(dy, -dx)
    aspect_cos = np.cos(aspect)
    aspect_sin = np.sin(aspect)

    # 地形粗糙度指数 TRI (3x3 标准差)
    if HAS_SCIPY:
        tri = ndi.generic_filter(z, np.nanstd, size=3)
    else:
        tri = np.abs(z - _box_mean(z, 1))
    tri = np.nan_to_num(tri, nan=0.0)

    # 多尺度 TPI = 中心 - 邻域均值
    def tpi_at(radius_px):
        if HAS_SCIPY:
            size = 2 * radius_px + 1
            mean = ndi.uniform_filter(z, size=size, mode="reflect")
            # 中心权重修正 (uniform_filter 含中心), 近似即可
            return z - mean
        return z - _box_mean(z, radius_px)

    # 曲率 (Laplacian)
    if HAS_SCIPY:
        curv = ndi.laplace(z) / (pixel_m ** 2)
    else:
        curv = (np.roll(z, 1, 0) + np.roll(z, -1, 0) +
                np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z) / (pixel_m ** 2)
    curv = np.nan_to_num(curv, nan=0.0)

    # 粗糙度 (多尺度标准差)
    def rough_at(radius_px):
        if HAS_SCIPY:
            return ndi.generic_filter(z, np.nanstd, size=2 * radius_px + 1)
        return np.abs(z - _box_mean(z, radius_px))

    # 将米尺度换算为像元半径 (近似, 仅用于形态学尺度)
    def px_of_m(m):
        return max(1, int(round(m / pixel_m)))

    tpi_100 = tpi_at(px_of_m(100))
    tpi_300 = tpi_at(px_of_m(300))
    tpi_900 = tpi_at(px_of_m(900))
    rough_3 = rough_at(px_of_m(100))
    rough_9 = rough_at(px_of_m(300))
    rough_27 = rough_at(px_of_m(900))

    factors = {
        "elevation": z,
        "slope": slope_deg,
        "aspect_cos": aspect_cos,
        "aspect_sin": aspect_sin,
        "tri": tri,
        "tpi_100": tpi_100, "tpi_300": tpi_300, "tpi_900": tpi_900,
        "profile_curvature": curv, "plan_curvature": curv,
        "rough_3": rough_3, "rough_9": rough_9, "rough_27": rough_27,
    }
    return factors


def _box_mean(arr, r):
    """无 scipy 时的简单盒均值近似 (r=1 => 3x3), 输出尺寸与 arr 完全一致。"""
    if r <= 0:
        return arr
    p = sliding_window_view(arr, (2 * r + 1, 2 * r + 1))
    m = np.nanmean(p, axis=(-2, -1))  # 形状 (H-2r, W-2r)
    # 将结果回填到中心 (r, r) 区域, 四周边界用全局均值补齐, 保证尺寸一致
    out = np.full(arr.shape, np.nan, dtype=arr.dtype)
    out[r:-r, r:-r] = m
    fill = np.nanmean(arr) if np.isfinite(np.nanmean(arr)) else 0.0
    out = np.where(np.isfinite(out), out, fill)
    return out


def _minmax01(band):
    """将单波段归一化到 [0,1]。"""
    b = np.asarray(band, dtype=np.float32)
    bmin, bmax = np.nanmin(b), np.nanmax(b)
    if not np.isfinite(bmin) or not np.isfinite(bmax) or (bmax - bmin) < 1e-9:
        return np.zeros_like(b)
    return np.clip((b - bmin) / (bmax - bmin), 0.0, 1.0).astype(np.float32)


def build_feature_stack(dem, terrain, line_mask=None, osm=None):
    """
    组装 26 维特征栈 (H, W, 26), 顺序严格对应 v3 config.FEATURE_BANDS。
    osm: compute_osm_feature_bands 返回的波段 dict (补齐 13-25); 为 None 时
         缺失波段以 0 填充 (模型仍可前向推理, 输出有效代价图)。
    line_mask: 可选, 二值数组标记现有输电线路位置 (用于 dist_existing_line, 波段 15)。
    """
    H, W = dem.shape
    stack = np.zeros((H, W, 26), dtype=np.float32)

    # 地形 13 维
    stack[:, :, 0] = _minmax01(terrain["elevation"])
    stack[:, :, 1] = _minmax01(terrain["slope"])
    stack[:, :, 2] = _minmax01(terrain["aspect_cos"])
    stack[:, :, 3] = _minmax01(terrain["aspect_sin"])
    stack[:, :, 4] = _minmax01(terrain["tri"])
    stack[:, :, 5] = _minmax01(terrain["tpi_100"])
    stack[:, :, 6] = _minmax01(terrain["tpi_300"])
    stack[:, :, 7] = _minmax01(terrain["tpi_900"])
    stack[:, :, 8] = _minmax01(terrain["profile_curvature"])
    stack[:, :, 9] = _minmax01(terrain["plan_curvature"])
    stack[:, :, 10] = _minmax01(terrain["rough_3"])
    stack[:, :, 11] = _minmax01(terrain["rough_9"])
    stack[:, :, 12] = _minmax01(terrain["rough_27"])

    # dist_existing_line (band 15): 距现有线路越近成本越低
    if line_mask is not None and HAS_SCIPY:
        dist = ndi.distance_transform_edt(~line_mask.astype(bool))
        # 归一化 (以约 2km 为衰减尺度)
        stack[:, :, 15] = np.clip(dist / 20.0, 0, 1).astype(np.float32)
        # 注: dist 单位为像元, /20 表示约 20 像元(视分辨率)内的现有线路附近成本更低

    # OSM / 风险波段 13,14,16,17,18,19,20,21,22,23,24,25 (接入 shared/data_acquisition 补齐)
    if osm is not None:
        stack[:, :, 13] = np.clip(osm["dist_road"] / 5000.0, 0, 1)
        stack[:, :, 14] = np.clip(osm["dist_water"] / 5000.0, 0, 1)
        stack[:, :, 16] = np.clip(osm["dist_railway"] / 5000.0, 0, 1)
        stack[:, :, 17] = np.clip(osm["dist_fault"] / 10000.0, 0, 1)
        stack[:, :, 18] = osm["landuse_code"].astype(np.float32) / 8.0
        p99 = np.nanpercentile(osm["building_density"], 99) if np.any(osm["building_density"] > 0) else 1.0
        stack[:, :, 19] = np.clip(osm["building_density"] / max(p99, 1.0), 0, 1)
        stack[:, :, 20] = np.clip(osm["vegetation_height"] / 30.0, 0, 1)
        for i, name in enumerate(["typhoon_risk", "seismic_risk", "landslide_risk", "ice_cover_risk", "lightning_risk"]):
            stack[:, :, 21 + i] = np.clip(osm[name], 0, 1).astype(np.float32)
    # osm 为 None 时, 上述波段保持 0 (缺失波段=0 近似, 模型前向仍输出有效代价图)。
    return stack


def heuristic_cost_map(dem, terrain, osm=None):
    """启发式代价图 [0,1]: 坡度为主, 叠加曲率与高程异常; 可选融合 OSM/风险波段。"""
    slope_n = _minmax01(terrain["slope"])
    rough_n = np.clip(_minmax01(terrain["rough_27"]), 0, 1)
    curv_n = np.clip(_minmax01(np.abs(terrain["profile_curvature"])), 0, 1)
    # 高程极差大的地方(山脊/深谷)施工难 -> 用局部标准差
    cost = (0.65 * slope_n ** 1.5 + 0.20 * rough_n + 0.15 * curv_n)
    if osm is not None:
        # 近水域 -> 成本高 (避免跨河/临水施工)
        water_pen = 1.0 - np.clip(osm["dist_water"] / 5000.0, 0, 1)
        # 建筑密集 -> 成本高
        bdens = osm["building_density"]
        p99 = np.nanpercentile(bdens, 99) if np.any(bdens > 0) else 1.0
        bdens_pen = np.clip(bdens / max(p99, 1.0), 0, 1)
        # 高滑坡/地震风险 -> 成本高
        risk_pen = 0.5 * osm["landslide_risk"] + 0.5 * osm["seismic_risk"]
        cost = cost + 0.25 * water_pen + 0.15 * bdens_pen + 0.20 * risk_pen
    cost = np.clip(cost, 0, 1).astype(np.float32)
    return cost


# ============================================================================
# 5b. OSM / 风险特征波段 (接入 shared/data_acquisition.py, 补齐 26 维)
# ============================================================================
def _fill_nan(a):
    """NaN -> 0, 保持 float32。"""
    return np.nan_to_num(np.asarray(a, np.float32), nan=0.0).astype(np.float32)


def _import_shared_data_acquisition():
    """导入 shared/data_acquisition.py 并注入最小 config 垫片, 使其离线可用。
    返回模块对象; 缺失依赖或失败时返回 None。OSM 原始缓存已存在于
    DOWNLOADED_DIR, 故 OSM_CACHE_DAYS=9999 强制走本地缓存, 避免运行时联网。"""
    if not (HAS_GPD and HAS_SHAPELY and HAS_RASTERIO and HAS_SCIPY):
        print("    [warn] 缺 geopandas/shapely/rasterio, 跳过 OSM 波段")
        return None
    try:
        import types, importlib
        # 确保 shared 包可被找到: 直接运行脚本时 sys.path[0] 是 scripts/, 不含项目根
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        cfg = types.SimpleNamespace(
            DEM_PATH=BASE_DIR / "data" / "models" / "v3_dl",
            SHP_PATH=BASE_DIR / "shp_placeholder",
            TAIWAN_BBOX=(120.0, 21.9, 122.0, 25.4),
            WGS84="EPSG:4326",
            DOWNLOADED_DIR=DOWNLOADED_DIR,
            OSM_CACHE_DAYS=9999,
            METERS_PER_DEG=111320.0,
            OFFLINE=True,   # 离线模式: shared 模块据此跳过联网请求
        )
        sys.modules.setdefault("config", cfg)   # 供 shared 的 `import config as cfg` 解析
        da = importlib.import_module("shared.data_acquisition")
        return da
    except Exception as ex:
        print(f"    [warn] 导入 shared/data_acquisition 失败: {ex}")
        return None


try:
    from rasterio.features import rasterize as _rio_rasterize
except Exception:
    _rio_rasterize = None


def _pixel_meters(transform, lat_center):
    """窗口中心的单像元边长(米)。"""
    return _deg_pixel_m(transform.a, lat_center)


def _gdf_to_geoms_tags(gdf):
    """GeoDataFrame -> [(geom, tags), ...]; 无 tags 列时给空 dict。"""
    if gdf is None or len(gdf) == 0:
        return []
    out = []
    has_tags = "tags" in gdf.columns
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        tags = row["tags"] if has_tags else {}
        out.append((g, tags if isinstance(tags, dict) else {}))
    return out


def _landuse_tag_to_code(tags):
    """土地利用标签 -> 软约束代码, 与 v3 LANDUSE_SOFT_COST 对应。
    代码 1..8 为真实类别; 0 为 no-data 哨兵(未知/无标签), 归一化时按中性 (0.0) 处理。"""
    if not tags:
        return 0
    lu = (tags.get("landuse") or "").lower()
    nat = (tags.get("natural") or "").lower()
    mapping = {
        "forest": 1, "wood": 1, "scrub": 1, "heath": 1,
        "farmland": 2, "farmyard": 2, "orchard": 2, "vineyard": 2,
        "meadow": 2, "grass": 2, "grassland": 2, "greenfield": 2,
        "bare_rock": 3, "bare_ground": 3, "scree": 3, "sand": 3, "beach": 3,
        "residential": 4, "retail": 4, "commercial": 4, "urban": 4,
        "industrial": 5, "quarry": 5, "brownfield": 5, "construction": 5,
        "reservoir": 6, "basin": 6, "water": 6, "salt_pond": 6, "aquaculture": 6,
        "wetland": 7, "marsh": 7, "swamp": 7, "mud": 7,
        "cemetery": 8, "recreation_ground": 8, "village_green": 8,
        "military": 8, "allotments": 8, "plant_nursery": 8,
    }
    for key, code in mapping.items():
        if key in lu or key in nat:
            return code
    return 0


def _rasterize_on(geoms, transform, shape):
    if not geoms or _rio_rasterize is None:
        return np.zeros(shape, np.uint8)
    shapes = [(g, 1) for g, _ in geoms]
    try:
        return _rio_rasterize(shapes, out_shape=shape, transform=transform,
                              fill=0, all_touched=True, dtype=np.uint8)
    except Exception:
        return np.zeros(shape, np.uint8)


def _compute_distance_band(geoms, transform, shape, max_dist_m=5000.0):
    on = _rasterize_on(geoms, transform, shape)
    if on.sum() == 0:
        return np.full(shape, max_dist_m, np.float32)
    cell_m = _pixel_meters(transform, (transform.f + transform.e * shape[0]) / 2.0)
    dist_px = ndi.distance_transform_edt(1 - on)
    return np.clip(dist_px * cell_m, 0, max_dist_m).astype(np.float32)


def _compute_building_density(geoms, transform, shape):
    on = _rasterize_on(geoms, transform, shape)
    if on.sum() == 0:
        return np.zeros(shape, np.float32)
    cell_m = _pixel_meters(transform, (transform.f + transform.e * shape[0]) / 2.0)
    sigma = max(2, int(500 / cell_m))
    density = gaussian_filter(on.astype(np.float32), sigma=sigma)
    cell_area_km2 = (cell_m / 1000.0) ** 2
    return (density / cell_area_km2).astype(np.float32)


def _compute_landuse_code(geoms, transform, shape):
    result = np.full(shape, 0, np.uint8)   # 0 = no-data 哨兵 (中性)
    if not geoms:
        return result
    from collections import defaultdict
    groups = defaultdict(list)
    for g, tags in geoms:
        try:
            groups[_landuse_tag_to_code(tags)].append(g)
        except Exception:
            continue
    for code in sorted(groups.keys()):
        shapes = [(g, code) for g in groups[code]]
        try:
            m = _rio_rasterize(shapes, out_shape=shape, transform=transform,
                               fill=0, all_touched=True, dtype=np.uint8)
            result = np.where(m == code, code, result).astype(np.uint8)
        except Exception:
            continue
    return result


def _compute_vegetation_height(geoms, transform, shape):
    result = np.zeros(shape, np.float32)
    if not geoms:
        return result
    from collections import defaultdict
    groups = defaultdict(list)
    for g, tags in geoms:
        lu = (tags.get("landuse") or "").lower()
        nat = (tags.get("natural") or "").lower()
        if "wood" in nat or "forest" in nat or "tree" in nat:
            h = 20.0
        elif "scrub" in nat or "grass" in nat or "grassland" in nat or "heath" in nat:
            h = 5.0
        elif "wetland" in nat or "marsh" in nat:
            h = 2.0
        elif "forest" in lu or "wood" in lu:
            h = 20.0
        else:
            h = 3.0
        groups[h].append(g)
    for h, glist in groups.items():
        m = _rasterize_on([(g, {}) for g in glist], transform, shape)
        result = np.where(m > 0, h, result).astype(np.float32)
    return result


_OSM_PKL_CACHE = {}


def _load_raw_osm_geoms(pkl_name, geom_type, bbox):
    """读取 DOWNLOADED_DIR 下 raw Overpass JSON .pkl, 返回落在 bbox 内的 [(geom, tags), ...]。
    仅在 bbox 内解析, 避免构建全岛巨型 GeoDataFrame (土地覆盖/建筑缓存各约 190MB)。
    用于 shared.fetch_osm_* 返回全岛数据、在走廊内开销过大的两个重型图层。"""
    pkl_path = DOWNLOADED_DIR / f"{pkl_name}.pkl"
    if not HAS_SHAPELY or not pkl_path.exists():
        return []
    if pkl_path in _OSM_PKL_CACHE:
        data = _OSM_PKL_CACHE[pkl_path]
    else:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            _OSM_PKL_CACHE[pkl_path] = data
        except Exception:
            return []
    elems = data.get("elements", []) if isinstance(data, dict) else []
    if not elems:
        return []
    box_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    out = []
    for el in elems:
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {}) or {}
        g = el.get("geometry")
        if not g:
            continue
        coords = [(float(pt["lon"]), float(pt["lat"])) for pt in g]
        try:
            if geom_type == "polygon":
                if len(coords) < 3:
                    continue
                if coords[0] != coords[-1]:
                    coords = coords + [coords[0]]
                geom = Polygon(coords)
            else:
                if len(coords) < 2:
                    continue
                geom = LineString(coords)
        except Exception:
            continue
        if geom.is_empty or not box_poly.intersects(geom):
            continue
        out.append((geom, tags))
    return out


def compute_osm_feature_bands(dem, transform, bbox, da, terrain=None):
    """计算 OSM 派生波段(13-20) 与风险波段(21-25), 返回 dict: name->(H,W) 数组。
    优先经 shared.data_acquisition 的 fetch_* / build_*_risk; 覆冰(24)/雷击(25)
    为 DEM 派生代理(shared 无对应函数)。任一层缺失则回退为零波段(优雅降级)。"""
    H, W = dem.shape
    bands = {}
    def safe(fn, default):
        try:
            v = fn()
            return default if v is None else v
        except Exception as ex:
            print(f"    [warn] OSM 波段计算失败: {ex}")
            return default

    # 13,14,16,17 距离波段 (道路/水域/铁路/断裂带)
    roads = safe(lambda: _gdf_to_geoms_tags(da.fetch_osm_roads(bbox)), [])
    water = safe(lambda: _gdf_to_geoms_tags(da.fetch_osm_water(bbox)), [])
    railways = safe(lambda: _gdf_to_geoms_tags(da.fetch_osm_railways(bbox)), [])
    faults = safe(lambda: _gdf_to_geoms_tags(da.fetch_osm_faults(bbox)), [])
    bands["dist_road"] = _compute_distance_band(roads, transform, (H, W), 5000.0)
    bands["dist_water"] = _compute_distance_band(water, transform, (H, W), 5000.0)
    bands["dist_railway"] = _compute_distance_band(railways, transform, (H, W), 5000.0)
    bands["dist_fault"] = _compute_distance_band(faults, transform, (H, W), 10000.0)
    # 18,19,20 分类/密度/植被
    # 土地覆盖/建筑缓存达 ~190MB, shared.fetch_* 返回全岛 GeoDataFrame 开销过大,
    # 改用 raw 缓存 + bbox 即时过滤; 其余轻型图层仍走 shared.fetch_*。
    landuse = safe(lambda: _load_raw_osm_geoms("taiwan_landuse", "polygon", bbox), [])
    buildings = safe(lambda: _load_raw_osm_geoms("taiwan_buildings", "polygon", bbox), [])
    vegetation = safe(lambda: _gdf_to_geoms_tags(da.fetch_osm_vegetation(bbox)), [])
    bands["landuse_code"] = _compute_landuse_code(landuse, transform, (H, W))
    bands["building_density"] = _compute_building_density(buildings, transform, (H, W))
    bands["vegetation_height"] = _compute_vegetation_height(vegetation, transform, (H, W))
    # 21-23 风险代理 (经 shared.build_*_risk)
    bands["typhoon_risk"] = safe(lambda: _fill_nan(da.build_typhoon_risk(dem, transform, "")),
                                  np.zeros((H, W), np.float32))
    bands["seismic_risk"] = safe(lambda: _fill_nan(da.build_seismic_risk(dem, transform, "")),
                                  np.zeros((H, W), np.float32))
    bands["landslide_risk"] = safe(lambda: _fill_nan(da.build_landslide_risk(dem, transform, "")),
                                    np.zeros((H, W), np.float32))
    # 24,25 DEM 派生 (覆冰/雷击) —— shared 无对应函数, 由地形推导
    z = _fill_nan(dem).astype(np.float32)
    elev_n = np.clip(z / 3000.0, 0, 1)
    if terrain is not None and "slope" in terrain:
        slope_n = _minmax01(terrain["slope"])
    else:
        pm = _pixel_meters(transform, (transform.f + transform.e * H) / 2.0)
        dy, dx = np.gradient(z, pm, pm)
        slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
        slope_n = np.clip(slope / 45.0, 0, 1)
    bands["ice_cover_risk"] = np.clip((z - 2000) / 2000.0, 0, 1).astype(np.float32)
    bands["lightning_risk"] = np.clip(0.6 * slope_n + 0.4 * elev_n, 0, 1).astype(np.float32)
    return bands


# ============================================================================
# 6. 坐标转换 (地理 <-> 栅格)
# ============================================================================
def geo_to_grid(lat, lon, transform):
    col = int((lon - transform.c) / transform.a)
    row = int((lat - transform.f) / transform.e)
    return row, col


def grid_to_geo(row, col, transform):
    lon = transform.c + (col + 0.5) * transform.a
    lat = transform.f + (row + 0.5) * transform.e
    return lat, lon


def _deg_pixel_m(a, lat_deg):
    """地理坐标下单个像元的近似边长(米), 随纬度变化。"""
    return abs(a) * 111320.0 * max(math.cos(math.radians(lat_deg)), 1e-3)


def _dem_bounds(transform, H, W):
    """由仿射变换与网格尺寸返回 (left, bottom, right, top) 地理范围。"""
    left = transform.c
    top = transform.f
    right = left + transform.a * W
    bottom = top + transform.e * H
    return left, bottom, right, top


# ============================================================================
# 7. A* / Dijkstra 寻路 (纯 Python, 8 邻域 Moore 图)
# ============================================================================
def _find_nearest_valid(cost_raster, r, c, radius=80):
    H, W = cost_raster.shape
    for rad in range(1, radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and np.isfinite(cost_raster[nr, nc]):
                    return nr, nc
    return None, None


def astar_search(cost_raster, start_rc, end_rc, use_dijkstra=False):
    """
    A* (use_dijkstra=False) 或 Dijkstra (True) 在代价栅格上寻路。
    cost_raster: (H, W), inf = 不可通行; 像元步长成本 = cost * 距离系数 * pixel_m。
    返回: list[(row, col)] 路径, 或 None。
    """
    H, W = cost_raster.shape
    inf_mask = np.isinf(cost_raster)          # 预计算不可通行掩膜, 加速邻居判定
    sr, sc = start_rc
    er, ec = end_rc
    if not (0 <= sr < H and 0 <= sc < W) or not (0 <= er < H and 0 <= ec < W):
        return None
    if inf_mask[sr, sc]:
        sr, sc = _find_nearest_valid(cost_raster, sr, sc)
    if inf_mask[er, ec]:
        er, ec = _find_nearest_valid(cost_raster, er, ec)
    if sr is None or er is None:
        return None
    start_rc = (sr, sc)   # 起点可能已被吸附到最近可行像元, 统一键名为更新后的 (sr, sc)

    valid = cost_raster[np.isfinite(cost_raster)]
    c_min = max(float(np.percentile(valid, 1)), 1e-6) if valid.size else 1e-6

    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1)]
    ndist = [1.0, math.sqrt(2), 1.0, math.sqrt(2),
             1.0, math.sqrt(2), 1.0, math.sqrt(2)]

    def heuristic(r, c):
        if use_dijkstra:
            return 0.0
        dr, dc = abs(r - er), abs(c - ec)
        return c_min * (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))

    g = {start_rc: 0.0}
    open_set = [(heuristic(sr, sc), 0, start_rc)]
    came_from = {}
    closed = set()
    tb = 0
    while open_set:
        f, _, cur = heapq.heappop(open_set)
        if cur in closed:
            continue
        if cur == (er, ec):
            # 回溯
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(cur)
        cr, cc = cur
        for ni, (dr, dc) in enumerate(neighbors):
            nr, nc = cr + dr, cc + dc
            nbr = (nr, nc)
            if nbr in closed or not (0 <= nr < H and 0 <= nc < W):
                continue
            if inf_mask[nr, nc]:
                continue
            if dr != 0 and dc != 0:  # 禁止穿越对角线夹角的两个障碍
                if inf_mask[cr + dr, cc] or inf_mask[cr, cc + dc]:
                    continue
            step = cost_raster[nr, nc] * ndist[ni]
            tg = g[cur] + step
            if nbr not in g or tg < g[nbr]:
                g[nbr] = tg
                h = heuristic(nr, nc)
                # 有界 tie-breaker: 偏向目标方向, 且不破坏启发式可采纳性
                # (原实现用无限增长的全局计数 tb, 大网格下会严重拖慢收敛并影响最优性)
                heapq.heappush(open_set, (tg + h + 1e-4 * (abs(nr - er) + abs(nc - ec)), tb, nbr))
                came_from[nbr] = cur
                tb += 1
    return None


# ============================================================================
# 8. 路径平滑 (RDP 简化 + 等距重采样, 纯 Python)
# ============================================================================
def _rdp(points, epsilon):
    """RDP 路径简化 (显式栈迭代版)。

    原实现为纯递归, 在 DEM max_side 较大、栅格路径达数千点时, 最坏递归深度
    ≈ 路径点数, 会超过 Python 默认递归上限(1000)触发 RecursionError 崩溃。
    此处改为显式栈迭代, 彻底规避长蛇形路径的爆栈风险, 行为与递归版等价。
    """
    if len(points) < 3:
        return points
    n = len(points)
    keep = [False] * n
    keep[0] = True
    keep[n - 1] = True
    # 栈中存放待处理的子段 [s, e] 索引 (闭区间)
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        start = np.array(points[s]); end = np.array(points[e])
        line_vec = end - start
        ll = np.linalg.norm(line_vec)
        if ll < 1e-12:
            # 子段退化为点: 端点已标记, 无需保留内部点
            continue
        unit = line_vec / ll
        dmax, idx = 0.0, 0
        for i in range(s + 1, e):
            v = np.array(points[i]) - start
            proj = np.clip(np.dot(v, unit), 0, ll)
            closest = start + proj * unit
            d = np.linalg.norm(np.array(points[i]) - closest)
            if d > dmax:
                dmax, idx = d, i
        if dmax > epsilon:
            # 保留最远点并向两侧继续细分
            keep[idx] = True
            stack.append((s, idx))
            stack.append((idx, e))
        # 否则整段可简化为端点 (端点已标记 keep)
    return [points[i] for i in range(n) if keep[i]]


def smooth_path(grid_path, transform, spacing_m=90.0):
    """栅格路径 -> 地理坐标 -> RDP 简化 -> 等距重采样。返回 [(lon, lat), ...]。"""
    geo = [grid_to_geo(r, c, transform) for (r, c) in grid_path]  # (lat, lon)
    geo = [(lon, lat) for (lat, lon) in geo]                      # -> (lon, lat)
    if len(geo) < 3:
        return geo
    arr = np.array(geo, dtype=np.float64)
    eps_deg = spacing_m / 111000.0 * 2.0  # RDP 容差(度)
    simplified = _rdp(arr.tolist(), eps_deg)
    if len(simplified) < 2:
        return geo
    simplified = np.array(simplified)
    xs, ys = simplified[:, 0], simplified[:, 1]
    seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    spacing_deg = spacing_m / 111000.0
    n = max(int(total / spacing_deg), len(simplified))
    sd = np.linspace(0, total, n)
    xi = np.interp(sd, cum, xs)
    yi = np.interp(sd, cum, ys)
    return [(float(xi[i]), float(yi[i])) for i in range(n)]


# ============================================================================
# 9. 指标计算
# ============================================================================
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def path_length_km(coords_lonlat):
    total = 0.0
    for i in range(len(coords_lonlat) - 1):
        lon1, lat1 = coords_lonlat[i]
        lon2, lat2 = coords_lonlat[i + 1]
        total += haversine_km(lon1, lat1, lon2, lat2)
    return total


def sample_dem_along(coords_lonlat, dem, transform):
    """沿路线采样 DEM 高程 (像元中心最近邻)。返回高程列表。"""
    hs = []
    H, W = dem.shape
    for lon, lat in coords_lonlat:
        r, c = geo_to_grid(lat, lon, transform)
        r = min(max(r, 0), H - 1); c = min(max(c, 0), W - 1)
        v = dem[r, c]
        hs.append(float(v) if np.isfinite(v) else np.nan)
    return hs


def cumulative_relief(hs):
    """累计高差 = Σ|Δh| (总升降); 另返回净高差。"""
    hs = np.array([h for h in hs if np.isfinite(h)], dtype=np.float64)
    if hs.size < 2:
        return 0.0, 0.0
    cum = float(np.sum(np.abs(np.diff(hs))))
    net = float(hs[-1] - hs[0])
    return cum, net


def compute_metrics(planned_ll, orig_ll, dem, transform):
    p_len = path_length_km(planned_ll)
    o_len = path_length_km(orig_ll) if orig_ll else float("nan")
    p_hs = sample_dem_along(planned_ll, dem, transform)
    p_cum, p_net = cumulative_relief(p_hs)
    if orig_ll:
        o_hs = sample_dem_along(orig_ll, dem, transform)
        o_cum, o_net = cumulative_relief(o_hs)
    else:
        o_cum = float("nan"); o_net = float("nan")

    len_change = (p_len - o_len) / o_len * 100.0 if (orig_ll and o_len > 0) else float("nan")
    elev_improve = (o_cum - p_cum) / o_cum * 100.0 if (orig_ll and o_cum > 0) else float("nan")
    return {
        "planned_length_km": p_len, "orig_length_km": o_len,
        "planned_cum_relief_m": p_cum, "planned_net_relief_m": p_net,
        "orig_cum_relief_m": o_cum, "orig_net_relief_m": o_net,
        "length_change_pct": len_change, "elev_improve_pct": elev_improve,
    }


# ============================================================================
# 10. 原始线路选择 (自动选用与 DEM 重叠的现有线路)
# ============================================================================
def select_original_line(line_files, dem, transform, dem_crs, force_line=None):
    """
    从候选线路中挑选与 DEM 范围重叠、且两端点均落在 DEM 有限区域(陆地)的线路。
    优先选择真实线路 (*real_path*), 避免选中跨越全岛/端点落在海洋 void 的生成线路。
    返回: (orig_coords_lonlat, start_ll, end_ll, src_name) 或 (None, None, None, None)
    """
    if not (HAS_SHAPELY and HAS_GPD):
        return None, None, None, None
    H, W = dem.shape
    left, bottom, right, top = _dem_bounds(transform, H, W)
    dem_box = box(left, bottom, right, top)

    def endpoint_ok(lat, lon):
        if not (left - 1e-6 <= lon <= right + 1e-6 and bottom - 1e-6 <= lat <= top + 1e-6):
            return False
        r = int((lat - transform.f) / transform.e)
        c = int((lon - transform.c) / transform.a)
        r = min(max(r, 0), H - 1); c = min(max(c, 0), W - 1)
        return np.isfinite(dem[r, c])

    candidates = []
    for lf in line_files:
        try:
            gdf = gpd.read_file(lf)
        except Exception:
            continue
        try:
            if dem_crs and gdf.crs and str(gdf.crs) != str(dem_crs):
                gdf = gdf.to_crs(dem_crs)
            elif dem_crs and gdf.crs is None:
                gdf = gdf.set_crs(dem_crs, allow_override=True)
        except Exception:
            pass
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                if not geom.intersects(dem_box):
                    continue
            except Exception:
                continue
            sub = geom if geom.geom_type != "MultiLineString" else max(geom.geoms, key=lambda g: g.length)
            coords = list(sub.coords)
            if len(coords) < 2:
                continue
            s, e = coords[0], coords[-1]
            # 两端点必须在 DEM 有限(陆地)区域内, 否则易因落入海洋 void 而无解
            if not (endpoint_ok(s[1], s[0]) and endpoint_ok(e[1], e[0])):
                continue
            is_real = "real_path" in lf.name
            score = (100.0 if is_real else 0.0) + sub.length
            candidates.append((sub, lf.name, score))
    if not candidates:
        return None, None, None, None
    sub, name, _ = max(candidates, key=lambda t: t[2])
    coords = list(sub.coords)
    orig_ll = [(x, y) for (x, y) in coords]
    start_ll = (coords[0][1], coords[0][0])
    end_ll = (coords[-1][1], coords[-1][0])
    return orig_ll, start_ll, end_ll, name


# ============================================================================
# 11. 可视化 (对比图)
# ============================================================================
def plot_comparison(dem, transform, planned_ll, orig_ll, start_ll, end_ll, out_path):
    H, W = dem.shape
    z = _fill_nan(dem)
    ls = LightSource(azdeg=315, altdeg=45)
    hill = ls.hillshade(z, vert_exag=max(30.0, 4000.0 / max(np.nanmax(z) - np.nanmin(z), 1.0)))
    hill = np.clip(hill, 0, 1)

    left, bottom, right, top = _dem_bounds(transform, H, W)
    extent = [left, right, bottom, top]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    ax.imshow(hill, extent=extent, cmap="gray", origin="upper", aspect="auto")
    # 叠加一层淡色高程以体现山区
    ax.imshow(z, extent=extent, cmap="terrain", origin="upper", aspect="auto",
              alpha=0.35, vmin=np.nanpercentile(z, 2), vmax=np.nanpercentile(z, 98))

    if orig_ll:
        olon = [p[0] for p in orig_ll]; olat = [p[1] for p in orig_ll]
        ax.plot(olon, olat, color="red", linestyle="--", linewidth=2.0,
                label="原始线路 (Existing)", zorder=3)
    plon = [p[0] for p in planned_ll]; plat = [p[1] for p in planned_ll]
    ax.plot(plon, plat, color="#1f77ff", linestyle="-", linewidth=2.2,
            label="规划线路 (AI-Planned)", zorder=4)

    if start_ll:
        ax.scatter([start_ll[1]], [start_ll[0]], c="green", s=120, marker="o",
                   edgecolors="black", label="起点 (Start)", zorder=5)
    if end_ll:
        ax.scatter([end_ll[1]], [end_ll[0]], c="orange", s=140, marker="*",
                   edgecolors="black", label="终点 (End)", zorder=5)

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("复杂山区输电线路 AI 路径规划对比\nDEM + 原始线路(红虚) + 规划线路(蓝实)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label="Hillshade / Elevation")
    fig.tight_layout()
    if os.path.exists(out_path):
        print(f"  [提示] 输出文件已存在, 将覆盖: {out_path}")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  对比图已保存: {out_path}")


# ============================================================================
# 12. 矢量写出 (优先 geopandas .shp, 回退 .geojson)
# ============================================================================
def export_route(planned_ll, dem_crs, shp_path, geojson_path):
    line = LineString([(lon, lat) for (lon, lat) in planned_ll])
    if HAS_GPD:
        # 覆盖前提示 (低成本的输出覆盖保护)
        if os.path.exists(shp_path):
            print(f"  [提示] 输出文件已存在, 将覆盖: {shp_path}")
        gdf = gpd.GeoDataFrame({"name": ["planned_route"], "geometry": [line]}, crs=dem_crs or "EPSG:4326")
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        print(f"  矢量路径已保存: {shp_path}")
        return shp_path
    # 回退: 写 GeoJSON (纯文本, 通用可读)
    feat = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": dem_crs or "EPSG:4326"}},
        "features": [{
            "type": "Feature",
            "properties": {"name": "planned_route"},
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for (lon, lat) in planned_ll],
            },
        }],
    }
    if os.path.exists(geojson_path):
        print(f"  [提示] 输出文件已存在, 将覆盖: {geojson_path}")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(feat, f, ensure_ascii=False, indent=2)
    print(f"  矢量路径已保存(GeoJSON): {geojson_path}")
    return geojson_path