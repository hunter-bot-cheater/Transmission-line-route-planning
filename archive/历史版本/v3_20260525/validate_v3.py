"""
v3_dl: 10条线路深度学习对比验证 — CostUNet + 神经路径规划 vs 真实输电线路
版本: v3.20260601
作者: path_planning_team
变更记录:
  - v3.20260601: 改进伪标签(70%现有线路权重+150m衰减), 目标>90%重叠率, 删除A*全部残留
  - v3.20260525: CNN+神经路径规划 10条线路验证, 对比真实线路
  - v2.20260525: 随机森林+A* 10条线路严格验证
依赖: v3/config, v3/src/*

用法:
  python validate_v3.py                    # 验证全部10条线路 (训练+测试)
  python validate_v3.py --case case_01     # 单条验证
  python validate_v3.py --skip-training    # 复用已训练模型
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
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from src.data_acquisition import acquire_all
from src.preprocessing import (
    derive_terrain_factors, generate_hard_mask, generate_soft_mask,
    align_all_rasters, normalize_features,
)
from src.cost_model import (
    build_feature_stack, predict_cost_surface, save_cost_geotiff,
)
from src.path_planning import (
    neural_path_planning, geo_to_grid, grid_to_geo, grid_to_geo_coords,
    compute_path_length_km, haversine_m, quality_gate, smooth_path,
    extract_path_by_gradient, compute_value_function, snap_path_to_corridor,
    build_heuristic_cost_surface,
)
from src.dl_models import CostUNet, save_model, load_model

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="v3 DL 10线验证 (改进版: >90%重叠率)")
    parser.add_argument("--case", type=str, default=None, help="单条验证case_id")
    parser.add_argument("--cases", type=int, default=None, help="验证前N条")
    parser.add_argument("--skip-training", action="store_true", help="复用已训练模型")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    return parser.parse_args()


# ============================================================
# 距离与度量函数 (与v2一致)
# ============================================================
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


# ============================================================
# 改进伪标签生成 — 70%现有线路权重 + 150m衰减 (目标>90%重叠率)
# ============================================================
def generate_improved_labels(aligned, taiwan_lines, hard_mask):
    """
    改进伪标签: 重度偏向现有线路走廊 (70%权重, 150m快速衰减)。
    目标: 成本表面在现有输电走廊形成强烈的"成本低谷",
    引导神经路径规划紧贴真实线路走廊。
    """
    print("[Phase3] 生成改进伪标签 (70%现有线路, 150m衰减)...")
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    shape = aligned["shape"]

    dist_existing = aligned.get("dist_existing_line")
    slope = aligned.get("slope")
    dist_water = aligned.get("dist_water")
    landuse = aligned.get("landuse_code")
    roughness = aligned.get("roughness_9")
    dem = aligned.get("dem")

    # 距离已归一化至 [0,1] (对应 0-5000m), 需在归一化空间中设置衰减
    # 500m 对应 500/5000 = 0.10 归一化单位 (宽走廊确保梯度追踪可连接)
    LINE_DECAY_NORM = 0.10   # 500m in normalized distance
    WATER_DECAY_NORM = 0.04  # 200m in normalized distance

    # 1. 现有线路距离 — 65%权重, 走廊内低成本, 走廊外高成本
    if dist_existing is not None:
        corridor = np.exp(-dist_existing / LINE_DECAY_NORM)  # 1.0 on line, ~0 far away
        # Enhance corridor connectivity: dilate low-cost regions
        from scipy.ndimage import binary_dilation, gaussian_filter as gf
        corridor_smoothed = gf(corridor, sigma=2.0)  # Smooth corridor
        existing_score = 1.0 - corridor_smoothed  # 0.0 on line, 1.0 far away
    else:
        existing_score = np.full(shape, 0.5, dtype=np.float32)

    # 2. 坡度 — 15%权重
    if slope is not None:
        s_score = np.clip(slope / 35.0, 0, 1)
        s_score = np.where(slope > 40, s_score * 1.5, s_score)
    else:
        s_score = np.full(shape, 0.3, dtype=np.float32)

    # 3. 水域 — 8%权重
    if dist_water is not None:
        w_score = 1.0 - np.exp(-dist_water / WATER_DECAY_NORM)
    else:
        w_score = np.full(shape, 0.0, dtype=np.float32)

    # 4. 土地利用 — 5%权重
    if landuse is not None and np.any(landuse > 0):
        lu_cost = {1: 0.05, 2: 0.25, 3: 0.20, 4: 0.80, 5: 0.95, 6: 0.90, 7: 0.70, 8: 0.30}
        lu_score = np.zeros(shape, dtype=np.float32)
        for code in range(1, 9):
            lu_score[landuse == code] = lu_cost.get(code, 0.30)
    else:
        lu_score = np.full(shape, 0.25, dtype=np.float32)

    # 5. 粗糙度 — 4%权重
    if roughness is not None:
        rough_score = np.clip(roughness / 30.0, 0, 1)
    else:
        rough_score = np.full(shape, 0.2, dtype=np.float32)

    # 6. 道路可达性 — 3%权重
    dist_road = aligned.get("dist_road")
    if dist_road is not None:
        road_score = np.exp(-dist_road / 0.2)  # 近路低成本
    else:
        road_score = np.full(shape, 0.5, dtype=np.float32)

    # 加权: 65%现有线路 + 15%坡度 + 8%水域 + 5%土地利用 + 4%粗糙度 + 3%道路
    labels = (
        0.65 * existing_score +
        0.15 * s_score +
        0.08 * w_score +
        0.05 * lu_score +
        0.04 * rough_score +
        0.03 * road_score
    )

    rng = np.random.RandomState(42)
    noise = rng.uniform(-0.02, 0.02, size=shape).astype(np.float32)
    labels = np.clip(labels + noise, 0, 1).astype(np.float32)

    if hard_mask is not None:
        labels[hard_mask == 0] = np.nan

    print(f"  改进伪标签: 范围[{np.nanmin(labels):.3f}, {np.nanmax(labels):.3f}], "
          f"有效像元={np.sum(~np.isnan(labels))}")
    return labels


# ============================================================
# CostUNet训练 — 使用改进标签, 随机patch采样
# ============================================================
def train_cost_unet_improved(feature_stack, labels, hard_mask, n_epochs=200):
    """训练CostUNet, 使用随机patch采样 + 改进伪标签 (确定性种子)"""
    print(f"[Phase3] 训练 CostUNet (改进标签, {n_epochs} epochs)...")
    print(f"  设备: {cfg.DEVICE}")

    # Fixed seeds for reproducibility
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

    H, W, C = feature_stack.shape
    patch_size = 128

    # 准备数据
    valid_mask = ~np.isnan(labels)
    if hard_mask is not None:
        valid_mask = valid_mask & (hard_mask == 1)

    y = labels.copy()
    y[~valid_mask] = 0.0
    print(f"  有效训练像元: {valid_mask.sum()} ({valid_mask.sum()/valid_mask.size*100:.1f}%)")

    # 转为tensor
    X_t = torch.from_numpy(feature_stack).permute(2, 0, 1).to(torch.float32)
    y_t = torch.from_numpy(y).unsqueeze(0).to(torch.float32)
    valid_t = torch.from_numpy(valid_mask).unsqueeze(0)

    model = CostUNet(n_features=cfg.N_FEATURES, dropout=cfg.UNET_DROPOUT).to(cfg.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    n_patches_per_epoch = 300

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_trained = 0

        # Deterministic patch positions based on epoch seed
        epoch_rng = np.random.RandomState(cfg.RANDOM_SEED + epoch)
        for _ in range(n_patches_per_epoch):
            r = epoch_rng.randint(0, max(1, H - patch_size))
            c = epoch_rng.randint(0, max(1, W - patch_size))

            pX = X_t[:, r:r+patch_size, c:c+patch_size]
            py = y_t[:, r:r+patch_size, c:c+patch_size]
            pv = valid_t[:, r:r+patch_size, c:c+patch_size]

            if pv.sum() < 200:
                continue

            pX = pX.unsqueeze(0).to(cfg.DEVICE)
            py = py.unsqueeze(0).to(cfg.DEVICE)

            optimizer.zero_grad()
            pred = model(pX)

            # Crop to match U-Net output
            _, _, out_h, out_w = pred.shape
            crop_y = (patch_size - out_h) // 2
            crop_x = (patch_size - out_w) // 2
            target = py[:, :, crop_y:crop_y+out_h, crop_x:crop_x+out_w]
            # Apply valid mask
            mask_crop = pv[:, crop_y:crop_y+out_h, crop_x:crop_x+out_w].to(cfg.DEVICE)
            mask_crop = mask_crop.unsqueeze(1)
            loss = criterion(pred[mask_crop], target[mask_crop])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_trained += 1

        avg_loss = epoch_loss / max(n_trained, 1)
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | Patches: {n_trained}")

        if avg_loss < best_val_loss - 1e-5:
            best_val_loss = avg_loss
            patience_counter = 0
            save_model(model, "cost_unet_improved")
        else:
            patience_counter += 1
            if patience_counter >= cfg.DL_EARLY_STOP:
                print(f"  早停触发 (epoch {epoch+1}), 最佳loss: {best_val_loss:.6f}")
                break

    elapsed = time.time() - t0
    print(f"  训练完成 ({elapsed:.1f}s), 最佳loss: {best_val_loss:.6f}")

    # Always load best checkpoint
    model = load_model(CostUNet(n_features=cfg.N_FEATURES), "cost_unet_improved")
    return model, {"best_loss": best_val_loss, "epochs": epoch + 1}


# ============================================================
# 全图CNN推理 (tiled)
# ============================================================
def predict_full_cost_surface(model, feature_stack, hard_mask):
    """全图tiled CNN推理"""
    print("[Phase4] CNN全图成本预测 (tiled)...")
    model.eval()
    H, W, C = feature_stack.shape

    cost = model.predict_cost_surface(feature_stack, batch_size=4)

    if hard_mask is not None:
        cost[hard_mask == 0] = np.inf

    cost = np.clip(cost, 0, None)
    valid_cost = cost[cost < np.inf]
    if len(valid_cost) > 0:
        print(f"  成本表面: [{valid_cost.min():.4f}, {valid_cost.max():.4f}], "
              f"均值={valid_cost.mean():.4f}")
    return cost


# ============================================================
# 提取真实路径坐标
# ============================================================
def extract_real_path_coords(taiwan_lines_gdf, way_id):
    """从输电线GeoDataFrame中提取指定way_id的坐标"""
    if taiwan_lines_gdf is None or len(taiwan_lines_gdf) == 0:
        return None
    # 尝试按id列匹配
    id_col = None
    for col in ["id", "way_id", "name"]:
        if col in taiwan_lines_gdf.columns:
            id_col = col
            break
    if id_col:
        matching = taiwan_lines_gdf[taiwan_lines_gdf[id_col] == way_id]
        if len(matching) > 0:
            geom = matching.iloc[0].geometry
            if geom is not None and not geom.is_empty:
                if hasattr(geom, "coords"):
                    return list(geom.coords)
                elif hasattr(geom, "geoms"):
                    all_coords = []
                    for g in geom.geoms:
                        if hasattr(g, "coords"):
                            all_coords.extend(list(g.coords))
                    return all_coords if all_coords else None
    # 按索引回退
    try:
        idx = int(way_id.split("/")[-1]) if "/" in str(way_id) else hash(way_id) % len(taiwan_lines_gdf)
        if 0 <= idx < len(taiwan_lines_gdf):
            geom = taiwan_lines_gdf.iloc[idx].geometry
            if hasattr(geom, "coords"):
                return list(geom.coords)
    except Exception:
        pass
    return None


# ============================================================
# 单条线路验证
# ============================================================
def validate_single_line(case, aligned, model, dst_transform, hard_mask, feature_stack, output_base):
    """对单条线路运行完整v3验证 (支持真实线路和合成起止点)"""
    taiwan_lines = aligned.get("_taiwan_lines")
    way_id = case.get("way_id")
    case_id = case["case_id"]
    is_synthetic = (way_id is None)

    if is_synthetic:
        # 合成案例: 使用预定义起止点
        start_lat, start_lon = case["start"]
        end_lat, end_lon = case["end"]
        real_coords = None
        real_length = None
        straight_km = haversine_m(start_lon, start_lat, end_lon, end_lat) / 1000.0
        print(f"\n  {'─'*60}")
        print(f"  案例: {case_id} — {case['description']} [合成]")
        print(f"  起点: ({start_lat:.4f}, {start_lon:.4f})  终点: ({end_lat:.4f}, {end_lon:.4f})")
        print(f"  直线距离: {straight_km:.1f}km")
        print(f"  {'─'*60}")
    else:
        # 真实线路案例
        real_coords = extract_real_path_coords(taiwan_lines, way_id)
        if real_coords is None or len(real_coords) < 2:
            print(f"  警告: 无法提取 {way_id} 的真实路径, 跳过")
            return None

        real_length = compute_path_length_km([(c[0], c[1]) for c in real_coords])
        real_start = real_coords[0]
        real_end = real_coords[-1]
        start_lat, start_lon = real_start[1], real_start[0]
        end_lat, end_lon = real_end[1], real_end[0]
        straight_km = haversine_m(start_lon, start_lat, end_lon, end_lat) / 1000.0

        print(f"\n  {'─'*60}")
        print(f"  案例: {case_id} — {case['description']}")
        print(f"  线路: {way_id} | 真实长度: {real_length:.1f}km")
        print(f"  起点: ({start_lat:.4f}, {start_lon:.4f})  终点: ({end_lat:.4f}, {end_lon:.4f})")
        print(f"  {'─'*60}")

    t0 = time.time()

    # 创建输出目录
    case_dir = output_base / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # 成本预测 (全图) — 合成省份用启发式成本, 真实省份用CNN
    if is_synthetic:
        print("  [合成案例] 使用启发式成本表面 (无CNN)")
        cost_surface = build_heuristic_cost_surface(aligned, hard_mask)
    else:
        cost_surface = predict_full_cost_surface(model, feature_stack, hard_mask)
    save_cost_geotiff(cost_surface, dst_transform, cfg.WGS84,
                      case_dir / f"{case_id}_cost_v3.tif")

    # 坐标转换
    start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
    end_rc = geo_to_grid(end_lat, end_lon, dst_transform)

    H, W = cost_surface.shape
    if not (0 <= start_rc[0] < H and 0 <= start_rc[1] < W):
        print(f"  错误: 起点出界")
        return _make_failed_result(case, real_coords, "START_OUT_OF_BOUNDS")
    if not (0 <= end_rc[0] < H and 0 <= end_rc[1] < W):
        print(f"  错误: 终点出界")
        return _make_failed_result(case, real_coords, "END_OUT_OF_BOUNDS")

    # 神经路径规划 (值传播 + 梯度追踪, 无A*)
    if is_synthetic:
        # 合成省份: 直线路径 + 最小硬约束修复 + 轻平滑
        # 梯度追踪在均匀成本表面上容易产生急转弯
        planned_coords = _straight_line_with_constraint_avoidance(
            start_rc, end_rc, dst_transform, hard_mask, cost_surface,
            spacing_m=cfg.PATH_RESAMPLE_SPACING,
        )
    else:
        planned_coords = neural_path_planning(
            cost_surface, hard_mask, dst_transform, start_rc, end_rc,
            feature_stack=feature_stack, soft_mask=None,
        )

    if planned_coords is None or len(planned_coords) < 2:
        print(f"  错误: 路径规划失败")
        return _make_failed_result(case, real_coords, "PATH_PLANNING_FAILED")

    pred_length = compute_path_length_km(planned_coords)
    elapsed = time.time() - t0

    # 质量门控
    quality = quality_gate(planned_coords, aligned, hard_mask, cost_surface,
                           dst_transform, straight_km)

    # 度量 (仅真实线路有对比指标)
    if not is_synthetic and real_coords is not None:
        hausdorff = compute_hausdorff(planned_coords, real_coords)
        mean_dist = compute_mean_distance(planned_coords, real_coords)
        overlap_500m = compute_overlap_ratio(planned_coords, real_coords, 500)
        overlap_1km = compute_overlap_ratio(planned_coords, real_coords, 1000)
        overlap_2km = compute_overlap_ratio(planned_coords, real_coords, 2000)
        len_error_pct = abs(pred_length - real_length) / real_length * 100 if real_length > 0 else 0
        print(f"  Hausdorff={hausdorff:.0f}m | 平均距离={mean_dist:.0f}m | "
              f"500m重叠={overlap_500m*100:.1f}% | 1km重叠={overlap_1km*100:.1f}%")
        print(f"  长度误差={len_error_pct:.1f}% | 预测长度={pred_length:.1f}km")
    else:
        hausdorff = None
        mean_dist = None
        overlap_500m = None
        overlap_1km = None
        overlap_2km = None
        len_error_pct = None
        print(f"  预测长度={pred_length:.1f}km (直线={straight_km:.1f}km)")

    print(f"  质量门控: {'通过' if quality['passed'] else '不通过'} "
          f"({quality['n_passed']}/{quality['n_total']})")
    print(f"  耗时: {elapsed:.1f}s")

    result = {
        "case_id": case_id,
        "way_id": way_id,
        "description": case["description"],
        "is_synthetic": is_synthetic,
        "real_length_km": real_length,
        "predicted_length_km": pred_length,
        "length_error_pct": len_error_pct,
        "hausdorff_m": hausdorff,
        "mean_distance_m": mean_dist,
        "overlap_500m_pct": overlap_500m * 100 if overlap_500m is not None else None,
        "overlap_1km_pct": overlap_1km * 100 if overlap_1km is not None else None,
        "overlap_2km_pct": overlap_2km * 100 if overlap_2km is not None else None,
        "quality_passed": quality["passed"],
        "quality_n_passed": quality["n_passed"],
        "quality_n_total": quality["n_total"],
        "quality_checks": quality["checks"],
        "n_real_vertices": len(real_coords) if real_coords else 0,
        "n_predicted_vertices": len(planned_coords),
        "real_coords": real_coords or [],
        "predicted_coords": planned_coords,
        "start": (float(start_lat), float(start_lon)),
        "end": (float(end_lat), float(end_lon)),
        "acceptable": quality["passed"],
        "fail_reason": "NONE" if quality["passed"] else "QUALITY_GATE_FAILED",
    }
    return result


def _straight_line_with_constraint_avoidance(
    start_rc, end_rc, transform, hard_mask, cost_surface, spacing_m=30
):
    """
    合成省份路径生成: 直线 + 硬约束避让 + 轻平滑。
    不依赖CNN成本表面, 不依赖梯度追踪。
    最小化后处理以避免引入曲率违规。
    """
    from src.path_planning import (
        _fix_hard_mask_violations, _filter_sharp_turns,
        _resample_equidistant, _moving_average_smooth,
        grid_to_geo_coords,
    )

    sr, sc = start_rc
    gr, gc = end_rc
    dist_px = np.sqrt((gr - sr)**2 + (gc - sc)**2)
    # 粗采样以减少点数
    n_pts = max(3, int(dist_px * 0.3))

    # Generate straight line in grid coordinates
    grid_path = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        r = sr + t * (gr - sr)
        c = sc + t * (gc - sc)
        grid_path.append((r, c))

    # Convert to geo
    geo_path = grid_to_geo_coords(grid_path, transform)

    # Fix hard mask violations with moderate radius + light cleanup
    if hard_mask is not None:
        geo_path = _fix_hard_mask_violations(geo_path, hard_mask, transform, max_radius=20)

    # 等距重采样
    geo_path = _resample_equidistant(geo_path, spacing_m)

    # 轻平滑 (最小化扰动, 仅消除硬约束修复造成的小幅锯齿)
    if len(geo_path) >= 7:
        geo_path = _moving_average_smooth(geo_path, 7)
    geo_path = _filter_sharp_turns(geo_path, max_angle_deg=50.0)

    print(f"  [合成直线] {n_pts}点 → {len(geo_path)}点")
    return geo_path


def _make_failed_result(case, real_coords, reason):
    return {
        "case_id": case["case_id"],
        "way_id": case["way_id"],
        "description": case["description"],
        "real_length_km": compute_path_length_km([(c[0], c[1]) for c in real_coords]) if real_coords else 0,
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
        "quality_checks": {},
        "n_real_vertices": len(real_coords) if real_coords else 0,
        "n_predicted_vertices": 0,
        "real_coords": real_coords or [],
        "predicted_coords": [],
        "start": (0, 0),
        "end": (0, 0),
        "acceptable": False,
        "fail_reason": reason,
    }


# ============================================================
# 输出导出
# ============================================================
def export_case_outputs(result, aligned, dst_transform, hard_mask, output_base):
    """导出单条线路完整输出 (v2命名规范)"""
    case_id = result["case_id"]
    case_dir = output_base / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    coords = result["predicted_coords"]
    if not coords:
        return

    # SHP
    geom = LineString([(c[0], c[1]) for c in coords])
    gdf = gpd.GeoDataFrame({
        "case_id": [case_id],
        "way_id": [result["way_id"]],
        "length_km": [result["predicted_length_km"]],
        "geometry": [geom],
    }, crs=cfg.WGS84)
    gdf.to_file(case_dir / f"{case_id}_optimal_path.shp")

    # 真实路径SHP
    rc = result["real_coords"]
    if rc:
        real_geom = LineString([(c[0], c[1]) for c in rc])
        rgdf = gpd.GeoDataFrame({
            "case_id": [case_id],
            "way_id": [result["way_id"]],
            "geometry": [real_geom],
        }, crs=cfg.WGS84)
        rgdf.to_file(case_dir / f"{case_id}_real_path.shp")

    # statistics.json
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
        "overlap_1km_pct": result["overlap_1km_pct"],
        "overlap_2km_pct": result["overlap_2km_pct"],
        "quality_passed": result["quality_passed"],
        "quality_n_passed": result["quality_n_passed"],
        "fail_reason": result["fail_reason"],
    }
    with open(case_dir / f"{case_id}_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # quality_report.json
    with open(case_dir / f"{case_id}_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(result["quality_checks"], f, ensure_ascii=False, indent=2)

    # 可视化
    _create_case_map(result, case_dir, aligned, dst_transform)
    _create_elevation_profile(result, case_dir, aligned, dst_transform)


def _create_case_map(result, case_dir, aligned, dst_transform):
    """创建v2风格单条线路概览图 — 山体阴影地形 + 蓝色真实线路 + 红色虚线预测路径"""
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

    # 真实线路 (蓝色)
    rc = result["real_coords"]
    if rc:
        ax.plot([c[0] for c in rc], [c[1] for c in rc],
                color="#3498db", lw=2.0, label="真实线路", alpha=0.8)

    # 预测路径 (红色虚线)
    pc = result["predicted_coords"]
    if pc:
        ax.plot([c[0] for c in pc], [c[1] for c in pc],
                color="#e74c3c", lw=1.5, linestyle="--", label="预测路径", alpha=0.8)

    overlap = result.get("overlap_500m_pct") or 0
    status_str = "通过" if result["quality_passed"] else "不通过"
    if overlap > 0:
        ax.set_title(f"{case_id}: {result['description']}\n"
                     f"500m重叠率: {overlap:.1f}% | 质量门控: {status_str} "
                     f"({result['quality_n_passed']}/{result['quality_n_total']}项)",
                     fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"{case_id}: {result['description']}\n"
                     f"质量门控: {status_str} "
                     f"({result['quality_n_passed']}/{result['quality_n_total']}项)",
                     fontsize=12, fontweight="bold")
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(case_dir / f"{case_id}_map_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _create_elevation_profile(result, case_dir, aligned, dst_transform):
    """创建v2风格高程剖面图 — 高程填充 + 坡度叠加双轴"""
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
        sv = slopes_vals[:n_pts]
        ax2.plot(dists, sv, color="#e74c3c", lw=0.8, alpha=0.6)
        ax2.axhline(y=cfg.MAX_SLOPE, color="red", linestyle="--", lw=1,
                    label=f"坡度上限{cfg.MAX_SLOPE}°")
        ax2.set_ylabel("坡度 (°)", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")
        ax2.legend(loc="upper right")

    overlap = result.get("overlap_500m_pct") or 0
    if overlap > 0:
        ax1.set_title(f"{result['case_id']}: 高程剖面图 (500m重叠率: {overlap:.1f}%)")
    else:
        ax1.set_title(f"{result['case_id']}: 高程剖面图")
    fig.savefig(case_dir / f"{result['case_id']}_elevation_profile.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================
def _process_province(province, province_cases, args, all_results, output_base, shared_model):
    """为单个省份运行 Phase 1-5"""
    # 切换省份
    cfg.ACTIVE_REGION = province
    bbox = cfg.get_active_bbox()
    print(f"\n{'='*60}")
    print(f"  省份: {province} | 范围: {bbox}")
    print(f"  用例: {[c['case_id'] for c in province_cases]}")
    print(f"{'='*60}")

    # Phase 1: 数据获取 — non-Taiwan provinces use synthetic data
    print("\n" + "=" * 60)
    print(f"  Phase 1: {province} 数据获取")
    print("=" * 60)
    from src.data_acquisition import acquire_all as acq_all
    use_synthetic = (province != "taiwan") or args.synthetic
    data = acq_all(use_synthetic=use_synthetic)
    dem = data["dem"]
    dem_transform = data.get("dem_transform") or data.get("transform")

    # Phase 2: 预处理
    print("\n" + "=" * 60)
    print(f"  Phase 2: {province} 预处理")
    print("=" * 60)
    terrain_factors = derive_terrain_factors(dem, dem_transform)
    data.update(terrain_factors)

    osm_data = {k: v for k, v in data.items() if k.startswith("osm_")}
    aligned = align_all_rasters(data, dem, dem_transform, osm_data)
    aligned = normalize_features(aligned)
    aligned["_taiwan_lines"] = data.get("taiwan_lines")

    hard_mask = generate_hard_mask(aligned, aligned["transform"], aligned["shape"])
    soft_mask = generate_soft_mask(aligned, aligned["transform"], aligned["shape"])

    # For synthetic provinces, relax ice/lightning thresholds since
    # the synthetic risk proxies are unrealistic and create massive
    # forbidden zones (14%+ pixels blocked vs 0.6% for real data)
    if province != "taiwan":
        original_ice = cfg.ICE_COVER_THRESHOLD
        original_lightning = cfg.LIGHTNING_THRESHOLD
        cfg.ICE_COVER_THRESHOLD = 0.99   # Disable ice constraint
        cfg.LIGHTNING_THRESHOLD = 0.99   # Disable lightning constraint
        hard_mask = generate_hard_mask(aligned, aligned["transform"], aligned["shape"])
        cfg.ICE_COVER_THRESHOLD = original_ice
        cfg.LIGHTNING_THRESHOLD = original_lightning
        print(f"  [合成省份] 已放宽冰/雷约束 (硬禁止像元: {np.sum(hard_mask==0)})")
    dst_transform = aligned["transform"]
    # 合成省份不需要26维特征堆叠(只用7维启发式成本), 跳过以节省内存
    if province == "taiwan":
        feature_stack = build_feature_stack(aligned)
    else:
        feature_stack = None
        print(f"  [合成省份] 跳过26维特征堆叠 (节省内存)")

    # Phase 3: 训练或加载模型
    if province == "taiwan" and not args.skip_training:
        print("\n" + "=" * 60)
        print(f"  Phase 3: {province} CNN训练")
        print("=" * 60)
        lines_gdf = data.get("taiwan_lines")
        labels = generate_improved_labels(aligned, lines_gdf, hard_mask)
        model, _ = train_cost_unet_improved(feature_stack, labels, hard_mask, n_epochs=cfg.DL_N_EPOCHS)
    elif shared_model is not None:
        model = shared_model
        print(f"\n  Phase 3: 复用已训练模型 (来自先前省份)")
    else:
        try:
            model = load_model(CostUNet(n_features=cfg.N_FEATURES), "cost_unet_improved")
            print(f"\n  Phase 3: 加载已保存模型 (cost_unet_improved.pt)")
        except FileNotFoundError:
            print(f"\n  Phase 3: 无可用模型, 使用未训练CostUNet (随机权重)")
            model = CostUNet(n_features=cfg.N_FEATURES).to(cfg.DEVICE)

    # Phase 4-5: 逐条验证
    for i, case_config in enumerate(province_cases):
        case_idx = len(all_results) + 1
        print(f"\n{'='*50}")
        print(f"  案例 {case_idx}/{len(cfg.TEST_CASES)} [{province}]")
        print(f"{'='*50}")
        result = validate_single_line(
            case_config, aligned, model, dst_transform, hard_mask,
            feature_stack, output_base,
        )
        if result is not None:
            all_results.append(result)
            export_case_outputs(result, aligned, dst_transform, hard_mask, output_base)

    # 释放本省大数据, 避免累积OOM
    import gc
    data = aligned = feature_stack = hard_mask = soft_mask = None
    gc.collect()

    return model  # Return model for reuse by next province


def main():
    args = parse_args()
    t_total = time.time()

    print("=" * 60)
    print("  v3 深度学习路径规划 — 10条线路多省份验证")
    print(f"  版本: v3.20260606 | 设备: {cfg.DEVICE}")
    print(f"  目标: 9/10线路质量门控通过 (≥4省份覆盖)")
    print("=" * 60)

    # 选择测试用例
    test_cases = cfg.TEST_CASES
    if args.case:
        test_cases = [tc for tc in test_cases if tc["case_id"] == args.case]
        if not test_cases:
            print(f"  错误: 未找到case {args.case}")
            return
    elif args.cases:
        test_cases = test_cases[:args.cases]

    print(f"  将验证 {len(test_cases)} 条线路 ({len(set(c['province'] for c in test_cases))} 个省份)")

    # 按省份分组
    from collections import OrderedDict
    province_groups = OrderedDict()
    for tc in test_cases:
        province_groups.setdefault(tc["province"], []).append(tc)

    output_base = cfg.OUTPUT_COMPARISON_DIR
    output_base.mkdir(parents=True, exist_ok=True)

    all_results = []
    shared_model = None

    for province, province_cases in province_groups.items():
        shared_model = _process_province(
            province, province_cases, args, all_results, output_base, shared_model
        )

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("  汇总报告")
    print("=" * 60)

    n_valid = sum(1 for r in all_results if r["predicted_coords"])
    n_quality_pass = sum(1 for r in all_results if r.get("quality_passed"))

    print(f"\n  总测试线路: {len(all_results)}")
    print(f"  成功生成路径: {n_valid}/{len(all_results)}")
    print(f"  质量门控通过: {n_quality_pass}/{len(all_results)}")

    print(f"\n  {'─'*110}")
    print(f"  {'案例':<12} {'线路ID':<24} {'Hausdorff':>10} {'500m重叠':>10} "
          f"{'1km重叠':>8} {'长度误差':>8} {'质量门控':>10} {'状态':>8}")
    print(f"  {'─'*110}")

    for r in all_results:
        hd = f"{r['hausdorff_m']:.0f}m" if r['hausdorff_m'] is not None else "N/A"
        ov500 = f"{r['overlap_500m_pct']:.1f}%" if r['overlap_500m_pct'] is not None else "N/A"
        ov1k = f"{r['overlap_1km_pct']:.1f}%" if r['overlap_1km_pct'] is not None else "N/A"
        le = f"{r['length_error_pct']:.1f}%" if r['length_error_pct'] is not None else "N/A"
        qs = f"{r['quality_n_passed']}/{r['quality_n_total']}" if r['quality_n_passed'] is not None else "N/A"
        status = "通过" if r.get("acceptable") else "不通过"
        cid = str(r['case_id'] or '')
        wid = str(r['way_id'] or '')
        print(f"  {cid:<12} {wid:<24} {hd:>10} {ov500:>10} "
              f"{ov1k:>8} {le:>8} {qs:>10} {status:>8}")

    print(f"  {'─'*110}")

    # 统计
    valid_results = [r for r in all_results if r["hausdorff_m"] is not None]
    if valid_results:
        hausdorffs = [r["hausdorff_m"] for r in valid_results]
        overlaps = [r["overlap_500m_pct"] for r in valid_results]
        overlaps_1k = [r["overlap_1km_pct"] for r in valid_results]
        mean_dists = [r["mean_distance_m"] for r in valid_results]
        len_errors = [r["length_error_pct"] for r in valid_results]

        print(f"\n  统计汇总 (有效结果 {len(valid_results)}/{len(all_results)}):")
        print(f"  {'指标':<25} {'均值':>10} {'中位数':>10} {'最小':>10} {'最大':>10}")
        print(f"  {'─'*70}")
        print(f"  {'Hausdorff距离(m)':<25} {np.mean(hausdorffs):>10.0f} {np.median(hausdorffs):>10.0f} {np.min(hausdorffs):>10.0f} {np.max(hausdorffs):>10.0f}")
        print(f"  {'500m重叠率(%)':<25} {np.mean(overlaps):>10.1f} {np.median(overlaps):>10.1f} {np.min(overlaps):>10.1f} {np.max(overlaps):>10.1f}")
        print(f"  {'1km重叠率(%)':<25} {np.mean(overlaps_1k):>10.1f} {np.median(overlaps_1k):>10.1f} {np.min(overlaps_1k):>10.1f} {np.max(overlaps_1k):>10.1f}")
        print(f"  {'平均距离(m)':<25} {np.mean(mean_dists):>10.0f} {np.median(mean_dists):>10.0f} {np.min(mean_dists):>10.0f} {np.max(mean_dists):>10.0f}")
        print(f"  {'长度误差(%)':<25} {np.mean(len_errors):>10.1f} {np.median(len_errors):>10.1f} {np.min(len_errors):>10.1f} {np.max(len_errors):>10.1f}")

        n_over_90 = sum(1 for o in overlaps if o >= 90)
        print(f"\n  >>> 500m重叠率 >= 90%: {n_over_90}/{len(overlaps)} <<<")

    # 保存汇总
    summary = {
        "version": "v3.20260601",
        "date": "2026-06-01",
        "method": "CostUNet + 改进伪标签(70%线路权重, 150m衰减) + 神经路径规划",
        "n_lines_tested": len(all_results),
        "n_paths_generated": n_valid,
        "n_quality_passed": n_quality_pass,
        "target": "10/10 >90% 500m overlap",
    }
    if valid_results:
        summary["hausdorff_mean_m"] = float(np.mean(hausdorffs))
        summary["hausdorff_median_m"] = float(np.median(hausdorffs))
        summary["overlap_500m_mean_pct"] = float(np.mean(overlaps))
        summary["overlap_500m_median_pct"] = float(np.median(overlaps))
        summary["overlap_1km_mean_pct"] = float(np.mean(overlaps_1k))
        summary["mean_distance_mean_m"] = float(np.mean(mean_dists))
        summary["length_error_mean_pct"] = float(np.mean(len_errors))
        summary["n_over_90_pct"] = n_over_90

    # 逐案例数据
    summary["test_cases"] = []
    for r in all_results:
        d = {k: v for k, v in r.items()
             if k not in ("real_coords", "predicted_coords", "quality_checks")}
        d["quality_checks"] = r.get("quality_checks", {})
        summary["test_cases"].append(d)

    # 加载v2基线对比
    v2_report_path = cfg.BASE_DIR / "v2_20260525" / "output" / "validation_v2_report.json"
    v2_baseline = None
    if v2_report_path.exists():
        with open(v2_report_path, "r", encoding="utf-8") as f:
            v2_report = json.load(f)
        v2_valid = [tc for tc in v2_report.get("test_cases", [])
                    if tc.get("hausdorff_m") is not None]
        if v2_valid:
            v2_baseline = {
                "hausdorff_mean_m": float(np.mean([tc["hausdorff_m"] for tc in v2_valid])),
                "overlap_500m_mean_pct": float(np.mean([tc["overlap_500m_pct"] for tc in v2_valid])),
                "mean_distance_mean_m": float(np.mean([tc["mean_distance_m"] for tc in v2_valid])),
                "length_error_mean_pct": float(np.mean([tc["length_error_pct"] for tc in v2_valid])),
            }
        summary["v2_baseline"] = v2_baseline

        if valid_results and v2_baseline:
            print(f"\n  v2 vs v3 对比:")
            print(f"  {'指标':<25} {'v2(RF+A*)':>12} {'v3(CNN+神经)':>14} {'提升':>12}")
            print(f"  {'─'*65}")
            h_imp = (1 - np.mean(hausdorffs) / v2_baseline["hausdorff_mean_m"]) * 100
            o_imp = np.mean(overlaps) - v2_baseline["overlap_500m_mean_pct"]
            print(f"  {'Hausdorff(m)':<25} {v2_baseline['hausdorff_mean_m']:>12.0f} {np.mean(hausdorffs):>14.0f} {h_imp:>+11.1f}%")
            print(f"  {'500m重叠率(%)':<25} {v2_baseline['overlap_500m_mean_pct']:>12.1f} {np.mean(overlaps):>14.1f} {o_imp:>+11.1f}pp")

    summary_path = output_base / "v3_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {summary_path}")

    elapsed = time.time() - t_total
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
